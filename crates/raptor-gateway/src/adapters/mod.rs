//! Client adapters.
//!
//! * `/v1/messages` (Anthropic): translated to OpenAI chat, sent through the
//!   same path as `/v1/chat/completions` (model resolution, queue, cancel), and
//!   translated back. Claude Code uses this route.
//! * Generic OpenAI clients: [`normalize_openai_chat`] maps a few common
//!   variants (`max_completion_tokens`, `developer` role, legacy `functions`)
//!   onto what the upstream understands. Everything else passes through.

pub mod anthropic;

use std::convert::Infallible;
use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::{Extension, State};
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use futures_util::{Stream, StreamExt};
use raptor_core::RaptorError;
use serde_json::{json, Value};

use crate::auth::Principal;
use crate::error::{ApiError, Shape};
use crate::openai::upstream::Endpoint;
use crate::openai::{run_endpoint, sse_event_frame, sse_response, Reply, SseStream};
use crate::server::AppState;

/// Routes for the Anthropic-compatible surface.
///
/// Query strings (Claude Code sends `?beta=true`) and `anthropic-beta` /
/// `anthropic-version` headers are ignored: routing matches the path only, and
/// no handler reads those values.
pub(crate) fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/v1/messages", post(messages))
        .route("/v1/messages/count_tokens", post(count_tokens))
}

async fn messages(
    State(st): State<Arc<AppState>>,
    Extension(principal): Extension<Principal>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    match handle_messages(&st, &principal, &headers, &body).await {
        Ok(resp) => resp,
        // Every error on this route uses Anthropic's envelope.
        Err(e) => e.with_shape(Shape::Anthropic).into_response(),
    }
}

/// `POST /v1/messages/count_tokens`. Answers `{"input_tokens": n}` plus
/// `"approximate": true` and `"method"`. The count comes from the upstream's
/// `/tokenize` when it has one, and otherwise from a length estimate (about four
/// characters per token). Either way it excludes chat-template overhead, so it
/// is always reported as approximate.
async fn count_tokens(State(st): State<Arc<AppState>>, body: Bytes) -> Response {
    let run = async {
        let value: Value = serde_json::from_slice(&body)
            .map_err(|e| ApiError::bad_request(format!("Invalid JSON body: {e}")))?;
        let translated =
            anthropic::count_request_to_openai(&value).map_err(ApiError::from_raptor)?;
        let text = anthropic::input_text(&translated.body);
        let (tokens, method) = match st.upstream.tokenize(&text).await {
            Ok(Some(n)) => (n, "tokenize"),
            Ok(None) | Err(_) => (estimate_tokens(&text), "estimate"),
        };
        Ok::<Value, ApiError>(json!({
            "input_tokens": tokens,
            "approximate": true,
            "method": method,
        }))
    };
    match run.await {
        Ok(v) => Json(v).into_response(),
        Err(e) => e.with_shape(Shape::Anthropic).into_response(),
    }
}

/// Roughly one token per four characters, rounded up.
fn estimate_tokens(text: &str) -> u64 {
    let chars = text.chars().count() as u64;
    chars.div_ceil(4)
}

async fn handle_messages(
    st: &AppState,
    principal: &Principal,
    headers: &HeaderMap,
    body: &[u8],
) -> Result<Response, ApiError> {
    let value: Value = serde_json::from_slice(body)
        .map_err(|e| ApiError::bad_request(format!("Invalid JSON body: {e}")))?;
    let translated = anthropic::request_to_openai(&value).map_err(ApiError::from_raptor)?;
    let model = translated.model;
    match run_endpoint(
        st,
        principal,
        headers,
        translated.body,
        Endpoint::ChatCompletions,
    )
    .await?
    {
        Reply::Json(v) => {
            let message =
                anthropic::response_to_anthropic(&v, &model).map_err(ApiError::from_raptor)?;
            Ok(Json(message).into_response())
        }
        Reply::Stream(s) => Ok(sse_response(anthropic_frames(s, model))),
    }
}

/// Renders upstream OpenAI payloads as Anthropic named SSE events. On an
/// upstream or translation error, an `error` event is sent and no
/// `message_stop` follows.
fn anthropic_frames(
    mut inner: SseStream,
    model: String,
) -> impl Stream<Item = Result<Bytes, Infallible>> + Send {
    async_stream::stream! {
        let mut translator = anthropic::StreamTranslator::new(&model);
        let mut failed = false;
        while let Some(item) = inner.next().await {
            let payload = match item {
                Ok(p) if p == "[DONE]" => break,
                Ok(p) => p,
                Err(e) => {
                    yield Ok(error_frame(e));
                    failed = true;
                    break;
                }
            };
            match translator.push(&payload) {
                Ok(events) => {
                    for (name, data) in events {
                        yield Ok(Bytes::from(sse_event_frame(name, &data.to_string())));
                    }
                }
                Err(e) => {
                    yield Ok(error_frame(e));
                    failed = true;
                    break;
                }
            }
        }
        if !failed {
            for (name, data) in translator.finish() {
                yield Ok(Bytes::from(sse_event_frame(name, &data.to_string())));
            }
        }
    }
}

fn error_frame(err: RaptorError) -> Bytes {
    let body = ApiError::from_raptor(err)
        .with_shape(Shape::Anthropic)
        .body();
    Bytes::from(sse_event_frame("error", &body.to_string()))
}

/// Maps common generic-OpenAI-client request variants onto the upstream's
/// vocabulary. Only fills gaps; never overwrites a field the client set.
pub fn normalize_openai_chat(body: &mut Value) {
    let Some(obj) = body.as_object_mut() else {
        return;
    };
    // Newer clients send max_completion_tokens; llama-server reads max_tokens.
    if let Some(v) = obj.remove("max_completion_tokens") {
        obj.entry("max_tokens").or_insert(v);
    }
    // Legacy function calling: functions -> tools, function_call -> tool_choice.
    if let Some(funcs) = obj.remove("functions").and_then(|f| f.as_array().cloned()) {
        if !obj.contains_key("tools") {
            let tools: Vec<Value> = funcs
                .into_iter()
                .map(|f| json!({"type": "function", "function": f}))
                .collect();
            obj.insert("tools".into(), Value::Array(tools));
        }
    }
    if let Some(fc) = obj.remove("function_call") {
        if !obj.contains_key("tool_choice") {
            let choice = match fc {
                Value::String(s) => json!(s),
                Value::Object(o) => match o.get("name").and_then(Value::as_str) {
                    Some(name) => json!({"type": "function", "function": {"name": name}}),
                    None => json!("auto"),
                },
                _ => json!("auto"),
            };
            obj.insert("tool_choice".into(), choice);
        }
    }
    // `developer` is the newer name for the system role.
    if let Some(messages) = obj.get_mut("messages").and_then(Value::as_array_mut) {
        for m in messages.iter_mut() {
            if m.get("role").and_then(Value::as_str) == Some("developer") {
                m["role"] = json!("system");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::upstream::SseParser;

    const STREAM_IN: &str = include_str!("golden/openai_stream_text_then_tool.sse");
    const STREAM_OUT: &str = include_str!("golden/anthropic_stream_text_then_tool.sse");

    /// Parses `event:`/`data:` blocks into `(event, json)` pairs.
    fn parse_named_events(text: &str) -> Vec<(String, Value)> {
        text.split("\n\n")
            .filter(|block| !block.trim().is_empty())
            .map(|block| {
                let mut event = String::new();
                let mut data = String::new();
                for line in block.lines() {
                    if let Some(e) = line.strip_prefix("event: ") {
                        event = e.to_string();
                    } else if let Some(d) = line.strip_prefix("data: ") {
                        data = d.to_string();
                    }
                }
                (
                    event,
                    serde_json::from_str(&data).expect("golden data is JSON"),
                )
            })
            .collect()
    }

    #[test]
    fn golden_stream_translation_text_then_tool_use() {
        let mut parser = SseParser::default();
        let payloads = parser.feed(STREAM_IN.as_bytes()).unwrap();
        let mut translator = anthropic::StreamTranslator::new("claude-sonnet-4-5");
        let mut frames = String::new();
        for p in &payloads {
            if p == "[DONE]" {
                break;
            }
            for (name, data) in translator.push(p).unwrap() {
                frames.push_str(&sse_event_frame(name, &data.to_string()));
            }
        }
        for (name, data) in translator.finish() {
            frames.push_str(&sse_event_frame(name, &data.to_string()));
        }
        // Every frame is a complete named event.
        assert!(frames.starts_with("event: message_start\ndata: "));
        assert!(frames.ends_with("\n\n"));
        assert!(!frames.contains("[DONE]"));

        let got = parse_named_events(&frames);
        let expected = parse_named_events(STREAM_OUT);
        assert_eq!(got.len(), expected.len(), "event count");
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert_eq!(g, e, "event {i}");
        }
    }

    #[test]
    fn normalizes_max_completion_tokens_and_developer_role() {
        let mut body = json!({
            "model": "m",
            "max_completion_tokens": 64,
            "messages": [{"role": "developer", "content": "be brief"}, {"role": "user", "content": "hi"}]
        });
        normalize_openai_chat(&mut body);
        assert_eq!(body["max_tokens"], 64);
        assert!(body.get("max_completion_tokens").is_none());
        assert_eq!(body["messages"][0]["role"], "system");
    }

    #[test]
    fn does_not_overwrite_explicit_max_tokens() {
        let mut body =
            json!({"model": "m", "max_tokens": 10, "max_completion_tokens": 99, "messages": []});
        normalize_openai_chat(&mut body);
        assert_eq!(body["max_tokens"], 10);
    }

    #[test]
    fn maps_legacy_functions_to_tools() {
        let mut body = json!({
            "model": "m",
            "messages": [],
            "functions": [{"name": "f", "parameters": {"type": "object"}}],
            "function_call": {"name": "f"}
        });
        normalize_openai_chat(&mut body);
        assert_eq!(body["tools"][0]["type"], "function");
        assert_eq!(body["tools"][0]["function"]["name"], "f");
        assert_eq!(
            body["tool_choice"],
            json!({"type": "function", "function": {"name": "f"}})
        );
        assert!(body.get("functions").is_none());
    }

    #[test]
    fn normalization_ignores_non_objects() {
        let mut body = json!([1, 2]);
        normalize_openai_chat(&mut body);
        assert_eq!(body, json!([1, 2]));
    }
}
