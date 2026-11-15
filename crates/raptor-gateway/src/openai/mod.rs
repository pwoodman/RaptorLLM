//! OpenAI-compatible endpoints: `/v1/models`, `/v1/chat/completions`,
//! `/v1/completions`, `/v1/embeddings`.
//!
//! Requests are proxied to the [`Upstream`] with their JSON intact apart from
//! `model`, which is resolved to an installed or loaded id. Streamed responses
//! are re-framed as SSE: each upstream payload becomes one `data:` event, and
//! the stream always ends with exactly one `data: [DONE]`.

pub mod upstream;

use std::collections::BTreeMap;
use std::convert::Infallible;
use std::sync::Arc;

use axum::body::Body;
use axum::body::Bytes;
use axum::extract::{Extension, State};
use axum::http::{header, HeaderMap, HeaderValue};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::{Stream, StreamExt};
use serde_json::{json, Value};

use crate::auth::Principal;
use crate::context_api::{inject_context, WORKSPACE_HEADER};
use crate::error::ApiError;
use crate::queue::Permit;
use crate::server::AppState;

pub use upstream::{Completion, Endpoint, HttpUpstream, ModelCatalog, SseStream, Upstream};

/// Model-name prefixes that map to whichever model is loaded, so OpenAI and
/// Claude client defaults (`gpt-4o`, `claude-sonnet-...`) keep working.
const LOADED_MODEL_PREFIXES: [&str; 2] = ["gpt-", "claude-"];

/// Result of forwarding one request, before it is rendered in a client format.
pub(crate) enum Reply {
    /// Complete JSON body from the upstream.
    Json(Value),
    /// Upstream payload stream. Holds the queue permit until it ends or is dropped.
    Stream(SseStream),
}

/// Routes for the OpenAI-compatible API.
pub(crate) fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/embeddings", post(embeddings))
}

async fn list_models(State(st): State<Arc<AppState>>) -> Result<Json<Value>, ApiError> {
    let catalog = st
        .catalog
        .get(st.upstream.as_ref())
        .await
        .map_err(ApiError::from_raptor)?;
    let mut ids = catalog.installed.clone();
    if let Some(loaded) = &catalog.loaded {
        if !ids.contains(loaded) {
            ids.push(loaded.clone());
        }
    }
    for alias in st.aliases.keys() {
        if !ids.contains(alias) {
            ids.push(alias.clone());
        }
    }
    let data: Vec<Value> = ids
        .iter()
        .map(|id| json!({"id": id, "object": "model", "created": 0, "owned_by": "raptor"}))
        .collect();
    Ok(Json(json!({"object": "list", "data": data})))
}

async fn chat_completions(
    State(st): State<Arc<AppState>>,
    Extension(principal): Extension<Principal>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    openai_call(
        &st,
        &principal,
        &headers,
        &body,
        Endpoint::ChatCompletions,
        &["model", "messages"],
    )
    .await
    .unwrap_or_else(|e| e.into_response())
}

async fn completions(
    State(st): State<Arc<AppState>>,
    Extension(principal): Extension<Principal>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    openai_call(
        &st,
        &principal,
        &headers,
        &body,
        Endpoint::Completions,
        &["model", "prompt"],
    )
    .await
    .unwrap_or_else(|e| e.into_response())
}

async fn embeddings(
    State(st): State<Arc<AppState>>,
    Extension(principal): Extension<Principal>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    openai_call(
        &st,
        &principal,
        &headers,
        &body,
        Endpoint::Embeddings,
        &["model", "input"],
    )
    .await
    .unwrap_or_else(|e| e.into_response())
}

async fn openai_call(
    st: &AppState,
    principal: &Principal,
    headers: &HeaderMap,
    body: &Bytes,
    endpoint: Endpoint,
    required: &[&str],
) -> Result<Response, ApiError> {
    let mut body = parse_body(body, required)?;
    if endpoint == Endpoint::ChatCompletions {
        crate::adapters::normalize_openai_chat(&mut body);
    }
    match run_endpoint(st, principal, headers, body, endpoint).await? {
        Reply::Json(v) => Ok(Json(v).into_response()),
        Reply::Stream(s) => Ok(sse_response(openai_frames(s))),
    }
}

/// Parses a request body as a JSON object and checks required top-level fields.
pub(crate) fn parse_body(body: &[u8], required: &[&str]) -> Result<Value, ApiError> {
    let v: Value = serde_json::from_slice(body)
        .map_err(|e| ApiError::bad_request(format!("Invalid JSON body: {e}")))?;
    if !v.is_object() {
        return Err(ApiError::bad_request("Request body must be a JSON object."));
    }
    for field in required {
        if v.get(*field).is_none() {
            return Err(ApiError::bad_request(format!(
                "Missing required field '{field}'."
            )));
        }
    }
    Ok(v)
}

/// The shared path for every upstream call: model resolution, optional context
/// injection, queue admission, then the upstream call.
pub(crate) async fn run_endpoint(
    st: &AppState,
    principal: &Principal,
    headers: &HeaderMap,
    mut body: Value,
    endpoint: Endpoint,
) -> Result<Reply, ApiError> {
    let requested = body
        .get("model")
        .and_then(Value::as_str)
        .ok_or_else(|| ApiError::bad_request("'model' must be a string."))?
        .to_string();
    let catalog = st
        .catalog
        .get(st.upstream.as_ref())
        .await
        .map_err(ApiError::from_raptor)?;
    let resolved = resolve_model(&requested, &catalog, &st.aliases)?;
    body["model"] = Value::String(resolved);

    if endpoint == Endpoint::ChatCompletions {
        apply_hygiene(st, &mut body);
        apply_default_effort(st, &mut body).await;
        if let Some(ws) = headers.get(WORKSPACE_HEADER) {
            let ws = ws
                .to_str()
                .map_err(|_| ApiError::bad_request("x-raptor-workspace must be ASCII."))?;
            let ctx = st.context.as_deref().ok_or_else(|| {
                ApiError::new(
                    axum::http::StatusCode::SERVICE_UNAVAILABLE,
                    "server_error",
                    "Context injection was requested but no context service is configured.",
                )
                .with_code("context_unavailable")
            })?;
            inject_context(ctx, ws, &mut body).await?;
        }
    }

    let permit = st
        .scheduler
        .acquire_with(&principal.client_id, principal.limits.max_concurrent)
        .await
        .map_err(|r| r.into_api_error(st.retry_after_secs))?;
    match st
        .upstream
        .call(endpoint, body)
        .await
        .map_err(ApiError::from_raptor)?
    {
        Completion::Json(v) => Ok(Reply::Json(v)),
        Completion::Stream(s) => Ok(Reply::Stream(hold_permit(permit, s))),
    }
}

/// Agent hygiene for a chat request: drops reasoning from earlier assistant turns (when enabled)
/// and, once per gateway start, warns about a volatile token at the top of the system prompt.
/// The warning never changes the prompt.
fn apply_hygiene(st: &AppState, body: &mut Value) {
    if st.strip_history_reasoning {
        let n = crate::hygiene::strip_prior_reasoning(body);
        if n > 0 {
            tracing::debug!(
                messages = n,
                "removed reasoning from earlier assistant turns"
            );
        }
    }
    if let Some(kind) = crate::hygiene::volatile_prefix_token(body) {
        warn_volatile_once(&st.volatile_warned, kind);
    }
}

/// Adds the default thinking effort to a chat request that sets none, for a model whose chat
/// template supports it. The upstream is asked only when the request has no thinking fields of its
/// own, and a model without the capability leaves the request byte-identical.
async fn apply_default_effort(st: &AppState, body: &mut Value) {
    if crate::hygiene::controls_reasoning(body) {
        return;
    }
    let caps = st.upstream.thinking_caps().await;
    if let Some(level) = crate::thinking::map_effort(st.default_effort, &caps) {
        crate::hygiene::inject_reasoning_effort(body, &level);
    }
}

/// Logs the volatile-prompt warning the first time `warned` is clear. Returns whether it logged.
fn warn_volatile_once(warned: &std::sync::atomic::AtomicBool, kind: &str) -> bool {
    if warned.swap(true, std::sync::atomic::Ordering::Relaxed) {
        return false;
    }
    tracing::warn!(
        "the system prompt has a {kind} near its start; it changes on every request, which defeats prompt-prefix caching and makes each turn re-read the whole prompt. Remove it from the client's system prompt."
    );
    true
}

/// Maps a requested model name to an installed or loaded model id.
///
/// Order: exact installed/loaded id, then the explicit alias table, then the
/// loaded model for `gpt-*` and `claude-*` names. Anything else is 404.
pub fn resolve_model(
    requested: &str,
    catalog: &ModelCatalog,
    aliases: &BTreeMap<String, String>,
) -> Result<String, ApiError> {
    let known =
        |m: &str| catalog.installed.iter().any(|i| i == m) || catalog.loaded.as_deref() == Some(m);
    let not_found = |msg: String| ApiError::not_found(msg).with_code("model_not_found");

    if known(requested) {
        return Ok(requested.to_string());
    }
    if let Some(target) = aliases.get(requested) {
        return if known(target) {
            Ok(target.clone())
        } else {
            Err(not_found(format!(
                "Alias '{requested}' points to a model that is not installed."
            )))
        };
    }
    if LOADED_MODEL_PREFIXES
        .iter()
        .any(|p| requested.starts_with(p))
    {
        return match &catalog.loaded {
            Some(loaded) => Ok(loaded.clone()),
            None => Err(not_found(
                "No model is loaded to serve this request.".into(),
            )),
        };
    }
    Err(not_found(format!(
        "The model '{requested}' does not exist."
    )))
}

/// Keeps `permit` alive for as long as the stream is being consumed.
pub(crate) fn hold_permit(permit: Permit, mut inner: SseStream) -> SseStream {
    Box::pin(async_stream::stream! {
        let _permit = permit;
        while let Some(item) = inner.next().await {
            yield item;
        }
    })
}

/// Frames an upstream payload stream as OpenAI SSE. Stops at the upstream's
/// `[DONE]` or at the first error (which is sent as an error event), then
/// always sends exactly one `[DONE]`.
pub(crate) fn openai_frames(
    mut inner: SseStream,
) -> impl Stream<Item = Result<Bytes, Infallible>> + Send {
    async_stream::stream! {
        while let Some(item) = inner.next().await {
            match item {
                Ok(payload) if payload == "[DONE]" => break,
                Ok(payload) => yield Ok(Bytes::from(sse_data_frame(&payload))),
                Err(err) => {
                    let body = ApiError::from_raptor(err).body().to_string();
                    yield Ok(Bytes::from(sse_data_frame(&body)));
                    break;
                }
            }
        }
        yield Ok(Bytes::from_static(b"data: [DONE]\n\n"));
    }
}

/// One `data:` event. Multi-line payloads get one `data:` line per line, as SSE requires.
pub(crate) fn sse_data_frame(payload: &str) -> String {
    let mut out = String::with_capacity(payload.len() + 8);
    for line in payload.split('\n') {
        out.push_str("data: ");
        out.push_str(line);
        out.push('\n');
    }
    out.push('\n');
    out
}

/// One named event (`event:` plus `data:`), used by the Anthropic adapter.
pub(crate) fn sse_event_frame(event: &str, data: &str) -> String {
    format!("event: {event}\n{}", sse_data_frame(data))
}

/// Wraps a byte stream in an SSE response with the headers streaming clients expect.
pub(crate) fn sse_response<S>(stream: S) -> Response
where
    S: Stream<Item = Result<Bytes, Infallible>> + Send + 'static,
{
    let mut response = Response::new(Body::from_stream(stream));
    let h = response.headers_mut();
    h.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/event-stream"),
    );
    h.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    h.insert("x-accel-buffering", HeaderValue::from_static("no"));
    response
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cat(installed: &[&str], loaded: Option<&str>) -> ModelCatalog {
        ModelCatalog {
            installed: installed.iter().map(|s| s.to_string()).collect(),
            loaded: loaded.map(str::to_string),
        }
    }

    #[test]
    fn exact_and_loaded_names_resolve_to_themselves() {
        let c = cat(&["qwen-7b", "llama-8b"], Some("qwen-7b"));
        let none = BTreeMap::new();
        assert_eq!(resolve_model("qwen-7b", &c, &none).unwrap(), "qwen-7b");
        assert_eq!(resolve_model("llama-8b", &c, &none).unwrap(), "llama-8b");
    }

    #[test]
    fn gpt_and_claude_names_map_to_loaded_model() {
        let c = cat(&["qwen-7b"], Some("qwen-7b"));
        let none = BTreeMap::new();
        assert_eq!(resolve_model("gpt-4o", &c, &none).unwrap(), "qwen-7b");
        assert_eq!(
            resolve_model("claude-sonnet-4-5", &c, &none).unwrap(),
            "qwen-7b"
        );
    }

    #[test]
    fn gpt_name_without_loaded_model_is_404() {
        let c = cat(&["qwen-7b"], None);
        let err = resolve_model("gpt-4", &c, &BTreeMap::new()).unwrap_err();
        assert_eq!(err.status, axum::http::StatusCode::NOT_FOUND);
        assert_eq!(err.code, Some("model_not_found"));
    }

    #[test]
    fn explicit_alias_maps_to_installed_model() {
        let c = cat(&["qwen-7b", "llama-8b"], None);
        let mut aliases = BTreeMap::new();
        aliases.insert("fast".to_string(), "llama-8b".to_string());
        assert_eq!(resolve_model("fast", &c, &aliases).unwrap(), "llama-8b");
        aliases.insert("broken".to_string(), "missing".to_string());
        assert!(resolve_model("broken", &c, &aliases).is_err());
    }

    #[test]
    fn unknown_model_is_404() {
        let c = cat(&["qwen-7b"], Some("qwen-7b"));
        assert!(resolve_model("mistral-x", &c, &BTreeMap::new()).is_err());
    }

    #[test]
    fn the_volatile_prompt_warning_is_logged_once_per_session() {
        let warned = std::sync::atomic::AtomicBool::new(false);
        assert!(warn_volatile_once(&warned, "time of day"));
        assert!(!warn_volatile_once(&warned, "UUID"));
        assert!(!warn_volatile_once(&warned, "time of day"));
    }

    #[test]
    fn data_frames_handle_multiline_payloads() {
        assert_eq!(sse_data_frame("{\"a\":1}"), "data: {\"a\":1}\n\n");
        assert_eq!(sse_data_frame("a\nb"), "data: a\ndata: b\n\n");
    }

    #[test]
    fn parse_body_checks_object_and_fields() {
        assert!(parse_body(b"[1]", &[]).is_err());
        assert!(parse_body(b"{not json", &[]).is_err());
        assert!(parse_body(b"{\"model\":\"m\"}", &["model", "messages"]).is_err());
        assert!(parse_body(b"{\"model\":\"m\",\"messages\":[]}", &["model", "messages"]).is_ok());
    }
}
