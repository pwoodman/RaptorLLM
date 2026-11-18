//! Streaming chat through the same upstream the gateway uses. Each request gets its own
//! event channel, `chat://<requestId>`, carrying `chunk`, `done` or `error` payloads.

use crate::adapters::RuntimeUpstream;
use crate::commands::chat_event_name;
use crate::state::AppState;
use futures_util::StreamExt;
use raptor_core::{ChatMessage, RaptorError, Result, Role};
use raptor_gateway::{Completion, Endpoint};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter, State};
use tokio_util::sync::CancellationToken;

/// Upper bound on one chat request's total message text.
const MAX_CHAT_BYTES: usize = 8 * 1024 * 1024;
const MAX_REQUEST_ID: usize = 64;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ChatOptions {
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    /// Name of a preset saved for the loaded model. Its system prompt, sampling values and
    /// length limit apply; explicit `temperature` and `max_tokens` above still win over it.
    pub preset: Option<String>,
    /// Sampling values. Set by a preset unless given here.
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub repeat_penalty: Option<f32>,
    /// How hard a thinking model thinks for this reply. `None` uses the setting. Ignored (and
    /// nothing is sent) when the loaded model has no effort control.
    pub thinking: Option<raptor_gateway::Effort>,
    /// Write the answer as parallel sections (see `crate::sections`): a plan first, then every
    /// section at once. Faster for long answers when the server has several slots; thinking is
    /// off for these requests. Falls back to one normal reply when no usable plan comes back.
    pub long_answer: bool,
}

/// Payload on `chat://<requestId>`.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatEvent {
    /// A text fragment of the answer, in order.
    Chunk { text: String },
    /// A fragment of the model's reasoning (`reasoning_content`, thinking models). It is not
    /// part of the answer: the UI shows it separately and never counts it as the reply.
    Reasoning { text: String },
    /// The stream ended. `cancelled` is true when the user stopped it. `stats` is the speed of
    /// this reply, or null when the server reported no timings (then nothing is shown).
    Done {
        cancelled: bool,
        stats: Option<ChatStats>,
    },
    /// The stream failed. The partial text already sent stays on screen. `error` is the
    /// serialized `RaptorError` (`{"kind": ..., "message": ...}`).
    Error { error: Value },
}

/// Speed of one reply, from the gateway's `raptor` object (see `adapters::attach_speed`).
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct ChatStats {
    pub tokens_per_second: Option<f64>,
    pub prompt_tokens_per_second: Option<f64>,
    pub time_to_first_token_ms: Option<f64>,
    pub generated_tokens: Option<u64>,
}

impl ChatStats {
    /// Reads the `raptor` object of a reply or last stream chunk. None when absent or empty.
    fn from_reply(v: &Value) -> Option<Self> {
        let r = v.get("raptor")?;
        let f = |k: &str| r.get(k).and_then(Value::as_f64);
        let stats = Self {
            tokens_per_second: f("tokens_per_second"),
            prompt_tokens_per_second: f("prompt_tokens_per_second"),
            time_to_first_token_ms: f("time_to_first_token_ms"),
            generated_tokens: r.get("generated_tokens").and_then(Value::as_u64),
        };
        (stats != Self::default()).then_some(stats)
    }
}

/// Serializes an error for the event payload. `RaptorError` is not `Clone`, which the
/// event API requires, so it is sent as JSON.
fn error_json(e: &RaptorError) -> Value {
    serde_json::to_value(e)
        .unwrap_or_else(|_| json!({ "kind": "Internal", "message": e.to_string() }))
}

fn validate_request_id(id: &str) -> Result<()> {
    let ok = !id.is_empty()
        && id.len() <= MAX_REQUEST_ID
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_');
    if ok {
        Ok(())
    } else {
        Err(RaptorError::InvalidInput(
            "request id must be 1-64 letters, digits, '-' or '_'".into(),
        ))
    }
}

/// Starts a streaming completion against the loaded model. Returns once the stream has
/// started; text arrives on the `chat://<requestId>` event.
#[tauri::command]
pub async fn chat_start(
    app: AppHandle,
    state: State<'_, AppState>,
    request_id: String,
    messages: Vec<ChatMessage>,
    options: Option<ChatOptions>,
) -> Result<()> {
    let id = request_id.clone();
    begin_chat(
        &state,
        request_id,
        messages,
        options.unwrap_or_default(),
        move |event| {
            let _ = app.emit(&chat_event_name(&id), event);
        },
    )
    .await
}

/// Starts a stream and returns once it is registered. Every event (chunks, then exactly one
/// `Done` or `Error`) goes to `sink`. Used by the command and by the end-to-end gate test,
/// so both exercise the same code.
pub async fn begin_chat<S>(
    state: &AppState,
    request_id: String,
    messages: Vec<ChatMessage>,
    options: ChatOptions,
    sink: S,
) -> Result<()>
where
    S: Fn(ChatEvent) + Send + Sync + 'static,
{
    validate_request_id(&request_id)?;
    if messages.is_empty() {
        return Err(RaptorError::InvalidInput("no messages".into()));
    }
    let total: usize = messages.iter().map(|m| m.content.len()).sum();
    if total > MAX_CHAT_BYTES {
        return Err(RaptorError::InvalidInput(format!(
            "messages exceed {MAX_CHAT_BYTES} bytes"
        )));
    }
    let cancel = CancellationToken::new();
    {
        let mut chats = state.chats.lock().unwrap_or_else(|p| p.into_inner());
        if chats.contains_key(&request_id) {
            return Err(RaptorError::Conflict(format!(
                "request {request_id} is already running"
            )));
        }
        chats.insert(request_id.clone(), cancel.clone());
    }

    let long_answer = options.long_answer;
    let prepared = match resolve_preset(state, options.preset.as_deref()) {
        Ok(preset) => {
            let mut messages = messages;
            let mut options = options;
            if let Some(p) = preset.as_ref() {
                apply_preset_to_request(p, &mut messages, &mut options);
            }
            let chosen = options.thinking;
            let default_effort = state.settings.snapshot().thinking_effort;
            match prepare_body(&state.upstream, &messages, options).await {
                Ok(mut body) => {
                    if !long_answer {
                        apply_thinking(state.upstream.as_ref(), &mut body, chosen, default_effort)
                            .await;
                    }
                    Ok((body, messages))
                }
                Err(e) => Err(e),
            }
        }
        Err(e) => Err(e),
    };
    let (body, messages) = match prepared {
        Ok(b) => b,
        Err(e) => {
            forget(&state.chats, &request_id);
            return Err(e);
        }
    };

    let upstream = state.upstream.clone();
    let chats = state.chats.clone();
    let id = request_id.clone();
    tauri::async_runtime::spawn(async move {
        let event_sink = Arc::new(sink);
        let chunk_sink = event_sink.clone();
        let on_piece = move |piece| {
            chunk_sink(match piece {
                Piece::Text(text) => ChatEvent::Chunk { text },
                Piece::Reasoning(text) => ChatEvent::Reasoning { text },
            });
        };
        let result = if long_answer {
            run_sections(upstream, body, messages, cancel, on_piece).await
        } else {
            run_stream(upstream, body, cancel, on_piece).await
        };
        let payload = match result {
            Ok((cancelled, stats)) => ChatEvent::Done { cancelled, stats },
            Err(error) => ChatEvent::Error {
                error: error_json(&error),
            },
        };
        event_sink(payload);
        forget(&chats, &id);
    });
    Ok(())
}

/// The named preset for the loaded model, or `None` when no preset was asked for.
fn resolve_preset(state: &AppState, name: Option<&str>) -> Result<Option<raptor_models::Preset>> {
    let Some(name) = name else {
        return Ok(None);
    };
    let model = state.current_model_id().ok_or_else(|| {
        RaptorError::Conflict("no model is loaded; load one before using a preset".into())
    })?;
    state.models.presets().get(&model, name).map(Some)
}

/// Applies a preset to one request: its system prompt (unless the conversation already starts
/// with one), its sampling values, and its length limit. Explicit options win.
pub fn apply_preset_to_request(
    preset: &raptor_models::Preset,
    messages: &mut Vec<ChatMessage>,
    options: &mut ChatOptions,
) {
    if let Some(prompt) = &preset.system_prompt {
        let has_system = messages.first().is_some_and(|m| m.role == Role::System);
        if !has_system && !prompt.is_empty() {
            messages.insert(0, ChatMessage::new(Role::System, prompt.clone()));
        }
    }
    if options.temperature.is_none() {
        options.temperature = Some(preset.temperature);
    }
    if options.max_tokens.is_none() {
        options.max_tokens = preset.max_tokens;
    }
    options.top_p.get_or_insert(preset.top_p);
    options.top_k.get_or_insert(preset.top_k);
    options.repeat_penalty.get_or_insert(preset.repeat_penalty);
}

/// Stops a running chat stream. Returns false when no such request is running.
#[tauri::command]
pub async fn chat_cancel(state: State<'_, AppState>, request_id: String) -> Result<bool> {
    let token = state
        .chats
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .get(&request_id)
        .cloned();
    match token {
        Some(t) => {
            t.cancel();
            Ok(true)
        }
        None => Ok(false),
    }
}

fn forget(chats: &Mutex<HashMap<String, CancellationToken>>, request_id: &str) {
    chats
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .remove(request_id);
}

/// Builds the upstream request body. Refused when no model is loaded.
async fn prepare_body(
    upstream: &Arc<RuntimeUpstream>,
    messages: &[ChatMessage],
    options: ChatOptions,
) -> Result<Value> {
    use raptor_gateway::Upstream;
    let catalog = upstream.models().await?;
    let model = catalog.loaded.ok_or_else(|| {
        RaptorError::Conflict("no model is loaded; load one from the Models page".into())
    })?;
    let mut body = json!({
        "model": model,
        "messages": serde_json::to_value(messages)?,
        "stream": true,
    });
    if let Some(t) = options.temperature {
        if !(0.0..=2.0).contains(&t) {
            return Err(RaptorError::InvalidInput("temperature must be 0-2".into()));
        }
        body["temperature"] = json!(t);
    }
    if let Some(m) = options.max_tokens {
        if m == 0 {
            return Err(RaptorError::InvalidInput(
                "max_tokens must be positive".into(),
            ));
        }
        body["max_tokens"] = json!(m);
    }
    if let Some(p) = options.top_p {
        if !(p.is_finite() && p > 0.0 && p <= 1.0) {
            return Err(RaptorError::InvalidInput("top_p must be in (0, 1]".into()));
        }
        body["top_p"] = json!(p);
    }
    if let Some(k) = options.top_k {
        if k > 1000 {
            return Err(RaptorError::InvalidInput(
                "top_k must be at most 1000".into(),
            ));
        }
        body["top_k"] = json!(k);
    }
    if let Some(r) = options.repeat_penalty {
        if !(r.is_finite() && (0.5..=2.0).contains(&r)) {
            return Err(RaptorError::InvalidInput(
                "repeat_penalty must be between 0.5 and 2".into(),
            ));
        }
        body["repeat_penalty"] = json!(r);
    }
    Ok(body)
}

/// Adds `reasoning_effort` to the request body when the loaded model has an effort control: the
/// conversation's choice, else the setting, mapped to the model's own level. A model without one
/// gets no reasoning field at all, so its request is exactly what it was.
async fn apply_thinking(
    upstream: &dyn raptor_gateway::Upstream,
    body: &mut Value,
    chosen: Option<raptor_gateway::Effort>,
    default_effort: raptor_gateway::Effort,
) {
    let caps = upstream.thinking_caps().await;
    if let Some(level) = crate::thinking::level_for_request(&caps, chosen, default_effort) {
        body["reasoning_effort"] = json!(level);
    }
}

/// One decoded piece of an upstream reply. Reasoning and answer text stay apart.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Piece {
    Text(String),
    Reasoning(String),
}

/// Pulls the non-empty `reasoning_content` and `content` strings out of one choice object
/// (a stream delta or a complete message). Reasoning comes first, as the model emits it.
fn pieces_of(choice: &Value) -> Vec<Piece> {
    let reasoning = choice["reasoning_content"].as_str().unwrap_or("");
    let text = choice["content"].as_str().unwrap_or("");
    let mut out = Vec::new();
    if !reasoning.is_empty() {
        out.push(Piece::Reasoning(reasoning.to_string()));
    }
    if !text.is_empty() {
        out.push(Piece::Text(text.to_string()));
    }
    out
}

/// Pumps the upstream stream into `on_piece` until it ends, fails or is cancelled.
/// Returns `(cancelled, stats)`. Dropping the stream cancels the upstream request.
async fn run_stream<F>(
    upstream: Arc<RuntimeUpstream>,
    body: Value,
    cancel: CancellationToken,
    mut on_piece: F,
) -> Result<(bool, Option<ChatStats>)>
where
    F: FnMut(Piece) + Send,
{
    use raptor_gateway::Upstream;
    let mut stats: Option<ChatStats> = None;
    let completion = tokio::select! {
        _ = cancel.cancelled() => return Ok((true, None)),
        r = upstream.call(Endpoint::ChatCompletions, body) => r?,
    };
    match completion {
        Completion::Json(v) => {
            for piece in pieces_of(&v["choices"][0]["message"]) {
                on_piece(piece);
            }
            Ok((false, ChatStats::from_reply(&v)))
        }
        Completion::Stream(mut stream) => loop {
            let next = tokio::select! {
                _ = cancel.cancelled() => return Ok((true, stats)),
                n = stream.next() => n,
            };
            match next {
                None => return Ok((false, stats)),
                Some(Err(e)) => return Err(e),
                Some(Ok(payload)) => {
                    if payload.trim() == "[DONE]" {
                        return Ok((false, stats));
                    }
                    let chunk: Value = serde_json::from_str(&payload).map_err(|e| {
                        RaptorError::Runtime(format!("malformed stream chunk: {e}"))
                    })?;
                    for piece in pieces_of(&chunk["choices"][0]["delta"]) {
                        on_piece(piece);
                    }
                    if let Some(found) = ChatStats::from_reply(&chunk) {
                        stats = Some(found);
                    }
                }
            }
        },
    }
}

/// The body of one planning or section request: the prepared `body` with other messages, a
/// token limit no larger than the user's, streaming on and thinking off (these requests are for
/// speed, and a section's reasoning would not be shown).
fn section_body(body: &Value, messages: &[ChatMessage], max_tokens: u32) -> Result<Value> {
    let mut b = body.clone();
    b["messages"] = serde_json::to_value(messages)?;
    b["stream"] = json!(true);
    let limit = b["max_tokens"]
        .as_u64()
        .map_or(max_tokens, |m| m.min(u64::from(max_tokens)) as u32);
    b["max_tokens"] = json!(limit);
    if let Some(o) = b.as_object_mut() {
        o.remove("reasoning_effort");
    }
    b["chat_template_kwargs"] = json!({ "enable_thinking": false });
    Ok(b)
}

/// Writes a long answer as parallel sections (see `crate::sections`): one planning request,
/// then every section at once, shown in order through `on_piece`. The stats are the whole
/// answer's wall-clock speed (all tokens, plan included, over the time from the request to the
/// last section). Without a usable plan it answers with one normal reply instead.
async fn run_sections<F>(
    upstream: Arc<RuntimeUpstream>,
    body: Value,
    messages: Vec<ChatMessage>,
    cancel: CancellationToken,
    mut on_piece: F,
) -> Result<(bool, Option<ChatStats>)>
where
    F: FnMut(Piece) + Send,
{
    use crate::sections;
    let started = std::time::Instant::now();
    let Some(last_user) = messages.iter().rposition(|m| m.role == Role::User) else {
        return run_stream(upstream, body, cancel, on_piece).await;
    };
    let mut plan_messages = messages.clone();
    plan_messages[last_user]
        .content
        .push_str(sections::PLAN_INSTRUCTION);
    let mut plan_text = String::new();
    let (cancelled, plan_stats) = run_stream(
        upstream.clone(),
        section_body(&body, &plan_messages, sections::PLAN_MAX_TOKENS)?,
        cancel.clone(),
        |p| {
            if let Piece::Text(t) = p {
                plan_text.push_str(&t);
            }
        },
    )
    .await?;
    if cancelled {
        return Ok((true, None));
    }
    let Some(plan) = sections::parse_plan(&plan_text) else {
        return run_stream(upstream, body, cancel, on_piece).await;
    };

    let n = plan.points.len();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<(usize, Option<String>)>();
    let mut tasks = Vec::with_capacity(n);
    for i in 0..n {
        let mut msgs = messages.clone();
        msgs.push(ChatMessage::new(Role::Assistant, plan.as_text()));
        msgs.push(ChatMessage::new(Role::User, plan.section_prompt(i)));
        let section = section_body(&body, &msgs, sections::SECTION_MAX_TOKENS)?;
        let (done_tx, up, stop) = (tx.clone(), upstream.clone(), cancel.clone());
        let text_tx = tx.clone();
        tasks.push(tauri::async_runtime::spawn(async move {
            let r = run_stream(up, section, stop, move |p| {
                if let Piece::Text(t) = p {
                    let _ = text_tx.send((i, Some(t)));
                }
            })
            .await;
            let _ = done_tx.send((i, None));
            r
        }));
    }
    drop(tx);

    let mut order = sections::InOrder::new(n);
    let mut first_text_ms = None;
    // Every task sends its end marker, so the channel closes once all sections are done.
    while let Some((i, msg)) = rx.recv().await {
        let ready = match msg {
            Some(text) => order.push(i, &text),
            None => order.finish(i),
        };
        if let Some(text) = ready {
            first_text_ms.get_or_insert_with(|| started.elapsed().as_secs_f64() * 1000.0);
            on_piece(Piece::Text(text));
        }
    }

    let mut generated = plan_stats.and_then(|s| s.generated_tokens).unwrap_or(0);
    let mut any_cancelled = false;
    let mut first_error = None;
    for t in tasks {
        match t.await {
            Ok(Ok((c, s))) => {
                any_cancelled |= c;
                generated += s.and_then(|s| s.generated_tokens).unwrap_or(0);
            }
            Ok(Err(e)) => {
                first_error.get_or_insert(e);
            }
            Err(e) => {
                first_error
                    .get_or_insert(RaptorError::Internal(format!("a section task failed: {e}")));
            }
        }
    }
    if let Some(e) = first_error {
        return Err(e);
    }
    let seconds = started.elapsed().as_secs_f64();
    let stats = (generated > 0).then(|| ChatStats {
        tokens_per_second: (seconds > 0.0).then(|| generated as f64 / seconds),
        prompt_tokens_per_second: None,
        time_to_first_token_ms: first_text_ms,
        generated_tokens: Some(generated),
    });
    Ok((any_cancelled || cancel.is_cancelled(), stats))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_come_only_from_the_raptor_object_and_never_from_nothing() {
        let reply = json!({"raptor": {"tokens_per_second": 21.3, "time_to_first_token_ms": 410.0, "generated_tokens": 72}});
        let stats = ChatStats::from_reply(&reply).unwrap();
        assert_eq!(stats.tokens_per_second, Some(21.3));
        assert_eq!(stats.time_to_first_token_ms, Some(410.0));
        assert_eq!(stats.generated_tokens, Some(72));
        assert!(ChatStats::from_reply(&json!({"choices": []})).is_none());
        assert!(ChatStats::from_reply(&json!({"raptor": {}})).is_none());
    }

    use crate::adapters::fakes::{counting_connect, FakeLibrary, FakeServer, FakeUpstream};
    use crate::adapters::{InstalledModels, RuntimeUpstream, ServerProbe};
    use raptor_core::Role;
    use std::sync::Mutex;

    fn build(fake: FakeUpstream) -> Arc<RuntimeUpstream> {
        let server = Arc::new(FakeServer(Mutex::new(Some("http://127.0.0.1:8081".into()))));
        let count = Arc::new(Mutex::new(0usize));
        let probe: Arc<dyn ServerProbe> = server;
        let lib: Arc<dyn InstalledModels> = Arc::new(FakeLibrary(vec![]));
        Arc::new(RuntimeUpstream::new(
            probe,
            lib,
            counting_connect(Arc::new(fake), count),
        ))
    }

    /// Like `build`, keeping a handle on the fake so the request bodies can be read.
    fn build_shared(fake: Arc<FakeUpstream>) -> Arc<RuntimeUpstream> {
        let server = Arc::new(FakeServer(Mutex::new(Some("http://127.0.0.1:8081".into()))));
        let probe: Arc<dyn ServerProbe> = server;
        let lib: Arc<dyn InstalledModels> = Arc::new(FakeLibrary(vec![]));
        Arc::new(RuntimeUpstream::new(
            probe,
            lib,
            counting_connect(fake, Arc::new(Mutex::new(0usize))),
        ))
    }

    fn deltas(texts: &[&str]) -> Vec<Result<String>> {
        let mut items: Vec<Result<String>> = texts
            .iter()
            .map(|t| Ok(json!({"choices": [{"delta": {"content": t}}]}).to_string()))
            .collect();
        items.push(Ok("[DONE]".to_string()));
        items
    }

    #[test]
    fn a_section_request_streams_with_thinking_off_and_never_exceeds_the_users_limit() {
        let body = json!({"model": "m", "max_tokens": 100, "reasoning_effort": "high", "temperature": 0.2});
        let b = section_body(&body, &msg("q"), 450).unwrap();
        assert_eq!(b["max_tokens"], 100, "the user's smaller limit wins");
        assert_eq!(b["stream"], true);
        assert_eq!(b["temperature"], 0.2);
        assert!(b.get("reasoning_effort").is_none());
        assert_eq!(b["chat_template_kwargs"]["enable_thinking"], false);
        assert_eq!(b["messages"][0]["content"], "q");
        let unlimited = section_body(&json!({"model": "m"}), &msg("q"), 170).unwrap();
        assert_eq!(unlimited["max_tokens"], 170);
    }

    #[tokio::test]
    async fn a_long_answer_plans_then_writes_each_section_and_shows_them_in_order() {
        let fake = Arc::new(FakeUpstream {
            loaded: Some("m.gguf".into()),
            // The first streamed call (the plan) gets this; every later one gets "hi".
            streamed: Mutex::new(Some(deltas(&["Facts: Mara.\n1. alpha\n", "2. beta"]))),
            ..Default::default()
        });
        let up = build_shared(fake.clone());
        let mut got = String::new();
        let (cancelled, stats) = run_sections(
            up,
            json!({"model": "m.gguf", "stream": true}),
            msg("Write a story"),
            CancellationToken::new(),
            |p| {
                if let Piece::Text(t) = p {
                    got.push_str(&t);
                }
            },
        )
        .await
        .unwrap();
        assert!(!cancelled);
        assert_eq!(
            got, "hi\n\nhi",
            "two sections, in order, with a blank line between"
        );
        assert!(
            stats.is_none(),
            "the fake reports no token counts, so no speed is invented"
        );
        let bodies = fake.bodies.lock().unwrap();
        assert_eq!(bodies.len(), 3, "one plan and two sections");
        let plan_q = bodies[0]["messages"][0]["content"].as_str().unwrap();
        assert!(plan_q.starts_with("Write a story") && plan_q.ends_with("No other text."));
        assert_eq!(bodies[0]["max_tokens"], crate::sections::PLAN_MAX_TOKENS);
        // Parallel requests can reach the fake in either order. Check both section requests
        // independently of scheduling; the assembled response is asserted above.
        let mut section_bodies: Vec<_> = bodies[1..].iter().collect();
        section_bodies
            .sort_by_key(|body| body["messages"][2]["content"].as_str().unwrap().to_owned());
        for (i, b) in section_bodies.iter().enumerate() {
            let m = b["messages"].as_array().unwrap();
            assert_eq!(m.len(), 3);
            assert_eq!(
                m[0]["content"], "Write a story",
                "the user's question is unchanged"
            );
            assert!(m[1]["content"].as_str().unwrap().contains("Facts: Mara."));
            let ask = m[2]["content"].as_str().unwrap();
            assert!(
                ask.starts_with(&format!("Write section {} of 2", i + 1)),
                "{ask}"
            );
            assert_eq!(b["max_tokens"], crate::sections::SECTION_MAX_TOKENS);
        }
    }

    #[tokio::test]
    async fn without_a_usable_plan_a_long_answer_is_one_normal_reply() {
        let fake = Arc::new(FakeUpstream {
            loaded: Some("m.gguf".into()),
            streamed: Mutex::new(Some(deltas(&["Sure, here is a short answer."]))),
            ..Default::default()
        });
        let up = build_shared(fake.clone());
        let mut got = String::new();
        let original = json!({"model": "m.gguf", "stream": true, "reasoning_effort": "low"});
        run_sections(
            up,
            original.clone(),
            msg("hi"),
            CancellationToken::new(),
            |p| {
                if let Piece::Text(t) = p {
                    got.push_str(&t);
                }
            },
        )
        .await
        .unwrap();
        assert_eq!(
            got, "hi",
            "the normal reply is what is shown, not the failed plan"
        );
        let bodies = fake.bodies.lock().unwrap();
        assert_eq!(bodies.len(), 2);
        assert_eq!(
            bodies[1], original,
            "the fallback sends the request as prepared"
        );
    }

    fn upstream_with(stream_items: Vec<Result<String>>, loaded: bool) -> Arc<RuntimeUpstream> {
        build(FakeUpstream {
            loaded: loaded.then(|| "m.gguf".to_string()),
            streamed: Mutex::new(Some(stream_items)),
            ..Default::default()
        })
    }

    fn msg(text: &str) -> Vec<ChatMessage> {
        vec![ChatMessage::new(Role::User, text)]
    }

    #[test]
    fn request_ids_are_restricted() {
        assert!(validate_request_id("req-1_A").is_ok());
        for bad in ["", "a/b", "../x", "x y", &"a".repeat(65)] {
            assert!(validate_request_id(bad).is_err(), "{bad}");
        }
    }

    #[tokio::test]
    async fn body_requires_a_loaded_model_and_validates_options() {
        let no_model = upstream_with(vec![], false);
        let err = prepare_body(&no_model, &msg("hi"), ChatOptions::default()).await;
        assert!(matches!(err, Err(RaptorError::Conflict(_))));

        let up = upstream_with(vec![], true);
        let bad_t = prepare_body(
            &up,
            &msg("hi"),
            ChatOptions {
                temperature: Some(9.0),
                max_tokens: None,
                ..Default::default()
            },
        )
        .await;
        assert!(matches!(bad_t, Err(RaptorError::InvalidInput(_))));
        let ok = prepare_body(
            &up,
            &msg("hi"),
            ChatOptions {
                temperature: Some(0.5),
                max_tokens: Some(64),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(ok["model"], "m.gguf");
        assert_eq!(ok["stream"], true);
        assert_eq!(ok["max_tokens"], 64);
    }

    #[tokio::test]
    async fn stream_text_arrives_in_order_and_done_is_reported() {
        let items = vec![
            Ok("{\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}".to_string()),
            Ok("{\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}".to_string()),
            Ok("[DONE]".to_string()),
        ];
        let up = upstream_with(items, true);
        let mut got = String::new();
        let (cancelled, _) =
            run_stream(up, json!({"stream": true}), CancellationToken::new(), |p| {
                if let Piece::Text(t) = p {
                    got.push_str(&t)
                }
            })
            .await
            .unwrap();
        assert!(!cancelled);
        assert_eq!(got, "Hello");
    }

    /// A thinking model streams `reasoning_content` deltas before its answer. The two must
    /// arrive as separate pieces, in order, so the answer never contains the reasoning.
    #[tokio::test]
    async fn reasoning_deltas_are_separate_from_the_answer() {
        let items = vec![
            Ok(r#"{"choices":[{"delta":{"reasoning_content":"Let me "}}]}"#.to_string()),
            Ok(r#"{"choices":[{"delta":{"reasoning_content":"think."}}]}"#.to_string()),
            Ok(r#"{"choices":[{"delta":{"content":"Answer"}}]}"#.to_string()),
            Ok(r#"{"choices":[{"delta":{"content":"!"}}]}"#.to_string()),
            Ok("[DONE]".to_string()),
        ];
        let up = upstream_with(items, true);
        let mut pieces = Vec::new();
        let (cancelled, _) =
            run_stream(up, json!({"stream": true}), CancellationToken::new(), |p| {
                pieces.push(p)
            })
            .await
            .unwrap();
        assert!(!cancelled);
        assert_eq!(
            pieces,
            vec![
                Piece::Reasoning("Let me ".into()),
                Piece::Reasoning("think.".into()),
                Piece::Text("Answer".into()),
                Piece::Text("!".into()),
            ]
        );
    }

    /// A delta can carry reasoning and text together; reasoning is emitted first.
    #[test]
    fn a_single_delta_yields_reasoning_before_text_and_skips_empty_fields() {
        let both = json!({"reasoning_content": "r", "content": "t"});
        assert_eq!(
            pieces_of(&both),
            vec![Piece::Reasoning("r".into()), Piece::Text("t".into())]
        );
        let empty = json!({"reasoning_content": "", "content": ""});
        assert!(pieces_of(&empty).is_empty());
        assert!(pieces_of(&json!({"role": "assistant"})).is_empty());
    }

    #[tokio::test]
    async fn cancel_stops_the_stream_and_reports_cancelled() {
        // The fake's stream never ends, so only cancellation can finish this call.
        let up = build(FakeUpstream {
            loaded: Some("m".into()),
            pending: true,
            ..Default::default()
        });
        let cancel = CancellationToken::new();
        let c2 = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            c2.cancel();
        });
        let (cancelled, _) = run_stream(up, json!({"stream": true}), cancel, |_: Piece| {})
            .await
            .unwrap();
        assert!(cancelled);
    }

    #[tokio::test]
    async fn upstream_error_mid_stream_is_returned_not_swallowed() {
        let items = vec![
            Ok("{\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}".to_string()),
            Err(RaptorError::Network("connection reset".into())),
        ];
        let up = upstream_with(items, true);
        let mut got = String::new();
        let res = run_stream(up, json!({"stream": true}), CancellationToken::new(), |p| {
            if let Piece::Text(t) = p {
                got.push_str(&t)
            }
        })
        .await;
        assert!(matches!(res, Err(RaptorError::Network(_))));
        assert_eq!(got, "partial");
    }

    fn thinking_upstream(levels: &[&str]) -> Arc<RuntimeUpstream> {
        build(FakeUpstream {
            loaded: Some("m.gguf".into()),
            caps: raptor_gateway::ThinkingCaps {
                supported: !levels.is_empty(),
                levels: levels.iter().map(|l| l.to_string()).collect(),
            },
            ..Default::default()
        })
    }

    #[tokio::test]
    async fn a_thinking_model_gets_the_conversations_level_mapped_to_its_own() {
        use raptor_gateway::Effort;
        let up = thinking_upstream(&["low", "medium", "xhigh"]);
        for (chosen, expect) in [
            (Some(Effort::Low), "low"),
            (Some(Effort::Medium), "medium"),
            (Some(Effort::High), "xhigh"),
            (Some(Effort::Max), "xhigh"),
            // No choice on this conversation: the setting applies.
            (None, "medium"),
        ] {
            let mut body = prepare_body(&up, &msg("hi"), ChatOptions::default())
                .await
                .unwrap();
            apply_thinking(up.as_ref(), &mut body, chosen, Effort::Medium).await;
            assert_eq!(body["reasoning_effort"], expect, "{chosen:?}");
        }
    }

    #[tokio::test]
    async fn without_a_choice_the_setting_decides() {
        use raptor_gateway::Effort;
        let up = thinking_upstream(&["low", "medium", "xhigh"]);
        let mut body = prepare_body(&up, &msg("hi"), ChatOptions::default())
            .await
            .unwrap();
        apply_thinking(up.as_ref(), &mut body, None, Effort::Low).await;
        assert_eq!(body["reasoning_effort"], "low");
    }

    #[tokio::test]
    async fn a_model_without_effort_control_gets_an_unchanged_request() {
        use raptor_gateway::Effort;
        let up = thinking_upstream(&[]);
        let plain = prepare_body(&up, &msg("hi"), ChatOptions::default())
            .await
            .unwrap();
        let mut body = plain.clone();
        apply_thinking(up.as_ref(), &mut body, Some(Effort::Max), Effort::High).await;
        assert_eq!(body, plain);
        assert!(body.get("reasoning_effort").is_none());
    }

    #[test]
    fn the_chat_option_reads_a_lowercase_level_and_defaults_to_none() {
        let o: ChatOptions = serde_json::from_str(r#"{"thinking": "high"}"#).unwrap();
        assert_eq!(o.thinking, Some(raptor_gateway::Effort::High));
        let none: ChatOptions = serde_json::from_str(r#"{"temperature": 0.5}"#).unwrap();
        assert_eq!(none.thinking, None);
        let null: ChatOptions = serde_json::from_str(r#"{"thinking": null}"#).unwrap();
        assert_eq!(null.thinking, None);
    }
}

#[cfg(test)]
mod preset_tests {
    use super::*;
    use raptor_models::Preset;

    fn preset() -> Preset {
        let mut p = Preset::with_defaults("org/repo/m.gguf", "Review");
        p.system_prompt = Some("You review diffs.".into());
        p.temperature = 0.2;
        p.top_p = 0.9;
        p.top_k = 20;
        p.repeat_penalty = 1.2;
        p.max_tokens = Some(256);
        p
    }

    #[test]
    fn a_preset_adds_its_system_prompt_and_sampling_unless_options_say_otherwise() {
        let mut messages = vec![ChatMessage::new(Role::User, "review this")];
        let mut options = ChatOptions::default();
        apply_preset_to_request(&preset(), &mut messages, &mut options);
        assert_eq!(messages[0].role, Role::System);
        assert_eq!(messages[0].content, "You review diffs.");
        assert_eq!(options.temperature, Some(0.2));
        assert_eq!(options.max_tokens, Some(256));
        assert_eq!(options.top_p, Some(0.9));
        assert_eq!(options.top_k, Some(20));
        assert_eq!(options.repeat_penalty, Some(1.2));
    }

    #[test]
    fn explicit_options_and_an_existing_system_message_win_over_the_preset() {
        let mut messages = vec![
            ChatMessage::new(Role::System, "mine"),
            ChatMessage::new(Role::User, "hi"),
        ];
        let mut options = ChatOptions {
            temperature: Some(0.9),
            max_tokens: Some(8),
            top_p: Some(0.5),
            ..Default::default()
        };
        apply_preset_to_request(&preset(), &mut messages, &mut options);
        assert_eq!(messages.len(), 2, "no second system message");
        assert_eq!(messages[0].content, "mine");
        assert_eq!(options.temperature, Some(0.9));
        assert_eq!(options.max_tokens, Some(8));
        assert_eq!(options.top_p, Some(0.5));
    }

    #[test]
    fn sampling_values_reach_the_body_and_out_of_range_ones_are_refused() {
        let up = std::sync::Arc::new(RuntimeUpstream::new(
            std::sync::Arc::new(crate::adapters::fakes::FakeServer(std::sync::Mutex::new(
                Some("http://127.0.0.1:8081".into()),
            ))),
            std::sync::Arc::new(crate::adapters::fakes::FakeLibrary(vec![])),
            crate::adapters::fakes::counting_connect(
                std::sync::Arc::new(crate::adapters::fakes::FakeUpstream {
                    loaded: Some("m.gguf".into()),
                    ..Default::default()
                }),
                std::sync::Arc::new(std::sync::Mutex::new(0)),
            ),
        ));
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            let msgs = vec![ChatMessage::new(Role::User, "hi")];
            let ok = prepare_body(
                &up,
                &msgs,
                ChatOptions {
                    top_p: Some(0.5),
                    top_k: Some(40),
                    repeat_penalty: Some(1.1),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
            assert_eq!(ok["top_p"], 0.5);
            assert_eq!(ok["top_k"], 40);
            assert_eq!(ok["repeat_penalty"], serde_json::json!(1.1f32));
            for bad in [
                ChatOptions {
                    top_p: Some(0.0),
                    ..Default::default()
                },
                ChatOptions {
                    top_k: Some(5000),
                    ..Default::default()
                },
                ChatOptions {
                    repeat_penalty: Some(9.0),
                    ..Default::default()
                },
            ] {
                assert!(matches!(
                    prepare_body(&up, &msgs, bad).await,
                    Err(RaptorError::InvalidInput(_))
                ));
            }
        });
    }
}
