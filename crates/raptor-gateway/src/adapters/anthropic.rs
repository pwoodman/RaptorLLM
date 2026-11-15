//! Anthropic Messages API <-> OpenAI Chat Completions translation.
//!
//! Covered: text, `tool_use` / `tool_result` blocks, `tools`, `tool_choice`,
//! `system`, `stop_sequences`, sampling parameters, non-streamed responses, and
//! streamed responses (text and tool-call blocks, `input_json_delta`).
//!
//! Not translated (rejected or dropped): image and document blocks (rejected,
//! the local models here are text-only), `thinking` blocks (dropped), and
//! `metadata` / `service_tier` / `disable_parallel_tool_use` (dropped).

use raptor_core::{RaptorError, Result};
use serde_json::{json, Map, Value};

/// An Anthropic request rewritten as an OpenAI chat request.
#[derive(Debug, Clone, PartialEq)]
pub struct Translated {
    /// OpenAI-shaped body. `model` is still the client's name; the gateway resolves it.
    pub body: Value,
    /// The model name the client asked for, echoed back in responses.
    pub model: String,
    /// Whether the client asked for a streamed response.
    pub stream: bool,
}

fn invalid(msg: impl Into<String>) -> RaptorError {
    RaptorError::InvalidInput(msg.into())
}

/// Translates an Anthropic `/v1/messages` request body.
pub fn request_to_openai(body: &Value) -> Result<Translated> {
    translate_request(body, true)
}

/// Translates a `/v1/messages/count_tokens` body. That request has no
/// `max_tokens`, so the field is optional here and only the input is translated.
pub fn count_request_to_openai(body: &Value) -> Result<Translated> {
    translate_request(body, false)
}

/// Text the upstream would see as input: message text, tool-call arguments and
/// tool schemas, each as a line. Used for token counting.
pub fn input_text(openai_body: &Value) -> String {
    let mut out = String::new();
    if let Some(messages) = openai_body.get("messages").and_then(Value::as_array) {
        for m in messages {
            if let Some(s) = m.get("content").and_then(Value::as_str) {
                out.push_str(s);
                out.push('\n');
            }
            if let Some(calls) = m.get("tool_calls").and_then(Value::as_array) {
                for c in calls {
                    out.push_str(c["function"]["name"].as_str().unwrap_or(""));
                    out.push(' ');
                    out.push_str(c["function"]["arguments"].as_str().unwrap_or(""));
                    out.push('\n');
                }
            }
        }
    }
    if let Some(tools) = openai_body.get("tools") {
        out.push_str(&tools.to_string());
    }
    out
}

fn translate_request(body: &Value, require_max_tokens: bool) -> Result<Translated> {
    let obj = body
        .as_object()
        .ok_or_else(|| invalid("request body must be a JSON object"))?;
    let model = obj
        .get("model")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| invalid("'model' is required"))?
        .to_string();
    let max_tokens = match obj.get("max_tokens").and_then(Value::as_u64) {
        Some(0) => return Err(invalid("'max_tokens' must be a positive integer")),
        Some(n) => n,
        None if require_max_tokens => {
            return Err(invalid(
                "'max_tokens' is required and must be a positive integer",
            ))
        }
        None => 1,
    };
    let raw_messages = obj
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("'messages' is required and must be an array"))?;
    let stream = obj.get("stream").and_then(Value::as_bool).unwrap_or(false);

    let mut messages = Vec::new();
    if let Some(system) = obj.get("system") {
        let text = system_text(system)?;
        if !text.is_empty() {
            messages.push(json!({"role": "system", "content": text}));
        }
    }
    for m in raw_messages {
        translate_message(m, &mut messages)?;
    }

    let mut out = Map::new();
    out.insert("model".into(), json!(model));
    out.insert("max_tokens".into(), json!(max_tokens));
    out.insert("messages".into(), Value::Array(messages));
    out.insert("stream".into(), json!(stream));
    for key in ["temperature", "top_p", "top_k"] {
        if let Some(v) = obj.get(key) {
            out.insert(key.into(), v.clone());
        }
    }
    if let Some(stop) = obj.get("stop_sequences") {
        out.insert("stop".into(), stop.clone());
    }
    if let Some(tools) = obj.get("tools").and_then(Value::as_array) {
        let translated = tools
            .iter()
            .map(translate_tool)
            .collect::<Result<Vec<_>>>()?;
        out.insert("tools".into(), Value::Array(translated));
    }
    if let Some(choice) = obj.get("tool_choice") {
        out.insert("tool_choice".into(), translate_tool_choice(choice)?);
    }
    if stream {
        out.insert("stream_options".into(), json!({"include_usage": true}));
    }
    Ok(Translated {
        body: Value::Object(out),
        model,
        stream,
    })
}

fn system_text(system: &Value) -> Result<String> {
    match system {
        Value::String(s) => Ok(s.clone()),
        Value::Array(blocks) => {
            let mut parts = Vec::new();
            for b in blocks {
                match b.get("type").and_then(Value::as_str) {
                    Some("text") => parts.push(text_of(b)),
                    _ => return Err(invalid("system blocks must be of type 'text'")),
                }
            }
            Ok(parts.join("\n\n"))
        }
        _ => Err(invalid(
            "'system' must be a string or an array of text blocks",
        )),
    }
}

fn text_of(block: &Value) -> String {
    block
        .get("text")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string()
}

fn push_user_text(parts: &mut Vec<String>, out: &mut Vec<Value>) {
    if !parts.is_empty() {
        out.push(json!({"role": "user", "content": parts.join("\n")}));
        parts.clear();
    }
}

fn translate_message(m: &Value, out: &mut Vec<Value>) -> Result<()> {
    let role = m
        .get("role")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("each message needs a 'role'"))?;
    if role != "user" && role != "assistant" {
        return Err(invalid(format!("unsupported message role '{role}'")));
    }
    let content = m
        .get("content")
        .ok_or_else(|| invalid("each message needs 'content'"))?;
    let blocks = match content {
        Value::String(s) => {
            out.push(json!({"role": role, "content": s}));
            return Ok(());
        }
        Value::Array(blocks) => blocks,
        _ => return Err(invalid("message 'content' must be a string or an array")),
    };

    let mut text_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<Value> = Vec::new();
    for b in blocks {
        match b.get("type").and_then(Value::as_str) {
            Some("text") => text_parts.push(text_of(b)),
            Some("tool_use") if role == "assistant" => {
                let id = b.get("id").and_then(Value::as_str).unwrap_or("");
                let name = b.get("name").and_then(Value::as_str).unwrap_or("");
                let input = b.get("input").cloned().unwrap_or_else(|| json!({}));
                tool_calls.push(json!({
                    "id": id,
                    "type": "function",
                    "function": {"name": name, "arguments": input.to_string()},
                }));
            }
            Some("tool_result") if role == "user" => {
                // Results must precede any text that follows them in the same turn.
                push_user_text(&mut text_parts, out);
                let id = b.get("tool_use_id").and_then(Value::as_str).unwrap_or("");
                out.push(json!({
                    "role": "tool",
                    "tool_call_id": id,
                    "content": tool_result_text(b),
                }));
            }
            Some("thinking") | Some("redacted_thinking") => {}
            Some("image") | Some("document") => {
                return Err(RaptorError::Unsupported(
                    "image and document content is not supported by the local models".into(),
                ))
            }
            Some(other) => {
                return Err(RaptorError::Unsupported(format!(
                    "content block type '{other}' is not supported"
                )))
            }
            None => return Err(invalid("content block without a 'type'")),
        }
    }

    if role == "assistant" {
        let text: String = text_parts.concat();
        // OpenAI expects null content when the turn is only tool calls.
        let content = if text.is_empty() && !tool_calls.is_empty() {
            Value::Null
        } else {
            json!(text)
        };
        let mut msg = json!({"role": "assistant", "content": content});
        if !tool_calls.is_empty() {
            msg["tool_calls"] = Value::Array(tool_calls);
        }
        out.push(msg);
    } else {
        push_user_text(&mut text_parts, out);
    }
    Ok(())
}

fn tool_result_text(block: &Value) -> String {
    let body = match block.get("content") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter(|p| p.get("type").and_then(Value::as_str) == Some("text"))
            .map(text_of)
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    };
    if block
        .get("is_error")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        format!("Error: {body}")
    } else {
        body
    }
}

fn translate_tool(t: &Value) -> Result<Value> {
    let name = t
        .get("name")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| invalid("each tool needs a 'name'"))?;
    let mut function = json!({
        "name": name,
        "parameters": t.get("input_schema").cloned().unwrap_or_else(|| json!({"type": "object", "properties": {}})),
    });
    if let Some(desc) = t.get("description").and_then(Value::as_str) {
        function["description"] = json!(desc);
    }
    Ok(json!({"type": "function", "function": function}))
}

fn translate_tool_choice(c: &Value) -> Result<Value> {
    match c.get("type").and_then(Value::as_str) {
        Some("auto") => Ok(json!("auto")),
        Some("any") => Ok(json!("required")),
        Some("none") => Ok(json!("none")),
        Some("tool") => {
            let name = c
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid("tool_choice of type 'tool' needs a 'name'"))?;
            Ok(json!({"type": "function", "function": {"name": name}}))
        }
        _ => Err(invalid("unsupported tool_choice")),
    }
}

/// Maps an OpenAI `finish_reason` (plus whether tool calls were produced) to an
/// Anthropic `stop_reason`.
pub fn stop_reason(finish: Option<&str>, has_tool_use: bool) -> &'static str {
    match finish {
        Some("length") => "max_tokens",
        Some("tool_calls") | Some("function_call") => "tool_use",
        _ if has_tool_use => "tool_use",
        _ => "end_turn",
    }
}

fn message_id(resp: &Value) -> String {
    match resp.get("id").and_then(Value::as_str) {
        Some(id) => format!("msg_{id}"),
        None => "msg_raptor".to_string(),
    }
}

/// Parses tool-call arguments emitted by the model. They must be a JSON object.
fn parse_arguments(raw: &str) -> Result<Value> {
    if raw.trim().is_empty() {
        return Ok(json!({}));
    }
    match serde_json::from_str::<Value>(raw) {
        Ok(v @ Value::Object(_)) => Ok(v),
        _ => Err(RaptorError::Runtime(
            "upstream returned tool call arguments that are not a JSON object".into(),
        )),
    }
}

/// Converts a complete OpenAI chat response into an Anthropic message.
pub fn response_to_anthropic(resp: &Value, model: &str) -> Result<Value> {
    let choice = resp
        .get("choices")
        .and_then(|c| c.get(0))
        .ok_or_else(|| RaptorError::Runtime("upstream response has no choices".into()))?;
    let message = choice.get("message").unwrap_or(&Value::Null);
    let mut content = Vec::new();
    if let Some(text) = message.get("content").and_then(Value::as_str) {
        if !text.is_empty() {
            content.push(json!({"type": "text", "text": text}));
        }
    }
    let empty = Vec::new();
    let calls = message
        .get("tool_calls")
        .and_then(Value::as_array)
        .unwrap_or(&empty);
    for (i, call) in calls.iter().enumerate() {
        let id = call
            .get("id")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| format!("toolu_{i}"));
        let name = call["function"]["name"].as_str().unwrap_or("");
        let args = call["function"]["arguments"].as_str().unwrap_or("");
        content.push(
            json!({"type": "tool_use", "id": id, "name": name, "input": parse_arguments(args)?}),
        );
    }
    let has_tool_use = !calls.is_empty();
    let usage = resp.get("usage");
    let input_tokens = usage
        .and_then(|u| u.get("prompt_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let output_tokens = usage
        .and_then(|u| u.get("completion_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    Ok(json!({
        "id": message_id(resp),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason(choice.get("finish_reason").and_then(Value::as_str), has_tool_use),
        "stop_sequence": null,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }))
}

/// One Anthropic stream event: `(event name, data)`.
pub type Event = (&'static str, Value);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Open {
    Text { index: usize },
    Tool { call: u64, index: usize },
}

/// Turns OpenAI stream chunks into Anthropic stream events, one chunk at a time.
///
/// Content blocks are numbered in order of appearance. A text block closes when
/// a tool call starts, and a tool block closes when the next call starts or
/// the stream ends. Tool-call argument fragments are forwarded as
/// `input_json_delta` events in the order received.
#[derive(Debug)]
pub struct StreamTranslator {
    model: String,
    id: String,
    started: bool,
    next_index: usize,
    open: Option<Open>,
    stop: Option<String>,
    has_tool_use: bool,
    input_tokens: u64,
    output_tokens: u64,
}

impl StreamTranslator {
    /// Creates a translator that reports `model` in `message_start`.
    pub fn new(model: &str) -> Self {
        Self {
            model: model.to_string(),
            id: String::new(),
            started: false,
            next_index: 0,
            open: None,
            stop: None,
            has_tool_use: false,
            input_tokens: 0,
            output_tokens: 0,
        }
    }

    /// Consumes one upstream payload (the JSON of one `data:` line).
    pub fn push(&mut self, payload: &str) -> Result<Vec<Event>> {
        let chunk: Value = serde_json::from_str(payload)
            .map_err(|_| RaptorError::Runtime("upstream sent a malformed stream chunk".into()))?;
        let mut out = Vec::new();
        if !self.started {
            self.id = chunk
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            self.start(&mut out);
        }
        if let Some(usage) = chunk.get("usage").filter(|u| u.is_object()) {
            self.input_tokens = usage
                .get("prompt_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(self.input_tokens);
            self.output_tokens = usage
                .get("completion_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(self.output_tokens);
        }
        let Some(choice) = chunk.get("choices").and_then(|c| c.get(0)) else {
            return Ok(out);
        };
        let delta = choice.get("delta").unwrap_or(&Value::Null);
        if let Some(text) = delta
            .get("content")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
        {
            self.text_delta(text, &mut out);
        }
        if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
            for call in calls {
                self.tool_delta(call, &mut out);
            }
        }
        if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
            self.stop = Some(reason.to_string());
        }
        Ok(out)
    }

    /// Closes any open block and emits the message trailer. Call once, at the end.
    pub fn finish(&mut self) -> Vec<Event> {
        let mut out = Vec::new();
        if !self.started {
            self.start(&mut out);
        }
        self.close_open(&mut out);
        let reason = stop_reason(self.stop.as_deref(), self.has_tool_use);
        out.push((
            "message_delta",
            json!({
                "type": "message_delta",
                "delta": {"stop_reason": reason, "stop_sequence": null},
                "usage": {"output_tokens": self.output_tokens},
            }),
        ));
        out.push(("message_stop", json!({"type": "message_stop"})));
        out
    }

    fn start(&mut self, out: &mut Vec<Event>) {
        self.started = true;
        out.push((
            "message_start",
            json!({
                "type": "message_start",
                "message": {
                    "id": if self.id.is_empty() { "msg_raptor".to_string() } else { format!("msg_{}", self.id) },
                    "type": "message",
                    "role": "assistant",
                    "model": self.model,
                    "content": [],
                    "stop_reason": null,
                    "stop_sequence": null,
                    "usage": {"input_tokens": self.input_tokens, "output_tokens": 0},
                },
            }),
        ));
    }

    fn close_open(&mut self, out: &mut Vec<Event>) {
        if let Some(open) = self.open.take() {
            let index = match open {
                Open::Text { index } | Open::Tool { index, .. } => index,
            };
            out.push((
                "content_block_stop",
                json!({"type": "content_block_stop", "index": index}),
            ));
        }
    }

    fn text_delta(&mut self, text: &str, out: &mut Vec<Event>) {
        let index = match self.open {
            Some(Open::Text { index }) => index,
            _ => {
                self.close_open(out);
                let index = self.next_index;
                self.next_index += 1;
                out.push((
                    "content_block_start",
                    json!({"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}}),
                ));
                self.open = Some(Open::Text { index });
                index
            }
        };
        out.push((
            "content_block_delta",
            json!({"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": text}}),
        ));
    }

    fn tool_delta(&mut self, call: &Value, out: &mut Vec<Event>) {
        let call_no = call.get("index").and_then(Value::as_u64).unwrap_or(0);
        let index = match self.open {
            Some(Open::Tool {
                call: current,
                index,
            }) if current == call_no => index,
            _ => {
                self.close_open(out);
                let index = self.next_index;
                self.next_index += 1;
                let id = call
                    .get("id")
                    .and_then(Value::as_str)
                    .map(str::to_string)
                    .unwrap_or_else(|| format!("toolu_{call_no}"));
                let name = call["function"]["name"].as_str().unwrap_or("");
                out.push((
                    "content_block_start",
                    json!({
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {"type": "tool_use", "id": id, "name": name, "input": {}},
                    }),
                ));
                self.open = Some(Open::Tool {
                    call: call_no,
                    index,
                });
                self.has_tool_use = true;
                index
            }
        };
        if let Some(args) = call["function"]["arguments"]
            .as_str()
            .filter(|s| !s.is_empty())
        {
            out.push((
                "content_block_delta",
                json!({"type": "content_block_delta", "index": index, "delta": {"type": "input_json_delta", "partial_json": args}}),
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const REQ_IN: &str = include_str!("golden/anthropic_request_tools.json");
    const REQ_OUT: &str = include_str!("golden/openai_request_tools.json");
    const RESP_IN: &str = include_str!("golden/openai_response_tool_calls.json");
    const RESP_OUT: &str = include_str!("golden/anthropic_response_tool_use.json");

    #[test]
    fn golden_request_translation() {
        let input: Value = serde_json::from_str(REQ_IN).unwrap();
        let expected: Value = serde_json::from_str(REQ_OUT).unwrap();
        let t = request_to_openai(&input).unwrap();
        assert_eq!(t.body, expected);
        assert!(!t.stream);
        assert_eq!(t.model, "claude-sonnet-4-5");
    }

    #[test]
    fn golden_response_translation() {
        let input: Value = serde_json::from_str(RESP_IN).unwrap();
        let expected: Value = serde_json::from_str(RESP_OUT).unwrap();
        let got = response_to_anthropic(&input, "claude-sonnet-4-5").unwrap();
        assert_eq!(got, expected);
    }

    #[test]
    fn streaming_request_asks_for_usage() {
        let body = json!({"model": "m", "max_tokens": 5, "stream": true, "messages": [{"role": "user", "content": "hi"}]});
        let t = request_to_openai(&body).unwrap();
        assert!(t.stream);
        assert_eq!(t.body["stream_options"], json!({"include_usage": true}));
    }

    #[test]
    fn image_blocks_are_rejected_not_dropped() {
        let body = json!({"model": "m", "max_tokens": 5, "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}
        ]}]});
        let err = request_to_openai(&body).unwrap_err();
        assert!(matches!(err, RaptorError::Unsupported(_)));
    }

    #[test]
    fn missing_max_tokens_is_invalid() {
        let body = json!({"model": "m", "messages": []});
        assert!(matches!(
            request_to_openai(&body),
            Err(RaptorError::InvalidInput(_))
        ));
    }

    #[test]
    fn tool_choice_variants_map() {
        assert_eq!(
            translate_tool_choice(&json!({"type": "any"})).unwrap(),
            json!("required")
        );
        assert_eq!(
            translate_tool_choice(&json!({"type": "none"})).unwrap(),
            json!("none")
        );
        assert_eq!(
            translate_tool_choice(&json!({"type": "tool", "name": "f"})).unwrap(),
            json!({"type": "function", "function": {"name": "f"}})
        );
        assert!(translate_tool_choice(&json!({"type": "tool"})).is_err());
    }

    #[test]
    fn stop_reason_mapping() {
        assert_eq!(stop_reason(Some("stop"), false), "end_turn");
        assert_eq!(stop_reason(Some("length"), false), "max_tokens");
        assert_eq!(stop_reason(Some("tool_calls"), false), "tool_use");
        assert_eq!(stop_reason(None, true), "tool_use");
    }

    #[test]
    fn malformed_tool_arguments_are_a_runtime_error() {
        let resp = json!({"choices": [{"message": {"tool_calls": [
            {"id": "c", "type": "function", "function": {"name": "f", "arguments": "{oops"}}
        ]}, "finish_reason": "tool_calls"}]});
        assert!(matches!(
            response_to_anthropic(&resp, "m"),
            Err(RaptorError::Runtime(_))
        ));
    }

    #[test]
    fn tool_result_error_flag_is_marked() {
        let block =
            json!({"type": "tool_result", "tool_use_id": "t", "content": "boom", "is_error": true});
        assert_eq!(tool_result_text(&block), "Error: boom");
    }
}
