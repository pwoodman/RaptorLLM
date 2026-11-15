//! Request hygiene for coding agents, applied to chat requests before they reach the upstream.
//!
//! Two things make a long agent session slow or wasteful on a local model:
//! - reasoning from earlier turns that the client echoes back (`<think>...</think>` in the
//!   content, or a `reasoning_content` field). It is thousands of tokens per turn that the model
//!   does not need and that fill a 32K-64K window in a few steps;
//! - a volatile token (a clock time, a UUID) near the top of the system prompt, which changes the
//!   prompt from its first tokens and so defeats prompt-prefix caching on every request.
//!
//! [`strip_prior_reasoning`] removes the first kind. [`volatile_prefix_token`] only detects the
//! second: the prompt is never modified.

use serde_json::Value;

const OPEN: &str = "<think>";
const CLOSE: &str = "</think>";

/// How far into the system prompt a volatile token is looked for, in characters.
const PREFIX_CHARS: usize = 500;

/// Removes reasoning from assistant messages of earlier turns: every assistant message before the
/// last `user` message. Assistant messages after it belong to the turn in progress (a tool-call
/// loop) and keep their reasoning, as do user, system and tool messages. Removes `<think>..</think>`
/// blocks from text content and drops the `reasoning_content` and `reasoning` fields. An unclosed
/// `<think>` removes the rest of that text. Text with no reasoning is left byte-identical, so a
/// cached prompt prefix still matches. Returns the number of messages changed.
pub fn strip_prior_reasoning(body: &mut Value) -> usize {
    let Some(messages) = body.get_mut("messages").and_then(Value::as_array_mut) else {
        return 0;
    };
    let Some(last_user) = messages
        .iter()
        .rposition(|m| m.get("role").and_then(Value::as_str) == Some("user"))
    else {
        return 0;
    };
    let mut changed = 0;
    for m in messages.iter_mut().take(last_user) {
        if m.get("role").and_then(Value::as_str) != Some("assistant") {
            continue;
        }
        if strip_message(m) {
            changed += 1;
        }
    }
    changed
}

fn strip_message(m: &mut Value) -> bool {
    let mut changed = false;
    if let Some(obj) = m.as_object_mut() {
        for key in ["reasoning_content", "reasoning"] {
            changed |= obj.remove(key).is_some();
        }
    }
    match m.get_mut("content") {
        Some(Value::String(s)) => {
            if let Some(clean) = strip_think(s) {
                *s = clean;
                changed = true;
            }
        }
        Some(Value::Array(parts)) => {
            for part in parts {
                if part.get("type").and_then(Value::as_str) != Some("text") {
                    continue;
                }
                if let Some(Value::String(s)) = part.get_mut("text") {
                    if let Some(clean) = strip_think(s) {
                        *s = clean;
                        changed = true;
                    }
                }
            }
        }
        _ => {}
    }
    changed
}

/// `text` without its `<think>` blocks, or `None` when it has none. Whitespace right after a
/// block that opens the text is dropped with it (the blank line models put between thinking and
/// answer); everything else is kept as is.
fn strip_think(text: &str) -> Option<String> {
    let first = text.find(OPEN)?;
    let mut out = String::with_capacity(text.len());
    out.push_str(&text[..first]);
    let mut rest = &text[first..];
    let mut at_start = first == 0;
    while let Some(open) = rest.find(OPEN) {
        out.push_str(&rest[..open]);
        let after_open = &rest[open + OPEN.len()..];
        match after_open.find(CLOSE) {
            Some(close) => {
                rest = &after_open[close + CLOSE.len()..];
                if at_start {
                    rest = rest.trim_start();
                }
            }
            None => {
                // Cut off mid-thought: nothing after the tag is an answer.
                rest = "";
                break;
            }
        }
        at_start = false;
    }
    out.push_str(rest);
    Some(out)
}

/// The kind of volatile token found in the first [`PREFIX_CHARS`] characters of the first system
/// (or developer) message, if any: `"time of day"`, `"date and time"` or `"UUID"`. A date alone is
/// not reported: it changes once a day, not on every request.
pub fn volatile_prefix_token(body: &Value) -> Option<&'static str> {
    let messages = body.get("messages")?.as_array()?;
    let system = messages.iter().find(|m| {
        matches!(
            m.get("role").and_then(Value::as_str),
            Some("system") | Some("developer")
        )
    })?;
    let text = match system.get("content")? {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => return None,
    };
    let head: String = text.chars().take(PREFIX_CHARS).collect();
    find_volatile(head.as_bytes())
}

fn find_volatile(b: &[u8]) -> Option<&'static str> {
    let digit = |i: usize| b.get(i).is_some_and(u8::is_ascii_digit);
    let hex = |i: usize| b.get(i).is_some_and(u8::is_ascii_hexdigit);
    let is = |i: usize, c: u8| b.get(i) == Some(&c);
    let mut found = None;
    for i in 0..b.len() {
        // UUID: 8-4-4-4-12 hex digits.
        if hex(i)
            && (0..8).all(hex_at(b, i))
            && is(i + 8, b'-')
            && (9..13).all(hex_at(b, i))
            && is(i + 13, b'-')
            && (14..18).all(hex_at(b, i))
            && is(i + 18, b'-')
            && (19..23).all(hex_at(b, i))
            && is(i + 23, b'-')
            && (24..36).all(hex_at(b, i))
        {
            return Some("UUID");
        }
        // Date and time: 2026-11-14T12:34 or 2026-11-14 12:34.
        if digit(i)
            && (0..4).all(|k| digit(i + k))
            && is(i + 4, b'-')
            && digit(i + 5)
            && digit(i + 6)
            && is(i + 7, b'-')
            && digit(i + 8)
            && digit(i + 9)
            && (is(i + 10, b'T') || is(i + 10, b' '))
            && digit(i + 11)
            && digit(i + 12)
            && is(i + 13, b':')
            && digit(i + 14)
            && digit(i + 15)
        {
            return Some("date and time");
        }
        // Time of day with seconds: 12:34:56, not part of a longer digit run.
        if digit(i) && !(i > 0 && (digit(i - 1) || is(i - 1, b':'))) {
            // One or two hour digits, then :MM:SS.
            let h = if digit(i + 1) { 2 } else { 1 };
            let m = i + h;
            if is(m, b':')
                && digit(m + 1)
                && digit(m + 2)
                && is(m + 3, b':')
                && digit(m + 4)
                && digit(m + 5)
                && !digit(m + 6)
            {
                found = Some("time of day");
            }
        }
    }
    found
}

fn hex_at(b: &[u8], base: usize) -> impl Fn(usize) -> bool + '_ {
    move |k| b.get(base + k).is_some_and(u8::is_ascii_hexdigit)
}

/// Request fields with which a client already says how the model should think. A request that
/// has any of them is left alone: the client's choice is never overridden.
const REASONING_FIELDS: [&str; 6] = [
    "reasoning_effort",
    "reasoning",
    "thinking",
    "thinking_budget_tokens",
    "reasoning_budget",
    "enable_thinking",
];

/// True when `body` already controls thinking: one of [`REASONING_FIELDS`], or
/// `chat_template_kwargs` with `enable_thinking` or `reasoning_effort` in it.
pub fn controls_reasoning(body: &Value) -> bool {
    REASONING_FIELDS.iter().any(|k| body.get(*k).is_some())
        || body.get("chat_template_kwargs").is_some_and(|k| {
            k.get("enable_thinking").is_some() || k.get("reasoning_effort").is_some()
        })
}

/// Adds `reasoning_effort: level` to a chat request that does not control thinking itself.
/// Returns whether it did. Nothing else in the body changes. The caller passes a level only for a
/// model that supports it (see [`crate::thinking`]), so other requests stay byte-identical.
pub fn inject_reasoning_effort(body: &mut Value, level: &str) -> bool {
    if controls_reasoning(body) {
        return false;
    }
    match body.as_object_mut() {
        Some(obj) => {
            obj.insert("reasoning_effort".into(), Value::String(level.to_string()));
            true
        }
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn chat(messages: Value) -> Value {
        json!({"model": "m", "messages": messages})
    }

    #[test]
    fn reasoning_is_stripped_from_earlier_assistant_turns_only() {
        let mut body = chat(json!([
            {"role": "system", "content": "<think>keep</think> system"},
            {"role": "user", "content": "<think>keep</think> hi"},
            {"role": "assistant", "content": "<think>plan the reply</think>\n\nHello there."},
            {"role": "user", "content": "next"},
            {"role": "assistant", "content": "<think>current turn</think> calling", "tool_calls": []},
            {"role": "tool", "content": "<think>tool text</think>"}
        ]));
        assert_eq!(strip_prior_reasoning(&mut body), 1);
        let m = body["messages"].as_array().unwrap();
        assert_eq!(m[0]["content"], "<think>keep</think> system");
        assert_eq!(m[1]["content"], "<think>keep</think> hi");
        assert_eq!(m[2]["content"], "Hello there.");
        // After the last user message: the turn in progress keeps its reasoning.
        assert_eq!(m[4]["content"], "<think>current turn</think> calling");
        assert_eq!(m[5]["content"], "<think>tool text</think>");
    }

    #[test]
    fn multiple_blocks_and_unclosed_tags_are_handled() {
        let mut body = chat(json!([
            {"role": "assistant", "content": "A<think>x</think>B <think>y</think>C"},
            {"role": "assistant", "content": "answer so far<think>cut off mid-thou"},
            {"role": "user", "content": "go"}
        ]));
        assert_eq!(strip_prior_reasoning(&mut body), 2);
        let m = body["messages"].as_array().unwrap();
        // Only a block that opens the text takes the whitespace after it.
        assert_eq!(m[0]["content"], "AB C");
        assert_eq!(m[1]["content"], "answer so far");
    }

    #[test]
    fn reasoning_fields_and_text_parts_are_cleaned() {
        let mut body = chat(json!([
            {"role": "assistant", "content": [
                {"type": "text", "text": "<think>t</think>shown"},
                {"type": "image_url", "image_url": {"url": "<think>x</think>"}}
            ], "reasoning_content": "long chain of thought", "reasoning": "more"},
            {"role": "user", "content": "go"}
        ]));
        assert_eq!(strip_prior_reasoning(&mut body), 1);
        let m = &body["messages"][0];
        assert_eq!(m["content"][0]["text"], "shown");
        assert_eq!(m["content"][1]["image_url"]["url"], "<think>x</think>");
        assert!(m.get("reasoning_content").is_none() && m.get("reasoning").is_none());
    }

    #[test]
    fn text_without_reasoning_is_byte_identical_and_counts_as_unchanged() {
        let original = chat(json!([
            {"role": "assistant", "content": "  spaced \n text with <thinking> and </think> alone\n", "tool_calls": [{"id": "1"}]},
            {"role": "user", "content": "go"}
        ]));
        let mut body = original.clone();
        assert_eq!(strip_prior_reasoning(&mut body), 0);
        assert_eq!(body, original);
        // No user message at all: nothing is "prior".
        let mut only_assistant =
            chat(json!([{"role": "assistant", "content": "<think>x</think>y"}]));
        assert_eq!(strip_prior_reasoning(&mut only_assistant), 0);
        // Not a chat request.
        assert_eq!(
            strip_prior_reasoning(&mut json!({"prompt": "<think>x</think>"})),
            0
        );
    }

    #[test]
    fn stripping_is_stable_across_turns_so_the_prefix_cache_still_matches() {
        let history = |extra: Value| {
            let mut v = vec![
                json!({"role": "user", "content": "q1"}),
                json!({"role": "assistant", "content": "<think>t1</think>\n\na1"}),
                json!({"role": "user", "content": "q2"}),
                json!({"role": "assistant", "content": "<think>t2</think>a2"}),
                json!({"role": "user", "content": "q3"}),
            ];
            v.push(extra);
            chat(Value::Array(v))
        };
        let mut a = history(json!({"role": "assistant", "content": "<think>t3</think>a3"}));
        let mut b = history(json!({"role": "user", "content": "q4"}));
        strip_prior_reasoning(&mut a);
        strip_prior_reasoning(&mut b);
        // The shared history is identical in both requests after stripping.
        assert_eq!(
            a["messages"].as_array().unwrap()[..5],
            b["messages"].as_array().unwrap()[..5]
        );
        assert_eq!(a["messages"][1]["content"], "a1");
    }

    fn sys(text: &str) -> Value {
        chat(json!([{"role": "system", "content": text}, {"role": "user", "content": "hi"}]))
    }

    #[test]
    fn a_clock_time_a_timestamp_or_a_uuid_near_the_top_is_detected() {
        assert_eq!(
            volatile_prefix_token(&sys("You are a coder. Current time: 12:34:56. Be brief.")),
            Some("time of day")
        );
        assert_eq!(
            volatile_prefix_token(&sys("Now: 9:05:07 local")),
            Some("time of day")
        );
        assert_eq!(
            volatile_prefix_token(&sys("Session started 2026-11-14T08:15:30Z")),
            Some("date and time")
        );
        assert_eq!(
            volatile_prefix_token(&sys("Session started 2026-11-14 08:15")),
            Some("date and time")
        );
        assert_eq!(
            volatile_prefix_token(&sys("request id 123e4567-e89b-12d3-a456-426614174000 end")),
            Some("UUID")
        );
        // Array content and the developer role are read too.
        let parts = chat(json!([
            {"role": "developer", "content": [{"type": "text", "text": "at 23:59:01"}]}
        ]));
        assert_eq!(volatile_prefix_token(&parts), Some("time of day"));
    }

    #[test]
    fn stable_prompts_and_tokens_past_the_first_500_chars_are_not_flagged() {
        assert_eq!(volatile_prefix_token(&sys("You are a coder.")), None);
        // A date alone changes once a day, and a version or ratio is not a time.
        assert_eq!(
            volatile_prefix_token(&sys("Today is 2026-11-14. v1.2.3, ratio 10:30")),
            None
        );
        assert_eq!(
            volatile_prefix_token(&sys("addr 192.168.100.200:8080")),
            None
        );
        let late = format!("{} Current time: 12:34:56", "x".repeat(600));
        assert_eq!(volatile_prefix_token(&sys(&late)), None);
        // Only the system prompt counts, not user text.
        let user = chat(json!([{"role": "user", "content": "it is 12:34:56"}]));
        assert_eq!(volatile_prefix_token(&user), None);
        assert_eq!(volatile_prefix_token(&json!({"prompt": "12:34:56"})), None);
    }

    #[test]
    fn detection_does_not_modify_the_request() {
        let body = sys("Current time: 12:34:56");
        let before = body.clone();
        let _ = volatile_prefix_token(&body);
        assert_eq!(body, before);
    }

    #[test]
    fn a_default_effort_is_added_to_a_request_that_says_nothing_about_thinking() {
        let mut body = chat(json!([{"role": "user", "content": "hi"}]));
        assert!(inject_reasoning_effort(&mut body, "medium"));
        assert_eq!(body["reasoning_effort"], "medium");
        // Only that field was added.
        assert_eq!(body["messages"][0]["content"], "hi");
        assert_eq!(body.as_object().unwrap().len(), 3);
    }

    #[test]
    fn a_client_that_controls_thinking_is_never_overridden() {
        for extra in [
            json!({"reasoning_effort": "high"}),
            json!({"reasoning": {"effort": "low"}}),
            json!({"thinking": {"type": "enabled", "budget_tokens": 1024}}),
            json!({"thinking_budget_tokens": 64}),
            json!({"reasoning_budget": 0}),
            json!({"enable_thinking": false}),
            json!({"chat_template_kwargs": {"enable_thinking": false}}),
            json!({"chat_template_kwargs": {"reasoning_effort": "low"}}),
        ] {
            let mut body = chat(json!([{"role": "user", "content": "hi"}]));
            for (k, v) in extra.as_object().unwrap() {
                body[k] = v.clone();
            }
            let before = body.clone();
            assert!(!inject_reasoning_effort(&mut body, "medium"), "{extra}");
            assert_eq!(body, before, "{extra}");
        }
    }

    #[test]
    fn unrelated_template_kwargs_do_not_count_as_thinking_control() {
        let mut body = chat(json!([{"role": "user", "content": "hi"}]));
        body["chat_template_kwargs"] = json!({"custom_flag": true});
        assert!(inject_reasoning_effort(&mut body, "low"));
        assert_eq!(body["chat_template_kwargs"], json!({"custom_flag": true}));
    }

    #[test]
    fn a_body_that_is_not_an_object_is_left_alone() {
        let mut body = json!(["not", "an", "object"]);
        assert!(!inject_reasoning_effort(&mut body, "low"));
        assert_eq!(body, json!(["not", "an", "object"]));
    }
}
