//! Request-only history selection for chat. Each turn includes the user message and all
//! following assistant and tool messages, so tool call IDs and replies travel together.

use raptor_core::{RaptorError, Result};
use raptor_gateway::Upstream;
use serde_json::{json, Value};
use std::collections::HashSet;

const TEMPLATE_MARGIN: u64 = 32;
const DEFAULT_OUTPUT_RESERVE: u64 = 8192;

fn is_instruction(message: &Value) -> bool {
    matches!(message["role"].as_str(), Some("system" | "developer"))
}

/// A conservative bound for text when the runtime cannot render or tokenize a chat.
fn text_budget(value: &Value) -> u64 {
    value.to_string().len() as u64 + 16
}

fn terms(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !(c.is_alphanumeric() || c == '_' || c == '-'))
        .filter(|word| word.len() >= 3)
        .map(str::to_lowercase)
        .filter(|word| {
            !matches!(
                word.as_str(),
                "the" | "and" | "for" | "with" | "that" | "this" | "please" | "from" | "have"
            )
        })
}

struct Turn {
    start: usize,
    end: usize,
    bytes: u64,
    relevance: usize,
}

fn turns_at(messages: &[Value], starts: &[usize], query: &HashSet<String>) -> Vec<Turn> {
    starts
        .windows(2)
        .map(|pair| {
            let (start, end) = (pair[0], pair[1]);
            let text: Vec<_> = messages[start..end]
                .iter()
                .filter(|m| !is_instruction(m))
                .map(Value::to_string)
                .collect();
            let matching: HashSet<_> = text
                .iter()
                .flat_map(|s| terms(s))
                .filter(|term| query.contains(term))
                .collect();
            Turn {
                start,
                end,
                bytes: text.iter().map(|s| s.len() as u64 + 16).sum(),
                relevance: matching.len(),
            }
        })
        .collect()
}

fn with_history(body: &Value, messages: &[Value], protected: &[bool], turns: &[&Turn]) -> Value {
    let mut selected = protected.to_vec();
    for turn in turns {
        selected[turn.start..turn.end].fill(true);
    }
    let mut request: serde_json::Map<_, _> = body
        .as_object()
        .into_iter()
        .flatten()
        .filter(|(key, _)| *key != "messages")
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    request.insert(
        "messages".into(),
        Value::Array(
            messages
                .iter()
                .enumerate()
                .filter(|(i, message)| selected[*i] || is_instruction(message))
                .map(|(_, message)| message.clone())
                .collect(),
        ),
    );
    Value::Object(request)
}

async fn measure(up: &dyn Upstream, request: &Value, exact: bool) -> u64 {
    if exact {
        if let Some(count) = up.chat_prompt_tokens(request).await {
            return count;
        }
    }
    text_budget(request)
}

fn finish(mut body: Value, explicit: Option<u64>, count: u64, window: u64) -> Value {
    if explicit.is_none() {
        body["max_tokens"] = json!(window.saturating_sub(count + TEMPLATE_MARGIN));
    }
    body
}

/// Reserves output space and drops complete historical turns only when the prompt overflows.
/// Keeps instructions and the current request verbatim. A very long agent turn can also drop
/// earlier completed assistant/tool rounds, keeping the latest round and its replies together.
/// Returns the number of older turns or completed rounds omitted.
pub async fn fit(up: &dyn Upstream, body: Value, window: u32) -> Result<(Value, usize)> {
    let Some(messages) = body["messages"].as_array() else {
        return Ok((body, 0));
    };
    let Some(active) = messages
        .iter()
        .rposition(|message| message["role"] == "user")
    else {
        return Ok((body, 0));
    };
    let window = u64::from(window);
    let explicit = ["max_completion_tokens", "max_tokens", "n_predict"]
        .iter()
        .filter_map(|key| body[*key].as_u64().filter(|n| *n > 0))
        .max();
    let reserve = explicit.unwrap_or_else(|| DEFAULT_OUTPUT_RESERVE.min(window / 4));
    let prompt_limit = window.saturating_sub(reserve.saturating_add(TEMPLATE_MARGIN));
    let count = up.chat_prompt_tokens(&body).await;
    if let Some(tokens) = count.filter(|n| *n <= prompt_limit) {
        return Ok((finish(body, explicit, tokens, window), 0));
    }
    let full_bytes = text_budget(&body);
    let full = count.unwrap_or(full_bytes);
    if full <= prompt_limit {
        return Ok((finish(body, explicit, full, window), 0));
    }

    let mut protected = vec![false; messages.len()];
    protected[active..].fill(true);
    let mut required = with_history(&body, messages, &protected, &[]);
    let mut required_tokens = measure(up, &required, count.is_some()).await;
    let mut round_starts = Vec::new();
    if required_tokens > prompt_limit {
        round_starts
            .extend((active + 1..messages.len()).filter(|i| messages[*i]["role"] == "assistant"));
        if let (Some(first), Some(last)) = (round_starts.first(), round_starts.last()) {
            if first < last {
                protected[*first..*last].fill(false);
                required = with_history(&body, messages, &protected, &[]);
                required_tokens = measure(up, &required, count.is_some()).await;
            }
        }
    }
    if required_tokens > prompt_limit {
        return Err(RaptorError::InvalidInput(format!(
            "The current turn, instructions, and reply budget do not fit the {window}-token context. Split the input by file or section, shorten tool output, or reduce the requested reply length."
        )));
    }

    let content = &messages[active]["content"];
    let query_text = content
        .as_str()
        .map_or_else(|| content.to_string(), str::to_owned);
    let query: HashSet<_> = terms(&query_text).collect();
    let history_starts: Vec<_> = (0..active)
        .filter(|i| messages[*i]["role"] == "user")
        .chain(std::iter::once(active))
        .collect();
    let mut turns = turns_at(messages, &history_starts, &query);
    let old_turns = turns.len();
    turns.extend(turns_at(messages, &round_starts, &query));
    let mut ranked: Vec<_> = turns.iter().enumerate().collect();
    ranked.sort_by_key(|(i, turn)| {
        let tier = if *i == 0 && old_turns > 0 {
            0 // the original brief
        } else if *i >= old_turns || *i + 2 >= old_turns {
            1 // completed rounds of the current task and the most recent older turns
        } else {
            2
        };
        (
            tier,
            std::cmp::Reverse(turn.relevance),
            std::cmp::Reverse(*i),
        )
    });
    let history_bytes = full_bytes.saturating_sub(text_budget(&required)).max(1);
    let history_tokens = full.saturating_sub(required_tokens).max(1);
    let room = prompt_limit - required_tokens;
    let mut spent = 0u64;
    let mut chosen = Vec::new();
    for (_, turn) in ranked {
        let cost = (turn.bytes * history_tokens).div_ceil(history_bytes);
        if spent + cost <= room {
            chosen.push(turn);
            spent += cost;
        }
    }

    // Verify the actual template. If the estimate overpacked history, search the ranked
    // prefixes for a fitting request. The required-only request already passed the budget.
    let mut best = (required, required_tokens, 0);
    let mut low = 1;
    let mut high = chosen.len();
    let mut size = high;
    while low <= high {
        let request = with_history(&body, messages, &protected, &chosen[..size]);
        let tokens = measure(up, &request, count.is_some()).await;
        if tokens <= prompt_limit {
            best = (request, tokens, size);
            low = size + 1;
        } else {
            high = size - 1;
        }
        size = low + high.saturating_sub(low) / 2;
    }
    let (request, tokens, retained) = best;
    Ok((
        finish(request, explicit, tokens, window),
        turns.len() - retained,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use raptor_gateway::{Completion, Endpoint, ModelCatalog};

    struct TestTokenizer;

    #[async_trait]
    impl Upstream for TestTokenizer {
        async fn models(&self) -> Result<ModelCatalog> {
            unreachable!()
        }

        async fn call(&self, _: Endpoint, _: Value) -> Result<Completion> {
            unreachable!()
        }

        async fn chat_prompt_tokens(&self, body: &Value) -> Option<u64> {
            Some(text_budget(&body["messages"]))
        }
    }

    #[tokio::test]
    async fn preserves_active_tool_exchange_and_an_older_relevant_source() {
        let protected = json!([
            {"role":"user","content":"Update the Atlas export deck using the source"},
            {"role":"assistant","content":null,"tool_calls":[{"id":"call_7","type":"function","function":{"name":"read_file","arguments":"{}"}}]},
            {"role":"tool","tool_call_id":"call_7","content":"verified source"}
        ]);
        let mut messages = vec![
            json!({"role":"system","content":"Keep citations and the original filename"}),
            json!({"role":"user","content":"Deliver a six-slide executive presentation"}),
            json!({"role":"assistant","content":"Agreed"}),
            json!({"role":"user","content":"Atlas export deck sources"}),
            json!({"role":"assistant","content":"Use source A, section 3"}),
            json!({"role":"user","content":"unrelated build logs"}),
            json!({"role":"assistant","content":"compiler log ".repeat(200)}),
            json!({"role":"user","content":"Keep this recent design decision"}),
            json!({"role":"assistant","content":"Blue headings"}),
        ];
        messages.extend(protected.as_array().unwrap().iter().cloned());
        let body = json!({"messages":messages,"max_tokens":200});
        let (request, omitted) = fit(&TestTokenizer, body, 1400).await.unwrap();
        let kept = request["messages"].as_array().unwrap();
        assert_eq!(omitted, 1);
        assert_eq!(&kept[kept.len() - 3..], protected.as_array().unwrap());
        let text = request.to_string();
        for expected in ["Keep citations", "six-slide", "section 3", "Blue headings"] {
            assert!(text.contains(expected), "lost {expected}");
        }
        assert!(!text.contains("compiler log"));
        assert_eq!(request["max_tokens"], 200);
    }

    #[tokio::test]
    async fn leaves_room_for_output_without_editing_the_current_turn() {
        let body = json!({"messages":[{"role":"user","content":"Review the supplied code"}]});
        let (request, omitted) = fit(&TestTokenizer, body.clone(), 65_536).await.unwrap();
        assert_eq!(omitted, 0);
        assert_eq!(request["messages"], body["messages"]);
        assert!(request["max_tokens"].as_u64().unwrap() > 8_192);

        let huge = json!({"messages":[{"role":"user","content":"let x = 5; ".repeat(6000)}]});
        let err = fit(&TestTokenizer, huge, 65_536).await.unwrap_err();
        assert!(err
            .to_string()
            .contains("Split the input by file or section"));
    }

    #[tokio::test]
    async fn an_overlong_agent_turn_omits_a_completed_tool_round_as_a_unit() {
        let request = json!({
            "max_tokens": 128,
            "messages": [
                {"role":"user","content":"Fix src/export.rs and keep the public interface."},
                {"role":"assistant","tool_calls":[{"id":"old-call"}]},
                {"role":"tool","tool_call_id":"old-call","content":"build log ".repeat(500)},
                {"role":"assistant","tool_calls":[{"id":"new-call"}]},
                {"role":"tool","tool_call_id":"new-call","content":"src/export.rs: missing test case"}
            ]
        });
        let (fitted, omitted) = fit(&TestTokenizer, request.clone(), 1280).await.unwrap();
        let original = request["messages"].as_array().unwrap();
        assert_eq!(
            fitted["messages"],
            json!([original[0], original[3], original[4]])
        );
        assert_eq!(omitted, 1);
    }
}
