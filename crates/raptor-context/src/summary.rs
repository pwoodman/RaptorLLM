//! Model-backed summaries of conversation turns.
//!
//! The model writes only the prose summary. Structured items never go through it:
//! `Decision:` and `Convention:` lines are removed from the transcript before the prompt is
//! built, and they are recorded verbatim by the engine from the original messages. Failed
//! attempts are recorded the same way. Whatever the model returns is bounded in length and
//! stripped of those marker lines, so a model cannot rewrite a decision.

use crate::memory::{extract_decisions, looks_like_error, Summarizer};
use raptor_core::{ChatMessage, RaptorError, Result, Role};

/// Longest summary kept, in characters.
pub const MAX_SUMMARY_CHARS: usize = 1200;
/// The newest part of the transcript that is sent to the model, in characters.
pub const MAX_TRANSCRIPT_CHARS: usize = 12_000;
/// Output token cap passed to the generator.
pub const SUMMARY_MAX_TOKENS: u32 = 320;
/// Failed attempts recorded per summarisation.
pub const MAX_FAILED_ATTEMPTS: usize = 20;
/// Longest failed-attempt line kept, in characters.
pub const MAX_FAILED_CHARS: usize = 300;

/// Text generation. The shell implements it over llama-server chat; tests use a fake.
pub trait TextGenerator: Send + Sync {
    /// Generates a reply to `prompt`, of at most about `max_tokens` tokens.
    fn generate(&self, prompt: &str, max_tokens: u32) -> Result<String>;
}

/// A prepared summary request: the prompt, and the bound the reply must respect.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SummaryRequest {
    pub prompt: String,
    pub max_chars: usize,
}

/// True for a line that carries a structured item (`Decision:` or `Convention:`).
fn is_marker_line(line: &str) -> bool {
    let lower = line.trim().to_ascii_lowercase();
    lower.starts_with("decision:") || lower.starts_with("convention:")
}

/// Builds the prompt for `messages`. Decision and convention lines are left out of the
/// transcript, and the newest [`MAX_TRANSCRIPT_CHARS`] characters are kept.
pub fn build_summary_request(messages: &[ChatMessage]) -> SummaryRequest {
    let mut lines: Vec<String> = Vec::new();
    for m in messages.iter().filter(|m| m.role != Role::System) {
        let role = match m.role {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
            Role::System => continue,
        };
        for line in m.content.lines().filter(|l| !is_marker_line(l)) {
            if !line.trim().is_empty() {
                lines.push(format!("[{role}] {line}"));
            }
        }
    }
    let mut transcript = lines.join("\n");
    let total = transcript.chars().count();
    let mut omitted = false;
    if total > MAX_TRANSCRIPT_CHARS {
        let skip = total - MAX_TRANSCRIPT_CHARS;
        transcript = transcript.chars().skip(skip).collect();
        omitted = true;
    }
    let note = if omitted {
        "(earlier turns were omitted for length)\n"
    } else {
        ""
    };
    let prompt = format!(
        "You summarise a coding session so that a later assistant can continue it. \
Write at most {MAX_SUMMARY_CHARS} characters of plain prose: what was done, which files changed, \
what is still open, and what did not work. Do not restate decisions or conventions; they are \
recorded separately. Do not invent details that are not in the conversation.\n\n\
Conversation:\n{note}{transcript}\n\nSummary:"
    );
    SummaryRequest {
        prompt,
        max_chars: MAX_SUMMARY_CHARS,
    }
}

/// Cleans a model reply: drops any marker lines, trims, and bounds the length at a character
/// boundary. An empty reply is an error, so the caller records nothing for it.
pub fn parse_summary_response(raw: &str, max_chars: usize) -> Result<String> {
    let kept: Vec<&str> = raw.lines().filter(|l| !is_marker_line(l)).collect();
    let text = kept.join("\n");
    let text = text.trim();
    if text.is_empty() {
        return Err(RaptorError::Runtime(
            "summary model returned no text".into(),
        ));
    }
    if text.chars().count() <= max_chars {
        return Ok(text.to_string());
    }
    let cut: String = text.chars().take(max_chars.saturating_sub(3)).collect();
    Ok(format!("{}...", cut.trim_end()))
}

/// Failed attempts in the conversation: the first line of each tool output that reports a
/// failure. Duplicates are dropped, the newest [`MAX_FAILED_ATTEMPTS`] are kept, and each line
/// is cut to [`MAX_FAILED_CHARS`].
pub fn failed_attempts(messages: &[ChatMessage]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for m in messages
        .iter()
        .filter(|m| m.role == Role::Tool && looks_like_error(&m.content))
    {
        let Some(first) = m.content.lines().map(str::trim).find(|l| !l.is_empty()) else {
            continue;
        };
        let line: String = first.chars().take(MAX_FAILED_CHARS).collect();
        if !out.contains(&line) {
            out.push(line);
        }
    }
    let excess = out.len().saturating_sub(MAX_FAILED_ATTEMPTS);
    out.drain(..excess);
    out
}

/// Decisions and conventions in `messages`, as the engine records them. Exposed so callers
/// can check what will be kept verbatim.
pub fn verbatim_items(messages: &[ChatMessage]) -> Vec<(crate::memory::MemoryKind, String)> {
    extract_decisions(messages)
}

/// A [`Summarizer`] that asks a [`TextGenerator`] for the prose summary.
pub struct ModelSummarizer<T: TextGenerator> {
    generator: T,
    max_tokens: u32,
}

impl<T: TextGenerator> ModelSummarizer<T> {
    /// A summarizer over `generator`, using [`SUMMARY_MAX_TOKENS`].
    pub fn new(generator: T) -> Self {
        Self {
            generator,
            max_tokens: SUMMARY_MAX_TOKENS,
        }
    }
}

impl<T: TextGenerator> Summarizer for ModelSummarizer<T> {
    fn summarize(&self, messages: &[ChatMessage]) -> Result<String> {
        let req = build_summary_request(messages);
        let raw = self.generator.generate(&req.prompt, self.max_tokens)?;
        parse_summary_response(&raw, req.max_chars)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::MemoryKind;
    use std::sync::Mutex;

    /// Records the prompts it receives and replies with a fixed string, or fails.
    struct Fake {
        reply: Result<String>,
        seen: Mutex<Vec<String>>,
    }

    impl TextGenerator for Fake {
        fn generate(&self, prompt: &str, _max_tokens: u32) -> Result<String> {
            if let Ok(mut seen) = self.seen.lock() {
                seen.push(prompt.to_string());
            }
            match &self.reply {
                Ok(s) => Ok(s.clone()),
                Err(e) => Err(RaptorError::Network(e.to_string())),
            }
        }
    }

    fn fake(reply: Result<String>) -> Fake {
        Fake {
            reply,
            seen: Mutex::new(Vec::new()),
        }
    }

    fn convo() -> Vec<ChatMessage> {
        vec![
            ChatMessage::new(Role::System, "be brief"),
            ChatMessage::new(
                Role::Assistant,
                "Decision: keep refunds in ledger.rs\nConvention: use snake_case\nChanged src/ledger.rs.",
            ),
            ChatMessage::new(Role::Tool, "test refund_ok ... FAILED\nassertion failed"),
            ChatMessage::new(Role::Tool, "test refund_ok ... FAILED\nassertion failed"),
        ]
    }

    #[test]
    fn prompt_leaves_out_structured_lines_and_system_messages() {
        let req = build_summary_request(&convo());
        assert!(!req.prompt.contains("keep refunds in ledger.rs"));
        assert!(!req.prompt.contains("use snake_case"));
        assert!(!req.prompt.contains("be brief"));
        assert!(req.prompt.contains("Changed src/ledger.rs."));
        assert_eq!(req.max_chars, MAX_SUMMARY_CHARS);
    }

    #[test]
    fn prompt_keeps_only_the_newest_transcript() {
        let big: Vec<ChatMessage> = (0..2000)
            .map(|i| ChatMessage::new(Role::User, format!("turn {i} {}", "x".repeat(20))))
            .collect();
        let req = build_summary_request(&big);
        assert!(req.prompt.contains("earlier turns were omitted"));
        assert!(req.prompt.contains("turn 1999"));
        assert!(!req.prompt.contains("turn 0 "));
        assert!(req.prompt.chars().count() < MAX_TRANSCRIPT_CHARS + 1000);
    }

    #[test]
    fn response_drops_marker_lines_and_is_bounded() {
        let raw = "Decision: model-chosen rewrite\nWe changed the ledger.\n";
        assert_eq!(
            parse_summary_response(raw, 100).expect("ok"),
            "We changed the ledger."
        );
        let long = "é".repeat(50);
        let cut = parse_summary_response(&long, 10).expect("ok");
        assert_eq!(cut.chars().count(), 10);
        assert!(cut.ends_with("..."));
        assert!(parse_summary_response("Decision: only this\n  \n", 100).is_err());
    }

    #[test]
    fn failed_attempts_are_deduplicated_and_bounded() {
        let attempts = failed_attempts(&convo());
        assert_eq!(attempts, vec!["test refund_ok ... FAILED".to_string()]);
        let many: Vec<ChatMessage> = (0..30)
            .map(|i| ChatMessage::new(Role::Tool, format!("error {i}")))
            .collect();
        let kept = failed_attempts(&many);
        assert_eq!(kept.len(), MAX_FAILED_ATTEMPTS);
        assert_eq!(kept.last().map(String::as_str), Some("error 29"));
    }

    #[test]
    fn model_summarizer_returns_the_bounded_reply() {
        let s = ModelSummarizer::new(fake(Ok("Refunds now go through the ledger.".into())));
        assert_eq!(
            s.summarize(&convo()).expect("summary"),
            "Refunds now go through the ledger."
        );
    }

    #[test]
    fn generator_failure_surfaces_as_an_error() {
        let s = ModelSummarizer::new(fake(Err(RaptorError::Network("down".into()))));
        assert!(s.summarize(&convo()).is_err());
    }

    #[test]
    fn verbatim_items_are_the_structured_lines() {
        let items = verbatim_items(&convo());
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].0, MemoryKind::Decision);
        assert_eq!(items[0].1, "keep refunds in ledger.rs");
    }
}
