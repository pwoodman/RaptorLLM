//! Chat conversations persisted as one JSON file per conversation.
//!
//! Ids are generated here as 32 lowercase hex characters and are validated on every
//! lookup, so an id coming from the UI can never name a path outside the store.

use crate::fsutil::{random_hex, unix_now, write_atomic};
use raptor_core::{RaptorError, Result, Role};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub const MAX_TITLE_CHARS: usize = 200;
pub const MAX_MESSAGE_BYTES: usize = 4 * 1024 * 1024;
const FILE_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoredMessage {
    pub id: String,
    pub role: Role,
    pub content: String,
    pub created_at: u64,
    /// The model's reasoning for an assistant reply. Stored apart from `content`, never counted as
    /// the answer, and absent from files written before it existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    /// Speed of an assistant reply as the gateway reported it: `tokens_per_second`,
    /// `time_to_first_token_ms`, `prompt_tokens_per_second`, `generated_tokens`. Absent when the
    /// server gave no timings and in older files.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stats: Option<serde_json::Value>,
}

/// A message to append; the store assigns the id and timestamp.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NewMessage {
    pub role: Role,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stats: Option<serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Conversation {
    pub version: u32,
    pub id: String,
    pub title: String,
    pub created_at: u64,
    pub updated_at: u64,
    pub messages: Vec<StoredMessage>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConversationSummary {
    pub id: String,
    pub title: String,
    pub created_at: u64,
    pub updated_at: u64,
    pub message_count: usize,
}

/// Listing result. Files that cannot be read are reported, not hidden.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ConversationList {
    pub conversations: Vec<ConversationSummary>,
    pub problems: Vec<String>,
}

#[derive(Clone)]
pub struct ConversationStore {
    dir: PathBuf,
}

impl ConversationStore {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    fn file_for(&self, id: &str) -> Result<PathBuf> {
        validate_id(id)?;
        Ok(self.dir.join(format!("{id}.json")))
    }

    fn write(&self, conv: &Conversation) -> Result<()> {
        let bytes = serde_json::to_vec_pretty(conv)?;
        write_atomic(&self.file_for(&conv.id)?, &bytes)
    }

    /// Creates an empty conversation.
    pub fn create(&self, title: &str) -> Result<Conversation> {
        let title = clean_title(title)?;
        let now = unix_now();
        let conv = Conversation {
            version: FILE_VERSION,
            id: new_id()?,
            title,
            created_at: now,
            updated_at: now,
            messages: Vec::new(),
        };
        self.write(&conv)?;
        Ok(conv)
    }

    pub fn get(&self, id: &str) -> Result<Conversation> {
        let path = self.file_for(id)?;
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(RaptorError::NotFound(format!("conversation {id}")));
            }
            Err(e) => return Err(e.into()),
        };
        parse(&bytes, id)
    }

    pub fn rename(&self, id: &str, title: &str) -> Result<Conversation> {
        let mut conv = self.get(id)?;
        conv.title = clean_title(title)?;
        conv.updated_at = unix_now();
        self.write(&conv)?;
        Ok(conv)
    }

    /// Appends messages in order. Empty content is refused.
    pub fn append(&self, id: &str, messages: Vec<NewMessage>) -> Result<Conversation> {
        if messages.is_empty() {
            return Err(RaptorError::InvalidInput("no messages to append".into()));
        }
        let mut conv = self.get(id)?;
        let now = unix_now();
        for m in messages {
            let reasoning_len = m.reasoning.as_ref().map_or(0, String::len);
            if m.content.len() > MAX_MESSAGE_BYTES || reasoning_len > MAX_MESSAGE_BYTES {
                return Err(RaptorError::InvalidInput(format!(
                    "message exceeds {MAX_MESSAGE_BYTES} bytes"
                )));
            }
            conv.messages.push(StoredMessage {
                id: new_id()?,
                role: m.role,
                content: m.content,
                created_at: now,
                reasoning: m.reasoning.filter(|r| !r.is_empty()),
                stats: m.stats,
            });
        }
        conv.updated_at = now;
        self.write(&conv)?;
        Ok(conv)
    }

    /// Truncates the conversation to keep the first `count` messages.
    pub fn truncate(&self, id: &str, count: usize) -> Result<Conversation> {
        let mut conv = self.get(id)?;
        if count > conv.messages.len() {
            return Err(RaptorError::InvalidInput(format!(
                "cannot truncate to {count} messages; conversation has only {}",
                conv.messages.len()
            )));
        }
        conv.messages.truncate(count);
        conv.updated_at = unix_now();
        self.write(&conv)?;
        Ok(conv)
    }

    /// Replaces the content of an existing message.
    pub fn replace_message(
        &self,
        id: &str,
        message_id: &str,
        content: &str,
    ) -> Result<Conversation> {
        if content.len() > MAX_MESSAGE_BYTES {
            return Err(RaptorError::InvalidInput(format!(
                "message exceeds {MAX_MESSAGE_BYTES} bytes"
            )));
        }
        let mut conv = self.get(id)?;
        let msg = conv
            .messages
            .iter_mut()
            .find(|m| m.id == message_id)
            .ok_or_else(|| RaptorError::NotFound(format!("message {message_id}")))?;
        msg.content = content.to_string();
        conv.updated_at = unix_now();
        self.write(&conv)?;
        Ok(conv)
    }

    pub fn delete(&self, id: &str) -> Result<()> {
        let path = self.file_for(id)?;
        match std::fs::remove_file(&path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                Err(RaptorError::NotFound(format!("conversation {id}")))
            }
            Err(e) => Err(e.into()),
        }
    }

    /// Lists conversations, newest first. Unreadable files appear in `problems`.
    pub fn list(&self) -> Result<ConversationList> {
        let mut conversations = Vec::new();
        let mut problems = Vec::new();
        let entries = match std::fs::read_dir(&self.dir) {
            Ok(e) => e,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok(ConversationList {
                    conversations,
                    problems,
                });
            }
            Err(e) => return Err(e.into()),
        };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let Some(id) = name.strip_suffix(".json") else {
                continue;
            };
            if validate_id(id).is_err() {
                continue;
            }
            match std::fs::read(entry.path())
                .map_err(RaptorError::from)
                .and_then(|b| parse(&b, id))
            {
                Ok(c) => conversations.push(ConversationSummary {
                    id: c.id,
                    title: c.title,
                    created_at: c.created_at,
                    updated_at: c.updated_at,
                    message_count: c.messages.len(),
                }),
                Err(e) => problems.push(format!("{name}: {e}")),
            }
        }
        conversations.sort_by(|a, b| b.updated_at.cmp(&a.updated_at).then(a.id.cmp(&b.id)));
        Ok(ConversationList {
            conversations,
            problems,
        })
    }

    /// Pretty JSON of one conversation, for the user to save.
    pub fn export(&self, id: &str) -> Result<String> {
        let conv = self.get(id)?;
        Ok(serde_json::to_string_pretty(&conv)?)
    }

    /// Imports a conversation exported by this app. Always stored under a new id so an
    /// import can never overwrite an existing conversation.
    pub fn import(&self, json: &str) -> Result<Conversation> {
        if json.len() > MAX_MESSAGE_BYTES * 4 {
            return Err(RaptorError::InvalidInput("import file is too large".into()));
        }
        let mut conv: Conversation = serde_json::from_str(json).map_err(|e| {
            RaptorError::InvalidInput(format!("not a Raptor LLM conversation: {e}"))
        })?;
        if conv.version != FILE_VERSION {
            return Err(RaptorError::Unsupported(format!(
                "conversation format version {} is not supported",
                conv.version
            )));
        }
        conv.title = clean_title(&conv.title)?;
        conv.id = new_id()?;
        for m in &mut conv.messages {
            if m.content.len() > MAX_MESSAGE_BYTES {
                return Err(RaptorError::InvalidInput(
                    "imported message is too large".into(),
                ));
            }
            m.id = new_id()?;
        }
        self.write(&conv)?;
        Ok(conv)
    }
}

/// Largest query accepted by [`ConversationStore::search`], in characters.
pub const SEARCH_MAX_QUERY_CHARS: usize = 200;
/// Default and largest number of hits returned by one search.
pub const SEARCH_DEFAULT_LIMIT: usize = 50;
pub const SEARCH_MAX_LIMIT: usize = 200;
/// Conversation bytes a single search may read. The newest conversations are read first, so a
/// search over a very large history reports `truncated` instead of stalling the app.
pub const SEARCH_BYTE_BUDGET: u64 = 256 * 1024 * 1024;
/// Context shown on each side of a match, in bytes before whitespace folding.
const SNIPPET_CONTEXT_BYTES: usize = 60;

/// One message that contains the query.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SearchHit {
    pub conversation_id: String,
    pub title: String,
    pub message_id: String,
    pub role: Role,
    /// Text around the first match in the message, whitespace collapsed, with `…` where cut.
    pub snippet: String,
    pub updated_at: u64,
}

/// Result of a content search across all conversations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ConversationSearch {
    /// At most one hit per message, newest conversation first.
    pub hits: Vec<SearchHit>,
    /// Conversations whose content was read.
    pub conversations_scanned: usize,
    /// True when the limit or the byte budget stopped the search before every conversation.
    pub truncated: bool,
    /// Files that could not be read, so they were not searched.
    pub problems: Vec<String>,
}

impl ConversationStore {
    /// Case-insensitive search of every message in every conversation. Results are bounded by
    /// `limit` and by [`SEARCH_BYTE_BUDGET`]; the newest conversations come first.
    pub fn search(&self, query: &str, limit: usize) -> Result<ConversationSearch> {
        let q = query.trim();
        if q.is_empty() || q.chars().count() > SEARCH_MAX_QUERY_CHARS {
            return Err(RaptorError::InvalidInput(format!(
                "query must be 1-{SEARCH_MAX_QUERY_CHARS} characters"
            )));
        }
        if !(1..=SEARCH_MAX_LIMIT).contains(&limit) {
            return Err(RaptorError::InvalidInput(format!(
                "limit must be 1-{SEARCH_MAX_LIMIT}"
            )));
        }
        let needle = Needle::new(q);
        let list_entries = match std::fs::read_dir(&self.dir) {
            Ok(e) => e,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok(ConversationSearch {
                    hits: Vec::new(),
                    conversations_scanned: 0,
                    truncated: false,
                    problems: Vec::new(),
                });
            }
            Err(e) => return Err(e.into()),
        };
        let mut problems = Vec::new();
        let mut loaded: Vec<Conversation> = Vec::new();
        let mut bytes_read: u64 = 0;
        let mut budget_hit = false;
        for entry in list_entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let Some(id) = name.strip_suffix(".json") else {
                continue;
            };
            if validate_id(id).is_err() {
                continue;
            }
            let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
            if bytes_read.saturating_add(size) > SEARCH_BYTE_BUDGET {
                budget_hit = true;
                continue;
            }
            bytes_read += size;
            match std::fs::read(entry.path())
                .map_err(RaptorError::from)
                .and_then(|b| parse(&b, id))
            {
                Ok(c) => loaded.push(c),
                Err(e) => problems.push(format!("{name}: {e}")),
            }
        }
        loaded.sort_by(|a, b| b.updated_at.cmp(&a.updated_at).then(a.id.cmp(&b.id)));

        let mut hits = Vec::new();
        let mut scanned = 0usize;
        let mut limit_hit = false;
        'outer: for conv in &loaded {
            scanned += 1;
            for msg in &conv.messages {
                let Some((start, end)) = needle.find(&msg.content) else {
                    continue;
                };
                if hits.len() == limit {
                    limit_hit = true;
                    break 'outer;
                }
                hits.push(SearchHit {
                    conversation_id: conv.id.clone(),
                    title: conv.title.clone(),
                    message_id: msg.id.clone(),
                    role: msg.role,
                    snippet: snippet(&msg.content, start, end),
                    updated_at: conv.updated_at,
                });
            }
        }
        Ok(ConversationSearch {
            hits,
            conversations_scanned: scanned,
            truncated: limit_hit || budget_hit,
            problems,
        })
    }

    /// Writes the export of conversation `id` to `path`, atomically. The path must be absolute
    /// and end in `.json`. Returns the number of bytes written.
    pub fn export_to(&self, id: &str, path: &std::path::Path) -> Result<u64> {
        if !path.is_absolute() {
            return Err(RaptorError::InvalidInput(
                "export path must be absolute".into(),
            ));
        }
        let is_json = path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("json"));
        if !is_json {
            return Err(RaptorError::InvalidInput(
                "export file must end in .json".into(),
            ));
        }
        let text = self.export(id)?;
        write_atomic(path, text.as_bytes())?;
        Ok(text.len() as u64)
    }
}

/// A search term, folded for case-insensitive matching. ASCII text takes a byte path; other
/// text is compared one character at a time with its first lowercase form.
struct Needle {
    ascii: Option<Vec<u8>>,
    folded: Vec<char>,
}

fn fold(c: char) -> char {
    c.to_lowercase().next().unwrap_or(c)
}

impl Needle {
    fn new(q: &str) -> Self {
        let ascii = q
            .is_ascii()
            .then(|| q.bytes().map(|b| b.to_ascii_lowercase()).collect());
        Self {
            ascii,
            folded: q.chars().map(fold).collect(),
        }
    }

    /// Byte range of the first match in `text`.
    fn find(&self, text: &str) -> Option<(usize, usize)> {
        if let Some(needle) = &self.ascii {
            if !text.is_ascii() {
                return self.find_folded(text);
            }
            let hay = text.as_bytes();
            if needle.len() > hay.len() {
                return None;
            }
            return hay
                .windows(needle.len())
                .position(|w| w.eq_ignore_ascii_case(needle))
                .map(|i| (i, i + needle.len()));
        }
        self.find_folded(text)
    }

    fn find_folded(&self, text: &str) -> Option<(usize, usize)> {
        let n = self.folded.len();
        let chars: Vec<(usize, usize, char)> = text
            .char_indices()
            .map(|(i, c)| (i, i + c.len_utf8(), fold(c)))
            .collect();
        if n == 0 || chars.len() < n {
            return None;
        }
        (0..=chars.len() - n).find_map(|start| {
            let matched = self
                .folded
                .iter()
                .zip(&chars[start..start + n])
                .all(|(q, (_, _, c))| q == c);
            matched.then(|| (chars[start].0, chars[start + n - 1].1))
        })
    }
}

/// The text around a match, cut on character boundaries with whitespace collapsed.
fn snippet(content: &str, start: usize, end: usize) -> String {
    let mut from = start.saturating_sub(SNIPPET_CONTEXT_BYTES);
    while from > 0 && !content.is_char_boundary(from) {
        from -= 1;
    }
    let mut to = (end + SNIPPET_CONTEXT_BYTES).min(content.len());
    while to < content.len() && !content.is_char_boundary(to) {
        to += 1;
    }
    let body = content[from..to]
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    let lead = if from > 0 { "…" } else { "" };
    let tail = if to < content.len() { "…" } else { "" };
    format!("{lead}{body}{tail}")
}

fn parse(bytes: &[u8], id: &str) -> Result<Conversation> {
    let conv: Conversation = serde_json::from_slice(bytes)
        .map_err(|e| RaptorError::Integrity(format!("conversation {id} is corrupt: {e}")))?;
    if conv.id != id {
        return Err(RaptorError::Integrity(format!(
            "conversation file {id} contains id {}",
            conv.id
        )));
    }
    Ok(conv)
}

/// Accepts only the 32-character lowercase hex ids this store generates.
pub fn validate_id(id: &str) -> Result<()> {
    if id.len() == 32
        && id
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        Ok(())
    } else {
        Err(RaptorError::InvalidInput("invalid conversation id".into()))
    }
}

fn new_id() -> Result<String> {
    random_hex(16)
}

fn clean_title(title: &str) -> Result<String> {
    let t = title.trim();
    if t.is_empty() {
        return Err(RaptorError::InvalidInput("title must not be empty".into()));
    }
    if t.chars().count() > MAX_TITLE_CHARS {
        return Err(RaptorError::InvalidInput(format!(
            "title exceeds {MAX_TITLE_CHARS} characters"
        )));
    }
    if t.chars().any(char::is_control) {
        return Err(RaptorError::InvalidInput(
            "title contains control characters".into(),
        ));
    }
    Ok(t.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> (tempfile::TempDir, ConversationStore) {
        let dir = tempfile::tempdir().unwrap();
        let s = ConversationStore::new(dir.path().join("conversations"));
        (dir, s)
    }

    /// Reasoning is stored beside the answer, never inside it, and old files still load.
    #[test]
    fn reasoning_is_a_separate_optional_field() {
        let (_d, s) = store();
        let c = s.create("thinking").unwrap();
        let c = s
            .append(
                &c.id,
                vec![NewMessage {
                    role: Role::Assistant,
                    content: "Answer".into(),
                    reasoning: Some("Working it out".into()),
                    stats: None,
                }],
            )
            .unwrap();
        assert_eq!(c.messages[0].content, "Answer");
        assert_eq!(c.messages[0].reasoning.as_deref(), Some("Working it out"));
        let reloaded = s.get(&c.id).unwrap();
        assert_eq!(
            reloaded.messages[0].reasoning.as_deref(),
            Some("Working it out")
        );

        // An empty reasoning string is not stored, and a file without the field loads with None.
        let c2 = s
            .append(
                &c.id,
                vec![NewMessage {
                    role: Role::Assistant,
                    content: "Plain".into(),
                    reasoning: Some(String::new()),
                    stats: None,
                }],
            )
            .unwrap();
        assert_eq!(c2.messages[1].reasoning, None);
        let legacy: Conversation = serde_json::from_str(
            r#"{"version":1,"id":"x","title":"t","created_at":0,"updated_at":0,
                "messages":[{"id":"m","role":"assistant","content":"old","created_at":0}]}"#,
        )
        .unwrap();
        assert_eq!(legacy.messages[0].reasoning, None);
    }

    #[test]
    fn create_append_rename_and_list() {
        let (_d, s) = store();
        let c = s.create("  First chat ").unwrap();
        assert_eq!(c.title, "First chat");
        let c = s
            .append(
                &c.id,
                vec![
                    NewMessage {
                        role: Role::User,
                        content: "hi".into(),
                        reasoning: None,
                        stats: None,
                    },
                    NewMessage {
                        role: Role::Assistant,
                        content: "hello".into(),
                        reasoning: None,
                        stats: None,
                    },
                ],
            )
            .unwrap();
        assert_eq!(c.messages.len(), 2);
        assert_eq!(c.messages[1].role, Role::Assistant);
        let c = s.rename(&c.id, "Renamed").unwrap();
        assert_eq!(c.title, "Renamed");
        let list = s.list().unwrap();
        assert!(list.problems.is_empty());
        assert_eq!(list.conversations.len(), 1);
        assert_eq!(list.conversations[0].message_count, 2);
    }

    #[test]
    fn delete_removes_and_second_delete_is_not_found() {
        let (_d, s) = store();
        let c = s.create("x").unwrap();
        s.delete(&c.id).unwrap();
        assert!(matches!(s.get(&c.id), Err(RaptorError::NotFound(_))));
        assert!(matches!(s.delete(&c.id), Err(RaptorError::NotFound(_))));
    }

    #[test]
    fn hostile_ids_are_rejected_before_touching_disk() {
        let (_d, s) = store();
        for bad in [
            "../../etc/passwd",
            "..\\x",
            "",
            "ABCDEF0123456789abcdef0123456789",
            "a/b",
        ] {
            assert!(
                matches!(s.get(bad), Err(RaptorError::InvalidInput(_))),
                "{bad}"
            );
            assert!(s.delete(bad).is_err());
        }
    }

    #[test]
    fn corrupt_file_is_reported_by_list_and_get() {
        let (d, s) = store();
        let c = s.create("ok").unwrap();
        let bad_id = "0123456789abcdef0123456789abcdef";
        std::fs::create_dir_all(d.path().join("conversations")).unwrap();
        std::fs::write(
            d.path()
                .join("conversations")
                .join(format!("{bad_id}.json")),
            b"{oops",
        )
        .unwrap();
        let list = s.list().unwrap();
        assert_eq!(list.conversations.len(), 1);
        assert_eq!(list.conversations[0].id, c.id);
        assert_eq!(list.problems.len(), 1);
        assert!(matches!(s.get(bad_id), Err(RaptorError::Integrity(_))));
    }

    #[test]
    fn export_then_import_gets_a_new_id_and_keeps_messages() {
        let (_d, s) = store();
        let c = s.create("orig").unwrap();
        s.append(
            &c.id,
            vec![NewMessage {
                role: Role::User,
                content: "q".into(),
                reasoning: None,
                stats: None,
            }],
        )
        .unwrap();
        let json = s.export(&c.id).unwrap();
        let imported = s.import(&json).unwrap();
        assert_ne!(imported.id, c.id);
        assert_eq!(imported.messages.len(), 1);
        assert_eq!(imported.messages[0].content, "q");
        assert_eq!(s.list().unwrap().conversations.len(), 2);
    }

    #[test]
    fn import_rejects_foreign_json_and_control_characters() {
        let (_d, s) = store();
        assert!(matches!(
            s.import("{\"x\":1}"),
            Err(RaptorError::InvalidInput(_))
        ));
        let c = s.create("t").unwrap();
        let mut json: serde_json::Value = serde_json::from_str(&s.export(&c.id).unwrap()).unwrap();
        json["title"] = serde_json::Value::String("bad\u{7}title".into());
        assert!(s.import(&json.to_string()).is_err());
    }

    #[test]
    fn search_is_case_insensitive_and_returns_message_ids_with_snippets() {
        let (_d, s) = store();
        let c = s.create("Billing").unwrap();
        let c = s
            .append(
                &c.id,
                vec![
                    NewMessage {
                        role: Role::User,
                        content: "How do I issue a REFUND\nfor a partial order?".into(),
                        reasoning: None,
                        stats: None,
                    },
                    NewMessage {
                        role: Role::Assistant,
                        content: "Call the refund endpoint.".into(),
                        reasoning: None,
                        stats: None,
                    },
                ],
            )
            .unwrap();
        let found = s.search("refund", 10).unwrap();
        assert_eq!(found.hits.len(), 2);
        assert_eq!(found.hits[0].message_id, c.messages[0].id);
        assert_eq!(found.hits[0].conversation_id, c.id);
        assert!(found.hits[0].snippet.contains("REFUND for a partial"));
        assert_eq!(found.conversations_scanned, 1);
        assert!(!found.truncated);
        assert!(s.search("nothing-here", 10).unwrap().hits.is_empty());
        assert!(matches!(
            s.search("  ", 10),
            Err(RaptorError::InvalidInput(_))
        ));
        assert!(matches!(
            s.search("x", 0),
            Err(RaptorError::InvalidInput(_))
        ));
        assert!(matches!(
            s.search(&"x".repeat(SEARCH_MAX_QUERY_CHARS + 1), 10),
            Err(RaptorError::InvalidInput(_))
        ));
    }

    #[test]
    fn search_folds_non_ascii_text_and_cuts_snippets_on_char_boundaries() {
        let (_d, s) = store();
        let c = s.create("Ünïcode").unwrap();
        let long = format!("{} ÉCOLE {}", "é".repeat(80), "ß".repeat(80));
        let c = s
            .append(
                &c.id,
                vec![NewMessage {
                    role: Role::User,
                    content: long,
                    reasoning: None,
                    stats: None,
                }],
            )
            .unwrap();
        let found = s.search("école", 5).unwrap();
        assert_eq!(found.hits.len(), 1, "{found:?}");
        assert!(found.hits[0].snippet.contains("ÉCOLE"));
        assert!(found.hits[0].snippet.starts_with('…'));
        assert_eq!(c.messages.len(), 1);
    }

    #[test]
    fn search_is_bounded_by_limit_and_reads_many_conversations_quickly() {
        let (_d, s) = store();
        // 400 conversations, each with 5 messages, of which every third mentions the term.
        for i in 0..400 {
            let c = s.create(&format!("chat {i}")).unwrap();
            let messages: Vec<NewMessage> = (0..5)
                .map(|m| NewMessage {
                    role: Role::User,
                    content: if (i + m) % 3 == 0 {
                        format!("message {m} of chat {i} mentions the Needle term")
                    } else {
                        format!("message {m} of chat {i} is about something else entirely")
                    },
                    reasoning: None,
                    stats: None,
                })
                .collect();
            s.append(&c.id, messages).unwrap();
        }
        let started = std::time::Instant::now();
        let all = s.search("needle", SEARCH_MAX_LIMIT).unwrap();
        let elapsed = started.elapsed();
        assert_eq!(all.hits.len(), SEARCH_MAX_LIMIT);
        assert!(all.truncated, "the limit stopped the search");
        assert!(all.conversations_scanned <= 400);
        assert!(elapsed.as_secs() < 10, "search took {elapsed:?}");

        let few = s.search("needle", 3).unwrap();
        assert_eq!(few.hits.len(), 3);
        assert!(few.truncated);

        let none = s.search("zzz-absent", 50).unwrap();
        assert_eq!(none.conversations_scanned, 400);
        assert!(!none.truncated);
    }

    #[test]
    fn export_to_writes_json_atomically_and_refuses_bad_paths() {
        let (d, s) = store();
        let c = s.create("Export me").unwrap();
        let target = d.path().join("out").join("chat.json");
        let bytes = s.export_to(&c.id, &target).unwrap();
        let written = std::fs::read_to_string(&target).unwrap();
        assert_eq!(written.len() as u64, bytes);
        assert!(written.contains("Export me"));
        assert!(s.export_to(&c.id, &d.path().join("chat.txt")).is_err());
        assert!(s
            .export_to(&c.id, std::path::Path::new("relative.json"))
            .is_err());
        assert!(matches!(
            s.export_to("0".repeat(32).as_str(), &d.path().join("x.json")),
            Err(RaptorError::NotFound(_))
        ));
    }

    #[test]
    fn empty_title_and_empty_append_are_refused() {
        let (_d, s) = store();
        assert!(s.create("   ").is_err());
        let c = s.create("ok").unwrap();
        assert!(s.append(&c.id, vec![]).is_err());
    }

    #[test]
    fn truncate_and_replace_message() {
        let (_d, s) = store();
        let c = s.create("trunc test").unwrap();
        let c = s
            .append(
                &c.id,
                vec![
                    NewMessage {
                        role: Role::User,
                        content: "hello".into(),
                        reasoning: None,
                        stats: None,
                    },
                    NewMessage {
                        role: Role::Assistant,
                        content: "reply 1".into(),
                        reasoning: None,
                        stats: None,
                    },
                    NewMessage {
                        role: Role::User,
                        content: "again".into(),
                        reasoning: None,
                        stats: None,
                    },
                    NewMessage {
                        role: Role::Assistant,
                        content: "reply 2".into(),
                        reasoning: None,
                        stats: None,
                    },
                ],
            )
            .unwrap();
        assert_eq!(c.messages.len(), 4);

        // Replace reply 1
        let msg_id = c.messages[1].id.clone();
        let c = s
            .replace_message(&c.id, &msg_id, "replaced reply 1")
            .unwrap();
        assert_eq!(c.messages[1].content, "replaced reply 1");

        // Truncate to 3 messages (dropping reply 2)
        let c = s.truncate(&c.id, 3).unwrap();
        assert_eq!(c.messages.len(), 3);
        assert_eq!(c.messages[2].content, "again");

        // Truncate out of bounds is rejected
        assert!(matches!(
            s.truncate(&c.id, 10),
            Err(RaptorError::InvalidInput(_))
        ));
    }
}
