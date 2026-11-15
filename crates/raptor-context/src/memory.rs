//! Model-independent conversation memory.
//!
//! Entries record what was decided, agreed, summarised or tried, together with the
//! hashes of the files they describe. When a file changes, entries that describe it
//! are marked stale: they are reported in the manifest but never served as current.

use crate::text::normalize_signature;
use raptor_core::paths::safe_file_name;
use raptor_core::{ChatMessage, RaptorError, Result, Role};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

/// Longest text accepted for one memory entry.
pub const MAX_ENTRY_CHARS: usize = 4000;
/// An output that repeats this many times is reported as a loop.
const REPEAT_THRESHOLD: usize = 3;

/// What a memory entry records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryKind {
    /// A decision made during the session. Protected: kept verbatim, never summarised away.
    Decision,
    /// A project convention agreed in the session. Protected.
    Convention,
    /// A summary of earlier conversation.
    Summary,
    /// Work that was completed.
    CompletedWork,
    /// An approach that was tried and did not work.
    FailedAttempt,
}

impl MemoryKind {
    /// Decisions and conventions are always included in the context.
    pub fn is_protected(self) -> bool {
        matches!(self, MemoryKind::Decision | MemoryKind::Convention)
    }
}

/// A file an entry depends on, with the content hash at recording time.
/// An empty hash means the file did not exist when the entry was recorded.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceRef {
    pub path: String,
    pub hash: String,
}

/// One remembered item.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub id: String,
    pub kind: MemoryKind,
    pub text: String,
    pub sources: Vec<SourceRef>,
    pub stale: bool,
    /// Why the entry was marked stale, when it was.
    pub stale_reason: Option<String>,
    /// Insertion order within the store.
    pub seq: u64,
}

/// The memory of one session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryStore {
    pub session: String,
    next_seq: u64,
    pub entries: Vec<MemoryEntry>,
}

impl MemoryStore {
    /// An empty store for `session`. The session id must be a safe file name.
    pub fn new(session: &str) -> Result<Self> {
        safe_file_name(session)?;
        Ok(Self {
            session: session.to_string(),
            next_seq: 0,
            entries: Vec::new(),
        })
    }

    /// Loads a store from `path`, or returns an empty store when the file does not exist.
    pub fn load_or_new(path: &Path, session: &str) -> Result<Self> {
        if !path.is_file() {
            return Self::new(session);
        }
        let bytes = std::fs::read(path)?;
        let store: MemoryStore = serde_json::from_slice(&bytes).map_err(|e| {
            RaptorError::Integrity(format!(
                "memory store {} is unreadable: {e}",
                path.display()
            ))
        })?;
        if store.session != session {
            return Err(RaptorError::Integrity(format!(
                "memory store {} belongs to another session",
                path.display()
            )));
        }
        Ok(store)
    }

    /// Writes the store atomically.
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec(self)?)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Records an entry and returns its id.
    pub fn add(&mut self, kind: MemoryKind, text: &str, sources: Vec<SourceRef>) -> Result<String> {
        let text = text.trim();
        if text.is_empty() {
            return Err(RaptorError::InvalidInput("memory text is empty".into()));
        }
        if text.chars().count() > MAX_ENTRY_CHARS {
            return Err(RaptorError::InvalidInput(format!(
                "memory text exceeds {MAX_ENTRY_CHARS} characters"
            )));
        }
        let seq = self.next_seq;
        self.next_seq += 1;
        let id = format!("mem-{seq:06}");
        self.entries.push(MemoryEntry {
            id: id.clone(),
            kind,
            text: text.to_string(),
            sources,
            stale: false,
            stale_reason: None,
            seq,
        });
        Ok(id)
    }

    /// Marks entries stale when a source file's hash differs from `current`, or the file is gone.
    ///
    /// Returns the ids newly marked stale. Entries stay stale: re-recording the
    /// information is the way to make it current again.
    pub fn invalidate_stale(&mut self, current: &BTreeMap<String, String>) -> Vec<String> {
        mark_stale(&mut self.entries, current)
    }

    /// Entries that may be served as current.
    pub fn current(&self) -> impl Iterator<Item = &MemoryEntry> {
        self.entries.iter().filter(|e| !e.stale)
    }

    /// Entries marked stale.
    pub fn stale(&self) -> impl Iterator<Item = &MemoryEntry> {
        self.entries.iter().filter(|e| e.stale)
    }
}

/// Marks each current entry stale when one of its sources changed or disappeared, and returns
/// the ids newly marked. Shared by session and project memory.
fn mark_stale(entries: &mut [MemoryEntry], current: &BTreeMap<String, String>) -> Vec<String> {
    let mut newly = Vec::new();
    for entry in entries.iter_mut().filter(|e| !e.stale) {
        let reason = entry
            .sources
            .iter()
            .find_map(|src| match current.get(&src.path) {
                None => Some(format!("source {} no longer exists", src.path)),
                Some(hash) if *hash != src.hash => Some(format!(
                    "source {} changed since this entry was recorded",
                    src.path
                )),
                Some(_) => None,
            });
        if let Some(reason) = reason {
            entry.stale = true;
            entry.stale_reason = Some(reason);
            newly.push(entry.id.clone());
        }
    }
    newly
}

/// Upper bound on entries in one workspace's project memory.
pub const MAX_PROJECT_ENTRIES: usize = 500;
/// Upper bound on decisions and conventions. These are never evicted automatically, so a
/// new one is refused past this count and the caller must forget one first.
pub const MAX_PROTECTED_ENTRIES: usize = 200;
/// Version of the project memory file format.
pub const PROJECT_MEMORY_VERSION: u32 = 1;

/// What [`ProjectMemory::add`] did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Added {
    pub id: String,
    /// Entries evicted to stay within [`MAX_PROJECT_ENTRIES`], least valuable first.
    pub evicted: Vec<String>,
}

#[derive(Serialize, Deserialize)]
struct ProjectFile {
    version: u32,
    next_seq: u64,
    entries: Vec<MemoryEntry>,
}

/// Long-term memory of one workspace: decisions, conventions, summaries and failed attempts
/// that outlive a session. Stored as one versioned JSON file, written atomically.
///
/// Stale detection is the same as session memory: an entry is stale when a file it cites has
/// changed. Eviction, when the store is full, removes stale entries first, then failed
/// attempts, summaries and completed work, oldest first. Decisions and conventions are never
/// evicted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectMemory {
    next_seq: u64,
    entries: Vec<MemoryEntry>,
}

impl Default for ProjectMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl ProjectMemory {
    /// An empty store.
    pub fn new() -> Self {
        Self {
            next_seq: 0,
            entries: Vec::new(),
        }
    }

    /// Loads the store at `path`, or returns an empty store when the file does not exist.
    ///
    /// A file that cannot be read or has another version is an error, not a reset: it may
    /// hold decisions the user relies on.
    pub fn load_or_new(path: &Path) -> Result<Self> {
        if !path.is_file() {
            return Ok(Self::new());
        }
        let bytes = std::fs::read(path)?;
        let file: ProjectFile = serde_json::from_slice(&bytes).map_err(|e| {
            RaptorError::Integrity(format!(
                "project memory {} is unreadable ({e}); move it aside to reset",
                path.display()
            ))
        })?;
        if file.version != PROJECT_MEMORY_VERSION {
            return Err(RaptorError::Integrity(format!(
                "project memory {} is version {}, expected {PROJECT_MEMORY_VERSION}",
                path.display(),
                file.version
            )));
        }
        Ok(Self {
            next_seq: file.next_seq,
            entries: file.entries,
        })
    }

    /// Writes the store atomically (temp file, then rename).
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = ProjectFile {
            version: PROJECT_MEMORY_VERSION,
            next_seq: self.next_seq,
            entries: self.entries.clone(),
        };
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec(&file)?)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Records an entry. Evicts low-value entries when the store is over its cap.
    pub fn add(&mut self, kind: MemoryKind, text: &str, sources: Vec<SourceRef>) -> Result<Added> {
        let text = text.trim();
        if text.is_empty() {
            return Err(RaptorError::InvalidInput("memory text is empty".into()));
        }
        if text.chars().count() > MAX_ENTRY_CHARS {
            return Err(RaptorError::InvalidInput(format!(
                "memory text exceeds {MAX_ENTRY_CHARS} characters"
            )));
        }
        if kind.is_protected() && self.protected_count() >= MAX_PROTECTED_ENTRIES {
            return Err(RaptorError::Conflict(format!(
                "project memory already holds {MAX_PROTECTED_ENTRIES} decisions and conventions; forget one first"
            )));
        }
        let seq = self.next_seq;
        self.next_seq += 1;
        let id = format!("proj-{seq:06}");
        self.entries.push(MemoryEntry {
            id: id.clone(),
            kind,
            text: text.to_string(),
            sources,
            stale: false,
            stale_reason: None,
            seq,
        });
        let evicted = self.evict_to_cap();
        Ok(Added { id, evicted })
    }

    /// Removes an entry by id, including a decision or convention. This is the only way a
    /// protected entry leaves the store.
    pub fn forget(&mut self, id: &str) -> Result<()> {
        let before = self.entries.len();
        self.entries.retain(|e| e.id != id);
        if self.entries.len() == before {
            return Err(RaptorError::NotFound(format!(
                "no project memory entry {id}"
            )));
        }
        Ok(())
    }

    /// Marks entries stale when a cited file changed or disappeared. Returns the newly stale ids.
    pub fn invalidate_stale(&mut self, current: &BTreeMap<String, String>) -> Vec<String> {
        mark_stale(&mut self.entries, current)
    }

    /// Entries that may be served as current.
    pub fn current(&self) -> impl Iterator<Item = &MemoryEntry> {
        self.entries.iter().filter(|e| !e.stale)
    }

    /// Entries marked stale.
    pub fn stale(&self) -> impl Iterator<Item = &MemoryEntry> {
        self.entries.iter().filter(|e| e.stale)
    }

    /// All entries, stale or not, in insertion order.
    pub fn entries(&self) -> &[MemoryEntry] {
        &self.entries
    }

    /// Number of decisions and conventions stored.
    pub fn protected_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| e.kind.is_protected())
            .count()
    }

    /// Evicts entries until the store is within its cap. Protected entries are skipped
    /// entirely, so the cap can only be exceeded by protected entries, which `add` bounds.
    fn evict_to_cap(&mut self) -> Vec<String> {
        if self.entries.len() <= MAX_PROJECT_ENTRIES {
            return Vec::new();
        }
        let excess = self.entries.len() - MAX_PROJECT_ENTRIES;
        // Lowest value first: stale before current, then failed attempts, summaries and
        // completed work, then the oldest. The tuple is the eviction order.
        let mut candidates: Vec<(u8, u8, u64, String)> = self
            .entries
            .iter()
            .filter(|e| !e.kind.is_protected())
            .map(|e| {
                let stale_rank = u8::from(!e.stale);
                let kind_rank = match e.kind {
                    MemoryKind::FailedAttempt => 0,
                    MemoryKind::Summary => 1,
                    MemoryKind::CompletedWork => 2,
                    MemoryKind::Decision | MemoryKind::Convention => 3,
                };
                (stale_rank, kind_rank, e.seq, e.id.clone())
            })
            .collect();
        candidates.sort();
        let victims: BTreeSet<String> = candidates
            .into_iter()
            .take(excess)
            .map(|(_, _, _, id)| id)
            .collect();
        self.entries.retain(|e| !victims.contains(&e.id));
        victims.into_iter().collect()
    }
}

/// Produces a summary of conversation turns. Implemented by the runtime or a fake in tests.
pub trait Summarizer: Send + Sync {
    /// Summarises `messages`. Decisions are extracted separately, so the summary may omit them.
    fn summarize(&self, messages: &[ChatMessage]) -> Result<String>;
}

/// Lines of the form `Decision: ...` or `Convention: ...` in non-tool messages.
pub fn extract_decisions(messages: &[ChatMessage]) -> Vec<(MemoryKind, String)> {
    let mut out = Vec::new();
    for m in messages.iter().filter(|m| m.role != Role::Tool) {
        for line in m.content.lines() {
            let t = line.trim();
            // ASCII lowercasing keeps byte offsets, so the prefix length slices `t` safely.
            let lower = t.to_ascii_lowercase();
            if lower.starts_with("decision:") {
                out.push((
                    MemoryKind::Decision,
                    t["decision:".len()..].trim().to_string(),
                ));
            } else if lower.starts_with("convention:") {
                out.push((
                    MemoryKind::Convention,
                    t["convention:".len()..].trim().to_string(),
                ));
            }
        }
    }
    out.retain(|(_, text)| !text.is_empty());
    out
}

/// True when tool output looks like a failure (errors, panics, failing tests).
pub fn looks_like_error(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    [
        "error",
        "panicked",
        "failed",
        "failure",
        "traceback",
        "exception",
        "assertion",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

/// Paths from `hashes` that `text` mentions, as source references (sorted by path).
pub fn mentioned_sources(text: &str, hashes: &BTreeMap<String, String>) -> Vec<SourceRef> {
    hashes
        .iter()
        .filter(|(path, _)| text.contains(path.as_str()))
        .map(|(path, hash)| SourceRef {
            path: path.clone(),
            hash: hash.clone(),
        })
        .collect()
}

/// Warnings for repeated outputs and circular exchanges in the conversation.
pub fn detect_loops(messages: &[ChatMessage]) -> Vec<String> {
    let sigs: Vec<String> = messages
        .iter()
        .filter(|m| matches!(m.role, Role::Assistant | Role::Tool))
        .map(|m| normalize_signature(&m.content))
        .filter(|s| s.chars().count() >= 8)
        .collect();
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for s in &sigs {
        *counts.entry(s.as_str()).or_insert(0) += 1;
    }
    let mut warnings = Vec::new();
    for (sig, n) in counts {
        if n >= REPEAT_THRESHOLD {
            let shown: String = sig.chars().take(60).collect();
            warnings.push(format!(
                "repeated output seen {n} times, possible retry loop: \"{shown}\""
            ));
        }
    }
    if sigs.len() >= 6 {
        let w = &sigs[sigs.len() - 6..];
        if w[0] != w[1] && w[0] == w[2] && w[2] == w[4] && w[1] == w[3] && w[3] == w[5] {
            warnings.push(
                "last six outputs alternate between two states: possible circular reasoning"
                    .to_string(),
            );
        }
    }
    warnings
}

#[cfg(test)]
mod tests {
    use super::*;

    fn src(path: &str, hash: &str) -> SourceRef {
        SourceRef {
            path: path.into(),
            hash: hash.into(),
        }
    }

    fn hashes(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(p, h)| (p.to_string(), h.to_string()))
            .collect()
    }

    #[test]
    fn session_must_be_a_safe_name() {
        assert!(MemoryStore::new("../escape").is_err());
        assert!(MemoryStore::new("sess-1").is_ok());
    }

    #[test]
    fn add_validates_text_and_assigns_ids() {
        let mut m = MemoryStore::new("s").expect("store");
        assert!(m.add(MemoryKind::Decision, "   ", vec![]).is_err());
        let long = "x".repeat(MAX_ENTRY_CHARS + 1);
        assert!(m.add(MemoryKind::Summary, &long, vec![]).is_err());
        let id1 = m
            .add(MemoryKind::Decision, "use BM25", vec![])
            .expect("add");
        let id2 = m.add(MemoryKind::Summary, "summary", vec![]).expect("add");
        assert_eq!(id1, "mem-000000");
        assert_eq!(id2, "mem-000001");
    }

    #[test]
    fn summary_is_invalidated_when_its_file_changes() {
        let mut m = MemoryStore::new("s").expect("store");
        let id = m
            .add(
                MemoryKind::Summary,
                "refund.rs handles refunds",
                vec![src("src/refund.rs", "h1")],
            )
            .expect("add");
        assert!(m
            .invalidate_stale(&hashes(&[("src/refund.rs", "h1")]))
            .is_empty());
        let newly = m.invalidate_stale(&hashes(&[("src/refund.rs", "h2")]));
        assert_eq!(newly, vec![id.clone()]);
        assert_eq!(m.current().count(), 0);
        let entry = m.stale().next().expect("stale entry");
        assert!(entry
            .stale_reason
            .as_deref()
            .unwrap_or("")
            .contains("changed"));
    }

    #[test]
    fn deleted_source_makes_entry_stale() {
        let mut m = MemoryStore::new("s").expect("store");
        m.add(
            MemoryKind::Summary,
            "about gone.rs",
            vec![src("gone.rs", "h")],
        )
        .expect("add");
        m.invalidate_stale(&BTreeMap::new());
        let e = m.stale().next().expect("stale");
        assert!(e
            .stale_reason
            .as_deref()
            .unwrap_or("")
            .contains("no longer exists"));
    }

    #[test]
    fn entry_without_sources_never_goes_stale() {
        let mut m = MemoryStore::new("s").expect("store");
        m.add(MemoryKind::Convention, "use tabs", vec![])
            .expect("add");
        assert!(m.invalidate_stale(&BTreeMap::new()).is_empty());
        assert_eq!(m.current().count(), 1);
    }

    #[test]
    fn stale_entries_stay_stale_after_revert() {
        let mut m = MemoryStore::new("s").expect("store");
        m.add(MemoryKind::Summary, "x", vec![src("a.rs", "h1")])
            .expect("add");
        m.invalidate_stale(&hashes(&[("a.rs", "h2")]));
        assert!(m.invalidate_stale(&hashes(&[("a.rs", "h1")])).is_empty());
        assert_eq!(m.current().count(), 0);
    }

    #[test]
    fn store_round_trips_through_disk() {
        let t = tempfile::tempdir().expect("tempdir");
        let path = t.path().join("memory").join("s.json");
        let mut m = MemoryStore::new("s").expect("store");
        m.add(MemoryKind::Decision, "keep BM25", vec![src("a.rs", "h")])
            .expect("add");
        m.save(&path).expect("save");
        let loaded = MemoryStore::load_or_new(&path, "s").expect("load");
        assert_eq!(loaded, m);
    }

    #[test]
    fn load_rejects_other_session_and_corrupt_files() {
        let t = tempfile::tempdir().expect("tempdir");
        let path = t.path().join("s.json");
        MemoryStore::new("s")
            .expect("store")
            .save(&path)
            .expect("save");
        assert!(matches!(
            MemoryStore::load_or_new(&path, "other"),
            Err(RaptorError::Integrity(_))
        ));
        std::fs::write(&path, "{bad").expect("write");
        assert!(matches!(
            MemoryStore::load_or_new(&path, "s"),
            Err(RaptorError::Integrity(_))
        ));
    }

    #[test]
    fn missing_store_file_is_an_empty_store() {
        let t = tempfile::tempdir().expect("tempdir");
        let m = MemoryStore::load_or_new(&t.path().join("none.json"), "s").expect("load");
        assert!(m.entries.is_empty());
    }

    #[test]
    fn protected_kinds_are_decisions_and_conventions() {
        assert!(MemoryKind::Decision.is_protected());
        assert!(MemoryKind::Convention.is_protected());
        assert!(!MemoryKind::Summary.is_protected());
        assert!(!MemoryKind::FailedAttempt.is_protected());
    }

    #[test]
    fn decisions_are_extracted_from_non_tool_messages() {
        let msgs = vec![
            ChatMessage::new(
                Role::Assistant,
                "Thinking.\nDecision: use reciprocal rank fusion\nConvention: snake_case files",
            ),
            ChatMessage::new(Role::Tool, "Decision: ignored because tool output"),
            ChatMessage::new(Role::User, "decision:"),
        ];
        let d = extract_decisions(&msgs);
        assert_eq!(
            d,
            vec![
                (
                    MemoryKind::Decision,
                    "use reciprocal rank fusion".to_string()
                ),
                (MemoryKind::Convention, "snake_case files".to_string()),
            ]
        );
    }

    #[test]
    fn error_markers_are_detected() {
        assert!(looks_like_error("test result: FAILED. 1 failed"));
        assert!(looks_like_error("thread 'main' panicked at src/x.rs"));
        assert!(!looks_like_error("compiled successfully"));
    }

    #[test]
    fn mentioned_sources_match_known_paths_only() {
        let known = hashes(&[("src/a.rs", "h1"), ("src/b.rs", "h2")]);
        let refs = mentioned_sources("see src/a.rs for details", &known);
        assert_eq!(refs, vec![src("src/a.rs", "h1")]);
    }

    #[test]
    fn repeated_output_is_flagged() {
        let msgs: Vec<ChatMessage> = (0..3)
            .map(|_| ChatMessage::new(Role::Tool, "error: linker failed on line 4"))
            .collect();
        let w = detect_loops(&msgs);
        assert_eq!(w.len(), 1);
        assert!(w[0].contains("repeated output seen 3 times"));
    }

    #[test]
    fn digits_do_not_hide_a_repeat() {
        let msgs = vec![
            ChatMessage::new(Role::Tool, "timeout after 30 seconds"),
            ChatMessage::new(Role::Tool, "timeout after 31 seconds"),
            ChatMessage::new(Role::Tool, "timeout after 32 seconds"),
        ];
        assert_eq!(detect_loops(&msgs).len(), 1);
    }

    #[test]
    fn alternating_outputs_are_flagged_as_circular() {
        let mut msgs = Vec::new();
        for i in 0..6 {
            let text = if i % 2 == 0 {
                "try approach alpha again"
            } else {
                "try approach beta again"
            };
            msgs.push(ChatMessage::new(Role::Assistant, text));
        }
        let w = detect_loops(&msgs);
        assert!(w.iter().any(|x| x.contains("circular")), "{w:?}");
    }

    #[test]
    fn distinct_outputs_raise_no_warning() {
        let msgs = vec![
            ChatMessage::new(Role::Assistant, "first distinct answer"),
            ChatMessage::new(Role::Assistant, "second different answer"),
        ];
        assert!(detect_loops(&msgs).is_empty());
    }

    #[test]
    fn project_memory_round_trips_and_refuses_other_versions() {
        let t = tempfile::tempdir().expect("tempdir");
        let path = t.path().join("project.json");
        let mut m = ProjectMemory::new();
        m.add(MemoryKind::Decision, "use RRF", vec![src("a.rs", "h")])
            .expect("add");
        m.save(&path).expect("save");
        let loaded = ProjectMemory::load_or_new(&path).expect("load");
        assert_eq!(loaded, m);
        std::fs::write(&path, r#"{"version":99,"next_seq":0,"entries":[]}"#).expect("write");
        assert!(matches!(
            ProjectMemory::load_or_new(&path),
            Err(RaptorError::Integrity(_))
        ));
        std::fs::write(&path, "{bad").expect("write");
        assert!(matches!(
            ProjectMemory::load_or_new(&path),
            Err(RaptorError::Integrity(_))
        ));
    }

    #[test]
    fn project_eviction_keeps_every_decision_and_convention() {
        let mut m = ProjectMemory::new();
        let decisions: Vec<String> = (0..3)
            .map(|i| {
                m.add(MemoryKind::Decision, &format!("decision {i}"), vec![])
                    .expect("add")
                    .id
            })
            .collect();
        m.add(MemoryKind::Convention, "snake_case", vec![])
            .expect("add");
        let mut evicted_all = Vec::new();
        for i in 0..(MAX_PROJECT_ENTRIES + 100) {
            let added = m
                .add(MemoryKind::Summary, &format!("summary {i}"), vec![])
                .expect("add");
            evicted_all.extend(added.evicted);
        }
        assert_eq!(m.entries().len(), MAX_PROJECT_ENTRIES);
        for id in &decisions {
            assert!(m.entries().iter().any(|e| &e.id == id), "{id} kept");
        }
        assert_eq!(m.protected_count(), 4);
        // 4 protected + 600 summaries = 604 entries, so 104 are evicted, all summaries.
        assert_eq!(evicted_all.len(), 104);
        assert!(evicted_all.iter().all(|id| !decisions.contains(id)));
        // Oldest summaries go first.
        assert_eq!(evicted_all.first().map(String::as_str), Some("proj-000004"));
    }

    #[test]
    fn stale_entries_are_evicted_before_fresh_summaries() {
        let mut m = ProjectMemory::new();
        m.add(MemoryKind::Summary, "about a", vec![src("a.rs", "h1")])
            .expect("add");
        m.invalidate_stale(&BTreeMap::from([("a.rs".to_string(), "h2".to_string())]));
        for i in 0..MAX_PROJECT_ENTRIES {
            m.add(MemoryKind::Summary, &format!("fresh {i}"), vec![])
                .expect("add");
        }
        assert!(m.entries().iter().all(|e| !e.stale));
        assert_eq!(m.entries().len(), MAX_PROJECT_ENTRIES);
    }

    #[test]
    fn protected_entries_are_capped_and_can_be_forgotten() {
        let mut m = ProjectMemory::new();
        let mut first = String::new();
        for i in 0..MAX_PROTECTED_ENTRIES {
            let id = m
                .add(MemoryKind::Convention, &format!("rule {i}"), vec![])
                .expect("add")
                .id;
            if i == 0 {
                first = id;
            }
        }
        assert!(matches!(
            m.add(MemoryKind::Decision, "one too many", vec![]),
            Err(RaptorError::Conflict(_))
        ));
        assert!(m.add(MemoryKind::Summary, "still fine", vec![]).is_ok());
        m.forget(&first).expect("forget");
        assert!(m.add(MemoryKind::Decision, "fits now", vec![]).is_ok());
        assert!(m.forget("proj-999999").is_err());
    }
}
