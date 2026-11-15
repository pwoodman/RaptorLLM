//! Engine-level tests: semantic retrieval with a fake backend, degradation, incremental
//! embedding, long-term project memory across restarts, and the summary guarantees.

use crate::api::{ContextEngine, ContextRequest, EngineDeps, SearchRequest};
use crate::budget::{Category, TaskKind};
use crate::git::NoGit;
use crate::memory::{MemoryKind, Summarizer};
use crate::summary::{ModelSummarizer, TextGenerator};
use crate::vector::EmbeddingBackend;
use raptor_core::{ChatMessage, DataDirs, RaptorError, Result, Role};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, UNIX_EPOCH};

/// The file a synonym query should find. Its path sorts after the decoy, so recency alone
/// ranks the decoy first.
const TARGET: &str = "src/zbilling/payouts.rs";
const DECOY: &str = "src/aui/colors.rs";
const TARGET_SRC: &str = "pub fn reimburse_customer(amount: u64) -> u64 {\n    amount\n}\n";
const DECOY_SRC: &str = "pub fn theme_palette() -> u32 {\n    7\n}\n";

/// Hand-made concept vectors. Text about refunds or reimbursement maps to axis 0, text about
/// palettes or themes to axis 2, and everything else to axis 1.
#[derive(Default)]
struct Concepts {
    fail: AtomicBool,
    calls: AtomicUsize,
    texts: AtomicUsize,
}

impl Concepts {
    fn texts_embedded(&self) -> usize {
        self.texts.load(Ordering::SeqCst)
    }
}

impl EmbeddingBackend for Concepts {
    fn model_id(&self) -> &str {
        "concepts-test-1"
    }

    fn dimension(&self) -> usize {
        3
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.fail.load(Ordering::SeqCst) {
            return Err(RaptorError::Network("connection refused".into()));
        }
        self.texts.fetch_add(texts.len(), Ordering::SeqCst);
        Ok(texts
            .iter()
            .map(|t| {
                let lower = t.to_ascii_lowercase();
                if ["reimburse", "money back", "refund"]
                    .iter()
                    .any(|k| lower.contains(k))
                {
                    vec![1.0, 0.0, 0.0]
                } else if lower.contains("palette") || lower.contains("theme") {
                    vec![0.0, 0.0, 1.0]
                } else {
                    vec![0.0, 1.0, 0.0]
                }
            })
            .collect())
    }
}

/// A generator that always replies with the same text, or always fails.
struct Gen(std::result::Result<String, String>);

impl TextGenerator for Gen {
    fn generate(&self, _prompt: &str, _max_tokens: u32) -> Result<String> {
        match &self.0 {
            Ok(s) => Ok(s.clone()),
            Err(e) => Err(RaptorError::Network(e.clone())),
        }
    }
}

fn set_mtime(path: &Path, secs: u64) {
    let f = std::fs::OpenOptions::new()
        .write(true)
        .open(path)
        .expect("open for mtime");
    f.set_modified(UNIX_EPOCH + Duration::from_secs(secs))
        .expect("set mtime");
}

/// Writes the two files with identical mtimes, so recency ties break by path.
fn workspace(ws: &Path) {
    for (rel, body) in [(TARGET, TARGET_SRC), (DECOY, DECOY_SRC)] {
        let p = ws.join(rel);
        std::fs::create_dir_all(p.parent().expect("parent")).expect("mkdir");
        std::fs::write(&p, body).expect("write");
        set_mtime(&p, 1_700_000_000);
    }
}

fn engine_with(
    data: &Path,
    ws: &Path,
    backend: Option<Arc<dyn EmbeddingBackend>>,
    summarizer: Option<Box<dyn Summarizer>>,
) -> ContextEngine {
    let deps = EngineDeps {
        git: Box::new(NoGit),
        summarizer,
        ..EngineDeps::estimating()
    };
    let mut eng = ContextEngine::open(&DataDirs::new(data), ws, deps).expect("open engine");
    eng.set_embedding_backend(backend).expect("set backend");
    eng
}

fn engine(data: &Path, ws: &Path, backend: Option<Arc<dyn EmbeddingBackend>>) -> ContextEngine {
    engine_with(data, ws, backend, None)
}

fn request(task: &str) -> ContextRequest {
    ContextRequest {
        task: task.into(),
        task_kind: TaskKind::Feature,
        model_ctx: 4096,
        reserved_output: 256,
        ..ContextRequest::default()
    }
}

#[test]
fn synonym_query_finds_the_file_only_with_embeddings() {
    let t = tempfile::tempdir().expect("tempdir");
    let ws = t.path().join("ws");
    let data = t.path().join("data");
    workspace(&ws);
    // No word of the query appears in the target file.
    let query = SearchRequest {
        query: "get my money back for the broken order".into(),
        task_kind: TaskKind::Feature,
        limit: 3,
        ..SearchRequest::default()
    };

    let plain = engine(&data, &ws, None);
    let hits = plain.search(&query).expect("search");
    assert_ne!(
        hits.first().map(|h| h.path.as_str()),
        Some(TARGET),
        "without embeddings the synonym query must not find the target: {hits:?}"
    );

    let semantic = engine(&data, &ws, Some(Arc::new(Concepts::default())));
    let hits = semantic.search(&query).expect("search");
    assert_eq!(
        hits.first().map(|h| h.path.as_str()),
        Some(TARGET),
        "with embeddings the target ranks first: {hits:?}"
    );
    assert!(hits[0].why.iter().any(|w| w.contains("semantic")));
    assert!(semantic.index_status().semantic.last_error.is_none());
}

#[test]
fn failing_backend_degrades_and_never_fails_a_build() {
    let t = tempfile::tempdir().expect("tempdir");
    let ws = t.path().join("ws");
    let data = t.path().join("data");
    workspace(&ws);
    let backend = Arc::new(Concepts::default());
    backend.fail.store(true, Ordering::SeqCst);
    let dyn_backend: Arc<dyn EmbeddingBackend> = backend.clone();
    let mut eng = engine(&data, &ws, Some(dyn_backend));

    let resp = eng
        .build_context(request("reimburse the customer"))
        .expect("build_context must succeed while embeddings are down");
    assert!(
        resp.manifest
            .warnings
            .iter()
            .any(|w| w.contains("semantic retrieval degraded")),
        "{:?}",
        resp.manifest.warnings
    );
    let status = eng.index_status();
    assert!(status.semantic.enabled);
    assert!(status.semantic.last_error.is_some());
    assert_eq!(status.semantic.vectors, 0);
    assert!(status.semantic.chunks_without_vector > 0);

    // Lexical retrieval still finds the file.
    let hits = eng
        .search(&SearchRequest {
            query: "reimburse customer".into(),
            ..SearchRequest::default()
        })
        .expect("search");
    assert_eq!(hits.first().map(|h| h.path.as_str()), Some(TARGET));

    // When the backend recovers, the next update embeds the missing chunks.
    backend.fail.store(false, Ordering::SeqCst);
    eng.update().expect("update");
    eng.search(&SearchRequest {
        query: "reimburse customer".into(),
        ..SearchRequest::default()
    })
    .expect("search");
    let status = eng.index_status();
    assert!(status.semantic.last_error.is_none(), "{status:?}");
    assert_eq!(status.semantic.chunks_without_vector, 0);
}

#[test]
fn only_changed_chunks_are_embedded_again() {
    let t = tempfile::tempdir().expect("tempdir");
    let ws = t.path().join("ws");
    let data = t.path().join("data");
    workspace(&ws);
    let backend = Arc::new(Concepts::default());
    let dyn_backend: Arc<dyn EmbeddingBackend> = backend.clone();
    let mut eng = engine(&data, &ws, Some(dyn_backend));
    let first = backend.texts_embedded();
    assert!(first >= 2, "both files are embedded on the first sync");

    eng.update().expect("update");
    assert_eq!(
        backend.texts_embedded(),
        first,
        "nothing changed, nothing embedded"
    );

    std::fs::write(
        ws.join(TARGET),
        "pub fn reimburse_customer(amount: u64) -> u64 {\n    amount + 1\n}\n",
    )
    .expect("edit");
    eng.update().expect("update");
    assert_eq!(backend.texts_embedded(), first + 1, "only the edited chunk");
}

#[test]
fn switching_model_discards_the_stored_vectors() {
    let t = tempfile::tempdir().expect("tempdir");
    let ws = t.path().join("ws");
    let data = t.path().join("data");
    workspace(&ws);
    drop(engine(&data, &ws, Some(Arc::new(Concepts::default()))));

    struct Other;
    impl EmbeddingBackend for Other {
        fn model_id(&self) -> &str {
            "other-model"
        }
        fn dimension(&self) -> usize {
            3
        }
        fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| vec![1.0, 0.0, 0.0]).collect())
        }
    }
    let eng = engine(&data, &ws, Some(Arc::new(Other)));
    let status = eng.index_status();
    assert_eq!(status.semantic.model_id.as_deref(), Some("other-model"));
    assert!(status
        .semantic
        .store_notice
        .as_deref()
        .unwrap_or("")
        .contains("rebuilding"));
    assert_eq!(status.semantic.vectors, 2);
}

#[test]
fn project_decision_is_served_survives_restart_and_goes_stale_on_edit() {
    let t = tempfile::tempdir().expect("tempdir");
    let ws = t.path().join("ws");
    let data = t.path().join("data");
    workspace(&ws);

    let mut eng = engine(&data, &ws, None);
    let added = eng
        .remember_project(
            MemoryKind::Decision,
            "keep reimbursements in payouts.rs",
            &[TARGET.to_string()],
        )
        .expect("remember");
    assert!(added.evicted.is_empty());
    let resp = eng.build_context(request("fix a bug")).expect("build");
    assert!(resp
        .manifest
        .included
        .iter()
        .any(|i| i.id.starts_with("project:") && i.category == Category::Memory));
    drop(eng);

    // Reloaded at engine start.
    let mut eng = engine(&data, &ws, None);
    assert_eq!(eng.index_status().project_memory_entries, 1);

    std::fs::write(
        ws.join(TARGET),
        "pub fn reimburse_customer(amount: u64) -> u64 {\n    amount * 2\n}\n",
    )
    .expect("edit");
    let resp = eng.build_context(request("fix a bug")).expect("build");
    assert!(
        resp.manifest
            .excluded
            .iter()
            .any(|e| e.id.starts_with("project:") && e.why.starts_with("stale")),
        "{:?}",
        resp.manifest.excluded
    );
    assert!(!resp
        .manifest
        .included
        .iter()
        .any(|i| i.id.starts_with("project:")));
    assert_eq!(eng.index_status().project_memory_stale, 1);

    let id = added.id.clone();
    eng.forget_project_memory(&id).expect("forget");
    assert_eq!(eng.index_status().project_memory_entries, 0);
}

#[test]
fn the_project_memory_list_shows_entries_and_flags_stale_ones_immediately() {
    let t = tempfile::tempdir().expect("tempdir");
    let ws = t.path().join("ws");
    let data = t.path().join("data");
    workspace(&ws);
    let mut eng = engine(&data, &ws, None);
    assert!(eng.project_memory().expect("list").is_empty());

    eng.remember_project(
        MemoryKind::Convention,
        "payouts live in payouts.rs",
        &[TARGET.to_string()],
    )
    .expect("remember");
    let listed = eng.project_memory().expect("list");
    assert_eq!(listed.len(), 1);
    assert!(!listed[0].stale);
    assert_eq!(listed[0].sources[0].path, TARGET);

    std::fs::write(ws.join(TARGET), "pub fn changed() {}\n").expect("edit");
    let listed = eng.project_memory().expect("list");
    assert!(
        listed[0].stale,
        "listing refreshes staleness without a context build"
    );
    assert!(listed[0].stale_reason.is_some());
}

fn convo() -> Vec<ChatMessage> {
    vec![
        ChatMessage::new(Role::User, "fix the reimbursement path"),
        ChatMessage::new(
            Role::Assistant,
            "Decision: reimbursements stay in payouts.rs\nChanged src/zbilling/payouts.rs.",
        ),
        ChatMessage::new(Role::Tool, "test reimburse_ok FAILED\nassertion failed"),
    ]
}

#[test]
fn summary_failure_keeps_structured_entries_and_warns() {
    let t = tempfile::tempdir().expect("tempdir");
    let ws = t.path().join("ws");
    let data = t.path().join("data");
    workspace(&ws);
    let summarizer: Box<dyn Summarizer> =
        Box::new(ModelSummarizer::new(Gen(Err("connection refused".into()))));
    let mut eng = engine_with(&data, &ws, None, Some(summarizer));

    let messages = convo();
    let out = eng
        .summarize_session_report("s1", &messages)
        .expect("report");
    assert!(out
        .warnings
        .iter()
        .any(|w| w.contains("summary not recorded")));
    assert_eq!(out.ids.len(), 2, "decision and failed attempt are recorded");
    assert!(eng.summarize_session("s1", &messages).is_err());

    let resp = eng
        .build_context(ContextRequest {
            session_id: Some("s1".into()),
            messages: messages.clone(),
            ..request("continue")
        })
        .expect("build");
    let text: String = resp.messages.iter().map(|m| m.content.as_str()).collect();
    assert!(
        text.contains("reimbursements stay in payouts.rs"),
        "verbatim decision"
    );
    assert!(
        text.contains("test reimburse_ok FAILED"),
        "failed attempt kept"
    );
}

#[test]
fn model_summary_is_stale_once_a_file_it_cites_changes() {
    let t = tempfile::tempdir().expect("tempdir");
    let ws = t.path().join("ws");
    let data = t.path().join("data");
    workspace(&ws);
    let summarizer: Box<dyn Summarizer> = Box::new(ModelSummarizer::new(Gen(Ok(
        "Reimbursements moved; one test still fails.".into(),
    ))));
    let mut eng = engine_with(&data, &ws, None, Some(summarizer));
    let out = eng
        .summarize_session_report("s2", &convo())
        .expect("report");
    assert!(out.warnings.is_empty(), "{:?}", out.warnings);
    assert_eq!(out.ids.len(), 3, "summary, decision and failed attempt");

    let req = ContextRequest {
        session_id: Some("s2".into()),
        ..request("continue")
    };
    let resp = eng.build_context(req.clone()).expect("build");
    let before: String = resp.messages.iter().map(|m| m.content.as_str()).collect();
    assert!(before.contains("one test still fails"));

    std::fs::write(
        ws.join(TARGET),
        "pub fn reimburse_customer(amount: u64) -> u64 {\n    amount * 3\n}\n",
    )
    .expect("edit");
    let resp = eng.build_context(req).expect("build");
    assert!(resp
        .manifest
        .excluded
        .iter()
        .any(|e| e.id.starts_with("memory:") && e.why.starts_with("stale")));
    let after: String = resp.messages.iter().map(|m| m.content.as_str()).collect();
    assert!(
        !after.contains("one test still fails"),
        "stale summary not served"
    );
}

#[test]
fn restart_reuses_persisted_vectors_without_embedding() {
    let t = tempfile::tempdir().expect("tempdir");
    let ws = t.path().join("ws");
    let data = t.path().join("data");
    workspace(&ws);
    drop(engine(&data, &ws, Some(Arc::new(Concepts::default()))));
    let backend = Arc::new(Concepts::default());
    let dyn_backend: Arc<dyn EmbeddingBackend> = backend.clone();
    let _eng = engine(&data, &ws, Some(dyn_backend));
    assert_eq!(
        backend.texts_embedded(),
        0,
        "vectors were persisted, so a restart embeds nothing"
    );
}

#[test]
fn engine_paths_stay_inside_the_context_index_dir() {
    let t = tempfile::tempdir().expect("tempdir");
    let ws = t.path().join("ws");
    let data = t.path().join("data");
    workspace(&ws);
    let mut eng = engine(&data, &ws, Some(Arc::new(Concepts::default())));
    eng.update().expect("update");
    let mut found: Vec<PathBuf> = Vec::new();
    collect(&data, &mut found);
    assert!(found
        .iter()
        .any(|p| p.file_name().is_some_and(|n| n == "vectors-v1.bin")));
    assert!(found.iter().all(|p| p.starts_with(&data)));
}

fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                collect(&p, out);
            } else {
                out.push(p);
            }
        }
    }
}
