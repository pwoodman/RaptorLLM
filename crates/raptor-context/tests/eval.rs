//! Evaluation over a fixture repository built in a temp directory.
//!
//! The fixture has two relevant files (the refund/reconciliation code) that are OLD,
//! and forty unrelated UI files that were modified MORE RECENTLY. A recency-only
//! truncation spends the budget on the newest files and misses the relevant code.
//! The engine must find the relevant code under the same budget.

use raptor_context::budget::{Category, TaskKind};
use raptor_context::git::NoGit;
use raptor_context::memory::{MemoryKind, Summarizer};
use raptor_context::provenance::ContextManifest;
use raptor_context::recovery::STALE_PREFIX;
use raptor_context::retrieve::NoEmbedder;
use raptor_context::{
    ContextEngine, ContextRequest, EngineDeps, RecoverRequest, SearchRequest, SymbolQuery,
};
use raptor_core::{ChatMessage, DataDirs, EstimatingCounter, RaptorError, Role, TokenCounter};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, UNIX_EPOCH};

const GOLD: [&str; 2] = ["src/ledger/refund.rs", "src/ledger/reconcile.rs"];
const QUERY: &str = "refund reconciliation mismatch in the ledger";
const MODEL_CTX: usize = 1200;
const RESERVED: usize = 200;
const BUDGET: usize = MODEL_CTX - RESERVED;

const REFUND_RS: &str = "//! Refund issuing for settled orders.
use crate::ledger::reconcile::Batch;

pub struct Refund {
    pub order_id: String,
    pub amount_cents: u64,
}

/// Issues a refund for a settled order and records it in the ledger.
pub fn issue_refund(order_id: &str, amount_cents: u64) -> Refund {
    let refund = Refund { order_id: order_id.to_string(), amount_cents };
    record_refund(&refund);
    refund
}

fn record_refund(refund: &Refund) {
    println!(\"refund {} {}\", refund.order_id, refund.amount_cents);
}
";

const RECONCILE_RS: &str = "//! Reconciles refunds against the settlement batch.
use crate::ledger::refund::Refund;

pub struct Batch {
    pub refunds: Vec<Refund>,
    pub settled_cents: u64,
}

/// Compares the refunds in a batch with the settled total and reports any ledger mismatch.
pub fn reconcile_refunds(batch: &Batch) -> Result<(), String> {
    let refunded: u64 = batch.refunds.iter().map(|r| r.amount_cents).sum();
    if refunded != batch.settled_cents {
        return Err(format!(\"ledger mismatch: refunded {refunded}, settled {}\", batch.settled_cents));
    }
    Ok(())
}
";

const RECONCILE_TEST_RS: &str = "use shop::ledger::reconcile::reconcile_refunds;

#[test]
fn reconcile_refunds_reports_ledger_mismatch() {
    assert!(true);
}
";

/// Builds the fixture under `root`. Returns every file with its modification time.
fn fixture(root: &Path) -> Vec<(String, u64)> {
    let mut files = Vec::new();
    // Relevant code is old: 2001.
    for (rel, body) in [
        ("src/ledger/refund.rs", REFUND_RS),
        ("src/ledger/reconcile.rs", RECONCILE_RS),
        ("tests/reconcile_test.rs", RECONCILE_TEST_RS),
    ] {
        write(root, rel, body);
        files.push((rel.to_string(), 1_000_000_000));
    }
    // Unrelated UI code is newer, one second apart.
    for i in 0..40u64 {
        let rel = format!("src/ui/widget_{i}.rs");
        let body = format!(
            "//! Widget {i} rendering.\npub fn render_widget_{i}(theme: &str) -> String {{\n    format!(\"<div class='{{theme}}'>{i}</div>\")\n}}\n\npub fn theme_color_{i}() -> u32 {{\n    0x00ff{i:02x}\n}}\n"
        );
        write(root, &rel, &body);
        files.push((rel, 1_700_000_000 + i));
    }
    for (rel, secs) in &files {
        set_mtime(root, rel, *secs);
    }
    files
}

fn write(root: &Path, rel: &str, body: &str) {
    let p = root.join(rel);
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent).expect("mkdir");
    }
    std::fs::write(p, body).expect("write fixture file");
}

fn set_mtime(root: &Path, rel: &str, secs: u64) {
    let file = std::fs::File::options()
        .write(true)
        .open(root.join(rel))
        .expect("open for mtime");
    file.set_modified(UNIX_EPOCH + Duration::from_secs(secs))
        .expect("set mtime");
}

fn engine(data: &Path, ws: &Path, summarizer: Option<Box<dyn Summarizer>>) -> ContextEngine {
    let deps = EngineDeps {
        counter: Arc::new(EstimatingCounter),
        git: Box::new(NoGit),
        embedder: Arc::new(NoEmbedder),
        summarizer,
    };
    ContextEngine::open(&DataDirs::new(data), ws, deps).expect("open engine")
}

fn base_request(task: &str) -> ContextRequest {
    ContextRequest {
        task: task.to_string(),
        task_kind: TaskKind::Debug,
        messages: vec![ChatMessage::new(Role::User, task)],
        model_ctx: MODEL_CTX,
        reserved_output: RESERVED,
        ..ContextRequest::default()
    }
}

/// Recency-only truncation: newest files first, whole files, until the budget is spent.
fn recency_only_baseline(files: &[(String, u64)], root: &Path, budget: usize) -> BTreeSet<String> {
    let mut ordered = files.to_vec();
    ordered.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let counter = EstimatingCounter;
    let mut used = 0usize;
    let mut kept = BTreeSet::new();
    for (rel, _) in ordered {
        let text = std::fs::read_to_string(root.join(&rel)).expect("read fixture");
        let t = counter.count(&text);
        if used + t <= budget {
            used += t;
            kept.insert(rel);
        }
    }
    kept
}

fn retrieved_sources(m: &ContextManifest) -> BTreeSet<String> {
    m.included
        .iter()
        .filter(|i| i.category == Category::RetrievedCode)
        .map(|i| i.source.clone())
        .collect()
}

fn gold_recall(found: &BTreeSet<String>) -> f64 {
    let hit = GOLD.iter().filter(|g| found.contains(**g)).count();
    hit as f64 / GOLD.len() as f64
}

struct Fixture {
    _tmp: tempfile::TempDir,
    ws: PathBuf,
    data: PathBuf,
    files: Vec<(String, u64)>,
}

fn setup() -> Fixture {
    let tmp = tempfile::tempdir().expect("tempdir");
    let ws = tmp.path().join("shop");
    let data = tmp.path().join("data");
    std::fs::create_dir_all(&ws).expect("mkdir ws");
    let files = fixture(&ws);
    Fixture {
        _tmp: tmp,
        ws,
        data,
        files,
    }
}

#[test]
fn relevance_recall_beats_recency_only_truncation() {
    let fx = setup();
    let baseline = recency_only_baseline(&fx.files, &fx.ws, BUDGET);
    let baseline_recall = gold_recall(&baseline);

    let mut eng = engine(&fx.data, &fx.ws, None);
    let resp = eng.build_context(base_request(QUERY)).expect("build");
    let engine_recall = gold_recall(&retrieved_sources(&resp.manifest));
    // Printed with `cargo test -- --nocapture` so the comparison is visible.
    eprintln!("baseline (recency-only) recall {baseline_recall:.2}: {baseline:?}");
    eprintln!(
        "engine recall {engine_recall:.2}: {:?}",
        retrieved_sources(&resp.manifest)
    );

    assert_eq!(
        engine_recall,
        1.0,
        "engine must include both relevant files: {:?}",
        retrieved_sources(&resp.manifest)
    );
    assert!(
        baseline_recall < engine_recall,
        "baseline {baseline_recall} vs engine {engine_recall}"
    );
    assert!(
        resp.manifest.used_tokens <= BUDGET,
        "used {} of {BUDGET}",
        resp.manifest.used_tokens
    );
    assert_eq!(resp.manifest.over_budget_by, 0);
}

#[test]
fn protected_items_are_never_dropped_even_with_a_tiny_budget() {
    let fx = setup();
    let mut eng = engine(&fx.data, &fx.ws, None);
    let req = ContextRequest {
        task: "fix the refund reconciliation".into(),
        active_files: vec!["src/ledger/refund.rs".into()],
        errors: vec!["FAILED reconcile_refunds: expected 3 got 2".into()],
        model_ctx: 80,
        reserved_output: 20,
        ..ContextRequest::default()
    };
    let resp = eng.build_context(req).expect("build");
    let m = &resp.manifest;
    assert!(m
        .included
        .iter()
        .any(|i| i.id == "active:src/ledger/refund.rs" && i.category == Category::ActiveFiles));
    assert!(m
        .included
        .iter()
        .any(|i| i.category == Category::ErrorsTests));
    assert!(m.included.iter().any(|i| i.category == Category::Task));
    assert!(m.over_budget_by > 0, "over budget must be reported");
    assert!(
        m.warnings.iter().any(|w| w.contains("exceeds the budget")),
        "{:?}",
        m.warnings
    );
    assert!(resp
        .messages
        .iter()
        .any(|msg| msg.content.contains("expected 3 got 2")));
}

#[test]
fn stale_summary_is_invalidated_after_a_file_edit() {
    struct Fake;
    impl Summarizer for Fake {
        fn summarize(&self, messages: &[ChatMessage]) -> raptor_core::Result<String> {
            Ok(format!(
                "Refunds are issued by issue_refund in src/ledger/refund.rs ({} turns).",
                messages.len()
            ))
        }
    }
    let fx = setup();
    let convo = vec![
        ChatMessage::new(Role::User, "Look at src/ledger/refund.rs please"),
        ChatMessage::new(
            Role::Assistant,
            "Decision: keep refunds idempotent per order id",
        ),
    ];
    let mut eng = engine(&fx.data, &fx.ws, Some(Box::new(Fake)));
    let ids = eng.summarize_session("s1", &convo).expect("summarize");
    assert_eq!(ids.len(), 2, "one summary and one decision");

    let req = ContextRequest {
        session_id: Some("s1".into()),
        model_ctx: 4000,
        ..base_request("refund idempotent")
    };
    let before = eng.build_context(req.clone()).expect("build before");
    assert!(
        before
            .manifest
            .included
            .iter()
            .any(|i| i.id == "memory:mem-000000"),
        "summary served while current"
    );

    // Edit the file the summary describes.
    let path = fx.ws.join("src/ledger/refund.rs");
    let edited = std::fs::read_to_string(&path).expect("read") + "\n// audited 2026\n";
    std::fs::write(&path, edited).expect("edit");

    let after = eng.build_context(req).expect("build after");
    assert!(
        !after
            .manifest
            .included
            .iter()
            .any(|i| i.id == "memory:mem-000000"),
        "stale summary must not be served"
    );
    let excluded = after
        .manifest
        .excluded
        .iter()
        .find(|e| e.id == "memory:mem-000000")
        .expect("stale summary is reported");
    assert!(excluded.why.starts_with("stale:"), "{}", excluded.why);
    assert!(after.manifest.warnings.iter().any(|w| w.contains("stale")));
    // The decision names no file, so it never goes stale.
    assert!(after
        .manifest
        .included
        .iter()
        .any(|i| i.id == "memory:mem-000001"));
}

#[test]
fn every_included_and_excluded_item_is_explained() {
    let fx = setup();
    let mut eng = engine(&fx.data, &fx.ws, None);
    let mut req = base_request(QUERY);
    req.model_ctx = 300;
    req.reserved_output = 50;
    req.active_files = vec!["src/ledger/refund.rs".into()];
    let resp = eng.build_context(req).expect("build");
    assert!(resp.manifest.is_fully_explained());
    assert!(
        !resp.manifest.excluded.is_empty(),
        "a tight budget must exclude something"
    );
    assert!(resp.manifest.excluded.iter().all(|e| !e.why.is_empty()));
}

#[test]
fn excluded_chunks_can_be_recovered_and_stale_handles_are_refused() {
    let fx = setup();
    let mut eng = engine(&fx.data, &fx.ws, None);
    let mut req = base_request(QUERY);
    req.model_ctx = 260;
    req.reserved_output = 40;
    let resp = eng.build_context(req).expect("build");
    let handle = resp
        .manifest
        .excluded
        .iter()
        .find_map(|e| e.recover_handle.clone())
        .expect("a tight budget excludes a recoverable chunk");

    let got = eng
        .recover(RecoverRequest {
            handle: Some(handle.clone()),
            ..RecoverRequest::default()
        })
        .expect("recover");
    assert!(!got.items[0].text.is_empty());

    let path = fx.ws.join(&got.items[0].path);
    let edited = std::fs::read_to_string(&path).expect("read") + "\n// edited\n";
    std::fs::write(&path, edited).expect("edit");
    match eng.recover(RecoverRequest {
        handle: Some(handle),
        ..RecoverRequest::default()
    }) {
        Err(RaptorError::Conflict(msg)) => assert!(msg.starts_with(STALE_PREFIX), "{msg}"),
        other => panic!("expected stale conflict, got {other:?}"),
    }
}

#[test]
fn incremental_update_reparses_only_the_changed_file() {
    let fx = setup();
    let mut eng = engine(&fx.data, &fx.ws, None);
    let first = eng.update().expect("first update");
    assert_eq!(first.reparsed.len(), fx.files.len());

    write(
        &fx.ws,
        "src/ui/widget_7.rs",
        "pub fn render_widget_7() -> u8 {\n    7\n}\n",
    );
    let second = eng.update().expect("second update");
    assert_eq!(second.reparsed, vec!["src/ui/widget_7.rs".to_string()]);
    assert_eq!(second.unchanged, fx.files.len() - 1);
    assert!(second.removed.is_empty());
}

#[test]
fn output_is_deterministic_for_identical_input() {
    let fx = setup();
    let mut eng = engine(&fx.data, &fx.ws, None);
    let a = eng.build_context(base_request(QUERY)).expect("a");
    let b = eng.build_context(base_request(QUERY)).expect("b");
    assert_eq!(
        serde_json::to_string(&a).expect("json a"),
        serde_json::to_string(&b).expect("json b")
    );
}

#[test]
fn search_symbols_and_impact_answer_the_basics() {
    let fx = setup();
    let mut eng = engine(&fx.data, &fx.ws, None);
    eng.update().expect("update");

    let hits = eng
        .search(&SearchRequest {
            query: "issue refund".into(),
            limit: 3,
            ..SearchRequest::default()
        })
        .expect("search");
    assert_eq!(
        hits.first().map(|h| h.path.as_str()),
        Some("src/ledger/refund.rs"),
        "{hits:?}"
    );

    let syms = eng.symbols(&SymbolQuery {
        name_contains: "reconcile".into(),
        ..SymbolQuery::default()
    });
    assert!(syms.iter().any(|s| s.name == "reconcile_refunds"));

    let impact = eng.impact("src/ledger/reconcile.rs").expect("impact");
    assert!(
        impact
            .related_tests
            .contains(&"tests/reconcile_test.rs".to_string()),
        "{impact:?}"
    );
    assert!(impact
        .dependents
        .contains(&"src/ledger/refund.rs".to_string()));

    let status = eng.index_status();
    assert_eq!(status.files, fx.files.len());
    assert!(!status.git_repo);
}

#[test]
fn memory_decisions_survive_as_protected_entries() {
    let fx = setup();
    let mut eng = engine(&fx.data, &fx.ws, None);
    eng.remember(
        "s2",
        MemoryKind::Decision,
        "Keep reconciliation read-only",
        &["src/ledger/reconcile.rs".into()],
    )
    .expect("remember");
    let req = ContextRequest {
        session_id: Some("s2".into()),
        model_ctx: 300,
        reserved_output: 20,
        ..base_request("unrelated query words")
    };
    let resp = eng.build_context(req).expect("build");
    assert!(resp
        .manifest
        .included
        .iter()
        .any(|i| i.id == "memory:mem-000000" && i.category == Category::Memory));
}
