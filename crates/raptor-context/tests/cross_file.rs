//! Second evaluation scenario: a change in file A breaks a caller in file B, and B's test
//! lives in file C. The engine must retrieve the caller and the test even under a tight
//! budget, and protected items must never be dropped.
//!
//! Layout:
//!   src/pricing/tax.rs        A: `compute_tax` gained a rate parameter (active file)
//!   src/checkout/cart.rs      B: calls `compute_tax` with the old arity (broken caller)
//!   src/checkout/cart_test.rs C: tests `checkout_total` through `super::cart` (broken test)
//!   src/ui/widget_*.rs        forty unrelated files, modified more recently

use raptor_context::budget::{Category, TaskKind};
use raptor_context::git::NoGit;
use raptor_context::retrieve::NoEmbedder;
use raptor_context::{ContextEngine, ContextRequest, EngineDeps};
use raptor_core::{DataDirs, EstimatingCounter};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, UNIX_EPOCH};

const TAX_RS: &str = "//! Tax on a cart total. The rate is a parameter now (basis points).
pub fn compute_tax(amount_cents: u64, rate_bp: u64) -> u64 {
    amount_cents * rate_bp / 10_000
}
";

const CART_RS: &str = "//! Cart totals. Calls the tax module.
use crate::pricing::tax::compute_tax;

pub struct Cart {
    pub total_cents: u64,
}

pub fn checkout_total(cart: &Cart) -> u64 {
    cart.total_cents + compute_tax(cart.total_cents)
}
";

const CART_TEST_RS: &str = "use super::cart::{checkout_total, Cart};

#[test]
fn checkout_total_includes_tax() {
    assert_eq!(checkout_total(&Cart { total_cents: 100 }), 110);
}
";

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

/// Builds the scenario under `root`. The three related files are old; the UI files are new.
fn fixture(root: &Path) {
    let related = [
        ("src/pricing/tax.rs", TAX_RS),
        ("src/checkout/cart.rs", CART_RS),
        ("src/checkout/cart_test.rs", CART_TEST_RS),
    ];
    for (rel, body) in related {
        write(root, rel, body);
        set_mtime(root, rel, 1_000_000_000);
    }
    for i in 0..40u64 {
        let rel = format!("src/ui/widget_{i}.rs");
        let body = format!(
            "//! Widget {i} rendering.\npub fn render_widget_{i}(theme: &str) -> String {{\n    format!(\"<div class='{{theme}}'>{i}</div>\")\n}}\n"
        );
        write(root, &rel, &body);
        set_mtime(root, &rel, 1_700_000_000 + i);
    }
}

struct Scenario {
    _tmp: tempfile::TempDir,
    ws: PathBuf,
    data: PathBuf,
}

fn scenario() -> Scenario {
    let tmp = tempfile::tempdir().expect("tempdir");
    let ws = tmp.path().join("shop");
    std::fs::create_dir_all(&ws).expect("mkdir");
    fixture(&ws);
    Scenario {
        data: tmp.path().join("data"),
        ws,
        _tmp: tmp,
    }
}

fn engine(sc: &Scenario) -> ContextEngine {
    let deps = EngineDeps {
        counter: Arc::new(EstimatingCounter),
        git: Box::new(NoGit),
        embedder: Arc::new(NoEmbedder),
        summarizer: None,
    };
    ContextEngine::open(&DataDirs::new(&sc.data), &sc.ws, deps).expect("open engine")
}

fn refactor_request(model_ctx: usize, reserved_output: usize) -> ContextRequest {
    ContextRequest {
        task: "compute_tax now takes a rate; update the callers and their tests".into(),
        task_kind: TaskKind::Refactor,
        active_files: vec!["src/pricing/tax.rs".into()],
        errors: vec![
            "error[E0061]: this function takes 2 arguments but 1 argument was supplied, at src/checkout/cart.rs:9"
                .into(),
        ],
        model_ctx,
        reserved_output,
        ..ContextRequest::default()
    }
}

#[test]
fn impact_of_the_changed_file_names_the_caller_and_its_test() {
    let sc = scenario();
    let mut eng = engine(&sc);
    eng.update().expect("update");
    let impact = eng.impact("src/pricing/tax.rs").expect("impact");
    assert_eq!(impact.dependents, vec!["src/checkout/cart.rs"]);
    assert!(impact
        .transitive_dependents
        .contains_key("src/checkout/cart.rs"));
    let cart = eng.impact("src/checkout/cart.rs").expect("impact cart");
    assert_eq!(cart.related_tests, vec!["src/checkout/cart_test.rs"]);
}

#[test]
fn change_in_a_retrieves_the_caller_in_b_and_the_test_in_c_under_a_tight_budget() {
    let sc = scenario();
    let mut eng = engine(&sc);
    let resp = eng
        .build_context(refactor_request(1200, 200))
        .expect("build");
    let m = &resp.manifest;

    let sources: BTreeSet<String> = m.included.iter().map(|i| i.source.clone()).collect();
    assert!(
        sources.contains("src/checkout/cart.rs"),
        "the caller is in context: {sources:?}"
    );
    assert!(
        sources.contains("src/checkout/cart_test.rs"),
        "the test of the caller is in context: {sources:?}"
    );
    assert!(
        m.included
            .iter()
            .any(|i| i.id == "active:src/pricing/tax.rs" && i.category == Category::ActiveFiles),
        "the changed file is included in full"
    );
    assert!(m
        .included
        .iter()
        .any(|i| i.category == Category::ErrorsTests));
    assert!(m.included.iter().any(|i| i.category == Category::Task));
    // Unrelated files may fill budget that the chain did not use, but they must rank below it.
    let score = |src: &str| {
        m.included
            .iter()
            .filter(|i| i.source == src)
            .map(|i| i.score)
            .fold(f64::MIN, f64::max)
    };
    let chain = score("src/checkout/cart.rs").min(score("src/checkout/cart_test.rs"));
    for item in m
        .included
        .iter()
        .filter(|i| i.source.starts_with("src/ui/"))
    {
        assert!(
            item.score < chain,
            "{} (score {}) must rank below the dependency chain ({chain})",
            item.source,
            item.score
        );
    }
    assert!(m.is_fully_explained(), "every item carries a reason");
}

#[test]
fn caller_and_test_are_reached_through_the_graph_or_retrieval_not_only_by_name() {
    let sc = scenario();
    let mut eng = engine(&sc);
    let resp = eng
        .build_context(refactor_request(1200, 200))
        .expect("build");
    // The caller shares no name with the active file except the called function, so its
    // inclusion must come from the call edge or the downstream signal.
    let caller = resp
        .manifest
        .included
        .iter()
        .find(|i| i.source == "src/checkout/cart.rs")
        .expect("caller included");
    assert!(
        caller.why.contains("calls")
            || caller.why.contains("downstream")
            || caller.why.contains("hop")
            || caller.why.contains("adjacent"),
        "why: {}",
        caller.why
    );
}

#[test]
fn protected_items_are_kept_when_the_cross_file_context_does_not_fit() {
    let sc = scenario();
    let mut eng = engine(&sc);
    let resp = eng.build_context(refactor_request(60, 10)).expect("build");
    let m = &resp.manifest;
    assert!(m
        .included
        .iter()
        .any(|i| i.id == "active:src/pricing/tax.rs"));
    assert!(m
        .included
        .iter()
        .any(|i| i.category == Category::ErrorsTests));
    assert!(m.included.iter().any(|i| i.category == Category::Task));
    assert!(m.over_budget_by > 0, "over budget is reported loudly");
    assert!(
        m.warnings
            .iter()
            .any(|w| w.contains("nothing protected was dropped")),
        "{:?}",
        m.warnings
    );
    assert!(m.is_fully_explained());
}
