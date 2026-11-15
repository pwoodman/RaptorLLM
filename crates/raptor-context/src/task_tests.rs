//! Task-kind strategies: each kind must change what is retrieved and how the budget is shared.

use crate::api::{ContextEngine, ContextRequest, EngineDeps};
use crate::budget::{Category, TaskKind};
use crate::git::NoGit;
use crate::index::Index;
use crate::parse::default_extractors;
use crate::retrieve::{search, Hit, NoEmbedder, SearchParams};
use raptor_core::DataDirs;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;

fn write(root: &Path, rel: &str, body: &str) {
    let p = root.join(rel);
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent).expect("mkdir");
    }
    std::fs::write(p, body).expect("write");
}

fn index_of(root: &Path) -> Index {
    let mut idx = Index::open(
        root,
        None,
        default_extractors().expect("extractors"),
        Arc::new(NoEmbedder),
    )
    .expect("open");
    idx.update().expect("update");
    idx
}

fn run(
    idx: &Index,
    query: &str,
    kind: TaskKind,
    active: &[&str],
    recency: &BTreeMap<String, usize>,
) -> Vec<Hit> {
    let active: BTreeSet<String> = active.iter().map(|s| s.to_string()).collect();
    let none = BTreeSet::new();
    search(
        idx,
        &SearchParams {
            query,
            kind,
            limit: 20,
            active: &active,
            pinned: &none,
            recency,
            query_vector: None,
            vectors: None,
        },
    )
}

/// Position of the first hit in `path`, or `None`.
fn rank_of(hits: &[Hit], path: &str) -> Option<usize> {
    hits.iter().position(|h| h.path == path)
}

fn score_of(hits: &[Hit], path: &str) -> f64 {
    hits.iter()
        .find(|h| h.path == path)
        .map(|h| h.score)
        .unwrap_or(0.0)
}

#[test]
fn debug_ranks_failing_test_code_above_equal_prose() {
    let t = tempfile::tempdir().expect("tempdir");
    write(
        t.path(),
        "src/tax.rs",
        "pub fn compute_tax(x: u32) -> u32 {\n    x * 2\n}\n",
    );
    write(
        t.path(),
        "src/notes.rs",
        "// compute tax notes for the quarter\npub fn describe_quarter() {}\n",
    );
    write(
        t.path(),
        "tests/tax_test.rs",
        "use crate::tax::compute_tax;\n\n#[test]\nfn compute_tax_doubles() {\n    assert_eq!(compute_tax(2), 4);\n}\n",
    );
    let idx = index_of(t.path());
    let none = BTreeMap::new();
    let debug = run(&idx, "compute tax failed", TaskKind::Debug, &[], &none);
    let feature = run(&idx, "compute tax failed", TaskKind::Feature, &[], &none);

    let test_rank = rank_of(&debug, "tests/tax_test.rs").expect("test found");
    let notes_rank = rank_of(&debug, "src/notes.rs").expect("notes found");
    assert!(test_rank < notes_rank, "debug: {debug:?}");
    assert!(
        debug
            .iter()
            .find(|h| h.path == "tests/tax_test.rs")
            .is_some_and(|h| h.why.iter().any(|w| w.contains("test code"))),
        "the reason names the test"
    );
    // The same chunk scores lower without the debugging weights.
    assert!(
        score_of(&debug, "tests/tax_test.rs") > score_of(&feature, "tests/tax_test.rs"),
        "debug {} vs feature {}",
        score_of(&debug, "tests/tax_test.rs"),
        score_of(&feature, "tests/tax_test.rs")
    );
}

#[test]
fn debug_weights_recent_changes_more_than_architecture() {
    let t = tempfile::tempdir().expect("tempdir");
    write(t.path(), "src/old.rs", "pub fn old() {}\n");
    write(t.path(), "src/fresh.rs", "pub fn fresh() {}\n");
    let idx = index_of(t.path());
    // Recency only: the query matches nothing, so the recency signal is the sole reason.
    let recency = BTreeMap::from([("src/fresh.rs".to_string(), 0usize)]);
    let debug = run(&idx, "zzzz", TaskKind::Debug, &[], &recency);
    let arch = run(&idx, "zzzz", TaskKind::Architecture, &[], &recency);
    let d = score_of(&debug, "src/fresh.rs");
    let a = score_of(&arch, "src/fresh.rs");
    assert!(d > 0.0 && a > 0.0, "{debug:?} {arch:?}");
    assert!(
        d > 2.5 * a,
        "debug {d} should be about 3.3x architecture {a}"
    );
    assert!(debug
        .iter()
        .find(|h| h.path == "src/fresh.rs")
        .is_some_and(|h| h.why.iter().any(|w| w.contains("most recently changed"))));
}

#[test]
fn architecture_surfaces_the_dependency_hub_without_a_query_match() {
    let t = tempfile::tempdir().expect("tempdir");
    write(t.path(), "src/core.rs", "pub fn core() {}\n");
    write(
        t.path(),
        "src/a.rs",
        "use crate::core::core;\npub fn a() { core(); }\n",
    );
    write(
        t.path(),
        "src/b.rs",
        "use crate::core::core;\npub fn b() { core(); }\n",
    );
    write(
        t.path(),
        "src/c.rs",
        "use crate::core::core;\npub fn c() { core(); }\n",
    );
    write(t.path(), "src/leaf.rs", "pub fn leaf() {}\n");
    let idx = index_of(t.path());
    let none = BTreeMap::new();
    let arch = run(&idx, "overview", TaskKind::Architecture, &[], &none);
    assert_eq!(
        arch.first().map(|h| h.path.as_str()),
        Some("src/core.rs"),
        "{arch:?}"
    );
    assert!(arch[0].why.iter().any(|w| w.contains("imported by 3")));
    // Other kinds do not use the overview signal, and nothing matches the query.
    let feature = run(&idx, "overview", TaskKind::Feature, &[], &none);
    assert!(feature.is_empty(), "{feature:?}");
}

#[test]
fn refactor_ranks_callers_of_the_active_file_above_lexical_noise() {
    let t = tempfile::tempdir().expect("tempdir");
    write(
        t.path(),
        "src/pricing/tax.rs",
        "pub fn compute_tax(x: u32) -> u32 {\n    x * 2\n}\n",
    );
    write(
        t.path(),
        "src/checkout.rs",
        "use crate::pricing::tax::compute_tax;\n\npub fn checkout(x: u32) -> u32 {\n    compute_tax(x)\n}\n",
    );
    write(
        t.path(),
        "src/docs.rs",
        "// tax rate notes: tax rate, tax notes, rate rate\npub fn rate_notes() {}\n",
    );
    let idx = index_of(t.path());
    let none = BTreeMap::new();
    let refactor = run(
        &idx,
        "tax rate notes",
        TaskKind::Refactor,
        &["src/pricing/tax.rs"],
        &none,
    );
    let checkout = rank_of(&refactor, "src/checkout.rs").expect("caller found");
    let docs = rank_of(&refactor, "src/docs.rs").expect("noise found");
    assert!(checkout < docs, "{refactor:?}");
    assert!(refactor[checkout]
        .why
        .iter()
        .any(|w| w.contains("downstream") || w.contains("calls")));

    let feature = run(
        &idx,
        "tax rate notes",
        TaskKind::Feature,
        &["src/pricing/tax.rs"],
        &none,
    );
    assert!(
        score_of(&refactor, "src/checkout.rs") > score_of(&feature, "src/checkout.rs"),
        "downstream weight raises the caller under Refactor"
    );
}

#[test]
fn security_ranks_auth_and_input_handling_above_more_query_matches() {
    let t = tempfile::tempdir().expect("tempdir");
    write(
        t.path(),
        "src/profile/update.rs",
        "pub fn update_profile(token: &str, input: &str) -> bool {\n    validate_input(input) && check_token(token)\n}\n",
    );
    // More query terms, but no authentication or input handling.
    write(
        t.path(),
        "src/profile/layout.rs",
        "pub fn update_profile_layout() -> u32 {\n    // profile update profile update profile layout\n    0\n}\n",
    );
    let idx = index_of(t.path());
    let none = BTreeMap::new();
    let security = run(&idx, "update profile", TaskKind::Security, &[], &none);
    let feature = run(&idx, "update profile", TaskKind::Feature, &[], &none);
    assert!(
        rank_of(&feature, "src/profile/layout.rs") < rank_of(&feature, "src/profile/update.rs"),
        "without the security weight the lexical match wins: {feature:?}"
    );
    assert!(
        rank_of(&security, "src/profile/update.rs") < rank_of(&security, "src/profile/layout.rs"),
        "security weight lifts the auth file: {security:?}"
    );
    let auth = security
        .iter()
        .find(|h| h.path == "src/profile/update.rs")
        .expect("update.rs");
    assert!(auth.why.iter().any(|w| w.contains("authentication")));
}

#[test]
fn build_context_allocates_budget_by_task_kind() {
    let t = tempfile::tempdir().expect("tempdir");
    write(t.path(), "src/lib.rs", "pub fn a() {}\n");
    let data = t.path().join("data");
    let ws = t.path().join("ws");
    write(&ws, "src/lib.rs", "pub fn a() {}\n");
    let dirs = DataDirs { root: data };
    let deps = || EngineDeps {
        git: Box::new(NoGit),
        ..EngineDeps::estimating()
    };
    let mut eng = ContextEngine::open(&dirs, &ws, deps()).expect("open");
    let req = |kind: TaskKind| ContextRequest {
        task: "explain the lib".into(),
        task_kind: kind,
        model_ctx: 4096,
        reserved_output: 512,
        ..ContextRequest::default()
    };
    let debug = eng
        .build_context(req(TaskKind::Debug))
        .expect("debug")
        .manifest;
    let arch = eng
        .build_context(req(TaskKind::Architecture))
        .expect("arch")
        .manifest;
    let budget =
        |m: &crate::provenance::ContextManifest, c: Category| m.utilization_by_category[&c].budget;
    // The shares come from the task-kind table: Debug gives retrieved code 40% and graph
    // neighbours 25%; Architecture gives 35% and 40%. Both pools are the same, so the
    // caps keep that ordering whatever the protected items cost.
    let (dr, ar) = (
        budget(&debug, Category::RetrievedCode),
        budget(&arch, Category::RetrievedCode),
    );
    let (dg, ag) = (
        budget(&debug, Category::GraphNeighbours),
        budget(&arch, Category::GraphNeighbours),
    );
    assert!(dr > ar, "debug retrieved {dr:?} vs architecture {ar:?}");
    assert!(ag > dg, "architecture graph {ag:?} vs debug {dg:?}");
    assert_eq!(
        dr.zip(dg).map(|(r, g)| r * 250 / 400 == g),
        Some(true),
        "debug shares 400:250"
    );
}
