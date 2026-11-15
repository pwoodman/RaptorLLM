//! Index status reports which extractor each language uses, and where a file fell back to regex.

use crate::api::{ContextEngine, EngineDeps};
use crate::git::NoGit;
use crate::parse::{Backend, Language};
use raptor_core::{DataDirs, EstimatingCounter};
use std::path::Path;
use std::sync::Arc;

fn write(root: &Path, rel: &str, body: &str) {
    let p = root.join(rel);
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent).expect("mkdir");
    }
    std::fs::write(p, body).expect("write");
}

#[test]
fn status_names_the_extractor_per_language_and_counts_regex_fallbacks() {
    let t = tempfile::tempdir().expect("tempdir");
    let ws = t.path().join("ws");
    write(&ws, "src/good.rs", "pub fn good() -> u32 { 1 }\n");
    // Syntax error: tree-sitter reports an error node, so this file is indexed by regex.
    write(&ws, "src/broken.rs", "pub fn broken( {\n    let x = ;\n");
    write(&ws, "app/main.py", "def run():\n    return 1\n");
    let deps = EngineDeps {
        counter: Arc::new(EstimatingCounter),
        git: Box::new(NoGit),
        ..EngineDeps::estimating()
    };
    let mut eng =
        ContextEngine::open(&DataDirs::new(t.path().join("data")), &ws, deps).expect("open");
    eng.update().expect("update");
    let status = eng.index_status();

    assert_eq!(
        status.extractors.len(),
        8,
        "one entry per supported language"
    );
    let rust = status
        .extractors
        .iter()
        .find(|e| e.language == Language::Rust)
        .expect("rust entry");
    assert_eq!(rust.extractor, Backend::TreeSitter);
    assert_eq!(rust.files, 2);
    assert_eq!(rust.regex_fallback_files, 1, "broken.rs fell back to regex");

    let python = status
        .extractors
        .iter()
        .find(|e| e.language == Language::Python)
        .expect("python entry");
    assert_eq!(python.extractor, Backend::TreeSitter);
    assert_eq!(python.files, 1);
    assert_eq!(python.regex_fallback_files, 0);

    let json = serde_json::to_string(&status).expect("json");
    assert!(json.contains(r#""extractor":"tree-sitter""#), "{json}");
    assert_eq!(Backend::Regex.as_str(), "regex");
}
