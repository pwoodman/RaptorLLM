//! Monorepo-scale check. Not run by default because it writes about 20k files and takes
//! a while. Run with:
//!
//!     cargo test -p raptor-context --test scale --release -- --ignored --nocapture
//!
//! It prints cold index, no-op update, one-file update, search and build_context timings,
//! and asserts that an incremental update parses only the changed file.

use raptor_context::git::NoGit;
use raptor_context::{ContextEngine, ContextRequest, EngineDeps, SearchRequest};
use raptor_core::DataDirs;
use std::path::Path;
use std::time::Instant;

const PACKAGES: usize = 200;
const FILES_PER_PACKAGE: usize = 100;

/// One small Rust module: a struct, a function that calls its neighbour, an import, and a test.
fn module_source(pkg: usize, n: usize) -> String {
    format!(
        "use crate::pkg{pkg}::m{prev}::helper{prev};\n\npub struct Record{n} {{\n    pub id: u64,\n}}\n\npub fn process{n}(id: u64) -> u64 {{\n    let r = Record{n} {{ id }};\n    helper{prev}(r.id) + {n}\n}}\n\npub fn helper{n}(x: u64) -> u64 {{\n    x * 2\n}}\n\n#[cfg(test)]\nmod tests {{\n    use super::*;\n    #[test]\n    fn process{n}_works() {{\n        assert_eq!(process{n}(1), 2 + {n});\n    }}\n}}\n",
        prev = if n == 0 { FILES_PER_PACKAGE - 1 } else { n - 1 },
    )
}

fn generate(root: &Path) -> usize {
    let mut count = 0;
    for pkg in 0..PACKAGES {
        let dir = root.join(format!("packages/pkg{pkg}/src"));
        std::fs::create_dir_all(&dir).expect("mkdir");
        for n in 0..FILES_PER_PACKAGE {
            std::fs::write(dir.join(format!("m{n}.rs")), module_source(pkg, n)).expect("write");
            count += 1;
        }
    }
    count
}

#[test]
#[ignore = "writes ~20k files; run explicitly with --ignored"]
fn twenty_thousand_files_index_and_update_times() {
    let t = tempfile::tempdir().expect("tempdir");
    let ws = t.path().join("monorepo");
    let total = generate(&ws);
    eprintln!("generated {total} files");

    let dirs = DataDirs {
        root: t.path().join("data"),
    };
    let deps = || EngineDeps {
        git: Box::new(NoGit),
        ..EngineDeps::estimating()
    };

    let start = Instant::now();
    let mut eng = ContextEngine::open(&dirs, &ws, deps()).expect("open");
    let cold = eng.update().expect("cold update");
    let cold_ms = start.elapsed().as_millis();
    eprintln!(
        "cold index: {cold_ms} ms, parsed {} files, scanned {}",
        cold.reparsed.len(),
        cold.scanned
    );
    assert_eq!(cold.reparsed.len(), total);

    let start = Instant::now();
    let noop = eng.update().expect("no-op update");
    let noop_ms = start.elapsed().as_millis();
    eprintln!(
        "no-op update: {noop_ms} ms, parsed {}, read and hashed {}, stat-skipped {}, unchanged {}",
        noop.reparsed.len(),
        noop.hashed,
        noop.stat_skipped,
        noop.unchanged
    );
    assert!(noop.reparsed.is_empty());

    let changed = ws.join("packages/pkg7/src/m3.rs");
    std::fs::write(
        &changed,
        format!("{}\npub fn added() {{}}\n", module_source(7, 3)),
    )
    .expect("edit");
    let start = Instant::now();
    let one = eng.update().expect("one-file update");
    let one_ms = start.elapsed().as_millis();
    eprintln!(
        "one-file update: {one_ms} ms, parsed {} ({:?}), read and hashed {}, graph recomputed {}",
        one.reparsed.len(),
        one.reparsed,
        one.hashed,
        one.graph_recomputed
    );
    assert_eq!(one.reparsed, vec!["packages/pkg7/src/m3.rs".to_string()]);

    let mut search_ms = Vec::new();
    for q in [
        "process record",
        "helper multiply",
        "record id",
        "tests works",
    ] {
        let start = Instant::now();
        let hits = eng
            .search(&SearchRequest {
                query: q.into(),
                limit: 10,
                ..SearchRequest::default()
            })
            .expect("search");
        search_ms.push(start.elapsed().as_millis());
        assert!(!hits.is_empty(), "{q} finds something");
    }
    eprintln!("search ms per query: {search_ms:?}");

    let start = Instant::now();
    let resp = eng
        .build_context(ContextRequest {
            task: "why does process7 return the wrong id".into(),
            active_files: vec!["packages/pkg7/src/m3.rs".into()],
            model_ctx: 8192,
            reserved_output: 1024,
            ..ContextRequest::default()
        })
        .expect("build_context");
    eprintln!(
        "build_context: {} ms (includes a no-op re-scan), {} messages, {} included items",
        start.elapsed().as_millis(),
        resp.messages.len(),
        resp.manifest.included.len()
    );
    let status = eng.index_status();
    eprintln!(
        "status: files {}, chunks {}, symbols {}, graph edges {}",
        status.files, status.chunks, status.symbols, status.graph_edges
    );
}
