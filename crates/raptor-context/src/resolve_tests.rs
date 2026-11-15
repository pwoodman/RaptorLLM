//! End-to-end import resolution per language: fixture sources go through the default
//! extractors (tree-sitter) and the graph, and the test checks which files each file imports.

use crate::graph::Graph;
use crate::parse::{default_extractors, Language};

/// Builds a graph from `(path, source)` fixtures, using the language of each path.
fn graph_of(files: &[(&str, &str)]) -> Graph {
    let extractors = default_extractors().expect("extractors");
    let mut g = Graph::new();
    let upserts = files
        .iter()
        .map(|(path, src)| {
            let lang = Language::from_path(path);
            let ex = extractors
                .iter()
                .find(|e| e.language() == lang)
                .expect("extractor for language");
            (path.to_string(), lang, ex.extract(path, src))
        })
        .collect();
    g.apply(upserts, &[]);
    g
}

#[test]
fn rust_super_crate_and_mod_declarations_resolve() {
    let g = graph_of(&[
        ("src/ledger/refund.rs", "pub struct Refund;\npub fn issue() {}\n"),
        ("src/ledger/audit.rs", "pub fn log() {}\n"),
        ("src/ledger/reconcile.rs", "pub struct Batch;\n"),
        ("src/ledger/reconcile/audit.rs", "pub fn nested() {}\n"),
        (
            "src/ledger/service.rs",
            "use super::refund::{Refund, issue};\nuse crate::ledger::reconcile::Batch;\nmod audit;\nfn run() {}\n",
        ),
    ]);
    let imports = g.imports_of("src/ledger/service.rs");
    assert!(
        imports.contains(&"src/ledger/refund.rs".to_string()),
        "{imports:?}"
    );
    assert!(
        imports.contains(&"src/ledger/reconcile.rs".to_string()),
        "{imports:?}"
    );
    // `mod audit;` in service.rs names src/ledger/service/audit.rs, which does not exist,
    // so it must not link the sibling audit.rs that the module system would not pick.
    assert!(
        !imports.contains(&"src/ledger/audit.rs".to_string()),
        "{imports:?}"
    );
}

#[test]
fn rust_mod_rs_declares_children_in_its_directory() {
    let g = graph_of(&[
        ("src/ledger/reconcile.rs", "pub fn r() {}\n"),
        (
            "src/ledger/mod.rs",
            "pub mod reconcile;\nuse super::config::Settings;\n",
        ),
        ("src/config.rs", "pub struct Settings;\n"),
    ]);
    assert_eq!(
        g.imports_of("src/ledger/mod.rs"),
        vec!["src/config.rs", "src/ledger/reconcile.rs"]
    );
}

#[test]
fn python_from_dot_import_links_the_submodule() {
    let g = graph_of(&[
        ("pkg/sub/__init__.py", ""),
        ("pkg/sub/sibling.py", "def helper():\n    return 1\n"),
        ("pkg/sub/other.py", "def unused():\n    return 2\n"),
        (
            "pkg/sub/mod.py",
            "from . import sibling\nfrom .. import top\nfrom os import path\n",
        ),
        ("pkg/top.py", ""),
    ]);
    let imports = g.imports_of("pkg/sub/mod.py");
    assert!(
        imports.contains(&"pkg/sub/sibling.py".to_string()),
        "{imports:?}"
    );
    assert!(
        imports.contains(&"pkg/sub/__init__.py".to_string()),
        "{imports:?}"
    );
    assert!(imports.contains(&"pkg/top.py".to_string()), "{imports:?}");
    assert!(
        !imports.contains(&"pkg/sub/other.py".to_string()),
        "{imports:?}"
    );
}

#[test]
fn python_absolute_package_import_resolves_init() {
    let g = graph_of(&[
        ("pkg/core/__init__.py", ""),
        (
            "pkg/app.py",
            "from pkg.core import models\nimport pkg.core.models as m\n",
        ),
        ("pkg/core/models.py", "class User: pass\n"),
    ]);
    let imports = g.imports_of("pkg/app.py");
    assert!(
        imports.contains(&"pkg/core/__init__.py".to_string()),
        "{imports:?}"
    );
    assert!(
        imports.contains(&"pkg/core/models.py".to_string()),
        "{imports:?}"
    );
}

#[test]
fn typescript_index_files_extensions_and_parent_directories() {
    let g = graph_of(&[
        ("src/utils/index.ts", "export const u = 1;\n"),
        ("src/models/cart.ts", "export class Cart {}\n"),
        ("src/shared/fmt.tsx", "export const fmt = () => 1;\n"),
        (
            "src/pay.ts",
            "import { u } from './utils';\nimport { Cart } from './models/cart.js';\nimport lodash from 'lodash';\n",
        ),
        (
            "src/app/view.tsx",
            "import { fmt } from '../shared/fmt';\nimport { Cart } from '../models/cart';\n",
        ),
    ]);
    assert_eq!(
        g.imports_of("src/pay.ts"),
        vec!["src/models/cart.ts", "src/utils/index.ts"]
    );
    assert_eq!(
        g.imports_of("src/app/view.tsx"),
        vec!["src/models/cart.ts", "src/shared/fmt.tsx"]
    );
}

#[test]
fn javascript_directory_import_resolves_index() {
    let g = graph_of(&[
        ("lib/index.js", "module.exports = {};\n"),
        ("lib/run.js", "const x = require('.');\n"),
    ]);
    assert_eq!(g.imports_of("lib/run.js"), vec!["lib/index.js"]);
}

#[test]
fn go_import_path_matches_package_directory_by_longest_suffix() {
    let g = graph_of(&[
        ("internal/store/db.go", "package store\n"),
        ("legacy/store/db.go", "package store\n"),
        (
            "cmd/app/main.go",
            "package main\n\nimport \"example.com/app/internal/store\"\n\nfunc main() {}\n",
        ),
    ]);
    assert_eq!(
        g.imports_of("cmd/app/main.go"),
        vec!["internal/store/db.go"]
    );
}

#[test]
fn java_class_import_and_package_wildcard_follow_package_layout() {
    let g = graph_of(&[
        ("src/main/java/com/acme/model/Refund.java", "package com.acme.model;\npublic class Refund {}\n"),
        ("src/main/java/com/acme/util/Money.java", "package com.acme.util;\npublic class Money {}\n"),
        ("src/main/java/com/acme/util/Rounding.java", "package com.acme.util;\npublic class Rounding {}\n"),
        (
            "src/main/java/com/acme/ledger/Ledger.java",
            "package com.acme.ledger;\nimport com.acme.model.Refund;\nimport com.acme.util.*;\npublic class Ledger {}\n",
        ),
    ]);
    let imports = g.imports_of("src/main/java/com/acme/ledger/Ledger.java");
    assert!(
        imports.contains(&"src/main/java/com/acme/model/Refund.java".to_string()),
        "{imports:?}"
    );
    assert!(
        imports.contains(&"src/main/java/com/acme/util/Money.java".to_string()),
        "{imports:?}"
    );
    assert!(
        imports.contains(&"src/main/java/com/acme/util/Rounding.java".to_string()),
        "{imports:?}"
    );
}

#[test]
fn c_quoted_includes_resolve_relative_then_from_root() {
    let g = graph_of(&[
        ("src/util/strbuf.h", "int sb(void);\n"),
        (
            "src/main.c",
            "#include \"util/strbuf.h\"\nint main(void) { return sb(); }\n",
        ),
    ]);
    assert_eq!(g.imports_of("src/main.c"), vec!["src/util/strbuf.h"]);
}

/// Documents a known gap (see README): a path that starts with the crate name is not resolved,
/// because the prefix walk never drops the leading segment.
#[test]
fn crate_name_prefixed_paths_are_a_known_gap() {
    let g = graph_of(&[
        ("src/checkout/cart.rs", "pub struct Cart;\n"),
        ("tests/cart.rs", "use shop::checkout::cart::Cart;\n"),
    ]);
    assert!(g.imports_of("tests/cart.rs").is_empty());
}
