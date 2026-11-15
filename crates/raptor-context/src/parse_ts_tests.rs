//! Fixture tests for the tree-sitter extractor, one block per language.

use super::*;

fn ts(lang: Language, path: &str, src: &str) -> FileSymbols {
    extract(lang, path, src).unwrap_or_else(|| panic!("tree-sitter should parse {path}"))
}

fn sym<'a>(f: &'a FileSymbols, name: &str) -> &'a Symbol {
    f.symbols
        .iter()
        .find(|s| s.name == name)
        .unwrap_or_else(|| panic!("symbol {name} missing in {:?}", f.symbols))
}

fn modules(f: &FileSymbols) -> Vec<&str> {
    f.imports.iter().map(|i| i.module.as_str()).collect()
}

fn call_names(f: &FileSymbols) -> Vec<&str> {
    f.calls.iter().map(|c| c.name.as_str()).collect()
}

#[test]
fn rust_ranges_parents_and_test_attributes() {
    let src = "/// Ledger.\n#[derive(Debug)]\npub struct Ledger {\n    n: u32,\n}\n\nimpl Ledger {\n    pub fn post(&mut self) {\n        self.n += 1;\n    }\n}\n\nfn free() -> u32 { 1 }\n\n#[test]\nfn works() {\n    assert!(true);\n}\n";
    let f = ts(Language::Rust, "src/ledger.rs", src);
    let post = sym(&f, "post");
    assert_eq!(post.kind, SymbolKind::Method);
    assert_eq!(post.parent.as_deref(), Some("Ledger"));
    assert_eq!((post.start_line, post.end_line), (8, 10));
    let free = sym(&f, "free");
    assert_eq!(
        (free.kind, free.start_line, free.end_line),
        (SymbolKind::Function, 13, 13)
    );
    let works = sym(&f, "works");
    assert_eq!(works.kind, SymbolKind::Test);
    assert_eq!((works.start_line, works.end_line), (16, 18));
    let ledger = sym(&f, "Ledger");
    assert_eq!(ledger.kind, SymbolKind::Struct);
    assert_eq!((ledger.start_line, ledger.end_line), (3, 5));
    assert!(f
        .symbols
        .iter()
        .any(|s| s.name == "Ledger" && s.kind == SymbolKind::Impl));
}

#[test]
fn rust_trait_methods_and_consts() {
    let src = "pub trait Store {\n    fn get(&self, k: &str) -> u32;\n    fn put(&mut self) {}\n}\npub const LIMIT: usize = 4;\n";
    let f = ts(Language::Rust, "src/store.rs", src);
    assert_eq!(sym(&f, "get").kind, SymbolKind::Method);
    assert_eq!(sym(&f, "get").parent.as_deref(), Some("Store"));
    assert_eq!(sym(&f, "put").parent.as_deref(), Some("Store"));
    assert_eq!(sym(&f, "LIMIT").kind, SymbolKind::Const);
}

#[test]
fn rust_use_paths_resolve_crate_self_and_super() {
    let src = "use crate::ledger::reconcile::Batch;\nuse super::refund::{Refund, issue as go};\nuse self::helpers::*;\nuse std::fmt;\nmod audit;\nmod nested { pub fn f() {} }\n";
    let f = ts(Language::Rust, "src/ledger/reconcile.rs", src);
    let m = modules(&f);
    assert!(m.contains(&"ledger/reconcile/Batch"), "{m:?}");
    // super of src/ledger/reconcile.rs is the src/ledger module, so super::refund is a sibling file.
    assert!(m.contains(&"./refund/Refund"), "{m:?}");
    assert!(m.contains(&"./refund/issue"), "{m:?}");
    assert!(m.contains(&"./reconcile/helpers"), "{m:?}");
    assert!(m.contains(&"std/fmt"), "{m:?}");
    // `mod audit;` in a non-mod.rs file lives under the file's own module directory.
    assert!(m.contains(&"./reconcile/audit"), "{m:?}");
    assert!(!m.iter().any(|x| x.contains("nested")));
}

#[test]
fn rust_mod_rs_super_points_one_directory_up() {
    let src = "use super::util::Helper;\nmod child;\n";
    let f = ts(Language::Rust, "src/ledger/mod.rs", src);
    let m = modules(&f);
    // mod.rs owns src/ledger/, so its super module is src/.
    assert!(m.contains(&"../util/Helper"), "{m:?}");
    assert!(m.contains(&"./child"), "{m:?}");
}

#[test]
fn rust_calls_include_macro_arguments_and_skip_comments() {
    let src = "fn a() {\n    if x { b(1); }\n    println!(\"{}\", c());\n    v.push(d());\n    // legacy(x)\n    Ledger::open();\n}\n";
    let f = ts(Language::Rust, "src/a.rs", src);
    let names = call_names(&f);
    for want in ["b", "c", "push", "d", "open"] {
        assert!(names.contains(&want), "{want} in {names:?}");
    }
    assert!(!names.contains(&"legacy"));
    assert!(!names.contains(&"if"));
    let c_call = f.calls.iter().find(|c| c.name == "c").expect("c");
    assert_eq!(c_call.line, 3);
}

#[test]
fn python_methods_tests_decorators_and_ranges() {
    let src = "class Refunds:\n    @staticmethod\n    def apply(x):\n        return helper(x)\n\n    def test_apply(self):\n        pass\n\ndef top():\n    return 1\n";
    let f = ts(Language::Python, "app/refunds.py", src);
    let apply = sym(&f, "apply");
    assert_eq!(apply.kind, SymbolKind::Method);
    assert_eq!(apply.parent.as_deref(), Some("Refunds"));
    assert_eq!(
        (apply.start_line, apply.end_line),
        (2, 4),
        "decorator included"
    );
    assert_eq!(sym(&f, "test_apply").kind, SymbolKind::Test);
    assert_eq!(sym(&f, "test_apply").parent.as_deref(), Some("Refunds"));
    assert_eq!(sym(&f, "top").kind, SymbolKind::Function);
    assert_eq!(sym(&f, "top").end_line, 10);
    assert!(call_names(&f).contains(&"helper"));
}

#[test]
fn python_from_import_links_submodules_and_package() {
    let src = "import os.path\nfrom . import sibling\nfrom .pkg import tool\nfrom ..core.models import User\nfrom pkg.util import helper as h\n";
    let f = ts(Language::Python, "pkg/sub/mod.py", src);
    let m = modules(&f);
    assert!(m.contains(&"os/path"));
    assert!(m.contains(&"./sibling"), "submodule of the package: {m:?}");
    assert!(m.contains(&"./"), "package itself: {m:?}");
    assert!(m.contains(&"./pkg/tool"), "{m:?}");
    assert!(m.contains(&"../core/models"), "{m:?}");
    assert!(m.contains(&"pkg/util"), "{m:?}");
    assert!(m.contains(&"pkg/util/helper"), "{m:?}");
}

#[test]
fn typescript_symbols_imports_tests_and_arrow_functions() {
    let src = "import { pay } from './ledger/pay';\nimport type { Cart } from '../models/cart.js';\nconst lib = require('lodash');\nexport class CartView {\n  total(items: number[]) {\n    return pay(items);\n  }\n}\nexport const fmt = (x: number) => x.toFixed(2);\ninterface Priced { amount: number }\ntype Id = string;\ndescribe('cart totals', () => {\n  it('sums', () => {\n    expect(fmt(1)).toBe('1.00');\n  });\n});\n";
    let f = ts(Language::TypeScript, "src/cart.ts", src);
    let m = modules(&f);
    assert!(m.contains(&"./ledger/pay"), "{m:?}");
    assert!(m.contains(&"../models/cart.js"), "{m:?}");
    assert!(m.contains(&"lodash"), "{m:?}");
    let total = sym(&f, "total");
    assert_eq!(total.kind, SymbolKind::Method);
    assert_eq!(total.parent.as_deref(), Some("CartView"));
    assert_eq!(sym(&f, "CartView").kind, SymbolKind::Class);
    assert_eq!(sym(&f, "fmt").kind, SymbolKind::Function);
    assert_eq!(sym(&f, "Priced").kind, SymbolKind::Interface);
    assert_eq!(sym(&f, "Id").kind, SymbolKind::Type);
    let t = sym(&f, "sums");
    assert_eq!(t.kind, SymbolKind::Test);
    assert_eq!((t.start_line, t.end_line), (13, 15));
    assert!(!f.symbols.iter().any(|s| s.name == "describe"));
    let names = call_names(&f);
    assert!(names.contains(&"pay"));
    assert!(names.contains(&"toFixed"));
}

#[test]
fn tsx_grammar_is_used_for_tsx_files() {
    let src = "export function View(props: { n: number }) {\n  return <div>{props.n}</div>;\n}\n";
    let f = ts(Language::TypeScript, "web/View.tsx", src);
    assert_eq!(sym(&f, "View").kind, SymbolKind::Function);
}

#[test]
fn js_directory_imports_and_index_specifiers() {
    let src = "import x from '.';\nimport y from '..';\nimport z from './utils/index.js';\nexport function run() { return z(); }\n";
    let f = ts(Language::JavaScript, "src/app/main.js", src);
    let m = modules(&f);
    assert!(m.contains(&"./"), "{m:?}");
    assert!(m.contains(&"../"), "{m:?}");
    assert!(m.contains(&"./utils/index.js"), "{m:?}");
    assert_eq!(sym(&f, "run").kind, SymbolKind::Function);
}

#[test]
fn go_receivers_interfaces_and_block_imports() {
    let src = "package ledger\n\nimport (\n\t\"fmt\"\n\tapp \"example.com/app/internal/store\"\n)\n\ntype Book struct {\n\tn int\n}\n\ntype Poster interface {\n\tPost() error\n}\n\nfunc (b *Book) Post() error {\n\tfmt.Println(b.n)\n\treturn nil\n}\n\nfunc Helper() {}\n";
    let f = ts(Language::Go, "ledger/book.go", src);
    assert_eq!(modules(&f), vec!["fmt", "example.com/app/internal/store"]);
    let post = f
        .symbols
        .iter()
        .find(|s| s.name == "Post" && s.kind == SymbolKind::Method)
        .expect("method Post");
    assert_eq!(post.parent.as_deref(), Some("Book"));
    assert_eq!((post.start_line, post.end_line), (16, 19));
    assert_eq!(sym(&f, "Book").kind, SymbolKind::Struct);
    assert_eq!(sym(&f, "Poster").kind, SymbolKind::Interface);
    assert!(call_names(&f).contains(&"Println"));
}

#[test]
fn go_test_functions_in_test_files() {
    let src = "package x\n\nimport \"testing\"\n\nfunc TestRefund(t *testing.T) {\n}\n\nfunc helper() {}\n";
    let f = ts(Language::Go, "ledger/book_test.go", src);
    assert_eq!(sym(&f, "TestRefund").kind, SymbolKind::Test);
    assert_eq!(sym(&f, "helper").kind, SymbolKind::Function);
    assert!(f.is_test);
}

#[test]
fn java_class_methods_annotations_and_package_imports() {
    let src = "package com.acme.ledger;\n\nimport com.acme.model.Refund;\nimport com.acme.util.*;\nimport static com.acme.Money.round;\n\npublic class Ledger {\n    public int total(int a) {\n        return round(a);\n    }\n\n    @Test\n    public void checksTotal() {\n        new Refund(1);\n    }\n\n    @ParameterizedTest\n    void parameterised(int x) {}\n}\n";
    let f = ts(
        Language::Java,
        "src/main/java/com/acme/ledger/Ledger.java",
        src,
    );
    let m = modules(&f);
    assert!(m.contains(&"com/acme/model/Refund"), "{m:?}");
    assert!(
        m.contains(&"com/acme/util"),
        "wildcard keeps the package: {m:?}"
    );
    assert!(m.contains(&"com/acme/Money/round"), "{m:?}");
    let total = sym(&f, "total");
    assert_eq!(total.kind, SymbolKind::Method);
    assert_eq!(total.parent.as_deref(), Some("Ledger"));
    assert_eq!(sym(&f, "checksTotal").kind, SymbolKind::Test);
    assert_eq!(sym(&f, "parameterised").kind, SymbolKind::Test);
    let names = call_names(&f);
    assert!(names.contains(&"Refund"), "constructor call: {names:?}");
    assert!(names.contains(&"round"));
}

#[test]
fn c_includes_and_functions() {
    let src = "#include \"util/strbuf.h\"\n#include <stdio.h>\n\nstatic int add_one(int x)\n{\n    return x + 1;\n}\n";
    let f = ts(Language::C, "src/main.c", src);
    assert_eq!(modules(&f), vec!["./util/strbuf.h"]);
    let add = sym(&f, "add_one");
    assert_eq!(add.kind, SymbolKind::Function);
    assert_eq!((add.start_line, add.end_line), (4, 7));
}

#[test]
fn cpp_classes_methods_and_qualified_definitions() {
    let src = "#include \"ledger.h\"\n\nclass Book {\npublic:\n    int total() const;\n    void post(int amount) {\n        apply(amount);\n    }\n};\n\nint Book::total() const {\n    return 0;\n}\n\nnamespace shop {\nvoid close() {}\n}\n";
    let f = ts(Language::Cpp, "src/book.cpp", src);
    assert_eq!(modules(&f), vec!["./ledger.h"]);
    assert_eq!(sym(&f, "Book").kind, SymbolKind::Class);
    let post = sym(&f, "post");
    assert_eq!(post.kind, SymbolKind::Method);
    assert_eq!(post.parent.as_deref(), Some("Book"));
    let total = f
        .symbols
        .iter()
        .find(|s| s.name == "total" && s.start_line == 11)
        .expect("out-of-class total");
    assert_eq!(total.parent.as_deref(), Some("Book"));
    assert_eq!(total.kind, SymbolKind::Method);
    assert_eq!(sym(&f, "close").kind, SymbolKind::Function);
    assert!(call_names(&f).contains(&"apply"));
}

#[test]
fn syntax_errors_and_unknown_languages_are_not_trusted() {
    assert!(extract(Language::Rust, "a.rs", "fn broken( {\n").is_none());
    assert!(extract(Language::Python, "a.py", "def f(:\n").is_none());
    assert!(extract(Language::Other, "a.md", "fn x() {}").is_none());
}
