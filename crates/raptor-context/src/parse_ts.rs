//! Tree-sitter extraction: symbols with exact ranges and parents, imports, call sites
//! and test detection, read from concrete syntax trees.
//!
//! A file is only trusted when its tree has no syntax errors. Otherwise [`extract`]
//! returns `None` and the caller falls back to the regex extractor for that file, and
//! the fallback is recorded in the index (see `parse::Backend`).

use crate::parse::{
    is_test_path, python_key, CallRef, FileSymbols, Import, Language, Symbol, SymbolKind, KEYWORDS,
};
use std::collections::BTreeSet;
use tree_sitter::{Language as TsLanguage, Node, Parser};

/// Maximum call sites kept per file, so generated or minified files stay bounded.
const MAX_CALLS_PER_FILE: usize = 2000;

/// Grammar for `lang`. TypeScript uses the TSX grammar for `.tsx` files.
fn grammar(lang: Language, path: &str) -> Option<TsLanguage> {
    let tsx = path.to_ascii_lowercase().ends_with(".tsx");
    Some(match lang {
        Language::Rust => tree_sitter_rust::LANGUAGE.into(),
        Language::Python => tree_sitter_python::LANGUAGE.into(),
        Language::TypeScript if tsx => tree_sitter_typescript::LANGUAGE_TSX.into(),
        Language::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        Language::JavaScript => tree_sitter_javascript::LANGUAGE.into(),
        Language::Go => tree_sitter_go::LANGUAGE.into(),
        Language::Java => tree_sitter_java::LANGUAGE.into(),
        Language::C => tree_sitter_c::LANGUAGE.into(),
        Language::Cpp => tree_sitter_cpp::LANGUAGE.into(),
        Language::Other => return None,
    })
}

/// True when a tree-sitter grammar is bundled for `lang`.
pub fn supports(lang: Language) -> bool {
    grammar(lang, "").is_some()
}

/// Parses `source` and extracts its facts. `None` when no grammar applies, the parser
/// produces no tree, or the tree contains syntax errors (the caller must fall back).
pub fn extract(lang: Language, path: &str, source: &str) -> Option<FileSymbols> {
    let language = grammar(lang, path)?;
    let mut parser = Parser::new();
    parser.set_language(&language).ok()?;
    let tree = parser.parse(source, None)?;
    let root = tree.root_node();
    if root.has_error() {
        return None;
    }
    let mut cx = Cx {
        lang,
        path,
        src: source.as_bytes(),
        out: FileSymbols {
            is_test: is_test_path(path),
            ..FileSymbols::default()
        },
        seen_calls: BTreeSet::new(),
    };
    cx.visit(root, &Scope::default());
    Some(cx.out)
}

/// Enclosing type for declarations visited below it (methods get it as their parent).
#[derive(Clone, Default)]
struct Scope {
    container: Option<String>,
}

impl Scope {
    fn inside(name: Option<&str>) -> Self {
        Scope {
            container: name.map(str::to_string),
        }
    }
}

struct Cx<'a> {
    lang: Language,
    path: &'a str,
    src: &'a [u8],
    out: FileSymbols,
    seen_calls: BTreeSet<(String, u32)>,
}

/// Node kinds that are call sites in the supported grammars.
fn is_call_kind(kind: &str) -> bool {
    matches!(
        kind,
        "call_expression" | "call" | "method_invocation" | "object_creation_expression"
    )
}

/// 1-based inclusive line range of a node. A node that ends at column 0 ends on the line before.
fn line_range(n: Node<'_>) -> (u32, u32) {
    let start = n.start_position().row as u32 + 1;
    let end = n.end_position();
    let mut last = end.row as u32 + 1;
    if end.column == 0 && last > start {
        last -= 1;
    }
    (start, last)
}

fn strip_quotes(s: &str) -> &str {
    let s = s.trim();
    let b = s.as_bytes();
    if b.len() >= 2 && matches!(b[0], b'"' | b'\'' | b'`') && b[b.len() - 1] == b[0] {
        &s[1..s.len() - 1]
    } else {
        s
    }
}

fn dir_of(path: &str) -> &str {
    path.rsplit_once('/').map(|(d, _)| d).unwrap_or("")
}

fn join(a: &str, b: &str) -> String {
    match (a.is_empty(), b.is_empty()) {
        (true, _) => b.to_string(),
        (_, true) => a.to_string(),
        _ => format!("{a}/{b}"),
    }
}

/// Relative module key from directory `from_dir` to `target` (both repo-relative).
/// Always starts with `./` or `../`, the form the graph resolves against the importer.
fn rel_key(from_dir: &str, target: &str) -> String {
    let from: Vec<&str> = from_dir.split('/').filter(|s| !s.is_empty()).collect();
    let to: Vec<&str> = target.split('/').filter(|s| !s.is_empty()).collect();
    let common = from.iter().zip(&to).take_while(|(a, b)| a == b).count();
    let ups = from.len() - common;
    let rest = to[common..].join("/");
    if ups == 0 {
        format!("./{rest}")
    } else {
        format!("{}{rest}", "../".repeat(ups))
    }
}

/// Directory that holds a Rust module's child modules. `a/b.rs` owns `a/b/`;
/// `mod.rs`, `lib.rs` and `main.rs` own their directory.
fn rust_self_dir(path: &str) -> String {
    let file = path.rsplit('/').next().unwrap_or(path);
    let stem = file.rsplit_once('.').map(|(s, _)| s).unwrap_or(file);
    let dir = dir_of(path);
    if matches!(stem, "mod" | "lib" | "main") {
        dir.to_string()
    } else {
        join(dir, stem)
    }
}

fn parent_dir(p: &str) -> &str {
    p.rsplit_once('/').map(|(d, _)| d).unwrap_or("")
}

/// Module key for a Rust `use` path given as segments. `crate::` paths are root-relative,
/// `self::` and `super::` are resolved against the file's module position and emitted
/// relative to the file's directory. Other paths are kept as written (extern crates
/// and suffix-matched crate paths).
fn rust_module_key(path: &str, segs: &[String]) -> String {
    match segs.first().map(String::as_str) {
        Some("crate") => segs[1..].join("/"),
        Some("self") | Some("super") => {
            let mut base = rust_self_dir(path);
            let mut i = 0;
            if segs[0] == "self" {
                i = 1;
            } else {
                while i < segs.len() && segs[i] == "super" {
                    base = parent_dir(&base).to_string();
                    i += 1;
                }
            }
            let target = join(&base, &segs[i..].join("/"));
            rel_key(dir_of(path), &target)
        }
        _ => segs.join("/"),
    }
}

/// Type name of a Rust impl target, without generics or path qualifiers.
fn rust_type_name(cx: &Cx<'_>, n: Node<'_>) -> String {
    let inner = match n.kind() {
        "generic_type" | "reference_type" | "pointer_type" => n.child_by_field_name("type"),
        "scoped_type_identifier" => n.child_by_field_name("name"),
        _ => Some(n),
    };
    inner
        .map(|t| cx.text(t))
        .unwrap_or("")
        .split('<')
        .next()
        .unwrap_or("")
        .trim()
        .to_string()
}

/// Name of the function a C/C++ declarator declares, plus the scope of a qualified
/// definition (`Foo::bar` gives `Foo`).
fn c_declared_name<'a>(cx: &Cx<'a>, decl: Node<'_>) -> Option<(&'a str, Option<&'a str>)> {
    match decl.kind() {
        "identifier" | "field_identifier" | "type_identifier" | "destructor_name"
        | "operator_name" => Some((cx.text(decl), None)),
        "qualified_identifier" => {
            let scope = decl
                .child_by_field_name("scope")
                .map(|s| cx.text(s).rsplit("::").next().unwrap_or(""));
            let name = c_declared_name(cx, decl.child_by_field_name("name")?)?;
            Some((name.0, scope.filter(|s| !s.is_empty())))
        }
        "template_function" => c_declared_name(cx, decl.child_by_field_name("name")?),
        "function_declarator"
        | "pointer_declarator"
        | "reference_declarator"
        | "parenthesized_declarator"
        | "array_declarator"
        | "attributed_declarator" => c_declared_name(cx, decl.child_by_field_name("declarator")?),
        _ => None,
    }
}

/// Go receiver type name: `*T`, `T[P]` and `(T)` all give `T`.
fn go_receiver_type<'a>(cx: &Cx<'a>, n: Node<'_>) -> Option<&'a str> {
    match n.kind() {
        "parameter_list" => {
            let decl = (0..n.named_child_count())
                .filter_map(|i| n.named_child(i))
                .find(|c| c.kind() == "parameter_declaration")?;
            go_receiver_type(cx, decl.child_by_field_name("type")?)
        }
        "pointer_type" => go_receiver_type(cx, n.named_child(0)?),
        "generic_type" => go_receiver_type(cx, n.child_by_field_name("type")?),
        "type_identifier" => Some(cx.text(n)),
        _ => None,
    }
}

impl<'a> Cx<'a> {
    fn text(&self, n: Node<'_>) -> &'a str {
        n.utf8_text(self.src).unwrap_or("")
    }

    fn field_text(&self, n: Node<'_>, field: &str) -> Option<&'a str> {
        n.child_by_field_name(field)
            .map(|c| self.text(c))
            .filter(|s| !s.is_empty())
    }

    fn symbol(&mut self, name: &str, kind: SymbolKind, range: Node<'_>, parent: Option<&str>) {
        let (start_line, end_line) = line_range(range);
        self.out.symbols.push(Symbol {
            name: name.to_string(),
            kind,
            start_line,
            end_line,
            parent: parent.map(str::to_string),
        });
    }

    fn import(&mut self, module: String, at: Node<'_>) {
        if module.is_empty() {
            return;
        }
        let raw = self.text(at).lines().next().unwrap_or("").trim();
        let raw: String = raw.chars().take(200).collect();
        self.out.imports.push(Import {
            raw,
            module,
            line: at.start_position().row as u32 + 1,
        });
    }

    fn call(&mut self, name: &str, at: Node<'_>) {
        if name.is_empty() || KEYWORDS.contains(&name) || self.out.calls.len() >= MAX_CALLS_PER_FILE
        {
            return;
        }
        let line = at.start_position().row as u32 + 1;
        if self.seen_calls.insert((name.to_string(), line)) {
            self.out.calls.push(CallRef {
                name: name.to_string(),
                line,
            });
        }
    }

    /// Callee name of a call node, following qualified and field access.
    fn callee_name(&self, n: Node<'_>) -> Option<&'a str> {
        let field = |f: &str| n.child_by_field_name(f);
        match n.kind() {
            "identifier"
            | "field_identifier"
            | "property_identifier"
            | "type_identifier"
            | "destructor_name" => Some(self.text(n)),
            "scoped_identifier" | "qualified_identifier" | "scoped_type_identifier" => {
                self.callee_name(field("name")?)
            }
            "field_expression" | "selector_expression" => self.callee_name(field("field")?),
            "member_expression" => self.callee_name(field("property")?),
            "attribute" => self.callee_name(field("attribute")?),
            "generic_function" => self.callee_name(field("function")?),
            "template_function" => self.callee_name(field("name")?),
            "generic_type" => self.callee_name(n.named_child(0)?),
            _ => None,
        }
    }

    fn record_call(&mut self, n: Node<'_>) {
        let target = match n.kind() {
            "method_invocation" => n.child_by_field_name("name"),
            "object_creation_expression" => n.child_by_field_name("type"),
            _ => n.child_by_field_name("function"),
        };
        if let Some(name) = target.and_then(|t| self.callee_name(t)) {
            self.call(name, n);
        }
    }

    fn children(&mut self, n: Node<'_>, scope: &Scope) {
        for i in 0..n.child_count() {
            if let Some(c) = n.child(i) {
                self.visit(c, scope);
            }
        }
    }

    /// Visits one node. Language handlers return `true` when they handled the node
    /// (and visited whatever children they need); otherwise calls are recorded and the
    /// children are visited with the same scope.
    fn visit(&mut self, n: Node<'_>, scope: &Scope) {
        let handled = match self.lang {
            Language::Rust => self.rust(n, scope),
            Language::Python => self.python(n, scope),
            Language::TypeScript | Language::JavaScript => self.js(n, scope),
            Language::Go => self.go(n, scope),
            Language::Java => self.java(n, scope),
            Language::C | Language::Cpp => self.c(n, scope),
            Language::Other => false,
        };
        if handled {
            return;
        }
        if is_call_kind(n.kind()) {
            self.record_call(n);
        }
        self.children(n, scope);
    }

    // ----- Rust -----

    fn rust(&mut self, n: Node<'_>, scope: &Scope) -> bool {
        match n.kind() {
            "function_item" | "function_signature_item" => {
                let Some(name) = self.field_text(n, "name") else {
                    return false;
                };
                let kind = if self.rust_has_test_attr(n) {
                    SymbolKind::Test
                } else if scope.container.is_some() {
                    SymbolKind::Method
                } else {
                    SymbolKind::Function
                };
                self.symbol(name, kind, n, scope.container.as_deref());
                self.children(n, &Scope::default());
                true
            }
            "struct_item" | "enum_item" | "trait_item" | "type_item" | "union_item" => {
                let Some(name) = self.field_text(n, "name") else {
                    return false;
                };
                let kind = match n.kind() {
                    "enum_item" => SymbolKind::Enum,
                    "trait_item" => SymbolKind::Trait,
                    "type_item" => SymbolKind::Type,
                    _ => SymbolKind::Struct,
                };
                self.symbol(name, kind, n, None);
                if kind == SymbolKind::Trait {
                    self.children(n, &Scope::inside(Some(name)));
                    return true;
                }
                false
            }
            "impl_item" => {
                let ty = n
                    .child_by_field_name("type")
                    .map(|t| rust_type_name(self, t))
                    .unwrap_or_default();
                if !ty.is_empty() {
                    self.symbol(&ty, SymbolKind::Impl, n, None);
                }
                let inner = Scope::inside((!ty.is_empty()).then_some(ty.as_str()));
                self.children(n, &inner);
                true
            }
            "mod_item" => {
                let Some(name) = self.field_text(n, "name") else {
                    return false;
                };
                self.symbol(name, SymbolKind::Module, n, None);
                if n.child_by_field_name("body").is_none() {
                    // `mod x;` names a child file: a module declared here lives under this module's directory.
                    let target = join(&rust_self_dir(self.path), name);
                    let key = rel_key(dir_of(self.path), &target);
                    self.import(key, n);
                }
                self.children(n, &Scope::default());
                true
            }
            "const_item" | "static_item" => {
                if let Some(name) = self.field_text(n, "name") {
                    self.symbol(name, SymbolKind::Const, n, scope.container.as_deref());
                }
                false
            }
            "use_declaration" => {
                if let Some(arg) = n.child_by_field_name("argument") {
                    let mut paths = Vec::new();
                    self.use_paths(arg, &[], &mut paths);
                    for segs in paths {
                        let key = rust_module_key(self.path, &segs);
                        self.import(key, n);
                    }
                }
                true
            }
            "token_tree" => {
                // Calls inside macro arguments (`println!("{}", f(x))`) are not parsed
                // expressions, so an identifier followed by a parenthesised group is a call.
                for i in 0..n.child_count() {
                    let (Some(c), Some(next)) = (n.child(i), n.child(i + 1)) else {
                        continue;
                    };
                    if c.kind() == "identifier"
                        && next.kind() == "token_tree"
                        && self.text(next).starts_with('(')
                    {
                        let name = self.text(c);
                        self.call(name, c);
                    }
                }
                false
            }
            _ => false,
        }
    }

    fn rust_has_test_attr(&self, n: Node<'_>) -> bool {
        let mut prev = n.prev_sibling();
        while let Some(s) = prev {
            match s.kind() {
                "attribute_item" => {
                    let t = self.text(s);
                    if t.contains("test") && !t.contains("cfg") {
                        return true;
                    }
                }
                "line_comment" | "block_comment" => {}
                _ => break,
            }
            prev = s.prev_sibling();
        }
        false
    }

    /// Flattens a `use` tree into full segment lists (`use a::{b, c::d}` gives `a::b` and `a::c::d`).
    fn use_paths(&self, n: Node<'_>, prefix: &[String], out: &mut Vec<Vec<String>>) {
        let with = |segs: &str| -> Vec<String> {
            let mut v = prefix.to_vec();
            v.extend(
                segs.split("::")
                    .filter(|s| !s.is_empty())
                    .map(str::to_string),
            );
            v
        };
        match n.kind() {
            "scoped_use_list" => {
                let base = match n.child_by_field_name("path") {
                    Some(p) => with(self.text(p)),
                    None => prefix.to_vec(),
                };
                match n.child_by_field_name("list") {
                    Some(list) => self.use_paths(list, &base, out),
                    None => out.push(base),
                }
            }
            "use_list" => {
                for i in 0..n.named_child_count() {
                    if let Some(c) = n.named_child(i) {
                        self.use_paths(c, prefix, out);
                    }
                }
            }
            "use_as_clause" => {
                if let Some(p) = n.child_by_field_name("path") {
                    self.use_paths(p, prefix, out);
                }
            }
            "use_wildcard" => {
                let text = self.text(n);
                let base = text.trim_end_matches('*').trim_end_matches("::");
                out.push(with(base));
            }
            _ => out.push(with(self.text(n))),
        }
    }

    // ----- Python -----

    fn python(&mut self, n: Node<'_>, scope: &Scope) -> bool {
        match n.kind() {
            "function_definition" => {
                let Some(name) = self.field_text(n, "name") else {
                    return false;
                };
                let kind = if name.starts_with("test_") {
                    SymbolKind::Test
                } else if scope.container.is_some() {
                    SymbolKind::Method
                } else {
                    SymbolKind::Function
                };
                // Decorators belong to the declaration: the range starts at the first decorator.
                let range = match n.parent() {
                    Some(p) if p.kind() == "decorated_definition" => p,
                    _ => n,
                };
                self.symbol(name, kind, range, scope.container.as_deref());
                self.children(n, &Scope::default());
                true
            }
            "class_definition" => {
                let Some(name) = self.field_text(n, "name") else {
                    return false;
                };
                self.symbol(name, SymbolKind::Class, n, None);
                self.children(n, &Scope::inside(Some(name)));
                true
            }
            "import_statement" => {
                for i in 0..n.named_child_count() {
                    let Some(c) = n.named_child(i) else { continue };
                    let module = match c.kind() {
                        "dotted_name" => self.text(c),
                        "aliased_import" => match c.child_by_field_name("name") {
                            Some(name) => self.text(name),
                            None => continue,
                        },
                        _ => continue,
                    };
                    self.import(python_key(module), n);
                }
                true
            }
            "import_from_statement" => {
                let module_node = n.child_by_field_name("module_name");
                let mut base = String::new();
                if let Some(m) = module_node {
                    if m.kind() == "relative_import" {
                        for i in 0..m.named_child_count() {
                            let Some(c) = m.named_child(i) else { continue };
                            match c.kind() {
                                "import_prefix" => base.push_str(self.text(c)),
                                "dotted_name" => base.push_str(self.text(c)),
                                _ => {}
                            }
                        }
                    } else {
                        base.push_str(self.text(m));
                    }
                }
                self.import(python_key(&base), n);
                // `from pkg import sub` may name a submodule: link it as well. Names that
                // are not modules match nothing in the graph.
                for i in 0..n.named_child_count() {
                    let Some(c) = n.named_child(i) else { continue };
                    if module_node.is_some_and(|m| m.id() == c.id()) {
                        continue;
                    }
                    let name = match c.kind() {
                        "dotted_name" => self.text(c),
                        "aliased_import" => match c.child_by_field_name("name") {
                            Some(name) => self.text(name),
                            None => continue,
                        },
                        _ => continue,
                    };
                    let candidate = if base.ends_with('.') {
                        format!("{base}{name}")
                    } else {
                        format!("{base}.{name}")
                    };
                    self.import(python_key(&candidate), n);
                }
                true
            }
            _ => false,
        }
    }

    // ----- TypeScript / JavaScript -----

    fn js(&mut self, n: Node<'_>, scope: &Scope) -> bool {
        match n.kind() {
            "function_declaration" | "generator_function_declaration" => {
                let Some(name) = self.field_text(n, "name") else {
                    return false;
                };
                self.symbol(name, SymbolKind::Function, n, None);
                self.children(n, &Scope::default());
                true
            }
            "class_declaration" | "abstract_class_declaration" | "class" => {
                let name = self.field_text(n, "name");
                if let Some(name) = name {
                    self.symbol(name, SymbolKind::Class, n, None);
                }
                self.children(n, &Scope::inside(name));
                true
            }
            "method_definition" => {
                let Some(name) = self.field_text(n, "name") else {
                    return false;
                };
                self.symbol(name, SymbolKind::Method, n, scope.container.as_deref());
                self.children(n, &Scope::default());
                true
            }
            "interface_declaration" | "enum_declaration" | "type_alias_declaration" => {
                let Some(name) = self.field_text(n, "name") else {
                    return false;
                };
                let kind = match n.kind() {
                    "interface_declaration" => SymbolKind::Interface,
                    "enum_declaration" => SymbolKind::Enum,
                    _ => SymbolKind::Type,
                };
                self.symbol(name, kind, n, None);
                self.children(n, &Scope::default());
                true
            }
            "lexical_declaration" | "variable_declaration" => {
                for i in 0..n.named_child_count() {
                    let Some(d) = n.named_child(i) else { continue };
                    if d.kind() != "variable_declarator" {
                        continue;
                    }
                    let is_fn = d.child_by_field_name("value").is_some_and(|v| {
                        matches!(
                            v.kind(),
                            "arrow_function"
                                | "function_expression"
                                | "function"
                                | "generator_function"
                        )
                    });
                    if let (true, Some(name)) = (is_fn, self.field_text(d, "name")) {
                        self.symbol(name, SymbolKind::Function, n, None);
                    }
                    self.children(d, &Scope::default());
                }
                true
            }
            "import_statement" | "export_statement" => {
                if let Some(src) = n.child_by_field_name("source") {
                    let key = js_module_key(strip_quotes(self.text(src)));
                    self.import(key, n);
                }
                false
            }
            "call_expression" => {
                let Some(func) = n.child_by_field_name("function") else {
                    return false;
                };
                let first_string = n
                    .child_by_field_name("arguments")
                    .and_then(|a| a.named_child(0))
                    .filter(|a| a.kind() == "string")
                    .map(|a| strip_quotes(self.text(a)));
                if func.kind() == "import" {
                    if let Some(spec) = first_string {
                        self.import(js_module_key(spec), n);
                    }
                    return false;
                }
                if func.kind() != "identifier" {
                    return false;
                }
                match (self.text(func), first_string) {
                    ("require", Some(spec)) => {
                        self.import(js_module_key(spec), n);
                        true_after_children(self, n, scope)
                    }
                    (name @ ("describe" | "it" | "test"), Some(title)) => {
                        // Jest / Mocha / Vitest: `it('x', ...)` is a test. `describe` only groups.
                        if name != "describe" {
                            let title = title.to_string();
                            self.symbol(&title, SymbolKind::Test, n, None);
                        }
                        true_after_children(self, n, scope)
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }

    // ----- Go -----

    fn go(&mut self, n: Node<'_>, scope: &Scope) -> bool {
        let _ = scope;
        match n.kind() {
            "function_declaration" => {
                let Some(name) = self.field_text(n, "name") else {
                    return false;
                };
                let kind = if self.path.ends_with("_test.go") && name.starts_with("Test") {
                    SymbolKind::Test
                } else {
                    SymbolKind::Function
                };
                self.symbol(name, kind, n, None);
                self.children(n, &Scope::default());
                true
            }
            "method_declaration" => {
                let Some(name) = self.field_text(n, "name") else {
                    return false;
                };
                let recv = n
                    .child_by_field_name("receiver")
                    .and_then(|r| go_receiver_type(self, r));
                self.symbol(name, SymbolKind::Method, n, recv);
                self.children(n, &Scope::default());
                true
            }
            "type_spec" | "type_alias" => {
                let Some(name) = self.field_text(n, "name") else {
                    return false;
                };
                let kind = match n.child_by_field_name("type").map(|t| t.kind()) {
                    Some("struct_type") => SymbolKind::Struct,
                    Some("interface_type") => SymbolKind::Interface,
                    _ => SymbolKind::Type,
                };
                self.symbol(name, kind, n, None);
                true
            }
            "import_spec" => {
                if let Some(path) = n.child_by_field_name("path") {
                    let p = strip_quotes(self.text(path)).to_string();
                    self.import(p, n);
                }
                true
            }
            _ => false,
        }
    }

    // ----- Java -----

    fn java(&mut self, n: Node<'_>, scope: &Scope) -> bool {
        let _ = scope;
        match n.kind() {
            "class_declaration"
            | "record_declaration"
            | "interface_declaration"
            | "enum_declaration" => {
                let Some(name) = self.field_text(n, "name") else {
                    return false;
                };
                let kind = match n.kind() {
                    "interface_declaration" => SymbolKind::Interface,
                    "enum_declaration" => SymbolKind::Enum,
                    _ => SymbolKind::Class,
                };
                self.symbol(name, kind, n, None);
                self.children(n, &Scope::inside(Some(name)));
                true
            }
            "method_declaration" | "constructor_declaration" => {
                let Some(name) = self.field_text(n, "name") else {
                    return false;
                };
                let kind = if java_is_test(self, n) {
                    SymbolKind::Test
                } else {
                    SymbolKind::Method
                };
                self.symbol(name, kind, n, scope.container.as_deref());
                self.children(n, &Scope::default());
                true
            }
            "import_declaration" => {
                for i in 0..n.named_child_count() {
                    let Some(c) = n.named_child(i) else { continue };
                    if matches!(c.kind(), "scoped_identifier" | "identifier") {
                        // `com.acme.ledger` for a package wildcard, `com.acme.Refund` for a class.
                        let key = self.text(c).replace('.', "/");
                        self.import(key, n);
                    }
                }
                true
            }
            _ => false,
        }
    }

    // ----- C and C++ -----

    fn c(&mut self, n: Node<'_>, scope: &Scope) -> bool {
        match n.kind() {
            "function_definition" => {
                let Some(decl) = n.child_by_field_name("declarator") else {
                    return false;
                };
                let Some((name, qualifier)) = c_declared_name(self, decl) else {
                    return false;
                };
                let parent = qualifier.or(scope.container.as_deref());
                let kind = if parent.is_some() {
                    SymbolKind::Method
                } else {
                    SymbolKind::Function
                };
                let parent = parent.map(str::to_string);
                self.symbol(name, kind, n, parent.as_deref());
                self.children(n, &Scope::default());
                true
            }
            "struct_specifier" | "class_specifier" | "union_specifier" => {
                let (Some(name), Some(body)) =
                    (self.field_text(n, "name"), n.child_by_field_name("body"))
                else {
                    return false;
                };
                let kind = if n.kind() == "class_specifier" {
                    SymbolKind::Class
                } else {
                    SymbolKind::Struct
                };
                self.symbol(name, kind, n, None);
                self.children(body, &Scope::inside(Some(name)));
                true
            }
            "enum_specifier" => {
                if let (Some(name), Some(_)) =
                    (self.field_text(n, "name"), n.child_by_field_name("body"))
                {
                    self.symbol(name, SymbolKind::Enum, n, None);
                }
                false
            }
            "type_definition" => {
                if let Some(decl) = n.child_by_field_name("declarator") {
                    if matches!(decl.kind(), "type_identifier" | "identifier") {
                        self.symbol(self.text(decl), SymbolKind::Type, n, None);
                    }
                }
                false
            }
            "namespace_definition" => {
                if let Some(name) = self.field_text(n, "name") {
                    self.symbol(name, SymbolKind::Module, n, None);
                }
                self.children(n, &Scope::default());
                true
            }
            "preproc_include" => {
                if let Some(path) = n.child_by_field_name("path") {
                    // Only quoted includes name project files; `<...>` are system headers.
                    if path.kind() == "string_literal" {
                        let raw = strip_quotes(self.text(path));
                        let key = if raw.starts_with("./") || raw.starts_with("../") {
                            raw.to_string()
                        } else {
                            format!("./{raw}")
                        };
                        self.import(key, n);
                    }
                }
                true
            }
            _ => false,
        }
    }
}

/// Visits the children of `n` (so calls in its arguments and body are recorded) and
/// reports the node as handled. Used by call forms that are not calls to record.
fn true_after_children(cx: &mut Cx<'_>, n: Node<'_>, scope: &Scope) -> bool {
    cx.children(n, scope);
    true
}

/// Module key for a JS/TS import specifier. `.` and `..` name the directory itself.
fn js_module_key(spec: &str) -> String {
    match spec {
        "." => "./".to_string(),
        ".." => "../".to_string(),
        other => other.to_string(),
    }
}

/// JUnit-style test annotations on a Java method.
fn java_is_test(cx: &Cx<'_>, n: Node<'_>) -> bool {
    let Some(mods) = (0..n.named_child_count())
        .filter_map(|i| n.named_child(i))
        .find(|c| c.kind() == "modifiers")
    else {
        return false;
    };
    (0..mods.named_child_count())
        .filter_map(|i| mods.named_child(i))
        .filter(|a| matches!(a.kind(), "marker_annotation" | "annotation"))
        .filter_map(|a| a.child_by_field_name("name"))
        .any(|name| {
            let text = cx.text(name);
            matches!(
                text.rsplit('.').next().unwrap_or(text),
                "Test" | "ParameterizedTest" | "RepeatedTest" | "TestFactory"
            )
        })
}

#[cfg(test)]
#[path = "parse_ts_tests.rs"]
mod tests;
