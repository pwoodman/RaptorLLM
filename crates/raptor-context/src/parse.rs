//! Language detection and symbol / import / call extraction.
//!
//! Extraction sits behind the [`Extractor`] trait. [`HybridExtractor`] (the default per
//! language) parses with tree-sitter and falls back to the regex extractors for a file
//! that does not parse cleanly. Which backend produced each file is recorded, so the
//! index status can report it.

use crate::parse_ts;
use raptor_core::{RaptorError, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Source language, detected from the file extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Language {
    Rust,
    Python,
    TypeScript,
    JavaScript,
    Go,
    Java,
    C,
    Cpp,
    Other,
}

impl Language {
    /// Detects the language from the extension of a workspace-relative path.
    pub fn from_path(path: &str) -> Language {
        let name = path.rsplit('/').next().unwrap_or(path);
        let ext = name
            .rsplit_once('.')
            .map(|(_, e)| e.to_ascii_lowercase())
            .unwrap_or_default();
        match ext.as_str() {
            "rs" => Language::Rust,
            "py" | "pyi" => Language::Python,
            "ts" | "tsx" | "mts" | "cts" => Language::TypeScript,
            "js" | "jsx" | "mjs" | "cjs" => Language::JavaScript,
            "go" => Language::Go,
            "java" => Language::Java,
            "c" | "h" => Language::C,
            "cc" | "cpp" | "cxx" | "hpp" | "hh" | "hxx" => Language::Cpp,
            _ => Language::Other,
        }
    }
}

/// Kind of a declared symbol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SymbolKind {
    Function,
    Method,
    Struct,
    Class,
    Enum,
    Trait,
    Interface,
    Impl,
    Module,
    Type,
    Const,
    Test,
}

impl SymbolKind {
    /// Containers own methods declared inside their line range.
    pub fn is_container(self) -> bool {
        matches!(
            self,
            SymbolKind::Class
                | SymbolKind::Struct
                | SymbolKind::Trait
                | SymbolKind::Interface
                | SymbolKind::Impl
        )
    }

    /// Declarations that make good retrieval chunks on their own.
    pub fn is_chunk_worthy(self) -> bool {
        matches!(
            self,
            SymbolKind::Function
                | SymbolKind::Method
                | SymbolKind::Test
                | SymbolKind::Struct
                | SymbolKind::Class
                | SymbolKind::Enum
                | SymbolKind::Trait
                | SymbolKind::Interface
                | SymbolKind::Type
        )
    }
}

/// A declaration with its 1-based inclusive line range.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Symbol {
    pub name: String,
    pub kind: SymbolKind,
    pub start_line: u32,
    pub end_line: u32,
    pub parent: Option<String>,
}

/// An import as written, plus a normalised module key.
///
/// Keys use `/` separators. A key starting with `./` or `../` is relative to the
/// importing file's directory; any other key is matched against repo paths.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Import {
    pub raw: String,
    pub module: String,
    pub line: u32,
}

/// A best-effort call site (`name(`), resolved later by the graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CallRef {
    pub name: String,
    pub line: u32,
}

/// Everything extracted from one file.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileSymbols {
    pub symbols: Vec<Symbol>,
    pub imports: Vec<Import>,
    pub calls: Vec<CallRef>,
    /// True when the path itself marks the file as a test (`tests/`, `_test.go`, ...).
    pub is_test: bool,
}

/// Maximum call sites kept per file, so generated or minified files stay bounded.
const MAX_CALLS_PER_FILE: usize = 2000;

/// Pure path-based test detection.
pub fn is_test_path(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    let name = lower.rsplit('/').next().unwrap_or(&lower);
    lower
        .split('/')
        .any(|seg| matches!(seg, "tests" | "test" | "__tests__" | "spec"))
        || name.starts_with("test_")
        || name.contains("_test.")
        || name.contains(".test.")
        || name.contains(".spec.")
        || name.ends_with("_test.go")
}

/// The extraction backend that produced a file's facts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum Backend {
    /// Concrete syntax tree from a tree-sitter grammar.
    #[serde(rename = "tree-sitter")]
    TreeSitter,
    /// Regex rules. Used when no grammar is bundled, or a file did not parse cleanly.
    #[default]
    #[serde(rename = "regex")]
    Regex,
}

impl Backend {
    /// Stable name used in status output: `tree-sitter` or `regex`.
    pub fn as_str(self) -> &'static str {
        match self {
            Backend::TreeSitter => "tree-sitter",
            Backend::Regex => "regex",
        }
    }
}

/// Extracts structure from one source file. Implementations must be pure and deterministic.
pub trait Extractor: Send + Sync {
    /// The language this extractor handles.
    fn language(&self) -> Language;
    /// Extracts symbols, imports and calls from `source`, the content of `path`.
    fn extract(&self, path: &str, source: &str) -> FileSymbols;
    /// Like [`Extractor::extract`], and also reports which backend produced the facts.
    /// The default reports [`Backend::Regex`]; implementations that use another backend override it.
    fn extract_tagged(&self, path: &str, source: &str) -> (FileSymbols, Backend) {
        (self.extract(path, source), Backend::Regex)
    }
    /// The backend this extractor uses for files it can parse.
    fn backend(&self) -> Backend {
        Backend::Regex
    }
}

/// Tree-sitter first, regex for files that do not parse cleanly or when no grammar is bundled.
pub struct HybridExtractor {
    lang: Language,
    tree_sitter: bool,
    regex: RegexExtractor,
}

impl HybridExtractor {
    /// Extractor for `lang`. Fails only if a built-in regex pattern is invalid.
    pub fn new(lang: Language) -> Result<Self> {
        Ok(Self {
            lang,
            tree_sitter: parse_ts::supports(lang),
            regex: RegexExtractor::new(lang)?,
        })
    }
}

impl Extractor for HybridExtractor {
    fn language(&self) -> Language {
        self.lang
    }

    fn extract(&self, path: &str, source: &str) -> FileSymbols {
        self.extract_tagged(path, source).0
    }

    fn extract_tagged(&self, path: &str, source: &str) -> (FileSymbols, Backend) {
        if self.tree_sitter {
            if let Some(facts) = parse_ts::extract(self.lang, path, source) {
                return (facts, Backend::TreeSitter);
            }
        }
        (self.regex.extract(path, source), Backend::Regex)
    }

    fn backend(&self) -> Backend {
        if self.tree_sitter {
            Backend::TreeSitter
        } else {
            Backend::Regex
        }
    }
}

/// One extractor per supported language, in a stable order.
pub fn default_extractors() -> Result<Vec<Box<dyn Extractor>>> {
    let langs = [
        Language::Rust,
        Language::Python,
        Language::TypeScript,
        Language::JavaScript,
        Language::Go,
        Language::Java,
        Language::C,
        Language::Cpp,
    ];
    let mut out: Vec<Box<dyn Extractor>> = Vec::with_capacity(langs.len());
    for lang in langs {
        out.push(Box::new(HybridExtractor::new(lang)?));
    }
    Ok(out)
}

fn rx(pattern: &str) -> Result<Regex> {
    Regex::new(pattern).map_err(|e| RaptorError::Internal(format!("invalid extractor regex: {e}")))
}

struct Rule {
    re: Regex,
    kind: SymbolKind,
}

struct ImportRule {
    re: Regex,
    /// Prefix prepended to the captured path before key normalisation (Rust `mod x;`).
    prefix: &'static str,
}

/// Regex-based extractor for one language.
pub struct RegexExtractor {
    lang: Language,
    symbols: Vec<Rule>,
    imports: Vec<ImportRule>,
    go_block_import: Option<Regex>,
    call_re: Regex,
}

const RUST_VIS: &str = r"(?:pub(?:\([^)]*\))?\s+)?";

/// Words that look like calls or declarations but never name a symbol.
pub(crate) const KEYWORDS: &[&str] = &[
    "if", "for", "while", "match", "switch", "catch", "return", "else", "elif", "fn", "def",
    "function", "func", "class", "struct", "enum", "impl", "sizeof", "new", "with", "assert",
    "yield", "await", "super", "self", "this", "lambda", "typeof", "not", "and", "or", "in", "as",
    "do", "loop", "use", "mod", "where", "pub", "import", "from", "require", "include", "except",
    "raise", "throw", "void", "int", "delete", "const", "let", "var",
];

impl RegexExtractor {
    /// Compiles the rules for `lang`. Fails only if a built-in pattern is invalid.
    pub fn new(lang: Language) -> Result<Self> {
        let (symbols, imports, go_block_import) = match lang {
            Language::Rust => (rust_symbols()?, rust_imports()?, None),
            Language::Python => (python_symbols()?, python_imports()?, None),
            Language::TypeScript | Language::JavaScript => (js_symbols()?, js_imports()?, None),
            Language::Go => (
                go_symbols()?,
                go_imports()?,
                Some(rx(r#"^\s*(?:[\w.]+\s+)?"(?P<path>[^"]+)""#)?),
            ),
            Language::Java => (java_symbols()?, java_imports()?, None),
            Language::C | Language::Cpp => (c_symbols()?, c_imports()?, None),
            Language::Other => (Vec::new(), Vec::new(), None),
        };
        Ok(Self {
            lang,
            symbols,
            imports,
            go_block_import,
            call_re: rx(r"(?P<name>[A-Za-z_]\w*)\s*\(")?,
        })
    }

    /// Normalises a raw import path into a module key (see [`Import::module`]).
    fn module_key(&self, raw: &str) -> String {
        match self.lang {
            Language::Rust => rust_key(raw),
            Language::Python => python_key(raw),
            Language::Java => raw.trim_end_matches('.').replace('.', "/"),
            Language::C | Language::Cpp => {
                if raw.starts_with("./") || raw.starts_with("../") {
                    raw.to_string()
                } else {
                    format!("./{raw}")
                }
            }
            _ => raw.to_string(),
        }
    }

    fn match_symbol(&self, path: &str, lines: &[&str], idx: usize) -> Option<Symbol> {
        let line = lines[idx];
        for rule in &self.symbols {
            let Some(caps) = rule.re.captures(line) else {
                continue;
            };
            let name = caps.name("name")?.as_str().to_string();
            if KEYWORDS.contains(&name.as_str()) {
                return None;
            }
            let mut kind = rule.kind;
            let mut parent = None;
            if let Some(recv) = caps.name("recv") {
                kind = SymbolKind::Method;
                parent = Some(recv.as_str().to_string());
            }
            if matches!(kind, SymbolKind::Function | SymbolKind::Method)
                && self.is_test_decl(path, lines, idx, &name)
            {
                kind = SymbolKind::Test;
            }
            let end = block_end(lines, idx, self.lang == Language::Python);
            return Some(Symbol {
                name,
                kind,
                start_line: idx as u32 + 1,
                end_line: end as u32 + 1,
                parent,
            });
        }
        None
    }

    fn is_test_decl(&self, path: &str, lines: &[&str], idx: usize, name: &str) -> bool {
        match self.lang {
            Language::Python => name.starts_with("test_"),
            Language::Go => path.ends_with("_test.go") && name.starts_with("Test"),
            Language::Rust => preceding_attrs(lines, idx)
                .iter()
                .any(|a| a.contains("test")),
            Language::Java => preceding_attrs(lines, idx)
                .iter()
                .any(|a| a.starts_with("@Test") || a.starts_with("@ParameterizedTest")),
            _ => false,
        }
    }

    fn collect_imports(&self, lines: &[&str]) -> Vec<Import> {
        let mut out = Vec::new();
        let mut in_go_block = false;
        for (idx, line) in lines.iter().enumerate() {
            let lineno = idx as u32 + 1;
            let trimmed = line.trim();
            if self.lang == Language::Go {
                if trimmed.starts_with("import (") {
                    in_go_block = true;
                    continue;
                }
                if in_go_block && trimmed == ")" {
                    in_go_block = false;
                    continue;
                }
                if in_go_block {
                    if let Some(caps) = self
                        .go_block_import
                        .as_ref()
                        .and_then(|re| re.captures(line))
                    {
                        push_import(
                            &mut out,
                            self,
                            caps.name("path").map(|m| m.as_str()),
                            "",
                            trimmed,
                            lineno,
                        );
                    }
                    continue;
                }
            }
            for rule in &self.imports {
                if let Some(caps) = rule.re.captures(line) {
                    push_import(
                        &mut out,
                        self,
                        caps.name("path").map(|m| m.as_str()),
                        rule.prefix,
                        trimmed,
                        lineno,
                    );
                }
            }
        }
        out
    }

    fn collect_calls(&self, lines: &[&str]) -> Vec<CallRef> {
        let mut out = Vec::new();
        for (idx, line) in lines.iter().enumerate() {
            let t = line.trim_start();
            if t.starts_with("//")
                || t.starts_with('#')
                || t.starts_with('*')
                || t.starts_with("/*")
            {
                continue;
            }
            let mut seen_here = BTreeSet::new();
            for caps in self.call_re.captures_iter(line) {
                let Some(name) = caps.name("name").map(|m| m.as_str()) else {
                    continue;
                };
                if KEYWORDS.contains(&name) || !seen_here.insert(name.to_string()) {
                    continue;
                }
                out.push(CallRef {
                    name: name.to_string(),
                    line: idx as u32 + 1,
                });
                if out.len() >= MAX_CALLS_PER_FILE {
                    return out;
                }
            }
        }
        out
    }
}

fn push_import(
    out: &mut Vec<Import>,
    ex: &RegexExtractor,
    path: Option<&str>,
    prefix: &str,
    raw: &str,
    line: u32,
) {
    let Some(path) = path else { return };
    // An empty key (e.g. a bare `use crate;`) names nothing resolvable.
    let module = ex.module_key(&format!("{prefix}{path}"));
    if !module.is_empty() {
        out.push(Import {
            raw: raw.to_string(),
            module,
            line,
        });
    }
}

impl Extractor for RegexExtractor {
    fn language(&self) -> Language {
        self.lang
    }

    fn extract(&self, path: &str, source: &str) -> FileSymbols {
        let lines: Vec<&str> = source.lines().collect();
        let mut symbols = Vec::new();
        for idx in 0..lines.len() {
            if let Some(sym) = self.match_symbol(path, &lines, idx) {
                symbols.push(sym);
            }
        }
        assign_parents(&mut symbols);
        FileSymbols {
            symbols,
            imports: self.collect_imports(&lines),
            calls: self.collect_calls(&lines),
            is_test: is_test_path(path),
        }
    }
}

/// Leading indentation width (tabs count as one column, which is enough for block nesting).
fn indent(line: &str) -> usize {
    line.len() - line.trim_start().len()
}

/// Returns the 0-based index of the last line of the block starting at `start`.
///
/// Python uses indentation; brace languages count `{`/`}`. A declaration with no
/// body (a trait method or an extern prototype) ends on its own line.
fn block_end(lines: &[&str], start: usize, python: bool) -> usize {
    if python {
        let base = indent(lines[start]);
        let mut end = start;
        for (j, l) in lines.iter().enumerate().skip(start + 1) {
            if l.trim().is_empty() {
                continue;
            }
            if indent(l) <= base {
                break;
            }
            end = j;
        }
        return end;
    }
    let mut depth: i64 = 0;
    let mut opened = false;
    for (j, l) in lines.iter().enumerate().skip(start) {
        for ch in l.chars() {
            match ch {
                '{' => {
                    depth += 1;
                    opened = true;
                }
                '}' => depth -= 1,
                _ => {}
            }
        }
        if opened && depth <= 0 {
            return j;
        }
        if !opened && (l.contains(';') || j > start + 4) {
            return if l.contains(';') { j } else { start };
        }
    }
    start
}

/// Attribute or doc lines directly above a declaration, trimmed, nearest first.
fn preceding_attrs<'a>(lines: &[&'a str], idx: usize) -> Vec<&'a str> {
    let mut out = Vec::new();
    let mut j = idx;
    while j > 0 && out.len() < 4 {
        j -= 1;
        let t = lines[j].trim();
        if t.starts_with("#[") || t.starts_with('@') {
            out.push(t);
        } else if t.is_empty() || t.starts_with("///") {
            continue;
        } else {
            break;
        }
    }
    out
}

/// Gives methods their container as parent. Runs once per file.
fn assign_parents(symbols: &mut [Symbol]) {
    let containers: Vec<(usize, u32, u32, String)> = symbols
        .iter()
        .enumerate()
        .filter(|(_, s)| s.kind.is_container())
        .map(|(i, s)| (i, s.start_line, s.end_line, s.name.clone()))
        .collect();
    let snapshot: Vec<(SymbolKind, bool, u32, u32)> = symbols
        .iter()
        .map(|s| (s.kind, s.parent.is_some(), s.start_line, s.end_line))
        .collect();
    for (i, (kind, has_parent, start, end)) in snapshot.into_iter().enumerate() {
        if has_parent
            || !matches!(
                kind,
                SymbolKind::Function | SymbolKind::Method | SymbolKind::Test
            )
        {
            continue;
        }
        let innermost = containers
            .iter()
            .filter(|(ci, cs, ce, _)| *ci != i && *cs < start && end <= *ce)
            .max_by_key(|(_, cs, _, _)| *cs)
            .map(|(_, _, _, name)| name.clone());
        if let Some(parent) = innermost {
            let s = &mut symbols[i];
            s.parent = Some(parent);
            if s.kind == SymbolKind::Function {
                s.kind = SymbolKind::Method;
            }
        }
    }
}

fn rust_key(raw: &str) -> String {
    if let Some(rest) = raw.strip_prefix("crate::") {
        return rest.replace("::", "/");
    }
    if let Some(rest) = raw.strip_prefix("self::") {
        return format!("./{}", rest.replace("::", "/"));
    }
    if let Some(rest) = raw.strip_prefix("super::") {
        // Relative-to-parent-module resolution is not modelled; match by suffix instead.
        return rest.replace("::", "/");
    }
    if raw == "crate" || raw == "self" || raw == "super" {
        return String::new();
    }
    raw.replace("::", "/")
}

pub(crate) fn python_key(raw: &str) -> String {
    let dots = raw.chars().take_while(|c| *c == '.').count();
    let rest = raw[dots..].replace('.', "/");
    match dots {
        0 => rest,
        1 => format!("./{rest}"),
        n => format!("{}{rest}", "../".repeat(n - 1)),
    }
}

fn rust_symbols() -> Result<Vec<Rule>> {
    let k = |p: String, kind| -> Result<Rule> { Ok(Rule { re: rx(&p)?, kind }) };
    Ok(vec![
        k(
            format!(r"^\s*{RUST_VIS}(?:const\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+(?P<name>\w+)"),
            SymbolKind::Function,
        )?,
        k(
            format!(r"^\s*{RUST_VIS}struct\s+(?P<name>\w+)"),
            SymbolKind::Struct,
        )?,
        k(
            format!(r"^\s*{RUST_VIS}enum\s+(?P<name>\w+)"),
            SymbolKind::Enum,
        )?,
        k(
            format!(r"^\s*{RUST_VIS}trait\s+(?P<name>\w+)"),
            SymbolKind::Trait,
        )?,
        k(
            format!(r"^\s*{RUST_VIS}type\s+(?P<name>\w+)"),
            SymbolKind::Type,
        )?,
        k(
            format!(r"^\s*{RUST_VIS}mod\s+(?P<name>\w+)\s*[{{;]"),
            SymbolKind::Module,
        )?,
        k(
            format!(r"^\s*{RUST_VIS}(?:const|static)\s+(?P<name>[A-Za-z_]\w*)\s*:"),
            SymbolKind::Const,
        )?,
        k(
            r"^\s*impl(?:<[^{]*?>)?\s+(?:[\w:]+(?:<[^{]*?>)?\s+for\s+)?(?P<name>\w+)".to_string(),
            SymbolKind::Impl,
        )?,
    ])
}

fn rust_imports() -> Result<Vec<ImportRule>> {
    Ok(vec![
        ImportRule {
            re: rx(&format!(r"^\s*{RUST_VIS}use\s+(?P<path>[\w:]+)"))?,
            prefix: "",
        },
        ImportRule {
            re: rx(&format!(r"^\s*{RUST_VIS}mod\s+(?P<path>\w+)\s*;"))?,
            prefix: "self::",
        },
    ])
}

fn python_symbols() -> Result<Vec<Rule>> {
    Ok(vec![
        Rule {
            re: rx(r"^\s*(?:async\s+)?def\s+(?P<name>\w+)")?,
            kind: SymbolKind::Function,
        },
        Rule {
            re: rx(r"^\s*class\s+(?P<name>\w+)")?,
            kind: SymbolKind::Class,
        },
    ])
}

fn python_imports() -> Result<Vec<ImportRule>> {
    Ok(vec![
        ImportRule {
            re: rx(r"^\s*import\s+(?P<path>[\w.]+)")?,
            prefix: "",
        },
        ImportRule {
            re: rx(r"^\s*from\s+(?P<path>\.*[\w.]*)\s+import\b")?,
            prefix: "",
        },
    ])
}

fn js_symbols() -> Result<Vec<Rule>> {
    Ok(vec![
        Rule {
            re: rx(
                r#"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*(?P<name>\w+)"#,
            )?,
            kind: SymbolKind::Function,
        },
        Rule {
            re: rx(r#"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+(?P<name>\w+)"#)?,
            kind: SymbolKind::Class,
        },
        Rule {
            re: rx(r#"^\s*(?:export\s+)?interface\s+(?P<name>\w+)"#)?,
            kind: SymbolKind::Interface,
        },
        Rule {
            re: rx(r#"^\s*(?:export\s+)?type\s+(?P<name>\w+)\s*(?:<[^=]*>)?\s*="#)?,
            kind: SymbolKind::Type,
        },
        Rule {
            re: rx(
                r#"^\s*(?:export\s+)?const\s+(?P<name>\w+)\s*(?::[^=]+)?=\s*(?:async\s+)?(?:\([^)]*\)|\w+)\s*(?::[^=]+)?=>"#,
            )?,
            kind: SymbolKind::Function,
        },
        Rule {
            re: rx(r#"^\s*(?:describe|it|test)\(\s*['"`](?P<name>[^'"`]+)['"`]"#)?,
            kind: SymbolKind::Test,
        },
        Rule {
            re: rx(
                r"^\s+(?:(?:public|private|protected|static|async|override|readonly|get|set)\s+)*(?P<name>\w+)\s*\([^)]*\)\s*(?::\s*[^{]+)?\{\s*$",
            )?,
            kind: SymbolKind::Method,
        },
    ])
}

fn js_imports() -> Result<Vec<ImportRule>> {
    Ok(vec![ImportRule {
        re: rx(r#"(?:\bfrom|\bimport|\brequire\s*\()\s*\(?\s*['"](?P<path>[^'"]+)['"]"#)?,
        prefix: "",
    }])
}

fn go_symbols() -> Result<Vec<Rule>> {
    Ok(vec![
        Rule {
            re: rx(
                r"^func\s+(?:\((?:\w+\s+)?\*?(?P<recv>\w+)(?:\[[^\]]*\])?\)\s*)?(?P<name>\w+)\s*[\[(]",
            )?,
            kind: SymbolKind::Function,
        },
        Rule {
            re: rx(r"^type\s+(?P<name>\w+)\s+struct\b")?,
            kind: SymbolKind::Struct,
        },
        Rule {
            re: rx(r"^type\s+(?P<name>\w+)\s+interface\b")?,
            kind: SymbolKind::Interface,
        },
    ])
}

fn go_imports() -> Result<Vec<ImportRule>> {
    Ok(vec![ImportRule {
        re: rx(r#"^import\s+(?:[\w.]+\s+)?"(?P<path>[^"]+)""#)?,
        prefix: "",
    }])
}

fn java_symbols() -> Result<Vec<Rule>> {
    Ok(vec![
        Rule {
            re: rx(
                r"^\s*(?:(?:public|private|protected|static|final|abstract|sealed)\s+)*class\s+(?P<name>\w+)",
            )?,
            kind: SymbolKind::Class,
        },
        Rule {
            re: rx(r"^\s*(?:(?:public|private|protected|static)\s+)*interface\s+(?P<name>\w+)")?,
            kind: SymbolKind::Interface,
        },
        Rule {
            re: rx(r"^\s*(?:(?:public|private|protected|static)\s+)*enum\s+(?P<name>\w+)")?,
            kind: SymbolKind::Enum,
        },
        Rule {
            re: rx(
                r"^\s+(?:(?:public|private|protected|static|final|synchronized|abstract)\s+)+(?:<[^>]+>\s+)?[\w<>\[\],.?\s]+?\s+(?P<name>\w+)\s*\([^;]*$",
            )?,
            kind: SymbolKind::Method,
        },
    ])
}

fn java_imports() -> Result<Vec<ImportRule>> {
    Ok(vec![ImportRule {
        re: rx(r"^\s*import\s+(?:static\s+)?(?P<path>[\w.]+)")?,
        prefix: "",
    }])
}

fn c_symbols() -> Result<Vec<Rule>> {
    Ok(vec![
        Rule {
            re: rx(
                r"^\s*(?:typedef\s+)?(?:struct|class|enum|union)\s+(?P<name>\w+)\s*(?::[^{;]*)?\{?\s*$",
            )?,
            kind: SymbolKind::Struct,
        },
        Rule {
            re: rx(
                r"^\s*(?:static\s+|inline\s+|extern\s+|virtual\s+|const\s+)*[\w:<>*&~ ]+?[\s*&](?P<name>~?\w+)\s*\([^;]*\)\s*(?:const\s*)?\{?\s*$",
            )?,
            kind: SymbolKind::Function,
        },
    ])
}

fn c_imports() -> Result<Vec<ImportRule>> {
    Ok(vec![ImportRule {
        re: rx(r#"^\s*#\s*include\s+"(?P<path>[^"]+)""#)?,
        prefix: "",
    }])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ex(lang: Language) -> RegexExtractor {
        RegexExtractor::new(lang).expect("builtin regexes compile")
    }

    #[test]
    fn detects_languages_from_extensions() {
        assert_eq!(Language::from_path("src/lib.rs"), Language::Rust);
        assert_eq!(Language::from_path("a/b.PY"), Language::Python);
        assert_eq!(Language::from_path("web/app.tsx"), Language::TypeScript);
        assert_eq!(Language::from_path("x.h"), Language::C);
        assert_eq!(Language::from_path("x.hpp"), Language::Cpp);
        assert_eq!(Language::from_path("README.md"), Language::Other);
        assert_eq!(Language::from_path(".env"), Language::Other);
    }

    #[test]
    fn rust_functions_with_ranges_and_impl_parents() {
        let src = "pub struct Ledger {\n    n: u32,\n}\n\nimpl Ledger {\n    pub fn post(&mut self) {\n        self.n += 1;\n    }\n}\n\nfn free() -> u32 { 1 }\n";
        let f = ex(Language::Rust).extract("src/ledger.rs", src);
        let post = f
            .symbols
            .iter()
            .find(|s| s.name == "post")
            .expect("post found");
        assert_eq!(post.kind, SymbolKind::Method);
        assert_eq!(post.parent.as_deref(), Some("Ledger"));
        assert_eq!((post.start_line, post.end_line), (6, 8));
        let free = f
            .symbols
            .iter()
            .find(|s| s.name == "free")
            .expect("free found");
        assert_eq!(free.kind, SymbolKind::Function);
        assert_eq!(free.end_line, 11);
    }

    #[test]
    fn rust_test_attribute_marks_test() {
        let src = "#[test]\nfn works() {\n    assert!(true);\n}\n";
        let f = ex(Language::Rust).extract("src/x.rs", src);
        assert_eq!(f.symbols[0].kind, SymbolKind::Test);
    }

    #[test]
    fn rust_imports_and_mod_declarations() {
        let src = "use crate::ledger::reconcile::Ledger;\nuse std::fmt;\nmod refund;\npub mod audit;\nfn main() {}\n";
        let f = ex(Language::Rust).extract("src/main.rs", src);
        let mods: Vec<&str> = f.imports.iter().map(|i| i.module.as_str()).collect();
        assert!(mods.contains(&"ledger/reconcile/Ledger"), "{mods:?}");
        assert!(mods.contains(&"std/fmt"));
        assert!(mods.contains(&"./refund"), "{mods:?}");
        assert!(mods.contains(&"./audit"), "{mods:?}");
    }

    #[test]
    fn rust_calls_skip_keywords_and_macros() {
        let src = "fn a() {\n    if x { b(1); }\n    println!(\"{}\", c());\n    v.push(d());\n}\n";
        let f = ex(Language::Rust).extract("src/a.rs", src);
        let names: Vec<&str> = f.calls.iter().map(|c| c.name.as_str()).collect();
        assert!(names.contains(&"b"));
        assert!(names.contains(&"c"));
        assert!(names.contains(&"push"));
        assert!(!names.contains(&"if"));
        assert!(!names.contains(&"println"));
    }

    #[test]
    fn python_classes_methods_and_indent_ranges() {
        let src = "class Refunds:\n    def apply(self, x):\n        return x\n\n    def test_apply(self):\n        pass\n\ndef top():\n    return 1\n";
        let f = ex(Language::Python).extract("app/refunds.py", src);
        let apply = f.symbols.iter().find(|s| s.name == "apply").expect("apply");
        assert_eq!(apply.kind, SymbolKind::Method);
        assert_eq!(apply.parent.as_deref(), Some("Refunds"));
        assert_eq!((apply.start_line, apply.end_line), (2, 3));
        let t = f
            .symbols
            .iter()
            .find(|s| s.name == "test_apply")
            .expect("test");
        assert_eq!(t.kind, SymbolKind::Test);
        let top = f.symbols.iter().find(|s| s.name == "top").expect("top");
        assert_eq!(top.kind, SymbolKind::Function);
        assert_eq!(top.end_line, 9);
    }

    #[test]
    fn python_relative_and_absolute_imports() {
        let src = "import os.path\nfrom . import sibling\nfrom ..core.models import User\nfrom pkg.util import helper\n";
        let f = ex(Language::Python).extract("pkg/sub/mod.py", src);
        let mods: Vec<&str> = f.imports.iter().map(|i| i.module.as_str()).collect();
        assert!(mods.contains(&"os/path"));
        assert!(mods.contains(&"./"));
        assert!(mods.contains(&"../core/models"), "{mods:?}");
        assert!(mods.contains(&"pkg/util"));
    }

    #[test]
    fn typescript_symbols_imports_and_tests() {
        let src = "import { pay } from './ledger/pay';\nconst lib = require('lodash');\nexport class Cart {\n  total(items: number[]) {\n    return pay(items);\n  }\n}\nexport const fmt = (x: number) => x.toFixed(2);\ndescribe('cart totals', () => {\n  it('sums', () => {});\n});\n";
        let f = ex(Language::TypeScript).extract("src/cart.ts", src);
        let mods: Vec<&str> = f.imports.iter().map(|i| i.module.as_str()).collect();
        assert!(mods.contains(&"./ledger/pay"));
        assert!(mods.contains(&"lodash"));
        assert!(f
            .symbols
            .iter()
            .any(|s| s.name == "Cart" && s.kind == SymbolKind::Class));
        let total = f.symbols.iter().find(|s| s.name == "total").expect("total");
        assert_eq!(total.kind, SymbolKind::Method);
        assert_eq!(total.parent.as_deref(), Some("Cart"));
        assert!(f
            .symbols
            .iter()
            .any(|s| s.name == "fmt" && s.kind == SymbolKind::Function));
        assert!(f
            .symbols
            .iter()
            .any(|s| s.name == "cart totals" && s.kind == SymbolKind::Test));
        assert!(!f.symbols.iter().any(|s| s.name == "describe"));
    }

    #[test]
    fn go_functions_methods_and_block_imports() {
        let src = "package ledger\n\nimport (\n\t\"fmt\"\n\tapp \"example.com/app/internal/store\"\n)\n\ntype Book struct {\n\tn int\n}\n\nfunc (b *Book) Post() error {\n\tfmt.Println(b.n)\n\treturn nil\n}\n\nfunc Helper() {}\n";
        let f = ex(Language::Go).extract("ledger/book.go", src);
        let mods: Vec<&str> = f.imports.iter().map(|i| i.module.as_str()).collect();
        assert_eq!(mods, vec!["fmt", "example.com/app/internal/store"]);
        let post = f.symbols.iter().find(|s| s.name == "Post").expect("post");
        assert_eq!(post.kind, SymbolKind::Method);
        assert_eq!(post.parent.as_deref(), Some("Book"));
        assert!(f
            .symbols
            .iter()
            .any(|s| s.name == "Book" && s.kind == SymbolKind::Struct));
    }

    #[test]
    fn go_test_functions_in_test_files() {
        let src = "package x\nfunc TestRefund(t *testing.T) {\n}\nfunc helper() {}\n";
        let f = ex(Language::Go).extract("ledger/book_test.go", src);
        assert_eq!(f.symbols[0].kind, SymbolKind::Test);
        assert_eq!(f.symbols[1].kind, SymbolKind::Function);
        assert!(f.is_test);
    }

    #[test]
    fn java_class_methods_and_junit_tests() {
        let src = "import com.acme.ledger.Refund;\npublic class Ledger {\n    public int total(int a) {\n        return a;\n    }\n    @Test\n    public void checksTotal() {\n    }\n}\n";
        let f = ex(Language::Java).extract("src/Ledger.java", src);
        assert_eq!(f.imports[0].module, "com/acme/ledger/Refund");
        let total = f.symbols.iter().find(|s| s.name == "total").expect("total");
        assert_eq!(total.kind, SymbolKind::Method);
        let t = f
            .symbols
            .iter()
            .find(|s| s.name == "checksTotal")
            .expect("test");
        assert_eq!(t.kind, SymbolKind::Test);
    }

    #[test]
    fn c_includes_and_functions() {
        let src = "#include \"util/strbuf.h\"\n#include <stdio.h>\n\nstatic int add_one(int x)\n{\n    return x + 1;\n}\n";
        let f = ex(Language::C).extract("src/main.c", src);
        let mods: Vec<&str> = f.imports.iter().map(|i| i.module.as_str()).collect();
        assert_eq!(mods, vec!["./util/strbuf.h"]);
        let add = f
            .symbols
            .iter()
            .find(|s| s.name == "add_one")
            .expect("add_one");
        assert_eq!(add.kind, SymbolKind::Function);
        assert_eq!((add.start_line, add.end_line), (4, 7));
    }

    #[test]
    fn comments_do_not_produce_calls() {
        let src = "// legacy(x) call\nfn live() { run(); }\n";
        let f = ex(Language::Rust).extract("src/a.rs", src);
        assert!(f.calls.iter().all(|c| c.name != "legacy"));
        assert!(f.calls.iter().any(|c| c.name == "run"));
    }

    #[test]
    fn test_paths_are_detected() {
        assert!(is_test_path("tests/eval.rs"));
        assert!(is_test_path("src/foo_test.go"));
        assert!(is_test_path("web/cart.test.ts"));
        assert!(is_test_path("pkg/test_models.py"));
        assert!(!is_test_path("src/contest.rs"));
    }

    #[test]
    fn unknown_language_yields_nothing_structural() {
        let e = ex(Language::Other);
        let f = e.extract("notes.md", "fn looks_like_code() {}");
        assert!(f.symbols.is_empty());
    }

    #[test]
    fn default_extractors_cover_all_languages() {
        let all = default_extractors().expect("compile");
        assert_eq!(all.len(), 8);
        assert!(all.iter().any(|e| e.language() == Language::Go));
    }
}
