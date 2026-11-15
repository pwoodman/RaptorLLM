//! Dependency and symbol graph with incremental edge invalidation.
//!
//! Nodes are workspace-relative file paths and symbol ids (`path#line:Parent.name`).
//! Every edge is *owned* by the file it originates from, so re-resolving one file
//! replaces exactly its edges. A change to a file only re-resolves files that
//! reference a name or path segment the change touched (see `key_files`).

use crate::parse::{FileSymbols, Language, Symbol, SymbolKind};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// Relation kinds stored on edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeKind {
    /// File defines symbol.
    Defines,
    /// File imports file.
    Imports,
    /// Symbol (or file, at top level) calls symbol.
    Calls,
    /// Subject file is tested by test file.
    TestedBy,
}

/// A directed, typed relation between two nodes.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Edge {
    pub from: String,
    pub to: String,
    pub kind: EdgeKind,
}

/// Stable id of a symbol: `path#line:Parent.name`.
pub fn symbol_id(file: &str, sym: &Symbol) -> String {
    let qualified = match &sym.parent {
        Some(p) => format!("{p}.{}", sym.name),
        None => sym.name.clone(),
    };
    format!("{file}#{}:{qualified}", sym.start_line)
}

/// The file part of a symbol id.
pub fn file_of_symbol(id: &str) -> &str {
    id.split('#').next().unwrap_or(id)
}

fn dir_of(path: &str) -> &str {
    path.rsplit_once('/').map(|(d, _)| d).unwrap_or("")
}

fn last_segment(s: &str) -> &str {
    s.rsplit('/').next().unwrap_or(s)
}

const CODE_EXTS: &[&str] = &[
    ".ts", ".tsx", ".mts", ".cts", ".js", ".jsx", ".mjs", ".cjs", ".py", ".rs", ".go", ".java",
];

fn strip_known_ext(s: &str) -> &str {
    for ext in CODE_EXTS {
        if let Some(stripped) = s.strip_suffix(ext) {
            return stripped;
        }
    }
    s
}

/// The module path a file answers to: extension and `mod`/`index`/`__init__` removed.
/// C and C++ keep their extension because `#include` names it.
fn module_of(path: &str, lang: Language) -> String {
    if matches!(lang, Language::C | Language::Cpp) {
        return path.to_string();
    }
    let file = last_segment(path);
    let stem = file
        .rsplit_once('.')
        .filter(|(s, _)| !s.is_empty())
        .map(|(s, _)| s)
        .unwrap_or(file);
    if matches!(stem, "mod" | "index" | "__init__") {
        return dir_of(path).to_string();
    }
    match dir_of(path) {
        "" => stem.to_string(),
        d => format!("{d}/{stem}"),
    }
}

/// Joins a relative import onto `base_dir`, resolving `.` and `..`. None if it escapes the root.
fn normalize_join(base_dir: &str, rel: &str) -> Option<String> {
    let mut parts: Vec<&str> = base_dir.split('/').filter(|s| !s.is_empty()).collect();
    for seg in rel.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                parts.pop()?;
            }
            s => parts.push(s),
        }
    }
    Some(parts.join("/"))
}

/// Test base name for a test file following a naming convention, e.g. `cart.test.ts` -> `cart`.
fn test_base(path: &str) -> Option<String> {
    let file = last_segment(path);
    let stem = file.rsplit_once('.').map(|(s, _)| s).unwrap_or(file);
    if let Some(rest) = stem.strip_prefix("test_") {
        return Some(rest.to_string());
    }
    if let Some(rest) = stem.strip_suffix("_test") {
        return Some(rest.to_string());
    }
    if let Some(rest) = stem
        .strip_suffix(".test")
        .or_else(|| stem.strip_suffix(".spec"))
    {
        return Some(rest.to_string());
    }
    None
}

/// Calls to a name that matches more than this many symbols, with no local or
/// imported match, are dropped as too ambiguous to link.
const MAX_GLOBAL_CALL_TARGETS: usize = 3;

/// The graph. Built and maintained by [`Graph::apply`]; queried by the methods below.
#[derive(Debug, Default)]
pub struct Graph {
    facts: BTreeMap<String, FileSymbols>,
    languages: BTreeMap<String, Language>,
    file_symbol_ids: BTreeMap<String, Vec<String>>,
    symbol_file: BTreeMap<String, String>,
    symbol_name: BTreeMap<String, String>,
    names: BTreeMap<String, BTreeSet<String>>,
    /// file -> symbol name -> ids, so a call is resolved without scanning every definition of its name.
    defs: BTreeMap<String, BTreeMap<String, Vec<String>>>,
    /// Module key (see [`module_of`]) -> files. Exact lookups for relative imports.
    by_module: BTreeMap<String, BTreeSet<String>>,
    /// Every trailing path of a module key -> files. Suffix lookups for path-style imports.
    by_suffix: BTreeMap<String, BTreeSet<String>>,
    by_last: BTreeMap<String, BTreeSet<String>>,
    by_dir_last: BTreeMap<String, BTreeSet<String>>,
    file_keys: BTreeMap<String, BTreeSet<String>>,
    key_files: BTreeMap<String, BTreeSet<String>>,
    owned: BTreeMap<String, BTreeSet<Edge>>,
    out_adj: BTreeMap<String, BTreeSet<Edge>>,
    in_adj: BTreeMap<String, BTreeSet<Edge>>,
}

impl Graph {
    /// An empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of files in the graph.
    pub fn file_count(&self) -> usize {
        self.facts.len()
    }

    /// Number of symbol nodes.
    pub fn symbol_count(&self) -> usize {
        self.symbol_file.len()
    }

    /// Number of edges.
    pub fn edge_count(&self) -> usize {
        self.owned.values().map(BTreeSet::len).sum()
    }

    /// Number of import edges (file to file).
    pub fn import_edge_count(&self) -> usize {
        self.owned
            .values()
            .flat_map(|s| s.iter())
            .filter(|e| e.kind == EdgeKind::Imports)
            .count()
    }

    /// Applies file upserts and removals, re-resolving only the files they can affect.
    ///
    /// Returns how many files had their edges recomputed.
    pub fn apply(
        &mut self,
        upserts: Vec<(String, Language, FileSymbols)>,
        removed: &[String],
    ) -> usize {
        let mut affected: BTreeSet<String> = BTreeSet::new();
        let mut touched: BTreeSet<String> = BTreeSet::new();
        for path in removed {
            self.detach(path, &mut affected);
        }
        for (path, lang, syms) in upserts {
            self.detach(&path, &mut affected);
            self.attach(&path, lang, syms, &mut affected);
            touched.insert(path);
        }
        let mut dirty = touched;
        for key in &affected {
            if let Some(files) = self.key_files.get(key) {
                dirty.extend(files.iter().cloned());
            }
        }
        for file in &dirty {
            self.recompute(file);
        }
        dirty.len()
    }

    /// Removes everything the graph knows about `path`, recording the names and
    /// path segments it touched so dependents can be re-resolved.
    fn detach(&mut self, path: &str, affected: &mut BTreeSet<String>) {
        let Some(old) = self.facts.remove(path) else {
            return;
        };
        let lang = self.languages.remove(path).unwrap_or(Language::Other);
        for s in &old.symbols {
            affected.insert(s.name.clone());
        }
        affected.insert(last_segment(&module_of(path, lang)).to_string());
        affected.insert(last_segment(dir_of(path)).to_string());
        if let Some(ids) = self.file_symbol_ids.remove(path) {
            for id in ids {
                self.symbol_file.remove(&id);
                if let Some(name) = self.symbol_name.remove(&id) {
                    if let Some(set) = self.names.get_mut(&name) {
                        set.remove(&id);
                        if set.is_empty() {
                            self.names.remove(&name);
                        }
                    }
                }
            }
        }
        self.defs.remove(path);
        let module = module_of(path, lang);
        remove_from_index(&mut self.by_module, &module, path);
        for suffix in module_suffixes(&module) {
            remove_from_index(&mut self.by_suffix, suffix, path);
        }
        remove_from_index(&mut self.by_last, last_segment(&module), path);
        remove_from_index(&mut self.by_dir_last, last_segment(dir_of(path)), path);
        if let Some(keys) = self.file_keys.remove(path) {
            for k in keys {
                remove_from_index(&mut self.key_files, &k, path);
            }
        }
        self.remove_owned(path);
    }

    /// Registers `path`'s new facts and the names it now provides.
    fn attach(
        &mut self,
        path: &str,
        lang: Language,
        syms: FileSymbols,
        affected: &mut BTreeSet<String>,
    ) {
        let mut ids = Vec::with_capacity(syms.symbols.len());
        for s in &syms.symbols {
            let id = symbol_id(path, s);
            self.symbol_file.insert(id.clone(), path.to_string());
            self.symbol_name.insert(id.clone(), s.name.clone());
            self.names
                .entry(s.name.clone())
                .or_default()
                .insert(id.clone());
            affected.insert(s.name.clone());
            ids.push(id);
        }
        self.file_symbol_ids.insert(path.to_string(), ids);
        let module = module_of(path, lang);
        let stem = last_segment(&module).to_string();
        let dir_last = last_segment(dir_of(path)).to_string();
        affected.insert(stem.clone());
        affected.insert(dir_last.clone());
        self.by_module
            .entry(module.clone())
            .or_default()
            .insert(path.to_string());
        for suffix in module_suffixes(&module) {
            self.by_suffix
                .entry(suffix.to_string())
                .or_default()
                .insert(path.to_string());
        }
        let mut by_name: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for s in &syms.symbols {
            by_name
                .entry(s.name.clone())
                .or_default()
                .push(symbol_id(path, s));
        }
        self.defs.insert(path.to_string(), by_name);
        self.by_last
            .entry(stem)
            .or_default()
            .insert(path.to_string());
        self.by_dir_last
            .entry(dir_last)
            .or_default()
            .insert(path.to_string());
        let keys = referenced_keys(path, &syms);
        for k in &keys {
            self.key_files
                .entry(k.clone())
                .or_default()
                .insert(path.to_string());
        }
        self.file_keys.insert(path.to_string(), keys);
        self.languages.insert(path.to_string(), lang);
        self.facts.insert(path.to_string(), syms);
    }

    fn remove_owned(&mut self, file: &str) {
        let Some(edges) = self.owned.remove(file) else {
            return;
        };
        for e in edges {
            if let Some(set) = self.out_adj.get_mut(&e.from) {
                set.remove(&e);
            }
            if let Some(set) = self.in_adj.get_mut(&e.to) {
                set.remove(&e);
            }
        }
    }

    fn recompute(&mut self, file: &str) {
        if !self.facts.contains_key(file) {
            return;
        }
        self.remove_owned(file);
        let edges = self.compute_edges(file);
        for e in &edges {
            self.out_adj
                .entry(e.from.clone())
                .or_default()
                .insert(e.clone());
            self.in_adj
                .entry(e.to.clone())
                .or_default()
                .insert(e.clone());
        }
        self.owned.insert(file.to_string(), edges);
    }

    fn compute_edges(&self, file: &str) -> BTreeSet<Edge> {
        let mut edges = BTreeSet::new();
        let Some(facts) = self.facts.get(file) else {
            return edges;
        };
        let lang = self.languages.get(file).copied().unwrap_or(Language::Other);

        for s in &facts.symbols {
            edges.insert(Edge {
                from: file.to_string(),
                to: symbol_id(file, s),
                kind: EdgeKind::Defines,
            });
        }

        let mut imported: BTreeSet<String> = BTreeSet::new();
        for imp in &facts.imports {
            for target in self.resolve(file, lang, &imp.module) {
                if target != file {
                    imported.insert(target);
                }
            }
        }
        for target in &imported {
            edges.insert(Edge {
                from: file.to_string(),
                to: target.clone(),
                kind: EdgeKind::Imports,
            });
        }

        for call in &facts.calls {
            let caller =
                enclosing_symbol(file, facts, call.line).unwrap_or_else(|| file.to_string());
            for target in self.call_targets(file, &call.name, &imported) {
                if target != caller {
                    edges.insert(Edge {
                        from: caller.clone(),
                        to: target,
                        kind: EdgeKind::Calls,
                    });
                }
            }
        }

        if facts.is_test {
            let mut subjects: BTreeSet<String> = imported.clone();
            subjects.extend(self.name_subjects(file, lang));
            for subject in subjects {
                edges.insert(Edge {
                    from: subject,
                    to: file.to_string(),
                    kind: EdgeKind::TestedBy,
                });
            }
        }
        edges
    }

    /// Files an import key can refer to, in sorted order.
    fn resolve(&self, from: &str, lang: Language, module: &str) -> Vec<String> {
        if module.starts_with("./") || module.starts_with("../") {
            // Longest prefix first: `./refund/Refund` names type `Refund` inside module
            // `refund`, so when the full path is not a file the leading segments are tried.
            // A prefix must keep at least one real segment, so `./` alone never widens
            // into a whole directory.
            let segs: Vec<&str> = module.split('/').filter(|s| !s.is_empty()).collect();
            let ends = (1..=segs.len())
                .rev()
                .filter(|e| *e == segs.len() || *e >= 2);
            for end in ends {
                let Some(candidate) = normalize_join(dir_of(from), &segs[..end].join("/")) else {
                    return Vec::new();
                };
                let hits = self.files_with_module(&candidate, lang);
                if !hits.is_empty() {
                    return hits;
                }
            }
            if matches!(lang, Language::C | Language::Cpp) {
                // `#include "x.h"` may also name a path relative to an include root.
                return self.suffix_matches(module.trim_start_matches("./"), lang);
            }
            return Vec::new();
        }
        if matches!(lang, Language::JavaScript | Language::TypeScript) {
            // Bare specifiers name packages, which are outside the repository.
            return Vec::new();
        }
        let segs: Vec<&str> = module.split('/').filter(|s| !s.is_empty()).collect();
        if segs.is_empty() {
            return Vec::new();
        }
        if lang == Language::Go {
            let last = segs[segs.len() - 1];
            let Some(pool) = self.by_dir_last.get(last) else {
                return Vec::new();
            };
            for start in 0..segs.len() {
                let key = segs[start..].join("/");
                let hits: Vec<String> = pool
                    .iter()
                    .filter(|f| {
                        let d = dir_of(f);
                        d == key || d.ends_with(&format!("/{key}"))
                    })
                    .cloned()
                    .collect();
                if !hits.is_empty() {
                    return hits;
                }
            }
            return Vec::new();
        }
        if lang == Language::Java {
            // A class import names a file exactly; a package wildcard names its directory.
            let exact = self.suffix_matches(&segs.join("/"), lang);
            if !exact.is_empty() {
                return exact;
            }
            let package = self.java_package_files(&segs.join("/"));
            if !package.is_empty() {
                return package;
            }
        }
        for end in (1..=segs.len()).rev() {
            let key = segs[..end].join("/");
            let hits = self.suffix_matches(&key, lang);
            if !hits.is_empty() {
                return hits;
            }
        }
        Vec::new()
    }

    /// Java files whose directory is the package `key` (or ends with it).
    fn java_package_files(&self, key: &str) -> Vec<String> {
        let last = last_segment(key);
        let Some(pool) = self.by_dir_last.get(last) else {
            return Vec::new();
        };
        pool.iter()
            .filter(|f| {
                let d = dir_of(f);
                d == key || d.ends_with(&format!("/{key}"))
            })
            .cloned()
            .collect()
    }

    fn files_with_module(&self, candidate: &str, lang: Language) -> Vec<String> {
        let clean = if matches!(lang, Language::C | Language::Cpp) {
            candidate.to_string()
        } else {
            strip_known_ext(candidate).to_string()
        };
        self.by_module
            .get(&clean)
            .map(|set| set.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Files whose module key ends with `key` (or equals it), sorted.
    fn suffix_matches(&self, key: &str, lang: Language) -> Vec<String> {
        let clean = if matches!(lang, Language::C | Language::Cpp) {
            key.to_string()
        } else {
            strip_known_ext(key).to_string()
        };
        self.by_suffix
            .get(&clean)
            .map(|set| set.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Targets of a call to `name` made from `file`: the file's own definitions, then
    /// definitions in files it imports, then (only when few exist) any definition of the name.
    /// Each lookup is keyed by file and name, so a common name costs no more than a rare one.
    fn call_targets(&self, file: &str, name: &str, imported: &BTreeSet<String>) -> Vec<String> {
        let defined_in = |f: &str| -> Vec<String> {
            self.defs
                .get(f)
                .and_then(|m| m.get(name))
                .cloned()
                .unwrap_or_default()
        };
        let local = defined_in(file);
        if !local.is_empty() {
            return local;
        }
        let via_import: Vec<String> = imported.iter().flat_map(|f| defined_in(f)).collect();
        if !via_import.is_empty() {
            return via_import;
        }
        match self.names.get(name) {
            Some(ids) if ids.len() <= MAX_GLOBAL_CALL_TARGETS => ids.iter().cloned().collect(),
            _ => Vec::new(),
        }
    }

    /// Non-test files whose base name matches this test file's naming convention.
    fn name_subjects(&self, file: &str, lang: Language) -> Vec<String> {
        let Some(base) = test_base(file) else {
            return Vec::new();
        };
        let Some(pool) = self.by_last.get(&base) else {
            return Vec::new();
        };
        pool.iter()
            .filter(|f| f.as_str() != file)
            .filter(|f| self.languages.get(*f).copied() == Some(lang))
            .filter(|f| self.facts.get(*f).is_some_and(|s| !s.is_test))
            .cloned()
            .collect()
    }

    /// Files that `file` imports, sorted.
    pub fn imports_of(&self, file: &str) -> Vec<String> {
        self.out_targets(file, EdgeKind::Imports)
    }

    /// Files that import `file`, sorted (direct dependents).
    pub fn dependents_of(&self, file: &str) -> Vec<String> {
        self.in_sources(file, EdgeKind::Imports)
    }

    /// Number of files that import `file`. Edges are unique per (importer, target), so this counts importers.
    pub fn dependent_count(&self, file: &str) -> usize {
        self.in_adj
            .get(file)
            .map(|s| s.iter().filter(|e| e.kind == EdgeKind::Imports).count())
            .unwrap_or(0)
    }

    /// Test files that cover `file`, sorted.
    pub fn related_tests(&self, file: &str) -> Vec<String> {
        self.out_targets(file, EdgeKind::TestedBy)
    }

    /// Files that depend on `file` transitively, with their distance, up to `max_depth`.
    pub fn transitive_dependents(&self, file: &str, max_depth: usize) -> BTreeMap<String, usize> {
        let mut dist = BTreeMap::new();
        let mut queue = VecDeque::from([(file.to_string(), 0usize)]);
        let mut visited = BTreeSet::from([file.to_string()]);
        while let Some((node, d)) = queue.pop_front() {
            if d >= max_depth {
                continue;
            }
            for dep in self.dependents_of(&node) {
                if visited.insert(dep.clone()) {
                    dist.insert(dep.clone(), d + 1);
                    queue.push_back((dep, d + 1));
                }
            }
        }
        dist
    }

    /// Files within `k` hops of any seed (seeds excluded), with their hop distance.
    /// Hops follow imports and test relations in either direction.
    pub fn neighbourhood(&self, seeds: &[String], k: usize) -> BTreeMap<String, usize> {
        let mut dist: BTreeMap<String, usize> = BTreeMap::new();
        let mut queue: VecDeque<(String, usize)> = VecDeque::new();
        let seed_set: BTreeSet<&String> = seeds.iter().collect();
        for s in seeds {
            if self.facts.contains_key(s) {
                dist.insert(s.clone(), 0);
                queue.push_back((s.clone(), 0));
            }
        }
        while let Some((node, d)) = queue.pop_front() {
            if d >= k {
                continue;
            }
            for next in self.file_neighbours(&node) {
                if !dist.contains_key(&next) {
                    dist.insert(next.clone(), d + 1);
                    queue.push_back((next, d + 1));
                }
            }
        }
        dist.retain(|f, _| !seed_set.contains(f));
        dist
    }

    fn file_neighbours(&self, file: &str) -> BTreeSet<String> {
        let mut out: BTreeSet<String> = BTreeSet::new();
        for kind in [EdgeKind::Imports, EdgeKind::TestedBy] {
            out.extend(self.out_targets(file, kind));
            out.extend(self.in_sources(file, kind));
        }
        out
    }

    /// Symbol ids declared with `name`, sorted.
    pub fn symbol_ids_named(&self, name: &str) -> Vec<String> {
        self.names
            .get(name)
            .map(|s| s.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Call edges into any symbol named `name`: each edge's `from` is the calling symbol or file.
    pub fn callers(&self, name: &str) -> Vec<Edge> {
        let mut out = Vec::new();
        for id in self.symbol_ids_named(name) {
            if let Some(set) = self.in_adj.get(&id) {
                out.extend(set.iter().filter(|e| e.kind == EdgeKind::Calls).cloned());
            }
        }
        out.sort();
        out
    }

    /// Call edges out of any symbol named `name`.
    pub fn callees(&self, name: &str) -> Vec<Edge> {
        let mut out = Vec::new();
        for id in self.symbol_ids_named(name) {
            if let Some(set) = self.out_adj.get(&id) {
                out.extend(set.iter().filter(|e| e.kind == EdgeKind::Calls).cloned());
            }
        }
        out.sort();
        out
    }

    /// The file a symbol id belongs to.
    pub fn symbol_file(&self, id: &str) -> Option<&str> {
        self.symbol_file.get(id).map(String::as_str)
    }

    fn out_targets(&self, node: &str, kind: EdgeKind) -> Vec<String> {
        let mut v: Vec<String> = self
            .out_adj
            .get(node)
            .map(|s| {
                s.iter()
                    .filter(|e| e.kind == kind)
                    .map(|e| e.to.clone())
                    .collect()
            })
            .unwrap_or_default();
        v.sort();
        v.dedup();
        v
    }

    fn in_sources(&self, node: &str, kind: EdgeKind) -> Vec<String> {
        let mut v: Vec<String> = self
            .in_adj
            .get(node)
            .map(|s| {
                s.iter()
                    .filter(|e| e.kind == kind)
                    .map(|e| e.from.clone())
                    .collect()
            })
            .unwrap_or_default();
        v.sort();
        v.dedup();
        v
    }
}

/// Every trailing path of a module key: `a/b/c` gives `b/c`, `c` and `a/b/c`.
fn module_suffixes(module: &str) -> Vec<&str> {
    let mut out: Vec<&str> = module
        .match_indices('/')
        .map(|(i, _)| &module[i + 1..])
        .collect();
    if !module.is_empty() {
        out.push(module);
    }
    out
}

fn remove_from_index(index: &mut BTreeMap<String, BTreeSet<String>>, key: &str, path: &str) {
    if let Some(set) = index.get_mut(key) {
        set.remove(path);
        if set.is_empty() {
            index.remove(key);
        }
    }
}

/// Keys whose changes can affect how this file resolves: import path segments,
/// called names and the subject name of a test file.
fn referenced_keys(path: &str, syms: &FileSymbols) -> BTreeSet<String> {
    let mut keys = BTreeSet::new();
    for imp in &syms.imports {
        for seg in imp.module.split('/') {
            if !seg.is_empty() && seg != "." && seg != ".." {
                keys.insert(seg.to_string());
            }
        }
    }
    for c in &syms.calls {
        keys.insert(c.name.clone());
    }
    if syms.is_test {
        if let Some(base) = test_base(path) {
            keys.insert(base);
        }
    }
    keys
}

/// Innermost symbol of `file` whose range contains `line`, as a symbol id.
fn enclosing_symbol(file: &str, syms: &FileSymbols, line: u32) -> Option<String> {
    syms.symbols
        .iter()
        .filter(|s| s.start_line <= line && line <= s.end_line && s.kind != SymbolKind::Module)
        .max_by_key(|s| s.start_line)
        .map(|s| symbol_id(file, s))
}

#[cfg(test)]
pub(crate) mod testing {
    use super::*;
    use crate::parse::{CallRef, Import};

    pub fn sym(name: &str, kind: SymbolKind, start: u32, end: u32, parent: Option<&str>) -> Symbol {
        Symbol {
            name: name.into(),
            kind,
            start_line: start,
            end_line: end,
            parent: parent.map(String::from),
        }
    }

    pub fn import(module: &str) -> Import {
        Import {
            raw: format!("use {module}"),
            module: module.into(),
            line: 1,
        }
    }

    pub fn call(name: &str, line: u32) -> CallRef {
        CallRef {
            name: name.into(),
            line,
        }
    }

    pub fn facts(
        symbols: Vec<Symbol>,
        imports: Vec<Import>,
        calls: Vec<CallRef>,
        is_test: bool,
    ) -> FileSymbols {
        FileSymbols {
            symbols,
            imports,
            calls,
            is_test,
        }
    }

    pub fn up(path: &str, lang: Language, f: FileSymbols) -> (String, Language, FileSymbols) {
        (path.to_string(), lang, f)
    }
}

#[cfg(test)]
mod tests {
    use super::testing::*;
    use super::*;

    fn rust_graph() -> Graph {
        let mut g = Graph::new();
        g.apply(
            vec![
                up(
                    "src/ledger/reconcile.rs",
                    Language::Rust,
                    facts(
                        vec![sym("reconcile_refunds", SymbolKind::Function, 1, 3, None)],
                        vec![],
                        vec![],
                        false,
                    ),
                ),
                up(
                    "src/main.rs",
                    Language::Rust,
                    facts(
                        vec![sym("main", SymbolKind::Function, 3, 5, None)],
                        vec![import("ledger/reconcile")],
                        vec![call("reconcile_refunds", 4)],
                        false,
                    ),
                ),
            ],
            &[],
        );
        g
    }

    #[test]
    fn import_resolves_to_module_file() {
        let g = rust_graph();
        assert_eq!(g.imports_of("src/main.rs"), vec!["src/ledger/reconcile.rs"]);
        assert_eq!(
            g.dependents_of("src/ledger/reconcile.rs"),
            vec!["src/main.rs"]
        );
    }

    #[test]
    fn call_links_to_imported_symbol_and_caller_is_enclosing_symbol() {
        let g = rust_graph();
        let callers = g.callers("reconcile_refunds");
        assert_eq!(callers.len(), 1);
        assert_eq!(callers[0].from, "src/main.rs#3:main");
        assert_eq!(g.callees("main").len(), 1);
    }

    #[test]
    fn incremental_change_only_recomputes_referencing_files() {
        let mut g = rust_graph();
        // An unrelated file that shares no names or path segments with the change.
        g.apply(
            vec![up(
                "docs/notes.rs",
                Language::Rust,
                facts(
                    vec![sym("unrelated", SymbolKind::Function, 1, 1, None)],
                    vec![],
                    vec![],
                    false,
                ),
            )],
            &[],
        );
        // Renaming the callee touches the call key, so main.rs is re-resolved too.
        let recomputed = g.apply(
            vec![up(
                "src/ledger/reconcile.rs",
                Language::Rust,
                facts(
                    vec![sym("reconcile_all", SymbolKind::Function, 1, 3, None)],
                    vec![],
                    vec![],
                    false,
                ),
            )],
            &[],
        );
        assert_eq!(
            recomputed, 2,
            "reconcile.rs itself and main.rs which calls the old name"
        );
        assert!(g.callers("reconcile_refunds").is_empty());
    }

    #[test]
    fn removing_a_file_drops_its_edges_and_dependents_links() {
        let mut g = rust_graph();
        g.apply(vec![], &["src/ledger/reconcile.rs".to_string()]);
        assert!(g.imports_of("src/main.rs").is_empty());
        assert!(g.callers("reconcile_refunds").is_empty());
        assert_eq!(g.file_count(), 1);
    }

    #[test]
    fn transitive_dependents_respect_depth() {
        let mut g = Graph::new();
        g.apply(
            vec![
                up(
                    "a/core.rs",
                    Language::Rust,
                    facts(vec![], vec![], vec![], false),
                ),
                up(
                    "a/mid.rs",
                    Language::Rust,
                    facts(vec![], vec![import("core")], vec![], false),
                ),
                up(
                    "a/top.rs",
                    Language::Rust,
                    facts(vec![], vec![import("mid")], vec![], false),
                ),
            ],
            &[],
        );
        let d1 = g.transitive_dependents("a/core.rs", 1);
        assert_eq!(d1.keys().cloned().collect::<Vec<_>>(), vec!["a/mid.rs"]);
        let d2 = g.transitive_dependents("a/core.rs", 3);
        assert_eq!(d2.get("a/top.rs"), Some(&2));
    }

    #[test]
    fn tests_linked_by_import_and_by_name() {
        let mut g = Graph::new();
        g.apply(
            vec![
                up(
                    "src/cart.ts",
                    Language::TypeScript,
                    facts(
                        vec![sym("total", SymbolKind::Function, 1, 2, None)],
                        vec![],
                        vec![],
                        false,
                    ),
                ),
                up(
                    "src/misc.ts",
                    Language::TypeScript,
                    facts(vec![], vec![], vec![], false),
                ),
                up(
                    "tests/cart.test.ts",
                    Language::TypeScript,
                    facts(vec![], vec![import("../src/cart")], vec![], true),
                ),
                up(
                    "src/cart.spec.ts",
                    Language::TypeScript,
                    facts(vec![], vec![], vec![], true),
                ),
            ],
            &[],
        );
        assert_eq!(
            g.related_tests("src/cart.ts"),
            vec!["src/cart.spec.ts", "tests/cart.test.ts"]
        );
        assert!(g.related_tests("src/misc.ts").is_empty());
    }

    #[test]
    fn neighbourhood_is_multi_source_and_bounded() {
        let g = rust_graph();
        let n = g.neighbourhood(&["src/main.rs".to_string()], 1);
        assert_eq!(n.get("src/ledger/reconcile.rs"), Some(&1));
        assert!(!n.contains_key("src/main.rs"));
        let none = g.neighbourhood(&["src/main.rs".to_string()], 0);
        assert!(none.is_empty());
    }

    #[test]
    fn bare_js_packages_are_not_resolved_to_repo_files() {
        let mut g = Graph::new();
        g.apply(
            vec![
                up(
                    "src/react.ts",
                    Language::TypeScript,
                    facts(vec![], vec![], vec![], false),
                ),
                up(
                    "src/app.ts",
                    Language::TypeScript,
                    facts(vec![], vec![import("react")], vec![], false),
                ),
            ],
            &[],
        );
        assert!(g.imports_of("src/app.ts").is_empty());
    }

    #[test]
    fn relative_ts_import_resolves_index_and_extension() {
        let mut g = Graph::new();
        g.apply(
            vec![
                up(
                    "src/utils/index.ts",
                    Language::TypeScript,
                    facts(vec![], vec![], vec![], false),
                ),
                up(
                    "src/pay.ts",
                    Language::TypeScript,
                    facts(
                        vec![],
                        vec![import("./utils"), import("./utils/index.ts")],
                        vec![],
                        false,
                    ),
                ),
            ],
            &[],
        );
        assert_eq!(g.imports_of("src/pay.ts"), vec!["src/utils/index.ts"]);
    }

    #[test]
    fn python_package_import_resolves_init() {
        let mut g = Graph::new();
        g.apply(
            vec![
                up(
                    "pkg/core/__init__.py",
                    Language::Python,
                    facts(vec![], vec![], vec![], false),
                ),
                up(
                    "pkg/app.py",
                    Language::Python,
                    facts(vec![], vec![import("pkg/core")], vec![], false),
                ),
            ],
            &[],
        );
        assert_eq!(g.imports_of("pkg/app.py"), vec!["pkg/core/__init__.py"]);
    }

    #[test]
    fn go_import_matches_package_directory_by_suffix() {
        let mut g = Graph::new();
        g.apply(
            vec![
                up(
                    "internal/store/db.go",
                    Language::Go,
                    facts(vec![], vec![], vec![], false),
                ),
                up(
                    "cmd/app/main.go",
                    Language::Go,
                    facts(
                        vec![],
                        vec![import("example.com/app/internal/store")],
                        vec![],
                        false,
                    ),
                ),
            ],
            &[],
        );
        assert_eq!(
            g.imports_of("cmd/app/main.go"),
            vec!["internal/store/db.go"]
        );
    }

    #[test]
    fn c_include_resolves_relative_then_root() {
        let mut g = Graph::new();
        g.apply(
            vec![
                up(
                    "src/util/strbuf.h",
                    Language::C,
                    facts(vec![], vec![], vec![], false),
                ),
                up(
                    "src/main.c",
                    Language::C,
                    facts(vec![], vec![import("./util/strbuf.h")], vec![], false),
                ),
                up(
                    "lib/net/sock.h",
                    Language::C,
                    facts(vec![], vec![], vec![], false),
                ),
                up(
                    "src/x.c",
                    Language::C,
                    facts(vec![], vec![import("./net/sock.h")], vec![], false),
                ),
            ],
            &[],
        );
        assert_eq!(g.imports_of("src/main.c"), vec!["src/util/strbuf.h"]);
        assert_eq!(g.imports_of("src/x.c"), vec!["lib/net/sock.h"]);
    }

    #[test]
    fn ambiguous_global_calls_are_not_linked() {
        let mut g = Graph::new();
        let many: Vec<(String, Language, FileSymbols)> = (0..5)
            .map(|i| {
                up(
                    &format!("m{i}/dup.rs"),
                    Language::Rust,
                    facts(
                        vec![sym("helper", SymbolKind::Function, 1, 1, None)],
                        vec![],
                        vec![],
                        false,
                    ),
                )
            })
            .collect();
        g.apply(many, &[]);
        g.apply(
            vec![up(
                "z/caller.rs",
                Language::Rust,
                facts(vec![], vec![], vec![call("helper", 1)], false),
            )],
            &[],
        );
        assert!(g.callees("helper").is_empty());
    }

    #[test]
    fn symbol_ids_are_stable_and_qualified() {
        let s = sym("post", SymbolKind::Method, 6, 8, Some("Ledger"));
        assert_eq!(symbol_id("src/l.rs", &s), "src/l.rs#6:Ledger.post");
        assert_eq!(file_of_symbol("src/l.rs#6:Ledger.post"), "src/l.rs");
    }

    #[test]
    fn module_of_strips_mod_and_index_files() {
        assert_eq!(module_of("src/ledger/mod.rs", Language::Rust), "src/ledger");
        assert_eq!(module_of("web/index.ts", Language::TypeScript), "web");
        assert_eq!(
            module_of("src/a.test.ts", Language::TypeScript),
            "src/a.test"
        );
    }

    #[test]
    fn normalize_join_handles_dot_segments_and_escape() {
        assert_eq!(normalize_join("a/b", "../c/./d").as_deref(), Some("a/c/d"));
        assert_eq!(normalize_join("", "../x"), None);
    }

    #[test]
    fn test_base_follows_conventions() {
        assert_eq!(test_base("web/cart.test.ts").as_deref(), Some("cart"));
        assert_eq!(test_base("pkg/test_models.py").as_deref(), Some("models"));
        assert_eq!(test_base("ledger/book_test.go").as_deref(), Some("book"));
        assert_eq!(test_base("tests/eval.rs"), None);
    }

    #[test]
    fn edge_counts_reflect_owned_edges() {
        let g = rust_graph();
        assert_eq!(g.file_count(), 2);
        assert_eq!(g.symbol_count(), 2);
        assert_eq!(g.import_edge_count(), 1);
        assert!(g.edge_count() >= 4);
    }
}
