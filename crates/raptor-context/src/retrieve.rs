//! Hybrid retrieval over index chunks.
//!
//! Six ranked signals are fused with weighted reciprocal-rank fusion (RRF). The
//! weights depend on the task kind: a debugging task leans on recent changes, an
//! architecture task leans on graph neighbourhoods.

use crate::budget::TaskKind;
use crate::git::GitState;
use crate::graph::{file_of_symbol, symbol_id};
use crate::index::Index;
use crate::parse::SymbolKind;
use crate::text::term_set;
use crate::vector::VectorIndex;
use raptor_core::{RaptorError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Optional semantic embedder. Disabled by default ([`NoEmbedder`]).
///
/// When [`Embedder::dimensions`] is `Some`, chunk embeddings are computed during
/// indexing and the query is embedded at search time.
pub trait Embedder: Send + Sync {
    /// Vector width, or `None` when embedding is disabled.
    fn dimensions(&self) -> Option<usize>;
    /// Embeds `text`. Only called when [`Embedder::dimensions`] is `Some`.
    fn embed(&self, text: &str) -> Result<Vec<f32>>;
}

/// The embedder used when none is configured.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoEmbedder;

impl Embedder for NoEmbedder {
    fn dimensions(&self) -> Option<usize> {
        None
    }
    fn embed(&self, _text: &str) -> Result<Vec<f32>> {
        Err(RaptorError::Unsupported("no embedder is configured".into()))
    }
}

/// Per-signal weights for RRF. Each signal is one ranked list; a weight of zero switches it off.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Weights {
    pub lexical: f64,
    pub symbol: f64,
    /// Files within two hops of an active file, in either direction.
    pub graph: f64,
    pub recency: f64,
    pub pin: f64,
    pub embedding: f64,
    /// Callers and dependents of the active files (what a change to them affects).
    pub downstream: f64,
    /// Test code, and text that reports errors or failures.
    pub diagnostics: f64,
    /// Authentication, input-handling and secret-handling code.
    pub security: f64,
    /// Files that many other files import: the dependency overview.
    pub hubs: f64,
}

/// Signal weights for a task kind. The table is the only place these are tuned.
pub fn weights(kind: TaskKind) -> Weights {
    // lexical, symbol, graph, recency, pin, embedding, downstream, diagnostics, security, hubs
    let v: [f64; 10] = match kind {
        TaskKind::Debug => [1.0, 1.2, 0.8, 1.0, 1.0, 0.8, 0.4, 1.2, 0.0, 0.0],
        TaskKind::Architecture => [1.0, 0.8, 1.2, 0.3, 1.0, 1.0, 0.3, 0.0, 0.0, 1.2],
        TaskKind::Refactor => [1.0, 1.2, 1.2, 0.5, 1.0, 0.7, 1.5, 0.0, 0.0, 0.0],
        TaskKind::Feature => [1.0, 0.9, 0.9, 0.6, 1.0, 1.0, 0.3, 0.0, 0.0, 0.0],
        TaskKind::Ui => [1.0, 0.8, 0.7, 0.7, 1.0, 0.9, 0.2, 0.0, 0.0, 0.0],
        TaskKind::Security => [1.0, 1.0, 1.2, 0.4, 1.0, 0.8, 0.3, 0.0, 1.5, 0.0],
        TaskKind::Performance => [1.0, 0.9, 1.0, 0.6, 1.0, 0.8, 0.2, 0.0, 0.0, 0.0],
    };
    Weights {
        lexical: v[0],
        symbol: v[1],
        graph: v[2],
        recency: v[3],
        pin: v[4],
        embedding: v[5],
        downstream: v[6],
        diagnostics: v[7],
        security: v[8],
        hubs: v[9],
    }
}

/// Hops followed from an active file to its downstream dependents.
const DOWNSTREAM_HOPS: usize = 2;

/// Terms (after stemming) that mark failure or error text.
const ERROR_TERMS: &[&str] = &[
    "error",
    "panic",
    "panicked",
    "fail",
    "failed",
    "failure",
    "exception",
    "traceback",
    "assert",
    "assertion",
    "mismatch",
    "unwrap",
    "bug",
];

/// Terms (after stemming) that mark authentication, input or secret handling.
const SECURITY_TERMS: &[&str] = &[
    "auth",
    "authenticate",
    "authorize",
    "authorization",
    "login",
    "logout",
    "password",
    "token",
    "session",
    "cookie",
    "csrf",
    "sanitize",
    "sanitise",
    "escape",
    "validate",
    "validation",
    "input",
    "sql",
    "crypto",
    "encrypt",
    "decrypt",
    "secret",
    "permission",
    "credential",
    "jwt",
    "oauth",
    "middleware",
    "xss",
    "injection",
];

/// Reciprocal-rank-fusion damping constant.
const RRF_K: f64 = 60.0;
/// Longest list kept per signal, so large repositories stay bounded.
const LIST_CAP: usize = 200;
/// Files considered "recent" for the recency signal.
const RECENT_FILES: usize = 50;
/// Graph radius around active files.
const GRAPH_HOPS: usize = 2;
/// BM25 parameters.
const BM25_K1: f64 = 1.2;
const BM25_B: f64 = 0.75;

/// Stable reference to one chunk: file path and position in that file's chunk list.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ChunkKey {
    pub path: String,
    pub idx: usize,
}

/// One retrieval result with the reasons it ranked.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Hit {
    pub path: String,
    pub start_line: u32,
    pub end_line: u32,
    pub symbol: Option<String>,
    /// Content hash of the file when the hit was produced.
    pub hash: String,
    pub score: f64,
    pub why: Vec<String>,
}

/// Inputs to one search.
pub struct SearchParams<'a> {
    pub query: &'a str,
    pub kind: TaskKind,
    pub limit: usize,
    /// Files being edited: seeds for the graph signal.
    pub active: &'a BTreeSet<String>,
    /// Files the caller wants favoured.
    pub pinned: &'a BTreeSet<String>,
    /// File rank by recency, 0 is most recent. See [`recency_ranking`].
    pub recency: &'a BTreeMap<String, usize>,
    /// Query embedding, when an embedder is configured.
    pub query_vector: Option<&'a [f32]>,
    /// Persisted chunk vectors from an [`crate::vector::EmbeddingBackend`]. When `None`, the
    /// embedding signal reads the per-chunk vectors that the legacy [`Embedder`] stored.
    pub vectors: Option<&'a VectorIndex>,
}

/// Ranks files by recency: uncommitted changes first, then recent commits, then
/// filesystem modification time. Only ranks leave this function, never wall-clock times.
pub fn recency_ranking(index: &Index, git: &GitState) -> BTreeMap<String, usize> {
    let mut order: Vec<String> = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for path in git.changed.iter().chain(git.recent.iter()) {
        if index.file(path).is_some() && seen.insert(path.clone()) {
            order.push(path.clone());
        }
    }
    let mut rest: Vec<(&String, u64)> = index
        .files()
        .iter()
        .filter(|(p, _)| !seen.contains(*p))
        .map(|(p, f)| (p, f.record.mtime))
        .collect();
    rest.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    order.extend(rest.into_iter().map(|(p, _)| p.clone()));
    order.into_iter().enumerate().map(|(i, p)| (p, i)).collect()
}

/// Cosine similarity; 0 for mismatched or zero vectors.
pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum();
    let na: f64 = a
        .iter()
        .map(|x| f64::from(*x) * f64::from(*x))
        .sum::<f64>()
        .sqrt();
    let nb: f64 = b
        .iter()
        .map(|x| f64::from(*x) * f64::from(*x))
        .sum::<f64>()
        .sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// Identifier-like words of a query, lowercased, for exact symbol matching.
fn query_identifiers(query: &str) -> BTreeSet<String> {
    query
        .split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .filter(|w| w.chars().count() >= 2)
        .map(|w| w.to_lowercase())
        .collect()
}

/// Runs hybrid retrieval over `index`. Deterministic: ties break by path, then line.
///
/// Signals work from file-level candidates or the inverted index, so the cost follows the
/// query and the active files rather than the size of the repository. The embedding signal
/// is the exception: it scores every chunk, and only runs when an embedder is configured.
pub fn search(index: &Index, p: &SearchParams<'_>) -> Vec<Hit> {
    let qterms: Vec<String> = term_set(p.query).into_iter().collect();
    let idents = query_identifiers(p.query);
    let w = weights(p.kind);
    // The overview signal ranks files without any query, so it keeps an empty query alive.
    if index.postings().live() == 0
        || (qterms.is_empty()
            && idents.is_empty()
            && p.query_vector.is_none()
            && p.active.is_empty()
            && p.pinned.is_empty()
            && w.hubs <= 0.0)
    {
        return Vec::new();
    }

    let lexical = lexical_ranking(index, &qterms);
    let symbol = symbol_ranking(index, &qterms, &idents);
    let graph = graph_ranking(index, p.active);
    let recency = recency_list(index, p.recency);
    let pinned = pinned_list(index, p.pinned);
    let embedding = match p.vectors {
        Some(store) => vector_ranking(index, store, p.query_vector),
        None => embedding_ranking(index, p.query_vector),
    };
    let downstream = if w.downstream > 0.0 {
        downstream_ranking(index, p.active)
    } else {
        Vec::new()
    };
    let diagnostics = if w.diagnostics > 0.0 {
        diagnostics_ranking(index, &qterms)
    } else {
        Vec::new()
    };
    let security = if w.security > 0.0 {
        security_ranking(index)
    } else {
        Vec::new()
    };
    let hubs = if w.hubs > 0.0 {
        hubs_ranking(index)
    } else {
        Vec::new()
    };

    let signals: [(f64, &[(ChunkKey, String)]); 10] = [
        (w.lexical, &lexical),
        (w.symbol, &symbol),
        (w.graph, &graph),
        (w.recency, &recency),
        (w.pin, &pinned),
        (w.embedding, &embedding),
        (w.downstream, &downstream),
        (w.diagnostics, &diagnostics),
        (w.security, &security),
        (w.hubs, &hubs),
    ];
    let mut fused: BTreeMap<ChunkKey, (f64, Vec<String>)> = BTreeMap::new();
    for (weight, list) in signals {
        if weight <= 0.0 {
            continue;
        }
        for (rank, (key, reason)) in list.iter().enumerate() {
            let entry = fused.entry(key.clone()).or_insert((0.0, Vec::new()));
            entry.0 += weight / (RRF_K + rank as f64 + 1.0);
            entry.1.push(reason.clone());
        }
    }

    let mut hits: Vec<Hit> = fused
        .into_iter()
        .filter_map(|(key, (score, why))| {
            let file = index.file(&key.path)?;
            let chunk = file.chunks.get(key.idx)?;
            Some(Hit {
                path: key.path.clone(),
                start_line: chunk.start_line,
                end_line: chunk.end_line,
                symbol: chunk.symbol.clone(),
                hash: file.record.hash.clone(),
                score,
                why,
            })
        })
        .collect();
    hits.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.path.cmp(&b.path))
            .then_with(|| a.start_line.cmp(&b.start_line))
    });
    hits.truncate(p.limit);
    hits
}

fn idf(n: f64, df: f64) -> f64 {
    (1.0 + (n - df + 0.5) / (df + 0.5)).ln()
}

/// BM25 over the inverted index. Only chunks that share a term with the query are scored,
/// so the cost follows the query's postings rather than the size of the repository.
fn lexical_ranking(index: &Index, qterms: &[String]) -> Vec<(ChunkKey, String)> {
    let post = index.postings();
    let n = post.live() as f64;
    if qterms.is_empty() || n == 0.0 {
        return Vec::new();
    }
    let avgdl = post.avg_len().max(1.0);
    // slot -> (score, matched terms in query order)
    let mut acc: BTreeMap<u32, (f64, Vec<&str>)> = BTreeMap::new();
    for t in qterms {
        let Some(list) = post.postings(t) else {
            continue;
        };
        let df = list.len() as f64;
        for &(slot, tf) in list {
            let Some(s) = post.slot(slot) else {
                continue;
            };
            let tf = f64::from(tf);
            let dl = f64::from(s.len);
            let score = idf(n, df) * tf * (BM25_K1 + 1.0)
                / (tf + BM25_K1 * (1.0 - BM25_B + BM25_B * dl / avgdl));
            let entry = acc.entry(slot).or_insert((0.0, Vec::new()));
            entry.0 += score;
            entry.1.push(t.as_str());
        }
    }
    let key_of = |slot: u32| -> (&str, usize) {
        post.slot(slot)
            .map(|s| (s.path.as_str(), s.idx))
            .unwrap_or(("", 0))
    };
    let mut scored: Vec<(f64, u32)> = acc
        .iter()
        .filter(|(_, (score, _))| *score > 0.0)
        .map(|(slot, (score, _))| (*score, *slot))
        .collect();
    scored.sort_by(|a, b| {
        b.0.total_cmp(&a.0)
            .then_with(|| key_of(a.1).cmp(&key_of(b.1)))
    });
    scored
        .into_iter()
        .take(LIST_CAP)
        .map(|(_, slot)| {
            let (path, idx) = key_of(slot);
            let matched = acc
                .get(&slot)
                .map(|(_, m)| m.as_slice())
                .unwrap_or_default();
            let shown: Vec<&str> = matched.iter().take(4).copied().collect();
            (
                ChunkKey {
                    path: path.to_string(),
                    idx,
                },
                format!("matches terms: {}", shown.join(", ")),
            )
        })
        .collect()
}

/// ASCII case-insensitive substring test. `needle` must already be lowercase.
fn contains_ci(hay: &str, needle: &str) -> bool {
    let (h, n) = (hay.as_bytes(), needle.as_bytes());
    n.is_empty() || (h.len() >= n.len() && h.windows(n.len()).any(|w| w.eq_ignore_ascii_case(n)))
}

/// Chunks of symbols whose name is a query identifier (score 2) or contains a query term of
/// four or more characters (score 1). Every chunk of a matching symbol is listed.
fn symbol_ranking(
    index: &Index,
    qterms: &[String],
    idents: &BTreeSet<String>,
) -> Vec<(ChunkKey, String)> {
    let mut scored: Vec<(u32, ChunkKey, &str)> = Vec::new();
    for (path, file) in index.files() {
        let mut names: BTreeMap<&str, u32> = BTreeMap::new();
        for sym in file
            .facts
            .symbols
            .iter()
            .filter(|s| s.kind.is_chunk_worthy())
        {
            let score = if idents.iter().any(|i| sym.name.eq_ignore_ascii_case(i)) {
                2
            } else if qterms
                .iter()
                .any(|t| t.chars().count() >= 4 && contains_ci(&sym.name, t))
            {
                1
            } else {
                0
            };
            if score > 0 {
                let e = names.entry(sym.name.as_str()).or_insert(0);
                *e = (*e).max(score);
            }
        }
        if names.is_empty() {
            continue;
        }
        for (idx, chunk) in file.chunks.iter().enumerate() {
            let Some(name) = chunk.symbol.as_deref() else {
                continue;
            };
            if let Some(score) = names.get(name) {
                scored.push((
                    *score,
                    ChunkKey {
                        path: path.clone(),
                        idx,
                    },
                    name,
                ));
            }
        }
    }
    scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    scored
        .into_iter()
        .take(LIST_CAP)
        .map(|(_, key, name)| (key, format!("defines `{name}`, which matches the query")))
        .collect()
}

/// Expands files, nearest first (then by path), into chunk keys until the list cap is reached.
/// Files arrive already ordered, so the list is the same as sorting every chunk.
fn expand_files<'a>(
    index: &Index,
    files: impl Iterator<Item = (usize, &'a str, String)>,
) -> Vec<(ChunkKey, String)> {
    let mut out = Vec::new();
    for (rank, path, why) in files {
        let Some(file) = index.file(path) else {
            continue;
        };
        for idx in 0..file.chunks.len() {
            if out.len() >= LIST_CAP {
                return out;
            }
            out.push((
                ChunkKey {
                    path: path.to_string(),
                    idx,
                },
                if why.is_empty() {
                    format!("rank {rank}")
                } else {
                    why.clone()
                },
            ));
        }
    }
    out
}

fn graph_ranking(index: &Index, active: &BTreeSet<String>) -> Vec<(ChunkKey, String)> {
    if active.is_empty() {
        return Vec::new();
    }
    let seeds: Vec<String> = active.iter().cloned().collect();
    let dist = index.graph().neighbourhood(&seeds, GRAPH_HOPS);
    let mut files: Vec<(usize, &str)> = dist.iter().map(|(p, d)| (*d, p.as_str())).collect();
    files.sort();
    expand_files(
        index,
        files.into_iter().map(|(d, p)| {
            (
                d,
                p,
                format!("{d} hop(s) from an active file in the import/test graph"),
            )
        }),
    )
}

fn recency_list(index: &Index, recency: &BTreeMap<String, usize>) -> Vec<(ChunkKey, String)> {
    let mut files: Vec<(usize, &str)> = recency
        .iter()
        .filter(|(p, r)| **r < RECENT_FILES && index.file(p).is_some())
        .map(|(p, r)| (*r, p.as_str()))
        .collect();
    files.sort();
    expand_files(
        index,
        files.into_iter().map(|(r, p)| {
            let why = if r == 0 {
                "most recently changed file".to_string()
            } else {
                format!("recently changed or committed (rank {r})")
            };
            (r, p, why)
        }),
    )
}

fn pinned_list(index: &Index, pinned: &BTreeSet<String>) -> Vec<(ChunkKey, String)> {
    expand_files(
        index,
        pinned
            .iter()
            .map(|p| (0usize, p.as_str(), "pinned by the caller".to_string())),
    )
}

/// Cosine similarity of the query vector with every chunk embedding. Scans all chunks; it
/// runs only when an embedder is configured.
fn embedding_ranking(index: &Index, query: Option<&[f32]>) -> Vec<(ChunkKey, String)> {
    let Some(q) = query else {
        return Vec::new();
    };
    let mut scored: Vec<(f64, &str, usize)> = Vec::new();
    for (path, file) in index.files() {
        for (idx, c) in file.chunks.iter().enumerate() {
            if let Some(v) = c.embedding.as_deref() {
                let s = cosine(q, v);
                if s > 0.0 {
                    scored.push((s, path.as_str(), idx));
                }
            }
        }
    }
    scored.sort_by(|a, b| {
        b.0.total_cmp(&a.0)
            .then_with(|| (a.1, a.2).cmp(&(b.1, b.2)))
    });
    scored
        .into_iter()
        .take(LIST_CAP)
        .map(|(s, path, idx)| {
            (
                ChunkKey {
                    path: path.to_string(),
                    idx,
                },
                format!("semantic similarity {s:.2}"),
            )
        })
        .collect()
}

/// Cosine similarity of the query with the persisted vector of every chunk. A file contributes
/// only when its stored record matches the current file hash and chunk count, so vectors from
/// an older version of the file are never used.
fn vector_ranking(
    index: &Index,
    store: &VectorIndex,
    query: Option<&[f32]>,
) -> Vec<(ChunkKey, String)> {
    let Some(q) = query else {
        return Vec::new();
    };
    let mut scored: Vec<(f64, &str, usize)> = Vec::new();
    for (path, file) in index.files() {
        let Some(stored) = store.file(path) else {
            continue;
        };
        if stored.hash != file.record.hash || stored.keys.len() != file.chunks.len() {
            continue;
        }
        for (idx, key) in stored.keys.iter().enumerate() {
            let Some(v) = key.as_ref().and_then(|k| store.get(k)) else {
                continue;
            };
            let s = cosine(q, v);
            if s > 0.0 {
                scored.push((s, path.as_str(), idx));
            }
        }
    }
    scored.sort_by(|a, b| {
        b.0.total_cmp(&a.0)
            .then_with(|| (a.1, a.2).cmp(&(b.1, b.2)))
    });
    scored
        .into_iter()
        .take(LIST_CAP)
        .map(|(s, path, idx)| {
            (
                ChunkKey {
                    path: path.to_string(),
                    idx,
                },
                format!("semantic similarity {s:.2}"),
            )
        })
        .collect()
}

/// Keeps the nearest reason for a file: a smaller hop count replaces a larger one.
fn note_reach(map: &mut BTreeMap<String, (usize, String)>, path: String, hops: usize, why: String) {
    if map.get(&path).is_some_and(|(h, _)| *h <= hops) {
        return;
    }
    map.insert(path, (hops, why));
}

/// Files downstream of the active files: importers up to [`DOWNSTREAM_HOPS`] away, and files
/// that call a symbol an active file defines. This is the impact of a change.
fn downstream_ranking(index: &Index, active: &BTreeSet<String>) -> Vec<(ChunkKey, String)> {
    let g = index.graph();
    let mut reach: BTreeMap<String, (usize, String)> = BTreeMap::new();
    for a in active {
        for (path, hops) in g.transitive_dependents(a, DOWNSTREAM_HOPS) {
            let why = format!("{hops} hop(s) downstream of active file {a}");
            note_reach(&mut reach, path, hops, why);
        }
        let Some(entry) = index.file(a) else {
            continue;
        };
        for sym in entry
            .facts
            .symbols
            .iter()
            .filter(|s| s.kind.is_chunk_worthy())
        {
            let id = symbol_id(a, sym);
            for edge in g.callers(&sym.name) {
                if edge.to != id {
                    continue;
                }
                let caller = file_of_symbol(&edge.from).to_string();
                if &caller != a {
                    let why = format!("calls `{}`, defined in active file {a}", sym.name);
                    note_reach(&mut reach, caller, 1, why);
                }
            }
        }
    }
    let mut files: Vec<(usize, &str, String)> = reach
        .iter()
        .map(|(p, (hops, why))| (*hops, p.as_str(), why.clone()))
        .collect();
    files.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(b.1)));
    expand_files(index, files.into_iter())
}

/// Chunks that appear in the inverted index for any of `terms`, as sorted keys.
fn posting_keys<'a>(index: &Index, terms: impl Iterator<Item = &'a str>) -> BTreeSet<ChunkKey> {
    let post = index.postings();
    let mut out = BTreeSet::new();
    for t in terms {
        for &(slot, _) in post.postings(t).unwrap_or(&[]) {
            if let Some(s) = post.slot(slot) {
                out.insert(ChunkKey {
                    path: s.path.clone(),
                    idx: s.idx,
                });
            }
        }
    }
    out
}

/// Test code and chunks that report errors or failures, among chunks that match the query.
/// Score is the test weight (2) plus the distinct error terms (capped at 3). Restricting to
/// query matches keeps this a re-ranking of relevant code, not a general boost for tests.
fn diagnostics_ranking(index: &Index, qterms: &[String]) -> Vec<(ChunkKey, String)> {
    let mut test_ranges: BTreeMap<String, Vec<(u32, u32)>> = BTreeMap::new();
    let mut scored: Vec<(u32, ChunkKey, String)> = Vec::new();
    for key in posting_keys(index, qterms.iter().map(String::as_str)) {
        let Some(file) = index.file(&key.path) else {
            continue;
        };
        let Some(chunk) = file.chunks.get(key.idx) else {
            continue;
        };
        let ranges = test_ranges.entry(key.path.clone()).or_insert_with(|| {
            file.facts
                .symbols
                .iter()
                .filter(|s| s.kind == SymbolKind::Test)
                .map(|s| (s.start_line, s.end_line))
                .collect()
        });
        let in_test = file.facts.is_test
            || ranges
                .iter()
                .any(|(s, e)| *s <= chunk.start_line && chunk.start_line <= *e);
        let errors: Vec<&str> = ERROR_TERMS
            .iter()
            .copied()
            .filter(|t| chunk.terms.contains_key(*t))
            .collect();
        let score = if in_test { 2 } else { 0 } + errors.len().min(3) as u32;
        if score == 0 {
            continue;
        }
        let mut why: Vec<String> = Vec::new();
        if in_test {
            why.push("test code".to_string());
        }
        if !errors.is_empty() {
            let shown: Vec<&str> = errors.iter().take(3).copied().collect();
            why.push(format!("reports errors or failures ({})", shown.join(", ")));
        }
        scored.push((score, key, why.join("; ")));
    }
    scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    scored
        .into_iter()
        .take(LIST_CAP)
        .map(|(_, key, why)| (key, why))
        .collect()
}

/// Chunks that mention authentication, input or secret handling, most distinct terms first.
fn security_ranking(index: &Index) -> Vec<(ChunkKey, String)> {
    let mut scored: Vec<(usize, u32, ChunkKey, String)> = Vec::new();
    for key in posting_keys(index, SECURITY_TERMS.iter().copied()) {
        let Some(chunk) = index.file(&key.path).and_then(|f| f.chunks.get(key.idx)) else {
            continue;
        };
        let hits: Vec<&str> = SECURITY_TERMS
            .iter()
            .copied()
            .filter(|t| chunk.terms.contains_key(*t))
            .collect();
        if hits.is_empty() {
            continue;
        }
        let occurrences: u32 = hits
            .iter()
            .map(|t| chunk.terms.get(*t).copied().unwrap_or(0))
            .sum();
        let shown: Vec<&str> = hits.iter().take(4).copied().collect();
        let why = format!(
            "touches authentication or input handling ({})",
            shown.join(", ")
        );
        scored.push((hits.len(), occurrences, key, why));
    }
    scored.sort_by(|a, b| {
        b.0.cmp(&a.0)
            .then_with(|| b.1.cmp(&a.1))
            .then_with(|| a.2.cmp(&b.2))
    });
    scored
        .into_iter()
        .take(LIST_CAP)
        .map(|(_, _, key, why)| (key, why))
        .collect()
}

/// Chunks of the files that many others import, most-imported first: the dependency overview.
fn hubs_ranking(index: &Index) -> Vec<(ChunkKey, String)> {
    let g = index.graph();
    let mut files: Vec<(usize, &str)> = index
        .files()
        .keys()
        .filter_map(|p| {
            let d = g.dependent_count(p);
            (d > 0).then_some((d, p.as_str()))
        })
        .collect();
    files.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(b.1)));
    expand_files(
        index,
        files.into_iter().map(|(d, p)| {
            (
                d,
                p,
                format!("imported by {d} file(s): a hub of the dependency graph"),
            )
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::Index;
    use crate::parse::default_extractors;
    use std::path::Path;
    use std::sync::Arc;

    fn build(root: &Path) -> Index {
        let mut idx = Index::open(
            root,
            None,
            default_extractors().expect("ex"),
            Arc::new(NoEmbedder),
        )
        .expect("open");
        idx.update().expect("update");
        idx
    }

    fn write(root: &Path, rel: &str, body: &str) {
        let p = root.join(rel);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).expect("mkdir");
        }
        std::fs::write(p, body).expect("write");
    }

    fn params<'a>(
        query: &'a str,
        kind: TaskKind,
        active: &'a BTreeSet<String>,
        pinned: &'a BTreeSet<String>,
        recency: &'a BTreeMap<String, usize>,
    ) -> SearchParams<'a> {
        SearchParams {
            query,
            kind,
            limit: 10,
            active,
            pinned,
            recency,
            query_vector: None,
            vectors: None,
        }
    }

    #[test]
    fn weights_sum_positive_for_every_kind() {
        for kind in [
            TaskKind::Debug,
            TaskKind::Architecture,
            TaskKind::Refactor,
            TaskKind::Feature,
            TaskKind::Ui,
            TaskKind::Security,
            TaskKind::Performance,
        ] {
            let w = weights(kind);
            assert!(w.lexical > 0.0 && w.symbol > 0.0 && w.graph > 0.0);
        }
        assert!(weights(TaskKind::Debug).recency > weights(TaskKind::Architecture).recency);
        assert!(weights(TaskKind::Architecture).graph > weights(TaskKind::Debug).graph);
    }

    #[test]
    fn cosine_basics() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-9);
        assert_eq!(cosine(&[1.0, 0.0], &[0.0, 1.0]), 0.0);
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
        assert_eq!(cosine(&[1.0], &[1.0, 2.0]), 0.0);
    }

    #[test]
    fn query_identifiers_keep_underscored_names() {
        let ids = query_identifiers("why does reconcile_refunds fail?");
        assert!(ids.contains("reconcile_refunds"));
        assert!(ids.contains("fail"));
    }

    #[test]
    fn lexical_search_finds_the_relevant_file() {
        let t = tempfile::tempdir().expect("tempdir");
        write(
            t.path(),
            "src/ui/button.rs",
            "pub fn render_button() { draw(); }\n",
        );
        write(
            t.path(),
            "src/ledger/refund.rs",
            "pub fn issue_refund(amount: u64) -> u64 { amount }\n",
        );
        let idx = build(t.path());
        let none = BTreeSet::new();
        let rec = BTreeMap::new();
        let hits = search(
            &idx,
            &params("issue refund amount", TaskKind::Feature, &none, &none, &rec),
        );
        assert_eq!(
            hits.first().map(|h| h.path.as_str()),
            Some("src/ledger/refund.rs")
        );
        assert!(hits[0].why.iter().any(|w| w.starts_with("matches terms")));
    }

    #[test]
    fn symbol_match_is_reported_as_definition() {
        let t = tempfile::tempdir().expect("tempdir");
        write(t.path(), "a.rs", "pub fn reconcile_refunds() {}\n");
        write(t.path(), "b.rs", "pub fn other() {}\n");
        let idx = build(t.path());
        let none = BTreeSet::new();
        let rec = BTreeMap::new();
        let hits = search(
            &idx,
            &params("reconcile_refunds", TaskKind::Debug, &none, &none, &rec),
        );
        assert_eq!(hits[0].symbol.as_deref(), Some("reconcile_refunds"));
        assert!(hits[0].why.iter().any(|w| w.contains("defines")));
    }

    #[test]
    fn graph_signal_surfaces_neighbours_of_active_file() {
        let t = tempfile::tempdir().expect("tempdir");
        write(t.path(), "src/core/ledger.rs", "pub fn post() {}\n");
        write(
            t.path(),
            "src/api/handler.rs",
            "use crate::core::ledger;\npub fn handle() { ledger::post(); }\n",
        );
        write(t.path(), "src/zzz/unrelated.rs", "pub fn zzz() {}\n");
        let idx = build(t.path());
        let active = BTreeSet::from(["src/core/ledger.rs".to_string()]);
        let none = BTreeSet::new();
        let rec = BTreeMap::new();
        let hits = search(
            &idx,
            &params(
                "nothing matches this",
                TaskKind::Architecture,
                &active,
                &none,
                &rec,
            ),
        );
        assert!(
            hits.iter().any(|h| h.path == "src/api/handler.rs"),
            "{hits:?}"
        );
        assert!(hits.iter().all(|h| h.path != "src/zzz/unrelated.rs"));
    }

    #[test]
    fn recency_and_pins_contribute_without_query_match() {
        let t = tempfile::tempdir().expect("tempdir");
        write(t.path(), "a.rs", "fn a() {}\n");
        write(t.path(), "b.rs", "fn b() {}\n");
        let idx = build(t.path());
        let none = BTreeSet::new();
        let pinned = BTreeSet::from(["b.rs".to_string()]);
        let rec = BTreeMap::from([("a.rs".to_string(), 0usize)]);
        let hits = search(&idx, &params("zzzz", TaskKind::Debug, &none, &pinned, &rec));
        let paths: Vec<&str> = hits.iter().map(|h| h.path.as_str()).collect();
        assert!(paths.contains(&"a.rs"));
        assert!(paths.contains(&"b.rs"));
        assert!(hits
            .iter()
            .any(|h| h.why.iter().any(|w| w.contains("pinned"))));
    }

    #[test]
    fn recency_ranking_orders_changes_then_commits_then_mtime() {
        let t = tempfile::tempdir().expect("tempdir");
        write(t.path(), "a.rs", "fn a() {}\n");
        write(t.path(), "b.rs", "fn b() {}\n");
        write(t.path(), "c.rs", "fn c() {}\n");
        let idx = build(t.path());
        let git = GitState {
            is_repo: true,
            changed: BTreeSet::from(["c.rs".to_string()]),
            recent: vec!["b.rs".to_string()],
        };
        let rank = recency_ranking(&idx, &git);
        assert_eq!(rank.get("c.rs"), Some(&0));
        assert_eq!(rank.get("b.rs"), Some(&1));
        assert_eq!(rank.get("a.rs"), Some(&2));
    }

    #[test]
    fn results_are_deterministic_across_runs() {
        let t = tempfile::tempdir().expect("tempdir");
        write(t.path(), "a.rs", "fn refund() {}\n");
        write(t.path(), "b.rs", "fn refund() {}\n");
        let idx = build(t.path());
        let none = BTreeSet::new();
        let rec = BTreeMap::new();
        let first = search(
            &idx,
            &params("refund", TaskKind::Feature, &none, &none, &rec),
        );
        let second = search(
            &idx,
            &params("refund", TaskKind::Feature, &none, &none, &rec),
        );
        assert_eq!(first, second);
        assert_eq!(first[0].path, "a.rs", "ties break by path");
    }

    #[test]
    fn empty_query_returns_nothing() {
        let t = tempfile::tempdir().expect("tempdir");
        write(t.path(), "a.rs", "fn a() {}\n");
        let idx = build(t.path());
        let none = BTreeSet::new();
        let rec = BTreeMap::new();
        assert!(search(&idx, &params("", TaskKind::Feature, &none, &none, &rec)).is_empty());
    }

    #[test]
    fn embedding_signal_ranks_by_cosine() {
        struct Two;
        impl Embedder for Two {
            fn dimensions(&self) -> Option<usize> {
                Some(2)
            }
            fn embed(&self, text: &str) -> Result<Vec<f32>> {
                Ok(if text.contains("money") {
                    vec![1.0, 0.0]
                } else {
                    vec![0.0, 1.0]
                })
            }
        }
        let t = tempfile::tempdir().expect("tempdir");
        write(t.path(), "pay.rs", "fn money() {}\n");
        write(t.path(), "ui.rs", "fn colour() {}\n");
        let mut idx = Index::open(
            t.path(),
            None,
            default_extractors().expect("ex"),
            Arc::new(Two),
        )
        .expect("open");
        idx.update().expect("update");
        let none = BTreeSet::new();
        let rec = BTreeMap::new();
        let q = [1.0f32, 0.0];
        let p = SearchParams {
            query: "x",
            kind: TaskKind::Feature,
            limit: 5,
            active: &none,
            pinned: &none,
            recency: &rec,
            query_vector: Some(&q),
            vectors: None,
        };
        let hits = search(&idx, &p);
        assert_eq!(hits[0].path, "pay.rs");
        assert!(hits[0].why.iter().any(|w| w.contains("semantic")));
    }

    #[test]
    fn persisted_vectors_are_ignored_when_the_file_hash_differs() {
        use crate::vector::{EmbeddingBackend, VectorIndex};
        struct Axis;
        impl EmbeddingBackend for Axis {
            fn model_id(&self) -> &str {
                "axis"
            }
            fn dimension(&self) -> usize {
                2
            }
            fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
                Ok(texts.iter().map(|_| vec![1.0, 0.0]).collect())
            }
        }
        let t = tempfile::tempdir().expect("tempdir");
        write(t.path(), "pay.rs", "fn money() {}\n");
        let idx = build(t.path());
        let hash = idx.file("pay.rs").expect("file").record.hash.clone();
        let mut store = VectorIndex::empty("axis", 2, 100);
        let texts = vec!["fn money() {}".to_string()];
        store.sync_file(&Axis, "pay.rs", "stale-hash", &texts);
        let none = BTreeSet::new();
        let rec = BTreeMap::new();
        let q = [1.0f32, 0.0];
        let p = SearchParams {
            query: "x",
            kind: TaskKind::Feature,
            limit: 5,
            active: &none,
            pinned: &none,
            recency: &rec,
            query_vector: Some(&q),
            vectors: Some(&store),
        };
        let hits = search(&idx, &p);
        assert!(
            hits.iter()
                .all(|h| !h.why.iter().any(|w| w.contains("semantic"))),
            "{hits:?}"
        );
        assert_ne!(hash, "stale-hash");
    }

    #[test]
    fn recency_ranking_ignores_unindexed_git_paths() {
        let t = tempfile::tempdir().expect("tempdir");
        write(t.path(), "a.rs", "fn a() {}\n");
        let idx = build(t.path());
        let git = GitState {
            is_repo: true,
            changed: BTreeSet::from(["gone.rs".to_string()]),
            recent: vec![],
        };
        let rank = recency_ranking(&idx, &git);
        assert!(!rank.contains_key("gone.rs"));
        assert_eq!(rank.get("a.rs"), Some(&0));
    }
}
