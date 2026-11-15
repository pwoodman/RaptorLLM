//! The repository index: scanned files, parsed facts, retrieval chunks and the
//! graph, updated incrementally and persisted as a versioned JSON store.

use crate::graph::Graph;
use crate::parse::{is_test_path, Backend, Extractor, FileSymbols, Language};
use crate::postings::Postings;
use crate::retrieve::Embedder;
use crate::scan::{scan_each_with, Exclusion, FileRecord, ScanConfig, ScanEvent, ScannedFile};
use crate::text::{term_freq, terms};
use raptor_core::{RaptorError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Schema version of the persisted store. A different version is discarded and rebuilt.
/// Version 2 added the per-file extraction backend.
pub const INDEX_VERSION: u32 = 2;

/// Line windows used for ranges not covered by a symbol.
const WINDOW_LINES: usize = 60;
/// Symbol bodies longer than this are split into windows of this size.
const MAX_SYMBOL_CHUNK_LINES: usize = 150;

/// A retrievable slice of a file. Text is not stored: it is re-read from disk and
/// checked against the file hash when it is needed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Chunk {
    /// 1-based inclusive first line.
    pub start_line: u32,
    /// 1-based inclusive last line.
    pub end_line: u32,
    /// Enclosing declaration, when the chunk is a symbol body.
    pub symbol: Option<String>,
    /// Term frequencies, including the file path terms.
    pub terms: BTreeMap<String, u32>,
    /// Sum of term frequencies (document length for BM25).
    pub len: u32,
    /// Optional embedding, computed at parse time when an embedder is configured.
    pub embedding: Option<Vec<f32>>,
}

/// Everything known about one indexed file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileEntry {
    pub record: FileRecord,
    pub facts: FileSymbols,
    pub chunks: Vec<Chunk>,
    /// Backend that produced `facts`: tree-sitter, or regex (by choice or fallback).
    #[serde(default)]
    pub backend: Backend,
}

/// What one [`Index::update`] call did. Shows that only changed files were parsed.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct UpdateReport {
    /// Files found by the walk.
    pub scanned: usize,
    /// Files whose content hash changed (or are new), sorted. The only files parsed.
    pub reparsed: Vec<String>,
    /// Files that disappeared or became excluded, sorted.
    pub removed: Vec<String>,
    /// Files whose content hash matched the index and were not parsed.
    pub unchanged: usize,
    /// Paths excluded by the scan (secrets, binary, generated, ...).
    pub excluded: usize,
    /// Files whose graph edges were re-resolved.
    pub graph_recomputed: usize,
    /// Files read and hashed this update. Files skipped by the stat fast path are not counted.
    #[serde(default)]
    pub hashed: usize,
    /// Files whose size and mtime matched the index, so they were neither read nor hashed.
    #[serde(default)]
    pub stat_skipped: usize,
}

/// A recorded mtime is trusted only when the content was hashed at least this many seconds
/// after it. Mtimes have one-second resolution, so a later write in the same second could
/// otherwise keep the same size and mtime.
pub const RACY_WINDOW_SECS: u64 = 2;

#[derive(Serialize, Deserialize)]
struct StoredIndex {
    version: u32,
    root: String,
    files: Vec<FileEntry>,
    excluded: Vec<Exclusion>,
}

#[derive(Serialize)]
struct StoredIndexRef<'a> {
    version: u32,
    root: &'a str,
    files: Vec<&'a FileEntry>,
    excluded: &'a [Exclusion],
}

/// Indexed view of a workspace. Mutated only by [`Index::update`].
pub struct Index {
    root: PathBuf,
    store: Option<PathBuf>,
    cfg: ScanConfig,
    extractors: Vec<Box<dyn Extractor>>,
    embedder: Arc<dyn Embedder>,
    files: BTreeMap<String, FileEntry>,
    excluded: Vec<Exclusion>,
    graph: Graph,
    postings: Postings,
}

impl Index {
    /// Opens the index for `root`, loading `store` when it exists and matches this root and version.
    ///
    /// A store written by another schema version is discarded and rebuilt. A store that
    /// cannot be parsed is an error: it is never silently replaced.
    pub fn open(
        root: &Path,
        store: Option<PathBuf>,
        extractors: Vec<Box<dyn Extractor>>,
        embedder: Arc<dyn Embedder>,
    ) -> Result<Self> {
        let root_str = root.display().to_string();
        let mut files = BTreeMap::new();
        let mut excluded = Vec::new();
        if let Some(path) = store.as_ref().filter(|p| p.is_file()) {
            let bytes = std::fs::read(path)?;
            let stored: StoredIndex = serde_json::from_slice(&bytes).map_err(|e| {
                RaptorError::Integrity(format!(
                    "index store {} is unreadable ({e}); delete it to rebuild",
                    path.display()
                ))
            })?;
            if stored.version == INDEX_VERSION && stored.root == root_str {
                files = stored
                    .files
                    .into_iter()
                    .map(|f| (f.record.path.clone(), f))
                    .collect();
                excluded = stored.excluded;
            }
        }
        let mut graph = Graph::new();
        let upserts = files
            .values()
            .map(|f| (f.record.path.clone(), f.record.language, f.facts.clone()))
            .collect();
        graph.apply(upserts, &[]);
        let mut postings = Postings::default();
        for f in files.values() {
            postings.add_file(&f.record.path, &f.chunks);
        }
        Ok(Self {
            root: root.to_path_buf(),
            store,
            cfg: ScanConfig::default(),
            extractors,
            embedder,
            files,
            excluded,
            graph,
            postings,
        })
    }

    /// Re-scans the workspace. Parses only files whose content hash changed, removes
    /// files that disappeared, and re-resolves only the graph edges that could change.
    ///
    /// Uses the stat fast path: a file whose size and mtime match the index is not read.
    /// See [`Index::update_with`] for the trade-off and for a full rehash.
    pub fn update(&mut self) -> Result<UpdateReport> {
        self.update_with(true)
    }

    /// Re-scans the workspace. With `trust_stat`, a file whose size and mtime match its
    /// record is skipped without reading it. Without it, every file is read and hashed.
    ///
    /// The fast path cannot see a content change that keeps the size and the mtime, which
    /// happens only when the mtime is set back by hand or by a tool that preserves it. The
    /// records written within [`RACY_WINDOW_SECS`] of their mtime are never trusted.
    /// `trust_stat = false` is the way to catch those cases.
    pub fn update_with(&mut self, trust_stat: bool) -> Result<UpdateReport> {
        let mut seen: BTreeSet<String> = BTreeSet::new();
        let mut parsed: Vec<FileEntry> = Vec::new();
        // Path and the stat fields of a file whose content matched the index.
        let mut restat: Vec<(String, FileRecord)> = Vec::new();
        let mut unchanged = 0usize;
        let mut stat_skipped = 0usize;
        let excluded = {
            let files = &self.files;
            let extractors = &self.extractors;
            let embedder = self.embedder.as_ref();
            let known = |rel: &str| -> Option<(u64, u64)> {
                if !trust_stat {
                    return None;
                }
                files
                    .get(rel)
                    .filter(|f| {
                        f.record.hashed_at >= f.record.mtime.saturating_add(RACY_WINDOW_SECS)
                    })
                    .map(|f| (f.record.size, f.record.mtime))
            };
            scan_each_with(&self.root, &self.cfg, known, |event| match event {
                ScanEvent::Unchanged { path } => {
                    seen.insert(path);
                    unchanged += 1;
                    stat_skipped += 1;
                    Ok(())
                }
                ScanEvent::Read(sf) => {
                    seen.insert(sf.record.path.clone());
                    match files.get(&sf.record.path) {
                        Some(old) if old.record.hash == sf.record.hash => {
                            unchanged += 1;
                            restat.push((sf.record.path.clone(), sf.record));
                            Ok(())
                        }
                        _ => {
                            parsed.push(build_entry(sf, extractors, embedder)?);
                            Ok(())
                        }
                    }
                }
            })?
        };
        let hashed = seen.len() - stat_skipped;

        let removed: Vec<String> = self
            .files
            .keys()
            .filter(|p| !seen.contains(*p))
            .cloned()
            .collect();
        let mut reparsed: Vec<String> = parsed.iter().map(|e| e.record.path.clone()).collect();
        reparsed.sort();
        let upserts = parsed
            .iter()
            .map(|e| (e.record.path.clone(), e.record.language, e.facts.clone()))
            .collect();
        let graph_recomputed = self.graph.apply(upserts, &removed);

        for entry in parsed {
            if let Some(old) = self.files.get(&entry.record.path) {
                self.postings.remove_file(&entry.record.path, &old.chunks);
            }
            self.postings.add_file(&entry.record.path, &entry.chunks);
            self.files.insert(entry.record.path.clone(), entry);
        }
        for path in &removed {
            if let Some(old) = self.files.remove(path) {
                self.postings.remove_file(path, &old.chunks);
            }
        }
        for (path, fresh) in restat {
            if let Some(f) = self.files.get_mut(&path) {
                f.record.size = fresh.size;
                f.record.mtime = fresh.mtime;
                f.record.hashed_at = fresh.hashed_at;
            }
        }
        self.excluded = excluded;
        Ok(UpdateReport {
            scanned: seen.len(),
            reparsed,
            removed,
            unchanged,
            excluded: self.excluded.len(),
            graph_recomputed,
            hashed,
            stat_skipped,
        })
    }

    /// Writes the store atomically (temp file, then rename). No-op without a store path.
    pub fn save(&self) -> Result<()> {
        let Some(store) = &self.store else {
            return Ok(());
        };
        if let Some(parent) = store.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let root = self.root.display().to_string();
        let stored = StoredIndexRef {
            version: INDEX_VERSION,
            root: &root,
            files: self.files.values().collect(),
            excluded: &self.excluded,
        };
        let bytes = serde_json::to_vec(&stored)?;
        let tmp = store.with_extension("json.tmp");
        std::fs::write(&tmp, bytes)?;
        std::fs::rename(&tmp, store)?;
        Ok(())
    }

    /// Indexed files keyed by path.
    pub fn files(&self) -> &BTreeMap<String, FileEntry> {
        &self.files
    }

    /// One indexed file, if present.
    pub fn file(&self, path: &str) -> Option<&FileEntry> {
        self.files.get(path)
    }

    /// Current content hashes of indexed files.
    pub fn hashes(&self) -> BTreeMap<String, String> {
        self.files
            .iter()
            .map(|(p, f)| (p.clone(), f.record.hash.clone()))
            .collect()
    }

    /// Paths the scan excluded, with reasons.
    pub fn excluded(&self) -> &[Exclusion] {
        &self.excluded
    }

    /// The inverted index over chunk terms, used for BM25 candidate selection.
    pub fn postings(&self) -> &Postings {
        &self.postings
    }

    /// The dependency and symbol graph.
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Workspace root.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Total number of retrieval chunks.
    pub fn chunk_count(&self) -> usize {
        self.files.values().map(|f| f.chunks.len()).sum()
    }

    /// Total number of declared symbols.
    pub fn symbol_count(&self) -> usize {
        self.files.values().map(|f| f.facts.symbols.len()).sum()
    }

    /// Total number of import statements.
    pub fn import_count(&self) -> usize {
        self.files.values().map(|f| f.facts.imports.len()).sum()
    }

    /// The backend each configured language extractor uses for files it can parse.
    pub fn extractor_backends(&self) -> BTreeMap<Language, Backend> {
        self.extractors
            .iter()
            .map(|e| (e.language(), e.backend()))
            .collect()
    }
}

fn build_entry(
    sf: ScannedFile,
    extractors: &[Box<dyn Extractor>],
    embedder: &dyn Embedder,
) -> Result<FileEntry> {
    let (facts, backend) = match extractors
        .iter()
        .find(|e| e.language() == sf.record.language)
    {
        Some(ex) => ex.extract_tagged(&sf.record.path, &sf.text),
        None => (
            FileSymbols {
                is_test: is_test_path(&sf.record.path),
                ..FileSymbols::default()
            },
            Backend::Regex,
        ),
    };
    let chunks = build_chunks(&sf.record.path, &sf.text, &facts, embedder)?;
    Ok(FileEntry {
        record: sf.record,
        facts,
        chunks,
        backend,
    })
}

/// Splits a file into retrieval chunks: one per symbol body (long bodies windowed),
/// and fixed-size windows over any lines no symbol covers.
pub fn build_chunks(
    path: &str,
    text: &str,
    facts: &FileSymbols,
    embedder: &dyn Embedder,
) -> Result<Vec<Chunk>> {
    let lines: Vec<&str> = text.lines().collect();
    let n = lines.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut covered = vec![false; n];
    let mut ranges: Vec<(usize, usize, Option<String>)> = Vec::new();
    for sym in &facts.symbols {
        if !sym.kind.is_chunk_worthy() {
            continue;
        }
        let start = sym.start_line.max(1) as usize;
        if start > n {
            continue;
        }
        let end = (sym.end_line as usize).clamp(start, n);
        let mut cur = start;
        while cur <= end {
            let stop = (cur + MAX_SYMBOL_CHUNK_LINES - 1).min(end);
            ranges.push((cur, stop, Some(sym.name.clone())));
            cur = stop + 1;
        }
        for flag in covered.iter_mut().take(end).skip(start - 1) {
            *flag = true;
        }
    }
    let mut i = 0;
    while i < n {
        if covered[i] {
            i += 1;
            continue;
        }
        let begin = i;
        while i < n && !covered[i] && i - begin < WINDOW_LINES {
            i += 1;
        }
        ranges.push((begin + 1, i, None));
    }
    ranges.sort_by_key(|(s, e, _)| (*s, *e));

    let path_terms = terms(path);
    let mut chunks = Vec::with_capacity(ranges.len());
    for (start, end, symbol) in ranges {
        let body = lines[start - 1..end].join("\n");
        let mut tf = term_freq(&body);
        for t in &path_terms {
            *tf.entry(t.clone()).or_insert(0) += 1;
        }
        let len: u32 = tf.values().sum();
        let embedding = match embedder.dimensions() {
            Some(_) => Some(embedder.embed(&body)?),
            None => None,
        };
        chunks.push(Chunk {
            start_line: start as u32,
            end_line: end as u32,
            symbol,
            terms: tf,
            len,
            embedding,
        });
    }
    Ok(chunks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::{default_extractors, Symbol, SymbolKind};
    use crate::retrieve::NoEmbedder;

    fn open(root: &Path, store: Option<PathBuf>) -> Index {
        Index::open(
            root,
            store,
            default_extractors().expect("extractors"),
            Arc::new(NoEmbedder),
        )
        .expect("open")
    }

    fn write(root: &Path, rel: &str, body: &str) {
        let p = root.join(rel);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).expect("mkdir");
        }
        std::fs::write(p, body).expect("write");
    }

    #[test]
    fn chunks_cover_symbols_and_gaps() {
        let text = "use std::fmt;\n\nfn one() {\n    1\n}\n";
        let facts = FileSymbols {
            symbols: vec![Symbol {
                name: "one".into(),
                kind: SymbolKind::Function,
                start_line: 3,
                end_line: 5,
                parent: None,
            }],
            ..FileSymbols::default()
        };
        let chunks = build_chunks("src/x.rs", text, &facts, &NoEmbedder).expect("chunks");
        assert_eq!(chunks.len(), 2);
        assert_eq!((chunks[0].start_line, chunks[0].end_line), (1, 2));
        assert_eq!(chunks[0].symbol, None);
        assert_eq!(chunks[1].symbol.as_deref(), Some("one"));
    }

    #[test]
    fn chunk_terms_include_path_terms() {
        let facts = FileSymbols::default();
        let chunks = build_chunks("src/ledger/refund.rs", "let x = 1;", &facts, &NoEmbedder)
            .expect("chunks");
        assert!(chunks[0].terms.contains_key("refund"));
        assert!(chunks[0].terms.contains_key("ledger"));
    }

    #[test]
    fn long_symbol_bodies_are_windowed() {
        let body = (0..400)
            .map(|i| format!("line{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let facts = FileSymbols {
            symbols: vec![Symbol {
                name: "big".into(),
                kind: SymbolKind::Function,
                start_line: 1,
                end_line: 400,
                parent: None,
            }],
            ..FileSymbols::default()
        };
        let chunks = build_chunks("a.rs", &body, &facts, &NoEmbedder).expect("chunks");
        assert_eq!(chunks.len(), 3);
        assert!(chunks.iter().all(|c| c.symbol.as_deref() == Some("big")));
        assert_eq!(chunks.last().map(|c| c.end_line), Some(400));
    }

    #[test]
    fn empty_file_has_no_chunks() {
        let chunks =
            build_chunks("a.rs", "", &FileSymbols::default(), &NoEmbedder).expect("chunks");
        assert!(chunks.is_empty());
    }

    #[test]
    fn embedder_output_is_stored_per_chunk() {
        struct Fixed;
        impl Embedder for Fixed {
            fn dimensions(&self) -> Option<usize> {
                Some(2)
            }
            fn embed(&self, _text: &str) -> Result<Vec<f32>> {
                Ok(vec![1.0, 0.0])
            }
        }
        let chunks = build_chunks("a.rs", "x", &FileSymbols::default(), &Fixed).expect("chunks");
        assert_eq!(chunks[0].embedding.as_deref(), Some(&[1.0, 0.0][..]));
    }

    #[test]
    fn first_update_parses_everything_second_parses_nothing() {
        let t = tempfile::tempdir().expect("tempdir");
        write(t.path(), "src/a.rs", "pub fn a() {}\n");
        write(t.path(), "src/b.rs", "pub fn b() { a(); }\n");
        let mut idx = open(t.path(), None);
        let first = idx.update().expect("update");
        assert_eq!(first.reparsed, vec!["src/a.rs", "src/b.rs"]);
        let second = idx.update().expect("update");
        assert!(second.reparsed.is_empty());
        assert_eq!(second.unchanged, 2);
    }

    #[test]
    fn editing_one_file_reparses_only_that_file() {
        let t = tempfile::tempdir().expect("tempdir");
        write(t.path(), "src/a.rs", "pub fn a() {}\n");
        write(t.path(), "src/b.rs", "pub fn b() {}\n");
        write(t.path(), "src/c.rs", "pub fn c() {}\n");
        let mut idx = open(t.path(), None);
        idx.update().expect("update");
        write(t.path(), "src/b.rs", "pub fn b() { c(); }\n");
        let report = idx.update().expect("update");
        assert_eq!(report.reparsed, vec!["src/b.rs"]);
        assert_eq!(report.unchanged, 2);
        assert!(idx
            .graph()
            .callees("b")
            .iter()
            .any(|e| e.to.starts_with("src/c.rs")));
    }

    #[test]
    fn deleted_files_are_removed_from_index_and_graph() {
        let t = tempfile::tempdir().expect("tempdir");
        write(t.path(), "src/a.rs", "pub fn a() {}\n");
        write(t.path(), "src/b.rs", "pub fn b() { a(); }\n");
        let mut idx = open(t.path(), None);
        idx.update().expect("update");
        std::fs::remove_file(t.path().join("src/a.rs")).expect("rm");
        let report = idx.update().expect("update");
        assert_eq!(report.removed, vec!["src/a.rs"]);
        assert!(idx.file("src/a.rs").is_none());
        assert!(idx.graph().callees("b").is_empty());
    }

    #[test]
    fn newly_excluded_file_is_removed() {
        let t = tempfile::tempdir().expect("tempdir");
        write(t.path(), "config.rs", "pub fn c() {}\n");
        let mut idx = open(t.path(), None);
        idx.update().expect("update");
        std::fs::rename(t.path().join("config.rs"), t.path().join(".env")).expect("mv");
        let report = idx.update().expect("update");
        assert_eq!(report.removed, vec!["config.rs"]);
        assert!(idx.excluded().iter().any(|e| e.path == ".env"));
    }

    #[test]
    fn store_round_trips_and_skips_reparse() {
        let t = tempfile::tempdir().expect("tempdir");
        let store = t.path().join("store").join("index-v1.json");
        write(t.path(), "ws/src/a.rs", "pub fn a() {}\n");
        let root = t.path().join("ws");
        {
            let mut idx = open(&root, Some(store.clone()));
            idx.update().expect("update");
            idx.save().expect("save");
        }
        assert!(store.is_file());
        let mut reopened = open(&root, Some(store));
        let report = reopened.update().expect("update");
        assert!(report.reparsed.is_empty(), "loaded from store");
        assert_eq!(report.unchanged, 1);
    }

    #[test]
    fn corrupt_store_is_an_integrity_error() {
        let t = tempfile::tempdir().expect("tempdir");
        let store = t.path().join("index-v1.json");
        std::fs::write(&store, "{not json").expect("write");
        let res = Index::open(
            t.path(),
            Some(store),
            default_extractors().expect("ex"),
            Arc::new(NoEmbedder),
        );
        assert!(matches!(res, Err(RaptorError::Integrity(_))));
    }

    #[test]
    fn other_schema_version_is_rebuilt_not_loaded() {
        let t = tempfile::tempdir().expect("tempdir");
        let ws = t.path().join("ws");
        let store_dir = t.path().join("store");
        std::fs::create_dir_all(&store_dir).expect("mkdir");
        let store = store_dir.join("index-v1.json");
        let root = ws.to_string_lossy().into_owned();
        let body = serde_json::json!({ "version": 0, "root": root, "files": [], "excluded": [] });
        std::fs::write(&store, body.to_string()).expect("write");
        write(&ws, "a.rs", "fn a() {}\n");
        let mut idx = open(&ws, Some(store));
        assert_eq!(idx.update().expect("update").reparsed, vec!["a.rs"]);
    }

    #[test]
    fn hashes_and_counts_reflect_contents() {
        let t = tempfile::tempdir().expect("tempdir");
        write(t.path(), "a.rs", "fn a() { b(); }\nfn b() {}\n");
        let mut idx = open(t.path(), None);
        idx.update().expect("update");
        assert_eq!(idx.hashes().len(), 1);
        assert_eq!(idx.symbol_count(), 2);
        assert!(idx.chunk_count() >= 2);
    }

    fn set_mtime(root: &Path, rel: &str, secs: u64) {
        let f = std::fs::OpenOptions::new()
            .write(true)
            .open(root.join(rel))
            .expect("open");
        f.set_modified(std::time::UNIX_EPOCH + std::time::Duration::from_secs(secs))
            .expect("mtime");
    }

    #[test]
    fn stat_fast_path_skips_reading_unchanged_files() {
        let t = tempfile::tempdir().expect("tempdir");
        write(t.path(), "a.rs", "fn a() {}\n");
        write(t.path(), "b.rs", "fn b() {}\n");
        // Old mtimes: the hashes were taken long after them, so they are trusted.
        set_mtime(t.path(), "a.rs", 1_700_000_000);
        set_mtime(t.path(), "b.rs", 1_700_000_000);
        let mut idx = open(t.path(), None);
        let first = idx.update().expect("update");
        assert_eq!(first.hashed, 2);
        assert_eq!(first.stat_skipped, 0);
        let second = idx.update().expect("update");
        assert_eq!(second.hashed, 0, "nothing read");
        assert_eq!(second.stat_skipped, 2);
        assert_eq!(second.unchanged, 2);
        assert!(second.reparsed.is_empty());
    }

    #[test]
    fn touch_without_change_is_read_once_and_then_trusted_again() {
        let t = tempfile::tempdir().expect("tempdir");
        write(t.path(), "a.rs", "fn a() {}\n");
        write(t.path(), "b.rs", "fn b() {}\n");
        set_mtime(t.path(), "a.rs", 1_700_000_000);
        set_mtime(t.path(), "b.rs", 1_700_000_000);
        let mut idx = open(t.path(), None);
        idx.update().expect("update");
        set_mtime(t.path(), "a.rs", 1_700_000_500);
        let touched = idx.update().expect("update");
        assert_eq!(
            touched.hashed, 1,
            "the touched file is read to confirm content"
        );
        assert_eq!(touched.stat_skipped, 1);
        assert!(touched.reparsed.is_empty(), "same content is not reparsed");
        assert_eq!(touched.unchanged, 2);
        let again = idx.update().expect("update");
        assert_eq!(again.hashed, 0, "the new mtime is recorded and trusted");
    }

    #[test]
    fn recently_written_files_are_never_trusted_by_stat() {
        // Written now, so the mtime is within the racy window of the hash taken right after.
        let t = tempfile::tempdir().expect("tempdir");
        write(t.path(), "a.rs", "fn a() {}\n");
        let mut idx = open(t.path(), None);
        idx.update().expect("update");
        let again = idx.update().expect("update");
        assert_eq!(again.hashed, 1, "racy record is re-read");
        assert!(again.reparsed.is_empty());
    }

    #[test]
    fn size_change_is_detected_even_when_the_mtime_is_restored() {
        let t = tempfile::tempdir().expect("tempdir");
        write(t.path(), "a.rs", "fn a() {}\n");
        set_mtime(t.path(), "a.rs", 1_700_000_000);
        let mut idx = open(t.path(), None);
        idx.update().expect("update");
        write(t.path(), "a.rs", "fn a() { b(); }\n");
        set_mtime(t.path(), "a.rs", 1_700_000_000);
        let report = idx.update().expect("update");
        assert_eq!(report.reparsed, vec!["a.rs"]);
    }

    /// Documented edge: a content change that keeps both the size and the mtime is invisible
    /// to the stat fast path. Only a full rehash catches it.
    #[test]
    fn same_size_and_mtime_edit_is_missed_by_fast_path_and_caught_by_full_rehash() {
        let t = tempfile::tempdir().expect("tempdir");
        write(t.path(), "a.rs", "fn a() {}\n");
        set_mtime(t.path(), "a.rs", 1_700_000_000);
        let mut idx = open(t.path(), None);
        idx.update().expect("update");
        write(t.path(), "a.rs", "fn b() {}\n");
        set_mtime(t.path(), "a.rs", 1_700_000_000);
        let fast = idx.update().expect("update");
        assert!(fast.reparsed.is_empty(), "fast path trusts size and mtime");
        let full = idx.update_with(false).expect("full update");
        assert_eq!(full.reparsed, vec!["a.rs"], "full rehash sees the edit");
        assert!(idx.graph().callees("b").is_empty());
        assert!(idx
            .file("a.rs")
            .is_some_and(|f| f.facts.symbols.iter().any(|s| s.name == "b")));
    }
}
