//! Persisted vector index for semantic retrieval.
//!
//! Vectors are keyed by the SHA-256 of the chunk text, so identical chunks share one vector
//! and a re-scan embeds only chunks whose text is new. Each indexed file keeps the list of its
//! chunk keys together with the file hash they were computed from. A file whose hash changes
//! is re-keyed, and vectors that no file references are pruned.
//!
//! The store is one binary file, `vectors-v{VECTOR_VERSION}.bin`, with a magic number, the
//! model id and dimension it was built with, and a trailing SHA-256 over the body. A store for
//! another model or dimension is discarded and rebuilt. A store that fails its checksum is
//! discarded with a notice. Neither case fails retrieval.

use crate::text::terms;
use raptor_core::{RaptorError, Result};
use sha2::{Digest as _, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

/// Version of the on-disk vector store format.
pub const VECTOR_VERSION: u32 = 1;
/// Default cap on stored vectors. Memory is about `vectors * dimension * 4` bytes.
pub const DEFAULT_MAX_VECTORS: usize = 100_000;
/// Chunk texts sent to the backend in one request.
pub const EMBED_BATCH: usize = 32;
/// Chunk text longer than this many characters is truncated before embedding.
pub const MAX_EMBED_CHARS: usize = 2000;

const MAGIC: &[u8; 4] = b"RVEC";

/// SHA-256 of a chunk's text.
pub type Digest = [u8; 32];

/// Key recorded for a blank chunk. It has nothing to embed, so it never has a vector, but it is
/// not "missing" either: counting it as missing put a false warning in every context manifest.
pub const BLANK_KEY: Digest = [0u8; 32];

/// Turns text into vectors. Implemented by the shell over llama-server `/v1/embeddings`,
/// or by [`HashingEmbedder`] and fakes in tests. Synchronous by design.
pub trait EmbeddingBackend: Send + Sync {
    /// Identifies the model. A change of model id discards the stored vectors.
    fn model_id(&self) -> &str;
    /// Vector width. Must be greater than zero and constant for the backend's lifetime.
    fn dimension(&self) -> usize;
    /// Embeds each text, returning one vector per input, in order.
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
}

/// A deterministic embedder for tests and offline runs. It hashes each term into a fixed
/// number of signed buckets and normalises the result. It matches shared words only, so it is
/// not a semantic model.
#[derive(Debug, Clone)]
pub struct HashingEmbedder {
    dim: usize,
    model: String,
}

impl HashingEmbedder {
    /// An embedder with `dim` buckets. `dim` must be at least 8.
    pub fn new(dim: usize) -> Result<Self> {
        if dim < 8 {
            return Err(RaptorError::InvalidInput(
                "hashing embedder needs at least 8 dimensions".into(),
            ));
        }
        Ok(Self {
            dim,
            model: format!("raptor-hashing-test-{dim}"),
        })
    }

    fn embed_one(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0f32; self.dim];
        for term in terms(text) {
            let h = Sha256::digest(term.as_bytes());
            let bucket =
                u64::from_le_bytes(h[0..8].try_into().unwrap_or([0; 8])) as usize % self.dim;
            let sign = if h[8] & 1 == 0 { 1.0 } else { -1.0 };
            v[bucket] += sign;
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }
}

impl EmbeddingBackend for HashingEmbedder {
    fn model_id(&self) -> &str {
        &self.model
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| self.embed_one(t)).collect())
    }
}

/// Per-file record: the file hash the keys were computed from, and one key per chunk in
/// chunk order. `None` means that chunk has no vector (capacity reached or embedding failed).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredFile {
    pub hash: String,
    pub keys: Vec<Option<Digest>>,
}

/// What one [`VectorIndex::sync_file`] call did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SyncStats {
    /// Vectors computed by the backend.
    pub embedded: usize,
    /// Chunks whose text already had a vector.
    pub reused: usize,
    /// Chunks left without a vector because the store is at capacity.
    pub capped: usize,
    /// Chunks left without a vector because the backend failed.
    pub failed: usize,
    /// The backend error, when one occurred. Retrieval continues without these vectors.
    pub error: Option<String>,
}

/// Persisted, per-file-invalidated vector store for one model.
#[derive(Debug)]
pub struct VectorIndex {
    model: String,
    dim: usize,
    max_vectors: usize,
    vectors: BTreeMap<Digest, Vec<f32>>,
    files: BTreeMap<String, StoredFile>,
    /// Files whose last sync failed. They are retried on the next update, not persisted.
    retry: BTreeSet<String>,
    dirty: bool,
    notice: Option<String>,
}

impl VectorIndex {
    /// An empty store for `model` with `dim`-wide vectors and at most `max_vectors` of them.
    pub fn empty(model: &str, dim: usize, max_vectors: usize) -> Self {
        Self {
            model: model.to_string(),
            dim,
            max_vectors,
            vectors: BTreeMap::new(),
            files: BTreeMap::new(),
            retry: BTreeSet::new(),
            dirty: false,
            notice: None,
        }
    }

    /// Loads the store at `path` if it matches `model`, `dim` and the format version.
    ///
    /// Never fails. A store for another model, or one that is corrupt, is discarded and
    /// `notice` says why. The caller then re-embeds the files.
    pub fn open(path: &Path, model: &str, dim: usize, max_vectors: usize) -> Self {
        let mut store = Self::empty(model, dim, max_vectors);
        if !path.is_file() {
            return store;
        }
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(e) => {
                store.notice = Some(format!("vector store unreadable, rebuilding: {e}"));
                return store;
            }
        };
        match decode(&bytes) {
            Ok((stored_model, stored_dim, vectors, files)) => {
                if stored_model != model || stored_dim != dim {
                    store.notice = Some(format!(
                        "vector store was built with model {stored_model} ({stored_dim}-d); rebuilding for {model} ({dim}-d)"
                    ));
                    return store;
                }
                store.vectors = vectors;
                store.files = files;
                // The retry set is not saved. A file with a chunk that has no vector (a failed or
                // partial sync before a restart) must be tried again, or it would look unchanged
                // and never be embedded. Files that only have blank chunks clear themselves on
                // the first sync, which sends nothing.
                store.retry = store
                    .files
                    .iter()
                    .filter(|(_, f)| f.keys.iter().any(Option::is_none))
                    .map(|(p, _)| p.clone())
                    .collect();
                store
            }
            Err(e) => {
                store.notice = Some(format!("vector store discarded, rebuilding: {e}"));
                store
            }
        }
    }

    /// Model id the vectors belong to.
    pub fn model_id(&self) -> &str {
        &self.model
    }

    /// Vector width.
    pub fn dimension(&self) -> usize {
        self.dim
    }

    /// Capacity in vectors.
    pub fn capacity(&self) -> usize {
        self.max_vectors
    }

    /// Stored vectors.
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// True when no vectors are stored.
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Why the store was discarded at open, if it was.
    pub fn notice(&self) -> Option<&str> {
        self.notice.as_deref()
    }

    /// The stored record for `path`.
    pub fn file(&self, path: &str) -> Option<&StoredFile> {
        self.files.get(path)
    }

    /// The vector for a chunk key.
    pub fn get(&self, key: &Digest) -> Option<&[f32]> {
        self.vectors.get(key).map(Vec::as_slice)
    }

    /// True when `path` must be synced: its hash or chunk count differs, it has no record,
    /// or its last sync failed.
    pub fn needs_file(&self, path: &str, hash: &str, chunk_count: usize) -> bool {
        if self.retry.contains(path) {
            return true;
        }
        match self.files.get(path) {
            Some(f) => f.hash != hash || f.keys.len() != chunk_count,
            None => true,
        }
    }

    /// Chunks in `files` that have no usable vector. `files` yields `(path, hash, chunk_count)`.
    pub fn missing_chunks<'a>(
        &self,
        files: impl Iterator<Item = (&'a str, &'a str, usize)>,
    ) -> usize {
        files
            .map(|(path, hash, n)| match self.files.get(path) {
                Some(f) if f.hash == hash && f.keys.len() == n => {
                    n - f.keys.iter().filter(|k| k.is_some()).count()
                }
                _ => n,
            })
            .sum()
    }

    /// Forgets `path`. Its vectors stay until [`VectorIndex::prune`] finds them unreferenced.
    pub fn remove_file(&mut self, path: &str) {
        self.retry.remove(path);
        if self.files.remove(path).is_some() {
            self.dirty = true;
        }
    }

    /// Embeds the chunks of one file that have no vector yet, and records its keys.
    ///
    /// `texts` are the chunk texts in chunk order. Chunks with a known text reuse their
    /// vector. The rest are sent to `backend` in batches of [`EMBED_BATCH`], and the store
    /// stops at its capacity. A backend failure is reported in the returned stats and does
    /// not return an error: the chunks stay unembedded and the file is retried later.
    pub fn sync_file(
        &mut self,
        backend: &dyn EmbeddingBackend,
        path: &str,
        hash: &str,
        texts: &[String],
    ) -> SyncStats {
        let mut stats = SyncStats::default();
        if backend.dimension() != self.dim || backend.model_id() != self.model {
            stats.failed = texts.len();
            stats.error = Some("backend does not match the vector store's model".into());
            self.retry.insert(path.to_string());
            self.files.insert(
                path.to_string(),
                StoredFile {
                    hash: hash.to_string(),
                    keys: vec![None; texts.len()],
                },
            );
            return stats;
        }
        let digests: Vec<Digest> = texts
            .iter()
            .map(|t| Sha256::digest(t.as_bytes()).into())
            .collect();
        let mut pending: Vec<(Digest, &str)> = Vec::new();
        let mut queued: BTreeSet<Digest> = BTreeSet::new();
        for (text, d) in texts.iter().zip(&digests) {
            // A blank chunk has nothing to embed, and llama-server answers HTTP 400 ("Input
            // content cannot be empty") for the whole batch, so it is never sent.
            if text.trim().is_empty() {
                continue;
            }
            if !self.vectors.contains_key(d) && queued.insert(*d) {
                pending.push((*d, text.as_str()));
            }
        }
        let mut capped_from: Option<usize> = None;
        let mut next = 0usize;
        while next < pending.len() {
            let room = self.max_vectors.saturating_sub(self.vectors.len());
            if room == 0 {
                capped_from = Some(next);
                break;
            }
            let take = room.min(EMBED_BATCH).min(pending.len() - next);
            let batch = &pending[next..next + take];
            let inputs: Vec<&str> = batch
                .iter()
                .map(|(_, t)| truncate_chars(t, MAX_EMBED_CHARS))
                .collect();
            let outcome = backend.embed_batch(&inputs).and_then(|vs| {
                if vs.len() != inputs.len() || vs.iter().any(|v| v.len() != self.dim) {
                    Err(RaptorError::Runtime(format!(
                        "backend returned {} vectors for {} texts, expected width {}",
                        vs.len(),
                        inputs.len(),
                        self.dim
                    )))
                } else {
                    Ok(vs)
                }
            });
            match outcome {
                Ok(vs) => {
                    for ((d, _), v) in batch.iter().zip(vs) {
                        self.vectors.insert(*d, v);
                    }
                    stats.embedded += take;
                    self.dirty = true;
                    next += take;
                }
                Err(e) => {
                    stats.error = Some(e.to_string());
                    break;
                }
            }
        }
        let keys: Vec<Option<Digest>> = digests
            .iter()
            .zip(texts)
            .map(|(d, text)| {
                if text.trim().is_empty() {
                    Some(BLANK_KEY)
                } else {
                    self.vectors.contains_key(d).then_some(*d)
                }
            })
            .collect();
        // Per chunk, not per unique text: a repeated text is one vector but several chunks.
        for ((key, d), text) in keys.iter().zip(&digests).zip(texts) {
            if text.trim().is_empty() {
                // Nothing to embed: neither a failure nor a reason to retry the file.
            } else if key.is_some() {
                if !queued.contains(d) {
                    stats.reused += 1;
                }
            } else if capped_from.is_some() {
                stats.capped += 1;
            } else {
                stats.failed += 1;
            }
        }
        if stats.error.is_some() || stats.failed > 0 {
            self.retry.insert(path.to_string());
        } else {
            self.retry.remove(path);
        }
        let unchanged = self
            .files
            .get(path)
            .is_some_and(|f| f.hash == hash && f.keys == keys);
        if !unchanged {
            self.dirty = true;
        }
        self.files.insert(
            path.to_string(),
            StoredFile {
                hash: hash.to_string(),
                keys,
            },
        );
        stats
    }

    /// Drops vectors that no file references. Returns how many were dropped.
    pub fn prune(&mut self) -> usize {
        let referenced: BTreeSet<Digest> = self
            .files
            .values()
            .flat_map(|f| f.keys.iter().flatten().copied())
            .collect();
        let before = self.vectors.len();
        self.vectors.retain(|k, _| referenced.contains(k));
        let dropped = before - self.vectors.len();
        if dropped > 0 {
            self.dirty = true;
        }
        dropped
    }

    /// Writes the store atomically (temp file, then rename). Does nothing when unchanged.
    pub fn save(&mut self, path: &Path) -> Result<()> {
        if !self.dirty && path.is_file() {
            return Ok(());
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let bytes = encode(&self.model, self.dim, &self.vectors, &self.files);
        let tmp = path.with_extension("bin.tmp");
        std::fs::write(&tmp, bytes)?;
        std::fs::rename(&tmp, path)?;
        self.dirty = false;
        Ok(())
    }
}

fn truncate_chars(text: &str, max: usize) -> &str {
    match text.char_indices().nth(max) {
        Some((i, _)) => &text[..i],
        None => text,
    }
}

fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn encode(
    model: &str,
    dim: usize,
    vectors: &BTreeMap<Digest, Vec<f32>>,
    files: &BTreeMap<String, StoredFile>,
) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    put_u32(&mut out, VECTOR_VERSION);
    put_u32(&mut out, dim as u32);
    put_u32(&mut out, model.len() as u32);
    out.extend_from_slice(model.as_bytes());
    put_u32(&mut out, vectors.len() as u32);
    for (key, v) in vectors {
        out.extend_from_slice(key);
        for x in v {
            out.extend_from_slice(&x.to_le_bytes());
        }
    }
    put_u32(&mut out, files.len() as u32);
    for (path, f) in files {
        put_u32(&mut out, path.len() as u32);
        out.extend_from_slice(path.as_bytes());
        put_u32(&mut out, f.hash.len() as u32);
        out.extend_from_slice(f.hash.as_bytes());
        put_u32(&mut out, f.keys.len() as u32);
        for k in &f.keys {
            match k {
                Some(d) => {
                    out.push(1);
                    out.extend_from_slice(d);
                }
                None => out.push(0),
            }
        }
    }
    let checksum: [u8; 32] = Sha256::digest(&out).into();
    out.extend_from_slice(&checksum);
    out
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|e| *e <= self.buf.len())
            .ok_or_else(|| RaptorError::Integrity("truncated vector store".into()))?;
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn digest(&mut self) -> Result<Digest> {
        let mut d = [0u8; 32];
        d.copy_from_slice(self.take(32)?);
        Ok(d)
    }

    fn string(&mut self) -> Result<String> {
        let n = self.u32()? as usize;
        String::from_utf8(self.take(n)?.to_vec())
            .map_err(|_| RaptorError::Integrity("vector store has invalid UTF-8".into()))
    }
}

type Decoded = (
    String,
    usize,
    BTreeMap<Digest, Vec<f32>>,
    BTreeMap<String, StoredFile>,
);

fn decode(bytes: &[u8]) -> Result<Decoded> {
    if bytes.len() < 32 + 4 + 4 {
        return Err(RaptorError::Integrity("vector store is too short".into()));
    }
    let (body, sum) = bytes.split_at(bytes.len() - 32);
    let expect: [u8; 32] = Sha256::digest(body).into();
    if expect[..] != sum[..] {
        return Err(RaptorError::Integrity(
            "vector store checksum mismatch".into(),
        ));
    }
    let mut r = Reader { buf: body, pos: 0 };
    if r.take(4)? != MAGIC {
        return Err(RaptorError::Integrity("not a vector store".into()));
    }
    let version = r.u32()?;
    if version != VECTOR_VERSION {
        return Err(RaptorError::Integrity(format!(
            "vector store version {version} is not {VECTOR_VERSION}"
        )));
    }
    let dim = r.u32()? as usize;
    let model = r.string()?;
    let n = r.u32()? as usize;
    let mut vectors = BTreeMap::new();
    for _ in 0..n {
        let key = r.digest()?;
        let raw = r.take(dim * 4)?;
        let v: Vec<f32> = (0..dim)
            .map(|i| {
                f32::from_le_bytes([raw[4 * i], raw[4 * i + 1], raw[4 * i + 2], raw[4 * i + 3]])
            })
            .collect();
        vectors.insert(key, v);
    }
    let nf = r.u32()? as usize;
    let mut files = BTreeMap::new();
    for _ in 0..nf {
        let path = r.string()?;
        let hash = r.string()?;
        let nk = r.u32()? as usize;
        let mut keys = Vec::with_capacity(nk.min(1 << 20));
        for _ in 0..nk {
            let flag = r.take(1)?[0];
            keys.push(match flag {
                0 => None,
                1 => Some(r.digest()?),
                _ => return Err(RaptorError::Integrity("bad key flag".into())),
            });
        }
        files.insert(path, StoredFile { hash, keys });
    }
    if r.pos != body.len() {
        return Err(RaptorError::Integrity(
            "trailing bytes in vector store".into(),
        ));
    }
    Ok((model, dim, vectors, files))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    /// Counts calls and returns vectors whose value is the text length.
    struct Counting {
        calls: AtomicUsize,
        dim: usize,
        fail: AtomicBool,
    }

    impl Counting {
        fn new(dim: usize) -> Self {
            Self {
                calls: AtomicUsize::new(0),
                dim,
                fail: AtomicBool::new(false),
            }
        }
        fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
        fn set_fail(&self, on: bool) {
            self.fail.store(on, Ordering::SeqCst);
        }
    }

    impl EmbeddingBackend for Counting {
        fn model_id(&self) -> &str {
            "counting-1"
        }
        fn dimension(&self) -> usize {
            self.dim
        }
        fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if self.fail.load(Ordering::SeqCst) {
                return Err(RaptorError::Network("connection refused".into()));
            }
            Ok(texts
                .iter()
                .map(|t| vec![t.len() as f32; self.dim])
                .collect())
        }
    }

    fn texts(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn hashing_embedder_is_deterministic_and_normalised() {
        let e = HashingEmbedder::new(64).expect("embedder");
        let a = e.embed_batch(&["refund the order"]).expect("embed");
        let b = e.embed_batch(&["refund the order"]).expect("embed");
        assert_eq!(a, b);
        let norm: f32 = a[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
        assert!(HashingEmbedder::new(4).is_err());
    }

    #[test]
    fn sync_embeds_once_and_reuses_identical_text() {
        let b = Counting::new(4);
        let mut s = VectorIndex::empty("counting-1", 4, 100);
        let first = s.sync_file(&b, "a.rs", "h1", &texts(&["x", "y"]));
        assert_eq!(first.embedded, 2);
        assert_eq!(b.calls(), 1);
        // Another file with the same chunk text reuses the vector.
        let second = s.sync_file(&b, "b.rs", "h2", &texts(&["x"]));
        assert_eq!(second.embedded, 0);
        assert_eq!(second.reused, 1);
        assert_eq!(b.calls(), 1);
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn changed_file_is_rekeyed_and_orphans_are_pruned() {
        let b = Counting::new(4);
        let mut s = VectorIndex::empty("counting-1", 4, 100);
        s.sync_file(&b, "a.rs", "h1", &texts(&["old text"]));
        assert!(!s.needs_file("a.rs", "h1", 1));
        assert!(s.needs_file("a.rs", "h2", 1));
        s.sync_file(&b, "a.rs", "h2", &texts(&["new text"]));
        assert_eq!(s.prune(), 1);
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn a_partly_embedded_file_is_retried_after_a_restart() {
        struct Down;
        impl EmbeddingBackend for Down {
            fn model_id(&self) -> &str {
                "m"
            }
            fn dimension(&self) -> usize {
                2
            }
            fn embed_batch(&self, _: &[&str]) -> Result<Vec<Vec<f32>>> {
                Err(RaptorError::Runtime("server down".into()))
            }
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.bin");
        let mut store = VectorIndex::empty("m", 2, 100);
        let stats = store.sync_file(&Down, "a.rs", "h1", &["fn a() {}".to_string()]);
        assert!(stats.error.is_some());
        store.save(&path).unwrap();

        let reopened = VectorIndex::open(&path, "m", 2, 100);
        assert!(
            reopened.needs_file("a.rs", "h1", 1),
            "unchanged hash, but a chunk has no vector: it must be synced again"
        );
    }

    #[test]
    fn blank_chunks_are_never_sent_and_do_not_fail_the_file() {
        use std::sync::Mutex;
        struct Spy(Mutex<Vec<String>>);
        impl EmbeddingBackend for Spy {
            fn model_id(&self) -> &str {
                "m"
            }
            fn dimension(&self) -> usize {
                2
            }
            fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
                self.0
                    .lock()
                    .unwrap()
                    .extend(texts.iter().map(|t| t.to_string()));
                Ok(texts.iter().map(|_| vec![1.0, 0.0]).collect())
            }
        }
        let spy = Spy(Mutex::new(Vec::new()));
        let mut store = VectorIndex::empty("m", 2, 100);
        let texts = vec![
            "fn a() {}".to_string(),
            "  \n\t".to_string(),
            String::new(),
            "fn b() {}".to_string(),
        ];
        let stats = store.sync_file(&spy, "a.rs", "h1", &texts);
        assert_eq!(stats.embedded, 2);
        assert_eq!(stats.failed, 0, "blank chunks are not failures");
        assert!(stats.error.is_none());
        assert_eq!(
            spy.0.lock().unwrap().len(),
            2,
            "only real text reached the backend"
        );
        assert!(!store.retry.contains("a.rs"));
        let file = store.file("a.rs").unwrap();
        assert_eq!(file.keys[1], Some(BLANK_KEY));
        assert_eq!(
            store.missing_chunks(std::iter::once(("a.rs", "h1", 4))),
            0,
            "blank chunks are not reported as missing"
        );
    }

    #[test]
    fn batches_are_bounded() {
        let b = Counting::new(4);
        let mut s = VectorIndex::empty("counting-1", 4, 1000);
        let many: Vec<String> = (0..EMBED_BATCH * 2 + 5)
            .map(|i| format!("chunk {i}"))
            .collect();
        let stats = s.sync_file(&b, "big.rs", "h", &many);
        assert_eq!(stats.embedded, many.len());
        assert_eq!(b.calls(), 3);
    }

    #[test]
    fn capacity_leaves_chunks_unembedded_and_reports_it() {
        let b = Counting::new(4);
        let mut s = VectorIndex::empty("counting-1", 4, 2);
        let stats = s.sync_file(&b, "a.rs", "h", &texts(&["one", "two", "three"]));
        assert_eq!(stats.embedded, 2);
        assert_eq!(stats.capped, 1);
        assert_eq!(stats.failed, 0);
        assert_eq!(s.len(), 2);
        assert_eq!(s.missing_chunks(std::iter::once(("a.rs", "h", 3))), 1);
    }

    #[test]
    fn backend_failure_is_reported_not_raised_and_retried() {
        let b = Counting::new(4);
        b.set_fail(true);
        let mut s = VectorIndex::empty("counting-1", 4, 100);
        let stats = s.sync_file(&b, "a.rs", "h", &texts(&["x"]));
        assert!(stats.error.as_deref().unwrap_or("").contains("connection"));
        assert_eq!(stats.failed, 1);
        assert_eq!(s.len(), 0);
        assert!(s.needs_file("a.rs", "h", 1), "failed file is retried");
        b.set_fail(false);
        let retry = s.sync_file(&b, "a.rs", "h", &texts(&["x"]));
        assert_eq!(retry.embedded, 1);
        assert!(!s.needs_file("a.rs", "h", 1));
    }

    #[test]
    fn wrong_width_from_backend_is_a_failure() {
        struct Wrong;
        impl EmbeddingBackend for Wrong {
            fn model_id(&self) -> &str {
                "wrong"
            }
            fn dimension(&self) -> usize {
                4
            }
            fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
                Ok(texts.iter().map(|_| vec![1.0; 3]).collect())
            }
        }
        let mut s = VectorIndex::empty("wrong", 4, 100);
        let stats = s.sync_file(&Wrong, "a.rs", "h", &texts(&["x"]));
        assert!(stats.error.is_some());
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn store_round_trips_and_is_checked_on_load() {
        let t = tempfile::tempdir().expect("tempdir");
        let path = t.path().join("vectors-v1.bin");
        let b = Counting::new(4);
        let mut s = VectorIndex::empty("counting-1", 4, 100);
        s.sync_file(&b, "a.rs", "h1", &texts(&["alpha", "beta"]));
        s.save(&path).expect("save");
        let loaded = VectorIndex::open(&path, "counting-1", 4, 100);
        assert!(loaded.notice().is_none());
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.file("a.rs"), s.file("a.rs"));
    }

    #[test]
    fn other_model_or_dimension_discards_the_store() {
        let t = tempfile::tempdir().expect("tempdir");
        let path = t.path().join("vectors-v1.bin");
        let b = Counting::new(4);
        let mut s = VectorIndex::empty("counting-1", 4, 100);
        s.sync_file(&b, "a.rs", "h1", &texts(&["alpha"]));
        s.save(&path).expect("save");
        let other_model = VectorIndex::open(&path, "other-model", 4, 100);
        assert!(other_model.is_empty());
        assert!(other_model.notice().unwrap_or("").contains("rebuilding"));
        let other_dim = VectorIndex::open(&path, "counting-1", 8, 100);
        assert!(other_dim.is_empty());
    }

    #[test]
    fn corrupt_store_is_discarded_with_a_notice() {
        let t = tempfile::tempdir().expect("tempdir");
        let path = t.path().join("vectors-v1.bin");
        let b = Counting::new(4);
        let mut s = VectorIndex::empty("counting-1", 4, 100);
        s.sync_file(&b, "a.rs", "h1", &texts(&["alpha"]));
        s.save(&path).expect("save");
        let mut bytes = std::fs::read(&path).expect("read");
        bytes[10] ^= 0xff;
        std::fs::write(&path, bytes).expect("write");
        let loaded = VectorIndex::open(&path, "counting-1", 4, 100);
        assert!(loaded.is_empty());
        assert!(loaded.notice().unwrap_or("").contains("checksum"));
    }

    #[test]
    fn truncated_store_is_discarded_not_panicking() {
        let t = tempfile::tempdir().expect("tempdir");
        let path = t.path().join("vectors-v1.bin");
        std::fs::write(&path, b"RVEC").expect("write");
        let loaded = VectorIndex::open(&path, "m", 4, 10);
        assert!(loaded.is_empty());
        assert!(loaded.notice().is_some());
    }

    #[test]
    fn save_is_skipped_when_nothing_changed() {
        let t = tempfile::tempdir().expect("tempdir");
        let path = t.path().join("v.bin");
        let b = Counting::new(4);
        let mut s = VectorIndex::empty("counting-1", 4, 100);
        s.sync_file(&b, "a.rs", "h1", &texts(&["alpha"]));
        s.save(&path).expect("save");
        let before = std::fs::metadata(&path).expect("meta").modified().ok();
        s.sync_file(&b, "a.rs", "h1", &texts(&["alpha"]));
        s.save(&path).expect("save");
        let after = std::fs::metadata(&path).expect("meta").modified().ok();
        assert_eq!(before, after);
    }
}
