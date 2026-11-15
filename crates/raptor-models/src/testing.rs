//! Test-only fakes and byte-level fixture builders. No network, no real disk-space probe.

use crate::download::{DiskSpace, HttpFetch, Provenance, Registrar, RemoteResponse, StagedShard};
use crate::hf::{HfApi, HfModelSummary, HfRepoInfo, RepoFile, RepoSnapshot, SearchSort};
use crate::library::ModelSource;
use async_trait::async_trait;
use futures_util::stream::{self, BoxStream, StreamExt};
use raptor_core::{RaptorError, Result};
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Builds GGUF v3 headers byte by byte, so tests can craft valid and hostile inputs.
#[derive(Default)]
pub struct GgufWriter {
    kvs: Vec<u8>,
    count: u64,
}

impl GgufWriter {
    pub fn new() -> Self {
        Self::default()
    }

    fn push_str(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    pub fn kv_str(&mut self, key: &str, value: &str) {
        let mut v = Vec::new();
        Self::push_str(&mut v, value);
        self.kv_raw(key, 8, &v);
    }

    pub fn kv_u32(&mut self, key: &str, value: u32) {
        self.kv_raw(key, 4, &value.to_le_bytes());
    }

    pub fn kv_array_str(&mut self, key: &str, values: &[&str]) {
        let mut v = Vec::new();
        v.extend_from_slice(&8u32.to_le_bytes());
        v.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for s in values {
            Self::push_str(&mut v, s);
        }
        self.kv_raw(key, 9, &v);
    }

    /// Writes a key with an arbitrary type tag and raw value bytes.
    pub fn kv_raw(&mut self, key: &str, ty: u32, value: &[u8]) {
        Self::push_str(&mut self.kvs, key);
        self.kvs.extend_from_slice(&ty.to_le_bytes());
        self.kvs.extend_from_slice(value);
        self.count += 1;
    }

    /// Writes only an array header (type 9) with a declared element type and count.
    pub fn kv_raw_array_header(&mut self, key: &str, elem_ty: u32, count: u64) {
        let mut v = Vec::new();
        v.extend_from_slice(&elem_ty.to_le_bytes());
        v.extend_from_slice(&count.to_le_bytes());
        self.kv_raw(key, 9, &v);
    }

    /// Writes an array nested `depth` levels deep (each level holds one array).
    pub fn kv_nested_array_depth(&mut self, key: &str, depth: u32) {
        let mut v = Vec::new();
        for _ in 0..depth {
            v.extend_from_slice(&9u32.to_le_bytes());
            v.extend_from_slice(&1u64.to_le_bytes());
        }
        v.extend_from_slice(&4u32.to_le_bytes());
        v.extend_from_slice(&0u64.to_le_bytes());
        self.kv_raw(key, 9, &v);
    }

    /// Appends raw bytes that are not a key/value entry.
    pub fn raw(&mut self, bytes: &[u8]) {
        self.kvs.extend_from_slice(bytes);
    }

    /// Full file bytes: header plus key/values, then 4 KiB of fake tensor data.
    pub fn finish(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        out.extend_from_slice(&self.count.to_le_bytes());
        out.extend_from_slice(&self.kvs);
        out.extend_from_slice(&vec![0u8; 4096]);
        out
    }
}

/// A small but valid GGUF file for `arch`, with the given quant `file_type`.
pub fn gguf_file(arch: &str, file_type: u32) -> Vec<u8> {
    let mut w = GgufWriter::new();
    w.kv_str("general.architecture", arch);
    w.kv_str("general.name", "Fixture");
    w.kv_u32("general.file_type", file_type);
    w.kv_u32(&format!("{arch}.context_length"), 4096);
    w.kv_u32(&format!("{arch}.block_count"), 4);
    w.kv_u32(&format!("{arch}.embedding_length"), 256);
    w.kv_u32(&format!("{arch}.attention.head_count"), 4);
    w.kv_u32(&format!("{arch}.attention.head_count_kv"), 2);
    w.kv_str("tokenizer.ggml.model", "gpt2");
    w.kv_str("tokenizer.chat_template", "{{ messages }}");
    w.finish()
}

/// One shard of a split GGUF: the same metadata plus `split.no` (0-based) and `split.count`.
pub fn gguf_shard(arch: &str, split_no: u32, split_count: u32) -> Vec<u8> {
    let mut w = GgufWriter::new();
    w.kv_str("general.architecture", arch);
    w.kv_u32("general.file_type", 15);
    w.kv_u32(&format!("{arch}.context_length"), 4096);
    w.kv_u32(&format!("{arch}.block_count"), 4);
    w.kv_u32(&format!("{arch}.embedding_length"), 256);
    w.kv_u32(&format!("{arch}.attention.head_count"), 4);
    w.kv_u32(&format!("{arch}.attention.head_count_kv"), 2);
    w.kv_u32("split.no", split_no);
    w.kv_u32("split.count", split_count);
    w.finish()
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    hex::encode(Sha256::digest(bytes))
}

/// Commit sha the fake Hub reports for every repo unless a test overrides it.
pub const FAKE_SHA: &str = "0123456789abcdef0123456789abcdef01234567";

/// Fake Hub: returns canned search results, file listings, a pinned head sha and tags.
pub struct FakeHf {
    pub search: Vec<HfModelSummary>,
    pub files: Vec<RepoFile>,
    /// Head commit `repo_snapshot` reports. Empty string means "no sha published".
    pub revision: String,
    pub tags: Vec<String>,
}

impl FakeHf {
    /// A fake with the default head sha and no tags.
    pub fn new(search: Vec<HfModelSummary>, files: Vec<RepoFile>) -> Self {
        Self {
            search,
            files,
            revision: FAKE_SHA.to_string(),
            tags: Vec::new(),
        }
    }
}

#[async_trait]
impl HfApi for FakeHf {
    async fn search_gguf(&self, _q: &str, _s: SearchSort, _l: u32) -> Result<Vec<HfModelSummary>> {
        Ok(self.search.clone())
    }
    async fn list_repo_files(&self, _repo: &str) -> Result<Vec<RepoFile>> {
        Ok(self.files.clone())
    }
    async fn list_repo_files_at(&self, _repo: &str, revision: &str) -> Result<Vec<RepoFile>> {
        crate::hf::validate_commit_sha(revision)?;
        Ok(self.files.clone())
    }
    async fn repo_info(&self, repo: &str) -> Result<HfRepoInfo> {
        Ok(HfRepoInfo {
            id: repo.to_string(),
            pipeline_tag: None,
            library_name: None,
            license: None,
            tags: self.tags.clone(),
            gated: false,
        })
    }
    async fn repo_snapshot(&self, repo: &str) -> Result<RepoSnapshot> {
        if self.revision.is_empty() {
            return Err(RaptorError::Network(
                "repo response has no commit sha".into(),
            ));
        }
        Ok(RepoSnapshot {
            info: self.repo_info(repo).await?,
            revision: self.revision.clone(),
        })
    }
    fn file_url(&self, repo: &str, file_name: &str) -> Result<String> {
        crate::hf::resolve_url("https://fake.hf.test", repo, file_name)
    }
    fn file_url_at(&self, repo: &str, revision: &str, file_name: &str) -> Result<String> {
        crate::hf::resolve_url_at("https://fake.hf.test", repo, revision, file_name)
    }
}

pub struct FixedDisk(pub u64);

impl DiskSpace for FixedDisk {
    fn available_bytes(&self, _dir: &Path) -> Result<u64> {
        Ok(self.0)
    }
}

/// Two fake volumes: the downloads directory has one amount of space, everything else
/// another, and the two never share a volume.
pub struct SplitVolumeDisk {
    downloads: PathBuf,
    downloads_free: u64,
    other_free: u64,
}

impl SplitVolumeDisk {
    pub fn new(downloads: PathBuf, downloads_free: u64, other_free: u64) -> Self {
        Self {
            downloads,
            downloads_free,
            other_free,
        }
    }
}

impl DiskSpace for SplitVolumeDisk {
    fn available_bytes(&self, dir: &Path) -> Result<u64> {
        Ok(if dir.starts_with(&self.downloads) {
            self.downloads_free
        } else {
            self.other_free
        })
    }

    fn same_volume(&self, a: &Path, b: &Path) -> bool {
        a.starts_with(&self.downloads) == b.starts_with(&self.downloads)
    }
}

/// Fault injected into one `open` call, relative to the bytes that call serves.
#[derive(Debug, Clone, Copy)]
pub enum Fault {
    /// Stream `n` bytes, then fail with a network error.
    FailAfter(usize),
    /// Stream `n` bytes, then never produce another chunk (a stalled connection).
    StallAfter(usize),
}

/// In-memory HTTP server with optional Range support and scripted faults.
pub struct FakeFetch {
    pub body: Vec<u8>,
    pub honor_range: bool,
    faults: Mutex<VecDeque<Fault>>,
    opens: Mutex<Vec<u64>>,
}

const CHUNK: usize = 1000;

impl FakeFetch {
    pub fn new(body: Vec<u8>) -> Self {
        Self {
            body,
            honor_range: true,
            faults: Mutex::new(VecDeque::new()),
            opens: Mutex::new(Vec::new()),
        }
    }

    pub fn with_faults(self, faults: Vec<Fault>) -> Self {
        *self.faults.lock().unwrap() = faults.into();
        self
    }

    pub fn ignoring_range(mut self) -> Self {
        self.honor_range = false;
        self
    }

    /// Offsets requested so far, in order.
    pub fn opens(&self) -> Vec<u64> {
        self.opens.lock().unwrap().clone()
    }
}

#[async_trait]
impl HttpFetch for FakeFetch {
    async fn open(&self, _url: &str, offset: u64) -> Result<RemoteResponse> {
        self.opens.lock().unwrap().push(offset);
        let fault = self.faults.lock().unwrap().pop_front();
        let resumed = self.honor_range && offset > 0;
        let len = self.body.len();
        let start = if resumed {
            (offset as usize).min(len)
        } else {
            0
        };
        let (limit, stall, fail) = match fault {
            Some(Fault::FailAfter(n)) => ((start + n).min(len), false, true),
            Some(Fault::StallAfter(n)) => ((start + n).min(len), true, false),
            None => (len, false, false),
        };
        let mut items: Vec<Result<Vec<u8>>> = Vec::new();
        let mut pos = start;
        while pos < limit {
            let end = (pos + CHUNK).min(limit);
            items.push(Ok(self.body[pos..end].to_vec()));
            pos = end;
        }
        let body: BoxStream<'static, Result<Vec<u8>>> = if fail {
            stream::iter(
                items
                    .into_iter()
                    .chain(std::iter::once(Err(RaptorError::Network(
                        "injected mid-stream failure".into(),
                    )))),
            )
            .boxed()
        } else if stall {
            stream::iter(items).chain(stream::pending()).boxed()
        } else {
            stream::iter(items).boxed()
        };
        Ok(RemoteResponse {
            total_len: Some(len as u64),
            resumed,
            body,
        })
    }
}

/// Fetch transport that serves a different body per URL, for shard sets.
#[derive(Default)]
pub struct RoutedFetch {
    routes: HashMap<String, Arc<FakeFetch>>,
}

impl RoutedFetch {
    pub fn new() -> Self {
        Self::default()
    }

    /// Serves `body` (with range support) at `url`.
    pub fn route(&mut self, url: &str, body: Vec<u8>) {
        self.routes
            .insert(url.to_string(), Arc::new(FakeFetch::new(body)));
    }

    /// Serves a preconfigured fake (for example one with injected faults) at `url`.
    pub fn route_fake(&mut self, url: &str, fake: Arc<FakeFetch>) {
        self.routes.insert(url.to_string(), fake);
    }
}

#[async_trait]
impl HttpFetch for RoutedFetch {
    async fn open(&self, url: &str, offset: u64) -> Result<RemoteResponse> {
        match self.routes.get(url) {
            Some(fake) => fake.open(url, offset).await,
            None => Err(RaptorError::NotFound(format!("no fake route for {url}"))),
        }
    }
}

/// Registrar that accepts everything and records nothing.
pub struct AcceptAll;

impl Registrar for AcceptAll {
    fn register(
        &self,
        _path: &Path,
        _sha256: &str,
        _source: &ModelSource,
        _replace: bool,
    ) -> Result<()> {
        Ok(())
    }

    fn register_set(
        &self,
        _shards: &[StagedShard],
        _source: &ModelSource,
        _replace: bool,
        _provenance: &Provenance,
    ) -> Result<()> {
        Ok(())
    }
}

/// Registrar that always refuses, to exercise post-promotion cleanup.
pub struct RefuseAll;

impl Registrar for RefuseAll {
    fn register(
        &self,
        _path: &Path,
        _sha256: &str,
        _source: &ModelSource,
        _replace: bool,
    ) -> Result<()> {
        Err(RaptorError::Conflict("refused by test registrar".into()))
    }

    fn register_set(
        &self,
        _shards: &[StagedShard],
        _source: &ModelSource,
        _replace: bool,
        _provenance: &Provenance,
    ) -> Result<()> {
        Err(RaptorError::Conflict("refused by test registrar".into()))
    }
}

/// Registrar that counts calls, so a test can assert a set is registered once, as one unit.
#[derive(Default)]
pub struct RecordingRegistrar {
    singles: AtomicUsize,
    sets: AtomicUsize,
}

impl RecordingRegistrar {
    pub fn single_calls(&self) -> usize {
        self.singles.load(Ordering::SeqCst)
    }

    pub fn set_calls(&self) -> usize {
        self.sets.load(Ordering::SeqCst)
    }
}

impl Registrar for RecordingRegistrar {
    fn register(
        &self,
        _path: &Path,
        _sha256: &str,
        _source: &ModelSource,
        _replace: bool,
    ) -> Result<()> {
        self.singles.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn register_set(
        &self,
        _shards: &[StagedShard],
        _source: &ModelSource,
        _replace: bool,
        _provenance: &Provenance,
    ) -> Result<()> {
        self.sets.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}
