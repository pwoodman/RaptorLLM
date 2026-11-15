//! Download manager: queued, resumable, verified transfers promoted into the library.
//!
//! Single file: wait for a concurrency slot, check the destination, reconcile the `.part` file
//! with its sidecar record, check disk space, optionally reject an unsupported architecture
//! from the first bytes, stream with HTTP Range resume, verify sha256 against the Hub's LFS
//! hash, atomically rename into place, then register.
//!
//! Shard set: the same pipeline runs for each shard in order under one job. Only when every
//! shard is verified are all of them renamed into place and registered as one entry. If any
//! step after verification fails, promoted shards are moved back to the downloads directory,
//! so the library never holds part of a set and an interrupted set can still resume.
//!
//! Transport, disk space and registration are traits, so tests run with fakes.

use crate::gguf::{self, HeaderProbe};
use crate::hf::{map_http_status, safe_repo_path, validate_repo_id};
use crate::library::ModelSource;
use crate::shard::MAX_SHARDS;
use crate::util::{fingerprint, is_sha256_hex, lock, remove_if_exists, sha256_file, write_atomic};
use async_trait::async_trait;
use futures_util::stream::BoxStream;
use futures_util::StreamExt;
use raptor_core::paths::safe_file_name;
use raptor_core::{RaptorError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{oneshot, watch, OwnedSemaphorePermit, Semaphore};

/// One open HTTP response for a byte range.
pub struct RemoteResponse {
    /// Full size of the resource, if the server said.
    pub total_len: Option<u64>,
    /// True only when the server honored the requested Range (206).
    pub resumed: bool,
    pub body: BoxStream<'static, Result<Vec<u8>>>,
}

/// Transport for downloads. Production uses [`HttpFetcher`]; tests use a fake.
#[async_trait]
pub trait HttpFetch: Send + Sync {
    /// Opens a GET for `url`, starting at `offset` (sends a Range header when offset > 0).
    async fn open(&self, url: &str, offset: u64) -> Result<RemoteResponse>;
}

/// Free-space query used by the disk preflight.
pub trait DiskSpace: Send + Sync {
    /// Bytes available to the caller on the volume holding `dir` (or its nearest existing parent).
    fn available_bytes(&self, dir: &Path) -> Result<u64>;

    /// True when both paths are known to be on one volume. Defaults to [`same_volume`],
    /// which answers false whenever it cannot tell (the conservative answer).
    fn same_volume(&self, a: &Path, b: &Path) -> bool {
        same_volume(a, b)
    }
}

/// Free space from the operating system (`fs2::available_space`, which uses `statvfs` on Unix
/// and `GetDiskFreeSpaceEx` on Windows).
pub struct SystemDiskSpace;

impl DiskSpace for SystemDiskSpace {
    fn available_bytes(&self, dir: &Path) -> Result<u64> {
        let probe = existing_ancestor(dir)
            .ok_or_else(|| RaptorError::Io(format!("no existing parent for {}", dir.display())))?;
        fs2::available_space(&probe)
            .map_err(|e| RaptorError::Io(format!("cannot read free disk space: {e}")))
    }
}

/// The nearest path (the path itself or an ancestor) that exists on disk.
fn existing_ancestor(path: &Path) -> Option<PathBuf> {
    let mut cur = path;
    loop {
        if cur.exists() {
            return Some(cur.to_path_buf());
        }
        cur = cur.parent()?;
    }
}

/// True when both paths are on the same volume. Unix compares device ids; Windows compares
/// the drive or UNC prefix. Returns false when either path cannot be inspected.
pub fn same_volume(a: &Path, b: &Path) -> bool {
    let (Some(a), Some(b)) = (existing_ancestor(a), existing_ancestor(b)) else {
        return false;
    };
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        match (fs::metadata(&a), fs::metadata(&b)) {
            (Ok(x), Ok(y)) => x.dev() == y.dev(),
            _ => false,
        }
    }
    #[cfg(windows)]
    {
        use std::path::Component;
        match (a.components().next(), b.components().next()) {
            (Some(Component::Prefix(x)), Some(Component::Prefix(y))) => x
                .as_os_str()
                .to_string_lossy()
                .eq_ignore_ascii_case(&y.as_os_str().to_string_lossy()),
            _ => false,
        }
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (a, b);
        false
    }
}

/// What a finished job records about where its bytes came from.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Provenance {
    /// Commit sha the bytes were fetched from.
    pub revision: Option<String>,
    /// Hub tags of the repo at download time.
    pub hub_tags: Vec<String>,
}

/// One verified shard handed to [`Registrar::register_set`]. `path` is its final location.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StagedShard {
    pub path: PathBuf,
    pub sha256: String,
}

/// Records a verified file in the library after it is promoted.
pub trait Registrar: Send + Sync {
    /// `replace` is true only when the caller already enforced an explicit overwrite.
    fn register(
        &self,
        path: &Path,
        sha256: &str,
        source: &ModelSource,
        replace: bool,
    ) -> Result<()>;

    /// Like [`Registrar::register`], also recording provenance. The default ignores it.
    fn register_with(
        &self,
        path: &Path,
        sha256: &str,
        source: &ModelSource,
        replace: bool,
        provenance: &Provenance,
    ) -> Result<()> {
        let _ = provenance;
        self.register(path, sha256, source, replace)
    }

    /// Records a complete shard set as one entry. Must be all-or-nothing. The default refuses.
    fn register_set(
        &self,
        shards: &[StagedShard],
        source: &ModelSource,
        replace: bool,
        provenance: &Provenance,
    ) -> Result<()> {
        let _ = (shards, source, replace, provenance);
        Err(RaptorError::Unsupported(
            "this registrar does not accept shard sets".into(),
        ))
    }
}

/// reqwest-backed transport. The token, if any, is sent only as a bearer header.
pub struct HttpFetcher {
    client: reqwest::Client,
    token: Option<String>,
}

impl HttpFetcher {
    pub fn new(token: Option<String>) -> Result<Self> {
        let client = reqwest::Client::builder()
            .user_agent(concat!("RaptorLLM/", env!("CARGO_PKG_VERSION")))
            .connect_timeout(Duration::from_secs(15))
            .build()
            .map_err(|e| RaptorError::Network(format!("cannot build HTTP client: {e}")))?;
        Ok(Self {
            client,
            token: token.filter(|t| !t.trim().is_empty()),
        })
    }
}

impl std::fmt::Debug for HttpFetcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HttpFetcher")
            .field("token", &self.token.as_ref().map(|_| "<redacted>"))
            .finish()
    }
}

#[async_trait]
impl HttpFetch for HttpFetcher {
    async fn open(&self, url: &str, offset: u64) -> Result<RemoteResponse> {
        let mut req = self.client.get(url);
        if offset > 0 {
            req = req.header(reqwest::header::RANGE, format!("bytes={offset}-"));
        }
        if let Some(t) = &self.token {
            req = req.bearer_auth(t);
        }
        let resp = req.send().await.map_err(|e| {
            RaptorError::Network(format!("download request failed: {}", e.without_url()))
        })?;
        let status = resp.status();
        if status == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
            return Err(RaptorError::InvalidInput(
                "server rejected the resume range; restarting the download".into(),
            ));
        }
        if status != reqwest::StatusCode::OK && status != reqwest::StatusCode::PARTIAL_CONTENT {
            let retry = resp
                .headers()
                .get(reqwest::header::RETRY_AFTER)
                .and_then(|v| v.to_str().ok())
                .map(str::to_string);
            return Err(map_http_status(status.as_u16(), retry));
        }
        let resumed = status == reqwest::StatusCode::PARTIAL_CONTENT;
        let total_len = if resumed {
            resp.headers()
                .get(reqwest::header::CONTENT_RANGE)
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.rsplit_once('/'))
                .and_then(|(_, total)| total.parse::<u64>().ok())
        } else {
            resp.content_length()
        };
        let body = resp
            .bytes_stream()
            .map(|chunk| {
                chunk.map(|b| b.to_vec()).map_err(|e| {
                    RaptorError::Network(format!("download interrupted: {}", e.without_url()))
                })
            })
            .boxed();
        Ok(RemoteResponse {
            total_len,
            resumed,
            body,
        })
    }
}

/// Byte prefixes tried, in order, when looking for the architecture at the start of a file.
const PROBE_STEPS: [u64; 3] = [1 << 20, 8 << 20, 32 << 20];

/// Reads just enough of a remote GGUF to learn its architecture, before any big transfer.
pub async fn probe_remote_header(fetch: &dyn HttpFetch, url: &str) -> Result<HeaderProbe> {
    let mut last = None;
    for want in PROBE_STEPS {
        let prefix = read_prefix(fetch, url, want).await?;
        match gguf::probe(&prefix)? {
            HeaderProbe::Complete(meta) => return Ok(HeaderProbe::Complete(meta)),
            HeaderProbe::Truncated {
                architecture: Some(a),
            } => {
                return Ok(HeaderProbe::Truncated {
                    architecture: Some(a),
                })
            }
            t @ HeaderProbe::Truncated { architecture: None } => last = Some(t),
        }
    }
    Ok(last.unwrap_or(HeaderProbe::Truncated { architecture: None }))
}

async fn read_prefix(fetch: &dyn HttpFetch, url: &str, want: u64) -> Result<Vec<u8>> {
    let mut resp = fetch.open(url, 0).await?;
    let mut buf: Vec<u8> = Vec::new();
    while (buf.len() as u64) < want {
        match resp.body.next().await {
            Some(chunk) => buf.extend_from_slice(&chunk?),
            None => break,
        }
    }
    let keep = usize::try_from(want).unwrap_or(usize::MAX).min(buf.len());
    buf.truncate(keep);
    Ok(buf)
}

/// Phase of a download job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobState {
    Queued,
    Running,
    Paused,
    Verifying,
    Completed,
    Failed,
    Cancelled,
}

/// Snapshot of a job, published through a watch channel.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DownloadProgress {
    pub key: String,
    pub state: JobState,
    pub downloaded: u64,
    pub total: u64,
    pub attempts: u32,
}

/// Final result of a job that did not fail. For a shard set, `path` is the first shard and
/// `sha256` is the set fingerprint (see `util::fingerprint`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DownloadOutcome {
    Completed { path: PathBuf, sha256: String },
    Cancelled,
}

/// Everything needed to fetch and verify one file.
#[derive(Debug, Clone)]
pub struct DownloadRequest {
    pub repo: String,
    /// Repo-relative path, e.g. `model-Q4_K_M.gguf` or `Q4/model-00001-of-00002.gguf`.
    pub file_name: String,
    pub url: String,
    /// Lowercase or uppercase hex sha256 from the Hub.
    pub expected_sha256: String,
    pub expected_size: u64,
    /// Final location. Its file name must equal the last segment of `file_name`.
    pub dest: PathBuf,
    /// Permit replacing an existing file at `dest`.
    pub overwrite: bool,
    pub source: ModelSource,
    /// Reject unsupported architectures from the first bytes, before the full transfer.
    pub verify_architecture: bool,
}

/// Tunables for the manager.
#[derive(Debug, Clone)]
pub struct DownloadConfig {
    pub downloads_dir: PathBuf,
    pub max_concurrent: usize,
    /// Free space that must remain after the transfer.
    pub min_free_margin: u64,
    /// Network attempts per job before giving up (the partial file is kept).
    pub max_attempts: u32,
    pub retry_backoff: Duration,
}

impl DownloadConfig {
    /// Defaults: 2 concurrent jobs, 256 MiB free-space margin, 5 attempts, 2 s backoff.
    pub fn new(downloads_dir: impl Into<PathBuf>) -> Self {
        Self {
            downloads_dir: downloads_dir.into(),
            max_concurrent: 2,
            min_free_margin: 256 * 1024 * 1024,
            max_attempts: 5,
            retry_backoff: Duration::from_secs(2),
        }
    }
}

/// An interrupted download found on disk.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PartialDownload {
    pub key: String,
    pub repo: String,
    pub file_name: String,
    pub expected_size: u64,
    pub downloaded_bytes: u64,
}

/// Result of scanning the downloads directory at startup.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RecoveryReport {
    pub partials: Vec<PartialDownload>,
    /// Unreadable records and orphaned `.part` files, described for the UI.
    pub problems: Vec<String>,
}

/// Handle to a started job.
pub struct DownloadHandle {
    manager: DownloadManager,
    key: String,
    progress: watch::Receiver<DownloadProgress>,
    done: oneshot::Receiver<Result<DownloadOutcome>>,
}

impl DownloadHandle {
    /// The job key (`owner--repo--file`; for a set, the first shard's key).
    pub fn key(&self) -> &str {
        &self.key
    }

    /// A receiver that always holds the latest progress snapshot.
    pub fn progress(&self) -> watch::Receiver<DownloadProgress> {
        self.progress.clone()
    }

    /// Pauses a running job. The partial file is kept; the connection is closed.
    pub fn pause(&self) -> Result<()> {
        self.manager.send_control(&self.key, Control::Pause)
    }

    /// Resumes a paused job from its partial file.
    pub fn resume(&self) -> Result<()> {
        self.manager.send_control(&self.key, Control::Run)
    }

    /// Cancels the job and deletes its partial files.
    pub fn cancel(&self) -> Result<()> {
        self.manager.send_control(&self.key, Control::Cancel)
    }

    /// Waits for the job to finish.
    pub async fn finish(self) -> Result<DownloadOutcome> {
        self.done
            .await
            .map_err(|_| RaptorError::Internal("download task ended without a result".into()))?
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Control {
    Run,
    Pause,
    Cancel,
}

#[derive(Default)]
struct ActiveSet {
    keys: HashSet<String>,
    dests: HashSet<PathBuf>,
}

/// Snapshot of one job for status displays. Byte counts come from the job's
/// progress channel, so they are as current as the last update.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct JobStatus {
    /// Job key, `owner--repo--file`.
    pub key: String,
    pub repo: String,
    pub file_name: String,
    pub state: JobState,
    pub downloaded: u64,
    pub total: u64,
    pub attempts: u32,
}

/// What `list_jobs` needs to know about one job.
struct JobEntry {
    repo: String,
    file_name: String,
    progress: watch::Receiver<DownloadProgress>,
}

struct Inner {
    cfg: DownloadConfig,
    fetch: Arc<dyn HttpFetch>,
    disk: Arc<dyn DiskSpace>,
    registrar: Arc<dyn Registrar>,
    slots: Arc<Semaphore>,
    active: Mutex<ActiveSet>,
    controls: Mutex<HashMap<String, watch::Sender<Control>>>,
    jobs: Mutex<BTreeMap<String, JobEntry>>,
}

/// Queue of verified downloads. Cheap to clone; clones share the same queue.
#[derive(Clone)]
pub struct DownloadManager {
    inner: Arc<Inner>,
}

/// Job key for a Hub file: `owner--repo--file`, with `/` in the path written as `%2F`.
/// Validates both parts. Used as the stem of the `.part` and sidecar files.
pub fn job_key(repo: &str, file_name: &str) -> Result<String> {
    validate_repo_id(repo)?;
    safe_repo_path(file_name)?;
    let key = format!(
        "{}--{}",
        repo.replace('/', "--"),
        file_name.replace('/', "%2F")
    );
    safe_file_name(&key)?;
    Ok(key)
}

fn basename(file_name: &str) -> &str {
    file_name.rsplit('/').next().unwrap_or(file_name)
}

fn validate_request(req: &mut DownloadRequest) -> Result<()> {
    validate_repo_id(&req.repo)?;
    safe_repo_path(&req.file_name)?;
    if !req.file_name.to_ascii_lowercase().ends_with(".gguf") {
        return Err(RaptorError::InvalidInput(
            "only .gguf files can be downloaded".into(),
        ));
    }
    if !is_sha256_hex(&req.expected_sha256) {
        return Err(RaptorError::InvalidInput(
            "expected sha256 must be 64 hex characters".into(),
        ));
    }
    req.expected_sha256 = req.expected_sha256.to_ascii_lowercase();
    if req.expected_size == 0 {
        return Err(RaptorError::InvalidInput(
            "expected size must be greater than zero".into(),
        ));
    }
    if !req.url.starts_with("https://") {
        return Err(RaptorError::InvalidInput(
            "download URL must use https".into(),
        ));
    }
    if req.dest.file_name().and_then(|n| n.to_str()) != Some(basename(&req.file_name)) {
        return Err(RaptorError::InvalidInput(
            "destination file name does not match the requested file".into(),
        ));
    }
    Ok(())
}

/// What a job runs.
enum Plan {
    Single {
        req: DownloadRequest,
        provenance: Provenance,
    },
    Set {
        reqs: Vec<DownloadRequest>,
        provenance: Provenance,
    },
}

impl DownloadManager {
    pub fn new(
        cfg: DownloadConfig,
        fetch: Arc<dyn HttpFetch>,
        disk: Arc<dyn DiskSpace>,
        registrar: Arc<dyn Registrar>,
    ) -> Self {
        let slots = Arc::new(Semaphore::new(cfg.max_concurrent.max(1)));
        Self {
            inner: Arc::new(Inner {
                cfg,
                fetch,
                disk,
                registrar,
                slots,
                active: Mutex::new(ActiveSet::default()),
                controls: Mutex::new(HashMap::new()),
                jobs: Mutex::new(BTreeMap::new()),
            }),
        }
    }

    /// Queues a download and returns immediately. Must be called inside a Tokio runtime.
    ///
    /// Refuses (without touching disk) a destination that exists unless `overwrite`, and
    /// a second job for the same file or destination while one is active.
    pub fn start(&self, req: DownloadRequest) -> Result<DownloadHandle> {
        self.start_with(req, Provenance::default())
    }

    /// Like [`DownloadManager::start`], recording `provenance` (pinned revision, Hub tags)
    /// with the library entry.
    pub fn start_with(
        &self,
        mut req: DownloadRequest,
        provenance: Provenance,
    ) -> Result<DownloadHandle> {
        validate_request(&mut req)?;
        let key = job_key(&req.repo, &req.file_name)?;
        if req.dest.exists() && !req.overwrite {
            return Err(RaptorError::Conflict(format!(
                "{} already exists",
                req.file_name
            )));
        }
        let (repo, file_name, total, dest) = (
            req.repo.clone(),
            req.file_name.clone(),
            req.expected_size,
            req.dest.clone(),
        );
        self.spawn_job(
            Plan::Single { req, provenance },
            vec![key],
            vec![dest],
            repo,
            file_name,
            total,
        )
    }

    /// Queues a shard set as one job. Shards run in the order given; nothing is registered
    /// until every shard is verified, and then all of them are registered together.
    ///
    /// Every request must come from the same repo and carry a distinct file and destination.
    /// Refuses (without touching disk) any destination that exists unless its request
    /// allows overwrite.
    pub fn start_set(
        &self,
        mut reqs: Vec<DownloadRequest>,
        provenance: Provenance,
    ) -> Result<DownloadHandle> {
        if reqs.is_empty() || reqs.len() > MAX_SHARDS as usize {
            return Err(RaptorError::InvalidInput(format!(
                "a shard set needs between 1 and {MAX_SHARDS} shards"
            )));
        }
        for r in reqs.iter_mut() {
            validate_request(r)?;
        }
        let repo = reqs[0].repo.clone();
        if reqs.iter().any(|r| r.repo != repo) {
            return Err(RaptorError::InvalidInput(
                "shards of one set must come from one repo".into(),
            ));
        }
        let keys: Vec<String> = reqs
            .iter()
            .map(|r| job_key(&r.repo, &r.file_name))
            .collect::<Result<_>>()?;
        let dests: Vec<PathBuf> = reqs.iter().map(|r| r.dest.clone()).collect();
        let distinct_keys: HashSet<&String> = keys.iter().collect();
        let distinct_dests: HashSet<&PathBuf> = dests.iter().collect();
        if distinct_keys.len() != keys.len() || distinct_dests.len() != dests.len() {
            return Err(RaptorError::InvalidInput(
                "shard set lists the same file or destination twice".into(),
            ));
        }
        for r in &reqs {
            if r.dest.exists() && !r.overwrite {
                return Err(RaptorError::Conflict(format!(
                    "{} already exists",
                    r.file_name
                )));
            }
        }
        let total: u64 = reqs.iter().map(|r| r.expected_size).sum();
        let file_name = reqs[0].file_name.clone();
        self.spawn_job(
            Plan::Set { reqs, provenance },
            keys,
            dests,
            repo,
            file_name,
            total,
        )
    }

    fn spawn_job(
        &self,
        plan: Plan,
        keys: Vec<String>,
        dests: Vec<PathBuf>,
        repo: String,
        file_name: String,
        total: u64,
    ) -> Result<DownloadHandle> {
        let head = keys[0].clone();
        {
            let mut active = lock(&self.inner.active)?;
            if keys.iter().any(|k| active.keys.contains(k))
                || dests.iter().any(|d| active.dests.contains(d))
            {
                return Err(RaptorError::Conflict(format!(
                    "{head} is already downloading"
                )));
            }
            active.keys.extend(keys.iter().cloned());
            active.dests.extend(dests.iter().cloned());
        }

        let (ctl_tx, ctl_rx) = watch::channel(Control::Run);
        lock(&self.inner.controls)?.insert(head.clone(), ctl_tx);
        let (prog_tx, prog_rx) = watch::channel(DownloadProgress {
            key: head.clone(),
            state: JobState::Queued,
            downloaded: 0,
            total,
            attempts: 0,
        });
        let (done_tx, done_rx) = oneshot::channel();
        lock(&self.inner.jobs)?.insert(
            head.clone(),
            JobEntry {
                repo,
                file_name,
                progress: prog_rx.clone(),
            },
        );

        let inner = Arc::clone(&self.inner);
        let task_head = head.clone();
        tokio::spawn(async move {
            let mut control = ctl_rx;
            let outcome = match plan {
                Plan::Single { req, provenance } => {
                    run_single(
                        &inner,
                        &req,
                        &task_head,
                        &prog_tx,
                        &mut control,
                        &provenance,
                    )
                    .await
                }
                Plan::Set { reqs, provenance } => {
                    run_set(&inner, &reqs, &prog_tx, &mut control, &provenance).await
                }
            };
            let state = match &outcome {
                Ok(DownloadOutcome::Completed { .. }) => JobState::Completed,
                Ok(DownloadOutcome::Cancelled) => JobState::Cancelled,
                Err(_) => JobState::Failed,
            };
            prog_tx.send_modify(|p| p.state = state);
            release(&inner, &keys, &dests, &task_head);
            let _ = done_tx.send(outcome);
        });

        Ok(DownloadHandle {
            manager: self.clone(),
            key: head,
            progress: prog_rx,
            done: done_rx,
        })
    }

    /// Snapshot of every job started on this manager, sorted by key. Finished jobs
    /// stay listed with their final state (`Completed`, `Failed` or `Cancelled`)
    /// until the same file is started again.
    pub fn list_jobs(&self) -> Result<Vec<JobStatus>> {
        let jobs = lock(&self.inner.jobs)?;
        Ok(jobs
            .iter()
            .map(|(key, entry)| {
                let p = entry.progress.borrow().clone();
                JobStatus {
                    key: key.clone(),
                    repo: entry.repo.clone(),
                    file_name: entry.file_name.clone(),
                    state: p.state,
                    downloaded: p.downloaded,
                    total: p.total,
                    attempts: p.attempts,
                }
            })
            .collect())
    }

    /// Pauses a queued or running job by key.
    pub fn pause(&self, key: &str) -> Result<()> {
        self.send_control(key, Control::Pause)
    }

    /// Resumes a paused job by key.
    pub fn resume(&self, key: &str) -> Result<()> {
        self.send_control(key, Control::Run)
    }

    /// Cancels a job by key and deletes its partial files.
    pub fn cancel(&self, key: &str) -> Result<()> {
        self.send_control(key, Control::Cancel)
    }

    /// Scans the downloads directory for interrupted transfers. Call at startup. Partials
    /// are reported, never promoted; a matching `start` resumes one from its bytes.
    pub fn recover(&self) -> Result<RecoveryReport> {
        let dir = &self.inner.cfg.downloads_dir;
        let mut report = RecoveryReport {
            partials: Vec::new(),
            problems: Vec::new(),
        };
        if !dir.is_dir() {
            return Ok(report);
        }
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().into_owned();
            if let Some(key) = name.strip_suffix(".part.json") {
                match read_sidecar(&entry.path()) {
                    Ok(Some(meta)) => report.partials.push(PartialDownload {
                        key: key.to_string(),
                        repo: meta.repo,
                        file_name: meta.file_name,
                        expected_size: meta.expected_size,
                        downloaded_bytes: part_len(&dir.join(format!("{key}.part")))?,
                    }),
                    Ok(None) => {}
                    Err(e) => report
                        .problems
                        .push(format!("unreadable download record {name}: {e}")),
                }
            } else if let Some(key) = name.strip_suffix(".part") {
                if !dir.join(format!("{key}.part.json")).exists() {
                    report
                        .problems
                        .push(format!("orphaned partial file {name} (no record)"));
                }
            }
        }
        report.partials.sort_by(|a, b| a.key.cmp(&b.key));
        Ok(report)
    }

    /// Deletes an interrupted transfer's files. Refused while that job is active.
    pub fn discard_partial(&self, key: &str) -> Result<()> {
        safe_file_name(key)?;
        if lock(&self.inner.active)?.keys.contains(key) {
            return Err(RaptorError::Conflict(format!(
                "{key} is active; cancel it first"
            )));
        }
        let dir = &self.inner.cfg.downloads_dir;
        discard_files(
            &dir.join(format!("{key}.part")),
            &dir.join(format!("{key}.part.json")),
        )
    }

    fn send_control(&self, key: &str, c: Control) -> Result<()> {
        let controls = lock(&self.inner.controls)?;
        let tx = controls
            .get(key)
            .ok_or_else(|| RaptorError::NotFound(format!("no active download: {key}")))?;
        let _previous = tx.send_replace(c);
        Ok(())
    }
}

fn release(inner: &Inner, keys: &[String], dests: &[PathBuf], head: &str) {
    if let Ok(mut active) = inner.active.lock() {
        for k in keys {
            active.keys.remove(k);
        }
        for d in dests {
            active.dests.remove(d);
        }
    }
    if let Ok(mut controls) = inner.controls.lock() {
        controls.remove(head);
    }
}

/// Persisted alongside each `.part`. A resume is allowed only when this matches exactly.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct JobMeta {
    repo: String,
    file_name: String,
    expected_sha256: String,
    expected_size: u64,
}

impl From<&DownloadRequest> for JobMeta {
    fn from(req: &DownloadRequest) -> Self {
        Self {
            repo: req.repo.clone(),
            file_name: req.file_name.clone(),
            expected_sha256: req.expected_sha256.clone(),
            expected_size: req.expected_size,
        }
    }
}

/// Outcome of one streaming attempt.
enum StreamEnd {
    Complete,
    Paused,
    Cancelled,
}

/// Result of fetching and verifying one file, before any promotion.
enum Fetched {
    Verified(String),
    Cancelled,
}

/// Where a shard sits inside its job, for progress: bytes done before it and the job total.
#[derive(Debug, Clone, Copy)]
struct Scope {
    base: u64,
    total: u64,
}

async fn next_control(rx: &mut watch::Receiver<Control>) -> Control {
    match rx.changed().await {
        Ok(()) => *rx.borrow_and_update(),
        // The sender lives in the manager until the job ends, so this is unreachable in practice.
        Err(_) => std::future::pending().await,
    }
}

/// Waits for a concurrency slot. `None` means cancelled while queued.
async fn acquire_slot(
    inner: &Inner,
    control: &mut watch::Receiver<Control>,
) -> Result<Option<OwnedSemaphorePermit>> {
    loop {
        if *control.borrow() == Control::Cancel {
            return Ok(None);
        }
        tokio::select! {
            permit = Arc::clone(&inner.slots).acquire_owned() => {
                return permit
                    .map(Some)
                    .map_err(|_| RaptorError::Internal("download queue is closed".into()));
            }
            c = next_control(control) => {
                if c == Control::Cancel {
                    return Ok(None);
                }
            }
        }
    }
}

/// Blocks while paused. Returns false if the job was cancelled.
async fn wait_if_paused(
    control: &mut watch::Receiver<Control>,
    progress: &watch::Sender<DownloadProgress>,
) -> bool {
    loop {
        let current = *control.borrow();
        match current {
            Control::Run => return true,
            Control::Cancel => return false,
            Control::Pause => {
                progress.send_modify(|p| p.state = JobState::Paused);
                next_control(control).await;
            }
        }
    }
}

async fn run_single(
    inner: &Inner,
    req: &DownloadRequest,
    key: &str,
    progress: &watch::Sender<DownloadProgress>,
    control: &mut watch::Receiver<Control>,
    provenance: &Provenance,
) -> Result<DownloadOutcome> {
    let Some(_slot) = acquire_slot(inner, control).await? else {
        return Ok(DownloadOutcome::Cancelled);
    };
    let dir = inner.cfg.downloads_dir.clone();
    fs::create_dir_all(&dir)?;
    let part = dir.join(format!("{key}.part"));
    let sidecar = dir.join(format!("{key}.part.json"));

    if req.dest.exists() && !req.overwrite {
        return Err(RaptorError::Conflict(format!(
            "{} already exists",
            req.file_name
        )));
    }
    prepare_partial(&part, &sidecar, &JobMeta::from(req))?;

    let result = single_body(inner, req, &part, &sidecar, progress, control, provenance).await;
    if matches!(result, Err(RaptorError::Integrity(_))) {
        // A permanent integrity failure must not leave bytes that could be resumed later.
        discard_files(&part, &sidecar)?;
    }
    result
}

async fn single_body(
    inner: &Inner,
    req: &DownloadRequest,
    part: &Path,
    sidecar: &Path,
    progress: &watch::Sender<DownloadProgress>,
    control: &mut watch::Receiver<Control>,
    provenance: &Provenance,
) -> Result<DownloadOutcome> {
    let remaining = req.expected_size.saturating_sub(part_len(part)?);
    preflight(inner, &[(req, remaining)])?;
    let scope = Scope {
        base: 0,
        total: req.expected_size,
    };
    match fetch_verified(
        inner,
        req,
        part,
        sidecar,
        progress,
        control,
        scope,
        req.verify_architecture,
    )
    .await?
    {
        Fetched::Cancelled => Ok(DownloadOutcome::Cancelled),
        Fetched::Verified(sha) => promote_single(inner, req, part, sidecar, &sha, provenance),
    }
}

async fn run_set(
    inner: &Inner,
    reqs: &[DownloadRequest],
    progress: &watch::Sender<DownloadProgress>,
    control: &mut watch::Receiver<Control>,
    provenance: &Provenance,
) -> Result<DownloadOutcome> {
    let Some(_slot) = acquire_slot(inner, control).await? else {
        return Ok(DownloadOutcome::Cancelled);
    };
    let dir = inner.cfg.downloads_dir.clone();
    fs::create_dir_all(&dir)?;
    for r in reqs {
        if r.dest.exists() && !r.overwrite {
            return Err(RaptorError::Conflict(format!(
                "{} already exists",
                r.file_name
            )));
        }
    }

    let mut files: Vec<(PathBuf, PathBuf)> = Vec::with_capacity(reqs.len());
    for r in reqs {
        let key = job_key(&r.repo, &r.file_name)?;
        let part = dir.join(format!("{key}.part"));
        let sidecar = dir.join(format!("{key}.part.json"));
        prepare_partial(&part, &sidecar, &JobMeta::from(r))?;
        files.push((part, sidecar));
    }
    let mut items = Vec::with_capacity(reqs.len());
    for (r, (part, _)) in reqs.iter().zip(&files) {
        items.push((r, r.expected_size.saturating_sub(part_len(part)?)));
    }
    preflight(inner, &items)?;

    let total: u64 = reqs.iter().map(|r| r.expected_size).sum();
    let mut verified: Vec<String> = Vec::with_capacity(reqs.len());
    let mut base = 0u64;
    for (i, (r, (part, sidecar))) in reqs.iter().zip(&files).enumerate() {
        let scope = Scope { base, total };
        let fetched = fetch_verified(
            inner,
            r,
            part,
            sidecar,
            progress,
            control,
            scope,
            i == 0 && r.verify_architecture,
        )
        .await;
        match fetched {
            Ok(Fetched::Verified(sha)) => {
                verified.push(sha);
                base += r.expected_size;
            }
            Ok(Fetched::Cancelled) => {
                discard_all(&files)?;
                return Ok(DownloadOutcome::Cancelled);
            }
            Err(e) => {
                if matches!(e, RaptorError::Integrity(_)) {
                    discard_files(part, sidecar)?;
                }
                return Err(e);
            }
        }
    }
    promote_set(inner, reqs, &files, &verified, provenance)
}

/// Streams (resuming as needed) and verifies one file. Leaves the verified bytes in `part`.
/// On cancellation it deletes this file's partial state and reports `Cancelled`.
#[allow(clippy::too_many_arguments)]
async fn fetch_verified(
    inner: &Inner,
    req: &DownloadRequest,
    part: &Path,
    sidecar: &Path,
    progress: &watch::Sender<DownloadProgress>,
    control: &mut watch::Receiver<Control>,
    scope: Scope,
    probe_architecture: bool,
) -> Result<Fetched> {
    if probe_architecture {
        let probe = probe_remote_header(inner.fetch.as_ref(), &req.url).await?;
        let arch = match &probe {
            HeaderProbe::Complete(meta) => meta.architecture.clone(),
            HeaderProbe::Truncated { architecture } => architecture.clone(),
        };
        if let Err(e) = gguf::ensure_supported(arch.as_deref()) {
            discard_files(part, sidecar)?;
            return Err(e);
        }
    }

    let mut attempts = 0u32;
    loop {
        let have = part_len(part)?;
        if have == req.expected_size {
            break;
        }
        if have > req.expected_size {
            return Err(RaptorError::Integrity(
                "partial file is larger than the listed size".into(),
            ));
        }
        if !wait_if_paused(control, progress).await {
            discard_files(part, sidecar)?;
            return Ok(Fetched::Cancelled);
        }
        let attempt_count = attempts;
        progress.send_modify(|p| {
            p.state = JobState::Running;
            p.downloaded = scope.base + have;
            p.total = scope.total;
            p.attempts = attempt_count;
        });
        match stream_once(inner, req, part, progress, control, scope).await {
            Ok(StreamEnd::Complete | StreamEnd::Paused) => {}
            Ok(StreamEnd::Cancelled) => {
                discard_files(part, sidecar)?;
                return Ok(Fetched::Cancelled);
            }
            Err(RaptorError::InvalidInput(_)) if have > 0 => {
                // The server rejected our resume offset: start the file over.
                remove_if_exists(part)?;
            }
            Err(e @ RaptorError::Network(_)) => {
                attempts += 1;
                if attempts >= inner.cfg.max_attempts {
                    return Err(e);
                }
                tokio::time::sleep(inner.cfg.retry_backoff).await;
                if *control.borrow() == Control::Cancel {
                    discard_files(part, sidecar)?;
                    return Ok(Fetched::Cancelled);
                }
            }
            Err(e) => return Err(e),
        }
    }

    progress.send_modify(|p| p.state = JobState::Verifying);
    let hashed = part.to_path_buf();
    let got = tokio::task::spawn_blocking(move || sha256_file(&hashed))
        .await
        .map_err(|e| RaptorError::Internal(format!("hashing task failed: {e}")))??;
    if got != req.expected_sha256 {
        discard_files(part, sidecar)?;
        return Err(RaptorError::Integrity(format!(
            "sha256 mismatch for {}: expected {}, got {}",
            req.file_name, req.expected_sha256, got
        )));
    }
    Ok(Fetched::Verified(got))
}

fn promote_single(
    inner: &Inner,
    req: &DownloadRequest,
    part: &Path,
    sidecar: &Path,
    sha: &str,
    provenance: &Provenance,
) -> Result<DownloadOutcome> {
    if req.dest.exists() && !req.overwrite {
        discard_files(part, sidecar)?;
        return Err(RaptorError::Conflict(format!(
            "{} already exists",
            req.file_name
        )));
    }
    if let Some(parent) = req.dest.parent() {
        fs::create_dir_all(parent)?;
    }
    promote(part, &req.dest)?;
    remove_if_exists(sidecar)?;
    if let Err(e) =
        inner
            .registrar
            .register_with(&req.dest, sha, &req.source, req.overwrite, provenance)
    {
        remove_if_exists(&req.dest)?;
        return Err(e);
    }
    Ok(DownloadOutcome::Completed {
        path: req.dest.clone(),
        sha256: sha.to_string(),
    })
}

/// Promotes every verified shard, then registers the set. Any failure moves the promoted
/// shards back to the downloads directory and keeps their records, so the set can resume.
fn promote_set(
    inner: &Inner,
    reqs: &[DownloadRequest],
    files: &[(PathBuf, PathBuf)],
    verified: &[String],
    provenance: &Provenance,
) -> Result<DownloadOutcome> {
    let overwrite = reqs.iter().any(|r| r.overwrite);
    let mut promoted: Vec<usize> = Vec::new();
    let mut failure: Option<RaptorError> = None;
    for (i, r) in reqs.iter().enumerate() {
        if r.dest.exists() && !r.overwrite {
            failure = Some(RaptorError::Conflict(format!(
                "{} already exists",
                r.file_name
            )));
            break;
        }
        let step = match r.dest.parent() {
            Some(parent) => fs::create_dir_all(parent).map_err(RaptorError::from),
            None => Ok(()),
        }
        .and_then(|()| promote(&files[i].0, &r.dest));
        if let Err(e) = step {
            failure = Some(e);
            break;
        }
        promoted.push(i);
    }
    if failure.is_none() {
        let shards: Vec<StagedShard> = reqs
            .iter()
            .zip(verified)
            .map(|(r, sha)| StagedShard {
                path: r.dest.clone(),
                sha256: sha.clone(),
            })
            .collect();
        if let Err(e) =
            inner
                .registrar
                .register_set(&shards, &reqs[0].source, overwrite, provenance)
        {
            failure = Some(e);
        }
    }
    if let Some(e) = failure {
        for i in promoted.iter().rev() {
            restore_part(&reqs[*i].dest, &files[*i].0);
        }
        return Err(e);
    }
    for (_, sidecar) in files {
        remove_if_exists(sidecar)?;
    }
    Ok(DownloadOutcome::Completed {
        path: reqs[0].dest.clone(),
        sha256: fingerprint(verified),
    })
}

/// Moves a promoted file back to its `.part` name. If that is impossible (for example the
/// bytes were copied across volumes), the promoted file is removed instead.
fn restore_part(dest: &Path, part: &Path) {
    if fs::rename(dest, part).is_err() {
        let _ = fs::remove_file(dest);
    }
}

fn discard_all(files: &[(PathBuf, PathBuf)]) -> Result<()> {
    for (part, sidecar) in files {
        discard_files(part, sidecar)?;
    }
    Ok(())
}

/// One streaming attempt from the bytes already on disk.
async fn stream_once(
    inner: &Inner,
    req: &DownloadRequest,
    part: &Path,
    progress: &watch::Sender<DownloadProgress>,
    control: &mut watch::Receiver<Control>,
    scope: Scope,
) -> Result<StreamEnd> {
    let offset = part_len(part)?;
    let mut resp = inner.fetch.open(&req.url, offset).await?;
    if let Some(total) = resp.total_len {
        if total != req.expected_size {
            return Err(RaptorError::Integrity(format!(
                "server reports {total} bytes but the listing said {}",
                req.expected_size
            )));
        }
    }
    let (mut file, mut written) = if offset > 0 && resp.resumed {
        (OpenOptions::new().append(true).open(part)?, offset)
    } else {
        // The server ignored the Range header (or this is a fresh start): begin again.
        (File::create(part)?, 0)
    };
    loop {
        tokio::select! {
            biased;
            c = next_control(control) => match c {
                Control::Run => {}
                Control::Pause => {
                    file.sync_data()?;
                    return Ok(StreamEnd::Paused);
                }
                Control::Cancel => return Ok(StreamEnd::Cancelled),
            },
            chunk = resp.body.next() => match chunk {
                None => break,
                Some(chunk) => {
                    let bytes = chunk?;
                    let next_len = written + bytes.len() as u64;
                    if next_len > req.expected_size {
                        return Err(RaptorError::Integrity(
                            "server sent more data than the listed size".into(),
                        ));
                    }
                    file.write_all(&bytes)?;
                    written = next_len;
                    progress.send_modify(|p| p.downloaded = scope.base + written);
                }
            }
        }
    }
    file.sync_data()?;
    if written != req.expected_size {
        return Err(RaptorError::Network(format!(
            "connection closed at {written} of {} bytes",
            req.expected_size
        )));
    }
    Ok(StreamEnd::Complete)
}

/// Checks free space before a transfer. `items` pairs each request with the bytes it still
/// has to write. Those bytes land on the downloads volume. A destination on another volume
/// receives whole files at promotion (a copy), so that volume must hold every cross-volume
/// file in full. The cross-volume check uses the sum over all such files, which is
/// conservative when a set spans several volumes.
fn preflight(inner: &Inner, items: &[(&DownloadRequest, u64)]) -> Result<()> {
    let margin = inner.cfg.min_free_margin;
    let dl = inner.cfg.downloads_dir.as_path();
    let needed: u64 = items
        .iter()
        .map(|(_, rem)| *rem)
        .sum::<u64>()
        .saturating_add(margin);
    check_free(inner, dl, needed)?;

    let cross: Vec<&DownloadRequest> = items
        .iter()
        .map(|(r, _)| *r)
        .filter(|r| {
            let dest_dir = r.dest.parent().unwrap_or(dl);
            !inner.disk.same_volume(dl, dest_dir)
        })
        .collect();
    if let Some(first) = cross.first() {
        let full: u64 = cross
            .iter()
            .map(|r| r.expected_size)
            .sum::<u64>()
            .saturating_add(margin);
        let dest_dir = first.dest.parent().unwrap_or(dl);
        check_free(inner, dest_dir, full)?;
    }
    Ok(())
}

fn check_free(inner: &Inner, dir: &Path, needed: u64) -> Result<()> {
    let available = inner.disk.available_bytes(dir)?;
    if available < needed {
        return Err(RaptorError::Io(format!(
            "not enough free disk space on {}: need {needed} bytes, have {available}",
            dir.display()
        )));
    }
    Ok(())
}

/// Moves a verified `.part` to its final path. Falls back to copy-then-rename inside the
/// destination directory when the two live on different volumes.
fn promote(part: &Path, dest: &Path) -> Result<()> {
    if fs::rename(part, dest).is_ok() {
        return Ok(());
    }
    let mut tmp_name = dest.as_os_str().to_owned();
    tmp_name.push(".importing");
    let tmp = PathBuf::from(tmp_name);
    let copied = fs::copy(part, &tmp).and_then(|_| fs::rename(&tmp, dest));
    if let Err(e) = copied {
        let _ = fs::remove_file(&tmp);
        return Err(e.into());
    }
    remove_if_exists(part)
}

/// Makes the `.part` file consistent with `meta`. A matching record keeps the bytes for
/// resume; anything else (missing, different, corrupt record, orphan bytes) starts clean.
fn prepare_partial(part: &Path, sidecar: &Path, meta: &JobMeta) -> Result<()> {
    match read_sidecar(sidecar) {
        Ok(Some(existing)) if existing == *meta => {}
        Ok(_) => discard_files(part, sidecar)?,
        Err(e) => {
            tracing::warn!("discarding unreadable download record: {e}");
            discard_files(part, sidecar)?;
        }
    }
    let json = serde_json::to_vec_pretty(meta)?;
    write_atomic(sidecar, &json)
}

fn read_sidecar(path: &Path) -> Result<Option<JobMeta>> {
    match fs::read(path) {
        Ok(bytes) => serde_json::from_slice(&bytes)
            .map(Some)
            .map_err(|e| RaptorError::Integrity(format!("record is corrupt: {e}"))),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.into()),
    }
}

fn part_len(part: &Path) -> Result<u64> {
    match fs::metadata(part) {
        Ok(m) => Ok(m.len()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(0),
        Err(e) => Err(e.into()),
    }
}

fn discard_files(part: &Path, sidecar: &Path) -> Result<()> {
    remove_if_exists(part)?;
    remove_if_exists(sidecar)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{
        gguf_file, sha256_hex, AcceptAll, FakeFetch, Fault, FixedDisk, RefuseAll, RoutedFetch,
        SplitVolumeDisk,
    };
    use tempfile::TempDir;

    const KEY: &str = "owner--repo--model-Q4_K_M.gguf";

    struct Env {
        _tmp: TempDir,
        downloads: PathBuf,
        models: PathBuf,
    }

    fn env() -> Env {
        let tmp = tempfile::tempdir().unwrap();
        let downloads = tmp.path().join("downloads");
        let models = tmp.path().join("models");
        Env {
            _tmp: tmp,
            downloads,
            models,
        }
    }

    fn cfg(env: &Env) -> DownloadConfig {
        DownloadConfig {
            downloads_dir: env.downloads.clone(),
            max_concurrent: 2,
            min_free_margin: 0,
            max_attempts: 3,
            retry_backoff: Duration::ZERO,
        }
    }

    fn pattern(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i % 251) as u8).collect()
    }

    fn request(env: &Env, body: &[u8], verify_architecture: bool) -> DownloadRequest {
        DownloadRequest {
            repo: "owner/repo".into(),
            file_name: "model-Q4_K_M.gguf".into(),
            url: "https://fake.hf.test/owner/repo/resolve/main/model-Q4_K_M.gguf".into(),
            expected_sha256: sha256_hex(body),
            expected_size: body.len() as u64,
            dest: env
                .models
                .join("owner")
                .join("repo")
                .join("model-Q4_K_M.gguf"),
            overwrite: false,
            source: ModelSource::HuggingFace {
                repo: "owner/repo".into(),
                file_name: "model-Q4_K_M.gguf".into(),
            },
            verify_architecture,
        }
    }

    fn manager(
        env: &Env,
        fetch: Arc<FakeFetch>,
        registrar: Arc<dyn Registrar>,
        disk: u64,
    ) -> DownloadManager {
        DownloadManager::new(cfg(env), fetch, Arc::new(FixedDisk(disk)), registrar)
    }

    fn part_path(env: &Env) -> PathBuf {
        env.downloads.join(format!("{KEY}.part"))
    }

    fn sidecar_path(env: &Env) -> PathBuf {
        env.downloads.join(format!("{KEY}.part.json"))
    }

    #[tokio::test]
    async fn completes_and_promotes_verified_file() {
        let env = env();
        let body = pattern(5000);
        let fetch = Arc::new(FakeFetch::new(body.clone()));
        let mgr = manager(&env, fetch.clone(), Arc::new(AcceptAll), u64::MAX);
        let handle = mgr.start(request(&env, &body, false)).unwrap();
        let outcome = handle.finish().await.unwrap();
        match outcome {
            DownloadOutcome::Completed { path, sha256 } => {
                assert_eq!(
                    path,
                    env.models
                        .join("owner")
                        .join("repo")
                        .join("model-Q4_K_M.gguf")
                );
                assert_eq!(sha256, sha256_hex(&body));
            }
            other => panic!("unexpected {other:?}"),
        }
        assert_eq!(
            fs::read(
                env.models
                    .join("owner")
                    .join("repo")
                    .join("model-Q4_K_M.gguf")
            )
            .unwrap(),
            body
        );
        assert!(!part_path(&env).exists());
        assert!(!sidecar_path(&env).exists());
        assert_eq!(fetch.opens(), vec![0]);
    }

    #[tokio::test]
    async fn resumes_with_range_after_mid_stream_failure() {
        let env = env();
        let body = pattern(5000);
        let fetch =
            Arc::new(FakeFetch::new(body.clone()).with_faults(vec![Fault::FailAfter(2000)]));
        let mgr = manager(&env, fetch.clone(), Arc::new(AcceptAll), u64::MAX);
        let outcome = mgr
            .start(request(&env, &body, false))
            .unwrap()
            .finish()
            .await
            .unwrap();
        assert!(matches!(outcome, DownloadOutcome::Completed { .. }));
        assert_eq!(
            fetch.opens(),
            vec![0, 2000],
            "second attempt must resume at the byte it stopped"
        );
        assert_eq!(
            fs::read(env.models.join("owner/repo/model-Q4_K_M.gguf")).unwrap(),
            body
        );
    }

    #[tokio::test]
    async fn restarts_cleanly_when_server_ignores_range() {
        let env = env();
        let body = pattern(4000);
        let fetch = Arc::new(
            FakeFetch::new(body.clone())
                .with_faults(vec![Fault::FailAfter(1500)])
                .ignoring_range(),
        );
        let mgr = manager(&env, fetch.clone(), Arc::new(AcceptAll), u64::MAX);
        let outcome = mgr
            .start(request(&env, &body, false))
            .unwrap()
            .finish()
            .await
            .unwrap();
        assert!(matches!(outcome, DownloadOutcome::Completed { .. }));
        assert_eq!(fetch.opens(), vec![0, 1500]);
        assert_eq!(
            fs::read(env.models.join("owner/repo/model-Q4_K_M.gguf")).unwrap(),
            body
        );
    }

    #[tokio::test]
    async fn hash_mismatch_deletes_everything_and_promotes_nothing() {
        let env = env();
        let body = pattern(3000);
        let fetch = Arc::new(FakeFetch::new(body.clone()));
        let mgr = manager(&env, fetch, Arc::new(AcceptAll), u64::MAX);
        let mut req = request(&env, &body, false);
        req.expected_sha256 = sha256_hex(b"something else entirely");
        match mgr.start(req).unwrap().finish().await {
            Err(RaptorError::Integrity(m)) => assert!(m.contains("sha256 mismatch")),
            other => panic!("expected integrity failure, got {other:?}"),
        }
        assert!(!env.models.join("owner/repo/model-Q4_K_M.gguf").exists());
        assert!(!part_path(&env).exists());
        assert!(!sidecar_path(&env).exists());
    }

    #[tokio::test]
    async fn server_size_disagreement_is_integrity() {
        let env = env();
        let body = pattern(3000);
        let fetch = Arc::new(FakeFetch::new(body.clone()));
        let mgr = manager(&env, fetch, Arc::new(AcceptAll), u64::MAX);
        let mut req = request(&env, &body, false);
        req.expected_size = 2999;
        assert!(matches!(
            mgr.start(req).unwrap().finish().await,
            Err(RaptorError::Integrity(_))
        ));
        assert!(!part_path(&env).exists());
    }

    #[tokio::test]
    async fn pause_keeps_bytes_and_resume_continues_from_them() {
        let env = env();
        let body = pattern(6000);
        let fetch =
            Arc::new(FakeFetch::new(body.clone()).with_faults(vec![Fault::StallAfter(3000)]));
        let mgr = manager(&env, fetch.clone(), Arc::new(AcceptAll), u64::MAX);
        let handle = mgr.start(request(&env, &body, false)).unwrap();
        let mut progress = handle.progress();
        progress.wait_for(|p| p.downloaded >= 3000).await.unwrap();
        handle.pause().unwrap();
        progress
            .wait_for(|p| p.state == JobState::Paused)
            .await
            .unwrap();
        assert_eq!(fs::metadata(part_path(&env)).unwrap().len(), 3000);
        handle.resume().unwrap();
        let outcome = handle.finish().await.unwrap();
        assert!(matches!(outcome, DownloadOutcome::Completed { .. }));
        assert_eq!(fetch.opens(), vec![0, 3000]);
        assert_eq!(
            fs::read(env.models.join("owner/repo/model-Q4_K_M.gguf")).unwrap(),
            body
        );
    }

    #[tokio::test]
    async fn cancel_deletes_partial_bytes() {
        let env = env();
        let body = pattern(6000);
        let fetch =
            Arc::new(FakeFetch::new(body.clone()).with_faults(vec![Fault::StallAfter(2000)]));
        let mgr = manager(&env, fetch, Arc::new(AcceptAll), u64::MAX);
        let handle = mgr.start(request(&env, &body, false)).unwrap();
        let mut progress = handle.progress();
        progress.wait_for(|p| p.downloaded >= 2000).await.unwrap();
        handle.cancel().unwrap();
        assert_eq!(handle.finish().await.unwrap(), DownloadOutcome::Cancelled);
        assert!(!part_path(&env).exists());
        assert!(!sidecar_path(&env).exists());
        assert!(!env.models.join("owner/repo/model-Q4_K_M.gguf").exists());
    }

    #[tokio::test]
    async fn exhausted_retries_keep_partial_for_recovery() {
        let env = env();
        let body = pattern(5000);
        let fetch = Arc::new(FakeFetch::new(body.clone()).with_faults(vec![
            Fault::FailAfter(1000),
            Fault::FailAfter(1000),
            Fault::FailAfter(1000),
        ]));
        let mgr = manager(&env, fetch.clone(), Arc::new(AcceptAll), u64::MAX);
        let err = mgr
            .start(request(&env, &body, false))
            .unwrap()
            .finish()
            .await
            .unwrap_err();
        assert!(matches!(err, RaptorError::Network(_)));
        assert_eq!(fetch.opens(), vec![0, 1000, 2000]);
        assert_eq!(fs::metadata(part_path(&env)).unwrap().len(), 3000);
        assert!(sidecar_path(&env).exists());
    }

    #[tokio::test]
    async fn recovered_partial_resumes_after_restart() {
        let env = env();
        let body = pattern(5000);
        let first = Arc::new(FakeFetch::new(body.clone()).with_faults(vec![
            Fault::FailAfter(1000),
            Fault::FailAfter(1000),
            Fault::FailAfter(1000),
        ]));
        let mgr = manager(&env, first, Arc::new(AcceptAll), u64::MAX);
        let _ = mgr
            .start(request(&env, &body, false))
            .unwrap()
            .finish()
            .await;

        // Simulate an app restart: a brand new manager over the same downloads directory.
        let report = manager(
            &env,
            Arc::new(FakeFetch::new(body.clone())),
            Arc::new(AcceptAll),
            u64::MAX,
        )
        .recover()
        .unwrap();
        assert_eq!(report.partials.len(), 1);
        assert_eq!(report.partials[0].downloaded_bytes, 3000);
        assert_eq!(report.partials[0].file_name, "model-Q4_K_M.gguf");
        assert!(report.problems.is_empty());

        let second = Arc::new(FakeFetch::new(body.clone()));
        let mgr2 = manager(&env, second.clone(), Arc::new(AcceptAll), u64::MAX);
        let outcome = mgr2
            .start(request(&env, &body, false))
            .unwrap()
            .finish()
            .await
            .unwrap();
        assert!(matches!(outcome, DownloadOutcome::Completed { .. }));
        assert_eq!(second.opens(), vec![3000]);
        assert_eq!(
            fs::read(env.models.join("owner/repo/model-Q4_K_M.gguf")).unwrap(),
            body
        );
    }

    #[tokio::test]
    async fn stale_partial_from_other_version_is_never_resumed() {
        let env = env();
        let old = pattern(4000);
        let first = Arc::new(
            FakeFetch::new(old.clone())
                .with_faults(vec![Fault::FailAfter(1000)])
                .ignoring_range(),
        );
        let mgr = DownloadManager::new(
            DownloadConfig {
                max_attempts: 1,
                ..cfg(&env)
            },
            first,
            Arc::new(FixedDisk(u64::MAX)),
            Arc::new(AcceptAll),
        );
        let _ = mgr
            .start(request(&env, &old, false))
            .unwrap()
            .finish()
            .await;
        assert_eq!(fs::metadata(part_path(&env)).unwrap().len(), 1000);

        let new = pattern(4000)
            .into_iter()
            .map(|b| b ^ 0x5a)
            .collect::<Vec<u8>>();
        let fetch = Arc::new(FakeFetch::new(new.clone()));
        let mgr2 = manager(&env, fetch.clone(), Arc::new(AcceptAll), u64::MAX);
        let outcome = mgr2
            .start(request(&env, &new, false))
            .unwrap()
            .finish()
            .await
            .unwrap();
        assert!(matches!(outcome, DownloadOutcome::Completed { .. }));
        assert_eq!(
            fetch.opens(),
            vec![0],
            "a record for a different sha must start from zero"
        );
        assert_eq!(
            fs::read(env.models.join("owner/repo/model-Q4_K_M.gguf")).unwrap(),
            new
        );
    }

    #[tokio::test]
    async fn existing_destination_is_conflict_without_network() {
        let env = env();
        let body = pattern(1000);
        let dest = env.models.join("owner/repo/model-Q4_K_M.gguf");
        fs::create_dir_all(dest.parent().unwrap()).unwrap();
        fs::write(&dest, b"user's existing model").unwrap();
        let fetch = Arc::new(FakeFetch::new(body.clone()));
        let mgr = manager(&env, fetch.clone(), Arc::new(AcceptAll), u64::MAX);
        assert!(matches!(
            mgr.start(request(&env, &body, false)),
            Err(RaptorError::Conflict(_))
        ));
        assert!(fetch.opens().is_empty());
        assert_eq!(fs::read(&dest).unwrap(), b"user's existing model");
    }

    #[tokio::test]
    async fn overwrite_replaces_existing_destination() {
        let env = env();
        let body = pattern(1000);
        let dest = env.models.join("owner/repo/model-Q4_K_M.gguf");
        fs::create_dir_all(dest.parent().unwrap()).unwrap();
        fs::write(&dest, b"old").unwrap();
        let mgr = manager(
            &env,
            Arc::new(FakeFetch::new(body.clone())),
            Arc::new(AcceptAll),
            u64::MAX,
        );
        let mut req = request(&env, &body, false);
        req.overwrite = true;
        mgr.start(req).unwrap().finish().await.unwrap();
        assert_eq!(fs::read(&dest).unwrap(), body);
    }

    #[tokio::test]
    async fn second_job_for_same_file_is_conflict_until_first_ends() {
        let env = env();
        let body = pattern(4000);
        let fetch =
            Arc::new(FakeFetch::new(body.clone()).with_faults(vec![Fault::StallAfter(1000)]));
        let mgr = manager(&env, fetch, Arc::new(AcceptAll), u64::MAX);
        let first = mgr.start(request(&env, &body, false)).unwrap();
        let mut progress = first.progress();
        progress.wait_for(|p| p.downloaded >= 1000).await.unwrap();
        assert!(matches!(
            mgr.start(request(&env, &body, false)),
            Err(RaptorError::Conflict(_))
        ));
        assert!(matches!(
            mgr.discard_partial(KEY),
            Err(RaptorError::Conflict(_))
        ));
        first.cancel().unwrap();
        assert_eq!(first.finish().await.unwrap(), DownloadOutcome::Cancelled);
        // Once the first job ends, the same request is accepted again.
        assert!(mgr.start(request(&env, &body, false)).is_ok());
    }

    #[tokio::test]
    async fn list_jobs_reports_live_and_finished_jobs_with_byte_counts() {
        let env = env();
        let body = pattern(4000);
        let fetch =
            Arc::new(FakeFetch::new(body.clone()).with_faults(vec![Fault::StallAfter(1000)]));
        let mgr = manager(&env, fetch, Arc::new(AcceptAll), u64::MAX);
        assert!(mgr.list_jobs().unwrap().is_empty());

        let first = mgr.start(request(&env, &body, false)).unwrap();
        let mut progress = first.progress();
        progress.wait_for(|p| p.downloaded >= 1000).await.unwrap();
        let live = mgr.list_jobs().unwrap();
        assert_eq!(live.len(), 1);
        assert_eq!(live[0].key, KEY);
        assert_eq!(live[0].repo, "owner/repo");
        assert_eq!(live[0].file_name, "model-Q4_K_M.gguf");
        assert_eq!(live[0].state, JobState::Running);
        assert!(live[0].downloaded >= 1000);
        assert_eq!(live[0].total, body.len() as u64);
        let json = serde_json::to_value(&live[0]).unwrap();
        assert_eq!(json["state"], "running");
        assert_eq!(json["file_name"], "model-Q4_K_M.gguf");

        first.cancel().unwrap();
        first.finish().await.unwrap();
        let ended = mgr.list_jobs().unwrap();
        assert_eq!(ended.len(), 1, "finished jobs stay listed");
        assert_eq!(ended[0].state, JobState::Cancelled);
        assert_eq!(ended[0].key, KEY);
    }

    #[tokio::test]
    async fn path_traversal_in_file_name_is_rejected_before_anything_runs() {
        let env = env();
        let body = pattern(100);
        let fetch = Arc::new(FakeFetch::new(body.clone()));
        let mgr = manager(&env, fetch.clone(), Arc::new(AcceptAll), u64::MAX);
        let mut req = request(&env, &body, false);
        req.file_name = "../../evil.gguf".into();
        assert!(matches!(mgr.start(req), Err(RaptorError::InvalidInput(_))));
        let mut req = request(&env, &body, false);
        req.file_name = "a/../evil.gguf".into();
        assert!(matches!(mgr.start(req), Err(RaptorError::InvalidInput(_))));
        let mut req = request(&env, &body, false);
        req.file_name = "not-a-model.bin".into();
        assert!(matches!(mgr.start(req), Err(RaptorError::InvalidInput(_))));
        let mut req = request(&env, &body, false);
        req.dest = env.models.join("elsewhere.gguf");
        assert!(matches!(mgr.start(req), Err(RaptorError::InvalidInput(_))));
        let mut req = request(&env, &body, false);
        req.url = "http://plain.example/x.gguf".into();
        assert!(matches!(mgr.start(req), Err(RaptorError::InvalidInput(_))));
        let mut req = request(&env, &body, false);
        req.expected_sha256 = "zz".into();
        assert!(matches!(mgr.start(req), Err(RaptorError::InvalidInput(_))));
        assert!(fetch.opens().is_empty());
        assert!(!env.downloads.join("..").join("evil.gguf").exists());
    }

    #[tokio::test]
    async fn insufficient_disk_space_fails_before_any_transfer() {
        let env = env();
        let body = pattern(5000);
        let fetch = Arc::new(FakeFetch::new(body.clone()));
        let cfg = DownloadConfig {
            min_free_margin: 1_000_000,
            ..cfg(&env)
        };
        let mgr = DownloadManager::new(
            cfg,
            fetch.clone(),
            Arc::new(FixedDisk(10_000)),
            Arc::new(AcceptAll),
        );
        match mgr
            .start(request(&env, &body, false))
            .unwrap()
            .finish()
            .await
        {
            Err(RaptorError::Io(m)) => assert!(m.contains("not enough free disk space")),
            other => panic!("expected disk space error, got {other:?}"),
        }
        assert!(fetch.opens().is_empty());
    }

    #[tokio::test]
    async fn known_unsupported_architecture_is_rejected_from_the_first_bytes() {
        let env = env();
        let body = gguf_file("clip", 15);
        let fetch = Arc::new(FakeFetch::new(body.clone()));
        let mgr = manager(&env, fetch, Arc::new(AcceptAll), u64::MAX);
        match mgr
            .start(request(&env, &body, true))
            .unwrap()
            .finish()
            .await
        {
            Err(RaptorError::Unsupported(m)) => assert!(m.contains("clip")),
            other => panic!("expected unsupported, got {other:?}"),
        }
        assert!(!part_path(&env).exists());
        assert!(!env.models.join("owner/repo/model-Q4_K_M.gguf").exists());
    }

    #[tokio::test]
    async fn unknown_architecture_is_not_blocked_by_the_download_check() {
        let env = env();
        let body = gguf_file("made-up-arch", 15);
        let mgr = manager(
            &env,
            Arc::new(FakeFetch::new(body.clone())),
            Arc::new(AcceptAll),
            u64::MAX,
        );
        let outcome = mgr
            .start(request(&env, &body, true))
            .unwrap()
            .finish()
            .await
            .unwrap();
        assert!(matches!(outcome, DownloadOutcome::Completed { .. }));
    }

    #[tokio::test]
    async fn supported_architecture_downloads_with_check_enabled() {
        let env = env();
        let body = gguf_file("llama", 15);
        let mgr = manager(
            &env,
            Arc::new(FakeFetch::new(body.clone())),
            Arc::new(AcceptAll),
            u64::MAX,
        );
        let outcome = mgr
            .start(request(&env, &body, true))
            .unwrap()
            .finish()
            .await
            .unwrap();
        assert!(matches!(outcome, DownloadOutcome::Completed { .. }));
    }

    #[tokio::test]
    async fn corrupt_gguf_fails_the_architecture_check() {
        let env = env();
        let body = b"<html>this is not a model file at all</html>".to_vec();
        let mgr = manager(
            &env,
            Arc::new(FakeFetch::new(body.clone())),
            Arc::new(AcceptAll),
            u64::MAX,
        );
        assert!(matches!(
            mgr.start(request(&env, &body, true))
                .unwrap()
                .finish()
                .await,
            Err(RaptorError::Integrity(_))
        ));
        assert!(!part_path(&env).exists());
    }

    #[tokio::test]
    async fn registrar_refusal_removes_the_promoted_file() {
        let env = env();
        let body = pattern(2000);
        let mgr = manager(
            &env,
            Arc::new(FakeFetch::new(body.clone())),
            Arc::new(RefuseAll),
            u64::MAX,
        );
        assert!(matches!(
            mgr.start(request(&env, &body, false))
                .unwrap()
                .finish()
                .await,
            Err(RaptorError::Conflict(_))
        ));
        assert!(!env.models.join("owner/repo/model-Q4_K_M.gguf").exists());
    }

    #[tokio::test]
    async fn recovery_reports_orphans_and_corrupt_records() {
        let env = env();
        fs::create_dir_all(&env.downloads).unwrap();
        fs::write(env.downloads.join("orphan--x.gguf.part"), b"bytes").unwrap();
        fs::write(env.downloads.join("bad--y.gguf.part.json"), b"{not json").unwrap();
        let mgr = manager(
            &env,
            Arc::new(FakeFetch::new(vec![])),
            Arc::new(AcceptAll),
            u64::MAX,
        );
        let report = mgr.recover().unwrap();
        assert!(report.partials.is_empty());
        assert_eq!(report.problems.len(), 2, "{:?}", report.problems);
    }

    #[tokio::test]
    async fn discard_partial_removes_interrupted_files() {
        let env = env();
        let body = pattern(4000);
        let fetch = Arc::new(FakeFetch::new(body.clone()).with_faults(vec![Fault::FailAfter(500)]));
        let cfg = DownloadConfig {
            max_attempts: 1,
            ..cfg(&env)
        };
        let mgr = DownloadManager::new(
            cfg,
            fetch,
            Arc::new(FixedDisk(u64::MAX)),
            Arc::new(AcceptAll),
        );
        let _ = mgr
            .start(request(&env, &body, false))
            .unwrap()
            .finish()
            .await;
        assert!(part_path(&env).exists());
        mgr.discard_partial(KEY).unwrap();
        assert!(!part_path(&env).exists());
        assert!(!sidecar_path(&env).exists());
        assert!(mgr.discard_partial("../escape").is_err());
    }

    #[tokio::test]
    async fn control_of_unknown_job_is_not_found() {
        let env = env();
        let mgr = manager(
            &env,
            Arc::new(FakeFetch::new(vec![])),
            Arc::new(AcceptAll),
            u64::MAX,
        );
        assert!(matches!(mgr.pause("nope"), Err(RaptorError::NotFound(_))));
        assert!(matches!(mgr.cancel("nope"), Err(RaptorError::NotFound(_))));
    }

    #[test]
    fn job_key_is_stable_and_validated() {
        assert_eq!(
            job_key("owner/repo", "a.gguf").unwrap(),
            "owner--repo--a.gguf"
        );
        assert_eq!(
            job_key("owner/repo", "Q4/a.gguf").unwrap(),
            "owner--repo--Q4%2Fa.gguf"
        );
        assert!(job_key("owner/../repo", "a.gguf").is_err());
        assert!(job_key("owner/repo", "../a.gguf").is_err());
    }

    #[test]
    fn promote_moves_file_and_keeps_name() {
        let tmp = tempfile::tempdir().unwrap();
        let part = tmp.path().join("p.part");
        let dest = tmp.path().join("final").join("m.gguf");
        fs::create_dir_all(dest.parent().unwrap()).unwrap();
        fs::write(&part, b"data").unwrap();
        promote(&part, &dest).unwrap();
        assert!(!part.exists());
        assert_eq!(fs::read(&dest).unwrap(), b"data");
    }

    #[test]
    fn remote_header_probe_reads_only_a_prefix() {
        let body = gguf_file("qwen2", 15);
        let fetch = FakeFetch::new(body);
        let probe = futures_block_on(probe_remote_header(&fetch, "https://x.test/m.gguf")).unwrap();
        match probe {
            HeaderProbe::Complete(meta) => assert_eq!(meta.architecture.as_deref(), Some("qwen2")),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn same_volume_is_true_for_a_temp_dir_and_its_child() {
        let tmp = tempfile::tempdir().unwrap();
        let child = tmp.path().join("not-yet-created").join("deeper");
        assert!(same_volume(tmp.path(), &child));
    }

    fn set_request(env: &Env, name: &str, body: &[u8], index: usize) -> DownloadRequest {
        DownloadRequest {
            repo: "owner/repo".into(),
            file_name: format!("Q4/{name}-0000{index}-of-00002.gguf"),
            url: format!(
                "https://fake.hf.test/owner/repo/resolve/sha/Q4/{name}-0000{index}-of-00002.gguf"
            ),
            expected_sha256: sha256_hex(body),
            expected_size: body.len() as u64,
            dest: env
                .models
                .join("owner")
                .join("repo")
                .join("Q4")
                .join(format!("{name}-0000{index}-of-00002.gguf")),
            overwrite: false,
            source: ModelSource::HuggingFace {
                repo: "owner/repo".into(),
                file_name: format!("Q4/{name}-0000{index}-of-00002.gguf"),
            },
            verify_architecture: false,
        }
    }

    fn routed(bodies: &[(&str, Vec<u8>)]) -> Arc<RoutedFetch> {
        let mut fetch = RoutedFetch::new();
        for (url, body) in bodies {
            fetch.route(url, body.clone());
        }
        Arc::new(fetch)
    }

    fn set_urls() -> [&'static str; 2] {
        [
            "https://fake.hf.test/owner/repo/resolve/sha/Q4/s-00001-of-00002.gguf",
            "https://fake.hf.test/owner/repo/resolve/sha/Q4/s-00002-of-00002.gguf",
        ]
    }

    #[tokio::test]
    async fn shard_set_downloads_as_one_job_and_registers_once() {
        let env = env();
        let a = pattern(3000);
        let b = pattern(2500);
        let urls = set_urls();
        let fetch = routed(&[(urls[0], a.clone()), (urls[1], b.clone())]);
        let registrar = Arc::new(crate::testing::RecordingRegistrar::default());
        let mgr = DownloadManager::new(
            cfg(&env),
            fetch,
            Arc::new(FixedDisk(u64::MAX)),
            registrar.clone(),
        );
        let reqs = vec![set_request(&env, "s", &a, 1), set_request(&env, "s", &b, 2)];
        let handle = mgr.start_set(reqs, Provenance::default()).unwrap();
        let outcome = handle.finish().await.unwrap();
        let DownloadOutcome::Completed { path, sha256 } = outcome else {
            panic!("expected completion");
        };
        assert_eq!(path, env.models.join("owner/repo/Q4/s-00001-of-00002.gguf"));
        assert_eq!(
            sha256,
            fingerprint(&[sha256_hex(&a), sha256_hex(&b)]),
            "set fingerprint is over the ordered shard digests"
        );
        assert_eq!(fs::read(&path).unwrap(), a);
        assert_eq!(
            fs::read(env.models.join("owner/repo/Q4/s-00002-of-00002.gguf")).unwrap(),
            b
        );
        assert_eq!(registrar.set_calls(), 1);
        assert_eq!(registrar.single_calls(), 0);
        assert!(!env
            .downloads
            .join(format!(
                "{}.part",
                "owner--repo--Q4%2Fs-00001-of-00002.gguf"
            ))
            .exists());
    }

    #[tokio::test]
    async fn shard_set_is_all_or_nothing_on_a_bad_shard() {
        let env = env();
        let a = pattern(3000);
        let b = pattern(2500);
        let urls = set_urls();
        let fetch = routed(&[(urls[0], a.clone()), (urls[1], b.clone())]);
        let mgr = DownloadManager::new(
            cfg(&env),
            fetch,
            Arc::new(FixedDisk(u64::MAX)),
            Arc::new(AcceptAll),
        );
        let mut reqs = vec![set_request(&env, "s", &a, 1), set_request(&env, "s", &b, 2)];
        reqs[1].expected_sha256 = sha256_hex(b"not the bytes served");
        let err = mgr
            .start_set(reqs, Provenance::default())
            .unwrap()
            .finish()
            .await
            .unwrap_err();
        assert!(matches!(err, RaptorError::Integrity(_)));
        assert!(!env
            .models
            .join("owner/repo/Q4/s-00001-of-00002.gguf")
            .exists());
        assert!(!env
            .models
            .join("owner/repo/Q4/s-00002-of-00002.gguf")
            .exists());
    }

    #[tokio::test]
    async fn shard_set_registration_failure_restores_parts_for_resume() {
        let env = env();
        let a = pattern(3000);
        let b = pattern(2500);
        let urls = set_urls();
        let fetch = routed(&[(urls[0], a.clone()), (urls[1], b.clone())]);
        let mgr = DownloadManager::new(
            cfg(&env),
            fetch,
            Arc::new(FixedDisk(u64::MAX)),
            Arc::new(RefuseAll),
        );
        let reqs = vec![set_request(&env, "s", &a, 1), set_request(&env, "s", &b, 2)];
        let err = mgr
            .start_set(reqs, Provenance::default())
            .unwrap()
            .finish()
            .await
            .unwrap_err();
        assert!(matches!(err, RaptorError::Conflict(_)));
        assert!(!env
            .models
            .join("owner/repo/Q4/s-00001-of-00002.gguf")
            .exists());
        assert!(!env
            .models
            .join("owner/repo/Q4/s-00002-of-00002.gguf")
            .exists());
        let kept = env
            .downloads
            .join("owner--repo--Q4%2Fs-00001-of-00002.gguf.part");
        assert_eq!(
            fs::read(&kept).unwrap(),
            a,
            "verified shard is kept for a later resume"
        );
        assert!(env
            .downloads
            .join("owner--repo--Q4%2Fs-00001-of-00002.gguf.part.json")
            .exists());
    }

    #[tokio::test]
    async fn shard_set_cancel_discards_every_shard() {
        let env = env();
        let a = pattern(3000);
        let b = pattern(2500);
        let urls = set_urls();
        let stalling =
            Arc::new(FakeFetch::new(a.clone()).with_faults(vec![Fault::StallAfter(1000)]));
        let mut fetch = RoutedFetch::new();
        fetch.route_fake(urls[0], stalling);
        fetch.route(urls[1], b.clone());
        let mgr = DownloadManager::new(
            cfg(&env),
            Arc::new(fetch),
            Arc::new(FixedDisk(u64::MAX)),
            Arc::new(AcceptAll),
        );
        let reqs = vec![set_request(&env, "s", &a, 1), set_request(&env, "s", &b, 2)];
        let handle = mgr.start_set(reqs, Provenance::default()).unwrap();
        let mut progress = handle.progress();
        progress.wait_for(|p| p.downloaded >= 1000).await.unwrap();
        handle.cancel().unwrap();
        assert_eq!(handle.finish().await.unwrap(), DownloadOutcome::Cancelled);
        let leftovers = fs::read_dir(&env.downloads).map(|d| d.count()).unwrap_or(0);
        assert_eq!(leftovers, 0, "cancel removes every shard's partial state");
    }

    #[tokio::test]
    async fn shard_set_progress_spans_all_shards() {
        let env = env();
        let a = pattern(3000);
        let b = pattern(2000);
        let urls = set_urls();
        let fetch = routed(&[(urls[0], a.clone()), (urls[1], b.clone())]);
        let mgr = DownloadManager::new(
            cfg(&env),
            fetch,
            Arc::new(FixedDisk(u64::MAX)),
            Arc::new(AcceptAll),
        );
        let reqs = vec![set_request(&env, "s", &a, 1), set_request(&env, "s", &b, 2)];
        let handle = mgr.start_set(reqs, Provenance::default()).unwrap();
        let progress = handle.progress();
        assert_eq!(progress.borrow().total, 5000);
        handle.finish().await.unwrap();
        let last = progress.borrow().clone();
        assert_eq!(last.downloaded, 5000, "second shard ends at the set total");
        assert_eq!(last.state, JobState::Completed);
    }

    #[tokio::test]
    async fn shard_set_refuses_duplicates_and_existing_destinations_before_transfer() {
        let env = env();
        let a = pattern(300);
        let fetch = Arc::new(FakeFetch::new(a.clone()));
        let mgr = DownloadManager::new(
            cfg(&env),
            fetch.clone(),
            Arc::new(FixedDisk(u64::MAX)),
            Arc::new(AcceptAll),
        );
        let dup = vec![set_request(&env, "s", &a, 1), set_request(&env, "s", &a, 1)];
        assert!(matches!(
            mgr.start_set(dup, Provenance::default()),
            Err(RaptorError::InvalidInput(_))
        ));
        let mixed_repo = {
            let mut r = set_request(&env, "s", &a, 2);
            r.repo = "other/repo".into();
            vec![set_request(&env, "s", &a, 1), r]
        };
        assert!(matches!(
            mgr.start_set(mixed_repo, Provenance::default()),
            Err(RaptorError::InvalidInput(_))
        ));
        let existing = env.models.join("owner/repo/Q4/s-00002-of-00002.gguf");
        fs::create_dir_all(existing.parent().unwrap()).unwrap();
        fs::write(&existing, b"keep me").unwrap();
        let reqs = vec![set_request(&env, "s", &a, 1), set_request(&env, "s", &a, 2)];
        assert!(matches!(
            mgr.start_set(reqs, Provenance::default()),
            Err(RaptorError::Conflict(_))
        ));
        assert!(fetch.opens().is_empty());
        assert_eq!(fs::read(&existing).unwrap(), b"keep me");
    }

    #[tokio::test]
    async fn cross_volume_preflight_needs_room_for_every_shard_in_full() {
        let a = pattern(3000);
        let b = pattern(2500);
        let total = (a.len() + b.len()) as u64;
        let urls = set_urls();
        for (dest_free, expect_ok) in [(total, true), (total - 1, false)] {
            let env = env();
            let disk = SplitVolumeDisk::new(env.downloads.clone(), total, dest_free);
            let mgr = DownloadManager::new(
                cfg(&env),
                routed(&[(urls[0], a.clone()), (urls[1], b.clone())]),
                Arc::new(disk),
                Arc::new(AcceptAll),
            );
            let result = mgr
                .start_set(
                    vec![set_request(&env, "s", &a, 1), set_request(&env, "s", &b, 2)],
                    Provenance::default(),
                )
                .unwrap()
                .finish()
                .await;
            if expect_ok {
                assert!(result.is_ok(), "{result:?}");
            } else {
                match result {
                    Err(RaptorError::Io(m)) => {
                        assert!(m.contains("not enough free disk space"));
                        assert!(m.contains("models"), "the destination volume is named: {m}");
                    }
                    other => panic!("expected disk space error, got {other:?}"),
                }
            }
        }
    }

    fn futures_block_on<F: std::future::Future>(f: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(f)
    }
}
