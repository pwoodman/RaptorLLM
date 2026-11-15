//! Hugging Face Hub access: model search, repo file listing, repo info and revision pinning.
//!
//! The parsers are pure functions over JSON bodies and are tested on fixtures. Network access
//! sits behind [`HfApi`], so tests never touch the internet. Every repo id, file path and
//! revision that comes from the Hub is validated before it is used in a URL or a file name.
//! Hub API calls retry a bounded number of times with backoff; file transfers have their own
//! resume logic in `download`.

use crate::quant::parse_quant_label;
use crate::shard::is_split_shard;
use crate::util::is_sha256_hex;
use async_trait::async_trait;
use raptor_core::paths::safe_file_name;
use raptor_core::{RaptorError, Result};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::future::Future;
use std::sync::Mutex;
use std::time::Duration;

pub const DEFAULT_BASE_URL: &str = "https://huggingface.co";
pub const MAX_SEARCH_LIMIT: u32 = 100;
/// Deepest subfolder accepted in a repo path (`a/b/c/file.gguf`).
pub const MAX_PATH_DEPTH: usize = 4;
const MAX_QUERY_CHARS: usize = 200;
/// Tree listing pages followed before giving up on a very large repo.
const MAX_TREE_PAGES: usize = 50;

/// Sort order for search results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchSort {
    Downloads,
    Likes,
    LastModified,
}

impl SearchSort {
    fn api_value(self) -> &'static str {
        match self {
            SearchSort::Downloads => "downloads",
            SearchSort::Likes => "likes",
            SearchSort::LastModified => "lastModified",
        }
    }
}

/// One search hit. Only GGUF repos are requested.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HfModelSummary {
    pub id: String,
    pub author: Option<String>,
    pub downloads: u64,
    pub likes: u64,
    pub tags: Vec<String>,
    pub last_modified: Option<String>,
    pub pipeline_tag: Option<String>,
}

/// A GGUF file in a repo, with the LFS sha256 used for verification.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepoFile {
    /// Repo-relative path, e.g. `model-Q4_K_M.gguf` or `Q4_K_M/model-00001-of-00002.gguf`.
    /// Every segment has passed `safe_file_name`.
    pub file_name: String,
    pub size: u64,
    /// Lowercase hex sha256 from the LFS pointer. `None` means it cannot be verified.
    pub sha256: Option<String>,
    /// Quant label parsed from the file's own name (not its folder).
    pub quant_label: Option<String>,
    /// True for `...-00001-of-00002.gguf` style shards. See `shard::detect_shard_sets`.
    pub split_shard: bool,
}

/// Basic repo metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HfRepoInfo {
    pub id: String,
    pub pipeline_tag: Option<String>,
    pub library_name: Option<String>,
    pub license: Option<String>,
    pub tags: Vec<String>,
    pub gated: bool,
}

/// Repo metadata together with the exact commit that `main` points to right now.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepoSnapshot {
    pub info: HfRepoInfo,
    /// Lowercase 40-character commit sha. Downloads are pinned to it.
    pub revision: String,
}

/// Rate-limit state reported by the Hub's `RateLimit` headers. Every field is optional
/// because the Hub does not always send them. Values are as reported; the format is parsed
/// from the documented structured-header style and has not been checked against live traffic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct HubRateLimit {
    /// Requests left in the current window.
    pub remaining: Option<u64>,
    /// Seconds until the window resets.
    pub reset_secs: Option<u64>,
    /// Requests allowed per window.
    pub quota: Option<u64>,
    /// Window length in seconds.
    pub window_secs: Option<u64>,
}

/// Bounded retry for Hub API calls. Delays double from `base_delay` up to `max_delay`.
/// A `Retry-After` longer than `max_delay` is not waited for; the error is returned instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetryPolicy {
    /// Total attempts, including the first. Values below 1 behave as 1.
    pub max_attempts: u32,
    pub base_delay: Duration,
    pub max_delay: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_delay: Duration::from_millis(750),
            max_delay: Duration::from_secs(8),
        }
    }
}

/// Outcome of one attempt inside [`run_with_retry`].
pub(crate) enum Attempt<T> {
    Done(T),
    /// Transient failure. `after` is the server's Retry-After, when it sent one.
    Retry {
        after: Option<Duration>,
        error: RaptorError,
    },
    /// Permanent failure; never retried.
    Fail(RaptorError),
}

/// Runs `op` until it succeeds, fails permanently, or the policy is exhausted. Dropping the
/// returned future cancels any pending backoff sleep, so callers can cancel a retry loop.
pub(crate) async fn run_with_retry<T, F, Fut>(policy: RetryPolicy, mut op: F) -> Result<T>
where
    F: FnMut(u32) -> Fut,
    Fut: Future<Output = Attempt<T>>,
{
    let max = policy.max_attempts.max(1);
    let mut attempt = 0u32;
    loop {
        match op(attempt).await {
            Attempt::Done(v) => return Ok(v),
            Attempt::Fail(e) => return Err(e),
            Attempt::Retry { after, error } => {
                attempt += 1;
                if attempt >= max {
                    return Err(error);
                }
                if after.is_some_and(|d| d > policy.max_delay) {
                    return Err(error);
                }
                let backoff = policy
                    .base_delay
                    .saturating_mul(2u32.saturating_pow(attempt - 1))
                    .min(policy.max_delay);
                let delay = after.unwrap_or(backoff).min(policy.max_delay);
                tracing::warn!(attempt, "Hugging Face request failed transiently; retrying");
                if !delay.is_zero() {
                    tokio::time::sleep(delay).await;
                }
            }
        }
    }
}

/// Hub operations. Implemented over HTTP in production and by fakes in tests.
#[async_trait]
pub trait HfApi: Send + Sync {
    /// Searches GGUF repos, sorted, capped at `limit` (clamped to 1..=100).
    async fn search_gguf(
        &self,
        query: &str,
        sort: SearchSort,
        limit: u32,
    ) -> Result<Vec<HfModelSummary>>;
    /// Lists the GGUF files of a repo's `main` revision, including subfolders.
    async fn list_repo_files(&self, repo: &str) -> Result<Vec<RepoFile>>;
    /// Lists the GGUF files at an exact 40-character commit sha.
    async fn list_repo_files_at(&self, repo: &str, revision: &str) -> Result<Vec<RepoFile>>;
    /// Fetches repo metadata.
    async fn repo_info(&self, repo: &str) -> Result<HfRepoInfo>;
    /// Repo metadata plus the commit sha that `main` currently points to.
    async fn repo_snapshot(&self, repo: &str) -> Result<RepoSnapshot>;
    /// Direct download URL for one file on `main`. Validates both inputs.
    fn file_url(&self, repo: &str, file_name: &str) -> Result<String>;
    /// Direct download URL for one file at an exact commit sha.
    fn file_url_at(&self, repo: &str, revision: &str, file_name: &str) -> Result<String>;
    /// Rate-limit state from the most recent Hub response, if one carried it.
    fn last_rate_limit(&self) -> Option<HubRateLimit> {
        None
    }
}

/// Checks `owner/name`: two non-empty segments of ASCII letters, digits, `-`, `_`, `.`,
/// no leading dot, no `..`, and no extra slashes.
pub fn validate_repo_id(repo: &str) -> Result<()> {
    let invalid = || RaptorError::InvalidInput(format!("invalid repo id: {repo}"));
    let (owner, name) = repo.split_once('/').ok_or_else(invalid)?;
    for part in [owner, name] {
        let ok = !part.is_empty()
            && part.len() <= 96
            && !part.starts_with('.')
            && !part.contains("..")
            && part
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.');
        if !ok {
            return Err(invalid());
        }
    }
    Ok(())
}

/// Checks a full 40-character hex commit sha (either case).
pub fn validate_commit_sha(revision: &str) -> Result<()> {
    if revision.len() == 40 && revision.bytes().all(|b| b.is_ascii_hexdigit()) {
        Ok(())
    } else {
        Err(RaptorError::InvalidInput(
            "revision must be a full 40-character commit sha".into(),
        ))
    }
}

/// Checks a repo-relative path: 1 to `MAX_PATH_DEPTH` segments, each a safe file name.
/// Rejects empty segments, `.` and `..`, backslashes and drive prefixes.
pub fn safe_repo_path(path: &str) -> Result<&str> {
    let invalid = || RaptorError::InvalidInput(format!("unsafe repo path: {path}"));
    if path.is_empty() || path.len() > 1024 {
        return Err(invalid());
    }
    let segments: Vec<&str> = path.split('/').collect();
    if segments.len() > MAX_PATH_DEPTH || segments.iter().any(|s| safe_file_name(s).is_err()) {
        return Err(invalid());
    }
    Ok(path)
}

/// Percent-encodes everything except RFC 3986 unreserved characters.
pub fn percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(char::from(b));
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// Encodes each segment of a validated repo path, keeping the `/` separators.
fn encode_path(path: &str) -> String {
    path.split('/')
        .map(percent_encode)
        .collect::<Vec<_>>()
        .join("/")
}

/// Builds the search URL. The query is encoded; no token ever goes in the URL.
pub fn search_url(base: &str, query: &str, sort: SearchSort, limit: u32) -> Result<String> {
    let q = query.trim();
    if q.chars().count() > MAX_QUERY_CHARS {
        return Err(RaptorError::InvalidInput(format!(
            "search query longer than {MAX_QUERY_CHARS} characters"
        )));
    }
    let limit = limit.clamp(1, MAX_SEARCH_LIMIT);
    let mut url = format!(
        "{base}/api/models?filter=gguf&sort={}&direction=-1&limit={limit}",
        sort.api_value()
    );
    if !q.is_empty() {
        url.push_str("&search=");
        url.push_str(&percent_encode(q));
    }
    Ok(url)
}

/// Direct download URL for a file on the `main` revision.
pub fn resolve_url(base: &str, repo: &str, file_name: &str) -> Result<String> {
    resolve_url_at(base, repo, "main", file_name)
}

/// Direct download URL for a file at `revision` (`main` or a full commit sha).
pub fn resolve_url_at(base: &str, repo: &str, revision: &str, file_name: &str) -> Result<String> {
    validate_repo_id(repo)?;
    if revision != "main" {
        validate_commit_sha(revision)?;
    }
    safe_repo_path(file_name)?;
    Ok(format!(
        "{base}/{repo}/resolve/{revision}/{}",
        encode_path(file_name)
    ))
}

/// Maps an HTTP status to a typed error. Messages never include credentials.
pub fn map_http_status(status: u16, retry_after: Option<String>) -> RaptorError {
    match status {
        401 | 403 => RaptorError::Unauthorized(format!(
            "Hugging Face returned HTTP {status}; the repo may be gated or the token lacks access"
        )),
        404 => RaptorError::NotFound("repo or file not found on Hugging Face".into()),
        429 => {
            let hint = retry_after
                .map(|v| format!(", retry after {v}s"))
                .unwrap_or_default();
            RaptorError::Network(format!("rate limited by Hugging Face (HTTP 429){hint}"))
        }
        _ => RaptorError::Network(format!("Hugging Face returned HTTP {status}")),
    }
}

/// Statuses worth retrying: rate limiting and server-side hiccups.
pub fn is_transient_status(status: u16) -> bool {
    matches!(status, 429 | 500 | 502 | 503 | 504)
}

/// Parses the Hub's structured rate-limit headers. `RateLimit` carries `r=` (remaining) and
/// `t=` (seconds to reset); `RateLimit-Policy` carries `q=` (quota) and `w=` (window).
/// Returns `None` when neither header yields a value.
pub fn parse_rate_limit(rate: Option<&str>, policy: Option<&str>) -> Option<HubRateLimit> {
    let mut out = HubRateLimit {
        remaining: None,
        reset_secs: None,
        quota: None,
        window_secs: None,
    };
    for (key, value) in params(rate) {
        match key {
            "r" => out.remaining = value.parse().ok(),
            "t" => out.reset_secs = value.parse().ok(),
            _ => {}
        }
    }
    for (key, value) in params(policy) {
        match key {
            "q" => out.quota = value.parse().ok(),
            "w" => out.window_secs = value.parse().ok(),
            _ => {}
        }
    }
    let any = out.remaining.is_some()
        || out.reset_secs.is_some()
        || out.quota.is_some()
        || out.window_secs.is_some();
    any.then_some(out)
}

/// `key=value` pairs of a structured header, split on `;`. Bare tokens such as `"api"` are skipped.
fn params(header: Option<&str>) -> Vec<(&str, &str)> {
    header
        .unwrap_or("")
        .split(';')
        .filter_map(|part| {
            let (k, v) = part.trim().split_once('=')?;
            Some((k.trim(), v.trim().trim_matches('"')))
        })
        .collect()
}

/// The `rel="next"` URL of a `Link` header, if any.
pub fn next_link(link: &str) -> Option<String> {
    link.split(',').find_map(|part| {
        if !part.contains("rel=\"next\"") {
            return None;
        }
        let start = part.find('<')? + 1;
        let end = part[start..].find('>')? + start;
        Some(part[start..end].to_string())
    })
}

#[derive(Deserialize)]
struct RawModel {
    id: Option<String>,
    #[serde(rename = "modelId")]
    model_id: Option<String>,
    author: Option<String>,
    downloads: Option<u64>,
    likes: Option<u64>,
    tags: Option<Vec<String>>,
    #[serde(rename = "lastModified")]
    last_modified: Option<String>,
    pipeline_tag: Option<String>,
}

/// Parses a search response. Entries without a usable repo id are dropped so they
/// can never reach the UI or a download.
pub fn parse_search(body: &str) -> Result<Vec<HfModelSummary>> {
    let raw: Vec<RawModel> = serde_json::from_str(body)
        .map_err(|e| RaptorError::Network(format!("unexpected search response: {e}")))?;
    Ok(raw
        .into_iter()
        .filter_map(|m| {
            let id = m.id.or(m.model_id)?;
            validate_repo_id(&id).ok()?;
            Some(HfModelSummary {
                id,
                author: m.author,
                downloads: m.downloads.unwrap_or(0),
                likes: m.likes.unwrap_or(0),
                tags: m.tags.unwrap_or_default(),
                last_modified: m.last_modified,
                pipeline_tag: m.pipeline_tag,
            })
        })
        .collect())
}

#[derive(Deserialize)]
struct RawLfs {
    oid: Option<String>,
    size: Option<u64>,
}

#[derive(Deserialize)]
struct RawEntry {
    #[serde(rename = "type")]
    kind: String,
    path: String,
    size: Option<u64>,
    lfs: Option<RawLfs>,
}

/// Parses one tree listing page into GGUF files, subfolders included.
///
/// Entries whose path fails `safe_repo_path` (for example `../x.gguf`, `a\b.gguf` or a
/// path deeper than `MAX_PATH_DEPTH`) are skipped, so a hostile repo cannot steer a write
/// outside the library.
pub fn parse_tree(body: &str) -> Result<Vec<RepoFile>> {
    let raw: Vec<RawEntry> = serde_json::from_str(body)
        .map_err(|e| RaptorError::Network(format!("unexpected tree response: {e}")))?;
    let mut out = Vec::new();
    for e in raw {
        if e.kind != "file" || !e.path.to_ascii_lowercase().ends_with(".gguf") {
            continue;
        }
        if safe_repo_path(&e.path).is_err() {
            tracing::warn!("skipping repo entry with an unsafe path");
            continue;
        }
        let base = e.path.rsplit('/').next().unwrap_or(&e.path);
        let size = e.lfs.as_ref().and_then(|l| l.size).or(e.size).unwrap_or(0);
        let sha256 = e
            .lfs
            .and_then(|l| l.oid)
            .filter(|h| is_sha256_hex(h))
            .map(|h| h.to_ascii_lowercase());
        out.push(RepoFile {
            quant_label: parse_quant_label(base).map(str::to_string),
            split_shard: is_split_shard(base),
            file_name: e.path.clone(),
            size,
            sha256,
        });
    }
    Ok(out)
}

#[derive(Deserialize)]
struct RawRepo {
    id: Option<String>,
    #[serde(rename = "modelId")]
    model_id: Option<String>,
    pipeline_tag: Option<String>,
    library_name: Option<String>,
    tags: Option<Vec<String>>,
    gated: Option<serde_json::Value>,
    sha: Option<String>,
}

/// Parses repo metadata. `license` comes from the `license:<id>` tag; `gated` is true for
/// any gated value (`"auto"`, `"manual"`, or `true`).
pub fn parse_repo_info(body: &str) -> Result<HfRepoInfo> {
    let raw: RawRepo = serde_json::from_str(body)
        .map_err(|e| RaptorError::Network(format!("unexpected repo response: {e}")))?;
    let id = raw
        .id
        .or(raw.model_id)
        .ok_or_else(|| RaptorError::Network("repo response has no id".into()))?;
    let tags = raw.tags.unwrap_or_default();
    let license = tags
        .iter()
        .find_map(|t| t.strip_prefix("license:"))
        .map(str::to_string);
    let gated = !matches!(
        raw.gated,
        None | Some(serde_json::Value::Null) | Some(serde_json::Value::Bool(false))
    );
    Ok(HfRepoInfo {
        id,
        pipeline_tag: raw.pipeline_tag,
        library_name: raw.library_name,
        license,
        tags,
        gated,
    })
}

/// Parses repo metadata and the head commit sha. A response without a valid sha is an error:
/// an unpinned download is never started.
pub fn parse_repo_snapshot(body: &str) -> Result<RepoSnapshot> {
    let info = parse_repo_info(body)?;
    let raw: RawRepo = serde_json::from_str(body)
        .map_err(|e| RaptorError::Network(format!("unexpected repo response: {e}")))?;
    let revision = raw
        .sha
        .ok_or_else(|| RaptorError::Network("repo response has no commit sha".into()))?
        .to_ascii_lowercase();
    validate_commit_sha(&revision)
        .map_err(|_| RaptorError::Network("repo response has a malformed commit sha".into()))?;
    Ok(RepoSnapshot { info, revision })
}

/// HTTP implementation of [`HfApi`]. The token, if any, is sent only as a bearer header and
/// only to the configured base URL; pagination links to other hosts are refused.
pub struct HttpHfApi {
    client: reqwest::Client,
    base_url: String,
    token: Option<String>,
    retry: RetryPolicy,
    rate: Mutex<Option<HubRateLimit>>,
}

impl HttpHfApi {
    pub fn new(token: Option<String>) -> Result<Self> {
        Self::with_base_url(DEFAULT_BASE_URL, token)
    }

    pub fn with_base_url(base_url: &str, token: Option<String>) -> Result<Self> {
        let client = reqwest::Client::builder()
            .user_agent(concat!("RaptorLLM/", env!("CARGO_PKG_VERSION")))
            .connect_timeout(Duration::from_secs(15))
            .timeout(Duration::from_secs(60))
            .build()
            .map_err(|e| RaptorError::Network(format!("cannot build HTTP client: {e}")))?;
        Ok(Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            token: token.filter(|t| !t.trim().is_empty()),
            retry: RetryPolicy::default(),
            rate: Mutex::new(None),
        })
    }

    /// Replaces the retry policy used for Hub API calls.
    pub fn with_retry_policy(mut self, policy: RetryPolicy) -> Self {
        self.retry = policy;
        self
    }

    fn remember_rate_limit(&self, headers: &reqwest::header::HeaderMap) {
        let get = |name: &str| headers.get(name).and_then(|v| v.to_str().ok());
        if let Some(rl) = parse_rate_limit(get("ratelimit"), get("ratelimit-policy")) {
            if let Ok(mut slot) = self.rate.lock() {
                *slot = Some(rl);
            }
        }
    }

    /// One GET with retries. Returns the body and the `Link: rel="next"` URL, if any.
    async fn get_page(&self, url: &str) -> Result<(String, Option<String>)> {
        run_with_retry(self.retry, |_attempt| async move {
            let mut req = self
                .client
                .get(url)
                .header(reqwest::header::ACCEPT, "application/json");
            if let Some(t) = &self.token {
                req = req.bearer_auth(t);
            }
            let resp = match req.send().await {
                Ok(r) => r,
                Err(e) => {
                    return Attempt::Retry {
                        after: None,
                        error: RaptorError::Network(format!(
                            "request to Hugging Face failed: {}",
                            e.without_url()
                        )),
                    }
                }
            };
            let headers = resp.headers().clone();
            self.remember_rate_limit(&headers);
            let status = resp.status();
            if !status.is_success() {
                let retry = headers
                    .get(reqwest::header::RETRY_AFTER)
                    .and_then(|v| v.to_str().ok())
                    .map(str::to_string);
                let after = retry
                    .as_deref()
                    .and_then(|v| v.trim().parse::<u64>().ok())
                    .map(Duration::from_secs);
                let error = map_http_status(status.as_u16(), retry);
                return if is_transient_status(status.as_u16()) {
                    Attempt::Retry { after, error }
                } else {
                    Attempt::Fail(error)
                };
            }
            let next = headers
                .get(reqwest::header::LINK)
                .and_then(|v| v.to_str().ok())
                .and_then(next_link);
            match resp.text().await {
                Ok(body) => Attempt::Done((body, next)),
                Err(e) => Attempt::Retry {
                    after: None,
                    error: RaptorError::Network(format!(
                        "reading Hugging Face response failed: {}",
                        e.without_url()
                    )),
                },
            }
        })
        .await
    }

    /// Follows tree pagination. A next link must stay on the configured base URL.
    async fn get_all_pages(&self, first: String) -> Result<Vec<String>> {
        let mut pages = Vec::new();
        let mut next = Some(first);
        while let Some(url) = next.take() {
            if pages.len() >= MAX_TREE_PAGES {
                return Err(RaptorError::Network(
                    "repo listing has more pages than this client follows".into(),
                ));
            }
            let (body, link) = self.get_page(&url).await?;
            pages.push(body);
            if let Some(link) = link {
                if !link.starts_with(&format!("{}/", self.base_url)) {
                    return Err(RaptorError::Network(
                        "Hugging Face pagination pointed outside its own host; refusing to follow"
                            .into(),
                    ));
                }
                next = Some(link);
            }
        }
        Ok(pages)
    }

    async fn tree(&self, repo: &str, revision: &str) -> Result<Vec<RepoFile>> {
        validate_repo_id(repo)?;
        let url = format!(
            "{}/api/models/{repo}/tree/{revision}?recursive=true",
            self.base_url
        );
        let mut files = Vec::new();
        for page in self.get_all_pages(url).await? {
            files.extend(parse_tree(&page)?);
        }
        Ok(files)
    }
}

impl fmt::Debug for HttpHfApi {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HttpHfApi")
            .field("base_url", &self.base_url)
            .field("token", &self.token.as_ref().map(|_| "<redacted>"))
            .field("retry", &self.retry)
            .finish()
    }
}

#[async_trait]
impl HfApi for HttpHfApi {
    async fn search_gguf(
        &self,
        query: &str,
        sort: SearchSort,
        limit: u32,
    ) -> Result<Vec<HfModelSummary>> {
        let url = search_url(&self.base_url, query, sort, limit)?;
        let (body, _) = self.get_page(&url).await?;
        parse_search(&body)
    }

    async fn list_repo_files(&self, repo: &str) -> Result<Vec<RepoFile>> {
        self.tree(repo, "main").await
    }

    async fn list_repo_files_at(&self, repo: &str, revision: &str) -> Result<Vec<RepoFile>> {
        validate_commit_sha(revision)?;
        self.tree(repo, revision).await
    }

    async fn repo_info(&self, repo: &str) -> Result<HfRepoInfo> {
        validate_repo_id(repo)?;
        let url = format!("{}/api/models/{repo}", self.base_url);
        let (body, _) = self.get_page(&url).await?;
        parse_repo_info(&body)
    }

    async fn repo_snapshot(&self, repo: &str) -> Result<RepoSnapshot> {
        validate_repo_id(repo)?;
        let url = format!("{}/api/models/{repo}", self.base_url);
        let (body, _) = self.get_page(&url).await?;
        parse_repo_snapshot(&body)
    }

    fn file_url(&self, repo: &str, file_name: &str) -> Result<String> {
        resolve_url(&self.base_url, repo, file_name)
    }

    fn file_url_at(&self, repo: &str, revision: &str, file_name: &str) -> Result<String> {
        resolve_url_at(&self.base_url, repo, revision, file_name)
    }

    fn last_rate_limit(&self) -> Option<HubRateLimit> {
        self.rate.lock().ok().and_then(|slot| *slot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEARCH: &str = r#"[
        {"_id":"1","id":"bartowski/Llama-3.2-3B-Instruct-GGUF","modelId":"bartowski/Llama-3.2-3B-Instruct-GGUF",
         "author":"bartowski","downloads":12345,"likes":210,"tags":["gguf","llama"],
         "pipeline_tag":"text-generation","lastModified":"2026-01-02T03:04:05.000Z"},
        {"id":"../evil/repo","downloads":1},
        {"modelId":"someone/minimal"},
        {"id":"no-slash-here"}
    ]"#;

    const TREE: &str = r#"[
        {"type":"file","oid":"abc","size":120,"path":"README.md"},
        {"type":"file","oid":"def","size":4000000000,"path":"Llama-3.2-3B-Q4_K_M.gguf",
         "lfs":{"oid":"ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789","size":4000000000}},
        {"type":"file","oid":"ghi","size":1000,"path":"sub/Llama-Q8_0.gguf"},
        {"type":"file","oid":"jkl","size":1000,"path":"../../etc/passwd.gguf"},
        {"type":"file","oid":"mno","size":1000,"path":"..\\evil.gguf"},
        {"type":"file","oid":"mno2","size":1000,"path":"a/b/c/d/e/too-deep.gguf"},
        {"type":"file","oid":"mno3","size":1000,"path":"a//b.gguf"},
        {"type":"directory","path":"folder.gguf"},
        {"type":"file","oid":"pqr","size":2000000000,"path":"model-Q2_K-00001-of-00002.gguf",
         "lfs":{"oid":"not-a-real-hash","size":2000000000}},
        {"type":"file","oid":"stu","size":500,"path":"Tiny-IQ3_XS.gguf"}
    ]"#;

    const REPO: &str = r#"{"id":"owner/model","modelId":"owner/model","pipeline_tag":"text-generation",
        "library_name":"gguf","tags":["gguf","license:apache-2.0","en"],"gated":"auto",
        "sha":"0123456789abcdef0123456789abcdef01234567"}"#;

    fn find<'a>(files: &'a [RepoFile], path: &str) -> &'a RepoFile {
        files.iter().find(|f| f.file_name == path).unwrap()
    }

    #[test]
    fn parses_search_fixture() {
        let hits = parse_search(SEARCH).unwrap();
        assert_eq!(hits.len(), 2, "unsafe and id-less entries must be dropped");
        assert_eq!(hits[0].id, "bartowski/Llama-3.2-3B-Instruct-GGUF");
        assert_eq!(hits[0].downloads, 12345);
        assert_eq!(hits[0].tags, vec!["gguf", "llama"]);
        assert_eq!(hits[1].id, "someone/minimal");
        assert_eq!(hits[1].downloads, 0);
    }

    #[test]
    fn search_rejects_non_json() {
        assert!(matches!(
            parse_search("<html>"),
            Err(RaptorError::Network(_))
        ));
    }

    #[test]
    fn parses_tree_including_subfolders_and_skips_hostile_entries() {
        let files = parse_tree(TREE).unwrap();
        let names: Vec<&str> = files.iter().map(|f| f.file_name.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "Llama-3.2-3B-Q4_K_M.gguf",
                "sub/Llama-Q8_0.gguf",
                "model-Q2_K-00001-of-00002.gguf",
                "Tiny-IQ3_XS.gguf"
            ]
        );
    }

    #[test]
    fn path_traversal_names_never_reach_results() {
        let files = parse_tree(TREE).unwrap();
        for f in &files {
            assert!(f.file_name.split('/').all(|s| s != ".." && !s.is_empty()));
            assert!(!f.file_name.contains('\\'));
        }
    }

    #[test]
    fn tree_extracts_lfs_hash_size_and_quant() {
        let files = parse_tree(TREE).unwrap();
        let main = find(&files, "Llama-3.2-3B-Q4_K_M.gguf");
        assert_eq!(main.size, 4_000_000_000);
        assert_eq!(main.quant_label.as_deref(), Some("Q4_K_M"));
        assert_eq!(
            main.sha256.as_deref(),
            Some("abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789")
        );
        assert!(!main.split_shard);
    }

    #[test]
    fn invalid_lfs_hash_becomes_unverifiable_not_trusted() {
        let files = parse_tree(TREE).unwrap();
        let shard = find(&files, "model-Q2_K-00001-of-00002.gguf");
        assert_eq!(shard.sha256, None);
        assert!(shard.split_shard);
    }

    #[test]
    fn non_lfs_file_uses_plain_size_and_has_no_hash() {
        let files = parse_tree(TREE).unwrap();
        let tiny = find(&files, "Tiny-IQ3_XS.gguf");
        assert_eq!(tiny.size, 500);
        assert_eq!(tiny.sha256, None);
    }

    #[test]
    fn quant_label_comes_from_the_file_not_its_folder() {
        let files =
            parse_tree(r#"[{"type":"file","oid":"x","size":5,"path":"Q8_0/model.gguf"}]"#).unwrap();
        assert_eq!(files[0].quant_label, None);
        assert_eq!(files[0].file_name, "Q8_0/model.gguf");
    }

    #[test]
    fn split_shard_detection() {
        assert!(is_split_shard("model-Q2_K-00001-of-00002.gguf"));
        assert!(is_split_shard("Big-F16-00003-of-00010.GGUF"));
        assert!(!is_split_shard("model-Q4_K_M.gguf"));
        assert!(!is_split_shard("model-of-00002.gguf"));
        assert!(!is_split_shard("model-00001-of-x.gguf"));
    }

    #[test]
    fn repo_id_validation() {
        assert!(validate_repo_id("bartowski/Llama-3.2-3B-Instruct-GGUF").is_ok());
        assert!(validate_repo_id("a/b").is_ok());
        for bad in [
            "",
            "noslash",
            "/x",
            "x/",
            "../x",
            "a/../b",
            "a/b/c",
            ".hidden/x",
            "a b/c",
            "a/b\\c",
            "a/..",
            "a/b?x=1",
        ] {
            assert!(validate_repo_id(bad).is_err(), "{bad} should be rejected");
        }
    }

    #[test]
    fn repo_path_validation() {
        assert!(safe_repo_path("Q4_K_M/model-00001-of-00002.gguf").is_ok());
        assert!(
            safe_repo_path("a/b/c/x.gguf").is_ok(),
            "four segments is the limit"
        );
        for bad in [
            "",
            "../x.gguf",
            "a/../x.gguf",
            "a//x.gguf",
            "/abs.gguf",
            "a\\b.gguf",
            "C:x.gguf",
            "a/b/c/d/x.gguf",
            "a/b/c/d/e/x.gguf",
            "a/.",
        ] {
            assert!(safe_repo_path(bad).is_err(), "{bad} should be rejected");
        }
    }

    #[test]
    fn percent_encoding_is_strict() {
        assert_eq!(
            percent_encode("llama 3 & more/x"),
            "llama%203%20%26%20more%2Fx"
        );
        assert_eq!(percent_encode("Q4_K_M.gguf"), "Q4_K_M.gguf");
    }

    #[test]
    fn search_url_encodes_query_and_clamps_limit() {
        let url = search_url("https://h.example", " llama 3 ", SearchSort::Likes, 500).unwrap();
        assert_eq!(
            url,
            "https://h.example/api/models?filter=gguf&sort=likes&direction=-1&limit=100&search=llama%203"
        );
        let no_query = search_url("https://h.example", "  ", SearchSort::Downloads, 0).unwrap();
        assert!(no_query.ends_with("limit=1"));
        assert!(!no_query.contains("search="));
    }

    #[test]
    fn search_url_rejects_huge_query() {
        let q = "x".repeat(MAX_QUERY_CHARS + 1);
        assert!(search_url(DEFAULT_BASE_URL, &q, SearchSort::Downloads, 10).is_err());
    }

    #[test]
    fn resolve_url_on_main_and_on_pinned_revision() {
        assert_eq!(
            resolve_url(DEFAULT_BASE_URL, "owner/repo", "m-Q4_K_M.gguf").unwrap(),
            "https://huggingface.co/owner/repo/resolve/main/m-Q4_K_M.gguf"
        );
        let sha = "0123456789abcdef0123456789abcdef01234567";
        assert_eq!(
            resolve_url_at(
                DEFAULT_BASE_URL,
                "owner/repo",
                sha,
                "Q4/m 1-00001-of-00002.gguf"
            )
            .unwrap(),
            format!("https://huggingface.co/owner/repo/resolve/{sha}/Q4/m%201-00001-of-00002.gguf")
        );
    }

    #[test]
    fn resolve_url_rejects_traversal_and_bad_revisions() {
        assert!(resolve_url(DEFAULT_BASE_URL, "owner/repo", "../x.gguf").is_err());
        assert!(resolve_url(DEFAULT_BASE_URL, "owner/repo", "a/../b.gguf").is_err());
        assert!(resolve_url(DEFAULT_BASE_URL, "../repo", "x.gguf").is_err());
        assert!(resolve_url_at(DEFAULT_BASE_URL, "owner/repo", "main;rm", "x.gguf").is_err());
        assert!(resolve_url_at(DEFAULT_BASE_URL, "owner/repo", "abc", "x.gguf").is_err());
    }

    #[test]
    fn http_status_mapping() {
        assert!(matches!(
            map_http_status(401, None),
            RaptorError::Unauthorized(_)
        ));
        assert!(matches!(
            map_http_status(403, None),
            RaptorError::Unauthorized(_)
        ));
        assert!(matches!(
            map_http_status(404, None),
            RaptorError::NotFound(_)
        ));
        match map_http_status(429, Some("60".into())) {
            RaptorError::Network(m) => assert!(m.contains("retry after 60s")),
            other => panic!("unexpected {other:?}"),
        }
        assert!(matches!(
            map_http_status(503, None),
            RaptorError::Network(_)
        ));
        assert!(is_transient_status(429) && is_transient_status(503));
        assert!(!is_transient_status(404) && !is_transient_status(401));
    }

    #[test]
    fn repo_info_parses_license_and_gating() {
        let info = parse_repo_info(REPO).unwrap();
        assert_eq!(info.id, "owner/model");
        assert_eq!(info.license.as_deref(), Some("apache-2.0"));
        assert_eq!(info.library_name.as_deref(), Some("gguf"));
        assert!(info.gated);

        let open = parse_repo_info(r#"{"id":"o/m","gated":false,"tags":[]}"#).unwrap();
        assert!(!open.gated);
        assert_eq!(open.license, None);
        assert!(parse_repo_info(r#"{"tags":[]}"#).is_err());
    }

    #[test]
    fn repo_snapshot_pins_the_head_commit() {
        let snap = parse_repo_snapshot(REPO).unwrap();
        assert_eq!(snap.revision, "0123456789abcdef0123456789abcdef01234567");
        assert_eq!(snap.info.id, "owner/model");
        assert!(parse_repo_snapshot(r#"{"id":"o/m","tags":[]}"#).is_err());
        assert!(parse_repo_snapshot(r#"{"id":"o/m","sha":"main"}"#).is_err());
        assert!(parse_repo_snapshot(
            r#"{"id":"o/m","sha":"ABCDEF0123456789ABCDEF0123456789ABCDEF01"}"#
        )
        .is_ok());
    }

    #[test]
    fn rate_limit_headers_are_parsed() {
        let rl =
            parse_rate_limit(Some("\"api\";r=297;t=233"), Some("\"api\";q=500;w=300")).unwrap();
        assert_eq!(rl.remaining, Some(297));
        assert_eq!(rl.reset_secs, Some(233));
        assert_eq!(rl.quota, Some(500));
        assert_eq!(rl.window_secs, Some(300));
        let partial = parse_rate_limit(Some("r=0"), None).unwrap();
        assert_eq!(partial.remaining, Some(0));
        assert_eq!(partial.reset_secs, None);
        assert_eq!(parse_rate_limit(None, None), None);
        assert_eq!(parse_rate_limit(Some("garbage"), None), None);
    }

    #[test]
    fn link_header_next_page_is_extracted() {
        let link = "<https://huggingface.co/api/models/x/tree/main?cursor=abc>; rel=\"next\"";
        assert_eq!(
            next_link(link).as_deref(),
            Some("https://huggingface.co/api/models/x/tree/main?cursor=abc")
        );
        assert_eq!(next_link("<https://h.example/a>; rel=\"prev\""), None);
    }

    fn fast_policy(max_attempts: u32) -> RetryPolicy {
        RetryPolicy {
            max_attempts,
            base_delay: Duration::ZERO,
            max_delay: Duration::from_secs(8),
        }
    }

    #[tokio::test]
    async fn retry_recovers_after_transient_failures() {
        let mut calls = 0u32;
        let out = run_with_retry(fast_policy(3), |_| {
            calls += 1;
            let n = calls;
            async move {
                if n < 3 {
                    Attempt::Retry {
                        after: None,
                        error: RaptorError::Network("flaky".into()),
                    }
                } else {
                    Attempt::Done(n)
                }
            }
        })
        .await
        .unwrap();
        assert_eq!(out, 3);
        assert_eq!(calls, 3);
    }

    #[tokio::test]
    async fn retry_is_bounded_and_returns_the_last_error() {
        let mut calls = 0u32;
        let err = run_with_retry::<(), _, _>(fast_policy(3), |_| {
            calls += 1;
            async {
                Attempt::Retry {
                    after: None,
                    error: RaptorError::Network("still down".into()),
                }
            }
        })
        .await
        .unwrap_err();
        assert_eq!(calls, 3);
        assert!(matches!(err, RaptorError::Network(m) if m == "still down"));
    }

    #[tokio::test]
    async fn permanent_failures_are_never_retried() {
        let mut calls = 0u32;
        let err = run_with_retry::<(), _, _>(fast_policy(5), |_| {
            calls += 1;
            async { Attempt::Fail(RaptorError::NotFound("gone".into())) }
        })
        .await
        .unwrap_err();
        assert_eq!(calls, 1);
        assert!(matches!(err, RaptorError::NotFound(_)));
    }

    #[tokio::test]
    async fn long_retry_after_is_not_waited_for() {
        let mut calls = 0u32;
        let err = run_with_retry::<(), _, _>(fast_policy(5), |_| {
            calls += 1;
            async {
                Attempt::Retry {
                    after: Some(Duration::from_secs(3600)),
                    error: RaptorError::Network("rate limited".into()),
                }
            }
        })
        .await
        .unwrap_err();
        assert_eq!(
            calls, 1,
            "a Retry-After beyond the cap ends the loop at once"
        );
        assert!(matches!(err, RaptorError::Network(_)));
    }

    #[test]
    fn debug_output_never_shows_token() {
        let api = HttpHfApi::new(Some("hf_supersecret".into())).unwrap();
        let shown = format!("{api:?}");
        assert!(!shown.contains("hf_supersecret"));
        assert!(shown.contains("<redacted>"));
    }

    #[test]
    fn blank_token_is_treated_as_absent() {
        let api = HttpHfApi::new(Some("   ".into())).unwrap();
        assert!(format!("{api:?}").contains("token: None"));
    }
}
