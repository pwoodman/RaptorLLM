//! llama-server process lifecycle: start, health-gated readiness, crash
//! detection with bounded backoff restarts, graceful shutdown of the whole
//! process tree, and a bounded log capture.
//!
//! The process itself is behind [`Spawner`], so every state transition below is
//! tested with a fake child and paused tokio time.

use crate::config::LaunchConfig;
use crate::http::HttpClient;
use crate::sys::PortCheck;
use async_trait::async_trait;
use raptor_core::{RaptorError, Result};
use serde::Serialize;
use std::collections::VecDeque;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncRead, BufReader};
use tokio::sync::{mpsc, oneshot, Mutex as AsyncMutex};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

/// Lifecycle state of the managed server.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum ServerState {
    Stopped,
    Starting,
    /// Health and model endpoints answered 200.
    Ready,
    /// Was ready; health checks are failing but the process is alive.
    Degraded {
        reason: String,
    },
    /// The process exited or never became ready. A restart may be pending.
    Crashed {
        reason: String,
    },
}

/// Snapshot returned by [`ServerManager::status`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ServerStatus {
    pub state: ServerState,
    pub endpoint: Option<String>,
    pub pid: Option<u32>,
    pub restarts: u32,
}

/// Which pipe a log line came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum LogStream {
    Stdout,
    Stderr,
}

/// One line of llama-server output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LogLine {
    pub stream: LogStream,
    pub text: String,
}

/// What to run. Arguments and environment come from [`LaunchConfig`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpawnSpec {
    pub program: PathBuf,
    pub args: Vec<String>,
    pub env: Vec<(String, String)>,
}

/// How a child ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProcessExit {
    pub code: Option<i32>,
}

/// Control surface for a running child.
#[async_trait]
pub trait ProcessControl: Send + Sync {
    fn pid(&self) -> Option<u32>;
    /// Asks the whole tree to stop. Sends a graceful signal first and forces
    /// a kill after `grace`. Returns as soon as the request is accepted; wait
    /// on [`ProcessHandle::exit`] for completion.
    async fn request_stop(&self, grace: Duration) -> Result<()>;
}

/// A started child: control, log stream, and exit notification.
pub struct ProcessHandle {
    pub control: Box<dyn ProcessControl>,
    pub logs: mpsc::UnboundedReceiver<LogLine>,
    pub exit: oneshot::Receiver<ProcessExit>,
}

/// Starts child processes. The real implementation is [`SystemSpawner`].
#[async_trait]
pub trait Spawner: Send + Sync {
    async fn spawn(&self, spec: &SpawnSpec) -> Result<ProcessHandle>;
}

/// Receives every log line as it arrives (for example a structured log file).
pub trait LogSink: Send + Sync {
    fn record(&self, line: &LogLine);
}

/// Discards log lines.
#[derive(Debug, Default, Clone, Copy)]
pub struct NullSink;

impl LogSink for NullSink {
    fn record(&self, _line: &LogLine) {}
}

/// Appends each line as one JSON object to a file. Either one fixed file, or a daily rolling
/// set (`llama-server-YYYY-MM-DD.log`) that keeps the last seven days.
pub struct FileLogSink {
    target: SinkTarget,
}

enum SinkTarget {
    Single(Mutex<std::fs::File>),
    Rolling(raptor_core::logfiles::RollingLog),
}

impl FileLogSink {
    /// Opens (creating if needed) `path` for appending.
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        Ok(Self {
            target: SinkTarget::Single(Mutex::new(file)),
        })
    }

    /// Writes `llama-server-YYYY-MM-DD.log` files in `dir`, one per day, keeping the last
    /// [`raptor_core::logfiles::KEEP_DAYS`] days.
    pub fn rolling(dir: &Path) -> Result<Self> {
        let log = raptor_core::logfiles::RollingLog::new(
            dir,
            "llama-server",
            raptor_core::logfiles::KEEP_DAYS,
        )?;
        Ok(Self {
            target: SinkTarget::Rolling(log),
        })
    }
}

impl LogSink for FileLogSink {
    fn record(&self, line: &LogLine) {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let entry = serde_json::json!({ "ts_ms": ts, "stream": line.stream, "text": line.text });
        let written = match &self.target {
            SinkTarget::Single(file) => writeln!(lock(file), "{entry}").map_err(Into::into),
            SinkTarget::Rolling(log) => log.write_bytes(
                format!(
                    "{entry}
"
                )
                .as_bytes(),
            ),
        };
        if let Err(e) = written {
            tracing::warn!(error = %e, "could not append to llama-server log file");
        }
    }
}

/// Why llama-server failed, inferred from its log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LaunchDiagnosis {
    OutOfMemory,
    UnsupportedArchitecture(String),
    ModelFileMissing,
    PortInUse,
    Other(String),
}

impl LaunchDiagnosis {
    /// Plain-language explanation shown to the user.
    pub fn message(&self) -> String {
        match self {
            LaunchDiagnosis::OutOfMemory => {
                "out of memory: the model or context does not fit in VRAM/RAM; reduce GPU layers or context".into()
            }
            LaunchDiagnosis::UnsupportedArchitecture(l) => {
                format!("unsupported model architecture ({l}); the runtime may be too old for this model")
            }
            LaunchDiagnosis::ModelFileMissing => "model file not found or unreadable".into(),
            LaunchDiagnosis::PortInUse => "port already in use by another process".into(),
            LaunchDiagnosis::Other(l) => format!("load error: {l}"),
        }
    }
}

fn classify_specific(line: &str) -> Option<LaunchDiagnosis> {
    let lo = line.to_ascii_lowercase();
    if lo.contains("couldn't bind http server socket")
        || lo.contains("address already in use")
        || lo.contains("only one usage of each socket address")
    {
        Some(LaunchDiagnosis::PortInUse)
    } else if [
        "out of memory",
        "cudamalloc failed",
        "hipmalloc failed",
        "unable to allocate",
        "erroroutofdevicememory",
        "failed to allocate",
    ]
    .iter()
    .any(|p| lo.contains(p))
    {
        Some(LaunchDiagnosis::OutOfMemory)
    } else if lo.contains("unknown model architecture")
        || lo.contains("unsupported model architecture")
    {
        Some(LaunchDiagnosis::UnsupportedArchitecture(
            line.trim().to_string(),
        ))
    } else if lo.contains("no such file or directory")
        || lo.contains("cannot find the file")
        || (lo.contains("failed to open") && lo.contains("gguf"))
    {
        Some(LaunchDiagnosis::ModelFileMissing)
    } else {
        None
    }
}

/// Infers the most specific failure cause from a llama-server log. Specific
/// causes win over the generic "error loading model" line.
pub fn diagnose_log(text: &str) -> Option<LaunchDiagnosis> {
    if let Some(d) = text.lines().find_map(classify_specific) {
        return Some(d);
    }
    text.lines()
        .find(|l| {
            let lo = l.to_ascii_lowercase();
            lo.contains("error loading model") || lo.contains("failed to load model")
        })
        .map(|l| LaunchDiagnosis::Other(l.trim().to_string()))
}

/// Exponential backoff between automatic restarts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RestartPolicy {
    /// Consecutive restarts allowed. The first launch does not count.
    pub max_restarts: u32,
    pub base_backoff: Duration,
    pub max_backoff: Duration,
    /// A child that stays Ready this long resets the consecutive-restart count
    /// (and the reported restart count), so an occasional crash does not
    /// exhaust the budget for the rest of the session.
    pub stable_window: Duration,
}

impl Default for RestartPolicy {
    fn default() -> Self {
        Self {
            max_restarts: 3,
            base_backoff: Duration::from_secs(2),
            max_backoff: Duration::from_secs(30),
            stable_window: Duration::from_secs(60),
        }
    }
}

impl RestartPolicy {
    /// Delay before restart `attempt` (1-based): base * 2^(attempt-1), capped.
    pub fn backoff(&self, attempt: u32) -> Duration {
        let shift = attempt.saturating_sub(1).min(20);
        self.base_backoff
            .saturating_mul(2u32.pow(shift))
            .min(self.max_backoff)
    }
}

/// Timing knobs. The defaults suit large models on a desktop GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Timing {
    /// Maximum time from spawn to Ready.
    pub startup_timeout: Duration,
    /// Delay between readiness probes during startup.
    pub poll_interval: Duration,
    /// Delay between health checks while running.
    pub health_interval: Duration,
    /// Grace period before a forced kill on stop.
    pub stop_grace: Duration,
    /// Consecutive failed health checks before Degraded.
    pub failures_before_degraded: u32,
}

impl Default for Timing {
    fn default() -> Self {
        Self {
            startup_timeout: Duration::from_secs(180),
            poll_interval: Duration::from_millis(500),
            health_interval: Duration::from_secs(5),
            stop_grace: Duration::from_secs(10),
            failures_before_degraded: 2,
        }
    }
}

/// Bounded ring buffer of recent log lines.
#[derive(Debug, Clone)]
pub struct LogBuffer {
    cap: usize,
    lines: VecDeque<LogLine>,
}

impl LogBuffer {
    /// Creates a buffer that keeps the newest `cap` lines.
    pub fn new(cap: usize) -> Self {
        Self {
            cap: cap.max(1),
            lines: VecDeque::new(),
        }
    }

    /// Appends a line, dropping the oldest when full.
    pub fn push(&mut self, line: LogLine) {
        if self.lines.len() == self.cap {
            self.lines.pop_front();
        }
        self.lines.push_back(line);
    }

    /// The newest `n` lines, oldest first.
    pub fn tail(&self, n: usize) -> Vec<LogLine> {
        let skip = self.lines.len().saturating_sub(n);
        self.lines.iter().skip(skip).cloned().collect()
    }

    /// All retained lines joined with newlines.
    pub fn text(&self) -> String {
        self.lines
            .iter()
            .map(|l| l.text.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Drops every retained line.
    pub fn clear(&mut self) {
        self.lines.clear();
    }

    /// Number of retained lines.
    pub fn len(&self) -> usize {
        self.lines.len()
    }

    /// True when no lines are retained.
    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }
}

/// Dependencies injected into the manager.
#[derive(Clone)]
pub struct ManagerDeps {
    pub spawner: Arc<dyn Spawner>,
    pub http: Arc<dyn HttpClient>,
    pub ports: Arc<dyn PortCheck>,
    pub sink: Arc<dyn LogSink>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    // A poisoned lock still holds consistent data for this simple state, so recover it.
    m.lock().unwrap_or_else(|p| p.into_inner())
}

const LOG_CAPACITY: usize = 2000;

struct Shared {
    status: Mutex<ServerStatus>,
    log: Mutex<LogBuffer>,
    active: Mutex<Option<CancellationToken>>,
    sink: Arc<dyn LogSink>,
}

impl Shared {
    fn new(sink: Arc<dyn LogSink>) -> Self {
        Self {
            status: Mutex::new(ServerStatus {
                state: ServerState::Stopped,
                endpoint: None,
                pid: None,
                restarts: 0,
            }),
            log: Mutex::new(LogBuffer::new(LOG_CAPACITY)),
            active: Mutex::new(None),
            sink,
        }
    }

    fn update(&self, f: impl FnOnce(&mut ServerStatus)) {
        f(&mut lock(&self.status));
    }

    fn snapshot(&self) -> ServerStatus {
        lock(&self.status).clone()
    }

    fn push_log(&self, line: LogLine) {
        self.sink.record(&line);
        lock(&self.log).push(line);
    }

    fn log_text(&self) -> String {
        lock(&self.log).text()
    }

    /// Starts a fresh log for a new server process. Without this the GPU verdict, which reads the
    /// whole buffer, also counted the previous model's offload lines and buffer sizes (measured:
    /// granite showed 26,510 MiB on a 16 GB card, the sum of flint and granite).
    fn clear_log(&self) {
        lock(&self.log).clear();
    }

    fn set_active(&self, token: CancellationToken) {
        *lock(&self.active) = Some(token);
    }

    fn cancel_active(&self) {
        if let Some(t) = lock(&self.active).clone() {
            t.cancel();
        }
    }
}

#[derive(Clone)]
struct Ctx {
    shared: Arc<Shared>,
    deps: ManagerDeps,
    restart: RestartPolicy,
    timing: Timing,
    cancel: CancellationToken,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Health {
    Ready,
    Loading,
    Down(String),
}

async fn probe_health(http: &dyn HttpClient, endpoint: &str) -> Health {
    let t = Duration::from_secs(2);
    match http.get(&format!("{endpoint}/health"), t).await {
        Ok(r) if r.status == 200 => {}
        Ok(r) if r.status == 503 => return Health::Loading,
        Ok(r) => return Health::Down(format!("/health returned HTTP {}", r.status)),
        Err(e) => return Health::Down(e.to_string()),
    }
    match http.get(&format!("{endpoint}/v1/models"), t).await {
        Ok(r) if r.status == 200 => Health::Ready,
        Ok(r) if r.status == 503 => Health::Loading,
        Ok(r) => Health::Down(format!("/v1/models returned HTTP {}", r.status)),
        Err(e) => Health::Down(e.to_string()),
    }
}

fn startup_error(ctx: &Ctx, what: &str) -> RaptorError {
    let text = ctx.shared.log_text();
    match diagnose_log(&text) {
        Some(d) => RaptorError::Runtime(format!("{what}: {}", d.message())),
        None => RaptorError::Runtime(format!("{what}; no diagnosis in the log")),
    }
}

/// Why readiness failed. `Exited` means the child's exit was already consumed,
/// so it must not be awaited again.
enum WaitError {
    Exited(RaptorError),
    Failed(RaptorError),
}

/// Spawns one child and waits for Ready. On failure a still-running child is
/// stopped, so nothing is left behind.
async fn start_child(ctx: &Ctx, cfg: &LaunchConfig, spec: &SpawnSpec) -> Result<ProcessHandle> {
    let mut handle = ctx.deps.spawner.spawn(spec).await?;
    let pid = handle.control.pid();
    ctx.shared.update(|s| s.pid = pid);
    match wait_ready(ctx, cfg, &mut handle).await {
        Ok(()) => Ok(handle),
        Err(WaitError::Exited(e)) => Err(e),
        Err(WaitError::Failed(e)) => {
            shutdown(handle, ctx.timing.stop_grace).await;
            Err(e)
        }
    }
}

async fn wait_ready(
    ctx: &Ctx,
    cfg: &LaunchConfig,
    handle: &mut ProcessHandle,
) -> std::result::Result<(), WaitError> {
    let endpoint = cfg.endpoint();
    let deadline = tokio::time::Instant::now() + ctx.timing.startup_timeout;
    let mut logs_open = true;
    loop {
        if probe_health(ctx.deps.http.as_ref(), &endpoint).await == Health::Ready {
            // Keep whatever startup output is already queued so diagnostics see it.
            while let Ok(line) = handle.logs.try_recv() {
                ctx.shared.push_log(line);
            }
            return Ok(());
        }
        if tokio::time::Instant::now() >= deadline {
            return Err(WaitError::Failed(startup_error(
                ctx,
                &format!("server not ready after {:?}", ctx.timing.startup_timeout),
            )));
        }
        tokio::select! {
            biased;
            _ = ctx.cancel.cancelled() => {
                return Err(WaitError::Failed(RaptorError::Runtime("start cancelled".into())));
            }
            line = handle.logs.recv(), if logs_open => match line {
                Some(l) => ctx.shared.push_log(l),
                None => logs_open = false,
            },
            exit = &mut handle.exit => {
                let code = exit.ok().and_then(|e| e.code);
                return Err(WaitError::Exited(startup_error(
                    ctx,
                    &format!("process exited during startup (code {code:?})"),
                )));
            }
            _ = tokio::time::sleep(ctx.timing.poll_interval) => {}
        }
    }
}

enum Monitor {
    Cancelled,
    Exited(Option<i32>),
}

/// Watches a Ready child: drains logs, health-checks it, and reports exit.
async fn monitor(
    ctx: &Ctx,
    endpoint: &str,
    handle: &mut ProcessHandle,
    attempts: &mut u32,
) -> Monitor {
    let mut interval = tokio::time::interval(ctx.timing.health_interval);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    interval.tick().await; // the first tick is immediate; the child was just probed
    let mut failures = 0u32;
    let mut logs_open = true;
    // Once this child has run for `stable_window`, forgive earlier crashes.
    let stable = tokio::time::sleep(ctx.restart.stable_window);
    tokio::pin!(stable);
    let mut stable_done = false;
    loop {
        tokio::select! {
            biased;
            _ = ctx.cancel.cancelled() => return Monitor::Cancelled,
            exit = &mut handle.exit => {
                return Monitor::Exited(exit.ok().and_then(|e| e.code));
            }
            _ = &mut stable, if !stable_done => {
                stable_done = true;
                *attempts = 0;
                ctx.shared.update(|s| s.restarts = 0);
            }
            line = handle.logs.recv(), if logs_open => match line {
                Some(l) => ctx.shared.push_log(l),
                None => logs_open = false,
            },
            _ = interval.tick() => {
                match probe_health(ctx.deps.http.as_ref(), endpoint).await {
                    Health::Ready => {
                        failures = 0;
                        ctx.shared.update(|s| {
                            if matches!(s.state, ServerState::Degraded { .. }) {
                                s.state = ServerState::Ready;
                            }
                        });
                    }
                    other => {
                        failures += 1;
                        if failures >= ctx.timing.failures_before_degraded {
                            let reason = match other {
                                Health::Down(d) => d,
                                _ => "health endpoint reports loading".to_string(),
                            };
                            ctx.shared.update(|s| {
                                if s.state == ServerState::Ready {
                                    s.state = ServerState::Degraded { reason };
                                }
                            });
                        }
                    }
                }
            }
        }
    }
}

/// Requests a graceful stop and waits (bounded) for the child to exit.
async fn shutdown(handle: ProcessHandle, grace: Duration) {
    let ProcessHandle { control, exit, .. } = handle;
    if let Err(e) = control.request_stop(grace).await {
        tracing::warn!(error = %e, "stop request failed");
    }
    if tokio::time::timeout(grace + Duration::from_secs(5), exit)
        .await
        .is_err()
    {
        tracing::error!("llama-server did not report exit after a forced kill request");
    }
}

/// Owns the child across its lifetime: monitors it, and restarts it with
/// backoff after an unexpected exit, up to `RestartPolicy::max_restarts`.
async fn supervise(ctx: Ctx, cfg: LaunchConfig, spec: SpawnSpec, first: ProcessHandle) {
    let endpoint = cfg.endpoint();
    let mut attempts = 0u32;
    let mut current = Some(first);
    loop {
        if let Some(mut handle) = current.take() {
            match monitor(&ctx, &endpoint, &mut handle, &mut attempts).await {
                Monitor::Cancelled => {
                    shutdown(handle, ctx.timing.stop_grace).await;
                    ctx.shared.update(|s| {
                        s.state = ServerState::Stopped;
                        s.pid = None;
                        s.endpoint = None;
                    });
                    return;
                }
                Monitor::Exited(code) => {
                    let diag = diagnose_log(&ctx.shared.log_text())
                        .map(|d| format!(": {}", d.message()))
                        .unwrap_or_default();
                    let reason = format!(
                        "process exited (code {}){diag}",
                        code.map_or_else(|| "unknown".to_string(), |c| c.to_string())
                    );
                    tracing::warn!(%reason, "llama-server crashed");
                    ctx.shared.update(|s| {
                        s.state = ServerState::Crashed { reason };
                        s.pid = None;
                    });
                }
            }
        }

        if attempts >= ctx.restart.max_restarts {
            tracing::error!(attempts, "restart limit reached; server stays crashed");
            return;
        }
        attempts += 1;
        let delay = ctx.restart.backoff(attempts);
        tokio::select! {
            _ = ctx.cancel.cancelled() => {
                ctx.shared.update(|s| s.state = ServerState::Stopped);
                return;
            }
            _ = tokio::time::sleep(delay) => {}
        }

        ctx.shared.update(|s| {
            s.state = ServerState::Starting;
            s.restarts = attempts;
        });
        match start_child(&ctx, &cfg, &spec).await {
            Ok(h) => {
                current = Some(h);
                ctx.shared.update(|s| s.state = ServerState::Ready);
            }
            Err(e) => {
                if ctx.cancel.is_cancelled() {
                    ctx.shared.update(|s| s.state = ServerState::Stopped);
                    return;
                }
                let reason = format!("restart attempt {attempts} failed: {e}");
                tracing::warn!(%reason);
                ctx.shared
                    .update(|s| s.state = ServerState::Crashed { reason });
            }
        }
    }
}

/// Config, binary and the extra environment (not the config's own) of the last launch.
type LastLaunch = (LaunchConfig, PathBuf, Vec<(String, String)>);

/// Manages one llama-server process at a time.
pub struct ServerManager {
    deps: ManagerDeps,
    restart: RestartPolicy,
    timing: Timing,
    shared: Arc<Shared>,
    /// Serializes start/stop and holds the supervisor task.
    session: AsyncMutex<Option<JoinHandle<()>>>,
    /// Last launch, kept for restarts.
    last: Mutex<Option<LastLaunch>>,
}

impl ServerManager {
    /// Creates a manager in the Stopped state.
    pub fn new(deps: ManagerDeps, restart: RestartPolicy, timing: Timing) -> Self {
        let shared = Arc::new(Shared::new(deps.sink.clone()));
        Self {
            deps,
            restart,
            timing,
            shared,
            session: AsyncMutex::new(None),
            last: Mutex::new(None),
        }
    }

    fn ctx(&self, cancel: CancellationToken) -> Ctx {
        Ctx {
            shared: self.shared.clone(),
            deps: self.deps.clone(),
            restart: self.restart,
            timing: self.timing,
            cancel,
        }
    }

    /// Starts llama-server with `cfg` and returns once it is Ready.
    ///
    /// Fails without spawning when the port is busy, and fails with a log-based
    /// diagnosis when the process exits or does not become ready in time.
    pub async fn start(&self, cfg: LaunchConfig, binary: &Path) -> Result<ServerStatus> {
        self.start_with_env(cfg, binary, Vec::new()).await
    }

    /// Like [`ServerManager::start`], with `extra_env` added to the child's
    /// environment (for example a PATH that includes the HIP SDK bin folders).
    /// The variables are kept for restarts.
    pub async fn start_with_env(
        &self,
        cfg: LaunchConfig,
        binary: &Path,
        extra_env: Vec<(String, String)>,
    ) -> Result<ServerStatus> {
        cfg.validate()?;
        let mut session = self.session.lock().await;
        if session.as_ref().is_some_and(|h| !h.is_finished()) {
            return Err(RaptorError::Conflict(
                "server is already running; stop it first".into(),
            ));
        }
        *session = None;
        if !self.deps.ports.is_free(&cfg.host, cfg.port) {
            return Err(RaptorError::Conflict(format!(
                "port {} on {} is already in use",
                cfg.port, cfg.host
            )));
        }

        let cancel = CancellationToken::new();
        self.shared.clear_log();
        self.shared.set_active(cancel.clone());
        *lock(&self.last) = Some((cfg.clone(), binary.to_path_buf(), extra_env.clone()));
        let mut env = cfg.env();
        env.extend(extra_env);
        let endpoint = cfg.endpoint();
        self.shared.update(|s| {
            s.state = ServerState::Starting;
            s.endpoint = Some(endpoint.clone());
            s.pid = None;
            s.restarts = 0;
        });

        let spec = SpawnSpec {
            program: binary.to_path_buf(),
            args: cfg.args(),
            env,
        };
        let ctx = self.ctx(cancel.clone());
        match start_child(&ctx, &cfg, &spec).await {
            Ok(handle) => {
                let pid = handle.control.pid();
                self.shared.update(|s| {
                    s.state = ServerState::Ready;
                    s.pid = pid;
                });
                let task = tokio::spawn(supervise(ctx, cfg, spec, handle));
                *session = Some(task);
                Ok(self.shared.snapshot())
            }
            Err(e) => {
                let state = if cancel.is_cancelled() {
                    ServerState::Stopped
                } else {
                    ServerState::Crashed {
                        reason: e.to_string(),
                    }
                };
                self.shared.update(|s| {
                    s.state = state;
                    s.pid = None;
                    s.endpoint = None;
                });
                Err(e)
            }
        }
    }

    /// Stops the server (idempotent). Cancels any start or restart in progress.
    pub async fn stop(&self) -> Result<()> {
        self.shared.cancel_active();
        let mut session = self.session.lock().await;
        if let Some(task) = session.take() {
            let limit = self.timing.stop_grace + Duration::from_secs(15);
            match tokio::time::timeout(limit, task).await {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    return Err(RaptorError::Internal(format!(
                        "supervisor task failed: {e}"
                    )))
                }
                Err(_) => {
                    return Err(RaptorError::Runtime(
                        "llama-server did not stop in time".into(),
                    ))
                }
            }
        }
        self.shared.update(|s| {
            s.state = ServerState::Stopped;
            s.pid = None;
            s.endpoint = None;
        });
        Ok(())
    }

    /// Stops and starts again with the last launch parameters.
    pub async fn restart(&self) -> Result<ServerStatus> {
        let (cfg, binary, extra_env) = lock(&self.last)
            .clone()
            .ok_or_else(|| RaptorError::NotFound("no previous launch to restart".into()))?;
        self.stop().await?;
        self.start_with_env(cfg, &binary, extra_env).await
    }

    /// Current state snapshot. Never blocks on I/O.
    pub fn status(&self) -> ServerStatus {
        self.shared.snapshot()
    }

    /// The newest `n` log lines, oldest first.
    pub fn log_tail(&self, n: usize) -> Vec<LogLine> {
        lock(&self.shared.log).tail(n)
    }

    /// All retained log lines as one string (for diagnostics and GPU verification).
    pub fn log_text(&self) -> String {
        self.shared.log_text()
    }
}

/// Reads one pipe line by line (lossy UTF-8) into the channel until EOF.
async fn pump<R: AsyncRead + Unpin + Send + 'static>(
    reader: R,
    stream: LogStream,
    tx: mpsc::UnboundedSender<LogLine>,
) {
    let mut reader = BufReader::new(reader);
    let mut buf = Vec::new();
    loop {
        buf.clear();
        match reader.read_until(b'\n', &mut buf).await {
            Ok(0) | Err(_) => break,
            Ok(_) => {
                let text = String::from_utf8_lossy(&buf)
                    .trim_end_matches(['\r', '\n'])
                    .to_string();
                if tx.send(LogLine { stream, text }).is_err() {
                    break;
                }
            }
        }
    }
}

/// Sends a stop signal to the whole process tree. Non-forced first, forced later.
async fn signal_tree(pid: Option<u32>, force: bool) {
    let Some(pid) = pid else { return };
    let mut cmd = if cfg!(windows) {
        let mut c = tokio::process::Command::new("taskkill");
        c.arg("/PID").arg(pid.to_string()).arg("/T");
        if force {
            c.arg("/F");
        }
        c
    } else {
        let mut c = tokio::process::Command::new("kill");
        c.arg(if force { "-KILL" } else { "-TERM" })
            .arg(format!("-{pid}"));
        c
    };
    cmd.stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    #[cfg(windows)]
    cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    match cmd.status().await {
        Ok(st) if st.success() => {}
        Ok(st) => {
            tracing::debug!(pid, force, status = %st, "signal_tree returned non-zero (process may have exited)")
        }
        Err(e) => tracing::warn!(pid, force, error = %e, "could not signal process tree"),
    }
}

async fn own_child(
    mut child: tokio::process::Child,
    pid: Option<u32>,
    mut stop_rx: mpsc::UnboundedReceiver<Duration>,
    exit_tx: oneshot::Sender<ProcessExit>,
) {
    let status = tokio::select! {
        st = child.wait() => st,
        req = stop_rx.recv() => match req {
            Some(grace) => {
                signal_tree(pid, false).await;
                match tokio::time::timeout(grace, child.wait()).await {
                    Ok(st) => st,
                    Err(_) => {
                        signal_tree(pid, true).await;
                        child.wait().await
                    }
                }
            }
            None => child.wait().await,
        },
    };
    let code = status.ok().and_then(|s| s.code());
    let _ = exit_tx.send(ProcessExit { code });
}

struct SystemControl {
    pid: Option<u32>,
    stop_tx: mpsc::UnboundedSender<Duration>,
}

#[async_trait]
impl ProcessControl for SystemControl {
    fn pid(&self) -> Option<u32> {
        self.pid
    }

    async fn request_stop(&self, grace: Duration) -> Result<()> {
        // A closed channel means the child already exited, which is the goal.
        let _ = self.stop_tx.send(grace);
        Ok(())
    }
}

/// Real spawner. `kill_on_drop` plus a process group (Unix) or taskkill /T (Windows)
/// keeps llama-server and its helpers from outliving Raptor LLM.
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemSpawner;

#[async_trait]
impl Spawner for SystemSpawner {
    async fn spawn(&self, spec: &SpawnSpec) -> Result<ProcessHandle> {
        use std::process::Stdio;
        let mut cmd = tokio::process::Command::new(&spec.program);
        cmd.args(&spec.args)
            .envs(spec.env.clone())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        #[cfg(unix)]
        cmd.process_group(0);
        #[cfg(windows)]
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW

        let mut child = cmd.spawn().map_err(|e| {
            RaptorError::Runtime(format!("cannot start {}: {e}", spec.program.display()))
        })?;
        let pid = child.id();
        let (log_tx, logs) = mpsc::unbounded_channel();
        if let Some(out) = child.stdout.take() {
            tokio::spawn(pump(out, LogStream::Stdout, log_tx.clone()));
        }
        if let Some(err) = child.stderr.take() {
            tokio::spawn(pump(err, LogStream::Stderr, log_tx.clone()));
        }
        drop(log_tx);

        let (stop_tx, stop_rx) = mpsc::unbounded_channel();
        let (exit_tx, exit) = oneshot::channel();
        tokio::spawn(own_child(child, pid, stop_rx, exit_tx));
        Ok(ProcessHandle {
            control: Box::new(SystemControl { pid, stop_tx }),
            logs,
            exit,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LaunchConfig;
    use crate::manifest::Backend;
    use crate::testing::{FakeHttp, FakeLaunch, FakePorts, FakeSpawner, Step};

    fn cfg() -> LaunchConfig {
        LaunchConfig::builder("C:/m/model.gguf", Backend::Cpu)
            .build()
            .unwrap()
    }

    fn timing() -> Timing {
        Timing {
            startup_timeout: Duration::from_secs(5),
            poll_interval: Duration::from_millis(500),
            health_interval: Duration::from_secs(1),
            stop_grace: Duration::from_secs(2),
            failures_before_degraded: 2,
        }
    }

    fn policy(max: u32) -> RestartPolicy {
        RestartPolicy {
            max_restarts: max,
            base_backoff: Duration::from_secs(1),
            max_backoff: Duration::from_secs(4),
            stable_window: Duration::from_secs(60),
        }
    }

    fn healthy_http() -> FakeHttp {
        FakeHttp::new()
            .route("/health", vec![Step::Status(200)])
            .route("/v1/models", vec![Step::Status(200)])
    }

    fn manager(
        spawner: Arc<FakeSpawner>,
        http: FakeHttp,
        ports_free: bool,
        max_restarts: u32,
    ) -> ServerManager {
        let deps = ManagerDeps {
            spawner,
            http: Arc::new(http),
            ports: Arc::new(FakePorts { free: ports_free }),
            sink: Arc::new(NullSink),
        };
        ServerManager::new(deps, policy(max_restarts), timing())
    }

    #[test]
    fn backoff_doubles_and_caps() {
        let p = RestartPolicy {
            max_restarts: 9,
            base_backoff: Duration::from_secs(2),
            max_backoff: Duration::from_secs(30),
            stable_window: Duration::from_secs(60),
        };
        assert_eq!(p.backoff(1), Duration::from_secs(2));
        assert_eq!(p.backoff(2), Duration::from_secs(4));
        assert_eq!(p.backoff(3), Duration::from_secs(8));
        assert_eq!(p.backoff(5), Duration::from_secs(30));
        assert_eq!(p.backoff(50), Duration::from_secs(30));
        assert_eq!(p.backoff(0), Duration::from_secs(2));
    }

    #[test]
    fn diagnosis_table() {
        let cases: &[(&str, Option<LaunchDiagnosis>)] = &[
            (
                "ggml_backend_cuda: cudaMalloc failed: out of memory",
                Some(LaunchDiagnosis::OutOfMemory),
            ),
            (
                "llama_model_load: error loading model: unable to allocate CUDA0 buffer",
                Some(LaunchDiagnosis::OutOfMemory),
            ),
            (
                "error: unknown model architecture: 'qwen9000'",
                Some(LaunchDiagnosis::UnsupportedArchitecture(
                    "error: unknown model architecture: 'qwen9000'".into(),
                )),
            ),
            (
                "gguf_init_from_file: failed to open 'x.gguf': No such file or directory",
                Some(LaunchDiagnosis::ModelFileMissing),
            ),
            (
                "srv  start: couldn't bind HTTP server socket, hostname: 127.0.0.1, port: 8080",
                Some(LaunchDiagnosis::PortInUse),
            ),
            ("main: server is listening on http://127.0.0.1:8080", None),
        ];
        for (line, want) in cases {
            assert_eq!(diagnose_log(line), *want, "line: {line}");
        }
        assert!(matches!(
            diagnose_log("llama_model_load_from_file: failed to load model"),
            Some(LaunchDiagnosis::Other(_))
        ));
        assert_eq!(diagnose_log(""), None);
    }

    #[test]
    fn specific_cause_wins_over_generic_error() {
        let log = "llama_model_load: error loading model: something odd\nggml: hipMalloc failed: out of memory\n";
        assert_eq!(diagnose_log(log), Some(LaunchDiagnosis::OutOfMemory));
    }

    #[test]
    fn log_buffer_is_bounded_and_ordered() {
        let mut b = LogBuffer::new(3);
        for i in 0..5 {
            b.push(LogLine {
                stream: LogStream::Stdout,
                text: format!("l{i}"),
            });
        }
        assert_eq!(b.len(), 3);
        assert_eq!(b.text(), "l2\nl3\nl4");
        let tail = b.tail(2);
        assert_eq!(
            tail.iter().map(|l| l.text.as_str()).collect::<Vec<_>>(),
            vec!["l3", "l4"]
        );
        assert!(!b.is_empty());
    }

    #[tokio::test(start_paused = true)]
    async fn start_reaches_ready_and_stop_returns_to_stopped() {
        let spawner = Arc::new(FakeSpawner::new(vec![FakeLaunch::lines(&[
            "main: server is listening",
        ])]));
        let mgr = manager(spawner.clone(), healthy_http(), true, 3);
        let st = mgr.start(cfg(), Path::new("llama-server")).await.unwrap();
        assert_eq!(st.state, ServerState::Ready);
        assert_eq!(st.endpoint.as_deref(), Some("http://127.0.0.1:8080"));
        assert!(st.pid.is_some());
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert!(mgr.log_text().contains("server is listening"));
        mgr.stop().await.unwrap();
        assert_eq!(mgr.status().state, ServerState::Stopped);
        assert_eq!(mgr.status().endpoint, None);
        assert_eq!(spawner.spawn_count(), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn a_new_server_starts_with_a_fresh_log() {
        let spawner = Arc::new(FakeSpawner::new(vec![
            FakeLaunch::lines(&["load_tensors: ROCm0 model buffer size = 14220.00 MiB"]),
            FakeLaunch::lines(&["load_tensors: ROCm0 model buffer size = 12289.00 MiB"]),
        ]));
        let mgr = manager(spawner, healthy_http(), true, 0);
        mgr.start(cfg(), Path::new("llama-server")).await.unwrap();
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert!(mgr.log_text().contains("14220"));
        mgr.stop().await.unwrap();

        mgr.start(cfg(), Path::new("llama-server")).await.unwrap();
        tokio::time::sleep(Duration::from_millis(10)).await;
        let log = mgr.log_text();
        assert!(log.contains("12289"));
        assert!(
            !log.contains("14220"),
            "the previous model's lines must not count: {log}"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn passes_args_and_env_without_secret_in_args() {
        let spawner = Arc::new(FakeSpawner::new(vec![]));
        let mgr = manager(spawner.clone(), healthy_http(), true, 0);
        let cfg = LaunchConfig::builder("C:/m/model.gguf", Backend::Cpu)
            .host("0.0.0.0")
            .api_key("tok123")
            .build()
            .unwrap();
        mgr.start(cfg, Path::new("llama-server")).await.unwrap();
        let spec = spawner.last_spec().unwrap();
        assert!(spec.args.iter().any(|a| a == "--host"));
        assert!(!spec.args.iter().any(|a| a.contains("tok123")));
        assert_eq!(
            spec.env,
            vec![("LLAMA_API_KEY".to_string(), "tok123".to_string())]
        );
        mgr.stop().await.unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn waits_through_503_loading_until_ready() {
        let http = FakeHttp::new()
            .route(
                "/health",
                vec![Step::Status(503), Step::Status(503), Step::Status(200)],
            )
            .route("/v1/models", vec![Step::Status(200)]);
        let spawner = Arc::new(FakeSpawner::new(vec![]));
        let mgr = manager(spawner.clone(), http, true, 0);
        let st = mgr.start(cfg(), Path::new("llama-server")).await.unwrap();
        assert_eq!(st.state, ServerState::Ready);
        assert_eq!(spawner.spawn_count(), 1);
        mgr.stop().await.unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn port_in_use_fails_before_spawning() {
        let spawner = Arc::new(FakeSpawner::new(vec![]));
        let mgr = manager(spawner.clone(), healthy_http(), false, 0);
        let err = mgr
            .start(cfg(), Path::new("llama-server"))
            .await
            .unwrap_err();
        assert!(matches!(err, RaptorError::Conflict(m) if m.contains("8080")));
        assert_eq!(spawner.spawn_count(), 0);
        assert_eq!(mgr.status().state, ServerState::Stopped);
    }

    #[tokio::test(start_paused = true)]
    async fn health_timeout_stops_child_and_reports_oom_diagnosis() {
        let http = FakeHttp::new().route("/health", vec![Step::Status(503)]);
        let spawner = Arc::new(FakeSpawner::new(vec![FakeLaunch::lines(&[
            "llama_model_load: error loading model: unable to allocate CUDA0 buffer",
        ])]));
        let mgr = manager(spawner.clone(), http, true, 0);
        let err = mgr
            .start(cfg(), Path::new("llama-server"))
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("not ready after"), "{msg}");
        assert!(msg.contains("out of memory"), "{msg}");
        assert!(matches!(mgr.status().state, ServerState::Crashed { .. }));
        assert_eq!(
            spawner.stop_count(),
            1,
            "the half-started child must be stopped"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn process_exit_during_startup_is_reported_with_missing_file_diagnosis() {
        let http = FakeHttp::new().route("/health", vec![Step::Refused]);
        let spawner = Arc::new(FakeSpawner::new(vec![FakeLaunch {
            lines: vec![
                "gguf_init_from_file: failed to open 'x.gguf': No such file or directory".into(),
            ],
            exit_after: Some(Duration::from_millis(100)),
            exit_code: 1,
            spawn_error: None,
        }]));
        let mgr = manager(spawner, http, true, 0);
        let err = mgr
            .start(cfg(), Path::new("llama-server"))
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("exited during startup"), "{msg}");
        assert!(msg.contains("model file not found"), "{msg}");
    }

    #[tokio::test(start_paused = true)]
    async fn spawn_failure_is_surfaced_as_crashed_state() {
        let spawner = Arc::new(FakeSpawner::new(vec![FakeLaunch::failing(
            "no such binary",
        )]));
        let mgr = manager(spawner, healthy_http(), true, 0);
        assert!(mgr.start(cfg(), Path::new("missing")).await.is_err());
        assert!(
            matches!(mgr.status().state, ServerState::Crashed { reason } if reason.contains("no such binary"))
        );
    }

    #[tokio::test(start_paused = true)]
    async fn crash_then_auto_restart_recovers() {
        let spawner = Arc::new(FakeSpawner::new(vec![FakeLaunch {
            lines: vec![],
            exit_after: Some(Duration::from_secs(1)),
            exit_code: 1,
            spawn_error: None,
        }]));
        let mgr = manager(spawner.clone(), healthy_http(), true, 3);
        mgr.start(cfg(), Path::new("llama-server")).await.unwrap();
        // Crash at t=1s, restart after 1s backoff, ready again shortly after.
        tokio::time::sleep(Duration::from_secs(5)).await;
        let st = mgr.status();
        assert_eq!(st.state, ServerState::Ready);
        assert_eq!(st.restarts, 1);
        assert_eq!(spawner.spawn_count(), 2);
        mgr.stop().await.unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn crash_loop_stops_at_restart_limit() {
        let crash = || FakeLaunch {
            lines: vec![],
            exit_after: Some(Duration::from_millis(100)),
            exit_code: 2,
            spawn_error: None,
        };
        let spawner = Arc::new(FakeSpawner::new(vec![crash(), crash(), crash(), crash()]));
        let mgr = manager(spawner.clone(), healthy_http(), true, 2);
        mgr.start(cfg(), Path::new("llama-server")).await.unwrap();
        tokio::time::sleep(Duration::from_secs(30)).await;
        let st = mgr.status();
        assert!(
            matches!(st.state, ServerState::Crashed { ref reason } if reason.contains("code 2")),
            "{st:?}"
        );
        assert_eq!(st.restarts, 2);
        assert_eq!(spawner.spawn_count(), 3, "one launch plus two restarts");
    }

    #[tokio::test(start_paused = true)]
    async fn stop_during_restart_backoff_cancels_the_restart() {
        let spawner = Arc::new(FakeSpawner::new(vec![FakeLaunch {
            lines: vec![],
            exit_after: Some(Duration::from_millis(100)),
            exit_code: 1,
            spawn_error: None,
        }]));
        let mgr = manager(spawner.clone(), healthy_http(), true, 3);
        mgr.start(cfg(), Path::new("llama-server")).await.unwrap();
        tokio::time::sleep(Duration::from_millis(500)).await; // crashed, waiting in backoff
        mgr.stop().await.unwrap();
        tokio::time::sleep(Duration::from_secs(10)).await;
        assert_eq!(
            spawner.spawn_count(),
            1,
            "no restart after an explicit stop"
        );
        assert_eq!(mgr.status().state, ServerState::Stopped);
    }

    #[tokio::test(start_paused = true)]
    async fn repeated_health_failures_degrade_then_recover() {
        let http = FakeHttp::new()
            .route(
                "/health",
                vec![
                    Step::Status(200),
                    Step::Status(503),
                    Step::Status(503),
                    Step::Status(200),
                ],
            )
            .route("/v1/models", vec![Step::Status(200)]);
        let spawner = Arc::new(FakeSpawner::new(vec![]));
        let mgr = manager(spawner.clone(), http, true, 0);
        mgr.start(cfg(), Path::new("llama-server")).await.unwrap();
        tokio::time::sleep(Duration::from_millis(2500)).await;
        assert!(
            matches!(mgr.status().state, ServerState::Degraded { .. }),
            "{:?}",
            mgr.status()
        );
        tokio::time::sleep(Duration::from_millis(1000)).await;
        assert_eq!(mgr.status().state, ServerState::Ready);
        assert_eq!(
            spawner.spawn_count(),
            1,
            "degradation must not restart the process"
        );
        mgr.stop().await.unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn second_start_while_running_is_conflict() {
        let spawner = Arc::new(FakeSpawner::new(vec![]));
        let mgr = manager(spawner.clone(), healthy_http(), true, 0);
        mgr.start(cfg(), Path::new("llama-server")).await.unwrap();
        let err = mgr
            .start(cfg(), Path::new("llama-server"))
            .await
            .unwrap_err();
        assert!(matches!(err, RaptorError::Conflict(_)));
        assert_eq!(spawner.spawn_count(), 1);
        mgr.stop().await.unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn restart_reuses_last_launch_and_stop_is_idempotent() {
        let spawner = Arc::new(FakeSpawner::new(vec![]));
        let mgr = manager(spawner.clone(), healthy_http(), true, 0);
        assert!(matches!(mgr.restart().await, Err(RaptorError::NotFound(_))));
        mgr.start(cfg(), Path::new("llama-server")).await.unwrap();
        let st = mgr.restart().await.unwrap();
        assert_eq!(st.state, ServerState::Ready);
        assert_eq!(spawner.spawn_count(), 2);
        mgr.stop().await.unwrap();
        mgr.stop().await.unwrap();
        assert_eq!(mgr.status().state, ServerState::Stopped);
    }

    #[tokio::test(start_paused = true)]
    async fn child_output_is_forwarded_to_sink() {
        use crate::testing::MemorySink;
        let sink = Arc::new(MemorySink::default());
        let spawner = Arc::new(FakeSpawner::new(vec![FakeLaunch::lines(&[
            "hello one",
            "hello two",
        ])]));
        let deps = ManagerDeps {
            spawner,
            http: Arc::new(healthy_http()),
            ports: Arc::new(FakePorts { free: true }),
            sink: sink.clone(),
        };
        let mgr = ServerManager::new(deps, policy(0), timing());
        mgr.start(cfg(), Path::new("llama-server")).await.unwrap();
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert_eq!(
            sink.texts(),
            vec!["hello one".to_string(), "hello two".to_string()]
        );
        let tail = mgr.log_tail(1);
        assert_eq!(tail[0].text, "hello two");
        mgr.stop().await.unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn restart_count_resets_after_sustained_ready() {
        let spawner = Arc::new(FakeSpawner::new(vec![FakeLaunch {
            lines: vec![],
            exit_after: Some(Duration::from_secs(1)),
            exit_code: 1,
            spawn_error: None,
        }]));
        let deps = ManagerDeps {
            spawner: spawner.clone(),
            http: Arc::new(healthy_http()),
            ports: Arc::new(FakePorts { free: true }),
            sink: Arc::new(NullSink),
        };
        let restart = RestartPolicy {
            max_restarts: 3,
            base_backoff: Duration::from_secs(1),
            max_backoff: Duration::from_secs(4),
            stable_window: Duration::from_secs(3),
        };
        let mgr = ServerManager::new(deps, restart, timing());
        mgr.start(cfg(), Path::new("llama-server")).await.unwrap();
        // Crash at t=1s, restarted at t=2s, Ready and then stable from t=2s.
        tokio::time::sleep(Duration::from_secs(3)).await;
        assert_eq!(mgr.status().restarts, 1, "still inside the stable window");
        tokio::time::sleep(Duration::from_secs(3)).await;
        let st = mgr.status();
        assert_eq!(st.state, ServerState::Ready);
        assert_eq!(st.restarts, 0, "sustained Ready must reset the counter");
        assert_eq!(spawner.spawn_count(), 2);
        mgr.stop().await.unwrap();
    }

    /// Runs a harmless shell command on the host OS (no GPU, no network).
    fn shell_spec(win: &str, unix: &str) -> SpawnSpec {
        if cfg!(windows) {
            SpawnSpec {
                program: "cmd.exe".into(),
                args: vec!["/C".into(), win.into()],
                env: vec![],
            }
        } else {
            SpawnSpec {
                program: "sh".into(),
                args: vec!["-c".into(), unix.into()],
                env: vec![],
            }
        }
    }

    #[tokio::test]
    async fn system_spawner_captures_output_and_exit_code() {
        let spec = shell_spec("echo raptor-ok & exit /b 3", "echo raptor-ok; exit 3");
        let mut h = SystemSpawner.spawn(&spec).await.unwrap();
        assert!(h.control.pid().is_some());
        let mut lines = Vec::new();
        let exit = tokio::time::timeout(Duration::from_secs(30), async {
            loop {
                tokio::select! {
                    Some(l) = h.logs.recv() => lines.push(l.text),
                    e = &mut h.exit => break e,
                }
            }
        })
        .await
        .expect("child finished in time")
        .expect("exit reported");
        assert_eq!(exit.code, Some(3));
        assert!(lines.iter().any(|l| l.contains("raptor-ok")), "{lines:?}");
    }

    #[tokio::test]
    async fn system_spawner_stop_terminates_a_long_running_child() {
        let spec = shell_spec("ping -n 60 127.0.0.1 > nul", "sleep 60");
        let h = SystemSpawner.spawn(&spec).await.unwrap();
        let started = std::time::Instant::now();
        h.control
            .request_stop(Duration::from_secs(1))
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(30), h.exit)
            .await
            .expect("stop must finish well before the child's own 60s")
            .expect("exit reported");
        assert!(started.elapsed() < Duration::from_secs(30));
    }
}
