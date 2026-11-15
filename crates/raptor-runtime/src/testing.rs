//! Fakes for every external effect. Compiled only for tests.
//! No network, no real processes, no GPU.

use crate::hardware::{HardwareInfo, HardwareProbe};
use crate::http::{HttpClient, HttpResponse};
use crate::install::Fetcher;
use crate::process::{
    LogLine, LogSink, LogStream, ProcessControl, ProcessExit, ProcessHandle, SpawnSpec, Spawner,
};
use crate::sys::{CommandRunner, PortCheck};
use async_trait::async_trait;
use raptor_core::{RaptorError, Result};
use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

/// Hardware probe that returns a fixed snapshot.
pub struct FakeProbe {
    pub hw: HardwareInfo,
}

#[async_trait]
impl HardwareProbe for FakeProbe {
    async fn probe(&self) -> Result<HardwareInfo> {
        Ok(self.hw.clone())
    }
}

/// Command runner keyed by program name. Records the environment of the last call.
#[derive(Default)]
pub struct FakeRunner {
    outputs: Mutex<HashMap<String, std::result::Result<String, String>>>,
    /// Answer for any program without its own entry (engine tests use install paths).
    fallback: Mutex<Option<String>>,
    last_env: Mutex<Vec<(String, String)>>,
}

impl FakeRunner {
    pub fn with_ok(self, program: &str, stdout: &str) -> Self {
        lock(&self.outputs).insert(program.into(), Ok(stdout.into()));
        self
    }
    /// Every program without its own entry succeeds with `stdout`.
    pub fn with_ok_for_all(self, stdout: &str) -> Self {
        *lock(&self.fallback) = Some(stdout.into());
        self
    }
    pub fn with_err(self, program: &str, msg: &str) -> Self {
        lock(&self.outputs).insert(program.into(), Err(msg.into()));
        self
    }
    /// Environment passed to the most recent call.
    pub fn last_env(&self) -> Vec<(String, String)> {
        lock(&self.last_env).clone()
    }
}

#[async_trait]
impl CommandRunner for FakeRunner {
    async fn run(&self, program: &str, args: &[String], timeout: Duration) -> Result<String> {
        self.run_with_env(program, args, &[], timeout).await
    }

    async fn run_with_env(
        &self,
        program: &str,
        _args: &[String],
        env: &[(String, String)],
        _timeout: Duration,
    ) -> Result<String> {
        *lock(&self.last_env) = env.to_vec();
        match lock(&self.outputs).get(program) {
            Some(Ok(out)) => return Ok(out.clone()),
            Some(Err(msg)) => return Err(RaptorError::Runtime(msg.clone())),
            None => {}
        }
        match lock(&self.fallback).as_ref() {
            Some(out) => Ok(out.clone()),
            None => Err(RaptorError::Runtime(format!("{program} not found"))),
        }
    }
}

/// Port check with a fixed answer.
pub struct FakePorts {
    pub free: bool,
}

impl PortCheck for FakePorts {
    fn is_free(&self, _host: &str, _port: u16) -> bool {
        self.free
    }
}

/// Log sink that keeps every line.
#[derive(Default)]
pub struct MemorySink {
    lines: Mutex<Vec<LogLine>>,
}

impl MemorySink {
    pub fn texts(&self) -> Vec<String> {
        lock(&self.lines).iter().map(|l| l.text.clone()).collect()
    }
}

impl LogSink for MemorySink {
    fn record(&self, line: &LogLine) {
        lock(&self.lines).push(line.clone());
    }
}

/// Fetcher that serves in-memory bytes keyed by URL.
#[derive(Default)]
pub struct FakeFetcher {
    files: Mutex<HashMap<String, Vec<u8>>>,
    calls: AtomicUsize,
}

impl FakeFetcher {
    pub fn serve(self, url: &str, bytes: Vec<u8>) -> Self {
        lock(&self.files).insert(url.into(), bytes);
        self
    }
    pub fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl Fetcher for FakeFetcher {
    async fn fetch(&self, url: &str, dest: &Path, cancel: &CancellationToken) -> Result<u64> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if cancel.is_cancelled() {
            return Err(RaptorError::Runtime("operation cancelled".into()));
        }
        let bytes = lock(&self.files)
            .get(url)
            .cloned()
            .ok_or_else(|| RaptorError::Network(format!("HTTP 404 for {url}")))?;
        std::fs::write(dest, &bytes)?;
        Ok(bytes.len() as u64)
    }
}

/// Scripted behaviour for one fake child process.
#[derive(Debug, Clone, Default)]
pub struct FakeLaunch {
    /// Lines printed as soon as the child starts.
    pub lines: Vec<String>,
    /// Exit on its own after this long. `None` runs until stopped.
    pub exit_after: Option<Duration>,
    pub exit_code: i32,
    /// When set, `spawn` fails with this message.
    pub spawn_error: Option<String>,
}

impl FakeLaunch {
    pub fn lines(lines: &[&str]) -> Self {
        Self {
            lines: lines.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }
    }
    pub fn failing(msg: &str) -> Self {
        Self {
            spawn_error: Some(msg.into()),
            ..Default::default()
        }
    }
}

async fn sleep_opt(d: Option<Duration>) {
    match d {
        Some(d) => tokio::time::sleep(d).await,
        None => std::future::pending::<()>().await,
    }
}

/// Spawner that plays back a queue of scripts. An empty queue yields a child
/// that prints nothing and runs until stopped.
pub struct FakeSpawner {
    scripts: Mutex<VecDeque<FakeLaunch>>,
    specs: Mutex<Vec<SpawnSpec>>,
    spawns: AtomicUsize,
    stops: Arc<AtomicUsize>,
}

impl FakeSpawner {
    pub fn new(scripts: Vec<FakeLaunch>) -> Self {
        Self {
            scripts: Mutex::new(scripts.into()),
            specs: Mutex::new(Vec::new()),
            spawns: AtomicUsize::new(0),
            stops: Arc::new(AtomicUsize::new(0)),
        }
    }
    pub fn spawn_count(&self) -> usize {
        self.spawns.load(Ordering::SeqCst)
    }
    pub fn stop_count(&self) -> usize {
        self.stops.load(Ordering::SeqCst)
    }
    pub fn last_spec(&self) -> Option<SpawnSpec> {
        lock(&self.specs).last().cloned()
    }
}

struct FakeControl {
    pid: u32,
    stop_tx: mpsc::UnboundedSender<Duration>,
    stops: Arc<AtomicUsize>,
}

#[async_trait]
impl ProcessControl for FakeControl {
    fn pid(&self) -> Option<u32> {
        Some(self.pid)
    }
    async fn request_stop(&self, grace: Duration) -> Result<()> {
        self.stops.fetch_add(1, Ordering::SeqCst);
        let _ = self.stop_tx.send(grace);
        Ok(())
    }
}

#[async_trait]
impl Spawner for FakeSpawner {
    async fn spawn(&self, spec: &SpawnSpec) -> Result<ProcessHandle> {
        let n = self.spawns.fetch_add(1, Ordering::SeqCst);
        lock(&self.specs).push(spec.clone());
        let script = lock(&self.scripts).pop_front().unwrap_or_default();
        if let Some(msg) = script.spawn_error {
            return Err(RaptorError::Runtime(msg));
        }
        let FakeLaunch {
            lines,
            exit_after,
            exit_code,
            ..
        } = script;
        let (log_tx, logs) = mpsc::unbounded_channel();
        let (exit_tx, exit) = oneshot::channel();
        let (stop_tx, mut stop_rx) = mpsc::unbounded_channel::<Duration>();
        tokio::spawn(async move {
            for text in lines {
                let _ = log_tx.send(LogLine {
                    stream: LogStream::Stderr,
                    text,
                });
            }
            tokio::select! {
                _ = sleep_opt(exit_after) => {
                    let _ = exit_tx.send(ProcessExit { code: Some(exit_code) });
                }
                Some(_) = stop_rx.recv() => {
                    let _ = exit_tx.send(ProcessExit { code: Some(0) });
                }
            }
        });
        let control = FakeControl {
            pid: 1000 + n as u32,
            stop_tx,
            stops: self.stops.clone(),
        };
        Ok(ProcessHandle {
            control: Box::new(control),
            logs,
            exit,
        })
    }
}

/// Builds an in-memory zip archive from (name, bytes) entries.
pub fn make_zip(entries: &[(&str, &[u8])]) -> Vec<u8> {
    use std::io::Write;
    let mut w = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    let opts = zip::write::SimpleFileOptions::default().unix_permissions(0o755);
    for (name, data) in entries {
        w.start_file(*name, opts).expect("start zip entry");
        w.write_all(data).expect("write zip entry");
    }
    w.finish().expect("finish zip").into_inner()
}

/// Builds an in-memory tar.gz archive from (name, bytes) regular-file entries.
pub fn make_tar_gz(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    let mut b = tar::Builder::new(gz);
    for (name, data) in entries {
        let mut h = tar::Header::new_gnu();
        h.set_size(data.len() as u64);
        h.set_mode(0o755);
        b.append_data(&mut h, name, *data)
            .expect("append tar entry");
    }
    b.into_inner()
        .expect("finish tar")
        .finish()
        .expect("finish gzip")
}

/// One scripted HTTP answer. A `Refused` step models a connection error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    Status(u16),
    Refused,
}

struct Route {
    suffix: String,
    steps: Vec<Step>,
    cursor: usize,
}

/// HTTP client keyed by URL suffix. Each route plays its steps in order and
/// repeats the last one. Unmatched URLs are refused.
#[derive(Default)]
pub struct FakeHttp {
    routes: Mutex<Vec<Route>>,
}

impl FakeHttp {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn route(self, suffix: &str, steps: Vec<Step>) -> Self {
        lock(&self.routes).push(Route {
            suffix: suffix.into(),
            steps,
            cursor: 0,
        });
        self
    }
}

#[async_trait]
impl HttpClient for FakeHttp {
    async fn get(&self, url: &str, _timeout: Duration) -> Result<HttpResponse> {
        let mut routes = lock(&self.routes);
        let Some(route) = routes.iter_mut().find(|r| url.ends_with(&r.suffix)) else {
            return Err(RaptorError::Network(format!("connection refused: {url}")));
        };
        let idx = route.cursor.min(route.steps.len().saturating_sub(1));
        route.cursor += 1;
        match route.steps.get(idx).copied() {
            Some(Step::Status(status)) => Ok(HttpResponse {
                status,
                body: String::new(),
            }),
            _ => Err(RaptorError::Network(format!("connection refused: {url}"))),
        }
    }
}
