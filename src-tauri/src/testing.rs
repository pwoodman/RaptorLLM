//! Fakes for the shell's composition tests: hardware, downloads, HTTP, child processes and the
//! tokenizer, plus a GGUF header writer and an in-memory runtime archive. Nothing here touches
//! the GPU, the network or a real llama-server.

use async_trait::async_trait;
use raptor_core::{RaptorError, Result, TokenCounter};
use raptor_runtime::config::GIB;
use raptor_runtime::hardware::{CpuInfo, GpuInfo, GpuVendor, HardwareInfo, HardwareProbe, Os};
use raptor_runtime::hash::sha256_hex;
use raptor_runtime::http::{HttpClient, HttpResponse};
use raptor_runtime::install::Fetcher;
use raptor_runtime::manifest::{Backend, Manifest};
use raptor_runtime::process::{
    LogLine, LogStream, ProcessControl, ProcessExit, ProcessHandle, SpawnSpec, Spawner,
};
use raptor_runtime::sys::{CommandRunner, PortCheck};
use raptor_runtime::tokenizer::TokenizeTransport;
use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

/// URL the fake runtime archive is served from.
pub const HIP_URL: &str = "https://example.invalid/llama-hip-win-b7100.zip";

/// Log lines of a llama-server that offloaded every layer to an RX 9060 XT (HIP).
pub const HIP_LOG: &[&str] = &[
    "ggml_cuda_init: found 1 ROCm devices:",
    "  Device 0: AMD Radeon RX 9060 XT, gfx1200 (0x1200), VMM: no, Wave Size: 32",
    "load_tensors: offloaded 33/33 layers to GPU",
    "load_tensors:        ROCm0 model buffer size =  4168.09 MiB",
    "main: server is listening on http://127.0.0.1:8080",
];

/// What a Vulkan llama-server prints for a fully offloaded model.
pub const VULKAN_LOG: &[&str] = &[
    "ggml_vulkan: Found 1 Vulkan devices:",
    "load_tensors: offloaded 33/33 layers to GPU",
    "load_tensors:      Vulkan0 model buffer size =  4168.09 MiB",
    "main: server is listening on http://127.0.0.1:8080",
];

pub const VULKAN_URL: &str = "https://example.invalid/llama-vulkan-win-b7100.zip";

/// A Vulkan runtime manifest whose sha256 matches `zip`.
pub fn vulkan_manifest(zip: &[u8]) -> Manifest {
    Manifest {
        id: "llama-vulkan-win".into(),
        version: "b7100".into(),
        llama_cpp_build: "b7100".into(),
        backend: Backend::Vulkan,
        os: Os::Windows,
        gfx_archs: Vec::new(),
        url: VULKAN_URL.into(),
        sha256: sha256_hex(zip),
        min_driver: None,
        signature: None,
    }
}

/// `llama-server --list-devices` output for a machine whose ROCm and Vulkan runtimes both see the GPU.
pub const BOTH_DEVICES: &str = "Available devices:\n  ROCm0: AMD Radeon RX 9060 XT (16304 MiB, 16158 MiB free)\n  Vulkan0: AMD Radeon RX 9060 XT (16304 MiB, 16158 MiB free)\n";

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

/// An AMD RX 9060 XT (gfx1200) with 16 GiB VRAM, 12 GiB of it free, on a 32 GiB machine.
pub fn fixed_hardware() -> HardwareInfo {
    HardwareInfo {
        os: Os::Windows,
        cpu: CpuInfo {
            brand: "Ryzen".into(),
            cores: 8,
            threads: 16,
            features: vec![],
        },
        ram_total: 32 * GIB,
        ram_avail: 20 * GIB,
        gpus: vec![GpuInfo {
            vendor: GpuVendor::Amd,
            name: "AMD Radeon RX 9060 XT".into(),
            vram_total: 16 * GIB,
            vram_free: Some(12 * GIB),
            arch: Some("gfx1200".into()),
            driver: Some("32.0.12019.1024".into()),
        }],
    }
}

/// Hardware probe that returns a fixed snapshot and counts its calls.
pub struct FakeProbe {
    pub hw: HardwareInfo,
    pub calls: AtomicUsize,
}

impl FakeProbe {
    pub fn new(hw: HardwareInfo) -> Self {
        Self {
            hw,
            calls: AtomicUsize::new(0),
        }
    }
}

#[async_trait]
impl HardwareProbe for FakeProbe {
    async fn probe(&self) -> Result<HardwareInfo> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(self.hw.clone())
    }
}

/// Fetcher that serves in-memory bytes by URL and counts every call.
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

/// HTTP client where llama-server is always healthy and loaded.
pub struct HealthyHttp;

#[async_trait]
impl HttpClient for HealthyHttp {
    async fn get(&self, url: &str, _timeout: Duration) -> Result<HttpResponse> {
        if url.ends_with("/health") || url.ends_with("/v1/models") {
            return Ok(HttpResponse {
                status: 200,
                body: String::new(),
            });
        }
        Err(RaptorError::Network(format!("connection refused: {url}")))
    }
}

/// Port check that always reports the port as free.
pub struct FreePorts;

impl PortCheck for FreePorts {
    fn is_free(&self, _host: &str, _port: u16) -> bool {
        true
    }
}

/// Command runner for the device preflight: every program answers `--list-devices`
/// with `output`. The default is one ROCm GPU, as on the development machine.
pub struct FakeDeviceRunner {
    pub output: String,
}

impl Default for FakeDeviceRunner {
    fn default() -> Self {
        Self {
            output:
                "Available devices:\n  ROCm0: AMD Radeon RX 9060 XT (16304 MiB, 16158 MiB free)\n"
                    .into(),
        }
    }
}

#[async_trait]
impl CommandRunner for FakeDeviceRunner {
    async fn run(&self, _program: &str, _args: &[String], _timeout: Duration) -> Result<String> {
        Ok(self.output.clone())
    }
}

/// Scripted behaviour of one fake child process.
#[derive(Debug, Clone, Default)]
pub struct FakeLaunch {
    /// Lines printed as soon as the child starts.
    pub lines: Vec<String>,
    /// Exit on its own after this long (a crash). `None` runs until stopped.
    pub exit_after: Option<Duration>,
    /// When set, `spawn` fails with this message.
    pub spawn_error: Option<String>,
}

impl FakeLaunch {
    /// A child that runs until stopped and reports a fully offloaded HIP model.
    pub fn hip() -> Self {
        Self {
            lines: HIP_LOG.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }
    }

    /// A child that runs until stopped and reports a fully offloaded Vulkan model.
    pub fn vulkan() -> Self {
        Self {
            lines: VULKAN_LOG.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }
    }

    /// The same child, but it crashes after `d`.
    pub fn crashing_after(mut self, d: Duration) -> Self {
        self.exit_after = Some(d);
        self
    }

    /// A child that cannot start.
    pub fn failing(msg: &str) -> Self {
        Self {
            spawn_error: Some(msg.into()),
            ..Default::default()
        }
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

/// Spawner that plays back a queue of scripts. An empty queue yields a silent child that
/// runs until stopped.
pub struct FakeSpawner {
    scripts: Mutex<VecDeque<FakeLaunch>>,
    spawns: AtomicUsize,
    stops: Arc<AtomicUsize>,
}

impl FakeSpawner {
    pub fn new(scripts: Vec<FakeLaunch>) -> Self {
        Self {
            scripts: Mutex::new(scripts.into()),
            spawns: AtomicUsize::new(0),
            stops: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub fn spawn_count(&self) -> usize {
        self.spawns.load(Ordering::SeqCst)
    }

    /// Stop requests sent to the children so far.
    pub fn stop_count(&self) -> usize {
        self.stops.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl Spawner for FakeSpawner {
    async fn spawn(&self, _spec: &SpawnSpec) -> Result<ProcessHandle> {
        let n = self.spawns.fetch_add(1, Ordering::SeqCst);
        let script = lock(&self.scripts).pop_front().unwrap_or_default();
        if let Some(msg) = script.spawn_error {
            return Err(RaptorError::Runtime(msg));
        }
        let (log_tx, logs) = mpsc::unbounded_channel();
        // The startup lines are queued before the handle is returned, so the readiness wait
        // drains them deterministically.
        for text in script.lines {
            let _ = log_tx.send(LogLine {
                stream: LogStream::Stderr,
                text,
            });
        }
        let (exit_tx, exit) = oneshot::channel();
        let (stop_tx, mut stop_rx) = mpsc::unbounded_channel::<Duration>();
        let exit_after = script.exit_after;
        tokio::spawn(async move {
            let _log_tx = log_tx;
            tokio::select! {
                _ = async {
                    match exit_after {
                        Some(d) => tokio::time::sleep(d).await,
                        None => std::future::pending::<()>().await,
                    }
                } => {
                    let _ = exit_tx.send(ProcessExit { code: Some(1) });
                }
                Some(_) = stop_rx.recv() => {
                    let _ = exit_tx.send(ProcessExit { code: Some(0) });
                }
            }
        });
        Ok(ProcessHandle {
            control: Box::new(FakeControl {
                pid: 1000 + n as u32,
                stop_tx,
                stops: self.stops.clone(),
            }),
            logs,
            exit,
        })
    }
}

/// Tokenizer transport that returns a fixed token count for every text.
pub struct FixedTransport(pub usize);

impl TokenizeTransport for FixedTransport {
    fn count_tokens(&self, _base: &str, _text: &str) -> Result<usize> {
        Ok(self.0)
    }
}

/// Token counter with a fixed answer and a fixed exactness flag.
pub struct FixedTokenizer {
    n: usize,
    exact: bool,
}

impl FixedTokenizer {
    pub fn exact(n: usize) -> Self {
        Self { n, exact: true }
    }
}

impl TokenCounter for FixedTokenizer {
    fn count(&self, _text: &str) -> usize {
        self.n
    }

    fn is_exact(&self) -> bool {
        self.exact
    }
}

/// A zip archive holding a fake `llama-server.exe` under the release folder.
pub fn llama_zip() -> Vec<u8> {
    use std::io::Write;
    let mut w = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    let opts = zip::write::SimpleFileOptions::default().unix_permissions(0o755);
    w.start_file("llama-b7100/llama-server.exe", opts)
        .expect("start zip entry");
    w.write_all(b"MZ fake binary").expect("write zip entry");
    w.finish().expect("finish zip").into_inner()
}

/// A HIP runtime manifest whose sha256 matches `zip`.
pub fn hip_manifest(zip: &[u8]) -> Manifest {
    Manifest {
        id: "llama-hip-win".into(),
        version: "b7100".into(),
        llama_cpp_build: "b7100".into(),
        backend: Backend::Hip,
        os: Os::Windows,
        gfx_archs: vec!["gfx1200".into()],
        url: HIP_URL.into(),
        sha256: sha256_hex(zip),
        min_driver: None,
        signature: None,
    }
}

fn push_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn kv_str(out: &mut Vec<u8>, key: &str, value: &str) {
    push_str(out, key);
    out.extend_from_slice(&8u32.to_le_bytes());
    push_str(out, value);
}

fn kv_u32(out: &mut Vec<u8>, key: &str, value: u32) {
    push_str(out, key);
    out.extend_from_slice(&4u32.to_le_bytes());
    out.extend_from_slice(&value.to_le_bytes());
}

/// Writes a GGUF v3 file with the header fields the shell needs for sizing: a Llama
/// architecture with 32 layers, 8 KV heads and 128-wide keys. There is no tensor data.
pub fn write_gguf(path: &Path) {
    write_gguf_with(path, &[]);
}

/// Like [`write_gguf`], with extra `u32` header fields (key, value), e.g. an MTP layer count.
pub fn write_gguf_with(path: &Path, extra: &[(&str, u32)]) {
    let mut kvs = Vec::new();
    kv_str(&mut kvs, "general.architecture", "llama");
    kv_str(&mut kvs, "general.name", "Tiny Test Llama");
    kv_u32(&mut kvs, "llama.context_length", 8192);
    kv_u32(&mut kvs, "llama.block_count", 32);
    kv_u32(&mut kvs, "llama.embedding_length", 4096);
    kv_u32(&mut kvs, "llama.attention.head_count", 32);
    kv_u32(&mut kvs, "llama.attention.head_count_kv", 8);
    kv_u32(&mut kvs, "llama.attention.key_length", 128);
    for (key, value) in extra {
        kv_u32(&mut kvs, key, *value);
    }
    let mut file = b"GGUF".to_vec();
    file.extend_from_slice(&3u32.to_le_bytes());
    file.extend_from_slice(&0u64.to_le_bytes());
    file.extend_from_slice(&(8 + extra.len() as u64).to_le_bytes());
    file.extend_from_slice(&kvs);
    std::fs::write(path, file).expect("write gguf");
}

/// GPU memory readings in order; the last one repeats. Empty means every read fails.
pub struct FakeGpuMemory {
    readings: Mutex<Vec<raptor_runtime::gpumem::GpuMemoryUsage>>,
}

impl FakeGpuMemory {
    pub fn new(readings: &[(u64, u64)]) -> Self {
        Self {
            readings: Mutex::new(
                readings
                    .iter()
                    .map(|(d, s)| raptor_runtime::gpumem::GpuMemoryUsage {
                        dedicated_mib: *d,
                        shared_mib: *s,
                    })
                    .collect(),
            ),
        }
    }
}

#[async_trait::async_trait]
impl raptor_runtime::gpumem::GpuMemoryProbe for FakeGpuMemory {
    async fn read(&self) -> raptor_core::Result<raptor_runtime::gpumem::GpuMemoryUsage> {
        let mut r = self.readings.lock().unwrap_or_else(|p| p.into_inner());
        match r.len() {
            0 => Err(raptor_core::RaptorError::Unsupported(
                "no GPU memory counters".into(),
            )),
            1 => Ok(r[0]),
            _ => Ok(r.remove(0)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gguf_fixture_parses_with_the_model_crate() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("tiny.gguf");
        write_gguf(&p);
        let meta = raptor_models::gguf::parse_file(&p).unwrap();
        assert_eq!(meta.block_count, Some(32));
        assert_eq!(meta.head_dim(), Some(128));
    }

    #[test]
    fn llama_zip_matches_the_manifest_hash() {
        let zip = llama_zip();
        assert_eq!(hip_manifest(&zip).sha256, sha256_hex(&zip));
    }

    #[tokio::test]
    async fn fake_spawner_crashes_on_schedule_and_stops_on_request() {
        let spawner = FakeSpawner::new(vec![
            FakeLaunch::hip().crashing_after(Duration::from_millis(5))
        ]);
        let spec = SpawnSpec {
            program: "x".into(),
            args: vec![],
            env: vec![],
        };
        let handle = spawner.spawn(&spec).await.unwrap();
        let exit = handle.exit.await.unwrap();
        assert_eq!(exit.code, Some(1));

        let handle = spawner.spawn(&spec).await.unwrap();
        handle.control.request_stop(Duration::ZERO).await.unwrap();
        assert_eq!(handle.exit.await.unwrap().code, Some(0));
        assert_eq!(spawner.stop_count(), 1);
        assert_eq!(spawner.spawn_count(), 2);
    }
}
