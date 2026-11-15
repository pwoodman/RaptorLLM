//! Device preflight. Before a GPU-backed launch, ask llama-server which
//! accelerators it can actually initialize (`llama-server --list-devices`), and
//! if a GPU backend has none, explain why in a typed [`BackendLoadFailure`].
//!
//! Without this check a HIP runtime whose DLLs fail to load serves requests on
//! the CPU with no error at all. On Windows the most common cause is the
//! `hipblas.dll` / `libhipblas.dll` naming mismatch between the llama.cpp ROCm
//! zip and the AMD HIP SDK, which is diagnosed precisely below. Nothing is ever
//! copied or renamed to work around it.

use crate::manifest::{version_key, Backend};
use crate::sys::CommandRunner;
use raptor_core::{RaptorError, Result};
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::time::Duration;

const LIST_DEVICES_TIMEOUT: Duration = Duration::from_secs(30);

/// One accelerator as llama-server lists it, for example
/// `ROCm0: AMD Radeon RX 9060 XT (16304 MiB, 16158 MiB free)`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RuntimeDevice {
    /// llama.cpp device id, e.g. `ROCm0`, `CUDA0`, `Vulkan1`.
    pub id: String,
    pub name: String,
    pub total_mib: Option<u64>,
    pub free_mib: Option<u64>,
    /// Backend implied by the id prefix. `None` for an unknown prefix.
    pub backend: Option<Backend>,
}

/// Parses the `Available devices:` section of `llama-server --list-devices`.
/// Pure: no I/O. `(none)` and unrecognized lines are skipped.
pub fn parse_list_devices(text: &str) -> Vec<RuntimeDevice> {
    let mut in_section = false;
    let mut out = Vec::new();
    for raw in text.lines() {
        let line = raw.trim_end();
        if line.trim() == "Available devices:" {
            in_section = true;
            continue;
        }
        if !in_section {
            continue;
        }
        if !line.starts_with(char::is_whitespace) && !line.trim().is_empty() {
            break;
        }
        let entry = line.trim();
        if entry.is_empty() || entry == "(none)" {
            continue;
        }
        if let Some(dev) = parse_entry(entry) {
            out.push(dev);
        }
    }
    out
}

fn parse_entry(entry: &str) -> Option<RuntimeDevice> {
    let (id, rest) = entry.split_once(": ")?;
    if id.is_empty() || !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return None;
    }
    // Memory is the trailing "(N MiB, M MiB free)" group. Any other parenthesis
    // belongs to the name, e.g. "Some GPU (gfx9)".
    let memory = rest.rfind(" (").and_then(|i| {
        let (total, free) = parse_memory(rest[i + 2..].strip_suffix(')')?);
        total.map(|t| (i, t, free))
    });
    let (name, total_mib, free_mib) = match memory {
        Some((i, total, free)) => (rest[..i].trim().to_string(), Some(total), free),
        None => (rest.trim().to_string(), None, None),
    };
    Some(RuntimeDevice {
        id: id.to_string(),
        name,
        total_mib,
        free_mib,
        backend: backend_for_id(id),
    })
}

/// "16304 MiB, 16158 MiB free" -> (Some(16304), Some(16158))
fn parse_memory(s: &str) -> (Option<u64>, Option<u64>) {
    let mut parts = s.split(", ");
    let total = parts.next().and_then(leading_number);
    let free = parts.next().and_then(leading_number);
    (total, free)
}

fn leading_number(s: &str) -> Option<u64> {
    s.split_whitespace().next()?.parse().ok()
}

fn backend_for_id(id: &str) -> Option<Backend> {
    if id.starts_with("ROCm") || id.starts_with("HIP") {
        Some(Backend::Hip)
    } else if id.starts_with("CUDA") {
        Some(Backend::Cuda)
    } else if id.starts_with("Vulkan") {
        Some(Backend::Vulkan)
    } else if id.starts_with("CPU") {
        Some(Backend::Cpu)
    } else {
        None
    }
}

/// Where the AMD HIP SDK may be installed. Windows only: on other systems the
/// search is empty because HIP SDK bin folders do not apply.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct HipSdkSearch {
    /// Folders that contain versioned SDK installs (`<root>\<version>\bin`).
    pub roots: Vec<PathBuf>,
    /// The `HIP_PATH` variable, checked first when it points at an install.
    pub hip_path: Option<PathBuf>,
}

impl HipSdkSearch {
    /// The real search on Windows: `%ProgramFiles%\AMD\ROCm` and `HIP_PATH`.
    /// Empty on other platforms.
    pub fn system() -> Self {
        if !cfg!(windows) {
            return Self::default();
        }
        let program_files = std::env::var_os("ProgramFiles")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(r"C:\Program Files"));
        Self {
            roots: vec![program_files.join("AMD").join("ROCm")],
            hip_path: std::env::var_os("HIP_PATH").map(PathBuf::from),
        }
    }

    /// Existing SDK `bin` folders: `HIP_PATH\bin` first, then each root's
    /// versions newest first. Duplicates are removed.
    pub fn bin_dirs(&self) -> Vec<PathBuf> {
        let mut out: Vec<PathBuf> = Vec::new();
        if let Some(hp) = &self.hip_path {
            push_dir(hp.join("bin"), &mut out);
        }
        for root in &self.roots {
            let Ok(entries) = std::fs::read_dir(root) else {
                continue;
            };
            let mut versions: Vec<(Vec<u64>, PathBuf)> = entries
                .flatten()
                .filter_map(|e| {
                    let name = e.file_name().to_string_lossy().into_owned();
                    let bin = e.path().join("bin");
                    bin.is_dir().then(|| (version_key(&name), bin))
                })
                .collect();
            versions.sort_by(|a, b| b.0.cmp(&a.0));
            for (_, bin) in versions {
                push_dir(bin, &mut out);
            }
        }
        out
    }
}

/// Adds `p` when it is an existing folder not already listed.
fn push_dir(p: PathBuf, out: &mut Vec<PathBuf>) {
    if p.is_dir() && !out.contains(&p) {
        out.push(p);
    }
}

/// The PATH value for a child process: `prefix` folders first, then the current
/// PATH. Empty when there is nothing to prepend, so the child inherits PATH unchanged.
pub fn launch_env(prefix: &[PathBuf], current_path: Option<&str>) -> Result<Vec<(String, String)>> {
    if prefix.is_empty() {
        return Ok(Vec::new());
    }
    let mut parts: Vec<PathBuf> = prefix.to_vec();
    if let Some(cur) = current_path {
        parts.extend(std::env::split_paths(cur));
    }
    let joined = std::env::join_paths(parts).map_err(|e| {
        RaptorError::InvalidInput(format!("an SDK folder path cannot be placed on PATH: {e}"))
    })?;
    Ok(vec![(
        "PATH".to_string(),
        joined.to_string_lossy().into_owned(),
    )])
}

/// Facts about the DLLs a HIP runtime folder needs. Gathered from the file
/// system only; nothing is loaded or executed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct HipDependencies {
    pub runtime_dir: PathBuf,
    /// Whether `ggml-hip.dll` names `hipblas.dll` as an import. `None` when `ggml-hip.dll` is absent.
    pub backend_imports_hipblas: Option<bool>,
    /// `hipblas.dll` is in the runtime folder.
    pub hipblas_in_runtime_dir: bool,
    /// First SDK bin folder that contains `hipblas.dll`.
    pub hipblas_in_sdk: Option<PathBuf>,
    /// First SDK bin folder that contains `libhipblas.dll`, the name HIP SDK 7.x ships.
    pub libhipblas_in_sdk: Option<PathBuf>,
}

impl HipDependencies {
    /// True when `hipblas.dll` can be found by the loader (runtime folder or SDK on PATH).
    pub fn hipblas_resolvable(&self) -> bool {
        self.hipblas_in_runtime_dir || self.hipblas_in_sdk.is_some()
    }
}

/// Inspects a HIP runtime folder and the SDK bin folders that will be on PATH.
pub fn check_hip_dependencies(runtime_dir: &Path, sdk_bin_dirs: &[PathBuf]) -> HipDependencies {
    let backend_imports_hipblas = std::fs::read(runtime_dir.join("ggml-hip.dll"))
        .ok()
        .map(|bytes| names_hipblas_import(&bytes));
    let find = |file: &str| {
        sdk_bin_dirs
            .iter()
            .find(|d| d.join(file).is_file())
            .cloned()
    };
    HipDependencies {
        runtime_dir: runtime_dir.to_path_buf(),
        backend_imports_hipblas,
        hipblas_in_runtime_dir: runtime_dir.join("hipblas.dll").is_file(),
        hipblas_in_sdk: find("hipblas.dll"),
        libhipblas_in_sdk: find("libhipblas.dll"),
    }
}

/// True when the DLL bytes contain `hipblas.dll` as a name, not as the tail of
/// `libhipblas.dll`. PE import names are stored as plain ASCII, so a byte scan
/// finds them without a PE parser.
fn names_hipblas_import(bytes: &[u8]) -> bool {
    const NEEDLE: &[u8] = b"hipblas.dll";
    bytes.windows(NEEDLE.len()).enumerate().any(|(i, w)| {
        w.eq_ignore_ascii_case(NEEDLE) && !(i >= 3 && bytes[i - 3..i].eq_ignore_ascii_case(b"lib"))
    })
}

/// Why a GPU backend has no usable device.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureCause {
    /// `ggml-hip.dll` needs `hipblas.dll`, which is absent, but the SDK ships `libhipblas.dll`.
    HipblasNameMismatch,
    /// `ggml-hip.dll` needs `hipblas.dll` and no folder provides it.
    HipblasMissing,
    /// No HIP SDK bin folder was found at all.
    HipSdkNotFound,
    /// The DLLs resolve but llama-server initializes no device of this backend.
    NoDeviceVisible,
}

/// Typed diagnosis for a GPU backend that initialized no device. Shown to the
/// user as the load error and in the engine status. It never means "running on CPU".
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BackendLoadFailure {
    pub backend: Backend,
    pub cause: FailureCause,
    /// One or two plain sentences naming the cause.
    pub message: String,
    /// Concrete next steps, most direct first.
    pub suggestions: Vec<String>,
}

impl BackendLoadFailure {
    /// The message followed by the suggestions, as one text for error reporting.
    pub fn full_text(&self) -> String {
        let mut text = self.message.clone();
        for s in &self.suggestions {
            text.push(' ');
            text.push_str(s);
        }
        text
    }
}

/// Explains zero devices for `backend`. Pure: the caller gathers the facts.
pub fn diagnose_no_devices(
    backend: Backend,
    sdk_bin_dirs: &[PathBuf],
    deps: Option<&HipDependencies>,
) -> BackendLoadFailure {
    if backend != Backend::Hip {
        return BackendLoadFailure {
            backend,
            cause: FailureCause::NoDeviceVisible,
            message: format!(
                "llama-server initialized no {} device, so the model would run on the CPU.",
                backend.label()
            ),
            suggestions: vec![
                "Check the GPU driver for this backend, or switch to a runtime that matches the hardware.".into(),
                "Load with an explicit CPU configuration if CPU inference is what you want.".into(),
            ],
        };
    }
    if let Some(d) = deps {
        if d.backend_imports_hipblas == Some(true) && !d.hipblas_resolvable() {
            if d.libhipblas_in_sdk.is_some() {
                return BackendLoadFailure {
                    backend,
                    cause: FailureCause::HipblasNameMismatch,
                    message: format!(
                        "The HIP runtime in {} needs hipblas.dll, but the AMD HIP SDK provides libhipblas.dll under a different name. The HIP DLLs fail to load silently, so llama-server would run on the CPU.",
                        d.runtime_dir.display()
                    ),
                    suggestions: vec![
                        "Install the HIP SDK release that matches this runtime build and still ships hipblas.dll, then restart Raptor LLM.".into(),
                        "Or use the Vulkan runtime, which needs no HIP SDK.".into(),
                        "Raptor LLM does not copy or rename DLLs. Renaming libhipblas.dll by hand is an untested combination.".into(),
                    ],
                };
            }
            if sdk_bin_dirs.is_empty() {
                return sdk_not_found();
            }
            return BackendLoadFailure {
                backend,
                cause: FailureCause::HipblasMissing,
                message: format!(
                    "The HIP runtime in {} needs hipblas.dll, and no HIP SDK bin folder provides it, so the HIP backend cannot load.",
                    d.runtime_dir.display()
                ),
                suggestions: vec![
                    "Install the AMD HIP SDK that matches this runtime build, then restart Raptor LLM.".into(),
                    "Or use the Vulkan runtime, which needs no HIP SDK.".into(),
                ],
            };
        }
    }
    if sdk_bin_dirs.is_empty() {
        return sdk_not_found();
    }
    BackendLoadFailure {
        backend,
        cause: FailureCause::NoDeviceVisible,
        message: "The HIP DLLs load, but llama-server sees no ROCm device, so the model would run on the CPU.".into(),
        suggestions: vec![
            "Update the AMD graphics driver and confirm the GPU architecture is supported by the installed HIP SDK.".into(),
            "Or use the Vulkan runtime.".into(),
        ],
    }
}

fn sdk_not_found() -> BackendLoadFailure {
    BackendLoadFailure {
        backend: Backend::Hip,
        cause: FailureCause::HipSdkNotFound,
        message: "No AMD HIP SDK bin folder was found (checked HIP_PATH and %ProgramFiles%\\AMD\\ROCm), so the HIP backend cannot load its DLLs.".into(),
        suggestions: vec![
            "Install the AMD HIP SDK that matches this runtime build, then restart Raptor LLM.".into(),
            "Or use the Vulkan runtime, which needs no HIP SDK.".into(),
        ],
    }
}

/// Result of asking llama-server which devices it can use.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DevicePreflight {
    pub backend: Backend,
    pub binary: PathBuf,
    /// Every device llama-server listed, of any backend.
    pub devices: Vec<RuntimeDevice>,
    /// SDK bin folders prepended to the child's PATH (HIP only).
    pub sdk_bin_dirs: Vec<PathBuf>,
    /// DLL facts for HIP runtimes.
    pub hip_dependencies: Option<HipDependencies>,
    /// Set when the requested backend has no device. `None` means at least one usable device.
    pub failure: Option<BackendLoadFailure>,
}

impl DevicePreflight {
    /// True when the requested backend has at least one device.
    pub fn has_usable_device(&self) -> bool {
        self.failure.is_none()
    }
}

/// Runs `binary --list-devices` with `env` and parses the result.
pub async fn list_devices(
    runner: &dyn CommandRunner,
    binary: &Path,
    env: &[(String, String)],
) -> Result<Vec<RuntimeDevice>> {
    let program = binary.to_string_lossy().into_owned();
    let out = runner
        .run_with_env(
            &program,
            &["--list-devices".to_string()],
            env,
            LIST_DEVICES_TIMEOUT,
        )
        .await?;
    Ok(parse_list_devices(&out))
}

/// Full preflight for `backend` on `binary`. `sdk_bin_dirs` and `env` come from
/// [`HipSdkSearch::bin_dirs`] and [`launch_env`]. Returns `Err` only when the
/// binary cannot be run at all. "No devices" is a normal result with `failure` set.
pub async fn preflight(
    runner: &dyn CommandRunner,
    binary: &Path,
    backend: Backend,
    sdk_bin_dirs: Vec<PathBuf>,
    env: &[(String, String)],
) -> Result<DevicePreflight> {
    let devices = list_devices(runner, binary, env).await?;
    let hip_dependencies = if backend == Backend::Hip {
        let dir = binary.parent().unwrap_or_else(|| Path::new("."));
        Some(check_hip_dependencies(dir, &sdk_bin_dirs))
    } else {
        None
    };
    let has_device = devices.iter().any(|d| d.backend == Some(backend));
    // A CPU runtime needs no device; only GPU backends can fail this check.
    let failure = (backend.is_gpu() && !has_device)
        .then(|| diagnose_no_devices(backend, &sdk_bin_dirs, hip_dependencies.as_ref()));
    Ok(DevicePreflight {
        backend,
        binary: binary.to_path_buf(),
        devices,
        sdk_bin_dirs,
        hip_dependencies,
        failure,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::FakeRunner;
    use std::fs;

    const RX9060: &str =
        "Available devices:\n  ROCm0: AMD Radeon RX 9060 XT (16304 MiB, 16158 MiB free)\n";

    #[test]
    fn parses_rocm_device_with_memory() {
        let devs = parse_list_devices(RX9060);
        assert_eq!(devs.len(), 1);
        assert_eq!(devs[0].id, "ROCm0");
        assert_eq!(devs[0].name, "AMD Radeon RX 9060 XT");
        assert_eq!(devs[0].total_mib, Some(16304));
        assert_eq!(devs[0].free_mib, Some(16158));
        assert_eq!(devs[0].backend, Some(Backend::Hip));
    }

    #[test]
    fn none_listed_yields_no_devices() {
        assert!(parse_list_devices("Available devices:\n  (none)\n").is_empty());
        assert!(parse_list_devices("").is_empty());
    }

    #[test]
    fn parses_cuda_vulkan_and_multiple_devices_and_ignores_noise() {
        let text = "load_backend: loaded ggml-hip\nAvailable devices:\n  CUDA0: NVIDIA GeForce RTX 4070 (12282 MiB, 11000 MiB free)\n  Vulkan1: AMD Radeon 780M (512 MiB, 400 MiB free)\n";
        let devs = parse_list_devices(text);
        assert_eq!(devs.len(), 2);
        assert_eq!(devs[0].backend, Some(Backend::Cuda));
        assert_eq!(devs[0].name, "NVIDIA GeForce RTX 4070");
        assert_eq!(devs[1].backend, Some(Backend::Vulkan));
        assert_eq!(devs[1].free_mib, Some(400));
    }

    #[test]
    fn device_without_memory_keeps_its_name() {
        let devs = parse_list_devices("Available devices:\n  ROCm0: Some GPU (gfx9)\n");
        assert_eq!(devs[0].name, "Some GPU (gfx9)");
        assert_eq!(devs[0].total_mib, None);
    }

    #[test]
    fn unknown_prefix_has_no_backend() {
        let devs = parse_list_devices("Available devices:\n  Foo0: Mystery (1 MiB, 1 MiB free)\n");
        assert_eq!(devs[0].backend, None);
    }

    #[test]
    fn launch_env_prepends_sdk_dirs_and_keeps_current_path() {
        let current = std::env::join_paths([PathBuf::from("/usr/bin")])
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let env = launch_env(&[PathBuf::from("/sdk/bin")], Some(current.as_str())).unwrap();
        assert_eq!(env[0].0, "PATH");
        let parts: Vec<PathBuf> = std::env::split_paths(&env[0].1).collect();
        assert_eq!(parts[0], PathBuf::from("/sdk/bin"));
        assert_eq!(parts[1], PathBuf::from("/usr/bin"));
    }

    #[test]
    fn launch_env_is_empty_without_sdk_dirs() {
        assert!(launch_env(&[], Some("/usr/bin")).unwrap().is_empty());
    }

    #[test]
    fn hipblas_import_detection_ignores_libhipblas() {
        assert!(names_hipblas_import(b"xx\0hipblas.dll\0yy"));
        assert!(names_hipblas_import(b"HIPBLAS.DLL"));
        assert!(!names_hipblas_import(b"\0libhipblas.dll\0"));
        assert!(!names_hipblas_import(b"amdhip64_7.dll"));
    }

    #[test]
    fn sdk_search_prefers_hip_path_then_newest_version() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("ROCm");
        for v in ["6.4", "7.1", "7.2", "7.10"] {
            fs::create_dir_all(root.join(v).join("bin")).unwrap();
        }
        fs::create_dir_all(root.join("no-bin")).unwrap();
        let hip = dir.path().join("custom");
        fs::create_dir_all(hip.join("bin")).unwrap();
        let search = HipSdkSearch {
            roots: vec![root.clone(), dir.path().join("missing")],
            hip_path: Some(hip.clone()),
        };
        let bins = search.bin_dirs();
        assert_eq!(bins[0], hip.join("bin"));
        assert_eq!(bins[1], root.join("7.10").join("bin"));
        assert_eq!(bins[2], root.join("7.2").join("bin"));
        assert_eq!(bins[3], root.join("7.1").join("bin"));
        assert_eq!(bins[4], root.join("6.4").join("bin"));
        assert_eq!(bins.len(), 5);
    }

    #[test]
    fn non_windows_search_is_empty_by_default() {
        let search = HipSdkSearch::default();
        assert!(search.bin_dirs().is_empty());
    }

    #[test]
    fn dependency_check_finds_name_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let runtime = dir.path().join("bin");
        let sdk = dir.path().join("sdk");
        fs::create_dir_all(&runtime).unwrap();
        fs::create_dir_all(&sdk).unwrap();
        fs::write(runtime.join("ggml-hip.dll"), b"MZ....hipblas.dll....").unwrap();
        fs::write(sdk.join("libhipblas.dll"), b"x").unwrap();
        let deps = check_hip_dependencies(&runtime, std::slice::from_ref(&sdk));
        assert_eq!(deps.backend_imports_hipblas, Some(true));
        assert!(!deps.hipblas_resolvable());
        assert_eq!(deps.libhipblas_in_sdk, Some(sdk.clone()));
        let failure = diagnose_no_devices(Backend::Hip, &[sdk], Some(&deps));
        assert_eq!(failure.cause, FailureCause::HipblasNameMismatch);
        assert!(failure.message.contains("hipblas.dll"));
        assert!(failure.message.contains("libhipblas.dll"));
        assert!(failure.suggestions.iter().any(|s| s.contains("Vulkan")));
    }

    #[test]
    fn hipblas_in_runtime_dir_resolves_the_import() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("ggml-hip.dll"), b"hipblas.dll").unwrap();
        fs::write(dir.path().join("hipblas.dll"), b"x").unwrap();
        let sdk = dir.path().join("sdk");
        fs::create_dir_all(&sdk).unwrap();
        let deps = check_hip_dependencies(dir.path(), std::slice::from_ref(&sdk));
        assert!(deps.hipblas_resolvable());
        assert_eq!(
            diagnose_no_devices(Backend::Hip, std::slice::from_ref(&sdk), Some(&deps)).cause,
            FailureCause::NoDeviceVisible
        );
    }

    #[test]
    fn missing_sdk_and_missing_dlls_are_distinct_causes() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("ggml-hip.dll"), b"hipblas.dll").unwrap();
        let deps = check_hip_dependencies(dir.path(), &[]);
        assert_eq!(
            diagnose_no_devices(Backend::Hip, &[], Some(&deps)).cause,
            FailureCause::HipSdkNotFound
        );
        let sdk = dir.path().join("sdk");
        fs::create_dir_all(&sdk).unwrap();
        let deps = check_hip_dependencies(dir.path(), std::slice::from_ref(&sdk));
        assert_eq!(
            diagnose_no_devices(Backend::Hip, std::slice::from_ref(&sdk), Some(&deps)).cause,
            FailureCause::HipblasMissing
        );
    }

    #[test]
    fn cuda_zero_devices_is_a_plain_no_device_failure() {
        let f = diagnose_no_devices(Backend::Cuda, &[], None);
        assert_eq!(f.cause, FailureCause::NoDeviceVisible);
        assert!(f.message.contains("cuda"));
    }

    #[tokio::test]
    async fn preflight_with_device_has_no_failure() {
        let runner = FakeRunner::default().with_ok("llama-server", RX9060);
        let pf = preflight(
            &runner,
            Path::new("llama-server"),
            Backend::Hip,
            vec![],
            &[],
        )
        .await
        .unwrap();
        assert!(pf.has_usable_device());
        assert_eq!(pf.devices.len(), 1);
    }

    #[tokio::test]
    async fn preflight_without_devices_fails_with_diagnosis() {
        let runner =
            FakeRunner::default().with_ok("llama-server", "Available devices:\n  (none)\n");
        let pf = preflight(
            &runner,
            Path::new("llama-server"),
            Backend::Hip,
            vec![],
            &[],
        )
        .await
        .unwrap();
        assert!(!pf.has_usable_device());
        assert_eq!(pf.failure.unwrap().cause, FailureCause::HipSdkNotFound);
    }

    #[tokio::test]
    async fn preflight_cpu_device_does_not_satisfy_hip() {
        let runner = FakeRunner::default().with_ok(
            "llama-server",
            "Available devices:\n  CPU: Ryzen (65000 MiB, 50000 MiB free)\n",
        );
        let pf = preflight(
            &runner,
            Path::new("llama-server"),
            Backend::Hip,
            vec![],
            &[],
        )
        .await
        .unwrap();
        assert!(!pf.has_usable_device());
    }

    #[tokio::test]
    async fn preflight_passes_environment_and_errors_when_binary_cannot_run() {
        let env = vec![("PATH".to_string(), "/sdk/bin".to_string())];
        let runner = FakeRunner::default().with_ok("llama-server", RX9060);
        preflight(
            &runner,
            Path::new("llama-server"),
            Backend::Hip,
            vec![],
            &env,
        )
        .await
        .unwrap();
        assert_eq!(runner.last_env(), env);

        let err = preflight(
            &FakeRunner::default(),
            Path::new("gone"),
            Backend::Hip,
            vec![],
            &[],
        )
        .await
        .unwrap_err();
        assert!(matches!(err, RaptorError::Runtime(_)));
    }
}
