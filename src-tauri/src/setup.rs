//! First-run checklist for the UI: what was detected, what is installed and what is missing.
//!
//! Nothing here downloads or installs. A compatible runtime that is not installed is reported
//! as available, and the user confirms the download separately through `runtime_ensure`.

use raptor_runtime::config::GIB;
use raptor_runtime::devices::DevicePreflight;
use raptor_runtime::hardware::HardwareInfo;
use raptor_runtime::manifest::Backend;
use raptor_runtime::RuntimeSummary;
use serde::Serialize;
use std::path::Path;

/// Free space below which the storage check fails. A single 7-8 GB model plus a runtime fits.
pub const MIN_FREE_BYTES: u64 = 10 * GIB;

/// One row of the checklist.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SetupCheck {
    pub id: &'static str,
    pub label: &'static str,
    pub ok: bool,
    pub detail: String,
}

/// A runtime the index offers for this machine, not yet installed.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AvailableRuntime {
    pub id: String,
    pub version: String,
    pub backend: Backend,
    pub degraded: bool,
    pub reason: String,
}

/// Everything the checklist is built from. The shell fills it in; the builder is pure.
#[derive(Debug, Clone, Default)]
pub struct SetupInput {
    pub hardware: Option<HardwareInfo>,
    /// The runtime selected in this session, if any.
    pub selected: Option<RuntimeSummary>,
    /// Runtimes installed on disk: `(id, version, backend)`.
    pub installed: Vec<(String, String, Backend)>,
    /// What the index offers for this machine, if the index is readable.
    pub available: Option<AvailableRuntime>,
    pub models_installed: usize,
    pub storage_free_bytes: Option<u64>,
    /// Why performance memory cannot be used, if it cannot.
    pub perf_store_error: Option<String>,
    /// Startup notes from the shell, such as settings recovery.
    pub notes: Vec<String>,
    /// Device preflight of the selected runtime. A `failure` makes the GPU row fail.
    pub device_preflight: Option<DevicePreflight>,
}

/// The full checklist plus the facts behind it.
#[derive(Debug, Clone, Serialize)]
pub struct SetupStatus {
    pub checks: Vec<SetupCheck>,
    pub hardware: Option<HardwareInfo>,
    pub selected_runtime: Option<RuntimeSummary>,
    pub installed_runtimes: Vec<String>,
    pub available_runtime: Option<AvailableRuntime>,
    /// True when a compatible runtime exists but nothing is installed. The UI must ask first.
    pub runtime_needs_confirmation: bool,
    /// True when the chosen runtime is CPU-only because no GPU build is compatible.
    pub degraded: bool,
    pub models_installed: usize,
    pub storage_free_bytes: Option<u64>,
    pub notes: Vec<String>,
    /// Device preflight of the selected runtime: which devices llama-server can use,
    /// and the typed diagnosis when a GPU backend has none.
    pub device_preflight: Option<DevicePreflight>,
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / GIB as f64
}

fn hardware_detail(hw: &HardwareInfo) -> String {
    match hw.gpus.first() {
        Some(g) => format!(
            "{} ({}), {:.0} GiB VRAM, {:.0} GiB RAM",
            g.name,
            g.arch.as_deref().unwrap_or("arch unknown"),
            gib(g.vram_total),
            gib(hw.ram_total)
        ),
        None => format!("no GPU found, {:.0} GiB RAM", gib(hw.ram_total)),
    }
}

/// Builds the checklist. Rows are in the order a first-run guide reads them.
pub fn build(input: SetupInput) -> SetupStatus {
    let installed_runtimes: Vec<String> = input
        .installed
        .iter()
        .map(|(id, version, _)| format!("{id} {version}"))
        .collect();
    let runtime_installed = input.selected.is_some() || !input.installed.is_empty();

    let hardware_check = SetupCheck {
        id: "hardware",
        label: "Hardware detected",
        ok: input.hardware.is_some(),
        detail: match &input.hardware {
            Some(hw) => hardware_detail(hw),
            None => "not detected yet".into(),
        },
    };

    let runtime_check = SetupCheck {
        id: "runtime",
        label: "llama.cpp runtime installed",
        ok: runtime_installed,
        detail: match (&input.selected, input.installed.first(), &input.available) {
            (Some(s), _, _) => format!("{} {} selected", s.id, s.version),
            (None, Some(_), _) => format!(
                "{} installed but not selected yet; runtime_ensure selects it",
                installed_runtimes[0]
            ),
            (None, None, Some(a)) => format!(
                "not installed. {} {} is compatible; installing it needs your confirmation",
                a.id, a.version
            ),
            (None, None, None) => "not installed, and no compatible runtime is listed".into(),
        },
    };

    // The GPU check describes the runtime in use, or the one that would be installed.
    // A device preflight that found no device overrides everything: the GPU build
    // is present but cannot run the model on the GPU.
    let no_device = input
        .device_preflight
        .as_ref()
        .and_then(|p| p.failure.as_ref());
    let (gpu_ok, gpu_detail) = match (no_device, &input.selected, &input.available) {
        (Some(f), _, _) => (false, format!("GPU unavailable: {}", f.message)),
        (None, Some(s), _) if s.degraded || !s.backend.is_gpu() => {
            (false, format!("CPU only: {}", s.reason))
        }
        (None, Some(s), _) => (true, format!("GPU build {} {} in use", s.id, s.version)),
        (None, None, Some(a)) if a.degraded || !a.backend.is_gpu() => {
            (false, format!("CPU only: {}", a.reason))
        }
        (None, None, Some(a)) => (
            true,
            format!("GPU build {} {} is compatible", a.id, a.version),
        ),
        (None, None, None) => (false, "not determined yet".into()),
    };
    let gpu_check = SetupCheck {
        id: "gpu",
        label: "GPU acceleration",
        ok: gpu_ok,
        detail: gpu_detail,
    };

    let model_check = SetupCheck {
        id: "model",
        label: "Model installed",
        ok: input.models_installed > 0,
        detail: match input.models_installed {
            0 => "no model in the library yet".into(),
            1 => "1 model in the library".into(),
            n => format!("{n} models in the library"),
        },
    };

    let storage_check = match input.storage_free_bytes {
        Some(free) => SetupCheck {
            id: "storage",
            label: "Free storage",
            ok: free >= MIN_FREE_BYTES,
            detail: format!("{:.1} GiB free", gib(free)),
        },
        None => SetupCheck {
            id: "storage",
            label: "Free storage",
            ok: false,
            detail: "could not be measured on this platform".into(),
        },
    };

    let perf_check = SetupCheck {
        id: "performance_memory",
        label: "Performance memory",
        ok: input.perf_store_error.is_none(),
        detail: match &input.perf_store_error {
            Some(e) => format!("unavailable: {e}"),
            None => "readable".into(),
        },
    };

    let degraded = input.selected.as_ref().is_some_and(|s| s.degraded)
        || (input.selected.is_none() && input.available.as_ref().is_some_and(|a| a.degraded));

    SetupStatus {
        checks: vec![
            hardware_check,
            runtime_check,
            gpu_check,
            model_check,
            storage_check,
            perf_check,
        ],
        hardware: input.hardware,
        selected_runtime: input.selected,
        installed_runtimes,
        runtime_needs_confirmation: !runtime_installed && input.available.is_some(),
        available_runtime: input.available,
        degraded,
        models_installed: input.models_installed,
        device_preflight: input.device_preflight,
        storage_free_bytes: input.storage_free_bytes,
        notes: input.notes,
    }
}

/// Free bytes available to the user on the volume holding `path`. `None` when the platform
/// cannot report it.
#[cfg(windows)]
pub fn free_bytes(path: &Path) -> Option<u64> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;
    let wide: Vec<u16> = path
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    let mut available: u64 = 0;
    // SAFETY: `wide` is NUL-terminated and lives across the call. The other out-pointers may
    // be null, which the API allows.
    let ok = unsafe {
        GetDiskFreeSpaceExW(
            wide.as_ptr(),
            &mut available,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };
    (ok != 0).then_some(available)
}

/// Free space is not measured on this platform yet.
#[cfg(not(windows))]
pub fn free_bytes(_path: &Path) -> Option<u64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::fixed_hardware;

    fn summary(backend: Backend, degraded: bool) -> RuntimeSummary {
        RuntimeSummary {
            id: "llama-hip-win".into(),
            version: "b7100".into(),
            backend,
            degraded,
            reason: "compatible".into(),
            binary: "x/llama-server.exe".into(),
        }
    }

    fn available(backend: Backend, degraded: bool) -> AvailableRuntime {
        AvailableRuntime {
            id: "llama-hip-win".into(),
            version: "b12340".into(),
            backend,
            degraded,
            reason: if degraded {
                "no GPU build is compatible".into()
            } else {
                "compatible".into()
            },
        }
    }

    fn check<'a>(s: &'a SetupStatus, id: &str) -> &'a SetupCheck {
        s.checks.iter().find(|c| c.id == id).expect("check present")
    }

    #[test]
    fn fresh_machine_asks_for_confirmation_before_any_install() {
        let s = build(SetupInput {
            hardware: Some(fixed_hardware()),
            available: Some(available(Backend::Hip, false)),
            storage_free_bytes: Some(200 * GIB),
            ..Default::default()
        });
        assert!(check(&s, "hardware").ok);
        assert!(!check(&s, "runtime").ok);
        assert!(check(&s, "runtime")
            .detail
            .contains("needs your confirmation"));
        assert!(check(&s, "gpu").ok, "a compatible GPU build is reported");
        assert!(s.runtime_needs_confirmation);
        assert!(!check(&s, "model").ok);
        assert!(s.selected_runtime.is_none());
    }

    #[test]
    fn cpu_fallback_is_reported_as_degraded_not_as_gpu() {
        let s = build(SetupInput {
            hardware: Some(fixed_hardware()),
            selected: Some(summary(Backend::Cpu, true)),
            storage_free_bytes: Some(200 * GIB),
            ..Default::default()
        });
        assert!(s.degraded);
        assert!(!check(&s, "gpu").ok);
        assert!(check(&s, "gpu").detail.starts_with("CPU only"));
        assert!(check(&s, "runtime").ok);
        assert!(!s.runtime_needs_confirmation);
    }

    #[test]
    fn installed_but_unselected_runtime_counts_as_installed() {
        let s = build(SetupInput {
            installed: vec![("llama-hip-win".into(), "b7100".into(), Backend::Hip)],
            available: Some(available(Backend::Hip, false)),
            ..Default::default()
        });
        assert!(check(&s, "runtime").ok);
        assert!(check(&s, "runtime").detail.contains("not selected yet"));
        assert!(!s.runtime_needs_confirmation);
        assert_eq!(
            s.installed_runtimes,
            vec!["llama-hip-win b7100".to_string()]
        );
    }

    #[test]
    fn low_storage_and_unmeasured_storage_both_fail_the_check() {
        let low = build(SetupInput {
            storage_free_bytes: Some(MIN_FREE_BYTES - 1),
            ..Default::default()
        });
        assert!(!check(&low, "storage").ok);
        let unknown = build(SetupInput::default());
        assert!(!check(&unknown, "storage").ok);
        assert!(check(&unknown, "storage")
            .detail
            .contains("could not be measured"));
        let enough = build(SetupInput {
            storage_free_bytes: Some(MIN_FREE_BYTES),
            ..Default::default()
        });
        assert!(check(&enough, "storage").ok);
    }

    #[test]
    fn unusable_performance_memory_is_reported() {
        let s = build(SetupInput {
            perf_store_error: Some("perf store is corrupt".into()),
            ..Default::default()
        });
        assert!(!check(&s, "performance_memory").ok);
        assert!(check(&s, "performance_memory").detail.contains("corrupt"));
    }

    #[test]
    fn checklist_has_every_row_in_order() {
        let s = build(SetupInput::default());
        let ids: Vec<&str> = s.checks.iter().map(|c| c.id).collect();
        assert_eq!(
            ids,
            vec![
                "hardware",
                "runtime",
                "gpu",
                "model",
                "storage",
                "performance_memory"
            ]
        );
    }

    #[cfg(windows)]
    #[test]
    fn free_space_of_a_temp_directory_is_measured() {
        let dir = tempfile::tempdir().unwrap();
        let free = free_bytes(dir.path()).expect("Windows reports free space");
        assert!(free > 0);
    }
}
