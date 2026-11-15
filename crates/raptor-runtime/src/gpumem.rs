//! How much of the GPU's memory is in use and how much of a load went into *shared* system memory.
//!
//! Windows lets a GPU allocation spill past dedicated VRAM into shared system memory instead of
//! failing, and the model keeps working but much slower: Qwen3.8-27B at a 167k window put 2.3 GB
//! into shared memory and generation fell from 47 to about 10 tok/s with no error anywhere. A load
//! that spilled must therefore say so. (For a sparse Mixture-of-experts model the spill is the
//! plan, because only a few experts are read per token, and the caller knows that.)
//!
//! Read from the `\GPU Adapter Memory(*)` performance counters through PowerShell. The reading is
//! per adapter; the adapter using the most dedicated memory is the one that matters. Other
//! programs also use shared memory, so a spill is the *growth* in shared usage across a load.

use crate::sys::CommandRunner;
use async_trait::async_trait;
use raptor_core::{RaptorError, Result};
use serde::Serialize;
use std::sync::Arc;
use std::time::Duration;

/// Growth in shared GPU memory above this many MiB during a load counts as a spill. Smaller
/// changes are the driver's own bookkeeping (measured: 483 -> 631 MiB between idle and a loaded
/// model that fit).
pub const SPILL_WARN_MIB: u64 = 256;

/// GPU memory in use, MiB.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct GpuMemoryUsage {
    pub dedicated_mib: u64,
    pub shared_mib: u64,
}

/// Reads GPU memory use. Behind a trait so loads are tested without a GPU.
#[async_trait]
pub trait GpuMemoryProbe: Send + Sync {
    async fn read(&self) -> Result<GpuMemoryUsage>;
}

/// Parses `dedicated=<MiB> shared=<MiB>`.
pub fn parse_usage(text: &str) -> Option<GpuMemoryUsage> {
    let field = |name: &str| -> Option<u64> {
        let at = text.find(&format!("{name}="))? + name.len() + 1;
        let digits: String = text[at..]
            .chars()
            .take_while(char::is_ascii_digit)
            .collect();
        digits.parse().ok()
    };
    Some(GpuMemoryUsage {
        dedicated_mib: field("dedicated")?,
        shared_mib: field("shared")?,
    })
}

/// The PowerShell that prints the usage of the adapter with the most dedicated memory in use.
const SCRIPT: &str = r#"$c = Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage','\GPU Adapter Memory(*)\Shared Usage' -ErrorAction SilentlyContinue
$a = @{}
foreach ($s in $c.CounterSamples) {
  $key = ($s.Path -split '\\')[-2]
  if (-not $a.ContainsKey($key)) { $a[$key] = @{ d = 0.0; s = 0.0 } }
  if ($s.Path -like '*dedicated usage') { $a[$key].d = $s.CookedValue } else { $a[$key].s = $s.CookedValue }
}
$best = $a.Values | Sort-Object { $_.d } -Descending | Select-Object -First 1
if ($best) { 'dedicated={0} shared={1}' -f [int]($best.d / 1MB), [int]($best.s / 1MB) }"#;

/// Reads the Windows counters. Other systems report that the reading is unavailable.
pub struct WindowsGpuMemory {
    runner: Arc<dyn CommandRunner>,
}

impl WindowsGpuMemory {
    pub fn new(runner: Arc<dyn CommandRunner>) -> Self {
        Self { runner }
    }
}

#[async_trait]
impl GpuMemoryProbe for WindowsGpuMemory {
    async fn read(&self) -> Result<GpuMemoryUsage> {
        if !cfg!(windows) {
            return Err(RaptorError::Unsupported(
                "GPU memory use is only read on Windows".into(),
            ));
        }
        let out = self
            .runner
            .run(
                "powershell",
                &[
                    "-NoProfile".to_string(),
                    "-NonInteractive".to_string(),
                    "-Command".to_string(),
                    SCRIPT.to_string(),
                ],
                Duration::from_secs(20),
            )
            .await?;
        parse_usage(&out).ok_or_else(|| {
            RaptorError::Runtime(format!("unreadable GPU memory counters: {}", out.trim()))
        })
    }
}

/// Peak GPU memory seen while some work ran.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct PeakUsage {
    /// Highest dedicated use read, MiB (includes whatever else uses the GPU).
    pub peak_dedicated_mib: u64,
    /// Highest growth of shared memory over the reading taken before the work, MiB.
    pub peak_shared_growth_mib: u64,
    /// True when the growth passed [`SPILL_WARN_MIB`].
    pub spilled: bool,
}

/// Runs `work` while reading GPU memory every `interval`, and returns its result with the peak.
/// The peak is `None` when no reading could be taken (counters unavailable, or `work` finished
/// before the first one). A short job can therefore be missed; the benchmark runs are minutes long.
pub async fn sample_while<T>(
    probe: &dyn GpuMemoryProbe,
    interval: Duration,
    work: impl std::future::Future<Output = T>,
) -> (T, Option<PeakUsage>) {
    let baseline = probe.read().await.ok();
    tokio::pin!(work);
    let mut peak: Option<PeakUsage> = None;
    let out = loop {
        tokio::select! {
            out = &mut work => break out,
            _ = tokio::time::sleep(interval) => {
                if let Ok(now) = probe.read().await {
                    let growth = baseline.map_or(0, |b| now.shared_mib.saturating_sub(b.shared_mib));
                    let p = peak.get_or_insert(PeakUsage {
                        peak_dedicated_mib: 0,
                        peak_shared_growth_mib: 0,
                        spilled: false,
                    });
                    p.peak_dedicated_mib = p.peak_dedicated_mib.max(now.dedicated_mib);
                    p.peak_shared_growth_mib = p.peak_shared_growth_mib.max(growth);
                    p.spilled = p.peak_shared_growth_mib > SPILL_WARN_MIB;
                }
            }
        }
    };
    (out, peak)
}

/// What a load did to GPU memory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SpillReport {
    /// Dedicated (VRAM) use after the load, MiB. Includes whatever else is using the GPU.
    pub dedicated_mib: u64,
    /// Growth in shared system memory across the load, MiB.
    pub shared_growth_mib: u64,
    /// True when `shared_growth_mib` is above [`SPILL_WARN_MIB`].
    pub spilled: bool,
}

/// Compares a reading before the server started with one after it was ready.
pub fn spill_between(before: GpuMemoryUsage, after: GpuMemoryUsage) -> SpillReport {
    let growth = after.shared_mib.saturating_sub(before.shared_mib);
    SpillReport {
        dedicated_mib: after.dedicated_mib,
        shared_growth_mib: growth,
        spilled: growth > SPILL_WARN_MIB,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_counter_line_is_parsed() {
        assert_eq!(
            parse_usage("dedicated=15609 shared=2281\r\n"),
            Some(GpuMemoryUsage {
                dedicated_mib: 15609,
                shared_mib: 2281
            })
        );
        assert_eq!(parse_usage("dedicated=1"), None);
        assert_eq!(parse_usage(""), None);
        assert_eq!(parse_usage("dedicated=x shared=2"), None);
    }

    #[test]
    fn a_spill_is_the_growth_in_shared_memory_above_the_noise() {
        let idle = GpuMemoryUsage {
            dedicated_mib: 586,
            shared_mib: 483,
        };
        // Measured: a Qwen load that fit moved shared memory 483 -> 631 MiB.
        let fit = spill_between(
            idle,
            GpuMemoryUsage {
                dedicated_mib: 13_807,
                shared_mib: 631,
            },
        );
        assert!(!fit.spilled && fit.shared_growth_mib == 148);
        // Measured: the 167k window with MTP moved it 483 -> 2,281 MiB.
        let spill = spill_between(
            idle,
            GpuMemoryUsage {
                dedicated_mib: 15_609,
                shared_mib: 2_281,
            },
        );
        assert!(spill.spilled);
        assert_eq!(spill.shared_growth_mib, 1_798);
        assert_eq!(spill.dedicated_mib, 15_609);
    }

    #[test]
    fn shared_memory_that_shrank_is_not_a_spill() {
        let r = spill_between(
            GpuMemoryUsage {
                dedicated_mib: 100,
                shared_mib: 2_000,
            },
            GpuMemoryUsage {
                dedicated_mib: 9_000,
                shared_mib: 500,
            },
        );
        assert_eq!(r.shared_growth_mib, 0);
        assert!(!r.spilled);
    }

    struct Seq(std::sync::Mutex<Vec<GpuMemoryUsage>>);

    #[async_trait]
    impl GpuMemoryProbe for Seq {
        async fn read(&self) -> Result<GpuMemoryUsage> {
            let mut v = self.0.lock().unwrap();
            if v.len() > 1 {
                Ok(v.remove(0))
            } else {
                v.first()
                    .copied()
                    .ok_or_else(|| RaptorError::Runtime("none".into()))
            }
        }
    }

    fn usage(d: u64, s: u64) -> GpuMemoryUsage {
        GpuMemoryUsage {
            dedicated_mib: d,
            shared_mib: s,
        }
    }

    #[tokio::test(start_paused = true)]
    async fn the_peak_is_the_highest_reading_while_the_work_runs() {
        let probe = Seq(std::sync::Mutex::new(vec![
            usage(500, 400),
            usage(9_000, 450),
            usage(15_000, 1_900),
            usage(12_000, 500),
        ]));
        let (out, peak) = sample_while(&probe, Duration::from_secs(2), async {
            tokio::time::sleep(Duration::from_secs(7)).await;
            42
        })
        .await;
        assert_eq!(out, 42);
        let p = peak.unwrap();
        assert_eq!(p.peak_dedicated_mib, 15_000);
        assert_eq!(p.peak_shared_growth_mib, 1_500);
        assert!(p.spilled);
    }

    #[tokio::test(start_paused = true)]
    async fn no_readings_means_no_peak_not_a_zero() {
        let probe = Seq(std::sync::Mutex::new(vec![]));
        let (_, peak) = sample_while(&probe, Duration::from_secs(1), async {
            tokio::time::sleep(Duration::from_secs(3)).await;
        })
        .await;
        assert!(peak.is_none());
    }

    #[tokio::test]
    async fn work_that_finishes_before_the_first_reading_has_no_peak() {
        let probe = Seq(std::sync::Mutex::new(vec![usage(1, 1)]));
        let (out, peak) = sample_while(&probe, Duration::from_secs(60), async { 7 }).await;
        assert_eq!(out, 7);
        assert!(peak.is_none());
    }

    #[tokio::test]
    async fn the_windows_probe_runs_powershell_and_parses_its_line() {
        use crate::testing::FakeRunner;
        let runner =
            Arc::new(FakeRunner::default().with_ok("powershell", "dedicated=700 shared=500\n"));
        let probe = WindowsGpuMemory::new(runner);
        if cfg!(windows) {
            assert_eq!(probe.read().await.unwrap().dedicated_mib, 700);
        } else {
            assert!(probe.read().await.is_err());
        }
    }

    #[tokio::test]
    async fn garbage_from_the_counters_is_an_error_not_zero() {
        use crate::testing::FakeRunner;
        let runner = Arc::new(FakeRunner::default().with_ok("powershell", "no counters"));
        let probe = WindowsGpuMemory::new(runner);
        assert!(probe.read().await.is_err());
    }
}
