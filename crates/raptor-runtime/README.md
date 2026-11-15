# raptor-runtime

Runtime engine for Raptor LLM: hardware detection, llama.cpp runtime selection and
installation, llama-server lifecycle, GPU verification from logs, memory-aware
launch configuration, performance memory, and token counting.
Spec: `docs/specs/runtime.md`.

Depends only on `raptor-core`. Every external effect (processes, HTTP, downloads,
hardware probes, command tools, port checks) is behind a trait with a fake in
`src/testing.rs`, so the test suite needs no GPU, network, or llama-server.

## Public API

Facade (`lib.rs`):

- `RuntimeEngine::with_system_deps(&DataDirs) -> Result<RuntimeEngine>`: real wiring.
- `RuntimeEngine::new(&DataDirs, EngineDeps)`: explicit dependencies.
- `detect_hardware(&self) -> Result<HardwareInfo>`: probe and cache hardware.
- `ensure_runtime(&self, &[Manifest], &CancellationToken) -> Result<Selection>`: select the best runtime, install it if needed (idempotent).
- `recommend_config(&self, &ModelMeta, ctx_target) -> Result<Recommendation>`: VRAM/RAM-aware config with a rationale (dense estimate).
- `recommend_config_with_arch(&self, &ModelMeta, &ModelArch, ctx_target, Option<&FitProjection>) -> Result<Recommendation>`: the same for a known architecture (hybrid KV layers, recurrent state, offloadable layers), optionally calibrated by llama.cpp's projection.
- `probe_fit(&self, &LaunchConfig) -> Result<FitProjection>`: runs `llama-fit-params --fit-print on` from the selected runtime with every layer on the GPU. Loads no weights, starts no server.
- `use_local_runtime(&self, &Path, Backend) -> Result<RuntimeSummary>`: use a llama-server binary this engine did not install (developer build or test fixture). Nothing is hash-checked. The shell does not call it.
- `preflight_devices(&self) -> Result<DevicePreflight>`: runs `llama-server --list-devices` for the selected runtime (HIP SDK folders on PATH) and stores the result. Loads no model.
- `load_model(&self, LaunchConfig) -> Result<EngineStatus>`: for a GPU backend, runs the device preflight first. Zero devices of that backend refuses the load with the typed diagnosis, unless `LaunchConfig.allow_cpu_fallback` is set. Otherwise it starts llama-server, returns once Ready, with a GPU verdict.
- `unload(&self) -> Result<()>`: stop the server.
- `status(&self) -> EngineStatus`: runtime, server state, GPU verdict, and `device_preflight` (latest device check, with any `BackendLoadFailure`).
- `rollback(&self) -> Result<RuntimeSummary>`: make the previous installed runtime current (reversible). Refused while the server runs; replaces the cached selection.
- `uninstall(&self, runtime_id) -> Result<()>`: delete `runtimes/<id>` only. Never touches models. Refused for the runtime in use while the server runs.
- `log_tail(&self, n) -> Vec<LogLine>`: newest llama-server log lines, oldest first.

Modules:

- `hardware`: `HardwareInfo`, `CpuInfo`, `GpuInfo`, `GpuVendor`, `Os`, `HardwareProbe`, `SystemProbe`; pure parsers `parse_windows_probe`, `parse_cpuinfo`, `parse_meminfo`, `parse_lspci`, `parse_rocminfo`, `build_linux_gpus`; `amd_arch_from_name`.
- `manifest`: `Manifest`, `Backend`, `parse_manifest_index`, `select_runtime(&HardwareInfo, &[Manifest]) -> Result<Selection>`, `Selection { degraded, reason, rejected, .. }`.
- `install`: `Fetcher` trait and `ReqwestFetcher`; `RuntimeInstaller::{install, current, installed_version, rollback, uninstall}`; `InstalledRuntime` (serde); `safe_entry_path`, `locate_binary`, `extract_archive`.
- `config`: `LaunchConfig` (builder, `validate`, `args`, `env`, `endpoint`, `embeddings` for `--embeddings`, off by default), `KvCacheType`, `ModelMeta`, `ModelArch` (`kv_layers`, `recurrent_bytes`, `offload_layers`; `ModelArch::dense`), `Recommendation`, `MemoryEstimate`, `recommend`, `recommend_with_arch`, `headroom`.
- `fit`: `FitProjection` (`ctx`, `model_mib`, `context_mib`, `compute_mib`, `total_mib`), `parse_fit_projection` (accepts the `--fit-print` rows and the server's log table rows), `probe_args`, `probe`, `FIT_PARAMS_BINARY`, `SERVER_DEFAULT_SLOTS`.
- `devices`: `DevicePreflight`, `RuntimeDevice`, `parse_list_devices`, `preflight`, `list_devices`, `HipSdkSearch` (Windows: `HIP_PATH` then `%ProgramFiles%\AMD\ROCm\<version>\bin`, newest first), `launch_env` (PATH with SDK folders first), `HipDependencies` / `check_hip_dependencies` (reads `ggml-hip.dll` for a `hipblas.dll` import; never copies or renames DLLs), `BackendLoadFailure`, `FailureCause`, `diagnose_no_devices`.
- `process`: `ServerManager::{start, start_with_env, stop, restart, status, log_tail, log_text}`, `ServerState`, `ServerStatus`, `RestartPolicy`, `Timing`, `Spawner` trait and `SystemSpawner`, `LogSink` with `FileLogSink`, `diagnose_log`.
- `verify`: `parse_server_log(&str) -> LogFacts`, `verify_gpu(Backend, &LogFacts) -> GpuVerdict`, `GpuTelemetry` trait, `RocmSmiTelemetry`, `parse_rocm_smi_json`, `apply_telemetry`.
- `perfmem`: `PerfStore::{open, record, best_known, lookup, save}`, `BenchRecord`, `RecordKey`, `ConfigSnapshot`, `RecordOutcome`, `compare`, `machine_fingerprint`, `migrate`.
- `tokenizer`: `ServerTokenCounter` (implements `raptor_core::TokenCounter`), `TokenizeTransport`, `LoopbackTransport`.
- `sys` / `http` / `hash`: `CommandRunner`, `PortCheck`, `HttpClient`, `sha256_file`, and related helpers.

## Status

**Implemented and tested:** 215 unit tests pass (`cargo test -p raptor-runtime`), plus the
real-hardware gate (`tests/gate_real_hardware.rs`, `#[ignore]`, see its header) and one manual
test (`--ignored real_probe`). `clippy -D warnings` and `fmt` are clean for this crate.

**Real-hardware gate (RX 9060 XT gfx1200, Windows, HIP SDK 7.1 and 7.2, llama.cpp b12340, Qwen3.8-27B-UD-IQ3_S, 66 layers):**

- `exp` runtime (GPU works): device `ROCm0` listed; `Verified`, 66/66 layers offloaded; ready in 6.4 s; one 20-token chat at 14.6 tok/s wall, 21.1 tok/s server-reported; 37 log lines (3.6 KB) per request at `-lv 4`; pid gone after stop.
- Stock `bin` runtime with the default SDK search (`HIP_PATH` then ROCm 7.2 first): also `Verified` 66/66 at 21.1 tok/s. SDK 7.2 ships `hipblas.dll`, so the stock zip works with it.
- Stock `bin` runtime with HIP SDK 7.1 as the only SDK (ships `libhipblas.dll` only): the load is refused with `hipblas_name_mismatch`, no process starts, status shows `Stopped`.

- **hardware:** Windows probe runs one PowerShell call and reads the display-class registry for VRAM (`Win32_VideoController.AdapterRAM` is 32-bit and wrong for 16 GB cards). Verified on the development machine: it reports the RX 9060 XT with arch `gfx1200` and about 16 GB VRAM. Linux probe reads /proc and /sys and uses `lspci -Dnn` and `rocminfo`.
- **manifest:** HTTPS and SHA-256 validation, index parsing with duplicate detection, selection by backend rank then version. A GPU build is never selected for an incompatible arch or driver. Without a compatible GPU build, a CPU build is selected with `degraded = true` and an explanation.
- **install:** download (streamed, cancellable), SHA-256 verification before extraction, zip and tar.gz extraction with zip-slip and link rejection, staging then atomic rename, idempotent reinstall, previous version kept, `rollback` (reversible). Partial files are removed on any failure.
- **devices:** `--list-devices` output parsed (`ROCm0: AMD Radeon RX 9060 XT (16304 MiB, 16158 MiB free)`). A GPU backend with zero devices of its kind gets a typed `BackendLoadFailure`: `hipblas_name_mismatch` (ggml-hip.dll imports `hipblas.dll`, the SDK has only `libhipblas.dll`), `hipblas_missing`, `hip_sdk_not_found`, or `no_device_visible`, each with suggestions. CPU runtimes skip the check.
- **memory estimate and llama.cpp projection (b12340, measured):** llama.cpp projects device memory itself. The server log at load prints a `common_memory_breakdown_print` row, and `llama-fit-params --fit-print on` prints the same as `ROCm0 <model> <context> <compute>` on stdout with no weights loaded (about 0.6 s). With `-np 4` (the server default here), the context column matches the server's own log exactly (854 MiB at ctx 4096; 1110 at ctx 8192). Fit flags of `llama-server`: `-fit on` (default) adjusts only unset `-ngl`/`-c`, `-fitt` is the free-memory margin per device (default 1024 MiB), `-fitc` is the smallest context it may choose (default 4096). Raptor always sets `-ngl` and `-c`, so the server never adjusts them and the launch stays deterministic. Qwen3.8-27B-UD-IQ3_S at ctx 8192 projects 12253 MiB (10616 weights, 1110 KV plus recurrent state, 527 compute). `recommend_with_arch` calibrates weights, recurrent state and compute from that projection and applies the headroom rule to the calibrated total. A projection that does not match the model is ignored and the rationale says so.
- **Real gate, Qwen3.8-27B-UD-IQ3_S, same session:** the shell's old config (65 of 66 layers, ctx 8192) verified `partial offload 65/66`, 14.8 tok/s server-reported (10.5 tok/s wall, 20 tokens). The calibrated recommendation (66 of 66, ctx 8192, from `gate_recommended_config_full_offload`) verified `offloaded 66/66`, 21.0 tok/s server-reported (14.3 tok/s wall). The existing ctx 4096 gate still verifies 66/66 at 21.0 tok/s. Single short requests, one run each.
- **config:** builder validates cross-field rules (quantized KV needs flash attention, GPU layers need a GPU backend, non-loopback host needs an API key). The API key goes in the environment, not argv. Every launch passes `-lv 4`: the llama-server default (3) omits the device and offload lines that verification needs. Measured on b12340: levels 0 to 3 emit no such lines, level 4 emits both. `recommend` reduces context before layers, falls back from F16 to q8_0 KV only when F16 cannot fit, keeps 10% or 512 MiB headroom, and returns an error rather than a CPU config when no GPU layers fit.
- **process:** start with Ready gating on `/health` and `/v1/models`, port pre-check, log-based diagnosis (OOM, unsupported arch, missing file, port in use), crash detection, exponential-backoff restarts up to a limit, Degraded on repeated health failures, bounded log ring buffer plus JSON-lines sink, graceful-then-forced tree kill. Covered by fakes under paused time, plus real-process tests for output, exit code and stop.
- **verify:** parsers for HIP, CUDA, Vulkan and CPU log shapes. `Verified` needs the requested backend's device plus offloaded layers or device buffers. Zero offloaded layers for a GPU request gives `CpuOnly` with a `WARNING`. Missing evidence gives `Unverified`, never "GPU active". Telemetry can only add evidence or downgrade.
- **perfmem:** versioned JSON store (v1 to v2 migration), atomic save, same-config merge with weighted averages, unstable runs never replace stable ones, `best_known` returns stable records only, corrupt files are an error and are not overwritten.
- **tokenizer:** `/tokenize` over a blocking loopback-only HTTP client (refuses non-loopback), chunked-response decoding, cache, and a flagged `EstimatingCounter` fallback.

**Stubbed or not verified (open issues):**

- `ensure_runtime` promotes the newest installed version that matches, so a later `ensure_runtime` (for example at startup) undoes a `rollback`. A persistent rollback needs its own selection rule.
- Log lines reach the ring through a separate pump task, so `status()` right after `load_model` can miss the GPU lines and report `Unverified` for a moment. The gate test waits for the lines before it reads the verdict.
- The HIP SDK search prefers the newest installed version. Mixing SDK and runtime versions is not tested beyond the 7.2 and 7.1 runs above.
- `use_local_runtime` skips hash verification by design (developer builds). It is not reachable from the shell.
- The `hipblas.dll` import check is a byte scan of `ggml-hip.dll` (no PE parser). It is accurate for the stock zip, not proven for every build.
- `devices` on Linux is untested on real hardware; the HIP SDK search is empty there.

- The Linux probe, rocminfo, lspci, rocm-smi and sysfs parsers are tested only with hand-written fixtures in the documented formats. They have not been run against real Linux output.
- The llama.cpp log fixtures are representative, not captured from a pinned build. Assumed flags to confirm against the pinned runtime: `-fa on|off`, `--cache-type-k/-v`, `LLAMA_API_KEY`, and `/tokenize` defaulting to no special tokens.
- Manifest `signature` is reserved and not verified.
- No download size cap (zip-bomb guard) and no cross-process install lock.
- `RocmSmiTelemetry` exists but the facade does not poll it.
- `perfmem` is not wired into the facade and there is no benchmark runner.
- `recommend` (the dense path, `recommend_config` with `ModelArch::dense`) uses the GGUF-file-size estimate with a fixed 384 MiB compute overhead and assumes every layer holds KV. For Qwen3.8-27B-UD-IQ3_S that gives 65 of 66 layers at ctx 8192. The shell no longer calls it on the load or preview path: `src-tauri` (`AppState::recommend_for_entry`) builds `ModelArch` from `raptor_models::runtime_hints` and calls `recommend_config_with_arch`, passing llama.cpp's `probe_fit` projection on a GPU runtime (about 0.6 s, no weights). A failed projection keeps the GGUF estimate and the rationale says so. `ModelArch.recurrent_bytes` is the one-slot hint times `SERVER_DEFAULT_SLOTS` (4).
- `recommend_with_arch` reads free VRAM from the hardware probe, which reports none on Windows. The estimate then uses total VRAM (marked optimistic in the rationale). `devices` preflight has the real free figure (16158 MiB) and is not used yet for this.
- The GGUF-based path does not parse GGUF; the caller supplies `ModelMeta` and `ModelArch`.
- The restart counter does not reset during a session.
- On Windows, `taskkill` without `/F` usually fails for console processes, so a stop typically reaches the forced kill after the grace period.
- Windows arch comes from a name table (`amd_arch_from_name`) when the platform does not report it.
- `ServerTokenCounter::is_exact` reflects the most recent call.
