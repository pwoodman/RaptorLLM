//! Lifecycle tests for `AppState` on fakes: load, benchmark, known-good selection, crash
//! recovery, shutdown and the setup checklist. No GPU, network or llama-server is involved.

use super::*;
use crate::adapters::fakes::{counting_connect, FakeUpstream};
use crate::testing::{
    fixed_hardware, hip_manifest, llama_zip, write_gguf, FakeFetcher, FakeLaunch, FakeProbe,
    FakeSpawner, FixedTransport, FreePorts, HealthyHttp, HIP_URL,
};
use raptor_runtime::hardware::HardwareProbe;
use raptor_runtime::process::NullSink;
use serde_json::json;
use std::time::Duration;

/// The bundled runtime index, used by the setup checklist.
const BUNDLED_INDEX: &str = include_str!("../../resources/runtime-manifest.json");

struct Harness {
    /// The archive both fake runtimes download.
    zip: Vec<u8>,
    _dir: tempfile::TempDir,
    state: AppState,
    spawner: Arc<FakeSpawner>,
    fetcher: Arc<FakeFetcher>,
    probe: Arc<FakeProbe>,
    upstream: Arc<FakeUpstream>,
    model_id: String,
}

fn timing() -> Timing {
    Timing {
        startup_timeout: Duration::from_secs(5),
        poll_interval: Duration::from_millis(5),
        health_interval: Duration::from_secs(3600),
        stop_grace: Duration::from_millis(50),
        failures_before_degraded: 2,
    }
}

/// Builds a shell whose runtime is installed (unless `install` is false) and whose library
/// holds one tiny model.
async fn harness(scripts: Vec<FakeLaunch>, max_restarts: u32, install: bool) -> Harness {
    harness_with_devices(scripts, max_restarts, install, None).await
}

/// Like `harness`, with the output `llama-server --list-devices` prints in the fake (`None` keeps
/// the RX 9060 XT listing).
async fn harness_with_devices(
    scripts: Vec<FakeLaunch>,
    max_restarts: u32,
    install: bool,
    devices: Option<&str>,
) -> Harness {
    let dir = tempfile::tempdir().unwrap();
    let dirs = DataDirs::new(dir.path().join("RaptorLLM"));
    dirs.ensure().unwrap();
    let settings = Arc::new(SettingsStore::with_settings(
        dirs.config().join("settings.json"),
        Settings::default(),
    ));
    let zip = llama_zip();
    let fetcher = Arc::new(
        FakeFetcher::default()
            .serve(HIP_URL, zip.clone())
            .serve(crate::testing::VULKAN_URL, zip.clone()),
    );
    let spawner = Arc::new(FakeSpawner::new(scripts));
    let probe = Arc::new(FakeProbe::new(fixed_hardware()));
    let upstream = Arc::new(FakeUpstream {
        loaded: Some("tiny.gguf".into()),
        ..Default::default()
    });
    let probe_dyn: Arc<dyn HardwareProbe> = probe.clone();
    let deps = ShellDeps {
        install_progress: Default::default(),
        gpu_memory: Arc::new(crate::testing::FakeGpuMemory::new(&[
            (600, 480),
            (9_000, 520),
        ])),
        engine: EngineDeps {
            probe: probe_dyn,
            fetcher: fetcher.clone(),
            spawner: spawner.clone(),
            http: Arc::new(HealthyHttp),
            ports: Arc::new(FreePorts),
            sink: Arc::new(NullSink),
            restart: RestartPolicy {
                max_restarts,
                base_backoff: Duration::from_millis(5),
                max_backoff: Duration::from_millis(10),
                stable_window: Duration::from_secs(3600),
            },
            timing: timing(),
            runner: Arc::new(match devices {
                Some(out) => crate::testing::FakeDeviceRunner {
                    output: out.to_string(),
                },
                None => crate::testing::FakeDeviceRunner::default(),
            }),
            hip_sdk: raptor_runtime::devices::HipSdkSearch::default(),
        },
        connect: counting_connect(upstream.clone(), Arc::new(Mutex::new(0))),
        tokenizer: Arc::new(FixedTransport(300)),
        llama_http: Arc::new(crate::semantic::fakes::FakeLlama::new()),
    };
    let state = AppState::from_parts(dirs.clone(), settings, vec![], deps).unwrap();
    if install {
        state
            .runtime
            .ensure_runtime(&[hip_manifest(&zip)], &CancellationToken::new())
            .await
            .unwrap();
    }
    let src = dir.path().join("tiny.gguf");
    write_gguf(&src);
    let entry = state.models.library().import_copy(&src).unwrap();
    Harness {
        zip,
        _dir: dir,
        state,
        spawner,
        fetcher,
        probe,
        upstream,
        model_id: entry.id,
    }
}

/// Makes the next benchmark stream `tokens` generated tokens.
fn queue_reply(up: &FakeUpstream, tokens: usize) {
    let mut items: Vec<raptor_core::Result<String>> = (0..tokens)
        .map(|i| Ok(json!({"choices": [{"delta": {"content": format!("t{i}")}}]}).to_string()))
        .collect();
    items.push(Ok("[DONE]".to_string()));
    *lock(&up.streamed) = Some(items);
}

/// Polls until `done` holds, or fails after five seconds.
async fn wait_until(mut done: impl FnMut() -> bool) {
    for _ in 0..500 {
        if done() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("condition not reached in time");
}

#[tokio::test]
async fn first_load_uses_the_recommendation_and_records_a_stable_benchmark() {
    let h = harness(vec![FakeLaunch::hip()], 0, true).await;
    queue_reply(&h.upstream, 3);
    let result = h.state.load_model(&h.model_id, None, None).await.unwrap();

    assert_eq!(result.config_source, ConfigSource::Recommended);
    assert!(!result.rationale.is_empty());
    assert!(result.benchmark.stable, "{:?}", result.benchmark.reasons);
    assert_eq!(result.benchmark.recorded, "inserted");
    assert!(result
        .benchmark
        .gpu_verdict
        .as_ref()
        .is_some_and(|v| v.is_verified()));
    assert_eq!(
        h.state.current_model_id().as_deref(),
        Some(h.model_id.as_str())
    );

    let history = h.state.perf_history(Some(&h.model_id)).unwrap();
    assert_eq!(history.len(), 1);
    let rec = &history[0].record;
    assert!(rec.stable);
    assert!(rec.gen_tps > 0.0 && rec.ttft_ms >= 0.0);
    assert_eq!(rec.runtime_id, "llama-hip-win");
    let expected_sha = h.state.models.library().get(&h.model_id).unwrap().sha256;
    assert_eq!(rec.model_hash, expected_sha, "keyed by the model's sha256");
}

#[tokio::test]
async fn second_load_prefers_the_known_good_configuration() {
    let h = harness(vec![FakeLaunch::hip(), FakeLaunch::hip()], 0, true).await;
    queue_reply(&h.upstream, 3);
    h.state.load_model(&h.model_id, None, None).await.unwrap();
    h.state.unload_model().await.unwrap();

    queue_reply(&h.upstream, 3);
    let second = h.state.load_model(&h.model_id, None, None).await.unwrap();
    assert_eq!(second.config_source, ConfigSource::KnownGood);
    assert!(second.rationale.iter().any(|r| r.contains("Known-good")));
    assert_eq!(second.benchmark.recorded, "merged");
    let history = h.state.perf_history(None).unwrap();
    assert_eq!(history.len(), 1, "same config is merged, not duplicated");
    assert_eq!(history[0].record.samples, 2);
}

#[tokio::test]
async fn an_unstable_rerun_is_recorded_as_such_and_never_replaces_the_known_good() {
    let h = harness(vec![FakeLaunch::hip(), FakeLaunch::hip()], 0, true).await;
    queue_reply(&h.upstream, 3);
    h.state.load_model(&h.model_id, None, None).await.unwrap();
    h.state.unload_model().await.unwrap();

    *lock(&h.upstream.streamed) = Some(vec![Err(RaptorError::Network("connection reset".into()))]);
    let second = h.state.load_model(&h.model_id, None, None).await.unwrap();
    assert!(!second.benchmark.stable);
    assert!(second
        .benchmark
        .reasons
        .iter()
        .any(|r| r.contains("benchmark failed") && r.contains("connection reset")));
    assert_eq!(second.benchmark.recorded, "kept_stable");
    assert!(second.benchmark.notes.iter().any(|n| n.contains("kept")));

    let best = h
        .state
        .perf_best_known(&h.model_id)
        .await
        .unwrap()
        .expect("the stable record survives");
    assert!(best.record.stable);
    assert_eq!(best.record.samples, 1);
    assert_eq!(h.state.perf_history(None).unwrap().len(), 1);
}

#[tokio::test]
async fn a_different_context_uses_the_recommendation_and_says_so() {
    let h = harness(vec![FakeLaunch::hip(), FakeLaunch::hip()], 0, true).await;
    queue_reply(&h.upstream, 3);
    let first = h
        .state
        .load_model(&h.model_id, Some(65_536), None)
        .await
        .unwrap();
    h.state.unload_model().await.unwrap();

    queue_reply(&h.upstream, 3);
    let second = h
        .state
        .load_model(&h.model_id, Some(98_304), None)
        .await
        .unwrap();
    assert_eq!(first.config_source, ConfigSource::Recommended);
    assert_eq!(second.config_source, ConfigSource::Recommended);
    assert!(second
        .rationale
        .iter()
        .any(|r| r.contains("65536-token context")));

    let a = first.benchmark.entry.expect("entry").id;
    let b = second.benchmark.entry.expect("entry").id;
    let cmp = h.state.perf_compare(&a, &b).unwrap();
    assert!(cmp.comparison.gen_tps_delta_pct.is_finite());
    assert!(h.state.perf_compare(&a, "missing").is_err());
    assert_eq!(h.state.perf_history(None).unwrap().len(), 2);
}

#[tokio::test]
async fn a_crash_leaves_no_stale_loaded_model_and_a_retry_loads_again() {
    let h = harness(
        vec![
            FakeLaunch::hip().crashing_after(Duration::from_millis(300)),
            FakeLaunch::hip(),
        ],
        0,
        true,
    )
    .await;
    queue_reply(&h.upstream, 3);
    h.state.load_model(&h.model_id, None, None).await.unwrap();
    assert_eq!(
        h.state.current_model_id().as_deref(),
        Some(h.model_id.as_str())
    );

    let runtime = h.state.runtime.clone();
    wait_until(|| matches!(runtime.status().server.state, ServerState::Crashed { .. })).await;
    assert!(
        h.state.current_model_id().is_none(),
        "a crashed server must not report a loaded model"
    );

    queue_reply(&h.upstream, 3);
    let again = h.state.load_model(&h.model_id, None, None).await.unwrap();
    assert_eq!(again.status.server.state, ServerState::Ready);
    assert_eq!(
        h.state.current_model_id().as_deref(),
        Some(h.model_id.as_str())
    );
    assert_eq!(h.spawner.spawn_count(), 2, "the retry started a new server");
}

#[tokio::test]
async fn a_failed_start_leaves_nothing_loaded_and_is_reported() {
    // Every variant of the speed plan fails, so the load is refused with the last error.
    let h = harness(
        vec![FakeLaunch::failing("cannot start the server"); 4],
        0,
        true,
    )
    .await;
    let err = h
        .state
        .load_model(&h.model_id, None, None)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("cannot start the server"), "{err}");
    assert_eq!(
        h.spawner.spawn_count(),
        4,
        "both the speculative and plain launches with multiple slots, then with one slot"
    );
    assert!(h.state.current_model_id().is_none());
    assert!(lock(&h.state.loaded).is_none());
    assert!(
        h.state.perf_history(None).unwrap().is_empty(),
        "nothing was benchmarked"
    );
}

#[tokio::test]
async fn a_launch_that_fails_with_speculation_is_retried_without_it_and_says_so() {
    let h = harness(
        vec![
            FakeLaunch::failing("failed to create MTP context"),
            FakeLaunch::hip(),
        ],
        0,
        true,
    )
    .await;
    queue_reply(&h.upstream, 2);
    let result = h.state.load_model(&h.model_id, None, None).await.unwrap();
    assert_eq!(h.spawner.spawn_count(), 2);
    assert!(
        result
            .rationale
            .iter()
            .any(|r| r.contains("did not start") && r.contains("MTP")),
        "{:?}",
        result.rationale
    );
    let config = lock(&h.state.loaded).as_ref().unwrap().config.clone();
    assert_eq!(
        config.speculative, None,
        "the variant that started has no speculation"
    );
}

/// Saves a tuning for the harness's model, as `tune_model` would.
async fn save_tuning(h: &Harness, gpu_layers: u32, ubatch: u32) {
    let entry = h.state.models.library().get(&h.model_id).unwrap();
    let hw = h.state.hardware_for_use().await.unwrap();
    let runtime = h.state.runtime.status().runtime.unwrap();
    let key = h.state.tuned_key(&hw, &entry.sha256, &runtime);
    lock(&h.state.tuned)
        .put(
            &key,
            crate::tuned::TunedEntry {
                qualification: None,
                gpu_layers,
                ubatch_size: ubatch,
                prompt_tps: 2000.0,
                gen_tps: 60.0,
                seconds: 6.0,
                baseline_seconds: 9.0,
                measured_at: 1,
                profile: Some(crate::tuned::TuneProfile {
                    kv_cache: raptor_runtime::config::KvCacheType::F16,
                    flash_attn: true,
                }),
                baseline_live: None,
                best_live: None,
                long: None,
            },
        )
        .unwrap();
}

#[tokio::test]
async fn legacy_tuning_without_full_window_proof_does_not_earn_the_badge() {
    let h = harness(vec![FakeLaunch::hip()], 0, true).await;
    queue_reply(&h.upstream, 2);
    save_tuning(&h, 20, 256).await;
    let result = h.state.load_model(&h.model_id, None, None).await.unwrap();
    assert!(!result.tuning_applied);
}

async fn save_qualified(h: &Harness, id: &str) {
    let entry = h.state.models.library().get(id).unwrap();
    let path = h.state.models.library().file_path(id).unwrap();
    let hw = h.state.hardware_for_use().await.unwrap();
    let runtime = h.state.runtime.status().runtime.unwrap();
    let rec = h
        .state
        .recommend_for_entry(&entry, &path, 65536)
        .await
        .unwrap();
    let mut config = qualification::stock_config(rec.config.with_model(path).unwrap(), 65536);
    config.gpu_layers = 20;
    config.ubatch_size = 256;
    config.kv_cache = raptor_runtime::config::KvCacheType::Q4_0;
    config.flash_attn = true;
    let key = h.state.tuned_key(&hw, &entry.sha256, &runtime);
    lock(&h.state.tuned)
        .put(
            &key,
            crate::tuned::TunedEntry {
                qualification: Some(crate::tuned::Qualification {
                    context_size: 65536,
                    signature: qualification::signature(&config),
                    pairs: vec![(15.0, 10.0); 3],
                }),
                gpu_layers: 20,
                ubatch_size: 256,
                measured_at: 1,
                profile: Some(crate::tuned::TuneProfile {
                    kv_cache: config.kv_cache,
                    flash_attn: true,
                }),
                prompt_tps: 0.0,
                gen_tps: 0.0,
                seconds: 10.0,
                baseline_seconds: 15.0,
                baseline_live: None,
                best_live: None,
                long: None,
            },
        )
        .unwrap();
}

#[tokio::test]
async fn verified_settings_apply_only_at_the_measured_window_and_slot_count() {
    let h = harness(vec![FakeLaunch::hip(); 3], 0, true).await;
    let id = add_model(&h, &[("llama.context_length", 131072)]);
    save_qualified(&h, &id).await;
    for (ctx, slots, applied) in [
        (65536, None, true),
        (98_304, None, false),
        (65536, Some(2), false),
    ] {
        queue_reply(&h.upstream, 2);
        let load = h
            .state
            .load_model_ext(
                &id,
                Some(ctx),
                None,
                LoadExtras {
                    slots,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(load.tuning_applied, applied);
        if applied {
            let config = lock(&h.state.loaded).as_ref().unwrap().config.clone();
            assert_eq!(
                (
                    config.ctx_size,
                    config.parallel_slots,
                    config.gpu_layers,
                    config.ubatch_size
                ),
                (65536, Some(1), 20, 256)
            );
        }
        h.state.unload_model().await.unwrap();
    }
}

#[tokio::test]
async fn an_ordinary_fallback_does_not_inherit_qualification() {
    // Two failures exhaust the verified profile's attempts (speculation variant and plain);
    // the ordinary plan's attempt loads, and the verification must not travel with it.
    let h = harness(
        vec![
            FakeLaunch::failing("out of memory"),
            FakeLaunch::failing("out of memory"),
            FakeLaunch::hip(),
        ],
        0,
        true,
    )
    .await;
    let id = add_model(&h, &[("llama.context_length", 131072)]);
    save_qualified(&h, &id).await;
    queue_reply(&h.upstream, 2);
    let result = h.state.load_model(&id, None, None).await.unwrap();
    assert!(!result.tuning_applied);
    assert_eq!(h.spawner.spawn_count(), 3);
}

#[tokio::test]
async fn without_a_saved_tuning_nothing_is_claimed_as_tuned() {
    let h = harness(vec![FakeLaunch::hip()], 0, true).await;
    queue_reply(&h.upstream, 2);
    let result = h.state.load_model(&h.model_id, None, None).await.unwrap();
    assert!(!result.tuning_applied);
    assert!(!result
        .rationale
        .iter()
        .any(|r| r.starts_with("Tuned on this machine")));
}

#[tokio::test]
async fn tuning_requires_an_idle_server_and_a_model_that_supports_the_floor() {
    let h = harness(vec![FakeLaunch::hip()], 0, true).await;
    queue_reply(&h.upstream, 2);
    h.state.load_model(&h.model_id, None, None).await.unwrap();
    let err = h.state.tune_model(&h.model_id, false).await.unwrap_err();
    assert!(err.to_string().contains("unload the model"), "{err}");

    h.state.unload_model().await.unwrap();
    let err = h.state.tune_model(&h.model_id, false).await.unwrap_err();
    assert!(
        err.to_string().contains("required") || err.to_string().contains("vocabulary"),
        "{err}"
    );
    let key = h.state.tuned_key(
        &h.state.hardware_for_use().await.unwrap(),
        &h.state.models.library().get(&h.model_id).unwrap().sha256,
        &h.state.runtime.status().runtime.unwrap(),
    );
    assert!(
        lock(&h.state.tuned).get(&key).is_none(),
        "a failed run saves nothing"
    );
}

#[tokio::test]
async fn the_parallel_slot_setting_reaches_the_server_launch() {
    let h = harness(vec![FakeLaunch::hip()], 0, true).await;
    queue_reply(&h.upstream, 2);
    h.state
        .settings
        .update(|s| {
            s.parallel_slots = 1;
            Ok(())
        })
        .unwrap();
    h.state.load_model(&h.model_id, None, None).await.unwrap();
    let config = lock(&h.state.loaded).as_ref().unwrap().config.clone();
    assert_eq!(config.parallel_slots, Some(1));
    assert!(config.args().windows(2).any(|w| w == ["-np", "1"]));
}

#[tokio::test]
async fn a_larger_context_option_sets_the_cache_type_window_slots_and_every_layer() {
    use raptor_runtime::config::KvCacheType;
    let h = harness(vec![FakeLaunch::hip()], 0, true).await;
    queue_reply(&h.upstream, 2);
    let result = h
        .state
        .load_model_ext(
            &h.model_id,
            Some(98_304),
            None,
            LoadExtras {
                kv_cache: Some(KvCacheType::Q8_0),
                slots: Some(1),
                speculation: Some(false),
            },
        )
        .await
        .unwrap();
    let config = lock(&h.state.loaded).as_ref().unwrap().config.clone();
    assert_eq!(config.kv_cache, KvCacheType::Q8_0);
    assert!(config.flash_attn, "a quantized cache needs flash attention");
    assert_eq!(config.ctx_size, 98_304);
    assert_eq!(config.parallel_slots, Some(1));
    assert!(config
        .args()
        .windows(2)
        .any(|w| w == ["--cache-type-k", "q8_0"]));
    assert!(
        result
            .rationale
            .iter()
            .any(|r| r.contains("98304 tokens") && r.contains("q8_0")),
        "{:?}",
        result.rationale
    );
}

#[tokio::test]
async fn without_a_projection_tool_only_the_default_window_is_offered() {
    let h = harness(vec![FakeLaunch::hip()], 0, true).await;
    let options = h.state.context_options(&h.model_id).await.unwrap();
    assert_eq!(options.len(), 1);
    assert_eq!(options[0].id, "default");
    assert!(options[0].ctx >= 256);
}

fn script_gpu_memory(h: &Harness, readings: &[(u64, u64)]) {
    *lock(&h.state.gpu_memory) = Arc::new(crate::testing::FakeGpuMemory::new(readings));
}

#[tokio::test]
async fn a_load_that_fits_says_nothing_spilled() {
    let h = harness(vec![FakeLaunch::hip()], 0, true).await;
    queue_reply(&h.upstream, 2);
    script_gpu_memory(&h, &[(586, 483), (13_807, 631)]);
    let result = h.state.load_model(&h.model_id, None, None).await.unwrap();
    let m = result.gpu_memory.expect("a reading was available");
    assert!(!m.spilled && !m.expected);
    assert_eq!(m.dedicated_mib, 13_807);
    assert!(
        result
            .rationale
            .iter()
            .any(|r| r.contains("nothing spilled")),
        "{:?}",
        result.rationale
    );
}

#[tokio::test]
async fn a_dense_model_that_spills_into_shared_memory_is_warned_about() {
    let h = harness(vec![FakeLaunch::hip()], 0, true).await;
    queue_reply(&h.upstream, 2);
    script_gpu_memory(&h, &[(586, 483), (15_609, 2_281)]);
    let result = h.state.load_model(&h.model_id, None, None).await.unwrap();
    let m = result.gpu_memory.unwrap();
    assert!(m.spilled && !m.expected, "{m:?}");
    assert_eq!(m.shared_growth_mib, 1_798);
    assert!(
        result
            .rationale
            .iter()
            .any(|r| r.starts_with("WARNING") && r.contains("1798 MiB")),
        "{:?}",
        result.rationale
    );
}

#[tokio::test]
async fn an_unreadable_gpu_memory_counter_is_reported_as_unknown_not_as_fine() {
    let h = harness(vec![FakeLaunch::hip()], 0, true).await;
    queue_reply(&h.upstream, 2);
    script_gpu_memory(&h, &[]);
    let result = h.state.load_model(&h.model_id, None, None).await.unwrap();
    assert!(result.gpu_memory.is_none());
    assert!(!result
        .rationale
        .iter()
        .any(|r| r.contains("nothing spilled")));
}

#[tokio::test]
async fn the_context_matrix_is_refused_while_loaded_and_saves_nothing_when_nothing_can_be_measured()
{
    let h = harness(vec![FakeLaunch::hip()], 0, true).await;
    queue_reply(&h.upstream, 2);
    h.state.load_model(&h.model_id, None, None).await.unwrap();
    let err = h
        .state
        .context_matrix(&h.model_id, 32_768)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("unload the model first"), "{err}");

    h.state.unload_model().await.unwrap();
    let err = h
        .state
        .context_matrix(&h.model_id, 32_768)
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("nothing could be measured"),
        "{err}"
    );
    assert!(h
        .state
        .last_context_matrix(&h.model_id)
        .await
        .unwrap()
        .is_none());
}

#[tokio::test]
async fn by_default_two_slots_are_used() {
    let h = harness(vec![FakeLaunch::hip()], 0, true).await;
    queue_reply(&h.upstream, 2);
    h.state.load_model(&h.model_id, None, None).await.unwrap();
    let config = lock(&h.state.loaded).as_ref().unwrap().config.clone();
    assert_eq!(config.parallel_slots, Some(2));
}

#[tokio::test]
async fn a_normal_load_asks_for_speculative_decoding() {
    let h = harness(vec![FakeLaunch::hip()], 0, true).await;
    queue_reply(&h.upstream, 2);
    let result = h.state.load_model(&h.model_id, None, None).await.unwrap();
    let config = lock(&h.state.loaded).as_ref().unwrap().config.clone();
    assert!(config.speculative.is_some());
    assert!(config.args().iter().any(|a| a == "--spec-type"));
    assert!(result
        .rationale
        .iter()
        .any(|r| r.contains("Speculative decoding")));
}

#[tokio::test]
async fn shutdown_stops_the_server_and_the_background_loops() {
    let h = harness(vec![FakeLaunch::hip()], 0, true).await;
    queue_reply(&h.upstream, 3);
    h.state.load_model(&h.model_id, None, None).await.unwrap();
    let background = h.state.background.clone();
    assert!(!background.is_cancelled());

    h.state.shutdown().await;

    assert!(background.is_cancelled());
    assert_eq!(h.state.runtime.status().server.state, ServerState::Stopped);
    assert!(h.spawner.stop_count() >= 1, "the child was asked to stop");
    assert!(h.state.current_model_id().is_none());
}

#[tokio::test]
async fn perf_run_benchmark_needs_a_ready_model() {
    let h = harness(vec![FakeLaunch::hip()], 0, true).await;
    assert!(matches!(
        h.state.perf_run_benchmark().await,
        Err(RaptorError::Conflict(_))
    ));
    queue_reply(&h.upstream, 3);
    h.state.load_model(&h.model_id, None, None).await.unwrap();
    queue_reply(&h.upstream, 3);
    let run = h.state.perf_run_benchmark().await.unwrap();
    assert!(run.stable, "{:?}", run.reasons);
    assert_eq!(run.recorded, "merged");
}

#[tokio::test]
async fn hardware_is_probed_once_and_then_cached() {
    let h = harness(vec![], 0, false).await;
    let before = h.probe.calls.load(std::sync::atomic::Ordering::SeqCst);
    h.state.hardware_for_use().await.unwrap();
    h.state.hardware_for_use().await.unwrap();
    let after = h.probe.calls.load(std::sync::atomic::Ordering::SeqCst);
    assert_eq!(after, before + 1, "the second call reads the cache");
    assert!(h.state.cached_hardware().is_some());
}

#[tokio::test]
async fn setup_status_reports_a_compatible_runtime_without_downloading_it() {
    let h = harness(vec![], 0, false).await;
    std::fs::write(h.state.runtime_manifest_path(), BUNDLED_INDEX).unwrap();

    let status = h.state.setup_status().await;
    let by_id = |id: &str| status.checks.iter().find(|c| c.id == id).unwrap().clone();
    assert!(by_id("hardware").ok);
    assert!(!by_id("runtime").ok, "nothing is installed yet");
    assert!(
        status.available_runtime.is_some(),
        "a compatible build is listed"
    );
    assert!(status.runtime_needs_confirmation);
    assert!(
        by_id("model").ok,
        "the fixture imports one model into the library"
    );
    assert_eq!(status.models_installed, 1);
    assert_eq!(h.fetcher.calls(), 0, "the checklist never downloads");
    assert_eq!(h.spawner.spawn_count(), 0);
}

#[tokio::test]
async fn restoring_the_runtime_never_downloads_one_that_is_not_installed() {
    let h = harness(vec![], 0, false).await;
    std::fs::write(h.state.runtime_manifest_path(), BUNDLED_INDEX).unwrap();

    assert!(!h.state.restore_installed_runtime().await.unwrap());
    assert!(h.state.runtime.status().runtime.is_none());
    assert_eq!(
        h.fetcher.calls(),
        0,
        "a missing runtime waits for the user's confirmation"
    );
}

#[tokio::test]
async fn restoring_the_runtime_keeps_an_already_selected_one() {
    let h = harness(vec![], 0, true).await;
    assert!(
        h.state.runtime.status().runtime.is_some(),
        "the harness installs one"
    );
    let before = h.fetcher.calls();
    assert!(h.state.restore_installed_runtime().await.unwrap());
    assert_eq!(h.fetcher.calls(), before);
}

#[tokio::test]
async fn setup_status_without_an_index_explains_what_is_missing() {
    let h = harness(vec![], 0, false).await;
    let status = h.state.setup_status().await;
    assert!(status.available_runtime.is_none());
    assert!(
        status.notes.iter().any(|n| n.contains("no runtime index")),
        "{:?}",
        status.notes
    );
    assert!(!status.runtime_needs_confirmation);
}

/// Turns on `allow_cpu_fallback` for the harness's shell.
fn allow_cpu_fallback(h: &Harness) {
    h.state
        .settings
        .update(|s| {
            s.allow_cpu_fallback = true;
            Ok(())
        })
        .unwrap();
}

#[tokio::test]
async fn a_gpu_without_a_device_is_refused_with_the_typed_diagnosis_and_nothing_starts() {
    let h = harness_with_devices(
        vec![FakeLaunch::hip()],
        0,
        true,
        Some("Available devices:\n"),
    )
    .await;
    let err = h
        .state
        .load_model(&h.model_id, None, None)
        .await
        .expect_err("no device must refuse the load");
    assert!(
        matches!(err.error, RaptorError::Runtime(_)),
        "{:?}",
        err.error
    );
    let failure = err
        .backend_failure
        .clone()
        .expect("the typed diagnosis is attached");
    assert!(
        matches!(
            failure.cause,
            raptor_runtime::devices::FailureCause::HipSdkNotFound
                | raptor_runtime::devices::FailureCause::NoDeviceVisible
        ),
        "{:?}",
        failure.cause
    );
    assert!(!failure.suggestions.is_empty());
    assert_eq!(h.spawner.spawn_count(), 0, "no llama-server is started");
    assert!(lock(&h.state.loaded).is_none());

    // The wire shape the UI reads: kind and message as before, plus backend_failure.
    let wire = serde_json::to_value(&err).unwrap();
    assert_eq!(wire["kind"], "Runtime");
    assert_eq!(
        wire["backend_failure"]["cause"],
        serde_json::json!(match failure.cause {
            raptor_runtime::devices::FailureCause::HipSdkNotFound => "hip_sdk_not_found",
            _ => "no_device_visible",
        })
    );
}

#[tokio::test]
async fn the_configured_embedding_model_is_launched_with_embeddings_and_pooling() {
    let h = harness(vec![FakeLaunch::hip()], 0, true).await;
    queue_reply(&h.upstream, 2);
    h.state
        .settings
        .update(|s| {
            s.embedding_model = Some(h.model_id.clone());
            Ok(())
        })
        .unwrap();
    h.state.load_model(&h.model_id, None, None).await.unwrap();
    let config = h
        .state
        .loaded
        .lock()
        .unwrap()
        .as_ref()
        .unwrap()
        .config
        .clone();
    assert!(config.embeddings, "semantic retrieval needs /v1/embeddings");
    assert_eq!(
        config.pooling.as_deref(),
        Some("mean"),
        "a chat model has no pooling of its own"
    );
    assert_eq!(
        config.ubatch_size, config.batch_size,
        "long chunks must fit one micro-batch"
    );
    let args = config.args();
    assert!(args.iter().any(|a| a == "--embeddings"));
}

#[tokio::test]
async fn a_model_that_is_not_the_embedding_model_is_launched_as_before() {
    let h = harness(vec![FakeLaunch::hip()], 0, true).await;
    queue_reply(&h.upstream, 2);
    h.state.load_model(&h.model_id, None, None).await.unwrap();
    let config = h
        .state
        .loaded
        .lock()
        .unwrap()
        .as_ref()
        .unwrap()
        .config
        .clone();
    assert!(!config.embeddings && config.pooling.is_none());
}

#[tokio::test]
async fn cpu_fallback_runs_on_cpu_only_when_allowed_and_says_so() {
    let h = harness_with_devices(
        vec![FakeLaunch::hip()],
        0,
        true,
        Some("Available devices:\n"),
    )
    .await;
    allow_cpu_fallback(&h);
    queue_reply(&h.upstream, 2);
    let result = h.state.load_model(&h.model_id, None, None).await.unwrap();
    assert!(result.cpu_fallback, "the fallback is reported loudly");
    assert!(result
        .rationale
        .iter()
        .any(|r| r.starts_with("CPU FALLBACK")));
    assert_eq!(h.spawner.spawn_count(), 1);
    assert!(
        result
            .status
            .device_preflight
            .as_ref()
            .is_some_and(|p| p.failure.is_some()),
        "the cause stays visible next to the fallback"
    );
}

#[tokio::test]
async fn a_normal_gpu_load_never_reports_cpu_fallback() {
    let h = harness(vec![FakeLaunch::hip()], 0, true).await;
    allow_cpu_fallback(&h);
    queue_reply(&h.upstream, 2);
    let result = h.state.load_model(&h.model_id, None, None).await.unwrap();
    assert!(!result.cpu_fallback);
    assert!(result.preset.is_none());
}

#[tokio::test]
async fn a_short_native_window_is_refused_before_launching_chat() {
    let h = harness(vec![], 0, true).await;
    let short = add_model(&h, &[("llama.context_length", 32_768)]);
    let err = h.state.load_model(&short, None, None).await.unwrap_err();
    assert!(matches!(err.error, RaptorError::Unsupported(_)));
    assert!(err.error.to_string().contains("65536"));
    assert_eq!(h.spawner.spawn_count(), 0);
}

#[tokio::test]
async fn a_preset_supplies_the_context_length_and_is_named_in_the_result() {
    let h = harness(vec![FakeLaunch::hip()], 0, true).await;
    let mut preset = raptor_models::Preset::with_defaults(&h.model_id, "Long");
    preset.n_ctx = 2048;
    h.state.models.presets().save(preset).unwrap();
    queue_reply(&h.upstream, 2);
    let result = h
        .state
        .load_model(&h.model_id, None, Some("Long"))
        .await
        .unwrap();
    assert_eq!(result.preset.as_deref(), Some("Long"));
    assert_eq!(
        lock(&h.state.loaded).as_ref().unwrap().config.ctx_size,
        65_536
    );

    // An explicit context above the floor still wins over the preset.
    queue_reply(&h.upstream, 2);
    h.state
        .load_model(&h.model_id, Some(73_728), Some("Long"))
        .await
        .unwrap();
    assert_eq!(
        lock(&h.state.loaded).as_ref().unwrap().config.ctx_size,
        73_728
    );
}

#[tokio::test]
async fn an_unknown_preset_is_refused_before_anything_is_stopped() {
    let h = harness(vec![FakeLaunch::hip()], 0, true).await;
    let err = h
        .state
        .load_model(&h.model_id, None, Some("Missing"))
        .await
        .expect_err("unknown preset");
    assert!(
        matches!(err.error, RaptorError::NotFound(_)),
        "{:?}",
        err.error
    );
    assert!(err.backend_failure.is_none());
    assert_eq!(h.spawner.spawn_count(), 0);
}

#[tokio::test]
async fn the_offloadable_layer_count_includes_the_mtp_layer() {
    // Qwen3.8-27B: 65 blocks (64 main + 1 NextN), and llama-server offloads 66 with the output
    // layer. Using the main stack alone said 65 and left one layer on the CPU.
    let h = harness(vec![], 0, true).await;
    let mut entry = h.state.models.library().get(&h.model_id).unwrap();
    entry.gguf.block_count = Some(65);
    entry.gguf.nextn_predict_layers = Some(1);
    assert_eq!(super::offload_layers_of(&entry), Some(66));
    entry.gguf.block_count = Some(40);
    entry.gguf.nextn_predict_layers = None;
    assert_eq!(super::offload_layers_of(&entry), Some(41));
    entry.gguf.block_count = None;
    assert_eq!(super::offload_layers_of(&entry), None);
}

// --- Per-model backend choice (ROCm / Vulkan) ---------------------------------------------------

use crate::backend_choice::BackendPref;
use crate::testing::{vulkan_manifest, write_gguf_with, BOTH_DEVICES};

/// Installs the fake Vulkan runtime next to the ROCm one, without selecting it.
async fn install_vulkan(h: &Harness) {
    h.state
        .runtime
        .install_runtime(&vulkan_manifest(&h.zip), &CancellationToken::new())
        .await
        .unwrap();
}

/// Adds a second model whose header carries `extra` fields, and returns its library id.
fn add_model(h: &Harness, extra: &[(&str, u32)]) -> String {
    let path = h._dir.path().join("second.gguf");
    write_gguf_with(&path, extra);
    h.state.models.library().import_copy(&path).unwrap().id
}

fn loaded_backend(h: &Harness) -> (Backend, String) {
    let l = lock(&h.state.loaded).clone().unwrap();
    (l.backend, l.runtime_id)
}

#[tokio::test]
async fn a_model_set_to_vulkan_runs_on_vulkan_and_the_next_model_returns_to_rocm() {
    let h = harness_with_devices(
        vec![FakeLaunch::vulkan(), FakeLaunch::hip()],
        0,
        true,
        Some(BOTH_DEVICES),
    )
    .await;
    install_vulkan(&h).await;
    h.state
        .set_backend_pref(&h.model_id, BackendPref::Vulkan)
        .unwrap();

    queue_reply(&h.upstream, 3);
    let first = h.state.load_model(&h.model_id, None, None).await.unwrap();
    assert_eq!(
        loaded_backend(&h),
        (Backend::Vulkan, "llama-vulkan-win".to_string())
    );
    assert!(first
        .rationale
        .iter()
        .any(|r| r.contains("Backend: Vulkan, as chosen")));
    h.state.unload_model().await.unwrap();

    // The same model back on Auto: a model that fits keeps the normal (ROCm) runtime.
    h.state
        .set_backend_pref(&h.model_id, BackendPref::Auto)
        .unwrap();
    assert!(h.state.settings.snapshot().backend_prefs.is_empty());
    queue_reply(&h.upstream, 3);
    h.state.load_model(&h.model_id, None, None).await.unwrap();
    assert_eq!(
        loaded_backend(&h),
        (Backend::Hip, "llama-hip-win".to_string())
    );
}

#[tokio::test]
async fn vulkan_chosen_but_not_installed_is_refused_and_nothing_starts() {
    let h = harness(vec![FakeLaunch::hip()], 0, true).await;
    h.state
        .set_backend_pref(&h.model_id, BackendPref::Vulkan)
        .unwrap();
    let err = h
        .state
        .load_model(&h.model_id, None, None)
        .await
        .expect_err("Vulkan is not installed");
    assert!(
        err.error
            .to_string()
            .contains("Vulkan runtime is not installed"),
        "{}",
        err.error
    );
    assert_eq!(h.spawner.spawn_count(), 0, "no silent run on ROCm");
    assert!(h.state.current_model_id().is_none());
    // The runtime in use is untouched.
    assert_eq!(
        h.state.runtime.status().runtime.unwrap().backend,
        Backend::Hip
    );
}

#[tokio::test]
async fn a_model_with_an_mtp_head_uses_ngram_speculation_on_vulkan_and_says_so() {
    let h = harness_with_devices(
        vec![FakeLaunch::vulkan(), FakeLaunch::hip()],
        0,
        true,
        Some(BOTH_DEVICES),
    )
    .await;
    install_vulkan(&h).await;
    let mtp = add_model(&h, &[("llama.nextn_predict_layers", 1)]);
    h.state.set_backend_pref(&mtp, BackendPref::Vulkan).unwrap();

    queue_reply(&h.upstream, 3);
    let on_vulkan = h.state.load_model(&mtp, None, None).await.unwrap();
    let config = lock(&h.state.loaded).as_ref().unwrap().config.clone();
    assert_eq!(
        config.speculative,
        Some(raptor_runtime::config::SpecMode::Ngram)
    );
    assert!(!config.args().iter().any(|a| a == "draft-mtp"));
    assert!(on_vulkan
        .rationale
        .iter()
        .any(|r| r.contains("multi-token-prediction") && r.contains("Vulkan")));
    h.state.unload_model().await.unwrap();

    // The same model on ROCm still gets its MTP head.
    h.state.set_backend_pref(&mtp, BackendPref::Rocm).unwrap();
    queue_reply(&h.upstream, 3);
    h.state.load_model(&mtp, None, None).await.unwrap();
    let config = lock(&h.state.loaded).as_ref().unwrap().config.clone();
    assert_eq!(
        config.speculative,
        Some(raptor_runtime::config::SpecMode::Mtp)
    );
}

#[tokio::test]
async fn the_backend_view_reports_what_a_load_would_use_and_why_a_choice_fails() {
    let h = harness(vec![], 0, true).await;
    let view = h.state.backend_view(&h.model_id).await.unwrap();
    assert_eq!(view.preference, BackendPref::Auto);
    assert_eq!(view.effective, "rocm");
    assert!(view.rocm_installed && !view.vulkan_installed && !view.spills);

    h.state
        .set_backend_pref(&h.model_id, BackendPref::Vulkan)
        .unwrap();
    let view = h.state.backend_view(&h.model_id).await.unwrap();
    assert_eq!(view.effective, "unavailable");
    assert!(view.note.as_deref().unwrap().contains("not installed"));

    install_vulkan(&h).await;
    let view = h.state.backend_view(&h.model_id).await.unwrap();
    assert_eq!(
        (view.effective.as_str(), view.vulkan_installed),
        ("vulkan", true)
    );
    assert!(h
        .state
        .set_backend_pref("no-such-model", BackendPref::Rocm)
        .is_err());
}

#[tokio::test]
async fn only_a_sparse_mixture_of_experts_model_larger_than_vram_counts_as_spilling() {
    let h = harness(vec![], 0, true).await;
    let hw = h.state.hardware_for_use().await.unwrap();
    let mut entry = h.state.models.library().get(&h.model_id).unwrap();
    assert!(!AppState::model_spills(&entry, &hw), "a tiny dense model");
    entry.size_bytes = 24 * 1024 * 1024 * 1024;
    assert!(
        !AppState::model_spills(&entry, &hw),
        "dense and large: all weights are read per token"
    );
    entry.gguf.expert_count = Some(256);
    entry.gguf.expert_used_count = Some(8);
    assert!(
        AppState::model_spills(&entry, &hw),
        "flint-like: 8 of 256 experts"
    );
    entry.size_bytes = 8 * 1024 * 1024 * 1024;
    assert!(!AppState::model_spills(&entry, &hw), "sparse but it fits");
}
