//! Verifies the log parser against excerpts captured from a real llama.cpp b12340 ROCm
//! build on an RX 9060 XT (gfx1200).
use raptor_runtime::manifest::Backend;
use raptor_runtime::verify::{parse_server_log, verify_gpu, GpuVerdict};

#[test]
fn real_gpu_log_is_verified_with_offload_count() {
    let log = include_str!("fixtures/real_b12340_hip_gpu_lv4.txt");
    let facts = parse_server_log(log);
    let verdict = verify_gpu(Backend::Hip, &facts);
    match verdict {
        GpuVerdict::Verified {
            offloaded_layers,
            total_layers,
            ..
        } => {
            assert_eq!(offloaded_layers, Some(66));
            assert_eq!(total_layers, Some(66));
        }
        other => panic!("expected Verified, got {other:?}"),
    }
}

#[test]
fn real_default_verbosity_log_without_device_is_never_verified() {
    let log = include_str!("fixtures/real_b12340_hip_nodevice_default.txt");
    let verdict = verify_gpu(Backend::Hip, &parse_server_log(log));
    assert!(
        !matches!(verdict, GpuVerdict::Verified { .. }),
        "CPU-only run must not be reported as GPU: {verdict:?}"
    );
}
