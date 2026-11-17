# Windows GPU compatibility and validation

The runtime manifest, hardware match, `llama-server --list-devices` preflight, verified layer offload, and a real chat are separate gates. A family is not hardware-validated until all four stages pass on that hardware. The preflight and load verdicts run in the app; the ignored real-hardware gate also sends a chat completion and records measured throughput.

| GPU family | Preferred runtime | Runtime package in manifest | Device and model gates | Hardware validation |
|---|---|---:|---|---|
| AMD Radeon RX 6000 series and newer with a listed `gfx` target | HIP/ROCm | Yes | HIP SDK dependency check, backend device preflight, load-time offload verdict | RX 9060 XT has existing real-app evidence in `ROADMAP_STATUS.md`; other RX families not run here |
| NVIDIA RTX 40 series | CUDA 13.5 | Yes | CUDA device preflight, load-time offload verdict | Not run |
| NVIDIA RTX 50 series / Blackwell | CUDA 13.5 | Yes | CUDA device preflight, load-time offload verdict | Not run |
| Intel Arc A770 | SYCL | Yes | SYCL device preflight, load-time offload verdict | Not run |
| Intel Arc B580 | SYCL | Yes | SYCL device preflight, load-time offload verdict | Not run |
| Intel Arc Pro B70 and other Arc Pro B-series | SYCL | Yes | SYCL device preflight, load-time offload verdict | Not run; require a physical-card gate before claiming validation |
| Any Vulkan-capable adapter | Vulkan | Yes | Vulkan device preflight, load-time offload verdict | Per-model and per-driver validation pending |

“Runtime package in manifest” means Raptor can select and install the upstream Windows x64 build after its SHA-256 check. It does not mean every board, driver, or model has passed the real-hardware gate. On Auto, Raptor prefers the native backend for the detected GPU with the most VRAM; the measured AMD sparse-MoE Vulkan rule remains limited to AMD systems. Fit and offload counts are then estimated for the selected model, and the server must confirm actual GPU use.

## Real-hardware gate

On a Windows system with the target GPU and an extracted matching runtime, set `RAPTOR_TEST_RUNTIME_DIR`, `RAPTOR_TEST_MODEL`, and `RAPTOR_TEST_BACKEND` (`hip`, `cuda`, `sycl`, or `vulkan`), then run:

```powershell
cargo test -p raptor-runtime --test gate_real_hardware -- --ignored --nocapture gate_selected_backend_verified
```

The gate requires a visible backend device, starts the model, checks the offload verdict, sends a short chat request, records response speed, and verifies shutdown. Run it once for each GPU model, runtime version, and driver combination promoted to validated support. The AMD HIP-specific gates remain available for the existing ROCm SDK diagnostics.
