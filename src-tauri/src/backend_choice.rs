//! Which llama.cpp backend runs a model. Pure, so every rule is testable.
//!
//! Measured on the RX 9060 XT (llama-bench b12360, flash attention on, docs/EXPERIMENTS.md):
//! Vulkan generates faster on sparse Mixture-of-experts models that spill out of VRAM (Ornith
//! 44.2 -> 72.8 tok/s, flint 27.9 -> 38.4) and ties on a dense model that fits (Qwen 21.6 vs 21.1),
//! while ROCm reads a prompt 1.2 to 3 times faster everywhere. So `Auto` keeps ROCm, and moves a
//! spilling sparse-MoE model to Vulkan only when the Vulkan runtime is installed. The user can
//! force either backend per model. A forced backend that is not installed is an error that says
//! so: it never quietly runs on the other one.
//!
//! Multi-token-prediction (`--spec-type draft-mtp`) has not been verified on the Vulkan build, so
//! a Vulkan load uses n-gram speculation instead and says so.

use raptor_runtime::manifest::Backend;
use serde::{Deserialize, Serialize};

/// The backend a user asks for on one model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BackendPref {
    /// ROCm, except a spilling sparse-MoE model, which goes to Vulkan when it is installed.
    #[default]
    Auto,
    Rocm,
    Cuda,
    Sycl,
    Vulkan,
}

/// What the rule needs to know.
#[derive(Debug, Clone, Copy)]
pub struct Inputs {
    pub pref: BackendPref,
    /// Auto's Vulkan speed rule is only measured for AMD hardware today.
    pub amd_gpu: bool,
    /// A sparse Mixture-of-experts model larger than VRAM: its cold experts live in shared memory.
    pub spills: bool,
    pub rocm_installed: bool,
    pub cuda_installed: bool,
    pub sycl_installed: bool,
    pub vulkan_installed: bool,
}

/// The decision. `backend` is `None` when the normal runtime selection should be used (ROCm, or
/// whatever the machine's best runtime is). `notes` go into the load report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Choice {
    pub backend: Option<Backend>,
    pub notes: Vec<String>,
}

/// The measured numbers quoted next to the selector and in notes.
pub const MEASURED: &str = "Measured on the RX 9060 XT with llama.cpp b12360: Vulkan generates faster \
on large sparse Mixture-of-experts models that do not fit in VRAM (Ornith 44 -> 73 tok/s, flint 28 -> 38) \
and ties on a dense model that fits (Qwen 21.6 vs 21.1). ROCm reads a prompt 1.2 to 3 times faster on every \
model, which matters most for long prompts and coding agents. Plain decoding only: speculative decoding \
was not compared.";

/// Applies the preference. `Err` carries the message to show: the chosen backend is not installed.
pub fn resolve(i: &Inputs) -> std::result::Result<Choice, String> {
    match i.pref {
        BackendPref::Rocm => {
            if !i.rocm_installed {
                return Err(
                    "You chose ROCm for this model, but no ROCm runtime is installed. \
                    Install it from Settings > Dependencies, or choose Auto."
                        .into(),
                );
            }
            Ok(Choice {
                backend: Some(Backend::Hip),
                notes: vec!["Backend: ROCm, as chosen for this model.".into()],
            })
        }
        BackendPref::Cuda => {
            if !i.cuda_installed {
                return Err("You chose CUDA for this model, but no CUDA runtime is installed. Install it from Settings > Dependencies, or choose Auto.".into());
            }
            Ok(Choice {
                backend: Some(Backend::Cuda),
                notes: vec!["Backend: CUDA, as chosen for this model.".into()],
            })
        }
        BackendPref::Sycl => {
            if !i.sycl_installed {
                return Err("You chose Intel SYCL for this model, but no SYCL runtime is installed. Install it from Settings > Dependencies, or choose Auto.".into());
            }
            Ok(Choice {
                backend: Some(Backend::Sycl),
                notes: vec!["Backend: Intel SYCL, as chosen for this model.".into()],
            })
        }
        BackendPref::Vulkan => {
            if !i.vulkan_installed {
                return Err(
                    "You chose Vulkan for this model, but the Vulkan runtime is not \
                    installed. Install it from Settings > Dependencies, or choose Auto or a native backend for \
                    this model."
                        .into(),
                );
            }
            Ok(Choice {
                backend: Some(Backend::Vulkan),
                notes: vec!["Backend: Vulkan, as chosen for this model.".into()],
            })
        }
        BackendPref::Auto => {
            if i.amd_gpu && i.spills && i.vulkan_installed {
                Ok(Choice {
                    backend: Some(Backend::Vulkan),
                    notes: vec![
                        "Backend: Vulkan (Auto). This sparse Mixture-of-experts model is larger \
                         than VRAM, and Vulkan generated faster for that kind of model (Ornith 44 \
                         -> 73 tok/s, flint 28 -> 38) while ROCm reads prompts faster. Choose ROCm \
                         for this model if long prompts matter more."
                            .into(),
                    ],
                })
            } else if i.amd_gpu && i.spills {
                Ok(Choice {
                    backend: None,
                    notes: vec![
                        "Hint: this sparse Mixture-of-experts model is larger than VRAM. Generation \
                         was faster on the Vulkan runtime (Ornith 44 -> 73 tok/s, flint 28 -> 38 \
                         on this GPU). It is not installed; Settings > Dependencies installs it."
                            .into(),
                    ],
                })
            } else {
                Ok(Choice {
                    backend: None,
                    notes: Vec::new(),
                })
            }
        }
    }
}

/// True when MTP must be left out of the speed plan for `backend`: it is unverified on Vulkan.
pub fn mtp_unverified(backend: Backend) -> bool {
    backend == Backend::Vulkan
}

/// Added to the load report when a model with an MTP head runs on Vulkan.
pub const MTP_ON_VULKAN_NOTE: &str = "This model has a multi-token-prediction head, but MTP has not been \
verified on the Vulkan runtime, so n-gram speculation is used instead. On ROCm the head roughly doubles \
generation speed for models that have one.";

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs(pref: BackendPref, spills: bool, rocm: bool, vulkan: bool) -> Inputs {
        Inputs {
            pref,
            amd_gpu: true,
            spills,
            rocm_installed: rocm,
            cuda_installed: false,
            sycl_installed: false,
            vulkan_installed: vulkan,
        }
    }

    #[test]
    fn auto_keeps_the_normal_runtime_for_a_model_that_fits() {
        let c = resolve(&inputs(BackendPref::Auto, false, true, true)).unwrap();
        assert_eq!(c.backend, None);
        assert!(c.notes.is_empty());
    }

    #[test]
    fn auto_moves_a_spilling_sparse_model_to_vulkan_when_it_is_installed() {
        let c = resolve(&inputs(BackendPref::Auto, true, true, true)).unwrap();
        assert_eq!(c.backend, Some(Backend::Vulkan));
        assert!(c.notes[0].contains("73"), "{:?}", c.notes);
    }

    #[test]
    fn auto_without_vulkan_stays_on_the_normal_runtime_and_says_what_is_missing() {
        let c = resolve(&inputs(BackendPref::Auto, true, true, false)).unwrap();
        assert_eq!(c.backend, None);
        assert!(c.notes[0].contains("not installed"));
    }

    #[test]
    fn a_forced_backend_is_used_even_when_auto_would_choose_otherwise() {
        let v = resolve(&inputs(BackendPref::Vulkan, false, true, true)).unwrap();
        assert_eq!(v.backend, Some(Backend::Vulkan));
        let r = resolve(&inputs(BackendPref::Rocm, true, true, true)).unwrap();
        assert_eq!(r.backend, Some(Backend::Hip));
    }

    #[test]
    fn a_forced_backend_that_is_not_installed_is_an_error_not_a_fallback() {
        let e = resolve(&inputs(BackendPref::Vulkan, true, true, false)).unwrap_err();
        assert!(e.contains("Vulkan") && e.contains("not installed"), "{e}");
        let e = resolve(&inputs(BackendPref::Rocm, false, false, true)).unwrap_err();
        assert!(e.contains("ROCm") && e.contains("no ROCm runtime"), "{e}");
    }

    #[test]
    fn mtp_is_held_back_only_on_vulkan() {
        assert!(mtp_unverified(Backend::Vulkan));
        assert!(!mtp_unverified(Backend::Hip));
        assert!(!mtp_unverified(Backend::Cuda));
    }

    #[test]
    fn the_preference_round_trips_as_a_lowercase_word() {
        assert_eq!(
            serde_json::to_string(&BackendPref::Vulkan).unwrap(),
            "\"vulkan\""
        );
        assert_eq!(
            serde_json::from_str::<BackendPref>("\"auto\"").unwrap(),
            BackendPref::Auto
        );
        assert_eq!(BackendPref::default(), BackendPref::Auto);
    }
}
