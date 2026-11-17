//! The speed plan for a load: which faster variants to try first, and in what order to fall back.
//! Pure, so every rule is tested. Each rule comes from a measurement on the RX 9060 XT
//! (llama.cpp b12360); the numbers are in the notes the plan writes into the load rationale.
//!
//! 1. Speculative decoding (`SpecMode`): the model's own MTP head when its GGUF has one, else a
//!    matching DFlash draft model from the library, otherwise n-gram drafting. Qwen3.8-27B
//!    21.5 -> 47.4 tok/s; Nanbeige4.3-33B code edits 31.7-38.8 with DFlash; granite 20.4 -> 25.9.
//! 2. Sparse Mixture-of-experts models that do not fit in VRAM run faster with every layer
//!    requested on the GPU, letting the driver keep the cold experts in shared system memory,
//!    than with a layer split: flint (24 GB, 6 of 256 experts per token) 26.6 -> 35.0 tok/s,
//!    Ornith 47.8 -> 68.4 tok/s. Only 1/16 or fewer experts are read per token, so little
//!    crosses the PCIe bus. Dense models are not eligible: all their weights are read every token.
//!    The micro-batch is 256 then, because 512 halved Ornith's prompt speed (1,707 -> 862 tok/s).
//!
//! Every variant can fail to load (a missing MTP head, out of memory), so the plan is a list:
//! the first that loads wins and the notes say which one it was.

use raptor_runtime::config::{KvCacheType, LaunchConfig, SpecMode};

const GIB: u64 = 1024 * 1024 * 1024;

/// What the plan needs to know about the model and the machine.
#[derive(Debug, Clone)]
pub struct Facts {
    pub windows: bool,
    pub gpu_backend: bool,
    pub embeddings: bool,
    /// False when the user chose a window that has no room for the draft context (the maximum
    /// option turns MTP off to gain about twice the window, at about half the generation speed).
    pub allow_speculation: bool,
    /// Experts per layer and experts used per token, when the model is a Mixture of experts.
    pub experts: Option<(u64, u64)>,
    /// The GGUF declares NextN/MTP layers.
    pub has_mtp_head: bool,
    /// A DFlash draft model in the library that matches this model (same width and vocabulary).
    pub dflash_draft: Option<std::path::PathBuf>,
    /// The layer count comes from a tuning measured on this machine, so general layer rules
    /// (`dense_spill_applies`) do not override it.
    pub layers_measured: bool,
    /// The context window asked for, capped at the model's trained length (0 when unknown). The
    /// recommendation may have shrunk it to fit a layer split.
    pub requested_ctx: u32,
    /// Layers `-ngl` can place: the repeating layers plus the output layer.
    pub offload_layers: u32,
    pub model_bytes: u64,
    pub vram_total: u64,
    pub ram_total: u64,
    pub ram_avail: u64,
    /// Logical processors (hardware threads) of the CPU.
    pub cpu_threads: u32,
}

/// One launch to try.
#[derive(Debug, Clone, PartialEq)]
pub struct Attempt {
    pub config: LaunchConfig,
    /// Why this variant is being tried, for the load rationale.
    pub notes: Vec<String>,
}

/// The GGUF architecture name of a DFlash block-diffusion draft model.
pub const DFLASH_ARCH: &str = "dflash";

/// True when `draft` is a DFlash draft that can serve `target`: it reads the target's hidden
/// states and shares its embeddings and output head, so the width and the vocabulary must match.
/// (llama.cpp checks the token texts again when it loads the pair, and refuses a mismatch.)
pub fn dflash_matches(
    target: &raptor_models::gguf::GgufMeta,
    draft: &raptor_models::gguf::GgufMeta,
) -> bool {
    draft.architecture.as_deref() == Some(DFLASH_ARCH)
        && target.architecture.as_deref() != Some(DFLASH_ARCH)
        && target.embedding_length.is_some()
        && draft.embedding_length == target.embedding_length
        && target.vocab_size.is_some()
        && draft.vocab_size == target.vocab_size
}

/// A model that reads at most 1/16 of its experts per token.
pub fn is_sparse_moe(experts: Option<(u64, u64)>) -> bool {
    matches!(experts, Some((count, used)) if count > 1 && used > 0 && used * 16 <= count)
}

/// True when the whole model can sit in VRAM plus the driver's shared system memory (Windows lets
/// the GPU use up to half of system RAM) with RAM left over for the OS, and the config today
/// leaves layers on the CPU.
pub fn oversubscription_applies(base: &LaunchConfig, f: &Facts) -> bool {
    let shared = f.ram_total / 2;
    let spill = f.model_bytes.saturating_sub(f.vram_total);
    f.windows
        && f.gpu_backend
        && !f.embeddings
        && is_sparse_moe(f.experts)
        && base.gpu_layers < f.offload_layers
        && f.model_bytes <= f.vram_total + shared
        && f.ram_avail >= spill + 6 * GIB
}

/// How far over VRAM (percent of VRAM) a dense model may be and still be run with every layer on
/// the GPU, letting the driver spill the rest into shared memory.
pub const DENSE_SPILL_MAX_PERCENT: u64 = 10;

/// A dense model slightly larger than VRAM runs faster with every layer requested on the GPU (the
/// driver spills a little into shared memory) than with layers on the CPU. Measured on Windows,
/// Nanbeige4.3-33B all-IQ4_XS (16.4 GiB on a 16 GiB card), llama-bench tg128 at 53 / 51 / 50 / 49
/// / 48 layers: 11.23 / 11.33 / 11.16 / 10.71 / 9.78 tok/s, and the 45-layer split the plan made
/// before 8.9. Far over VRAM it is the other way: the 19.4 GiB Q4_K_M (22% over) ran 4.8 tok/s
/// with every layer vs 5.1-5.3 with 40-48, so the rule stops at `DENSE_SPILL_MAX_PERCENT`.
pub fn dense_spill_applies(base: &LaunchConfig, f: &Facts) -> bool {
    f.windows
        && f.gpu_backend
        && !f.embeddings
        && !is_sparse_moe(f.experts)
        && !f.layers_measured
        && base.gpu_layers < f.offload_layers
        && f.model_bytes >= f.vram_total - f.vram_total * DENSE_SPILL_MAX_PERCENT / 100
        && f.model_bytes <= f.vram_total + f.vram_total * DENSE_SPILL_MAX_PERCENT / 100
}

/// Windows this long or longer make a spilling sparse-MoE model prefer a 4-bit KV cache.
pub const LONG_WINDOW_CTX: u32 = 32_768;

/// The KV cache type a spilling sparse-MoE model should use at a window of `LONG_WINDOW_CTX` or
/// more: `q4_0`. Measured at a 32K context (llama-bench, 3 runs, ROCm b12374): Ornith generates
/// 46.9 tok/s with q4_0, 24.0 with f16 and 10.8 with q8_0; flint 16.9 / 15.5 / 12.3. The smaller
/// cache means a smaller spill into shared memory. `None` when this model/window is not that case
/// or already uses q4_0.
pub fn long_window_cache(base: &LaunchConfig, f: &Facts) -> Option<KvCacheType> {
    (base.ctx_size >= LONG_WINDOW_CTX
        && base.kv_cache != KvCacheType::Q4_0
        && oversubscription_applies(base, f))
    .then_some(KvCacheType::Q4_0)
}

/// With layers left on the CPU the CPU does real work every token, and generation scales with
/// logical processors, not just physical cores (granite with 25 layers on the CPU: 3.6 tok/s at
/// 8 threads, 4.1 at 16). With every layer on the GPU the thread count made no difference (Qwen,
/// flint), so it is left alone there.
fn with_threads(mut c: LaunchConfig, f: &Facts) -> LaunchConfig {
    if c.gpu_layers < f.offload_layers && f.cpu_threads > c.threads {
        c.threads = f.cpu_threads;
    }
    c
}

fn with_spec(mut c: LaunchConfig, mode: Option<SpecMode>, f: &Facts) -> LaunchConfig {
    c.speculative = mode;
    c.draft_model = match mode {
        Some(SpecMode::Dflash) => f.dflash_draft.clone(),
        _ => None,
    };
    c
}

/// Most request slots a DFlash draft is used with. Each busy slot drafts and verifies its own block,
/// so with many parallel requests the verify batches grow and the draft's memory adds spill. Measured
/// in the app (Nanbeige4.3-33B, 8 slots, a Long answer of 8 parallel sections): 11.0 tok/s with the
/// draft vs 30.7 without (shared-memory spill 4,310 vs 1,243 MiB). With one slot the draft won
/// everywhere (whole-file edit 32.7 vs 7.0 tok/s). Two slots were not measured.
pub const DFLASH_MAX_SLOTS: u32 = 2;

/// The ordered launches to try for `base` (the recommended or known-good configuration).
pub fn plan(base: &LaunchConfig, f: &Facts) -> Vec<Attempt> {
    let slots = base
        .parallel_slots
        .unwrap_or(raptor_runtime::fit::SERVER_DEFAULT_SLOTS);
    let draft_skipped = f.dflash_draft.is_some() && !f.has_mtp_head && slots > DFLASH_MAX_SLOTS;
    let primary = if f.embeddings || !f.allow_speculation {
        None
    } else if f.has_mtp_head {
        Some(SpecMode::Mtp)
    } else if f.dflash_draft.is_some() && !draft_skipped {
        Some(SpecMode::Dflash)
    } else {
        Some(SpecMode::Ngram)
    };
    let spec_note = |m: Option<SpecMode>| -> Option<String> {
        m.map(|m| match m {
            SpecMode::Mtp => {
                "Speculative decoding with the model's own multi-token-prediction head \
                              stacked with prompt n-grams (measured on Qwen3.8-27B: 21.5 -> 47.4 \
                              tok/s on one edit prompt; code rewrites about 200 tok/s)."
                    .to_string()
            }
            SpecMode::Ngram => "Speculative decoding from n-grams of the prompt and reply \
                                (measured: granite 20.4 -> 25.9 tok/s on a first reply)."
                .to_string(),
            SpecMode::Dflash => format!(
                "Speculative decoding with the DFlash draft model {} stacked with prompt n-grams \
                 (measured on Nanbeige4.3-33B: whole-file code edits 31.7-38.8 tok/s, new code \
                 16.7 vs 14.2 plain; prose about even).",
                f.dflash_draft
                    .as_deref()
                    .and_then(|p| p.file_name())
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default()
            ),
        })
    };

    let mut variants: Vec<(bool, Option<SpecMode>)> = Vec::new();
    // Measured layer and micro-batch choices take precedence over spill heuristics.
    let over =
        !f.layers_measured && (oversubscription_applies(base, f) || dense_spill_applies(base, f));
    let moe = is_sparse_moe(f.experts);
    let mut specs = vec![primary];
    if matches!(primary, Some(SpecMode::Mtp | SpecMode::Dflash)) {
        specs.push(Some(SpecMode::Ngram));
    }
    specs.push(None);
    specs.dedup();
    if over {
        for s in &specs[..specs.len().min(2)] {
            variants.push((true, *s));
        }
    }
    for s in &specs {
        variants.push((false, *s));
    }

    let mut out: Vec<Attempt> = Vec::new();
    for (oversub, spec) in variants {
        let mut config = with_spec(base.clone(), spec, f);
        let mut notes = Vec::new();
        if oversub && moe {
            config.gpu_layers = f.offload_layers;
            config.ubatch_size = config.ubatch_size.min(256);
            notes.push(format!(
                "Sparse Mixture-of-experts model larger than VRAM: all {} layers are requested on \
                 the GPU and the driver keeps the cold experts in shared system memory (measured \
                 faster than a layer split: flint 26.6 -> 35.0 tok/s, Ornith 47.8 -> 68.4 tok/s). \
                 Micro-batch 256.",
                f.offload_layers
            ));
        } else if oversub {
            config.gpu_layers = f.offload_layers;
            // The base window was shrunk to fit a layer split. With every layer on the GPU the
            // model spills a little anyway, so the window asked for comes back (a q4_0 cache for
            // a 52-layer 33B is about 60 KB per token: 4K more tokens is about 240 MB).
            if f.requested_ctx > config.ctx_size {
                notes.push(format!(
                    "Context window {} tokens as asked (the layer split had reduced it to {}).",
                    f.requested_ctx, config.ctx_size
                ));
                config.ctx_size = f.requested_ctx;
            }
            notes.push(format!(
                "This model is slightly larger than the VRAM budget: all {} layers are requested \
                 on the GPU and the driver keeps a little in shared system memory, which was \
                 faster than leaving layers on the CPU (measured on a 33B model 3% over VRAM: \
                 11.2 tok/s with every layer vs 8.9 with a 45-of-53 split).",
                f.offload_layers
            ));
        }
        notes.extend(spec_note(spec));
        if draft_skipped && spec.is_some() {
            notes.push(format!(
                "The DFlash draft is not used with {slots} request slots (it is used with \
                 {DFLASH_MAX_SLOTS} or fewer): with parallel requests every slot drafts its own \
                 block, measured 11.0 vs 30.7 tok/s for a Long answer at 8 slots."
            ));
        }
        let before = config.threads;
        config = with_threads(config, f);
        if config.threads != before {
            notes.push(format!(
                "Layers stay on the CPU, so it uses all {} logical processors (measured +13% generation speed over physical cores only).",
                config.threads
            ));
        }
        if !out.iter().any(|a| a.config == config) {
            out.push(Attempt { config, notes });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use raptor_runtime::manifest::Backend;

    fn base(ngl: u32) -> LaunchConfig {
        LaunchConfig::builder("C:/m.gguf", Backend::Hip)
            .gpu_layers(ngl)
            .ctx_size(8192)
            .threads(8)
            .build()
            .unwrap()
    }

    fn facts() -> Facts {
        Facts {
            windows: true,
            gpu_backend: true,
            embeddings: false,
            allow_speculation: true,
            experts: None,
            has_mtp_head: false,
            dflash_draft: None,
            layers_measured: false,
            requested_ctx: 0,
            offload_layers: 50,
            model_bytes: 24 * GIB,
            vram_total: 16 * GIB,
            ram_total: 64 * GIB,
            ram_avail: 40 * GIB,
            cpu_threads: 16,
        }
    }

    #[test]
    fn measured_sparse_moe_settings_are_not_overridden_by_spill_heuristics() {
        let f = Facts {
            experts: Some((256, 6)),
            layers_measured: true,
            ..facts()
        };
        let mut config = base(30);
        config.ubatch_size = 512;
        assert!(oversubscription_applies(&config, &f));
        for attempt in plan(&config, &f) {
            assert_eq!(attempt.config.gpu_layers, 30);
            assert_eq!(attempt.config.ubatch_size, 512);
        }
    }

    #[test]
    fn sparse_means_one_sixteenth_or_fewer_experts_per_token() {
        assert!(is_sparse_moe(Some((256, 6))));
        assert!(is_sparse_moe(Some((128, 8))));
        assert!(
            !is_sparse_moe(Some((8, 2))),
            "Mixtral-style models read too much per token"
        );
        assert!(!is_sparse_moe(None));
        assert!(!is_sparse_moe(Some((1, 1))));
    }

    #[test]
    fn a_dense_model_gets_speculation_and_never_oversubscription() {
        let plan = plan(&base(30), &facts());
        assert!(
            plan.iter().all(|a| a.config.gpu_layers == 30),
            "dense models keep the layer split"
        );
        assert_eq!(plan[0].config.speculative, Some(SpecMode::Ngram));
        assert_eq!(
            plan.last().unwrap().config.speculative,
            None,
            "the last resort has no speculation"
        );
    }

    #[test]
    fn a_model_with_an_mtp_head_tries_it_first_then_ngram_then_nothing() {
        let f = Facts {
            has_mtp_head: true,
            ..facts()
        };
        let specs: Vec<_> = plan(&base(50), &f)
            .iter()
            .map(|a| a.config.speculative)
            .collect();
        assert_eq!(specs, [Some(SpecMode::Mtp), Some(SpecMode::Ngram), None]);
    }

    #[test]
    fn a_dense_model_slightly_over_vram_tries_every_layer_first_and_keeps_the_split_as_fallback() {
        let f = Facts {
            model_bytes: 17 * GIB, // 6% over a 16 GiB card
            ..facts()
        };
        let attempts = plan(&base(45), &f);
        assert_eq!(attempts[0].config.gpu_layers, 50, "every layer first");
        assert_eq!(
            attempts[0].config.ubatch_size,
            base(45).ubatch_size,
            "the micro-batch rule is for sparse MoE models only"
        );
        assert!(attempts[0]
            .notes
            .iter()
            .any(|n| n.contains("slightly larger than the VRAM budget")));
        assert_eq!(attempts.last().unwrap().config.gpu_layers, 45);
        assert_eq!(
            attempts[0].config.ctx_size, 8192,
            "no requested window known: unchanged"
        );
        let asked = Facts {
            requested_ctx: 16_384,
            ..f.clone()
        };
        let a = plan(&base(45), &asked);
        assert_eq!(
            a[0].config.ctx_size, 16_384,
            "every layer gets the window asked for back"
        );
        assert!(a[0]
            .notes
            .iter()
            .any(|n| n.contains("16384 tokens as asked")));
        assert_eq!(
            a.last().unwrap().config.ctx_size,
            8192,
            "the split keeps the fitted window"
        );
        // Far over VRAM, or not on Windows, the split stays.
        let far = Facts {
            model_bytes: 20 * GIB,
            ..facts()
        };
        assert!(plan(&base(45), &far)
            .iter()
            .all(|a| a.config.gpu_layers == 45));
        let linux = Facts {
            windows: false,
            ..f.clone()
        };
        assert!(plan(&base(45), &linux)
            .iter()
            .all(|a| a.config.gpu_layers == 45));
        let tuned = Facts {
            layers_measured: true,
            ..f.clone()
        };
        assert!(
            plan(&base(45), &tuned)
                .iter()
                .all(|a| a.config.gpu_layers == 45),
            "a measured tuning wins"
        );
        let fits = Facts {
            model_bytes: 12 * GIB,
            ..f.clone()
        };
        assert!(
            plan(&base(45), &fits)
                .iter()
                .all(|a| a.config.gpu_layers == 45),
            "a model well under VRAM keeps what it was given"
        );
    }

    #[test]
    fn a_dflash_draft_matches_only_a_target_of_the_same_width_and_vocabulary() {
        use raptor_models::gguf::GgufMeta;
        let target = GgufMeta {
            architecture: Some("llama".into()),
            embedding_length: Some(7168),
            vocab_size: Some(166_912),
            ..Default::default()
        };
        let draft = GgufMeta {
            architecture: Some("dflash".into()),
            ..target.clone()
        };
        assert!(dflash_matches(&target, &draft));
        let narrower = GgufMeta {
            embedding_length: Some(3072),
            ..draft.clone()
        };
        assert!(!dflash_matches(&target, &narrower));
        let other_vocab = GgufMeta {
            vocab_size: Some(166_144),
            ..draft.clone()
        };
        assert!(!dflash_matches(&target, &other_vocab));
        let not_a_draft = GgufMeta {
            architecture: Some("llama".into()),
            ..draft.clone()
        };
        assert!(!dflash_matches(&target, &not_a_draft));
        assert!(!dflash_matches(&draft, &draft), "a draft is not a target");
        let unknown = GgufMeta {
            vocab_size: None,
            ..target.clone()
        };
        assert!(!dflash_matches(&unknown, &draft), "no vocabulary, no match");
    }

    #[test]
    fn a_model_with_a_dflash_draft_tries_it_first_then_ngram_then_nothing() {
        let f = Facts {
            dflash_draft: Some("C:/models/draft.gguf".into()),
            ..facts()
        };
        let mut one = base(50);
        one.parallel_slots = Some(1);
        let attempts = plan(&one, &f);
        let specs: Vec<_> = attempts.iter().map(|a| a.config.speculative).collect();
        assert_eq!(specs, [Some(SpecMode::Dflash), Some(SpecMode::Ngram), None]);
        assert_eq!(
            attempts[0].config.draft_model.as_deref(),
            Some(std::path::Path::new("C:/models/draft.gguf"))
        );
        assert!(attempts[0].notes.iter().any(|n| n.contains("draft.gguf")));
        assert!(
            attempts[1..].iter().all(|a| a.config.draft_model.is_none()),
            "the fallbacks do not load the draft"
        );
        // Many request slots: n-grams instead, and the notes say why.
        let mut wide = base(50);
        wide.parallel_slots = Some(8);
        let attempts8 = plan(&wide, &f);
        assert_eq!(attempts8[0].config.speculative, Some(SpecMode::Ngram));
        assert!(attempts8.iter().all(|a| a.config.draft_model.is_none()));
        assert!(attempts8[0]
            .notes
            .iter()
            .any(|n| n.contains("8 request slots")));
        let mut two = base(50);
        two.parallel_slots = Some(2);
        assert_eq!(plan(&two, &f)[0].config.speculative, Some(SpecMode::Dflash));
        // A model's own MTP head still wins over a separate draft.
        let both = Facts {
            has_mtp_head: true,
            ..f
        };
        assert_eq!(
            plan(&base(50), &both)[0].config.speculative,
            Some(SpecMode::Mtp)
        );
    }

    #[test]
    fn a_sparse_moe_that_spills_tries_all_layers_on_the_gpu_first_and_falls_back_to_the_split() {
        let f = Facts {
            experts: Some((256, 6)),
            ..facts()
        };
        let attempts = plan(&base(30), &f);
        assert_eq!(
            attempts[0].config.gpu_layers, 50,
            "all layers requested first"
        );
        assert_eq!(attempts[0].config.ubatch_size, 256);
        assert!(attempts[0]
            .notes
            .iter()
            .any(|n| n.contains("shared system memory")));
        let last = attempts.last().unwrap();
        assert_eq!(
            last.config.gpu_layers, 30,
            "the conservative split is the last resort"
        );
        assert_eq!(last.config.speculative, None);
        assert!(attempts
            .iter()
            .any(|a| a.config.gpu_layers == 30 && a.config.speculative.is_some()));
    }

    fn moe_facts() -> Facts {
        Facts {
            experts: Some((256, 6)),
            ..facts()
        }
    }

    fn at(ctx: u32, kv: KvCacheType, ngl: u32) -> LaunchConfig {
        let mut c = base(ngl);
        c.ctx_size = ctx;
        c.kv_cache = kv;
        c
    }

    #[test]
    fn a_spilling_moe_at_a_long_window_gets_the_4_bit_cache() {
        let f = moe_facts();
        for kv in [KvCacheType::F16, KvCacheType::Q8_0] {
            assert_eq!(
                long_window_cache(&at(32_768, kv, 30), &f),
                Some(KvCacheType::Q4_0),
                "{kv:?}"
            );
        }
        assert_eq!(
            long_window_cache(&at(131_072, KvCacheType::Q8_0, 30), &f),
            Some(KvCacheType::Q4_0)
        );
    }

    #[test]
    fn the_cache_is_left_alone_for_short_windows_dense_models_and_models_that_fit() {
        let f = moe_facts();
        assert_eq!(
            long_window_cache(&at(8_192, KvCacheType::F16, 30), &f),
            None
        );
        assert_eq!(
            long_window_cache(&at(32_767, KvCacheType::F16, 30), &f),
            None
        );
        assert_eq!(
            long_window_cache(&at(32_768, KvCacheType::Q4_0, 30), &f),
            None,
            "already 4-bit"
        );
        // Dense: the rule is for sparse Mixture-of-experts only.
        assert_eq!(
            long_window_cache(&at(32_768, KvCacheType::F16, 30), &facts()),
            None
        );
        // Fully offloaded: nothing spills, so the usual cache rules apply.
        assert_eq!(
            long_window_cache(&at(32_768, KvCacheType::F16, 50), &f),
            None
        );
    }

    #[test]
    fn oversubscription_needs_windows_a_gpu_room_and_a_partial_offload() {
        let f = Facts {
            experts: Some((256, 6)),
            ..facts()
        };
        assert!(oversubscription_applies(&base(30), &f));
        assert!(!oversubscription_applies(
            &base(30),
            &Facts {
                windows: false,
                ..f.clone()
            }
        ));
        assert!(!oversubscription_applies(
            &base(30),
            &Facts {
                gpu_backend: false,
                ..f.clone()
            }
        ));
        assert!(
            !oversubscription_applies(&base(50), &f),
            "already fully offloaded"
        );
        // 40 GiB of weights cannot fit 16 GiB VRAM plus half of 64 GiB of RAM.
        assert!(!oversubscription_applies(
            &base(30),
            &Facts {
                model_bytes: 60 * GIB,
                ..f.clone()
            }
        ));
        // Not enough free RAM to hold the spill and leave room for the system.
        assert!(!oversubscription_applies(
            &base(30),
            &Facts {
                ram_avail: 10 * GIB,
                ..f.clone()
            }
        ));
        assert!(!oversubscription_applies(
            &base(30),
            &Facts {
                embeddings: true,
                ..f
            }
        ));
    }

    #[test]
    fn threads_use_every_logical_processor_only_when_layers_stay_on_the_cpu() {
        let f = facts();
        let split = plan(&base(30), &f);
        assert!(
            split.iter().all(|a| a.config.threads == 16),
            "{:?}",
            split.iter().map(|a| a.config.threads).collect::<Vec<_>>()
        );
        assert!(split[0]
            .notes
            .iter()
            .any(|n| n.contains("logical processors")));
        let full = plan(&base(50), &f);
        assert!(
            full.iter().all(|a| a.config.threads == base(50).threads),
            "no change when everything is on the GPU"
        );
    }

    #[test]
    fn speculation_can_be_switched_off_for_a_window_that_needs_the_room() {
        let f = Facts {
            has_mtp_head: true,
            allow_speculation: false,
            ..facts()
        };
        let attempts = plan(&base(50), &f);
        assert!(attempts.iter().all(|a| a.config.speculative.is_none()));
    }

    #[test]
    fn the_embeddings_mode_never_gets_speculation() {
        let f = Facts {
            embeddings: true,
            has_mtp_head: true,
            ..facts()
        };
        let attempts = plan(&base(50), &f);
        assert_eq!(attempts.len(), 1);
        assert_eq!(attempts[0].config.speculative, None);
    }

    #[test]
    fn identical_attempts_are_listed_once() {
        let f = facts();
        let attempts = plan(&base(50), &f);
        for (i, a) in attempts.iter().enumerate() {
            assert!(!attempts[..i].iter().any(|b| b.config == a.config));
        }
    }
}
