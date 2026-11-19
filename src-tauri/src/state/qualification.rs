use super::*;
use crate::settings::MIN_CONTEXT_TARGET;
use crate::tuned::Qualification;
use raptor_runtime::config::KvCacheType;
use serde_json::json;
use std::sync::atomic::Ordering::Relaxed;
use std::time::{Duration, Instant};

const OUTPUT: u32 = 64;

#[derive(Debug, Clone, Serialize)]
pub struct Trial {
    pub gpu_layers: u32,
    pub ubatch: u32,
    pub kv_cache: KvCacheType,
    pub seconds: f64,
}

impl Trial {
    fn new(config: &LaunchConfig, seconds: f64) -> Self {
        Self {
            gpu_layers: config.gpu_layers,
            ubatch: config.ubatch_size,
            kv_cache: config.kv_cache,
            seconds,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct TuneReport {
    pub context_size: u32,
    pub output_tokens: u32,
    pub results: Vec<Trial>,
    pub errors: Vec<String>,
    pub confirmation: Vec<(f64, f64)>,
    pub finalist: Option<usize>,
    pub saved: bool,
    pub retained_profile: bool,
}

/// Hash the actual runtime arguments to bind proof to the tested configuration.
pub(super) fn signature(config: &LaunchConfig) -> String {
    raptor_runtime::hash::sha256_hex(config.args().join("\0").as_bytes())
}

fn candidates(
    base: &LaunchConfig,
    facts: &crate::tuning::Facts,
    recommended: u32,
    full: u32,
    extended: bool,
) -> Vec<LaunchConfig> {
    let mut configs = vec![base.clone()];
    let mut add = |layers, ubatch: u32, cache| {
        let mut c = base.clone();
        c.gpu_layers = layers;
        c.ubatch_size = ubatch.min(c.batch_size);
        c.kv_cache = cache;
        if cache != KvCacheType::F16 {
            c.flash_attn = true;
        }
        if !configs.contains(&c) {
            configs.push(c);
        }
    };
    // The big full-window lever first: the plan's shared-memory spill variants (every layer on
    // the GPU; sparse MoE and near-VRAM dense models), then a shorter KV cache doing the same.
    if base.backend.is_gpu() {
        for attempt in crate::tuning::plan(base, facts) {
            if attempt.config.gpu_layers > base.gpu_layers {
                add(
                    attempt.config.gpu_layers,
                    attempt.config.ubatch_size,
                    attempt.config.kv_cache,
                );
            }
        }
        add(full, base.ubatch_size, KvCacheType::Q8_0);
        // Dense models: a 4-bit KV cache cost ~40% of prompt speed at the same layer count on
        // HIP (Nanbeige4.3-33B Q4_K_M, ROCm b12374), because flash attention dequantizes every
        // K/V read. It only pays where the cache itself is the spill — sparse MoE models.
        if crate::tuning::is_sparse_moe(facts.experts) {
            add(full, 256, KvCacheType::Q4_0);
        }
    }
    add(base.gpu_layers, 256, base.kv_cache);
    if crate::tuning::is_sparse_moe(facts.experts) {
        add(base.gpu_layers, base.ubatch_size, KvCacheType::Q4_0);
    }
    if extended {
        let mid = recommended.midpoint(full);
        for layers in [recommended, mid, base.gpu_layers, full] {
            for ub in [128, 256, 512, 1024] {
                for kv in [KvCacheType::Q4_0, KvCacheType::Q8_0] {
                    add(layers, ub, kv);
                }
            }
        }
    }
    configs
}

fn check_usage(reply: &serde_json::Value, window: u32) -> Result<()> {
    let prompt = u64::from(window - OUTPUT);
    let usage = &reply["usage"];
    if usage["prompt_tokens"].as_u64() != Some(prompt)
        || usage["completion_tokens"].as_u64() != Some(u64::from(OUTPUT))
        || reply["timings"]["prompt_n"].as_u64() != Some(prompt)
        || usage["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .unwrap_or(0)
            != 0
    {
        return Err(RaptorError::Runtime(format!("Server did not confirm {prompt} freshly evaluated tokens and {OUTPUT} output tokens in one request (usage {usage}, prompt_n {}).", reply["timings"]["prompt_n"])));
    }
    Ok(())
}

/// The stock configuration the candidate must beat: the recommended layer split and window,
/// one request slot, a 16-bit cache and no speculative decoding — what a plain llama.cpp
/// launcher does. The required 50% proof is the Raptor edge over this.
pub(super) fn stock_config(mut base: LaunchConfig, ctx: u32) -> LaunchConfig {
    base.ctx_size = ctx;
    base.parallel_slots = Some(1);
    base.allow_cpu_fallback = false;
    base.kv_cache = KvCacheType::F16;
    base.speculative = None;
    base.draft_model = None;
    base
}

impl AppState {
    async fn trial(&self, config: &LaunchConfig, seed: u32, vocab: u32) -> Result<f64> {
        let result = self.time_full_window(config, seed, vocab).await;
        self.tune_steps.0.fetch_add(1, Relaxed);
        result
    }

    async fn time_full_window(&self, config: &LaunchConfig, seed: u32, vocab: u32) -> Result<f64> {
        self.runtime.load_model(config.clone()).await?;
        let measured = async {
            if config.backend.is_gpu()
                && !matches!(
                    self.runtime.status().gpu_verdict,
                    Some(raptor_runtime::verify::GpuVerdict::Verified { .. })
                )
            {
                return Err(RaptorError::Runtime("GPU offload was not verified".into()));
            }
            // Token IDs avoid template-dependent lengths. Different seeds defeat prefix reuse.
            let mut state = seed;
            let prompt: Vec<u32> = (0..config.ctx_size - OUTPUT)
                .map(|_| {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    32 + state % (vocab - 32)
                })
                .collect();
            let body = json!({"prompt": prompt, "max_tokens": OUTPUT, "ignore_eos": true,
                "cache_prompt": false, "temperature": 0, "stream": false, "seed": 42});
            let started = Instant::now();
            let response = tokio::time::timeout(
                Duration::from_secs(1800),
                self.upstream
                    .call(raptor_gateway::Endpoint::Completions, body),
            )
            .await
            .map_err(|_| {
                RaptorError::Runtime("Full-window request timed out after 30 minutes".into())
            })??;
            let elapsed = started.elapsed().as_secs_f64();
            let raptor_gateway::Completion::Json(reply) = response else {
                return Err(RaptorError::Runtime(
                    "Expected a completion with token usage".into(),
                ));
            };
            check_usage(&reply, config.ctx_size)?;
            Ok(elapsed)
        }
        .await;
        self.runtime.unload().await?;
        measured
    }

    pub async fn tune_model(&self, model_id: &str, deep: bool) -> Result<TuneReport> {
        self.tune_model_at(model_id, deep, None, None).await
    }

    pub async fn tune_model_at(
        &self,
        model_id: &str,
        deep: bool,
        context: Option<u32>,
        preset: Option<&str>,
    ) -> Result<TuneReport> {
        self.restore_installed_runtime().await?;
        let _guard = self
            .runtime_lock
            .try_lock()
            .map_err(|_| RaptorError::Conflict("Another model task is running".into()))?;
        if self.runtime.status().server.state != ServerState::Stopped {
            return Err(RaptorError::Conflict(
                "unload the model before tuning".into(),
            ));
        }
        let library = self.models.library();
        let entry = library.get(model_id)?;
        let path = library.file_path(model_id)?;
        let target = match (context, preset) {
            (Some(ctx), _) => ctx,
            (_, Some(name)) => self.models.presets().get(model_id, name)?.n_ctx,
            _ => self.settings.snapshot().context_target,
        };
        if target > crate::settings::MAX_CONTEXT_TARGET {
            return Err(RaptorError::InvalidInput(format!(
                "Context exceeds the {}-token ceiling",
                crate::settings::MAX_CONTEXT_TARGET
            )));
        }
        let ctx = target.max(MIN_CONTEXT_TARGET);
        if entry
            .gguf
            .context_length
            .is_some_and(|n| n < u64::from(ctx))
        {
            return Err(RaptorError::InvalidInput(format!(
                "This model's trained window is below the required {ctx} tokens."
            )));
        }
        let vocab = entry
            .gguf
            .vocab_size
            .and_then(|n| u32::try_from(n).ok())
            .filter(|&n| n > 32)
            .ok_or_else(|| {
                RaptorError::InvalidInput(
                    "Full-window tuning requires the GGUF vocabulary size.".into(),
                )
            })?;
        let hw = self.hardware_for_use().await?;
        self.apply_backend_choice(model_id, &entry, &hw).await?;
        let runtime = self
            .runtime
            .status()
            .runtime
            .ok_or_else(|| RaptorError::Conflict("Install a runtime before tuning".into()))?;
        let draft = self.planned_draft_bytes(&entry, runtime.backend, 1, true);
        let base = self
            .recommend_for_entry_with(&entry, &path, ctx, draft)
            .await?
            .config
            .with_model(path)?;
        let recommended = base.gpu_layers;
        let baseline = qualification::stock_config(base, ctx);
        let facts = crate::tuning::Facts {
            requested_ctx: ctx,
            ..self.speed_facts(&entry, &baseline, &hw, baseline.backend)
        };
        let configs = candidates(
            &baseline,
            &facts,
            recommended,
            offload_layers_of(&entry).unwrap_or(recommended),
            deep,
        );
        self.tune_steps.0.store(0, Relaxed);
        self.tune_steps.1.store(configs.len() as u32, Relaxed);
        let base_time = self.trial(&baseline, 1, vocab).await.map_err(|e| {
            RaptorError::Runtime(format!("Baseline failed full-window qualification: {e}"))
        })?;
        let mut report = TuneReport {
            context_size: ctx,
            output_tokens: OUTPUT,
            results: vec![Trial::new(&baseline, base_time)],
            errors: Vec::new(),
            confirmation: Vec::new(),
            finalist: None,
            saved: false,
            retained_profile: false,
        };
        // Screen candidates in the order above; the first one that already reaches the required
        // boost stops the quick sweep. An explicit extended search screens the full grid.
        let mut ranked = Vec::new();
        for c in &configs[1..] {
            match self.trial(c, 1, vocab).await {
                Ok(t) => {
                    if t < base_time {
                        ranked.push((report.results.len(), c, t));
                    }
                    report.results.push(Trial::new(c, t));
                    if !deep && t * 1.5 <= base_time {
                        break;
                    }
                }
                Err(e) => report.errors.push(format!(
                    "{} layers, micro-batch {}, {} cache: {e}",
                    c.gpu_layers,
                    c.ubatch_size,
                    c.kv_cache.llama_arg()
                )),
            }
        }
        // The three pairs are the gate (>= 1.50x in every one), so only candidates whose screen
        // already came close enough are measured again.
        ranked.retain(|&(_, _, t)| t <= base_time / 1.45);
        ranked.sort_by(|a, b| a.2.total_cmp(&b.2));
        let key = self.tuned_key(&hw, &entry.sha256, &runtime);
        for (rank, (row, best, _seconds)) in ranked.into_iter().enumerate() {
            self.tune_steps
                .1
                .store(self.tune_steps.0.load(Relaxed) + 6, Relaxed);
            let mut pairs = Vec::with_capacity(3);
            for seed in 2..=4 {
                let candidate_first = seed % 2 == 0;
                let (first, second) = if candidate_first {
                    (best, &baseline)
                } else {
                    (&baseline, best)
                };
                let pair = async {
                    let a = self.trial(first, seed, vocab).await?;
                    let b = self.trial(second, seed, vocab).await?;
                    Ok::<_, RaptorError>(if candidate_first { (b, a) } else { (a, b) })
                }
                .await;
                match pair {
                    Ok(pair) => pairs.push(pair),
                    Err(e) => {
                        report.errors.push(format!(
                            "Confirmation for candidate {row}, pair {}: {e}",
                            seed - 1
                        ));
                        break;
                    }
                }
            }
            let qualification = Qualification {
                context_size: ctx,
                signature: signature(best),
                pairs: pairs.clone(),
            };
            // Keep the fastest candidate's evidence unless another actually qualifies.
            if rank == 0 || qualification.passed() {
                report.confirmation = pairs;
                report.finalist = Some(row);
            }
            if !qualification.passed() {
                continue;
            }
            let seconds = median(report.confirmation.iter().map(|p| p.1).collect());
            let baseline_seconds = median(report.confirmation.iter().map(|p| p.0).collect());
            lock(&self.tuned).put(
                &key,
                crate::tuned::TunedEntry {
                    gpu_layers: best.gpu_layers,
                    ubatch_size: best.ubatch_size,
                    seconds,
                    baseline_seconds,
                    measured_at: crate::fsutil::unix_now(),
                    prompt_tps: 0.0,
                    gen_tps: 0.0,
                    long: None,
                    profile: Some(crate::tuned::TuneProfile {
                        kv_cache: best.kv_cache,
                        flash_attn: best.flash_attn,
                    }),
                    qualification: Some(qualification),
                    baseline_live: None,
                    best_live: None,
                },
            )?;
            report.saved = true;
            break;
        }
        if !report.saved {
            report.retained_profile = lock(&self.tuned)
                .get(&key)
                .and_then(|t| t.qualification)
                .is_some_and(|q| q.context_size == ctx && q.passed());
        }
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu_base() -> LaunchConfig {
        LaunchConfig::builder("m.gguf", raptor_runtime::manifest::Backend::Hip)
            .gpu_layers(30)
            .ctx_size(65_536)
            .batch_size(2048)
            .build()
            .unwrap()
    }

    fn facts() -> crate::tuning::Facts {
        crate::tuning::Facts {
            windows: true,
            gpu_backend: true,
            embeddings: false,
            allow_speculation: true,
            experts: None,
            has_mtp_head: false,
            dflash_draft: None,
            layers_measured: false,
            requested_ctx: 65_536,
            offload_layers: 40,
            model_bytes: 17 * 1024 * 1024 * 1024,
            vram_total: 16 * 1024 * 1024 * 1024,
            ram_total: 64 * 1024 * 1024 * 1024,
            ram_avail: 40 * 1024 * 1024 * 1024,
            cpu_threads: 16,
        }
    }

    #[test]
    fn candidates_start_at_the_stock_split_and_try_spill_then_cache() {
        let base = stock_config(gpu_base(), 65_536);
        let list = candidates(&base, &facts(), 30, 40, false);
        assert_eq!(list[0], base);
        assert_eq!(list[1].gpu_layers, 40);
        assert_eq!(list[1].speculative, None);
        assert!(list
            .iter()
            .any(|c| c.kv_cache == KvCacheType::Q8_0 && c.gpu_layers == 40));
        // Dense model: no q4_0 cache candidates — measured ~40% slower prompts on HIP.
        assert!(list.iter().all(|c| c.kv_cache != KvCacheType::Q4_0));
        assert!(list
            .iter()
            .all(|c| c.parallel_slots == Some(1) && c.ctx_size == 65_536));
    }

    #[test]
    fn sparse_moe_candidates_keep_the_four_bit_cache_lever() {
        let base = LaunchConfig::builder("m.gguf", raptor_runtime::manifest::Backend::Hip)
            .gpu_layers(30)
            .ctx_size(65_536)
            .batch_size(2048)
            .build()
            .unwrap();
        let facts = crate::tuning::Facts {
            experts: Some((256, 6)),
            ..facts()
        };
        let list = candidates(&base, &facts, 30, 40, false);
        assert!(list
            .iter()
            .any(|c| c.kv_cache == KvCacheType::Q4_0 && c.gpu_layers == 40));
        assert!(list
            .iter()
            .any(|c| c.kv_cache == KvCacheType::Q4_0 && c.gpu_layers == 30));
    }

    #[test]
    fn the_baseline_is_a_plain_single_slot_sixteen_bit_launch() {
        let stock = stock_config(
            LaunchConfig::builder("m.gguf", raptor_runtime::manifest::Backend::Hip)
                .gpu_layers(30)
                .kv_cache(KvCacheType::Q8_0)
                .speculative(raptor_runtime::config::SpecMode::Ngram)
                .build()
                .unwrap(),
            65_536,
        );
        assert_eq!(
            (
                stock.kv_cache,
                stock.speculative,
                stock.parallel_slots,
                stock.allow_cpu_fallback,
                stock.ctx_size
            ),
            (KvCacheType::F16, None, Some(1), false, 65_536)
        );
    }

    #[test]
    fn full_window_requires_uncached_prompt_and_complete_output() {
        let mut v = json!({"usage":{"prompt_tokens":65472,"completion_tokens":64},"timings":{"prompt_n":65472}});
        assert!(check_usage(&v, 65536).is_ok());
        v["timings"]["prompt_n"] = json!(373);
        assert!(check_usage(&v, 65536).is_err());
        v["timings"]["prompt_n"] = json!(65472);
        v["usage"]["completion_tokens"] = json!(63);
        assert!(check_usage(&v, 65536).is_err());
    }
}
