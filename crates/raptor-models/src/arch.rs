//! Memory-relevant architecture facts derived from a GGUF header.
//!
//! The formulas follow llama.cpp's memory layout for the architectures that need them:
//! attention layers hold a KV cache, recurrent or SSM layers hold a fixed-size state that does
//! not grow with context. Each formula below was checked against a real header (Qwen3.5-family
//! `qwen35`, 64 main layers plus one NextN layer) against llama.cpp's own log.

use crate::gguf::GgufMeta;

/// Bytes per element of the f32 recurrent state (llama.cpp keeps SSM and conv state in f32).
const STATE_ELEM_BYTES: u64 = 4;
/// Bytes per element of the f16 KV cache.
const KV_ELEM_BYTES_F16: u64 = 2;

/// Plain numbers the runtime needs for its memory estimate, derived from one header.
///
/// A zero field means the header did not carry the input (for example no `block_count`).
/// The shell should then fall back to its dense estimate rather than trust zeros.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArchHints {
    /// Layers that hold a KV cache.
    pub kv_layers: u32,
    /// Recurrent and SSM state bytes for one slot, independent of context.
    pub recurrent_bytes_per_slot: u64,
    /// Layers `-ngl` offloads: all repeating layers (NextN included, as llama.cpp counts them)
    /// plus the output layer.
    pub offload_layers: u32,
}

/// Derives the hints the runtime maps into its architecture model. Never panics.
pub fn runtime_hints(meta: &GgufMeta) -> ArchHints {
    ArchHints {
        kv_layers: to_u32(meta.kv_layer_count().unwrap_or(0)),
        recurrent_bytes_per_slot: meta.recurrent_state_bytes(1).unwrap_or(0),
        offload_layers: to_u32(
            meta.block_count
                .filter(|b| *b > 0)
                .and_then(|b| b.checked_add(1))
                .unwrap_or(0),
        ),
    }
}

fn to_u32(v: u64) -> u32 {
    u32::try_from(v).unwrap_or(u32::MAX)
}

impl GgufMeta {
    /// True for a Mixture-of-experts model (`expert_count` above 1).
    pub fn is_moe(&self) -> bool {
        self.expert_count.is_some_and(|n| n > 1)
    }

    /// True when the header carries SSM dimensions, i.e. some layers hold recurrent state.
    pub fn has_recurrent_state(&self) -> bool {
        self.ssm_inner_size.is_some() || self.ssm_state_size.is_some()
    }

    /// Repeating layers of the main stack: `block_count` minus the NextN layers.
    /// `None` when `block_count` is absent or zero.
    pub fn main_layer_count(&self) -> Option<u64> {
        let blocks = self.block_count.filter(|b| *b > 0)?;
        let main = blocks.saturating_sub(self.nextn_predict_layers.unwrap_or(0));
        (main > 0).then_some(main)
    }

    /// Number of main-stack layers that hold a KV cache.
    ///
    /// Order of evidence:
    /// 1. A per-layer `head_count_kv` array: count the non-zero entries of the main stack.
    /// 2. `full_attention_interval = k`: every k-th layer is attention, so `main / k`.
    /// 3. SSM dimensions without an interval: no attention layers, so 0.
    /// 4. Otherwise a dense model: every main layer holds KV.
    ///
    /// NextN layers are excluded. This matches the Qwen3.5 header, where llama.cpp reports
    /// 16 attention layers out of 64 main layers and a 65th NextN layer.
    pub fn kv_layer_count(&self) -> Option<u64> {
        let main = self.main_layer_count()?;
        if let Some(per_layer) = &self.head_count_kv_layers {
            let take = usize::try_from(main).unwrap_or(usize::MAX);
            let count = per_layer.iter().take(take).filter(|h| **h > 0).count();
            return u64::try_from(count).ok();
        }
        if let Some(k) = self.full_attention_interval.filter(|k| *k > 0) {
            return Some(main / k);
        }
        if self.has_recurrent_state() {
            return Some(0);
        }
        Some(main)
    }

    /// Number of main-stack layers that hold recurrent or SSM state (main layers minus KV layers).
    /// Zero for models without SSM dimensions.
    pub fn recurrent_layer_count(&self) -> Option<u64> {
        if !self.has_recurrent_state() {
            return Some(0);
        }
        let main = self.main_layer_count()?;
        main.checked_sub(self.kv_layer_count()?)
    }

    /// Recurrent-state bytes for `n_slots` parallel sequences, independent of context length.
    ///
    /// Per recurrent layer and slot, with `conv_dim = inner + 2 * groups * state` and f32 elements:
    /// - conv state: `(conv_kernel - 1) * conv_dim * 4`
    /// - SSM state: `state_size * inner * 4`
    ///
    /// The total is `n_slots * recurrent_layers * (conv + ssm)`. For Qwen3.5-family models this
    /// gives 22.5 MiB of conv state and 576 MiB of SSM state for 4 slots, matching llama.cpp.
    /// `ssm.time_step_rank` is not needed: for these models the SSM state is `state * inner`.
    /// Returns `None` on missing SSM keys or arithmetic overflow.
    pub fn recurrent_state_bytes(&self, n_slots: u64) -> Option<u64> {
        let layers = self.recurrent_layer_count()?;
        if layers == 0 || n_slots == 0 {
            return Some(0);
        }
        let conv_kernel = self.ssm_conv_kernel?;
        let state = self.ssm_state_size?;
        let inner = self.ssm_inner_size?;
        let groups = self.ssm_group_count.unwrap_or(0);
        let conv_dim = groups
            .checked_mul(state)?
            .checked_mul(2)?
            .checked_add(inner)?;
        let conv = conv_kernel
            .saturating_sub(1)
            .checked_mul(conv_dim)?
            .checked_mul(STATE_ELEM_BYTES)?;
        let ssm = state.checked_mul(inner)?.checked_mul(STATE_ELEM_BYTES)?;
        conv.checked_add(ssm)?
            .checked_mul(layers)?
            .checked_mul(n_slots)
    }

    /// KV cache bytes per context token with an f16 cache, over the KV layers only.
    ///
    /// Per KV layer: `kv_heads * (key_dim + value_dim) * 2`. Unlike [`GgufMeta::kv_bytes_per_token`],
    /// this excludes recurrent layers and uses per-layer head counts when the header has them.
    pub fn kv_cache_bytes_per_token_f16(&self) -> Option<u64> {
        let key_dim = self.head_dim()?;
        let value_dim = self.value_length.filter(|v| *v > 0).unwrap_or(key_dim);
        let per_head = key_dim
            .checked_add(value_dim)?
            .checked_mul(KV_ELEM_BYTES_F16)?;
        if let Some(per_layer) = &self.head_count_kv_layers {
            let take = usize::try_from(self.main_layer_count()?).unwrap_or(usize::MAX);
            return per_layer.iter().take(take).try_fold(0u64, |acc, heads| {
                acc.checked_add(heads.checked_mul(per_head)?)
            });
        }
        let kv_heads = self.head_count_kv.or(self.head_count).filter(|h| *h > 0)?;
        let layers = self.kv_layer_count()?;
        layers.checked_mul(kv_heads)?.checked_mul(per_head)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf;
    use crate::testing::GgufWriter;
    use std::path::{Path, PathBuf};

    const MIB: u64 = 1 << 20;

    /// Little helper: an unsigned 32-bit integer array (type 9, element type 4).
    fn u32_array(w: &mut GgufWriter, key: &str, values: &[u32]) {
        let mut v = Vec::new();
        v.extend_from_slice(&4u32.to_le_bytes());
        v.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for x in values {
            v.extend_from_slice(&x.to_le_bytes());
        }
        w.kv_raw(key, 9, &v);
    }

    /// Dense llama-style header: 32 layers, 8 KV heads of 128 dims.
    fn dense_header() -> Vec<u8> {
        let mut w = GgufWriter::new();
        w.kv_str("general.architecture", "llama");
        w.kv_u32("llama.block_count", 32);
        w.kv_u32("llama.embedding_length", 4096);
        w.kv_u32("llama.attention.head_count", 32);
        w.kv_u32("llama.attention.head_count_kv", 8);
        w.finish()
    }

    /// Qwen3.5-shaped hybrid, scaled down: 8 main layers + 1 NextN, attention every 4th layer.
    fn hybrid_header() -> Vec<u8> {
        let mut w = GgufWriter::new();
        w.kv_str("general.architecture", "qwen35");
        w.kv_u32("qwen35.block_count", 9);
        w.kv_u32("qwen35.nextn_predict_layers", 1);
        w.kv_u32("qwen35.embedding_length", 5120);
        w.kv_u32("qwen35.attention.head_count", 24);
        w.kv_u32("qwen35.attention.head_count_kv", 4);
        w.kv_u32("qwen35.attention.key_length", 256);
        w.kv_u32("qwen35.attention.value_length", 256);
        w.kv_u32("qwen35.full_attention_interval", 4);
        w.kv_u32("qwen35.ssm.conv_kernel", 4);
        w.kv_u32("qwen35.ssm.state_size", 128);
        w.kv_u32("qwen35.ssm.group_count", 16);
        w.kv_u32("qwen35.ssm.time_step_rank", 48);
        w.kv_u32("qwen35.ssm.inner_size", 6144);
        w.finish()
    }

    /// Mixture-of-experts dense attention, 8 experts routed per token.
    fn moe_header() -> Vec<u8> {
        let mut w = GgufWriter::new();
        w.kv_str("general.architecture", "qwen3moe");
        w.kv_u32("qwen3moe.block_count", 48);
        w.kv_u32("qwen3moe.embedding_length", 2048);
        w.kv_u32("qwen3moe.attention.head_count", 32);
        w.kv_u32("qwen3moe.attention.head_count_kv", 4);
        w.kv_u32("qwen3moe.expert_count", 128);
        w.kv_u32("qwen3moe.expert_used_count", 8);
        w.finish()
    }

    #[test]
    fn dense_model_every_layer_holds_kv() {
        let meta = gguf::parse(&dense_header()).unwrap();
        assert!(!meta.is_moe());
        assert!(!meta.has_recurrent_state());
        assert_eq!(meta.kv_layer_count(), Some(32));
        assert_eq!(meta.recurrent_layer_count(), Some(0));
        assert_eq!(meta.recurrent_state_bytes(4), Some(0));
        // 32 layers * 8 heads * (128 + 128) * 2 bytes.
        assert_eq!(meta.kv_cache_bytes_per_token_f16(), Some(131_072));
        assert_eq!(
            runtime_hints(&meta),
            ArchHints {
                kv_layers: 32,
                recurrent_bytes_per_slot: 0,
                offload_layers: 33,
            }
        );
    }

    #[test]
    fn hybrid_model_counts_attention_and_excludes_nextn() {
        let meta = gguf::parse(&hybrid_header()).unwrap();
        assert_eq!(meta.main_layer_count(), Some(8));
        assert_eq!(meta.kv_layer_count(), Some(2));
        assert_eq!(meta.recurrent_layer_count(), Some(6));
        // 2 layers * 4 heads * (256 + 256) * 2 bytes.
        assert_eq!(meta.kv_cache_bytes_per_token_f16(), Some(8192));
        // Per layer: conv (3 * (6144 + 2*16*128) * 4) + ssm (128 * 6144 * 4) = 3_268_608 bytes.
        // Six recurrent layers and two slots.
        assert_eq!(meta.recurrent_state_bytes(2), Some(2 * 6 * 3_268_608));
        assert_eq!(
            runtime_hints(&meta),
            ArchHints {
                kv_layers: 2,
                recurrent_bytes_per_slot: 6 * 3_268_608,
                offload_layers: 10,
            }
        );
    }

    #[test]
    fn per_layer_head_count_kv_array_sets_kv_layers() {
        // Granite-style hybrid: zero KV heads mark the recurrent layers.
        let mut w = GgufWriter::new();
        w.kv_str("general.architecture", "granitehybrid");
        w.kv_u32("granitehybrid.block_count", 4);
        w.kv_u32("granitehybrid.embedding_length", 2048);
        w.kv_u32("granitehybrid.attention.head_count", 16);
        u32_array(
            &mut w,
            "granitehybrid.attention.head_count_kv",
            &[0, 8, 0, 8],
        );
        w.kv_u32("granitehybrid.ssm.conv_kernel", 4);
        w.kv_u32("granitehybrid.ssm.state_size", 16);
        w.kv_u32("granitehybrid.ssm.inner_size", 64);
        let meta = gguf::parse(&w.finish()).unwrap();
        assert_eq!(
            meta.head_count_kv_layers.as_deref(),
            Some(&[0, 8, 0, 8][..])
        );
        assert_eq!(meta.head_count_kv, None);
        assert_eq!(meta.kv_layer_count(), Some(2));
        assert_eq!(meta.recurrent_layer_count(), Some(2));
        // Conv: 3 * (64 + 0) * 4 = 768. SSM: 16 * 64 * 4 = 4096. Two layers, one slot.
        assert_eq!(meta.recurrent_state_bytes(1), Some(2 * (768 + 4096)));
        // KV: 8 heads * (128 + 128) * 2 bytes on each attention layer (head dim 128 from 2048/16).
        assert_eq!(meta.kv_cache_bytes_per_token_f16(), Some(2 * 8 * 256 * 2));
    }

    #[test]
    fn pure_ssm_model_has_no_kv_layers() {
        let mut w = GgufWriter::new();
        w.kv_str("general.architecture", "mamba");
        w.kv_u32("mamba.block_count", 4);
        w.kv_u32("mamba.ssm.conv_kernel", 4);
        w.kv_u32("mamba.ssm.state_size", 16);
        w.kv_u32("mamba.ssm.inner_size", 64);
        let meta = gguf::parse(&w.finish()).unwrap();
        assert_eq!(meta.kv_layer_count(), Some(0));
        assert_eq!(meta.recurrent_layer_count(), Some(4));
        // Group count absent: conv_dim = inner. Conv 3*64*4 = 768, SSM 16*64*4 = 4096, four layers.
        assert_eq!(meta.recurrent_state_bytes(1), Some(4 * (768 + 4096)));
        assert_eq!(meta.kv_cache_bytes_per_token_f16(), None);
    }

    #[test]
    fn moe_flag_and_expert_counts() {
        let meta = gguf::parse(&moe_header()).unwrap();
        assert!(meta.is_moe());
        assert_eq!(meta.expert_count, Some(128));
        assert_eq!(meta.expert_used_count, Some(8));
        assert_eq!(meta.kv_layer_count(), Some(48));
        let dense = gguf::parse(&dense_header()).unwrap();
        assert!(!dense.is_moe());
    }

    #[test]
    fn missing_block_count_gives_zero_hints() {
        let mut w = GgufWriter::new();
        w.kv_str("general.architecture", "llama");
        let meta = gguf::parse(&w.finish()).unwrap();
        assert_eq!(meta.kv_layer_count(), None);
        assert_eq!(
            runtime_hints(&meta),
            ArchHints {
                kv_layers: 0,
                recurrent_bytes_per_slot: 0,
                offload_layers: 0,
            }
        );
    }

    #[test]
    fn overflowing_ssm_sizes_return_none() {
        let mut w = GgufWriter::new();
        w.kv_str("general.architecture", "mamba");
        w.kv_u32("mamba.block_count", 4);
        w.kv_u32("mamba.ssm.conv_kernel", 4);
        w.kv_raw("mamba.ssm.state_size", 10, &u64::MAX.to_le_bytes());
        w.kv_u32("mamba.ssm.inner_size", 64);
        let meta = gguf::parse(&w.finish()).unwrap();
        assert_eq!(meta.recurrent_state_bytes(1), None);
        assert_eq!(runtime_hints(&meta).recurrent_bytes_per_slot, 0);
    }

    /// Directory of local sample models. Set `RAPTOR_GGUF_SAMPLES` to override.
    fn sample_dir() -> Option<PathBuf> {
        let dir = std::env::var_os("RAPTOR_GGUF_SAMPLES")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(r"D:\AI_Storage\lmstudio\lmstudio-community"));
        dir.is_dir().then_some(dir)
    }

    /// Parses a sample file when present. Returns `None` (and logs why) when it is absent.
    fn sample(dir: &Path, rel: &str) -> Option<GgufMeta> {
        let path = dir.join(rel);
        if !path.is_file() {
            eprintln!("skipping: {} not found", path.display());
            return None;
        }
        Some(gguf::parse_file(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display())))
    }

    #[test]
    fn real_qwen35_header_matches_llama_cpp_log() {
        let Some(dir) = sample_dir() else {
            eprintln!("skipping: sample directory not found");
            return;
        };
        let Some(meta) = sample(&dir, "Qwen 3.8 27b/Qwen3.8-27B-UD-IQ3_S.gguf") else {
            return;
        };
        assert_eq!(meta.architecture.as_deref(), Some("qwen35"));
        assert_eq!(meta.block_count, Some(65));
        assert_eq!(meta.nextn_predict_layers, Some(1));
        assert_eq!(meta.kv_layer_count(), Some(16));
        assert_eq!(meta.recurrent_layer_count(), Some(48));
        // llama.cpp: KV f16 256 MiB at ctx 4096.
        assert_eq!(
            meta.kv_cache_bytes_per_token_f16().map(|b| b * 4096),
            Some(256 * MIB)
        );
        // llama.cpp: recurrent state 598.5 MiB for 4 slots (R 22.5 MiB + S 576 MiB).
        assert_eq!(meta.recurrent_state_bytes(4), Some(598 * MIB + MIB / 2));
        // llama.cpp: 66 offloadable layers including the output layer.
        assert_eq!(runtime_hints(&meta).offload_layers, 66);
        assert!(!meta.is_moe());
    }

    #[test]
    fn other_local_samples_parse_and_report_facts() {
        let Some(dir) = sample_dir() else {
            return;
        };
        if let Some(meta) = sample(&dir, "Ornith 1.5/Ornith-1.5-35B-A3B-AD-IQ3_S-IQ3_XXS.gguf") {
            assert_eq!(meta.architecture.as_deref(), Some("qwen35moe"));
            assert!(meta.is_moe());
            assert_eq!(meta.kv_layer_count(), Some(10));
            assert_eq!(meta.recurrent_layer_count(), Some(30));
            eprintln!("ornith hints: {:?}", runtime_hints(&meta));
        }
        if let Some(meta) = sample(
            &dir,
            "granite-4.2-30b.i1-IQ3_M/granite-4.2-30b.i1-IQ3_M.gguf",
        ) {
            assert_eq!(meta.architecture.as_deref(), Some("granite"));
            assert_eq!(meta.kv_layer_count(), Some(64));
            assert!(!meta.has_recurrent_state());
            eprintln!("granite hints: {:?}", runtime_hints(&meta));
        }
        if let Some(meta) = sample(&dir, "flint-GGUF/flint-Q4_K_M.gguf") {
            assert_eq!(meta.architecture.as_deref(), Some("flint"));
            assert!(meta.is_moe());
            eprintln!("flint hints: {:?}", runtime_hints(&meta));
        }
        if let Some(meta) = sample(
            &dir,
            "bartowski--Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-IQ4_XS/Qwen3.8-Flash-Next-IQ4_XS-00001-of-00003.gguf.part",
        ) {
            assert_eq!(meta.architecture.as_deref(), Some("qwen4exp"));
            assert!(meta.is_moe());
            assert_eq!(meta.kv_layer_count(), Some(12));
            eprintln!("qwen4exp hints: {:?}", runtime_hints(&meta));
        }
    }
}
