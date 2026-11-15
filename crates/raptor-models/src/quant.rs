//! Quantization knowledge: label parsing from file names, approximate
//! bits-per-weight, a plain-input compatibility rating and quant recommendations.
//! Deliberately independent of raptor-runtime: callers pass hardware numbers in.

use serde::{Deserialize, Serialize};

/// Approximate bits per weight for a GGML quant type. These are rounded estimates
/// used for ranking; the real size is always the file size reported by the Hub.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuantSpec {
    pub label: &'static str,
    pub bits_per_weight: f32,
}

const TABLE: &[QuantSpec] = &[
    QuantSpec {
        label: "F32",
        bits_per_weight: 32.0,
    },
    QuantSpec {
        label: "F16",
        bits_per_weight: 16.0,
    },
    QuantSpec {
        label: "BF16",
        bits_per_weight: 16.0,
    },
    QuantSpec {
        label: "Q8_0",
        bits_per_weight: 8.5,
    },
    QuantSpec {
        label: "Q6_K",
        bits_per_weight: 6.56,
    },
    QuantSpec {
        label: "Q5_K_M",
        bits_per_weight: 5.69,
    },
    QuantSpec {
        label: "Q5_K_S",
        bits_per_weight: 5.54,
    },
    QuantSpec {
        label: "Q5_1",
        bits_per_weight: 6.0,
    },
    QuantSpec {
        label: "Q5_0",
        bits_per_weight: 5.5,
    },
    QuantSpec {
        label: "Q4_K_M",
        bits_per_weight: 4.89,
    },
    QuantSpec {
        label: "Q4_K_S",
        bits_per_weight: 4.58,
    },
    QuantSpec {
        label: "Q4_1",
        bits_per_weight: 5.0,
    },
    QuantSpec {
        label: "Q4_0",
        bits_per_weight: 4.55,
    },
    QuantSpec {
        label: "Q3_K_L",
        bits_per_weight: 4.27,
    },
    QuantSpec {
        label: "Q3_K_M",
        bits_per_weight: 3.89,
    },
    QuantSpec {
        label: "Q3_K_S",
        bits_per_weight: 3.5,
    },
    QuantSpec {
        label: "Q2_K",
        bits_per_weight: 3.35,
    },
    QuantSpec {
        label: "IQ4_NL",
        bits_per_weight: 4.5,
    },
    QuantSpec {
        label: "IQ4_XS",
        bits_per_weight: 4.25,
    },
    QuantSpec {
        label: "IQ3_M",
        bits_per_weight: 3.66,
    },
    QuantSpec {
        label: "IQ3_S",
        bits_per_weight: 3.44,
    },
    QuantSpec {
        label: "IQ3_XS",
        bits_per_weight: 3.3,
    },
    QuantSpec {
        label: "IQ3_XXS",
        bits_per_weight: 3.06,
    },
    QuantSpec {
        label: "IQ2_M",
        bits_per_weight: 2.7,
    },
    QuantSpec {
        label: "IQ2_S",
        bits_per_weight: 2.5,
    },
    QuantSpec {
        label: "IQ2_XS",
        bits_per_weight: 2.31,
    },
    QuantSpec {
        label: "IQ2_XXS",
        bits_per_weight: 2.06,
    },
    QuantSpec {
        label: "IQ1_M",
        bits_per_weight: 1.75,
    },
    QuantSpec {
        label: "IQ1_S",
        bits_per_weight: 1.56,
    },
];

/// Looks up a quant label (case-insensitive) in the known table.
pub fn lookup(label: &str) -> Option<&'static QuantSpec> {
    TABLE.iter().find(|q| q.label.eq_ignore_ascii_case(label))
}

/// Finds the quant label inside a file name such as `Llama-3-8B-Q4_K_M.gguf`.
/// Matching is done on whole `_`-separated segments, so `BF16` never matches `F16`
/// and `Qwen2.5` never matches anything. The leftmost match wins.
pub fn parse_quant_label(file_name: &str) -> Option<&'static str> {
    let upper = file_name.to_ascii_uppercase();
    for token in upper.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')) {
        if token.is_empty() {
            continue;
        }
        let segs: Vec<&str> = token.split('_').collect();
        for start in 0..segs.len() {
            // Longest candidate first so `Q4_K_M` wins over a shorter accidental hit.
            for end in (start + 1..=segs.len()).rev() {
                let candidate = segs[start..end].join("_");
                if let Some(q) = lookup(&candidate) {
                    return Some(q.label);
                }
            }
        }
    }
    None
}

/// Overhead reserved for the runtime, compute buffers and the OS on top of weights and KV cache.
pub const RUNTIME_OVERHEAD_BYTES: u64 = 768 * 1024 * 1024;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// How well a model fits the hardware. Ordered best to worst.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CompatRating {
    /// Fits in VRAM with at least 20% headroom.
    Great,
    /// Fits fully resident: barely in VRAM, or in RAM (CPU inference, slower).
    Ok,
    /// Only fits with partial GPU offload, or with almost no headroom.
    Tight,
    /// Exceeds RAM plus VRAM combined.
    WontFit,
}

/// Plain hardware input. The caller measures it; this crate does not probe hardware.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HardwareSummary {
    pub ram_bytes: u64,
    pub vram_bytes: u64,
}

/// Plain model input for fit estimates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelFootprint {
    pub file_bytes: u64,
    /// KV cache cost per context token (see `GgufMeta::kv_bytes_per_token`).
    pub kv_bytes_per_token: u64,
}

/// Estimated resident bytes needed for `ctx` tokens of context.
pub fn required_bytes(fp: ModelFootprint, ctx: u64) -> u64 {
    fp.file_bytes
        .saturating_add(fp.kv_bytes_per_token.saturating_mul(ctx))
        .saturating_add(RUNTIME_OVERHEAD_BYTES)
}

/// Rates a model at a context length against the hardware. Rules:
/// - VRAM >= need + 20%: `Great`; VRAM >= need: `Ok`.
/// - RAM >= need + 20% (CPU inference): `Ok`.
/// - RAM >= need, or RAM + VRAM >= need (partial offload): `Tight`.
/// - Otherwise `WontFit`.
pub fn compat_rating(fp: ModelFootprint, ctx: u64, hw: HardwareSummary) -> CompatRating {
    let need = required_bytes(fp, ctx);
    let headroom = need.saturating_add(need / 5);
    if hw.vram_bytes >= headroom {
        CompatRating::Great
    } else if hw.vram_bytes >= need || hw.ram_bytes >= headroom {
        CompatRating::Ok
    } else if hw.ram_bytes >= need || hw.ram_bytes.saturating_add(hw.vram_bytes) >= need {
        CompatRating::Tight
    } else {
        CompatRating::WontFit
    }
}

/// One downloadable variant of a repo, as listed by the Hub.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuantOption {
    pub file_name: String,
    pub file_bytes: u64,
}

/// A ranked suggestion for one quant option.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct QuantRecommendation {
    pub file_name: String,
    pub label: Option<&'static str>,
    pub bits_per_weight: Option<f32>,
    pub file_bytes: u64,
    pub rating: CompatRating,
    pub reason: String,
}

/// Ranks quant options for the hardware. Order: best fit rating first, then highest
/// quality (bits per weight) within a rating, then file name for stability.
/// `ctx_target` is clamped to at least 1 token.
pub fn recommend_quant(
    options: &[QuantOption],
    kv_bytes_per_token: u64,
    ctx_target: u64,
    hw: HardwareSummary,
) -> Vec<QuantRecommendation> {
    let ctx = ctx_target.max(1);
    let mut out: Vec<QuantRecommendation> = options
        .iter()
        .map(|o| {
            let label = parse_quant_label(&o.file_name);
            let bits = label.and_then(lookup).map(|q| q.bits_per_weight);
            let fp = ModelFootprint {
                file_bytes: o.file_bytes,
                kv_bytes_per_token,
            };
            let rating = compat_rating(fp, ctx, hw);
            let need = required_bytes(fp, ctx);
            QuantRecommendation {
                file_name: o.file_name.clone(),
                label,
                bits_per_weight: bits,
                file_bytes: o.file_bytes,
                rating,
                reason: reason(rating, need, ctx, hw),
            }
        })
        .collect();
    out.sort_by(|a, b| {
        a.rating
            .cmp(&b.rating)
            .then_with(|| {
                b.bits_per_weight
                    .unwrap_or(0.0)
                    .total_cmp(&a.bits_per_weight.unwrap_or(0.0))
            })
            .then_with(|| a.file_name.cmp(&b.file_name))
    });
    out
}

fn reason(rating: CompatRating, need: u64, ctx: u64, hw: HardwareSummary) -> String {
    let need_gib = need as f64 / GIB;
    let ram_gib = hw.ram_bytes as f64 / GIB;
    let vram_gib = hw.vram_bytes as f64 / GIB;
    match rating {
        CompatRating::Great => format!(
            "fits in VRAM with headroom (about {need_gib:.1} GiB at {ctx} tokens of context)"
        ),
        CompatRating::Ok if hw.vram_bytes >= need => {
            "fits in VRAM with little headroom".to_string()
        }
        CompatRating::Ok => "fits in RAM without GPU offload, so generation is slower".to_string(),
        CompatRating::Tight => {
            "fits only with partial GPU offload or with very little headroom".to_string()
        }
        CompatRating::WontFit => format!(
            "needs about {need_gib:.1} GiB at {ctx} tokens; {ram_gib:.1} GiB RAM and {vram_gib:.1} GiB VRAM available"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: u64 = 1024 * 1024 * 1024;

    #[test]
    fn parses_common_labels() {
        assert_eq!(
            parse_quant_label("Llama-3-8B-Instruct-Q4_K_M.gguf"),
            Some("Q4_K_M")
        );
        assert_eq!(parse_quant_label("model.IQ3_XS.gguf"), Some("IQ3_XS"));
        assert_eq!(parse_quant_label("mistral-7b-q8_0.gguf"), Some("Q8_0"));
        assert_eq!(parse_quant_label("phi-3-f16.gguf"), Some("F16"));
        assert_eq!(
            parse_quant_label("model-Q6_K-00001-of-00002.gguf"),
            Some("Q6_K")
        );
    }

    #[test]
    fn label_segments_must_match_whole() {
        assert_eq!(parse_quant_label("model-BF16.gguf"), Some("BF16"));
        assert_eq!(parse_quant_label("Qwen2.5-7B.gguf"), None);
        assert_eq!(parse_quant_label("XQ4_K_M.gguf"), None);
        assert_eq!(parse_quant_label("readme.md"), None);
    }

    #[test]
    fn lookup_is_case_insensitive() {
        let q = lookup("q4_k_m").expect("known label");
        assert_eq!(q.label, "Q4_K_M");
        assert!(lookup("Q9_Z").is_none());
    }

    #[test]
    fn bits_table_is_sane() {
        assert!(
            lookup("Q8_0").unwrap().bits_per_weight > lookup("Q4_K_M").unwrap().bits_per_weight
        );
        assert!(lookup("IQ1_S").unwrap().bits_per_weight < 2.0);
    }

    #[test]
    fn required_bytes_includes_kv_and_overhead() {
        let fp = ModelFootprint {
            file_bytes: G,
            kv_bytes_per_token: 1024,
        };
        assert_eq!(
            required_bytes(fp, 1000),
            G + 1024 * 1000 + RUNTIME_OVERHEAD_BYTES
        );
    }

    #[test]
    fn rating_great_needs_vram_headroom() {
        let fp = ModelFootprint {
            file_bytes: 4 * G,
            kv_bytes_per_token: 0,
        };
        let need = required_bytes(fp, 4096);
        let hw = HardwareSummary {
            ram_bytes: 0,
            vram_bytes: need + need / 5,
        };
        assert_eq!(compat_rating(fp, 4096, hw), CompatRating::Great);
    }

    #[test]
    fn rating_ok_when_barely_in_vram() {
        let fp = ModelFootprint {
            file_bytes: 4 * G,
            kv_bytes_per_token: 0,
        };
        let need = required_bytes(fp, 4096);
        let hw = HardwareSummary {
            ram_bytes: 0,
            vram_bytes: need,
        };
        assert_eq!(compat_rating(fp, 4096, hw), CompatRating::Ok);
    }

    #[test]
    fn rating_ok_for_cpu_with_headroom() {
        let fp = ModelFootprint {
            file_bytes: 4 * G,
            kv_bytes_per_token: 0,
        };
        let need = required_bytes(fp, 4096);
        let hw = HardwareSummary {
            ram_bytes: need + need / 5,
            vram_bytes: 0,
        };
        assert_eq!(compat_rating(fp, 4096, hw), CompatRating::Ok);
    }

    #[test]
    fn rating_tight_for_partial_offload() {
        let fp = ModelFootprint {
            file_bytes: 8 * G,
            kv_bytes_per_token: 0,
        };
        let need = required_bytes(fp, 4096);
        let hw = HardwareSummary {
            ram_bytes: need / 2,
            vram_bytes: need / 2,
        };
        assert_eq!(compat_rating(fp, 4096, hw), CompatRating::Tight);
    }

    #[test]
    fn rating_wont_fit_when_over_total_memory() {
        let fp = ModelFootprint {
            file_bytes: 70 * G,
            kv_bytes_per_token: 0,
        };
        let hw = HardwareSummary {
            ram_bytes: 16 * G,
            vram_bytes: 8 * G,
        };
        assert_eq!(compat_rating(fp, 4096, hw), CompatRating::WontFit);
    }

    #[test]
    fn rating_order_is_best_first() {
        assert!(CompatRating::Great < CompatRating::Ok);
        assert!(CompatRating::Ok < CompatRating::Tight);
        assert!(CompatRating::Tight < CompatRating::WontFit);
    }

    #[test]
    fn recommendations_rank_fit_then_quality() {
        let opts = vec![
            QuantOption {
                file_name: "m-Q8_0.gguf".into(),
                file_bytes: 8 * G,
            },
            QuantOption {
                file_name: "m-Q4_K_M.gguf".into(),
                file_bytes: 4 * G,
            },
            QuantOption {
                file_name: "m-Q2_K.gguf".into(),
                file_bytes: 3 * G,
            },
            QuantOption {
                file_name: "m-Q5_K_M.gguf".into(),
                file_bytes: 5 * G,
            },
        ];
        // 8 GiB VRAM: Q4_K_M and Q2_K can be Great/Ok, Q8_0 cannot fit in VRAM with headroom.
        let hw = HardwareSummary {
            ram_bytes: 32 * G,
            vram_bytes: 8 * G,
        };
        let recs = recommend_quant(&opts, 0, 4096, hw);
        assert_eq!(recs.len(), 4);
        assert_eq!(recs[0].label, Some("Q5_K_M"));
        assert_eq!(recs[0].rating, CompatRating::Great);
        assert!(recs[0].reason.contains("headroom"));
        // Ratings never get better as we go down the list.
        for pair in recs.windows(2) {
            assert!(pair[0].rating <= pair[1].rating);
        }
    }

    #[test]
    fn recommendations_handle_unknown_labels() {
        let opts = vec![QuantOption {
            file_name: "weird.gguf".into(),
            file_bytes: G,
        }];
        let recs = recommend_quant(
            &opts,
            0,
            0,
            HardwareSummary {
                ram_bytes: 64 * G,
                vram_bytes: 0,
            },
        );
        assert_eq!(recs[0].label, None);
        assert_eq!(recs[0].bits_per_weight, None);
        assert_eq!(recs[0].rating, CompatRating::Ok);
    }

    #[test]
    fn wont_fit_reason_names_the_shortfall() {
        let opts = vec![QuantOption {
            file_name: "big-F16.gguf".into(),
            file_bytes: 140 * G,
        }];
        let recs = recommend_quant(
            &opts,
            0,
            4096,
            HardwareSummary {
                ram_bytes: 16 * G,
                vram_bytes: 0,
            },
        );
        assert_eq!(recs[0].rating, CompatRating::WontFit);
        assert!(recs[0].reason.starts_with("needs about"));
    }
}
