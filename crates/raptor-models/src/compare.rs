//! Side-by-side model comparison, task tags and task ranking.
//!
//! Most numbers here are arithmetic on header metadata and the quant table, so they are
//! exact in their inputs but approximate in their meaning. Items marked **heuristic** are
//! guesses from names, Hub tags and metadata; they are for ordering and hints, not facts.
//! Fit numbers use the same model as `quant::compat_rating`: weights + KV cache at the
//! requested context + a fixed runtime overhead.

use crate::library::ModelDetails;
use crate::quant::{
    compat_rating, lookup, parse_quant_label, required_bytes, CompatRating, HardwareSummary,
    ModelFootprint,
};
use raptor_core::{RaptorError, Result};
use serde::{Deserialize, Serialize};

/// Context length used when the caller does not choose one.
pub const DEFAULT_CTX: u64 = 4096;
/// Largest context a comparison or ranking will estimate for.
pub const MAX_COMPARE_CTX: u64 = 1_048_576;
/// Score a task needs before a model is tagged with it.
const TAG_THRESHOLD: u32 = 2;

/// What a model is good for. Tags are **heuristic**; see [`task_tags`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Task {
    Coding,
    Reasoning,
    General,
    Agentic,
}

/// One task tag with the score behind it and the evidence that produced it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskTag {
    pub task: Task,
    pub score: u32,
    /// Human-readable reasons, for example `repo tag 'code'` or `name token 'r1'`.
    pub evidence: Vec<String>,
}

/// Plain description of a model. Built from an installed model with
/// [`ModelProfile::from_details`], or filled in by hand for a Hub candidate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelProfile {
    pub id: String,
    /// Display name: the file name without its `.gguf` extension.
    pub name: String,
    pub architecture: Option<String>,
    /// Total file bytes (all shards).
    pub file_bytes: u64,
    pub quant_label: Option<String>,
    /// Trained context length from the header, if present.
    pub context_train: Option<u64>,
    /// KV cache bytes per context token (f16). `None` when the header lacks the fields.
    pub kv_bytes_per_token: Option<u64>,
    pub has_chat_template: bool,
    /// Hub tags of the repo, empty when unknown.
    pub tags: Vec<String>,
}

impl ModelProfile {
    /// Profile of an installed model, using its header and recorded Hub tags.
    pub fn from_details(details: &ModelDetails) -> Self {
        let e = &details.entry;
        Self {
            id: e.id.clone(),
            name: strip_gguf(&e.file_name).to_string(),
            architecture: e.gguf.architecture.clone(),
            file_bytes: e.size_bytes,
            quant_label: e
                .gguf
                .quant_label
                .clone()
                .or_else(|| parse_quant_label(&e.file_name).map(str::to_string)),
            context_train: e.gguf.context_length,
            kv_bytes_per_token: e.gguf.kv_bytes_per_token(),
            has_chat_template: e.gguf.has_chat_template,
            tags: details.hub_tags.clone(),
        }
    }
}

fn strip_gguf(name: &str) -> &str {
    name.strip_suffix(".gguf").unwrap_or(name)
}

/// Approximate parameter count from the file size and the quant's bits per weight.
/// **Heuristic**: the header does not carry a parameter count, and the quant table is rounded.
pub fn params_estimate(file_bytes: u64, bits_per_weight: f32) -> Option<u64> {
    if bits_per_weight <= 0.0 || file_bytes == 0 {
        return None;
    }
    Some((file_bytes as f64 * 8.0 / f64::from(bits_per_weight)).round() as u64)
}

/// Parameter size written in a model name, such as `7B` in `Qwen2.5-7B-Instruct` (in billions).
/// **Heuristic**: reads the first `<number>B` token of `-`/`_`-separated segments.
pub fn params_from_name(name: &str) -> Option<f32> {
    name.split(['-', '_']).find_map(|tok| {
        let number = tok.strip_suffix(['B', 'b'])?;
        if number.is_empty() || !number.bytes().all(|b| b.is_ascii_digit() || b == b'.') {
            return None;
        }
        number.parse::<f32>().ok().filter(|v| *v > 0.0)
    })
}

/// Tags a model for tasks. **Heuristic.** Signals and their weights:
/// - Hub repo tags containing `code` (coding), `reasoning`/`thinking`/`qwq`/`r1` (reasoning),
///   `tool`/`function-calling`/`agent` (agentic), `instruct`/`chat` (general): +2 each.
/// - Name tokens: `code`/`coder`/`codestral` (coding), `r1`/`qwq`/`reason`/`think` (reasoning),
///   `agent`/`hermes`/`functionary`/`tool` (agentic), `instruct`/`chat`/`it` (general): +2 each.
/// - Metadata: a chat template (general) +1; a trained context of 32k or more (agentic and
///   coding) +1 each; 13B parameters or more (reasoning and general) +1 each.
///
/// A task is tagged when its score reaches 2, so metadata alone cannot tag a model for coding,
/// reasoning or agentic work. Tags are sorted by score. No signal gives no tags.
pub fn task_tags(p: &ModelProfile) -> Vec<TaskTag> {
    let mut acc: Vec<(Task, u32, Vec<String>)> =
        [Task::Coding, Task::Reasoning, Task::General, Task::Agentic]
            .into_iter()
            .map(|t| (t, 0, Vec::new()))
            .collect();
    let mut add = |task: Task, points: u32, why: String| {
        if let Some(slot) = acc.iter_mut().find(|(t, _, _)| *t == task) {
            slot.1 += points;
            slot.2.push(why);
        }
    };

    let tag_rules: [(&str, Task, &str); 12] = [
        ("code", Task::Coding, "code"),
        ("reasoning", Task::Reasoning, "reasoning"),
        ("thinking", Task::Reasoning, "thinking"),
        ("qwq", Task::Reasoning, "qwq"),
        ("r1", Task::Reasoning, "r1"),
        ("tool", Task::Agentic, "tool use"),
        ("function-calling", Task::Agentic, "function calling"),
        ("agent", Task::Agentic, "agent"),
        ("instruct", Task::General, "instruct"),
        ("chat", Task::General, "chat"),
        ("conversational", Task::General, "conversational"),
        ("reason", Task::Reasoning, "reason"),
    ];
    for tag in &p.tags {
        let lower = tag.to_ascii_lowercase();
        for (needle, task, label) in tag_rules {
            if lower.contains(needle) {
                add(task, 2, format!("repo tag '{tag}' ({label})"));
                break;
            }
        }
    }

    let name = p.name.to_ascii_lowercase();
    let tokens: Vec<&str> = name
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| !t.is_empty())
        .collect();
    let has = |t: &str| tokens.contains(&t);
    let has_prefix = |p: &str| tokens.iter().any(|x| x.starts_with(p));
    if has_prefix("code") || has("codestral") {
        add(Task::Coding, 2, "name token 'code'".into());
    }
    if has("r1") || has("qwq") || has_prefix("reason") || has_prefix("think") {
        add(Task::Reasoning, 2, "name token 'reasoning/r1/qwq'".into());
    }
    if has_prefix("agent") || has("hermes") || has("functionary") || has("tool") {
        add(Task::Agentic, 2, "name token 'agent/tool'".into());
    }
    if has("instruct") || has("chat") || has("it") {
        add(Task::General, 2, "name token 'instruct/chat/it'".into());
    }

    if p.has_chat_template {
        add(Task::General, 1, "chat template in header".into());
    }
    if p.context_train.is_some_and(|c| c >= 32_768) {
        add(Task::Agentic, 1, "trained context of 32k or more".into());
        add(Task::Coding, 1, "trained context of 32k or more".into());
    }
    if params_from_name(&p.name).is_some_and(|b| b >= 13.0) {
        add(
            Task::Reasoning,
            1,
            "13B parameters or more (from name)".into(),
        );
        add(
            Task::General,
            1,
            "13B parameters or more (from name)".into(),
        );
    }

    let mut tags: Vec<TaskTag> = acc
        .into_iter()
        .filter(|(_, score, _)| *score >= TAG_THRESHOLD)
        .map(|(task, score, evidence)| TaskTag {
            task,
            score,
            evidence,
        })
        .collect();
    tags.sort_by(|a, b| b.score.cmp(&a.score).then(a.task.cmp(&b.task)));
    tags
}

/// One column of a comparison.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ComparisonRow {
    pub id: String,
    pub name: String,
    pub architecture: Option<String>,
    pub quant_label: Option<String>,
    /// Bits per weight from the quant table (approximate).
    pub bits_per_weight: Option<f32>,
    /// **Heuristic**: file size divided by bits per weight.
    pub params_estimate: Option<u64>,
    /// **Heuristic**: parameter size written in the name, in billions.
    pub params_from_name_billions: Option<f32>,
    pub file_bytes: u64,
    pub context_train: Option<u64>,
    pub chat_template: bool,
    /// Weights + KV cache at the comparison context + runtime overhead. `None` when the
    /// header lacks the KV fields.
    pub estimated_bytes_at_ctx: Option<u64>,
    /// True when the requested context is longer than the model was trained for.
    pub ctx_exceeds_train: bool,
    /// Fit rating for the hardware given to the comparison. `None` without hardware or KV data.
    pub compat: Option<CompatRating>,
    /// **Heuristic** task tags.
    pub tags: Vec<TaskTag>,
}

/// A structured side-by-side view.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Comparison {
    pub ctx: u64,
    pub hardware: Option<HardwareSummary>,
    /// Rows in the order the models were given.
    pub rows: Vec<ComparisonRow>,
}

/// Compares models at a context length, optionally rating fit for `hardware`.
pub fn compare(
    models: &[ModelProfile],
    ctx: u64,
    hardware: Option<HardwareSummary>,
) -> Result<Comparison> {
    if ctx == 0 || ctx > MAX_COMPARE_CTX {
        return Err(RaptorError::InvalidInput(format!(
            "context must be between 1 and {MAX_COMPARE_CTX} tokens"
        )));
    }
    let rows = models
        .iter()
        .map(|p| {
            let bpw = p
                .quant_label
                .as_deref()
                .and_then(lookup)
                .map(|q| q.bits_per_weight);
            let footprint = p.kv_bytes_per_token.map(|kv| ModelFootprint {
                file_bytes: p.file_bytes,
                kv_bytes_per_token: kv,
            });
            ComparisonRow {
                id: p.id.clone(),
                name: p.name.clone(),
                architecture: p.architecture.clone(),
                quant_label: p.quant_label.clone(),
                bits_per_weight: bpw,
                params_estimate: bpw.and_then(|b| params_estimate(p.file_bytes, b)),
                params_from_name_billions: params_from_name(&p.name),
                file_bytes: p.file_bytes,
                context_train: p.context_train,
                chat_template: p.has_chat_template,
                estimated_bytes_at_ctx: footprint.map(|fp| required_bytes(fp, ctx)),
                ctx_exceeds_train: p.context_train.is_some_and(|c| ctx > c),
                compat: match (footprint, hardware) {
                    (Some(fp), Some(hw)) => Some(compat_rating(fp, ctx, hw)),
                    _ => None,
                },
                tags: task_tags(p),
            }
        })
        .collect();
    Ok(Comparison {
        ctx,
        hardware,
        rows,
    })
}

/// A candidate's place in a task ranking.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RankedCandidate {
    pub id: String,
    /// Ordering score: task score scaled by fit, plus a small quality term. Higher is better.
    pub score: f32,
    /// The **heuristic** task score (0 when the task was not detected).
    pub task_score: u32,
    pub rating: Option<CompatRating>,
    pub bits_per_weight: Option<f32>,
    pub reasons: Vec<String>,
}

/// Ranks candidates for a task at the default context. See [`rank_for_at`].
pub fn rank_for(
    task: Task,
    candidates: &[ModelProfile],
    hardware: HardwareSummary,
) -> Vec<RankedCandidate> {
    rank_for_at(task, candidates, hardware, DEFAULT_CTX)
}

/// Ranks candidates for a task at `ctx` tokens.
///
/// Order: models that cannot fit at all go last. Otherwise by task score (**heuristic**) times
/// a fit factor (Great 1.0, Ok 0.85, Tight 0.5; unknown KV size counts as Ok with a reason),
/// then by bits per weight (higher quality first), then by id for stability.
pub fn rank_for_at(
    task: Task,
    candidates: &[ModelProfile],
    hardware: HardwareSummary,
    ctx: u64,
) -> Vec<RankedCandidate> {
    let ctx = ctx.clamp(1, MAX_COMPARE_CTX);
    let mut out: Vec<RankedCandidate> = candidates
        .iter()
        .map(|p| {
            let tags = task_tags(p);
            let tag = tags.iter().find(|t| t.task == task);
            let task_score = tag.map(|t| t.score).unwrap_or(0);
            let mut reasons: Vec<String> = match tag {
                Some(t) => t
                    .evidence
                    .iter()
                    .map(|e| format!("tagged {task:?}: {e}"))
                    .collect(),
                None => vec![format!("no {task:?} signal found (heuristic)")],
            };
            let bpw = p
                .quant_label
                .as_deref()
                .and_then(lookup)
                .map(|q| q.bits_per_weight);
            let (rating, factor) = match p.kv_bytes_per_token {
                Some(kv) => {
                    let fp = ModelFootprint {
                        file_bytes: p.file_bytes,
                        kv_bytes_per_token: kv,
                    };
                    let r = compat_rating(fp, ctx, hardware);
                    (Some(r), fit_factor(r))
                }
                None => {
                    // Without KV fields the fit check covers weights only; say so.
                    let fp = ModelFootprint {
                        file_bytes: p.file_bytes,
                        kv_bytes_per_token: 0,
                    };
                    let r = compat_rating(fp, ctx, hardware);
                    reasons.push("KV size unknown; fit uses weights and overhead only".into());
                    (Some(r), fit_factor(r) * 0.9)
                }
            };
            if rating == Some(CompatRating::WontFit) {
                reasons.push("does not fit the given memory".into());
            }
            let quality = bpw.unwrap_or(0.0) / 100.0;
            let base = if rating == Some(CompatRating::WontFit) {
                0.0
            } else {
                task_score as f32 * factor
            };
            RankedCandidate {
                id: p.id.clone(),
                score: base + quality,
                task_score,
                rating,
                bits_per_weight: bpw,
                reasons,
            }
        })
        .collect();
    out.sort_by(|a, b| {
        let a_fits = a.rating != Some(CompatRating::WontFit);
        let b_fits = b.rating != Some(CompatRating::WontFit);
        b_fits
            .cmp(&a_fits)
            .then_with(|| b.score.total_cmp(&a.score))
            .then_with(|| {
                b.bits_per_weight
                    .unwrap_or(0.0)
                    .total_cmp(&a.bits_per_weight.unwrap_or(0.0))
            })
            .then_with(|| a.id.cmp(&b.id))
    });
    out
}

fn fit_factor(rating: CompatRating) -> f32 {
    match rating {
        CompatRating::Great => 1.0,
        CompatRating::Ok => 0.85,
        CompatRating::Tight => 0.5,
        CompatRating::WontFit => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: u64 = 1024 * 1024 * 1024;

    fn profile(name: &str) -> ModelProfile {
        ModelProfile {
            id: format!("owner/repo/{name}.gguf"),
            name: name.to_string(),
            architecture: Some("llama".into()),
            file_bytes: 4 * G,
            quant_label: Some("Q4_K_M".into()),
            context_train: Some(8192),
            kv_bytes_per_token: Some(131_072),
            has_chat_template: true,
            tags: vec![],
        }
    }

    #[test]
    fn params_from_name_reads_billions() {
        assert_eq!(params_from_name("Qwen2.5-7B-Instruct"), Some(7.0));
        assert_eq!(params_from_name("Llama-3.2-3B"), Some(3.0));
        assert_eq!(params_from_name("gemma-2-9b-it"), Some(9.0));
        assert_eq!(params_from_name("Mixtral-8x7B"), None);
        assert_eq!(params_from_name("phi-3-mini"), None);
    }

    #[test]
    fn params_estimate_inverts_bits_per_weight() {
        let est = params_estimate(4_000_000_000, 4.0).unwrap();
        assert_eq!(est, 8_000_000_000);
        assert_eq!(params_estimate(0, 4.0), None);
        assert_eq!(params_estimate(10, 0.0), None);
    }

    #[test]
    fn coder_names_are_tagged_for_coding() {
        let tags = task_tags(&profile("Qwen2.5-Coder-7B-Instruct-Q4_K_M"));
        let coding = tags.iter().find(|t| t.task == Task::Coding).unwrap();
        assert!(coding.score >= 2);
        assert!(coding.evidence.iter().any(|e| e.contains("code")));
    }

    #[test]
    fn repo_tags_tag_reasoning_and_agentic() {
        let mut p = profile("DeepSeek-R1-Distill-Qwen-7B-Q4_K_M");
        p.tags = vec!["reasoning".into(), "tool-use".into()];
        let tags = task_tags(&p);
        assert!(tags.iter().any(|t| t.task == Task::Reasoning));
        assert!(tags.iter().any(|t| t.task == Task::Agentic));
    }

    #[test]
    fn metadata_alone_cannot_tag_coding_or_reasoning() {
        let mut p = profile("plain-model-Q4_K_M");
        p.context_train = Some(131_072);
        p.has_chat_template = false;
        let tags = task_tags(&p);
        assert!(tags.iter().all(|t| t.task != Task::Coding));
        assert!(tags.iter().all(|t| t.task != Task::Reasoning));
    }

    #[test]
    fn a_plain_model_with_no_signal_has_no_tags() {
        let mut p = profile("zzz-Q4_K_M");
        p.has_chat_template = false;
        p.context_train = None;
        assert!(task_tags(&p).is_empty());
    }

    #[test]
    fn compare_fills_estimates_and_fit() {
        let hw = HardwareSummary {
            ram_bytes: 32 * G,
            vram_bytes: 8 * G,
        };
        let mut q8 = profile("b-Q8_0");
        q8.quant_label = Some("Q8_0".into());
        let c = compare(&[profile("a-Q4_K_M"), q8], 4096, Some(hw)).unwrap();
        assert_eq!(c.rows.len(), 2);
        assert_eq!(c.rows[0].bits_per_weight, Some(4.89));
        assert!(c.rows[0].estimated_bytes_at_ctx.unwrap() > c.rows[0].file_bytes);
        assert_eq!(c.rows[0].compat, Some(CompatRating::Great));
        assert!(!c.rows[0].ctx_exceeds_train);
        assert_eq!(c.rows[1].bits_per_weight, Some(8.5));
    }

    #[test]
    fn compare_flags_context_beyond_training_and_missing_kv() {
        let mut p = profile("short-Q4_K_M");
        p.kv_bytes_per_token = None;
        let c = compare(&[p], 16_384, None).unwrap();
        let row = &c.rows[0];
        assert!(row.ctx_exceeds_train);
        assert_eq!(row.estimated_bytes_at_ctx, None);
        assert_eq!(row.compat, None);
    }

    #[test]
    fn compare_rejects_nonsense_context() {
        assert!(compare(&[], 0, None).is_err());
        assert!(compare(&[], MAX_COMPARE_CTX + 1, None).is_err());
    }

    #[test]
    fn rank_prefers_task_match_that_fits() {
        let mut coder = profile("Coder-7B-Q4_K_M");
        coder.id = "coder".into();
        let mut chat = profile("Chat-7B-Instruct-Q4_K_M");
        chat.id = "chat".into();
        let hw = HardwareSummary {
            ram_bytes: 32 * G,
            vram_bytes: 16 * G,
        };
        let ranked = rank_for(Task::Coding, &[chat.clone(), coder.clone()], hw);
        assert_eq!(ranked[0].id, "coder");
        assert!(ranked[0].task_score > 0);
        assert_eq!(ranked[1].task_score, 0);
    }

    #[test]
    fn rank_puts_models_that_cannot_fit_last() {
        let mut big = profile("Coder-70B-Q8_0");
        big.id = "big".into();
        big.file_bytes = 70 * G;
        let mut small = profile("Coder-7B-Q4_K_M");
        small.id = "small".into();
        let hw = HardwareSummary {
            ram_bytes: 16 * G,
            vram_bytes: 0,
        };
        let ranked = rank_for(Task::Coding, &[big, small], hw);
        assert_eq!(ranked[0].id, "small");
        assert_eq!(ranked[1].rating, Some(CompatRating::WontFit));
        assert!(ranked[1].reasons.iter().any(|r| r.contains("does not fit")));
    }

    #[test]
    fn rank_is_deterministic_for_ties() {
        let mut a = profile("x-Q4_K_M");
        a.id = "b".into();
        let mut b = profile("x-Q4_K_M");
        b.id = "a".into();
        let hw = HardwareSummary {
            ram_bytes: 64 * G,
            vram_bytes: 0,
        };
        let ranked = rank_for(Task::General, &[a, b], hw);
        assert_eq!(ranked[0].id, "a");
    }
}
