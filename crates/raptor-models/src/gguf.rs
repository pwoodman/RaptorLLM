//! Pure GGUF header parser. Reads key/value metadata only; tensor data is never touched.
//!
//! Hardened for untrusted input: every length is checked against the bytes actually
//! present and a hard cap before use, nothing is pre-allocated from file-supplied
//! counts, nesting is depth-limited, and truncated input is reported (never a panic).
//! `probe` works on a byte prefix so an unsupported architecture can be rejected
//! before a multi-gigabyte download starts.

use raptor_core::{RaptorError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

pub const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const MIN_VERSION: u32 = 2;
const MAX_VERSION: u32 = 3;
/// Key names are short identifiers; anything longer is corrupt.
const MAX_KEY_BYTES: u64 = 256;
/// Largest string value we will materialize (chat templates are far below this).
const MAX_STRING_BYTES: u64 = 1 << 20;
/// Arrays may nest, but not arbitrarily deep.
const MAX_ARRAY_DEPTH: u32 = 4;
const MAX_KV_COUNT: u64 = 1 << 20;
const MAX_ARCH_BYTES: usize = 64;
/// File-based parsing never reads more than this much of a file.
pub const MAX_HEADER_READ_BYTES: u64 = 64 << 20;

/// Architectures Raptor has tested, or confirmed against the pinned llama.cpp (b12340) on a
/// local model file: `qwen35`, `qwen35moe`, `qwen4exp`, `flint` and `granite` were loaded by
/// that build's `llama-fit-params`, which reads the same header and rejects unknown names.
/// Maintained by hand. An architecture missing here is `ArchSupport::Unknown`, not refused.
pub const SUPPORTED_ARCHITECTURES: &[&str] = &[
    "qwen35",
    "qwen35moe",
    "qwen4exp",
    "flint",
    "llama",
    "falcon",
    "gpt2",
    "gptj",
    "gptneox",
    "mpt",
    "starcoder",
    "starcoder2",
    "refact",
    "bloom",
    "stablelm",
    "qwen2",
    "qwen2moe",
    "qwen3",
    "qwen3moe",
    "phi2",
    "phi3",
    "gemma",
    "gemma2",
    "gemma3",
    "command-r",
    "olmo",
    "olmo2",
    "deepseek2",
    "granite",
    "granitemoe",
    "internlm2",
    "minicpm",
    "minicpm3",
    "exaone",
    "nemotron",
    "chatglm",
    "baichuan",
    "xverse",
];

/// Architectures llama.cpp (b12340) is known to refuse as a text model, with the reason.
/// `clip` is a vision projector (mmproj): llama-server answers "CLIP cannot be used as main
/// model, use it with --mmproj instead". Only this list blocks a download or an import.
pub const KNOWN_UNSUPPORTED_ARCHITECTURES: &[(&str, &str)] = &[(
    "clip",
    "it is a vision projector (mmproj) file, not a text model; llama-server refuses it as the main model",
)];

/// Shown for an architecture that is neither tested nor refused. Loading is still attempted.
pub const UNKNOWN_ARCHITECTURE_NOTE: &str =
    "not in Raptor's tested list; llama-server will decide when you load it";

/// How Raptor classifies a GGUF architecture name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArchSupport {
    /// In `SUPPORTED_ARCHITECTURES`.
    Known,
    /// In `KNOWN_UNSUPPORTED_ARCHITECTURES`. Blocks a download or an import.
    KnownUnsupported,
    /// Neither list. Never blocks a file; llama-server decides at load time.
    Unknown,
}

/// Architecture name, its support class and a human note (absent for `Known`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArchInfo {
    pub name: Option<String>,
    pub support: ArchSupport,
    pub note: Option<String>,
}

/// Classifies an architecture name. Pure: no I/O.
pub fn classify_architecture(arch: &str) -> ArchSupport {
    if SUPPORTED_ARCHITECTURES.contains(&arch) {
        ArchSupport::Known
    } else if KNOWN_UNSUPPORTED_ARCHITECTURES
        .iter()
        .any(|(name, _)| *name == arch)
    {
        ArchSupport::KnownUnsupported
    } else {
        ArchSupport::Unknown
    }
}

/// Support facts for a model's architecture, for details and the UI.
pub fn arch_info(arch: Option<&str>) -> ArchInfo {
    let Some(name) = arch else {
        return ArchInfo {
            name: None,
            support: ArchSupport::Unknown,
            note: Some("the GGUF header has no general.architecture".into()),
        };
    };
    let support = classify_architecture(name);
    let note = match support {
        ArchSupport::Known => None,
        ArchSupport::Unknown => Some(UNKNOWN_ARCHITECTURE_NOTE.into()),
        ArchSupport::KnownUnsupported => refusal_reason(name).map(str::to_string),
    };
    ArchInfo {
        name: Some(name.to_string()),
        support,
        note,
    }
}

fn refusal_reason(arch: &str) -> Option<&'static str> {
    KNOWN_UNSUPPORTED_ARCHITECTURES
        .iter()
        .find(|(name, _)| *name == arch)
        .map(|(_, reason)| *reason)
}

/// Errors only for a `KnownUnsupported` architecture. `Unknown` and `Known` pass.
pub fn refuse_known_unsupported(arch: &str) -> Result<()> {
    match refusal_reason(arch) {
        Some(reason) => Err(RaptorError::Unsupported(format!(
            "architecture '{arch}' cannot be loaded as a model: {reason}"
        ))),
        None => Ok(()),
    }
}

/// Metadata extracted from a GGUF header. Fields are `None` when the file does not carry them.
///
/// Adding a field breaks struct literals that do not use `..Default::default()`.
/// Serde defaults keep indexes written before a field existed loadable.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GgufMeta {
    pub version: u32,
    pub kv_count: u64,
    pub architecture: Option<String>,
    pub name: Option<String>,
    /// Raw `general.file_type` (llama.cpp ftype enum).
    pub file_type: Option<u32>,
    /// Quant label derived from `file_type`, e.g. `Q4_K_M`.
    pub quant_label: Option<String>,
    pub context_length: Option<u64>,
    /// Total repeating layers, including any NextN (MTP) layers in `nextn_predict_layers`.
    pub block_count: Option<u64>,
    pub embedding_length: Option<u64>,
    pub head_count: Option<u64>,
    /// Scalar KV head count. Absent when the header writes a per-layer array (see below).
    pub head_count_kv: Option<u64>,
    /// Per-layer KV head counts when the header stores `attention.head_count_kv` as an array.
    /// A zero entry means that layer holds no KV cache (for example a recurrent layer).
    pub head_count_kv_layers: Option<Vec<u64>>,
    pub key_length: Option<u64>,
    /// Per-head value dimension (`attention.value_length`); equals `key_length` when absent.
    pub value_length: Option<u64>,
    /// Hybrid models: every Nth layer is full attention, the others are recurrent or SSM.
    pub full_attention_interval: Option<u64>,
    /// Trailing layers used only for multi-token prediction. They are not part of the main stack.
    pub nextn_predict_layers: Option<u64>,
    /// `ssm.conv_kernel`: the causal conv width; recurrent state keeps `conv_kernel - 1` steps.
    pub ssm_conv_kernel: Option<u64>,
    /// `ssm.state_size`: the SSM state dimension (`d_state`).
    pub ssm_state_size: Option<u64>,
    /// `ssm.inner_size`: the expanded inner width (`d_inner`).
    pub ssm_inner_size: Option<u64>,
    /// `ssm.group_count`: B/C groups in Mamba-2 style layers. Absent means 0.
    pub ssm_group_count: Option<u64>,
    /// `ssm.time_step_rank`: number of delta-net or SSM heads. Informational for the formulas here.
    pub ssm_time_step_rank: Option<u64>,
    /// Mixture-of-experts: total experts per MoE layer.
    pub expert_count: Option<u64>,
    /// Mixture-of-experts: experts routed per token.
    pub expert_used_count: Option<u64>,
    pub tokenizer_model: Option<String>,
    pub has_chat_template: bool,
    /// Entries in `tokenizer.ggml.tokens`. A draft model can only serve a target with the same
    /// vocabulary, so this is how a DFlash draft is matched to its target.
    pub vocab_size: Option<u64>,
}

impl GgufMeta {
    /// Per-head dimension: explicit `key_length`, else `embedding_length / head_count`.
    pub fn head_dim(&self) -> Option<u64> {
        if let Some(k) = self.key_length.filter(|k| *k > 0) {
            return Some(k);
        }
        match (self.embedding_length, self.head_count) {
            (Some(e), Some(h)) if h > 0 => Some(e / h),
            _ => None,
        }
    }

    /// Estimated KV cache bytes per context token, assuming an f16 cache (K and V).
    pub fn kv_bytes_per_token(&self) -> Option<u64> {
        let layers = self.block_count.filter(|v| *v > 0)?;
        let kv_heads = self.head_count_kv.or(self.head_count).filter(|v| *v > 0)?;
        let head_dim = self.head_dim().filter(|v| *v > 0)?;
        2u64.checked_mul(layers)?
            .checked_mul(kv_heads)?
            .checked_mul(head_dim)?
            .checked_mul(2)
    }
}

/// Result of probing a header that may be only a prefix of the file.
// Boxing `Complete` would change a public type; the enum is short-lived, so the size gap is fine.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq)]
pub enum HeaderProbe {
    /// The whole key/value section was present.
    Complete(GgufMeta),
    /// The bytes ended first. `architecture` is set if it appeared before the cut.
    Truncated { architecture: Option<String> },
}

#[derive(Debug)]
enum Stop {
    Truncated,
    Corrupt(String),
}

type Walk<T> = std::result::Result<T, Stop>;

/// Parses a complete header from `bytes`. Truncated input is an error here.
pub fn parse(bytes: &[u8]) -> Result<GgufMeta> {
    match probe(bytes)? {
        HeaderProbe::Complete(meta) => Ok(meta),
        HeaderProbe::Truncated { .. } => Err(RaptorError::Integrity(
            "truncated GGUF header: not enough bytes to read all metadata".into(),
        )),
    }
}

/// Parses as much of the header as `bytes` allows. Corrupt input is an error;
/// running out of bytes is reported as `Truncated`, never as a panic.
pub fn probe(bytes: &[u8]) -> Result<HeaderProbe> {
    let mut acc = Accumulator::default();
    match walk(bytes, &mut acc) {
        Ok(()) => Ok(HeaderProbe::Complete(acc.into_meta())),
        Err(Stop::Truncated) => Ok(HeaderProbe::Truncated {
            architecture: acc.architecture,
        }),
        Err(Stop::Corrupt(msg)) => Err(RaptorError::Integrity(format!("corrupt GGUF: {msg}"))),
    }
}

/// Parses the header of a file, reading at most `MAX_HEADER_READ_BYTES`.
pub fn parse_file(path: &Path) -> Result<GgufMeta> {
    parse_file_with_split(path).map(|(meta, _)| meta)
}

/// Shard metadata a split GGUF file may carry (`split.no` is 0-based, `split.count` is the
/// number of shards in the set). Both are `None` for a single, unsplit file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SplitInfo {
    pub no: Option<u64>,
    pub count: Option<u64>,
}

/// Like [`parse_file`], also returning the split keys of the same header read.
pub fn parse_file_with_split(path: &Path) -> Result<(GgufMeta, SplitInfo)> {
    let file = std::fs::File::open(path)?;
    let mut buf = Vec::new();
    file.take(MAX_HEADER_READ_BYTES).read_to_end(&mut buf)?;
    let mut acc = Accumulator::default();
    match walk(&buf, &mut acc) {
        Ok(()) => {
            let split = SplitInfo {
                no: acc.split_no,
                count: acc.split_count,
            };
            Ok((acc.into_meta(), split))
        }
        Err(Stop::Truncated) => Err(RaptorError::Integrity(
            "truncated GGUF header: not enough bytes to read all metadata".into(),
        )),
        Err(Stop::Corrupt(msg)) => Err(RaptorError::Integrity(format!("corrupt GGUF: {msg}"))),
    }
}

/// True when the architecture is in `SUPPORTED_ARCHITECTURES` (tested). Not a refusal test:
/// see [`classify_architecture`] and [`ensure_supported`].
pub fn is_architecture_supported(arch: &str) -> bool {
    classify_architecture(arch) == ArchSupport::Known
}

/// Gate for a download (checked from the first bytes). Refuses a missing architecture and a
/// `KnownUnsupported` one. An `Unknown` architecture passes: llama-server decides at load time.
pub fn ensure_supported(arch: Option<&str>) -> Result<()> {
    match arch {
        None => Err(RaptorError::Integrity(
            "cannot determine the model architecture from the GGUF header".into(),
        )),
        Some(a) => refuse_known_unsupported(a),
    }
}

/// Maps llama.cpp's `general.file_type` enum to a quant label.
pub fn quant_from_file_type(file_type: u32) -> Option<&'static str> {
    Some(match file_type {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        7 => "Q8_0",
        8 => "Q5_0",
        9 => "Q5_1",
        10 => "Q2_K",
        11 => "Q3_K_S",
        12 => "Q3_K_M",
        13 => "Q3_K_L",
        14 => "Q4_K_S",
        15 => "Q4_K_M",
        16 => "Q5_K_S",
        17 => "Q5_K_M",
        18 => "Q6_K",
        19 => "IQ2_XXS",
        20 => "IQ2_XS",
        21 => "Q2_K_S",
        22 => "IQ3_XS",
        23 => "IQ3_XXS",
        24 => "IQ1_S",
        25 => "IQ4_NL",
        26 => "IQ3_S",
        27 => "IQ3_M",
        28 => "IQ2_S",
        29 => "IQ2_M",
        30 => "IQ4_XS",
        31 => "IQ1_M",
        32 => "BF16",
        36 => "TQ1_0",
        37 => "TQ2_0",
        _ => return None,
    })
}

/// Bounds-checked little-endian reader over a byte slice.
struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: u64) -> Walk<&'a [u8]> {
        let n = usize::try_from(n).map_err(|_| Stop::Truncated)?;
        let end = self.pos.checked_add(n).ok_or(Stop::Truncated)?;
        let slice = self.buf.get(self.pos..end).ok_or(Stop::Truncated)?;
        self.pos = end;
        Ok(slice)
    }

    fn array<const N: usize>(&mut self) -> Walk<[u8; N]> {
        let slice = self.take(N as u64)?;
        <[u8; N]>::try_from(slice).map_err(|_| Stop::Truncated)
    }

    fn u8(&mut self) -> Walk<u8> {
        Ok(self.array::<1>()?[0])
    }

    fn u32(&mut self) -> Walk<u32> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn u64(&mut self) -> Walk<u64> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    /// Reads a length-prefixed UTF-8 string, rejecting lengths above `max` before reading.
    fn string(&mut self, max: u64) -> Walk<String> {
        let len = self.u64()?;
        if len > max {
            return Err(Stop::Corrupt(format!(
                "string length {len} exceeds cap {max}"
            )));
        }
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|_| Stop::Corrupt("string is not valid UTF-8".into()))
    }

    /// Skips a length-prefixed string without materializing it.
    fn skip_string(&mut self, max: u64) -> Walk<()> {
        let len = self.u64()?;
        if len > max {
            return Err(Stop::Corrupt(format!(
                "string length {len} exceeds cap {max}"
            )));
        }
        self.take(len)?;
        Ok(())
    }
}

/// Scalar values we may need to inspect; everything else is skipped.
enum Scalar {
    /// Non-negative integer (any integer type), or `None` for negative values.
    Num(Option<u64>),
    Str(String),
    Bool,
    Other,
}

fn fixed_size(ty: u32) -> Option<u64> {
    match ty {
        0 | 1 | 7 => Some(1),
        2 | 3 => Some(2),
        4..=6 => Some(4),
        10..=12 => Some(8),
        _ => None,
    }
}

fn read_value(cur: &mut Cursor<'_>, ty: u32, depth: u32) -> Walk<Scalar> {
    Ok(match ty {
        0 => Scalar::Num(Some(u64::from(cur.u8()?))),
        1 => Scalar::Num(u64::try_from(cur.u8()? as i8).ok()),
        2 => Scalar::Num(Some(u64::from(u16::from_le_bytes(cur.array()?)))),
        3 => Scalar::Num(u64::try_from(i16::from_le_bytes(cur.array()?)).ok()),
        4 => Scalar::Num(Some(u64::from(cur.u32()?))),
        5 => Scalar::Num(u64::try_from(i32::from_le_bytes(cur.array()?)).ok()),
        6 => {
            cur.take(4)?;
            Scalar::Other
        }
        7 => {
            let b = cur.u8()?;
            if b > 1 {
                return Err(Stop::Corrupt("boolean value is neither 0 nor 1".into()));
            }
            Scalar::Bool
        }
        8 => Scalar::Str(cur.string(MAX_STRING_BYTES)?),
        9 => {
            skip_array(cur, depth)?;
            Scalar::Other
        }
        10 => Scalar::Num(Some(cur.u64()?)),
        11 => Scalar::Num(u64::try_from(i64::from_le_bytes(cur.array()?)).ok()),
        12 => {
            cur.take(8)?;
            Scalar::Other
        }
        other => return Err(Stop::Corrupt(format!("unknown value type {other}"))),
    })
}

/// Skips an array value. Every loop iteration consumes at least 8 bytes, so the
/// element count can never drive work beyond the input length.
fn skip_array(cur: &mut Cursor<'_>, depth: u32) -> Walk<()> {
    if depth >= MAX_ARRAY_DEPTH {
        return Err(Stop::Corrupt("arrays nested too deeply".into()));
    }
    let elem = cur.u32()?;
    let count = cur.u64()?;
    skip_elements(cur, elem, count, depth)
}

/// Byte size of an integer element type, or `None` for non-integer types.
fn int_size(ty: u32) -> Option<u64> {
    match ty {
        0 | 1 | 2 | 3 | 4 | 5 | 10 | 11 => fixed_size(ty),
        _ => None,
    }
}

/// Reads the elements of an array whose header (`elem`, `count`) is already consumed, keeping
/// them when they are all non-negative integers and the count is within `MAX_LAYER_ARRAY`.
/// Every other array is skipped and yields `None`.
fn read_int_array(
    cur: &mut Cursor<'_>,
    elem: u32,
    count: u64,
    depth: u32,
) -> Walk<Option<Vec<u64>>> {
    let Some(size) = int_size(elem) else {
        skip_elements(cur, elem, count, depth)?;
        return Ok(None);
    };
    let total = count.checked_mul(size).ok_or(Stop::Truncated)?;
    let raw = cur.take(total)?;
    if count > MAX_LAYER_ARRAY {
        return Ok(None);
    }
    let mut sub = Cursor { buf: raw, pos: 0 };
    let mut out = Vec::with_capacity(count as usize);
    for _ in 0..count {
        match read_value(&mut sub, elem, 0)? {
            Scalar::Num(Some(n)) => out.push(n),
            _ => return Ok(None),
        }
    }
    Ok(Some(out))
}

/// Skips the `count` elements of an array with element type `elem`.
fn skip_elements(cur: &mut Cursor<'_>, elem: u32, count: u64, depth: u32) -> Walk<()> {
    match elem {
        8 => {
            for _ in 0..count {
                cur.skip_string(MAX_STRING_BYTES)?;
            }
        }
        9 => {
            for _ in 0..count {
                skip_array(cur, depth + 1)?;
            }
        }
        other => {
            let size = fixed_size(other)
                .ok_or_else(|| Stop::Corrupt(format!("unknown array element type {other}")))?;
            let total = count.checked_mul(size).ok_or(Stop::Truncated)?;
            cur.take(total)?;
        }
    }
    Ok(())
}

/// Suffixes of per-architecture numeric keys we keep (full key is `<arch>.<suffix>`).
const NUMERIC_SUFFIXES: &[&str] = &[
    ".context_length",
    ".block_count",
    ".embedding_length",
    ".attention.head_count",
    ".attention.head_count_kv",
    ".attention.key_length",
    ".attention.value_length",
    ".full_attention_interval",
    ".nextn_predict_layers",
    ".ssm.conv_kernel",
    ".ssm.state_size",
    ".ssm.inner_size",
    ".ssm.group_count",
    ".ssm.time_step_rank",
    ".expert_count",
    ".expert_used_count",
];

/// Suffix of the key that some hybrid models write as one value per layer instead of a scalar.
const LAYER_ARRAY_SUFFIX: &str = ".attention.head_count_kv";
/// Longest per-layer integer array we keep. Real models have a few hundred layers at most.
const MAX_LAYER_ARRAY: u64 = 4096;

#[derive(Default)]
struct Accumulator {
    version: u32,
    kv_count: u64,
    architecture: Option<String>,
    name: Option<String>,
    file_type: Option<u32>,
    tokenizer_model: Option<String>,
    has_chat_template: bool,
    vocab_size: Option<u64>,
    /// Keyed by full key. Stored unconditionally for matching suffixes so that key
    /// order in the file does not matter relative to `general.architecture`.
    numbers: HashMap<String, u64>,
    split_no: Option<u64>,
    split_count: Option<u64>,
    /// Integer arrays kept for `LAYER_ARRAY_SUFFIX` keys, keyed by full key.
    layer_arrays: HashMap<String, Vec<u64>>,
}

impl Accumulator {
    fn observe(&mut self, key: &str, value: Scalar) -> Walk<()> {
        match (key, value) {
            ("general.architecture", Scalar::Str(arch)) => {
                if !valid_arch_name(&arch) {
                    return Err(Stop::Corrupt("invalid architecture name".into()));
                }
                self.architecture = Some(arch);
            }
            ("general.name", Scalar::Str(name)) => self.name = Some(name),
            ("general.file_type", Scalar::Num(Some(n))) => {
                self.file_type = u32::try_from(n).ok();
            }
            ("split.no", Scalar::Num(Some(n))) => self.split_no = Some(n),
            ("split.count", Scalar::Num(Some(n))) => self.split_count = Some(n),
            ("tokenizer.ggml.model", Scalar::Str(model)) => self.tokenizer_model = Some(model),
            ("tokenizer.chat_template" | "tokenizer.chat_templates", _) => {
                self.has_chat_template = true;
            }
            (k, Scalar::Num(Some(n))) if NUMERIC_SUFFIXES.iter().any(|s| k.ends_with(s)) => {
                self.numbers.insert(k.to_string(), n);
            }
            _ => {}
        }
        Ok(())
    }

    fn num(&self, suffix: &str) -> Option<u64> {
        let arch = self.architecture.as_deref()?;
        self.numbers.get(&format!("{arch}.{suffix}")).copied()
    }

    fn into_meta(mut self) -> GgufMeta {
        let head_count_kv_layers = self.architecture.as_deref().and_then(|arch| {
            self.layer_arrays
                .remove(&format!("{arch}{LAYER_ARRAY_SUFFIX}"))
        });
        GgufMeta {
            version: self.version,
            kv_count: self.kv_count,
            context_length: self.num("context_length"),
            block_count: self.num("block_count"),
            embedding_length: self.num("embedding_length"),
            head_count: self.num("attention.head_count"),
            head_count_kv: self.num("attention.head_count_kv"),
            head_count_kv_layers,
            key_length: self.num("attention.key_length"),
            value_length: self.num("attention.value_length"),
            full_attention_interval: self.num("full_attention_interval"),
            nextn_predict_layers: self.num("nextn_predict_layers"),
            ssm_conv_kernel: self.num("ssm.conv_kernel"),
            ssm_state_size: self.num("ssm.state_size"),
            ssm_inner_size: self.num("ssm.inner_size"),
            ssm_group_count: self.num("ssm.group_count"),
            ssm_time_step_rank: self.num("ssm.time_step_rank"),
            expert_count: self.num("expert_count"),
            expert_used_count: self.num("expert_used_count"),
            quant_label: self
                .file_type
                .and_then(quant_from_file_type)
                .map(str::to_string),
            architecture: self.architecture,
            name: self.name,
            file_type: self.file_type,
            tokenizer_model: self.tokenizer_model,
            has_chat_template: self.has_chat_template,
            vocab_size: self.vocab_size,
        }
    }
}

fn valid_arch_name(arch: &str) -> bool {
    !arch.is_empty()
        && arch.len() <= MAX_ARCH_BYTES
        && arch
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-')
}

fn walk(bytes: &[u8], acc: &mut Accumulator) -> Walk<()> {
    let mut cur = Cursor { buf: bytes, pos: 0 };
    if cur.take(4)? != GGUF_MAGIC {
        return Err(Stop::Corrupt("bad magic, not a GGUF file".into()));
    }
    let version = cur.u32()?;
    if !(MIN_VERSION..=MAX_VERSION).contains(&version) {
        return Err(Stop::Corrupt(format!("unsupported GGUF version {version}")));
    }
    let _tensor_count = cur.u64()?;
    let kv_count = cur.u64()?;
    if kv_count > MAX_KV_COUNT {
        return Err(Stop::Corrupt(format!(
            "key/value count {kv_count} exceeds cap"
        )));
    }
    acc.version = version;
    acc.kv_count = kv_count;
    for _ in 0..kv_count {
        let key = cur.string(MAX_KEY_BYTES)?;
        let ty = cur.u32()?;
        if ty == 9 && key.ends_with(LAYER_ARRAY_SUFFIX) {
            let elem = cur.u32()?;
            let count = cur.u64()?;
            if let Some(values) = read_int_array(&mut cur, elem, count, 0)? {
                acc.layer_arrays.insert(key, values);
            }
            continue;
        }
        if ty == 9 && key == "tokenizer.ggml.tokens" {
            let elem = cur.u32()?;
            let count = cur.u64()?;
            skip_elements(&mut cur, elem, count, 0)?;
            acc.vocab_size = Some(count);
            continue;
        }
        let value = read_value(&mut cur, ty, 0)?;
        acc.observe(&key, value)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::GgufWriter;

    fn llama_header() -> Vec<u8> {
        let mut w = GgufWriter::new();
        w.kv_str("general.architecture", "llama");
        w.kv_str("general.name", "Tiny Llama");
        w.kv_u32("general.file_type", 15);
        w.kv_u32("llama.context_length", 8192);
        w.kv_u32("llama.block_count", 32);
        w.kv_u32("llama.embedding_length", 4096);
        w.kv_u32("llama.attention.head_count", 32);
        w.kv_u32("llama.attention.head_count_kv", 8);
        w.kv_str("tokenizer.ggml.model", "gpt2");
        w.kv_str("tokenizer.chat_template", "{% for m in messages %}");
        w.finish()
    }

    #[test]
    fn parses_llama_header() {
        let meta = parse(&llama_header()).unwrap();
        assert_eq!(meta.version, 3);
        assert_eq!(meta.architecture.as_deref(), Some("llama"));
        assert_eq!(meta.name.as_deref(), Some("Tiny Llama"));
        assert_eq!(meta.quant_label.as_deref(), Some("Q4_K_M"));
        assert_eq!(meta.context_length, Some(8192));
        assert_eq!(meta.block_count, Some(32));
        assert_eq!(meta.embedding_length, Some(4096));
        assert_eq!(meta.head_count, Some(32));
        assert_eq!(meta.head_count_kv, Some(8));
        assert_eq!(meta.tokenizer_model.as_deref(), Some("gpt2"));
        assert!(meta.has_chat_template);
        assert_eq!(meta.kv_count, 10);
    }

    #[test]
    fn head_dim_and_kv_cache_size() {
        let meta = parse(&llama_header()).unwrap();
        assert_eq!(meta.head_dim(), Some(128));
        // 2 (K,V) * 32 layers * 8 kv heads * 128 dim * 2 bytes = 131072 bytes per token.
        assert_eq!(meta.kv_bytes_per_token(), Some(131_072));
    }

    #[test]
    fn explicit_key_length_wins_over_embedding_division() {
        let mut w = GgufWriter::new();
        w.kv_str("general.architecture", "qwen2");
        w.kv_u32("qwen2.block_count", 2);
        w.kv_u32("qwen2.embedding_length", 100);
        w.kv_u32("qwen2.attention.head_count", 4);
        w.kv_u32("qwen2.attention.key_length", 64);
        let meta = parse(&w.finish()).unwrap();
        assert_eq!(meta.head_dim(), Some(64));
        assert_eq!(meta.head_count_kv, None);
        // Falls back to head_count when kv heads are absent.
        assert_eq!(meta.kv_bytes_per_token(), Some(2 * 2 * 4 * 64 * 2));
    }

    #[test]
    fn numeric_keys_may_precede_architecture() {
        let mut w = GgufWriter::new();
        w.kv_u32("llama.context_length", 4096);
        w.kv_str("general.architecture", "llama");
        let meta = parse(&w.finish()).unwrap();
        assert_eq!(meta.context_length, Some(4096));
    }

    #[test]
    fn missing_fields_are_none_not_errors() {
        let mut w = GgufWriter::new();
        w.kv_str("general.architecture", "gpt2");
        let meta = parse(&w.finish()).unwrap();
        assert_eq!(meta.context_length, None);
        assert_eq!(meta.kv_bytes_per_token(), None);
        assert!(!meta.has_chat_template);
    }

    #[test]
    fn bad_magic_is_an_error() {
        let mut bytes = llama_header();
        bytes[0] = b'X';
        assert!(matches!(parse(&bytes), Err(RaptorError::Integrity(_))));
    }

    #[test]
    fn unsupported_version_is_an_error() {
        let mut bytes = llama_header();
        bytes[4..8].copy_from_slice(&1u32.to_le_bytes());
        assert!(matches!(parse(&bytes), Err(RaptorError::Integrity(m)) if m.contains("version 1")));
    }

    #[test]
    fn every_truncation_point_is_an_error_not_a_panic() {
        let full = llama_header();
        // The writer appends 4 KiB of fake tensor data after the key/values; the header ends before it.
        let kv_end = full.len() - 4096;
        for cut in 0..kv_end {
            let r = parse(&full[..cut]);
            assert!(r.is_err(), "cut at {cut} should not parse");
            // Probe must also never panic; it reports Truncated or Corrupt.
            let _ = probe(&full[..cut]);
        }
        assert!(parse(&full[..kv_end]).is_ok());
        assert!(parse(&full).is_ok());
    }

    #[test]
    fn the_vocabulary_size_is_the_token_array_length() {
        let mut w = GgufWriter::new();
        w.kv_str("general.architecture", "dflash");
        w.kv_array_str("tokenizer.ggml.tokens", &["a", "b", "c", "d", "e"]);
        w.kv_str("tokenizer.ggml.model", "llama");
        let meta = parse(&w.finish()).unwrap();
        assert_eq!(meta.vocab_size, Some(5));
        assert_eq!(
            meta.tokenizer_model.as_deref(),
            Some("llama"),
            "keys after the array still parse"
        );
        assert_eq!(parse(&llama_header()).unwrap().vocab_size, None);
    }

    #[test]
    fn probe_reports_architecture_from_prefix() {
        let mut w = GgufWriter::new();
        w.kv_str("general.architecture", "mistral-next");
        w.kv_array_str("tokenizer.ggml.tokens", &["a", "b", "c", "d", "e"]);
        w.kv_str("tokenizer.ggml.model", "llama");
        let full = w.finish();
        // Cut inside the last key/value (its value is 5 bytes long): architecture was already seen.
        let kv_end = full.len() - 4096;
        let cut = kv_end - 20;
        match probe(&full[..cut]).unwrap() {
            HeaderProbe::Truncated { architecture } => {
                assert_eq!(architecture.as_deref(), Some("mistral-next"));
            }
            other => panic!("expected truncated probe, got {other:?}"),
        }
    }

    #[test]
    fn probe_before_architecture_has_none() {
        let full = llama_header();
        match probe(&full[..30]).unwrap() {
            HeaderProbe::Truncated { architecture } => assert_eq!(architecture, None),
            other => panic!("expected truncated probe, got {other:?}"),
        }
    }

    #[test]
    fn huge_string_length_is_rejected_without_allocating() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(GGUF_MAGIC);
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        // Key length of u64::MAX must be refused by the cap, not attempted.
        bytes.extend_from_slice(&u64::MAX.to_le_bytes());
        assert!(matches!(parse(&bytes), Err(RaptorError::Integrity(_))));
    }

    #[test]
    fn oversized_key_is_rejected() {
        let mut w = GgufWriter::new();
        let long = "k".repeat(300);
        w.kv_str(&long, "v");
        assert!(matches!(parse(&w.finish()), Err(RaptorError::Integrity(m)) if m.contains("cap")));
    }

    #[test]
    fn huge_kv_count_is_rejected() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(GGUF_MAGIC);
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&u64::MAX.to_le_bytes());
        assert!(matches!(parse(&bytes), Err(RaptorError::Integrity(_))));
    }

    #[test]
    fn huge_array_count_in_fixed_type_is_truncation_not_overflow() {
        let mut w = GgufWriter::new();
        w.kv_raw_array_header("tokenizer.ggml.scores", 6, u64::MAX);
        let bytes = w.finish();
        // Declared count * 4 overflows u64: must be an error, never a panic or huge read.
        assert!(parse(&bytes).is_err());
    }

    #[test]
    fn huge_string_array_count_stops_at_end_of_input() {
        let mut w = GgufWriter::new();
        w.kv_raw_array_header("tokenizer.ggml.tokens", 8, u64::MAX / 2);
        w.raw(b"abcdefgh");
        let bytes = w.finish();
        assert!(parse(&bytes).is_err());
    }

    #[test]
    fn deeply_nested_arrays_are_rejected() {
        let mut w = GgufWriter::new();
        w.kv_nested_array_depth("weird.nested", 10);
        let bytes = w.finish();
        assert!(matches!(parse(&bytes), Err(RaptorError::Integrity(m)) if m.contains("nested")));
    }

    #[test]
    fn unknown_value_type_is_rejected() {
        let mut w = GgufWriter::new();
        w.kv_raw("weird.key", 99, &[0u8; 8]);
        assert!(
            matches!(parse(&w.finish()), Err(RaptorError::Integrity(m)) if m.contains("type 99"))
        );
    }

    #[test]
    fn invalid_utf8_string_is_rejected() {
        let mut w = GgufWriter::new();
        w.kv_raw("general.name", 8, &{
            let mut v = 2u64.to_le_bytes().to_vec();
            v.extend_from_slice(&[0xff, 0xfe]);
            v
        });
        assert!(parse(&w.finish()).is_err());
    }

    #[test]
    fn invalid_architecture_name_is_rejected() {
        let mut w = GgufWriter::new();
        w.kv_str("general.architecture", "../../etc");
        assert!(matches!(parse(&w.finish()), Err(RaptorError::Integrity(_))));
    }

    #[test]
    fn bad_bool_is_rejected() {
        let mut w = GgufWriter::new();
        w.kv_raw("flag", 7, &[7u8]);
        assert!(parse(&w.finish()).is_err());
    }

    #[test]
    fn architecture_support_list() {
        assert!(is_architecture_supported("llama"));
        assert!(is_architecture_supported("qwen35"));
        assert!(!is_architecture_supported("made-up-arch"));
        assert!(ensure_supported(Some("llama")).is_ok());
        assert!(ensure_supported(Some("qwen35moe")).is_ok());
        assert!(matches!(
            ensure_supported(None),
            Err(RaptorError::Integrity(_))
        ));
    }

    #[test]
    fn three_outcomes_known_known_unsupported_unknown() {
        assert_eq!(classify_architecture("qwen35"), ArchSupport::Known);
        assert_eq!(classify_architecture("granite"), ArchSupport::Known);
        assert_eq!(classify_architecture("clip"), ArchSupport::KnownUnsupported);
        assert_eq!(classify_architecture("zzfake-arch"), ArchSupport::Unknown);
    }

    #[test]
    fn only_known_unsupported_blocks() {
        // Unknown names never block a download or an import.
        assert!(ensure_supported(Some("zzfake-arch")).is_ok());
        assert!(refuse_known_unsupported("zzfake-arch").is_ok());
        // A known-unsupported name blocks with its reason.
        match ensure_supported(Some("clip")) {
            Err(RaptorError::Unsupported(m)) => {
                assert!(m.contains("clip") && m.contains("mmproj"), "{m}")
            }
            other => panic!("expected unsupported, got {other:?}"),
        }
        assert!(refuse_known_unsupported("clip").is_err());
    }

    #[test]
    fn arch_info_carries_the_user_facing_note() {
        let known = arch_info(Some("qwen35"));
        assert_eq!(known.support, ArchSupport::Known);
        assert_eq!(known.note, None);
        let unknown = arch_info(Some("zzfake-arch"));
        assert_eq!(unknown.support, ArchSupport::Unknown);
        assert_eq!(
            unknown.note.as_deref(),
            Some("not in Raptor's tested list; llama-server will decide when you load it")
        );
        let refused = arch_info(Some("clip"));
        assert_eq!(refused.support, ArchSupport::KnownUnsupported);
        assert!(refused.note.unwrap().contains("mmproj"));
        assert_eq!(arch_info(None).support, ArchSupport::Unknown);
    }

    #[test]
    fn file_type_mapping() {
        assert_eq!(quant_from_file_type(15), Some("Q4_K_M"));
        assert_eq!(quant_from_file_type(7), Some("Q8_0"));
        assert_eq!(quant_from_file_type(9999), None);
    }

    #[test]
    fn parse_file_reads_header_from_disk() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("m.gguf");
        std::fs::write(&p, llama_header()).unwrap();
        assert_eq!(
            parse_file(&p).unwrap().architecture.as_deref(),
            Some("llama")
        );
    }

    #[test]
    fn hybrid_and_moe_keys_are_exposed() {
        let mut w = GgufWriter::new();
        w.kv_str("general.architecture", "qwen35moe");
        w.kv_u32("qwen35moe.block_count", 41);
        w.kv_u32("qwen35moe.nextn_predict_layers", 1);
        w.kv_u32("qwen35moe.full_attention_interval", 4);
        w.kv_u32("qwen35moe.attention.key_length", 256);
        w.kv_u32("qwen35moe.attention.value_length", 128);
        w.kv_u32("qwen35moe.ssm.conv_kernel", 4);
        w.kv_u32("qwen35moe.ssm.state_size", 128);
        w.kv_u32("qwen35moe.ssm.inner_size", 4096);
        w.kv_u32("qwen35moe.ssm.group_count", 16);
        w.kv_u32("qwen35moe.ssm.time_step_rank", 32);
        w.kv_u32("qwen35moe.expert_count", 256);
        w.kv_u32("qwen35moe.expert_used_count", 8);
        let meta = parse(&w.finish()).unwrap();
        assert_eq!(meta.nextn_predict_layers, Some(1));
        assert_eq!(meta.full_attention_interval, Some(4));
        assert_eq!(meta.value_length, Some(128));
        assert_eq!(meta.ssm_conv_kernel, Some(4));
        assert_eq!(meta.ssm_state_size, Some(128));
        assert_eq!(meta.ssm_inner_size, Some(4096));
        assert_eq!(meta.ssm_group_count, Some(16));
        assert_eq!(meta.ssm_time_step_rank, Some(32));
        assert_eq!(meta.expert_count, Some(256));
        assert_eq!(meta.expert_used_count, Some(8));
        assert_eq!(meta.head_count_kv_layers, None);
    }

    #[test]
    fn per_layer_kv_array_is_kept_and_oversized_array_is_skipped() {
        let mut small = GgufWriter::new();
        small.kv_str("general.architecture", "hyb");
        small.kv_raw_array_header("hyb.attention.head_count_kv", 4, 2);
        small.raw(&[0, 0, 0, 0, 8, 0, 0, 0]);
        let meta = parse(&small.finish()).unwrap();
        assert_eq!(meta.head_count_kv_layers, Some(vec![0, 8]));

        // Over the cap: the bytes are consumed and the array is dropped, not kept.
        let mut big = GgufWriter::new();
        big.kv_str("general.architecture", "hyb");
        let n = MAX_LAYER_ARRAY + 1;
        big.kv_raw_array_header("hyb.attention.head_count_kv", 4, n);
        big.raw(&vec![1u8; (n * 4) as usize]);
        big.kv_u32("hyb.block_count", 3);
        let meta = parse(&big.finish()).unwrap();
        assert_eq!(meta.head_count_kv_layers, None);
        assert_eq!(meta.block_count, Some(3));
    }

    #[test]
    fn negative_per_layer_kv_entry_drops_the_array() {
        let mut w = GgufWriter::new();
        w.kv_str("general.architecture", "hyb");
        w.kv_raw_array_header("hyb.attention.head_count_kv", 5, 2);
        w.raw(&[8, 0, 0, 0, 0xff, 0xff, 0xff, 0xff]);
        let meta = parse(&w.finish()).unwrap();
        assert_eq!(meta.head_count_kv_layers, None);
    }

    #[test]
    fn truncated_per_layer_kv_array_is_an_error_not_a_panic() {
        let mut w = GgufWriter::new();
        w.kv_str("general.architecture", "hyb");
        w.kv_raw_array_header("hyb.attention.head_count_kv", 4, u64::MAX / 2);
        assert!(parse(&w.finish()).is_err());
    }

    #[test]
    fn parse_file_rejects_non_gguf() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("fake.gguf");
        std::fs::write(&p, b"<html>not a model</html>").unwrap();
        assert!(parse_file(&p).is_err());
    }
}
