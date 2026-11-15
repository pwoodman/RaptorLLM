//! Provenance: why every item was included or excluded, and how the budget was used.

use crate::budget::Category;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Inclusive 1-based line range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LineRange {
    pub start: u32,
    pub end: u32,
}

/// An item that is in the returned context.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ManifestItem {
    /// Stable identifier, e.g. `chunk:src/a.rs:10-40` or `history:3`.
    pub id: String,
    /// Human-readable source: a path, `task`, `history turn 3`, ...
    pub source: String,
    pub category: Category,
    /// Content hash of the source file, when the item comes from a file.
    pub hash: Option<String>,
    pub range: Option<LineRange>,
    /// Tokens the item occupies in the returned context.
    pub tokens: usize,
    pub score: f64,
    /// Why the item is in the context.
    pub why: String,
    /// True when the item was elided to fit; the omitted lines are marked in the text.
    pub truncated: bool,
}

/// An item that was considered and left out, with the reason and a way to get it back.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExcludedItem {
    pub id: String,
    pub source: String,
    pub category: Category,
    /// Tokens the full item would have needed.
    pub tokens: usize,
    /// Why the item is out. Never empty.
    pub why: String,
    /// Pass to `recover` to fetch the item later, when it can be fetched.
    pub recover_handle: Option<String>,
}

/// Token use of one category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CategoryUsage {
    pub used: usize,
    /// Cap for capped categories. `None` for protected categories, which have no cap.
    pub budget: Option<usize>,
}

/// The full account of one context build.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextManifest {
    /// Tokens available after the reserved output.
    pub budget_tokens: usize,
    /// Tokens used by the returned context.
    pub used_tokens: usize,
    /// Protected tokens beyond the budget. Non-zero means the context is over budget.
    pub over_budget_by: usize,
    /// True when token counts come from the model's tokenizer, false for estimates.
    pub tokens_exact: bool,
    pub included: Vec<ManifestItem>,
    pub excluded: Vec<ExcludedItem>,
    /// Every category, always present, so consumers see a stable table.
    pub utilization_by_category: BTreeMap<Category, CategoryUsage>,
    /// Conditions the caller should look at: over-budget, stale memory, git failures, loops.
    pub warnings: Vec<String>,
}

impl ContextManifest {
    /// True when every included and excluded item carries a non-empty reason.
    pub fn is_fully_explained(&self) -> bool {
        self.included.iter().all(|i| !i.why.trim().is_empty())
            && self.excluded.iter().all(|e| !e.why.trim().is_empty())
    }
}
