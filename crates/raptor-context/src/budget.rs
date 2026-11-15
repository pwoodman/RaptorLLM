//! Context categories, task kinds and budget allocation.
//!
//! Protected categories (instructions, task, active files, diffs, errors) are never
//! capped. Everything else shares the remaining pool according to the task kind.

use raptor_core::{RaptorError, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Where a context item comes from. Each category is a line in the manifest's utilisation table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Category {
    /// System prompt and system messages. Protected.
    Instructions,
    /// The active task statement. Protected.
    Task,
    /// Files being edited, in full. Protected.
    ActiveFiles,
    /// Caller-supplied diffs. Protected.
    Diffs,
    /// Failing tests and unresolved errors. Protected.
    ErrorsTests,
    /// Conversation memory. Decisions and conventions are protected, summaries are not.
    Memory,
    /// Retrieved code chunks.
    RetrievedCode,
    /// Signature views of files adjacent to active files in the graph.
    GraphNeighbours,
    /// Earlier conversation turns.
    History,
}

impl Category {
    /// Every category, in rendering order.
    pub const ALL: [Category; 9] = [
        Category::Instructions,
        Category::Task,
        Category::ActiveFiles,
        Category::Diffs,
        Category::ErrorsTests,
        Category::Memory,
        Category::RetrievedCode,
        Category::GraphNeighbours,
        Category::History,
    ];

    /// Section heading used when the context is rendered into a system message.
    pub fn heading(self) -> &'static str {
        match self {
            Category::Instructions => "Instructions",
            Category::Task => "Task",
            Category::ActiveFiles => "Active files",
            Category::Diffs => "Diffs",
            Category::ErrorsTests => "Errors and failing tests",
            Category::Memory => "Memory",
            Category::RetrievedCode => "Retrieved code",
            Category::GraphNeighbours => "Related code (graph)",
            Category::History => "History",
        }
    }
}

/// The kind of work the agent is doing. Drives retrieval weights and budget shares.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskKind {
    Debug,
    Architecture,
    Refactor,
    #[default]
    Feature,
    Ui,
    Security,
    Performance,
}

/// Share of the unprotected pool per category, in permille. Each row sums to 1000.
pub fn shares(kind: TaskKind) -> [(Category, u32); 4] {
    use Category::*;
    match kind {
        TaskKind::Debug => [
            (RetrievedCode, 400),
            (GraphNeighbours, 250),
            (Memory, 100),
            (History, 250),
        ],
        TaskKind::Architecture => [
            (RetrievedCode, 350),
            (GraphNeighbours, 400),
            (Memory, 150),
            (History, 100),
        ],
        TaskKind::Refactor => [
            (RetrievedCode, 350),
            (GraphNeighbours, 350),
            (Memory, 100),
            (History, 200),
        ],
        TaskKind::Feature => [
            (RetrievedCode, 400),
            (GraphNeighbours, 200),
            (Memory, 150),
            (History, 250),
        ],
        TaskKind::Ui => [
            (RetrievedCode, 450),
            (GraphNeighbours, 150),
            (Memory, 150),
            (History, 250),
        ],
        TaskKind::Security => [
            (RetrievedCode, 450),
            (GraphNeighbours, 250),
            (Memory, 100),
            (History, 200),
        ],
        TaskKind::Performance => [
            (RetrievedCode, 450),
            (GraphNeighbours, 200),
            (Memory, 100),
            (History, 250),
        ],
    }
}

/// Per-category token caps for an unprotected `pool`, rounded down.
pub fn allocate(kind: TaskKind, pool: usize) -> BTreeMap<Category, usize> {
    shares(kind)
        .iter()
        .map(|(cat, permille)| (*cat, pool.saturating_mul(*permille as usize) / 1000))
        .collect()
}

/// Tokens available for context: the model window minus the reserved output.
///
/// Errors when nothing is left, since no context could be produced.
pub fn available_tokens(model_ctx: usize, reserved_output: usize) -> Result<usize> {
    let available = model_ctx.saturating_sub(reserved_output);
    if available == 0 {
        return Err(RaptorError::InvalidInput(format!(
            "model context {model_ctx} minus reserved output {reserved_output} leaves no room for context"
        )));
    }
    Ok(available)
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL_KINDS: [TaskKind; 7] = [
        TaskKind::Debug,
        TaskKind::Architecture,
        TaskKind::Refactor,
        TaskKind::Feature,
        TaskKind::Ui,
        TaskKind::Security,
        TaskKind::Performance,
    ];

    #[test]
    fn every_share_row_sums_to_one_thousand() {
        for kind in ALL_KINDS {
            let total: u32 = shares(kind).iter().map(|(_, p)| p).sum();
            assert_eq!(total, 1000, "{kind:?}");
        }
    }

    #[test]
    fn allocation_never_exceeds_pool() {
        for kind in ALL_KINDS {
            for pool in [0usize, 1, 7, 999, 4096] {
                let caps = allocate(kind, pool);
                let sum: usize = caps.values().sum();
                assert!(sum <= pool, "{kind:?} {pool}");
            }
        }
    }

    #[test]
    fn debug_favours_retrieved_code_over_architecture() {
        let debug = allocate(TaskKind::Debug, 1000);
        let arch = allocate(TaskKind::Architecture, 1000);
        assert!(arch[&Category::GraphNeighbours] > debug[&Category::GraphNeighbours]);
        assert_eq!(debug[&Category::RetrievedCode], 400);
    }

    #[test]
    fn allocation_covers_only_capped_categories() {
        let caps = allocate(TaskKind::Feature, 100);
        assert!(!caps.contains_key(&Category::Instructions));
        assert!(!caps.contains_key(&Category::ActiveFiles));
        assert_eq!(caps.len(), 4);
    }

    #[test]
    fn available_tokens_subtracts_output_reserve() {
        assert_eq!(available_tokens(4096, 512).expect("ok"), 3584);
    }

    #[test]
    fn available_tokens_rejects_no_room() {
        assert!(matches!(
            available_tokens(100, 100),
            Err(RaptorError::InvalidInput(_))
        ));
        assert!(matches!(
            available_tokens(10, 500),
            Err(RaptorError::InvalidInput(_))
        ));
    }

    #[test]
    fn every_category_has_a_heading() {
        for c in Category::ALL {
            assert!(!c.heading().is_empty());
        }
    }

    #[test]
    fn default_task_kind_is_feature() {
        assert_eq!(TaskKind::default(), TaskKind::Feature);
    }
}
