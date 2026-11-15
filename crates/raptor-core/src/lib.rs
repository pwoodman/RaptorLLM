//! raptor-core: the tiny shared vocabulary every Raptor LLM subsystem depends on.
//! Subsystems must NOT depend on each other directly; they talk through types
//! defined here (or through the API Gateway). Keep this crate small and stable.

pub mod chat;
pub mod error;
pub mod logfiles;
pub mod paths;

pub use chat::{ChatMessage, Role};
pub use error::{RaptorError, Result};
pub use paths::DataDirs;

/// Counts tokens for a specific model. Implemented by the runtime (via llama-server
/// `/tokenize`) or by a fallback estimator; consumed by the context engine.
pub trait TokenCounter: Send + Sync {
    fn count(&self, text: &str) -> usize;
    /// True when counts come from the model's real tokenizer, false for estimates.
    fn is_exact(&self) -> bool;
}

/// Conservative estimator (~3.5 chars/token, rounded up). Never claims exactness.
#[derive(Debug, Default, Clone, Copy)]
pub struct EstimatingCounter;

impl TokenCounter for EstimatingCounter {
    fn count(&self, text: &str) -> usize {
        (text.chars().count() * 2).div_ceil(7)
    }
    fn is_exact(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn estimator_is_inexact_and_nonzero() {
        let c = EstimatingCounter;
        assert!(!c.is_exact());
        assert_eq!(c.count(""), 0);
        assert!(c.count("hello world") > 0);
    }
}
