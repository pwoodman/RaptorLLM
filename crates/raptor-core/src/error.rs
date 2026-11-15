use serde::Serialize;

pub type Result<T> = std::result::Result<T, RaptorError>;

/// Error type shared across subsystem boundaries. Serializable so the UI and
/// API gateway can show real causes; never swallow errors into success.
#[derive(Debug, Clone, thiserror::Error, Serialize)]
#[serde(tag = "kind", content = "message")]
pub enum RaptorError {
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("conflict: {0}")]
    Conflict(String),
    /// A handle or cached view no longer matches the underlying state (for
    /// example a context id from before a re-index). The caller should refresh
    /// and retry; the gateway maps it to 409 Conflict.
    #[error("stale: {0}")]
    Stale(String),
    #[error("io error: {0}")]
    Io(String),
    #[error("network error: {0}")]
    Network(String),
    #[error("integrity check failed: {0}")]
    Integrity(String),
    #[error("runtime error: {0}")]
    Runtime(String),
    #[error("unsupported: {0}")]
    Unsupported(String),
    #[error("unauthorized: {0}")]
    Unauthorized(String),
    #[error("internal error: {0}")]
    Internal(String),
}

impl From<std::io::Error> for RaptorError {
    fn from(e: std::io::Error) -> Self {
        RaptorError::Io(e.to_string())
    }
}
impl From<serde_json::Error> for RaptorError {
    fn from(e: serde_json::Error) -> Self {
        RaptorError::Internal(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errors_clone_and_stale_serializes_with_its_kind() {
        let e = RaptorError::Stale("context id from before re-index".into());
        let copy = e.clone();
        assert_eq!(copy.to_string(), e.to_string());
        assert_eq!(e.to_string(), "stale: context id from before re-index");
        let json = serde_json::to_value(&e).unwrap();
        assert_eq!(json["kind"], "Stale");
        assert_eq!(json["message"], "context id from before re-index");
    }
}
