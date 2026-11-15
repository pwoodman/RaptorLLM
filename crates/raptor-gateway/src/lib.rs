//! raptor-gateway: the local API gateway for Raptor LLM.
//!
//! Serves an OpenAI-compatible API, Anthropic `/v1/messages` (for Claude Code),
//! an MCP endpoint, and `/raptor/context/*`, all in front of an inference
//! [`Upstream`] and an optional [`ContextService`]. Both are traits this crate
//! defines and the desktop shell implements. The crate depends on no other
//! `raptor-*` crate. See `docs/specs/gateway.md` and the crate README.

pub mod adapters;
pub mod auth;
pub mod context_api;
pub mod error;
pub mod hygiene;
pub mod mcp;
pub mod openai;
pub mod queue;
pub mod server;
pub mod thinking;
pub mod tls;

pub use auth::{
    generate_secret, AccessPolicy, ApiKeyRecord, Authenticator, KeyInfo, KeyLimits, KeyStore,
    Principal, StoredApiKey,
};
pub use context_api::{
    BuildRequest, ContextPack, ContextService, ImpactInput, ImpactQuery, ImpactRequest,
    ImpactTarget, RecoverRequest, SearchRequest, SymbolsRequest, WORKSPACE_HEADER,
};
pub use error::ApiError;
pub use openai::{Completion, Endpoint, HttpUpstream, ModelCatalog, SseStream, Upstream};
pub use queue::{Metrics, Permit, QueueConfig, Rejection, Scheduler};
pub use server::{
    build, BindMode, Built, Gateway, GatewayConfig, GatewayDeps, Handle, ListenerStatus,
};
pub use thinking::{level_map, map_effort, Effort, ThinkingCaps};
pub use tls::{display_fingerprint, fingerprint_of_der, TlsConfig, TlsIdentity};
