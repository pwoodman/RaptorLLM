//! HTTP error responses. Every error leaves the gateway in the shape of the API
//! the client is speaking: OpenAI-style `{"error":{...}}` for everything except
//! the Anthropic `/v1/messages` route, which gets Anthropic's
//! `{"type":"error","error":{...}}` envelope.

use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use raptor_core::RaptorError;
use serde_json::{json, Value};

/// Which wire format an error must be rendered in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    OpenAi,
    Anthropic,
}

/// A failure that is rendered as an HTTP response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApiError {
    pub status: StatusCode,
    /// OpenAI error type (`invalid_request_error`, `server_error`, ...).
    pub kind: &'static str,
    /// Machine-readable code, e.g. `invalid_api_key`, `queue_full`.
    pub code: Option<&'static str>,
    pub message: String,
    /// Seconds for the `Retry-After` header, when the client should retry later.
    pub retry_after: Option<u64>,
    pub shape: Shape,
}

impl ApiError {
    /// Creates an OpenAI-shaped error with no code and no retry hint.
    pub fn new(status: StatusCode, kind: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            kind,
            code: None,
            message: message.into(),
            retry_after: None,
            shape: Shape::OpenAi,
        }
    }

    /// 400 `invalid_request_error`.
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "invalid_request_error", message)
    }

    /// 401 `authentication_error` with code `invalid_api_key`.
    pub fn unauthorized(message: impl Into<String>) -> Self {
        let mut e = Self::new(StatusCode::UNAUTHORIZED, "authentication_error", message);
        e.code = Some("invalid_api_key");
        e
    }

    /// 403 `permission_error`.
    pub fn forbidden(message: impl Into<String>) -> Self {
        Self::new(StatusCode::FORBIDDEN, "permission_error", message)
    }

    /// 404 `invalid_request_error` (OpenAI uses this type for unknown models).
    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new(StatusCode::NOT_FOUND, "invalid_request_error", message)
    }

    /// Sets the machine-readable code.
    pub fn with_code(mut self, code: &'static str) -> Self {
        self.code = Some(code);
        self
    }

    /// Sets the `Retry-After` hint in seconds.
    pub fn with_retry_after(mut self, secs: u64) -> Self {
        self.retry_after = Some(secs);
        self
    }

    /// Switches the rendering shape.
    pub fn with_shape(mut self, shape: Shape) -> Self {
        self.shape = shape;
        self
    }

    /// Maps a shared `RaptorError` to an HTTP status. Upstream and runtime
    /// failures become 502 so clients can tell them apart from gateway bugs.
    pub fn from_raptor(err: RaptorError) -> Self {
        let message = err.to_string();
        match err {
            RaptorError::InvalidInput(_) | RaptorError::Unsupported(_) => {
                Self::bad_request(message)
            }
            RaptorError::NotFound(_) => Self::not_found(message),
            RaptorError::Conflict(_) | RaptorError::Stale(_) => {
                Self::new(StatusCode::CONFLICT, "invalid_request_error", message)
            }
            RaptorError::Unauthorized(_) => Self::unauthorized(message),
            RaptorError::Network(_) | RaptorError::Runtime(_) => {
                let mut e = Self::new(StatusCode::BAD_GATEWAY, "server_error", message);
                e.code = Some("upstream_error");
                e
            }
            RaptorError::Io(_) | RaptorError::Integrity(_) | RaptorError::Internal(_) => {
                Self::new(StatusCode::INTERNAL_SERVER_ERROR, "server_error", message)
            }
        }
    }

    /// Renders the JSON body for this error in its shape.
    pub fn body(&self) -> Value {
        match self.shape {
            Shape::OpenAi => json!({
                "error": {
                    "message": self.message,
                    "type": self.kind,
                    "param": null,
                    "code": self.code,
                }
            }),
            Shape::Anthropic => json!({
                "type": "error",
                "error": {
                    "type": anthropic_kind(self.kind),
                    "message": self.message,
                }
            }),
        }
    }
}

/// Maps an OpenAI error type to the closest Anthropic error type.
fn anthropic_kind(kind: &str) -> &str {
    match kind {
        "server_error" => "api_error",
        other => other,
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = self.body().to_string();
        let mut response = Response::new(axum::body::Body::from(body));
        *response.status_mut() = self.status;
        let headers = response.headers_mut();
        headers.insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );
        if let Some(secs) = self.retry_after {
            if let Ok(v) = HeaderValue::from_str(&secs.to_string()) {
                headers.insert(header::RETRY_AFTER, v);
            }
        }
        if self.status == StatusCode::UNAUTHORIZED {
            headers.insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
        }
        response
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_shape_has_error_object() {
        let e = ApiError::unauthorized("missing key");
        let v = e.body();
        assert_eq!(v["error"]["type"], "authentication_error");
        assert_eq!(v["error"]["code"], "invalid_api_key");
        assert_eq!(v["error"]["message"], "missing key");
    }

    #[test]
    fn anthropic_shape_uses_envelope() {
        let e = ApiError::bad_request("bad").with_shape(Shape::Anthropic);
        let v = e.body();
        assert_eq!(v["type"], "error");
        assert_eq!(v["error"]["type"], "invalid_request_error");
    }

    #[test]
    fn upstream_failures_are_502() {
        let e = ApiError::from_raptor(RaptorError::Runtime("boom".into()));
        assert_eq!(e.status, StatusCode::BAD_GATEWAY);
    }

    #[test]
    fn retry_after_header_is_set() {
        let r = ApiError::new(StatusCode::TOO_MANY_REQUESTS, "rate_limit_error", "slow")
            .with_retry_after(7)
            .into_response();
        assert_eq!(
            r.headers()
                .get(header::RETRY_AFTER)
                .and_then(|v| v.to_str().ok()),
            Some("7")
        );
    }
}
