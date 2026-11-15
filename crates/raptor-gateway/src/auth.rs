//! API key authentication.
//!
//! Keys are stored only as `SHA-256(salt || secret)` with a per-key random
//! salt. Presented keys are hashed the same way and compared in constant time
//! against every stored record (no early exit), so timing does not reveal which
//! key, or how many leading bytes, matched. Key ids are labels and may be
//! logged; secrets never are.

use std::collections::HashSet;
use std::fmt;
use std::net::IpAddr;
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::http::{header, HeaderMap};
use raptor_core::{RaptorError, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

use crate::error::ApiError;

const SALT_LEN: usize = 16;
const DIGEST_LEN: usize = 32;
const MIN_SECRET_LEN: usize = 16;
const MAX_SECRET_LEN: usize = 256;
const MAX_LABEL_LEN: usize = 128;
/// Prefix that makes generated keys recognisable in config files and secret scanners.
pub const KEY_PREFIX: &str = "nxk_";

/// Per-key limits. They hook into the same accounting as the queue's per-client
/// limits, under the client identity `key:<id>`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct KeyLimits {
    /// Running plus waiting requests this key may hold. `None` uses the queue's
    /// `per_client_limit`. Must be at least 1 when set.
    pub max_concurrent: Option<usize>,
    /// Requests this key may start in any rolling 60 seconds. `None` is
    /// unlimited. Must be at least 1 when set.
    pub requests_per_minute: Option<u32>,
}

impl KeyLimits {
    fn validate(&self) -> Result<()> {
        if self.max_concurrent == Some(0) || self.requests_per_minute == Some(0) {
            return Err(RaptorError::InvalidInput(
                "per-key limits must be at least 1 when set".into(),
            ));
        }
        Ok(())
    }
}

/// A stored API key: an id label plus a salted digest of the secret.
///
/// Persist it with serde (or [`StoredApiKey`]). The serialized form holds the
/// id, label, creation time, limits, salt and digest, never the secret, so a
/// settings file cannot authenticate on its own.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "StoredApiKey", into = "StoredApiKey")]
pub struct ApiKeyRecord {
    id: String,
    label: String,
    created_unix: u64,
    limits: KeyLimits,
    salt: [u8; SALT_LEN],
    digest: [u8; DIGEST_LEN],
}

/// Serialized form of an [`ApiKeyRecord`]: `{"key_id", "salt", "digest"}` with
/// salt and digest as lowercase hex, plus optional metadata. Contains no secret.
///
/// The metadata fields are optional so records stored by older versions still load.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoredApiKey {
    pub key_id: String,
    pub salt: String,
    pub digest: String,
    /// Human label. Defaults to the id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// Creation time, seconds since the Unix epoch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_unix: Option<u64>,
    /// Per-key concurrency limit. See [`KeyLimits::max_concurrent`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_concurrent: Option<usize>,
    /// Per-key request rate. See [`KeyLimits::requests_per_minute`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requests_per_minute: Option<u32>,
}

impl ApiKeyRecord {
    /// Hashes `secret` under a fresh random salt. The plaintext is not kept.
    ///
    /// Errors when the id is not a short `[A-Za-z0-9_-]` label, or the secret
    /// is outside 16..=256 printable non-space ASCII characters.
    pub fn new(id: impl Into<String>, secret: &str) -> Result<Self> {
        let id = id.into();
        validate_id(&id)?;
        let ok_secret = (MIN_SECRET_LEN..=MAX_SECRET_LEN).contains(&secret.len())
            && secret.bytes().all(|b| b.is_ascii_graphic());
        if !ok_secret {
            return Err(RaptorError::InvalidInput(
                "api key secret must be 16-256 printable ASCII characters".into(),
            ));
        }
        let mut salt = [0u8; SALT_LEN];
        getrandom::getrandom(&mut salt)
            .map_err(|e| RaptorError::Internal(format!("rng failure: {e}")))?;
        let digest = digest_of(&salt, secret);
        Ok(Self {
            label: id.clone(),
            id,
            created_unix: now_unix(),
            limits: KeyLimits::default(),
            salt,
            digest,
        })
    }

    /// Generates a new random secret and returns it together with its record.
    /// The caller must show the secret to the user once; it cannot be recovered.
    pub fn generate(id: impl Into<String>) -> Result<(Self, String)> {
        let secret = generate_secret()?;
        let record = Self::new(id, &secret)?;
        Ok((record, secret))
    }

    /// Sets the human label (1-128 characters, no control characters).
    pub fn with_label(mut self, label: impl Into<String>) -> Result<Self> {
        let label = label.into();
        if label.is_empty() || label.len() > MAX_LABEL_LEN || label.chars().any(char::is_control) {
            return Err(RaptorError::InvalidInput(
                "api key label must be 1-128 characters without control characters".into(),
            ));
        }
        self.label = label;
        Ok(self)
    }

    /// Sets the per-key limits.
    pub fn with_limits(mut self, limits: KeyLimits) -> Result<Self> {
        limits.validate()?;
        self.limits = limits;
        Ok(self)
    }

    /// Rebuilds a record from its stored form. Validates the id, and that the
    /// salt and digest decode to 16 and 32 bytes. The secret is not needed.
    pub fn from_stored(stored: StoredApiKey) -> Result<Self> {
        validate_id(&stored.key_id)?;
        let salt: [u8; SALT_LEN] = decode_fixed(&stored.salt, "salt")?;
        let digest: [u8; DIGEST_LEN] = decode_fixed(&stored.digest, "digest")?;
        let limits = KeyLimits {
            max_concurrent: stored.max_concurrent,
            requests_per_minute: stored.requests_per_minute,
        };
        limits.validate()?;
        let label = stored.label.unwrap_or_else(|| stored.key_id.clone());
        let record = Self {
            id: stored.key_id,
            label: String::new(),
            created_unix: stored.created_unix.unwrap_or(0),
            limits,
            salt,
            digest,
        };
        record.with_label(label)
    }

    /// The serialized form: id, label, creation time, limits, salt and digest.
    /// Never the secret.
    pub fn to_stored(&self) -> StoredApiKey {
        StoredApiKey {
            key_id: self.id.clone(),
            salt: hex::encode(self.salt),
            digest: hex::encode(self.digest),
            label: Some(self.label.clone()),
            created_unix: Some(self.created_unix),
            max_concurrent: self.limits.max_concurrent,
            requests_per_minute: self.limits.requests_per_minute,
        }
    }

    /// True when `presented` is the secret this record was made from. The
    /// comparison is constant time over the digest.
    pub fn verify(&self, presented: &str) -> bool {
        let digest = digest_of(&self.salt, presented);
        let eq: bool = digest.ct_eq(&self.digest).into();
        eq
    }

    /// The non-secret id of this key.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// The human label.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Creation time, seconds since the Unix epoch. `0` for records that
    /// predate the field.
    pub fn created_at(&self) -> u64 {
        self.created_unix
    }

    /// The per-key limits.
    pub fn limits(&self) -> KeyLimits {
        self.limits
    }
}

impl From<ApiKeyRecord> for StoredApiKey {
    fn from(rec: ApiKeyRecord) -> Self {
        rec.to_stored()
    }
}

impl TryFrom<StoredApiKey> for ApiKeyRecord {
    type Error = RaptorError;
    fn try_from(stored: StoredApiKey) -> Result<Self> {
        Self::from_stored(stored)
    }
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Non-secret description of a stored key, for listings and the UI.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct KeyInfo {
    pub id: String,
    pub label: String,
    /// Seconds since the Unix epoch.
    pub created_at: u64,
    pub limits: KeyLimits,
}

/// The live set of API keys, shared between the listener and its owner.
///
/// Clones share one set. Adding, revoking or replacing keys takes effect for the
/// next request, with no listener restart. Requests already in flight finish.
#[derive(Clone, Default)]
pub struct KeyStore {
    inner: Arc<RwLock<Vec<ApiKeyRecord>>>,
}

impl KeyStore {
    /// Creates a store. Rejects duplicate ids, because revocation is by id.
    pub fn new(keys: Vec<ApiKeyRecord>) -> Result<Self> {
        check_unique(&keys)?;
        Ok(Self::from_vec(keys))
    }

    /// Creates a store without the duplicate check. Duplicates do not weaken
    /// authentication, and this keeps [`Authenticator::new`] infallible.
    pub(crate) fn from_vec(keys: Vec<ApiKeyRecord>) -> Self {
        Self {
            inner: Arc::new(RwLock::new(keys)),
        }
    }

    /// Adds a key. Errors with `Conflict` when the id is already in use.
    pub fn add(&self, record: ApiKeyRecord) -> Result<()> {
        let mut keys = self.write();
        if keys.iter().any(|k| k.id == record.id) {
            return Err(RaptorError::Conflict(format!(
                "api key id '{}' is already in use",
                record.id
            )));
        }
        keys.push(record);
        Ok(())
    }

    /// Revokes the key with `id`. Returns `false` when no such key exists.
    pub fn revoke(&self, id: &str) -> bool {
        let mut keys = self.write();
        let before = keys.len();
        keys.retain(|k| k.id != id);
        keys.len() != before
    }

    /// Atomically replaces every key. An invalid set (duplicate ids) leaves the
    /// current keys in place.
    pub fn replace(&self, keys: Vec<ApiKeyRecord>) -> Result<()> {
        check_unique(&keys)?;
        *self.write() = keys;
        Ok(())
    }

    /// Non-secret descriptions of the current keys.
    pub fn list(&self) -> Vec<KeyInfo> {
        self.read()
            .iter()
            .map(|k| KeyInfo {
                id: k.id.clone(),
                label: k.label.clone(),
                created_at: k.created_unix,
                limits: k.limits,
            })
            .collect()
    }

    /// Number of keys.
    pub fn len(&self) -> usize {
        self.read().len()
    }

    /// True when no keys are stored.
    pub fn is_empty(&self) -> bool {
        self.read().is_empty()
    }

    /// Checks `presented` against every key without exiting early, so timing
    /// does not show which key matched. Returns the matching key's id and limits.
    fn match_key(&self, presented: &str) -> Option<(String, KeyLimits)> {
        let keys = self.read();
        let mut found = None;
        for rec in keys.iter() {
            if rec.verify(presented) && found.is_none() {
                found = Some((rec.id.clone(), rec.limits));
            }
        }
        found
    }

    fn read(&self) -> RwLockReadGuard<'_, Vec<ApiKeyRecord>> {
        // A poisoned lock still holds a whole key list; keep serving.
        self.inner.read().unwrap_or_else(|e| e.into_inner())
    }

    fn write(&self) -> RwLockWriteGuard<'_, Vec<ApiKeyRecord>> {
        self.inner.write().unwrap_or_else(|e| e.into_inner())
    }
}

impl fmt::Debug for KeyStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KeyStore")
            .field("keys", &self.len())
            .finish()
    }
}

fn check_unique(keys: &[ApiKeyRecord]) -> Result<()> {
    let mut seen = HashSet::new();
    for k in keys {
        if !seen.insert(k.id.as_str()) {
            return Err(RaptorError::InvalidInput(format!(
                "duplicate api key id '{}'",
                k.id
            )));
        }
    }
    Ok(())
}

fn validate_id(id: &str) -> Result<()> {
    if id.is_empty()
        || id.len() > 64
        || !id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return Err(RaptorError::InvalidInput(
            "api key id must be 1-64 chars of [A-Za-z0-9_-]".into(),
        ));
    }
    Ok(())
}

fn decode_fixed<const N: usize>(hex_str: &str, what: &str) -> Result<[u8; N]> {
    let bytes = hex::decode(hex_str)
        .map_err(|e| RaptorError::Integrity(format!("api key {what} is not hex: {e}")))?;
    bytes
        .try_into()
        .map_err(|_| RaptorError::Integrity(format!("api key {what} must be {N} bytes")))
}

impl fmt::Debug for ApiKeyRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ApiKeyRecord")
            .field("id", &self.id)
            .field("secret", &"<redacted>")
            .finish()
    }
}

/// Generates a 256-bit random secret of the form `nxk_<64 hex chars>`.
pub fn generate_secret() -> Result<String> {
    let mut raw = [0u8; 32];
    getrandom::getrandom(&mut raw)
        .map_err(|e| RaptorError::Internal(format!("rng failure: {e}")))?;
    Ok(format!("{KEY_PREFIX}{}", hex::encode(raw)))
}

fn digest_of(salt: &[u8; SALT_LEN], secret: &str) -> [u8; DIGEST_LEN] {
    let mut h = Sha256::new();
    h.update(salt);
    h.update(secret.as_bytes());
    h.finalize().into()
}

/// Which access rules apply to a listener.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AccessPolicy {
    /// Loopback peers may skip the key when no key is presented.
    pub allow_keyless_loopback: bool,
}

/// Identity of an authenticated request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Principal {
    /// Stable identity used for per-client queue limits: `key:<id>` or `local:<ip>`.
    pub client_id: String,
    /// True when a stored API key matched.
    pub authenticated: bool,
    /// Limits of the matched key. Defaults for keyless loopback.
    pub limits: KeyLimits,
}

/// Checks requests against the configured keys and access policy.
#[derive(Clone)]
pub struct Authenticator {
    keys: KeyStore,
    policy: AccessPolicy,
}

impl Authenticator {
    /// Builds an authenticator over `keys` with the given policy.
    pub fn new(keys: Vec<ApiKeyRecord>, policy: AccessPolicy) -> Self {
        Self {
            keys: KeyStore::from_vec(keys),
            policy,
        }
    }

    /// Builds an authenticator over a shared, live key store.
    pub fn with_store(keys: KeyStore, policy: AccessPolicy) -> Self {
        Self { keys, policy }
    }

    /// Decides whether a request may proceed.
    ///
    /// * A presented key (`Authorization: Bearer` or `x-api-key`) must match a
    ///   stored key, or the request is refused even from loopback.
    /// * With no key presented, loopback peers pass when keyless loopback is
    ///   allowed, but only with a loopback `Host` header. A web page that
    ///   DNS-rebinds its own hostname to 127.0.0.1 reaches the gateway as a
    ///   loopback peer, and the foreign `Host` is what exposes it. Every other
    ///   keyless request gets 401.
    pub fn authorize(
        &self,
        peer: Option<IpAddr>,
        headers: &HeaderMap,
    ) -> std::result::Result<Principal, ApiError> {
        if let Some(presented) = presented_secret(headers) {
            return match self.keys.match_key(presented) {
                Some((id, limits)) => Ok(Principal {
                    client_id: format!("key:{id}"),
                    authenticated: true,
                    limits,
                }),
                None => Err(ApiError::unauthorized("Invalid API key.")),
            };
        }
        match peer {
            Some(ip) if self.policy.allow_keyless_loopback && ip.to_canonical().is_loopback() => {
                if !host_is_local(headers) {
                    return Err(ApiError::forbidden("Host header is not a loopback name"));
                }
                Ok(Principal {
                    client_id: format!("local:{ip}"),
                    authenticated: false,
                    limits: KeyLimits::default(),
                })
            }
            _ => Err(ApiError::unauthorized(
                "An API key is required for this request.",
            )),
        }
    }
}

/// Extracts the presented secret. An `Authorization` header with a non-Bearer
/// scheme counts as a presented (and therefore invalid) credential.
fn presented_secret(headers: &HeaderMap) -> Option<&str> {
    if let Some(v) = headers.get(header::AUTHORIZATION) {
        let s = v.to_str().unwrap_or("").trim();
        return Some(match s.split_once(' ') {
            Some((scheme, tok)) if scheme.eq_ignore_ascii_case("bearer") => tok.trim(),
            _ => "",
        });
    }
    headers
        .get("x-api-key")
        .map(|v| v.to_str().unwrap_or("").trim())
}

/// True when the `Host` header names a loopback host, or is absent (HTTP/1.0).
pub fn host_is_local(headers: &HeaderMap) -> bool {
    let Some(v) = headers.get(header::HOST) else {
        return true;
    };
    let Ok(s) = v.to_str() else {
        return false;
    };
    let host = match s.strip_prefix('[') {
        Some(rest) => rest.split(']').next().unwrap_or(""),
        None => s.rsplit_once(':').map(|(h, _)| h).unwrap_or(s),
    };
    matches!(
        host.to_ascii_lowercase().as_str(),
        "localhost" | "127.0.0.1" | "::1"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    const SECRET: &str = "nxk_test_secret_0123456789";
    const LOCAL: IpAddr = IpAddr::V4(std::net::Ipv4Addr::LOCALHOST);

    fn lan_ip() -> IpAddr {
        IpAddr::V4(std::net::Ipv4Addr::new(192, 168, 1, 20))
    }

    fn policy() -> AccessPolicy {
        AccessPolicy {
            allow_keyless_loopback: true,
        }
    }

    fn with_header(name: &'static str, value: &str) -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert(name, HeaderValue::from_str(value).unwrap());
        h
    }

    #[test]
    fn stores_only_salted_digest() {
        let rec = ApiKeyRecord::new("ci", SECRET).unwrap();
        assert_ne!(rec.digest.as_slice(), SECRET.as_bytes());
        let dbg = format!("{rec:?}");
        assert!(!dbg.contains(SECRET));
        assert!(dbg.contains("<redacted>"));
    }

    #[test]
    fn rejects_bad_ids_and_secrets() {
        assert!(ApiKeyRecord::new("", SECRET).is_err());
        assert!(ApiKeyRecord::new("has space", SECRET).is_err());
        assert!(ApiKeyRecord::new("ok", "short").is_err());
        assert!(ApiKeyRecord::new("ok", "has space in the secret 1234").is_err());
    }

    #[test]
    fn generated_secrets_are_prefixed_and_unique() {
        let a = generate_secret().unwrap();
        let b = generate_secret().unwrap();
        assert!(a.starts_with(KEY_PREFIX));
        assert_eq!(a.len(), KEY_PREFIX.len() + 64);
        assert_ne!(a, b);
    }

    #[test]
    fn bearer_key_authenticates_from_lan() {
        let rec = ApiKeyRecord::new("laptop", SECRET).unwrap();
        let auth = Authenticator::new(vec![rec], policy());
        let p = auth
            .authorize(
                Some(lan_ip()),
                &with_header("authorization", &format!("Bearer {SECRET}")),
            )
            .unwrap();
        assert_eq!(p.client_id, "key:laptop");
        assert!(p.authenticated);
    }

    #[test]
    fn bearer_scheme_is_case_insensitive_and_x_api_key_works() {
        let rec = ApiKeyRecord::new("k", SECRET).unwrap();
        let auth = Authenticator::new(vec![rec], policy());
        assert!(auth
            .authorize(
                None,
                &with_header("authorization", &format!("BEARER {SECRET}"))
            )
            .is_ok());
        assert!(auth
            .authorize(None, &with_header("x-api-key", SECRET))
            .is_ok());
    }

    #[test]
    fn wrong_key_is_refused_even_on_loopback() {
        let rec = ApiKeyRecord::new("k", SECRET).unwrap();
        let auth = Authenticator::new(vec![rec], policy());
        let err = auth
            .authorize(
                Some(LOCAL),
                &with_header("authorization", "Bearer nxk_wrong_wrong_wrong_1"),
            )
            .unwrap_err();
        assert_eq!(err.status, axum::http::StatusCode::UNAUTHORIZED);
    }

    #[test]
    fn any_of_several_keys_matches() {
        let a = ApiKeyRecord::new("a", "aaaaaaaaaaaaaaaa1").unwrap();
        let b = ApiKeyRecord::new("b", "bbbbbbbbbbbbbbbb2").unwrap();
        let auth = Authenticator::new(vec![a, b], policy());
        let p = auth
            .authorize(None, &with_header("x-api-key", "bbbbbbbbbbbbbbbb2"))
            .unwrap();
        assert_eq!(p.client_id, "key:b");
    }

    #[test]
    fn keyless_loopback_allowed_but_remote_needs_key() {
        let auth = Authenticator::new(vec![], policy());
        assert!(auth.authorize(Some(LOCAL), &HeaderMap::new()).is_ok());
        let err = auth
            .authorize(Some(lan_ip()), &HeaderMap::new())
            .unwrap_err();
        assert_eq!(err.status, axum::http::StatusCode::UNAUTHORIZED);
        // No ConnectInfo at all is treated as remote.
        assert!(auth.authorize(None, &HeaderMap::new()).is_err());
    }

    #[test]
    fn keyless_loopback_can_be_disabled() {
        let auth = Authenticator::new(
            vec![],
            AccessPolicy {
                allow_keyless_loopback: false,
            },
        );
        assert!(auth.authorize(Some(LOCAL), &HeaderMap::new()).is_err());
    }

    #[test]
    fn non_bearer_authorization_is_invalid_not_anonymous() {
        let auth = Authenticator::new(vec![], policy());
        let err = auth
            .authorize(Some(LOCAL), &with_header("authorization", "Basic abc"))
            .unwrap_err();
        assert_eq!(err.status, axum::http::StatusCode::UNAUTHORIZED);
    }

    #[test]
    fn keyless_loopback_requires_loopback_host_header() {
        let rec = ApiKeyRecord::new("k", SECRET).unwrap();
        let auth = Authenticator::new(vec![rec], policy());
        // A DNS-rebinding page reaches us as loopback with a foreign Host.
        let evil = auth.authorize(Some(LOCAL), &with_header("host", "evil.example:8080"));
        assert_eq!(evil.unwrap_err().status, axum::http::StatusCode::FORBIDDEN);
        // A client that presents a valid key is authenticated by that key.
        let keyed = auth.authorize(Some(lan_ip()), &{
            let mut h = with_header("host", "box.lan:8790");
            h.insert("x-api-key", HeaderValue::from_static(SECRET));
            h
        });
        assert!(keyed.is_ok());
        assert!(auth
            .authorize(Some(LOCAL), &with_header("host", "localhost:8080"))
            .is_ok());
        assert!(auth
            .authorize(Some(LOCAL), &with_header("host", "[::1]:8080"))
            .is_ok());
        assert!(auth
            .authorize(Some(LOCAL), &with_header("host", "127.0.0.1:1"))
            .is_ok());
        assert!(auth.authorize(Some(LOCAL), &HeaderMap::new()).is_ok());
    }

    #[test]
    fn persisted_form_holds_no_secret_and_reloads_to_a_verifying_key() {
        let rec = ApiKeyRecord::new("laptop", SECRET).unwrap();
        let json = serde_json::to_string(&rec).unwrap();
        assert!(!json.contains(SECRET), "secret leaked into {json}");
        assert!(json.contains("\"key_id\":\"laptop\""));
        let reloaded: ApiKeyRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(reloaded, rec);
        assert!(reloaded.verify(SECRET));
        assert!(!reloaded.verify("nxk_wrong_wrong_wrong_1"));
        assert!(!reloaded.verify(&SECRET[..SECRET.len() - 1]));
    }

    #[test]
    fn stored_form_is_id_salt_and_digest_only() {
        let rec = ApiKeyRecord::new("ci", SECRET).unwrap();
        let stored = rec.to_stored();
        assert_eq!(stored.key_id, "ci");
        assert_eq!(stored.salt.len(), SALT_LEN * 2);
        assert_eq!(stored.digest.len(), DIGEST_LEN * 2);
        let back = ApiKeyRecord::from_stored(stored.clone()).unwrap();
        assert!(back.verify(SECRET));
        let v = serde_json::to_value(&stored).unwrap();
        let mut keys: Vec<&str> = v.as_object().unwrap().keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            ["created_unix", "digest", "key_id", "label", "salt"],
            "no secret-bearing field may appear"
        );
    }

    #[test]
    fn legacy_stored_form_loads_with_defaults() {
        let legacy = r#"{"key_id":"old","salt":"00000000000000000000000000000000","digest":"0000000000000000000000000000000000000000000000000000000000000000"}"#;
        let rec: ApiKeyRecord = serde_json::from_str(legacy).unwrap();
        assert_eq!(rec.label(), "old");
        assert_eq!(rec.created_at(), 0);
        assert_eq!(rec.limits(), KeyLimits::default());
    }

    #[test]
    fn labels_and_limits_are_validated() {
        let rec = ApiKeyRecord::new("ci", SECRET).unwrap();
        assert!(rec.clone().with_label("").is_err());
        assert!(rec.clone().with_label("bad\nlabel").is_err());
        assert!(rec.clone().with_label("x".repeat(129)).is_err());
        let ok = rec.clone().with_label("Build bot").unwrap();
        assert_eq!(ok.label(), "Build bot");
        let zero = KeyLimits {
            max_concurrent: Some(0),
            requests_per_minute: None,
        };
        assert!(rec.clone().with_limits(zero).is_err());
        let limited = rec
            .with_limits(KeyLimits {
                max_concurrent: Some(2),
                requests_per_minute: Some(30),
            })
            .unwrap();
        let back: ApiKeyRecord =
            serde_json::from_str(&serde_json::to_string(&limited).unwrap()).unwrap();
        assert_eq!(back.limits(), limited.limits());
        assert_eq!(back.label(), "ci");
    }

    #[test]
    fn keys_can_be_added_revoked_and_replaced_live() {
        let store =
            KeyStore::new(vec![ApiKeyRecord::new("a", "aaaaaaaaaaaaaaaa1").unwrap()]).unwrap();
        let auth = Authenticator::with_store(store.clone(), policy());
        let hdr = |s: &str| with_header("x-api-key", s);
        assert!(auth.authorize(None, &hdr("aaaaaaaaaaaaaaaa1")).is_ok());

        store
            .add(ApiKeyRecord::new("b", "bbbbbbbbbbbbbbbb2").unwrap())
            .unwrap();
        assert!(auth.authorize(None, &hdr("bbbbbbbbbbbbbbbb2")).is_ok());
        assert!(store
            .add(ApiKeyRecord::new("b", "cccccccccccccccc3").unwrap())
            .is_err());

        assert!(store.revoke("a"));
        assert!(!store.revoke("a"));
        assert!(auth.authorize(None, &hdr("aaaaaaaaaaaaaaaa1")).is_err());
        assert!(auth.authorize(None, &hdr("bbbbbbbbbbbbbbbb2")).is_ok());

        let listed = store.list();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, "b");
        assert!(!format!("{listed:?}").contains("bbbbbbbbbbbbbbbb2"));

        store
            .replace(vec![ApiKeyRecord::new("c", "cccccccccccccccc3").unwrap()])
            .unwrap();
        assert!(auth.authorize(None, &hdr("bbbbbbbbbbbbbbbb2")).is_err());
        assert!(auth.authorize(None, &hdr("cccccccccccccccc3")).is_ok());
    }

    #[test]
    fn invalid_replacement_keeps_current_keys() {
        let store = KeyStore::new(vec![ApiKeyRecord::new("a", SECRET).unwrap()]).unwrap();
        let dup = vec![
            ApiKeyRecord::new("x", "xxxxxxxxxxxxxxxx1").unwrap(),
            ApiKeyRecord::new("x", "xxxxxxxxxxxxxxxx2").unwrap(),
        ];
        assert!(store.replace(dup).is_err());
        assert_eq!(store.list()[0].id, "a");
        assert!(KeyStore::new(vec![]).unwrap().is_empty());
    }

    #[test]
    fn matched_key_carries_its_limits() {
        let rec = ApiKeyRecord::new("k", SECRET)
            .unwrap()
            .with_limits(KeyLimits {
                max_concurrent: Some(1),
                requests_per_minute: Some(5),
            })
            .unwrap();
        let auth = Authenticator::new(vec![rec], policy());
        let p = auth
            .authorize(None, &with_header("x-api-key", SECRET))
            .unwrap();
        assert_eq!(p.limits.max_concurrent, Some(1));
        assert_eq!(p.limits.requests_per_minute, Some(5));
    }

    #[test]
    fn two_records_for_one_secret_use_different_salts() {
        let a = ApiKeyRecord::new("a", SECRET).unwrap();
        let b = ApiKeyRecord::new("a", SECRET).unwrap();
        assert_ne!(a.to_stored().salt, b.to_stored().salt);
        assert_ne!(a.to_stored().digest, b.to_stored().digest);
        assert!(a.verify(SECRET) && b.verify(SECRET));
    }

    #[test]
    fn generated_record_verifies_its_returned_secret_once() {
        let (rec, secret) = ApiKeyRecord::generate("new-phone").unwrap();
        assert!(secret.starts_with(KEY_PREFIX));
        assert!(rec.verify(&secret));
        let stored = serde_json::to_string(&rec).unwrap();
        assert!(!stored.contains(&secret));
        let auth = Authenticator::new(vec![serde_json::from_str(&stored).unwrap()], policy());
        let p = auth
            .authorize(None, &with_header("x-api-key", &secret))
            .unwrap();
        assert_eq!(p.client_id, "key:new-phone");
    }

    #[test]
    fn malformed_stored_records_are_rejected() {
        let good = ApiKeyRecord::new("k", SECRET).unwrap().to_stored();
        let bad_id = StoredApiKey {
            key_id: "has space".into(),
            ..good.clone()
        };
        let bad_hex = StoredApiKey {
            salt: "zz".repeat(SALT_LEN),
            ..good.clone()
        };
        let short_digest = StoredApiKey {
            digest: "ab".repeat(DIGEST_LEN - 1),
            ..good.clone()
        };
        let short_salt = StoredApiKey {
            salt: "ab".repeat(SALT_LEN - 1),
            ..good
        };
        for bad in [bad_id, bad_hex, short_digest, short_salt] {
            assert!(ApiKeyRecord::from_stored(bad.clone()).is_err(), "{bad:?}");
        }
        let json = r#"{"key_id":"k","salt":"00","digest":"00"}"#;
        assert!(serde_json::from_str::<ApiKeyRecord>(json).is_err());
    }

    #[test]
    fn mapped_ipv4_loopback_counts_as_loopback() {
        let auth = Authenticator::new(vec![], policy());
        let mapped: IpAddr = "::ffff:127.0.0.1".parse().unwrap();
        assert!(auth.authorize(Some(mapped), &HeaderMap::new()).is_ok());
    }
}
