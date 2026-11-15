//! Thin system boundaries: one-shot tool execution and local port checks.
//! Both are traits so probes and the process manager can be tested with fakes.

use async_trait::async_trait;
use raptor_core::{RaptorError, Result};
use std::net::TcpListener;
use std::time::Duration;

/// Runs short-lived helper tools (PowerShell, lspci, rocminfo, rocm-smi).
#[async_trait]
pub trait CommandRunner: Send + Sync {
    /// Runs `program` with `args` and returns its stdout. A non-zero exit or a
    /// missing program is an error; stderr is included (truncated) in the message.
    async fn run(&self, program: &str, args: &[String], timeout: Duration) -> Result<String>;

    /// Like [`CommandRunner::run`], with extra variables merged over the inherited
    /// environment. Runners that cannot set a child environment refuse non-empty `env`
    /// instead of silently ignoring it.
    async fn run_with_env(
        &self,
        program: &str,
        args: &[String],
        env: &[(String, String)],
        timeout: Duration,
    ) -> Result<String> {
        if env.is_empty() {
            return self.run(program, args, timeout).await;
        }
        Err(RaptorError::Runtime(format!(
            "this command runner cannot set the environment for {program}"
        )))
    }
}

/// Reports whether a TCP port can be bound on a host right now.
pub trait PortCheck: Send + Sync {
    fn is_free(&self, host: &str, port: u16) -> bool;
}

/// Real port check: attempts to bind and immediately releases the socket.
#[derive(Debug, Default, Clone, Copy)]
pub struct TcpPortCheck;

impl PortCheck for TcpPortCheck {
    fn is_free(&self, host: &str, port: u16) -> bool {
        TcpListener::bind((host, port)).is_ok()
    }
}

/// Real command runner backed by `tokio::process`.
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemCommandRunner;

#[async_trait]
impl CommandRunner for SystemCommandRunner {
    async fn run(&self, program: &str, args: &[String], timeout: Duration) -> Result<String> {
        self.run_with_env(program, args, &[], timeout).await
    }

    async fn run_with_env(
        &self,
        program: &str,
        args: &[String],
        env: &[(String, String)],
        timeout: Duration,
    ) -> Result<String> {
        let mut cmd = tokio::process::Command::new(program);
        cmd.args(args)
            .envs(env.iter().map(|(k, v)| (k.as_str(), v.as_str())))
            .stdin(std::process::Stdio::null())
            .kill_on_drop(true);
        #[cfg(windows)]
        {
            // CREATE_NO_WINDOW: never flash a console window from the GUI.
            cmd.creation_flags(0x0800_0000);
        }
        let output = tokio::time::timeout(timeout, cmd.output())
            .await
            .map_err(|_| RaptorError::Runtime(format!("{program} timed out after {timeout:?}")))?
            .map_err(|e| RaptorError::Runtime(format!("cannot run {program}: {e}")))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let snippet: String = stderr.trim().chars().take(400).collect();
            return Err(RaptorError::Runtime(format!(
                "{program} exited with {}: {snippet}",
                output.status
            )));
        }
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    }
}

/// Builds the argument list for `powershell -EncodedCommand`. Encoding the
/// script avoids every Windows command-line quoting pitfall.
pub fn powershell_encoded_args(script: &str) -> Vec<String> {
    let utf16: Vec<u8> = script
        .encode_utf16()
        .flat_map(|u| u.to_le_bytes())
        .collect();
    vec![
        "-NoProfile".into(),
        "-NonInteractive".into(),
        "-ExecutionPolicy".into(),
        "Bypass".into(),
        "-EncodedCommand".into(),
        base64_encode(&utf16),
    ]
}

/// Standard base64 (RFC 4648, with padding).
pub fn base64_encode(input: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
        let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(ALPHABET[(n >> 18) as usize & 63] as char);
        out.push(ALPHABET[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            ALPHABET[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_rfc4648_vectors() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"Man"), "TWFu");
    }

    #[test]
    fn powershell_script_is_utf16le_base64() {
        let args = powershell_encoded_args("A");
        assert_eq!(args.last().map(String::as_str), Some("QQA="));
        assert!(args.iter().any(|a| a == "-EncodedCommand"));
    }

    #[test]
    fn port_check_detects_bound_port() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        assert!(!TcpPortCheck.is_free("127.0.0.1", port));
        drop(listener);
        assert!(TcpPortCheck.is_free("127.0.0.1", port));
    }
}
