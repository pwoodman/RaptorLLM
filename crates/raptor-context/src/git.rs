//! Git awareness: changed files and recent history, behind the [`GitProbe`] trait.
//!
//! The real probe shells out to the `git` CLI. A missing or failing `git` is an
//! error, not an empty result, so the engine can report it in the manifest.

use raptor_core::{RaptorError, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::Path;
use std::process::Command;

/// Git state relevant to ranking. Paths are workspace-relative with `/` separators.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GitState {
    /// True when the workspace sits inside a git work tree.
    pub is_repo: bool,
    /// Modified, staged or untracked files, sorted.
    pub changed: BTreeSet<String>,
    /// Files touched by recent commits, most recent first, without duplicates.
    pub recent: Vec<String>,
}

/// Reads git state for a workspace. Implementations must not mutate the repository.
pub trait GitProbe: Send + Sync {
    /// Returns the state of the work tree containing `root`.
    fn state(&self, root: &Path) -> Result<GitState>;
}

/// Probe that never finds a repository. Use for tests and non-git workspaces.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoGit;

impl GitProbe for NoGit {
    fn state(&self, _root: &Path) -> Result<GitState> {
        Ok(GitState::default())
    }
}

/// Probe backed by the `git` executable on `PATH`.
#[derive(Debug, Default, Clone, Copy)]
pub struct GitCli;

/// How many commits of history feed the recency list.
const RECENT_COMMITS: &str = "200";

fn run_git(root: &Path, args: &[&str]) -> Result<(bool, String)> {
    let mut cmd = Command::new("git");
    cmd.arg("-C").arg(root).args(args);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // CREATE_NO_WINDOW: never flash a console window from the GUI.
        cmd.creation_flags(0x0800_0000);
    }
    let output = cmd
        .output()
        .map_err(|e| RaptorError::Runtime(format!("cannot run git: {e}")))?;
    Ok((
        output.status.success(),
        String::from_utf8_lossy(&output.stdout).into_owned(),
    ))
}

/// Parses `git status --porcelain=v1 -z` output into changed paths relative to `prefix`.
///
/// Renames carry the old path as a separate NUL-terminated token, which is skipped.
pub fn parse_status_z(output: &str, prefix: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let mut tokens = output.split('\0');
    while let Some(tok) = tokens.next() {
        if tok.len() < 4 {
            continue;
        }
        let (code, path) = tok.split_at(3);
        if code.starts_with('R') || code.starts_with('C') {
            // The next token is the original path of a rename or copy.
            tokens.next();
        }
        if let Some(rel) = path.strip_prefix(prefix) {
            if !rel.is_empty() {
                out.insert(rel.replace('\\', "/"));
            }
        }
    }
    out
}

/// Parses `git log --name-only --format=` output into unique paths, newest first.
pub fn parse_log_names(output: &str, prefix: &str) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for line in output.lines().map(str::trim).filter(|l| !l.is_empty()) {
        if let Some(rel) = line.strip_prefix(prefix) {
            if seen.insert(rel.to_string()) {
                out.push(rel.replace('\\', "/"));
            }
        }
    }
    out
}

impl GitProbe for GitCli {
    fn state(&self, root: &Path) -> Result<GitState> {
        let (inside, prefix_out) = run_git(root, &["rev-parse", "--show-prefix"])?;
        if !inside {
            return Ok(GitState::default());
        }
        let prefix = prefix_out.trim().to_string();
        let (ok, status) = run_git(
            root,
            &["status", "--porcelain=v1", "-z", "--untracked-files=all"],
        )?;
        if !ok {
            return Err(RaptorError::Runtime("git status failed".into()));
        }
        let (ok, log) = run_git(
            root,
            &["log", "-n", RECENT_COMMITS, "--name-only", "--format="],
        )?;
        if !ok {
            return Err(RaptorError::Runtime("git log failed".into()));
        }
        Ok(GitState {
            is_repo: true,
            changed: parse_status_z(&status, &prefix),
            recent: parse_log_names(&log, &prefix),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_parses_modified_untracked_and_renames() {
        let out = " M src/a.rs\0?? new.txt\0R  src/new_name.rs\0src/old_name.rs\0A  b.rs\0";
        let set = parse_status_z(out, "");
        assert!(set.contains("src/a.rs"));
        assert!(set.contains("new.txt"));
        assert!(set.contains("src/new_name.rs"));
        assert!(!set.contains("src/old_name.rs"));
        assert!(set.contains("b.rs"));
        assert_eq!(set.len(), 4);
    }

    #[test]
    fn status_strips_subdirectory_prefix_and_drops_outside_paths() {
        let out = " M app/src/a.rs\0 M other/x.rs\0";
        let set = parse_status_z(out, "app/");
        assert_eq!(set.into_iter().collect::<Vec<_>>(), vec!["src/a.rs"]);
    }

    #[test]
    fn log_names_are_unique_and_ordered() {
        let out = "src/a.rs\n\nsrc/b.rs\nsrc/a.rs\n";
        assert_eq!(parse_log_names(out, ""), vec!["src/a.rs", "src/b.rs"]);
    }

    #[test]
    fn no_git_is_empty_not_error() {
        let st = NoGit.state(Path::new(".")).expect("ok");
        assert!(!st.is_repo);
        assert!(st.changed.is_empty());
    }
}
