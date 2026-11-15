//! Split GGUF shard sets: `model-Q4_K_M-00001-of-00005.gguf` and friends.
//!
//! A set is identified by its directory, its prefix (everything before `-NNNNN-of-`) and its
//! total. It is complete only when every index from 1 to the total is present exactly once.
//! Pure functions over file names, so every rule here is tested without touching disk.

use crate::hf::RepoFile;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Upper bound on shards in one set. Real splits use a few dozen at most.
pub const MAX_SHARDS: u32 = 1000;

/// A parsed shard file name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardName {
    /// Everything before `-NNNNN-of-`, original case preserved.
    pub prefix: String,
    /// 1-based position in the set.
    pub index: u32,
    pub total: u32,
}

/// Parses `prefix-NNNNN-of-NNNNN.gguf`. Returns `None` for anything else, including an
/// index of zero or an index above the total. Works on a bare file name only.
pub fn parse_shard_name(file_name: &str) -> Option<ShardName> {
    let n = file_name.len();
    if n < 5 || !file_name.get(n - 5..)?.eq_ignore_ascii_case(".gguf") {
        return None;
    }
    let stem = &file_name[..n - 5];
    let of = stem.rfind("-of-")?;
    let total_digits = &stem[of + 4..];
    let head = &stem[..of];
    let dash = head.rfind('-')?;
    let index_digits = &head[dash + 1..];
    let prefix = &head[..dash];
    if prefix.is_empty() || !is_digits(index_digits) || !is_digits(total_digits) {
        return None;
    }
    let index: u32 = index_digits.parse().ok()?;
    let total: u32 = total_digits.parse().ok()?;
    if total == 0 || total > MAX_SHARDS || index == 0 || index > total {
        return None;
    }
    Some(ShardName {
        prefix: prefix.to_string(),
        index,
        total,
    })
}

fn is_digits(s: &str) -> bool {
    !s.is_empty() && s.len() <= 5 && s.bytes().all(|b| b.is_ascii_digit())
}

/// Loose listing check: `...-<digits>-of-<digits>.gguf`, with no range rules applied.
/// Use [`parse_shard_name`] when the index and total must make sense.
pub fn is_split_shard(file_name: &str) -> bool {
    let lower = file_name.to_ascii_lowercase();
    let Some(stem) = lower.strip_suffix(".gguf") else {
        return false;
    };
    let Some((head, total)) = stem.rsplit_once("-of-") else {
        return false;
    };
    let Some((_, index)) = head.rsplit_once('-') else {
        return false;
    };
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    digits(index) && digits(total)
}

/// A set of shards found in a repo listing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardSet {
    /// Repo-relative name without the index suffix, e.g. `Q4_K_M/model-Q4_K_M`.
    pub name: String,
    pub total: u32,
    /// Present shards, ordered by index.
    pub shards: Vec<RepoFile>,
    /// Indices from 1 to `total` that are absent. Empty means complete.
    pub missing: Vec<u32>,
    /// Sum of the listed shard sizes.
    pub size_bytes: u64,
}

impl ShardSet {
    /// True when every shard of the set is listed.
    pub fn is_complete(&self) -> bool {
        self.missing.is_empty()
    }

    /// True when `path` (repo-relative) is one of this set's listed shards.
    pub fn contains(&self, path: &str) -> bool {
        self.shards.iter().any(|s| s.file_name == path)
    }
}

/// Groups shard files of a listing into sets. Files that are not shards are ignored.
/// A group with two files claiming the same index is ambiguous and is left out entirely.
pub fn detect_shard_sets(files: &[RepoFile]) -> Vec<ShardSet> {
    // Keyed by (directory with trailing slash, prefix, total).
    type GroupKey = (String, String, u32);
    let mut groups: BTreeMap<GroupKey, Vec<(u32, &RepoFile)>> = BTreeMap::new();
    for f in files {
        let (dir, base) = match f.file_name.rsplit_once('/') {
            Some((d, b)) => (format!("{d}/"), b),
            None => (String::new(), f.file_name.as_str()),
        };
        if let Some(shard) = parse_shard_name(base) {
            groups
                .entry((dir, shard.prefix, shard.total))
                .or_default()
                .push((shard.index, f));
        }
    }
    let mut sets = Vec::new();
    for ((dir, prefix, total), mut members) in groups {
        members.sort_by_key(|(i, _)| *i);
        if members.windows(2).any(|w| w[0].0 == w[1].0) {
            continue;
        }
        let present: Vec<u32> = members.iter().map(|(i, _)| *i).collect();
        let missing: Vec<u32> = (1..=total).filter(|i| !present.contains(i)).collect();
        let shards: Vec<RepoFile> = members.into_iter().map(|(_, f)| f.clone()).collect();
        let size_bytes = shards.iter().map(|s| s.size).sum();
        sets.push(ShardSet {
            name: format!("{dir}{prefix}"),
            total,
            shards,
            missing,
            size_bytes,
        });
    }
    sets
}

/// The set (complete or not) that contains `path`, if `path` is a shard.
pub fn set_containing(files: &[RepoFile], path: &str) -> Option<ShardSet> {
    detect_shard_sets(files)
        .into_iter()
        .find(|s| s.contains(path))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(path: &str, size: u64) -> RepoFile {
        RepoFile {
            file_name: path.to_string(),
            size,
            sha256: Some("a".repeat(64)),
            quant_label: None,
            split_shard: true,
        }
    }

    #[test]
    fn parses_shard_names() {
        let s = parse_shard_name("Big-Q2_K-00002-of-00005.gguf").unwrap();
        assert_eq!(s.prefix, "Big-Q2_K");
        assert_eq!((s.index, s.total), (2, 5));
        assert!(parse_shard_name("big-00001-of-00001.GGUF").is_some());
    }

    #[test]
    fn rejects_non_shard_names() {
        for bad in [
            "model-Q4_K_M.gguf",
            "model-of-00002.gguf",
            "model-00001-of-x.gguf",
            "model-00000-of-00002.gguf",
            "model-00003-of-00002.gguf",
            "model-00001-of-00000.gguf",
            "-00001-of-00002.gguf",
            "model-00001-of-00002.bin",
            "model-000001-of-000002.gguf",
            "model-00001-of-00002",
        ] {
            assert!(parse_shard_name(bad).is_none(), "{bad} must not parse");
        }
    }

    #[test]
    fn complete_set_is_detected_in_order() {
        let files = vec![
            file("m-00002-of-00002.gguf", 20),
            file("m-00001-of-00002.gguf", 10),
            file("other.gguf", 1),
        ];
        let sets = detect_shard_sets(&files);
        assert_eq!(sets.len(), 1);
        assert!(sets[0].is_complete());
        assert_eq!(sets[0].name, "m");
        assert_eq!(sets[0].shards[0].file_name, "m-00001-of-00002.gguf");
        assert_eq!(sets[0].size_bytes, 30);
    }

    #[test]
    fn missing_shards_are_reported_by_index() {
        let files = vec![
            file("m-00001-of-00004.gguf", 1),
            file("m-00004-of-00004.gguf", 1),
        ];
        let sets = detect_shard_sets(&files);
        assert_eq!(sets[0].missing, vec![2, 3]);
        assert!(!sets[0].is_complete());
    }

    #[test]
    fn subfolders_form_separate_sets() {
        let files = vec![
            file("Q4/m-00001-of-00002.gguf", 1),
            file("Q4/m-00002-of-00002.gguf", 1),
            file("Q8/m-00001-of-00002.gguf", 1),
        ];
        let sets = detect_shard_sets(&files);
        assert_eq!(sets.len(), 2);
        let q4 = sets.iter().find(|s| s.name == "Q4/m").unwrap();
        assert!(q4.is_complete());
        let q8 = sets.iter().find(|s| s.name == "Q8/m").unwrap();
        assert_eq!(q8.missing, vec![2]);
        assert!(set_containing(&files, "Q8/m-00001-of-00002.gguf").is_some());
    }

    #[test]
    fn duplicate_index_makes_group_ambiguous() {
        let files = vec![
            file("m-00001-of-00002.gguf", 1),
            file("m-00001-of-00002.GGUF", 1),
            file("m-00002-of-00002.gguf", 1),
        ];
        assert!(detect_shard_sets(&files).is_empty());
    }

    #[test]
    fn loose_split_detection_matches_listing_rules() {
        assert!(is_split_shard("model-Q2_K-00001-of-00002.gguf"));
        assert!(is_split_shard("Big-F16-00003-of-00010.GGUF"));
        assert!(!is_split_shard("model-Q4_K_M.gguf"));
        assert!(!is_split_shard("model-of-00002.gguf"));
        assert!(!is_split_shard("model-00001-of-x.gguf"));
    }

    #[test]
    fn single_files_are_not_sets() {
        let files = vec![file("m-Q4_K_M.gguf", 1)];
        assert!(detect_shard_sets(&files).is_empty());
    }
}
