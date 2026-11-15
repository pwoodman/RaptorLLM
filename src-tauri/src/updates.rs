//! Dependency updates: newer llama.cpp runtime builds, and whether the AMD HIP SDK that the
//! AMD GPU build needs is present. Checking only reads the GitHub release metadata. Nothing is
//! downloaded or installed until the user confirms (see `commands::system`).
//!
//! The expected SHA-256 of an update comes from the `digest` GitHub publishes for each release
//! asset, over the same TLS connection as the download. It is checked before anything is
//! extracted. It is not an independent signature, and the UI says where it came from.

use async_trait::async_trait;
use raptor_core::{RaptorError, Result};
use raptor_runtime::manifest::Manifest;
use serde::Serialize;
use serde_json::Value;
use std::path::PathBuf;
use std::time::Duration;

/// Where llama.cpp releases are listed. GitHub's own "latest release" points at a `v1.0.0`
/// marker with no binaries, and llama.cpp publishes every `b<number>` build as a pre-release,
/// so the list is read and the highest build number wins.
pub const RELEASES_URL: &str =
    "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=30";

/// Oldest AMD HIP SDK the stock AMD build loads against (measured: 7.1 fails with a
/// `hipblas.dll` name mismatch, 7.2 works).
pub const HIP_SDK_MIN: (u32, u32) = (7, 2);

#[derive(Debug, Clone, PartialEq)]
pub struct ReleaseAsset {
    pub name: String,
    pub url: String,
    pub size_bytes: Option<u64>,
    /// Lowercase hex, from GitHub's `sha256:` digest. None when GitHub published none.
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Release {
    pub tag: String,
    pub prerelease: bool,
    pub assets: Vec<ReleaseAsset>,
}

/// Source of the newest release. Production reads GitHub; tests use a fake.
#[async_trait]
pub trait ReleaseSource: Send + Sync {
    async fn latest(&self) -> Result<Release>;
}

/// Stands in when the HTTP client could not be built, so the check reports why.
pub struct Unavailable(pub String);

#[async_trait]
impl ReleaseSource for Unavailable {
    async fn latest(&self) -> Result<Release> {
        Err(RaptorError::Network(format!(
            "update check unavailable: {}",
            self.0
        )))
    }
}

pub struct GitHubReleases {
    client: reqwest::Client,
}

impl GitHubReleases {
    pub fn new() -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(20))
            .user_agent(concat!("RaptorLLM/", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|e| RaptorError::Network(format!("http client: {e}")))?;
        Ok(Self { client })
    }
}

#[async_trait]
impl ReleaseSource for GitHubReleases {
    async fn latest(&self) -> Result<Release> {
        let resp = self
            .client
            .get(RELEASES_URL)
            .header("Accept", "application/vnd.github+json")
            .send()
            .await
            .map_err(|e| RaptorError::Network(format!("could not reach GitHub: {e}")))?;
        let status = resp.status();
        if !status.is_success() {
            return Err(RaptorError::Network(format!(
                "GitHub returned HTTP {} for the llama.cpp release list",
                status.as_u16()
            )));
        }
        let body: Value = resp
            .json()
            .await
            .map_err(|e| RaptorError::Network(format!("unreadable release data: {e}")))?;
        newest_build(&body)
    }
}

/// The release with the highest `b<number>` tag in a GitHub release list. Marker releases
/// such as `v1.0.0` are skipped.
pub fn newest_build(list: &Value) -> Result<Release> {
    let best = list
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|r| Some((build_number(r.get("tag_name")?.as_str()?)?, r)))
        .max_by_key(|(n, _)| *n)
        .ok_or_else(|| RaptorError::Network("the release list has no llama.cpp build".into()))?;
    parse_release(best.1)
}

/// Reads a GitHub release object. Assets without a name or URL are skipped.
pub fn parse_release(v: &Value) -> Result<Release> {
    let tag = v
        .get("tag_name")
        .and_then(Value::as_str)
        .ok_or_else(|| RaptorError::Network("release data has no tag_name".into()))?
        .to_string();
    let assets = v
        .get("assets")
        .and_then(Value::as_array)
        .map(|list| {
            list.iter()
                .filter_map(|a| {
                    Some(ReleaseAsset {
                        name: a.get("name")?.as_str()?.to_string(),
                        url: a.get("browser_download_url")?.as_str()?.to_string(),
                        size_bytes: a.get("size").and_then(Value::as_u64),
                        sha256: a
                            .get("digest")
                            .and_then(Value::as_str)
                            .and_then(|d| d.strip_prefix("sha256:"))
                            .filter(|h| h.len() == 64 && h.bytes().all(|b| b.is_ascii_hexdigit()))
                            .map(str::to_ascii_lowercase),
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    Ok(Release {
        tag,
        prerelease: v
            .get("prerelease")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        assets,
    })
}

/// `b12340` becomes 12340. None for anything that is not `b<digits>`.
pub fn build_number(tag: &str) -> Option<u64> {
    tag.strip_prefix('b')?.parse().ok()
}

/// One installed runtime and the newer build of it, if there is one.
#[derive(Debug, Clone, Serialize)]
pub struct RuntimeUpdate {
    pub id: String,
    pub current_version: String,
    pub latest_version: String,
    pub available: bool,
    /// Why nothing is offered, or where the checksum came from.
    pub note: String,
    pub url: Option<String>,
    pub sha256: Option<String>,
    pub size_bytes: Option<u64>,
    /// The index entry that installing this update adds. Not sent to the UI.
    #[serde(skip)]
    pub manifest: Option<Manifest>,
}

/// For each runtime id in `index` (newest entry per id), finds the same build in `release`.
/// The asset name is the old one with the build tag swapped. `wanted` limits it to ids the user
/// has installed. A build is offered only when its asset is in the release with a checksum.
pub fn plan_updates(
    index: &[Manifest],
    release: &Release,
    wanted: &[String],
) -> Vec<RuntimeUpdate> {
    let latest_n = build_number(&release.tag);
    let mut newest: Vec<&Manifest> = Vec::new();
    for m in index {
        if !wanted.contains(&m.id) {
            continue;
        }
        match newest.iter().position(|n| n.id == m.id) {
            Some(i) => {
                if build_number(&m.version) > build_number(&newest[i].version) {
                    newest[i] = m;
                }
            }
            None => newest.push(m),
        }
    }
    newest
        .into_iter()
        .map(|m| {
            let mut update = RuntimeUpdate {
                id: m.id.clone(),
                current_version: m.version.clone(),
                latest_version: release.tag.clone(),
                available: false,
                note: String::new(),
                url: None,
                sha256: None,
                size_bytes: None,
                manifest: None,
            };
            let newer = match (latest_n, build_number(&m.version)) {
                (Some(l), Some(c)) => l > c,
                _ => false,
            };
            if latest_n.is_none() || build_number(&m.version).is_none() {
                update.note = format!(
                    "Cannot compare {} with {}: not llama.cpp build numbers",
                    m.version, release.tag
                );
                return update;
            }
            if !newer {
                update.note = "Up to date".into();
                return update;
            }
            let old_name = m.url.rsplit('/').next().unwrap_or("");
            let new_name = old_name.replace(&m.version, &release.tag);
            match release.assets.iter().find(|a| a.name == new_name) {
                None => {
                    update.note = format!("{} has no {new_name} build yet", release.tag);
                }
                Some(asset) => match &asset.sha256 {
                    None => {
                        update.note = format!(
                            "{} was published without a checksum, so it is not offered",
                            asset.name
                        );
                    }
                    Some(sha) => {
                        let mut next = m.clone();
                        next.version = release.tag.clone();
                        next.llama_cpp_build = release.tag.clone();
                        next.url = asset.url.clone();
                        next.sha256 = sha.clone();
                        update.available = true;
                        update.note =
                            "Checksum from the GitHub release; verified before use".into();
                        update.url = Some(asset.url.clone());
                        update.sha256 = Some(sha.clone());
                        update.size_bytes = asset.size_bytes;
                        update.manifest = Some(next);
                    }
                },
            }
            update
        })
        .collect()
}

/// Runtime id of the Vulkan build. It sits next to `llama-hip-win` under the runtimes folder.
pub const VULKAN_RUNTIME_ID: &str = "llama-vulkan-win";

/// The Vulkan runtime of a release: whether it is on disk, and the verified download that adds it.
/// Shown in Settings > Dependencies. Installing is a separate, confirmed step.
#[derive(Debug, Clone, Serialize)]
pub struct VulkanOffer {
    pub installed_version: Option<String>,
    pub latest_version: String,
    pub available: bool,
    /// Why nothing is offered, or where the checksum came from.
    pub note: String,
    pub url: Option<String>,
    pub sha256: Option<String>,
    pub size_bytes: Option<u64>,
    /// The index entry that installing adds. Not sent to the UI.
    #[serde(skip)]
    pub manifest: Option<Manifest>,
}

/// Looks for `llama-<tag>-bin-win-vulkan-x64.zip` in `release`. It is offered when the Vulkan
/// runtime is not installed or an older build is, and only with a checksum from the release. The
/// Vulkan build needs no HIP SDK, only the GPU driver's Vulkan loader.
pub fn plan_vulkan(release: &Release, installed: Option<&str>, windows: bool) -> VulkanOffer {
    let mut offer = VulkanOffer {
        installed_version: installed.map(str::to_string),
        latest_version: release.tag.clone(),
        available: false,
        note: String::new(),
        url: None,
        sha256: None,
        size_bytes: None,
        manifest: None,
    };
    if !windows {
        offer.note = "The Vulkan runtime is only offered for Windows here.".into();
        return offer;
    }
    if let (Some(have), Some(latest)) =
        (installed.and_then(build_number), build_number(&release.tag))
    {
        if have >= latest {
            offer.note = "Up to date".into();
            return offer;
        }
    }
    let name = format!("llama-{}-bin-win-vulkan-x64.zip", release.tag);
    match release.assets.iter().find(|a| a.name == name) {
        None => offer.note = format!("{} has no {name} build", release.tag),
        Some(asset) => match &asset.sha256 {
            None => {
                offer.note = format!(
                    "{} was published without a checksum, so it is not offered",
                    asset.name
                );
            }
            Some(sha) => {
                offer.available = true;
                offer.note = "Checksum from the GitHub release; verified before use".into();
                offer.url = Some(asset.url.clone());
                offer.sha256 = Some(sha.clone());
                offer.size_bytes = asset.size_bytes;
                offer.manifest = Some(Manifest {
                    id: VULKAN_RUNTIME_ID.into(),
                    version: release.tag.clone(),
                    llama_cpp_build: release.tag.clone(),
                    backend: raptor_runtime::manifest::Backend::Vulkan,
                    os: raptor_runtime::hardware::Os::Windows,
                    gfx_archs: Vec::new(),
                    url: asset.url.clone(),
                    sha256: sha.clone(),
                    min_driver: None,
                    signature: None,
                });
            }
        },
    }
    offer
}

/// State of the AMD HIP SDK, which the AMD GPU runtime loads its libraries from.
#[derive(Debug, Clone, Serialize)]
pub struct HipSdkStatus {
    /// Versions found, newest first (e.g. `7.2`).
    pub versions: Vec<String>,
    /// True when one of them is at least [`HIP_SDK_MIN`].
    pub ok: bool,
    pub required: String,
    pub guidance: String,
    pub download_page: &'static str,
}

pub const HIP_SDK_PAGE: &str = "https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html";

/// Version of an SDK `bin` folder like `C:\Program Files\AMD\ROCm\7.2\bin`.
fn version_of_bin_dir(dir: &std::path::Path) -> Option<(u32, u32)> {
    let name = dir.parent()?.file_name()?.to_str()?;
    let mut parts = name.split('.');
    Some((parts.next()?.parse().ok()?, parts.next()?.parse().ok()?))
}

pub fn hip_sdk_status(bin_dirs: &[PathBuf]) -> HipSdkStatus {
    let mut found: Vec<(u32, u32)> = bin_dirs
        .iter()
        .filter_map(|d| version_of_bin_dir(d))
        .collect();
    found.sort_unstable_by(|a, b| b.cmp(a));
    found.dedup();
    let ok = found.iter().any(|v| *v >= HIP_SDK_MIN);
    let required = format!("{}.{}", HIP_SDK_MIN.0, HIP_SDK_MIN.1);
    let guidance = if ok {
        "A compatible AMD HIP SDK is installed.".to_string()
    } else if found.is_empty() {
        format!(
            "No AMD HIP SDK was found. The AMD GPU build needs HIP SDK {required} or newer. \
             Raptor does not run AMD's installer for you: download it from AMD, install it, then check again."
        )
    } else {
        format!(
            "The newest AMD HIP SDK found is too old for the AMD GPU build, which needs {required} or newer. \
             Install a newer one from AMD, then check again."
        )
    };
    HipSdkStatus {
        versions: found.iter().map(|(a, b)| format!("{a}.{b}")).collect(),
        ok,
        required,
        guidance,
        download_page: HIP_SDK_PAGE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use raptor_runtime::hardware::Os;
    use raptor_runtime::manifest::Backend;
    use serde_json::json;

    fn manifest(id: &str, version: &str, file: &str) -> Manifest {
        Manifest {
            id: id.into(),
            version: version.into(),
            llama_cpp_build: version.into(),
            backend: Backend::Hip,
            os: Os::Windows,
            gfx_archs: vec!["gfx1200".into()],
            url: format!(
                "https://github.com/ggml-org/llama.cpp/releases/download/{version}/{file}"
            ),
            sha256: "0".repeat(64),
            min_driver: None,
            signature: None,
        }
    }

    fn release() -> Release {
        parse_release(&json!({
            "tag_name": "b12400",
            "prerelease": false,
            "assets": [
                {"name": "llama-b12400-bin-win-rocm-10.0-x64.zip", "size": 300,
                 "browser_download_url": "https://github.com/ggml-org/llama.cpp/releases/download/b12400/llama-b12400-bin-win-rocm-10.0-x64.zip",
                 "digest": format!("sha256:{}", "AB".repeat(32))},
                {"name": "llama-b12400-bin-win-vulkan-x64.zip", "size": 200,
                 "browser_download_url": "https://example.test/v.zip", "digest": null},
                {"name": "no-url"}
            ]
        }))
        .unwrap()
    }

    #[test]
    fn a_release_is_parsed_with_lowercase_checksums_and_bad_assets_skipped() {
        let r = release();
        assert_eq!(r.tag, "b12400");
        assert_eq!(r.assets.len(), 2);
        assert_eq!(
            r.assets[0].sha256.as_deref(),
            Some("ab".repeat(32).as_str())
        );
        assert_eq!(r.assets[1].sha256, None);
        assert!(parse_release(&json!({})).is_err());
    }

    fn vulkan_release(digest: serde_json::Value) -> Release {
        parse_release(&json!({
            "tag_name": "b12400",
            "assets": [
                {"name": "llama-b12400-bin-win-vulkan-x64.zip", "size": 40_000_000,
                 "browser_download_url": "https://github.com/ggml-org/llama.cpp/releases/download/b12400/llama-b12400-bin-win-vulkan-x64.zip",
                 "digest": digest}
            ]
        }))
        .unwrap()
    }

    #[test]
    fn the_vulkan_runtime_is_offered_with_its_checksum_when_it_is_not_installed() {
        let r = vulkan_release(json!(format!("sha256:{}", "cd".repeat(32))));
        let o = plan_vulkan(&r, None, true);
        assert!(o.available, "{}", o.note);
        assert_eq!(o.size_bytes, Some(40_000_000));
        let m = o.manifest.unwrap();
        assert_eq!(m.id, VULKAN_RUNTIME_ID);
        assert_eq!(m.version, "b12400");
        assert_eq!(m.backend, Backend::Vulkan);
        assert_eq!(m.sha256, "cd".repeat(32));
        assert!(m.gfx_archs.is_empty());
        m.validate().unwrap();
    }

    #[test]
    fn the_vulkan_runtime_is_not_offered_without_a_checksum_or_when_current() {
        let none = plan_vulkan(&vulkan_release(json!(null)), None, true);
        assert!(!none.available && none.note.contains("without a checksum"));
        let r = vulkan_release(json!(format!("sha256:{}", "cd".repeat(32))));
        let current = plan_vulkan(&r, Some("b12400"), true);
        assert!(!current.available && current.note == "Up to date");
        let older = plan_vulkan(&r, Some("b12360"), true);
        assert!(older.available, "an older build is offered the update");
        let other_os = plan_vulkan(&r, None, false);
        assert!(!other_os.available);
        let missing = plan_vulkan(&release(), None, true);
        assert!(!missing.available, "{}", missing.note);
    }

    #[test]
    fn the_newest_build_is_picked_from_the_list_and_marker_releases_are_skipped() {
        let list = json!([
            {"tag_name": "v1.0.0", "prerelease": false, "assets": []},
            {"tag_name": "b12359", "prerelease": true, "assets": []},
            {"tag_name": "b12360", "prerelease": true, "assets": [
                {"name": "llama-b12360-bin-win-cpu-x64.zip", "browser_download_url": "https://x.test/a.zip"}]},
            {"tag_name": "b12347", "prerelease": true, "assets": []}
        ]);
        let r = newest_build(&list).unwrap();
        assert_eq!(r.tag, "b12360");
        assert_eq!(r.assets.len(), 1);
        assert!(newest_build(&json!([{"tag_name": "v1.0.0"}])).is_err());
        assert!(newest_build(&json!({})).is_err());
    }

    #[test]
    fn a_newer_build_is_offered_only_with_its_asset_and_checksum() {
        let index = vec![
            manifest(
                "llama-hip-win",
                "b12340",
                "llama-b12340-bin-win-rocm-10.0-x64.zip",
            ),
            manifest(
                "llama-vulkan-win",
                "b12340",
                "llama-b12340-bin-win-vulkan-x64.zip",
            ),
            manifest(
                "llama-cpu-win",
                "b12340",
                "llama-b12340-bin-win-cpu-x64.zip",
            ),
        ];
        let wanted = vec!["llama-hip-win".to_string(), "llama-vulkan-win".to_string()];
        let plan = plan_updates(&index, &release(), &wanted);
        assert_eq!(plan.len(), 2, "only installed runtimes are checked");

        let hip = &plan[0];
        assert!(hip.available);
        let m = hip.manifest.as_ref().unwrap();
        assert_eq!(
            (m.version.as_str(), m.llama_cpp_build.as_str()),
            ("b12400", "b12400")
        );
        assert!(m.url.contains("/b12400/llama-b12400-bin-win-rocm"));
        assert_eq!(m.sha256, "ab".repeat(32));
        assert_eq!(
            m.gfx_archs,
            vec!["gfx1200".to_string()],
            "other fields carry over"
        );
        assert!(m.validate().is_ok());

        let vulkan = &plan[1];
        assert!(!vulkan.available && vulkan.manifest.is_none());
        assert!(
            vulkan.note.contains("without a checksum"),
            "{}",
            vulkan.note
        );
    }

    #[test]
    fn nothing_is_offered_when_current_or_ahead_of_the_stable_release() {
        let index = vec![manifest(
            "llama-hip-win",
            "b12340",
            "llama-b12340-bin-win-rocm-10.0-x64.zip",
        )];
        let wanted = vec!["llama-hip-win".to_string()];
        let mut old = release();
        old.tag = "b12100".into();
        let plan = plan_updates(&index, &old, &wanted);
        assert!(!plan[0].available);
        assert_eq!(plan[0].note, "Up to date");
        let mut same = release();
        same.tag = "b12340".into();
        assert!(!plan_updates(&index, &same, &wanted)[0].available);
    }

    #[test]
    fn the_newest_index_entry_per_runtime_is_the_one_compared() {
        let index = vec![
            manifest(
                "llama-hip-win",
                "b12340",
                "llama-b12340-bin-win-rocm-10.0-x64.zip",
            ),
            manifest(
                "llama-hip-win",
                "b12400",
                "llama-b12400-bin-win-rocm-10.0-x64.zip",
            ),
        ];
        let plan = plan_updates(&index, &release(), &["llama-hip-win".to_string()]);
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].current_version, "b12400");
        assert!(!plan[0].available);
    }

    #[test]
    fn an_update_entry_round_trips_through_the_runtime_index_parser() {
        let index = vec![manifest(
            "llama-hip-win",
            "b12340",
            "llama-b12340-bin-win-rocm-10.0-x64.zip",
        )];
        let plan = plan_updates(&index, &release(), &["llama-hip-win".to_string()]);
        let entry = serde_json::to_value(plan[0].manifest.as_ref().unwrap()).unwrap();
        let text = json!({"schema_version": 1, "runtimes": [entry]}).to_string();
        let parsed = crate::state::parse_runtime_index(&text).unwrap();
        assert_eq!(parsed.manifests.len(), 1);
        assert_eq!(parsed.manifests[0].version, "b12400");
        assert!(parsed.refused.is_empty());
    }

    #[test]
    fn the_hip_sdk_check_reads_versions_from_bin_folders() {
        let dirs = |vs: &[&str]| -> Vec<PathBuf> {
            vs.iter()
                .map(|v| PathBuf::from(format!(r"C:\Program Files\AMD\ROCm\{v}\bin")))
                .collect()
        };
        let ok = hip_sdk_status(&dirs(&["7.1", "7.2"]));
        assert!(ok.ok);
        assert_eq!(ok.versions, ["7.2", "7.1"]);
        let old = hip_sdk_status(&dirs(&["7.1"]));
        assert!(!old.ok && old.guidance.contains("too old"));
        let none = hip_sdk_status(&[]);
        assert!(!none.ok && none.guidance.contains("No AMD HIP SDK"));
    }
}
