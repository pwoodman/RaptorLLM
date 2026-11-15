//! `ModelManager`: the facade the desktop app and gateway call. It wires Hub search,
//! download queueing, the library and presets together. It holds no UI or runtime logic.
//!
//! Downloads are pinned: `start_download` asks the Hub for the commit `main` points to, lists
//! files at that commit and fetches every byte from that commit's URLs. The commit is stored
//! with the library entry, so [`ModelManager::check_updates`] can later tell whether the repo
//! has moved on. Nothing is ever replaced automatically.

use crate::compare::{self, Comparison, ModelProfile, RankedCandidate, Task};
use crate::download::{
    DiskSpace, DownloadConfig, DownloadHandle, DownloadManager, DownloadRequest, HttpFetch,
    HttpFetcher, Provenance, SystemDiskSpace,
};
use crate::hf::{
    safe_repo_path, validate_commit_sha, validate_repo_id, HfApi, HfModelSummary, HfRepoInfo,
    HttpHfApi, HubRateLimit, RepoFile, SearchSort,
};
use crate::library::{Library, ModelDetails, ModelSource};
use crate::presets::PresetStore;
use crate::quant::HardwareSummary;
use crate::shard::{parse_shard_name, set_containing};
use raptor_core::{DataDirs, RaptorError, Result};
use serde::Serialize;
use std::collections::HashMap;
use std::sync::Arc;

const PRESETS_FILE: &str = "model-presets.json";

/// Whether an installed Hub model's pinned commit still matches the repo head.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum UpdateStatus {
    /// The repo head is the commit the model was downloaded from.
    UpToDate,
    /// The repo head is a different commit. The files may or may not have changed.
    Changed,
    /// The model was installed before commits were recorded; nothing to compare.
    NotRecorded,
}

/// Result of [`ModelManager::check_updates`] for one installed model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct UpdateCheck {
    pub id: String,
    pub repo: String,
    /// Commit the model was downloaded from.
    pub pinned_revision: Option<String>,
    /// Commit `main` points to now.
    pub head_revision: String,
    pub status: UpdateStatus,
}

/// Entry point for everything model-related.
pub struct ModelManager {
    hf: Arc<dyn HfApi>,
    library: Arc<Library>,
    presets: PresetStore,
    downloads: DownloadManager,
}

impl ModelManager {
    /// Builds a manager from explicit parts. Tests pass fakes; production uses [`ModelManager::with_hugging_face`].
    pub fn new(
        dirs: &DataDirs,
        hf: Arc<dyn HfApi>,
        fetch: Arc<dyn HttpFetch>,
        disk: Arc<dyn DiskSpace>,
    ) -> Result<Self> {
        let library = Arc::new(Library::open(dirs)?);
        let presets = PresetStore::open(dirs.config().join(PRESETS_FILE))?;
        let downloads = DownloadManager::new(
            DownloadConfig::new(dirs.downloads()),
            fetch,
            disk,
            library.clone(),
        );
        Ok(Self {
            hf,
            library,
            presets,
            downloads,
        })
    }

    /// Production wiring: Hugging Face over HTTPS, real disk space. The token (optional,
    /// for gated repos) is never logged or placed in a URL.
    pub fn with_hugging_face(dirs: &DataDirs, token: Option<String>) -> Result<Self> {
        let hf = Arc::new(HttpHfApi::new(token.clone())?);
        let fetch = Arc::new(HttpFetcher::new(token)?);
        Self::new(dirs, hf, fetch, Arc::new(SystemDiskSpace))
    }

    /// Searches GGUF repos on the Hub.
    pub async fn search(
        &self,
        query: &str,
        sort: SearchSort,
        limit: u32,
    ) -> Result<Vec<HfModelSummary>> {
        self.hf.search_gguf(query, sort, limit).await
    }

    /// GGUF files of a repo's `main` revision (subfolders included), with sizes, sha256 and
    /// quant labels. Shards appear one by one; see `shard::detect_shard_sets`.
    pub async fn repo_files(&self, repo: &str) -> Result<Vec<RepoFile>> {
        validate_repo_id(repo)?;
        self.hf.list_repo_files(repo).await
    }

    /// Repo metadata (license, gating, tags).
    pub async fn repo_info(&self, repo: &str) -> Result<HfRepoInfo> {
        validate_repo_id(repo)?;
        self.hf.repo_info(repo).await
    }

    /// Rate-limit state from the most recent Hub response, if the Hub sent one.
    pub fn hub_rate_limit(&self) -> Option<HubRateLimit> {
        self.hf.last_rate_limit()
    }

    /// Queues a verified, pinned download of one GGUF file from a repo into the library.
    ///
    /// `file_path` is a repo-relative path (a top-level name works too). If it names a shard of
    /// a split model, the whole set is downloaded as one job and registered as one entry; an
    /// incomplete set is refused before any transfer. Refuses files without a published sha256,
    /// and an existing library entry unless `overwrite` is set.
    pub async fn start_download(
        &self,
        repo: &str,
        file_path: &str,
        overwrite: bool,
    ) -> Result<DownloadHandle> {
        validate_repo_id(repo)?;
        safe_repo_path(file_path)?;
        let snapshot = self.hf.repo_snapshot(repo).await?;
        validate_commit_sha(&snapshot.revision)?;
        let files = self.hf.list_repo_files_at(repo, &snapshot.revision).await?;
        let provenance = Provenance {
            revision: Some(snapshot.revision.clone()),
            hub_tags: snapshot.info.tags.clone(),
        };
        let listed = files
            .iter()
            .find(|f| f.file_name == file_path)
            .ok_or_else(|| {
                RaptorError::NotFound(format!("{file_path} is not a GGUF file in {repo}"))
            })?;

        let base = file_path.rsplit('/').next().unwrap_or(file_path);
        if parse_shard_name(base).is_some_and(|s| s.total > 1) {
            let set = set_containing(&files, file_path).ok_or_else(|| {
                RaptorError::NotFound(format!("no shard set found for {file_path}"))
            })?;
            if !set.is_complete() {
                let missing: Vec<String> = set
                    .missing
                    .iter()
                    .map(|i| format!("{i} of {}", set.total))
                    .collect();
                return Err(RaptorError::NotFound(format!(
                    "shard set {} is incomplete: missing shard {}",
                    set.name,
                    missing.join(", ")
                )));
            }
            let first_id = format!("{repo}/{}", set.shards[0].file_name);
            if !overwrite && self.library.contains_id(&first_id)? {
                return Err(RaptorError::Conflict(format!(
                    "{} is already in the library; pass overwrite to replace it",
                    set.shards[0].file_name
                )));
            }
            let mut reqs = Vec::with_capacity(set.shards.len());
            for (i, shard) in set.shards.iter().enumerate() {
                reqs.push(self.request_for(repo, &snapshot.revision, shard, overwrite, i == 0)?);
            }
            return self.downloads.start_set(reqs, provenance);
        }

        let id = format!("{repo}/{file_path}");
        let req = self.request_for(repo, &snapshot.revision, listed, overwrite, true)?;
        if !overwrite && (self.library.contains_id(&id)? || req.dest.exists()) {
            return Err(RaptorError::Conflict(format!(
                "{file_path} is already in the library; pass overwrite to replace it"
            )));
        }
        self.downloads.start_with(req, provenance)
    }

    /// Builds one request for a listed file at a pinned commit.
    fn request_for(
        &self,
        repo: &str,
        revision: &str,
        file: &RepoFile,
        overwrite: bool,
        verify_architecture: bool,
    ) -> Result<DownloadRequest> {
        if file.size == 0 {
            return Err(RaptorError::Unsupported(format!(
                "Hugging Face reports no size for {}",
                file.file_name
            )));
        }
        let sha256 = file.sha256.clone().ok_or_else(|| {
            RaptorError::Unsupported(format!(
                "Hugging Face publishes no sha256 for {}; refusing an unverifiable download",
                file.file_name
            ))
        })?;
        Ok(DownloadRequest {
            repo: repo.to_string(),
            file_name: file.file_name.clone(),
            url: self.hf.file_url_at(repo, revision, &file.file_name)?,
            expected_sha256: sha256,
            expected_size: file.size,
            dest: self.library.managed_path(repo, &file.file_name)?,
            overwrite,
            source: ModelSource::HuggingFace {
                repo: repo.to_string(),
                file_name: file.file_name.clone(),
            },
            verify_architecture,
        })
    }

    /// Compares the repo head of every installed Hub model with the commit it was downloaded
    /// from. Makes no download. Returns a clean error when Hugging Face cannot be reached;
    /// nothing is changed either way.
    pub async fn check_updates(&self) -> Result<Vec<UpdateCheck>> {
        let mut heads: HashMap<String, String> = HashMap::new();
        let mut out = Vec::new();
        for entry in self.library.list()? {
            let ModelSource::HuggingFace { repo, .. } = &entry.source else {
                continue;
            };
            let details = self.library.details(&entry.id)?;
            let head = match heads.get(repo) {
                Some(h) => h.clone(),
                None => {
                    let snapshot = self.hf.repo_snapshot(repo).await?;
                    validate_commit_sha(&snapshot.revision)?;
                    heads.insert(repo.clone(), snapshot.revision.clone());
                    snapshot.revision
                }
            };
            let status = match &details.revision {
                None => UpdateStatus::NotRecorded,
                Some(pinned) if *pinned == head => UpdateStatus::UpToDate,
                Some(_) => UpdateStatus::Changed,
            };
            out.push(UpdateCheck {
                id: entry.id.clone(),
                repo: repo.clone(),
                pinned_revision: details.revision,
                head_revision: head,
                status,
            });
        }
        Ok(out)
    }

    /// Removes an installed model and its presets. Every shard of a split model is removed.
    /// Referenced files stay on disk.
    pub fn delete_model(&self, id: &str) -> Result<()> {
        self.library.delete(id)?;
        self.presets.delete_for_model(id)?;
        Ok(())
    }

    /// Side-by-side comparison of installed models at `ctx` tokens, optionally rated for
    /// `hardware`. Every id must be installed.
    pub fn compare(
        &self,
        ids: &[&str],
        ctx: u64,
        hardware: Option<HardwareSummary>,
    ) -> Result<Comparison> {
        let profiles = self.profiles(ids)?;
        compare::compare(&profiles, ctx, hardware)
    }

    /// Ranks installed models for a task. See [`compare::rank_for_at`] for the heuristics.
    pub fn rank_installed(
        &self,
        task: Task,
        hardware: HardwareSummary,
        ctx: u64,
    ) -> Result<Vec<RankedCandidate>> {
        let ids: Vec<String> = self.library.list()?.into_iter().map(|e| e.id).collect();
        let refs: Vec<&str> = ids.iter().map(String::as_str).collect();
        let profiles = self.profiles(&refs)?;
        Ok(compare::rank_for_at(task, &profiles, hardware, ctx))
    }

    /// Full details of an installed model: files, pinned revision, Hub tags.
    pub fn model_details(&self, id: &str) -> Result<ModelDetails> {
        self.library.details(id)
    }

    fn profiles(&self, ids: &[&str]) -> Result<Vec<ModelProfile>> {
        ids.iter()
            .map(|id| {
                self.library
                    .details(id)
                    .map(|d| ModelProfile::from_details(&d))
            })
            .collect()
    }

    /// The local model library.
    pub fn library(&self) -> &Library {
        self.library.as_ref()
    }

    /// Per-model presets.
    pub fn presets(&self) -> &PresetStore {
        &self.presets
    }

    /// The download queue.
    pub fn downloads(&self) -> &DownloadManager {
        &self.downloads
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::download::{DownloadOutcome, JobState};
    use crate::hf::RepoFile;
    use crate::presets::Preset;
    use crate::testing::{
        gguf_file, gguf_shard, sha256_hex, FakeFetch, FakeHf, FixedDisk, RoutedFetch, FAKE_SHA,
    };
    use async_trait::async_trait;
    use raptor_core::Result as CoreResult;

    struct Env {
        _tmp: tempfile::TempDir,
        dirs: DataDirs,
    }

    fn env() -> Env {
        let tmp = tempfile::tempdir().unwrap();
        let dirs = DataDirs::new(tmp.path().join("RaptorLLM"));
        dirs.ensure().unwrap();
        Env { _tmp: tmp, dirs }
    }

    fn repo_file(name: &str, body: &[u8], split: bool) -> RepoFile {
        RepoFile {
            file_name: name.to_string(),
            size: body.len() as u64,
            sha256: Some(sha256_hex(body)),
            quant_label: crate::quant::parse_quant_label(name).map(str::to_string),
            split_shard: split,
        }
    }

    fn fake_hf(files: Vec<RepoFile>) -> FakeHf {
        FakeHf::new(
            vec![HfModelSummary {
                id: "owner/repo".into(),
                author: Some("owner".into()),
                downloads: 7,
                likes: 1,
                tags: vec!["gguf".into()],
                last_modified: None,
                pipeline_tag: Some("text-generation".into()),
            }],
            files,
        )
    }

    fn manager(env: &Env, files: Vec<RepoFile>, body: Vec<u8>) -> ModelManager {
        ModelManager::new(
            &env.dirs,
            Arc::new(fake_hf(files)),
            Arc::new(FakeFetch::new(body)),
            Arc::new(FixedDisk(u64::MAX)),
        )
        .unwrap()
    }

    /// The URL the fake Hub hands out for a file at the fake head sha.
    fn url(repo: &str, file: &str) -> String {
        format!("https://fake.hf.test/{repo}/resolve/{FAKE_SHA}/{file}")
    }

    #[tokio::test]
    async fn end_to_end_download_lands_in_library_with_metadata_and_pin() {
        let env = env();
        let body = gguf_file("llama", 15);
        let mgr = manager(
            &env,
            vec![repo_file("tiny-Q4_K_M.gguf", &body, false)],
            body.clone(),
        );
        let handle = mgr
            .start_download("owner/repo", "tiny-Q4_K_M.gguf", false)
            .await
            .unwrap();
        assert!(matches!(
            handle.finish().await.unwrap(),
            DownloadOutcome::Completed { .. }
        ));

        let installed = mgr.library().list().unwrap();
        assert_eq!(installed.len(), 1);
        assert_eq!(installed[0].id, "owner/repo/tiny-Q4_K_M.gguf");
        assert_eq!(installed[0].gguf.architecture.as_deref(), Some("llama"));
        assert_eq!(installed[0].sha256, sha256_hex(&body));
        assert!(matches!(
            installed[0].source,
            ModelSource::HuggingFace { .. }
        ));
        let details = mgr.model_details("owner/repo/tiny-Q4_K_M.gguf").unwrap();
        assert_eq!(details.revision.as_deref(), Some(FAKE_SHA));
        mgr.library().verify("owner/repo/tiny-Q4_K_M.gguf").unwrap();
    }

    #[tokio::test]
    async fn download_is_pinned_to_the_commit_not_main() {
        let env = env();
        let body = gguf_file("llama", 15);
        // Only the commit-pinned URL is routed. A download from `main` would fail.
        let mut routes = RoutedFetch::new();
        routes.route(&url("owner/repo", "p-Q4_K_M.gguf"), body.clone());
        let mgr = ModelManager::new(
            &env.dirs,
            Arc::new(fake_hf(vec![repo_file("p-Q4_K_M.gguf", &body, false)])),
            Arc::new(routes),
            Arc::new(FixedDisk(u64::MAX)),
        )
        .unwrap();
        mgr.start_download("owner/repo", "p-Q4_K_M.gguf", false)
            .await
            .unwrap()
            .finish()
            .await
            .unwrap();
        assert_eq!(
            mgr.model_details("owner/repo/p-Q4_K_M.gguf")
                .unwrap()
                .revision
                .as_deref(),
            Some(FAKE_SHA)
        );
    }

    #[tokio::test]
    async fn known_unsupported_architecture_never_enters_library() {
        let env = env();
        let body = gguf_file("clip", 15);
        let mgr = manager(&env, vec![repo_file("x-Q4_K_M.gguf", &body, false)], body);
        let handle = mgr
            .start_download("owner/repo", "x-Q4_K_M.gguf", false)
            .await
            .unwrap();
        assert!(matches!(
            handle.finish().await,
            Err(RaptorError::Unsupported(_))
        ));
        assert!(mgr.library().list().unwrap().is_empty());
    }

    #[tokio::test]
    async fn refuses_unhashed_files_and_incomplete_sets_before_transfer() {
        let env = env();
        let body = gguf_file("llama", 15);
        let mut unhashed = repo_file("plain-Q8_0.gguf", &body, false);
        unhashed.sha256 = None;
        let partial_set = repo_file("big-Q2_K-00001-of-00002.gguf", &body, true);
        let mgr = manager(&env, vec![partial_set, unhashed], body);
        match mgr
            .start_download("owner/repo", "big-Q2_K-00001-of-00002.gguf", false)
            .await
        {
            Err(RaptorError::NotFound(m)) => assert!(m.contains("incomplete"), "{m}"),
            Err(e) => panic!("expected incomplete set, got {e:?}"),
            Ok(_) => panic!("an incomplete set must not start"),
        }
        assert!(matches!(
            mgr.start_download("owner/repo", "plain-Q8_0.gguf", false).await,
            Err(RaptorError::Unsupported(m)) if m.contains("sha256")
        ));
    }

    #[tokio::test]
    async fn complete_shard_set_downloads_as_one_entry() {
        let env = env();
        let a = gguf_shard("llama", 0, 2);
        let b = gguf_shard("llama", 1, 2);
        let fetch = {
            let mut f = RoutedFetch::new();
            f.route(&url("owner/repo", "Q4/big-00001-of-00002.gguf"), a.clone());
            f.route(&url("owner/repo", "Q4/big-00002-of-00002.gguf"), b.clone());
            Arc::new(f)
        };
        let files = vec![
            repo_file("Q4/big-00001-of-00002.gguf", &a, true),
            repo_file("Q4/big-00002-of-00002.gguf", &b, true),
        ];
        let mgr = ModelManager::new(
            &env.dirs,
            Arc::new(fake_hf(files)),
            fetch,
            Arc::new(FixedDisk(u64::MAX)),
        )
        .unwrap();
        // Asking for any shard downloads the set.
        mgr.start_download("owner/repo", "Q4/big-00002-of-00002.gguf", false)
            .await
            .unwrap()
            .finish()
            .await
            .unwrap();
        let list = mgr.library().list().unwrap();
        assert_eq!(list.len(), 1, "a split model is one entry");
        assert_eq!(list[0].id, "owner/repo/Q4/big-00001-of-00002.gguf");
        assert_eq!(list[0].size_bytes, (a.len() + b.len()) as u64);
        let details = mgr.model_details(&list[0].id).unwrap();
        assert_eq!(details.files.len(), 2);
        assert_eq!(details.revision.as_deref(), Some(FAKE_SHA));
        mgr.library().verify(&list[0].id).unwrap();

        mgr.delete_model(&list[0].id).unwrap();
        assert!(!env
            .dirs
            .models()
            .join("owner/repo/Q4/big-00002-of-00002.gguf")
            .exists());
    }

    #[tokio::test]
    async fn installed_model_is_conflict_unless_overwrite_requested() {
        let env = env();
        let body = gguf_file("llama", 15);
        let mgr = manager(
            &env,
            vec![repo_file("dup-Q4_K_M.gguf", &body, false)],
            body.clone(),
        );
        mgr.start_download("owner/repo", "dup-Q4_K_M.gguf", false)
            .await
            .unwrap()
            .finish()
            .await
            .unwrap();
        assert!(matches!(
            mgr.start_download("owner/repo", "dup-Q4_K_M.gguf", false)
                .await,
            Err(RaptorError::Conflict(_))
        ));
        let again = mgr
            .start_download("owner/repo", "dup-Q4_K_M.gguf", true)
            .await
            .unwrap();
        assert!(matches!(
            again.finish().await.unwrap(),
            DownloadOutcome::Completed { .. }
        ));
        assert_eq!(mgr.library().list().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn delete_model_removes_file_and_presets() {
        let env = env();
        let body = gguf_file("llama", 15);
        let mgr = manager(
            &env,
            vec![repo_file("gone-Q4_K_M.gguf", &body, false)],
            body,
        );
        mgr.start_download("owner/repo", "gone-Q4_K_M.gguf", false)
            .await
            .unwrap()
            .finish()
            .await
            .unwrap();
        let id = "owner/repo/gone-Q4_K_M.gguf";
        mgr.presets()
            .create(Preset::with_defaults(id, "default"))
            .unwrap();
        mgr.delete_model(id).unwrap();
        assert!(mgr.library().list().unwrap().is_empty());
        assert!(mgr.presets().list_for_model(id).unwrap().is_empty());
    }

    #[tokio::test]
    async fn search_and_repo_info_pass_through_validation() {
        let env = env();
        let mgr = manager(&env, vec![], vec![]);
        let hits = mgr
            .search("llama", SearchSort::Downloads, 10)
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, "owner/repo");
        assert_eq!(mgr.repo_info("owner/repo").await.unwrap().id, "owner/repo");
        assert!(mgr.repo_info("bad").await.is_err());
        assert!(mgr.repo_files("../bad").await.is_err());
    }

    #[tokio::test]
    async fn unknown_file_and_hostile_names_are_refused() {
        let env = env();
        let body = gguf_file("llama", 15);
        let mgr = manager(&env, vec![repo_file("a-Q4_K_M.gguf", &body, false)], body);
        assert!(matches!(
            mgr.start_download("owner/repo", "missing.gguf", false)
                .await,
            Err(RaptorError::NotFound(_))
        ));
        assert!(matches!(
            mgr.start_download("../etc", "a-Q4_K_M.gguf", false).await,
            Err(RaptorError::InvalidInput(_))
        ));
        assert!(matches!(
            mgr.start_download("owner/repo", "../../a.gguf", false)
                .await,
            Err(RaptorError::InvalidInput(_))
        ));
    }

    #[tokio::test]
    async fn recovered_partials_are_visible_through_the_facade() {
        let env = env();
        let body = gguf_file("llama", 15);
        let mgr = manager(&env, vec![repo_file("r-Q4_K_M.gguf", &body, false)], body);
        let report = mgr.downloads().recover().unwrap();
        assert!(report.partials.is_empty());
    }

    #[tokio::test]
    async fn check_updates_reports_changed_and_up_to_date_without_downloading() {
        let env = env();
        let body = gguf_file("llama", 15);
        let mgr = manager(
            &env,
            vec![repo_file("u-Q4_K_M.gguf", &body, false)],
            body.clone(),
        );
        mgr.start_download("owner/repo", "u-Q4_K_M.gguf", false)
            .await
            .unwrap()
            .finish()
            .await
            .unwrap();

        let same = manager(&env, vec![], vec![]);
        let checks = same.check_updates().await.unwrap();
        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].status, UpdateStatus::UpToDate);
        assert_eq!(checks[0].pinned_revision.as_deref(), Some(FAKE_SHA));

        // The repo moved on: a new head commit, no download performed.
        let moved_hf = FakeHf {
            revision: "fedcba9876543210fedcba9876543210fedcba98".into(),
            ..fake_hf(vec![])
        };
        let moved = ModelManager::new(
            &env.dirs,
            Arc::new(moved_hf),
            Arc::new(FakeFetch::new(vec![])),
            Arc::new(FixedDisk(u64::MAX)),
        )
        .unwrap();
        let checks = moved.check_updates().await.unwrap();
        assert_eq!(checks[0].status, UpdateStatus::Changed);
        assert_eq!(
            checks[0].head_revision,
            "fedcba9876543210fedcba9876543210fedcba98"
        );
        assert_eq!(
            moved.library().list().unwrap().len(),
            1,
            "never replaced automatically"
        );
    }

    #[tokio::test]
    async fn check_updates_is_a_clean_error_when_offline() {
        struct Offline;
        #[async_trait]
        impl HfApi for Offline {
            async fn search_gguf(
                &self,
                _: &str,
                _: SearchSort,
                _: u32,
            ) -> CoreResult<Vec<HfModelSummary>> {
                Err(RaptorError::Network("offline".into()))
            }
            async fn list_repo_files(&self, _: &str) -> CoreResult<Vec<RepoFile>> {
                Err(RaptorError::Network("offline".into()))
            }
            async fn list_repo_files_at(&self, _: &str, _: &str) -> CoreResult<Vec<RepoFile>> {
                Err(RaptorError::Network("offline".into()))
            }
            async fn repo_info(&self, _: &str) -> CoreResult<HfRepoInfo> {
                Err(RaptorError::Network("offline".into()))
            }
            async fn repo_snapshot(&self, _: &str) -> CoreResult<crate::hf::RepoSnapshot> {
                Err(RaptorError::Network("offline".into()))
            }
            fn file_url(&self, _: &str, _: &str) -> CoreResult<String> {
                Err(RaptorError::Network("offline".into()))
            }
            fn file_url_at(&self, _: &str, _: &str, _: &str) -> CoreResult<String> {
                Err(RaptorError::Network("offline".into()))
            }
        }
        let env = env();
        let body = gguf_file("llama", 15);
        let online = manager(&env, vec![repo_file("o-Q4_K_M.gguf", &body, false)], body);
        online
            .start_download("owner/repo", "o-Q4_K_M.gguf", false)
            .await
            .unwrap()
            .finish()
            .await
            .unwrap();
        let offline = ModelManager::new(
            &env.dirs,
            Arc::new(Offline),
            Arc::new(FakeFetch::new(vec![])),
            Arc::new(FixedDisk(u64::MAX)),
        )
        .unwrap();
        assert!(matches!(
            offline.check_updates().await,
            Err(RaptorError::Network(_))
        ));
        assert_eq!(
            offline.library().list().unwrap().len(),
            1,
            "installed models untouched"
        );
    }

    #[tokio::test]
    async fn compare_and_rank_use_installed_metadata() {
        let env = env();
        let body = gguf_file("llama", 15);
        let mgr = manager(&env, vec![repo_file("c-Q4_K_M.gguf", &body, false)], body);
        mgr.start_download("owner/repo", "c-Q4_K_M.gguf", false)
            .await
            .unwrap()
            .finish()
            .await
            .unwrap();
        let id = "owner/repo/c-Q4_K_M.gguf";
        let hw = HardwareSummary {
            ram_bytes: 16 * 1024 * 1024 * 1024,
            vram_bytes: 0,
        };
        let cmp = mgr.compare(&[id], 2048, Some(hw)).unwrap();
        assert_eq!(cmp.rows.len(), 1);
        assert_eq!(cmp.rows[0].quant_label.as_deref(), Some("Q4_K_M"));
        assert_eq!(cmp.rows[0].context_train, Some(4096));
        assert!(mgr.compare(&["owner/repo/none.gguf"], 2048, None).is_err());
        let ranked = mgr.rank_installed(Task::General, hw, 2048).unwrap();
        assert_eq!(ranked.len(), 1);
    }

    #[tokio::test]
    async fn recorded_job_states_are_terminal_after_finish() {
        let env = env();
        let body = gguf_file("llama", 15);
        let mgr = manager(&env, vec![repo_file("s-Q4_K_M.gguf", &body, false)], body);
        let handle = mgr
            .start_download("owner/repo", "s-Q4_K_M.gguf", false)
            .await
            .unwrap();
        let progress = handle.progress();
        handle.finish().await.unwrap();
        assert_eq!(progress.borrow().state, JobState::Completed);
    }
}
