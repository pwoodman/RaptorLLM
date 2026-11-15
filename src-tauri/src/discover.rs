//! Where other local-AI apps keep GGUF models, so a user's existing downloads can be offered
//! without hunting for the folder. Only folders that exist are reported, and scanning reads
//! headers only; nothing is copied or hashed until the user adds a model.

use std::path::{Path, PathBuf};

/// A folder worth scanning and who it belongs to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KnownFolder {
    pub source: String,
    pub path: PathBuf,
}

/// The usual model folders of LM Studio, Jan, GPT4All, the llama.cpp cache and the Hugging Face
/// cache, under `home` (and `local_app_data` on Windows). Pure: existence is checked by the caller.
pub fn known_model_folders(home: &Path, local_app_data: Option<&Path>) -> Vec<KnownFolder> {
    let mut out = vec![
        ("LM Studio", home.join(".lmstudio").join("models")),
        (
            "LM Studio",
            home.join(".cache").join("lm-studio").join("models"),
        ),
        ("Jan", home.join("jan").join("models")),
        ("Jan", home.join("jan").join("data").join("models")),
        ("llama.cpp", home.join(".cache").join("llama.cpp")),
        (
            "Hugging Face cache",
            home.join(".cache").join("huggingface").join("hub"),
        ),
    ];
    if let Some(local) = local_app_data {
        out.push(("GPT4All", local.join("nomic.ai").join("GPT4All")));
        out.push(("llama.cpp", local.join("llama.cpp")));
    }
    out.into_iter()
        .map(|(source, path)| KnownFolder {
            source: source.to_string(),
            path,
        })
        .collect()
}

/// The `downloadsFolder` of an LM Studio `settings.json`, if it names an absolute folder.
pub fn parse_downloads_folder(settings_json: &str) -> Option<PathBuf> {
    let v: serde_json::Value = serde_json::from_str(settings_json).ok()?;
    let folder = PathBuf::from(v.get("downloadsFolder")?.as_str()?);
    folder.is_absolute().then_some(folder)
}

/// Where LM Studio itself says it keeps models, from its settings files under `home`
/// (`.lmstudio/settings.json`) and `roaming_app_data` (`LM Studio/settings.json`). This finds a
/// custom folder such as `D:\AI_Storage\lmstudio`. Unreadable or missing files are skipped.
pub fn lm_studio_configured_folders(
    home: &Path,
    roaming_app_data: Option<&Path>,
) -> Vec<KnownFolder> {
    let mut files = vec![home.join(".lmstudio").join("settings.json")];
    if let Some(roaming) = roaming_app_data {
        files.push(roaming.join("LM Studio").join("settings.json"));
    }
    files
        .iter()
        .filter_map(|f| std::fs::read_to_string(f).ok())
        .filter_map(|text| parse_downloads_folder(&text))
        .map(|path| KnownFolder {
            source: "LM Studio".into(),
            path,
        })
        .collect()
}

/// Case-insensitive, separator-insensitive form of a path, for comparing folders.
fn key(p: &Path) -> String {
    p.to_string_lossy().to_lowercase().replace('\\', "/")
}

/// Known folders plus the ones the user scanned before (`added`), existing ones only, no
/// duplicates (compared case-insensitively on Windows paths).
pub fn candidate_folders(
    known: Vec<KnownFolder>,
    added: &[PathBuf],
    exists: impl Fn(&Path) -> bool,
) -> Vec<KnownFolder> {
    let mut out: Vec<KnownFolder> = Vec::new();
    let same = |a: &Path, b: &Path| key(a) == key(b);
    for folder in known.into_iter().chain(added.iter().map(|p| KnownFolder {
        source: "Scanned by you before".into(),
        path: p.clone(),
    })) {
        if exists(&folder.path) && !out.iter().any(|f| same(&f.path, &folder.path)) {
            out.push(folder);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_usual_folders_of_other_apps_are_listed() {
        let list = known_model_folders(Path::new("/h"), Some(Path::new("/local")));
        let sources: Vec<&str> = list.iter().map(|f| f.source.as_str()).collect();
        for want in [
            "LM Studio",
            "Jan",
            "llama.cpp",
            "Hugging Face cache",
            "GPT4All",
        ] {
            assert!(sources.contains(&want), "{want} missing");
        }
        assert!(list.iter().any(|f| key(&f.path) == "/h/.lmstudio/models"));
    }

    #[test]
    fn the_lm_studio_downloads_folder_is_read_from_its_settings() {
        assert_eq!(
            parse_downloads_folder(r#"{"a":1,"downloadsFolder":"D:\\AI_Storage\\lmstudio"}"#),
            Some(PathBuf::from(r"D:\AI_Storage\lmstudio"))
        );
        assert_eq!(
            parse_downloads_folder(r#"{"downloadsFolder":"relative/dir"}"#),
            None
        );
        assert_eq!(parse_downloads_folder(r#"{"other":true}"#), None);
        assert_eq!(parse_downloads_folder("not json"), None);
    }

    #[test]
    fn only_existing_folders_are_offered_without_duplicates() {
        let known = known_model_folders(Path::new("/h"), None);
        let added = vec![
            PathBuf::from("/h/.LMStudio/models"), // same folder as a known one, other case
            PathBuf::from("/data/ai"),
            PathBuf::from("/gone"),
        ];
        let exists = |p: &Path| {
            let s = key(p);
            s == "/h/.lmstudio/models" || s == "/data/ai"
        };
        let list = candidate_folders(known, &added, exists);
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].source, "LM Studio");
        assert_eq!(list[1].path, PathBuf::from("/data/ai"));
        assert_eq!(list[1].source, "Scanned by you before");
    }
}
