# Raptor LLM app updates

The desktop app checks only the GitHub Releases endpoint configured in `src-tauri/tauri.conf.json`. Update packages are signed and verified by the Tauri updater. Runtime downloads (such as llama.cpp) are a separate feature and do not update Raptor LLM itself.

## Release a version

1. Commit and push the version changes in `package.json`, `src-tauri/tauri.conf.json`, and `src-tauri/Cargo.toml`.
2. Run `./scripts/prepare-update-release.ps1`. It checks the signing key and version, safely adds the GitHub secret if it is missing, then asks before it pushes a matching `vX.Y.Z` tag.
3. The Windows Release workflow builds the NSIS installer, signs the updater bundle, and publishes the GitHub Release automatically. It rejects tags that do not match all app versions.

The guided script needs GitHub CLI (`gh`) installed and signed in with permission to manage Actions secrets and releases. If sign-in has expired, it explains how to reconnect. Users who prefer the GitHub website can configure the same secret under repository Settings → Secrets and variables → Actions, then push a matching version tag. The workflow publishes the release automatically either way.

The updater signing key is a long-lived trust anchor. The guided script reads `.secrets/tauri-updater.key` without printing it and never replaces an existing GitHub secret. The matching public key is in the Tauri config. Never commit or publish the private key; keep an encrypted offline backup because losing it prevents future updates for already-installed clients. The local key lives under the ignored `.secrets/` directory. Do not generate a replacement after publishing; changing the key needs a planned app update that trusts both keys during transition.

## First updater-enabled release

Builds made before this updater was added cannot update themselves. Users on those builds must install the first updater-enabled release once from the GitHub Release page. Later versions can be checked and installed from Settings → About.

The app never installs without an explicit user click. It checks for updates on demand, validates the signed metadata/package, downloads the selected release, then lets the Windows installer restart the app. If no update is found, the current version is shown.
