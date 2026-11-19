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

Users can check and install updates manually from Settings → About. Starting with 0.1.3,
automatic idle updates are enabled by default and can be disabled there. While Raptor is open,
it checks hourly and uses Windows session-wide keyboard/mouse inactivity to require at least
five idle minutes before downloading an update. It verifies the signed package, displays a
30-second postponable countdown, then rechecks OS inactivity and Raptor work before installing
and restarting. Returning to the PC or starting work cancels the countdown. API requests,
model operations, downloads, indexing, unsaved chat text and open dialogs postpone installation.
Systems without supported OS idle detection keep manual installation available.

The interaction follows [Microsoft Edge's idle-restart approach](https://support.microsoft.com/en-us/edge/automatically-update-microsoft-edge-when-your-device-is-idle),
with an explicit idle threshold and Raptor-specific work guards. No scheduler is installed;
the background checks run only while Raptor is open. The existing updater public key remains
the trust anchor for both manual and automatic updates.

## Publish an existing locally signed build

Pushing a branch or building installers locally does not publish an update. The configured
`releases/latest/download/latest.json` endpoint exists only after a non-draft, non-prerelease
GitHub Release includes that asset. Every update also needs a higher app version than the
installed build; rebuilding version 0.1.1 cannot update an installation of 0.1.1.

When the signing key is kept locally, build both Windows installers with the existing key, then
run `node scripts/create-update-manifest.mjs` with an optional release notes file path. It creates `target/release/bundle/latest.json`
from the generated installer signatures and versioned download URLs. The manifest supports
existing NSIS and MSI installations and the generic Windows x64 target.

Create a draft GitHub Release at the build's committed source, upload both installers, both
`.sig` files and `latest.json`, then publish it as the latest stable release. Verify the public
`latest.json` endpoint and its installer URLs after publication. The private signing key stays
local and must never be included among the release assets.
