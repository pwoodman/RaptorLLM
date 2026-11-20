param(
  [switch]$PushTag
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
  $keyPath = Join-Path $repoRoot ".secrets/tauri-updater.key"
  $publicPath = "$keyPath.pub"
  if (!(Test-Path -LiteralPath $keyPath) -or !(Test-Path -LiteralPath $publicPath)) {
    throw "The updater signing key is missing. Ask the repository maintainer to restore the original key into .secrets/tauri-updater.key. Never create a replacement after an updater release has shipped."
  }

  $tauri = Get-Content -Raw -LiteralPath "src-tauri/tauri.conf.json" | ConvertFrom-Json
  $publicKey = (Get-Content -Raw -LiteralPath $publicPath).Trim()
  if ($tauri.plugins.updater.pubkey -ne $publicKey) {
    throw "The local public key does not match src-tauri/tauri.conf.json. Stop here; replacing the signing key can strand installed users."
  }

  $version = (Get-Content -Raw -LiteralPath "package.json" | ConvertFrom-Json).version
  & node "scripts/check-release-tag.mjs" "v$version"
  if ($LASTEXITCODE -ne 0) { throw "App version check failed." }

  if (!(Get-Command gh -ErrorAction SilentlyContinue)) {
    throw "GitHub CLI is not installed. Install it from https://cli.github.com/, then run this script again."
  }
  & gh auth status --hostname github.com *> $null
  if ($LASTEXITCODE -ne 0) {
    throw "GitHub CLI sign-in is missing or expired. Run 'gh auth login --hostname github.com', choose HTTPS, sign in, then rerun this script."
  }

  $remote = (& git remote get-url origin).Trim()
  if ($LASTEXITCODE -ne 0 -or $remote -notmatch "github\.com[:/](?<repo>[^/]+/[^/.]+)(?:\.git)?$") {
    throw "Could not determine the GitHub repository from the origin remote."
  }
  $repository = $Matches.repo
  $secretNames = & gh secret list --repo $repository --json name --jq '.[].name'
  if ($LASTEXITCODE -ne 0) { throw "Could not read repository Actions secrets. Check that your GitHub account can manage repository secrets." }

  if ($secretNames -contains "TAURI_SIGNING_PRIVATE_KEY") {
    Write-Host "Signing secret already exists; leaving it untouched."
    Write-Host "If it was created from this app's key, setup is ready. Secret contents cannot be read back for safety."
  } else {
    Write-Host "Adding the updater signing key to GitHub Actions without displaying it..."
    Get-Content -Raw -LiteralPath $keyPath | & gh secret set TAURI_SIGNING_PRIVATE_KEY --repo $repository
    if ($LASTEXITCODE -ne 0) { throw "GitHub rejected the signing secret. No release tag was created." }
    Write-Host "Signing secret added."
  }

  if (!$PushTag) {
    Write-Host "Updater setup is ready. To publish v$version after committing and pushing version changes, run this script with -PushTag."
    return
  }

  $changes = & git status --porcelain
  if ($changes) { throw "Commit and push all changes first. The release tag is not created while the working tree is dirty." }
  $tag = "v$version"
  $localTag = (& git tag --list $tag)
  if ($localTag) { throw "Local tag $tag already exists; refusing to reuse a release tag." }
  $remoteTag = & git ls-remote --tags origin "refs/tags/$tag"
  if ($LASTEXITCODE -ne 0) { throw "Could not check GitHub for an existing $tag tag." }
  if ($remoteTag) { throw "GitHub tag $tag already exists; release tags cannot be reused." }

  Write-Host "This will push the current branch and publish $tag as a GitHub Release."
  $answer = Read-Host "Type $tag to continue"
  if ($answer -ne $tag) { Write-Host "Cancelled. No tag was created."; return }

  & git push
  if ($LASTEXITCODE -ne 0) { throw "Branch push failed. No release tag was created." }
  & git tag -a $tag -m "Raptor LLM $tag"
  if ($LASTEXITCODE -ne 0) { throw "Could not create release tag $tag." }
  & git push origin $tag
  if ($LASTEXITCODE -ne 0) { throw "Tag push failed. The local tag $tag remains; resolve Git access and run 'git push origin $tag'." }
  Write-Host "Tag pushed. GitHub Actions will build and publish the Windows release automatically."
} finally {
  Pop-Location
}
