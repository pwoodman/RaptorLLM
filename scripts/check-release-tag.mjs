import { readFile } from "node:fs/promises";

const tag = process.env.GITHUB_REF_NAME ?? process.argv[2];
if (!tag || !/^v\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/.test(tag)) {
  throw new Error(`Expected a semantic version tag such as v0.1.1; received ${tag ?? "nothing"}`);
}
const expected = tag.slice(1);
const packageJson = JSON.parse(await readFile("package.json", "utf8"));
const tauriJson = JSON.parse(await readFile("src-tauri/tauri.conf.json", "utf8"));
const cargoToml = await readFile("src-tauri/Cargo.toml", "utf8");
const cargoVersion = cargoToml.match(/^version\s*=\s*"([^"]+)"/m)?.[1];
const versions = { "package.json": packageJson.version, "tauri.conf.json": tauriJson.version, "Cargo.toml": cargoVersion };
const mismatches = Object.entries(versions).filter(([, version]) => version !== expected);
if (mismatches.length) {
  throw new Error(`Release ${tag} requires version ${expected} in all app manifests; found ${mismatches.map(([file, version]) => `${file}=${version}`).join(", ")}`);
}
console.log(`Release tag ${tag} matches all app versions.`);
