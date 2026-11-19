import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

// Run after a signed Windows build. No signing key is read or included in this public file.
const pkg = JSON.parse(await readFile("package.json", "utf8"));
const version = pkg.version;
const notesPath = process.argv[2];
const notes = notesPath ? (await readFile(notesPath, "utf8")).trim() : `Raptor LLM ${version}`;
const base = `https://github.com/pwoodman/RaptorLLM/releases/download/v${version}/`;
const bundleDir = resolve("target/release/bundle");
async function platform(folder, filename) {
  const artifact = resolve(bundleDir, folder, filename);
  await readFile(artifact);
  const signature = (await readFile(`${artifact}.sig`, "utf8")).trim();
  if (!signature) throw new Error(`Missing updater signature for ${filename}`);
  // GitHub normalizes spaces in uploaded release asset names to periods.
  return { signature, url: base + encodeURIComponent(filename.replace(/ /g, ".")) };
}
const nsis = await platform("nsis", `Raptor LLM_${version}_x64-setup.exe`);
const msi = await platform("msi", `Raptor LLM_${version}_x64_en-US.msi`);
const manifest = {
  version,
  notes,
  pub_date: new Date().toISOString(),
  platforms: { "windows-x86_64": nsis, "windows-x86_64-nsis": nsis, "windows-x86_64-msi": msi },
};
const output = resolve(bundleDir, "latest.json");
await writeFile(output, JSON.stringify(manifest, null, 2) + "\n");
console.log(`Created signed-package update manifest for v${version}: ${output}`);
