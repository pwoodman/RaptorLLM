import { describe, expect, it } from "vitest";
import type { HardwareInfo, RepoFile } from "../api";
import { budgetFromHardware, downloadBlocker, rateCompat, recommendFile } from "./compat";

const GIB = 1024 ** 3;
const rx9060 = { vramBytes: 16 * GIB, ramAvailBytes: 32 * GIB };

function file(name: string, size: number, overrides: Partial<RepoFile> = {}): RepoFile {
  return { file_name: name, size, sha256: "a".repeat(64), quant_label: null, split_shard: false, ...overrides };
}

describe("rateCompat", () => {
  it("rates a small model as Great when it fits VRAM with headroom", () => {
    expect(rateCompat(4 * GIB, rx9060).rating).toBe("Great");
  });

  it("rates a model that barely fits VRAM as Ok", () => {
    // Footprint is size * 1.1 + 512 MiB. 12.5 GiB -> about 14.25 GiB, between 80% and 95% of 16 GiB.
    expect(rateCompat(12.5 * GIB, rx9060).rating).toBe("Ok");
  });

  it("rates a model larger than VRAM but within VRAM plus half of RAM as Tight", () => {
    const result = rateCompat(20 * GIB, rx9060);
    expect(result.rating).toBe("Tight");
    expect(result.reason).toContain("system RAM");
  });

  it("rates a model that exceeds every budget as WontFit", () => {
    expect(rateCompat(200 * GIB, rx9060).rating).toBe("WontFit");
  });

  it("explains CPU-only fits instead of claiming GPU fit", () => {
    const cpuOnly = { vramBytes: 0, ramAvailBytes: 16 * GIB };
    const result = rateCompat(6 * GIB, cpuOnly);
    expect(result.rating).toBe("Tight");
    expect(result.reason).toContain("No GPU detected");
  });
});

describe("recommendFile", () => {
  it("picks the largest Great or Ok file", () => {
    const files = [file("small-Q4.gguf", 4 * GIB), file("mid-Q5.gguf", 9 * GIB), file("huge-Q8.gguf", 60 * GIB)];
    expect(recommendFile(files, rx9060)?.file_name).toBe("mid-Q5.gguf");
  });

  it("never recommends a file that cannot be verified or is a split shard", () => {
    const files = [
      file("unverified.gguf", 12 * GIB, { sha256: null }),
      file("shard-00001.gguf", 12 * GIB, { split_shard: true }),
    ];
    expect(recommendFile(files, rx9060)).toBeNull();
  });
});

describe("downloadBlocker", () => {
  it("explains why a file cannot be downloaded", () => {
    expect(downloadBlocker(file("a.gguf", 1, { sha256: null }))).toContain("cannot be verified");
    expect(downloadBlocker(file("a.gguf", 1, { split_shard: true }))).toContain("Split shards");
    expect(downloadBlocker(file("a.gguf", 1))).toBeNull();
  });
});

describe("budgetFromHardware", () => {
  it("uses the largest GPU and the available RAM", () => {
    const hw: HardwareInfo = {
      os: "windows",
      cpu: { brand: "cpu", cores: 8, threads: 16, features: [] },
      ram_total: 64 * GIB,
      ram_avail: 40 * GIB,
      gpus: [
        { vendor: "intel", name: "iGPU", vram_total: 2 * GIB, vram_free: null, arch: null, driver: null },
        { vendor: "amd", name: "RX", vram_total: 16 * GIB, vram_free: null, arch: "gfx1200", driver: null },
      ],
    };
    expect(budgetFromHardware(hw)).toEqual({ vramBytes: 16 * GIB, ramAvailBytes: 40 * GIB });
  });
});
