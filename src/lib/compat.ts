import type { HardwareInfo, RepoFile } from "../api";

export type CompatRating = "Great" | "Ok" | "Tight" | "WontFit";

export interface HardwareBudget {
  /** Largest GPU VRAM reported. 0 when no GPU is detected. */
  vramBytes: number;
  ramAvailBytes: number;
}

export interface CompatResult {
  rating: CompatRating;
  footprintBytes: number;
  reason: string;
}

const MIB = 1024 * 1024;
/** Memory beyond the weights (KV cache, buffers). Deliberately rough. */
const OVERHEAD_RATIO = 0.1;
const FIXED_RESERVE_BYTES = 512 * MIB;

function estimateFootprint(sizeBytes: number): number {
  return sizeBytes * (1 + OVERHEAD_RATIO) + FIXED_RESERVE_BYTES;
}

export function budgetFromHardware(hw: HardwareInfo): HardwareBudget {
  const vramBytes = hw.gpus.reduce((max, gpu) => Math.max(max, gpu.vram_total), 0);
  return { vramBytes, ramAvailBytes: hw.ram_avail };
}

/**
 * Rates a GGUF file from its size alone. This is a planning hint for the browser, not the
 * runtime's memory estimator, which needs the model's layer and KV-cache dimensions.
 */
export function rateCompat(sizeBytes: number, hw: HardwareBudget): CompatResult {
  const footprintBytes = estimateFootprint(sizeBytes);
  const vram = Math.max(0, hw.vramBytes);
  const ram = Math.max(0, hw.ramAvailBytes);

  if (vram > 0 && footprintBytes <= vram * 0.8) {
    return { rating: "Great", footprintBytes, reason: "Fits in VRAM with headroom." };
  }
  if (vram > 0 && footprintBytes <= vram * 0.95) {
    return { rating: "Ok", footprintBytes, reason: "Fits in VRAM with little headroom." };
  }
  if (footprintBytes <= vram + ram * 0.5) {
    const reason =
      vram > 0
        ? "Only partly fits in VRAM. The rest runs from system RAM, which is slower."
        : "No GPU detected. Runs from system RAM on the CPU, which is slower.";
    return { rating: "Tight", footprintBytes, reason };
  }
  return { rating: "WontFit", footprintBytes, reason: "Larger than VRAM plus half of available RAM." };
}

/** The largest file rated Great or Ok. Split shards and unverifiable files are never recommended. */
export function recommendFile(files: RepoFile[], hw: HardwareBudget): RepoFile | null {
  let best: RepoFile | null = null;
  for (const file of files) {
    if (file.split_shard || file.sha256 === null) continue;
    const { rating } = rateCompat(file.size, hw);
    if (rating !== "Great" && rating !== "Ok") continue;
    if (best === null || file.size > best.size) best = file;
  }
  return best;
}

/** Why a file cannot be downloaded from this app yet, or null when it can. */
export function downloadBlocker(file: RepoFile): string | null {
  if (file.split_shard) return "Split shards are not downloadable yet.";
  if (file.sha256 === null) return "No SHA-256 is published, so the file cannot be verified.";
  return null;
}
