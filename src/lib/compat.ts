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

/** Prefer Q4 through Q6 among viable verified files, then the largest within that tier. */
function preferredQuant(file: RepoFile): boolean {
  return /(?:^|[-_.])i?q[456](?:[-_.]|$)/i.test(file.quant_label ?? file.file_name);
}

/**
 * Picks the file to offer first. A file that fits with headroom ("Great") comes before one that
 * only just fits ("Ok"): the KV cache and the compute buffers need that room, and a model that
 * fills the card spills into system memory and slows down. Measured on a 16 GiB card with a 33B
 * model: the 10.1 GiB file ran 22.7 tok/s, the 12.2 GiB file 20.7 tok/s, the 16.4 GiB file
 * 9.1 tok/s.
 *
 * Inside one rating, Q4 to Q6 wins first: that is the quality the user asked for and it only
 * costs speed while the file still fits with headroom. Within that tier the larger file of the
 * same model comes first, because it carries the most bits. The low-bit files are ordered the
 * other way round, by size ascending: below Q4 the file name stops saying how good a quant is
 * (IQ2_XXS, IQ2_XS and IQ2_S are different quantizers, not just different bit rates), and the
 * smallest file is the fastest and leaves the most room for the KV cache.
 */
export function recommendFile(files: RepoFile[], hw: HardwareBudget): RepoFile | null {
  const FIT_ORDER: CompatRating[] = ["Great", "Ok", "Tight", "WontFit"];
  let best: RepoFile | null = null;
  let bestFit = FIT_ORDER.length;
  let bestPreferred = false;
  let bestSize = -1;
  for (const file of files) {
    if (file.split_shard || file.sha256 === null) continue;
    const { rating } = rateCompat(file.size, hw);
    if (rating !== "Great" && rating !== "Ok") continue;
    const fit = FIT_ORDER.indexOf(rating);
    const preferred = preferredQuant(file);
    const better =
      best === null ||
      fit < bestFit ||
      (fit === bestFit && preferred && !bestPreferred) ||
      (fit === bestFit && preferred === bestPreferred &&
        (preferred ? file.size > bestSize : file.size < bestSize));
    if (better) {
      best = file;
      bestFit = fit;
      bestPreferred = preferred;
      bestSize = file.size;
    }
  }
  return best;
}

/** Why a file cannot be downloaded from this app yet, or null when it can. */
export function downloadBlocker(file: RepoFile): string | null {
  if (file.split_shard) return "Split shards are not downloadable yet.";
  if (file.sha256 === null) return "No SHA-256 is published, so the file cannot be verified.";
  return null;
}
