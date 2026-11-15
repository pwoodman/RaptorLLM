const UNITS = ["B", "KB", "MB", "GB", "TB"] as const;

/** Binary units (1 GB = 1024^3 bytes), the same figures the OS shows for files. */
export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return "unknown size";
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < UNITS.length - 1) {
    value /= 1024;
    unit += 1;
  }
  const digits = unit === 0 || value >= 100 ? 0 : 1;
  return `${value.toFixed(digits)} ${UNITS[unit]}`;
}

export function formatCount(n: number): string {
  return n.toLocaleString("en-US");
}
