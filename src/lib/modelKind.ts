export interface ModelKindMetadata {
  architecture: string | null;
  expert_count?: number | null;
}

export function modelKind(meta: ModelKindMetadata): string {
  const experts = meta.expert_count ?? null;
  const isMoe = (experts !== null && experts > 1) || /moe|exp/i.test(meta.architecture ?? "");
  if (!isMoe) return "Dense";
  return experts ? "MoE · " + experts + " experts" : "MoE";
}
