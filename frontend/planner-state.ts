export interface GenerationSettings {
  guidanceStrength: number;
  radialSoftCap: number;
  orbitalSoftCap: number;
  distributorSoftCap: number;
  demandMode: "OBSERVED_BLEND" | "GRAVITY_ONLY";
}
export interface PlannerProjectV2 {
  schemaVersion: 2;
  id: string;
  network: { stations: unknown[]; lines: unknown[] };
  generationSettings: GenerationSettings;
  corridorProvenance: Record<string, string>;
  [key: string]: unknown;
}
export function migrateProject(project: Record<string, any>): PlannerProjectV2 {
  if (![1, 2].includes(project.schemaVersion) || !project.id || !project.network) {
    throw new Error("Unsupported planner project file");
  }
  return {
    ...project,
    schemaVersion: 2,
    generationSettings: project.generationSettings ?? { guidanceStrength: 0.08, radialSoftCap: 3, orbitalSoftCap: 3, distributorSoftCap: 3, demandMode: "OBSERVED_BLEND" },
    corridorProvenance: project.corridorProvenance ?? {},
  };
}

export interface CorridorSummary { id: string; lengthMeters: number }

export function toggleCandidate(ids: readonly string[], id: string, enabled: boolean): string[] {
  const next = new Set(ids);
  if (enabled) next.add(id); else next.delete(id);
  return [...next].sort((a, b) => Number(a.slice(1)) - Number(b.slice(1)));
}

export function localBundleMetrics(candidates: readonly CorridorSummary[], ids: readonly string[]) {
  const chosen = new Set(ids);
  return {
    lineCount: chosen.size,
    lengthMeters: candidates.filter((candidate) => chosen.has(candidate.id)).reduce((sum, candidate) => sum + candidate.lengthMeters, 0),
  };
}
