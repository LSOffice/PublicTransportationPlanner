import { describe, expect, it } from "vitest";
import { migrateProject, toggleCandidate, localBundleMetrics } from "./planner-state";
import { decodeGenerationEvent } from "./stream";

describe("project migration", () => {
  it("adds v2 generation provenance to v1 projects", () => {
    const project = migrateProject({ schemaVersion: 1, id: "old", network: { stations: [], lines: [] } });
    expect(project.schemaVersion).toBe(2);
    expect(project.generationSettings.guidanceStrength).toBe(0.08);
    expect(project.corridorProvenance).toEqual({});
  });
  it("preserves v2 settings", () => {
    const settings = { guidanceStrength: 0, radialSoftCap: 2, orbitalSoftCap: 1, distributorSoftCap: 3, demandMode: "GRAVITY_ONLY" };
    expect(migrateProject({ schemaVersion: 2, id: "new", network: { stations: [], lines: [] }, generationSettings: settings }).generationSettings).toEqual(settings);
  });
});
describe("stream decoding", () => {
  it("reads a monotonic event payload", () => expect(decodeGenerationEvent('{"id":2,"type":"PLAN_FORMED","message":"ready"}').id).toBe(2));
  it("rejects malformed frames", () => expect(() => decodeGenerationEvent('{"id":0}')).toThrow());
});
describe("candidate bundle", () => {
  it("toggles corridors without duplicating IDs", () => {
    expect(toggleCandidate(["C2"], "C1", true)).toEqual(["C1", "C2"]);
    expect(toggleCandidate(["C1", "C2"], "C2", false)).toEqual(["C1"]);
  });
  it("sums local selected length", () => {
    expect(localBundleMetrics([{ id: "C1", lengthMeters: 1000 }, { id: "C2", lengthMeters: 2000 }], ["C2"]))
      .toEqual({ lineCount: 1, lengthMeters: 2000 });
  });
});
