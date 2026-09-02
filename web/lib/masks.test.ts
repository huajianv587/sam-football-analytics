import { describe, expect, it } from "vitest";
import { decodeRle, decodeRleCrop, nearestFrameValue, projectMaskCrop, rleContains, smallestTrackAt } from "./masks";
import type { Track } from "./types";

describe("decodeRle", () => {
  it("decodes alternating zero and one runs", () => {
    expect(Array.from(decodeRle({ size: [2, 3], counts: [2, 3, 1] }))).toEqual([0, 0, 1, 1, 1, 0]);
  });

  it("uses RLE for canvas click hit-testing", () => {
    const mask = { size: [2, 3] as [number, number], counts: [2, 3, 1] };
    expect(rleContains(mask, 2, 0)).toBe(true);
    expect(rleContains(mask, 0, 0)).toBe(false);
  });

  it("decodes cropped RLE into full-frame coordinates", () => {
    const mask = {
      size: [4, 5] as [number, number],
      bbox: [2, 1, 4, 3] as [number, number, number, number],
      counts: [1, 2, 1],
    };
    expect(Array.from(decodeRle(mask))).toEqual([
      0, 0, 0, 0, 0,
      0, 0, 0, 1, 0,
      0, 0, 1, 0, 0,
      0, 0, 0, 0, 0,
    ]);
    expect(rleContains(mask, 3, 1)).toBe(true);
    expect(rleContains(mask, 1, 1)).toBe(false);
  });

  it("returns only the drawable crop for canvas rendering", () => {
    const decoded = decodeRleCrop({
      size: [720, 1280],
      bbox: [20, 30, 22, 32],
      counts: [1, 2, 1],
    });
    expect({ x: decoded.x, y: decoded.y, width: decoded.width, height: decoded.height }).toEqual({
      x: 20,
      y: 30,
      width: 2,
      height: 2,
    });
    expect(Array.from(decoded.pixels)).toEqual([0, 1, 1, 0]);
  });
});

describe("smallestTrackAt", () => {
  it("selects the smallest overlapping detection box", () => {
    const makeTrack = (objectId: number, bbox: [number, number, number, number]) => ({
      object_id: objectId,
      trajectory: [{ frame: 4, bbox }],
    }) as Track;
    const selected = smallestTrackAt(
      [makeTrack(1, [0, 0, 100, 100]), makeTrack(2, [20, 20, 60, 80])],
      4,
      30,
      30,
    );
    expect(selected?.object_id).toBe(2);
  });

  it("keeps detection hit targets when a gated mask has no trajectory sample", () => {
    const track = {
      object_id: 7,
      detections: [{ frame: 12, time: 0.8, bbox: [10, 20, 30, 60] }],
      trajectory: [],
    } as unknown as Track;
    expect(smallestTrackAt([track], 12, 20, 30)?.object_id).toBe(7);
  });

  it("uses the nearest verified mask frame when the current frame is gated", () => {
    const values = new Map([[4, "before"], [10, "after"]]);
    expect(nearestFrameValue(values, 7)).toEqual({ frame: 4, value: "before", interpolated: true });
    expect(nearestFrameValue(values, 10)).toEqual({ frame: 10, value: "after", interpolated: false });
  });

  it("moves and scales a verified mask crop with the current detection box", () => {
    expect(projectMaskCrop(
      { x: 12, y: 24, width: 6, height: 12 },
      [10, 20, 20, 40],
      [30, 50, 50, 90],
    )).toEqual({ x: 34, y: 58, width: 12, height: 24 });
  });
});
