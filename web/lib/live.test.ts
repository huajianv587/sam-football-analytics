import { describe, expect, it } from "vitest";
import { liveFramePacket, smallestLiveTrackAt, type LiveTrack } from "./live";

const track = (trackId: number, bbox: [number, number, number, number]): LiveTrack => ({
  track_id: trackId,
  bbox,
  confidence: 0.9,
  class_name: "person",
  mask: [],
  mask_source: "lightweight",
  trail: [],
  speed_px_s: 0,
  speed_kmh: null,
});

describe("live protocol", () => {
  it("encodes the binary frame header in network byte order", () => {
    const packet = liveFramePacket(42, 12.5, new Uint8Array([1, 2, 3]).buffer);
    const view = new DataView(packet);
    expect(view.getUint32(0, false)).toBe(42);
    expect(view.getFloat64(4, false)).toBe(12.5);
    expect(Array.from(new Uint8Array(packet, 12))).toEqual([1, 2, 3]);
  });

  it("selects the smallest overlapping live track", () => {
    expect(smallestLiveTrackAt([
      track(1, [0, 0, 100, 100]),
      track(2, [20, 20, 40, 60]),
    ], 30, 30)?.track_id).toBe(2);
  });
});
