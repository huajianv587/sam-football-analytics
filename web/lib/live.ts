export type LiveDisplayMode = "all_masks" | "selected_only" | "boxes";

export type LiveTrack = {
  track_id: number;
  bbox: [number, number, number, number];
  confidence: number;
  class_name: string;
  mask: Array<[number, number]>;
  mask_source: "lightweight" | "sam";
  trail: Array<[number, number]>;
  speed_px_s: number;
  speed_kmh: number | null;
};

export type LiveFrame = {
  type: "frame";
  frame_id: number;
  width: number;
  height: number;
  inference_ms: number;
  processing_fps: number;
  selected_id: number | null;
  tracks: LiveTrack[];
};

export type LiveStatus = {
  type: "status";
  state: "ready" | "loading" | "error";
  message: string;
};

export function liveFramePacket(frameId: number, timestamp: number, jpeg: ArrayBuffer) {
  const packet = new ArrayBuffer(12 + jpeg.byteLength);
  const view = new DataView(packet);
  view.setUint32(0, frameId, false);
  view.setFloat64(4, timestamp, false);
  new Uint8Array(packet, 12).set(new Uint8Array(jpeg));
  return packet;
}

export function smallestLiveTrackAt(tracks: LiveTrack[], x: number, y: number) {
  return tracks
    .filter(({ bbox }) => x >= bbox[0] && x <= bbox[2] && y >= bbox[1] && y <= bbox[3])
    .sort((first, second) => area(first.bbox) - area(second.bbox))[0] ?? null;
}

function area(box: [number, number, number, number]) {
  return (box[2] - box[0]) * (box[3] - box[1]);
}
