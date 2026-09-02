import type { RleMask, Track } from "./types";

export function decodeRle(rle: RleMask) {
  const [height, width] = rle.size;
  const decoded = decodeRleCrop(rle);
  if (!rle.bbox) return decoded.pixels;
  const output = new Uint8Array(height * width);
  for (let row = 0; row < decoded.height; row += 1) {
    output.set(
      decoded.pixels.subarray(row * decoded.width, (row + 1) * decoded.width),
      (decoded.y + row) * width + decoded.x,
    );
  }
  return output;
}

export function decodeRleCrop(rle: RleMask) {
  const [height, width] = rle.size;
  if (!rle.bbox) {
    return { pixels: decodeRuns(rle.counts, width * height), x: 0, y: 0, width, height };
  }
  const [x1, y1, x2, y2] = rle.bbox;
  const cropWidth = x2 - x1;
  const cropHeight = y2 - y1;
  return {
    pixels: decodeRuns(rle.counts, cropWidth * cropHeight),
    x: x1,
    y: y1,
    width: cropWidth,
    height: cropHeight,
  };
}

function decodeRuns(counts: number[], length: number, output = new Uint8Array(length)) {
  let offset = 0;
  let value = 0;
  for (const count of counts) {
    if (value) output.fill(1, offset, offset + count);
    offset += count;
    value = 1 - value;
  }
  return output;
}

export function rleContains(rle: RleMask, x: number, y: number) {
  let target = y * rle.size[1] + x;
  if (rle.bbox) {
    const [x1, y1, x2, y2] = rle.bbox;
    if (x < x1 || x >= x2 || y < y1 || y >= y2) return false;
    target = (y - y1) * (x2 - x1) + (x - x1);
  }
  let offset = 0;
  let value = 0;
  for (const count of rle.counts) {
    if (target < offset + count) return value === 1;
    offset += count;
    value = 1 - value;
  }
  return false;
}

export async function gunzipJson<T>(response: Response): Promise<T> {
  if (!response.body) throw new Error("The mask manifest is empty.");
  const stream = response.body.pipeThrough(new DecompressionStream("gzip"));
  return new Response(stream).json() as Promise<T>;
}

export function smallestTrackAt(tracks: Track[], frame: number, x: number, y: number) {
  return tracks
    .map((track) => ({
      track,
      point: (track.detections ?? track.trajectory).find((sample) => sample.frame === frame),
    }))
    .filter(({ point }) => point && x >= point.bbox[0] && x <= point.bbox[2] && y >= point.bbox[1] && y <= point.bbox[3])
    .sort((first, second) => {
      const a = first.point!.bbox;
      const b = second.point!.bbox;
      return (a[2] - a[0]) * (a[3] - a[1]) - (b[2] - b[0]) * (b[3] - b[1]);
    })[0]?.track ?? null;
}

export function nearestFrameValue<T>(values: Map<number, T>, frame: number) {
  const exact = values.get(frame);
  if (exact !== undefined) return { frame, value: exact, interpolated: false };
  let nearestFrame: number | null = null;
  let nearestDistance = Number.POSITIVE_INFINITY;
  for (const candidate of values.keys()) {
    const distance = Math.abs(candidate - frame);
    if (distance < nearestDistance) {
      nearestFrame = candidate;
      nearestDistance = distance;
    }
  }
  return nearestFrame === null
    ? null
    : { frame: nearestFrame, value: values.get(nearestFrame)!, interpolated: true };
}

export function projectMaskCrop(
  crop: { x: number; y: number; width: number; height: number },
  sourceBox: [number, number, number, number] | undefined,
  targetBox: [number, number, number, number] | undefined,
) {
  if (!sourceBox || !targetBox) return crop;
  const sourceWidth = Math.max(1, sourceBox[2] - sourceBox[0]);
  const sourceHeight = Math.max(1, sourceBox[3] - sourceBox[1]);
  const scaleX = (targetBox[2] - targetBox[0]) / sourceWidth;
  const scaleY = (targetBox[3] - targetBox[1]) / sourceHeight;
  return {
    x: targetBox[0] + (crop.x - sourceBox[0]) * scaleX,
    y: targetBox[1] + (crop.y - sourceBox[1]) * scaleY,
    width: crop.width * scaleX,
    height: crop.height * scaleY,
  };
}
