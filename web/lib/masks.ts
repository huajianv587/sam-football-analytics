import type { RleMask } from "./types";

export function decodeRle(rle: RleMask) {
  const [height, width] = rle.size;
  const output = new Uint8Array(height * width);
  let offset = 0;
  let value = 0;
  for (const count of rle.counts) {
    if (value) output.fill(1, offset, offset + count);
    offset += count;
    value = 1 - value;
  }
  return output;
}

export function rleContains(rle: RleMask, x: number, y: number) {
  const target = y * rle.size[1] + x;
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
