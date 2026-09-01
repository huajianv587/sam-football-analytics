import { describe, expect, it } from "vitest";
import { decodeRle, rleContains } from "./masks";

describe("decodeRle", () => {
  it("decodes alternating zero and one runs", () => {
    expect(Array.from(decodeRle({ size: [2, 3], counts: [2, 3, 1] }))).toEqual([0, 0, 1, 1, 1, 0]);
  });

  it("uses RLE for canvas click hit-testing", () => {
    const mask = { size: [2, 3] as [number, number], counts: [2, 3, 1] };
    expect(rleContains(mask, 2, 0)).toBe(true);
    expect(rleContains(mask, 0, 0)).toBe(false);
  });
});
