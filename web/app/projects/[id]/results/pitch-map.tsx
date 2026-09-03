"use client";

import { useEffect, useRef } from "react";
import type { Track, TrajectoryPoint } from "@/lib/types";

const FIELD = { width: 105, height: 68 };
const GRID = { columns: 84, rows: 54 };

function metricPoint(point: TrajectoryPoint) {
  const value = point.smoothed_pitch ?? point.pitch;
  if (!value || !Number.isFinite(value[0]) || !Number.isFinite(value[1])) return null;
  if (value[0] < 0 || value[0] > FIELD.width || value[1] < 0 || value[1] > FIELD.height) return null;
  return value;
}

function activityColor(value: number) {
  if (value <= 0) return "rgb(18, 61, 47)";
  const stops = [
    [0, [20, 83, 65]], [0.2, [23, 116, 95]], [0.45, [39, 164, 137]],
    [0.7, [94, 205, 130]], [0.88, [184, 242, 99]], [1, [255, 238, 133]],
  ] as const;
  for (let index = 1; index < stops.length; index += 1) {
    const [upper, upperColor] = stops[index];
    if (value <= upper) {
      const [lower, lowerColor] = stops[index - 1];
      const fraction = (value - lower) / (upper - lower);
      const rgb = lowerColor.map((channel, channelIndex) => Math.round(channel + (upperColor[channelIndex] - channel) * fraction));
      return `rgb(${rgb.join(",")})`;
    }
  }
  return "rgb(255,238,133)";
}

function drawPitch(ctx: CanvasRenderingContext2D, width: number, height: number) {
  const pad = 14;
  const left = pad, top = pad, right = width - pad, bottom = height - pad;
  ctx.strokeStyle = "rgba(230, 255, 242, .86)";
  ctx.lineWidth = 2;
  ctx.strokeRect(left, top, right - left, bottom - top);
  ctx.beginPath(); ctx.moveTo((left + right) / 2, top); ctx.lineTo((left + right) / 2, bottom); ctx.stroke();
  ctx.beginPath(); ctx.arc((left + right) / 2, (top + bottom) / 2, (bottom - top) * 0.13, 0, Math.PI * 2); ctx.stroke();
  ctx.fillStyle = "rgba(230,255,242,.86)"; ctx.beginPath(); ctx.arc((left + right) / 2, (top + bottom) / 2, 3, 0, Math.PI * 2); ctx.fill();
  for (const side of ["left", "right"] as const) {
    const areaWidth = (right - left) * (16.5 / FIELD.width);
    const x = side === "left" ? left : right - areaWidth;
    ctx.strokeRect(x, top + (bottom - top) * (13.84 / FIELD.height), areaWidth, (bottom - top) * (40.32 / FIELD.height));
    const goalWidth = (right - left) * (5.5 / FIELD.width);
    const goalX = side === "left" ? left - goalWidth : right;
    ctx.strokeRect(goalX, top + (bottom - top) * (24.84 / FIELD.height), goalWidth, (bottom - top) * (18.32 / FIELD.height));
  }
}

function metricSegments(track: Track) {
  const segments: Array<Array<[number, number]>> = [];
  let current: Array<[number, number]> = [];
  let previousFrame: number | null = null;
  for (const point of track.trajectory) {
    const pitch = metricPoint(point);
    if (!pitch) { if (current.length) segments.push(current); current = []; previousFrame = null; continue; }
    if (previousFrame !== null && point.frame - previousFrame > 15) {
      if (current.length) segments.push(current);
      current = [];
    }
    current.push(pitch);
    previousFrame = point.frame;
  }
  if (current.length) segments.push(current);
  return segments;
}

export function PitchMap({ tracks, selectedId }: { tracks: Track[]; selectedId: number | null }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const selectedTracks = selectedId === null ? tracks : tracks.filter((track) => track.object_id === selectedId);
  const validSamples = selectedTracks.reduce((total, track) => total + track.trajectory.reduce((count, point) => count + (metricPoint(point) ? 1 : 0), 0), 0);
  const activeTracks = selectedTracks.filter((track) => track.trajectory.some(metricPoint)).length;

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context) return;
    const width = canvas.width, height = canvas.height, inset = 14;
    const pitchWidth = width - inset * 2, pitchHeight = height - inset * 2;
    context.clearRect(0, 0, width, height);
    context.fillStyle = "#123f30"; context.fillRect(0, 0, width, height);

    const density = new Float32Array(GRID.columns * GRID.rows);
    let maximum = 0;
    for (const track of selectedTracks) {
      for (const point of track.trajectory) {
        const pitch = metricPoint(point);
        if (!pitch) continue;
        const column = Math.min(GRID.columns - 1, Math.max(0, Math.floor((pitch[0] / FIELD.width) * GRID.columns)));
        const row = Math.min(GRID.rows - 1, Math.max(0, Math.floor((pitch[1] / FIELD.height) * GRID.rows)));
        const index = row * GRID.columns + column;
        density[index] += 1; maximum = Math.max(maximum, density[index]);
      }
    }
    const blurred = new Float32Array(density.length);
    for (let row = 0; row < GRID.rows; row += 1) {
      for (let column = 0; column < GRID.columns; column += 1) {
        let total = 0, weight = 0;
        for (let dy = -1; dy <= 1; dy += 1) for (let dx = -1; dx <= 1; dx += 1) {
          const sourceRow = row + dy, sourceColumn = column + dx;
          if (sourceRow < 0 || sourceRow >= GRID.rows || sourceColumn < 0 || sourceColumn >= GRID.columns) continue;
          const distance = Math.abs(dx) + Math.abs(dy), sourceWeight = distance === 0 ? 4 : distance === 1 ? 2 : 1;
          total += density[sourceRow * GRID.columns + sourceColumn] * sourceWeight; weight += sourceWeight;
        }
        blurred[row * GRID.columns + column] = weight ? total / weight : 0;
      }
    }
    const scale = maximum || 1, cellWidth = pitchWidth / GRID.columns, cellHeight = pitchHeight / GRID.rows;
    for (let row = 0; row < GRID.rows; row += 1) for (let column = 0; column < GRID.columns; column += 1) {
      const ratio = Math.min(1, Math.sqrt(blurred[row * GRID.columns + column] / scale));
      if (ratio <= 0.02) continue;
      context.fillStyle = activityColor(ratio); context.globalAlpha = 0.24 + ratio * 0.66;
      context.fillRect(inset + column * cellWidth, inset + row * cellHeight, cellWidth + 0.5, cellHeight + 0.5);
    }
    context.globalAlpha = 1;
    const toCanvas = (point: [number, number]) => [inset + (point[0] / FIELD.width) * pitchWidth, inset + (point[1] / FIELD.height) * pitchHeight] as const;
    for (const track of selectedTracks) {
      const selected = track.object_id === selectedId;
      const segments = metricSegments(track);
      context.strokeStyle = selected ? "#ffffff" : "rgba(218,255,195,.24)"; context.lineWidth = selected ? 3 : 1;
      context.lineCap = "round"; context.lineJoin = "round";
      for (const path of segments) {
        if (path.length < 2) continue;
        context.beginPath();
        path.forEach((point, index) => { const [x, y] = toCanvas(point); if (index === 0) context.moveTo(x, y); else context.lineTo(x, y); }); context.stroke();
      }
      const lastPoint = segments.at(-1)?.at(-1);
      if (selected && lastPoint) { const [x, y] = toCanvas(lastPoint); context.fillStyle = "#ffffff"; context.beginPath(); context.arc(x, y, 4, 0, Math.PI * 2); context.fill(); }
    }
    drawPitch(context, width, height);
  }, [selectedId, selectedTracks]);

  return <div className="pitch-map-wrap"><div className="pitch-map-canvas-wrap"><canvas ref={canvasRef} className="pitch-map" width={840} height={544} aria-label="Pitch activity heatmap" /><div className="pitch-map-scale" aria-hidden="true"><span>LOW</span><i /><span>HIGH</span></div></div><div className="pitch-map-meta"><span>{selectedId === null ? "ALL TRACKS" : `TRACK ${selectedId}`}</span><span>{activeTracks} active · {validSamples.toLocaleString()} metric samples</span></div>{validSamples === 0 && <p className="pitch-map-empty">Metric calibration unavailable · pixel trajectory remains available in the video view.</p>}</div>;
}
