"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { decodeRle, rleContains } from "@/lib/masks";
import type { MaskManifest, Track } from "@/lib/types";

const COLORS = [[184,255,98], [117,216,255], [255,189,102], [206,134,255], [255,123,115], [80,227,178]];

export function MaskVideoPlayer({
  videoUrl,
  manifest,
  tracks,
  selectedId,
  showMasks,
  showTrajectory,
  onSelect,
  onTime,
}: {
  videoUrl: string;
  manifest: MaskManifest;
  tracks: Track[];
  selectedId: number | null;
  showMasks: boolean;
  showTrajectory: boolean;
  onSelect: (objectId: number) => void;
  onTime: (time: number) => void;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [frameIndex, setFrameIndex] = useState(0);
  const frame = manifest.frames[Math.min(frameIndex, manifest.frames.length - 1)];
  const trackMap = useMemo(() => new Map(tracks.map((track) => [track.object_id, track])), [tracks]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context || !frame) return;
    context.clearRect(0, 0, canvas.width, canvas.height);
    if (showMasks) {
      const image = context.createImageData(canvas.width, canvas.height);
      Object.entries(frame.objects).forEach(([idText, rle]) => {
        const id = Number(idText);
        const mask = decodeRle(rle);
        const color = COLORS[(id - 1) % COLORS.length];
        const alpha = selectedId === null || selectedId === id ? 125 : 42;
        for (let pixel = 0; pixel < mask.length; pixel += 1) {
          if (!mask[pixel]) continue;
          const offset = pixel * 4;
          image.data[offset] = color[0]; image.data[offset + 1] = color[1]; image.data[offset + 2] = color[2]; image.data[offset + 3] = alpha;
        }
      });
      context.putImageData(image, 0, 0);
    }
    if (showTrajectory && selectedId !== null) {
      const track = trackMap.get(selectedId);
      const path = track?.trajectory.filter((point) => point.frame <= frameIndex) ?? [];
      if (path.length > 1) {
        context.strokeStyle = "#b8ff62"; context.lineWidth = 4; context.lineCap = "round"; context.lineJoin = "round";
        context.beginPath(); path.forEach((point, index) => index ? context.lineTo(...point.foot) : context.moveTo(...point.foot)); context.stroke();
      }
    }
    Object.keys(frame.objects).forEach((idText) => {
      const id = Number(idText);
      const point = trackMap.get(id)?.trajectory.find((sample) => sample.frame === frameIndex);
      if (!point) return;
      const labelY = Math.max(0, point.bbox[1] - 27);
      context.font = "700 18px sans-serif";
      context.fillStyle = "rgba(5,12,10,.86)";
      context.fillRect(point.bbox[0], labelY, 58, 25);
      context.fillStyle = `rgb(${COLORS[(id - 1) % COLORS.length].join(",")})`;
      context.fillText(`ID ${id}`, point.bbox[0] + 6, labelY + 19);
    });
  }, [frame, frameIndex, selectedId, showMasks, showTrajectory, trackMap]);

  useEffect(() => draw(), [draw]);

  function syncTime() {
    const time = videoRef.current?.currentTime ?? 0;
    setFrameIndex(Math.min(Math.round(time * manifest.fps), manifest.frames.length - 1));
    onTime(time);
  }

  function click(event: React.MouseEvent<HTMLCanvasElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    const x = Math.floor(((event.clientX - rect.left) / rect.width) * manifest.width);
    const y = Math.floor(((event.clientY - rect.top) / rect.height) * manifest.height);
    for (const [id, rle] of Object.entries(frame?.objects ?? {})) {
      if (rleContains(rle, x, y)) { onSelect(Number(id)); return; }
    }
  }

  return (
    <div className="video-stage">
      <video ref={videoRef} src={videoUrl} crossOrigin="anonymous" controls playsInline onTimeUpdate={syncTime} onSeeked={syncTime} />
      <canvas ref={canvasRef} width={manifest.width} height={manifest.height} onClick={click} />
    </div>
  );
}
