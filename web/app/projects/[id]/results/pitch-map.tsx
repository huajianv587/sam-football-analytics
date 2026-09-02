"use client";

import { useEffect, useRef } from "react";
import type { Track } from "@/lib/types";

export function PitchMap({ tracks, selectedId }: { tracks: Track[]; selectedId: number | null }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#174c36"; ctx.fillRect(0, 0, canvas.width, canvas.height);
    const visible = selectedId === null ? tracks : tracks.filter((track) => track.object_id === selectedId);
    visible.forEach((track, trackIndex) => {
      track.trajectory.filter((point, index) => index % 3 === 0 && ("smoothed_pitch" in point ? point.smoothed_pitch : point.pitch) !== null).forEach((point) => {
        const pitch = "smoothed_pitch" in point ? point.smoothed_pitch : point.pitch;
        if (!pitch) return;
        const x = (pitch[0] / 105) * canvas.width;
        const y = (pitch[1] / 68) * canvas.height;
        const gradient = ctx.createRadialGradient(x, y, 0, x, y, 28);
        gradient.addColorStop(0, trackIndex % 2 ? "rgba(117,216,255,.32)" : "rgba(184,255,98,.35)");
        gradient.addColorStop(1, "rgba(184,255,98,0)");
        ctx.fillStyle = gradient; ctx.fillRect(x - 28, y - 28, 56, 56);
      });
    });
    ctx.strokeStyle = "rgba(255,255,255,.62)"; ctx.lineWidth = 2; ctx.strokeRect(10, 10, canvas.width - 20, canvas.height - 20);
    ctx.beginPath(); ctx.moveTo(canvas.width / 2, 10); ctx.lineTo(canvas.width / 2, canvas.height - 10); ctx.stroke();
    ctx.beginPath(); ctx.arc(canvas.width / 2, canvas.height / 2, 48, 0, Math.PI * 2); ctx.stroke();
  }, [selectedId, tracks]);
  return <canvas ref={canvasRef} className="pitch-map" width={840} height={544} />;
}
