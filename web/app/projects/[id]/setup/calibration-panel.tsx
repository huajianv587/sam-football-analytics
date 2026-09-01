"use client";

import { RotateCcw } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import type { CalibrationPair } from "@/lib/types";

export function CalibrationPanel({ videoUrl, pairs, onChange }: { videoUrl: string; pairs: CalibrationPair[]; onChange: (pairs: CalibrationPair[]) => void }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const frameCanvas = useRef<HTMLCanvasElement>(null);
  const pitchCanvas = useRef<HTMLCanvasElement>(null);
  const [pending, setPending] = useState<[number, number] | null>(null);
  const [ready, setReady] = useState(false);

  const drawFrame = useCallback(() => {
    const canvas = frameCanvas.current;
    const video = videoRef.current;
    if (!canvas || !video || !ready) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    pairs.forEach((pair, index) => drawPoint(ctx, pair.video[0] * canvas.width, pair.video[1] * canvas.height, index + 1));
    if (pending) drawPoint(ctx, pending[0] * canvas.width, pending[1] * canvas.height, pairs.length + 1, "#ffbd66");
  }, [pairs, pending, ready]);

  const drawPitch = useCallback(() => {
    const canvas = pitchCanvas.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "rgba(255,255,255,.72)";
    ctx.lineWidth = 2;
    ctx.strokeRect(12, 12, canvas.width - 24, canvas.height - 24);
    ctx.beginPath(); ctx.moveTo(canvas.width / 2, 12); ctx.lineTo(canvas.width / 2, canvas.height - 12); ctx.stroke();
    ctx.beginPath(); ctx.arc(canvas.width / 2, canvas.height / 2, 54, 0, Math.PI * 2); ctx.stroke();
    ctx.strokeRect(12, canvas.height * .25, 92, canvas.height * .5);
    ctx.strokeRect(canvas.width - 104, canvas.height * .25, 92, canvas.height * .5);
    pairs.forEach((pair, index) => drawPoint(ctx, (pair.pitch[0] / 105) * canvas.width, (pair.pitch[1] / 68) * canvas.height, index + 1));
  }, [pairs]);

  useEffect(() => { drawFrame(); drawPitch(); }, [drawFrame, drawPitch]);

  function selectVideo(event: React.MouseEvent<HTMLCanvasElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    setPending([(event.clientX - rect.left) / rect.width, (event.clientY - rect.top) / rect.height]);
  }

  function selectPitch(event: React.MouseEvent<HTMLCanvasElement>) {
    if (!pending || pairs.length >= 8) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const pitch: [number, number] = [((event.clientX - rect.left) / rect.width) * 105, ((event.clientY - rect.top) / rect.height) * 68];
    onChange([...pairs, { video: pending, pitch }]);
    setPending(null);
  }

  return (
    <div>
      <video ref={videoRef} src={videoUrl} crossOrigin="anonymous" muted playsInline style={{ display: "none" }} onLoadedData={() => { setReady(true); if (videoRef.current) videoRef.current.currentTime = 0; }} />
      <div className="calibration-grid">
        <div><p className="muted micro">01 / Select a pitch landmark in the video</p><canvas ref={frameCanvas} className="pitch-canvas" width={640} height={360} onClick={selectVideo} /></div>
        <div><p className="muted micro">02 / Select the matching point on the reference pitch</p><canvas ref={pitchCanvas} className="pitch-canvas" width={630} height={408} onClick={selectPitch} /></div>
      </div>
      <div className="toolbar"><span className="muted micro">{pairs.length} point pairs set. A minimum of four is required.</span><span className="toolbar-spacer" /><button className="button button-secondary compact" disabled={!pairs.length} onClick={() => { onChange([]); setPending(null); }}><RotateCcw size={14} /> RESET</button></div>
    </div>
  );
}

function drawPoint(ctx: CanvasRenderingContext2D, x: number, y: number, index: number, color = "#b8ff62") {
  ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x, y, 8, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = "#07100e"; ctx.font = "800 10px sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(String(index), x, y);
}
