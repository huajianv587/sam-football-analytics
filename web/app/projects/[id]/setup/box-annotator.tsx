"use client";

import { RotateCcw, Undo2 } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import type { PromptBox } from "@/lib/types";

const COLORS = ["#b8ff62", "#75d8ff", "#ffbd66", "#ce86ff", "#ff7b73", "#50e3b2"];

export function BoxAnnotator({ videoUrl, boxes, onChange }: { videoUrl: string; boxes: PromptBox[]; onChange: (boxes: PromptBox[]) => void }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const startRef = useRef<[number, number] | null>(null);
  const draftRef = useRef<[number, number, number, number] | null>(null);
  const [draft, setDraft] = useState<[number, number, number, number] | null>(null);
  const [ready, setReady] = useState(false);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video || !ready) return;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    [...boxes, ...(draft ? [{ object_id: boxes.length + 1, box: draft }] : [])].forEach((item, index) => {
      const [x1, y1, x2, y2] = item.box;
      const color = COLORS[index % COLORS.length];
      context.strokeStyle = color;
      context.lineWidth = 3;
      context.fillStyle = `${color}22`;
      context.fillRect(x1 * canvas.width, y1 * canvas.height, (x2 - x1) * canvas.width, (y2 - y1) * canvas.height);
      context.strokeRect(x1 * canvas.width, y1 * canvas.height, (x2 - x1) * canvas.width, (y2 - y1) * canvas.height);
      context.fillStyle = color;
      context.font = "700 16px sans-serif";
      context.fillText(`ID ${item.object_id}`, x1 * canvas.width + 6, y1 * canvas.height + 20);
    });
  }, [boxes, draft, ready]);

  useEffect(() => draw(), [draw]);

  function point(event: React.PointerEvent<HTMLCanvasElement>): [number, number] {
    const rect = event.currentTarget.getBoundingClientRect();
    return [(event.clientX - rect.left) / rect.width, (event.clientY - rect.top) / rect.height];
  }

  function pointerDown(event: React.PointerEvent<HTMLCanvasElement>) {
    event.currentTarget.setPointerCapture(event.pointerId);
    startRef.current = point(event);
  }

  function pointerMove(event: React.PointerEvent<HTMLCanvasElement>) {
    if (!startRef.current) return;
    const [sx, sy] = startRef.current;
    const [x, y] = point(event);
    const nextDraft: [number, number, number, number] = [Math.min(sx, x), Math.min(sy, y), Math.max(sx, x), Math.max(sy, y)];
    draftRef.current = nextDraft;
    setDraft(nextDraft);
  }

  function pointerUp() {
    const completedDraft = draftRef.current;
    if (completedDraft && completedDraft[2] - completedDraft[0] > 0.01 && completedDraft[3] - completedDraft[1] > 0.02) {
      onChange([...boxes, { object_id: boxes.length + 1, box: completedDraft }]);
    }
    startRef.current = null;
    draftRef.current = null;
    setDraft(null);
  }

  return (
    <>
      <video ref={videoRef} src={videoUrl} crossOrigin="anonymous" muted playsInline style={{ display: "none" }} onLoadedData={() => { if (videoRef.current) videoRef.current.currentTime = 0; setReady(true); }} />
      <div className="video-stage">
        <canvas ref={canvasRef} width={1280} height={720} onPointerDown={pointerDown} onPointerMove={pointerMove} onPointerUp={pointerUp} />
      </div>
      <div className="toolbar">
        <span className="muted micro">Drag to draw a box. Each subject receives a persistent object ID.</span><span className="toolbar-spacer" />
        <button className="button button-secondary compact" disabled={!boxes.length} onClick={() => onChange(boxes.slice(0, -1))}><Undo2 size={14} /> UNDO</button>
        <button className="button button-secondary compact" disabled={!boxes.length} onClick={() => onChange([])}><RotateCcw size={14} /> CLEAR</button>
      </div>
    </>
  );
}

export function promptColor(index: number) { return COLORS[index % COLORS.length]; }
