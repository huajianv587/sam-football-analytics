"use client";

import { Camera, CircleStop, Play, Radio, Upload, Users } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import {
  liveFramePacket,
  smallestLiveTrackAt,
  type LiveDisplayMode,
  type LiveFrame,
  type LiveStatus,
  type LiveTrack,
} from "@/lib/live";

const COLORS = ["#a7f45f", "#75d8ff", "#ffbd66", "#ce86ff", "#ff7b73", "#50e3b2"];
const TARGET_FPS = 15;

export function LiveAnalyzer() {
  const liveUrl = process.env.NEXT_PUBLIC_LIVE_WS_URL ?? "ws://127.0.0.1:8010/v1/live/ws";
  const videoRef = useRef<HTMLVideoElement>(null);
  const captureRef = useRef<HTMLCanvasElement>(null);
  const outputRef = useRef<HTMLCanvasElement>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const runningRef = useRef(false);
  const frameIdRef = useRef(0);
  const lastSentRef = useRef(0);
  const pendingRef = useRef(new Map<number, ImageBitmap>());
  const tracksRef = useRef<LiveTrack[]>([]);
  const [sourceUrl, setSourceUrl] = useState("");
  const [sourceName, setSourceName] = useState("");
  const [state, setState] = useState<"idle" | "connecting" | "ready" | "running" | "error">("idle");
  const [message, setMessage] = useState("Select a video or field camera to start.");
  const [displayMode, setDisplayMode] = useState<LiveDisplayMode>("all_masks");
  const displayModeRef = useRef<LiveDisplayMode>("all_masks");
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const selectedIdRef = useRef<number | null>(null);
  const [tracks, setTracks] = useState<LiveTrack[]>([]);
  const [performance, setPerformance] = useState({ fps: 0, latency: 0 });

  useEffect(() => () => {
    runningRef.current = false;
    socketRef.current?.close();
    for (const bitmap of pendingRef.current.values()) bitmap.close();
    const stream = videoRef.current?.srcObject as MediaStream | null;
    for (const track of stream?.getTracks() ?? []) track.stop();
  }, []);
  useEffect(() => () => { if (sourceUrl) URL.revokeObjectURL(sourceUrl); }, [sourceUrl]);
  useEffect(() => { displayModeRef.current = displayMode; }, [displayMode]);
  useEffect(() => { selectedIdRef.current = selectedId; }, [selectedId]);

  function chooseVideo(file: File | undefined) {
    if (!file) return;
    stop();
    if (sourceUrl) URL.revokeObjectURL(sourceUrl);
    const url = URL.createObjectURL(file);
    setSourceUrl(url);
    setSourceName(file.name);
    setState("idle");
    setMessage("Video ready. Connect to the A40 live service.");
  }

  async function useCamera() {
    stop();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: false });
      if (!videoRef.current) return;
      videoRef.current.srcObject = stream;
      setSourceUrl("");
      setSourceName("Field camera");
      await videoRef.current.play();
      connect();
    } catch (error) {
      setState("error");
      setMessage(error instanceof Error ? error.message : "Camera permission was denied.");
    }
  }

  function connect() {
    const video = videoRef.current;
    if (!video || (!sourceUrl && !video.srcObject)) return;
    stopSocket();
    setState("connecting");
    setMessage("Connecting to the A40 inference service…");
    const socket = new WebSocket(liveUrl);
    socketRef.current = socket;
    socket.binaryType = "arraybuffer";
    socket.onmessage = (event) => {
      const payload = JSON.parse(event.data) as LiveFrame | LiveStatus;
      if (payload.type === "status") {
        setMessage(payload.message);
        if (payload.state === "error") setState("error");
        if (payload.state === "ready") {
          setState("ready");
          runningRef.current = true;
          void video.play();
          void sendNextFrame();
        }
        return;
      }
      renderFrame(payload);
      setTracks(payload.tracks);
      tracksRef.current = payload.tracks;
      setPerformance({ fps: payload.processing_fps, latency: payload.inference_ms });
      setState("running");
      setMessage(`${payload.tracks.length} people · ${payload.processing_fps.toFixed(1)} processing FPS`);
      window.setTimeout(() => void sendNextFrame(), 0);
    };
    socket.onerror = () => {
      setState("error");
      setMessage(`Unable to reach ${liveUrl}. Start the A40 live service and SSH tunnel.`);
    };
    socket.onclose = () => { runningRef.current = false; };
  }

  async function sendNextFrame() {
    const socket = socketRef.current;
    const video = videoRef.current;
    const capture = captureRef.current;
    if (!runningRef.current || !socket || socket.readyState !== WebSocket.OPEN || !video || !capture) return;
    if (video.ended) return stop();
    if (video.readyState < 2 || !video.videoWidth) {
      return window.setTimeout(() => void sendNextFrame(), 30);
    }
    const wait = Math.max(0, 1000 / TARGET_FPS - (performanceNow() - lastSentRef.current));
    if (wait > 1) return window.setTimeout(() => void sendNextFrame(), wait);
    const width = Math.min(1280, video.videoWidth);
    const height = Math.round(video.videoHeight * width / video.videoWidth);
    capture.width = width;
    capture.height = height;
    capture.getContext("2d", { alpha: false })?.drawImage(video, 0, 0, width, height);
    const frameId = ++frameIdRef.current;
    pendingRef.current.set(frameId, await createImageBitmap(capture));
    const blob = await canvasBlob(capture);
    const jpeg = await blob.arrayBuffer();
    lastSentRef.current = performanceNow();
    socket.send(liveFramePacket(frameId, performanceNow() / 1000, jpeg));
  }

  function renderFrame(frame: LiveFrame) {
    const canvas = outputRef.current;
    const bitmap = pendingRef.current.get(frame.frame_id);
    if (!canvas || !bitmap) return;
    canvas.width = frame.width;
    canvas.height = frame.height;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.drawImage(bitmap, 0, 0, frame.width, frame.height);
    bitmap.close();
    pendingRef.current.delete(frame.frame_id);
    for (const [pendingId, pending] of pendingRef.current) {
      if (pendingId < frame.frame_id) { pending.close(); pendingRef.current.delete(pendingId); }
    }
    for (const track of frame.tracks) drawTrack(context, track, displayModeRef.current, selectedIdRef.current);
  }

  function selectAt(event: React.MouseEvent<HTMLCanvasElement>) {
    const canvas = event.currentTarget;
    const rect = canvas.getBoundingClientRect();
    const track = smallestLiveTrackAt(
      tracksRef.current,
      (event.clientX - rect.left) * canvas.width / rect.width,
      (event.clientY - rect.top) * canvas.height / rect.height,
    );
    if (!track) return;
    setSelectedId(track.track_id);
    socketRef.current?.send(JSON.stringify({ type: "select", track_id: track.track_id }));
  }

  function clearSelection() {
    setSelectedId(null);
    socketRef.current?.send(JSON.stringify({ type: "select", track_id: null }));
  }

  function stopSocket() {
    runningRef.current = false;
    socketRef.current?.close();
    socketRef.current = null;
    for (const bitmap of pendingRef.current.values()) bitmap.close();
    pendingRef.current.clear();
  }

  function stop() {
    stopSocket();
    const video = videoRef.current;
    video?.pause();
    if (video?.srcObject) {
      for (const track of (video.srcObject as MediaStream).getTracks()) track.stop();
      video.srcObject = null;
    }
    setTracks([]);
    tracksRef.current = [];
    setSelectedId(null);
    if (state !== "idle") setState("idle");
  }

  const selected = tracks.find((track) => track.track_id === selectedId) ?? null;
  const active = state === "connecting" || state === "ready" || state === "running";
  return (
    <div className="container live-shell">
      <header className="offline-heading">
        <div><span className="eyebrow">GENERIC PERSON INTELLIGENCE / LIVE</span><h1>Segment everyone. Track every ID. Refine one with SAM.</h1><p>Sport-agnostic person instance segmentation for video files and field cameras. Lightweight Masks stay live for everyone; SAM follows the selected Track.</p></div>
        <span className={`live-indicator ${state}`}><Radio size={13} /> {state.toUpperCase()}</span>
      </header>

      <div className="live-grid">
        <section className="panel live-main">
          <div className="live-stage">
            <video ref={videoRef} src={sourceUrl || undefined} muted playsInline loop className={state === "running" ? "source-hidden" : ""} />
            <canvas ref={outputRef} onClick={selectAt} />
            {!sourceName && <div className="live-empty"><Users size={38} /><strong>Select a real-time source</strong><span>Any sport · any scene containing people</span></div>}
          </div>
          <canvas ref={captureRef} hidden />
          <div className="live-controls">
            <label className="button button-secondary"><Upload size={14} /> VIDEO FILE<input hidden type="file" accept="video/*" onChange={(event) => chooseVideo(event.target.files?.[0])} /></label>
            <button className="button button-secondary" onClick={useCamera}><Camera size={14} /> FIELD CAMERA</button>
            {!active ? <button className="button button-primary" disabled={!sourceName} onClick={connect}><Play size={14} /> START LIVE</button> : <button className="button button-danger" onClick={stop}><CircleStop size={14} /> STOP</button>}
            <span className="live-source-name">{sourceName || "No source"}</span>
          </div>
          <div className="tabs live-tabs">
            {(["all_masks", "selected_only", "boxes"] as LiveDisplayMode[]).map((mode) => <button key={mode} className={`tab ${displayMode === mode ? "active" : ""}`} onClick={() => setDisplayMode(mode)}>{mode.replace("_", " ").toUpperCase()}</button>)}
          </div>
          <p className="live-message">{message}</p>
        </section>

        <aside className="live-side">
          <section className="panel live-summary">
            <span className="eyebrow">LIVE TELEMETRY</span>
            <div className="stat-grid"><div className="stat"><span>PEOPLE</span><strong>{tracks.length}</strong></div><div className="stat"><span>PROCESSING</span><strong>{performance.fps.toFixed(1)} FPS</strong></div><div className="stat"><span>INFERENCE</span><strong>{performance.latency.toFixed(0)} ms</strong></div><div className="stat"><span>MODE</span><strong>{displayMode === "boxes" ? "BOX" : "MASK"}</strong></div></div>
          </section>
          <section className="panel player-card">
            {selected ? <>
              <div className="player-identity"><span className="jersey-badge">{selected.track_id}</span><div><h2>Track {selected.track_id}</h2><p>{selected.mask_source === "sam" ? "SAM refined Mask" : "Lightweight Mask"}</p></div></div>
              <div className="metric-row"><span>Identity</span><strong>Unidentified</strong></div>
              <div className="metric-row"><span>Confidence</span><strong>{Math.round(selected.confidence * 100)}%</strong></div>
              <div className="metric-row"><span>Pixel speed</span><strong>{selected.speed_px_s.toFixed(1)} px/s</strong></div>
              <div className="metric-row"><span>Metric speed</span><strong>{selected.speed_kmh === null ? "Calibration required" : `${selected.speed_kmh.toFixed(1)} km/h`}</strong></div>
              <button className="button button-secondary live-clear" onClick={clearSelection}>CLEAR SELECTION</button>
            </> : <div className="live-unselected"><Users size={30} /><h3>Select any person</h3><p>All people use lightweight Masks. Clicking a Track activates SAM refinement for that person only.</p></div>}
          </section>
          <p className="micro muted">RTSP/HLS ingest uses the same WebSocket result protocol but runs at the GPU worker. Browser camera and video are implemented first for measurable end-to-end latency.</p>
        </aside>
      </div>
    </div>
  );
}

function drawTrack(context: CanvasRenderingContext2D, track: LiveTrack, mode: LiveDisplayMode, selectedId: number | null) {
  const selected = track.track_id === selectedId;
  const showMask = mode === "all_masks" || (mode === "selected_only" && selected);
  const color = selected ? "#a7f45f" : COLORS[Math.abs(track.track_id) % COLORS.length];
  if (showMask && track.mask.length > 2) {
    context.beginPath();
    track.mask.forEach(([x, y], index) => index ? context.lineTo(x, y) : context.moveTo(x, y));
    context.closePath();
    context.globalAlpha = selected && track.mask_source === "sam" ? 0.58 : 0.28;
    context.fillStyle = color;
    context.fill();
    context.globalAlpha = 1;
  }
  const [x1, y1, x2, y2] = track.bbox;
  context.strokeStyle = color;
  context.lineWidth = selected ? 3 : 1.2;
  context.strokeRect(x1, y1, x2 - x1, y2 - y1);
  context.fillStyle = "rgba(3, 10, 8, .85)";
  context.fillRect(x1, Math.max(0, y1 - 18), 50, 18);
  context.fillStyle = color;
  context.font = "700 12px sans-serif";
  context.fillText(`ID ${track.track_id}`, x1 + 5, Math.max(13, y1 - 5));
  if (selected && track.trail.length > 1) {
    context.beginPath();
    track.trail.forEach(([x, y], index) => index ? context.lineTo(x, y) : context.moveTo(x, y));
    context.strokeStyle = "#a7f45f";
    context.lineWidth = 3;
    context.stroke();
  }
}

function canvasBlob(canvas: HTMLCanvasElement) {
  return new Promise<Blob>((resolve, reject) => canvas.toBlob((blob) => blob ? resolve(blob) : reject(new Error("Unable to encode frame")), "image/jpeg", 0.82));
}

function performanceNow() {
  return window.performance.now();
}
