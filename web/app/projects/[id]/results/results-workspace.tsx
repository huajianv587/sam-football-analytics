"use client";

import { Activity, Download, Eye, Footprints, Gauge, Layers3, Route, ScanLine, ShieldCheck } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Topbar } from "@/components/topbar";
import { gunzipJson } from "@/lib/masks";
import type { MaskManifest, Project, Track } from "@/lib/types";
import { MaskVideoPlayer } from "./mask-video-player";
import { PitchMap } from "./pitch-map";

type Metrics = { device: string; frames: number; objects: number; elapsed_seconds: number; effective_fps: number; gpu_peak_memory_mb: number; occlusion_events: number };

export function ResultsWorkspace({ project, tracks, urls }: { project: Project; tracks: Track[]; urls: { video: string; foreground: string; masks: string; metrics: string } }) {
  const [manifest, setManifest] = useState<MaskManifest | null>(null);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [selectedId, setSelectedId] = useState<number | null>(tracks[0]?.object_id ?? null);
  const [showMasks, setShowMasks] = useState(true);
  const [showTrajectory, setShowTrajectory] = useState(true);
  const [mode, setMode] = useState<"original" | "foreground">("original");
  const [time, setTime] = useState(0);
  const [error, setError] = useState("");

  useEffect(() => {
    Promise.all([
      fetch(urls.masks).then((response) => gunzipJson<MaskManifest>(response)),
      fetch(urls.metrics).then((response) => response.json() as Promise<Metrics>),
    ]).then(([maskData, metricData]) => { setManifest(maskData); setMetrics(metricData); }).catch((reason) => setError(reason instanceof Error ? reason.message : "Unable to load results."));
  }, [urls.masks, urls.metrics]);

  const selected = useMemo(() => tracks.find((track) => track.object_id === selectedId) ?? null, [selectedId, tracks]);
  const current = selected?.trajectory.length
    ? selected.trajectory.reduce((nearest, point) => Math.abs(point.time - time) < Math.abs(nearest.time - time) ? point : nearest)
    : null;

  return (
    <main className="page-shell"><Topbar signedIn /><div className="container content">
      <div className="page-heading"><div><span className="eyebrow">INTERACTIVE RESULT</span><h1>{project.title}</h1><p>{tracks.length} persistent subjects · select a player in the video to inspect performance</p></div><div className="nav-actions"><span className="status-pill completed"><ShieldCheck size={13} />A40 COMPLETE</span><a className="button button-primary" href={urls.foreground} download><Download size={16} /> DOWNLOAD FOREGROUND MP4</a></div></div>
      {error && <div className="panel setup-note">{error}</div>}
      <div className="result-grid">
        <div className="form-stack">
          <section className="panel result-video-panel">
            {manifest ? <MaskVideoPlayer videoUrl={mode === "original" ? urls.video : urls.foreground} manifest={manifest} tracks={tracks} selectedId={selectedId} showMasks={showMasks && mode === "original"} showTrajectory={showTrajectory} onSelect={setSelectedId} onTime={setTime} /> : <div className="video-stage" style={{ display: "grid", placeItems: "center" }}><span className="muted">Loading masks…</span></div>}
            <div className="toolbar">
              <div className="tabs"><button className={`tab ${mode === "original" ? "active" : ""}`} onClick={() => setMode("original")}>ORIGINAL</button><button className={`tab ${mode === "foreground" ? "active" : ""}`} onClick={() => setMode("foreground")}>FOREGROUND</button></div>
              <span className="toolbar-spacer" />
              <button className={`button compact ${showMasks ? "button-primary" : "button-secondary"}`} onClick={() => setShowMasks((value) => !value)}><Layers3 size={14} /> Mask</button>
              <button className={`button compact ${showTrajectory ? "button-primary" : "button-secondary"}`} onClick={() => setShowTrajectory((value) => !value)}><Route size={14} /> TRAJECTORY</button>
            </div>
          </section>
          <section className="panel"><div className="panel-title"><h2>Pitch activity heatmap</h2><span>105M × 68M</span></div><PitchMap tracks={tracks} selectedId={selectedId} /></section>
        </div>
        <aside className="form-stack">
          <section className="panel player-card">
            {selected ? <><div className="player-identity"><div className="jersey-badge">{selected.jersey_number ?? `#${selected.object_id}`}</div><div><h2>{selected.player_name ?? "Unidentified player"}</h2><p>{selected.team} · {selected.role === "referee" ? "OFFICIAL" : "PLAYER"} · TRACK ID {selected.object_id}</p></div></div>
              <div className="stat-grid">
                <div className="stat"><span><Gauge size={12} /> CURRENT SPEED</span><strong>{current?.speed_kmh.toFixed(1) ?? "0.0"} <small>km/h</small></strong></div>
                <div className="stat"><span><Activity size={12} /> TOP SPEED</span><strong>{selected.metrics.max_speed_kmh.toFixed(1)} <small>km/h</small></strong></div>
                <div className="stat"><span><Footprints size={12} /> DISTANCE</span><strong>{selected.metrics.distance_m.toFixed(1)} <small>m</small></strong></div>
                <div className="stat"><span><ScanLine size={12} /> OCR CONFIDENCE</span><strong>{Math.min(selected.metrics.ocr_confidence, 1).toFixed(2)}</strong></div>
              </div></> : <div className="empty-state"><Eye size={28} /><h2>Select a player</h2><p>Inspect identity, speed, and occlusion metrics.</p></div>}
          </section>
          {selected && <section className="panel"><div className="panel-title"><h3>OCCLUSION STABILITY</h3></div><div className="metric-row"><span>Occlusion events</span><strong>{selected.metrics.occlusion_count}</strong></div><div className="metric-row"><span>ID retained</span><strong>{selected.metrics.id_retained ? "YES" : "NO"}</strong></div><div className="metric-row"><span>Area recovery</span><strong>{(selected.metrics.area_recovery_ratio * 100).toFixed(0)}%</strong></div><div className="metric-row"><span>Recovery frames</span><strong>{selected.metrics.recovery_frames ?? "—"}</strong></div><div className="metric-row"><span>Max centroid shift</span><strong>{selected.metrics.max_centroid_jump_px.toFixed(1)} px</strong></div><div className="metric-row"><span>Average speed</span><strong>{selected.metrics.average_speed_kmh.toFixed(1)} km/h</strong></div></section>}
          {metrics && <section className="panel"><div className="panel-title"><h3>INFERENCE PROFILE</h3></div><div className="metric-row"><span>Device</span><strong>{metrics.device}</strong></div><div className="metric-row"><span>Frames processed</span><strong>{metrics.frames}</strong></div><div className="metric-row"><span>Effective FPS</span><strong>{metrics.effective_fps}</strong></div><div className="metric-row"><span>Peak GPU memory</span><strong>{metrics.gpu_peak_memory_mb.toFixed(0)} MB</strong></div><div className="metric-row"><span>Occlusion events</span><strong>{metrics.occlusion_events}</strong></div><div className="metric-row"><span>Elapsed time</span><strong>{metrics.elapsed_seconds}s</strong></div></section>}
        </aside>
      </div>
    </div></main>
  );
}
