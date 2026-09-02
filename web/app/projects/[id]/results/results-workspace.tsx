"use client";

import { Activity, Download, Eye, Footprints, Gauge, Layers3, Route, ScanLine, ShieldCheck, Sparkles } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Topbar } from "@/components/topbar";
import { gunzipJson } from "@/lib/masks";
import type { MaskManifest, Project, RosterPlayer, Track, TrackMaskManifest } from "@/lib/types";
import { MaskVideoPlayer } from "./mask-video-player";
import { PitchMap } from "./pitch-map";

type Metrics = {
  device: string;
  frames: number;
  objects: number;
  elapsed_seconds: number;
  effective_fps: number;
  gpu_peak_memory_mb: number;
  gpu_memory_used_peak_mb: number;
  gpu_utilization_average_percent: number;
  occlusion_events: number;
  calibration_valid_rate: number;
};

type ResultUrls = {
  video: string;
  foreground: string;
  metrics: string;
  legacyMasks: string | null;
  masksByTrack: Record<number, string>;
};

type RefinementResult = {
  state: "base_ready" | "queued" | "running" | "large_ready" | "failed";
  mask_url: string | null;
  message: string | null;
};

export function ResultsWorkspace({ project, tracks, roster, urls }: { project: Project; tracks: Track[]; roster: RosterPlayer[]; urls: ResultUrls }) {
  const [trackData, setTrackData] = useState(tracks);
  const [maskCache, setMaskCache] = useState(() => new Map<number, TrackMaskManifest>());
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [showMasks, setShowMasks] = useState(true);
  const [showTrajectory, setShowTrajectory] = useState(true);
  const [mode, setMode] = useState<"original" | "foreground">("original");
  const [time, setTime] = useState(0);
  const [error, setError] = useState("");
  const legacyCache = useRef<MaskManifest | null>(null);
  const apiUrl = process.env.NEXT_PUBLIC_INFERENCE_API_URL ?? "http://127.0.0.1:8000";

  useEffect(() => {
    fetch(urls.metrics)
      .then((response) => response.json() as Promise<Metrics>)
      .then(setMetrics)
      .catch((reason) => setError(reason instanceof Error ? reason.message : "Unable to load metrics."));
  }, [urls.metrics]);

  useEffect(() => {
    if (selectedId === null) return;
    const cached = maskCache.get(selectedId);
    if (cached) return;
    let cancelled = false;
    async function loadMask() {
      let next: TrackMaskManifest;
      if (urls.masksByTrack[selectedId!]) {
        next = await fetch(urls.masksByTrack[selectedId!]).then((response) => gunzipJson<TrackMaskManifest>(response));
      } else if (urls.legacyMasks) {
        legacyCache.current ??= await fetch(urls.legacyMasks).then((response) => gunzipJson<MaskManifest>(response));
        const legacy = legacyCache.current!;
        const frames = legacy.frames.flatMap((frame) => frame.objects[String(selectedId)] ? [{ index: frame.index, rle: frame.objects[String(selectedId)] }] : []);
        next = { track_id: selectedId!, fps: legacy.fps, width: legacy.width, height: legacy.height, first_frame: frames[0]?.index ?? 0, last_frame: frames.at(-1)?.index ?? 0, frames };
      } else {
        throw new Error("This track has no mask artifact.");
      }
      if (!cancelled) {
        setMaskCache((current) => new Map(current).set(selectedId!, next));
      }
    }
    loadMask().catch((reason) => !cancelled && setError(reason instanceof Error ? reason.message : "Unable to load player mask."));
    return () => { cancelled = true; };
  }, [maskCache, selectedId, urls.legacyMasks, urls.masksByTrack]);

  const manifest = selectedId === null ? null : maskCache.get(selectedId) ?? null;
  const loadingMask = selectedId !== null && manifest === null;

  const selected = useMemo(() => trackData.find((track) => track.object_id === selectedId) ?? null, [selectedId, trackData]);
  const current = selected?.trajectory.find((point) => point.frame === Math.round(time * 15)) ?? null;

  useEffect(() => {
    if (!selected || !["queued", "running"].includes(selected.metrics.mask_refinement_status ?? "")) return;
    let cancelled = false;
    async function refreshRefinement() {
      const response = await fetch(`${apiUrl}/v1/projects/${project.id}/tracks/${selected!.object_id}/refine`);
      const result = await response.json() as RefinementResult & { detail?: string };
      if (!response.ok) throw new Error(result.detail ?? "Unable to read refinement status.");
      if (cancelled) return;
      setTrackData((items) => items.map((track) => track.object_id === selected!.object_id ? {
        ...track,
        metrics: {
          ...track.metrics,
          mask_model_tier: result.state === "large_ready" ? "large" : track.metrics.mask_model_tier,
          mask_refinement_status: result.state,
          mask_refinement_error: result.message,
        },
      } : track));
      if (result.state === "large_ready" && result.mask_url) {
        const refined = await fetch(result.mask_url, { cache: "no-store" }).then((maskResponse) => gunzipJson<TrackMaskManifest>(maskResponse));
        if (!cancelled) setMaskCache((currentCache) => new Map(currentCache).set(selected!.object_id, refined));
      }
      if (result.state === "failed") setError(result.message ?? "SAM Large refinement failed.");
    }
    void refreshRefinement().catch((reason) => !cancelled && setError(reason instanceof Error ? reason.message : "Unable to refresh refinement."));
    const timer = window.setInterval(() => {
      void refreshRefinement().catch((reason) => !cancelled && setError(reason instanceof Error ? reason.message : "Unable to refresh refinement."));
    }, 4000);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [apiUrl, project.id, selected]);

  async function changeIdentity(value: string) {
    if (!selected) return;
    const response = await fetch(`${apiUrl}/v1/projects/${project.id}/tracks/${selected.object_id}/identity`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ roster_id: value ? Number(value) : null }),
    });
    const result = await response.json();
    if (!response.ok) return setError(result.detail ?? "Unable to update identity.");
    setTrackData((items) => items.map((track) => track.object_id === selected.object_id ? { ...track, ...result } : track));
  }

  async function refineSelectedMask() {
    if (!selected) return;
    setError("");
    setTrackData((items) => items.map((track) => track.object_id === selected.object_id ? {
      ...track,
      metrics: { ...track.metrics, mask_refinement_status: "queued" },
    } : track));
    const response = await fetch(`${apiUrl}/v1/projects/${project.id}/tracks/${selected.object_id}/refine`, { method: "POST" });
    const result = await response.json() as RefinementResult & { detail?: string };
    if (!response.ok) setError(result.detail ?? "Unable to submit SAM Large refinement.");
  }

  function selectTrack(objectId: number) {
    setError("");
    setSelectedId(objectId);
  }

  const roleLabel = selected?.role === "referee" ? "OFFICIAL" : selected?.role === "goalkeeper" ? "GOALKEEPER" : "PLAYER";
  const teamLabel = selected?.team && selected.team.toLowerCase() !== "unknown" ? `${selected.team} · ` : "";
  return (
    <main className="page-shell"><Topbar /><div className="container content">
      <div className="page-heading"><div><span className="eyebrow">INTERACTIVE RESULT</span><h1>{project.title}</h1><p>{trackData.length} persistent tracks · click a detection box to load one pixel mask</p></div><div className="nav-actions"><span className="status-pill completed"><ShieldCheck size={13} />A40 COMPLETE</span><a className="button button-primary" href={urls.foreground} download><Download size={16} /> DOWNLOAD FOREGROUND MP4</a></div></div>
      {error && <div className="panel setup-note">{error}</div>}
      <div className="result-grid">
        <div className="form-stack">
          <section className="panel result-video-panel">
            <MaskVideoPlayer videoUrl={mode === "original" ? urls.video : urls.foreground} manifest={manifest} tracks={trackData} selectedId={selectedId} showMasks={showMasks && mode === "original"} showTrajectory={showTrajectory} onSelect={selectTrack} onTime={setTime} />
            <div className="toolbar"><div className="tabs"><button className={`tab ${mode === "original" ? "active" : ""}`} onClick={() => setMode("original")}>ORIGINAL</button><button className={`tab ${mode === "foreground" ? "active" : ""}`} onClick={() => setMode("foreground")}>FOREGROUND</button></div><span className="toolbar-spacer">{loadingMask ? "LOADING SELECTED MASK…" : selectedId === null ? "CLICK A TRACK" : `TRACK ${selectedId} CACHED`}</span><button className={`button compact ${showMasks ? "button-primary" : "button-secondary"}`} onClick={() => setShowMasks((value) => !value)}><Layers3 size={14} /> MASK</button><button className={`button compact ${showTrajectory ? "button-primary" : "button-secondary"}`} onClick={() => setShowTrajectory((value) => !value)}><Route size={14} /> TRAJECTORY</button></div>
          </section>
          <section className="panel"><div className="panel-title"><h2>Pitch activity heatmap</h2><span>{selected?.metrics.metric_calibration_available === false ? "METRIC CALIBRATION UNAVAILABLE" : "105M × 68M"}</span></div><PitchMap tracks={trackData} selectedId={selectedId} /></section>
        </div>
        <aside className="form-stack">
          <section className="panel player-card">
            {selected ? <><div className="player-identity"><div className="jersey-badge">{selected.jersey_number ?? `#${selected.object_id}`}</div><div><h2>{selected.player_name ?? "Unidentified"}</h2><p>{teamLabel}{roleLabel} · TRACK ID {selected.object_id}</p></div></div>
              <div className="stat-grid"><div className="stat"><span><Gauge size={12} /> CURRENT SPEED</span><strong>{formatMetric(current?.speed_kmh)} <small>km/h</small></strong></div><div className="stat"><span><Activity size={12} /> TOP SPEED</span><strong>{formatMetric(selected.metrics.max_speed_kmh)} <small>km/h</small></strong></div><div className="stat"><span><Footprints size={12} /> DISTANCE</span><strong>{formatMetric(selected.metrics.metric_calibration_available === false ? null : selected.metrics.distance_m)} <small>m</small></strong></div><div className="stat"><span><ScanLine size={12} /> IDENTITY CONFIDENCE</span><strong>{selected.identity_source === "unidentified" ? "—" : selected.identity_confidence.toFixed(2)}</strong></div></div>
              <div className="field" style={{ marginTop: 16 }}><label>PIXEL MASK · {(selected.metrics.mask_model_tier ?? "base_plus").replace("_", " ").toUpperCase()}</label><button className="button button-secondary" onClick={refineSelectedMask} disabled={["queued", "running", "large_ready"].includes(selected.metrics.mask_refinement_status ?? "base_ready")}><Sparkles size={14} />{selected.metrics.mask_refinement_status === "large_ready" ? "SAM LARGE READY" : ["queued", "running"].includes(selected.metrics.mask_refinement_status ?? "") ? "SAM LARGE REFINING…" : "REFINE SELECTED WITH SAM LARGE"}</button></div>
              {selected.role !== "referee" && roster.length > 0 && <div className="field" style={{ marginTop: 16 }}><label>ROSTER IDENTITY · {selected.identity_source.toUpperCase()}</label><select className="input" value={selected.roster_id ?? ""} onChange={(event) => changeIdentity(event.target.value)}><option value="">Restore automatic / Unidentified</option>{roster.map((player) => <option key={player.id} value={player.id}>{player.team} · #{player.squad_number} · {player.player_name}</option>)}</select></div>}
            </> : <div className="empty-state"><Eye size={28} /><h2>Select a player</h2><p>All tracks stay lightweight until you click one. Its SAM mask and full trajectory then load on demand.</p></div>}
          </section>
          {selected && <section className="panel"><div className="panel-title"><h3>TRACK STABILITY</h3></div><div className="metric-row"><span>Detection confidence</span><strong>{selected.detector_confidence?.toFixed(2) ?? "—"}</strong></div><div className="metric-row"><span>Pixel mask coverage</span><strong>{selected.metrics.mask_coverage_ratio === undefined ? "—" : `${(selected.metrics.mask_coverage_ratio * 100).toFixed(0)}%`}</strong></div><div className="metric-row"><span>Occlusion events</span><strong>{selected.metrics.occlusion_count}</strong></div><div className="metric-row"><span>ID retained</span><strong>{selected.metrics.id_retained ? "YES" : "NO"}</strong></div><div className="metric-row"><span>Area recovery</span><strong>{(selected.metrics.area_recovery_ratio * 100).toFixed(0)}%</strong></div><div className="metric-row"><span>Recovery frames</span><strong>{selected.metrics.recovery_frames ?? "—"}</strong></div><div className="metric-row"><span>Max centroid shift</span><strong>{selected.metrics.max_centroid_jump_px.toFixed(1)} px</strong></div><div className="metric-row"><span>Average speed</span><strong>{formatMetric(selected.metrics.average_speed_kmh)} km/h</strong></div></section>}
          {metrics && <section className="panel"><div className="panel-title"><h3>INFERENCE PROFILE</h3></div><div className="metric-row"><span>Device</span><strong>{metrics.device}</strong></div><div className="metric-row"><span>Frames processed</span><strong>{metrics.frames}</strong></div><div className="metric-row"><span>Effective FPS</span><strong>{metrics.effective_fps}</strong></div><div className="metric-row"><span>Peak GPU memory</span><strong>{metrics.gpu_peak_memory_mb.toFixed(0)} MB</strong></div><div className="metric-row"><span>Average GPU</span><strong>{metrics.gpu_utilization_average_percent.toFixed(0)}%</strong></div><div className="metric-row"><span>Calibration valid</span><strong>{(metrics.calibration_valid_rate * 100).toFixed(0)}%</strong></div><div className="metric-row"><span>Elapsed time</span><strong>{metrics.elapsed_seconds}s</strong></div></section>}
        </aside>
      </div>
    </div></main>
  );
}

function formatMetric(value: number | null | undefined) {
  return value === null || value === undefined ? "—" : value.toFixed(1);
}
