"use client";

import { Camera, CheckCircle2, CloudUpload, Plane, Play, Radio, RotateCcw } from "lucide-react";
import { useEffect, useState } from "react";
import { BoxAnnotator } from "@/app/projects/[id]/setup/box-annotator";
import { CalibrationPanel } from "@/app/projects/[id]/setup/calibration-panel";
import type { CalibrationPair, PromptBox } from "@/lib/types";
import { MAX_VIDEO_BYTES } from "@/lib/upload";

const QUICK_CALIBRATION: CalibrationPair[] = [
  { video: [0, 0], pitch: [0, 0] },
  { video: [1, 0], pitch: [105, 0] },
  { video: [1, 1], pitch: [105, 68] },
  { video: [0, 1], pitch: [0, 68] },
];

type JobState = "idle" | "queued" | "running" | "completed" | "failed";

export function OfflineAnalyzer() {
  const apiUrl = process.env.NEXT_PUBLIC_INFERENCE_API_URL ?? "http://127.0.0.1:8000";
  const [file, setFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState("");
  const [boxes, setBoxes] = useState<PromptBox[]>([]);
  const [pairs, setPairs] = useState<CalibrationPair[]>(QUICK_CALIBRATION);
  const [jobId, setJobId] = useState("");
  const [state, setState] = useState<JobState>("idle");
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState("");

  useEffect(() => () => { if (videoUrl) URL.revokeObjectURL(videoUrl); }, [videoUrl]);

  useEffect(() => {
    if (!jobId || (state !== "queued" && state !== "running")) return;
    const timer = window.setInterval(async () => {
      const response = await fetch(`${apiUrl}/v1/offline/jobs/${jobId}`);
      if (!response.ok) return;
      const job = await response.json();
      setState(job.state);
      setProgress(job.progress);
      setMessage(job.message ?? "");
    }, 4000);
    return () => window.clearInterval(timer);
  }, [apiUrl, jobId, state]);

  async function chooseFile(nextFile: File | undefined) {
    if (!nextFile) return;
    if (nextFile.type !== "video/mp4") return setMessage("The offline workflow currently accepts MP4 video only.");
    if (nextFile.size > MAX_VIDEO_BYTES) return setMessage("Video size must be 50 MB or less.");
    const url = URL.createObjectURL(nextFile);
    const duration = await readDuration(url);
    if (duration > 60) { URL.revokeObjectURL(url); return setMessage("The offline workflow supports clips up to 60 seconds."); }
    setFile(nextFile);
    setVideoUrl(url);
    setBoxes([]);
    setPairs(QUICK_CALIBRATION);
    setJobId("");
    setState("idle");
    setProgress(0);
    setMessage("");
  }

  function reset() {
    setFile(null);
    setVideoUrl("");
    setBoxes([]);
    setPairs(QUICK_CALIBRATION);
    setJobId("");
    setState("idle");
    setMessage("");
  }

  async function analyze() {
    if (!file || !boxes.length || pairs.length < 4) return;
    setState("queued");
    setProgress(5);
    setMessage("");
    const body = new FormData();
    body.append("video", file);
    body.append("prompts", JSON.stringify(boxes));
    body.append("calibration", JSON.stringify(pairs));
    body.append("title", file.name.replace(/\.mp4$/i, ""));
    try {
      const response = await fetch(`${apiUrl}/v1/offline/jobs`, { method: "POST", body });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail ?? "Unable to submit the analysis job.");
      setJobId(result.project_id);
      setState(result.state);
      setProgress(result.progress);
    } catch (error) {
      setState("failed");
      setMessage(error instanceof Error ? error.message : "Unable to reach the local analysis service.");
    }
  }

  const processing = state === "queued" || state === "running";
  return (
    <div className="container offline-shell">
      <header className="offline-heading">
        <div><span className="eyebrow">VIDEO INTELLIGENCE / OFFLINE PROCESSING</span><h1>Upload footage. Select players. Run analysis.</h1><p>Process continuous match footage now. Live streams, field cameras, and drone feeds connect in the next phase.</p></div>
        <div className="source-modes" aria-label="Input sources">
          <span className="source-mode active"><Play size={14} /> FILE</span>
          <span className="source-mode"><Radio size={14} /> LIVE STREAM</span>
          <span className="source-mode"><Camera size={14} /> FIELD CAMERA</span>
          <span className="source-mode"><Plane size={14} /> DRONE FEED</span>
        </div>
      </header>

      {!videoUrl ? (
        <section className="panel simple-upload">
          <CloudUpload size={42} />
          <h2>Select match footage</h2>
          <p>H.264 MP4 · up to 60 seconds · 50 MB maximum · continuous shot</p>
          <label className="button button-primary">SELECT VIDEO<input type="file" accept="video/mp4,.mp4" onChange={(event) => chooseFile(event.target.files?.[0])} /></label>
          {message && <p className="form-error">{message}</p>}
        </section>
      ) : (
        <div className="simple-workspace">
          <section className="panel annotation-panel">
            <div className="panel-title"><div><h2>Select players on the first frame</h2><p className="muted micro">Draw one box per subject. SAM propagates masks and persistent IDs across the clip.</p></div><span>{boxes.length} SUBJECTS</span></div>
            <BoxAnnotator videoUrl={videoUrl} boxes={boxes} onChange={setBoxes} />
          </section>

          <details className="panel calibration-details">
            <summary>Metric calibration / optional</summary>
            <p className="muted micro">Quick mode maps the frame corners to a standard pitch. Replace the four point pairs for accurate metric speed.</p>
            <CalibrationPanel videoUrl={videoUrl} pairs={pairs} onChange={setPairs} />
          </details>

          <section className="panel action-bar">
            <div>
              <strong>{file?.name}</strong>
              <span>{processing ? `A40 PROCESSING · ${progress}%` : state === "completed" ? "ANALYSIS COMPLETE · RESULTS SAVED" : state === "failed" ? message : boxes.length ? "READY TO ANALYZE" : "SELECT AT LEAST ONE PLAYER"}</span>
              {processing && <div className="progress"><span style={{ width: `${progress}%` }} /></div>}
            </div>
            <div className="nav-actions">
              <button className="button button-secondary" onClick={reset} disabled={processing}><RotateCcw size={15} /> REPLACE VIDEO</button>
              <button className="button button-primary" onClick={analyze} disabled={!boxes.length || pairs.length < 4 || processing || state === "completed"}>{state === "completed" ? <CheckCircle2 size={16} /> : <Play size={16} />}{processing ? "ANALYZING…" : state === "completed" ? "COMPLETE" : "RUN OFFLINE ANALYSIS"}</button>
            </div>
          </section>
        </div>
      )}
    </div>
  );
}

function readDuration(url: string) {
  return new Promise<number>((resolve, reject) => {
    const video = document.createElement("video");
    video.preload = "metadata";
    video.onloadedmetadata = () => resolve(video.duration);
    video.onerror = () => reject(new Error("Unable to read video metadata."));
    video.src = url;
  });
}
