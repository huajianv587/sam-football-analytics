"use client";

import { Camera, CheckCircle2, CloudUpload, Plane, Play, Radio, RotateCcw } from "lucide-react";
import { useEffect, useRef, useState } from "react";

const MAX_VIDEO_BYTES = 50 * 1024 * 1024;
const LAST_JOB_KEY = "pitchvision:last-job";

type JobState = "idle" | "queued" | "running" | "completed" | "failed";

const STAGE_LABELS: Record<string, string> = {
  queued: "Queued",
  normalize: "Normalize",
  reconstruct: "Detect / Track / Calibrate",
  detect: "Detect",
  track: "Track",
  calibrate: "Calibrate",
  segment: "Segment",
  identify: "Identify",
  upload: "Upload",
  completed: "Complete",
  failed: "Failed",
};

export function OfflineAnalyzer() {
  const apiUrl = process.env.NEXT_PUBLIC_INFERENCE_API_URL ?? "http://127.0.0.1:8000";
  const [file, setFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState("");
  const [jobId, setJobId] = useState("");
  const [state, setState] = useState<JobState>("idle");
  const [stage, setStage] = useState("queued");
  const [progress, setProgress] = useState(0);
  const [trackCount, setTrackCount] = useState(0);
  const [message, setMessage] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => () => { if (videoUrl) URL.revokeObjectURL(videoUrl); }, [videoUrl]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      const savedJob = window.localStorage.getItem(LAST_JOB_KEY);
      if (savedJob) {
        setJobId(savedJob);
        setState("queued");
        return;
      }
      fetch(`${apiUrl}/v1/offline/latest`)
        .then((response) => response.ok ? response.json() : null)
        .then((job) => {
          if (!job) return;
          window.localStorage.setItem(LAST_JOB_KEY, job.project_id);
          setJobId(job.project_id);
          setState(job.state);
          setStage(job.stage);
          setProgress(job.progress);
          setTrackCount(job.track_count);
          setMessage(job.message ?? "");
        })
        .catch(() => setMessage("Analysis service is offline. You can still preview the video, then start FastAPI to analyze it."));
    }, 0);
    return () => window.clearTimeout(timer);
  }, [apiUrl]);

  useEffect(() => {
    if (!jobId || (state !== "queued" && state !== "running")) return;
    async function refreshJob() {
      try {
        const response = await fetch(`${apiUrl}/v1/offline/jobs/${jobId}`);
        if (!response.ok) return;
        const job = await response.json();
        setState(job.state);
        setStage(job.stage);
        setProgress(job.progress);
        setTrackCount(job.track_count);
        setMessage(job.message ?? "");
      } catch {
        setState("failed");
        setMessage("Analysis service is offline. Start FastAPI, then choose the video again.");
      }
    }
    void refreshJob();
    const timer = window.setInterval(refreshJob, 4000);
    return () => window.clearInterval(timer);
  }, [apiUrl, jobId, state]);

  async function chooseFile(nextFile: File | undefined) {
    if (!nextFile) return;
    // Clear the native value so selecting the same file again still fires
    // change after a failed validation or a cancelled analysis.
    if (fileInputRef.current) fileInputRef.current.value = "";
    const extension = nextFile.name.toLowerCase().match(/\.([a-z0-9]+)$/)?.[1];
    const knownVideoExtension = ["mp4", "mov", "m4v", "webm", "mkv", "avi"].includes(extension ?? "");
    // Some camera exports (including extensionless `videoplayback` files) have an
    // empty MIME type. Let the user pick them, then validate by MIME/extension.
    if (!nextFile.type.startsWith("video/") && !knownVideoExtension && nextFile.type !== "application/octet-stream") {
      return setMessage(`“${nextFile.name}” is not recognized as a video file.`);
    }
    if (nextFile.size > MAX_VIDEO_BYTES) return setMessage(`“${nextFile.name}” is ${(nextFile.size / 1024 / 1024).toFixed(1)} MB. The upload limit is 50 MB; trim or export a shorter clip first.`);
    const url = URL.createObjectURL(nextFile);
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    // Set the preview before reading metadata so a valid local file is visible
    // immediately, even when Safari/Chrome takes time to inspect its codec.
    setFile(nextFile);
    setVideoUrl(url);
    setJobId("");
    setState("idle");
    setProgress(0);
    setTrackCount(0);
    setMessage("");
    window.localStorage.removeItem(LAST_JOB_KEY);
    let duration: number | null = null;
    try {
      duration = await readDuration(url);
    } catch {
      setMessage(`“${nextFile.name}” is selected, but the browser cannot preview this codec. FastAPI will attempt FFmpeg normalization when you start analysis.`);
      return;
    }
    if (duration > 60) {
      URL.revokeObjectURL(url);
      setFile(null);
      setVideoUrl("");
      return setMessage(`This clip is ${Math.ceil(duration)} seconds. The offline workflow supports clips up to 60 seconds.`);
    }
  }

  function reset() {
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    window.localStorage.removeItem(LAST_JOB_KEY);
    setFile(null);
    setVideoUrl("");
    setJobId("");
    setState("idle");
    setProgress(0);
    setMessage("");
  }

  async function analyze() {
    if (!file) return;
    setState("queued");
    setStage("upload");
    setProgress(0);
    setMessage("");
    try {
      const form = new FormData();
      form.set("video", file);
      form.set("title", file.name.replace(/\.mp4$/i, ""));
      form.set("match_label", "Unspecified Match");
      form.set("team_a", "Team A");
      form.set("team_b", "Team B");
      setProgress(5);
      const response = await fetch(`${apiUrl}/v1/offline/jobs`, {
        method: "POST",
        body: form,
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail ?? "Unable to submit the analysis job.");
      setJobId(result.project_id);
      window.localStorage.setItem(LAST_JOB_KEY, result.project_id);
      setState(result.state);
      setStage(result.stage);
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
        <div><span className="eyebrow">VIDEO INTELLIGENCE / OFFLINE PROCESSING</span><h1>Upload footage. Detect everyone. Inspect any player.</h1><p>Automatic detection, persistent IDs, pitch calibration, and SAM 2.1 masks — no manual annotation.</p></div>
        <div className="source-modes" aria-label="Input sources"><span className="source-mode active"><Play size={14} /> FILE</span><span className="source-mode"><Radio size={14} /> LIVE STREAM</span><span className="source-mode"><Camera size={14} /> FIELD CAMERA</span><span className="source-mode"><Plane size={14} /> DRONE FEED</span></div>
      </header>

      <div className="workflow-strip" aria-label="Analysis workflow">
        <span><b>01</b> Upload clip</span><i />
        <span><b>02</b> Detect + track</span><i />
        <span><b>03</b> Inspect with SAM</span>
      </div>

      {!videoUrl && !jobId ? (
        <section className="panel simple-upload">
          <div className="upload-layout">
            <div className="upload-copy"><span className="upload-icon"><CloudUpload size={24} /></span><div><h2>Select match footage</h2><p>H.264 MP4 · up to 60 seconds · 50 MB maximum</p></div></div>
            <div className="upload-action"><label className="button button-primary upload-trigger"><span>SELECT VIDEO</span><input ref={fileInputRef} type="file" accept="video/*,.mkv,.avi,.m4v" onChange={(event) => { const selected = event.currentTarget.files?.[0]; if (selected) void chooseFile(selected); }} onInput={(event) => { const selected = event.currentTarget.files?.[0]; if (selected) void chooseFile(selected); }} /></label><span>H.264 MP4 · up to 60 seconds · 50 MB</span></div>
          </div>
          {message && <p className="form-error">{message}</p>}
        </section>
      ) : videoUrl ? (
        <div className="simple-workspace">
          <section className="panel result-video-panel"><div className="video-stage"><video src={videoUrl} controls playsInline /></div></section>
          <section className="panel action-bar">
            <div><strong>{file?.name}</strong><span>{processing ? `${STAGE_LABELS[stage] ?? stage} · ${progress}%${trackCount ? ` · ${trackCount} tracks` : ""}` : state === "completed" ? "ANALYSIS COMPLETE · RESULTS SAVED" : state === "failed" ? message : "READY · ALL PLAYERS WILL BE DETECTED AUTOMATICALLY"}</span>{processing && <div className="progress"><span style={{ width: `${progress}%` }} /></div>}</div>
            <div className="nav-actions"><button className="button button-secondary" onClick={reset} disabled={processing}><RotateCcw size={15} /> REPLACE VIDEO</button>{state === "completed" ? <a className="button button-primary" href={`/projects/${jobId}/results`}><CheckCircle2 size={16} /> OPEN RESULTS</a> : <button className="button button-primary" onClick={analyze} disabled={processing}><Play size={16} />{processing ? "ANALYZING…" : "RUN AUTO ANALYSIS"}</button>}</div>
          </section>
        </div>
      ) : (
        <section className="panel simple-upload">
          {state === "completed" ? <CheckCircle2 size={42} /> : <CloudUpload size={42} />}
          <h2>{state === "completed" ? "Analysis complete" : state === "failed" ? "Analysis failed" : "Analysis in progress"}</h2>
          <p>{state === "failed" ? message : `${STAGE_LABELS[stage] ?? stage} · ${progress}%${trackCount ? ` · ${trackCount} tracks` : ""}`}</p>
          {(state === "queued" || state === "running") && <div className="progress" style={{ width: "min(520px, 100%)" }}><span style={{ width: `${progress}%` }} /></div>}
          <div className="nav-actions">
            {state === "completed" && <button className="button button-secondary" onClick={reset}><RotateCcw size={15} /> NEW ANALYSIS</button>}
            {state === "completed" ? <a className="button button-primary" href={`/projects/${jobId}/results`}><CheckCircle2 size={16} /> OPEN RESULTS</a> : <button className="button button-secondary" onClick={reset} disabled={state === "queued" || state === "running"}><RotateCcw size={15} /> SELECT ANOTHER VIDEO</button>}
          </div>
        </section>
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
