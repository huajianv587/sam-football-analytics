"use client";

import { Check, CloudUpload, Cpu, Film, ScanSearch } from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { Topbar } from "@/components/topbar";
import { createClient } from "@/lib/supabase/client";
import type { Project, ProjectStatus } from "@/lib/types";
import { MAX_VIDEO_BYTES, uploadVideo } from "@/lib/upload";

const STAGES = ["normalize", "reconstruct", "segment", "identify", "upload"];

export function SetupWorkspace({ project, userId, initialVideoUrl }: { project: Project; userId: string; initialVideoUrl: string | null }) {
  const router = useRouter();
  const [videoUrl, setVideoUrl] = useState(initialVideoUrl);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<ProjectStatus>(project.status);
  const [stage, setStage] = useState(project.stage ?? "draft");
  const [progress, setProgress] = useState(project.progress ?? 0);
  const [trackCount, setTrackCount] = useState(0);
  const [message, setMessage] = useState(project.error_message ?? "");
  const apiUrl = process.env.NEXT_PUBLIC_INFERENCE_API_URL ?? "http://127.0.0.1:8000";

  useEffect(() => {
    if (!(status === "queued" || status === "running")) return;
    const timer = window.setInterval(async () => {
      const { data } = await createClient().auth.getSession();
      const response = await fetch(`${apiUrl}/v1/jobs/${project.id}`, { headers: { Authorization: `Bearer ${data.session?.access_token}` } });
      if (!response.ok) return;
      const job = await response.json();
      setStatus(job.state);
      setStage(job.stage);
      setProgress(job.progress);
      setTrackCount(job.track_count);
      setMessage(job.message ?? "");
      if (job.state === "completed") {
        window.clearInterval(timer);
        router.push(`/projects/${project.id}/results`);
        router.refresh();
      }
      if (job.state === "failed") window.clearInterval(timer);
    }, 4000);
    return () => window.clearInterval(timer);
  }, [apiUrl, project.id, router, status]);

  async function chooseFile(file: File | undefined) {
    if (!file) return;
    if (file.type !== "video/mp4") return setMessage("Select an MP4 video.");
    if (file.size > MAX_VIDEO_BYTES) return setMessage("Video size must be 50 MB or less.");
    const localUrl = URL.createObjectURL(file);
    const duration = await readDuration(localUrl);
    if (duration > 60) {
      URL.revokeObjectURL(localUrl);
      return setMessage("Video duration must be 60 seconds or less.");
    }
    setBusy(true);
    setMessage("");
    setVideoUrl(localUrl);
    const path = `${userId}/${project.id}/source.mp4`;
    try {
      const supabase = createClient();
      await uploadVideo(supabase, file, path, setUploadProgress);
      const { error } = await supabase.from("projects").update({ source_path: path, analysis_mode: "auto_all" }).eq("id", project.id);
      if (error) throw error;
      const { data } = await supabase.auth.getSession();
      const response = await fetch(`${apiUrl}/v1/jobs`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${data.session?.access_token}` },
        body: JSON.stringify({ project_id: project.id, source_path: path, analysis_mode: "auto_all" }),
      });
      const job = await response.json();
      if (!response.ok) throw new Error(job.detail ?? "Unable to submit the A40 job.");
      setStatus(job.state);
      setStage(job.stage);
      setProgress(job.progress);
    } catch (error) {
      setStatus("failed");
      setMessage(error instanceof Error ? error.message : "Upload or submission failed.");
    } finally {
      setBusy(false);
    }
  }

  const processing = status === "queued" || status === "running";
  const currentStage = STAGES.indexOf(stage);
  return (
    <main className="page-shell"><Topbar signedIn /><div className="container content">
      <div className="page-heading"><div><span className="eyebrow">AUTOMATIC MATCH ANALYSIS</span><h1>{project.title}</h1><p>{project.team_a} vs {project.team_b} · upload once, detect everyone automatically</p></div><span className={`status-pill ${status}`}><Cpu size={13} />{processing ? `${stage.toUpperCase()} · ${progress}%` : status === "failed" ? "FAILED" : "READY"}</span></div>
      <div className="workspace-grid">
        <div className="workspace-main">
          {!videoUrl ? <section className="panel upload-box"><CloudUpload size={34} /><h2>Upload continuous match footage</h2><p>MP4 · up to 60 seconds · 50 MB maximum · no manual boxes required</p><label className="button button-primary">SELECT AND ANALYZE<input type="file" accept="video/mp4" onChange={(event) => chooseFile(event.target.files?.[0])} /></label></section> : <section className="panel result-video-panel"><div className="video-stage"><video src={videoUrl} controls playsInline /></div>{(busy || processing) && <div className="progress"><span style={{ width: `${busy ? uploadProgress : progress}%` }} /></div>}</section>}
        </div>
        <aside className="workspace-side">
          <section className="panel"><div className="panel-title"><h3>PIPELINE</h3><span>{trackCount ? `${trackCount} TRACKS` : "AUTO"}</span></div><div className="step-list">
            <div className={`step ${videoUrl ? "done" : "active"}`}><span className="step-number">{videoUrl ? <Check size={14} /> : 1}</span><Film size={16} /> Upload footage</div>
            {STAGES.map((item, index) => <div className={`step ${currentStage > index || status === "completed" ? "done" : currentStage === index ? "active" : ""}`} key={item}><span className="step-number">{currentStage > index || status === "completed" ? <Check size={14} /> : index + 2}</span>{item === "reconstruct" ? <ScanSearch size={16} /> : <Cpu size={16} />}{item === "reconstruct" ? "Detect / Track / Calibrate" : item[0].toUpperCase() + item.slice(1)}</div>)}
          </div></section>
          {message && <section className="panel"><p className="form-error" style={{ margin: 0 }}>{message}</p></section>}
        </aside>
      </div>
    </div></main>
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
