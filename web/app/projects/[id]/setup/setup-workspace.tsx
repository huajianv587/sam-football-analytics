"use client";

import { Check, CloudUpload, Cpu, Film, ScanSearch, Send } from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { Topbar } from "@/components/topbar";
import { createClient } from "@/lib/supabase/client";
import type { CalibrationPair, Project, PromptBox } from "@/lib/types";
import { MAX_VIDEO_BYTES, uploadVideo } from "@/lib/upload";
import { BoxAnnotator, promptColor } from "./box-annotator";
import { CalibrationPanel } from "./calibration-panel";

export function SetupWorkspace({ project, userId, initialVideoUrl }: { project: Project; userId: string; initialVideoUrl: string | null }) {
  const router = useRouter();
  const [videoUrl, setVideoUrl] = useState(initialVideoUrl);
  const [sourcePath, setSourcePath] = useState(project.source_path);
  const [boxes, setBoxes] = useState<PromptBox[]>(project.prompts ?? []);
  const [pairs, setPairs] = useState<CalibrationPair[]>(project.calibration ?? []);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState(project.status);
  const [message, setMessage] = useState(project.error_message ?? "");
  const apiUrl = process.env.NEXT_PUBLIC_INFERENCE_API_URL ?? "http://127.0.0.1:8000";

  useEffect(() => {
    if (!(["queued", "running"] as string[]).includes(status)) return;
    const timer = window.setInterval(async () => {
      const { data } = await createClient().auth.getSession();
      const response = await fetch(`${apiUrl}/v1/jobs/${project.id}`, { headers: { Authorization: `Bearer ${data.session?.access_token}` } });
      if (!response.ok) return;
      const job = await response.json();
      setStatus(job.state);
      setMessage(job.message ?? "");
      if (job.state === "completed") { window.clearInterval(timer); router.push(`/projects/${project.id}/results`); router.refresh(); }
      if (job.state === "failed") window.clearInterval(timer);
    }, 5000);
    return () => window.clearInterval(timer);
  }, [apiUrl, project.id, router, status]);

  async function chooseFile(file: File | undefined) {
    if (!file) return;
    if (file.type !== "video/mp4") return setMessage("Select an MP4 video.");
    if (file.size > MAX_VIDEO_BYTES) return setMessage("Video size must be 50 MB or less.");
    const localUrl = URL.createObjectURL(file);
    const duration = await readDuration(localUrl);
    if (duration > 60) { URL.revokeObjectURL(localUrl); return setMessage("Video duration must be 60 seconds or less."); }
    setBusy(true); setMessage(""); setVideoUrl(localUrl);
    const path = `${userId}/${project.id}/source.mp4`;
    try {
      const supabase = createClient();
      await uploadVideo(supabase, file, path, setUploadProgress);
      const { error } = await supabase.from("projects").update({ source_path: path }).eq("id", project.id);
      if (error) throw error;
      setSourcePath(path);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Upload failed.");
    } finally { setBusy(false); }
  }

  async function submitJob() {
    if (!sourcePath || !boxes.length || pairs.length < 4) return;
    setBusy(true); setMessage("");
    try {
      const supabase = createClient();
      const { error } = await supabase.from("projects").update({ prompts: boxes, calibration: pairs }).eq("id", project.id);
      if (error) throw error;
      const { data } = await supabase.auth.getSession();
      const response = await fetch(`${apiUrl}/v1/jobs`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${data.session?.access_token}` },
        body: JSON.stringify({ project_id: project.id, source_path: sourcePath, prompts: boxes, calibration: pairs }),
      });
      if (!response.ok) throw new Error((await response.json()).detail ?? "Unable to submit the A40 job.");
      const job = await response.json();
      setStatus(job.state);
    } catch (error) { setMessage(error instanceof Error ? error.message : "Submission failed."); }
    finally { setBusy(false); }
  }

  const ready = Boolean(sourcePath && boxes.length && pairs.length >= 4);
  const processing = status === "queued" || status === "running";
  return (
    <main className="page-shell"><Topbar signedIn /><div className="container content">
      <div className="page-heading"><div><span className="eyebrow">PROJECT SETUP</span><h1>{project.title}</h1><p>{project.team_a} vs {project.team_b} · {project.match_label}</p></div><span className={`status-pill ${status}`}><Cpu size={13} />{processing ? "A40 PROCESSING" : status === "failed" ? "FAILED" : "READY TO ANNOTATE"}</span></div>
      <div className="workspace-grid">
        <div className="workspace-main">
          {!videoUrl ? (
            <section className="panel upload-box"><CloudUpload size={34} /><h2>Upload continuous match footage</h2><p>MP4 · 10–60 seconds · 50 MB maximum · no camera cuts</p><label className="button button-primary">SELECT VIDEO<input type="file" accept="video/mp4" onChange={(event) => chooseFile(event.target.files?.[0])} /></label>{busy && <div className="progress"><span style={{ width: `${uploadProgress}%` }} /></div>}</section>
          ) : (
            <div className="form-stack">
              <section className="panel"><div className="panel-title"><h2>01 / Select players and officials</h2><span>{boxes.length} SUBJECTS</span></div><BoxAnnotator videoUrl={videoUrl} boxes={boxes} onChange={setBoxes} /></section>
              <section className="panel"><div className="panel-title"><h2>02 / Four-point pitch calibration</h2><span>{pairs.length}/4 PAIRS</span></div><CalibrationPanel videoUrl={videoUrl} pairs={pairs} onChange={setPairs} /></section>
            </div>
          )}
        </div>
        <aside className="workspace-side">
          <section className="panel"><div className="panel-title"><h3>PIPELINE</h3></div><div className="step-list">
            <div className={`step ${sourcePath ? "done" : "active"}`}><span className="step-number">{sourcePath ? <Check size={14} /> : 1}</span><Film size={16} /> Upload footage</div>
            <div className={`step ${boxes.length ? "done" : sourcePath ? "active" : ""}`}><span className="step-number">{boxes.length ? <Check size={14} /> : 2}</span><ScanSearch size={16} /> First-frame prompts</div>
            <div className={`step ${pairs.length >= 4 ? "done" : boxes.length ? "active" : ""}`}><span className="step-number">{pairs.length >= 4 ? <Check size={14} /> : 3}</span> Metric calibration</div>
            <div className={`step ${processing ? "active" : ""}`}><span className="step-number">4</span><Cpu size={16} /> A40 inference</div>
          </div></section>
          {boxes.length > 0 && <section className="panel"><div className="panel-title"><h3>OBJECT IDs</h3><span>{boxes.length}</span></div><div className="prompt-list">{boxes.map((box, index) => <div className="prompt-row" key={box.object_id}><span><i className="object-color" style={{ background: promptColor(index) }} />Object {box.object_id}</span><span className="muted">{Math.round((box.box[2] - box.box[0]) * 100)}% width</span></div>)}</div></section>}
          {message && <section className="panel"><p className="form-error" style={{ margin: 0 }}>{message}</p></section>}
          <button className="button button-primary" disabled={!ready || busy || processing} onClick={submitJob}><Send size={16} />{processing ? "A40 PROCESSING" : busy ? "SUBMITTING…" : "RUN A40 ANALYSIS"}</button>
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
