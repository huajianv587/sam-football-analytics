"use client";

import { ArrowRight, FolderPlus, LogOut, Plus, Sparkles } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { Topbar } from "@/components/topbar";
import { createClient } from "@/lib/supabase/client";
import type { Project } from "@/lib/types";

const labels = { draft: "SETUP", queued: "QUEUED", running: "A40 PROCESSING", completed: "COMPLETE", failed: "FAILED" };

export function ProjectsDashboard({ configured, userId, initialProjects }: { configured: boolean; userId: string; initialProjects: Project[] }) {
  const router = useRouter();
  const [showForm, setShowForm] = useState(false);
  const [title, setTitle] = useState("Automatic Football Analysis");
  const [matchLabel, setMatchLabel] = useState("Unspecified Match");
  const [teamA, setTeamA] = useState("Team A");
  const [teamB, setTeamB] = useState("Team B");
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);

  async function createProject(event: React.FormEvent) {
    event.preventDefault();
    setSaving(true);
    const { data, error: insertError } = await createClient().from("projects").insert({
      owner_id: userId,
      title,
      match_label: matchLabel,
      team_a: teamA,
      team_b: teamB,
      analysis_mode: "auto_all",
    }).select("id").single();
    setSaving(false);
    if (insertError) return setError(insertError.message);
    router.push(`/projects/${data.id}/setup`);
  }

  async function signOut() {
    await createClient().auth.signOut();
    router.push("/login");
    router.refresh();
  }

  return (
    <main className="page-shell">
      <Topbar signedIn />
      <div className="container content">
        <div className="page-heading">
          <div><span className="eyebrow">ANALYSIS WORKSPACE</span><h1>Match analysis projects</h1><p>Manage footage, A40 jobs, and interactive results.</p></div>
          <div className="nav-actions">
            <button className="button button-ghost" onClick={signOut}><LogOut size={16} /> SIGN OUT</button>
            <button className="button button-primary" disabled={!configured} onClick={() => setShowForm((value) => !value)}><Plus size={17} /> NEW PROJECT</button>
          </div>
        </div>

        {!configured && <div className="panel setup-note">Supabase is not configured. Complete the cloud setup in SETUP.md.</div>}

        {showForm && (
          <form className="panel form-stack" style={{ marginBottom: 20 }} onSubmit={createProject}>
            <div className="panel-title"><h2>New analysis</h2><span>CONTINUOUS SHOT · 60 SEC MAX</span></div>
            <div className="calibration-grid">
              <div className="field"><label>PROJECT NAME</label><input className="input" value={title} onChange={(event) => setTitle(event.target.value)} required /></div>
              <div className="field"><label>MATCH</label><input className="input" value={matchLabel} onChange={(event) => setMatchLabel(event.target.value)} required /></div>
              <div className="field"><label>TEAM A</label><input className="input" value={teamA} onChange={(event) => setTeamA(event.target.value)} required /></div>
              <div className="field"><label>TEAM B</label><input className="input" value={teamB} onChange={(event) => setTeamB(event.target.value)} required /></div>
            </div>
            {error && <p className="form-error">{error}</p>}
            <div><button className="button button-primary" disabled={saving}>{saving ? "CREATING…" : "CREATE AND UPLOAD"}</button></div>
          </form>
        )}

        {initialProjects.length ? (
          <div className="project-grid">
            {initialProjects.map((project) => (
              <Link className="panel project-card" key={project.id} href={project.status === "completed" ? `/projects/${project.id}/results` : `/projects/${project.id}/setup`}>
                <div><span className={`status-pill ${project.status}`}><Sparkles size={12} />{labels[project.status]}</span></div>
                <h3>{project.title}</h3><p>{project.team_a} vs {project.team_b}</p>
                <div className="project-card-footer"><span>{new Date(project.created_at).toLocaleDateString("en-GB")}</span><span>OPEN <ArrowRight size={13} style={{ display: "inline" }} /></span></div>
              </Link>
            ))}
          </div>
        ) : (
          <div className="panel empty-state"><FolderPlus size={34} /><h2>No analysis projects yet</h2><p>Create a project and upload a continuous match clip. Players, officials, IDs, and masks are generated automatically.</p><button className="button button-primary" disabled={!configured} onClick={() => setShowForm(true)}>CREATE FIRST PROJECT</button></div>
        )}
      </div>
    </main>
  );
}
