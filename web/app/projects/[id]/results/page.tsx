import { notFound, redirect } from "next/navigation";
import type { Project, RosterPlayer, Track } from "@/lib/types";
import { ResultsWorkspace } from "./results-workspace";

export default async function ResultsPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const apiUrl = process.env.NEXT_PUBLIC_INFERENCE_API_URL ?? "http://127.0.0.1:8000";
  const response = await fetch(`${apiUrl}/v1/projects/${id}/results`, { cache: "no-store" });
  if (response.status === 404) notFound();
  if (!response.ok) throw new Error("Unable to load analysis results.");
  const bundle = await response.json() as {
    project: Project;
    tracks: Track[];
    roster: RosterPlayer[];
    urls: {
      video: string;
      foreground: string;
      metrics: string;
      legacyMasks: string | null;
      masksByTrack: Record<number, string>;
    };
  };
  if (bundle.project.status !== "completed") redirect("/");

  return (
    <ResultsWorkspace
      project={bundle.project}
      tracks={bundle.tracks}
      roster={bundle.roster}
      urls={bundle.urls}
    />
  );
}
