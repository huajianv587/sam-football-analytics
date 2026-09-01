import { notFound, redirect } from "next/navigation";
import { createClient } from "@/lib/supabase/server";
import type { Project, Track } from "@/lib/types";
import { ResultsWorkspace } from "./results-workspace";

export default async function ResultsPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const supabase = await createClient();
  if (!supabase) notFound();
  const [{ data: project }, { data: tracks }] = await Promise.all([
    supabase.from("projects").select("*").eq("id", id).single(),
    supabase.from("tracks").select("*").eq("project_id", id).order("object_id"),
  ]);
  if (!project) notFound();
  if (project.status !== "completed") redirect(`/projects/${id}/setup`);

  const paths = [project.normalized_video_path, project.foreground_video_path, project.mask_manifest_path, project.metrics_path];
  if (paths.some((path) => !path)) notFound();
  const signed = await Promise.all(paths.map((path) => supabase.storage.from("artifacts").createSignedUrl(path, 3600)));
  if (signed.some(({ data }) => !data?.signedUrl)) notFound();
  return (
    <ResultsWorkspace
      project={project as Project}
      tracks={(tracks ?? []) as Track[]}
      urls={{
        video: signed[0].data!.signedUrl,
        foreground: signed[1].data!.signedUrl,
        masks: signed[2].data!.signedUrl,
        metrics: signed[3].data!.signedUrl,
      }}
    />
  );
}
