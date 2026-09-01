import { notFound } from "next/navigation";
import { createClient } from "@/lib/supabase/server";
import type { Project } from "@/lib/types";
import { SetupWorkspace } from "./setup-workspace";

export default async function SetupPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const supabase = await createClient();
  if (!supabase) notFound();
  const { data: project } = await supabase.from("projects").select("*").eq("id", id).single();
  if (!project) notFound();
  let videoUrl: string | null = null;
  if (project.source_path) {
    const { data } = await supabase.storage.from("videos").createSignedUrl(project.source_path, 3600);
    videoUrl = data?.signedUrl ?? null;
  }
  const { data: claims } = await supabase.auth.getClaims();
  return <SetupWorkspace project={project as Project} userId={String(claims?.claims?.sub ?? "")} initialVideoUrl={videoUrl} />;
}
