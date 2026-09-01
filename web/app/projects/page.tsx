import { createClient } from "@/lib/supabase/server";
import type { Project } from "@/lib/types";
import { ProjectsDashboard } from "./projects-dashboard";

export default async function ProjectsPage() {
  const supabase = await createClient();
  if (!supabase) return <ProjectsDashboard configured={false} userId="" initialProjects={[]} />;
  const [{ data: claims }, { data: projects }] = await Promise.all([
    supabase.auth.getClaims(),
    supabase.from("projects").select("*").order("created_at", { ascending: false }),
  ]);
  return (
    <ProjectsDashboard
      configured
      userId={String(claims?.claims?.sub ?? "")}
      initialProjects={(projects ?? []) as Project[]}
    />
  );
}
