create extension if not exists pgcrypto;

create table if not exists public.projects (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null references auth.users(id) on delete cascade,
  title text not null,
  match_label text not null default '2026 FIFA World Cup Final',
  team_a text not null default 'Spain',
  team_b text not null default 'Argentina',
  source_path text,
  normalized_video_path text,
  mask_manifest_path text,
  foreground_video_path text,
  metrics_path text,
  prompts jsonb not null default '[]'::jsonb,
  calibration jsonb not null default '[]'::jsonb,
  slurm_job_id text,
  status text not null default 'draft' check (status in ('draft', 'queued', 'running', 'completed', 'failed')),
  error_message text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.tracks (
  id uuid primary key default gen_random_uuid(),
  project_id uuid not null references public.projects(id) on delete cascade,
  object_id integer not null,
  role text not null default 'player',
  team text not null default 'unknown',
  jersey_number integer,
  player_name text,
  dominant_color jsonb,
  trajectory jsonb not null default '[]'::jsonb,
  speed_series jsonb not null default '[]'::jsonb,
  metrics jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  unique (project_id, object_id)
);

create table if not exists public.roster (
  id bigint generated always as identity primary key,
  match_label text not null,
  team text not null,
  squad_number integer not null check (squad_number between 1 and 99),
  player_name text not null,
  position text not null,
  unique (match_label, team, squad_number)
);

create or replace function public.set_updated_at()
returns trigger
language plpgsql
set search_path = ''
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists projects_set_updated_at on public.projects;
create trigger projects_set_updated_at
before update on public.projects
for each row execute function public.set_updated_at();

alter table public.projects enable row level security;
alter table public.tracks enable row level security;
alter table public.roster enable row level security;

create policy "owners manage projects"
on public.projects for all
to authenticated
using ((select auth.uid()) = owner_id)
with check ((select auth.uid()) = owner_id);

create policy "owners read tracks"
on public.tracks for select
to authenticated
using (exists (
  select 1 from public.projects
  where projects.id = tracks.project_id and projects.owner_id = (select auth.uid())
));

create policy "roster is readable after login"
on public.roster for select
to authenticated
using (true);

grant select, insert, update, delete on public.projects to authenticated;
grant select on public.tracks to authenticated;
grant select on public.roster to authenticated;

insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values
  ('videos', 'videos', false, 52428800, array['video/mp4']),
  ('artifacts', 'artifacts', false, 52428800, array['video/mp4', 'application/json', 'application/gzip'])
on conflict (id) do update set
  public = excluded.public,
  file_size_limit = excluded.file_size_limit,
  allowed_mime_types = excluded.allowed_mime_types;

create policy "owners upload videos"
on storage.objects for insert
to authenticated
with check (bucket_id = 'videos' and (storage.foldername(name))[1] = (select auth.uid()::text));

create policy "owners read videos"
on storage.objects for select
to authenticated
using (bucket_id = 'videos' and (storage.foldername(name))[1] = (select auth.uid()::text));

create policy "owners replace videos"
on storage.objects for update
to authenticated
using (bucket_id = 'videos' and (storage.foldername(name))[1] = (select auth.uid()::text));

create policy "owners read artifacts"
on storage.objects for select
to authenticated
using (bucket_id = 'artifacts' and (storage.foldername(name))[1] = (select auth.uid()::text));
