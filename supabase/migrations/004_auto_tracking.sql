alter table public.projects
  add column if not exists analysis_mode text,
  add column if not exists stage text,
  add column if not exists progress integer,
  add column if not exists calibration_path text;

update public.projects
set analysis_mode = 'manual_sam'
where analysis_mode is null;

update public.projects
set stage = case
    when status = 'completed' then 'completed'
    when status = 'failed' then 'failed'
    when status = 'running' then 'segment'
    else status
  end,
  progress = case when status = 'completed' then 100 else 0 end
where stage is null or progress is null;

alter table public.projects
  alter column analysis_mode set default 'auto_all',
  alter column analysis_mode set not null,
  alter column stage set default 'draft',
  alter column stage set not null,
  alter column progress set default 0,
  alter column progress set not null;

alter table public.projects
  drop constraint if exists projects_analysis_mode_check;

alter table public.projects
  add constraint projects_analysis_mode_check
  check (analysis_mode in ('manual_sam', 'auto_all'));

alter table public.projects
  drop constraint if exists projects_progress_check;

alter table public.projects
  add constraint projects_progress_check
  check (progress between 0 and 100);

alter table public.tracks
  add column if not exists mask_path text,
  add column if not exists detections jsonb not null default '[]'::jsonb,
  add column if not exists first_frame integer,
  add column if not exists last_frame integer,
  add column if not exists detector_confidence real,
  add column if not exists auto_roster_id bigint references public.roster(id) on delete set null,
  add column if not exists roster_id bigint references public.roster(id) on delete set null,
  add column if not exists identity_source text,
  add column if not exists identity_confidence real;

update public.tracks
set identity_source = case when player_name is null then 'unidentified' else 'automatic' end,
  identity_confidence = coalesce(identity_confidence, 0)
where identity_source is null or identity_confidence is null;

alter table public.tracks
  alter column identity_source set default 'unidentified',
  alter column identity_source set not null,
  alter column identity_confidence set default 0,
  alter column identity_confidence set not null;

alter table public.tracks
  drop constraint if exists tracks_identity_source_check;

alter table public.tracks
  add constraint tracks_identity_source_check
  check (identity_source in ('automatic', 'manual', 'unidentified'));

alter table public.tracks
  drop constraint if exists tracks_identity_confidence_check;

alter table public.tracks
  add constraint tracks_identity_confidence_check
  check (identity_confidence between 0 and 1);

create index if not exists tracks_project_frames_idx
on public.tracks (project_id, first_frame, last_frame);

grant select, insert, update, delete
on table public.projects
to service_role;

grant select, insert, update, delete
on table public.tracks
to service_role;
