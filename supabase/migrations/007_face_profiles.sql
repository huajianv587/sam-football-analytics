-- Private face profiles for optional household / team identity matching.
-- Embeddings are kept as JSON so this migration does not require pgvector;
-- deployments can move the column to vector(512) when pgvector is enabled.
create table if not exists public.face_profiles (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null,
  label text not null check (char_length(label) between 1 and 120),
  embedding jsonb not null,
  photo_path text,
  consent_at timestamptz not null default now(),
  created_at timestamptz not null default now()
);

alter table public.face_profiles enable row level security;

drop policy if exists "owners manage face profiles" on public.face_profiles;
create policy "owners manage face profiles"
on public.face_profiles for all
using (owner_id = (select auth.uid()))
with check (owner_id = (select auth.uid()));

grant select, insert, update, delete on public.face_profiles to authenticated;

create index if not exists face_profiles_owner_idx
on public.face_profiles (owner_id, created_at desc);

insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values ('face-photos', 'face-photos', false, 5242880, array['image/jpeg', 'image/png', 'image/webp'])
on conflict (id) do update set public = excluded.public;

drop policy if exists "owners upload face photos" on storage.objects;
create policy "owners upload face photos"
on storage.objects for insert
to authenticated
with check (bucket_id = 'face-photos' and (storage.foldername(name))[1] = (select auth.uid()::text));

drop policy if exists "owners read face photos" on storage.objects;
create policy "owners read face photos"
on storage.objects for select
to authenticated
using (bucket_id = 'face-photos' and (storage.foldername(name))[1] = (select auth.uid()::text));
