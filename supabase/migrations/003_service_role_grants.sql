-- Backend access for the Supabase secret key.
grant usage on schema public to service_role;

grant select, insert, update, delete
on table public.projects
to service_role;

grant select, insert, update, delete
on table public.tracks
to service_role;

grant select
on table public.roster
to service_role;

grant usage, select
on sequence public.roster_id_seq
to service_role;
