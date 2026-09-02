-- Random demo footage must never inherit a specific fixture identity.
-- Existing projects and the verified 2026 final roster remain unchanged.
alter table public.projects
  alter column team_a set default 'Team A',
  alter column team_b set default 'Team B';
