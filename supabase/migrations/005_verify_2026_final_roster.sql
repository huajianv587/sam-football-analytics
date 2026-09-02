-- Verified against FIFA's final World Cup 2026 squad list, version 1,
-- published 19 July 2026 after the Spain v Argentina final.
-- Source: https://fdp.fifa.org/assetspublic/ce281/pdf/SquadLists-English.pdf
update public.roster
set player_name = 'Marcos Senesi', position = 'DF'
where match_label = '2026 FIFA World Cup Final'
  and team = 'Argentina'
  and squad_number = 2;

update public.roster
set player_name = 'Nicolás González', position = 'MF'
where match_label = '2026 FIFA World Cup Final'
  and team = 'Argentina'
  and squad_number = 15;

update public.roster
set player_name = 'Yeremy Pino', position = 'FW'
where match_label = '2026 FIFA World Cup Final'
  and team = 'Spain'
  and squad_number = 11;

update public.roster
set player_name = 'Martín Zubimendi', position = 'MF'
where match_label = '2026 FIFA World Cup Final'
  and team = 'Spain'
  and squad_number = 18;

update public.roster
set player_name = 'Dani Olmo', position = 'FW'
where match_label = '2026 FIFA World Cup Final'
  and team = 'Spain'
  and squad_number = 10;
