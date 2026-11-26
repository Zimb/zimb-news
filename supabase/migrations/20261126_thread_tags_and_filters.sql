-- Sujets de fond des histoires (attribués par Claude), filtres par catégorie et par sujet, sujets de la semaine.
-- Compatible avec l'ancien site et l'ancien worker : nouveaux paramètres facultatifs, colonnes ajoutées en fin de résultat.

alter table public.threads add column if not exists tags text[] not null default '{}';
create index if not exists threads_tags_gin on public.threads using gin (tags);
create index if not exists threads_category_last_seen on public.threads (category, last_seen desc);

-- ---------------------------------------------------------------- feed
drop function if exists public.feed(timestamptz, bigint, integer);
create function public.feed(before_ts timestamptz default null, before_id bigint default null, lim integer default 20,
                            cat text default null, tag text default null)
returns table(id bigint, title text, summary text, category text, first_seen timestamptz, last_seen timestamptz,
              updated_at timestamptz, n_sources_total integer, n_articles_total integer, n_versions integer,
              sources jsonb, tags text[])
language sql stable set search_path to ''
as $function$
  select t.id, t.title, t.summary, t.category, t.first_seen, t.last_seen, t.updated_at,
         (select count(distinct a.domain)::int from public.thread_articles ta join public.articles a on a.url = ta.article_url where ta.thread_id = t.id),
         (select count(*)::int from public.thread_articles ta where ta.thread_id = t.id and not ta.is_mirror),
         (select count(*)::int from public.thread_versions v where v.thread_id = t.id),
         (select jsonb_agg(jsonb_build_object('url', x.url, 'domain', x.domain, 'title', x.title_guess, 'at', x.at) order by x.at desc)
          from (select a.url, a.domain, a.title_guess, coalesce(a.observed_at, ta.added_at) as at
                from public.thread_articles ta join public.articles a on a.url = ta.article_url
                where ta.thread_id = t.id order by 4 desc limit 15) x),
         t.tags
  from public.threads t
  where (before_ts is null or (t.last_seen, t.id) < (before_ts, coalesce(before_id, 9223372036854775807)))
    and (cat is null or t.category = cat)
    and (tag is null or t.tags @> array[lower(tag)])
  order by t.last_seen desc, t.id desc
  limit least(greatest(lim, 1), 50)
$function$;
grant execute on function public.feed(timestamptz, bigint, integer, text, text) to anon, authenticated, service_role;

-- ---------------------------------------------------------------- top_stories
drop function if exists public.top_stories(integer, integer);
create function public.top_stories(hours integer default 3, lim integer default 10,
                                   cat text default null, tag text default null)
returns table(id bigint, title text, summary text, category text, first_seen timestamptz, last_seen timestamptz,
              updated_at timestamptz, score numeric, n_sources_window integer, n_articles_window integer,
              n_sources_total integer, n_articles_total integer, n_versions integer, sources jsonb, tags text[])
language sql stable set search_path to ''
as $function$
  select tr.id, tr.title, tr.summary, tr.category, tr.first_seen, tr.last_seen, tr.updated_at,
         tr.score, tr.n_sources_window, tr.n_articles_window, tr.n_sources_total, tr.n_articles_total,
         (select count(*)::int from public.thread_versions v where v.thread_id = tr.id),
         tr.sources, t.tags
  -- sans filtre, trending ne calcule que les `lim` premières ; avec filtre, on filtre dans tout le classement
  from public.trending(hours, case when cat is null and tag is null then lim else 1000 end) tr
  join public.threads t on t.id = tr.id
  where (cat is null or tr.category = cat)
    and (tag is null or t.tags @> array[lower(tag)])
  order by tr.score desc, tr.n_sources_window desc, tr.last_seen desc
  limit lim
$function$;
grant execute on function public.top_stories(integer, integer, text, text) to anon, authenticated, service_role;

-- ---------------------------------------------------------------- week_topics
-- Sujets portés par au moins `min_threads` histoires distinctes sur la période
create or replace function public.week_topics(days integer default 7, lim integer default 12, min_threads integer default 2)
returns table(tag text, n_threads integer, n_days integer, last_seen timestamptz, categories text[])
language sql stable set search_path to ''
as $function$
  with x as (
    select unnest(t.tags) as tag, t.id, t.last_seen, t.category
    from public.threads t
    where t.last_seen > now() - make_interval(days => days)
  ),
  agg as (
    select x.tag, count(distinct x.id)::int as n_threads, max(x.last_seen) as last_seen,
           array_agg(distinct x.category) as categories, array_agg(distinct x.id) as ids
    from x group by x.tag
    having count(distinct x.id) >= min_threads
  )
  select agg.tag, agg.n_threads,
         (select count(distinct (ta.added_at at time zone 'Europe/Paris')::date)::int
          from public.thread_articles ta
          where ta.thread_id = any(agg.ids) and ta.added_at > now() - make_interval(days => days)),
         agg.last_seen, agg.categories
  from agg
  order by 2 desc, 3 desc, 4 desc
  limit least(greatest(lim, 1), 100)
$function$;
grant execute on function public.week_topics(integer, integer, integer) to anon, authenticated, service_role;

-- ---------------------------------------------------------------- active_threads_v2 (worker uniquement)
drop function if exists public.active_threads_v2(integer, integer);
create function public.active_threads_v2(hours integer default 6, lim integer default 40)
returns table(id bigint, title text, summary text, category text, last_seen timestamptz, updates text[], tags text[])
language sql stable security definer set search_path to ''
as $function$
  select t.id, t.title, left(t.summary, 220), t.category, t.last_seen,
         coalesce((select array_agg(u.text order by u.created_at desc)
                   from (select text, created_at from public.thread_updates
                         where thread_id = t.id order by created_at desc limit 3) u), '{}'),
         t.tags
  from public.threads t
  where t.last_seen > now() - make_interval(hours => hours)
  order by t.last_seen desc
  limit lim
$function$;
revoke all on function public.active_threads_v2(integer, integer) from public, anon, authenticated;
grant execute on function public.active_threads_v2(integer, integer) to service_role;

-- ---------------------------------------------------------------- ingest_run_v2 : enregistre les sujets
create or replace function public.ingest_run_v2(payload jsonb)
returns bigint
language plpgsql security definer set search_path to ''
as $function$
declare
  v_run_id bigint;
  v_at timestamptz := coalesce((payload->>'window_end')::timestamptz, now());
  g jsonb;
  v_thread_id bigint;
  v_title text;
  v_summary text;
  v_update text;
  v_tags text[];
begin
  insert into public.runs (started_at, finished_at, window_start, window_end, source_files,
                           n_articles, n_stories, model, input_tokens, output_tokens, status, error)
  values (coalesce((payload->>'started_at')::timestamptz, now()), now(),
          (payload->>'window_start')::timestamptz, v_at,
          coalesce(array(select jsonb_array_elements_text(payload->'source_files')), '{}'),
          coalesce((payload->>'n_articles')::int, 0),
          coalesce(jsonb_array_length(payload->'groups'), 0),
          payload->>'model', (payload->>'input_tokens')::int, (payload->>'output_tokens')::int,
          coalesce(payload->>'status', 'ok'), payload->>'error')
  returning id into v_run_id;

  insert into public.seen_urls (url)
  select jsonb_array_elements_text(coalesce(payload->'seen_urls', '[]'::jsonb))
  on conflict do nothing;

  insert into public.articles (url, domain, title_guess, snippet, observed_at, first_run_id)
  select a->>'url', a->>'domain', a->>'title_guess', a->>'snippet', (a->>'observed_at')::timestamptz, v_run_id
  from jsonb_array_elements(coalesce(payload->'articles', '[]'::jsonb)) a
  on conflict (url) do nothing;

  for g in select * from jsonb_array_elements(coalesce(payload->'groups', '[]'::jsonb)) loop
    v_thread_id := null;
    v_title := nullif(btrim(g->>'title'), '');
    v_summary := nullif(btrim(g->>'summary'), '');
    v_update := nullif(btrim(g->>'update'), '');
    v_tags := array(select distinct lower(btrim(x))
                    from jsonb_array_elements_text(coalesce(g->'tags', '[]'::jsonb)) x
                    where btrim(x) <> '');

    if coalesce((g->>'thread_id')::bigint, 0) > 0 then
      update public.threads t set
        last_seen = greatest(t.last_seen, v_at),
        title = coalesce(v_title, t.title),
        summary = coalesce(v_summary, t.summary),
        category = coalesce(nullif(g->>'category', ''), t.category),
        tags = case when cardinality(v_tags) > 0 then v_tags else t.tags end,   -- rattachement automatique : sujets conservés
        updated_at = case when (v_title is not null and v_title is distinct from t.title)
                            or (v_summary is not null and v_summary is distinct from t.summary)
                            or v_update is not null
                          then v_at else t.updated_at end,
        updated_run_id = v_run_id
      where t.id = (g->>'thread_id')::bigint
      returning t.id into v_thread_id;

      if v_thread_id is not null and v_update is not null then
        insert into public.thread_updates (thread_id, run_id, text, created_at)
        values (v_thread_id, v_run_id, v_update, v_at);
      end if;
    end if;

    if v_thread_id is null then   -- nouvelle histoire, ou identifiant inconnu
      insert into public.threads (title, summary, category, tags, first_seen, last_seen, updated_at, created_run_id, updated_run_id)
      values (coalesce(v_title, 'Sans titre'), coalesce(v_summary, ''), g->>'category', v_tags, v_at, v_at, v_at, v_run_id, v_run_id)
      returning id into v_thread_id;
    end if;

    insert into public.thread_articles (article_url, thread_id, run_id, is_mirror, added_at)
    select m->>'url', v_thread_id, v_run_id, coalesce((m->>'mirror')::boolean, false), v_at
    from jsonb_array_elements(coalesce(g->'members', '[]'::jsonb)) m
    where exists (select 1 from public.articles a where a.url = m->>'url')
    on conflict (article_url) do nothing;
  end loop;

  return v_run_id;
end;
$function$;
