"""Worker zimb-news.

Toutes les 15 min (GitHub Actions) :
1. liste les derniers fichiers GDELT publiés sur Hugging Face (openalphalab/gdelt-news)
2. lit les articles en français avec DuckDB et écarte les adresses déjà traitées
3. regroupe les reprises d'un même article (titres d'un même groupe de presse)
4. envoie à Claude les nouveaux articles et les histoires déjà suivies : il rattache chaque
   article à une histoire existante (en mettant son résumé à jour) ou crée une nouvelle histoire
5. enregistre le résultat dans Supabase (fonction ingest_run_v2)

Variables d'environnement :
  SUPABASE_URL, SUPABASE_SECRET_KEY   (obligatoires)
  ANTHROPIC_API_KEY                   (obligatoire sauf DRY_RUN=1)
  HF_TOKEN                            (facultatif, évite les limites de débit Hugging Face)
  CLAUDE_MODEL   (défaut claude-opus-5-5)   CLAUDE_EFFORT (défaut medium)
  WINDOW_MINUTES (défaut 15)                ACTIVE_HOURS (défaut 6 : histoires proposées au rattachement)
  DRY_RUN=1 : lit GDELT et les histoires actives, affiche le prompt, n'appelle pas Claude et n'écrit rien
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import sys
import tempfile
import unicodedata
from pathlib import Path
from urllib.parse import urlparse

import anthropic
import duckdb
import httpx

REPO = "openalphalab/gdelt-news"
HF_API = f"https://huggingface.co/api/datasets/{REPO}/tree/main/data"
HF_FILE = f"https://huggingface.co/datasets/{REPO}/resolve/main/"

MODEL = os.environ.get("CLAUDE_MODEL") or "claude-opus-5-5"
EFFORT = os.environ.get("CLAUDE_EFFORT") or "medium"
WINDOW = dt.timedelta(minutes=int(os.environ.get("WINDOW_MINUTES") or 15))
ACTIVE_HOURS = int(os.environ.get("ACTIVE_HOURS") or 6)
DRY_RUN = os.environ.get("DRY_RUN") == "1"
MAX_ARTICLES = 400
MAX_THREADS = 40
SNIPPET_CHARS = 320

CATEGORIES = ["politique", "international", "économie", "société", "faits divers",
              "sport", "culture", "sciences et tech", "santé", "environnement", "autre"]


def log(*args):
    print(*args, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- Hugging Face

def hf_headers() -> dict:
    token = os.environ.get("HF_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def parse_stamp(s: str) -> dt.datetime:
    return dt.datetime.strptime(s, "%Y%m%d%H%M%S").replace(tzinfo=dt.timezone.utc)


def list_recent_files(http: httpx.Client, now: dt.datetime) -> list[tuple[str, dt.datetime, dt.datetime]]:
    """Fichiers Parquet des dossiers du jour et de la veille (UTC) : (chemin, début, fin)."""
    files = []
    for day in (now, now - dt.timedelta(days=1)):
        r = http.get(f"{HF_API}/{day:%Y/%m/%d}", headers=hf_headers())
        if r.status_code == 404:
            continue
        r.raise_for_status()
        for entry in r.json():
            m = re.search(r"(\d{14})-(\d{14})\.parquet$", entry.get("path", ""))
            if m:
                files.append((entry["path"], parse_stamp(m.group(1)), parse_stamp(m.group(2))))
    return sorted(files, key=lambda f: f[2])


def download(http: httpx.Client, path: str, dest: Path) -> Path:
    out = dest / Path(path).name
    with http.stream("GET", HF_FILE + path, headers=hf_headers(), follow_redirects=True) as r:
        r.raise_for_status()
        with out.open("wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
    return out


# --------------------------------------------------------------------------- Articles

def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def domain_of(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def slug_of(url: str) -> str:
    """Dernier segment d'URL, sans extension ni identifiant numérique."""
    seg = [p for p in urlparse(url).path.split("/") if p]
    if not seg:
        return ""
    s = seg[-1].lower()
    s = re.sub(r"\.(php|html?|aspx?)$", "", s)
    for _ in range(3):  # plusieurs identifiants peuvent s'enchaîner (4265337-20261125-…)
        s = re.sub(r"[_.-]?\d{5,}[_.-]?$", "", s)   # identifiant final (…-13669313, .3000531.)
        s = re.sub(r"^\d{5,}[_.-]?", "", s)         # identifiant initial (4265337-…)
        s = re.sub(r"[_-][a-z0-9]{20,}$", "", s)    # identifiant opaque (…-6FKAWUKAX…)
    return s.strip("-_.")


def title_from_slug(slug: str) -> str:
    words = re.sub(r"[-_]+", " ", slug).strip()
    return (words[:1].upper() + words[1:])[:160] if words else ""


def load_rows(paths: list[Path]) -> list[dict]:
    """Articles en français des fichiers, une ligne par adresse (sans regroupement)."""
    rows = duckdb.connect().execute(
        """
        SELECT source_url, date, text
        FROM read_parquet(?)
        WHERE language = 'fr' AND length(text) > 500
        QUALIFY row_number() OVER (PARTITION BY source_url ORDER BY length(text) DESC) = 1
        """,
        [[str(p) for p in paths]],
    ).fetchall()
    out = []
    for url, date, text in rows:
        slug = slug_of(url)
        if len(slug) < 15:          # page d'accueil, rubrique, page de service
            continue
        clean = re.sub(r"\s+", " ", text).strip()
        out.append({
            "url": url,
            "domain": domain_of(url).split(":")[0],
            "title_guess": title_from_slug(slug),
            "snippet": clean[:SNIPPET_CHARS],
            "observed_at": date,
            "_slug": slug,
            "_fp": hashlib.sha1(strip_accents(clean[:400].lower()).encode()).hexdigest(),
        })
    return out


def group_mirrors(rows: list[dict]) -> list[dict]:
    """Regroupe les reprises syndiquées : même slug ou même début de texte."""
    articles, index = [], {}
    for art in rows:
        key = index.get(art["_slug"])
        if key is None:
            key = index.get(art["_fp"])
        if key is not None:
            articles[key]["mirrors"].append(art)
            continue
        art["mirrors"] = []
        index[art["_slug"]] = index[art["_fp"]] = len(articles)
        articles.append(art)
    return articles[:MAX_ARTICLES]


def public_fields(a: dict) -> dict:
    return {k: a[k] for k in ("url", "domain", "title_guess", "snippet", "observed_at")}


# --------------------------------------------------------------------------- Claude

SYSTEM = f"""Tu es secrétaire de rédaction pour un site qui suit les histoires dont la presse francophone parle le plus en ce moment.

On te donne deux listes :
- les HISTOIRES DÉJÀ SUIVIES ces dernières heures (identifiant, catégorie, titre, résumé abrégé, évolutions déjà connues) ;
- les NOUVEAUX ARTICLES repérés par GDELT ces 15 dernières minutes : identifiant, média, titre déduit de l'URL, début du texte. Le texte est reconstitué automatiquement : il peut commencer au milieu d'une phrase, être dans le désordre ou contenir des répétitions.

Ta tâche :
1. Pour chaque nouvel article d'actualité, décide s'il traite d'une histoire déjà suivie (même événement précis, ou sa suite directe) ou d'une nouvelle histoire. Rattache-le à l'histoire existante dès que c'est le cas : il ne faut jamais créer de doublon d'une histoire déjà suivie, même si l'angle ou la formulation diffère.
2. Regroupe entre eux les nouveaux articles qui parlent d'une même nouvelle histoire. Un regroupement porte sur un événement précis, pas seulement sur un thème général.
3. Écarte ce qui n'est pas de l'actualité : programmes TV, horoscopes, jeux, promotions, recettes, conseils pratiques intemporels, pages de service. Un article peut n'appartenir à aucun groupe.
4. Pour une NOUVELLE histoire (existing_story_id = 0), donne en français :
   - title : titre factuel de 90 caractères maximum ;
   - summary : résumé neutre de 2 à 3 phrases, fondé uniquement sur les extraits, sans rien inventer ni extrapoler ;
   - update : chaîne vide ;
   - category : parmi {", ".join(CATEGORIES)} ;
   - article_ids : les identifiants des nouveaux articles du groupe.
5. Pour une histoire DÉJÀ SUIVIE (existing_story_id = son identifiant), ne réécris pas le résumé :
   - title : chaîne vide, sauf si un fait nouveau change la nature de l'histoire (bilan, décision, dénouement) ; alors un nouveau titre ;
   - summary : chaîne vide ;
   - update : UNE phrase courte (25 mots maximum) avec le fait nouveau apporté par les nouveaux articles, seulement s'il n'est ni dans le résumé ni dans les évolutions déjà connues ; sinon chaîne vide ;
   - category : la catégorie actuelle ;
   - article_ids : les identifiants des nouveaux articles rattachés.
6. Une nouvelle histoire couverte par un seul article n'est retenue que si elle est manifestement importante. N'inclus une histoire existante que si au moins un nouvel article s'y rattache."""

SCHEMA = {
    "type": "object",
    "properties": {
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "existing_story_id": {"type": "integer"},
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "update": {"type": "string"},
                    "category": {"type": "string", "enum": CATEGORIES},
                    "article_ids": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["existing_story_id", "title", "summary", "update", "category", "article_ids"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["groups"],
    "additionalProperties": False,
}


def build_prompt(articles: list[dict], threads: list[dict]) -> str:
    def thread_line(t: dict) -> str:
        line = f"[#{t['id']}] ({t['category']}) {t['title']} — {t['summary']}"
        if t.get("updates"):
            line += "\n    Évolutions déjà connues : " + " | ".join(t["updates"])
        return line
    known = "\n".join(thread_line(t) for t in threads) or "(aucune)"
    lines = []
    for i, a in enumerate(articles):
        extra = f" (+{len(a['mirrors'])} reprises)" if a["mirrors"] else ""
        lines.append(f"[{i}] {a['domain']}{extra} | {a['title_guess']} | {a['snippet']}")
    return f"HISTOIRES DÉJÀ SUIVIES :\n{known}\n\nNOUVEAUX ARTICLES :\n" + "\n".join(lines)


def ask_claude(prompt: str) -> tuple[list[dict], dict]:
    client = anthropic.Anthropic()
    params = dict(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
        output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}},
    )
    try:
        # Repli automatique sur un autre modèle si le modèle principal refuse la requête
        resp = client.beta.messages.create(
            **params, betas=["server-side-fallback-2026-07-01"], fallbacks="default"
        )
    except anthropic.BadRequestError as e:
        log(f"Repli serveur non accepté pour {MODEL} ({e.message}) : appel standard.")
        resp = client.messages.create(**params)

    usage = {"model": resp.model, "input_tokens": resp.usage.input_tokens,
             "output_tokens": resp.usage.output_tokens}
    if resp.stop_reason == "refusal":
        raise RuntimeError("Claude a refusé la requête")
    if resp.stop_reason == "max_tokens":
        raise RuntimeError("Réponse tronquée (max_tokens atteint)")
    text = next(b.text for b in resp.content if b.type == "text")
    return json.loads(text)["groups"], usage


def build_groups(raw: list[dict], articles: list[dict], threads: list[dict]) -> list[dict]:
    """Valide la réponse : identifiants connus, chaque article dans un seul groupe."""
    known_ids = {t["id"] for t in threads}
    used, groups = set(), []
    for g in raw:
        ids = [i for i in dict.fromkeys(g["article_ids"]) if 0 <= i < len(articles) and i not in used]
        if not ids:
            continue
        used.update(ids)
        thread_id = g["existing_story_id"] if g["existing_story_id"] in known_ids else 0
        members = []
        for i in ids:
            members.append({"url": articles[i]["url"], "mirror": False, "_a": articles[i]})
            members += [{"url": m["url"], "mirror": True, "_a": m} for m in articles[i]["mirrors"]]
        title, summary, update = g["title"].strip()[:200], g["summary"].strip(), g["update"].strip()[:300]
        if not thread_id and not (title and summary):
            # Nouvelle histoire incomplète : on ne peut pas l'afficher, on l'ignore
            log(f"  ! groupe ignoré (nouvelle histoire sans titre ou résumé) : {g['article_ids']}")
            continue
        groups.append({
            "thread_id": thread_id,
            "title": title,
            "summary": summary if not thread_id else "",   # le résumé d'une histoire suivie n'est jamais réécrit
            "update": update if thread_id else "",
            "category": g["category"],
            "members": members,
        })
    return groups


# --------------------------------------------------------------------------- Supabase

def supabase_headers() -> dict:
    key = os.environ["SUPABASE_SECRET_KEY"]
    h = {"apikey": key, "Content-Type": "application/json"}
    if key.startswith("eyJ"):  # ancienne clé service_role (JWT)
        h["Authorization"] = f"Bearer {key}"
    return h


def rpc(http: httpx.Client, name: str, body: dict):
    r = http.post(f"{os.environ['SUPABASE_URL']}/rest/v1/rpc/{name}", json=body, headers=supabase_headers())
    if r.status_code >= 400:
        raise RuntimeError(f"Supabase {name} {r.status_code}: {r.text[:500]}")
    return r.json()


def last_window_end(http: httpx.Client) -> dt.datetime | None:
    r = http.get(
        f"{os.environ['SUPABASE_URL']}/rest/v1/runs",
        params={"select": "window_end", "status": "in.(ok,no_data)", "order": "id.desc", "limit": "1"},
        headers=supabase_headers(),
    )
    r.raise_for_status()
    rows = r.json()
    return dt.datetime.fromisoformat(rows[0]["window_end"]) if rows and rows[0]["window_end"] else None


def known_urls(http: httpx.Client, urls: list[str]) -> set[str]:
    known = set()
    for i in range(0, len(urls), 500):
        for row in rpc(http, "known_urls", {"urls": urls[i:i + 500]}):
            known.add(row if isinstance(row, str) else next(iter(row.values())))
    return known


# --------------------------------------------------------------------------- Main

def main() -> int:
    started = dt.datetime.now(dt.timezone.utc)
    required = ["SUPABASE_URL", "SUPABASE_SECRET_KEY"] + ([] if DRY_RUN else ["ANTHROPIC_API_KEY"])
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        log(f"Configuration incomplète, exécution ignorée. Manquant : {', '.join(missing)}")
        return 0
    http = httpx.Client(timeout=60)

    files = list_recent_files(http, started)
    if not files:
        log("Aucun fichier GDELT trouvé.")
        return 0
    window_end = files[-1][2]
    window_start = window_end - WINDOW
    selected = [f for f in files if f[2] > window_start]

    previous = last_window_end(http)
    if previous and window_end <= previous and not DRY_RUN:
        log(f"Rien de nouveau depuis {previous:%H:%M} UTC.")
        return 0

    log(f"Fenêtre {window_start:%Y-%m-%d %H:%M} → {window_end:%H:%M} UTC, {len(selected)} fichier(s)")
    with tempfile.TemporaryDirectory() as tmp:
        rows = load_rows([download(http, f[0], Path(tmp)) for f in selected])
    known = known_urls(http, [r["url"] for r in rows])
    fresh = [r for r in rows if r["url"] not in known]
    articles = group_mirrors(fresh)
    threads = rpc(http, "active_threads_v2", {"hours": ACTIVE_HOURS, "lim": MAX_THREADS})
    log(f"{len(rows)} articles en français, {len(known)} déjà vus, "
        f"{len(articles)} nouveaux articles distincts, {len(threads)} histoires actives")

    base = {
        "started_at": started.isoformat(),
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "source_files": [f[0] for f in selected],
        "n_articles": len(articles),
        "seen_urls": [r["url"] for r in fresh],
    }
    prompt = build_prompt(articles, threads)

    if DRY_RUN:
        log(f"DRY_RUN : prompt de {len(prompt):,} caractères. Aperçu :\n{prompt[:2500]}")
        return 0

    if not articles:
        rpc(http, "ingest_run_v2", {"payload": {**base, "status": "no_data", "groups": []}})
        log("Aucun nouvel article.")
        return 0

    try:
        raw, usage = ask_claude(prompt)
    except Exception as e:  # trace de l'échec en base (sans marquer les articles comme vus), puis échec du job
        rpc(http, "ingest_run_v2", {"payload": {**base, "seen_urls": [], "status": "error",
                                                "error": str(e)[:1000], "model": MODEL, "groups": []}})
        raise

    groups = build_groups(raw, articles, threads)
    payload = {
        **base, **usage, "status": "ok",
        "articles": [public_fields(m["_a"]) for g in groups for m in g["members"]],
        "groups": [{**g, "members": [{"url": m["url"], "mirror": m["mirror"]} for m in g["members"]]}
                   for g in groups],
    }
    run_id = rpc(http, "ingest_run_v2", {"payload": payload})

    titles = {t["id"]: t["title"] for t in threads}
    n_new = sum(1 for g in groups if not g["thread_id"])
    log(f"Run {run_id} : {len(groups)} groupes ({n_new} nouvelles histoires, {len(groups) - n_new} complétées), "
        f"{usage['input_tokens']} tokens en entrée, {usage['output_tokens']} en sortie ({usage['model']})")
    for g in groups:
        n = sum(1 for m in g["members"] if not m["mirror"])
        if g["thread_id"]:
            changed = f" (nouveau titre : {g['title']})" if g["title"] and g["title"] != titles[g["thread_id"]] else ""
            evo = f" — évolution : {g['update']}" if g["update"] else ""
            log(f"  + #{g['thread_id']} ← {n} article(s) : {titles[g['thread_id']]}{changed}{evo}")
        else:
            log(f"  * nouvelle ← {n} article(s) : {g['title']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
