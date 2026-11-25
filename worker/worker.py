"""Worker zimb-news.

Toutes les 15 min (GitHub Actions) :
1. liste les derniers fichiers GDELT publiés sur Hugging Face (openalphalab/gdelt-news)
2. lit les articles en français avec DuckDB
3. supprime les doublons (même article republié par plusieurs titres d'un groupe de presse)
4. demande à Claude de regrouper les articles par histoire et de les résumer
5. enregistre le résultat dans Supabase (fonction ingest_run)

Variables d'environnement :
  SUPABASE_URL, SUPABASE_SECRET_KEY   (obligatoires)
  ANTHROPIC_API_KEY                   (obligatoire sauf DRY_RUN=1)
  HF_TOKEN                            (facultatif, évite les limites de débit Hugging Face)
  CLAUDE_MODEL   (défaut claude-opus-5-5)   CLAUDE_EFFORT (défaut medium)
  WINDOW_MINUTES (défaut 15)                DRY_RUN=1 : n'appelle ni Claude ni Supabase en écriture
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
DRY_RUN = os.environ.get("DRY_RUN") == "1"
MAX_ARTICLES = 400
TOP_STORIES = 10
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
        url = f"{HF_API}/{day:%Y/%m/%d}"
        r = http.get(url, headers=hf_headers())
        if r.status_code == 404:
            continue
        r.raise_for_status()
        for entry in r.json():
            path = entry.get("path", "")
            m = re.search(r"(\d{14})-(\d{14})\.parquet$", path)
            if m:
                files.append((path, parse_stamp(m.group(1)), parse_stamp(m.group(2))))
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


def load_articles(paths: list[Path]) -> list[dict]:
    con = duckdb.connect()
    rows = con.execute(
        """
        SELECT source_url, date, text
        FROM read_parquet(?)
        WHERE language = 'fr' AND length(text) > 500
        QUALIFY row_number() OVER (PARTITION BY source_url ORDER BY length(text) DESC) = 1
        """,
        [[str(p) for p in paths]],
    ).fetchall()

    articles, seen = [], {}
    for url, date, text in rows:
        slug = slug_of(url)
        if len(slug) < 15:          # page d'accueil, rubrique, page de service
            continue
        clean = re.sub(r"\s+", " ", text).strip()
        fingerprint = hashlib.sha1(strip_accents(clean[:400].lower()).encode()).hexdigest()
        art = {
            "url": url,
            "domain": domain_of(url).split(":")[0],
            "title_guess": title_from_slug(slug),
            "snippet": clean[:SNIPPET_CHARS],
            "observed_at": date,
        }
        # Doublon syndiqué : même slug ou même début de texte qu'un article déjà vu
        key = seen.get(slug) or seen.get(fingerprint)
        if key is not None:
            articles[key]["mirrors"].append(art)
            continue
        art["mirrors"] = []
        seen[slug] = seen[fingerprint] = len(articles)
        articles.append(art)
    return articles[:MAX_ARTICLES]


# --------------------------------------------------------------------------- Claude

SYSTEM = f"""Tu es secrétaire de rédaction pour un site qui présente les histoires dont la presse francophone parle le plus en ce moment.

On te donne des articles repérés par GDELT ces 15 dernières minutes. Pour chacun : un identifiant, le média, un titre déduit de l'URL et le début du texte. Le texte est reconstitué automatiquement : il peut commencer au milieu d'une phrase, être dans le désordre ou contenir des répétitions.

Ta tâche :
1. Regroupe les articles qui traitent du MÊME événement ou de la même histoire précise (pas seulement du même thème général). Un article peut n'appartenir à aucune histoire.
2. Écarte ce qui n'est pas de l'actualité : programmes TV, horoscopes, jeux, promotions, recettes, conseils pratiques intemporels, pages de service.
3. Pour chaque histoire, rédige en français :
   - un titre factuel de 90 caractères maximum ;
   - un résumé neutre de 2 à 3 phrases, fondé uniquement sur les extraits fournis, sans rien inventer ni extrapoler. Si un point est incertain dans les extraits, ne l'affirme pas ;
   - une catégorie parmi : {", ".join(CATEGORIES)}.
4. Donne au maximum 15 histoires, en privilégiant celles couvertes par le plus d'articles distincts. Une histoire couverte par un seul article n'est retenue que si elle est manifestement importante."""

SCHEMA = {
    "type": "object",
    "properties": {
        "stories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "category": {"type": "string", "enum": CATEGORIES},
                    "article_ids": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["title", "summary", "category", "article_ids"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["stories"],
    "additionalProperties": False,
}


def build_prompt(articles: list[dict]) -> str:
    lines = []
    for i, a in enumerate(articles):
        extra = f" (+{len(a['mirrors'])} reprises)" if a["mirrors"] else ""
        lines.append(f"[{i}] {a['domain']}{extra} | {a['title_guess']} | {a['snippet']}")
    return "Articles :\n\n" + "\n".join(lines)


def ask_claude(articles: list[dict]) -> tuple[list[dict], dict]:
    client = anthropic.Anthropic()
    params = dict(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM,
        messages=[{"role": "user", "content": build_prompt(articles)}],
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
    return json.loads(text)["stories"], usage


def rank_stories(raw: list[dict], articles: list[dict]) -> list[dict]:
    stories = []
    for s in raw:
        ids = sorted({i for i in s["article_ids"] if 0 <= i < len(articles)})
        if not ids:
            continue
        members = [articles[i] for i in ids]
        all_arts = members + [m for a in members for m in a["mirrors"]]
        stories.append({
            "title": s["title"].strip()[:200],
            "summary": s["summary"].strip(),
            "category": s["category"],
            "n_articles": len(members),                       # articles distincts
            "n_sources": len({a["domain"] for a in all_arts}),  # médias, reprises comprises
            "urls": [a["url"] for a in all_arts],
            "_articles": all_arts,
            # médias ayant écrit leur propre article (les reprises syndiquées ne comptent pas)
            "_independent": len({a["domain"] for a in members}),
        })
    # D'abord le nombre de médias indépendants (étendue de la couverture), puis les articles distincts
    stories.sort(key=lambda s: (s.pop("_independent"), s["n_articles"], s["n_sources"]), reverse=True)
    for rank, s in enumerate(stories[:TOP_STORIES], 1):
        s["rank"] = rank
    return stories[:TOP_STORIES]


# --------------------------------------------------------------------------- Supabase

def supabase_headers() -> dict:
    key = os.environ["SUPABASE_SECRET_KEY"]
    h = {"apikey": key, "Content-Type": "application/json"}
    if key.startswith("eyJ"):  # ancienne clé service_role (JWT)
        h["Authorization"] = f"Bearer {key}"
    return h


def last_window_end(http: httpx.Client) -> dt.datetime | None:
    r = http.get(
        f"{os.environ['SUPABASE_URL']}/rest/v1/runs",
        params={"select": "window_end", "status": "eq.ok", "order": "id.desc", "limit": "1"},
        headers=supabase_headers(),
    )
    r.raise_for_status()
    rows = r.json()
    return dt.datetime.fromisoformat(rows[0]["window_end"]) if rows and rows[0]["window_end"] else None


def ingest(http: httpx.Client, payload: dict) -> int:
    r = http.post(
        f"{os.environ['SUPABASE_URL']}/rest/v1/rpc/ingest_run",
        json={"payload": payload},
        headers=supabase_headers(),
    )
    if r.status_code >= 400:
        raise RuntimeError(f"Supabase {r.status_code}: {r.text[:500]}")
    return r.json()


# --------------------------------------------------------------------------- Main

def main() -> int:
    started = dt.datetime.now(dt.timezone.utc)
    if not DRY_RUN:
        missing = [k for k in ("SUPABASE_URL", "SUPABASE_SECRET_KEY", "ANTHROPIC_API_KEY") if not os.environ.get(k)]
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

    if not DRY_RUN:
        previous = last_window_end(http)
        if previous and window_end <= previous:
            log(f"Rien de nouveau depuis {previous:%H:%M} UTC.")
            return 0

    log(f"Fenêtre {window_start:%Y-%m-%d %H:%M} → {window_end:%H:%M} UTC, {len(selected)} fichier(s)")
    with tempfile.TemporaryDirectory() as tmp:
        paths = [download(http, f[0], Path(tmp)) for f in selected]
        articles = load_articles(paths)
    log(f"{len(articles)} articles distincts en français")

    base = {
        "started_at": started.isoformat(),
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "source_files": [f[0] for f in selected],
        "n_articles": len(articles),
    }

    if DRY_RUN:
        prompt = build_prompt(articles)
        log(f"DRY_RUN : prompt de {len(prompt):,} caractères. Aperçu :\n{prompt[:1500]}")
        return 0

    if len(articles) < 5:
        ingest(http, {**base, "status": "no_data", "stories": []})
        log("Trop peu d'articles, exécution enregistrée sans histoire.")
        return 0

    try:
        raw, usage = ask_claude(articles)
    except Exception as e:  # on garde une trace de l'échec en base, puis on échoue le job
        ingest(http, {**base, "status": "error", "error": str(e)[:1000], "model": MODEL, "stories": []})
        raise

    stories = rank_stories(raw, articles)
    used = {a["url"]: a for s in stories for a in s.pop("_articles")}
    payload = {
        **base, **usage, "status": "ok",
        "articles": [{k: a[k] for k in ("url", "domain", "title_guess", "snippet", "observed_at")}
                     for a in used.values()],
        "stories": stories,
    }
    run_id = ingest(http, payload)
    log(f"Run {run_id} : {len(stories)} histoires, {usage['input_tokens']} tokens en entrée, "
        f"{usage['output_tokens']} en sortie ({usage['model']})")
    for s in stories:
        log(f"  {s['rank']:>2}. [{s['n_articles']} art. / {s['n_sources']} médias] {s['title']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
