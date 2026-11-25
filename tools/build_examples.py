"""Génère la galerie d'exemples (site/exemples.json) : une synthèse sur 3 jours par sujet d'exemple.

1. récupère les sources de chaque sujet via l'API de veille (route /sources, sans IA) ;
2. envoie toutes les synthèses à Claude en un seul lot (Message Batches API, -50 %) ;
3. écrit site/exemples.json, lu par la page publique site/exemples.html.

Variables : ANTHROPIC_API_KEY, NEWS_API_KEY (clé zn_…), NEWS_API_URL (facultatif),
            EXAMPLES_MODEL (défaut claude-opus-5-5), EXAMPLES_EFFORT (défaut medium).
Le texte des consignes reprend celui de supabase/functions/news-api/index.ts (DIGEST_SYSTEM) :
le garder synchronisé.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import anthropic
import httpx
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

ROOT = Path(__file__).resolve().parent.parent
API = os.environ.get("NEWS_API_URL") or "https://rtygalgkixjttiezdcym.supabase.co/functions/v1/news-api"
MODEL = os.environ.get("EXAMPLES_MODEL") or "claude-opus-5-5"
EFFORT = os.environ.get("EXAMPLES_EFFORT") or "medium"
DAYS = 3

DIGEST_SYSTEM = """Tu rédiges en français la veille quotidienne d'ingénieurs et de chercheurs en IA sur un sujet précis. Les sources récentes sont de plusieurs types : versions publiées de logiciels (GitHub), modèles publiés (Hugging Face), annonces de labos (blogs), articles de recherche (arXiv, Hugging Face Daily Papers), discussions (Hacker News). Chaque source a un identifiant [S1], [S2]… et parfois une ligne « Signaux » (code publié, votes, poids publiés, version publiée).

Règles :
- Ne retiens que ce qui concerne vraiment le sujet ; ignore les sources hors sujet.
- Fonde-toi uniquement sur les sources fournies, sans rien inventer. Présente les résultats d'articles comme des affirmations des auteurs.
- Donne les chiffres quand les sources en fournissent (accélération, mémoire, précision, taille de modèle, benchmark et point de comparaison). N'en invente jamais.
- Indique quand c'est le cas que le code ou les poids sont publiés : c'est décisif pour des praticiens. Privilégie les sources aux signaux forts.
- Garde les termes techniques en anglais quand c'est l'usage (KV cache, speculative decoding, fine-tuning…). Pas de vulgarisation : le lecteur est du métier.
- Regroupe les sources qui parlent de la même avancée. S'il y a peu de sources pertinentes, dis-le : ne remplis pas artificiellement.

Produis :
- headline : un titre de 100 caractères maximum qui résume la période ;
- overview : 2 à 4 phrases sur ce qui ressort de la période ;
- releases : de 0 à 6 sorties concrètes utilisables tout de suite (nouvelle version d'un logiciel, modèle ou poids publiés, outil open source), chacune avec un titre, une description de 1 à 2 phrases (ce qui change, pour qui) et les identifiants des sources ;
- highlights : de 0 à 6 avancées de recherche, de la plus importante à la moins importante, chacune avec un titre, un résumé de 2 à 4 phrases avec les chiffres disponibles, une phrase « pourquoi c'est important », et les identifiants des sources ;
- trends : 0 à 4 tendances ou signaux faibles en une phrase chacune."""

_ENTRY = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "source_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "summary", "source_ids"],
    "additionalProperties": False,
}
DIGEST_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "overview": {"type": "string"},
        "releases": {"type": "array", "items": _ENTRY},
        "highlights": {
            "type": "array",
            "items": {
                **_ENTRY,
                "properties": {**_ENTRY["properties"], "why_it_matters": {"type": "string"}},
                "required": ["title", "summary", "why_it_matters", "source_ids"],
            },
        },
        "trends": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["headline", "overview", "releases", "highlights", "trends"],
    "additionalProperties": False,
}


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def render_items(items: list[dict]) -> str:
    out = []
    for i in items:
        head = f"[{i['id']}] ({i['source']}, {i['date'][:10]}{', ' + i['score'] if i.get('score') else ''}) {i['title']}"
        sig = f"\nSignaux : {' ; '.join(i['signals'])}" if i.get("signals") else ""
        out.append(f"{head}{sig}\n{i['text']}")
    return "\n\n".join(out)


def pick(items: list[dict], ids: list[str]) -> list[dict]:
    by_id = {i["id"]: i for i in items}
    keep = ("id", "source", "title", "url", "date", "score", "signals")
    return [{k: by_id[x].get(k) for k in keep} for x in dict.fromkeys(ids) if x in by_id]


def main() -> int:
    wanted = json.loads((ROOT / "tools" / "example_topics.json").read_text(encoding="utf-8"))
    http = httpx.Client(timeout=120, headers={"x-api-key": os.environ["NEWS_API_KEY"]})
    known = {t["name"]: t for t in http.get(f"{API}/topics").raise_for_status().json()["topics"]}

    # 1. Sources de chaque sujet (pause entre les appels : arXiv demande de l'espacer)
    topics, requests = [], []
    for w in wanted:
        t = known.get(w["name"])
        if not t:
            log(f"Sujet absent de l'API, ignoré : {w['name']}")
            continue
        r = http.get(f"{API}/topics/{t['slug']}/sources", params={"days": DAYS})
        r.raise_for_status()
        src = r.json()
        log(f"{t['slug']:55} {len(src['items']):3} sources {src['errors'] or ''}")
        topics.append({"slug": t["slug"], "name": t["name"], "description": t["description"],
                       "items": src["items"], "errors": src["errors"]})
        if src["items"]:
            prompt = (f"Sujet : {t['name']}\nPérimètre : {t['description']}\nPériode : {DAYS} dernier(s) jour(s)"
                      f"\n\nSOURCES :\n\n{render_items(src['items'])}")
            requests.append(Request(
                custom_id=t["slug"][:64],
                params=MessageCreateParamsNonStreaming(
                    model=MODEL,
                    max_tokens=16000,
                    system=DIGEST_SYSTEM,
                    messages=[{"role": "user", "content": prompt}],
                    output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": DIGEST_SCHEMA}},
                ),
            ))
        time.sleep(3)

    # 2. Un seul lot pour toutes les synthèses
    client = anthropic.Anthropic()
    batch = client.messages.batches.create(requests=requests)
    log(f"Lot {batch.id} : {len(requests)} synthèses avec {MODEL}")
    deadline = time.time() + 50 * 60
    while True:
        batch = client.messages.batches.retrieve(batch.id)
        if batch.processing_status == "ended":
            break
        if time.time() > deadline:
            log(f"Lot toujours en cours après 50 min ({batch.id}) : relancer plus tard.")
            return 1
        log(f"  en cours : {batch.request_counts.processing} restantes")
        time.sleep(30)
    log(f"Terminé : {batch.request_counts.succeeded} réussies, {batch.request_counts.errored} en erreur")

    # 3. Résultats
    results, tok_in, tok_out = {}, 0, 0
    for res in client.messages.batches.results(batch.id):
        if res.result.type != "succeeded":
            log(f"  {res.custom_id} : {res.result.type}")
            continue
        msg = res.result.message
        if msg.stop_reason in ("refusal", "max_tokens"):
            log(f"  {res.custom_id} : arrêt {msg.stop_reason}")
            continue
        text = next((b.text for b in msg.content if b.type == "text"), "")
        results[res.custom_id] = json.loads(text)
        tok_in += msg.usage.input_tokens
        tok_out += msg.usage.output_tokens

    gallery = []
    for t in topics:
        d = results.get(t["slug"][:64])
        if d is None:
            continue
        items = t["items"]
        used = [x for h in d["releases"] + d["highlights"] for x in h["source_ids"]]
        gallery.append({
            "slug": t["slug"], "name": t["name"], "description": t["description"],
            "n_items": len(items),
            "content": {
                **d,
                "releases": [{**h, "sources": pick(items, h["source_ids"])} for h in d["releases"]],
                "highlights": [{**h, "sources": pick(items, h["source_ids"])} for h in d["highlights"]],
                "n_sources_used": len(set(used)),
            },
        })

    out = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "days": DAYS, "model": MODEL, "topics": gallery,
    }
    path = ROOT / "site" / "exemples.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    # Tarif du lot : moitié du tarif standard (Opus 5.5 : 4 $ / 20 $ par million de tokens)
    price = {"claude-opus-5-5": (4, 20), "claude-sonnet-5-5": (2, 10), "claude-haiku-5-5": (0.1, 0.5)}.get(MODEL, (4, 20))
    cost = (tok_in * price[0] + tok_out * price[1]) / 2 / 1e6
    log(f"{len(gallery)} synthèses écrites dans {path.name} · {tok_in:,} tokens en entrée, {tok_out:,} en sortie "
        f"· coût estimé {cost:.2f} $")
    return 0


if __name__ == "__main__":
    sys.exit(main())
