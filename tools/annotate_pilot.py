"""Pilote d'annotation d'articles arXiv : mesure le coût réel et la qualité par modèle.

Récupère N articles IA récents sur arXiv (titre, résumé, commentaire des auteurs), demande à
chaque modèle une fiche structurée, puis affiche : tokens réels par article, coût extrapolé au
tarif des lots (Message Batches, -50 %), accord entre modèles et part des chiffres extraits
qu'on retrouve tels quels dans le résumé.

Variables : ANTHROPIC_API_KEY ; PILOT_N (défaut 30) ; PILOT_MODELS (liste séparée par des virgules).
Écrit annotate_pilot.json (toutes les fiches) dans le dossier courant.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import anthropic
import httpx

N = int(os.environ.get("PILOT_N") or 30)
MODELS = [m.strip() for m in (os.environ.get("PILOT_MODELS") or
                              "claude-haiku-5-5,claude-sonnet-5-5,claude-opus-5-5").split(",") if m.strip()]
EFFORT = os.environ.get("PILOT_EFFORT") or "low"
TOTAL_2026 = 130_342   # articles cs.AI, cs.LG, cs.CL, cs.CV, stat.ML du 1er janvier au 25 novembre 2026
PRICES = {  # $ par million de tokens (entrée, sortie), tarif standard
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-opus-5-5": (4.0, 20.0),
}
QUERY = "(cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:cs.CV OR cat:stat.ML)"
NS = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}

SYSTEM = """Tu remplis la fiche structurée d'un article de recherche en IA à partir de son titre, de son résumé et du commentaire des auteurs. Le lecteur final est ingénieur ou chercheur.

Règles :
- N'utilise que le texte fourni. N'invente rien : si une information n'y est pas, laisse le champ vide ou la liste vide.
- tldr_fr : une phrase en français (30 mots maximum) qui dit ce que l'article apporte, termes techniques usuels en anglais.
- contribution : le type principal de contribution.
- topics : 1 à 4 expressions courtes en anglais, en minuscules, sous leur forme la plus courante (« speculative decoding », « retrieval-augmented generation »).
- modalities : les modalités de données concernées.
- claims : 0 à 3 résultats chiffrés annoncés par les auteurs. Recopie la valeur telle qu'elle est écrite (« 2.7x », « +13.7% », « 81.4% »). baseline = le point de comparaison s'il est donné, benchmark = le jeu de test s'il est donné, sinon chaîne vide.
- benchmarks : jeux de données ou benchmarks d'évaluation nommés.
- models : modèles nommés, utilisés ou introduits (« Llama-3.1-8B », « Qwen3-14B »).
- code : « released » si un dépôt ou une mention explicite indique que le code est disponible, « promised » s'il est annoncé pour plus tard, sinon « not_mentioned ». code_url : l'adresse du dépôt si elle figure dans le texte, sinon chaîne vide.
- weights : même logique pour les poids du modèle."""

SCHEMA = {
    "type": "object",
    "properties": {
        "tldr_fr": {"type": "string"},
        "contribution": {"type": "string", "enum": [
            "method", "model", "benchmark_or_dataset", "system_or_tool", "empirical_study",
            "survey", "theory", "application", "position"]},
        "topics": {"type": "array", "items": {"type": "string"}},
        "modalities": {"type": "array", "items": {"type": "string", "enum": [
            "text", "image", "video", "audio", "code", "robotics", "tabular", "graph",
            "time_series", "3d", "science", "other"]}},
        "claims": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "metric": {"type": "string"}, "value": {"type": "string"},
                "baseline": {"type": "string"}, "benchmark": {"type": "string"},
            },
            "required": ["metric", "value", "baseline", "benchmark"],
            "additionalProperties": False,
        }},
        "benchmarks": {"type": "array", "items": {"type": "string"}},
        "models": {"type": "array", "items": {"type": "string"}},
        "code": {"type": "string", "enum": ["released", "promised", "not_mentioned"]},
        "code_url": {"type": "string"},
        "weights": {"type": "string", "enum": ["released", "promised", "not_mentioned"]},
    },
    "required": ["tldr_fr", "contribution", "topics", "modalities", "claims", "benchmarks",
                 "models", "code", "code_url", "weights"],
    "additionalProperties": False,
}


def log(*a):
    print(*a, flush=True)


def fetch_papers(n: int) -> list[dict]:
    url = "https://export.arxiv.org/api/query"
    params = {"search_query": QUERY, "sortBy": "submittedDate", "sortOrder": "descending", "max_results": n}
    for attempt in range(5):
        r = httpx.get(url, params=params, timeout=60, headers={"User-Agent": "zimb-news/1.0 (pilote d'annotation)"})
        if r.status_code == 200 and "<entry>" in r.text:
            break
        log(f"arXiv {r.status_code}, nouvel essai dans {10 * (attempt + 1)} s")
        time.sleep(10 * (attempt + 1))
    else:
        raise RuntimeError("arXiv indisponible")
    papers = []
    for e in ET.fromstring(r.text).findall("a:entry", NS):
        txt = lambda tag: re.sub(r"\s+", " ", (e.findtext(tag, default="", namespaces=NS) or "")).strip()
        papers.append({
            "id": txt("a:id").rsplit("/", 1)[-1],
            "title": txt("a:title"), "abstract": txt("a:summary"), "comment": txt("x:comment"),
            "category": (e.find("x:primary_category", NS).get("term") if e.find("x:primary_category", NS) is not None else ""),
        })
    return papers


def paper_text(p: dict) -> str:
    return (f"Titre : {p['title']}\nCatégorie : {p['category']}\nRésumé : {p['abstract']}"
            + (f"\nCommentaire des auteurs : {p['comment']}" if p["comment"] else ""))


def annotate(client: anthropic.Anthropic, model: str, p: dict) -> dict:
    t = time.time()
    try:
        # Appel direct pour le pilote ; le traitement complet passera par les lots (mêmes paramètres)
        resp = client.messages.create(
            model=model, max_tokens=4000, system=SYSTEM,
            messages=[{"role": "user", "content": paper_text(p)}],
            output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}},
        )
    except anthropic.RateLimitError as e:
        return {"error": f"limite de débit : {e.message}"}
    except anthropic.APIStatusError as e:
        return {"error": f"{e.status_code} : {e.message}"}
    except anthropic.APIConnectionError as e:
        return {"error": f"connexion : {e}"}
    if resp.stop_reason in ("refusal", "max_tokens"):
        return {"error": f"arrêt {resp.stop_reason}"}
    text = next((b.text for b in resp.content if b.type == "text"), "")
    return {
        "data": json.loads(text), "seconds": round(time.time() - t, 1),
        "input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens,
        "json_chars": len(text),
    }


def numbers(s: str) -> list[str]:
    return re.findall(r"\d+(?:[.,]\d+)?", s)


def main() -> int:
    papers = fetch_papers(N)
    log(f"{len(papers)} articles arXiv, résumé moyen {sum(len(p['abstract']) for p in papers) // len(papers)} caractères\n")
    client = anthropic.Anthropic(max_retries=5)
    results: dict[str, list[dict]] = {}
    for model in MODELS:
        with ThreadPoolExecutor(max_workers=4) as pool:
            results[model] = list(pool.map(lambda p: annotate(client, model, p), papers))
        ok = [r for r in results[model] if "data" in r]
        errors = [r["error"] for r in results[model] if "error" in r]
        if not ok:
            log(f"{model} : aucune réponse ({errors[:2]})")
            continue
        tin = sum(r["input_tokens"] for r in ok) / len(ok)
        tout = sum(r["output_tokens"] for r in ok) / len(ok)
        json_tok = sum(r["json_chars"] for r in ok) / len(ok) / 4
        pin, pout = PRICES.get(model, (4.0, 20.0))
        per_paper = (tin * pin + tout * pout) / 2 / 1e6        # tarif des lots : moitié prix
        # Les chiffres extraits figurent-ils tels quels dans le texte de l'article ?
        claims = [(c, p) for r, p in zip(results[model], papers) if "data" in r for c in r["data"]["claims"]]
        grounded = sum(1 for c, p in claims
                       if numbers(c["value"]) and all(x in paper_text(p) for x in numbers(c["value"])))
        log(f"=== {model} : {len(ok)}/{len(papers)} fiches, {len(errors)} erreur(s)")
        log(f"  tokens par article : {tin:.0f} en entrée, {tout:.0f} en sortie (dont ~{json_tok:.0f} de JSON, le reste en réflexion)")
        log(f"  temps moyen : {sum(r['seconds'] for r in ok) / len(ok):.1f} s")
        log(f"  coût en lot : {per_paper * 1000:.2f} $ pour 1 000 articles, {per_paper * TOTAL_2026:.0f} $ pour les {TOTAL_2026:,} articles de 2026")
        log(f"  chiffres : {len(claims)} extraits ({len(claims) / len(ok):.1f} par article), "
            f"{grounded}/{len(claims)} retrouvés tels quels dans le texte")
        log(f"  code publié : {sum(r['data']['code'] == 'released' for r in ok)} · "
            f"poids publiés : {sum(r['data']['weights'] == 'released' for r in ok)}\n")

    # Accord avec le modèle le plus capable de la liste (le dernier)
    ref = MODELS[-1]
    for model in MODELS[:-1]:
        pairs = [(a["data"], b["data"]) for a, b in zip(results.get(model, []), results.get(ref, []))
                 if "data" in a and "data" in b]
        if not pairs:
            continue
        agree = lambda f: sum(a[f] == b[f] for a, b in pairs)
        topic_overlap = sum(bool(set(a["topics"]) & set(b["topics"])) for a, b in pairs)
        log(f"Accord {model} / {ref} sur {len(pairs)} articles : contribution {agree('contribution')}, "
            f"code {agree('code')}, poids {agree('weights')}, au moins un topic commun {topic_overlap}")

    log("\n--- Trois exemples ---")
    for i in range(min(3, len(papers))):
        log(f"\n[{papers[i]['id']}] {papers[i]['title']}")
        for model in MODELS:
            r = results.get(model, [{}] * len(papers))[i]
            log(f"  {model} : {json.dumps(r.get('data', r.get('error')), ensure_ascii=False)}")

    with open("annotate_pilot.json", "w", encoding="utf-8") as f:
        json.dump({"papers": papers, "results": results}, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
