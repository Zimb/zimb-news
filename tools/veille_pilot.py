"""Pilote de veille : les articles de recherche prometteurs d'une période, pour un domaine donné.

Entonnoir (identique pour tous les profils) :
  1. Collecte (OpenAlex) selon les filtres du profil.
  2. Filtre sans IA : règles simples sur les métadonnées.
  3. Fiche de tri par IA (résumé seul) : thème, type, résultat chiffré, limites, et une grille de
     quatre notes de 0 à 3 propre au profil.
  4. Lecture du texte complet des meilleurs (copie OpenAlex) et rédaction de la fiche publiable.

Profils : « planete » (agriculture, énergie, eau, climat…) et « dev-ia » (développement logiciel assisté par IA).

Variables : OPENALEX_API_KEY, ANTHROPIC_API_KEY ; PROFILE ; DATE_FROM et DATE_TO (AAAA-MM-JJ, DATE_TO
facultative) ; SCREEN_MODEL (défaut claude-haiku-5-5) ; WRITE_MODEL (défaut claude-opus-5-5) ;
TOP_FULLTEXT (défaut 10) ; DRY_RUN=1 (ni IA ni téléchargement de texte).
Écrit veille_<profil>.json et veille_<profil>.md (sans les résumés des articles).
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import anthropic
import httpx

OA = "https://api.openalex.org"
SELECT = ("id,doi,title,type,publication_date,primary_location,primary_topic,sustainable_development_goals,"
          "authorships,institutions_distinct_count,countries_distinct_count,open_access,best_oa_location,"
          "has_content,content_urls,referenced_works_count,funders,abstract_inverted_index")
PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.0, 10.0), "claude-opus-5-5": (4.0, 20.0)}

_default_day = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=3)).strftime("%Y-%m-%d")
PROFILE_NAME = os.environ.get("PROFILE") or "planete"
DATE_FROM = os.environ.get("DATE_FROM") or _default_day
DATE_TO = os.environ.get("DATE_TO") or DATE_FROM
SCREEN_MODEL = os.environ.get("SCREEN_MODEL") or "claude-haiku-5-5"
WRITE_MODEL = os.environ.get("WRITE_MODEL") or "claude-opus-5-5"
TOP_FULLTEXT = int(os.environ.get("TOP_FULLTEXT") or 10)
MAX_ARTICLES = int(os.environ.get("MAX_ARTICLES") or 2000)
PROMISING = int(os.environ.get("PROMISING_SCORE") or 8)     # seuil sur 12
DRY_RUN = os.environ.get("DRY_RUN") == "1"


# --------------------------------------------------------------------------- Profils

PLANET_SDGS = {2: "Faim zéro", 6: "Eau", 7: "Énergie propre", 12: "Consommation responsable",
               13: "Climat", 14: "Vie aquatique", 15: "Vie terrestre"}
PLANET_FIELDS = {
    "Agricultural and Biological Sciences", "Environmental Science", "Energy", "Engineering",
    "Materials Science", "Earth and Planetary Sciences", "Chemical Engineering", "Chemistry", "Veterinary",
}


def planet_stage1(d: dict, abstract: str, w: dict) -> str | None:
    if d["domaine"] not in PLANET_FIELDS:
        return "domaine hors périmètre"
    if d["score_onu"] < 0.5:
        return "score ONU faible"
    if (w.get("institutions_distinct_count") or 0) < 1:
        return "institution inconnue"
    if len(abstract) < 400:
        return "résumé trop court"
    return None


def dev_stage1(d: dict, abstract: str, w: dict) -> str | None:
    if w.get("type") not in ("preprint", "article", "conference-paper", "review"):
        return "logiciel, jeu de données ou autre type"
    if any(x in d["revue"].lower() for x in ("zenodo", "open science framework", "figshare", "ssrn", "researchgate")):
        return "dépôt libre sans relecture (Zenodo, OSF…)"
    if len(abstract) < 400:
        return "résumé trop court"
    return None


PROFILES = {
    "planete": {
        "titre": "Avancées pour la planète",
        "filters": ["type:article", "has_abstract:true", "is_retracted:false", "language:en",
                    "primary_location.source.is_core:true",
                    "sustainable_development_goals.id:" + "|".join(f"https://metadata.un.org/sdg/{i}" for i in PLANET_SDGS)],
        "search": "",
        "stage1": planet_stage1,
        "sujet": "une veille des avancées utiles à la planète (agriculture et alimentation durables, énergie, eau, climat, biodiversité, matériaux et recyclage)",
        "hors_sujet": "true si l'article n'apporte rien à ces sujets (médecine, gestion d'entreprise, sciences sociales sans lien concret, etc.)",
        "themes": ["agriculture", "élevage et aquaculture", "alimentation", "énergie solaire", "stockage d'énergie",
                   "autres énergies", "eau", "climat et émissions", "biodiversité et écosystèmes",
                   "matériaux et chimie verte", "déchets et recyclage", "autre"],
        "stades": ["recherche fondamentale", "laboratoire", "prototype ou pilote", "essai en conditions réelles",
                   "déploiement", "modélisation", "revue de littérature"],
        "stade_label": "stade",
        "echelle": "durée, nombre de sites, taille d'échantillon, si indiqués (« 2 saisons, 3 sites »)",
        "criteres": {
            "preuve": "0 = opinion, revue narrative, aucune donnée ; 1 = simulation ou modèle seul, ou expérience sans comparaison claire ; 2 = expérience contrôlée avec comparateur (laboratoire, serre, prototype) ou analyse de données réelles ; 3 = essai en conditions réelles sur plusieurs sites, saisons ou années, ou méta-analyse.",
            "effet": "0 = aucun chiffre ; 1 = chiffre sans point de comparaison ; 2 = gain chiffré par rapport à une référence nommée ; 3 = gain important et chiffré sur le critère qui compte (rendement, émissions, coût, efficacité) par rapport à la pratique actuelle.",
            "proximite": "0 = fondamental ; 1 = preuve de concept en laboratoire ; 2 = prototype, pilote ou essai de terrain ; 3 = utilisable avec les équipements ou pratiques existants.",
            "portee": "0 = anecdotique ou très local ; 1 = niche ; 2 = une filière ou une région entière ; 3 = problème majeur à grande échelle (culture de base, source d'énergie importante, grosses émissions).",
        },
        "extras_doc": "",
        "extras_schema": {},
        "lecteur": "des professionnels et des lecteurs curieux",
        "consigne_redaction": "Dis clairement où en est le travail et ce qui manque avant un usage réel.",
    },
    "dev-ia": {
        "titre": "Développement logiciel assisté par IA",
        "filters": ["has_abstract:true", "is_retracted:false", "language:en"],
        "search": ('("large language model" OR "large language models" OR LLM OR LLMs OR "generative AI" OR "AI assistant" '
                   'OR "AI agents" OR "coding agent" OR "coding agents" OR Copilot) AND ("code generation" OR '
                   '"software engineering" OR "software development" OR "program repair" OR "code review" OR '
                   '"test generation" OR "coding agent" OR "coding agents" OR "coding assistant" OR "SWE-bench" OR '
                   '"programming assistant" OR "pull request" OR "developer productivity" OR "vibe coding")'),
        "stage1": dev_stage1,
        "sujet": "une veille sur le développement logiciel assisté par IA (assistants et agents de programmation, génération, réparation, test et revue de code, pratiques et productivité des développeurs)",
        "hors_sujet": "true si l'article ne porte pas sur l'usage de l'IA pour écrire, tester, corriger, relire ou maintenir du logiciel (par exemple un LLM appliqué à un autre métier, ou l'enseignement général)",
        "themes": ["agents de programmation", "génération de code", "réparation et débogage", "tests",
                   "revue de code", "compréhension et recherche de code", "sécurité du code généré",
                   "pratiques et productivité des développeurs", "évaluation et benchmarks",
                   "formation à la programmation", "autre"],
        "stades": ["outil ou agent", "méthode", "benchmark ou jeu de données", "étude d'usage réel",
                   "étude avec des développeurs", "analyse de sécurité ou de fiabilité", "revue ou prise de position"],
        "stade_label": "type de contribution",
        "echelle": "benchmarks, nombre de tâches, de dépôts ou de participants, si indiqués (« SWE-bench Verified, 500 tâches »)",
        "criteres": {
            "preuve": "0 = vision ou prise de position sans donnée ; 1 = démonstration ou évaluation sur quelques exemples ou tâches jouets ; 2 = évaluation sur un benchmark reconnu avec comparaison à d'autres approches, ou étude contrôlée avec des développeurs ; 3 = preuve à grande échelle en conditions réelles (données de déploiement industriel, nombreux dépôts ou pull requests réels, plusieurs benchmarks avec ablations, large étude utilisateurs).",
            "effet": "0 = aucun chiffre ; 1 = chiffre sans point de comparaison ; 2 = gain chiffré par rapport à une approche nommée ; 3 = écart important sur un critère qui compte (taux de résolution, temps gagné, défauts, coût) par rapport à une référence forte et actuelle. Un résultat négatif solide et chiffré compte autant qu'un gain.",
            "utilisable": "0 = idée ; 1 = méthode décrite sans code ni données ; 2 = code, données ou benchmark publiés, ou protocole reproductible ; 3 = outil utilisable aujourd'hui, intégré à un IDE ou une chaîne CI, ou déployé en production.",
            "nouveaute": "0 = réplication ou variante mineure ; 1 = amélioration incrémentale d'une approche connue ; 2 = méthode, benchmark ou constat nouveau sur une question ouverte ; 3 = résultat qui contredit une idée reçue ou ouvre une capacité nouvelle.",
        },
        "extras_doc": ("- modeles : les modèles d'IA nommés, évalués ou utilisés (« GPT-5 », « Claude Sonnet 4.5 », « Qwen3-Coder »).\n"
                       "- code_publie : « oui » si un dépôt ou une mention explicite indique que le code, l'outil ou les données sont disponibles, « annoncé » s'ils sont promis, sinon « non mentionné »."),
        "extras_schema": {
            "modeles": {"type": "array", "items": {"type": "string"}},
            "code_publie": {"type": "string", "enum": ["oui", "annoncé", "non mentionné"]},
        },
        "lecteur": "des développeurs et des responsables techniques",
        "consigne_redaction": "Précise les modèles et les benchmarks utilisés, si le code ou l'outil est disponible, et ce qu'un développeur peut en tirer aujourd'hui.",
    },
}

if PROFILE_NAME not in PROFILES:
    sys.exit(f"Profil inconnu : {PROFILE_NAME} (choix : {', '.join(PROFILES)})")
P = PROFILES[PROFILE_NAME]
CRIT = list(P["criteres"])

SCREEN_SYSTEM = f"""Tu tries des articles de recherche pour {P['sujet']}. Tu ne disposes que du titre, de la source et du résumé. Tu remplis une fiche factuelle, sans rien inventer : si une information n'est pas dans le texte, laisse la chaîne vide.

Champs :
- hors_sujet : {P['hors_sujet']}.
- theme : parmi {", ".join(P['themes'])}.
- stade ({P['stade_label']}) : parmi {", ".join(P['stades'])}.
- tldr_fr : une phrase en français, 30 mots maximum, qui dit ce que l'étude apporte. Termes techniques usuels en anglais.
- resultat : le principal résultat chiffré. mesure = ce qui est mesuré ; valeur = le chiffre tel qu'il est écrit ; comparaison = le point de référence s'il est donné.
- echelle : {P['echelle']}.
- limites : la principale limite, énoncée par les auteurs ou évidente (15 mots maximum).
{P['extras_doc']}

Grille, quatre notes entières de 0 à 3. Note ce que le résumé établit, pas ce qu'il promet. Les adjectifs (« novel », « promising », « significant ») ne valent rien sans donnée.
""" + "\n".join(f"- {k} : {v}" for k, v in P["criteres"].items())

SCREEN_SCHEMA = {
    "type": "object",
    "properties": {
        "hors_sujet": {"type": "boolean"},
        "theme": {"type": "string", "enum": P["themes"]},
        "stade": {"type": "string", "enum": P["stades"]},
        "tldr_fr": {"type": "string"},
        "resultat": {
            "type": "object",
            "properties": {"mesure": {"type": "string"}, "valeur": {"type": "string"}, "comparaison": {"type": "string"}},
            "required": ["mesure", "valeur", "comparaison"], "additionalProperties": False,
        },
        "echelle": {"type": "string"},
        "limites": {"type": "string"},
        **P["extras_schema"],
        **{k: {"type": "integer"} for k in CRIT},
    },
    "required": ["hors_sujet", "theme", "stade", "tldr_fr", "resultat", "echelle", "limites",
                 *P["extras_schema"], *CRIT],
    "additionalProperties": False,
}

WRITE_SYSTEM = f"""Tu rédiges en français une fiche de veille sur un article de recherche, pour {P['lecteur']}. Tu disposes des métadonnées et, selon les cas, du texte complet ou seulement du résumé.

Règles :
- Appuie-toi uniquement sur le texte fourni. N'invente aucun chiffre, aucune institution, aucune conclusion.
- Présente les résultats comme ceux des auteurs. Aucune formule promotionnelle : pas de « révolutionnaire », « prometteur », « majeur ».
- Donne les chiffres avec leur point de comparaison et leurs conditions.
- {P['consigne_redaction']}
- Si le texte complet nuance ou contredit le résumé, dis-le.
- Garde les termes techniques en anglais quand c'est l'usage.

Champs :
- titre : titre factuel de 90 caractères maximum.
- en_bref : deux phrases, l'essentiel pour quelqu'un qui ne lira que ça.
- ce_que_montre : 3 à 4 phrases, les résultats avec leurs chiffres.
- methode : 1 à 2 phrases sur le protocole et son échelle.
- stade ({P['stade_label']}) : parmi {", ".join(P['stades'])}.
- pourquoi_important : 1 à 2 phrases, sans exagération.
- limites : 1 à 2 phrases, celles que les auteurs reconnaissent et celles que le protocole impose.
- chiffres_confirmes : true si les chiffres clés du résumé se retrouvent dans le texte complet ; false si le texte complet n'est pas fourni ou s'ils n'y sont pas.
- note_verification : une phrase sur cette vérification.
- financement : financeurs et liens d'intérêts déclarés (y compris l'appartenance des auteurs à l'entreprise dont l'outil est évalué), ou chaîne vide.
- confiance : « élevée », « moyenne » ou « faible », selon la solidité de la preuve."""

WRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "titre": {"type": "string"}, "en_bref": {"type": "string"}, "ce_que_montre": {"type": "string"},
        "methode": {"type": "string"}, "stade": {"type": "string", "enum": P["stades"]},
        "pourquoi_important": {"type": "string"}, "limites": {"type": "string"},
        "chiffres_confirmes": {"type": "boolean"}, "note_verification": {"type": "string"},
        "financement": {"type": "string"}, "confiance": {"type": "string", "enum": ["élevée", "moyenne", "faible"]},
    },
    "required": ["titre", "en_bref", "ce_que_montre", "methode", "stade", "pourquoi_important", "limites",
                 "chiffres_confirmes", "note_verification", "financement", "confiance"],
    "additionalProperties": False,
}


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------- OpenAlex

def collect(http: httpx.Client) -> list[dict]:
    filters = [f"from_publication_date:{DATE_FROM}", f"to_publication_date:{DATE_TO}", *P["filters"]]
    if P["search"]:
        filters.append(f"title_and_abstract.search:{P['search']}")
    cursor, out = "*", []
    while cursor and len(out) < MAX_ARTICLES:
        r = http.get(f"{OA}/works", params={"filter": ",".join(filters), "select": SELECT, "per_page": 100, "cursor": cursor})
        r.raise_for_status()
        d = r.json()
        if not d["results"]:
            break
        out += d["results"]
        cursor = d["meta"].get("next_cursor")
    return out


def abstract_of(w: dict) -> str:
    idx = w.get("abstract_inverted_index") or {}
    return " ".join(word for _, word in sorted((p, word) for word, ps in idx.items() for p in ps))


def describe(w: dict) -> dict:
    """Fiche d'identité compacte d'un article (sans le résumé)."""
    inst, countries = [], []
    for a in w.get("authorships") or []:
        for i in a.get("institutions") or []:
            if i.get("display_name") and i["display_name"] not in inst:
                inst.append(i["display_name"])
        for c in a.get("countries") or []:
            if c not in countries:
                countries.append(c)
    sdgs = {int(s["id"].rsplit("/", 1)[-1]): s.get("score") or 0
            for s in w.get("sustainable_development_goals") or [] if s.get("id")}
    planet = {PLANET_SDGS[k]: round(v, 2) for k, v in sdgs.items() if k in PLANET_SDGS}
    topic = w.get("primary_topic") or {}
    src = (w.get("primary_location") or {}).get("source") or {}
    return {
        "id": w["id"].rsplit("/", 1)[-1], "doi": w.get("doi"), "titre_original": w.get("title") or "",
        "type": w.get("type") or "", "date": w.get("publication_date"), "revue": src.get("display_name") or "",
        "domaine": (topic.get("field") or {}).get("display_name") or "",
        "sujet": topic.get("display_name") or "",
        "objectifs": planet, "score_onu": max(planet.values(), default=0),
        "institutions": inst[:4], "pays": countries[:6],
        "acces_ouvert": bool((w.get("open_access") or {}).get("is_oa")),
        "texte_complet_disponible": bool((w.get("has_content") or {}).get("grobid_xml")),
        "signaux": {
            "financeur": bool(w.get("funders")),
            "international": (w.get("countries_distinct_count") or 0) > 1,
            "bibliographie_30": (w.get("referenced_works_count") or 0) >= 30,
        },
        "lien": w.get("doi") or w["id"],
    }


def tei_to_text(xml: str, limit: int = 60000) -> str:
    """Texte lisible d'un article au format TEI (GROBID) : sections, paragraphes, tableaux."""
    ns = {"t": "http://www.tei-c.org/ns/1.0"}
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return ""
    clean = lambda el: re.sub(r"\s+", " ", "".join(el.itertext())).strip()
    parts = []
    body = root.find(".//t:text/t:body", ns)
    for div in (body.findall("t:div", ns) if body is not None else []):
        head = div.find("t:head", ns)
        if head is not None:
            parts.append(f"\n## {clean(head)}")
        parts += [clean(p) for p in div.findall("t:p", ns)]
    for fig in root.findall(".//t:figure", ns):
        head = fig.find("t:head", ns)
        desc = fig.find("t:figDesc", ns)
        rows = [" | ".join(clean(c) for c in row.findall("t:cell", ns)) for row in fig.findall(".//t:row", ns)]
        label = " ".join(x for x in (clean(head) if head is not None else "", clean(desc) if desc is not None else "") if x)
        if label or rows:
            parts.append(f"\n[{label}]" + ("\n" + "\n".join(rows[:25]) if rows else ""))
    text = "\n".join(p for p in parts if p)
    text = re.sub(r"J o u r n a l P r e -?\s?p r o o f", "", text)   # filigrane des épreuves
    return text[:limit]


def fulltext(http: httpx.Client, w: dict) -> str:
    url = (w.get("content_urls") or {}).get("grobid_xml")
    if not url:
        return ""
    for _ in range(6):   # un compte récent est limité à 0,10 $ par tranche de 5 minutes
        r = http.get(url)
        if r.status_code == 429:
            time.sleep(65)
            continue
        return tei_to_text(r.text) if r.status_code == 200 else ""
    return ""


# --------------------------------------------------------------------------- Claude

def ask(client: anthropic.Anthropic, model: str, system: str, prompt: str, schema: dict,
        max_tokens: int, effort: str, fallback: bool):
    params = dict(model=model, max_tokens=max_tokens, system=system,
                  messages=[{"role": "user", "content": prompt}],
                  output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}})
    if fallback:
        try:   # repli automatique sur un autre modèle si le modèle principal refuse la requête
            resp = client.beta.messages.create(**params, betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        except anthropic.BadRequestError:
            resp = client.messages.create(**params)
    else:
        resp = client.messages.create(**params)
    if resp.stop_reason in ("refusal", "max_tokens"):
        raise RuntimeError(f"arrêt {resp.stop_reason}")
    text = next(b.text for b in resp.content if b.type == "text")
    return json.loads(text), resp.usage


def screen(client, d: dict, abstract: str):
    prompt = f"Titre : {d['titre_original']}\nSource : {d['revue']} ({d['type']})\nRésumé : {abstract}"
    try:
        data, usage = ask(client, SCREEN_MODEL, SCREEN_SYSTEM, prompt, SCREEN_SCHEMA, 3000, "low", fallback=False)
    except (anthropic.APIError, RuntimeError, json.JSONDecodeError, StopIteration) as e:
        return None, (0, 0), str(e)[:120]
    for k in CRIT:
        data[k] = max(0, min(3, int(data[k])))
    data["score"] = 0 if data["hors_sujet"] else sum(data[k] for k in CRIT)
    return data, (usage.input_tokens, usage.output_tokens), None


def write_fiche(client, d: dict, abstract: str, text: str):
    prompt = (f"Titre : {d['titre_original']}\nSource : {d['revue']} ({d['type']}, {d['date']})\n"
              f"Institutions : {', '.join(d['institutions']) or 'non renseignées'}\n"
              f"Résumé des auteurs : {abstract}\n\n"
              + (f"TEXTE COMPLET :\n{text}" if text else "Le texte complet n'est pas disponible : rédige d'après le résumé."))
    try:
        data, usage = ask(client, WRITE_MODEL, WRITE_SYSTEM, prompt, WRITE_SCHEMA, 6000, "medium", fallback=True)
    except (anthropic.APIError, RuntimeError, json.JSONDecodeError, StopIteration) as e:
        return None, (0, 0), str(e)[:120]
    data["source"] = "texte complet" if text else "résumé seul"
    return data, (usage.input_tokens, usage.output_tokens), None


def cost(model: str, tin: int, tout: int) -> float:
    pin, pout = PRICES.get(model, (4.0, 20.0))
    return (tin * pin + tout * pout) / 1e6


# --------------------------------------------------------------------------- Rapport

def period() -> str:
    return f"du {DATE_FROM}" if DATE_FROM == DATE_TO else f"du {DATE_FROM} au {DATE_TO}"


def report(stats: dict, promising: list[dict], fiches: list[dict]) -> str:
    notes = lambda t: ", ".join(f"{k} {t[k]}" for k in CRIT)
    L = [f"# {P['titre']} : articles {period()}", "",
         f"- Articles collectés : **{stats['collectes']}**",
         f"- Après le filtre sans IA : **{stats['apres_filtre']}** ({', '.join(f'{k} : {v}' for k, v in stats['rejets'].items()) or 'aucun rejet'})",
         f"- Jugés hors sujet par l'IA : **{stats['hors_sujet']}**",
         f"- Prometteurs (note d'au moins {PROMISING} sur 12) : **{len(promising)}**",
         f"- Coût : tri {stats['cout_tri']:.2f} $, rédaction {stats['cout_redaction']:.2f} $", ""]
    if fiches:
        L += ["## Les fiches rédigées", ""]
        for f in fiches:
            a, t, r = f["article"], f["tri"], f["fiche"]
            who = " · ".join(x for x in (a["revue"], ", ".join(a["institutions"][:2]), ", ".join(a["pays"][:3])) if x)
            L += [f"### {r['titre']}",
                  f"*{who} · note {t['score']}/12 ({notes(t)}) · {r['stade']} · confiance {r['confiance']} · "
                  f"rédigé d'après le {r['source']}*", "",
                  f"**En bref.** {r['en_bref']}", "", f"**Ce que montre l'étude.** {r['ce_que_montre']}", "",
                  f"**Méthode.** {r['methode']}", "", f"**Pourquoi c'est important.** {r['pourquoi_important']}", "",
                  f"**Limites.** {r['limites']}", "",
                  f"**Vérification.** {r['note_verification']}" + (f" Financement et liens d'intérêts : {r['financement']}" if r["financement"] else ""),
                  "", f"[Lire l'étude]({a['lien']})", ""]
    L += ["## Tous les articles prometteurs", "", "| Note | Thème | Type | En une phrase | Résultat | Source |",
          "|---|---|---|---|---|---|"]
    esc = lambda s: str(s).replace("|", "/").replace("\n", " ")
    for p in promising:
        t, a = p["tri"], p["article"]
        res = " ".join(x for x in (t["resultat"]["valeur"], f"({t['resultat']['comparaison']})" if t["resultat"]["comparaison"] else "") if x)
        code = " · code publié" if t.get("code_publie") == "oui" else ""
        L.append(f"| {t['score']} | {esc(t['theme'])} | {esc(t['stade'])} | [{esc(t['tldr_fr'])}]({a['lien']}){code} | {esc(res)} | {esc(a['revue'])} |")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- Main

def main() -> int:
    http = httpx.Client(timeout=90, headers={"Authorization": f"Bearer {os.environ['OPENALEX_API_KEY']}"})
    works = collect(http)
    log(f"[{PROFILE_NAME}] {period()} : {len(works)} travaux collectés")

    candidates, rejets = [], {}
    for w in works:
        d, abstract = describe(w), abstract_of(w)
        why = P["stage1"](d, abstract, w)
        if why:
            rejets[why] = rejets.get(why, 0) + 1
        else:
            candidates.append((w, d, abstract))
    log(f"Filtre sans IA : {len(candidates)} retenus, rejets {rejets}")
    stats = {"profil": PROFILE_NAME, "du": DATE_FROM, "au": DATE_TO, "collectes": len(works),
             "apres_filtre": len(candidates), "rejets": rejets, "hors_sujet": 0, "cout_tri": 0.0, "cout_redaction": 0.0}
    if DRY_RUN:
        with_text = sum(1 for _, d, _ in candidates if d["texte_complet_disponible"])
        log(f"DRY_RUN : {with_text} candidats ont un texte complet chez OpenAlex.")
        for _, d, _ in candidates[:8]:
            log(f"  - [{d['type']}] {d['titre_original'][:100]} ({d['revue']})")
        return 0

    client = anthropic.Anthropic(max_retries=8)
    with ThreadPoolExecutor(max_workers=6) as pool:
        screened = list(pool.map(lambda c: screen(client, c[1], c[2]), candidates))
    rows, tin, tout, errors = [], 0, 0, 0
    for (w, d, abstract), (tri, usage, err) in zip(candidates, screened):
        tin, tout = tin + usage[0], tout + usage[1]
        if tri is None:
            errors += 1
            continue
        rows.append({"article": d, "tri": tri, "_w": w, "_abstract": abstract})
    stats["cout_tri"] = cost(SCREEN_MODEL, tin, tout)
    stats["hors_sujet"] = sum(1 for r in rows if r["tri"]["hors_sujet"])
    stats["erreurs_tri"] = errors
    signals = lambda r: sum(r["article"]["signaux"].values())
    rows.sort(key=lambda r: (r["tri"]["score"], r["tri"][CRIT[0]], signals(r)), reverse=True)
    promising = [r for r in rows if r["tri"]["score"] >= PROMISING]
    log(f"Tri par IA : {len(rows)} fiches, {errors} erreur(s), {stats['hors_sujet']} hors sujet, "
        f"{len(promising)} prometteurs · {tin:,} tokens en entrée, {tout:,} en sortie · {stats['cout_tri']:.2f} $")
    dist = {}
    for r in rows:
        dist[r["tri"]["score"]] = dist.get(r["tri"]["score"], 0) + 1
    log("Répartition des notes :", dict(sorted(dist.items(), reverse=True)))

    # Rédaction : les meilleurs, en préférant ceux dont on peut lire le texte complet à note égale
    top = sorted(promising, key=lambda r: (r["tri"]["score"], r["article"]["texte_complet_disponible"],
                                           r["tri"][CRIT[0]], signals(r)), reverse=True)[:TOP_FULLTEXT]
    fiches, win, wout = [], 0, 0
    for r in top:
        text = fulltext(http, r["_w"])
        fiche, usage, err = write_fiche(client, r["article"], r["_abstract"], text)
        win, wout = win + usage[0], wout + usage[1]
        if fiche:
            fiches.append({"article": r["article"], "tri": r["tri"], "fiche": fiche})
            log(f"  fiche [{r['tri']['score']}/12, {fiche['source']}, confiance {fiche['confiance']}] {fiche['titre']}")
        else:
            log(f"  échec de rédaction pour {r['article']['id']} : {err}")
    stats["cout_redaction"] = cost(WRITE_MODEL, win, wout)
    log(f"Rédaction : {len(fiches)} fiches · {win:,} tokens en entrée, {wout:,} en sortie · {stats['cout_redaction']:.2f} $")

    public = lambda r: {"article": r["article"], "tri": r["tri"]}
    with open(f"veille_{PROFILE_NAME}.json", "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "criteres": CRIT, "fiches": fiches, "prometteurs": [public(r) for r in promising],
                   "autres": [public(r) for r in rows if r["tri"]["score"] < PROMISING]}, f, ensure_ascii=False, indent=1)
    md = report(stats, promising, fiches)
    with open(f"veille_{PROFILE_NAME}.md", "w", encoding="utf-8") as f:
        f.write(md)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
