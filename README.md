# Zimb News

Les histoires dont la presse francophone parle le plus ces 15 dernières minutes, regroupées et résumées par Claude.
Site : https://news.zimb.app

## Fonctionnement

```
GitHub Actions (toutes les 15 min)                Supabase (Postgres)               news.zimb.app
  worker/worker.py                          ──►   threads / thread_articles   ◄──   site/index.html
  1. derniers fichiers GDELT (Hugging Face)       articles / seen_urls / runs       (GitHub Pages)
  2. articles en français jamais vus              rpc trending(hours, lim)
  3. regroupement des reprises syndiquées         lecture publique seule (RLS)
  4. Claude rattache aux histoires suivies
     ou crée de nouvelles histoires
```

- Source : [openalphalab/gdelt-news](https://huggingface.co/datasets/openalphalab/gdelt-news), texte reconstitué depuis GDELT Web News NGrams 3.0.
- **Histoires persistantes** : chaque passage envoie à Claude les nouveaux articles et les histoires actives des 6 dernières heures. Il rattache les articles aux histoires existantes (et met leur résumé à jour si un fait nouveau apparaît) ou crée de nouvelles histoires.
- **Pas de double traitement** : les adresses déjà vues (`seen_urls`) ne sont jamais renvoyées à Claude.
- **Classement** (`trending`) : somme, sur les médias ayant écrit leur propre article sur l'histoire ces 3 dernières heures, d'un poids qui va de 1 (article de moins de 30 min) à 0,25 (fin de fenêtre). Les reprises syndiquées sont affichées mais ne comptent pas dans le score.
- Seuls de courts extraits (320 caractères) sont stockés, et ils ne sont pas exposés publiquement.
- **Rien n'est écrasé** : chaque titre et résumé est conservé dans `thread_versions` (trigger sur `threads`). Le site affiche l'historique des versions.
- **Site** : « En ce moment » (rpc `top_stories`, 10 histoires) puis le fil complet de toutes les histoires (rpc `feed`, pagination par curseur, défilement infini). Le rafraîchissement automatique ne touche pas au fil : un bouton signale les histoires mises à jour.
- **Rubriques et sujets** : Claude attribue à chaque histoire une catégorie et 1 à 3 sujets de fond (`threads.tags`, par exemple « droits de douane »), en réutilisant à l'identique les sujets déjà employés dans la semaine, qui lui sont rappelés. « Sujets de la semaine » (rpc `week_topics`) liste les sujets portés par au moins deux histoires sur 7 jours. `top_stories` et `feed` acceptent les filtres facultatifs `cat` et `tag`, repris dans l'adresse du site (`?cat=économie`, `?tag=droits de douane`).
- Migrations SQL récentes : `supabase/migrations/`.
- Un seul appel à Claude par passage : tous les nouveaux articles de la fenêtre (jusqu'à 400) et les histoires actives sont envoyés ensemble.
- Les tables `stories` / `story_articles`, la vue `latest_stories` et la fonction `trending` (utilisée par `top_stories`) datent des premières versions ; les deux premières ne sont plus alimentées.

## API de veille tech & science (`supabase/functions/news-api`)

Fonction Edge Supabase, protégée par clé (`x-api-key`), page web : `site/sujets.html`.

```
Base : https://rtygalgkixjttiezdcym.supabase.co/functions/v1/news-api

GET  /topics                                 liste des sujets
POST /topics            {name, description?} crée un sujet (Claude génère les requêtes par source)
GET  /topics/:slug/digest?days=1|2|3         synthèse (cache 2 h ; &refresh=1 pour forcer)
POST /topics/:slug/ask  {question, days?}    question libre, réponse sourcée [S1]…
GET  /topics/:slug/sources?days=1|2|3        sources brutes, sans IA (gratuit)
GET  /usage                                  consommation de la clé sur 24 h
```

- Sources : versions GitHub des dépôts du sujet (flux Atom), modèles tendance Hugging Face, blogs de labos (OpenAI, DeepMind, Google Research, Hugging Face, NVIDIA, BAIR), Hugging Face Daily Papers, arXiv (résumé complet), Hacker News (recherche stricte dans les titres).
- Signaux transmis à l'IA et affichés : code publié (★), votes Hugging Face, poids publiés (modèle HF relié à l'article arXiv), version publiée. La synthèse sépare « sorties concrètes » et « avancées de recherche ».
- Clés : seule l'empreinte SHA-256 est stockée (`api_keys`) ; quota d'appels à l'IA par 24 h glissantes (`daily_ai_limit`, 60 par défaut).
- Secrets de la fonction (Supabase → Edge Functions → Secrets) : `ANTHROPIC_API_KEY` (obligatoire), `CLAUDE_MODEL` (défaut `claude-sonnet-5-5`), `CLAUDE_EFFORT` (défaut `low`).
- Créer une clé : générer une chaîne aléatoire `zn_…`, puis `insert into api_keys (name, key_hash) values ('nom', '<sha256 hex de la clé>')`.

```bash
curl -H "x-api-key: $ZN_KEY" ".../news-api/topics/informatique-quantique/digest?days=3"
```

## Configuration (Settings → Secrets and variables → Actions)

| Nom | Type | Rôle |
|---|---|---|
| `SUPABASE_URL` | variable | `https://rtygalgkixjttiezdcym.supabase.co` |
| `SUPABASE_SECRET_KEY` | secret | Clé secrète Supabase (`sb_secret_…`), utilisée uniquement pour écrire |
| `ANTHROPIC_API_KEY` | secret | Clé API Claude |
| `HF_TOKEN` | secret, facultatif | Token Hugging Face (lecture), évite les limites de débit |
| `CLAUDE_MODEL` | variable, facultatif | Défaut `claude-opus-5-5` |
| `CLAUDE_EFFORT` | variable, facultatif | Défaut `medium` (`low` pour réduire le coût) |
| `WINDOW_MINUTES` | variable, facultatif | Défaut `15` |
| `ACTIVE_HOURS` | variable, facultatif | Défaut `6` : ancienneté maximale des histoires proposées au rattachement |

## Tester en local

```bash
pip install -r worker/requirements.txt
DRY_RUN=1 python worker/worker.py      # lit GDELT et les histoires actives, affiche le prompt, sans appeler Claude ni écrire en base
                                       # (nécessite SUPABASE_URL et SUPABASE_SECRET_KEY)
```
