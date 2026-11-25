# Zimb News

Les histoires dont la presse francophone parle le plus ces 15 derniÃ¨res minutes, regroupÃ©es et rÃ©sumÃ©es par Claude.
Site : https://news.zimb.app

## Fonctionnement

```
GitHub Actions (toutes les 15 min)                Supabase (Postgres)               news.zimb.app
  worker/worker.py                          â”€â”€â–º   threads / thread_articles   â—„â”€â”€   site/index.html
  1. derniers fichiers GDELT (Hugging Face)       articles / seen_urls / runs       (GitHub Pages)
  2. articles en franÃ§ais jamais vus              rpc trending(hours, lim)
  3. regroupement des reprises syndiquÃ©es         lecture publique seule (RLS)
  4. Claude rattache aux histoires suivies
     ou crÃ©e de nouvelles histoires
```

- Source : [openalphalab/gdelt-news](https://huggingface.co/datasets/openalphalab/gdelt-news), texte reconstituÃ© depuis GDELT Web News NGrams 3.0.
- **Histoires persistantes** : chaque passage envoie Ã  Claude les nouveaux articles et les histoires actives des 6 derniÃ¨res heures. Il rattache les articles aux histoires existantes (et met leur rÃ©sumÃ© Ã  jour si un fait nouveau apparaÃ®t) ou crÃ©e de nouvelles histoires.
- **Pas de double traitement** : les adresses dÃ©jÃ  vues (`seen_urls`) ne sont jamais renvoyÃ©es Ã  Claude.
- **Classement** (`trending`) : somme, sur les mÃ©dias ayant Ã©crit leur propre article sur l'histoire ces 3 derniÃ¨res heures, d'un poids qui va de 1 (article de moins de 30 min) Ã  0,25 (fin de fenÃªtre). Les reprises syndiquÃ©es sont affichÃ©es mais ne comptent pas dans le score.
- Seuls de courts extraits (320 caractÃ¨res) sont stockÃ©s, et ils ne sont pas exposÃ©s publiquement.
- **Rien n'est Ã©crasÃ©** : chaque titre et rÃ©sumÃ© est conservÃ© dans `thread_versions` (trigger sur `threads`). Le site affiche l'historique des versions.
- **Site** : Â« En ce moment Â» (rpc `top_stories`, 10 histoires) puis le fil complet de toutes les histoires (rpc `feed`, pagination par curseur, dÃ©filement infini). Le rafraÃ®chissement automatique ne touche pas au fil : un bouton signale les histoires mises Ã  jour.
- Un seul appel Ã  Claude par passage : tous les nouveaux articles de la fenÃªtre (jusqu'Ã  400) et les histoires actives sont envoyÃ©s ensemble.
- Les tables `stories` / `story_articles`, la vue `latest_stories` et la fonction `trending` (utilisÃ©e par `top_stories`) datent des premiÃ¨res versions ; les deux premiÃ¨res ne sont plus alimentÃ©es.

## API de veille tech & science (`supabase/functions/news-api`)

Fonction Edge Supabase, protÃ©gÃ©e par clÃ© (`x-api-key`), page web : `site/sujets.html`.

```
Base : https://rtygalgkixjttiezdcym.supabase.co/functions/v1/news-api

GET  /topics                                 liste des sujets
POST /topics            {name, description?} crÃ©e un sujet (Claude gÃ©nÃ¨re les requÃªtes par source)
GET  /topics/:slug/digest?days=1|2|3         synthÃ¨se (cache 2 h ; &refresh=1 pour forcer)
POST /topics/:slug/ask  {question, days?}    question libre, rÃ©ponse sourcÃ©e [S1]â€¦
GET  /topics/:slug/sources?days=1|2|3        sources brutes, sans IA (gratuit)
GET  /usage                                  consommation de la clÃ© sur 24 h
```

- Sources : arXiv (toutes les sciences), Hugging Face Daily Papers (IA, votes de la communautÃ©), Hacker News (recherche stricte dans les titres).
- ClÃ©s : seule l'empreinte SHA-256 est stockÃ©e (`api_keys`) ; quota d'appels Ã  l'IA par 24 h glissantes (`daily_ai_limit`, 60 par dÃ©faut).
- Secrets de la fonction (Supabase â†’ Edge Functions â†’ Secrets) : `ANTHROPIC_API_KEY` (obligatoire), `CLAUDE_MODEL` (dÃ©faut `claude-sonnet-5-5`), `CLAUDE_EFFORT` (dÃ©faut `low`).
- CrÃ©er une clÃ© : gÃ©nÃ©rer une chaÃ®ne alÃ©atoire `zn_â€¦`, puis `insert into api_keys (name, key_hash) values ('nom', '<sha256 hex de la clÃ©>')`.

```bash
curl -H "x-api-key: $ZN_KEY" ".../news-api/topics/informatique-quantique/digest?days=3"
```

## Configuration (Settings â†’ Secrets and variables â†’ Actions)

| Nom | Type | RÃ´le |
|---|---|---|
| `SUPABASE_URL` | variable | `https://rtygalgkixjttiezdcym.supabase.co` |
| `SUPABASE_SECRET_KEY` | secret | ClÃ© secrÃ¨te Supabase (`sb_secret_â€¦`), utilisÃ©e uniquement pour Ã©crire |
| `ANTHROPIC_API_KEY` | secret | ClÃ© API Claude |
| `HF_TOKEN` | secret, facultatif | Token Hugging Face (lecture), Ã©vite les limites de dÃ©bit |
| `CLAUDE_MODEL` | variable, facultatif | DÃ©faut `claude-opus-5-5` |
| `CLAUDE_EFFORT` | variable, facultatif | DÃ©faut `medium` (`low` pour rÃ©duire le coÃ»t) |
| `WINDOW_MINUTES` | variable, facultatif | DÃ©faut `15` |
| `ACTIVE_HOURS` | variable, facultatif | DÃ©faut `6` : anciennetÃ© maximale des histoires proposÃ©es au rattachement |

## Tester en local

```bash
pip install -r worker/requirements.txt
DRY_RUN=1 python worker/worker.py      # lit GDELT et les histoires actives, affiche le prompt, sans appeler Claude ni Ã©crire en base
                                       # (nÃ©cessite SUPABASE_URL et SUPABASE_SECRET_KEY)
```
