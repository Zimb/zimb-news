# Zimb News

Les histoires dont la presse francophone parle le plus ces 15 dernières minutes, regroupées et résumées par Claude.
Site : https://news.zimb.app

## Fonctionnement

```
GitHub Actions (toutes les 15 min)            Supabase (Postgres)            news.zimb.app
  worker/worker.py                     ──►    runs / articles / stories ◄──  site/index.html
  1. derniers fichiers GDELT (Hugging Face)   vue latest_stories             (GitHub Pages)
  2. articles en français, sans doublons      lecture publique seule (RLS)
  3. Claude regroupe par histoire et résume
```

- Source : [openalphalab/gdelt-news](https://huggingface.co/datasets/openalphalab/gdelt-news), texte reconstitué depuis GDELT Web News NGrams 3.0.
- Classement : nombre d'articles **distincts** par histoire (les reprises d'un même article par plusieurs titres d'un groupe de presse ne comptent qu'une fois).
- Seuls de courts extraits (320 caractères) sont stockés, et ils ne sont pas exposés publiquement.

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

## Tester en local

```bash
pip install -r worker/requirements.txt
DRY_RUN=1 python worker/worker.py      # lit GDELT et affiche le prompt, sans appeler Claude ni écrire en base
```
