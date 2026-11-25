// news-api : veille IA sur des sujets tech et science (1 à 3 derniers jours)
//
// Authentification : en-tête  x-api-key: zn_…   (empreinte SHA-256 comparée à api_keys)
//
//   GET  /topics                         liste des sujets
//   POST /topics          {name, description?}    crée un sujet (Claude prépare les requêtes)
//   GET  /topics/:slug/digest?days=1|2|3          synthèse (cache 2 h, sinon régénérée)
//   POST /topics/:slug/ask {question, days?}      question libre, réponse sourcée
//   GET  /topics/:slug/sources?days=1|2|3         sources brutes, sans IA (gratuit)
//   GET  /usage                          consommation de la clé sur 24 h
//
// Sources : arXiv (toutes les sciences), Hugging Face Daily Papers (IA) et modèles tendance,
//           versions GitHub des dépôts du sujet, blogs de labos (RSS), Hacker News (tech).
// Signaux : code publié (★), votes Hugging Face, poids publiés (modèle HF lié à l'article arXiv).
// Secrets : ANTHROPIC_API_KEY (obligatoire), CLAUDE_MODEL (défaut claude-sonnet-5-5),
//           CLAUDE_EFFORT (défaut low). SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY sont fournis.

import Anthropic from "npm:@anthropic-ai/sdk@1.12.0";
import { createClient } from "npm:@supabase/supabase-js@2.119.2";

const MODEL = Deno.env.get("CLAUDE_MODEL") || "claude-sonnet-5-5";
const EFFORT = (Deno.env.get("CLAUDE_EFFORT") || "low") as "low" | "medium" | "high";
const DIGEST_TTL_MS = 2 * 60 * 60 * 1000;
const UA = "zimb-news/1.0 (veille tech et science)";

const db = createClient(Deno.env.get("SUPABASE_URL")!, Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!, {
  auth: { persistSession: false },
});

// --------------------------------------------------------------------------- HTTP

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "x-api-key, content-type, authorization, apikey",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
};

class HttpError extends Error {
  constructor(public status: number, message: string) { super(message); }
}

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { ...CORS, "Content-Type": "application/json; charset=utf-8" } });

async function sha256(s: string): Promise<string> {
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(s));
  return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

type ApiKey = { id: number; name: string; daily_ai_limit: number };

async function authenticate(req: Request): Promise<ApiKey> {
  const key = req.headers.get("x-api-key")?.trim();
  if (!key) throw new HttpError(401, "Clé manquante : en-tête x-api-key.");
  const { data, error } = await db.from("api_keys")
    .select("id, name, daily_ai_limit").eq("key_hash", await sha256(key)).is("revoked_at", null).maybeSingle();
  if (error) throw error;
  if (!data) throw new HttpError(401, "Clé invalide ou révoquée.");
  return data;
}

async function usageLast24h(keyId: number): Promise<number> {
  const since = new Date(Date.now() - 86_400_000).toISOString();
  const { count, error } = await db.from("api_usage")
    .select("id", { count: "exact", head: true }).eq("key_id", keyId).gte("at", since);
  if (error) throw error;
  return count ?? 0;
}

async function checkQuota(key: ApiKey) {
  const used = await usageLast24h(key.id);
  if (used >= key.daily_ai_limit) {
    throw new HttpError(429, `Quota atteint : ${used}/${key.daily_ai_limit} appels à l'IA sur 24 h.`);
  }
}

async function logUsage(key: ApiKey, kind: string, topicId: number | null, usage: Usage) {
  await db.from("api_usage").insert({
    key_id: key.id, kind, topic_id: topicId, input_tokens: usage.input_tokens, output_tokens: usage.output_tokens,
  });
}

// --------------------------------------------------------------------------- Claude

type Usage = { model: string; input_tokens: number; output_tokens: number };
let client: Anthropic | null = null;
function claude(): Anthropic {
  if (!Deno.env.get("ANTHROPIC_API_KEY")) {
    throw new HttpError(503, "IA non configurée : ajoutez le secret ANTHROPIC_API_KEY dans Supabase (Edge Functions → Secrets).");
  }
  return (client ??= new Anthropic());
}

async function askClaude<T>(system: string, prompt: string, schema: Record<string, unknown>, maxTokens = 8000): Promise<{ data: T; usage: Usage }> {
  const params = {
    model: MODEL,
    max_tokens: maxTokens,
    system,
    messages: [{ role: "user" as const, content: prompt }],
    output_config: { effort: EFFORT, format: { type: "json_schema" as const, schema } },
  };
  const anthropic = claude();
  let resp;
  try {
    // Repli automatique sur un autre modèle si le modèle principal refuse la requête
    resp = await anthropic.beta.messages.create({
      ...params, betas: ["server-side-fallback-2026-07-01"], fallbacks: "default",
    });
  } catch (e) {
    if (!(e instanceof Anthropic.BadRequestError)) throw e;
    resp = await anthropic.messages.create(params);
  }
  if (resp.stop_reason === "refusal") throw new HttpError(502, "L'IA a refusé de traiter cette demande.");
  if (resp.stop_reason === "max_tokens") throw new HttpError(502, "Réponse de l'IA tronquée.");
  const block = resp.content.find((b: { type: string }) => b.type === "text") as { text: string } | undefined;
  if (!block) throw new HttpError(502, "Réponse de l'IA vide.");
  return {
    data: JSON.parse(block.text) as T,
    usage: { model: resp.model, input_tokens: resp.usage.input_tokens, output_tokens: resp.usage.output_tokens },
  };
}

// --------------------------------------------------------------------------- Sources

type SourceName = "arXiv" | "Hugging Face" | "Hacker News" | "Modèle HF" | "GitHub" | "Blog";
type Item = {
  id: string; source: SourceName;
  title: string; url: string; date: string; text: string; score?: string;
  signals?: string[];      // code publié, votes, poids… (affichés et transmis à l'IA)
  weight?: number;         // pour trier les sources les plus signalées en premier
  arxivId?: string;
};
type Search = {
  arxiv_query: string; hn_queries: string[]; keywords: string[];
  github_repos?: string[]; hf_model_keywords?: string[];
};

// Blogs de labos et d'éditeurs dont le flux RSS/Atom a été vérifié
const LAB_FEEDS = [
  { name: "OpenAI", url: "https://openai.com/news/rss.xml" },
  { name: "Google DeepMind", url: "https://deepmind.google/blog/rss.xml" },
  { name: "Google Research", url: "https://research.google/blog/rss/" },
  { name: "Hugging Face", url: "https://huggingface.co/blog/feed.xml" },
  { name: "NVIDIA", url: "https://developer.nvidia.com/blog/feed" },
  { name: "Berkeley AI Research", url: "https://bair.berkeley.edu/blog/feed.xml" },
];

const arxivIdOf = (url: string) => url.match(/(\d{4}\.\d{4,5})/)?.[1];
const stripHtml = (s: string) => decode(decode(s).replace(/<[^>]+>/g, " "));

// Lecture minimale d'un flux RSS ou Atom (les premières entrées seulement, pour limiter le CPU)
function parseFeed(xml: string, max = 40) {
  const out: { title: string; link: string; date: string; text: string }[] = [];
  for (const m of xml.matchAll(/<(item|entry)[\s>]([\s\S]*?)<\/\1>/g)) {
    const e = m[2];
    const link = e.match(/<link[^>]*href="([^"]+)"/)?.[1] ?? tag(e, "link");
    const date = tag(e, "pubDate") || tag(e, "published") || tag(e, "updated") || tag(e, "dc:date");
    const text = stripHtml(xmlRaw(e, "description") || xmlRaw(e, "summary") || xmlRaw(e, "content"));
    out.push({ title: stripHtml(xmlRaw(e, "title")), link: decode(link), date, text });
    if (out.length >= max) break;
  }
  return out;
}

const matchesAny = (text: string, kws: string[]) => {
  const hay = text.toLowerCase();
  return kws.some((k) => k && hay.includes(k.toLowerCase()));
};
const validDate = (d: string) => !Number.isNaN(new Date(d).getTime());

const decode = (s: string) => s
  .replace(/<!\[CDATA\[([\s\S]*?)\]\]>/g, "$1")
  .replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&amp;/g, "&")
  .replace(/\s+/g, " ").trim();
const xmlRaw = (xml: string, name: string) => xml.match(new RegExp(`<${name}[^>]*>([\\s\\S]*?)</${name}>`))?.[1] ?? "";
const tag = (xml: string, name: string) => decode(xmlRaw(xml, name));

async function fetchArxiv(q: string, since: Date): Promise<Item[]> {
  const url = `https://export.arxiv.org/api/query?search_query=${encodeURIComponent(q)}` +
    `&sortBy=submittedDate&sortOrder=descending&max_results=40`;
  const r = await fetch(url, { headers: { "User-Agent": UA } });
  if (!r.ok) throw new Error(`arXiv ${r.status}`);
  const xml = await r.text();
  return [...xml.matchAll(/<entry>([\s\S]*?)<\/entry>/g)]
    .map(([, e]) => {
      const authors = [...e.matchAll(/<name>([^<]+)<\/name>/g)].map((m) => m[1]);
      const url = tag(e, "id").replace(/^http:/, "https:");
      const comment = tag(e, "arxiv:comment");
      const code = comment.match(/https?:\/\/(?:www\.)?github\.com\/[\w.-]+\/[\w.-]+/)?.[0];
      return {
        id: "", source: "arXiv" as const,
        title: tag(e, "title"),
        url,
        arxivId: arxivIdOf(url),
        date: tag(e, "published"),
        // Résumé complet : les gains chiffrés sont souvent dans les dernières phrases
        text: `${authors.slice(0, 3).join(", ")}${authors.length > 3 ? " et al." : ""} — ${tag(e, "summary").slice(0, 2000)}` +
          (comment ? ` [Commentaire des auteurs : ${comment.slice(0, 200)}]` : ""),
        signals: code ? [`code : ${code.replace(/^https?:\/\/(www\.)?/, "")}`] : [],
        weight: code ? 3 : 0,
      };
    })
    .filter((i) => i.title && new Date(i.date) >= since);
}

// Modèles tendance sur Hugging Face, filtrés par les mots-clés du sujet (nom et étiquettes)
async function fetchHfModels(keywords: string[], since: Date): Promise<Item[]> {
  if (!keywords.length) return [];
  const r = await fetch("https://huggingface.co/api/models?sort=trendingScore&direction=-1&limit=100", { headers: { "User-Agent": UA } });
  if (!r.ok) throw new Error(`Hugging Face modèles ${r.status}`);
  // deno-lint-ignore no-explicit-any
  const rows: any[] = await r.json();
  // Les modèles tendance sont souvent sortis dans la semaine : fenêtre élargie à 7 jours minimum
  const from = new Date(Math.min(since.getTime(), Date.now() - 7 * 86_400_000));
  return rows
    .filter((m) => m.createdAt && new Date(m.createdAt) >= from)
    .filter((m) => matchesAny(`${m.id} ${(m.tags ?? []).join(" ")} ${m.pipeline_tag ?? ""}`, keywords))
    .slice(0, 10)
    .map((m) => {
      const tags: string[] = m.tags ?? [];
      const paper = tags.find((t) => t.startsWith("arxiv:"))?.slice(6);
      const base = tags.find((t) => t.startsWith("base_model:") && !t.includes(":finetune:") && !t.includes(":quantized:"))?.slice(11);
      return {
        id: "", source: "Modèle HF" as const,
        title: m.id,
        url: `https://huggingface.co/${m.id}`,
        date: m.createdAt,
        arxivId: paper,
        score: `${m.likes ?? 0} likes, ${m.downloads ?? 0} téléchargements`,
        text: `Modèle publié sur Hugging Face${m.pipeline_tag ? ` (${m.pipeline_tag})` : ""}` +
          `${base ? `, dérivé de ${base}` : ""}. Étiquettes : ${tags.filter((t) => !/^(region|license|endpoints|base_model):/.test(t)).slice(0, 12).join(", ")}`,
        signals: ["poids publiés"],
        weight: 4 + Math.min(4, Math.log10(1 + (m.likes ?? 0))),
      };
    });
}

// Versions publiées des dépôts GitHub suivis par le sujet (flux Atom public, sans quota d'API)
async function fetchGithubReleases(repos: string[], since: Date): Promise<Item[]> {
  const valid = repos.filter((r) => /^[\w.-]+\/[\w.-]+$/.test(r)).slice(0, 6);
  const results = await Promise.allSettled(valid.map(async (repo) => {
    const r = await fetch(`https://github.com/${repo}/releases.atom`, { headers: { "User-Agent": UA } });
    if (!r.ok) return [];
    return parseFeed(await r.text(), 10)
      .filter((e) => validDate(e.date) && new Date(e.date) >= since)
      .slice(0, 3)
      .map((e) => ({
        id: "", source: "GitHub" as const,
        title: `${repo} ${e.title}`,
        url: e.link.startsWith("http") ? e.link : `https://github.com${e.link}`,
        date: new Date(e.date).toISOString(),
        text: `Notes de version : ${e.text.slice(0, 900)}`,
        signals: ["version publiée"],
        weight: 5,
      }));
  }));
  return results.flatMap((r) => (r.status === "fulfilled" ? r.value : []));
}

// Blogs de labos : articles récents qui mentionnent les mots-clés du sujet
async function fetchLabBlogs(keywords: string[], since: Date): Promise<Item[]> {
  const results = await Promise.allSettled(LAB_FEEDS.map(async (f) => {
    const r = await fetch(f.url, { headers: { "User-Agent": UA } });
    if (!r.ok) return [];
    return parseFeed(await r.text(), 30)
      .filter((e) => validDate(e.date) && new Date(e.date) >= since && matchesAny(`${e.title} ${e.text}`, keywords))
      .map((e) => ({
        id: "", source: "Blog" as const,
        title: `${f.name} — ${e.title}`,
        url: e.link,
        date: new Date(e.date).toISOString(),
        text: e.text.slice(0, 900),
        signals: [`annonce ${f.name}`],
        weight: 4,
      }));
  }));
  return results.flatMap((r) => (r.status === "fulfilled" ? r.value : [])).slice(0, 10);
}

type HfPaperSignal = { votes: number; repo?: string; stars?: number };

// Daily Papers : articles mis en avant par la communauté, et signaux pour recouper les articles arXiv
async function fetchHfPapers(keywords: string[], since: Date): Promise<{ items: Item[]; byArxiv: Map<string, HfPaperSignal> }> {
  const r = await fetch("https://huggingface.co/api/daily_papers?limit=100", { headers: { "User-Agent": UA } });
  if (!r.ok) throw new Error(`Hugging Face ${r.status}`);
  // deno-lint-ignore no-explicit-any
  const rows: any[] = await r.json();
  const byArxiv = new Map<string, HfPaperSignal>();
  for (const p of rows) {
    if (p.paper?.id) {
      byArxiv.set(p.paper.id, {
        votes: p.paper.upvotes ?? 0,
        repo: p.paper.githubRepo?.replace(/^https?:\/\/(www\.)?/, "") || undefined,
        stars: p.paper.githubStars ?? undefined,
      });
    }
  }
  const items = rows
    .filter((p) => new Date(p.publishedAt) >= since)
    .filter((p) => matchesAny(`${p.title} ${p.paper?.summary ?? ""} ${(p.paper?.ai_keywords ?? []).join(" ")}`, keywords))
    .map((p) => {
      const s = byArxiv.get(p.paper?.id) ?? { votes: 0 };
      return {
        id: "", source: "Hugging Face" as const,
        title: p.title,
        url: `https://huggingface.co/papers/${p.paper?.id}`,
        arxivId: p.paper?.id,
        date: p.publishedAt,
        score: `${s.votes} votes`,
        text: String(p.paper?.summary || p.paper?.ai_summary || "").slice(0, 2000),
        signals: s.repo ? [`code : ${s.repo}${s.stars ? ` (★${s.stars})` : ""}`] : [],
        weight: Math.min(6, s.votes / 5) + (s.repo ? 3 : 0),
      };
    });
  return { items, byArxiv };
}

async function fetchHackerNews(queries: string[], since: Date): Promise<Item[]> {
  const ts = Math.floor(since.getTime() / 1000);
  const seen = new Set<string>();
  const out: Item[] = [];
  for (const q of queries.slice(0, 3)) {
    // Recherche stricte dans les titres : sans cela, Algolia remonte beaucoup de bruit
    const url = `https://hn.algolia.com/api/v1/search?query=${encodeURIComponent(q)}&tags=story` +
      `&restrictSearchableAttributes=title&typoTolerance=false` +
      `&numericFilters=${encodeURIComponent(`created_at_i>${ts},points>3`)}&hitsPerPage=20`;
    const r = await fetch(url, { headers: { "User-Agent": UA } });
    if (!r.ok) continue;
    // deno-lint-ignore no-explicit-any
    const { hits } = await r.json() as { hits: any[] };
    for (const h of hits) {
      if (seen.has(h.objectID)) continue;
      seen.add(h.objectID);
      out.push({
        id: "", source: "Hacker News",
        title: h.title,
        url: h.url || `https://news.ycombinator.com/item?id=${h.objectID}`,
        date: h.created_at,
        score: `${h.points} points, ${h.num_comments ?? 0} commentaires`,
        text: `Discussion : https://news.ycombinator.com/item?id=${h.objectID}` +
          (h.story_text ? ` — ${String(h.story_text).replace(/<[^>]+>/g, " ").slice(0, 400)}` : ""),
      });
    }
  }
  return out;
}

async function gather(topic: { search: Search }, days: number) {
  const since = new Date(Date.now() - days * 86_400_000);
  const s = topic.search;
  const names = ["arXiv", "Hugging Face", "Modèles Hugging Face", "GitHub", "Blogs", "Hacker News"];
  const results = await Promise.allSettled([
    fetchArxiv(s.arxiv_query, since),
    fetchHfPapers(s.keywords, since),
    fetchHfModels(s.hf_model_keywords ?? [], since),
    fetchGithubReleases(s.github_repos ?? [], since),
    fetchLabBlogs(s.keywords, since),
    fetchHackerNews(s.hn_queries, since),
  ]);
  const errors = results.flatMap((r, i) => (r.status === "rejected" ? [`${names[i]} indisponible`] : []));
  const ok = <T,>(i: number, empty: T): T => (results[i].status === "fulfilled" ? (results[i] as PromiseFulfilledResult<T>).value : empty);
  const arxiv = ok<Item[]>(0, []);
  const hf = ok<{ items: Item[]; byArxiv: Map<string, HfPaperSignal> }>(1, { items: [], byArxiv: new Map() });
  const models = ok<Item[]>(2, []);
  const releases = ok<Item[]>(3, []);
  const blogs = ok<Item[]>(4, []);
  const hn = ok<Item[]>(5, []);

  // Recoupement : un article arXiv mis en avant sur HF, avec du code ou des poids publiés, pèse plus lourd
  const weightsByArxiv = new Map<string, string>();
  for (const m of models) if (m.arxivId) weightsByArxiv.set(m.arxivId, m.title);
  const hfIds = new Set(hf.items.map((i) => i.arxivId));
  for (const a of arxiv) {
    const sig = a.arxivId ? hf.byArxiv.get(a.arxivId) : undefined;
    if (sig) {
      a.signals!.push(`mis en avant sur Hugging Face (${sig.votes} votes)`);
      if (sig.repo && !a.signals!.some((x) => x.startsWith("code"))) {
        a.signals!.push(`code : ${sig.repo}${sig.stars ? ` (★${sig.stars})` : ""}`);
        a.weight! += 3;
      }
      a.weight! += Math.min(6, sig.votes / 5);
    }
    const w = a.arxivId && weightsByArxiv.get(a.arxivId);
    if (w) { a.signals!.push(`poids publiés : ${w}`); a.weight! += 4; }
  }
  for (const h of hf.items) {
    const w = h.arxivId && weightsByArxiv.get(h.arxivId);
    if (w) { h.signals!.push(`poids publiés : ${w}`); h.weight! += 4; }
  }

  const byWeight = (a: Item, b: Item) => (b.weight ?? 0) - (a.weight ?? 0) || b.date.localeCompare(a.date);
  const items = [
    ...releases.sort(byWeight).slice(0, 10),
    ...models.sort(byWeight).slice(0, 10),
    ...blogs.sort(byWeight).slice(0, 8),
    ...hf.items.sort(byWeight).slice(0, 12),
    // évite les doublons avec les Daily Papers déjà listés
    ...arxiv.filter((a) => !hfIds.has(a.arxivId)).sort(byWeight).slice(0, 30),
    ...hn.slice(0, 15),
  ];
  items.forEach((it, i) => (it.id = `S${i + 1}`));
  return { items, errors, since };
}

const renderItems = (items: Item[]) => items.map((i) =>
  `[${i.id}] (${i.source}, ${i.date.slice(0, 10)}${i.score ? `, ${i.score}` : ""}) ${i.title}` +
  (i.signals?.length ? `\nSignaux : ${i.signals.join(" ; ")}` : "") +
  `\n${i.text}`).join("\n\n");

const pick = (items: Item[], ids: string[]) => {
  const byId = new Map(items.map((i) => [i.id, i]));
  return [...new Set(ids)].map((id) => byId.get(id)).filter(Boolean)
    .map((i) => ({
      id: i!.id, source: i!.source, title: i!.title, url: i!.url, date: i!.date,
      score: i!.score ?? null, signals: i!.signals ?? [],
    }));
};

// --------------------------------------------------------------------------- Sujets

const TOPIC_SYSTEM = `Tu prépares la veille d'un sujet tech ou science. À partir du nom et de la description donnés (souvent en français), produis :
- arxiv_query : une requête pour l'API arXiv (syntaxe search_query), en anglais, précise mais pas trop étroite. Utilise les préfixes all:, ti:, abs:, cat: et les opérateurs AND / OR / ANDNOT, avec des guillemets pour les expressions. Exemple : (abs:"quantum error correction" OR abs:"logical qubit") AND cat:quant-ph
- hn_queries : 1 à 3 requêtes courtes en anglais pour la recherche Hacker News (2 à 4 mots chacune, termes qu'on trouve dans des titres d'articles) ;
- keywords : 4 à 10 mots-clés ou expressions en anglais, en minuscules, pour filtrer des titres et résumés d'articles de recherche et de billets de blog ;
- github_repos : 0 à 6 dépôts GitHub de référence pour ce sujet, au format « propriétaire/nom » exact (par exemple vllm-project/vllm, ggml-org/llama.cpp), dont les nouvelles versions intéressent les praticiens. Liste vide si tu n'es pas sûr des noms exacts ;
- hf_model_keywords : 3 à 8 termes en minuscules qui apparaissent dans les noms ou étiquettes des modèles Hugging Face pertinents (par exemple gguf, awq, fp8, mlx pour la quantification). Liste vide si le sujet ne concerne pas des modèles ;
- description : une phrase en français qui décrit précisément le périmètre du sujet.`;

const TOPIC_SCHEMA = {
  type: "object",
  properties: {
    arxiv_query: { type: "string" },
    hn_queries: { type: "array", items: { type: "string" } },
    keywords: { type: "array", items: { type: "string" } },
    github_repos: { type: "array", items: { type: "string" } },
    hf_model_keywords: { type: "array", items: { type: "string" } },
    description: { type: "string" },
  },
  required: ["arxiv_query", "hn_queries", "keywords", "github_repos", "hf_model_keywords", "description"],
  additionalProperties: false,
};

const slugify = (s: string) => s.normalize("NFKD").replace(/[̀-ͯ]/g, "").toLowerCase()
  .replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 60) || "sujet";

async function createTopic(key: ApiKey, body: { name?: string; description?: string }) {
  const name = body.name?.trim();
  if (!name || name.length > 120) throw new HttpError(400, "Champ « name » requis (120 caractères max).");
  const description = body.description?.trim().slice(0, 600) || "";
  await checkQuota(key);
  const { data, usage } = await askClaude<Search & { description: string }>(
    TOPIC_SYSTEM, `Nom du sujet : ${name}\nDescription : ${description || "(aucune)"}`, TOPIC_SCHEMA, 2000);
  await logUsage(key, "topic", null, usage);

  let slug = slugify(name);
  const { data: clash } = await db.from("topics").select("id").eq("slug", slug).maybeSingle();
  if (clash) slug = `${slug}-${crypto.randomUUID().slice(0, 4)}`;
  const { data: topic, error } = await db.from("topics").insert({
    slug, name, description: description || data.description, created_by: key.id,
    search: {
      arxiv_query: data.arxiv_query, hn_queries: data.hn_queries, keywords: data.keywords,
      github_repos: data.github_repos, hf_model_keywords: data.hf_model_keywords,
    },
  }).select("id, slug, name, description, search, created_at").single();
  if (error) throw error;
  return topic;
}

async function getTopic(slug: string) {
  const { data, error } = await db.from("topics").select("id, slug, name, description, search").eq("slug", slug).maybeSingle();
  if (error) throw error;
  if (!data) throw new HttpError(404, `Sujet « ${slug} » introuvable.`);
  return data as { id: number; slug: string; name: string; description: string; search: Search };
}

// --------------------------------------------------------------------------- Synthèse

const DIGEST_SYSTEM = `Tu rédiges en français la veille quotidienne d'ingénieurs et de chercheurs en IA sur un sujet précis. Les sources récentes sont de plusieurs types : versions publiées de logiciels (GitHub), modèles publiés (Hugging Face), annonces de labos (blogs), articles de recherche (arXiv, Hugging Face Daily Papers), discussions (Hacker News). Chaque source a un identifiant [S1], [S2]… et parfois une ligne « Signaux » (code publié, votes, poids publiés, version publiée).

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
- trends : 0 à 4 tendances ou signaux faibles en une phrase chacune.`;

const DIGEST_SCHEMA = {
  type: "object",
  properties: {
    headline: { type: "string" },
    overview: { type: "string" },
    releases: {
      type: "array",
      items: {
        type: "object",
        properties: {
          title: { type: "string" },
          summary: { type: "string" },
          source_ids: { type: "array", items: { type: "string" } },
        },
        required: ["title", "summary", "source_ids"],
        additionalProperties: false,
      },
    },
    highlights: {
      type: "array",
      items: {
        type: "object",
        properties: {
          title: { type: "string" },
          summary: { type: "string" },
          why_it_matters: { type: "string" },
          source_ids: { type: "array", items: { type: "string" } },
        },
        required: ["title", "summary", "why_it_matters", "source_ids"],
        additionalProperties: false,
      },
    },
    trends: { type: "array", items: { type: "string" } },
  },
  required: ["headline", "overview", "releases", "highlights", "trends"],
  additionalProperties: false,
};

type Digest = {
  headline: string; overview: string; trends: string[];
  releases: { title: string; summary: string; source_ids: string[] }[];
  highlights: { title: string; summary: string; why_it_matters: string; source_ids: string[] }[];
};

async function digest(key: ApiKey, slug: string, days: number, force: boolean) {
  const topic = await getTopic(slug);
  if (!force) {
    const { data: cached } = await db.from("topic_digests")
      .select("content, n_items, model, generated_at").eq("topic_id", topic.id).eq("days", days)
      .gte("generated_at", new Date(Date.now() - DIGEST_TTL_MS).toISOString())
      .order("generated_at", { ascending: false }).limit(1).maybeSingle();
    if (cached) return { topic: { slug, name: topic.name }, days, cached: true, ...cached };
  }

  await checkQuota(key);
  const { items, errors, since } = await gather(topic, days);
  let content: Record<string, unknown>;
  let usage: Usage | null = null;
  if (!items.length) {
    content = {
      headline: "Aucune nouveauté repérée sur la période",
      overview: `Aucune source récente ne correspond à ce sujet depuis le ${since.toLocaleDateString("fr-FR")}.`,
      releases: [], highlights: [], trends: [], sources: [], errors,
    };
  } else {
    const r = await askClaude<Digest>(
      DIGEST_SYSTEM,
      `Sujet : ${topic.name}\nPérimètre : ${topic.description}\nPériode : ${days} dernier(s) jour(s)\n\nSOURCES :\n\n${renderItems(items)}`,
      DIGEST_SCHEMA,
    );
    usage = r.usage;
    await logUsage(key, "digest", topic.id, usage);
    const used = [...r.data.releases, ...r.data.highlights].flatMap((h) => h.source_ids);
    content = {
      ...r.data,
      releases: r.data.releases.map((h) => ({ ...h, sources: pick(items, h.source_ids) })),
      highlights: r.data.highlights.map((h) => ({ ...h, sources: pick(items, h.source_ids) })),
      sources: pick(items, used),
      n_considered: items.length,
      errors,
    };
  }
  const row = {
    topic_id: topic.id, days, content, n_items: items.length,
    model: usage?.model ?? null, input_tokens: usage?.input_tokens ?? null, output_tokens: usage?.output_tokens ?? null,
  };
  const { data: saved, error } = await db.from("topic_digests").insert(row).select("generated_at").single();
  if (error) throw error;
  return { topic: { slug, name: topic.name }, days, cached: false, content, n_items: items.length, model: row.model, generated_at: saved.generated_at };
}

// --------------------------------------------------------------------------- Question libre

const ASK_SYSTEM = `Tu réponds en français à une question sur un sujet de veille tech ou science, uniquement à partir des sources récentes fournies (identifiants [S1], [S2]…).
- Cite les sources entre crochets dans le texte, par exemple [S3].
- Si les sources ne permettent pas de répondre, dis-le clairement et indique ce qu'elles couvrent ; n'utilise pas de connaissances extérieures pour combler.
- Présente les résultats de recherche comme des affirmations des auteurs.
- Le lecteur est ingénieur ou chercheur : donne les chiffres disponibles, précise si le code ou les poids sont publiés (ligne « Signaux »), garde les termes techniques usuels en anglais.
- Réponse claire et structurée, 150 à 400 mots.`;

const ASK_SCHEMA = {
  type: "object",
  properties: {
    answer: { type: "string" },
    source_ids: { type: "array", items: { type: "string" } },
  },
  required: ["answer", "source_ids"],
  additionalProperties: false,
};

async function ask(key: ApiKey, slug: string, body: { question?: string; days?: number }) {
  const question = body.question?.trim();
  if (!question || question.length > 500) throw new HttpError(400, "Champ « question » requis (500 caractères max).");
  const days = clampDays(body.days ?? 3);
  const topic = await getTopic(slug);
  await checkQuota(key);
  const { items, errors } = await gather(topic, days);
  if (!items.length) {
    return { answer: "Aucune source récente ne correspond à ce sujet sur la période : impossible de répondre.", sources: [], errors };
  }
  const { data, usage } = await askClaude<{ answer: string; source_ids: string[] }>(
    ASK_SYSTEM,
    `Sujet : ${topic.name}\nPérimètre : ${topic.description}\nPériode : ${days} dernier(s) jour(s)\n\nSOURCES :\n\n${renderItems(items)}\n\nQUESTION : ${question}`,
    ASK_SCHEMA, 6000,
  );
  await logUsage(key, "ask", topic.id, usage);
  return { topic: { slug, name: topic.name }, days, question, answer: data.answer, sources: pick(items, data.source_ids), errors, model: usage.model };
}

// --------------------------------------------------------------------------- Routeur

const clampDays = (d: unknown) => Math.min(3, Math.max(1, Math.round(Number(d)) || 3));

async function route(req: Request): Promise<Response> {
  if (req.method === "OPTIONS") return new Response(null, { headers: CORS });
  const url = new URL(req.url);
  const path = url.pathname.replace(/^.*?\/news-api/, "").replace(/\/+$/, "") || "/";
  const key = await authenticate(req);
  const body = req.method === "POST" ? await req.json().catch(() => ({})) : {};

  if (path === "/topics" && req.method === "GET") {
    const { data, error } = await db.from("topics").select("slug, name, description, created_at").order("created_at", { ascending: false });
    if (error) throw error;
    return json({ topics: data });
  }
  if (path === "/topics" && req.method === "POST") return json(await createTopic(key, body), 201);
  if (path === "/usage" && req.method === "GET") {
    return json({ key: key.name, used_24h: await usageLast24h(key.id), limit_24h: key.daily_ai_limit });
  }
  let m = path.match(/^\/topics\/([a-z0-9-]+)\/digest$/);
  if (m && req.method === "GET") {
    return json(await digest(key, m[1], clampDays(url.searchParams.get("days") ?? 1), url.searchParams.get("refresh") === "1"));
  }
  m = path.match(/^\/topics\/([a-z0-9-]+)\/sources$/);
  if (m && req.method === "GET") {
    const topic = await getTopic(m[1]);
    const days = clampDays(url.searchParams.get("days") ?? 1);
    const { items, errors } = await gather(topic, days);
    return json({ topic: { slug: topic.slug, name: topic.name }, days, search: topic.search, errors, items });
  }
  m = path.match(/^\/topics\/([a-z0-9-]+)\/ask$/);
  if (m && req.method === "POST") return json(await ask(key, m[1], body));
  m = path.match(/^\/topics\/([a-z0-9-]+)$/);
  if (m && req.method === "GET") return json(await getTopic(m[1]));
  throw new HttpError(404, `Route inconnue : ${req.method} ${path}`);
}

Deno.serve(async (req) => {
  try {
    return await route(req);
  } catch (e) {
    if (e instanceof HttpError) return json({ error: e.message }, e.status);
    if (e instanceof Anthropic.RateLimitError) return json({ error: "Limite de débit de l'IA atteinte, réessayez dans une minute." }, 429);
    if (e instanceof Anthropic.AuthenticationError) return json({ error: "Clé Claude manquante ou invalide côté serveur (secret ANTHROPIC_API_KEY)." }, 500);
    if (e instanceof Anthropic.APIError) return json({ error: `Erreur de l'IA (${e.status}).` }, 502);
    console.error(e);
    return json({ error: "Erreur interne." }, 500);
  }
});
