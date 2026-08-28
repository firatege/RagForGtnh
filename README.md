# RAG for GTNH

A production-shaped RAG pipeline that answers questions about **GT New Horizons**, a notoriously deep Minecraft tech-progression modpack (15 voltage tiers, thousands of interlocking recipes). Built to test how far a hybrid retrieval + LLM-synthesis stack can go on a real, messy, domain-specific knowledge base — not a toy FAQ bot.

Live deployment answers questions directly in-game: players type `ai! <question>` in the server chat, the bot retrieves and synthesizes an answer from the wiki corpus, and posts it back — see [`events/event.ts`](events/event.ts).

## Why this project

Most RAG demos retrieve from a handful of clean paragraphs. This one had to survive:
- ~130 unstructured PDF game-mechanics docs (grouped by tech tier: Stone Age → IV) plus a scraped wiki, with wildly inconsistent formatting
- highly technical, jargon-dense text where naive chunking destroys meaning (a machine's recipe table split mid-row is useless)
- questions that are really 2–3 questions stapled together ("what's the fastest way to LV and what do I need for the first EBF?")

Every design choice below is a direct answer to one of those problems.

## Pipeline

```
Ingest ─┬─ PDF corpus (pdf-ts) ──────────┐
        └─ Wiki scrape (axios+cheerio) ──┤
                                          ▼
                              LLM semantic chunking (Gemini 2.0 Flash)
                              350–450 chars, 1–2 sentence overlap,
                              paragraph-based fallback on failure
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    ▼                                           ▼
         local embedding (modernbert-embed-large,        keyword extraction
         1024-dim, mean-pooled, runs in-process           (Gemini, structured
         via transformers.js — no embedding API call)      JSON schema)
                    │                                           │
                    └─────────────────────┬─────────────────────┘
                                           ▼
                      Postgres + pgvector, HNSW index (vector_l2_ops)
                      two-level schema: sections (topic summaries) → chunks

Query ── query decomposition (Gemini: 1 question → 3-4 sub-queries) ──┐
                                                                       ▼
                                            per sub-query, in parallel:
                                            hybrid score = 0.7–0.9·cosine + 0.1–0.3·keyword_overlap
                                            (single CTE query, not two passes)
                                                                       │
                                            ± context-window join: pull
                                            neighboring chunk_ids around
                                            each top hit (small-to-big)
                                                                       ▼
                              grounded answer synthesis (Gemini) —
                              system prompt hard-constrains "GTNH questions:
                              answer only from provided chunks, no outside
                              knowledge" + explicit off-topic refusal
```

## What's actually interesting here

- **Hybrid retrieval in one SQL query, not two.** The embedding-similarity score and keyword-overlap score are computed and weighted inside the same CTE (`db/process.ts`), instead of running vector search and full-text search separately and merging in application code (RRF-style). Simpler to reason about, one round-trip to Postgres.
- **Context-window expansion at retrieval time.** After ranking chunks, a second CTE pulls every chunk within `±N` of each top hit's `chunk_id`, so the model gets surrounding context instead of an isolated 400-character fragment — a lightweight version of small-to-big retrieval without a separate parent-doc store.
- **Local embeddings, not an API.** `utils/embedding.ts` loads `lightonai/modernbert-embed-large` via `@huggingface/transformers` and runs inference in-process. No per-query embedding cost or latency to an external provider.
- **LLM-in-the-loop chunking with a hard fallback.** Chunk boundaries are decided by an LLM call constrained to a JSON schema (keeps recipe tables and multi-step instructions intact instead of splitting on fixed character counts). If the model call fails or returns malformed JSON, it falls back to deterministic paragraph-based chunking — the pipeline degrades, it doesn't break.
- **Query decomposition before retrieval.** Compound questions are split into 3–4 sub-queries by an LLM call before embedding, each retrieved independently and merged — closer to how a human would actually search a wiki for a multi-part question.
- **Rate limiting as a first-class concern.** A small async queue (`RateLimiter` in `db/process.ts`) throttles every Gemini call (embedding-adjacent keyword extraction, chunking, decomposition, synthesis) to stay under quota, instead of firing requests unbounded and handling 429s reactively.
- **Grounding is enforced in the prompt, not assumed.** The synthesis model's system instruction explicitly forbids outside knowledge for in-scope questions and requires a topic refusal otherwise — the failure mode of "confidently answers from parametric memory instead of retrieved context" is designed against, not left to chance.

## Stack

`Bun` + `TypeScript` · `PostgreSQL` + `pgvector` (HNSW) · `@huggingface/transformers` (local embeddings) · `Gemini 2.0 Flash` (chunking, keyword extraction, query decomposition, answer synthesis) · `cheerio`/`axios` (wiki scraping) · `pdf-ts` (PDF extraction)

## Known limitations

This was built to explore retrieval quality, not to be a hardened service — there's no automated eval set (retrieval is judged manually against known GTNH answers), no reranking stage, and the in-game chat integration polls rather than using a push/webhook model. The next iteration would add a golden Q&A set and a cross-encoder rerank pass before synthesis.
