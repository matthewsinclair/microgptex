# Session Restart Context

## Project

MicroGPTEx — a functional, pedagogical GPT trainer in Elixir. Faithful translation of Karpathy's MicroGPT.

## What's Done

All three steel threads are complete and live under `intent/st/COMPLETED/`. `intent st list` is empty.

- **ST0001** (implementation): 9 modules in `lib/microgptex.ex`, comprehensive tests, rich moduledocs, Livebook walkthrough + interactive notebook. All 9 WPs Done. See `intent/st/COMPLETED/ST0001/info.md`.
- **ST0002** (blog series): 4-part series in `docs/blog/`. All written, reviewed, polished with Australian/British English and depersonalised technical voice. All 5 WPs Done. See `intent/st/COMPLETED/ST0002/info.md`.
- **ST0003** (20 AI Concepts explainer): 24-doc set in `docs/20-concepts/`, refs verified, detroped, and published via the mdagg + pandoc/typst pipeline. See `intent/st/COMPLETED/ST0003/info.md`.

## Key Files

- `lib/microgptex.ex` — the entire implementation (~1,521 lines)
- `docs/blog/part{1,2,3,4}-*.md` — the 4 blog posts
- `docs/20-concepts/` — the 20 Concepts explainer set (24 docs + 21 images)
- `notebooks/walkthrough.livemd` — step-by-step Livebook walkthrough
- `notebooks/interactive.livemd` — Kino-based interactive explorations
- `docs/explainer/` — NotebookLM-generated explainer assets (see `ABOUT.md`)
- `README.md` — project overview
- `priv/config.yaml` — default configuration

## Outstanding Items

None. All deliverables are complete and published.

## Style Rules

- Australian/British English for all prose (normalise, tokeniser, etc.)
- Code identifiers stay American English (Elixir convention)
- Technical prose is depersonalised — no "I/me" in technical sections
- "I" permitted in personal motivation/reflection only (blog Part 1 intro and closing)
- Credit Karpathy and growingswe prominently
- xcancel never x.com for X/Twitter links
