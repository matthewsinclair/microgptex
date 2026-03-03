# Session Restart Context

## Project

MicroGPTEx — a functional, pedagogical GPT trainer in Elixir. Faithful translation of Karpathy's MicroGPT.

## What's Done

Both steel threads are complete:

- **ST0001** (implementation): 9 modules in `lib/microgptex.ex`, comprehensive tests, rich moduledocs, Livebook walkthrough + interactive notebook. See `intent/st/ST0001/info.md`.
- **ST0002** (blog series): 4-part series in `docs/blog/`. All written, reviewed, polished with Australian/British English and depersonalised technical voice. See `intent/st/ST0002/info.md`.

## Key Files

- `lib/microgptex.ex` — the entire implementation (~1,521 lines)
- `docs/blog/part{1,2,3,4}-*.md` — the 4 blog posts
- `notebooks/walkthrough.livemd` — step-by-step Livebook walkthrough
- `notebooks/interactive.livemd` — Kino-based interactive explorations
- `docs/explainer/` — NotebookLM-generated explainer assets (see `ABOUT.md`)
- `README.md` — project overview
- `priv/config.yaml` — default configuration

## Outstanding Items

1. **Push to GitHub** — code and blog posts need to be pushed; "Run in Livebook" badge URLs won't work until then
2. **Mermaid diagram fixes** — some Livebook Mermaid blocks show rendering errors ("Unsupported markdown: list")
3. **Blog publication** — posts are written but not published to any platform yet
4. **Interactive Livebook (WP-09)** — plan exists (`.claude/plans/snug-mixing-patterson.md`) for enhanced Kino widgets but not yet implemented

## Style Rules

- Australian/British English for all prose (normalise, tokeniser, etc.)
- Code identifiers stay American English (Elixir convention)
- Technical prose is depersonalised — no "I/me" in technical sections
- "I" permitted in personal motivation/reflection only (Part 1 intro and closing)
- Credit Karpathy and growingswe prominently
