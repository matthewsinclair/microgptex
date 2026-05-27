---
verblock: "27 May 2026:v0.3: matts - All three STs closed; no active work"
---

# Work In Progress

## Current Focus

No active work. All steel threads are complete and registered under `intent/st/COMPLETED/`. `intent st list` is empty.

## Completed Steel Threads

- **ST0001**: MicroGPTEx implementation — 9 modules, single file, zero deps, comprehensive tests, rich moduledocs, Livebook walkthrough + interactive (Kino) notebook. All 9 WPs Done. Status: Done.
- **ST0002**: Blog post series — 4-part series (autograd, model, attention, training), all written, reviewed, polished. Australian/British English, depersonalised technical voice, Karpathy/growingswe credit. All 5 WPs Done. Status: Done.
- **ST0003**: 20 AI Concepts explainer series — 24-doc set in `docs/20-concepts/` (ToC, README, 20 chapters, recap, references) + 21 images. Refs verified (zero hallucinated), detroped, and published via the mdagg + pandoc/typst pipeline. Status: Done.

## Deliverables

### Code

- `lib/microgptex.ex` — ~1,521 lines (558 code, 664 commentary, 299 blank)
- `test/microgptex_test.exs` — behavioural tests covering all modules

### Blog Posts

- `docs/blog/part1-autograd.md` — "What If Numbers Could Remember?"
- `docs/blog/part2-model.md` — "From Letters to Logits"
- `docs/blog/part3-attention.md` — "How Tokens Talk to Each Other"
- `docs/blog/part4-training.md` — "Learning and Dreaming"

### 20 Concepts

- `docs/20-concepts/` — 24 markdown docs + 21 images; published PDF set via mdagg + pandoc/typst

### Livebooks

- `notebooks/walkthrough.livemd` — step-by-step code walkthrough with Mermaid diagrams
- `notebooks/interactive.livemd` — Kino-based interactive explorations (softmax, temperature, gradients, training, attention)

### Explainer Assets

- `docs/explainer/` — NotebookLM-generated audio, video, infographic, slides, PDFs (see `ABOUT.md` for provenance)

## Upcoming Work

None tracked. Project deliverables are complete and published.

## Notes

- Livebook notebooks use conditional `Mix.install` — local path first, GitHub fallback
- All spelling is Australian/British English (normalise, tokeniser, etc.)
- Code identifiers remain American English (as per Elixir convention)
