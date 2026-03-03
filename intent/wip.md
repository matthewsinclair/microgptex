---
verblock: "03 Mar 2026:v0.2: matts - ST0002 complete, updated WIP"
---

# Work In Progress

## Current Focus

No active work. Both steel threads are complete.

## Completed Steel Threads

- **ST0001**: MicroGPTEx implementation — 9 modules, single file, zero deps, comprehensive tests, rich moduledocs, Livebook walkthrough. Status: Done.
- **ST0002**: Blog post series — 4-part series (autograd, model, attention, training), all written, reviewed, polished. Australian/British English, depersonalised technical voice, Karpathy/growingswe credit amplified. Status: Done.

## Deliverables

### Code

- `lib/microgptex.ex` — ~1,521 lines (558 code, 664 commentary, 299 blank)
- `test/microgptex_test.exs` — behavioural tests covering all modules

### Blog Posts

- `docs/blog/part1-autograd.md` — "What If Numbers Could Remember?"
- `docs/blog/part2-model.md` — "From Letters to Logits"
- `docs/blog/part3-attention.md` — "How Tokens Talk to Each Other"
- `docs/blog/part4-training.md` — "Learning and Dreaming"

### Livebooks

- `notebooks/walkthrough.livemd` — step-by-step code walkthrough (7 chapters, 12 Mermaid diagrams)
- `notebooks/interactive.livemd` — Kino-based interactive explorations (softmax, temperature, gradients, training, attention)

## Upcoming Work

- Push to GitHub and verify "Run in Livebook" badge URLs work
- Publish blog posts (platform TBD)
- Fix Mermaid diagram rendering issues in Livebook (some diagrams show "Unsupported markdown: list")
- Consider WP-09 interactive Livebook enhancements (plan exists at `.claude/plans/snug-mixing-patterson.md`)

## Notes

- Livebook notebooks use conditional `Mix.install` — local path first, GitHub fallback
- All spelling is Australian/British English (normalise, tokeniser, etc.)
- Code identifiers remain American English (as per Elixir convention)
