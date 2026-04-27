# Claude Code Session Restart

## Project: MicroGPTEx

Functional pedagogical GPT trainer in Elixir. Zero external deps. Single file: `lib/microgptex.ex` (~1,521 lines).

## Status: All Work Complete

- **ST0001** (implementation): Done. See `intent/st/ST0001/info.md`
- **ST0002** (blog series): Done. See `intent/st/ST0002/info.md`

## TODO

1. **Push to GitHub** -- nothing has been pushed yet. The "Run in Livebook" badge URLs in Part 4 and README won't work until the repo is public at `github.com/matthewsinclair/microgptex`.

2. **Fix Mermaid diagrams in Livebook** -- several diagrams in `notebooks/walkthrough.livemd` show "Unsupported markdown: list" errors. Audit all 12 Mermaid blocks for Livebook compatibility.

3. **Publish blog posts** -- 4 posts in `docs/blog/` are ready, platform TBD.

4. **Interactive Livebook (WP-09)** -- plan at `.claude/plans/snug-mixing-patterson.md`. Adds Kino sliders, VegaLite charts, and attention heatmaps to `notebooks/interactive.livemd`. Not yet started.

## Key References

- `intent/wip.md` -- current project state
- `intent/restart.md` -- session restart context with style rules
- `intent/st/ST0002/design.md` -- blog series design decisions
- `intent/st/ST0002/impl.md` -- what was built and editorial choices
- `docs/explainer/ABOUT.md` -- provenance note for NotebookLM-generated assets
- `CLAUDE.md` -- project guidelines (Intent v2.5.0)

## Style Rules (carry forward)

- Australian/British English for prose; American English for code identifiers
- Depersonalised technical voice (no "I/me" in technical sections)
- Credit Karpathy and growingswe prominently
