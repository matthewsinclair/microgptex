# Implementation — ST0002: MicroGPTEx Blog Post Series

## Status: Complete

All 4 blog posts written, reviewed, and polished. All work packages done.

## What Was Done

### WP-01: Part 1 — Autograd (`docs/blog/part1-autograd.md`)

- Full post covering Value struct, computation graphs, backward pass, fan-out, Elixir vs Python comparison, threaded RNG, bump test verification
- Scope/tooling disclaimer added (learning tool, not production; built with Claude Code and Intent)
- "Who made this?" author credit section at end
- All diagrams and code examples from the actual codebase

### WP-02: Part 2 — Model (`docs/blog/part2-model.md`)

- Full post covering tokeniser, math building blocks (dot, linear, softmax, rmsnorm, add_vec, relu_vec), model architecture, forward pass, lists-as-vectors rationale
- Added aside explaining strings vs atoms for parameter IDs (string interpolation for layer tags)

### WP-03: Part 3 — Attention (`docs/blog/part3-attention.md`)

- Full post covering Q/K/V projections, scaled dot-product attention, multi-head attention, KV cache (prepend + reverse pattern), residual connections, RMSNorm, complete transformer block diagram

### WP-04: Part 4 — Training & Generation (`docs/blog/part4-training.md`)

- Full post covering cross-entropy loss, Adam optimiser (with Elixir vs Python comparison), params round-trip, training loop as `Enum.reduce`, autoregressive sampling, temperature, full pipeline, reflections
- "Run in Livebook" badge URLs for both notebooks
- Expanded closing credit to Karpathy and growingswe

### WP-05: Review & Cross-references

- Consistent terminology across all 4 parts
- Cross-reference links between parts ("Part 1 covered...", "Part 2 added...")
- All spelling converted to Australian/British English (normalise, randomise, organise, tokeniser, etc.)
- All technical prose depersonalised (no "I/me" in technical sections)
- Removed all `---` horizontal rules
- Karpathy/growingswe credit amplified throughout

## Editorial Decisions

1. **Depersonalised technical voice**: "I" only in personal intro/reflection; technical prose uses "the model", "the forward pass", "the code"
2. **Series recaps**: Each part opens with "Part N covered..." rather than "I built..."
3. **Descriptive section headings**: "The foundation so far" / "Three more modules" / "The full attention mechanism" / "The complete picture" instead of "What I've built"
4. **Comment-to-code ratio**: Source file is 558 code lines (36.7%), 664 commentary lines (43.7%), 299 blank lines (19.7%) — ratio 1.19:1
