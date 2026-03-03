# Design — ST0002: MicroGPTEx Blog Post Series

## As-Built Design

### Structure

A 4-part blog post series explaining how GPT works through MicroGPTEx's Elixir implementation. Each post is self-contained but builds on the previous, following the bottom-up dependency order of the codebase itself.

### Files

```
docs/blog/
  part1-autograd.md   — "What If Numbers Could Remember?" (autograd, Value struct, backward pass, threaded RNG)
  part2-model.md      — "From Letters to Logits" (tokeniser, math building blocks, model architecture, forward pass)
  part3-attention.md  — "How Tokens Talk to Each Other" (Q/K/V, scaled dot-product, multi-head, KV cache, residual connections)
  part4-training.md   — "Learning and Dreaming" (cross-entropy, Adam optimiser, training loop, autoregressive sampling)
```

### Narrative Design

**Central question**: What does Elixir make explicit that Python hides?

- Part 1: Gradients are data (immutable maps), not side effects (mutable `.grad` fields)
- Part 2: All state flows through function arguments — no hidden globals, no mutation
- Part 3: Attention is just math — dot products, softmax, weighted sums
- Part 4: The training loop is a reduce; determinism is structural, not opt-in

### Voice and Style

- Third-person impersonal for all technical prose ("the model computes...", "the forward pass produces...")
- First person permitted only in personal motivation/reflection paragraphs (intro and closing of Part 1)
- Australian/British English spelling throughout (normalise, randomise, organise, tokeniser, etc.)
- Real code from MicroGPTEx, not pseudocode
- Elixir vs Python side-by-side comparisons where the idiom differs
- Each post ~2000-3000 words

### Credits

All posts credit Andrej Karpathy's MicroGPT and the growingswe.com walkthrough prominently. Attribution is woven into the narrative rather than relegated to footnotes. Part 1 includes a "Who made this?" section. Part 4's closing amplifies the credit.

### Companion Livebooks

Two Livebook notebooks complement the blog series:

- `notebooks/walkthrough.livemd` — step-by-step code walkthrough (zero deps beyond microgptex)
- `notebooks/interactive.livemd` — Kino-based interactive explorations (sliders, charts, heatmaps)

Both use conditional `Mix.install` that tries local path first, falls back to GitHub.

Stable "Run in Livebook" URLs point to the GitHub repo:

- `https://livebook.dev/run?url=https://github.com/matthewsinclair/microgptex/blob/main/notebooks/walkthrough.livemd`
- `https://livebook.dev/run?url=https://github.com/matthewsinclair/microgptex/blob/main/notebooks/interactive.livemd`
