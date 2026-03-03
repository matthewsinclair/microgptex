---
verblock: "02 Mar 2026:v0.1: matts - Initial version"
wp_id: WP-09
title: "Interactive Visualizations in Livebook"
scope: Large
status: WIP
---

# WP-09: Interactive Visualizations in Livebook

## Objective

Add interactive, manipulable visualizations to the Livebook walkthrough, inspired by the growingswe.com/blog/microgpt walkthrough. The growingswe page has sliders that adjust logit values and show softmax probabilities in real time, step-through carousels for computation graphs and training steps, and attention heatmaps with head selection tabs. We want the Livebook equivalent using Kino.

## What growingswe Does

The growingswe walkthrough uses custom React components for:

1. **Softmax demo** — sliders to adjust individual logit values, bar chart showing probability distribution updates in real time
2. **Temperature slider** — adjust temperature, see the probability distribution sharpen/flatten with entropy metric
3. **Cross-entropy loss curve** — slider for probability (0-1), real-time loss curve showing -log(p)
4. **Computation graph step-through** — forward pass builds the graph node by node, backward pass shows gradients propagating
5. **Attention heatmap** — matrix visualization with head selection tabs (Head 1/2/3/4)
6. **Training loss curve** — animated loss plot over training steps with generated samples at each checkpoint
7. **Name generation step-through** — character-by-character generation showing model confidence at each step

## What Kino Can Do

Livebook's [Kino](https://hexdocs.pm/kino/) library provides:

- **`Kino.Input`** — text inputs, sliders, selects, number inputs
- **`Kino.Control`** — buttons, forms
- **`Kino.Frame`** — dynamic output that can be updated in place
- **`Kino.Layout`** — tabs, grids for arranging multiple outputs
- **`Kino.VegaLite`** — Vega-Lite chart specifications (bar charts, line plots, heatmaps)
- **`Kino.Mermaid`** — programmatic Mermaid diagrams (dynamically generated)
- **`Kino.JS`** / **`Kino.JS.Live`** — custom JavaScript widgets for anything Kino doesn't cover natively

## Proposed Interactive Elements

### Tier 1: Kino + VegaLite (no custom JS)

These use standard Kino widgets and VegaLite charts:

- **Softmax explorer** — `Kino.Input.range` sliders for 3-4 logit values, `Kino.VegaLite` bar chart showing the probability distribution. User drags a slider, the chart updates.
- **Temperature explorer** — temperature slider + VegaLite bar chart showing how the distribution sharpens/flattens. Display entropy value.
- **Loss curve** — `Kino.VegaLite` line chart that updates during training via `Kino.VegaLite.push`. User sees the loss decrease in real time as training runs.
- **Attention weights heatmap** — `Kino.VegaLite` heatmap of attention scores, with `Kino.Layout.tabs` to switch between heads.

### Tier 2: Kino.Frame + dynamic updates

These use `Kino.Frame` to update output dynamically:

- **Training step-by-step** — button to advance one training step. Frame shows current loss, gradient norms, and learning rate. VegaLite chart accumulates loss points.
- **Generation step-through** — button to generate one token at a time. Frame shows the current sequence, probability distribution over next token, and which token was sampled.

### Tier 3: Kino.JS.Live (custom JavaScript)

These require custom JS for interaction patterns Kino doesn't cover:

- **Computation graph visualizer** — interactive DAG visualization where nodes light up during forward/backward pass. Would need a JS graph library (e.g., D3, dagre).
- **Animated backward pass** — step through topological order, highlighting each node as its gradient is computed.

## Approach

1. Start with Tier 1 (standard Kino + VegaLite) — highest value, lowest risk
2. Add Tier 2 if Tier 1 works well — incremental complexity
3. Tier 3 only if there's a compelling reason — custom JS is fragile and hard to maintain

## Dependencies

New Mix dependencies needed (dev/notebook only):

```elixir
# In the Livebook setup cell, not in mix.exs
Mix.install([
  {:microgptex, path: ".."},
  {:kino, "~> 0.14"},
  {:kino_vega_lite, "~> 0.1"}
])
```

This keeps the library itself zero-dependency. The interactive notebook is a separate artifact that adds Kino.

## Deliverables

- Enhanced `notebooks/walkthrough.livemd` with interactive visualizations
- Or a separate `notebooks/interactive.livemd` if the additions make the original too heavy
- Each interactive element has a static fallback (the existing code cells still work without Kino)

## Acceptance Criteria

- [ ] At least 4 interactive elements from Tier 1
- [ ] Softmax explorer with sliders and live bar chart
- [ ] Training loss curve that updates in real time
- [ ] All interactive cells work in Livebook 0.14+
- [ ] The base walkthrough (`walkthrough.livemd`) still works without Kino

## Estimate

Large scope. Tier 1 alone requires VegaLite chart specifications, Kino input wiring, and testing in Livebook. Tier 2 adds dynamic state management. Recommend starting with Tier 1 as a standalone deliverable and evaluating whether Tier 2/3 add enough value.

## Open Questions

- Separate notebook (`interactive.livemd`) vs enhancing the existing one?
- Is `kino_vega_lite` sufficient for heatmaps, or do we need `kino_explorer`?
- Should the interactive notebook duplicate the narrative text, or assume the reader has gone through the static walkthrough first?

## Dependencies

- Depends on WP-07 (base Livebook walkthrough)
- Depends on WP-08 (Mermaid diagrams — some may be replaced by Kino.Mermaid programmatic equivalents)
