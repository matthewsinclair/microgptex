---
verblock: "02 Mar 2026:v0.1: matts - Initial version"
wp_id: WP-08
title: "Mermaid Diagrams in Livebook"
scope: Medium
status: Not Started
---

# WP-08: Mermaid Diagrams in Livebook

## Objective

Add Mermaid diagrams to `notebooks/walkthrough.livemd` that visually explain what each code section does before the reader runs it. The growingswe walkthrough uses node-and-edge computation graphs, pipeline block diagrams, and attention heatmap-style visuals — Mermaid can replicate the structural diagrams natively in Livebook.

## What Mermaid Can Do in Livebook

Livebook renders Mermaid diagrams in markdown cells via fenced code blocks:

    ```mermaid
    graph TD
      A[Token] --> B[Embedding]
    ```

This gives us static structural diagrams — no dependencies, no Kino, just markdown.

## Proposed Diagrams

### Chapter 1: Autograd

- **Computation graph** — show a simple expression like `a*b + a` as a directed acyclic graph with nodes for operations and edges for data flow. Annotate edges with local gradients.
- **Backward pass** — same graph with gradient values flowing in reverse direction, showing how `backward/1` traverses the topology.
- **Fan-out** — graph for `a * a` showing the same node feeding two inputs to `mul`, with gradient accumulation at the junction.

### Chapter 4: Model Architecture

- **GPT forward pass pipeline** — the full data flow from token input through embedding, RMSNorm, attention block, MLP block, to output logits. Matches the ASCII art already in the `Model` moduledoc but rendered visually.
- **Multi-head attention detail** — Q/K/V projections splitting into heads, scaled dot-product, softmax, weighted sum, concatenation, output projection.
- **KV cache growth** — sequence diagram showing how the cache grows as tokens are processed: position 0 stores 1 KV pair, position 1 stores 2, etc.

### Chapter 5: Training

- **Training loop** — flowchart: pick doc → forward pass → backward → Adam update → repeat. Shows which modules are involved at each step.
- **Params round-trip** — data flow diagram: `Model.params/1` → `Value.backward/1` → `Adam.step/5` → `Model.update_params/2`, showing the `{tag, row, col}` IDs as the linking key.

### Chapter 7: Full Pipeline

- **Module dependency graph** — the 9-module dependency chain as a directed graph, showing which modules depend on which.

## Deliverables

- Mermaid diagram blocks added to each chapter of `notebooks/walkthrough.livemd`
- Each diagram placed immediately before the code it explains
- Brief caption text below each diagram connecting the visual to the code

## Acceptance Criteria

- [ ] Diagrams render correctly in Livebook (no broken Mermaid syntax)
- [ ] Each chapter with code has at least one structural diagram
- [ ] Diagrams match the actual code structure (not aspirational/hypothetical)
- [ ] No external dependencies — Mermaid only, rendered by Livebook's built-in support

## Estimate

Medium scope. Writing and iterating on ~8-10 Mermaid diagrams, testing each in Livebook. The main effort is getting the diagrams to be genuinely useful rather than decorative — each should convey something that's harder to see in the code alone.

## Dependencies

- Depends on WP-07 (Livebook walkthrough must exist first)
