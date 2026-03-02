---
verblock: "02 Mar 2026:v0.2: matts - As-built documentation"
intent_version: 2.4.0
status: WIP
slug: initial-version
created: 20260302
completed:
---

# ST0001: MicroGPTEx — Functional Pedagogical GPT Trainer

## Objective

Implement a functional, pedagogical GPT trainer in idiomatic Elixir — a faithful translation of Andrej Karpathy's MicroGPT demonstrating reverse-mode autograd, multi-head self-attention, Adam optimization, and autoregressive sampling using only scalar operations. Zero external dependencies.

## Context

Based on [Karpathy's MicroGPT](https://github.com/karpathy/microgpt) (2026-02-12) and the [growingswe walkthrough](https://growingswe.com/blog/microgpt). The Python original is ~200 lines of NumPy-style scalar ops that train a tiny character-level GPT on names.

The Elixir translation expands the pedagogical scope by making explicit what Python hides:

- **Immutable autograd** — `backward/1` returns a `%{id => gradient}` map instead of mutating `.grad` fields in-place. Gradient accumulation for fan-out (same value used multiple times) is handled by `Map.update/4`.
- **Threaded state** — RNG, KV cache, and optimizer moments flow explicitly through function arguments and return values as `{result, new_state}` tuples. Same seed always produces the same training run.
- **Pipe/pattern-match/with idioms** — the training pipeline expressed as composable, readable Elixir using multi-clause functions, guards, and `Enum.reduce/3`.

## Design Rationale

Elixir adds genuine value to this pedagogical project:

1. **Immutability makes data flow explicit** — every transformation produces a new value, making the training loop's data dependencies visible in the code itself
2. **Pattern matching replaces conditionals** — function clauses dispatch on structure, not if/else chains
3. **No hidden state** — unlike Python where `loss.backward()` mutates gradients in-place, Elixir's `backward/1` returns a gradient map, making the flow of derivatives through the computation graph an explicit data structure
4. **Deterministic by construction** — threaded RNG state means reproducibility is not opt-in but structural

## As-Built Architecture

Nine modules in a single file (`lib/microgptex.ex`), ordered bottom-up by dependency — each module can only depend on modules defined above it. The architecture enforces a clean IO boundary: eight inner modules are pure (no side effects), and the top-level `Microgptex` module is the only one that touches the outside world (config files, network, stdout).

Key architectural features:

- **Immutable gradient maps** — `backward/1` returns `%{id => gradient}` instead of mutating nodes in-place
- **Stable param IDs** — `{tag, row, col}` tuples so Adam's moment buffers survive across training steps
- **KV cache as map** — `%{layer_idx => %{keys: [..], values: [..]}}` for O(log n) layer access and pattern-matchable structure
- **`on_step` callback** — training loop is pure; IO happens only through a caller-provided callback
- **Threaded RNG** — `:rand.*_s` APIs ensure deterministic reproducibility by construction

## Work Packages

| WP  | Title                              | Status      |
| --- | ---------------------------------- | ----------- |
| 01  | Core Implementation                | Done        |
| 02  | Comprehensive Test Suite           | Done        |
| 03  | Elixir Idiom Audit & Remediation   | Done        |
| 04  | Socratic Code Review & Remediation | Done        |
| 05  | Public Repo Polish                 | Done        |
| 06  | Rich Moduledoc Text                | Not Started |
| 07  | LiveBook Walkthrough               | Not Started |

## Related Steel Threads

- None (first steel thread)
