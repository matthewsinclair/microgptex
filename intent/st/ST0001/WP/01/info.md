---
verblock: "02 Mar 2026:v0.1: matts - Initial version"
wp_id: WP-01
title: "Core Implementation"
scope: Large
status: Done
---

# WP-01: Core Implementation

## Objective

Implement the complete GPT training algorithm in idiomatic Elixir as a single-file, 9-module library: reverse-mode autograd, multi-head self-attention, Adam optimizer, character-level tokenizer, and autoregressive sampler — all using only scalar operations with zero external dependencies.

## Deliverables

- `lib/microgptex.ex` — 9 modules ordered bottom-up by dependency, from pure foundations (RNG, Value) through the GPT algorithm (Model, Adam, Trainer) to the IO shell (Microgptex)
- `priv/config.yaml` — default hyperparameters parsed by a simple manual YAML parser (no deps)
- `priv/data/.gitkeep` — training data downloaded on demand via `:httpc` at runtime

## What Each Module Does and Why

- **RNG** — Pure threaded RNG using `:rand.*_s` APIs. Every function returns `{result, new_rng}`. Box-Muller for normal samples, Fisher-Yates for shuffling. Threaded state guarantees deterministic reproducibility — same seed, same training run.
- **Value** — The autograd core. Each scalar operation creates a node storing its output, children, and local gradients. `backward/1` does a topological sort (root-first DFS) then folds through to build a `%{id => gradient}` map. `Map.update/4` handles fan-out accumulation.
- **Tokenizer** — Character-level encoding with BOS token for start/stop framing. Both `char_to_id` and `id_to_char` maps for O(1) encode and decode.
- **Math** — Vector/matrix operations over lists of Value nodes: dot product, linear transform, numerically-stable softmax (max subtracted as constant, not differentiated), RMSNorm.
- **Model** — GPT-2 architecture: token + position embeddings, RMSNorm, multi-head self-attention with KV cache, MLP with ReLU, residual connections. Params have `{tag, row, col}` IDs for stable Adam tracking.
- **Adam** — Pure optimizer: takes current state, returns new state + update map. Bias-corrected first and second moments.
- **Trainer** — Cross-entropy loss via forward pass, gradient computation via `backward`, Adam updates. Pure — IO happens through an `on_step` callback the caller provides.
- **Sampler** — Temperature-controlled autoregressive sampling via explicit tail recursion. BOS token detection uses multi-clause dispatch (`continue_or_stop`).
- **Microgptex** — The IO boundary. Config loading, data download, training, sampling, and stdout output all happen here and nowhere else.

## Key Design Choices

- **Param IDs**: `{tag, row, col}` tuples for stable Adam moment tracking
- **Intermediate IDs**: `make_ref()` for cheap unique graph node IDs
- **IO boundary**: pure core with IO only in top-level `Microgptex.run/1`
- **KV cache**: map-based for O(log n) access and pattern matching
- **Threaded state**: all RNG/cache/optimizer state flows as `{result, new_state}`

## Acceptance Criteria

- [x] All 9 modules compile without warnings
- [x] Forward pass produces vocab_size logits
- [x] Backward pass computes correct gradients (verified by tests)
- [x] Training loop decreases loss over steps
- [x] Sampler generates deterministic output given seed
- [x] `priv/config.yaml` parsed correctly with manual YAML parser
