# MicroGPTEx

A functional, pedagogical GPT trainer in Elixir — a faithful translation of
[Andrej Karpathy's MicroGPT](https://github.com/karpathy/microgpt) demonstrating
autograd, multi-head attention, Adam optimization, and autoregressive text generation
using only scalar operations. Zero external dependencies.

For real work, use [Nx](https://github.com/elixir-nx/nx) + [EXLA](https://github.com/elixir-nx/nx/tree/main/exla).
This is the pedagogical version.

## What This Is

MicroGPTEx implements the complete GPT training algorithm in idiomatic Elixir:

- **Reverse-mode autograd** on scalar values (the `Value` module)
- **Multi-head self-attention** with KV caching
- **Adam optimizer** with bias correction
- **Character-level tokenization** with BOS framing
- **Temperature-controlled sampling** for text generation
- **RMSNorm** layer normalization

The entire algorithm is expressed as pure functions with explicitly threaded state —
no process dictionaries, no ETS, no mutation. The only IO happens in the top-level
`Microgptex` module.

## Architecture

Nine modules in a single file, ordered bottom-up by dependency:

```
Microgptex.RNG        — Pure threaded RNG (:rand.*_s APIs, Box-Muller, Fisher-Yates)
Microgptex.Value      — Autograd scalar node (forward ops + reverse-mode backward)
Microgptex.Tokenizer  — Character-level tokenizer with BOS token
Microgptex.Math       — Vector/matrix ops on Value lists (dot, linear, softmax, rmsnorm)
Microgptex.Model      — GPT-2 model: init, forward pass, params, update_params
Microgptex.Adam       — Adam optimizer (pure state-in/state-out)
Microgptex.Trainer    — Loss computation + training loop
Microgptex.Sampler    — Temperature-controlled autoregressive sampling
Microgptex            — Top-level API: config loading, data download, run/1
```

## Quick Start

```bash
mix deps.get
mix compile
```

Then in IEx:

```elixir
iex -S mix
Microgptex.run()
```

This will:

1. Download the [names dataset](https://github.com/karpathy/makemore) (~1MB, cached)
2. Train a tiny GPT on character-level name generation (1000 steps)
3. Generate 20 new hallucinated names

Default config is in `priv/config.yaml`. Override at runtime:

```elixir
Microgptex.run(steps: 100, temperature: 0.8)
```

## Configuration

| Parameter       | Default | Description                           |
| --------------- | ------- | ------------------------------------- |
| `n_layer`       | 1       | Number of transformer layers          |
| `n_embd`        | 16      | Embedding dimension                   |
| `block_size`    | 16      | Maximum sequence length               |
| `n_head`        | 4       | Number of attention heads             |
| `learning_rate` | 0.01    | Base learning rate (linearly decayed) |
| `beta1`         | 0.85    | Adam first moment decay               |
| `beta2`         | 0.99    | Adam second moment decay              |
| `steps`         | 1000    | Training steps                        |
| `temperature`   | 0.5     | Sampling temperature                  |
| `num_samples`   | 20      | Number of names to generate           |

## Tests

```bash
mix test
```

Behavioral tests covering autograd correctness, gradient backpropagation, softmax/RMSNorm
numerics, tokenizer round-trips, model forward pass determinism, Adam updates, training
convergence, and sampling determinism. Every test asserts concrete expected values.

## How It Works

### Immutable Autograd

Python's micrograd mutates `.grad` fields in-place during `backward()`.
MicroGPTEx's `Value.backward/1` returns a `%{id => gradient}` map — an immutable
data structure built by folding over the topologically sorted computation graph.
Fan-out (same value used multiple times) is handled by `Map.update/4`.

### Threaded State

All state — RNG, KV cache, optimizer moments — flows explicitly through function
arguments and return values. Every function is `(state_in) -> {result, state_out}`.
Same seed always produces the same training run.

### The Training Loop

```
for each step:
  doc = pick training document
  loss = cross_entropy(model, doc)      # forward pass, builds computation graph
  grads = Value.backward(loss)          # reverse pass, produces gradient map
  {opt, updates} = Adam.step(opt, params, grads, lr)
  model = Model.update_params(model, updates)
```

## Credits

Based on [Andrej Karpathy's MicroGPT](https://github.com/karpathy/microgpt)
and the [growingswe walkthrough](https://growingswe.com/blog/microgpt).

## License

MIT
