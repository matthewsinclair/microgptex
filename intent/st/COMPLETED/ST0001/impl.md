# Implementation — ST0001: MicroGPTEx

## As-Built Summary

Single-file implementation (`lib/microgptex.ex`) with 9 modules ordered bottom-up by dependency. Tests cover autograd correctness, gradient backpropagation, softmax/RMSNorm numerics, tokenizer round-trips, model forward pass determinism, Adam update correctness, training convergence, and sampling determinism. Config lives in `priv/config.yaml` (simple key-value YAML, parsed without dependencies).

## Implementation Notes

### Ported from Karpathy's MicroGPT with Fixes

The primary reference is [Karpathy's MicroGPT](https://github.com/karpathy/microgpt) and the [growingswe walkthrough](https://growingswe.com/blog/microgpt). Key improvements in the Elixir translation:

1. **Bug fix in cumulative softmax** — Original only matches on first iteration, crashes on subsequent elements. Fixed with proper `Enum.scan/3`.
2. **Bug fix in `generate_one`** — Original's `reduce_while` has fragile return shape. Normalized to always return `{name_string, rng}`.
3. **KV cache as map** — Original uses list indexing. Changed to map for O(log n) access and pattern matching.
4. **`gpt/4` returns `{logits, kv_cache}`** — Original doesn't return the updated cache, losing it. Fixed so the caller threads it.
5. **Trainer is pure** — Original has IO inside the training loop. Moved to `on_step` callback pattern.

### Config Parsing

Simple key-value YAML parser — no library needed. Supports `key: value` lines where values are strings, integers, or floats. Comments and blank lines are ignored. Implemented as `parse_value/1` with pattern matching on string structure.

### Data Download

`Microgptex.run/1` downloads training data on first run via `:httpc` (part of OTP, no deps). Cached to `priv/data/input.txt`. HTTP errors produce descriptive error messages with status codes.

## As-Built Code Structure

### Module Dependency Graph

```
Microgptex.RNG      (standalone — pure threaded RNG)
      ↓
Microgptex.Value    (uses RNG for nothing — standalone autograd)
      ↓
Microgptex.Tokenizer (standalone — character-level encoding)
      ↓
Microgptex.Math     (uses Value — vector/matrix ops)
      ↓
Microgptex.Model    (uses RNG, Value, Math — GPT-2 model)
      ↓
Microgptex.Adam     (uses Value — optimizer)
      ↓
Microgptex.Trainer  (uses Value, Tokenizer, Math, Model, Adam — training loop)
      ↓
Microgptex.Sampler  (uses Value, RNG, Math, Model — sampling)
      ↓
Microgptex          (uses all — top-level API + IO boundary)
```

### Key Functions per Module

**RNG**: `seed/1`, `uniform01/1`, `uniform_int/2`, `normal/3`, `shuffle/2`

**Value**: `leaf/2`, `add/2`, `mul/2`, `pow/2`, `log/1`, `exp/1`, `relu/1`, `neg/1`, `sub/2`, `divide/2`, `sum/1`, `mean/1`, `scale_data/2`, `backward/1`

**Tokenizer**: `new/1`, `encode/2`, `decode/2`, `vocab_size` (derived)

**Math**: `dot/2`, `linear/3`, `softmax/1`, `rmsnorm/1`, `add_vec/2`, `relu_vec/1`, `slice/3`

**Model**: `init/2` (builds GPT-2 struct), `gpt/4` (forward pass), `params/1` (flat param list), `update_params/2` (apply Adam updates), `empty_kv_cache/1`

**Adam**: `new/1`, `step/5` (one optimizer step with bias correction)

**Trainer**: `loss_for_doc/3`, `train/2` (pure training loop with `on_step` callback)

**Sampler**: `generate/2` (produce N samples via autoregressive decoding)

**Microgptex**: `run/1` (top-level entry point with IO)

## Code Examples

### Training Loop (Pure)

```elixir
Enum.reduce(0..(steps - 1), {model, opt, rng}, fn step, {model, opt, rng} ->
  {doc, rng} = pick_doc(docs, step, rng)
  {loss, grads} = loss_and_grads(model, tokenizer, doc)
  lr_t = lr * (1.0 - step / max(steps, 1))
  {opt, updates} = Adam.step(opt, params, grads, lr_t)
  model = Model.update_params(model, updates)
  on_step.(step, loss.data)
  {model, opt, rng}
end)
```

### Backward Pass (Immutable Gradient Map)

```elixir
def backward(%Value{} = root) do
  topo = topo_sort(root)
  grads = %{root.id => 1.0}

  Enum.reduce(topo, grads, fn node, grads ->
    node_grad = Map.get(grads, node.id, 0.0)
    Enum.zip(node.children, node.local_grads)
    |> Enum.reduce(grads, fn {child, local}, acc ->
      Map.update(acc, child.id, local * node_grad, &(&1 + local * node_grad))
    end)
  end)
end
```

### Autoregressive Sampling (Explicit Recursion)

```elixir
defp generate_loop(model, tok, kv, pos, tokens, max_len, temp, rng) do
  if pos >= max_len do
    {Enum.reverse(tokens), rng}
  else
    last_token = hd(tokens)
    {logits, kv} = Model.gpt(model, last_token, pos, kv)
    {token_id, rng} = sample_token(logits, temp, rng)

    if token_id == tok.bos_id do
      {Enum.reverse(tokens), rng}
    else
      generate_loop(model, tok, kv, pos + 1, [token_id | tokens], max_len, temp, rng)
    end
  end
end
```

## Challenges & Solutions

- **Gradient fan-out**: When a value is used multiple times (e.g., `a * a`), gradients must accumulate. Solved with `Map.update/4` which adds to existing gradient.
- **Topological sort order**: DFS with prepend (`[v | topo]`) produces root-first order. Backward iterates directly — root processed first guarantees gradients flow correctly. Initially had a bug with `Enum.reverse` that was caught by tests.
- **Softmax numerical stability**: Subtracting `max(.data)` as a plain float constant (not a Value node) prevents underflow without adding unnecessary computation graph nodes.
- **Deterministic sampling**: All randomness flows through the threaded RNG state, making test results reproducible with a fixed seed.
- **Vocab size mismatch**: Model config `vocab_size` must exactly match tokenizer's `vocab_size` — enforced in tests by deriving model config from tokenizer.
- **`scale_data/2` for inference**: Sampling needs to divide logits by temperature but must not create autograd graph nodes. `V.scale_data/2` modifies only `.data`, leaving the graph untouched.
- **List building performance**: All `acc ++ [item]` patterns replaced with `[item | acc]` + `Enum.reverse/1` to avoid O(n^2) list concatenation.
