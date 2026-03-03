# Learning and Dreaming

## Training and Generation — Part 4 of Building GPT from Scratch

The model has structure but no knowledge — every weight is random noise. Training is the process that turns noise into understanding. Generation is where that understanding becomes creative.

In [Part 1](part1-autograd.md), I built the autograd engine. In [Part 2](part2-model.md), I built tokenization, math building blocks, and the model architecture. In [Part 3](part3-attention.md), I dug into the attention mechanism — how tokens look at each other, the KV cache, residual connections.

Now I make it learn.

## Cross-entropy loss: measuring surprise

Before the model can learn, I need a way to measure how wrong it is. That's the loss function.

For each position in a training document, the model predicts a probability distribution over the vocabulary. The cross-entropy loss measures how surprised the model is by the actual next character:

```
loss = -log(p(correct_token))
```

The intuition is simple:

- If the model assigns probability **1.0** to the right answer, loss is `-log(1.0) = 0`. Perfect.
- If the model assigns probability **1/27** (uniform over vocabulary), loss is `-log(1/27) = ln(27) ≈ 3.3`. Just guessing.
- If the model assigns probability **0.01** to the right answer, loss is `-log(0.01) = 4.6`. Confidently wrong.

Training drives the loss down by adjusting weights so the model assigns higher probability to the tokens that actually follow in the training data.

Here's `loss_for_doc/3` — it feeds each token through the model and accumulates the loss:

```elixir
def loss_for_doc(model, tokenizer, doc) do
  tokens = Tokenizer.encode(tokenizer, doc)
  kv_cache = Model.empty_kv_cache(model)

  token_pairs =
    tokens
    |> Enum.zip(tl(tokens))
    |> Enum.with_index()
    |> Enum.take(min(model.block_size, length(tokens) - 1))

  {losses, _kv_cache} =
    Enum.reduce(token_pairs, {[], kv_cache}, fn {{token_id, target_id}, pos_id},
                                                {acc, kv_cache} ->
      {logits, kv_cache} = Model.gpt(model, token_id, pos_id, kv_cache)
      probs = Math.softmax(logits)
      loss_t = V.neg(V.log(Enum.at(probs, target_id)))

      {[loss_t | acc], kv_cache}
    end)

  V.mean(Enum.reverse(losses))
end
```

For each (input, target) pair, the model does a forward pass, converts logits to probabilities via softmax, and measures `-log(p(target))`. The final loss is the mean across all positions.

The critical thing: this entire computation builds a `Value` computation graph. The loss node at the root is connected — through softmax, through linear projections, through attention, through embeddings — to every weight in the model. That graph is what `backward/1` will traverse to compute gradients.

## One gradient step

Each training step follows the same pattern — forward, backward, update:

```
pick training doc
    → forward pass (builds computation graph)
    → compute loss
    → Value.backward (gradient map)
    → Adam.step (compute updates)
    → Model.update_params (apply updates)
    → repeat
```

Here's what one step looks like in code:

```elixir
# 1. Forward pass — builds the computation graph
loss = Trainer.loss_for_doc(model, tok, "ab")

# 2. Backward pass — compute gradients for all parameters
grads = V.backward(loss)

# 3. Adam optimizer — compute parameter updates
params = Model.params(model)
opt = Adam.init(0.01, 0.85, 0.99, 1.0e-8)
{opt, updates} = Adam.step(opt, params, grads, 0.01)

# 4. Apply updates to the model
model = Model.update_params(model, updates)
```

The gradient map `grads` is `%{id => gradient}` — one entry per node in the computation graph. For each parameter, it says: "if you increase this weight by a tiny amount, the loss changes by this much." The optimizer uses these gradients to decide how to adjust each weight.

## The Adam optimizer

Plain gradient descent applies `weight -= lr * gradient` to every parameter uniformly. Adam is smarter — it tracks two statistics per parameter:

- **Momentum (m)** — exponential moving average of the gradient. Smooths out noise and maintains direction. If the gradient consistently points left, momentum builds up; if it oscillates, momentum stays small.
- **Velocity (v)** — exponential moving average of the squared gradient. Tracks magnitude. Parameters with large, noisy gradients get smaller effective learning rates.

The update formula:

```
m_t = β₁ · m_{t-1} + (1 - β₁) · g           # smoothed gradient
v_t = β₂ · v_{t-1} + (1 - β₂) · g²          # smoothed squared gradient
m̂   = m_t / (1 - β₁ᵗ)                        # bias correction
v̂   = v_t / (1 - β₂ᵗ)                        # bias correction
θ   = θ - lr · m̂ / (√v̂ + ε)                  # parameter update
```

Bias correction compensates for the zero initialization of `m` and `v`. Without it, the first few updates would be artificially small because the moving averages haven't warmed up yet.

Here's the core of `Adam.step/5`:

```elixir
def step(%Adam{} = opt, params, grads_by_id, lr_t) do
  t = opt.t + 1

  {m2, v2, updated} =
    Enum.reduce(params, {opt.m, opt.v, %{}}, fn %Value{id: id, data: data},
                                                {m, v, upd} ->
      g = Map.get(grads_by_id, id, 0.0)

      m_t = opt.beta1 * Map.get(m, id, 0.0) + (1.0 - opt.beta1) * g
      v_t = opt.beta2 * Map.get(v, id, 0.0) + (1.0 - opt.beta2) * (g * g)

      m_hat = m_t / (1.0 - :math.pow(opt.beta1, t * 1.0))
      v_hat = v_t / (1.0 - :math.pow(opt.beta2, t * 1.0))

      new_data = data - lr_t * m_hat / (:math.sqrt(v_hat) + opt.eps)

      {Map.put(m, id, m_t), Map.put(v, id, v_t), Map.put(upd, id, new_data)}
    end)

  {%Adam{opt | m: m2, v: v2, t: t}, updated}
end
```

Notice the return type: `{new_optimizer_state, %{param_id => new_value}}`. The optimizer doesn't touch the model — it returns a map of updates, and the caller applies them explicitly via `Model.update_params/2`.

In Python, `optimizer.step()` mutates model parameters in-place through shared object references. The optimizer holds pointers to the same parameter tensors as the model, so it can write directly. In Elixir, there's no shared mutable state. The optimizer and model communicate through ID-keyed maps — the same `{"wte", 2, 5}` ID links a parameter through the entire round-trip.

## The params round-trip

This round-trip is the heartbeat of training:

```
Model (nested maps of Values)
    → Model.params/1 — flatten into [%Value{id: {tag, r, c}}]
    → V.backward/1 — produce %{id => gradient}
    → Adam.step/5 — produce %{id => new_data}
    → Model.update_params/2 — walk the model tree, apply updates
    → Updated Model
    → (repeat)
```

The `{tag, row, col}` ID scheme is what makes this work. Each parameter has a deterministic, stable identity:

- `{"wte", 2, 5}` — token embedding matrix, row 2, column 5
- `{"layer0.attn_wq", 7, 3}` — layer 0 attention query weights, row 7, column 3

These IDs persist across training steps. Adam's momentum and velocity buffers (`m` and `v` maps) accumulate statistics keyed by these same IDs. If the IDs changed between steps, the optimizer would lose its memory and revert to plain gradient descent.

In Python, this persistence comes for free from shared mutable references — the optimizer holds the same Python objects as the model. In Elixir, it's explicit in the ID scheme. More verbose, but there's no ambiguity about which parameter is which.

## The training loop

The entire training loop is a single `Enum.reduce`:

```elixir
def train(config) do
  opt = Adam.init(lr, beta1, beta2, eps_adam)

  {model, opt} =
    Enum.reduce(0..(steps - 1), {model, opt}, fn step, {model, opt} ->
      doc = Enum.at(docs, rem(step, n_docs))
      loss = loss_for_doc(model, tokenizer, doc)
      grads = V.backward(loss)

      lr_t = lr * (1.0 - step / max(steps, 1))
      params = Model.params(model)
      {opt, updated_by_id} = Adam.step(opt, params, grads, lr_t)
      model = Model.update_params(model, updated_by_id)

      on_step.(step, loss.data)
      {model, opt}
    end)

  {model, opt}
end
```

Each iteration: pick a document, compute loss, backward pass, optimizer step, apply updates. The accumulator carries `{model, opt}` — both are updated at each step and threaded to the next.

The learning rate decays linearly from `lr` to 0 over the training run: `lr_t = lr * (1 - step/steps)`. This helps the model settle into a good minimum rather than bouncing around with large updates late in training.

The `on_step` callback is the IO boundary. The training loop itself is pure — given the same inputs, it always produces the same outputs. IO (printing progress, updating charts) happens only through the callback the caller provides. In tests, pass `fn _step, _loss -> :ok end` for silent execution. In production, pass a callback that prints to stdout. In Livebook, pass one that pushes to a VegaLite chart.

This "pure core, impure shell" pattern means the training loop can be tested by asserting on the model state directly — no stdout capturing, no mocking, no test fixtures.

## Autoregressive sampling

After training, I generate text by sampling from the model's predictions one token at a time. This is "autoregressive" generation: each generated token becomes the input for the next step.

The algorithm:

1. Start with BOS (beginning-of-sequence)
2. Forward pass → logits over vocabulary
3. Scale logits by temperature
4. Softmax → probability distribution
5. Sample one token from that distribution
6. If the token is BOS (end), stop. Otherwise, feed it back and repeat.

```
BOS → forward → softmax → sample 'e' → forward → softmax → sample 'm' → ... → sample BOS → stop
```

The KV cache carries forward between steps. Each forward pass only processes the newest token while reusing cached attention from all previous positions. This is why the model processes one token at a time — it's the natural pattern for autoregressive generation.

Temperature controls the sampling distribution (as I described in Part 2):

- **Low temperature** (0.1-0.5) — sharp distribution, picks the most likely token almost every time. Produces conservative, repetitive output.
- **High temperature** (1.5-3.0) — flat distribution, gives unlikely tokens a real chance. Produces creative, noisy output.
- **Temperature = 1.0** — the learned distribution as-is.

An Elixir detail worth noting: temperature scaling uses `V.scale_data/2` instead of `V.divide/2`. During inference, I don't need gradients — the model isn't learning, just generating. `scale_data` modifies the `.data` field directly without building autograd nodes, avoiding the overhead of graph construction that's only needed during training.

### Multi-clause termination

The sampler's exit conditions are expressed as function clauses rather than `if/break` statements:

```elixir
# Guard: continue while within block_size
defp generate_loop(model, tokenizer, rng, inv_temp, pos_id, token_id, kv_cache, chars)
     when pos_id < model.block_size do
  # ... forward pass, sample next token ...
  continue_or_stop(next_token, ...)
end

# Fallthrough: block_size reached, stop
defp generate_loop(_model, _tokenizer, rng, _inv_temp, _pos_id, _token_id, _kv_cache, chars) do
  {chars |> Enum.reverse() |> Enum.join(), rng}
end

# BOS token emitted: stop
defp continue_or_stop(bos, _model, %{bos: bos} = _tok, rng, _inv_t, _pos, _kv, chars) do
  {chars |> Enum.reverse() |> Enum.join(), rng}
end

# Any other token: continue generating
defp continue_or_stop(token, model, tokenizer, rng, inv_temp, pos_id, kv_cache, chars) do
  ch = Map.fetch!(tokenizer.id_to_char, token)
  generate_loop(model, tokenizer, rng, inv_temp, pos_id + 1, token, kv_cache, [ch | chars])
end
```

All exit conditions are visible from the function heads: the guard `when pos_id < model.block_size` for the length limit, and the pattern `continue_or_stop(bos, _, %{bos: bos}, ...)` for BOS detection. A reader can enumerate every way generation can stop without tracing through loop bodies.

## The full pipeline

Here's what happens when you put it all together — train a small model on a few names and generate new ones:

```elixir
docs = ["emma", "olivia", "liam", "noah", "ava", "sophia", "james", "william"]
tok = Tokenizer.build(docs)

{model, rng} = Model.init(%{
  n_layer: 1, n_embd: 16, block_size: 16,
  n_head: 4, vocab_size: tok.vocab_size,
  std: 0.08, seed: 42
})

{trained, _opt} = Trainer.train(%{
  docs: docs, tokenizer: tok, model: model,
  steps: 50, learning_rate: 0.01,
  beta1: 0.85, beta2: 0.99, eps_adam: 1.0e-8,
  on_step: fn step, loss ->
    if rem(step, 10) == 0, do: IO.puts("step #{step + 1}: loss=#{loss}")
  end
})

{samples, _rng} = Sampler.generate(trained, tok, rng, 10, 0.8)
Enum.each(samples, &IO.puts/1)
```

With only 50 steps on 8 names, the model won't produce great results. But you can see it learning — the loss drops from near the random baseline down toward something meaningful, and the generated output starts to show character patterns from the training data. Increase to 200 or 500 steps and the output improves noticeably. With the full names dataset (32K names) and 1000 steps, it generates plausible-sounding new names.

## What Elixir reveals

This isn't just a port from Python to Elixir. Karpathy's original is remarkably clean — that clarity is what made a faithful functional translation possible. But the translation reveals things about the algorithm's structure that the imperative version obscures.

**Immutable autograd.** Python's `backward()` mutates `.grad` fields via `+=` — the gradient computation is a side effect. Elixir's `backward/1` returns a gradient map — an immutable data structure you can hold, compare, and pass to multiple consumers. The fan-out accumulation is explicit in `Map.update/4`.

**Threaded state.** RNG state, KV cache, optimizer moments — everything flows through function arguments. Same seed, same training run. Not because you remembered to call `random.seed()`, but because the types enforce it.

**Pure core, impure shell.** Eight of nine modules are pure functions. IO only happens in the top-level `Microgptex` module, and the training loop's IO boundary is an `on_step` callback. The entire GPT algorithm — autograd, attention, optimization, sampling — is testable without mocking, capturing stdout, or managing fixtures.

**Pattern-matched dispatch.** Python uses `if/break` for loop termination and `isinstance` for type dispatch. Elixir uses function clauses and guards: all exit conditions are readable from the function heads. The sampler's two stop conditions (BOS emitted, block_size reached) are two function clauses, not nested conditionals.

**The params round-trip.** Python links parameters through shared mutable object references. Elixir links them through stable `{tag, row, col}` IDs in keyed maps. The data flow — model → flatten → gradients → optimizer → updates → model — is explicit at every step.

The same math, the same algorithm, the same results. But the data flow is visible in the code itself. When I got confused about how backpropagation works, I could inspect the gradient map. When I wondered whether the optimizer was updating the right parameters, I could check the ID keys. When a test failed, I could trace the entire computation because there was no hidden state to account for.

That's the pedagogical payoff: a functional implementation doesn't just _work_ — it _shows its work_.

## Try it yourself

The full source is in a single file: [`lib/microgptex.ex`](https://github.com/TODO/microgptex). Nine modules, ~1500 lines, zero external dependencies.

There are also two [Livebook](https://livebook.dev/) notebooks for hands-on exploration:

- **Code walkthrough** — "MicroGPTEx: How GPT Works, from Scratch." Step through the algorithm chapter by chapter, with executable code cells and Mermaid diagrams.
  <br/>[![Run in Livebook](https://livebook.dev/badge/v1/blue.svg)](https://livebook.dev/run?url=https://github.com/matthewsinclair/microgptex/blob/main/notebooks/walkthrough.livemd)

- **Interactive explorations** — "MicroGPTEx: Interactive Explorations." Drag sliders to reshape softmax distributions, watch training loss curves update in real time, explore attention heatmaps across heads.
  <br/>[![Run in Livebook](https://livebook.dev/badge/v1/blue.svg)](https://livebook.dev/run?url=https://github.com/matthewsinclair/microgptex/blob/main/notebooks/interactive.livemd)

None of this would exist without [Andrej Karpathy's MicroGPT](https://karpathy.github.io/2026/02/12/microgpt/) — a masterclass in fitting a complete GPT into minimal code. The interactive walkthrough at [growingswe.com](https://growingswe.com/blog/microgpt) is equally worth reading; it's what convinced me this algorithm could be understood by working through it hands-on, and directly inspired the Livebook notebooks.
