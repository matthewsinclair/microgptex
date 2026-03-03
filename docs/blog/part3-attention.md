# How Tokens Talk to Each Other

## Attention — Part 3 of Building GPT from Scratch

Self-attention is the mechanism that lets a GPT "look back" at previous tokens when predicting the next one. It's the key innovation of the transformer architecture — and it's simpler than you might think.

In [Part 1](part1-autograd.md), I built the autograd engine. In [Part 2](part2-model.md), I built the tokenizer, the math building blocks, and the model structure — embeddings, weight matrices, and a forward pass that processes one token at a time.

But I glossed over the most interesting part: what happens inside the transformer block? When the forward pass calls `attn_block/5`, what actually happens?

That's what this post is about.

## Why attention matters

Without attention, each token is processed independently. The model sees "b" at position 2, applies some weight matrices, and produces logits. It has no way to know what came before — whether the sequence is "rob" or "job" or "Bob." The model has no memory.

With attention, each token can "look at" all previous tokens and decide which ones are relevant. When predicting the letter after "qu," the model needs to know that "q" came before. When deciding whether a name is ending, it needs to see how long the sequence has been so far.

Attention provides this by computing a weighted average of all previous positions, where the weights are learned. The model decides _what_ to attend to by learning the projection matrices.

## Query, Key, Value: three projections

The attention mechanism starts with three linear projections of the input vector `x`:

```elixir
q = Math.linear(x, layer.attn_wq)   # query: "what am I looking for?"
k = Math.linear(x, layer.attn_wk)   # key:   "what do I contain?"
v = Math.linear(x, layer.attn_wv)   # value: "what do I offer?"
```

These three vectors serve different roles:

- **Query (q)** — represents the current token's question: "which previous tokens are relevant to me?"
- **Key (k)** — represents what each position advertises about itself: "here's what I contain"
- **Value (v)** — represents the information each position offers: "if you attend to me, here's what you get"

All three are the same shape (`n_embd`-dimensional vectors) produced by different weight matrices applied to the same input. The model learns these matrices during training, which means it learns _what_ to look for, _what_ to advertise, and _what_ to offer.

## Scaled dot-product attention

The core of attention is a similarity search: compare the current query against all cached keys, then use the resulting scores to weight the values.

Step by step:

**1. Score each position.** Compute the dot product of the query with each key. Positions whose keys are similar to the query get high scores:

```elixir
attn_logits =
  Enum.map(cached_keys, fn k_t ->
    V.divide(Math.dot(q_h, k_t), scale)
  end)
```

**2. Normalize to probabilities.** Apply softmax to convert scores into a probability distribution — attention weights that sum to 1.0:

```elixir
attn_weights = Math.softmax(attn_logits)
```

**3. Weighted sum of values.** Multiply each value vector by its attention weight and sum. Positions with high attention weights contribute more to the output:

```elixir
# For each dimension j in the output:
Enum.zip(attn_weights, v_h)
|> Enum.map(fn {w, v_t} -> V.mul(w, Enum.at(v_t, j)) end)
|> V.sum()
```

The result is a vector that blends information from all previous positions, weighted by relevance to the current query.

### The scaling factor

The dot products are divided by `sqrt(head_dim)` before softmax. Why?

```elixir
scale = :math.sqrt(model.head_dim * 1.0)
```

Without scaling, the dot products grow in magnitude as the dimension increases (more terms being summed). Large dot products push softmax into its saturated regime — the output becomes nearly one-hot, with almost all the attention on a single position. This makes gradients very small (softmax is nearly flat at the extremes), which slows or stalls training.

Dividing by `sqrt(d)` keeps the variance of the dot products roughly constant regardless of dimension, ensuring softmax operates in its useful range.

## Multi-head attention

A single attention computation can only focus on one thing at a time. Multi-head attention runs several attention computations in parallel, each on a different slice of the embedding:

```elixir
x_attn =
  Enum.flat_map(0..(model.n_head - 1), fn h ->
    hs = h * model.head_dim

    # Slice q, k, v for this head
    q_h = Math.slice(q, hs, model.head_dim)
    k_h = Enum.map(cached_keys, &Math.slice(&1, hs, model.head_dim))
    v_h = Enum.map(cached_values, &Math.slice(&1, hs, model.head_dim))

    # Scaled dot-product attention for this head
    attn_logits = Enum.map(k_h, fn k_t ->
      V.divide(Math.dot(q_h, k_t), scale)
    end)
    attn_weights = Math.softmax(attn_logits)

    # Weighted sum of values
    weighted_sum(attn_weights, v_h, model.head_dim)
  end)
```

With `n_embd = 8` and `n_head = 2`, each head operates on a 4-dimensional slice. Head 0 gets dimensions 0-3, Head 1 gets dimensions 4-7. Each head independently computes its own attention pattern — one head might learn to look at the previous consonant, another might learn to look at vowel patterns. The model figures out the division of labor during training.

After all heads compute their outputs, the results are concatenated back into a full `n_embd`-dimensional vector and passed through one more projection:

```elixir
x =
  x_attn
  |> Math.linear(layer.attn_wo)
  |> Math.add_vec(x_residual)
```

The `Wo` projection mixes information across heads — it's where the model combines the different perspectives into a single representation.

## The KV cache: remembering past positions

The model processes one token at a time. When it processes position 3, it needs to attend to positions 0, 1, and 2. But those tokens have already been processed — their key and value vectors were computed during earlier forward passes.

The KV cache stores these. Each forward pass computes `k` and `v` for the current position and prepends them to the cache:

```elixir
layer_cache = Map.fetch!(kv_cache, li)

layer_cache = %{
  layer_cache
  | keys: [k | layer_cache.keys],
    values: [v | layer_cache.values]
}

kv_cache = Map.put(kv_cache, li, layer_cache)
```

The cache grows by one entry per position:

```
After position 0:  keys: [K₀],           values: [V₀]
After position 1:  keys: [K₁, K₀],       values: [V₁, V₀]
After position 2:  keys: [K₂, K₁, K₀],   values: [V₂, V₁, V₀]
```

When computing attention, the keys and values are reversed to restore chronological order:

```elixir
cached_keys = Enum.reverse(layer_cache.keys)
cached_values = Enum.reverse(layer_cache.values)
```

Prepend + reverse is the standard functional pattern for building a list incrementally. Prepend is O(1) (just a cons cell), and the reverse happens once per attention computation.

The cache is structured as a map keyed by layer index: `%{0 => %{keys: [...], values: [...]}, 1 => %{keys: [...], values: [...]}}`. Each transformer layer has its own cache because each layer's Q/K/V projections produce different representations.

During training, the cache is threaded through `Enum.reduce` over all positions in the document — it grows as each position is processed. During generation, the same cache persists across multiple calls to `gpt/4`, one per generated token. This avoids recomputing attention over past positions, which is the whole point of the KV cache optimization.

## Residual connections and normalization

The attention output doesn't replace the input — it's _added_ to it:

```elixir
x = Math.add_vec(attention_output, x_residual)
```

This is a residual connection: `x + f(x)`. The input passes through both the attention computation _and_ a direct shortcut. Why?

Residual connections solve the vanishing gradient problem in deep networks. Without them, gradients must flow back through every layer during backpropagation. Each layer's operations can shrink the gradients, and after enough layers, they become too small to drive learning. The shortcut gives gradients a direct path — even if the attention computation zeroes out the gradient, the residual path preserves it.

In practice, residual connections are why deep networks can train at all.

The other stabilizing mechanism is RMSNorm, applied before both the attention and MLP blocks:

```elixir
# In attn_block:
x = Math.rmsnorm(x)
q = Math.linear(x, layer.attn_wq)
# ...

# In mlp_block:
x
|> Math.rmsnorm()
|> Math.linear(layer.mlp_fc1)
# ...
```

This is "pre-norm" style — normalize before the transformation, not after. It keeps activation magnitudes in a stable range, preventing the kind of blow-up that makes training diverge.

## The complete transformer block

Putting it all together, each transformer block does:

```
input x
  │
  ├── save as x_residual
  │
  ├── rmsnorm
  ├── Q/K/V projections
  ├── multi-head attention (with KV cache)
  ├── output projection (Wo)
  ├── + x_residual  ←── residual connection
  │
  ├── save as x_residual
  │
  ├── rmsnorm
  ├── fc1 (expand: n_embd → 4*n_embd)
  ├── relu
  ├── fc2 (contract: 4*n_embd → n_embd)
  ├── + x_residual  ←── residual connection
  │
  output x
```

The MLP block (`mlp_block/2`) is particularly clean in Elixir — a pure pipeline:

```elixir
defp mlp_block(x, layer) do
  x_residual = x

  x
  |> Math.rmsnorm()
  |> Math.linear(layer.mlp_fc1)
  |> Math.relu_vec()
  |> Math.linear(layer.mlp_fc2)
  |> Math.add_vec(x_residual)
end
```

Five operations, each feeding its output to the next, with a residual connection at the end. The `fc1` layer expands the representation to 4x the embedding dimension, ReLU introduces nonlinearity, and `fc2` contracts back. This expansion-contraction pattern gives the model more capacity to transform the representation within each block.

Multiple transformer blocks stack on top of each other. Each one refines the representation further — the output of block 0 is the input to block 1, and so on. In MicroGPTEx's default config there's just one layer, but the architecture supports any number.

## What I've covered

The full attention mechanism, broken down:

- **Q/K/V projections** — three different views of the same input, learned during training
- **Scaled dot-product attention** — similarity search via dot products, normalized by softmax, with `sqrt(d)` scaling for numerical stability
- **Multi-head attention** — parallel attention on different slices of the embedding, concatenated and projected
- **KV cache** — stores past keys and values to avoid redundant computation, threaded through the forward pass as an immutable map
- **Residual connections** — `x + f(x)` shortcuts that keep gradients flowing through deep networks
- **The complete transformer block** — attention + MLP, each with pre-norm and residual connections

All of this is built from the six math operations I described in Part 2, operating on `Value` nodes from Part 1. The autograd engine handles gradient computation through the entire stack automatically.

## What's next

In [Part 4: "Learning and Dreaming"](part4-training.md), I'll make the model learn. I'll cover cross-entropy loss (measuring how wrong the model is), the Adam optimizer (adjusting weights intelligently), the training loop (a single `Enum.reduce`), and autoregressive sampling (generating text one token at a time). It's where random noise becomes understanding — and understanding becomes creativity.
