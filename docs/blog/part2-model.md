# From Letters to Logits

## Text, Math, and the Model — Part 2 of Building GPT from Scratch

A neural network can't read letters. It needs numbers — and a small set of mathematical operations to transform them. In this post, I build the bricks and assemble the GPT architecture.

In [Part 1](part1-autograd.md), I built the autograd engine: a `Value` struct that tracks every computation in a directed acyclic graph, and a `backward/1` function that computes gradients by walking that graph in reverse. I also built a threaded RNG that guarantees deterministic reproducibility by construction.

Now I need three more things before I can train a model:

1. **A tokenizer** — to convert text into numbers the model can process
2. **Math building blocks** — dot products, matrix multiplication, softmax, normalization
3. **A model** — the GPT architecture itself, with all its weight matrices and the forward pass

These map to three modules in MicroGPTEx's dependency chain:

```
RNG → Value → Tokenizer → Math → Model → ...
              ^^^^^^^^    ^^^^   ^^^^^
              this post covers these three
```

## Tokenization: converting text to numbers

The tokenizer maps characters to integers and back. It's character-level — each unique character in the training data gets its own ID, and a special BOS (beginning-of-sequence) token marks the start and end of each name.

```elixir
alias Microgptex.Tokenizer

tok = Tokenizer.build(["alice", "bob", "charlie"])

tok.uchars      #=> ["a", "b", "c", "e", "h", "i", "l", "o", "r"]
tok.vocab_size  #=> 10  (9 chars + 1 BOS)
tok.bos         #=> 9   (BOS gets the last ID)
```

`build/1` extracts every unique character, sorts them, assigns sequential IDs, and reserves the last slot for BOS. The result is a struct with two maps — `char_to_id` for encoding and `id_to_char` for decoding:

```elixir
tok.char_to_id  #=> %{"a" => 0, "b" => 1, "c" => 2, ...}
tok.id_to_char  #=> %{0 => "a", 1 => "b", 2 => "c", ...}
```

Keeping both directions as maps gives O(1) lookup in either direction. This is a useful Elixir pattern whenever you have a fixed, bidirectional mapping — build both maps once and use assertive `Map.fetch!/2` access thereafter.

### Encoding and the BOS token

Each training document gets wrapped with BOS at both ends:

```elixir
Tokenizer.encode(tok, "bob")  #=> [9, 1, 7, 1, 9]
#                                  ^  b  o  b  ^
#                                  BOS         BOS
```

The leading BOS says "start of name." The trailing BOS says "I'm done." The model learns both meanings from the same token — BOS at the start is the prompt, BOS at the end is the target that teaches the model to stop generating.

This encoding defines the model's task. At each position, predict the next token:

```
Position 0: BOS → b     "After start-of-name, predict 'b'"
Position 1: b   → o     "After 'b', predict 'o'"
Position 2: o   → b     "After 'o', predict 'b'"
Position 3: b   → BOS   "After 'b', predict end-of-name"
```

Every training example is a sequence of these (input, target) pairs. The model sees the input token and must predict the target. The loss measures how wrong it is.

### Why character-level?

Production models use subword [tokenizers](https://huggingface.co/docs/transformers/en/tokenizer_summary) ([BPE](https://medium.com/@dhiyaadli/bpe-vs-wordpiece-vs-sentencepiece-a-beginner-friendly-guide-to-subword-tokenization-8047b39d82e0), [SentencePiece](https://github.com/google/sentencepiece)) with 30K-100K tokens. I use character-level because the vocabulary is tiny (~27 tokens for lowercase English names), the embedding matrices stay small, and the algorithm is identical — only the vocabulary size changes. No tokenizer training step needed.

## Math building blocks

The GPT model is built from a small set of mathematical operations. Each one operates on lists of `Value` nodes, so gradients flow through everything automatically.

### Dot product

The dot product measures similarity between two vectors: `sum(w_i * x_i)`. In attention, it computes how much one position "attends to" another.

```elixir
alias Microgptex.Math

query = [V.leaf(1.0, :q1), V.leaf(0.0, :q2), V.leaf(1.0, :q3)]
key   = [V.leaf(1.0, :k1), V.leaf(1.0, :k2), V.leaf(0.0, :k3)]

score = Math.dot(query, key)
score.data  #=> 1.0  (1*1 + 0*1 + 1*0)
```

The implementation is three lines: zip, multiply, sum:

```elixir
def dot(ws, xs) do
  Enum.zip(ws, xs)
  |> Enum.map(fn {w, x} -> V.mul(w, x) end)
  |> V.sum()
end
```

Because every `V.mul` and `V.sum` builds autograd nodes, gradients flow backward through the dot product for free.

### Linear transform

Matrix-vector multiplication is the core operation of every neural network layer. Each row of the weight matrix produces one output element via a dot product:

```elixir
def linear(x, w) do
  Enum.map(w, fn w_row -> dot(w_row, x) end)
end
```

That's it. One line. Every projection in the model — embedding lookups, attention Q/K/V, MLP layers, the final output head — is a call to `Math.linear/2`.

### Softmax

Softmax converts a vector of raw scores (logits) into a probability distribution that sums to 1.0. The larger the input, the higher its probability:

```elixir
logits = [V.leaf(2.0, :a), V.leaf(5.0, :b), V.leaf(1.0, :c)]
probs = Math.softmax(logits)

# probs ≈ [0.0420, 0.9362, 0.0218]
# The largest logit (5.0) dominates the distribution
```

The implementation uses the standard numerical stability trick — subtract the max before exponentiating to prevent overflow:

```elixir
def softmax(logits) do
  max_val = logits |> Enum.map(& &1.data) |> Enum.max()
  exps = Enum.map(logits, fn v -> V.exp(V.sub(v, max_val)) end)
  total = V.sum(exps)
  Enum.map(exps, &V.divide(&1, total))
end
```

A subtle detail: `max_val` is extracted as a raw float (`.data`), not as a `Value` node. This means the subtraction is treated as a constant shift during differentiation. That's correct — shifting all logits by a constant doesn't change the softmax output or its gradient. But it prevents `exp` from blowing up on large values.

### Temperature

Temperature is a scaling trick applied _before_ softmax. Divide the logits by a temperature value:

- **Low temperature** (e.g. 0.1) — the distribution becomes sharper, peaking hard on the most likely token. The model becomes confident and repetitive.
- **High temperature** (e.g. 2.0) — the distribution flattens, giving lower-probability tokens more of a chance. The model becomes creative and noisy.
- **Temperature = 1.0** — the unmodified distribution, as the model learned it.

```
logits = [2.0, 5.0, 1.0]

temp=0.1: [0.0000, 1.0000, 0.0000]  ← almost deterministic
temp=0.5: [0.0003, 0.9994, 0.0003]  ← very confident
temp=1.0: [0.0420, 0.9362, 0.0218]  ← the learned distribution
temp=2.0: [0.1402, 0.6439, 0.2160]  ← more spread out
```

Temperature doesn't change the model — it changes how I interpret the model's output. The model always produces the same logits for the same input; temperature just reshapes the probability distribution before sampling.

### RMSNorm

RMSNorm normalizes a vector so its root-mean-square is approximately 1.0:

```
rmsnorm(x) = x / sqrt(mean(x^2) + eps)
```

This prevents activations from growing or shrinking uncontrollably during the forward pass. Without normalization, deep networks tend to have exploding or vanishing activations, which makes training unstable.

```elixir
def rmsnorm(x, eps \\ 1.0e-5) do
  ms = x |> Enum.map(fn xi -> V.mul(xi, xi) end) |> V.mean()
  scale = V.pow(V.add(ms, eps), -0.5)
  Enum.map(x, fn xi -> V.mul(xi, scale) end)
end
```

RMSNorm is a simpler alternative to LayerNorm — it skips the mean-centering step. Modern architectures like LLaMA use it because it's cheaper to compute and works just as well in practice.

### The complete toolkit

Six functions are all the math I need for a GPT:

| Function     | What it does           | Where it's used              |
| ------------ | ---------------------- | ---------------------------- |
| `dot/2`      | Dot product            | Attention scores             |
| `linear/2`   | Matrix-vector multiply | Every projection             |
| `softmax/1`  | Scores → probabilities | Attention weights, sampling  |
| `rmsnorm/1`  | Normalize magnitudes   | Before attention, before MLP |
| `add_vec/2`  | Element-wise addition  | Residual connections         |
| `relu_vec/1` | Element-wise ReLU      | MLP activation               |

Each one produces `Value` nodes, so the autograd engine from Part 1 handles all gradient computation automatically.

## The GPT model

Now I have all the pieces to build the actual model. A GPT is a stack of weight matrices organized into a specific architecture. `Model.init/1` creates them all with random values:

```elixir
alias Microgptex.{Model, RNG}

cfg = %{
  n_layer: 1,      # transformer layers
  n_embd: 8,       # embedding dimension
  block_size: 8,    # max sequence length
  n_head: 2,        # attention heads (head_dim = 8/2 = 4)
  vocab_size: 5,    # tokens in vocabulary
  std: 0.08,        # weight initialization std dev
  seed: 42
}

{model, _rng} = Model.init(cfg)
```

The model state is a nested map of `Value` leaf nodes — the learnable parameters. Here's what's inside:

### Where the parameters live

```
Token embeddings (wte):     vocab_size × n_embd     =  5 × 8  =   40
Position embeddings (wpe):  block_size × n_embd     =  8 × 8  =   64
Language model head:        vocab_size × n_embd     =  5 × 8  =   40

Per transformer layer:
  Attention Q projection:   n_embd × n_embd         =  8 × 8  =   64
  Attention K projection:   n_embd × n_embd         =  8 × 8  =   64
  Attention V projection:   n_embd × n_embd         =  8 × 8  =   64
  Attention O projection:   n_embd × n_embd         =  8 × 8  =   64
  MLP fc1:                  (4 × n_embd) × n_embd   = 32 × 8  =  256
  MLP fc2:                  n_embd × (4 × n_embd)   =  8 × 32 =  256

Total: 40 + 64 + 40 + 64*4 + 256 + 256 = 912 parameters
```

Every one of these 912 parameters is a `Value` leaf node with a stable `{tag, row, col}` ID:

```elixir
# A weight in the token embedding matrix, row 2, column 5:
%Value{data: 0.0312, id: {"wte", 2, 5}, children: [], local_grads: []}
```

These stable IDs are critical. The training loop needs to:

1. Flatten all parameters into a list (`Model.params/1`)
2. Run the forward pass and compute loss
3. Call `V.backward(loss)` to get a `%{id => gradient}` map
4. Feed the gradients to the Adam optimizer (`Adam.step/5`)
5. Get back a `%{id => new_value}` map of updated parameter values
6. Walk the model tree and apply updates (`Model.update_params/2`)

The same `{"wte", 2, 5}` ID links a parameter through flatten → gradient → optimizer → update → back into the model. Without stable IDs, the optimizer's momentum and velocity buffers (which track per-parameter statistics across training steps) wouldn't know which parameter is which.

In Python, this linking happens implicitly through shared mutable object references — the optimizer holds a reference to the same object as the model, so `optimizer.step()` can mutate it directly. In Elixir, the linking is explicit through ID-keyed maps.

## The forward pass

The forward pass takes a single token and produces a prediction. Given `(token_id, position_id)`, it returns logits — a score for each possible next token:

```elixir
kv_cache = Model.empty_kv_cache(model)

{logits, kv_cache} = Model.gpt(model, 0, 0, kv_cache)
# logits is a list of 5 Values (one per vocab token)
```

Here's what happens inside `Model.gpt/4`:

```elixir
def gpt(model, token_id, pos_id, kv_cache) do
  %{wte: wte, wpe: wpe, lm_head: lm_head, layers: layers} = model.state

  # 1. Look up embeddings and normalize
  x =
    Enum.at(wte, token_id)
    |> Math.add_vec(Enum.at(wpe, pos_id))
    |> Math.rmsnorm()

  # 2. Pass through transformer blocks
  {x, kv_cache} =
    Enum.reduce(0..(model.n_layer - 1), {x, kv_cache}, fn li, {x, kv_cache} ->
      layer = Map.fetch!(layers, li)
      {x, kv_cache} = attn_block(model, x, li, layer, kv_cache)
      x = mlp_block(x, layer)
      {x, kv_cache}
    end)

  # 3. Project to vocabulary
  logits = Math.linear(x, lm_head)
  {logits, kv_cache}
end
```

Three steps:

1. **Embed**: look up the token's embedding vector from `wte`, add the position embedding from `wpe`, normalize with RMSNorm. This gives the model two pieces of information — _what_ token this is and _where_ it is in the sequence.

2. **Transform**: pass through each transformer block (attention + MLP). Each block refines the representation, incorporating information from previous tokens via the attention mechanism and the KV cache.

3. **Project**: multiply by `lm_head` to produce a score for each token in the vocabulary. These scores (logits) become probabilities after softmax.

An untrained model outputs near-random logits — it has no idea what comes next. After training, the logits encode which tokens are likely to follow. The token with the highest logit is the model's best guess; softmax turns these into a probability distribution for sampling.

The `kv_cache` flows in and out of the forward pass, carrying attention state from previous positions. I'll cover how this works in detail in Part 3, when I dig into the attention mechanism.

## Lists as vectors

One thing might look unusual to Elixir developers: the model uses plain lists as vectors and lists-of-lists as matrices. There's no Nx, no tensors, no special numeric types.

```elixir
# A vector is [%Value{}, %Value{}, ...]
# A matrix is [[%Value{}, ...], [%Value{}, ...], ...]
```

This means `Enum.at/2` for indexed access — O(n), not O(1). For production work, that's unacceptable. But for this pedagogical codebase, the dimensions are tiny: vocabulary ~27, embedding ~16, block size ~16. The O(n) cost is negligible.

The upside is transparency. The data structures are plain Elixir — no special types to learn, no opaque tensor operations, no shape-mismatch errors. You can `IO.inspect` any intermediate value and see exactly what's inside. Every operation is a simple `Enum.map` or `Enum.zip` that you can read and trace.

For real work, use Nx + EXLA. For understanding, use lists.

## Three more modules

With tokenization, math, and the model in place, the module count is up to five:

- **Tokenizer** — character-level encoding with BOS framing and O(1) bidirectional lookup maps
- **Math** — six operations (dot, linear, softmax, rmsnorm, add_vec, relu_vec) that produce `Value` nodes for automatic gradient computation
- **Model** — the GPT architecture: embeddings, transformer blocks, language model head, stable `{tag, row, col}` parameter IDs, and a forward pass that processes one token at a time

The model can do a forward pass — turn a token ID into logits over the vocabulary. But I glossed over the most interesting part: what happens inside the transformer block? How does the model "look at" previous tokens to decide what comes next?

## Up next

[Part 3: "How Tokens Talk to Each Other"](part3-attention.md) opens up the transformer block and explains multi-head self-attention — the mechanism that lets each token attend to all previous tokens. Query/key/value projections, scaled dot-product attention, the KV cache, and residual connections. It's simpler than it sounds.
