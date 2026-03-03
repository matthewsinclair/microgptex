---
verblock: "03 Mar 2026:v0.3: matts - All 4 parts written, reviewed, polished"
intent_version: 2.4.0
status: Done
slug: microgptex-blog-post
created: 20260303
completed:
---

# ST0002: MicroGPTEx Blog Post Series

## Objective

Write a 4-part blog post series explaining how GPT works, told through MicroGPTEx — a pure Elixir implementation of Andrej Karpathy's MicroGPT. Each post is self-contained but builds on the previous one.

## Context

MicroGPTEx is a ~1500-line Elixir implementation of the complete GPT training algorithm: reverse-mode autograd, multi-head self-attention, Adam optimization, and autoregressive sampling. Zero external dependencies. Nine modules in a single file.

The project was inspired by [Karpathy's MicroGPT](https://karpathy.github.io/2026/02/12/microgpt/) (2026-02-12) and the [growingswe walkthrough](https://growingswe.com/blog/microgpt). The growingswe post covers the same algorithm in Python in a single interactive article. These posts differ in three ways:

1. **Elixir perspective** — the functional lens reveals things Python hides (immutable autograd, threaded state, pattern-matched dispatch, pure core/impure shell)
2. **Multi-part series** — deeper treatment of each topic, more room for code and explanation
3. **Written companion** — complements the two Livebook notebooks (walkthrough + interactive) rather than replacing them

### Target audience

- Software engineers who know some Elixir (or are Elixir-curious) and want to understand how GPT works at the code level
- Python ML practitioners who'd gain insight from seeing the algorithm through a functional lens
- Anyone who's used ChatGPT and wants to understand what's happening under the hood

### Narrative thread

The central question running through all 4 parts: **what does Elixir make explicit that Python hides?**

- Part 1: Gradients are data (immutable maps), not side effects (mutable `.grad` fields)
- Part 2: All state flows through function arguments — no hidden globals, no mutation
- Part 3: Attention is just math — dot products, softmax, weighted sums — made transparent by the functional style
- Part 4: The training loop is a reduce, not a while-loop; determinism is structural, not opt-in

### Tone and style

- First person singular ("I", "I'll") — not "we" or "we'll"
- Conversational but technically precise
- Real code from MicroGPTEx (not pseudocode)
- Side-by-side Elixir vs Python where the idiom differs
- Each post ~2000-3000 words
- Diagrams where they aid understanding (can be Mermaid, ASCII, or described for the blog platform)
- Link to the Livebook notebooks for hands-on exploration

## Blog Structure

### Part 1: "What If Numbers Could Remember?" — Autograd in Elixir

**File**: `docs/blog/part1-autograd.md`

**Hook**: "You can train a GPT from scratch in ~1500 lines of Elixir. No Nx, no external dependencies. Just pure functions, pattern matching, and a data structure that remembers every calculation."

**Sections**:

1. **Introduction** — What is MicroGPTEx, why translate MicroGPT to Elixir, what you'll learn
   - Explicit credit to Karpathy's MicroGPT and the growingswe walkthrough
   - The 9-module architecture diagram (RNG → Value → ... → Microgptex)
   - "I'll build understanding bottom-up, the same way the code is organized"

2. **The Value struct** — Every number remembers where it came from
   - The `%Value{data, id, children, local_grads}` struct
   - Creating leaf nodes: `V.leaf(3.0, :a)`
   - Operations build a graph: `V.mul(a, b)` stores `local_grads: [b.data, a.data]`
   - Show the struct after a multiplication — the chain rule factors are right there

3. **Building computation graphs** — Forward pass
   - A tiny neuron: `relu(w1*x + w2)`
   - Diagram: the computation graph with values at each node
   - "Every operation in a neural network — every weight times every input, every activation function — builds this graph"

4. **The backward pass** — Computing gradients
   - The fundamental question: "if I nudge this weight, how much does the output change?"
   - `backward/1`: topological sort + fold with chain rule
   - Returns `%{id => gradient}` — an immutable data structure
   - Code walkthrough: the actual `backward/1` implementation (~15 lines)

5. **Fan-out** — When the same value is used twice
   - `a * a`: node `a` feeds both inputs
   - Gradient must be the sum from both paths
   - `Map.update/4` handles accumulation explicitly

6. **The Elixir difference: immutable autograd**
   - **Python**: `backward()` mutates `.grad` fields via `+=`. The accumulation is hidden behind shared mutable references. You have to know that `loss.backward()` is a side effect.
   - **Elixir**: `backward/1` returns a gradient map. You can print it, compare it, pass it to multiple consumers. The fan-out accumulation is visible in `Map.update/4`.
   - Side-by-side code comparison
   - "The same math, but the data flow is visible in the code itself"

7. **Threaded RNG** — Determinism by construction
   - RNG state as explicit argument: `{result, new_rng}`
   - Box-Muller transform for weight initialization
   - Fisher-Yates shuffle for training data
   - "Same seed, same training run — not because you remembered to call `random.seed()`, but because the types enforce it"

8. **Verifying autograd** — The bump test
   - Numerical differentiation as a sanity check
   - Code: compare autograd vs `(f(x+eps) - f(x-eps)) / 2eps`
   - "If these don't match, something is wrong with the chain rule implementation"

9. **What's next** — Preview of Part 2

### Part 2: "From Letters to Logits" — Text, Math, and the Model

**File**: `docs/blog/part2-model.md`

**Hook**: "A neural network can't read letters. It needs numbers — and a small set of mathematical operations to transform them. In this post, I build the bricks and assemble the GPT architecture."

**Sections**:

1. **Recap and setup** — Where I am in the build
   - I have autograd. Now I need: tokenization, math ops, and a model structure.
   - The 9-module dependency chain: I'm covering Tokenizer, Math, and Model

2. **Tokenization** — Converting text to numbers
   - Character-level: each character gets an integer ID
   - BOS (beginning-of-sequence) as start and stop marker
   - `Tokenizer.build/1`, `encode/2`, `decode/2`
   - The dual-map idiom: `char_to_id` + `id_to_char` for O(1) bidirectional lookup
   - "The model's task: given the tokens so far, predict the next one"
   - Training pairs diagram: `BOS→b, b→o, o→b, b→BOS` for "bob"

3. **Math building blocks** — Operations on Value lists
   - Dot product: measuring similarity (used in attention)
   - Linear transform: matrix-vector multiplication (the core of every layer)
   - Softmax: converting scores to probabilities
   - RMSNorm: keeping activations in a stable range
   - "These six functions (dot, linear, softmax, rmsnorm, add_vec, relu_vec) are all the math I need for a GPT"

4. **Softmax and temperature** — Controlling confidence
   - Softmax formula with numerical stability trick (subtract max)
   - Temperature: divide logits before softmax
   - Low temperature → sharp/confident. High temperature → flat/random
   - "Temperature doesn't change the model — it changes how we interpret the model's output"

5. **The GPT model** — What's inside
   - Model initialization: random weights from `N(0, std)`
   - The weight matrices: wte, wpe, lm_head, per-layer (Q, K, V, O, fc1, fc2)
   - Parameter count breakdown: where the numbers come from
   - Stable parameter IDs: `{tag, row, col}` tuples

6. **The forward pass** — One token at a time
   - Input: `(token_id, position_id)` → output: logits over vocabulary
   - Pipeline: embed → normalize → transformer block(s) → project to vocab
   - Diagram: the full forward pass flow
   - "An untrained model outputs near-random logits. After training, the logits encode which tokens are likely to come next"

7. **Elixir idiom: lists as vectors**
   - Why lists instead of Nx tensors: transparency over performance
   - O(n) indexed access is fine for vocab=27, embd=16
   - "For production, use Nx. For understanding, use lists."

8. **What's next** — Preview of Part 3 (attention)

### Part 3: "How Tokens Talk to Each Other" — Attention

**File**: `docs/blog/part3-attention.md`

**Hook**: "Self-attention is the mechanism that lets a GPT 'look back' at previous tokens when predicting the next one. It's the key innovation of the transformer — and it's simpler than you think."

**Sections**:

1. **Recap** — I have a model with random weights. It can do a forward pass. But I skipped the most interesting part: what happens inside the transformer block?

2. **The intuition** — Why attention matters
   - Without attention, each token is processed independently — the model has no memory
   - With attention, each token can "look at" all previous tokens and decide which ones are relevant
   - "When predicting the letter after 'qu', the model needs to know that 'q' came before"

3. **Query, Key, Value** — Three projections
   - The analogy: Q is "what am I looking for?", K is "what do I contain?", V is "what do I offer?"
   - Each is a linear projection of the same input vector
   - Code: `q = Math.linear(x, layer.attn_wq)` etc.

4. **Scaled dot-product attention** — The core computation
   - `score = dot(q, k) / sqrt(d)` — how much does this query match this key?
   - Softmax over scores → attention weights (probability distribution over past positions)
   - Weighted sum of values → the attention output
   - "The model learns what to attend to by learning the Q, K, V projection matrices"
   - The scaling factor `sqrt(d)`: why it matters for numerical stability

5. **Multi-head attention** — Multiple perspectives
   - Split the embedding into `n_head` slices
   - Each head independently computes attention on its slice
   - Concatenate and project back with `Wo`
   - "Head 1 might learn to look at the previous consonant. Head 2 might learn to look at the vowel pattern. The model figures this out during training."
   - Diagram: multi-head attention flow

6. **The KV cache** — Remembering past positions
   - During generation, I process one token at a time
   - The KV cache stores past keys and values to avoid recomputation
   - Elixir idiom: `%{layer_idx => %{keys: [..], values: [..]}}` — prepend + reverse
   - Sequence diagram showing cache growth

7. **Residual connections and normalization**
   - `x + f(x)`: the shortcut that prevents vanishing gradients
   - RMSNorm before attention and before MLP
   - "Residual connections are why deep networks can train at all"

8. **The complete transformer block**
   - Putting it all together: rmsnorm → attention → residual → rmsnorm → MLP → residual
   - The `attn_block/5` and `mlp_block/2` implementations
   - How blocks stack: each layer refines the representation

9. **What's next** — Preview of Part 4 (training and generation)

### Part 4: "Learning and Dreaming" — Training and Generation

**File**: `docs/blog/part4-training.md`

**Hook**: "The model has structure but no knowledge — every weight is random noise. Training is the process that turns noise into understanding. Generation is where that understanding becomes creative."

**Sections**:

1. **Recap** — I have the full model: autograd, tokenization, math, attention. Now I make it learn.

2. **Cross-entropy loss** — Measuring surprise
   - "If the model assigns probability 1.0 to the right answer, loss is 0. If it's guessing uniformly, loss is ln(vocab_size)."
   - `loss_for_doc/3`: feed tokens, compute softmax, measure `-log(p(correct))`
   - "The untrained model's loss is near the random baseline — it's just guessing"

3. **One gradient step** — Forward, backward, update
   - Forward pass builds the computation graph
   - `V.backward(loss)` → gradient map
   - The gradient map tells us: "change this weight by this much to reduce loss"
   - Diagram: forward → loss → backward → optimizer → update

4. **The Adam optimizer** — Smarter than gradient descent
   - Momentum (m): smoothed gradient direction
   - Velocity (v): smoothed gradient magnitude
   - Bias correction: compensating for zero initialization
   - The update formula
   - **Elixir idiom**: `step/5` returns `{new_opt, updates_map}` — no mutation, no shared references
   - Side-by-side with Python's `optimizer.step()` which mutates in-place

5. **The params round-trip** — How the pieces connect
   - model → `params/1` flatten → `backward/1` gradient map → `Adam.step/5` updates → `update_params/2` → model
   - The `{tag, row, col}` ID scheme: why stable IDs matter
   - Diagram: the round-trip cycle

6. **The training loop** — A single Enum.reduce
   - `Trainer.train/1`: reduce over steps, cycling through documents
   - Linear learning rate decay
   - The `on_step` callback: pure core, impure shell
   - "Python uses `for step in range(steps):`. Elixir uses `Enum.reduce(0..(steps-1), ...)`. Same loop, different data model."
   - Watching the loss decrease: text-based loss curve

7. **Autoregressive sampling** — One token at a time
   - Start with BOS, forward pass, softmax, sample, repeat
   - Temperature controls the distribution: low = conservative, high = creative
   - `V.scale_data/2` vs `V.divide/2`: no autograd needed during inference
   - Multi-clause termination: BOS emitted or block_size reached — readable from function heads

8. **The full pipeline** — Train + generate
   - Train on a small dataset of names, generate new ones
   - The progression: random noise → character patterns → plausible names
   - What more training steps and data would achieve

9. **Reflections: What Elixir reveals**
   - Immutable autograd: gradients are data, not side effects
   - Threaded state: determinism by construction
   - Pure core, impure shell: testable without mocking
   - Pattern-matched dispatch: all exit conditions readable from function heads
   - "This isn't just a port. The functional translation makes the algorithm's structure visible."

10. **Try it yourself** — Links and resources
    - GitHub repo link
    - Livebook walkthrough: step through the algorithm
    - Livebook interactive: sliders, charts, real-time exploration
    - Credit: Karpathy's MicroGPT, growingswe walkthrough

## LiveBooks

"Run in Livebook" links (stable, point to GitHub repo):

- Code walkthru: "MicroGPTEx: How GPT Works, from Scratch"
  <https://livebook.dev/run?url=https://github.com/matthewsinclair/microgptex/blob/main/notebooks/walkthrough.livemd>

- Interactive: "MicroGPTEx: Interactive Explorations"
  <https://livebook.dev/run?url=https://github.com/matthewsinclair/microgptex/blob/main/notebooks/interactive.livemd>

Use those links when referring to the LiveBooks. They open in the reader's own Livebook instance (local or hosted) and always point to the latest version in the repo.

## Work Packages

| WP  | Title                     | Status |
| --- | ------------------------- | ------ |
| 01  | Part 1: Autograd          | Done   |
| 02  | Part 2: Model             | Done   |
| 03  | Part 3: Attention         | Done   |
| 04  | Part 4: Training & Gen    | Done   |
| 05  | Review & Cross-references | Done   |

## Files

- `docs/blog/part1-autograd.md` — Part 1
- `docs/blog/part2-model.md` — Part 2
- `docs/blog/part3-attention.md` — Part 3
- `docs/blog/part4-training.md` — Part 4
- `docs/explainer/ABOUT.md` — provenance note for NotebookLM-generated assets
- `docs/explainer/` — NotebookLM-generated explainer assets (audio, video, infographic, slides, PDFs)

## Related Steel Threads

- ST0001: MicroGPTEx implementation (the codebase this blog explains)
