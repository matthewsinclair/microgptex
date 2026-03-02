# lib/microgptex.ex
#
# MicroGPTEx — A functional, pedagogical GPT trainer in Elixir.
#
# Faithfully translated from Andrej Karpathy's MicroGPT (2026-02-12), this file
# implements the complete GPT training algorithm using only scalar operations and
# reverse-mode automatic differentiation. Zero external dependencies.
#
# Nine modules, ordered bottom-up by dependency:
#
#   RNG → Value → Tokenizer → Math → Model → Adam → Trainer → Sampler → Microgptex
#
# Data structures: Lists serve as vectors and matrices here, matching the Python
# original's use of list indexing. For production Elixir, use Nx tensors. Lists give
# O(n) indexed access via Enum.at; this is acceptable for the tiny dimensions used
# in this pedagogical codebase (vocab ~27, embedding ~16).
#
# For real work, use Nx + EXLA. This is the pedagogical version.

# ==============================================================================
# Microgptex.RNG — Pure threaded random number generation
# ==============================================================================

defmodule Microgptex.RNG do
  @moduledoc """
  Pure, threaded random number generation.

  Every function takes an RNG state and returns `{result, new_rng}` — no process
  dictionary, no side effects. This makes all randomness deterministic and testable:
  same seed always produces the same sequence.

  Uses Erlang's `:rand.*_s` ("stateful" suffix) APIs with the `exsss` algorithm,
  which is fast and has good statistical properties for non-cryptographic use.

  ## Role in GPT training

  Randomness appears in three places in the training algorithm:

  - **Weight initialization** — `normal/3` draws from a Gaussian distribution via the
    Box-Muller transform to set initial parameter values. The standard deviation controls
    how "spread out" the initial weights are; too large and gradients explode, too small
    and the model can't learn.
  - **Training data shuffling** — `shuffle/2` implements Fisher-Yates to randomize the
    order in which training examples are presented. This prevents the optimizer from
    memorizing the sequence order.
  - **Sampling** — `uniform01/1` provides the random draw used to sample from the
    model's predicted probability distribution during text generation.

  ## Elixir idiom: threaded state

  Rather than storing RNG state in a process or ETS table, we thread it explicitly
  through every function that needs randomness. This pattern — `{result, new_state}` —
  is the functional equivalent of Python's stateful `random.random()`.

  The threading has a structural advantage: deterministic reproducibility is guaranteed
  by construction. Pass the same seed, get the same training run — no global state to
  corrupt, no test ordering sensitivity, no `random.seed()` calls to forget.
  """

  @typedoc "Opaque RNG state from `:rand.seed_s/2`"
  @type t :: :rand.state()

  @doc """
  Create a new RNG state from an integer seed.

  The triple `{seed, seed*101+7, seed*1009+23}` ensures the three sub-seeds
  for exsss are distinct even for small seed values.
  """
  @spec seed(integer()) :: t()
  def seed(seed_int) when is_integer(seed_int) do
    :rand.seed_s(:exsss, {seed_int, seed_int * 101 + 7, seed_int * 1009 + 23})
  end

  @doc "Draw a uniform float in [0, 1). Returns `{float, new_rng}`."
  @spec uniform01(t()) :: {float(), t()}
  def uniform01(rng), do: :rand.uniform_s(rng)

  @doc "Draw a uniform integer in [0, n). Returns `{int, new_rng}`."
  @spec uniform_int(t(), pos_integer()) :: {non_neg_integer(), t()}
  def uniform_int(rng, n) when is_integer(n) and n > 0 do
    {u, rng2} = uniform01(rng)
    {min(trunc(u * n), n - 1), rng2}
  end

  @doc """
  Draw from a normal distribution using the Box-Muller transform.

  Box-Muller converts two uniform samples into a standard normal sample:

      z = sqrt(-2 * ln(u1)) * cos(2π * u2)

  Then scale and shift: `mean + std * z`.
  """
  @spec normal(t(), number(), number()) :: {float(), t()}
  def normal(rng, mean \\ 0.0, std \\ 1.0) do
    {u1, rng1} = uniform01(rng)
    {u2, rng2} = uniform01(rng1)
    u1 = max(u1, 1.0e-12)

    z0 = :math.sqrt(-2.0 * :math.log(u1)) * :math.cos(2.0 * :math.pi() * u2)
    {mean + std * z0, rng2}
  end

  @doc """
  Fisher-Yates shuffle with threaded RNG.

  Produces a uniformly random permutation. The RNG state is threaded through
  each swap, so the shuffle is fully deterministic given the same seed.
  """
  @spec shuffle(list(), t()) :: {list(), t()}
  def shuffle([], rng), do: {[], rng}

  def shuffle(list, rng) when is_list(list) do
    arr = :array.from_list(list)
    n = :array.size(arr)

    {arr, rng} =
      Enum.reduce((n - 1)..1//-1, {arr, rng}, fn i, {arr, rng} ->
        {j, rng} = uniform_int(rng, i + 1)
        val_i = :array.get(i, arr)
        val_j = :array.get(j, arr)
        arr = :array.set(i, val_j, :array.set(j, val_i, arr))
        {arr, rng}
      end)

    {:array.to_list(arr), rng}
  end
end

# ==============================================================================
# Microgptex.Value — Autograd scalar node
# ==============================================================================

defmodule Microgptex.Value do
  @moduledoc """
  A scalar value with automatic differentiation (autograd).

  This is the heart of the project. Every number in the neural network — every weight,
  every activation, every loss value — is a `Value` node in a computation graph. When
  you write `V.mul(a, b)`, you get back a new `Value` whose `.data` is `a * b`, but
  which also remembers that it came from multiplying `a` and `b`, and stores the partial
  derivatives needed to propagate gradients backward.

  Each `Value` stores:
  - `data` — the scalar float result of the forward pass
  - `id` — a unique identifier (tuple for parameters, reference for intermediates)
  - `children` — the `Value` nodes that were inputs to the operation that produced this node
  - `local_grads` — the partial derivatives of this node's output w.r.t. each child

  ## How autograd works

  **Forward pass**: Build the graph by applying operations. Each op creates a new `Value`
  whose `children` are its inputs and whose `local_grads` are the chain rule factors
  computed eagerly from the input values. For multiplication, `V.mul(a, b)` stores
  `local_grads: [b.data, a.data]` because `d(ab)/da = b` and `d(ab)/db = a`.

  **Backward pass**: `backward/1` performs a topological sort (root-first DFS), then
  folds through the sorted nodes accumulating gradients via the chain rule. Each node's
  gradient is the product of its parent's gradient and the local gradient along that edge.
  The result is a `%{id => gradient}` map — the gradient of the loss w.r.t. every node
  in the graph.

  ## Fan-out and gradient accumulation

  When the same `Value` is used as input to multiple operations (e.g., `a * a`), its
  gradient must be the *sum* of the gradients flowing back through each use. This is
  handled by `Map.update/4`: if a node's ID is already in the gradient map, the new
  gradient is added to the existing one rather than replacing it.

  ## Elixir vs Python

  In Python's micrograd, `backward()` mutates each node's `.grad` field in-place —
  the `+=` operator silently accumulates gradients through shared mutable references.
  In Elixir, `backward/1` returns an immutable gradient map. The accumulation is
  explicit in `Map.update/4`, making the fan-out semantics visible in the code rather
  than hidden behind mutation order.

  A second difference: Python stores backward logic as closures (`_backward` lambdas)
  that capture the forward-pass variables. Elixir stores the chain rule factors eagerly
  as `local_grads` data on the node — making them inspectable at any time, not deferred
  until backward execution.

  ## Note on purity

  Intermediate node IDs use `make_ref()`, which is technically impure (it touches the
  VM's reference counter). This is a pragmatic trade-off: threading an ID counter through
  every arithmetic operation would triple the code size for zero pedagogical benefit.
  The refs are used only as identity tags — they do not affect computation results.
  """

  @type id :: {String.t(), non_neg_integer(), non_neg_integer()} | reference()

  @type t :: %__MODULE__{
          data: float(),
          id: id(),
          children: [t()],
          local_grads: [float()]
        }

  @enforce_keys [:data, :id]
  defstruct [:data, :id, children: [], local_grads: []]

  @doc """
  Create a leaf node — a learnable parameter or constant.

  Parameters use `{tag, row, col}` tuple IDs so the Adam optimizer's momentum/velocity
  buffers survive across training steps. Intermediate nodes use `make_ref()`.
  """
  @spec leaf(number(), id()) :: t()
  def leaf(data, id) when is_number(data) do
    %__MODULE__{data: data * 1.0, id: id, children: [], local_grads: []}
  end

  # Wrap a plain number as a Value with a throwaway ref ID.
  @spec wrap(t() | number()) :: t()
  defp wrap(%__MODULE__{} = v), do: v
  defp wrap(x) when is_number(x), do: leaf(x * 1.0, make_ref())

  @doc "Addition: `d(a+b)/da = 1`, `d(a+b)/db = 1`."
  @spec add(t() | number(), t() | number()) :: t()
  def add(a, b) do
    a = wrap(a)
    b = wrap(b)
    %__MODULE__{data: a.data + b.data, id: make_ref(), children: [a, b], local_grads: [1.0, 1.0]}
  end

  @doc "Multiplication: `d(a*b)/da = b`, `d(a*b)/db = a`."
  @spec mul(t() | number(), t() | number()) :: t()
  def mul(a, b) do
    a = wrap(a)
    b = wrap(b)

    %__MODULE__{
      data: a.data * b.data,
      id: make_ref(),
      children: [a, b],
      local_grads: [b.data, a.data]
    }
  end

  @doc "Power: `d(a^p)/da = p * a^(p-1)`. Exponent `p` is a plain number, not a Value."
  @spec pow(t() | number(), number()) :: t()
  def pow(a, p) when is_number(p) do
    a = wrap(a)

    %__MODULE__{
      data: :math.pow(a.data, p),
      id: make_ref(),
      children: [a],
      local_grads: [p * :math.pow(a.data, p - 1.0)]
    }
  end

  @doc "Natural log: `d(ln(a))/da = 1/a`."
  @spec log(t() | number()) :: t()
  def log(a) do
    a = wrap(a)

    %__MODULE__{
      data: :math.log(a.data),
      id: make_ref(),
      children: [a],
      local_grads: [1.0 / a.data]
    }
  end

  @doc "Exponential: `d(exp(a))/da = exp(a)`."
  @spec exp(t() | number()) :: t()
  def exp(a) do
    a = wrap(a)
    ea = :math.exp(a.data)
    %__MODULE__{data: ea, id: make_ref(), children: [a], local_grads: [ea]}
  end

  @doc "ReLU: `d(relu(a))/da = 1 if a > 0, else 0`."
  @spec relu(t() | number()) :: t()
  def relu(a) do
    a = wrap(a)
    {out, lg} = relu_data(a.data)
    %__MODULE__{data: out, id: make_ref(), children: [a], local_grads: [lg]}
  end

  defp relu_data(data) when data > 0.0, do: {data, 1.0}
  defp relu_data(_data), do: {0.0, 0.0}

  @doc "Negation: `-a`."
  @spec neg(t()) :: t()
  def neg(a), do: mul(a, -1.0)

  @doc "Subtraction: `a - b`."
  @spec sub(t(), t()) :: t()
  def sub(a, b), do: add(a, neg(b))

  @doc "Division: `a / b`, implemented as `a * b^(-1)`."
  @spec divide(t(), t() | number()) :: t()
  def divide(a, b), do: mul(a, pow(b, -1.0))

  @doc """
  Scale a Value's data directly, bypassing the computation graph.

  Used during inference (eg temperature scaling) where gradients are not needed.
  This avoids creating unnecessary autograd nodes.
  """
  @spec scale_data(t(), float()) :: t()
  def scale_data(%__MODULE__{} = v, factor) when is_number(factor) do
    %__MODULE__{v | data: v.data * factor}
  end

  @doc "Sum a list of Values."
  @spec sum([t()]) :: t()
  def sum([]), do: leaf(0.0, make_ref())
  def sum([x]), do: wrap(x)
  def sum(xs), do: Enum.reduce(xs, &add/2)

  @doc "Mean of a list of Values."
  @spec mean([t()]) :: t()
  def mean([_ | _] = xs) do
    divide(sum(xs), length(xs) * 1.0)
  end

  @doc """
  Reverse-mode automatic differentiation.

  Given a root node (typically the loss), computes gradients for all nodes in the
  computation graph. Returns `%{id => gradient}` where each gradient is `d(root)/d(node)`.

  The algorithm:
  1. Topologically sort the graph via DFS (root first, leaves last)
  2. Walk from root to leaves, seeding with `d(root)/d(root) = 1.0`
  3. For each node, propagate its gradient to children via the chain rule:
     `d(root)/d(child) += local_grad * d(root)/d(node)`

  `Map.update/4` handles fan-out: if a value is used multiple times,
  gradients accumulate (addition) rather than overwriting.
  """
  @spec backward(t()) :: %{id() => float()}
  def backward(%__MODULE__{} = root) do
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

  # Topological sort via DFS. Prepends each node after visiting its children,
  # producing root-first order (root at head, leaves at tail).
  @spec topo_sort(t()) :: [t()]
  defp topo_sort(%__MODULE__{} = root) do
    {topo, _visited} = build_topo(root, [], MapSet.new())
    topo
  end

  defp build_topo(%__MODULE__{id: id} = v, topo, visited) do
    case MapSet.member?(visited, id) do
      true ->
        {topo, visited}

      false ->
        visited = MapSet.put(visited, id)

        {topo, visited} =
          Enum.reduce(v.children, {topo, visited}, fn child, {t, vis} ->
            build_topo(child, t, vis)
          end)

        {[v | topo], visited}
    end
  end
end

# ==============================================================================
# Microgptex.Tokenizer — Character-level tokenization
# ==============================================================================

defmodule Microgptex.Tokenizer do
  @moduledoc """
  Character-level tokenizer with a BOS (beginning-of-sequence) token.

  Maps each unique character in the training data to an integer ID. The BOS token
  gets the highest ID and serves double duty as both start and end marker — the model
  learns to emit BOS when it's done generating.

  ## Role in GPT training

  The tokenizer defines the vocabulary, which determines the dimensions of the model's
  embedding and output layers. For the names dataset, the vocabulary is typically 27
  characters (a-z plus BOS), so the embedding matrix is 27×n_embd and the output
  projection is 27×n_embd.

  Each training document is encoded as `[BOS, c1, c2, ..., cn, BOS]`. The leading BOS
  is the prompt (the model sees "start of name"), and the trailing BOS is the target
  that teaches the model to signal "I'm done generating." During sampling, the model
  starts from BOS and generates characters until it emits BOS again.

  ## Why character-level?

  For a pedagogical GPT, character-level tokenization keeps things simple: no BPE,
  no sentencepiece, no subword merges. Each token is one character. The vocabulary
  is small, which makes the embedding matrices tiny and training fast. Production
  models use subword tokenizers (30K-100K tokens) to handle open vocabularies
  efficiently.

  ## Elixir idiom: dual maps for O(1) bidirectional lookup

  Both `char_to_id` (for encoding) and `id_to_char` (for decoding) are stored as
  maps. This avoids the O(n) cost of reversing a map at decode time — a pattern
  worth adopting whenever you need bidirectional lookup on a fixed mapping.
  """

  @type t :: %__MODULE__{
          uchars: [String.t()],
          bos: non_neg_integer(),
          vocab_size: pos_integer(),
          char_to_id: %{String.t() => non_neg_integer()},
          id_to_char: %{non_neg_integer() => String.t()}
        }

  @enforce_keys [:uchars, :bos, :vocab_size, :char_to_id, :id_to_char]
  defstruct [:uchars, :bos, :vocab_size, :char_to_id, :id_to_char]

  @doc """
  Build a tokenizer from a list of documents (strings).

  Extracts all unique characters, sorts them, assigns sequential IDs,
  and reserves the last ID for BOS.
  """
  @spec build([String.t()]) :: t()
  def build(docs) when is_list(docs) do
    uchars =
      docs
      |> Enum.join("")
      |> String.graphemes()
      |> MapSet.new()
      |> MapSet.to_list()
      |> Enum.sort()

    bos = length(uchars)
    char_to_id = Map.new(Enum.with_index(uchars))
    id_to_char = Map.new(Enum.with_index(uchars), fn {ch, id} -> {id, ch} end)

    %__MODULE__{
      uchars: uchars,
      bos: bos,
      vocab_size: bos + 1,
      char_to_id: char_to_id,
      id_to_char: id_to_char
    }
  end

  @doc """
  Encode a document string into a list of token IDs.

  Wraps with BOS at both ends: `[BOS, c1, c2, ..., cn, BOS]`.
  The leading BOS is the prompt; the trailing BOS is the target that signals "done".
  """
  @spec encode(t(), String.t()) :: [non_neg_integer()]
  def encode(%__MODULE__{bos: bos, char_to_id: c2i}, doc) when is_binary(doc) do
    chars = doc |> String.graphemes() |> Enum.map(&Map.fetch!(c2i, &1))
    [bos | chars] ++ [bos]
  end

  @doc """
  Decode a list of token IDs back to a string.

  BOS tokens and out-of-range IDs are silently dropped.
  """
  @spec decode(t(), [non_neg_integer()]) :: String.t()
  def decode(%__MODULE__{id_to_char: i2c}, token_ids) when is_list(token_ids) do
    Enum.map_join(token_ids, "", fn id -> Map.get(i2c, id, "") end)
  end
end

# ==============================================================================
# Microgptex.Math — Vector/matrix operations on Value lists
# ==============================================================================

defmodule Microgptex.Math do
  @moduledoc """
  Vector and matrix operations on lists of `Microgptex.Value` nodes.

  These are the building blocks for the neural network layers. Each operation
  produces new `Value` nodes, extending the computation graph so that gradients
  can flow backward through every mathematical step.

  ## Operations and where they appear in GPT

  - **`dot/2`** — Computes the dot product of two vectors (`sum(w_i * x_i)`).
    Used in attention to compute how much each position "attends to" every other
    position: `score = dot(query, key)`.
  - **`linear/3`** — Matrix-vector multiplication (`W · x`). The core building
    block for projections: embedding lookups, attention Q/K/V projections, MLP
    layers, and the final language model head all use linear transforms.
  - **`softmax/1`** — Converts a vector of logits into a probability distribution
    that sums to 1.0. Used twice: in attention (to weight values by relevance)
    and in sampling (to get token probabilities from the model's output logits).
    Numerically stabilized by subtracting the max logit as a raw float constant
    (not a `Value` node) so the subtraction is not differentiated — this is correct
    because the argmax doesn't change the gradient.
  - **`rmsnorm/1`** — Root-mean-square normalization, a simpler alternative to
    LayerNorm that skips the mean-centering step. Stabilizes training by keeping
    activation magnitudes in a consistent range.
  - **`add_vec/2`**, **`relu_vec/1`**, **`slice/3`** — Element-wise addition,
    ReLU activation, and subvector extraction used in the MLP and attention blocks.

  ## Elixir idiom: lists as vectors

  Vectors are `[%Value{}]` and matrices are `[[%Value{}]]` — plain Elixir lists.
  This matches the Python original and keeps the code transparent, but gives O(n)
  indexed access via `Enum.at/2`. For the tiny dimensions in this pedagogical
  codebase (vocab ~27, embedding ~16), this is acceptable. For production work,
  use `Nx` tensors.
  """

  alias Microgptex.Value, as: V

  @doc "Dot product of two Value vectors: `sum(w_i * x_i)`."
  @spec dot([V.t()], [V.t()]) :: V.t()
  def dot(ws, xs) when is_list(ws) and is_list(xs) do
    Enum.zip(ws, xs)
    |> Enum.map(fn {w, x} -> V.mul(w, x) end)
    |> V.sum()
  end

  @doc """
  Linear transform: multiply input vector `x` by weight matrix `w`.

  `w` is a list of row vectors. Each row produces one output element via dot product.
  This is the core operation of every neural network layer.
  """
  @spec linear([V.t()], [[V.t()]]) :: [V.t()]
  def linear(x, w) when is_list(x) and is_list(w) do
    Enum.map(w, fn w_row -> dot(w_row, x) end)
  end

  @doc """
  Softmax: convert logits to a probability distribution.

  `softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))`

  The `max` subtraction is a numerical stability trick: it prevents overflow in `exp`
  without changing the result (since it cancels in the ratio). We use `.data` for the
  max so it's treated as a constant and not differentiated — this is correct because
  the max doesn't change the gradient of the softmax.
  """
  @spec softmax([V.t()]) :: [V.t()]
  def softmax([_ | _] = logits) do
    max_val = logits |> Enum.map(& &1.data) |> Enum.max()
    exps = Enum.map(logits, fn v -> V.exp(V.sub(v, max_val)) end)
    total = V.sum(exps)
    Enum.map(exps, &V.divide(&1, total))
  end

  @doc """
  RMSNorm (Root Mean Square Normalization).

  `rmsnorm(x) = x / sqrt(mean(x^2) + eps)`

  A simpler alternative to LayerNorm that skips the mean-centering step.
  Used in modern architectures like LLaMA. The `eps` prevents division by zero.
  """
  @spec rmsnorm([V.t()], float()) :: [V.t()]
  def rmsnorm([_ | _] = x, eps \\ 1.0e-5) do
    ms = x |> Enum.map(fn xi -> V.mul(xi, xi) end) |> V.mean()
    scale = V.pow(V.add(ms, eps), -0.5)
    Enum.map(x, fn xi -> V.mul(xi, scale) end)
  end

  @doc "Element-wise addition of two Value vectors."
  @spec add_vec([V.t()], [V.t()]) :: [V.t()]
  def add_vec(a, b) when is_list(a) and is_list(b) do
    Enum.zip(a, b) |> Enum.map(fn {x, y} -> V.add(x, y) end)
  end

  @doc "Element-wise ReLU of a Value vector."
  @spec relu_vec([V.t()]) :: [V.t()]
  def relu_vec(x) when is_list(x), do: Enum.map(x, &V.relu/1)

  @doc "Slice a list: take `len` elements starting at index `from`."
  @spec slice(list(), non_neg_integer(), non_neg_integer()) :: list()
  def slice(v, from, len), do: v |> Enum.drop(from) |> Enum.take(len)
end

# ==============================================================================
# Microgptex.Model — GPT-2 architecture
# ==============================================================================

defmodule Microgptex.Model do
  @moduledoc """
  A minimal GPT-2 model: token embeddings, position embeddings, transformer blocks
  (multi-head self-attention + MLP with RMSNorm), and a language model head.

  ## What this implements

  GPT (Generative Pre-trained Transformer) is an autoregressive language model. Given
  a sequence of tokens, it predicts the probability distribution over the next token.
  Training minimizes the cross-entropy loss between the predicted and actual next tokens.

  The model processes one token at a time (not a full sequence at once). This is
  simpler than batched attention and matches the autoregressive generation pattern:
  each forward pass takes a single `(token_id, position_id)` and returns logits over
  the vocabulary.

  ## Architecture

      input token_id, pos_id
            │
      ┌─────┴──────┐
      │ wte[tok]   │  token embedding lookup
      │ + wpe[pos] │  position embedding lookup
      │ → rmsnorm  │
      └─────┬──────┘
            │
      ┌─────┴─────────────────┐
      │ Transformer Block ×L  │
      │  ├─ rmsnorm           │
      │  ├─ multi-head attn   │
      │  ├─ residual add      │
      │  ├─ rmsnorm           │
      │  ├─ MLP (fc1→relu→fc2)│
      │  └─ residual add      │
      └─────┬─────────────────┘
            │
      ┌─────┴─────┐
      │ lm_head   │  project to vocab logits
      └───────────┘

  **Multi-head attention** splits the embedding into `n_head` independent subspaces.
  Each head computes its own query/key/value projections, attention scores (scaled dot
  product), and weighted value sum. The heads are concatenated and projected back.
  This lets the model attend to different aspects of the input simultaneously.

  **Residual connections** add the block's input directly to its output (`x + f(x)`),
  giving gradients a "shortcut" path that prevents vanishing gradients in deep networks.

  ## KV Cache

  The KV cache is a `%{layer_idx => %{keys: [[Value]], values: [[Value]]}}` map.
  Each forward call prepends the current key/value vectors (reversed on read for
  correct positional ordering). During training, the cache threads through
  `Enum.reduce` over token positions. During inference, it grows one entry per
  generated token, avoiding redundant recomputation of attention over past positions.

  ## Elixir idiom: the params round-trip

  The model state is a nested map of `Value` leaf nodes. The training loop needs to:
  1. Flatten all params (`params/1`) for gradient computation
  2. Compute gradients (`Value.backward/1`) keyed by param ID
  3. Apply optimizer updates (`Adam.step/5`) keyed by param ID
  4. Walk the model tree applying new values (`update_params/2`)

  The `{tag, row, col}` ID scheme makes this round-trip work: the same parameter
  always has the same ID, so Adam's momentum/velocity buffers persist across steps.
  This is explicit where Python uses shared mutable object references implicitly.
  """

  alias Microgptex.{Math, RNG}
  alias Microgptex.Value, as: V

  @type model_config :: %{
          n_layer: pos_integer(),
          n_embd: pos_integer(),
          block_size: pos_integer(),
          n_head: pos_integer(),
          vocab_size: pos_integer(),
          std: float(),
          seed: integer()
        }

  @type t :: %__MODULE__{
          n_layer: pos_integer(),
          n_embd: pos_integer(),
          block_size: pos_integer(),
          n_head: pos_integer(),
          head_dim: pos_integer(),
          state: map()
        }

  @enforce_keys [:n_layer, :n_embd, :block_size, :n_head, :head_dim, :state]
  defstruct [:n_layer, :n_embd, :block_size, :n_head, :head_dim, :state]

  @doc """
  Initialize a new model with random weights.

  Takes a config map and returns `{model, rng}` where the RNG state has been
  advanced past all the random weight initializations.

  Weight matrices are initialized from `N(0, std)` where `std` is typically 0.08.
  """
  @spec init(model_config()) :: {t(), RNG.t()}
  def init(%{
        n_layer: n_layer,
        n_embd: n_embd,
        block_size: block_size,
        n_head: n_head,
        vocab_size: vocab_size,
        std: std,
        seed: seed
      }) do
    head_dim = div(n_embd, n_head)
    rng = RNG.seed(seed)

    {wte, rng} = matrix(rng, vocab_size, n_embd, std, "wte")
    {wpe, rng} = matrix(rng, block_size, n_embd, std, "wpe")
    {lm_head, rng} = matrix(rng, vocab_size, n_embd, std, "lm_head")

    {layers, rng} =
      Enum.reduce(0..(n_layer - 1), {%{}, rng}, fn li, {acc, rng} ->
        {attn_wq, rng} = matrix(rng, n_embd, n_embd, std, "layer#{li}.attn_wq")
        {attn_wk, rng} = matrix(rng, n_embd, n_embd, std, "layer#{li}.attn_wk")
        {attn_wv, rng} = matrix(rng, n_embd, n_embd, std, "layer#{li}.attn_wv")
        {attn_wo, rng} = matrix(rng, n_embd, n_embd, std, "layer#{li}.attn_wo")
        {mlp_fc1, rng} = matrix(rng, 4 * n_embd, n_embd, std, "layer#{li}.mlp_fc1")
        {mlp_fc2, rng} = matrix(rng, n_embd, 4 * n_embd, std, "layer#{li}.mlp_fc2")

        layer = %{
          attn_wq: attn_wq,
          attn_wk: attn_wk,
          attn_wv: attn_wv,
          attn_wo: attn_wo,
          mlp_fc1: mlp_fc1,
          mlp_fc2: mlp_fc2
        }

        {Map.put(acc, li, layer), rng}
      end)

    model = %__MODULE__{
      n_layer: n_layer,
      n_embd: n_embd,
      block_size: block_size,
      n_head: n_head,
      head_dim: head_dim,
      state: %{wte: wte, wpe: wpe, lm_head: lm_head, layers: layers}
    }

    {model, rng}
  end

  @doc """
  Forward pass for a single token at a given position.

  Takes the model, a token ID, a position ID, and the KV cache.
  Returns `{logits, updated_kv_cache}` where logits is a list of `vocab_size` Values.

  This processes one token at a time (not a full sequence), which is how autoregressive
  generation works: each token attends to all previous tokens via the KV cache.
  """
  @spec gpt(t(), non_neg_integer(), non_neg_integer(), map()) :: {[V.t()], map()}
  def gpt(%__MODULE__{} = model, token_id, pos_id, kv_cache) do
    %{wte: wte, wpe: wpe, lm_head: lm_head, layers: layers} = model.state

    x =
      Enum.at(wte, token_id)
      |> Math.add_vec(Enum.at(wpe, pos_id))
      |> Math.rmsnorm()

    {x, kv_cache} =
      Enum.reduce(0..(model.n_layer - 1), {x, kv_cache}, fn li, {x, kv_cache} ->
        layer = Map.fetch!(layers, li)
        {x, kv_cache} = attn_block(model, x, li, layer, kv_cache)
        x = mlp_block(x, layer)
        {x, kv_cache}
      end)

    logits = Math.linear(x, lm_head)
    {logits, kv_cache}
  end

  @doc """
  Create an empty KV cache for the model.

  Returns `%{0 => %{keys: [], values: []}, 1 => %{keys: [], values: []}, ...}`
  with one entry per transformer layer.
  """
  @spec empty_kv_cache(t()) :: map()
  def empty_kv_cache(%__MODULE__{n_layer: n_layer}) do
    Map.new(0..(n_layer - 1), fn li -> {li, %{keys: [], values: []}} end)
  end

  # Multi-head self-attention block with residual connection.
  defp attn_block(%__MODULE__{} = model, x, li, layer, kv_cache) do
    x_residual = x

    x = Math.rmsnorm(x)
    q = Math.linear(x, layer.attn_wq)
    k = Math.linear(x, layer.attn_wk)
    v = Math.linear(x, layer.attn_wv)

    # Append current k, v to the cache for this layer
    layer_cache = Map.fetch!(kv_cache, li)

    layer_cache = %{
      layer_cache
      | keys: [k | layer_cache.keys],
        values: [v | layer_cache.values]
    }

    kv_cache = Map.put(kv_cache, li, layer_cache)

    # Multi-head attention: split q/k/v into heads, compute attention, concatenate
    x_attn =
      Enum.flat_map(0..(model.n_head - 1), fn h ->
        hs = h * model.head_dim
        q_h = Math.slice(q, hs, model.head_dim)
        cached_keys = Enum.reverse(layer_cache.keys)
        cached_values = Enum.reverse(layer_cache.values)

        k_h = Enum.map(cached_keys, &Math.slice(&1, hs, model.head_dim))
        v_h = Enum.map(cached_values, &Math.slice(&1, hs, model.head_dim))

        # Scaled dot-product attention
        scale = :math.sqrt(model.head_dim * 1.0)

        attn_logits =
          Enum.map(k_h, fn k_t ->
            V.divide(Math.dot(q_h, k_t), scale)
          end)

        attn_weights = Math.softmax(attn_logits)

        # Weighted sum of values
        weighted_sum(attn_weights, v_h, model.head_dim)
      end)

    x =
      x_attn
      |> Math.linear(layer.attn_wo)
      |> Math.add_vec(x_residual)

    {x, kv_cache}
  end

  # Weighted sum of value vectors by attention weights, for one head.
  defp weighted_sum(attn_weights, v_h, head_dim) do
    Enum.map(0..(head_dim - 1), fn j ->
      Enum.zip(attn_weights, v_h)
      |> Enum.map(fn {w, v_t} -> V.mul(w, Enum.at(v_t, j)) end)
      |> V.sum()
    end)
  end

  # MLP block: rmsnorm → fc1 → relu → fc2, with residual connection.
  defp mlp_block(x, layer) do
    x_residual = x

    x
    |> Math.rmsnorm()
    |> Math.linear(layer.mlp_fc1)
    |> Math.relu_vec()
    |> Math.linear(layer.mlp_fc2)
    |> Math.add_vec(x_residual)
  end

  @doc """
  Flatten all learnable parameters into a list of `Value` nodes.

  Used to feed parameters to the optimizer. The order is deterministic:
  wte, wpe, lm_head, then layers in order (each layer's matrices in a fixed order).
  """
  @spec params(t()) :: [V.t()]
  def params(%__MODULE__{} = model) do
    %{wte: wte, wpe: wpe, lm_head: lm_head, layers: layers} = model.state

    layer_params =
      layers
      |> Enum.sort_by(fn {li, _} -> li end)
      |> Enum.flat_map(fn {_li, layer} ->
        [:attn_wq, :attn_wk, :attn_wv, :attn_wo, :mlp_fc1, :mlp_fc2]
        |> Enum.flat_map(fn key -> Map.fetch!(layer, key) end)
      end)

    List.flatten(wte) ++ List.flatten(wpe) ++ List.flatten(lm_head) ++ List.flatten(layer_params)
  end

  @doc """
  Apply parameter updates to the model.

  Takes a `%{param_id => new_data_value}` map (from the optimizer) and walks the
  model state tree, replacing each `Value`'s `.data` field where a matching ID is found.
  """
  @spec update_params(t(), %{V.id() => float()}) :: t()
  def update_params(%__MODULE__{} = model, updated_by_id) when is_map(updated_by_id) do
    update_value = fn %V{id: id} = v ->
      case Map.fetch(updated_by_id, id) do
        {:ok, new_data} -> %V{v | data: new_data}
        :error -> v
      end
    end

    %{model | state: update_state(model.state, update_value)}
  end

  defp update_state(state, update_fn) do
    state
    |> Map.update!(:wte, &map_matrix(&1, update_fn))
    |> Map.update!(:wpe, &map_matrix(&1, update_fn))
    |> Map.update!(:lm_head, &map_matrix(&1, update_fn))
    |> Map.update!(:layers, fn layers ->
      Map.new(layers, fn {li, layer} ->
        {li, Map.new(layer, fn {k, mat} -> {k, map_matrix(mat, update_fn)} end)}
      end)
    end)
  end

  defp map_matrix(mat, f), do: Enum.map(mat, fn row -> Enum.map(row, f) end)

  # Generate a random weight matrix: nout rows × nin cols.
  # Each Value gets a stable {tag, row, col} ID.
  defp matrix(rng, nout, nin, std, tag) do
    Enum.reduce(0..(nout - 1), {[], rng}, fn r, {rows, rng} ->
      {row, rng} =
        Enum.reduce(0..(nin - 1), {[], rng}, fn c, {acc, rng} ->
          {w, rng} = RNG.normal(rng, 0.0, std)
          {[V.leaf(w, {tag, r, c}) | acc], rng}
        end)

      {[Enum.reverse(row) | rows], rng}
    end)
    |> then(fn {rows, rng} -> {Enum.reverse(rows), rng} end)
  end
end

# ==============================================================================
# Microgptex.Adam — Adam optimizer
# ==============================================================================

defmodule Microgptex.Adam do
  @moduledoc """
  The Adam optimizer — pure state-in, state-out.

  Adam ("Adaptive Moment Estimation") is the standard optimizer for training
  neural networks. It maintains per-parameter exponential moving averages of:
  - `m` (first moment / momentum): smoothed gradient direction
  - `v` (second moment / velocity): smoothed squared gradient magnitude

  These are bias-corrected and combined to produce an adaptive learning rate for
  each parameter. The key insight: parameters with noisy gradients get smaller
  effective learning rates (high `v` → small step), while parameters with
  consistent gradient direction get momentum (high `m` → bigger step).

  ## The update formula

  For each parameter with gradient `g`:

      m_t = β₁ · m_{t-1} + (1 - β₁) · g           # smoothed gradient
      v_t = β₂ · v_{t-1} + (1 - β₂) · g²          # smoothed squared gradient
      m̂_t = m_t / (1 - β₁^t)                       # bias correction
      v̂_t = v_t / (1 - β₂^t)                       # bias correction
      θ_t = θ_{t-1} - lr · m̂_t / (√v̂_t + ε)       # parameter update

  Bias correction compensates for the fact that `m` and `v` are initialized to zero
  and therefore biased toward zero in early training steps. Without it, the first
  few updates would be too small.

  ## Pure functional design

  In Python, optimizer state lives as mutable attributes on an optimizer object, and
  `optimizer.step()` mutates model parameters in-place via shared references.

  Here, `step/5` takes the optimizer state and returns `{new_optimizer, updates_map}` —
  no mutation. The caller applies the updates explicitly via `Model.update_params/2`.
  The `%{id => value}` maps for `m` and `v` persist across training steps because
  parameter IDs are stable `{tag, row, col}` tuples.

  This two-phase design (compute updates, then apply them) makes the data flow
  visible: the optimizer does not need to know about the model's structure, and
  the model does not need to know about the optimizer's internals.
  """

  @type t :: %__MODULE__{
          lr: float(),
          beta1: float(),
          beta2: float(),
          eps: float(),
          m: %{Microgptex.Value.id() => float()},
          v: %{Microgptex.Value.id() => float()},
          t: non_neg_integer()
        }

  @enforce_keys [:lr, :beta1, :beta2, :eps]
  defstruct [:lr, :beta1, :beta2, :eps, m: %{}, v: %{}, t: 0]

  @doc "Create a new Adam optimizer with the given hyperparameters."
  @spec init(float(), float(), float(), float()) :: t()
  def init(lr, beta1, beta2, eps) do
    %__MODULE__{lr: lr, beta1: beta1, beta2: beta2, eps: eps}
  end

  @doc """
  Perform one Adam update step.

  Takes the current optimizer state, a list of parameter `Value` nodes, a gradient map
  `%{id => gradient}`, and the current learning rate (which may be decayed).

  Returns `{new_optimizer_state, %{param_id => new_data_value}}`.

  The update rule for each parameter:

      m_t = β₁ · m_{t-1} + (1 - β₁) · g
      v_t = β₂ · v_{t-1} + (1 - β₂) · g²
      m̂ = m_t / (1 - β₁ᵗ)          # bias correction
      v̂ = v_t / (1 - β₂ᵗ)
      θ = θ - lr · m̂ / (√v̂ + ε)
  """
  @spec step(t(), [Microgptex.Value.t()], %{Microgptex.Value.id() => float()}, float()) ::
          {t(), %{Microgptex.Value.id() => float()}}
  def step(%__MODULE__{} = opt, params, grads_by_id, lr_t) do
    t = opt.t + 1

    {m2, v2, updated} =
      Enum.reduce(params, {opt.m, opt.v, %{}}, fn %Microgptex.Value{id: id, data: data},
                                                  {m, v, upd} ->
        g = Map.get(grads_by_id, id, 0.0)

        m_t = opt.beta1 * Map.get(m, id, 0.0) + (1.0 - opt.beta1) * g
        v_t = opt.beta2 * Map.get(v, id, 0.0) + (1.0 - opt.beta2) * (g * g)

        m_hat = m_t / (1.0 - :math.pow(opt.beta1, t * 1.0))
        v_hat = v_t / (1.0 - :math.pow(opt.beta2, t * 1.0))

        new_data = data - lr_t * m_hat / (:math.sqrt(v_hat) + opt.eps)

        {Map.put(m, id, m_t), Map.put(v, id, v_t), Map.put(upd, id, new_data)}
      end)

    {%__MODULE__{opt | m: m2, v: v2, t: t}, updated}
  end
end

# ==============================================================================
# Microgptex.Trainer — Loss computation and training loop
# ==============================================================================

defmodule Microgptex.Trainer do
  @moduledoc """
  Loss computation and training loop.

  ## What cross-entropy loss means

  For each position in the training document, the model predicts a probability
  distribution over the vocabulary. The cross-entropy loss measures how surprised
  the model is by the actual next character: `loss = -log(p(correct_token))`.

  If the model assigns probability 1.0 to the right token, loss is 0.
  If the model assigns probability 1/27 (uniform over vocab), loss is ln(27) ≈ 3.3.
  Training drives the loss down by adjusting weights so the model assigns higher
  probability to the tokens that actually follow in the training data.

  ## The training loop

  The loop is a single `Enum.reduce` over steps. At each step:
  1. Pick a training document (cycling through the dataset)
  2. Forward-pass each token to build the computation graph and compute loss
  3. `Value.backward/1` to get the gradient map
  4. `Adam.step/5` to compute parameter updates
  5. `Model.update_params/2` to apply updates to the model
  6. Call the `on_step` callback with `(step, loss_value)`

  The learning rate decays linearly from `lr` to 0 over the training run, which
  helps the model settle into a good minimum rather than bouncing around.

  ## Elixir idiom: pure core with callback IO boundary

  The `on_step` callback keeps IO out of the core training logic. In tests, pass
  `fn _step, _loss -> :ok end` for silent, pure behavior. In production, pass a
  callback that writes progress to stdout (using iodata for zero-allocation IO).

  This separation means the training loop can be tested by asserting on model state
  directly — no stdout capturing, no mocking, no "suppress output" flags.
  """

  alias Microgptex.{Adam, Math, Model}
  alias Microgptex.Value, as: V

  @type train_config :: %{
          docs: [String.t()],
          tokenizer: Microgptex.Tokenizer.t(),
          model: Model.t(),
          steps: pos_integer(),
          learning_rate: float(),
          beta1: float(),
          beta2: float(),
          eps_adam: float(),
          on_step: (non_neg_integer(), float() -> any())
        }

  @doc """
  Compute the cross-entropy loss for a single document.

  Feeds each token through the model one at a time, accumulating the KV cache.
  At each position, the loss is `-log(softmax(logits)[target])` — how surprised
  the model is by the next token. The final loss is the mean over all positions.
  """
  @spec loss_for_doc(Model.t(), Microgptex.Tokenizer.t(), String.t()) :: V.t()
  def loss_for_doc(model, tokenizer, doc) do
    tokens = Microgptex.Tokenizer.encode(tokenizer, doc)
    n = min(model.block_size, length(tokens) - 1)
    kv_cache = Model.empty_kv_cache(model)

    # Zip adjacent token pairs to avoid indexed access into the token list
    token_pairs =
      tokens
      |> Enum.zip(tl(tokens))
      |> Enum.with_index()
      |> Enum.take(n)

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

  @doc """
  Run the training loop.

  Takes a config map with all training parameters and returns `{trained_model, optimizer}`.

  The `on_step` callback is called after each step with `(step_number, loss_float)`.
  Pass `fn _step, _loss -> :ok end` for silent training (eg in tests).
  """
  @spec train(train_config()) :: {Model.t(), Adam.t()}
  def train(%{
        docs: docs,
        tokenizer: tokenizer,
        model: model,
        steps: steps,
        learning_rate: lr,
        beta1: beta1,
        beta2: beta2,
        eps_adam: eps_adam,
        on_step: on_step
      }) do
    opt = Adam.init(lr, beta1, beta2, eps_adam)
    n_docs = length(docs)

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
end

# ==============================================================================
# Microgptex.Sampler — Autoregressive text generation
# ==============================================================================

defmodule Microgptex.Sampler do
  @moduledoc """
  Temperature-controlled autoregressive text generation.

  ## What autoregressive sampling means

  "Autoregressive" means the model generates one token at a time, feeding each
  generated token back as input for the next step. Starting from BOS:

  1. Forward-pass the current token through the model → logits over vocabulary
  2. Divide logits by temperature (lower = more confident/repetitive)
  3. Softmax to get a probability distribution
  4. Sample one token from that distribution
  5. Feed the sampled token back as input for the next position
  6. Stop when BOS is emitted (model says "done") or block_size is reached

  The KV cache avoids redundant computation: at each generation step, only the
  new token is forward-passed, and its key/value vectors are appended to the cache.
  Past tokens' attention computations are reused from the cache.

  ## Temperature

  Temperature controls the "confidence" of the sampling distribution:
  - `temperature = 1.0`: standard sampling, follows the learned distribution
  - `temperature < 1.0`: sharper distribution, more likely to pick the top token
  - `temperature > 1.0`: flatter distribution, more random/creative

  Temperature scaling uses `V.scale_data/2` rather than `V.divide/2` because
  we don't need gradients during inference — manipulating `.data` directly avoids
  building unnecessary computation graph nodes.

  ## Elixir idiom: explicit tail recursion with multi-clause termination

  Instead of Python's `while True: ... if done: break` loop, the sampler uses
  explicit tail recursion through `generate_loop/8`. The two exit conditions —
  BOS token emitted and maximum length reached — are expressed as separate function
  clauses: `continue_or_stop/8` dispatches on whether the token matches BOS via
  a pattern match, and `generate_loop/8` uses a guard (`when pos_id < block_size`)
  for the length limit. A reader can enumerate all exit paths by reading the
  function heads, without tracing through loop bodies.
  """

  alias Microgptex.{Math, Model, RNG}
  alias Microgptex.Value, as: V

  @doc """
  Generate `count` text samples from the model.

  Returns `{[string], new_rng}` where each string is a generated sequence
  (characters between the two BOS tokens).
  """
  @spec generate(Model.t(), Microgptex.Tokenizer.t(), RNG.t(), pos_integer(), float()) ::
          {[String.t()], RNG.t()}
  def generate(model, tokenizer, rng, count, temperature) do
    {acc, rng} =
      Enum.reduce(1..count, {[], rng}, fn _i, {acc, rng} ->
        {name, rng} = generate_one(model, tokenizer, rng, temperature)
        {[name | acc], rng}
      end)

    {Enum.reverse(acc), rng}
  end

  defp generate_one(model, tokenizer, rng, temperature) do
    inv_temp = 1.0 / temperature

    generate_loop(
      model,
      tokenizer,
      rng,
      inv_temp,
      0,
      tokenizer.bos,
      Model.empty_kv_cache(model),
      []
    )
  end

  defp generate_loop(model, tokenizer, rng, inv_temp, pos_id, token_id, kv_cache, chars)
       when pos_id < model.block_size do
    {logits, kv_cache} = Model.gpt(model, token_id, pos_id, kv_cache)

    # Temperature scaling via direct data manipulation (no autograd needed during inference)
    scaled_logits = Enum.map(logits, fn %V{} = l -> V.scale_data(l, inv_temp) end)
    probs = Math.softmax(scaled_logits)

    {next_token, rng} = sample_categorical(rng, probs)
    continue_or_stop(next_token, model, tokenizer, rng, inv_temp, pos_id, kv_cache, chars)
  end

  defp generate_loop(_model, _tokenizer, rng, _inv_temp, _pos_id, _token_id, _kv_cache, chars) do
    {chars |> Enum.reverse() |> Enum.join(), rng}
  end

  # Two clauses: BOS token stops generation, any other token continues.
  defp continue_or_stop(bos, _model, %{bos: bos} = _tok, rng, _inv_t, _pos, _kv, chars) do
    {chars |> Enum.reverse() |> Enum.join(), rng}
  end

  defp continue_or_stop(token, model, tokenizer, rng, inv_temp, pos_id, kv_cache, chars) do
    ch = Map.fetch!(tokenizer.id_to_char, token)

    generate_loop(model, tokenizer, rng, inv_temp, pos_id + 1, token, kv_cache, [
      ch | chars
    ])
  end

  defp sample_categorical(rng, probs) do
    {u, rng} = RNG.uniform01(rng)
    prob_values = Enum.map(probs, & &1.data)

    cdf = Enum.scan(prob_values, &(&1 + &2))
    total = List.last(cdf) || 1.0
    target = u * total

    idx =
      cdf
      |> Enum.with_index()
      |> Enum.find_value(length(cdf) - 1, fn {c, i} -> if c >= target, do: i end)

    {idx, rng}
  end
end

# ==============================================================================
# Microgptex — Top-level API and IO boundary
# ==============================================================================

defmodule Microgptex do
  @moduledoc """
  MicroGPTEx — A functional, pedagogical GPT trainer in Elixir.

  A faithful translation of Andrej Karpathy's MicroGPT demonstrating autograd,
  multi-head attention, Adam optimization, and autoregressive sampling.
  Zero external dependencies.

  ## Quick Start

      Microgptex.run()

  This loads config from `priv/config.yaml`, downloads training data if needed,
  trains the model, and generates sample text.

  ## The IO boundary

  This module is the *only* place in the entire codebase that performs IO:
  reading config files, downloading training data via `:httpc`, printing
  progress during training, and printing generated samples. Every other module
  is pure — given the same inputs, they always produce the same outputs with
  no side effects.

  This "pure core, impure shell" architecture means the entire GPT algorithm
  (autograd, attention, optimization, sampling) can be tested without mocking
  IO, capturing stdout, or managing test fixtures. The shell is thin: `run/1`
  is essentially a pipeline that wires together the pure modules and provides
  the IO callbacks they need.

  ## Architecture

  Nine modules, ordered bottom-up by dependency:

  - `Microgptex.RNG` — Pure threaded random number generation
  - `Microgptex.Value` — Autograd scalar node (forward + backward)
  - `Microgptex.Tokenizer` — Character-level tokenizer with BOS token
  - `Microgptex.Math` — Vector/matrix ops on Value lists
  - `Microgptex.Model` — GPT-2 model: init, forward, params, update
  - `Microgptex.Adam` — Adam optimizer (pure state-in/state-out)
  - `Microgptex.Trainer` — Loss computation + training loop
  - `Microgptex.Sampler` — Temperature-controlled autoregressive sampling
  - `Microgptex` — Top-level API and IO boundary
  """

  alias Microgptex.{Model, RNG, Sampler, Tokenizer, Trainer}

  @doc """
  Run the full training and generation pipeline.

  Loads config, downloads data, trains, and generates samples.
  All IO (printing, file downloads) happens here — the modules below are pure.
  """
  @spec run(keyword()) :: :ok
  def run(opts \\ []) do
    config = load_config()
    config = Keyword.merge(config, opts)

    data_path = ensure_data(Keyword.fetch!(config, :data_url))

    docs =
      data_path
      |> File.read!()
      |> String.split("\n", trim: true)
      |> Enum.map(&String.trim/1)
      |> Enum.reject(&(&1 == ""))

    seed = Keyword.fetch!(config, :seed)
    rng = RNG.seed(seed)
    {docs, _rng} = RNG.shuffle(docs, rng)
    IO.puts("num docs: #{length(docs)}")

    tokenizer = Tokenizer.build(docs)
    IO.puts("vocab size: #{tokenizer.vocab_size}")

    model_cfg = %{
      n_layer: Keyword.fetch!(config, :n_layer),
      n_embd: Keyword.fetch!(config, :n_embd),
      block_size: Keyword.fetch!(config, :block_size),
      n_head: Keyword.fetch!(config, :n_head),
      vocab_size: tokenizer.vocab_size,
      std: Keyword.fetch!(config, :init_std),
      seed: seed
    }

    {model, rng} = Model.init(model_cfg)
    IO.puts("num params: #{length(Model.params(model))}")

    steps = Keyword.fetch!(config, :steps)

    on_step = fn step, loss ->
      IO.write([
        "\rstep ",
        String.pad_leading(Integer.to_string(step + 1), 4),
        " / ",
        String.pad_leading(Integer.to_string(steps), 4),
        " | loss ",
        :erlang.float_to_binary(loss, decimals: 4)
      ])
    end

    {model, _opt} =
      Trainer.train(%{
        docs: docs,
        tokenizer: tokenizer,
        model: model,
        steps: steps,
        learning_rate: Keyword.fetch!(config, :learning_rate),
        beta1: Keyword.fetch!(config, :beta1),
        beta2: Keyword.fetch!(config, :beta2),
        eps_adam: Keyword.fetch!(config, :eps_adam),
        on_step: on_step
      })

    IO.puts("\n\n--- inference (new, hallucinated names) ---")

    temperature = Keyword.fetch!(config, :temperature)
    num_samples = Keyword.fetch!(config, :num_samples)

    {samples, _rng} = Sampler.generate(model, tokenizer, rng, num_samples, temperature)

    samples
    |> Enum.with_index(1)
    |> Enum.each(fn {s, i} ->
      IO.puts("sample #{String.pad_leading(Integer.to_string(i), 2)}: #{s}")
    end)

    :ok
  end

  @doc """
  Load configuration from `priv/config.yaml`.

  Simple key-value YAML parser — no library needed. Supports strings, integers,
  floats, and scientific notation. Comments and blank lines are ignored.
  """
  @spec load_config() :: keyword()
  def load_config do
    priv_dir = :code.priv_dir(:microgptex)
    path = Path.join(priv_dir, "config.yaml")

    path
    |> File.read!()
    |> String.split("\n")
    |> Enum.reject(fn line ->
      trimmed = String.trim(line)
      trimmed == "" or String.starts_with?(trimmed, "#")
    end)
    |> Enum.map(fn line ->
      [key, value] = String.split(line, ":", parts: 2)
      # Config keys are a fixed, known set defined in our own file
      key = key |> String.trim() |> String.to_existing_atom()
      value = value |> String.trim() |> parse_value()
      {key, value}
    end)
  end

  defp parse_value(str) do
    case Integer.parse(str) do
      {i, ""} -> i
      _ -> parse_value_float(str)
    end
  end

  defp parse_value_float(str) do
    case Float.parse(str) do
      {f, ""} -> f
      _ -> str
    end
  end

  defp ensure_data(url) do
    priv_dir = :code.priv_dir(:microgptex)
    data_dir = Path.join(priv_dir, "data")
    File.mkdir_p!(data_dir)
    path = Path.join(data_dir, "input.txt")

    if not File.exists?(path) do
      IO.puts("Downloading training data from #{url}...")
      http_opts = [ssl: ssl_opts()]

      case :httpc.request(:get, {String.to_charlist(url), []}, http_opts, []) do
        {:ok, {{_, 200, _}, _headers, body}} ->
          File.write!(path, body)
          IO.puts("Saved to #{path}")

        {:ok, {{_, status, _}, _, _}} ->
          raise "HTTP #{status} downloading #{url}"

        {:error, reason} ->
          raise "Failed to download #{url}: #{inspect(reason)}"
      end
    end

    path
  end

  # OTP 28 removed :public_key.pkix_verify_hostname_match_fun/1 which httpc's
  # default SSL setup relied on for RFC 6125 wildcard matching (*.example.com).
  # Provide our own match function for HTTPS wildcard certificate verification.
  defp ssl_opts do
    [
      verify: :verify_peer,
      cacerts: :public_key.cacerts_get(),
      customize_hostname_check: [match_fun: &https_wildcard_match/2]
    ]
  end

  defp https_wildcard_match({:dns_id, hostname}, {:dNSName, pattern}) do
    case String.split(to_string(pattern), ".", parts: 2) do
      ["*", domain] ->
        case String.split(to_string(hostname), ".", parts: 2) do
          [_label, rest] -> rest == domain
          _ -> false
        end

      _ ->
        to_string(hostname) == to_string(pattern)
    end
  end

  defp https_wildcard_match(_ref, _presented), do: :default
end
