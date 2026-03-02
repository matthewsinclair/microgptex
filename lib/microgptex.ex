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

  ## Elixir idiom: threaded state

  Rather than storing RNG state in a process or ETS table, we thread it explicitly
  through every function that needs randomness. This pattern — `{result, new_state}` —
  is the functional equivalent of Python's stateful `random.random()`.
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

  Each `Value` is a node in a computation graph. It stores:
  - `data` — the scalar float result of the forward pass
  - `id` — a unique identifier (tuple for parameters, reference for intermediates)
  - `children` — the `Value` nodes that were inputs to the operation that produced this node
  - `local_grads` — the partial derivatives of this node's output w.r.t. each child

  ## How autograd works

  **Forward pass**: Build the graph by applying operations. Each op creates a new `Value`
  whose `children` are its inputs and whose `local_grads` are the partial derivatives
  (chain rule factors) computed from the input values.

  **Backward pass**: `backward/1` traverses the graph in reverse topological order,
  accumulating gradients via the chain rule. The result is a `%{id => gradient}` map.

  ## Elixir vs Python

  In Python's micrograd, `backward()` mutates each node's `.grad` field in-place.
  In Elixir, `backward/1` returns an immutable gradient map — no mutation, no side effects.
  Gradient fan-out (same value used twice, like `a * a`) is handled naturally by
  `Map.update/4`, which accumulates rather than overwrites.

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

  ## Why character-level?

  For a pedagogical GPT, character-level tokenization keeps things simple: no BPE,
  no sentencepiece, no subword merges. Each token is one character. The vocabulary
  is small (27 for lowercase English names: a-z plus BOS), which makes the embedding
  matrices tiny and training fast.
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

  These are the building blocks for the neural network layers: dot products for
  attention scores, linear transforms for projections, softmax for probability
  distributions, and RMSNorm for layer normalization.

  All operations produce new `Value` nodes, extending the computation graph for
  automatic differentiation.
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

  The model state is a nested map of `Value` leaf nodes (the learnable parameters).
  Each parameter has a stable `{tag, row, col}` ID that persists across training
  steps, so the Adam optimizer's momentum and velocity buffers line up correctly.

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

  ## KV Cache

  The KV cache is a `%{layer_idx => %{keys: [[Value]], values: [[Value]]}}` map.
  Each forward call appends the current key/value vectors. During training, the cache
  threads through `Enum.reduce` over token positions. During inference, it grows
  one entry per generated token.
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

  Adam maintains per-parameter exponential moving averages of:
  - `m` (first moment / momentum): smoothed gradient direction
  - `v` (second moment / velocity): smoothed squared gradient magnitude

  These are bias-corrected and combined to produce an adaptive learning rate for
  each parameter. The key insight: parameters with noisy gradients get smaller
  effective learning rates (high `v` → small step), while parameters with
  consistent gradient direction get momentum (high `m` → bigger step).

  ## Pure functional design

  In Python, optimizer state lives as mutable attributes on an optimizer object.
  Here, `step/4` takes the optimizer state and returns a new one — no mutation.
  The `%{id => value}` maps for `m` and `v` persist across training steps because
  parameter IDs are stable `{tag, row, col}` tuples.
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

  The training loop is a `Enum.reduce` over steps. At each step:
  1. Pick a training document
  2. Compute the cross-entropy loss over the document
  3. Backpropagate to get gradients
  4. Adam update
  5. Call the `on_step` callback (the IO boundary)

  The `on_step` callback receives `(step, loss_value)` and returns `:ok`.
  This keeps IO out of the core training logic, making it testable.
  Note: the callback makes `train/1` technically impure — purity depends on what
  the caller passes. Use `fn _step, _loss -> :ok end` for pure behavior.
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

  Starting from BOS, the sampler repeatedly:
  1. Forward-pass the current token through the model
  2. Divide logits by temperature (lower = more confident/repetitive)
  3. Softmax to get probabilities
  4. Sample from the categorical distribution
  5. Stop when BOS is emitted (model says "done") or block_size is reached

  ## Temperature

  - `temperature = 1.0`: standard sampling, follows the learned distribution
  - `temperature < 1.0`: sharper distribution, more likely to pick the top token
  - `temperature > 1.0`: flatter distribution, more random/creative
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
    case Float.parse(str) do
      {f, ""} -> f
      _ -> parse_value_non_float(str)
    end
  end

  defp parse_value_non_float(str) do
    case Integer.parse(str) do
      {i, ""} -> i
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
      :ok = Application.ensure_started(:inets)
      :ok = Application.ensure_started(:ssl)

      case :httpc.request(:get, {String.to_charlist(url), []}, [], []) do
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
end
