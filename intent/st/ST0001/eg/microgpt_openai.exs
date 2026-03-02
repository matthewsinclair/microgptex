# atomic_gpt.exs
# A dependency-free, micrograd-style, single-file GPT trainer in Elixir.
# The core is written as pure functions that thread state explicitly.

defmodule AtomicGPT.RNG do
  @moduledoc false

  # Threaded RNG state (pure): we do not rely on process RNG state.
  # Uses :rand's *_s APIs.
  def seed(seed_int) when is_integer(seed_int) do
    # exsss is stable and good enough for this toy.
    :rand.seed_s(:exsss, {seed_int, seed_int * 101 + 7, seed_int * 1009 + 23})
  end

  def uniform01(rng) do
    :rand.uniform_s(rng)
  end

  def uniform_int(rng, n) when is_integer(n) and n > 0 do
    {u, rng2} = uniform01(rng)
    {min(trunc(u * n), n - 1), rng2}
  end

  # Box–Muller transform
  def normal(rng, mean \\ 0.0, std \\ 1.0) when is_number(mean) and is_number(std) and std >= 0 do
    {u1, rng1} = uniform01(rng)
    {u2, rng2} = uniform01(rng1)

    # avoid log(0)
    u1 = max(u1, 1.0e-12)

    z0 =
      :math.sqrt(-2.0 * :math.log(u1)) *
        :math.cos(2.0 * :math.pi() * u2)

    {mean + std * z0, rng2}
  end

  # Fisher–Yates shuffle, threaded RNG
  def shuffle(list, rng) when is_list(list) do
    list
    |> Enum.with_index()
    |> Enum.reduce({%{}, rng}, fn {x, i}, {acc, rng0} ->
      {j, rng1} = uniform_int(rng0, i + 1)
      acc =
        acc
        |> Map.put(i, Map.get(acc, j, x))
        |> Map.put(j, x)

      {acc, rng1}
    end)
    |> then(fn {acc, rng2} ->
      shuffled =
        0..(length(list) - 1)
        |> Enum.map(&Map.fetch!(acc, &1))

      {shuffled, rng2}
    end)
  end
end

defmodule AtomicGPT.Tokenizer do
  @moduledoc false
  defstruct [:uchars, :bos, :vocab_size, :char_to_id]

  def build(docs) when is_list(docs) do
    uchars =
      docs
      |> Enum.join("")
      |> String.graphemes()
      |> MapSet.new()
      |> MapSet.to_list()
      |> Enum.sort()

    bos = length(uchars)

    %__MODULE__{
      uchars: uchars,
      bos: bos,
      vocab_size: bos + 1,
      char_to_id: Map.new(Enum.with_index(uchars))
    }
  end

  def encode(%__MODULE__{bos: bos, char_to_id: c2i}, doc) when is_binary(doc) do
    [bos] ++
      (doc |> String.graphemes() |> Enum.map(&Map.fetch!(c2i, &1))) ++
      [bos]
  end

  def decode(%__MODULE__{uchars: uchars}, token_ids) when is_list(token_ids) do
    token_ids
    |> Enum.map(fn
      id when is_integer(id) and id >= 0 and id < length(uchars) -> Enum.at(uchars, id)
      _ -> ""
    end)
    |> Enum.join("")
  end
end

defmodule AtomicGPT.Value do
  @moduledoc false
  # micrograd-ish scalar node:
  # - data: scalar float
  # - id: stable identifier for gradient accumulation + optimiser buffers
  # - children: list of child Values
  # - local_grads: list of local derivative scalars aligned with children
  defstruct [:data, :id, children: [], local_grads: []]

  def leaf(data, id) when is_number(data) do
    %__MODULE__{data: data * 1.0, id: id, children: [], local_grads: []}
  end

  defp wrap(%__MODULE__{} = v), do: v
  defp wrap(x) when is_number(x), do: leaf(x * 1.0, make_ref())

  def add(a, b) do
    a = wrap(a)
    b = wrap(b)
    %__MODULE__{data: a.data + b.data, id: make_ref(), children: [a, b], local_grads: [1.0, 1.0]}
  end

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

  def pow(a, p) when is_number(p) do
    a = wrap(a)
    %__MODULE__{
      data: :math.pow(a.data, p),
      id: make_ref(),
      children: [a],
      local_grads: [p * :math.pow(a.data, p - 1.0)]
    }
  end

  def log(a) do
    a = wrap(a)
    %__MODULE__{
      data: :math.log(a.data),
      id: make_ref(),
      children: [a],
      local_grads: [1.0 / a.data]
    }
  end

  def exp(a) do
    a = wrap(a)
    ea = :math.exp(a.data)

    %__MODULE__{
      data: ea,
      id: make_ref(),
      children: [a],
      local_grads: [ea]
    }
  end

  def relu(a) do
    a = wrap(a)
    out = if a.data > 0.0, do: a.data, else: 0.0
    lg = if a.data > 0.0, do: 1.0, else: 0.0

    %__MODULE__{
      data: out,
      id: make_ref(),
      children: [a],
      local_grads: [lg]
    }
  end

  def neg(a), do: mul(a, -1.0)
  def sub(a, b), do: add(a, neg(b))
  def div(a, b), do: mul(a, pow(b, -1.0))

  def sum([]), do: leaf(0.0, make_ref())
  def sum([x]), do: wrap(x)
  def sum(xs), do: Enum.reduce(xs, &add/2)

  def mean(xs) when is_list(xs) and length(xs) > 0 do
    div(sum(xs), length(xs) * 1.0)
  end

  def backward(%__MODULE__{} = root) do
    topo = topo_sort(root)
    grads0 = %{root.id => 1.0}

    Enum.reduce(Enum.reverse(topo), grads0, fn %__MODULE__{} = v, grads ->
      vg = Map.get(grads, v.id, 0.0)

      Enum.zip(v.children, v.local_grads)
      |> Enum.reduce(grads, fn {child, local}, acc ->
        Map.update(acc, child.id, local * vg, &(&1 + local * vg))
      end)
    end)
  end

  defp topo_sort(%__MODULE__{} = root) do
    {topo, _visited} = build_topo(root, [], MapSet.new())
    topo
  end

  defp build_topo(%__MODULE__{id: id} = v, topo, visited) do
    if MapSet.member?(visited, id) do
      {topo, visited}
    else
      visited = MapSet.put(visited, id)

      {topo, visited} =
        Enum.reduce(v.children, {topo, visited}, fn child, {t, vis} ->
          build_topo(child, t, vis)
        end)

      {[v | topo], visited}
    end
  end
end

defmodule AtomicGPT.Math do
  @moduledoc false
  alias AtomicGPT.Value, as: V

  # dot product: sum_i (wi * xi)
  def dot(w_row, x) when is_list(w_row) and is_list(x) do
    Enum.zip(w_row, x)
    |> Enum.map(fn {wi, xi} -> V.mul(wi, xi) end)
    |> V.sum()
  end

  def linear(x, w) when is_list(x) and is_list(w) do
    Enum.map(w, fn w_row -> dot(w_row, x) end)
  end

  def softmax(logits) when is_list(logits) and logits != [] do
    max_val = logits |> Enum.map(& &1.data) |> Enum.max()

    exps =
      logits
      |> Enum.map(fn v -> V.exp(V.sub(v, max_val)) end)

    total = V.sum(exps)
    Enum.map(exps, &V.div(&1, total))
  end

  def rmsnorm(x, eps \\ 1.0e-5) when is_list(x) and length(x) > 0 do
    ms =
      x
      |> Enum.map(fn xi -> V.mul(xi, xi) end)
      |> V.mean()

    scale = V.pow(V.add(ms, eps), -0.5)
    Enum.map(x, fn xi -> V.mul(xi, scale) end)
  end

  def add_vec(a, b) when is_list(a) and is_list(b) do
    Enum.zip(a, b) |> Enum.map(fn {x, y} -> V.add(x, y) end)
  end

  def relu_vec(x) when is_list(x), do: Enum.map(x, &V.relu/1)

  def slice(v, from, len) do
    v |> Enum.drop(from) |> Enum.take(len)
  end
end

defmodule AtomicGPT.Model do
  @moduledoc false
  alias AtomicGPT.Math
  alias AtomicGPT.Value, as: V

  defstruct [:n_layer, :n_embd, :block_size, :n_head, :head_dim, :state]

  # state holds matrices of %Value{} leaves
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
    rng0 = AtomicGPT.RNG.seed(seed)

    {wte, rng1} = matrix(rng0, vocab_size, n_embd, std, "wte")
    {wpe, rng2} = matrix(rng1, block_size, n_embd, std, "wpe")
    {lm_head, rng3} = matrix(rng2, vocab_size, n_embd, std, "lm_head")

    {layers, rng4} =
      Enum.reduce(0..(n_layer - 1), {%{}, rng3}, fn li, {acc, rng} ->
        {attn_wq, rng} = matrix(rng, n_embd, n_embd, std, "layer#{li}.attn_wq")
        {attn_wk, rng} = matrix(rng, n_embd, n_embd, std, "layer#{li}.attn_wk")
        {attn_wv, rng} = matrix(rng, n_embd, n_embd, std, "layer#{li}.attn_wv")
        {attn_wo, rng} = matrix(rng, n_embd, n_embd, std, "layer#{li}.attn_wo")
        {mlp_fc1, rng} = matrix(rng, 4 * n_embd, n_embd, std, "layer#{li}.mlp_fc1")
        {mlp_fc2, rng} = matrix(rng, n_embd, 4 * n_embd, std, "layer#{li}.mlp_fc2")

        layer =
          %{
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

    {model, rng4}
  end

  # forward: single token at pos_id, with cached keys/values (lists per layer)
  def gpt(%__MODULE__{} = model, token_id, pos_id, keys, values) do
    %{wte: wte, wpe: wpe, lm_head: lm_head, layers: layers} = model.state

    tok_emb = Enum.at(wte, token_id)
    pos_emb = Enum.at(wpe, pos_id)

    x =
      tok_emb
      |> Math.add_vec(pos_emb)
      |> Math.rmsnorm()

    {x, keys, values} =
      Enum.reduce(0..(model.n_layer - 1), {x, keys, values}, fn li, {x, keys, values} ->
        layer = Map.fetch!(layers, li)
        {x, keys, values} = attn_block(model, x, li, layer, keys, values)
        {x, keys, values} = mlp_block(x, li, layer, keys, values)
        {x, keys, values}
      end)

    logits = Math.linear(x, lm_head)
    logits
  end

  defp attn_block(%__MODULE__{} = model, x, li, layer, keys, values) do
    x_residual = x

    x = Math.rmsnorm(x)
    q = Math.linear(x, layer.attn_wq)
    k = Math.linear(x, layer.attn_wk)
    v = Math.linear(x, layer.attn_wv)

    keys = update_cache(keys, li, k)
    values = update_cache(values, li, v)

    x_attn =
      0..(model.n_head - 1)
      |> Enum.flat_map(fn h ->
        hs = h * model.head_dim
        q_h = Math.slice(q, hs, model.head_dim)

        k_h =
          keys
          |> Enum.at(li)
          |> Enum.map(&Math.slice(&1, hs, model.head_dim))

        v_h =
          values
          |> Enum.at(li)
          |> Enum.map(&Math.slice(&1, hs, model.head_dim))

        attn_logits =
          0..(length(k_h) - 1)
          |> Enum.map(fn t ->
            # dot(q_h, k_h[t]) / sqrt(head_dim)
            dot =
              Enum.zip(q_h, Enum.at(k_h, t))
              |> Enum.map(fn {a, b} -> V.mul(a, b) end)
              |> V.sum()

            V.div(dot, :math.sqrt(model.head_dim * 1.0))
          end)

        attn_weights = Math.softmax(attn_logits)

        0..(model.head_dim - 1)
        |> Enum.map(fn j ->
          0..(length(v_h) - 1)
          |> Enum.map(fn t ->
            V.mul(Enum.at(attn_weights, t), Enum.at(Enum.at(v_h, t), j))
          end)
          |> V.sum()
        end)
      end)

    x =
      x_attn
      |> Math.linear(layer.attn_wo)
      |> Math.add_vec(x_residual)

    {x, keys, values}
  end

  defp mlp_block(x, _li, layer, keys, values) do
    x_residual = x

    x =
      x
      |> Math.rmsnorm()
      |> Math.linear(layer.mlp_fc1)
      |> Math.relu_vec()
      |> Math.linear(layer.mlp_fc2)
      |> Math.add_vec(x_residual)

    {x, keys, values}
  end

  defp update_cache(caches, li, item) do
    List.update_at(caches, li, fn layer_cache -> layer_cache ++ [item] end)
  end

  # matrix: nout rows, nin cols
  defp matrix(rng, nout, nin, std, tag) do
    Enum.reduce(0..(nout - 1), {[], rng}, fn r, {rows, rng0} ->
      {row, rng1} =
        Enum.reduce(0..(nin - 1), {[], rng0}, fn c, {acc, rngx} ->
          {w, rngy} = AtomicGPT.RNG.normal(rngx, 0.0, std)
          id = {tag, r, c}
          {[V.leaf(w, id) | acc], rngy}
        end)

      {[Enum.reverse(row) | rows], rng1}
    end)
    |> then(fn {rows, rng2} -> {Enum.reverse(rows), rng2} end)
  end

  def params(%__MODULE__{} = model) do
    %{wte: wte, wpe: wpe, lm_head: lm_head, layers: layers} = model.state

    [wte, wpe, lm_head | Map.values(layers) |> Enum.flat_map(&Map.values/1)]
    |> List.flatten()
    |> List.flatten()
  end

  def update_params(%__MODULE__{} = model, updated_by_id) when is_map(updated_by_id) do
    update_value = fn
      %AtomicGPT.Value{id: id} = v ->
        case Map.fetch(updated_by_id, id) do
          {:ok, new_data} -> %AtomicGPT.Value{v | data: new_data}
          :error -> v
        end
    end

    model
    |> Map.update!(:state, fn state ->
      state
      |> Map.update!(:wte, &map_matrix(&1, update_value))
      |> Map.update!(:wpe, &map_matrix(&1, update_value))
      |> Map.update!(:lm_head, &map_matrix(&1, update_value))
      |> Map.update!(:layers, fn layers ->
        Map.new(layers, fn {li, layer} ->
          {li,
           Map.new(layer, fn {k, mat} ->
             {k, map_matrix(mat, update_value)}
           end)}
        end)
      end)
    end)
  end

  defp map_matrix(mat, f) do
    Enum.map(mat, fn row -> Enum.map(row, f) end)
  end
end

defmodule AtomicGPT.Optim.Adam do
  @moduledoc false

  defstruct [:lr, :beta1, :beta2, :eps, m: %{}, v: %{}, t: 0]

  def init(lr, beta1, beta2, eps) do
    %__MODULE__{lr: lr, beta1: beta1, beta2: beta2, eps: eps, m: %{}, v: %{}, t: 0}
  end

  # Returns {opt2, updated_by_id} where updated_by_id maps param_id -> new_data
  def step(%__MODULE__{} = opt, params, grads_by_id, lr_t) do
    t = opt.t + 1

    {m2, v2, updated} =
      Enum.reduce(params, {opt.m, opt.v, %{}}, fn %AtomicGPT.Value{id: id, data: data}, {m, v, upd} ->
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

defmodule AtomicGPT.Train do
  @moduledoc false
  alias AtomicGPT.Math
  alias AtomicGPT.Value, as: V

  def loss_for_doc(model, tokenizer, doc) do
    tokens = AtomicGPT.Tokenizer.encode(tokenizer, doc)
    n = min(model.block_size, length(tokens) - 1)

    keys = Enum.map(1..model.n_layer, fn _ -> [] end)
    values = Enum.map(1..model.n_layer, fn _ -> [] end)

    {losses, _keys, _values} =
      Enum.reduce(0..(n - 1), {[], keys, values}, fn pos_id, {acc, keys, values} ->
        token_id = Enum.at(tokens, pos_id)
        target_id = Enum.at(tokens, pos_id + 1)

        logits = AtomicGPT.Model.gpt(model, token_id, pos_id, keys, values)
        probs = Math.softmax(logits)
        loss_t = V.neg(V.log(Enum.at(probs, target_id)))

        {acc ++ [loss_t], keys, values}
      end)

    V.mean(losses)
  end

  def train(%{
        docs: docs,
        tokenizer: tokenizer,
        model: model,
        steps: steps,
        learning_rate: lr,
        beta1: beta1,
        beta2: beta2,
        eps_adam: eps_adam
      }) do
    opt = AtomicGPT.Optim.Adam.init(lr, beta1, beta2, eps_adam)
    params = AtomicGPT.Model.params(model)

    Enum.reduce(0..(steps - 1), {model, opt}, fn step, {model, opt} ->
      doc = Enum.at(docs, rem(step, length(docs)))
      loss = loss_for_doc(model, tokenizer, doc)
      grads = V.backward(loss)

      lr_t = lr * (1.0 - step / max(steps, 1))

      {opt2, updated_by_id} =
        AtomicGPT.Optim.Adam.step(opt, params, grads, lr_t)

      model2 = AtomicGPT.Model.update_params(model, updated_by_id)

      # refresh params to new data for next optimiser step
      params2 = AtomicGPT.Model.params(model2)

      # side-effect boundary kept here (caller can remove if desired)
      IO.write(
        "step #{String.pad_leading(Integer.to_string(step + 1), 4)} / #{String.pad_leading(Integer.to_string(steps), 4)} | loss #{:io_lib.format("~.4f", [loss.data])}\r"
      )

      {model2, %AtomicGPT.Optim.Adam{opt2 | t: opt2.t} |> then(fn o -> o end)}
      |> then(fn {m, o} -> {m, o} end)
      |> then(fn {m, o} ->
        # carry updated params implicitly via model; keep params2 from being optimised away
        _ = params2
        {m, o}
      end)
    end)
    |> then(fn {model2, opt2} ->
      IO.write("\n")
      {model2, opt2}
    end)
  end
end

defmodule AtomicGPT.Sample do
  @moduledoc false
  alias AtomicGPT.Math

  def generate_names(model, tokenizer, rng, count, temperature) do
    Enum.reduce(1..count, {[], rng}, fn _i, {acc, rng0} ->
      {name, rng1} = generate_one(model, tokenizer, rng0, temperature)
      {acc ++ [name], rng1}
    end)
  end

  defp generate_one(model, tokenizer, rng, temperature) do
    keys = Enum.map(1..model.n_layer, fn _ -> [] end)
    values = Enum.map(1..model.n_layer, fn _ -> [] end)

    token_id = tokenizer.bos

    Enum.reduce_while(0..(model.block_size - 1), {[], token_id, keys, values, rng}, fn pos_id,
                                                                                       {chars, token_id, keys, values, rng0} ->
      logits = AtomicGPT.Model.gpt(model, token_id, pos_id, keys, values)
      probs = Math.softmax(Enum.map(logits, fn l -> %AtomicGPT.Value{l | data: l.data / temperature} end))

      {next_token, rng1} = sample_categorical(rng0, probs)

      if next_token == tokenizer.bos do
        {:halt, {Enum.join(chars), rng1}}
      else
        ch = Enum.at(tokenizer.uchars, next_token)
        {:cont, {chars ++ [ch], next_token, keys, values, rng1}}
      end
    end)
    |> case do
      {name, rng2} -> {name, rng2}
      other -> other
    end
  end

  defp sample_categorical(rng, probs) do
    # probs: list of %Value{data: p}
    {u, rng1} = AtomicGPT.RNG.uniform01(rng)

    probs
    |> Enum.map(& &1.data)
    |> cumulative()
    |> pick_from_cdf(u)
    |> then(fn idx -> {idx, rng1} end)
  end

  defp cumulative(ps) do
    Enum.reduce(ps, [], fn p, [] -> [p] end)
    |> case do
      [] ->
        []

      [first | rest] ->
        Enum.reduce(rest, [first], fn p, acc -> acc ++ [List.last(acc) + p] end)
    end
  end

  defp pick_from_cdf(cdf, u) do
    total = List.last(cdf) || 1.0
    target = u * total

    cdf
    |> Enum.with_index()
    |> Enum.find_value(length(cdf) - 1, fn {c, i} -> if c >= target, do: i, else: nil end)
  end
end

defmodule AtomicGPT.Main do
  @moduledoc false

  def run do
    seed = 42
    rng0 = AtomicGPT.RNG.seed(seed)

    docs =
      "input.txt"
      |> File.read!()
      |> String.split("\n", trim: true)
      |> Enum.map(&String.trim/1)
      |> Enum.reject(&(&1 == ""))

    {docs, _rng1} = AtomicGPT.RNG.shuffle(docs, rng0)
    IO.puts("num docs: #{length(docs)}")

    tokenizer = AtomicGPT.Tokenizer.build(docs)
    IO.puts("vocab size: #{tokenizer.vocab_size}")

    cfg = %{
      n_layer: 1,
      n_embd: 16,
      block_size: 16,
      n_head: 4,
      vocab_size: tokenizer.vocab_size,
      std: 0.08,
      seed: seed
    }

    {model, rng2} = AtomicGPT.Model.init(cfg)
    IO.puts("num params: #{length(AtomicGPT.Model.params(model))}")

    {model, _opt} =
      AtomicGPT.Train.train(%{
        docs: docs,
        tokenizer: tokenizer,
        model: model,
        steps: 1000,
        learning_rate: 0.01,
        beta1: 0.85,
        beta2: 0.99,
        eps_adam: 1.0e-8
      })

    IO.puts("\n--- inference (new, hallucinated names) ---")

    {samples, _rng3} = AtomicGPT.Sample.generate_names(model, tokenizer, rng2, 20, 0.5)

    samples
    |> Enum.with_index(1)
    |> Enum.each(fn {s, i} -> IO.puts("sample #{String.pad_leading(Integer.to_string(i), 2)}: #{s}") end)
  end
end

# If you want it to run as an .exs script, keep this:
AtomicGPT.Main.run()

If you want this to be more “blocked Elixir”, the next step is to split Model.gpt/5 into with-driven helpers (embed/4, attn/6, mlp/4, lm_head/2) and make the caches a struct with pattern-matched fields per layer/head.
