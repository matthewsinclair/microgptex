defmodule MicrogptexTest do
  @moduledoc "Tests for the MicroGPTEx pedagogical GPT trainer."

  use ExUnit.Case, async: true

  alias Microgptex.{Adam, Math, Model, RNG, Sampler, Tokenizer, Trainer}
  alias Microgptex.Value, as: V

  # ============================================================================
  # Value autograd tests
  # ============================================================================

  describe "Value forward ops" do
    test "add produces correct sum" do
      a = V.leaf(3.0, :a)
      b = V.leaf(4.0, :b)
      assert V.add(a, b).data == 7.0
    end

    test "mul produces correct product" do
      a = V.leaf(3.0, :a)
      b = V.leaf(4.0, :b)
      assert V.mul(a, b).data == 12.0
    end

    test "pow produces correct power" do
      a = V.leaf(3.0, :a)
      assert V.pow(a, 2).data == 9.0
    end

    test "log produces correct natural log" do
      a = V.leaf(:math.exp(1.0), :a)
      assert_in_delta V.log(a).data, 1.0, 1.0e-10
    end

    test "exp produces correct exponential" do
      a = V.leaf(1.0, :a)
      assert_in_delta V.exp(a).data, :math.exp(1.0), 1.0e-10
    end

    test "relu passes positive values" do
      a = V.leaf(5.0, :a)
      assert V.relu(a).data == 5.0
    end

    test "relu zeros negative values" do
      a = V.leaf(-3.0, :a)
      assert V.relu(a).data == 0.0
    end

    test "relu at exactly zero returns zero" do
      a = V.leaf(0.0, :a)
      assert V.relu(a).data == 0.0
    end

    test "neg negates a value" do
      a = V.leaf(5.0, :a)
      assert V.neg(a).data == -5.0
    end

    test "sub produces correct difference" do
      a = V.leaf(7.0, :a)
      b = V.leaf(3.0, :b)
      assert V.sub(a, b).data == 4.0
    end

    test "divide produces correct quotient" do
      a = V.leaf(10.0, :a)
      b = V.leaf(4.0, :b)
      assert V.divide(a, b).data == 2.5
    end

    test "sum over list" do
      vals = [V.leaf(1.0, :a), V.leaf(2.0, :b), V.leaf(3.0, :c)]
      assert V.sum(vals).data == 6.0
    end

    test "mean over list" do
      vals = [V.leaf(2.0, :a), V.leaf(4.0, :b), V.leaf(6.0, :c)]
      assert V.mean(vals).data == 4.0
    end

    test "scale_data multiplies data without creating graph nodes" do
      a = V.leaf(5.0, :a)
      scaled = V.scale_data(a, 0.5)
      assert scaled.data == 2.5
      assert scaled.children == []
    end
  end

  describe "Value backward (gradients)" do
    test "d(a+b)/da == 1.0 and d(a+b)/db == 1.0" do
      a = V.leaf(3.0, :a)
      b = V.leaf(4.0, :b)
      grads = V.backward(V.add(a, b))
      assert grads[:a] == 1.0
      assert grads[:b] == 1.0
    end

    test "d(a*b)/da == b and d(a*b)/db == a" do
      a = V.leaf(3.0, :a)
      b = V.leaf(4.0, :b)
      grads = V.backward(V.mul(a, b))
      assert grads[:a] == 4.0
      assert grads[:b] == 3.0
    end

    test "d(x^2)/dx == 2x via pow" do
      x = V.leaf(5.0, :x)
      grads = V.backward(V.pow(x, 2))
      assert_in_delta grads[:x], 10.0, 1.0e-10
    end

    test "fan-out: d(a*a)/da == 2*a" do
      a = V.leaf(3.0, :a)
      result = V.mul(a, a)
      grads = V.backward(result)
      assert grads[:a] == 6.0
    end

    test "composite: d(a*b + a*c)/da == b + c" do
      a = V.leaf(2.0, :a)
      b = V.leaf(3.0, :b)
      c = V.leaf(5.0, :c)
      result = V.add(V.mul(a, b), V.mul(a, c))
      grads = V.backward(result)
      assert grads[:a] == 8.0
    end

    test "d(exp(x))/dx == exp(x)" do
      x = V.leaf(2.0, :x)
      grads = V.backward(V.exp(x))
      assert_in_delta grads[:x], :math.exp(2.0), 1.0e-10
    end

    test "d(log(x))/dx == 1/x" do
      x = V.leaf(3.0, :x)
      grads = V.backward(V.log(x))
      assert_in_delta grads[:x], 1.0 / 3.0, 1.0e-10
    end

    test "d(relu(x))/dx == 1 for positive x" do
      x = V.leaf(5.0, :x)
      grads = V.backward(V.relu(x))
      assert grads[:x] == 1.0
    end

    test "d(relu(x))/dx == 0 for negative x" do
      x = V.leaf(-5.0, :x)
      grads = V.backward(V.relu(x))
      assert grads[:x] == 0.0
    end

    test "d(a*b + c)/da == b, d/db == a, d/dc == 1" do
      a = V.leaf(2.0, :a)
      b = V.leaf(3.0, :b)
      c = V.leaf(4.0, :c)
      result = V.add(V.mul(a, b), c)
      grads = V.backward(result)

      assert grads[:a] == 3.0
      assert grads[:b] == 2.0
      assert grads[:c] == 1.0
    end

    test "d(relu(0))/dx == 0 at zero boundary" do
      x = V.leaf(0.0, :x)
      grads = V.backward(V.relu(x))
      assert grads[:x] == 0.0
    end
  end

  # ============================================================================
  # Math tests
  # ============================================================================

  describe "Math" do
    test "softmax sums to 1.0 with correct individual probabilities" do
      logits = [V.leaf(1.0, :a), V.leaf(2.0, :b), V.leaf(3.0, :c)]
      probs = Math.softmax(logits)
      prob_vals = Enum.map(probs, & &1.data)

      # Sum to 1.0
      assert_in_delta Enum.sum(prob_vals), 1.0, 1.0e-6

      # Concrete expected value for softmax([1,2,3])[2] = e^3 / (e^1 + e^2 + e^3)
      e1 = :math.exp(1.0)
      e2 = :math.exp(2.0)
      e3 = :math.exp(3.0)
      expected_p3 = e3 / (e1 + e2 + e3)
      assert_in_delta Enum.at(prob_vals, 2), expected_p3, 1.0e-6
    end

    test "softmax assigns largest probability to largest input" do
      logits = [V.leaf(1.0, :a), V.leaf(5.0, :b), V.leaf(2.0, :c)]
      probs = Math.softmax(logits)
      prob_vals = Enum.map(probs, & &1.data)
      assert Enum.at(prob_vals, 1) > Enum.at(prob_vals, 0)
      assert Enum.at(prob_vals, 1) > Enum.at(prob_vals, 2)
    end

    test "rmsnorm output has near-unit RMS" do
      x = [V.leaf(3.0, :a), V.leaf(4.0, :b), V.leaf(5.0, :c)]
      normed = Math.rmsnorm(x)

      rms =
        normed
        |> Enum.map(fn v -> v.data * v.data end)
        |> Enum.sum()
        |> then(&:math.sqrt(&1 / 3))

      assert_in_delta rms, 1.0, 0.01
    end

    test "dot product of [1,2,3] . [4,5,6] == 32.0" do
      a = [V.leaf(1.0, :a1), V.leaf(2.0, :a2), V.leaf(3.0, :a3)]
      b = [V.leaf(4.0, :b1), V.leaf(5.0, :b2), V.leaf(6.0, :b3)]
      assert Math.dot(a, b).data == 32.0
    end

    test "linear transform through identity matrix preserves input" do
      x = [V.leaf(1.0, :x1), V.leaf(2.0, :x2)]

      w = [
        [V.leaf(1.0, :w11), V.leaf(0.0, :w12)],
        [V.leaf(0.0, :w21), V.leaf(1.0, :w22)]
      ]

      [r0, r1] = Math.linear(x, w)
      assert r0.data == 1.0
      assert r1.data == 2.0
    end

    test "add_vec element-wise addition" do
      a = [V.leaf(1.0, :a1), V.leaf(2.0, :a2)]
      b = [V.leaf(3.0, :b1), V.leaf(4.0, :b2)]
      result = Math.add_vec(a, b)
      assert Enum.at(result, 0).data == 4.0
      assert Enum.at(result, 1).data == 6.0
    end

    test "relu_vec zeros negatives and passes positives" do
      x = [V.leaf(-1.0, :a), V.leaf(2.0, :b), V.leaf(0.0, :c)]
      result = Math.relu_vec(x)
      assert Enum.at(result, 0).data == 0.0
      assert Enum.at(result, 1).data == 2.0
      assert Enum.at(result, 2).data == 0.0
    end

    test "slice extracts correct sublist" do
      list = [1, 2, 3, 4, 5]
      assert Math.slice(list, 1, 3) == [2, 3, 4]
      assert Math.slice(list, 0, 2) == [1, 2]
    end
  end

  # ============================================================================
  # Tokenizer tests
  # ============================================================================

  describe "Tokenizer" do
    test "vocab_size equals unique chars plus 1 (BOS)" do
      tok = Tokenizer.build(["abc", "bcd"])
      # unique chars: a, b, c, d → 4 + 1 BOS = 5
      assert tok.vocab_size == 5
    end

    test "encode wraps document with BOS tokens" do
      tok = Tokenizer.build(["ab"])
      encoded = Tokenizer.encode(tok, "ab")
      # "ab" → [BOS, a_id, b_id, BOS] where BOS=2, a=0, b=1
      assert encoded == [2, 0, 1, 2]
    end

    test "decode recovers original text from inner tokens" do
      tok = Tokenizer.build(["hello"])
      encoded = Tokenizer.encode(tok, "hello")
      # Drop the two BOS tokens to get original back
      inner = encoded |> Enum.drop(1) |> Enum.drop(-1)
      assert Tokenizer.decode(tok, inner) == "hello"
    end

    test "decode silently drops BOS tokens" do
      tok = Tokenizer.build(["test"])
      encoded = Tokenizer.encode(tok, "test")
      # Full encoded includes BOS at both ends; decode drops them
      decoded = Tokenizer.decode(tok, encoded)
      assert decoded == "test"
    end

    test "BOS token ID is length of unique chars" do
      tok = Tokenizer.build(["xyz"])
      # x, y, z → 3 unique chars, BOS = 3
      assert tok.bos == 3
    end

    test "id_to_char provides O(1) reverse lookup" do
      tok = Tokenizer.build(["ab"])
      assert tok.id_to_char[0] == "a"
      assert tok.id_to_char[1] == "b"
      assert Map.get(tok.id_to_char, tok.bos) == nil
    end
  end

  # ============================================================================
  # RNG tests
  # ============================================================================

  describe "RNG" do
    test "same seed produces same sequence" do
      rng1 = RNG.seed(42)
      rng2 = RNG.seed(42)
      {v1, _} = RNG.uniform01(rng1)
      {v2, _} = RNG.uniform01(rng2)
      assert v1 == v2
    end

    test "uniform01 produces values in [0, 1)" do
      rng = RNG.seed(123)

      {values, _rng} =
        Enum.reduce(1..100, {[], rng}, fn _i, {acc, rng} ->
          {v, rng} = RNG.uniform01(rng)
          {[v | acc], rng}
        end)

      min_val = Enum.min(values)
      max_val = Enum.max(values)
      assert min_val >= 0.0
      assert max_val < 1.0
    end

    test "normal samples have approximately zero mean" do
      rng = RNG.seed(42)

      {values, _rng} =
        Enum.reduce(1..1000, {[], rng}, fn _i, {acc, rng} ->
          {v, rng} = RNG.normal(rng)
          {[v | acc], rng}
        end)

      mean = Enum.sum(values) / length(values)
      assert_in_delta mean, 0.0, 0.1
    end

    test "shuffle preserves all elements" do
      list = Enum.to_list(1..20)
      rng = RNG.seed(42)
      {shuffled, _rng} = RNG.shuffle(list, rng)
      assert Enum.sort(shuffled) == Enum.sort(list)
    end

    test "same seed produces same shuffle" do
      list = Enum.to_list(1..10)
      {s1, _} = RNG.shuffle(list, RNG.seed(42))
      {s2, _} = RNG.shuffle(list, RNG.seed(42))
      assert s1 == s2
    end
  end

  # ============================================================================
  # Model tests
  # ============================================================================

  # Use a consistent tiny config for all model/training/sampling tests.
  # vocab_size is set dynamically per test to match the tokenizer.
  defp small_model_cfg(vocab_size) do
    %{
      n_layer: 1,
      n_embd: 4,
      block_size: 4,
      n_head: 2,
      vocab_size: vocab_size,
      std: 0.08,
      seed: 42
    }
  end

  describe "Model" do
    test "param count matches expected formula" do
      cfg = small_model_cfg(5)
      {model, _rng} = Model.init(cfg)
      params = Model.params(model)

      # wte: 5×4=20, wpe: 4×4=16, lm_head: 5×4=20
      # per layer: attn_wq/wk/wv/wo: 4×4×4=64, mlp_fc1: 16×4=64, mlp_fc2: 4×16=64
      # total: 20 + 16 + 20 + 64 + 64 + 64 = 248
      expected = 5 * 4 + 4 * 4 + 5 * 4 + 4 * 4 * 4 + 4 * 16 + 4 * 16
      assert length(params) == expected
    end

    test "forward produces concrete logit values for token 0 at position 0" do
      cfg = small_model_cfg(5)
      {model, _rng} = Model.init(cfg)
      kv_cache = Model.empty_kv_cache(model)
      {logits, _kv_cache} = Model.gpt(model, 0, 0, kv_cache)

      # Deterministic model (seed 42) produces these exact logit values
      [l0, l1, l2, l3, l4] = logits
      assert_in_delta l0.data, -0.3343, 0.001
      assert_in_delta l1.data, -0.0290, 0.001
      assert_in_delta l2.data, 0.0804, 0.001
      assert_in_delta l3.data, -0.2385, 0.001
      assert_in_delta l4.data, 0.0114, 0.001
    end

    test "forward is deterministic given same model" do
      cfg = small_model_cfg(5)
      {model, _rng} = Model.init(cfg)
      kv_cache1 = Model.empty_kv_cache(model)
      kv_cache2 = Model.empty_kv_cache(model)

      {logits1, _} = Model.gpt(model, 0, 0, kv_cache1)
      {logits2, _} = Model.gpt(model, 0, 0, kv_cache2)

      data1 = Enum.map(logits1, & &1.data)
      data2 = Enum.map(logits2, & &1.data)
      assert data1 == data2
    end

    test "update_params modifies data values" do
      cfg = small_model_cfg(5)
      {model, _rng} = Model.init(cfg)
      params = Model.params(model)
      first_param = hd(params)

      updates = %{first_param.id => 999.0}
      model2 = Model.update_params(model, updates)
      params2 = Model.params(model2)
      assert hd(params2).data == 999.0
    end

    test "empty_kv_cache has one entry per layer" do
      cfg = small_model_cfg(5)
      {model, _rng} = Model.init(cfg)
      kv_cache = Model.empty_kv_cache(model)
      assert map_size(kv_cache) == 1
      assert kv_cache[0] == %{keys: [], values: []}
    end
  end

  # ============================================================================
  # Adam tests
  # ============================================================================

  describe "Adam" do
    test "step produces concrete expected parameter update" do
      # Single parameter: data=1.0, gradient=0.5
      param = V.leaf(1.0, :p)
      opt = Adam.init(0.01, 0.9, 0.999, 1.0e-8)
      grads = %{:p => 0.5}

      {opt2, updates} = Adam.step(opt, [param], grads, 0.01)

      # After step 1: m = 0.1*0.5 = 0.05, v = 0.001*0.25 = 0.00025
      # m_hat = 0.05 / (1-0.9) = 0.5, v_hat = 0.00025 / (1-0.999) = 0.25
      # new = 1.0 - 0.01 * 0.5 / (sqrt(0.25) + 1e-8) = 1.0 - 0.01 * 1.0 = 0.99
      assert_in_delta updates[:p], 0.99, 1.0e-6
      assert opt2.t == 1
    end

    test "step with zero gradient leaves parameter unchanged" do
      param = V.leaf(5.0, :p)
      opt = Adam.init(0.01, 0.9, 0.999, 1.0e-8)
      grads = %{:p => 0.0}

      {_opt2, updates} = Adam.step(opt, [param], grads, 0.01)

      assert_in_delta updates[:p], 5.0, 1.0e-10
    end
  end

  # ============================================================================
  # Training tests
  # ============================================================================

  describe "Trainer" do
    setup do
      tok = Tokenizer.build(["ab"])
      cfg = small_model_cfg(tok.vocab_size)
      {model, _rng} = Model.init(cfg)
      %{tok: tok, model: model}
    end

    test "loss_for_doc returns cross-entropy in expected ballpark", %{tok: tok, model: model} do
      loss = Trainer.loss_for_doc(model, tok, "ab")
      # For a randomly initialized model over vocab_size=3, expected loss ≈ ln(3) ≈ 1.099
      assert_in_delta loss.data, :math.log(tok.vocab_size), 0.7
    end

    test "loss decreases after one gradient step", %{tok: tok, model: model} do
      loss_before = Trainer.loss_for_doc(model, tok, "ab")
      grads = V.backward(loss_before)
      params = Model.params(model)
      opt = Adam.init(0.01, 0.85, 0.99, 1.0e-8)
      {_opt, updates} = Adam.step(opt, params, grads, 0.01)
      model2 = Model.update_params(model, updates)

      loss_after = Trainer.loss_for_doc(model2, tok, "ab")
      assert loss_after.data < loss_before.data
    end

    test "training over multiple steps decreases loss", %{tok: tok, model: model} do
      loss_before = Trainer.loss_for_doc(model, tok, "ab")

      {model2, _opt} =
        Trainer.train(%{
          docs: ["ab"],
          tokenizer: tok,
          model: model,
          steps: 3,
          learning_rate: 0.01,
          beta1: 0.85,
          beta2: 0.99,
          eps_adam: 1.0e-8,
          on_step: fn _step, _loss -> :ok end
        })

      loss_after = Trainer.loss_for_doc(model2, tok, "ab")
      assert loss_after.data < loss_before.data
    end
  end

  # ============================================================================
  # Sampler tests
  # ============================================================================

  describe "Sampler" do
    setup do
      tok = Tokenizer.build(["abc"])
      cfg = small_model_cfg(tok.vocab_size)
      {model, rng} = Model.init(cfg)
      %{tok: tok, model: model, rng: rng}
    end

    test "generate produces the requested number of samples", %{tok: tok, model: model, rng: rng} do
      {[s0, s1, s2], _rng} = Sampler.generate(model, tok, rng, 3, 1.0)
      # Seeded model produces deterministic samples: two empty (BOS immediately) and "c"
      assert s0 == ""
      assert s1 == ""
      assert s2 == "c"
    end

    test "all generated characters are in the vocabulary", %{tok: tok, model: model, rng: rng} do
      {samples, _rng} = Sampler.generate(model, tok, rng, 5, 1.0)
      valid_chars = MapSet.new(tok.uchars)

      generated_chars =
        samples
        |> Enum.flat_map(&String.graphemes/1)
        |> MapSet.new()

      assert MapSet.subset?(generated_chars, valid_chars)
    end

    test "seeded sampling is deterministic", %{tok: tok, model: model, rng: rng} do
      {samples1, _} = Sampler.generate(model, tok, rng, 3, 0.5)
      {samples2, _} = Sampler.generate(model, tok, rng, 3, 0.5)
      assert samples1 == samples2
    end
  end
end
