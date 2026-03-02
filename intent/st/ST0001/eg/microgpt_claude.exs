# The most atomic way to train and run inference for a GPT in pure, dependency-free Elixir.
# This file is the complete algorithm. Everything else is just efficiency.
#
# Translated from @karpathy's Python implementation.
# For real work, use Nx + EXLA. This is the pedagogical version.

# Let there be order among chaos
:rand.seed(:exsss, {42, 42, 42})

# -----------------------------------------------------------------------------
# Let there be Autograd to recursively apply the chain rule through a computation graph
# -----------------------------------------------------------------------------

defmodule MicroGPT.Value do
  @moduledoc "Autograd-enabled scalar with computation graph tracking."

  @type t :: %__MODULE__{
          id: reference(),
          data: float(),
          children: [t()],
          local_grads: [float()]
        }

  defstruct [:id, :data, children: [], local_grads: []]

  @spec new(number()) :: t()
  def new(data), do: %__MODULE__{id: make_ref(), data: data * 1.0}

  @spec add(t(), t()) :: t()
  def add(%__MODULE__{} = a, %__MODULE__{} = b) do
    %__MODULE__{
      id: make_ref(),
      data: a.data + b.data,
      children: [a, b],
      local_grads: [1.0, 1.0]
    }
  end

  @spec mul(t(), t()) :: t()
  def mul(%__MODULE__{} = a, %__MODULE__{} = b) do
    %__MODULE__{
      id: make_ref(),
      data: a.data * b.data,
      children: [a, b],
      local_grads: [b.data, a.data]
    }
  end

  @spec pow_val(t(), number()) :: t()
  def pow_val(%__MODULE__{data: data} = v, exponent) do
    %__MODULE__{
      id: make_ref(),
      data: :math.pow(data, exponent),
      children: [v],
      local_grads: [exponent * :math.pow(data, exponent - 1)]
    }
  end

  @spec log_val(t()) :: t()
  def log_val(%__MODULE__{data: data} = v) do
    %__MODULE__{id: make_ref(), data: :math.log(data), children: [v], local_grads: [1.0 / data]}
  end

  @spec exp_val(t()) :: t()
  def exp_val(%__MODULE__{data: data} = v) do
    e = :math.exp(data)
    %__MODULE__{id: make_ref(), data: e, children: [v], local_grads: [e]}
  end

  @spec relu(t()) :: t()
  def relu(%__MODULE__{data: data} = v) do
    %__MODULE__{
      id: make_ref(),
      data: max(0.0, data),
      children: [v],
      local_grads: [if(data > 0, do: 1.0, else: 0.0)]
    }
  end

  @spec neg(t()) :: t()
  def neg(v), do: scale(v, -1.0)

  @spec sub(t(), t()) :: t()
  def sub(a, b), do: add(a, neg(b))

  @spec div_val(t(), t()) :: t()
  def div_val(a, b), do: mul(a, pow_val(b, -1))

  @spec scale(t(), number()) :: t()
  def scale(v, factor), do: mul(v, new(factor))

  @spec dot([t()], [t()]) :: t()
  def dot(xs, ys) do
    Enum.zip(xs, ys)
    |> Enum.map(fn {x, y} -> mul(x, y) end)
    |> sum_values()
  end

  @spec sum_values([t()]) :: t()
  def sum_values([sing
