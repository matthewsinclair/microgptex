# What If Numbers Could Remember?

## Autograd in Elixir — Part 1 of Building GPT from Scratch

You can train a GPT from scratch in ~1500 lines of Elixir. No Nx, no external dependencies. Just pure functions, pattern matching, and a data structure that remembers every calculation.

[MicroGPTEx](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) is a faithful translation of [Andrej Karpathy's](https://karpathy.github.io/) [MicroGPT](https://karpathy.github.io/2026/02/12/microgpt/) into idiomatic Elixir. The original Python implementation is ~200 lines and trains a tiny character-level GPT on human names. The Elixir version does the same thing — but the translation reveals things about the algorithm that Python hides behind mutation.

This is Part 1 of a 4-part series. I'll build understanding bottom-up, the same way the code is organized:

- [Part 1](./part1-autograd.md): **Autograd** — how the model learns from its mistakes (this post)
- [Part 2](./part2-model.md): **Text, math, and the model** — tokenization, building blocks, and the GPT architecture
- [Part 3](./part3-attention.md): **Attention** — how tokens talk to each other
- [Part 4](./part4-training.md): **Training and generation** — making it learn, making it create

Credit where it's due: the original Python code is Karpathy's, and the excellent interactive walkthrough at [growingswe.com](https://growingswe.com/blog/microgpt) is what inspired this Elixir translation. If you haven't read that walkthrough, it's worth your time regardless of which language you prefer.

This whole thing came about because I read Karpathy's original and asked myself: _"Do I really know how a GPT works?"_ The disturbing answer was an unequivocal "_No!_" So I set out to do two things. First, recreate it in Elixir and bring out the best of a pure-functional interpretation. Second, create something other people could use to build their own understanding. Karpathy's original is 200 lines of _code-as-art_ Python; this version is longer, but that's mainly due to substantially more exposition — favouring explanation over brevity.

### The architecture at a glance

MicroGPTEx is nine modules in a single file, ordered bottom-up by dependency:

```
RNG → Value → Tokenizer → Math → Model → Adam → Trainer → Sampler → Microgptex
```

Each module can only depend on the ones before it. The bottom eight are pure — no side effects, no IO. The top-level `Microgptex` module is the only one that touches the outside world.

In this post, I'll focus on the first two: `RNG` and `Value`. They're the foundation that everything else is built on.

## Every number remembers where it came from

The fundamental question of training a neural network is: _if I change this weight by a tiny amount, how much does the loss change?_ The answer is called the **gradient** — the derivative of the loss with respect to that weight.

Computing gradients by hand for thousands of parameters would be impractical. Automatic differentiation (autograd) does it for you by tracking every operation in a computation graph, then walking that graph backward to propagate gradients via the chain rule.

In MicroGPTEx, the autograd engine is the `Value` struct:

```elixir
defstruct [:data, :id, children: [], local_grads: []]
```

Four fields. That's the entire data structure:

- **`data`** — the scalar float value (the result of the forward computation)
- **`id`** — a unique identifier for this node
- **`children`** — the Value nodes that were inputs to the operation that produced this one
- **`local_grads`** — the partial derivatives of this node's output with respect to each child

Every number in the neural network — every weight, every activation, every loss value — is one of these nodes. When you perform an operation, you get back a new Value that remembers what produced it.

Here it is in action:

```elixir
alias Microgptex.Value, as: V

a = V.leaf(3.0, :a)
b = V.leaf(4.0, :b)
c = V.mul(a, b)

c.data         #=> 12.0
c.children     #=> [a, b]
c.local_grads  #=> [4.0, 3.0]
```

That last line is the key. The `local_grads` are `[4.0, 3.0]` because:

- `d(a * b) / da = b = 4.0`
- `d(a * b) / db = a = 3.0`

These are the chain rule factors — computed eagerly during the forward pass and stored right on the node. When I later need to compute gradients, they're already there.

Here's the multiplication operation in full:

```elixir
def mul(a, b) do
  a = wrap(a)
  b = wrap(b)

  %Value{
    data: a.data * b.data,
    id: make_ref(),
    children: [a, b],
    local_grads: [b.data, a.data]
  }
end
```

Every other operation follows the same pattern — compute the result, record the children, store the partial derivatives:

| Operation   | Formula   | Local gradients            |
| ----------- | --------- | -------------------------- |
| `add(a, b)` | a + b     | `[1.0, 1.0]`               |
| `mul(a, b)` | a \* b    | `[b, a]`                   |
| `pow(a, p)` | a^p       | `[p * a^(p-1)]`            |
| `log(a)`    | ln(a)     | `[1/a]`                    |
| `exp(a)`    | e^a       | `[e^a]`                    |
| `relu(a)`   | max(0, a) | `[1.0 if a > 0, else 0.0]` |

These six operations — plus negation, subtraction, and division built on top of them — are sufficient to implement the entire GPT algorithm.

## Building a computation graph

Here's something that looks like a real neural network computation: a tiny neuron with weights, an input, and an activation function.

```elixir
# A tiny "neuron": relu(w1 * x + w2)
x  = V.leaf(2.0, :x)
w1 = V.leaf(0.5, :w1)
w2 = V.leaf(-0.3, :w2)

weighted = V.add(V.mul(w1, x), w2)  # w1*x + w2 = 0.7
output = V.relu(weighted)           # relu(0.7) = 0.7
```

Each line builds a node in a directed acyclic graph. The leaf nodes (`x`, `w1`, `w2`) are inputs. The operation nodes (`mul`, `add`, `relu`) remember their children and store the chain rule factors:

```
w1 = 0.5 ─────┐
              ├── × → 1.0 ───┐
x  = 2.0 ─────┘              ├── add → 0.7 ──── ReLU → 0.7
                             │
w2 = -0.3 ───────────────────┘
```

Every operation in a neural network — every weight times every input, every activation function, every loss computation — builds this graph. The graph is the complete record of "how did we get this answer?"

## The backward pass

Now the real magic. Given an output node, `backward/1` traverses the graph in reverse and computes `d(output)/d(every node)`.

```elixir
grads = V.backward(output)

grads[:x]   #=> 0.5   — "if x increases by 1, output increases by 0.5"
grads[:w1]  #=> 2.0   — "if w1 increases by 1, output increases by 2.0"
grads[:w2]  #=> 1.0   — "if w2 increases by 1, output increases by 1.0"
```

These gradients are exactly what the optimizer needs. They answer: "which direction should I adjust each weight to reduce the loss?"

Here's the complete `backward/1` implementation:

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

That's ~12 lines for the entire backward pass. Here's what it does:

1. **Topological sort** — order the nodes from root to leaves (DFS, root first)
2. **Seed the root** — `d(output)/d(output) = 1.0`
3. **Fold through the sorted nodes** — for each node, propagate its gradient to its children via the chain rule: `child_grad += local_grad * node_grad`

The result is a `%{id => gradient}` map — an immutable data structure containing the gradient of the output with respect to every node in the graph.

Tracing through the neuron example, starting from `output` (the relu node) with gradient 1.0:

- **relu node** (grad = 1.0): relu's local gradient is 1.0 (because 0.7 > 0), so the add node gets `1.0 * 1.0 = 1.0`
- **add node** (grad = 1.0): addition's local gradients are `[1.0, 1.0]`, so the mul node gets `1.0 * 1.0 = 1.0` and w2 gets `1.0 * 1.0 = 1.0`
- **mul node** (grad = 1.0): multiplication's local gradients are `[b, a]` = `[2.0, 0.5]`, so w1 gets `1.0 * 2.0 = 2.0` and x gets `1.0 * 0.5 = 0.5`

The chain rule factors stored during the forward pass are simply multiplied together during the backward pass. That's all backpropagation is.

## Fan-out: when the same value is used twice

What happens when a Value appears in multiple places in the graph? Consider `a * a`:

```elixir
a = V.leaf(3.0, :a)
result = V.mul(a, a)    # a^2 = 9.0

grads = V.backward(result)
grads[:a]  #=> 6.0     (= 2 * a = 2 * 3.0)
```

Node `a` feeds _both_ inputs to the multiplication. The gradient from the left edge is `a.data = 3.0` and from the right edge is also `a.data = 3.0`. These must be **summed** to give the correct gradient of `2a = 6.0`.

This is fan-out, and it's handled by a single line in `backward/1`:

```elixir
Map.update(acc, child.id, local * node_grad, &(&1 + local * node_grad))
```

`Map.update/4` says: if this child's ID is already in the gradient map, _add_ the new gradient to the existing one. If it's the first time we've seen this child, store the gradient directly.

Fan-out appears constantly in real neural networks — any time the same weight matrix is used for multiple inputs, any time a value is reused in a residual connection. Getting the accumulation right is critical for correct training.

## The Elixir difference: immutable autograd

This is where the Elixir translation reveals something that Python hides.

In Python's [micrograd](https://github.com/karpathy/micrograd) (Karpathy's earlier autograd engine), `backward()` mutates each node's `.grad` field in-place:

```python
# Python (micrograd)
class Value:
    def __init__(self, data):
        self.data = data
        self.grad = 0.0          # mutable!
        self._backward = lambda: None

    # In backward():
    for v in reversed(topo):
        v._backward()            # mutates v.grad via +=
```

The `_backward` lambda captures the forward-pass variables and uses `+=` to accumulate gradients through shared mutable references. Fan-out works because Python's `+=` silently adds to the existing value. But you have to know that `loss.backward()` is a side effect — it doesn't return anything, it mutates every node in the graph.

In Elixir, there's no mutation. `backward/1` returns a gradient map:

```elixir
# Elixir (MicroGPTEx)
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

The gradient map is a value you can hold, print, compare, or pass to multiple consumers. The fan-out accumulation is visible in `Map.update/4` — not hidden behind mutation order.

There's a second difference worth noting. Python stores backward logic as closures — `_backward` lambdas that capture the forward-pass variables and execute later. You can't inspect what a closure will do until you call it. Elixir stores the chain rule factors eagerly as `local_grads` data on the node. They're inspectable at any time:

```elixir
c = V.mul(a, b)
c.local_grads  #=> [b.data, a.data] — right there, no execution needed
```

The same math, the same algorithm, the same results. But the data flow is visible in the code itself.

## Threaded RNG: determinism by construction

Before I move on from the foundation, there's one more module worth examining: `RNG`. Neural network training uses randomness in three places:

1. **Weight initialization** — random starting values for all parameters
2. **Training data shuffling** — randomise the order of training examples
3. **Sampling** — draw from the model's predicted probability distribution

In Python, you'd use `random.random()` or `numpy.random`, which maintain hidden global state. If you want reproducibility, you call `random.seed(42)` at the start and hope nothing else touches the global RNG in between.

MicroGPTEx threads the RNG state explicitly through every function:

```elixir
rng = RNG.seed(42)

# Draw a normal sample — returns {value, new_rng}
{weight, rng} = RNG.normal(rng, 0.0, 0.08)

# Shuffle a list — returns {shuffled_list, new_rng}
{docs, rng} = RNG.shuffle(docs, rng)

# Draw a uniform sample — returns {float, new_rng}
{u, rng} = RNG.uniform01(rng)
```

Every function takes an RNG state and returns `{result, new_rng}`. The `rng` variable is rebound at each step, threading the state through the computation. Under the hood, it's Erlang's `:rand.uniform_s/1` — the `_s` suffix means "stateful" (takes and returns state, rather than using process dictionary).

This pattern has a structural advantage. In Python:

```python
random.seed(42)
# ... 500 lines later, did something call random.random() in between?
# You have to check every function you called.
weight = random.gauss(0, 0.08)
```

In Elixir, the state is in the variable. You can see it flow:

```elixir
rng = RNG.seed(42)
{weight1, rng} = RNG.normal(rng, 0.0, 0.08)
{weight2, rng} = RNG.normal(rng, 0.0, 0.08)
# rng has been advanced exactly twice — this is visible in the code
```

Same seed, same training run. Not because you remembered to call `random.seed()`, but because the types enforce it. The threading is admittedly verbose — every function that needs randomness must accept and return the RNG state. But the payoff is that deterministic reproducibility is guaranteed by construction. No global state to corrupt, no test ordering sensitivity, no seeds to forget.

## Verifying autograd: the bump test

How do I know the chain rule factors are correct? I verify by comparing autograd against numerical differentiation — the "bump test." Nudge a value by a tiny epsilon and measure how the output changes:

```elixir
epsilon = 1.0e-5

a = V.leaf(2.0, :a)
b = V.leaf(3.0, :b)
expr = fn a, b -> V.add(V.mul(a, b), V.pow(a, 2)) end

# Autograd gradient
grads = V.backward(expr.(a, b))
autograd_da = grads[:a]    #=> 7.0  (= b + 2*a = 3 + 4)

# Numerical gradient: (f(a+eps) - f(a-eps)) / (2*eps)
f_plus  = expr.(V.leaf(2.0 + epsilon, :a), b).data
f_minus = expr.(V.leaf(2.0 - epsilon, :a), b).data
numerical_da = (f_plus - f_minus) / (2 * epsilon)
#=> 6.999999...

abs(autograd_da - numerical_da) < 1.0e-6  #=> true
```

The expression is `a*b + a^2`. The derivative with respect to `a` is `b + 2a` = `3 + 4 = 7`. Autograd gives us 7.0 exactly. The numerical approximation gives us 6.9999-something. They match.

This is the standard sanity check for any autograd implementation. If the analytical and numerical gradients diverge, there's a bug in the chain rule factors. MicroGPTEx's test suite uses this technique across all operations.

## The foundation so far

In ~250 lines of Elixir, the autograd foundation is in place:

- A **Value struct** that tracks every computation in a directed acyclic graph
- A **backward pass** that computes gradients for all parameters via the chain rule, returning an immutable `%{id => gradient}` map
- **Fan-out handling** via `Map.update/4` — explicit gradient accumulation where Python uses implicit mutation
- A **threaded RNG** that guarantees deterministic reproducibility by construction

This mirrors what Karpathy's [micrograd](https://github.com/karpathy/micrograd) does in Python — but with immutable data structures instead of mutation. Every weight in the neural network will be a Value node. Every forward pass will build a computation graph. Every training step will call `backward/1` to get gradients. The optimizer will use those gradients to adjust the weights. And the whole thing will be reproducible because the RNG state flows explicitly through every function.

## Up next

[Part 2: "From Letters to Logits"](part2-model.md) converts text into numbers, builds the mathematical operations that make up neural network layers, and assembles the GPT model architecture — following the same structure as Karpathy's original, but with stable parameter IDs that make the training loop work in a functional language.
