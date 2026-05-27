# Design — ST0001: MicroGPTEx

## Module Structure (As-Built)

Single file `lib/microgptex.ex` containing 9 modules, ordered bottom-up by dependency. Each module can only depend on those above it — enforcing a strict layering from pure foundations (RNG, Value) up through the algorithm (Model, Adam, Trainer) to the IO shell (Microgptex).

```
Microgptex.RNG        — Pure threaded RNG (:rand.*_s, Box-Muller, Fisher-Yates)
Microgptex.Value      — Autograd scalar node: forward ops + reverse-mode backward via gradient maps
Microgptex.Tokenizer  — Character-level tokenizer with BOS token and O(1) decode via id_to_char map
Microgptex.Math       — Vector/matrix ops on Value lists (dot, linear, softmax, rmsnorm)
Microgptex.Model      — GPT-2 struct: weight init, multi-head attention forward pass, param extraction
Microgptex.Adam       — Adam optimizer with bias correction (pure state-in/state-out)
Microgptex.Trainer    — Cross-entropy loss + training loop (pure — IO via on_step callback)
Microgptex.Sampler    — Temperature-controlled autoregressive sampling with explicit recursion
Microgptex            — Top-level API: config parsing, data download, run/1 (the IO boundary)
```

## Key Design Decisions

| Decision              | Choice                                                | Why                                                                  |
| --------------------- | ----------------------------------------------------- | -------------------------------------------------------------------- |
| Param IDs             | `{tag, row, col}` tuples                              | Stable across update cycles; Adam m/v maps survive training steps    |
| Intermediate node IDs | `make_ref()`                                          | Unique, cheap, no collisions                                         |
| RNG                   | Threaded `:rand.*_s` API                              | Pure functional — no process state, deterministic, testable          |
| IO boundary           | `on_step` callback in Trainer                         | Core logic stays pure; `run/1` provides the IO shell                 |
| Config                | `priv/config.yaml` (simple manual parser)             | No deps; key-value YAML only                                         |
| Training data         | `priv/data/input.txt` (downloaded on demand)          | Clean separation of code and data; `:httpc` is OTP-included          |
| KV cache              | Map-based `%{layer_idx => %{keys: [], values: []}}`   | O(log n) access, pattern-matchable, idiomatic Elixir                 |
| Softmax stability     | Subtract max `.data` as float constant                | Numerically correct; max not differentiated (no graph bloat)         |
| List-as-vector        | `[%Value{}]` for vectors, `[[%Value{}]]` for matrices | Matches Python original; O(n) access acceptable at pedagogical scale |
| Struct enforcement    | `@enforce_keys` on all public structs                 | Compile-time validation prevents missing-field bugs                  |
| Tokenizer decode      | `id_to_char` map alongside `char_to_id`               | O(1) decode vs O(n) `Enum.find` on reversed map                      |
| Documentation         | Extensive `@moduledoc`/`@doc` on all public functions | Dual-audience: GPT algorithm concepts + Elixir idioms                |

## Architecture Highlights

### Immutable Autograd

Python's `Value.backward()` mutates `.grad` in-place. Elixir's `backward/1` returns a `%{id => gradient}` map built by folding over topologically sorted nodes (root-first). Gradient accumulation via `Map.update/4` handles fan-out (same value used multiple times, e.g., `a * a` yields `d/da = 2a`).

The topological sort uses DFS with a visited set. Nodes are prepended (`[v | topo]`), producing root-first order. Backward iterates this directly — processing the root first guarantees every node's gradient is available before its children need it.

### Params Round-Trip

```
Model.params/1  →  flat list of %Value{id: {tag, row, col}}
      ↓
Value.backward/1  →  %{id => gradient} map
      ↓
Adam.step/4  →  {new_opt, %{id => new_data_value}}
      ↓
Model.update_params/2  →  model with updated Value.data fields
```

The `{tag, row, col}` ID scheme ensures Adam's moment maps (`m`, `v`) survive across training steps — same parameter always has the same ID.

### KV Cache Threading

The KV cache is a `%{layer_idx => %{keys: [[Value]], values: [[Value]]}}` map. Each call to `Model.gpt/4` takes `(model, token_id, pos_id, kv_cache)` and returns `{logits, updated_kv_cache}`. The Trainer's loss loop and Sampler's generation loop both thread the cache through `Enum.reduce/3`.

### IO Boundary

All eight inner modules are pure — no IO, no side effects. The only module that does IO is `Microgptex` (the top-level API), which:

- Reads config from `priv/config.yaml`
- Downloads training data via `:httpc`
- Passes an `on_step` callback that calls `IO.write/1` with iodata
- Prints generated samples

This makes the entire core testable without mocking IO.

## Alternatives Considered

1. **Multi-file module layout** — Rejected: single file is more pedagogical, matches the spirit of Karpathy's compact gist
2. **Nx/EXLA for tensor ops** — Rejected: the whole point is scalar-only autograd to show the algorithm at its most explicit
3. **Process-based RNG** — Rejected: breaks purity and determinism; threaded `_s` API is cleaner
4. **GenServer for training state** — Rejected: unnecessary for a single-threaded trainer; plain `Enum.reduce/3` is clearer
5. **`acc ++ [item]` list building** — Rejected during audit: O(n^2); replaced with prepend-and-reverse throughout
6. **`Enum.at` for tokenizer decode** — Rejected during audit: O(n); added `id_to_char` map for O(1) lookup
