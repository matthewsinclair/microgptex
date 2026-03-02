---
verblock: "02 Mar 2026:v0.1: matts - Initial version"
wp_id: WP-03
title: "Elixir Idiom Audit and Remediation"
scope: Medium
status: Done
---

# WP-03: Elixir Idiom Audit and Remediation

## Objective

Audit the codebase against `/intent-elixir-essentials` and `/intent-elixir-testing` skill rules, then fix all findings to ensure the code is idiomatic Elixir.

## Audits Performed

1. **Elixir Essentials audit** — multi-clause pattern matching, pipe usage, naming conventions, assertive data access, tagged tuples, debug artifacts
2. **Elixir Testing audit** — strong assertions, no control flow in tests, one focus per test, domain contracts, no mocks, no shape tests

## Code Findings Fixed

| Finding                             | Location                      | Fix                                        |
| ----------------------------------- | ----------------------------- | ------------------------------------------ |
| `acc ++ [item]` O(n^2) list concat  | Multiple locations            | `[item \| acc]` + `Enum.reverse/1`         |
| `Enum.at` for tokenizer decode O(n) | `Tokenizer.decode/2`          | Added `id_to_char` map for O(1) lookup     |
| Missing `@enforce_keys`             | Value, Tokenizer, Model, Adam | Added for compile-time validation          |
| `Enum.map \|> Enum.join`            | `Tokenizer.decode/2`          | Combined into `Enum.map_join/3`            |
| Excessive nesting in `attn_block`   | `Model` module                | Extracted `weighted_sum/3` helper          |
| `length(x) > 0` guard               | `Math.softmax/1`              | `[_ \| _] = logits` pattern match          |
| `unless` (deprecated)               | `Microgptex.ensure_data/2`    | `if not File.exists?(path)`                |
| Untyped config maps                 | `Microgptex` module           | `@type model_config`, `@type train_config` |
| `parse_value` double-parse          | Config parser                 | Single-pass with pattern matching          |

## Test Findings Fixed

| Finding                            | Fix                                            |
| ---------------------------------- | ---------------------------------------------- |
| Shape-only assertions              | Concrete value assertions                      |
| Nested `Enum.each` in sampler test | `MapSet.subset?`                               |
| Missing Adam direct tests          | Concrete expected-value tests                  |
| Missing Math helper tests          | `linear`, `add_vec`, `relu_vec`, `slice` tests |
| Shared `@small_cfg` mismatch       | `small_model_cfg(vocab_size)` function         |
| Missing relu@zero boundary         | Added with gradient assertion                  |
| Weak training assertions           | Assert loss decreases, ballpark cross-entropy  |

## Acceptance Criteria

- [x] All `/intent-elixir-essentials` rules pass
- [x] All `/intent-elixir-testing` rules pass
- [x] `mix credo --strict` — 0 issues
- [x] 56 tests pass after all changes

## Dependencies

- Depends on WP-01 + WP-02
