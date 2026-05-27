---
verblock: "02 Mar 2026:v0.1: matts - Initial version"
wp_id: WP-02
title: "Comprehensive Test Suite"
scope: Medium
status: Done
---

# WP-02: Comprehensive Test Suite

## Objective

Write behavioral tests proving that the GPT training algorithm works correctly end-to-end: autograd computes correct gradients, the forward pass produces deterministic results, training converges, and sampling generates valid output.

## What the Tests Prove

- **Autograd is correct** — each forward op (add, mul, pow, log, exp, relu) produces the right value, and `backward/1` computes the right gradients. Fan-out accumulation (`a*a → d/da = 2a`) is verified. Composite expressions (`a*b + a*c → d/da = b+c`) are verified with concrete expected values.
- **Math operations are numerically sound** — softmax sums to 1.0 with the correct individual probabilities, RMSNorm produces near-unit RMS, dot products match hand-computed values.
- **Tokenizer round-trips** — encode produces the exact expected token sequence (`"ab" → [2, 0, 1, 2]`), decode recovers the original text, BOS tokens are correctly placed.
- **Model forward pass is deterministic** — same seed, same model, same input produces bitwise-identical logit values. Concrete logit values are asserted (`l0 ≈ -0.3343`).
- **Adam produces correct updates** — a single Adam step with known inputs produces a specific expected parameter value (`≈ 0.99`). Zero gradients leave parameters unchanged.
- **Training converges** — loss decreases after one gradient step, and continues decreasing over multiple steps.
- **Sampling is deterministic and vocabulary-sound** — seeded sampling produces identical results across runs. All generated characters belong to the training vocabulary. Concrete sample values are asserted.

## Testing Principles

- **Concrete assertions only** — every test asserts specific numeric values, not shapes or types
- **No control flow in test bodies** — straight-line setup → action → assert
- **One focus per test** — each test verifies one behavioral property
- **Public API only** — tests use module functions, never internal state
- **Real code, no mocks** — all tests use actual modules
- **Shared setup via `setup` blocks** — common initialization in describe-level setup, not duplicated per test

## Acceptance Criteria

- [x] All behavioral properties proven with concrete expected values
- [x] No shape-only tests (no `is_struct`, `is_map`, `is_binary` assertions)
- [x] No control flow in test bodies
- [x] `@moduledoc` tag present on test module

## Dependencies

- Depends on WP-01 (tests need implementation)
