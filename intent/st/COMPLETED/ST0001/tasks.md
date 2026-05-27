# Tasks — ST0001: MicroGPTEx

## Completed

- [x] WP-01: Core implementation — 9-module GPT trainer with scalar autograd, multi-head attention, Adam, and autoregressive sampling
- [x] WP-02: Comprehensive test suite — behavioral tests proving autograd correctness, training convergence, and sampling determinism
- [x] WP-03: Elixir idiom audit — remediated all findings from `/intent-elixir-essentials` and `/intent-elixir-testing` audits
- [x] WP-04: Socratic code review — CTO/Tech Lead dialog comparing Elixir vs Python approaches, all findings fixed
- [x] WP-05: Public repo polish — README for public consumption, removed reference implementation files

- [x] WP-06: Rich moduledoc text — dual-audience `@moduledoc` explaining both GPT concepts and Elixir idioms
- [x] WP-07: LiveBook walkthrough — interactive notebook in growingswe.com/blog/microgpt style

## Remaining

- [ ] WP-08: Mermaid diagrams — structural diagrams in the Livebook (computation graphs, model architecture, training loop, attention flow)
- [ ] WP-09: Interactive visualizations — Kino + VegaLite widgets (softmax explorer, temperature slider, live loss curve, attention heatmap)

## Quality Gates

- [x] `mix compile` — clean
- [x] `mix test` — all passing
- [x] `mix format --check-formatted` — clean
- [x] `mix credo --strict` — clean

## Dependencies

- WP-02 depends on WP-01 (tests need implementation)
- WP-03 depends on WP-01 + WP-02 (audit needs code + tests)
- WP-04 depends on WP-01 + WP-02 (review needs code + tests)
- WP-05 depends on WP-01 through WP-04 (polish after all fixes)
- WP-06 depends on WP-05 (moduledoc after code is stable)
- WP-07 depends on WP-06 (LiveBook after all code work is done)
- WP-08 depends on WP-07 (diagrams enhance the existing walkthrough)
- WP-09 depends on WP-07 + WP-08 (interactivity builds on diagrams; some Mermaid may become Kino.Mermaid)
