# Tasks - ST0002: MicroGPTEx Blog Post Series

## Tasks

### WP-01: Part 1 — Autograd

- [ ] Write introduction and hook
- [ ] Write Value struct section with code examples
- [ ] Write computation graph / forward pass section
- [ ] Write backward pass section with code walkthrough
- [ ] Write fan-out section
- [ ] Write Elixir vs Python comparison (immutable autograd)
- [ ] Write threaded RNG section
- [ ] Write bump test / verification section
- [ ] Add diagrams (computation graph, fan-out)
- [ ] Review and polish

### WP-02: Part 2 — Model

- [ ] Write recap and setup
- [ ] Write tokenization section
- [ ] Write math building blocks section
- [ ] Write softmax and temperature section
- [ ] Write model structure section (init, params, IDs)
- [ ] Write forward pass section
- [ ] Write Elixir idiom: lists as vectors
- [ ] Add diagrams (training pairs, forward pass flow)
- [ ] Review and polish

### WP-03: Part 3 — Attention

- [ ] Write intuition / why attention matters
- [ ] Write Q/K/V projections section
- [ ] Write scaled dot-product attention section
- [ ] Write multi-head attention section
- [ ] Write KV cache section
- [ ] Write residual connections and normalization
- [ ] Write complete transformer block section
- [ ] Add diagrams (multi-head attention, KV cache growth)
- [ ] Review and polish

### WP-04: Part 4 — Training and Generation

- [ ] Write cross-entropy loss section
- [ ] Write one gradient step section
- [ ] Write Adam optimizer section with Elixir vs Python
- [ ] Write params round-trip section
- [ ] Write training loop section
- [ ] Write autoregressive sampling section
- [ ] Write full pipeline section
- [ ] Write reflections / what Elixir reveals
- [ ] Write try-it-yourself / links section
- [ ] Add diagrams (training loop, sampling flow)
- [ ] Review and polish

### WP-05: Review & Cross-references

- [ ] Ensure consistent terminology across all 4 parts
- [ ] Verify all code examples compile and match current source
- [ ] Add cross-reference links between parts
- [ ] Final read-through for flow and coherence
- [ ] Verify links to Livebook notebooks and GitHub repo

## Dependencies

- WP-01 → WP-02 → WP-03 → WP-04 → WP-05 (sequential, each builds on previous)
- All WPs depend on ST0001 being complete (the codebase the blog explains)
