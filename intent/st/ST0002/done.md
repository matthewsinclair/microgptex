# Done — ST0002: MicroGPTEx Blog Post Series

## WP-01: Part 1 — Autograd

- [x] Write introduction and hook
- [x] Write Value struct section with code examples
- [x] Write computation graph / forward pass section
- [x] Write backward pass section with code walkthrough
- [x] Write fan-out section
- [x] Write Elixir vs Python comparison (immutable autograd)
- [x] Write threaded RNG section
- [x] Write bump test / verification section
- [x] Add diagrams (computation graph, fan-out)
- [x] Review and polish

## WP-02: Part 2 — Model

- [x] Write recap and setup
- [x] Write tokenisation section
- [x] Write math building blocks section
- [x] Write softmax and temperature section
- [x] Write model structure section (init, params, IDs)
- [x] Write forward pass section
- [x] Write Elixir idiom: lists as vectors
- [x] Add diagrams (training pairs, forward pass flow)
- [x] Review and polish

## WP-03: Part 3 — Attention

- [x] Write intuition / why attention matters
- [x] Write Q/K/V projections section
- [x] Write scaled dot-product attention section
- [x] Write multi-head attention section
- [x] Write KV cache section
- [x] Write residual connections and normalisation
- [x] Write complete transformer block section
- [x] Add diagrams (multi-head attention, KV cache growth)
- [x] Review and polish

## WP-04: Part 4 — Training and Generation

- [x] Write cross-entropy loss section
- [x] Write one gradient step section
- [x] Write Adam optimiser section with Elixir vs Python
- [x] Write params round-trip section
- [x] Write training loop section
- [x] Write autoregressive sampling section
- [x] Write full pipeline section
- [x] Write reflections / what Elixir reveals
- [x] Write try-it-yourself / links section
- [x] Add diagrams (training loop, sampling flow)
- [x] Review and polish

## WP-05: Review & Cross-references

- [x] Ensure consistent terminology across all 4 parts
- [x] Verify all code examples compile and match current source
- [x] Add cross-reference links between parts
- [x] Final read-through for flow and coherence
- [x] Verify links to Livebook notebooks and GitHub repo

## Post-Write Polish (done during WP-05)

- [x] Remove all `---` horizontal rules from all 4 posts
- [x] Convert all spelling to Australian/British English
- [x] Depersonalise all technical prose (no "I/me" in technical sections)
- [x] Change "What I've built" headings to descriptive alternatives
- [x] Amplify Karpathy/growingswe credit throughout
- [x] Add scope/tooling disclaimers to Part 1 and README
- [x] Add "Who made this?" section to Part 1
- [x] Add strings-vs-atoms aside to Part 2
- [x] Update README (soft wrapping, credits, British English)
- [x] Update Livebook notebooks with conditional Mix.install (local/GitHub fallback)
- [x] Replace HuggingFace session URLs with stable "Run in Livebook" badge URLs
