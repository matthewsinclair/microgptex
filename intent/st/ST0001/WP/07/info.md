---
verblock: "02 Mar 2026:v0.1: matts - Initial version"
wp_id: WP-07
title: "LiveBook Walkthrough"
scope: Large
status: Not Started
---

# WP-07: LiveBook Walkthrough

## Objective

Create an interactive Livebook notebook that walks through the MicroGPTEx implementation step by step, in the style of [growingswe.com/blog/microgpt](https://growingswe.com/blog/microgpt). The notebook should be a standalone learning resource that explains the GPT training algorithm while letting the reader execute each piece interactively.

## Deliverables

- Livebook notebook file (`.livemd`) covering the full algorithm walkthrough
- Sections: autograd, tokenization, model architecture, forward pass, backward pass, Adam optimizer, training loop, sampling

## Acceptance Criteria

- [ ] Notebook runs end-to-end in Livebook without errors
- [ ] Each section is self-contained with explanatory text + executable code
- [ ] Visual output where appropriate (loss curves, generated samples)
- [ ] Style matches growingswe.com/blog/microgpt narrative approach

## Dependencies

- Depends on WP-06 (all code work must be complete first)
