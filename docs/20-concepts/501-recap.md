---
verblock: "26 May 2026:v0.1: Matthew Sinclair - Recap"
---

# The Whole Picture

## Recap · 20 Concepts To Understand AI in 2026

Twenty concepts, taken one at a time, can feel like twenty separate facts. They are not. They are one story told in four movements, each building on the last. This recap reassembles them.

### Part 1 gave us the machine

[Neural networks](101-neural-networks.md) are the substrate: layers of weighted connections, tuned by gradient descent, with all the "knowledge" living in the weights. Everything that follows is a way of feeding that machine or arranging it. Text cannot enter as text, so [tokenisation](102-tokenisation.md) cuts it into sub-word pieces, and [embeddings](103-embeddings.md) turn those pieces into vectors whose geometry encodes meaning. [Attention](104-attention.md) lets every token weigh every other, resolving meaning from context, and the [transformer](105-transformers.md) makes attention the whole architecture, processing a sequence in parallel rather than one step at a time. Foundations: how raw text becomes something a network can compute over.

### Part 2 showed us the behaviour

Scale that machine on next-token prediction and you get a [large language model](206-llms.md). Its quirks follow directly from how it works. The [context window](207-context-window.md) is the working memory it can attend to at once. [Temperature](208-temperature.md) is the dial between safe and surprising, governing how it samples from its own probabilities. [Hallucination](209-hallucination.md) is not a fault bolted on but a consequence of predicting plausible tokens rather than retrieving truth. And because framing changes the output, [prompt engineering](210-prompt-engineering.md) — clear instructions, examples, step-by-step requests — is a genuine lever rather than a trick.

### Part 3 showed us how raw models become useful

A pretrained model is capable but generic. [Transfer learning](311-transfer-learning.md) reuses its broad competence; [fine-tuning](312-fine-tuning.md) specialises it on focused data, including the instruction-tuning that makes it follow directions. [RLHF](313-rlhf.md), and its simpler successor DPO, aligns it with what people actually prefer — the step that separates "technically capable" from "genuinely helpful". Because all of this is expensive, [LoRA](314-lora.md) makes adaptation cheap by training tiny add-on layers, and [quantisation](315-quantisation.md) shrinks models to run on modest hardware.

### Part 4 showed us the real systems

The products people use are rarely a bare model. [RAG](416-rag.md) grounds answers in retrieved documents, directly countering hallucination, and it leans on [vector databases](417-vector-databases.md) that search by meaning using the embeddings from Part 1. [AI agents](418-ai-agents.md) turn a responder into a doer: plan, act, observe, adjust, repeat. [Chain of thought](419-chain-of-thought.md) makes reasoning explicit, and measurably better. And [diffusion models](420-diffusion-models.md) show the same generative leap arriving through a completely different mechanism, building images by reversing noise rather than predicting tokens.

### One picture

Read top to bottom, the through-line is simple. A neural network learns patterns from data (Part 1); scaled on language, it becomes an LLM with characteristic behaviours (Part 2); targeted training makes it useful and affordable (Part 3); and surrounding it with retrieval, tools, and reasoning turns it into the systems people actually use (Part 4). None of the twenty ideas is magic. Each is a piece of engineering, and together they explain how the tools on your screen really work.

Every source cited across these chapters is collected, and verified, in the [references](502-references.md).
