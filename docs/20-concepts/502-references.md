---
verblock: "26 May 2026:v0.2: Matthew Sinclair - Full references"
---

# References

Every source cited across the twenty chapters, grouped by chapter, plus the primary attribution for the series. Every link was checked to resolve, and confirmed to support the claim it is attached to, at the time of writing. Nothing here is included unverified.

## Primary attribution

- **Concept list** — adapted from a thread by @sairahul1 on X, read via xcancel: [part one](https://xcancel.com/sairahul1/status/2057740928908161461?s=43) and [part two](https://xcancel.com/sairahul1/status/2058103035969351896?s=43).
- **Anchor reference** — Yann Dubois, _Building Large Language Models_, Stanford CS229 guest lecture (Stanford Online): [youtube.com/watch?v=9vM4p9NN0Ts](https://www.youtube.com/watch?v=9vM4p9NN0Ts).

## Part 1 — How AI actually works

### 101 · Neural Networks

- 3Blue1Brown, _But what is a Neural Network?_ (Deep learning, chapter 1) — [youtube.com/watch?v=aircAruvnKk](https://www.youtube.com/watch?v=aircAruvnKk). Visual intuition for layers, weights, and activations.
- Goodfellow, Bengio & Courville, _Deep Learning_ (MIT Press, free online) — [deeplearningbook.org](https://www.deeplearningbook.org). The standard textbook for the underlying mathematics.
- Brown et al., _Language Models are Few-Shot Learners_ (2020) — [arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165). The GPT-3 paper; source for the verified 175-billion-parameter figure.
- Yann Dubois, _Building Large Language Models_, Stanford CS229 — [youtube.com/watch?v=9vM4p9NN0Ts](https://www.youtube.com/watch?v=9vM4p9NN0Ts).

### 102 · Tokenisation

- Sennrich, Haddow & Birch, _Neural Machine Translation of Rare Words with Subword Units_ (2016) — [arxiv.org/abs/1508.07909](https://arxiv.org/abs/1508.07909). Introduces byte-pair encoding (BPE) for sub-word tokenisation.
- Andrej Karpathy, _Let's build the GPT Tokenizer_ — [youtube.com/watch?v=zduSFxRajkE](https://www.youtube.com/watch?v=zduSFxRajkE). End-to-end build of a BPE tokeniser.
- OpenAI, _tiktoken_ — [github.com/openai/tiktoken](https://github.com/openai/tiktoken). The fast BPE tokeniser used by OpenAI's models.

### 103 · Embeddings

- Mikolov et al., _Efficient Estimation of Word Representations in Vector Space_ (2013) — [arxiv.org/abs/1301.3781](https://arxiv.org/abs/1301.3781). The word2vec paper; meaning as position in vector space.
- Jay Alammar, _The Illustrated Word2vec_ — [jalammar.github.io/illustrated-word2vec](https://jalammar.github.io/illustrated-word2vec/). Visual explainer for embeddings.

### 104 · Attention

- Bahdanau, Cho & Bengio, _Neural Machine Translation by Jointly Learning to Align and Translate_ (2015) — [arxiv.org/abs/1409.0473](https://arxiv.org/abs/1409.0473). An early attention mechanism for sequence models.
- Vaswani et al., _Attention Is All You Need_ (2017) — [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762). Query-key-value and multi-head attention.
- Jay Alammar, _The Illustrated Transformer_ — [jalammar.github.io/illustrated-transformer](https://jalammar.github.io/illustrated-transformer/). Visual walkthrough of the attention mechanics.

### 105 · Transformers

- Vaswani et al., _Attention Is All You Need_ (2017) — [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762). The transformer architecture.
- Phuong & Hutter, _Formal Algorithms for Transformers_ (2022) — [arxiv.org/abs/2207.09238](https://arxiv.org/abs/2207.09238). Precise, self-contained pseudocode for transformers.
- Yann Dubois, _Building Large Language Models_, Stanford CS229 — [youtube.com/watch?v=9vM4p9NN0Ts](https://www.youtube.com/watch?v=9vM4p9NN0Ts).

## Part 2 — How LLMs work

### 206 · Large Language Models

- Brown et al., _Language Models are Few-Shot Learners_ (2020) — [arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165). GPT-3; next-token prediction at scale.
- Hoffmann et al., _Training Compute-Optimal Large Language Models_ (2022) — [arxiv.org/abs/2203.15556](https://arxiv.org/abs/2203.15556). The Chinchilla scaling result.
- Yann Dubois, _Building Large Language Models_, Stanford CS229 — [youtube.com/watch?v=9vM4p9NN0Ts](https://www.youtube.com/watch?v=9vM4p9NN0Ts).

### 207 · Context Window

- Su et al., _RoFormer: Enhanced Transformer with Rotary Position Embedding_ (2021) — [arxiv.org/abs/2104.09864](https://arxiv.org/abs/2104.09864). RoPE, central to extending context length.
- Liu et al., _Lost in the Middle: How Language Models Use Long Contexts_ (2023) — [arxiv.org/abs/2307.03172](https://arxiv.org/abs/2307.03172). Models use information in the middle of long contexts poorly.

### 208 · Temperature

- Holtzman et al., _The Curious Case of Neural Text Degeneration_ (2019) — [arxiv.org/abs/1904.09751](https://arxiv.org/abs/1904.09751). Temperature, top-k, and nucleus (top-p) sampling.
- Yann Dubois, _Building Large Language Models_, Stanford CS229 — [youtube.com/watch?v=9vM4p9NN0Ts](https://www.youtube.com/watch?v=9vM4p9NN0Ts). Covers generation and decoding.

### 209 · Hallucination

- Ji et al., _Survey of Hallucination in Natural Language Generation_ (2022) — [arxiv.org/abs/2202.03629](https://arxiv.org/abs/2202.03629). Taxonomy and causes of hallucination.
- Lin, Hilton & Evans, _TruthfulQA: Measuring How Models Mimic Human Falsehoods_ (2021) — [arxiv.org/abs/2109.07958](https://arxiv.org/abs/2109.07958). Benchmark for truthfulness.

### 210 · Prompt Engineering

- Brown et al., _Language Models are Few-Shot Learners_ (2020) — [arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165). In-context (few-shot) prompting.
- Wei et al., _Chain-of-Thought Prompting Elicits Reasoning in Large Language Models_ (2022) — [arxiv.org/abs/2201.11903](https://arxiv.org/abs/2201.11903). Step-by-step prompting.
- Kojima et al., _Large Language Models are Zero-Shot Reasoners_ (2022) — [arxiv.org/abs/2205.11916](https://arxiv.org/abs/2205.11916). The "Let's think step by step" finding.

## Part 3 — How AI models improve

### 311 · Transfer Learning

- Howard & Ruder, _Universal Language Model Fine-tuning for Text Classification_ (2018) — [arxiv.org/abs/1801.06146](https://arxiv.org/abs/1801.06146). ULMFiT; transfer learning for NLP.
- Raffel et al., _Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer_ (2020) — [arxiv.org/abs/1910.10683](https://arxiv.org/abs/1910.10683). The T5 study.

### 312 · Fine-Tuning

- Devlin et al., _BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding_ (2019) — [arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805). The pretrain-then-fine-tune paradigm.
- Ouyang et al., _Training language models to follow instructions with human feedback_ (2022) — [arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155). InstructGPT; instruction fine-tuning.

### 313 · RLHF

- Christiano et al., _Deep reinforcement learning from human preferences_ (2017) — [arxiv.org/abs/1706.03741](https://arxiv.org/abs/1706.03741). The foundation of preference-based training.
- Ouyang et al., _Training language models to follow instructions with human feedback_ (2022) — [arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155). RLHF applied to align an LLM.
- Rafailov et al., _Direct Preference Optimization: Your Language Model is Secretly a Reward Model_ (2023) — [arxiv.org/abs/2305.18290](https://arxiv.org/abs/2305.18290). DPO; alignment without a separate reward model.

### 314 · LoRA

- Hu et al., _LoRA: Low-Rank Adaptation of Large Language Models_ (2021) — [arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685). Train small add-on matrices instead of all weights.
- Dettmers et al., _QLoRA: Efficient Finetuning of Quantized LLMs_ (2023) — [arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314). LoRA combined with quantisation.

### 315 · Quantisation

- Dettmers et al., _LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale_ (2022) — [arxiv.org/abs/2208.07339](https://arxiv.org/abs/2208.07339). 8-bit inference for large models.
- Frantar et al., _GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers_ (2022) — [arxiv.org/abs/2210.17323](https://arxiv.org/abs/2210.17323). Accurate low-bit post-training quantisation.
- Dettmers et al., _QLoRA: Efficient Finetuning of Quantized LLMs_ (2023) — [arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314). 4-bit fine-tuning.

## Part 4 — How real AI systems are built

### 416 · RAG

- Lewis et al., _Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks_ (2020) — [arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401). The RAG paper.
- Malkov & Yashunin, _Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs_ (2018) — [arxiv.org/abs/1603.09320](https://arxiv.org/abs/1603.09320). HNSW, the retrieval mechanism.

### 417 · Vector Databases

- Malkov & Yashunin, _Efficient and robust approximate nearest neighbor search using HNSW graphs_ (2018) — [arxiv.org/abs/1603.09320](https://arxiv.org/abs/1603.09320). Approximate nearest-neighbour search.
- Facebook AI Research, _FAISS_ — [github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss). Library for efficient similarity search over dense vectors.
- _pgvector_ — [github.com/pgvector/pgvector](https://github.com/pgvector/pgvector). Vector similarity search for PostgreSQL.

### 418 · AI Agents

- Yao et al., _ReAct: Synergizing Reasoning and Acting in Language Models_ (2022) — [arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629). The reason-act-observe loop.
- Schick et al., _Toolformer: Language Models Can Teach Themselves to Use Tools_ (2023) — [arxiv.org/abs/2302.04761](https://arxiv.org/abs/2302.04761). Learned tool use.
- Anthropic, _Building Effective AI Agents_ — [anthropic.com/research/building-effective-agents](https://www.anthropic.com/research/building-effective-agents). Practitioner guidance on agents versus workflows.

### 419 · Chain of Thought

- Wei et al., _Chain-of-Thought Prompting Elicits Reasoning in Large Language Models_ (2022) — [arxiv.org/abs/2201.11903](https://arxiv.org/abs/2201.11903). Reasoning before answering.
- Kojima et al., _Large Language Models are Zero-Shot Reasoners_ (2022) — [arxiv.org/abs/2205.11916](https://arxiv.org/abs/2205.11916). Zero-shot chain of thought.
- Wang et al., _Self-Consistency Improves Chain of Thought Reasoning in Language Models_ (2022) — [arxiv.org/abs/2203.11171](https://arxiv.org/abs/2203.11171). Sample multiple reasoning paths, take the majority.

### 420 · Diffusion Models

- Ho, Jain & Abbeel, _Denoising Diffusion Probabilistic Models_ (2020) — [arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239). The DDPM framework.
- Rombach et al., _High-Resolution Image Synthesis with Latent Diffusion Models_ (2022) — [arxiv.org/abs/2112.10752](https://arxiv.org/abs/2112.10752). Latent diffusion, the basis of Stable Diffusion.
- Lilian Weng, _What are Diffusion Models?_ — [lilianweng.github.io/posts/2021-07-11-diffusion-models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/). Mathematical walkthrough of diffusion.
