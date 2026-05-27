# Design - ST0003: 20 AI Concepts explainer series

## Structure

`docs/20-concepts/` — 24 markdown files + 21 images.

- Front matter: `000-toc.md` (table of contents, pairs with the `000.png` cover) and `README.md` (landing note).
- Four parts, section-banded 3-digit numbering where the file number = `<section><absolute-topic>`:
  - Part 1 — How AI actually works: `101`–`105`
  - Part 2 — How LLMs work: `206`–`210`
  - Part 3 — How AI models improve: `311`–`315`
  - Part 4 — How real AI systems are built: `416`–`420`
- Back matter: `501-recap.md`, `502-references.md`.
- Each chapter `NNN-slug.md` embeds the identically-numbered `NNN.png` (1:1 pairing).

## Per-chapter template

`verblock` frontmatter → evocative H1 → `## Concept N of 20 · Part X` → embedded image → one-paragraph hook → `What it is` / `How it works` / `State of the art in 2026` / `Why it matters` / (optional) `A common misconception` → italic forward-link footer to the next chapter. Target ~850-1150 words (2-3 pages).

## Conventions

- British/Australian spelling in prose; American for code identifiers and proper names.
- Depersonalised technical voice (no "I"/"me"/"we"); second-person "you" used sparingly.
- References: inline markdown links in the prose plus a consolidated `502-references.md` grouped by chapter. ZERO hallucinated references — every link verified to resolve and to support the claim it carries.
- Attribution: concept list from @sairahul1 (xcancel, never x.com); anchor reference is Stanford CS229 "Building Large Language Models" (Yann Dubois).
- em-dashes are permitted (house style).

## Alternatives considered

- Flat `001`–`020` numbering — rejected in favour of section-banded numbering so the file number encodes its part and pairs 1:1 with its image.
- Leaving `_sources` in history and only untracking going forward — rejected; the user wanted the 1.1 MB PDF purged from the pushed history (force-push).
- Drafting all chapters myself vs. parallel per-part agents — chose agents for throughput, with the author owning reference verification and a review/detrope pass.
