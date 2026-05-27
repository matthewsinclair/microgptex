# Implementation - ST0003: 20 AI Concepts explainer series

## As built

24 markdown files in `docs/20-concepts/` (ToC, README, 20 chapters, recap, references) plus 21 images (`000.png` cover + one `NNN.png` per chapter). `_sources/` (the `.md`, 1.1 MB `.pdf`, and a 581 MB CS229 `.mp4`) is kept on disk but gitignored.

## Commit trail (on `main`)

- `f88fcba` — ToC + topic images + local `_sources` ignore rule (after the history rewrite).
- `d123cae` — golden chapter 101 + seed references.
- `d76eee6` — chapters 102-420.
- `c99e4b0` — recap + consolidated references.
- `2d8c0e7` — detrope pass.

## Git surgery

The accidentally-committed `_sources/*.md` + 1.1 MB `*.pdf` lived only in the tip commit (`9754aaa`), already pushed to `upstream` (GitHub) and `local` (Dropbox). Fixed by amending the tip to drop them and re-point README to the corrected `docs/20-concepts/` path, then force-pushing both remotes with `--force-with-lease`. The 581 MB mp4 was never committed. A backup of `_sources/` sits at `~/microgptex-sources-backup-*` outside the repo.

## Reference verification

All references were pre-verified before drafting: 30 arXiv papers confirmed via their abstract-page titles; the non-arXiv sources confirmed by fetch. A post-draft URL audit confirmed every link in every chapter belongs to the verified pool — zero hallucinated references. The drafting agents were supplied the verified references and forbidden from adding any.

## Detrope

Mechanical pre-scan plus an ultrathink contextual pass over the 21 prose files. Trope density was low and the AI signal weak. Fixes applied: grandiose-stakes (104), doubled negative-parallelism (419), tapestry-and-landscape (105, 315), one negative-parallelism + magic adverb (311), one magic adverb (207). em-dash density was suppressed as house style (the approved golden chapter uses em-dashes).
