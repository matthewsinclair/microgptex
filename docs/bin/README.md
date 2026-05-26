# MicroGPTEx Documentation Publisher

Generates an aggregated Markdown file and a formatted PDF from a documentation set, using `mdagg` for aggregation and Pandoc + Typst for PDF generation. Adapted from the Lamplight publisher.

## Quick start

```bash
docs/bin/publish 20-concepts   # the "20 AI Concepts" explainer
docs/bin/publish blog          # the "Building GPT from Scratch" series
docs/bin/publish all           # both
docs/bin/publish --help
```

## Document sets

| Set           | Source              | Output base                                 |
| ------------- | ------------------- | ------------------------------------------- |
| `20-concepts` | `docs/20-concepts/` | `20-ai-concepts-to-understand-in-2026-v<V>` |
| `blog`        | `docs/blog/`        | `microgptex-building-gpt-from-scratch-v<V>` |

Each set writes `<base>.md` and `<base>.pdf` to its own `docs/<set>/_out/` directory.

## Options

```
docs/bin/publish {blog,20-concepts,all} [options]
```

- `--no-frontpage` — skip the logo on the title page
- `-v, --verbose` — show detailed output (including the file list and mdagg log)
- `-h, --help` — show this document (via `glow`/`bat` if available)

## Output

Two files per run, in the set's `_out/` directory (which is gitignored — regenerate any time):

```
docs/<set>/_out/<base>-v<VERSION>.md     # aggregated markdown
docs/<set>/_out/<base>-v<VERSION>.pdf    # typeset PDF
```

`VERSION` is read from a `VERSION` file if present, otherwise from `mix.exs` (currently `0.1.0`).

## How it works

```
Markdown files -> mdagg -> post-process -> Typst header -> Pandoc (typst) -> PDF
```

1. **Aggregate.** The file list is discovered by sorted filename — no committed config. `20-concepts` takes the `1xx`–`5xx` chapters plus recap and references (`000-toc.md` and `README.md` are excluded; the Typst title page and outline replace them). `blog` takes `part*.md`. A throwaway `mdagg.yaml` drives mdagg, with page breaks between files and front-matter + back-links stripped.
2. **Post-process** for PDF compatibility: linked and standalone remote images (eg the "Run in Livebook" badge) become plain links, since Typst cannot fetch URLs; HTML page-break divs become Typst pagebreaks; cross-document `.md` links become plain text (they cannot resolve inside a single PDF); `@`-mentions are escaped so Pandoc does not read them as citation keys; local images are symlinked into `_out/` under a name matching their **true** format and their refs rewritten (the topic images are JPEG bytes carrying `.png` names); the logo is symlinked in.
3. **Title page + ToC.** A Typst header block adds the logo, title, subtitle, version/date, and a native `#outline` table of contents.
4. **PDF.** `pandoc --pdf-engine=typst` with the `minimal.typst` template and the `fix-table-widths.lua` + `keep-together.lua` filters.
5. **Clean up.** All symlinks, Pandoc's `media-*` dir, and the temporary `mdagg.yaml` are removed; only the `.md` and `.pdf` remain.

## Dependencies

- `mdagg` — markdown aggregator (Utilz framework; on `PATH`)
- `pandoc` — document converter (3.x, with Typst support)
- `typst` — PDF engine

Optional: `glow` or `bat` for a rendered `--help`.

```bash
brew install pandoc typst
# mdagg: install Utilz and put its bin/ on your PATH
```

## Files

```
docs/bin/
  publish               # this script
  minimal.typst         # Pandoc template for Typst output
  fix-table-widths.lua  # removes fixed table widths (responsive columns)
  keep-together.lua     # .keep-together -> Typst unbreakable block
  README.md             # this document
```

## Adding a document set

1. Put markdown in `docs/<set>/` with sortable filenames.
2. Add a case to `get_config()`: `DOC_SUBDIR|OUTPUT_BASE|TITLE|SUBTITLE|DESC1|DESC2|GLOB`.
3. Add `<set>` to the argument parser and to the `all` loop.

## Notes

- The 20-concepts topic images are JPEG data with `.png` filenames. The publisher detects the true format and relabels on the fly, so no source files need renaming and they still render correctly on GitHub.
- The logo is `design/microgptex-logo.svg`. Use `--no-frontpage` to omit it.
