microgptex docs <publish|agents|all> -- build the documentation, and regenerate the derived kind.

    publish <docset>  docs/bin/publish -- aggregate a set and render its PDF
    agents            intent agents sync (rewrites the root AGENTS.md)
    all               agents (publish is deliberately not in it)

`docs publish` is the one that matters here. This project is a paper and the small implementation it describes, so the PDF is the build artefact -- there is no server to run and no release to ship. It calls `docs/bin/publish`, which aggregates a documentation set with `mdagg`, post-processes it for Typst, and renders the PDF through pandoc with a typst template and two lua pandoc filters. The Markdown aggregate and the PDF both land in `docs/<docset>/_out/`, which is gitignored: regenerate rather than commit.

The docset is REQUIRED and is passed straight through to the publisher, so the list of sets lives in one place -- the publisher's own case statement -- rather than being copied into a devbin option per set. Today that list is `blog`, `20-concepts` and `all`. Called with no docset, `docs publish` exits 1 on the publisher's usage line rather than guessing, and the failure seals like any other gate.

`publish` is excluded from `docs all` on purpose (`in_all: false` in bin/.devbin/config.yaml). `docs all` is meant to be a cheap, argument-free sweep of the generators; publishing shells out to pandoc and typst, takes real time, and cannot run without an argument. The exclusion is reported in the `all` output, never silent.

It needs three tools on PATH that nothing else here needs: `mdagg`, `pandoc` (3.x, with Typst support) and `typst`. `brew install pandoc typst` covers the last two. The version stamped into the output filename comes from mix.exs, which is this project's one version home -- the publisher looks for a VERSION file first and falls back, and there is none, so both paths give the same answer.

AGENTS.md is GENERATED. Never hand-edit it -- `intent agents sync` rewrites it from project state and a manual edit is lost on the next sync.

`docs treeindex` was REMOVED on 2026-08-25 (devbin#0032), along with the `docs.treeindex_dirs` key in bin/.devbin/config.yaml that fed it. Intent v3 retires `intent treeindex` with no replacement (executed at intent 861fa66c), so the option was one that lists cleanly, resolves cleanly and can never succeed once this estate moves to v3. It indexed lib and test -- the prose under docs/ was deliberately never in that list, since a code index is not what a documentation set wants. `docs all` is correspondingly one generator lighter, which is a real reduction rather than a rename.

    bin/mg docs publish 20-concepts
    bin/mg docs publish all --no-frontpage
    bin/mg docs all
