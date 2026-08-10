microgptex docs <publish|agents|treeindex|all> -- build the documentation, and regenerate the derived kind.

    publish <docset>  docs/bin/publish -- aggregate a set and render its PDF
    agents            intent agents sync (rewrites the root AGENTS.md)
    treeindex [dirs]  intent treeindex over subdirectories
    all               agents + treeindex (publish is deliberately not in it)

`docs publish` is the one that matters here. This project is a paper and the small implementation it describes, so the PDF is the build artefact -- there is no server to run and no release to ship. It calls `docs/bin/publish`, which aggregates a documentation set with `mdagg`, post-processes it for Typst, and renders the PDF through pandoc with a typst template and two lua pandoc filters. The Markdown aggregate and the PDF both land in `docs/<docset>/_out/`, which is gitignored: regenerate rather than commit.

The docset is REQUIRED and is passed straight through to the publisher, so the list of sets lives in one place -- the publisher's own case statement -- rather than being copied into a devbin option per set. Today that list is `blog`, `20-concepts` and `all`. Called with no docset, `docs publish` exits 1 on the publisher's usage line rather than guessing, and the failure seals like any other gate.

`publish` is excluded from `docs all` on purpose (`in_all: false` in bin/.devbin/config.yaml). `docs all` is meant to be a cheap, argument-free sweep of the generators; publishing shells out to pandoc and typst, takes real time, and cannot run without an argument. The exclusion is reported in the `all` output, never silent.

It needs three tools on PATH that nothing else here needs: `mdagg`, `pandoc` (3.x, with Typst support) and `typst`. `brew install pandoc typst` covers the last two. The version stamped into the output filename comes from mix.exs, which is this project's one version home -- the publisher looks for a VERSION file first and falls back, and there is none, so both paths give the same answer.

AGENTS.md is GENERATED. Never hand-edit it -- `intent agents sync` rewrites it from project state and a manual edit is lost on the next sync.

treeindex is NEVER run on the project root: it is a per-subdirectory index and the root sweep is both enormous and useless. Bare `docs treeindex` indexes the declared list (lib test -- `docs.treeindex_dirs` in bin/.devbin/config.yaml), and `docs treeindex .` is REFUSED rather than obeyed. The prose under docs/ is deliberately not in that list; a code index is not what a documentation set wants.

    bin/mg docs publish 20-concepts
    bin/mg docs publish all --no-frontpage
    bin/mg docs all
    bin/mg docs treeindex lib
