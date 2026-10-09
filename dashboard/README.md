# dashboard/ -- one corruption per screen

Every corruption of an inference run, reviewable case by case: the true
fact and the corruption, the neighbourhood graph with the corrupted edge
drawn in, the LLM's reasoning against KGSAGE's three goals, what the graph
measures, and what it records. The exporter turns one run record into a
view model; the React app in `app/` renders it.

## CLI

```bash
python dashboard/export.py                      # the newest run
python dashboard/export.py --run codex-s_r0.05_s0
python dashboard/export.py --serve              # export, then the Vite dev server
python dashboard/export.py --build              # export, then a static app/dist/
python dashboard/export.py --build --single     # one self-contained HTML
```

`--serve` and `--build` need Node; the first of either runs `npm install`
in `app/`. `--single` writes `app/dist/dashboard_<run>.html` with the data
inlined -- one file (about 20 MB for 1,827 cases) that opens from disk with
no server. Every build empties `app/dist/` first, so a later `--build`
removes that file -- run `--build --single` again for a fresh one. A record
still being explained exports fine: unjudged cases say so, and exporting
again picks up the rest.

## Reads and writes

| reads | |
|---|---|
| `outputs/inference/manifest.json` | which run is newest |
| `outputs/inference/<run>/corruptions.json` | the run record -- the dashboard's only data source |

Writes `outputs/dashboard/data/<run>.json` and `manifest.json`, nothing
else (plus the app's own build in `app/dist/`). `app/vite.config.js` serves
`outputs/dashboard/` as the static root, which is why the app fetches
`./data/manifest.json` and finds what the exporter wrote. Formats:
[../CONTRACTS.md](../CONTRACTS.md).

## Using it

- **Overview.** The run, the pipeline, four stat tiles, three small charts
  (verdicts; what fails; corroboration by measure), a table by slot and by
  relation, and the decks: overlapping views of one docket -- by verdict,
  by what went wrong (probably true, out of character, corroborated after
  all, wrong kind), by measured corroboration, by slot, by relation.
  Hovering a deck lights up the cases it holds.
- **A case.** Arrow keys page, Escape returns. The graph: anchor pinned
  left, true value upper right, replacement lower right; the solid edge is
  the true fact, the dashed one the corruption; diamonds are shared
  neighbours, drawn below the corrupted edge with the paths through them;
  grey nodes are context. Hover or focus for what a node or edge is, drag
  to rearrange, "2 hops" for more context, "list view" for every fact as
  text, SVG / PNG to save the picture as a figure.
- **Links.** `#/case/c0342` opens a case, `#/deck/probably-true/3` a deck
  position; `?theme=light|dark` and `?depth=2` set the view (put the query
  before the `#`). The theme button cycles auto, light and dark.

## Design notes

- **The frontend holds no dataset knowledge.** Every label, kind,
  measurement and verdict arrives in the view model. The app computes two
  things itself, by design: the deck filters (pure groupings of exported
  fields) and the graph layout.
- **The layout** (`app/src/graph/layout.js`) is d3-force run to rest before
  first paint: a seeded random source and fixed starting positions make it
  deterministic -- the same case draws the same picture, in any browser.
  The three focus entities are pinned, so paging keeps them on the same
  spots of the screen. A label-box force and a final clean-up pass keep
  labels apart (the fixed relation labels of the two highlighted edges
  included); shared neighbours hang on slack springs so they never sit on
  the corrupted edge.
- **Colour.** The KGMVAD dashboard's design system. Graph roles use the
  reference categorical slots 1 to 3 (blue, aqua, orange), validated for
  colour-blind separation in both themes; verdicts use status colours,
  always with an icon and a word; measured corroboration is an ordered
  scale in one hue. Dark mode is its own validated set of steps, not an
  automatic flip.
- **Export** clones the live SVG; its colours are concrete attributes
  (read from the CSS custom properties), so the file renders on its own.

## Checks

```bash
cd dashboard/app && node test/layout.test.mjs ../../outputs/dashboard/data/<run>.json
```

Lays out every ninth case at both depths, twice: deterministic positions,
pins held, finite coordinates, a shape for every edge, context on its own
entity's side, and label overlaps against the same layout without the
label force. `python check_boundaries.py` at the repository root proves the
dashboard imports and writes only what it may.

## Layout

```
export.py     the CLI: record -> view model, manifest; --serve/--build/--single
formats.py    the shared formats (identical to inference/formats.py)
paths.py      where things are
app/          Vite + React 19 + d3-force
  src/App.jsx              data loading, hash routes, keyboard, theme
  src/decks.js             the decks -- pure filters
  src/labels.js            ids to words, criteria and level names
  src/components/          Overview, CaseCard, Criteria, Exhibit, Facts
  src/graph/layout.js      the layout (pure, tested under Node)
  src/graph/Graph.jsx      the SVG graph: hover, drag, zoom, list view, export
  src/styles.css           tokens for both themes, then the components
  test/layout.test.mjs     the layout check
```
