# KGSAGE inference + dashboard: architecture plan

Draft 2, 9 October 2026. Decisions D1 to D9 were settled by the author on
9 October (section 11); three minor ones defaulted to the recommendation.
**Scope for the first build: CoDEx-S only.**

> **Built 10 October 2026.** M0 to M4 are done. Where the build departed
> from this plan -- the neighbourhood caps, the corroboration measurements
> the LLM needed, eight prompt versions, the label-aware layout -- and every
> measurement behind those choices is recorded in [DESIGN.md](DESIGN.md).
> This plan is kept as written, as the record of what was agreed.

## 0. The feature in one paragraph

Two new components beside the `kgsage/` package, laid out the way KGMVAD
lays out its components (one folder each, coupled only through files in
`outputs/`). **`inference/`** takes a trained checkpoint and a dataset,
corrupts a chosen percentage of its triples, and writes the corrupted
triple list plus one JSON record that carries every corruption with its
neighbourhood; a second CLI in the same folder hands each case to Gemini
through `google-genai` and writes the model's reasoning about the three
KGSAGE goals, plus a world-knowledge falseness check, back into that JSON.
**`dashboard/`** turns the JSON into a view model and renders it in a
React app: the corruptions, the reasoning, and the neighbourhood graph
with the corrupted edge drawn in, laid out in the browser by a React
graph component.

## 1. What exists, and what this supersedes

### The package API (one optional kwarg added, section 7)

`kgsage.load_kg`, `load_checkpoint(ckpt, types_path=)`,
`generate_negatives(triples, payload, id_maps, rng, max_resample,
support_max, guards)` and `render_stats`. The payload keys that are already
a contract (`real_triple_set`, `pool_masks`, `true_tails`/`true_heads`,
`kinds` once types are attached) give the new code everything it needs
without importing anything private.

Measured on 9 October on `outputs/checkpoints/run_codex-s_s0.pt` with
`data/codex-s` (pytorch env, `torch.set_num_threads(1)`):

| | |
|---|---|
| checkpoint vocabulary vs dataset | 2,034 / 2,034 entities, 42 / 42 relations, nothing missing |
| labels from KGMVAD's CoDEx definition files | 2,034 / 2,034 entities, 42 / 42 relations |
| guarded generation (kind + support 0.5 + unique, max_resample 32) | 7 ms / row, 1.5 % null, 98.5 % type-valid |
| unguarded (every recorded matrix run before 24 Sep) | 1 ms / row, 0.5 % null |

At ratio 0.05 on CoDEx-S (1,827 corruptions) generation is seconds, not
minutes. The LLM stage is the slow one.

### The CLIs the new path replaces (retired at M4, decision D7)

| existing `kgsage/cli` tool | what it does | replaced by |
|---|---|---|
| `gen_plain_tsv.py` | corruptions as a 3-column TSV, nulls and repeats dropped | `inference/corrupt.py` → `corruptions.tsv` |
| `gen_corruptions_csv.py` | labelled CSV, gated on a hand-written relation template list (FB15K-237 / WN18RR only; **cannot run on CoDEx**) | `corruptions.json`, dataset-agnostic |
| `gen_neighbourhood_context.py` | anchor's facts as paste-ready case blocks for a manual LLM run | `inference/explain.py` (automated; `--dry-run` prints the blocks) |
| `format_for_llm.py` | corrupted + control triple blocks | out of scope for now (section 12) |
| `ego_from_csv.py` | one matplotlib ego figure per row | the dashboard's graph component, with SVG / PNG export |
| `knockout_eval.py` | snapshot selection | **stays** |

Retirement happens only after the dashboard's export reproduces an ego
figure (M3 gate); git history is the archive, per the single-version policy.

### KGMVAD patterns mirrored

- one folder per component; a component imports only its own folder, the
  standard library, and (for `inference/`) the `kgsage` package;
- everything between components is a file in `outputs/`, each component
  writes only its own subfolder, and a `CONTRACTS.md` names every format;
- `dashboard/export.py` builds a view model, the React app holds no
  dataset knowledge, `--serve` / `--build` / `--single` (self-contained HTML);
- Gemini through a `.env` key, 4 s pacing for the free tier's 15
  requests / minute, retry options, and a record that is self-contained
  (model name, prompt version, timestamps inside the record).

## 2. Folder structure

```
KGSAGE/
  kgsage/                    THE PACKAGE -- unchanged except the slot kwarg (section 7)
  inference/                 component 1: corrupt, describe, explain
    README.md
    corrupt.py               CLI: checkpoint + dataset + ratio -> TSVs + corruptions.json
    explain.py               CLI: corruptions.json -> Gemini reasoning -> corruptions.json
    context.py               CoDEx definition reader: labels, descriptions, kinds (section 3.4)
    neighbourhood.py         ego extraction + the structural measurements (section 4.3)
    prompts.py               the explain prompt, versioned
    formats.py               TSV + JSON readers and writers, sha256, atomic write
    paths.py
  dashboard/                 component 2
    README.md
    export.py                corruptions.json -> view model + manifest;
                             --serve / --build / --single, as in KGMVAD
    paths.py
    app/                     Vite + React, scaffold copied from KGMVAD/dashboard/app
      package.json           + d3-force (the only graph dependency)
      src/App.jsx  decks.js  styles.css
      src/components/Overview.jsx  CaseCard.jsx  Criteria.jsx  Exhibit.jsx
      src/graph/Graph.jsx    the neighbourhood graph: d3-force layout, SVG render, export
  data/codex-s/
    train.txt valid.txt test.txt          existing (flat layout; the registry expects it)
    entities/en/entities.json             copied from KGMVAD, 10.0 MB   (D4)
    relations/en/relations.json           10 KB
    types/entity2types.json               1.7 MB
    types/en/types.json                   518 KB
  outputs/
    checkpoints/             existing
    eval/                    existing
    inference/<run>/         corrupt.py writes; explain.py enriches corruptions.json in place
    inference/manifest.json  run list, newest first
    dashboard/data/          export.py writes: <run>.json + manifest.json
  CONTRACTS.md               the formats
  check_boundaries.py        AST proof that the import rules hold
  .env.example               GOOGLE_API_KEY=
  docs/                      this plan; later a DESIGN.md of dated decisions
```

`pyproject.toml` installs only `kgsage*`; `inference/` and `dashboard/`
are repo furniture like `smoke_test.py` and `slurm/`, run from the repo
root as `python inference/corrupt.py`. Applications over the package,
never the package over them.

`inference/` keeps its name (D1). The package module `kgsage.inference`
was renamed `corruption_generation` in July 2026 for the paper's phase
names; the application folder is a different thing and the name is the
author's.

## 3. Stage A: `inference/corrupt.py`

### 3.1 Command

```bash
# pytorch env, from the KGSAGE repo root
python inference/corrupt.py --ckpt outputs/checkpoints/run_codex-s_s0.pt \
    --data codex-s --ratio 0.05 --seed 0
```

| flag | default | meaning |
|---|---|---|
| `--ckpt` | required | a candidate_v2 checkpoint |
| `--data` | required | registry name or directory; `codex-s` for this build |
| `--ratio` | `0.05` | corruptions as a share of the dataset's triples (all splits, deduplicated). 0.05 on CoDEx-S = 1,827, ADKGD's `r0.05` column size |
| `--n` | – | absolute count instead of a ratio |
| `--seed` | `0` | fixes the source sample, the generator's Gumbel draws, and the shuffle |
| `--sources <tsv>` | – | corrupt *these* true triples instead of a uniform sample (the paired-sources protocol from `docs/KGSAGE_COLUMN.md`, without ADKGD) |
| `--slot both/head/tail` | `both` | through the new package kwarg (D9) |
| `--guards` / `--no-guards` | **on** (D5) | kind + support 0.5 + unique, the v2 eval-column protocol; `--no-guards` reproduces the pre-24-Sep column |
| `--types <json>` | auto | `entity2types.json` for the kind guard, found at `data/<name>/types/` |
| `--support_max N` | off | the corroboration *mask* (enforce). Off on purpose: turning it on rejects 56 % of hand-verified negatives (`docs/KGSAGE_COLUMN.md`). Corroboration is always **measured** and recorded |
| `--hops 2 --max_neighbours 6` | ego caps | the `ego_from_csv.py` defaults |
| `--out` | `outputs/inference/<dataset>_r<ratio>_s<seed>[_<tag>]/` | run folder (D11) |
| `--tag` | – | suffix for a variant run of the same seed |
| `--max_resample 32` | | redraws before a null, the guarded setting |

Over-draw loop as in `gen_plain_tsv.py` (1.5× rounds, max 6): nulls and
repeats are dropped and counted, so the file holds exactly `--n` unique
false corruptions or says why it could not.

On Windows without CUDA the script sets `torch.set_num_threads(1)` and
disables mkldnn (the ADKGD fix), a no-op on the cluster.

### 3.2 Outputs (D6)

```
outputs/inference/codex-s_r0.05_s0/
  corruptions.tsv      h  r  t      the corruptions only. This IS the handover format:
                                    KGMVAD  inject --source frozen --file ... --kind kgsage
                                    ADKGD   --anomaly_file
  kg_corrupted.tsv     h  r  t      original dataset (all splits, deduped) + corruptions,
                                    shuffled by --seed   <- "original dataset + corrupted triples"
  labels.tsv           h  r  t  label  kind      answer key for kg_corrupted.tsv, one row per row,
                                    KGMVAD's ground_truth.tsv format (0 real / 1 kgsage)
  corruptions.json     the record (section 4); explain.py adds "reasoning" to it in place
```

`outputs/inference/manifest.json` lists runs newest-first so the dashboard
exporter can pick the latest without arguments. A rerun with the same
dataset, ratio and seed overwrites its folder; `--tag` keeps a variant.

### 3.3 What is recorded per case

Each corruption leaves with: both triples (ids and labels), which slot
changed, the anchor, the true filler and the picked candidate; the three
entities' labels, descriptions and kinds; the relation's label and
description; the structural measurements of section 4.3; and the
neighbourhood (nodes + edges) extracted the way `ego_from_csv.py` does it:
2-hop ego of both endpoints of the true triple capped at `max_neighbours`
per node, plus the candidate's 1-hop, with the true filler and the
candidate never dropped by the cap.

### 3.4 Labels: CoDEx-S only (D4)

KGMVAD's four CoDEx definition files are copied into `data/codex-s/`
beside the flat split files (the registry wants `codex-s/train.txt`, so
the KGMVAD `triples/` sub-folder is not copied) and **tracked in git**, as
KGMVAD tracks them: KGSAGE must run alone. `inference/context.py` reads
those four fixed relative paths and resolves ids to labels, descriptions
and kind labels; it is the dashboard copy of KGMVAD's `context.py`
without the reverse maps. Other datasets' label formats (FB15K-237
`entity2text.txt`, WordNet synsets) are out of scope for this build; a
dataset without definition files will still run with raw ids.

## 4. The JSON contract: `corruptions.json`

### 4.1 Header

```json
{
  "schema": "kgsage-inference/1",
  "run": "codex-s_r0.05_s0",
  "generated": "2026-10-09T15:30:00",
  "dataset": {"name": "codex-s", "path": "data/codex-s", "triples": 36543,
              "entities": 2034, "relations": 42, "kg_sha256": "…",
              "labels": "codex-json"},
  "checkpoint": {"path": "outputs/checkpoints/run_codex-s_s0.pt",
                 "sha256": "…", "arch": "candidate_v2"},
  "params": {"ratio": 0.05, "n": 1827, "seed": 0, "slot": "both",
             "sources": "uniform", "guards": {"kind": true, "support": 0.5,
             "unique": true}, "support_max": null, "max_resample": 32,
             "hops": 2, "max_neighbours": 6},
  "stats": {"requested": 1827, "emitted": 1827, "nulls": 27, "repeats": 41,
            "rounds": 2, "slot_h": 912, "slot_t": 915, "type_valid": 1801,
            "kind_match": 1790, "uncorroborated": 1210, "direct_edge": 38,
            "render_stats": "processed=… used_original(gen_failed)=…"},
  "files": {"corruptions_tsv": "corruptions.tsv",
            "kg_corrupted_tsv": "kg_corrupted.tsv", "labels_tsv": "labels.tsv"},
  "reasoning": null,
  "cases": [ … ]
}
```

The header-level `reasoning` is filled by `explain.py` with the run-wide
facts (model, prompt version, cases explained, calls made, errors,
started / finished) so the record says what was done to it.

### 4.2 One case

```json
{
  "id": "c0001",
  "slot": "tail",
  "true":      {"h": "Q42", "r": "P106", "t": "Q36180"},
  "corrupted": {"h": "Q42", "r": "P106", "t": "Q937857"},
  "anchor": "Q42", "true_filler": "Q36180", "candidate": "Q937857",
  "labels": {"h": "Douglas Adams", "r": "occupation",
             "true_t": "novelist", "corr_t": "footballer"},
  "relation": {"label": "occupation", "description": "…"},
  "entities": {
    "Q42":     {"label": "Douglas Adams", "description": "English writer…",
                "kinds": ["human"], "degree": 14},
    "Q36180":  {"label": "novelist", "description": "…", "kinds": ["profession"], "degree": 120},
    "Q937857": {"label": "footballer", "description": "…", "kinds": ["profession"], "degree": 61}
  },
  "structure": { … section 4.3 … },
  "neighbourhood": {
    "nodes": [{"id": "Q42", "label": "Douglas Adams", "role": "anchor", "hop": 0},
              {"id": "Q36180", "label": "novelist", "role": "true_filler", "hop": 1},
              {"id": "Q937857", "label": "footballer", "role": "candidate", "hop": null},
              {"id": "Q145", "label": "United Kingdom", "role": "context", "hop": 1}, …],
    "edges": [{"h": "Q42", "r": "P106", "t": "Q36180", "kind": "true"},
              {"h": "Q42", "r": "P106", "t": "Q937857", "kind": "corrupted"},
              {"h": "Q42", "r": "P27", "t": "Q145", "kind": "context"}, …]
  },
  "reasoning": null
}
```

Ids stay (exact; the TSVs and any detector speak them); labels travel
beside them (the only form a person or an LLM can judge). Node `role` is
one of `anchor`, `true_filler`, `candidate`, `context`. Edge labels for
the graph come from `relation.label` lookups the exporter adds, so the
app never opens a definition file.

### 4.3 The structural measurements

All recomputed from the dataset's all-splits graph in
`inference/neighbourhood.py` (not by calling the package's private
helpers), so the numbers are reproducible from the TSVs alone:

| field | meaning | which goal it speaks to |
|---|---|---|
| `candidate_in_pool` | the candidate has occupied this slot of this relation somewhere in the graph | type-valid (the generator's own notion) |
| `kind_match` | candidate shares a kind with the true filler (CoDEx criterion i, the kind guard) | type-valid |
| `slot_usual_kinds` | top-3 kinds of this slot's occupants, with counts | type-valid (KGMVAD's odd_types exhibit) |
| `shared_neighbours` | \|N(anchor) ∩ N(candidate)\|, undirected, all splits | uncorroborated |
| `direct_edge` | candidate is a 1-hop neighbour of the anchor | uncorroborated |
| `candidate_hop` | BFS distance anchor → candidate within `hops`, else null | uncorroborated |
| `support` | P(candidate \| v) maxed over the anchor's true values v under r: the share of v's holders that also hold the candidate (the `KGSAGE_COLUMN.md` probe metric; the support guard bans > 0.5) | plausible vs probably-true |
| `anchor_values` | the anchor's other true values under r (labels) | plausible / uncorroborated |
| `anchor_degree`, `candidate_degree`, `true_filler_degree` | | context for the LLM and the graph |

### 4.4 Size

About 1,800 cases × roughly 35 nodes and 45 edges each is 6 to 10 MB. The
KGMVAD dashboard already loads a 1,581-case record, and `--single` embeds
it in the HTML without trouble.

## 5. Stage B: `inference/explain.py`

### 5.1 Command

```bash
# torch-free; base conda (has google-genai 2.17) or pytorch env after
# pip install google-genai
python inference/explain.py --run outputs/inference/codex-s_r0.05_s0 \
    [--model gemini-3.5-flash-lite] [--batch 10] [--limit 20] [--pause 4]
    [--redo] [--dry-run]
```

Reads `corruptions.json`, explains every case whose `reasoning` is null,
writes the file back **atomically** (temp file + rename) after every batch.
Resumable by construction: a quota stop or a crash loses at most one batch,
and rerunning continues. `--limit` is for prompt iteration on 10 to 20
cases before spending a full run. `--dry-run` prints the exact case blocks
with no calls, the automated successor of `gen_neighbourhood_context.py`.

Key discovery exactly as `KGMVAD/detector/detect.py` does it:
`GOOGLE_API_KEY` from the environment or a `.env` beside the repo root.
`.env.example` is committed; `.env` goes into `.gitignore` (it is not
ignored today).

### 5.2 Stack: `google-genai` direct, structured output (D2)

A single `google-genai` call per batch with
`response_mime_type="application/json"` and a `response_schema`, which
returns the reasoning blocks as parsed JSON with no free-text parsing. No
tools, no ADK session machinery: every fact the model should see is in
the case record. The pacing (4 s between calls), the retry options
(`HttpRetryOptions(attempts=5, initial_delay=10, max_delay=70)`) and the
telemetry (calls made, errors, retries, a `truncated` status when the API
stopped the run) are lifted from KGMVAD's `agents/config.py`, `pacing.py`
and `telemetry.py`. Model name defaults to KGMVAD's
`gemini-3.5-flash-lite` and is recorded per case.

### 5.3 What the model sees per case

A case block, batched `--batch` at a time under one instruction:

```
CASE c0001  (tail slot corrupted; anchor = Douglas Adams)
True fact:       Douglas Adams --occupation-- novelist
Corrupted fact:  Douglas Adams --occupation-- footballer
Relation: occupation -- "occupation of a person; see also field of work…"
Entities:
  Douglas Adams  [human]        English writer and humorist (1952–2001)
  novelist       [profession]   writer of novels
  footballer     [profession]   person who plays association football
Facts about Douglas Adams (anchor), occupation first, up to 20:
  - Douglas Adams, occupation, novelist
  - Douglas Adams, occupation, screenwriter
  - Douglas Adams, country of citizenship, United Kingdom
  …
Facts about footballer (candidate), up to 10:
  - David Beckham, occupation, footballer
  …
Measured: candidate shares 0 neighbours with the anchor; no direct edge;
  the slot's usual kinds are profession ×1,204, artist ×311; the candidate
  is of kind profession; support P(footballer | novelist) = 0.01.
```

### 5.4 The criteria and the output block (D3)

The three KGSAGE goals as the paper states them (type-valid, plausible,
contradicted by the neighbourhood), plus the fourth, `false`, a
world-knowledge check:

```json
"reasoning": {
  "model": "gemini-3.5-flash-lite", "prompt_version": "explain/1",
  "at": "2026-10-09T16:02:11", "batch": 12,
  "criteria": {
    "type_valid":     {"holds": true,  "why": "A profession sits where a profession belongs; the slot's occupants are professions."},
    "plausible":      {"holds": true,  "why": "A person can hold this occupation; nothing in the fact is absurd on its face."},
    "uncorroborated": {"holds": true,  "why": "Nothing around Adams points at football: his recorded occupations, works and awards are all literary."},
    "false":          {"holds": true,  "confidence": "high", "why": "Adams was never a footballer."}
  },
  "verdict": "aligned",
  "summary": "A type-valid, plausible swap that the neighbourhood rules out and the world confirms false."
}
```

`verdict` is `aligned` (all three goals hold), `partial` (one fails) or
`misaligned` (two or more fail). The `false` criterion stays outside the
verdict: it is the check the 24 September column investigation found
missing (13 % of the v1 column was probably true). It is **triage**, not
the number the paper reports: `KGSAGE_COLUMN.md` option G prefers a
Wikidata SPARQL check over an LLM judge for the reported false-anomaly
rate, because an LLM judge shares a family with KGMVAD's observers. The
flag tells you which rows to send to Wikidata first.

### 5.5 Cost

1,827 cases at `--batch 10` is 183 calls; at 4 s pacing that is about 13
minutes on the free tier (15 requests / minute). Prompt length per call is
roughly 10 × 600 tokens of case text, well inside the context. Batch 20
halves the calls if a daily request cap bites.

## 6. Stage C: `dashboard/`

### 6.1 `export.py`

```bash
python dashboard/export.py                     # newest run in outputs/inference/
python dashboard/export.py --run outputs/inference/codex-s_r0.05_s0
python dashboard/export.py --serve             # export, then the Vite dev server
python dashboard/export.py --build [--single]  # static dist/, or one self-contained HTML
```

Reads `corruptions.json`, refuses an unknown schema version, attaches
relation labels to neighbourhood edges, computes the overview statistics
and the deck counts, and writes `outputs/dashboard/data/<run>.json` plus
`manifest.json`. No layout is computed here (D8); the exporter is thin.
The `--serve/--build/--single` block and `vite.config.js` (`publicDir` =
`outputs/dashboard`) are copied from KGMVAD. Cases without reasoning are
exported too, with the reasoning panel saying "not yet explained", so the
dashboard is usable the moment `corrupt.py` finishes.

### 6.2 Screens

**Overview.** Run header (dataset, checkpoint, ratio, seed, guards, LLM
model). Stat tiles: corruptions, head / tail split, type-valid %,
uncorroborated % (zero shared neighbours and no direct edge), and the LLM
summary: aligned / partial / misaligned / probably-true counts. A pipeline
strip (corrupt → describe → explain) in the KGMVAD style. Then the decks.

**Decks** (overlapping views over one docket, pure filters as in
KGMVAD's `decks.js`; D12 defaulted to this list):

| deck | dimension |
|---|---|
| every corruption | all |
| aligned / partial / misaligned | LLM verdict |
| fails type-valid / fails plausible / corroborated after all | which criterion failed |
| probably true | the `false` criterion did not hold |
| head slot / tail slot | slot |
| zero shared neighbours / shared > 0 / direct edge | measured corroboration |
| per relation (top 10 by count) | relation |

**Case card.** The fact pair with the changed slot highlighted (true fact
in green, corrupted in red, same typography as the KGMVAD fact header);
the neighbourhood graph; a four-row criteria checklist with a HOLDS /
FAILS stamp and the LLM's one sentence each, then the verdict and
summary; the structural exhibit (shared neighbours, direct edge, support,
usual kinds versus the candidate's kind, degrees) as pills and bars like
KGMVAD's `Evidence.jsx`; a provenance footer (run, checkpoint hash,
model, prompt version). Arrow keys page, Escape returns, as in KGMVAD.

### 6.3 The neighbourhood graph: a React component, laid out in the browser (D8)

`src/graph/Graph.jsx` is the author's "try some React component". First
attempt: our own SVG component with `d3-force` doing the layout (the
only graph dependency, about 30 KB, bundles into `--single` with no CDN):

- the three focus nodes are pinned (`fx`/`fy`): anchor left, true filler
  upper right, candidate lower right, the same arrangement
  `ego_from_csv.py` used, so the true edge and the corrupted edge always
  cross open space; everything else settles around them under link,
  charge and collision forces;
- deterministic: the simulation gets a seeded `randomSource` and starts
  from fixed initial positions, and it is run to completion
  synchronously (`tick()` in a loop) before first paint, so the same case
  draws the same picture every time and does not jitter;
- styling as the paper figures: `#6c8ebf` anchor-side head, `#82b366`
  true filler, `#b85450` candidate, greys for hop 1 and hop 2, green
  solid true edge, red dashed corrupted edge, arrow markers, relation
  labels on the two highlighted edges, other edge labels on hover;
- hover a node for label, description and kinds; drag to untangle a hub
  case; "export SVG / PNG" serialises the SVG node, which is what retires
  `ego_from_csv.py`.

Fallback if the hand-rolled component is not legible on hub cases after a
day: `cytoscape.js` through `react-cytoscapejs` (built-in pan / zoom,
`cose` layout with `randomize: false`, dashed line styles, `cytoscape-svg`
for export). `react-force-graph-2d` was considered and set aside: canvas
only, so no SVG export and hand-drawn edge labels.

When building, follow the `dataviz` skill for the tiles, bars and colour
choices.

## 7. Touch points inside `kgsage/` and the repo

- `generate_negatives(..., slot=None)`: a backwards-compatible kwarg
  (D9). `None` keeps today's 50/50 draw (every recorded run), `0` or `2`
  forces head or tail. One line in the slot step, the stats keys unchanged.
- `pyproject.toml`: an `inference` extra (`google-genai`) beside the
  existing `viz` extra.
- `.gitignore`: add `.env`, `dashboard/app/node_modules/`,
  `dashboard/app/dist/` (none is ignored today; `outputs/` already is).
  The copied CoDEx definition files are **tracked** (12.3 MB in all).
- README: layout block, usage section, the two new components; the
  "Helpers that read that CSV" paragraph goes, and `kgsage/cli/__init__.py`'s
  tool list shrinks to `knockout_eval`.
- `CONTRACTS.md` + `check_boundaries.py`: `inference/` may import
  `kgsage`, its own folder and the standard library (plus numpy / torch
  through kgsage, and `google-genai` in `explain.py` only); `dashboard/`
  imports its own folder and the standard library, never `kgsage` or
  `inference`; `explain.py` never imports torch; each component writes
  only its own `outputs/` subfolder.

## 8. Environments on this machine

| stage | interpreter | why |
|---|---|---|
| `corrupt.py` | `envs/pytorch/python.exe` | `kgsage` is editable-installed there; set `OMP_NUM_THREADS=1`, the script sets `torch.set_num_threads(1)` itself |
| `explain.py` | base `miniconda3/python.exe` | has `google-genai` 2.17; the script is torch-free by design. Or `pip install google-genai` into pytorch |
| `export.py` | either | stdlib only |
| the app | Node 24.4 / npm 11.4 | present |

## 9. Milestones

| | deliverable | gate | size |
|---|---|---|---|
| M0 | folders, READMEs, `CONTRACTS.md`, `paths.py` / `formats.py`, `.env.example`, gitignore, the four CoDEx definition files copied into `data/codex-s/` | `python inference/corrupt.py --help` runs from the repo root; `check_boundaries.py` passes on the empty scaffold; `smoke_test.py` still passes | ½ day |
| M1 | `corrupt.py` + `context.py` + `neighbourhood.py` + the `slot` kwarg | dummy_kg end to end; CoDEx-S at ratio 0.05 seed 0 writes 1,827 unique false rows; `corruptions.tsv` passes KGMVAD's `inject --source frozen` guards line (0 true, 0 duplicates); every case's neighbourhood contains both edges; structural fields match a hand-computed sample; `--slot tail` gives 100 % tail | 1 to 1½ days, ~450 lines |
| M2 | `explain.py` + `prompts.py` | `--dry-run` on 5 cases reads right; `--limit 20` live run returns schema-valid blocks; a kill mid-run resumes without loss; the record carries model, prompt version, call counts | 1 day, ~300 lines + prompt iteration |
| M3 | `export.py`, the React app, `Graph.jsx` with d3-force | overview and case card on the full CoDEx-S run; `--single` HTML opens from disk; the graph is legible on a 30-node case and on a hub case; the same case rendered twice is pixel-identical; SVG export of one case matches the old ego figure in content | 2 days, ~200 Python + ~800 JSX/CSS |
| M4 | boundary check, README, retire the five superseded CLIs (D7) | `smoke_test.py` passes; the two new READMEs explain each component alone; the retired tools are gone from the tree and named in the commit message | ½ day |

Roughly five working days, the LLM prompt iteration and the graph
legibility spike being the elastic parts.

## 10. Risks and traps

- **Guards versus history.** Guards on (the default now) is the v2 column
  of 24 September; `--no-guards` reproduces every run recorded before it.
  The JSON header records the setting, so no run is ambiguous.
- **The corroboration mask.** Easy to "turn on because the goal says
  uncorroborated". It stays a measurement by default: enforcing it
  selects for a property hand-verified negatives do not have
  (`docs/KGSAGE_COLUMN.md`, option C).
- **Head corruptions anchor on hubs.** A head swap keeps the tail (a
  place, a genre) as anchor, where "nothing around it points at the
  candidate" reads as neutral rather than contradicting. The case block
  names the anchor and the slot so the model, and the reader, judge the
  right entity; the slot deck makes the two populations comparable.
- **Nulls and repeats.** Dropped and counted, never shipped: a null is a
  true fact, and KGMVAD's injector refuses a file that contains one.
- **Private helpers.** `_cooccurrence` and `_support_banned` are private to
  the package; `neighbourhood.py` recomputes support from the TSVs with
  the same definition rather than importing them.
- **Windows torch.** `set_num_threads(1)` and the mkldnn switch, guarded by
  platform, or the run access-violates with no traceback.
- **Quota.** Free-tier Gemini is 15 requests / minute; batching, pacing and
  atomic resumable writes make a stopped run a paused run.
- **Two writers of one file.** `corrupt.py` creates `corruptions.json`,
  `explain.py` rewrites it (D6). Atomic rename plus the schema version
  keep it safe; `corrupt.py` refuses to overwrite a record that already
  carries reasoning unless `--force`.
- **Browser layout.** Force layouts are random by default; the seeded
  random source and synchronous settling are what make a case
  reproducible. If a library is swapped in (section 6.3 fallback), the
  same two properties must be checked again.
- **Clone trap.** Two KGSAGE clones exist; the editable install points at
  this workspace clone (verified 6 Sep). `python inference/corrupt.py`
  resolves `kgsage` through the install, not the working directory.

## 11. Decisions

Settled by the author on 9 October 2026:

| # | question | decision |
|---|---|---|
| D1 | folder name | `inference/` |
| D2 | LLM stack | `google-genai` direct, structured output; no ADK |
| D3 | criteria | the three goals plus `false` (world knowledge), outside the verdict |
| D4 | CoDEx labels | copy KGMVAD's four definition files into `data/codex-s/`, tracked; KGSAGE stands alone |
| D5 | guards default | on; `--no-guards` reproduces the old column |
| D6 | outputs | one `corruptions.json` enriched in place, atomic writes; all three TSVs |
| D7 | superseded CLIs | retire at M4, after the M3 gate; `knockout_eval.py` stays |
| D8 | graph layout | a React graph component laid out in the browser (d3-force first, cytoscape fallback), not a Python layout |
| D9 | `--slot` | add the backwards-compatible kwarg |
| scope | datasets | CoDEx-S only for this build |

Defaulted to the recommendation, no objection raised (say so if you want
otherwise):

| # | question | default |
|---|---|---|
| D10 | launcher | none; two scripts and one exporter |
| D11 | run naming | deterministic `<dataset>_r<ratio>_s<seed>`, optional `--tag` |
| D12 | decks and tiles | the section 6.2 list |

## 12. Out of scope for this build

- other datasets' label formats (FB15K-237 `entity2text.txt`, WordNet);
- the blind control mode (`format_for_llm.py`'s corrupted-versus-control
  protocol, automated) -- the schema leaves a `blind` slot beside
  `reasoning` for it;
- an ADK agent variant of `explain.py` with on-demand neighbourhood tools;
- a launcher script;
- Wikidata verification of the `false` flags (section 5.4 says why it is
  a separate, SPARQL-based step).

**Status: built, 10 October 2026 -- see [DESIGN.md](DESIGN.md).**
