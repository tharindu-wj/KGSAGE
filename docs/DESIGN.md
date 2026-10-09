# inference/ and dashboard/ -- the build record

Built 9 to 10 October 2026 from the approved plan
([INFERENCE_DASHBOARD_PLAN.md](INFERENCE_DASHBOARD_PLAN.md)), CoDEx-S only.
This page records what the build decided beyond the plan, and the
measurement behind each decision, in the order they were made.

## 1. What was delivered

| milestone | delivered |
|---|---|
| M0 | `inference/`, `dashboard/`, `CONTRACTS.md`, `check_boundaries.py`, `.env.example`, gitignore entries; the four CoDEx definition files copied into `data/codex-s/` (byte-identical to KGMVAD's) |
| M1 | `inference/corrupt.py`, `context.py`, `neighbourhood.py`; `generate_negatives(..., slot=None)` in the package |
| M2 | `inference/explain.py`, `prompts.py` (prompt `explain/8`) |
| M3 | `dashboard/export.py` and the React app with a d3-force graph component |
| M4 | the five superseded CLIs retired; README, data README, `pyproject.toml` extras, `kgsage/cli` docstring updated; offline checks in `inference/checks/` and `dashboard/app/test/` |

## 2. Decisions taken during the build

**The package change is invisible by default.** `slot=None` keeps every
recorded run's behaviour; the 50/50 coin is drawn even when a slot is
forced, so a forced slot leaves the caller's rng exactly where `None` does.
Proven against a baseline captured before the edit: identical corruptions,
stats and rng state for four configurations (unguarded at two seeds,
guarded, corroboration mask). `inference/checks/check_slot.py` keeps the
properties checked.

**The record normalises its tables.** Labels, descriptions, kinds and
degrees live once in top-level `entities` / `relations` tables, not inside
every node; header fields are indented and every case sits on one line.
Writes are atomic (temp file + rename, retried on a Windows sharing error),
because `explain.py` rewrites the record while the dashboard exporter may be
reading it.

**Must-show entities ride on top of the neighbourhood cap.** The retired
ego-figure walk counted kept entities against the cap of six. Once shared
neighbours were kept as well, a well-connected anchor's cap filled with
them and the 1-hop view showed no context at all (seen on the first case
card). Kept entries now come on top of the cap: 1-hop views went from no
context nodes to 13 at the least and about 18 on average.

**The outer ring gets its own cap.** With kept entries on top, every 1-hop
node also expanded six context neighbours: 93 nodes per case and a 27 MB
record. Nodes further out now expand three (`--outer_neighbours`): about 60
nodes at two hops, about 21 at one, and a 19 MB record.

**The LLM needed measurements, not adjectives** (prompts `explain/1` to
`explain/8`, each probed live before the next):

| version | change | what the probe showed |
|---|---|---|
| explain/1 | first wording | a borderline "why" could argue one way and "holds" the other |
| explain/2 | holds follows the sentence; plausible means "in character" | in 990 judged cases `uncorroborated` never failed: the model read it as "no direct link", true of every corruption by construction (run stopped to save quota) |
| explain/3 | the case block shows the paths through shared neighbours and any direct link | all five direct links failed; two-step paths still waved away as "not direct" |
| explain/4 | never argue from the missing link; a specific two-step path counts | swung too far: any non-hub path failed, diplomatic triangles included |
| explain/5 | a baseline: how many neighbours the slot's typical alternative shares | balanced but inconsistent at the threshold |
| explain/6 | the measure as one word: DIRECT, NOTABLE, ORDINARY, NONE | direct 5/5 failed; ordinary and none 14/14 held; notable overridden on semantic grounds -- but an influence chain still called "not direct" |
| explain/7 | "a chain of the corrupted relation counts" | no change: this model would not take it from wording |
| explain/8 | the chain is measured (`Graph.chains`) and becomes its own level, CHAIN | direct 5/5 and chain 8/8 failed; notable 2 of 7 failed where the path bears on the fact; ordinary and none all held -- adopted |

The measurements that came out of this are now first-class fields of the
record: `shared_baseline` (the slot's typical sharing and the share of
alternatives that match the replacement), `chains` (directed relations
only -- above 25% reciprocity a chain is a mere triangle), `corroboration`
(direct, chain, notable, ordinary, none), `bridges` and `direct_facts`. The
dashboard shows them beside the LLM's call, so a reader always sees both
the structural and the semantic view.

**The verdict is derived, never asked.** `aligned` / `partial` /
`misaligned` comes from the three goals' `holds`; the model writes only the
criteria and a summary, so the verdict cannot disagree with them.
"False in the world" stays outside it, as triage for a Wikidata check.

**The key is never copied.** The user's Gemini key lives in KGMVAD's
`.env`; `explain.py --env-file ../KGMVAD/.env` reads it in place, and only
its source is printed.

**The graph layout is label-aware.** The first screenshots showed shared
neighbours' labels piled on each other and on the corrupted edge. Three
changes, each measured by `dashboard/app/test/layout.test.mjs` over 203
cases at both depths: a label-box collision force plus a deterministic
clean-up pass (overlapping label pairs 2,411 without it, 0 with it); the
two highlighted edges' relation labels as fixed obstacles; and shared
neighbours on slack springs with a home below the corrupted edge -- two
pinned endpoints had been dragging them onto that very line. Final:
10 overlapping label pairs in 406 layouts, deterministic, pins held.

**Colours were computed, not chosen by eye** (the dataviz validator): the
graph roles are reference categorical slots 1 to 3, which pass every
all-pairs colour-blind gate in both themes (worst deutan separation 9.2
light, 9.4 dark); the old figures' red-against-green failed that gate at
1.8. Measured corroboration is an ordinal one-hue ramp, validated in both
themes. Verdicts wear status colours with an icon and a word.

**Old ego figure, compared and retired.** Rendered for the same case, the
retired `ego_from_csv.py` showed the same content (anchor, true value,
replacement, both edges, one- and two-hop context) with raw ids only, and a
wrong caption: it called the replacement "inside the neighbourhood (hop
0)", counting the replacement's own ego. The dashboard's SVG export,
checked separately, is well-formed, self-contained (no CSS variables) and
renders on its own.

## 3. The full CoDEx-S run

Run `codex-s_r0.05_s0`: 1,827 corruptions (5% of 36,543 facts), seed 0,
guards on, corroboration mask off. Judged by `gemini-3.5-flash-lite` under
`explain/8` in 183 calls (27 probe cases, then 1,800), with no malformed
answer and no retry: 1.47 M prompt and 0.40 M output tokens, about 32
minutes on the free tier.

| | cases | share |
|---|---|---|
| aligned (all three goals hold) | 1,110 | 61% |
| partial (one fails) | 705 | 39% |
| misaligned (two or more fail) | 12 | 1% |
| fails type-valid | 2 | 0.1% |
| fails plausible | 602 | 33% |
| fails uncorroborated | 125 | 7% |
| probably true in the world | 319 | 17% (high confidence 208, medium 66, low 45) |

The LLM's `uncorroborated` against the measured level:

| measured | held | failed |
|---|---|---|
| direct | 0 | 5 |
| chain | 0 | 8 |
| notable | 41 | 42 |
| ordinary | 375 | 69 |
| none | 1,286 | 1 |

- **The kind guard does its job:** two type-valid failures in 1,827.
- **Plausibility is where the generator fails**, and it concentrates by
  relation: 259 of 569 occupations and 71 of 101 citizenships are out of
  character -- people moved to professions and countries that do not fit
  them. That is the "semantically obvious to a knowledge-based judge"
  weakness `docs/KGSAGE_COLUMN.md` found from the detector side.
- **Probably true concentrates in diplomatic relations:** 221 of 309. The
  graph lacks many real diplomatic ties, so a "false" diplomatic corruption
  is often a true fact; that relation needs the Wikidata check before its
  corruptions serve as anomalies.
- Tail corruptions align more often than head ones (65% against 57%);
  probably true is level (18% against 17%).
- On the 990 cases both prompts judged, explain/2 to explain/8 moved
  uncorroborated failures from 0 to 72, plausible failures from 393 to 342,
  and probably true from 203 to 169; the "false" check agreed on 91%.

## 4. Checks, all passing on 10 October 2026

| check | what it proves |
|---|---|
| `inference/checks/check_run.py` | every guarantee and measurement of a run, recomputed from the raw splits (falseness, uniqueness, one slot, TSVs and hashes, neighbourhoods, kinds, support, shared neighbours, baseline, chains, levels, bridges, facts, tables) |
| `inference/checks/check_explain.py` | 35 checks of the LLM stage against a scripted fake model: malformed answers never written, quota stop, resume, `--ids`, Ctrl-C, verdict derivation, every case renders |
| `inference/checks/check_slot.py` | the `slot=` option: None unchanged, forced slots exact, rng stream preserved, bad values refused |
| `dashboard/app/test/layout.test.mjs` | deterministic layout, pins, finite positions, edge shapes, context on its own side, label overlaps |
| `check_boundaries.py` | 14 component rules; shown to fail on planted violations (an import, a transitive torch import, a write into another component's folder) |
| `smoke_test.py` | the package, after the CLI retirement |
| determinism | a second process with the same seed writes byte-identical TSVs and a record that differs only in its timestamp |
| handover | `corruptions.tsv` planted through a scratch copy of KGMVAD's injector: `GUARDS planted-but-actually-true 0 duplicates 0` |

## 5. Known limits

- **CoDEx-S only.** Other datasets run with raw ids until their label
  files are added (`inference/context.py` reads the CoDEx layout).
- **The LLM is a judge, not a measurement.** Its calls are reasoned but
  noisy at the margins: about 1% of plausibility reasons argued against
  their own mark under explain/2, and on NOTABLE cases it decides on the
  meaning of the paths. "False in the world" is triage; the reported
  false-anomaly rate still needs the Wikidata check
  (`docs/KGSAGE_COLUMN.md`, option G).
- **Summaries** sometimes use the schema's names (`type_valid`) rather than
  plain words.
- **Size.** The record and the view model are about 19 MB, the single-file
  dashboard about 20 MB; fine locally, too big to publish as a single page.
- **Dense hubs.** Two-hop views of a hub anchor (a country, an occupation)
  hold about 60 nodes; the list view is the readable twin.
