# The KGSAGE interface

Three parts, coupled only through files:

| | what it is | reads | writes |
|---|---|---|---|
| **`kgsage/`** | the package: training and the generation API | `data/` | `outputs/checkpoints/` (the trainer) |
| **`inference/`** | corrupt a dataset with a checkpoint; ask an LLM about each corruption | a checkpoint, `data/<dataset>/` | `outputs/inference/` |
| **`dashboard/`** | case-by-case review of one inference run | `outputs/inference/` | `outputs/dashboard/` |

The package is installed (`pip install -e .`); the other two are
applications run from the repository root. This page is their whole
coupling surface: honour a format here and any part can be rewritten or run
alone. `check_boundaries.py` enforces the rules mechanically.

---

## The rules

1. **The package knows nothing of the applications.** No file under
   `kgsage/` imports `inference`, `dashboard`, or any detector module
   (ADKGD's `kgsage_bridge`, `dataset`). KGSAGE stays standalone.
2. **Each application imports only what it may.** `inference/`: its own
   modules, the standard library, `kgsage`, `numpy`, `torch`, and
   `google-genai` in `explain.py` alone. `dashboard/`: its own modules and
   the standard library -- not `kgsage`, not `inference`. The app in
   `dashboard/app/` fetches nothing but `./data/`.
3. **The LLM stage is torch-free.** `inference/explain.py` and everything it
   imports never touch torch, numpy or kgsage: it reads only the run record,
   so it runs in any environment with `google-genai`.
4. **Nobody writes another part's folder.** `inference/` writes
   `outputs/inference/` and nothing else; `dashboard/` writes
   `outputs/dashboard/` (and its own build output, `dashboard/app/dist/`).
5. **Each application reads formats for itself.** Both carry a
   `formats.py`; the copies are compared, and they are identical today.

---

## Formats

### `outputs/inference/<run>/`

`<run>` is `<dataset>_r<ratio>_s<seed>` (or `_n<count>_`), plus every
non-default choice -- `_tail`, `_noguards`, `_mask<k>`, `_paired`, `_<tag>`
-- so two variants never overwrite each other.

**`corruptions.tsv`** -- the corruptions alone, one per line, three
tab-separated dataset ids, in case order:

```
Q9358	P737	Q8018
```

Every row changes exactly one entity slot of a true triple, keeps the
relation, is not a fact of the dataset in any split, and appears once. This
is the handover file: KGMVAD `inject --source frozen --file … --kind
kgsage`, ADKGD `--anomaly_file`.

**`kg_corrupted.tsv`** -- the dataset's triples (all splits,
deduplicated) with the corruptions mixed in, shuffled by `--seed`, same
three columns.

**`labels.tsv`** -- one row per `kg_corrupted.tsv` row, same order:
`head relation tail label kind`, with `1 kgsage` on a corruption and
`0 real` otherwise (KGMVAD's `ground_truth.tsv` format). A `0` means "not
planted", not "true": the dataset carries its source's own errors.

**`corruptions.json`** -- the run record, schema **`kgsage-inference/1`**.
Written by `inference/corrupt.py`; `inference/explain.py` fills the
`reasoning` fields in place, atomically (temp file + rename) after every
batch. Header fields are indented; every item of `relations`, `entities`
and `cases` sits on one line, so a grep for an id returns the whole item.

```
schema        "kgsage-inference/1"
run, generated
dataset       name, path, triples, entities, relations, labels
              ("codex-json" | "partial" | "ids"), splits {file: sha256}
checkpoint    path, sha256, arch, entities, relations
params        ratio, n, seed, slot, sources, guards, kind_guard_active, types,
              support_max, max_resample, hops, candidate_hops,
              max_neighbours, outer_neighbours, anchor_facts, candidate_facts
environment   python, torch, numpy, kgsage, device
stats         requested, emitted, short, rounds, drawn, nulls, repeats, real,
              surplus, passes, slot_head, slot_tail, candidate_in_pool,
              kind_match {yes, no, unknown}, uncorroborated, shared_any,
              direct_edge, support_above_half,
              corroboration {direct, chain, notable, ordinary, none},
              neighbourhood {nodes_mean, nodes_max, edges_mean, edges_max},
              generator {generate_negatives stats, summed}, render_stats [..]
files         {corruptions_tsv | kg_corrupted_tsv | labels_tsv:
               {path, rows, sha256}}
reasoning     null until explain.py runs; then the run-level summary below
relations     {id: {label, description}}
entities      {id: {label, description, kinds [labels], degree, wiki?}}
cases         [one object per corruption, below]
```

Every id a case mentions is in `entities` / `relations`; `degree` is the
number of facts an entity takes part in, all splits.

**A case:**

```
id            "c0001"...
slot          "head" | "tail" -- which entity was replaced
true          {h, r, t}       the true fact
corrupted     {h, r, t}       the corruption (same r)
anchor        the entity that kept its place
true_filler   the entity replaced
candidate     the replacement
text          {true, corrupted}  "<label> --<relation>-- <label>"
structure     the measurements, below
facts         {anchor: [[h,r,t]..], candidate: [[h,r,t]..]}
              the graph's own facts about each, the corrupted relation's
              first (up to 20 and 10) -- what the LLM is shown
neighbourhood {nodes: [{id, role, hop, near}], edges: [{h, r, t, kind}]}
reasoning     null, or the LLM's block, below
```

`neighbourhood`: the anchor's and the true value's ego to `hops` (default
2), the replacement's to `candidate_hops` (1); a focus entity expands up to
`max_neighbours` (6) context neighbours, any node further out up to
`outer_neighbours` (3), always on top of the entities the case must show
(the three focus entities and up to three shared neighbours, chains first).
Node `role`: `anchor`, `true_filler`, `candidate`, `shared` or `context`;
`hop` is 0 for the focus entities, 1 for shared neighbours, else the
distance to the nearest focus entity, named by `near`. Edge `kind`: `true`,
`corrupted` or `context`.

**`structure`** -- recomputed from the dataset's splits, never read back
from the generator:

| field | meaning |
|---|---|
| `candidate_in_pool` | the replacement has filled this slot of this relation before |
| `kind_match` | it shares a kind with the value it replaced (CoDEx's criterion i; the kind guard); null when either is untyped |
| `kind_peers`, `slot_occupants`, `slot_usual_kinds` | how many of the slot's occupants share a kind with it; how many there are; the top three kinds, `[[label, count]..]` |
| `shared_neighbours`, `shared_sample` | entities linked to both anchor and replacement; up to five, chains first, then least-connected |
| `shared_baseline` | `{typical, as_many, compared}`: the median shared-neighbour count between the anchor and the slot's other entities (not its true values), and the share of them sharing at least as many as the replacement |
| `chains` | shared neighbours X with anchor --r--> X --r--> replacement in the corrupted fact's direction (reversed for a head corruption), counted only for relations recorded both ways at most 25% of the time |
| `corroboration` | one word: `direct` (linked by another relation), `chain`, `notable` (more than typical and in the slot's top fifth), `ordinary`, `none` |
| `bridges` | for up to three shared neighbours: `{via, anchor_side, candidate_side}`, the facts on each side |
| `direct_edge`, `direct_facts` | linked by any relation; up to three such facts |
| `candidate_hop` | 1 direct, 2 through a shared neighbour, null beyond |
| `uncorroborated` | no shared neighbour and no direct link |
| `support` | `{share, value, holders, together}`: max over the anchor's values v (held by five or more) of the share of v's holders that also hold the replacement -- the support guard's own measure; null when no value is thick enough |
| `anchor_values` | the anchor's true values in the corrupted slot |

**A case's `reasoning`:**

```
model, prompt_version, at
verdict    "aligned" (all three goals hold) | "partial" (one fails) |
           "misaligned" (two or more) -- derived by explain.py, never asked
fails      [goal..]
criteria   {type_valid: {holds, why}, plausible: {holds, why},
            uncorroborated: {holds, why},
            false: {holds, confidence: high|medium|low, why}}
summary    one sentence
```

`false` is a separate world-knowledge check -- `holds: false` means the
LLM thinks the corrupted fact is actually true -- and it never enters the
verdict: it is triage for a Wikidata check, not a measurement.

**The header's `reasoning`** (rebuilt from the cases on every save):
`criteria`, `goals`, `explained`, `total`, `models`, `prompt_versions`,
`verdicts {aligned, partial, misaligned}`, `fails {goal: n}`,
`probably_true`, and `sessions` -- one per explain.py run: `started`,
`finished`, `status` (`completed`, `partial`, `truncated` = stopped by
quota, `failed`, `interrupted`, `stopped` = killed), `model`,
`prompt_version`, `batch`, `temperature`, `targeted`, `calls`, `explained`,
`retries`, `tokens {prompt, output, thoughts}`, `errors [..]`.

### `outputs/inference/manifest.json`

The run list, newest first: `[{run, folder, dataset, created, cases,
explained}]`. `corrupt.py` adds a run; `explain.py` keeps `explained`
current.

### `outputs/dashboard/data/<run>.json` and `manifest.json`

The view model the React app reads, schema **`kgsage-dashboard/1`**,
written compact by `dashboard/export.py`: the record's `run`, `dataset`,
`checkpoint` (sha shortened), `params`, a `generation` summary, a
`reasoning` summary, the overview `stats` it computes (verdicts, fails,
probably true, measured corroboration, by slot and by relation), and the
record's `relations`, `entities` and `cases` unchanged. `manifest.json`
lists exported runs newest first: `[{file, run, dataset, exported, cases,
explained}]`.

---

## Contracts with other repositories

KGSAGE has no code dependency on its siblings; they exchange files.

| to | file | how it is used |
|---|---|---|
| KGMVAD | `outputs/inference/<run>/corruptions.tsv` | `python injector/inject.py --source frozen --file … --kind kgsage` -- verified: its guards line reads zero true facts, zero duplicates |
| ADKGD | the same file | `--anomaly_file` (a frozen test column) |
| ADKGD | a checkpoint from `outputs/checkpoints/` | through its own `kgsage_bridge` |
