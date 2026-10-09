# inference/ -- corrupt a dataset, then ask why

Two command-line tools over the `kgsage` package. `corrupt.py` uses a
trained checkpoint to corrupt a chosen share of a dataset and records every
corruption with its neighbourhood; `explain.py` hands each corruption to an
LLM, which judges it against KGSAGE's three goals and writes its reasoning
back into the same record. The dashboard (`../dashboard/`) reads that
record. The formats are in [../CONTRACTS.md](../CONTRACTS.md).

Built and tested on **CoDEx-S**, the dataset whose definition files
(labels, descriptions, kinds) live in `data/codex-s/`.

## Run it

```bash
# 1. corrupt 5% of CoDEx-S (the pytorch env; seconds)
python inference/corrupt.py --ckpt outputs/checkpoints/run_codex-s_s0.pt \
    --data codex-s --ratio 0.05 --seed 0

# 2. the LLM's reasoning (any env with google-genai; ~30 min for 1,827 cases)
python inference/explain.py --run codex-s_r0.05_s0
python inference/explain.py --run codex-s_r0.05_s0 --limit 20   # try 20 first
python inference/explain.py --run codex-s_r0.05_s0 --dry-run    # print, call nothing
```

Run both from the repository root. On this machine, step 1 runs in
`miniconda3/envs/pytorch` and step 2 in the base env, which has
`google-genai`; `pip install -e ".[inference]"` adds it anywhere else.

## `corrupt.py`

Writes `outputs/inference/<run>/`, where `<run>` is
`<dataset>_r<ratio>_s<seed>` plus any non-default choice (`_tail`,
`_noguards`, `_mask<k>`, `_paired`, `_<tag>`):

| file | what it holds |
|---|---|
| `corruptions.tsv` | the corruptions alone, `head relation tail` -- the handover file a detector plants |
| `kg_corrupted.tsv` | the dataset (all splits, deduplicated) with the corruptions mixed in, shuffled by `--seed` |
| `labels.tsv` | the answer key for it, one row per row: `h r t 1 kgsage` or `h r t 0 real` |
| `corruptions.json` | the record: every corruption with its labels, measurements, facts and neighbourhood |

**What the files guarantee.** Every corruption changes exactly one entity
slot of a real triple, keeps the relation, is not a fact of the dataset in
any split, and appears once. Nulls (rows where the generator could only
return the original true triple) and repeats are dropped and counted; the
run asks for more rows than it needs. The same seed writes byte-identical
TSVs and the same record in any process.

| flag | default | |
|---|---|---|
| `--ratio` / `--n` | `0.05` | share of the dataset's triples, or a count |
| `--seed` | `0` | sources, generator noise, shuffle |
| `--slot` | `both` | `head` or `tail` forces one slot |
| `--guards` / `--no-guards` | on | kind + support 0.5 + unique, the 24 Sep 2026 v2 protocol; off reproduces older columns |
| `--support_max N` | off | the corroboration MASK; off on purpose -- corroboration is measured, not enforced (`docs/KGSAGE_COLUMN.md`, option C) |
| `--sources FILE` | uniform | corrupt these true triples instead (paired sources) |
| `--hops`, `--candidate_hops` | 2, 1 | ego depth around anchor and true value, and around the replacement |
| `--max_neighbours`, `--outer_neighbours` | 6, 3 | context a focus entity expands, and each node further out |
| `--tag`, `--out`, `--force` | | name a variant; place the run; overwrite a record that already holds LLM reasoning (refused otherwise) |

**The measurements** (`structure` in each case) are recomputed from the
dataset's splits by `neighbourhood.py`, never read back from the generator:
kind match and the slot's usual kinds; shared neighbours and a baseline --
how many the slot's other entities typically share with the anchor; chains,
where the corrupted relation runs twice through one entity; one
corroboration level (`direct`, `chain`, `notable`, `ordinary`, `none`); the
paths through the shared neighbours; and support, the support guard's own
measure. `CONTRACTS.md` defines each.

## `explain.py`

Judges the cases that have no reasoning yet, in batches of 10 (one call
each), against four criteria:

- **type-valid** -- the replacement is the kind of thing this slot holds;
- **plausible** -- the corrupted fact is in character for these entities;
- **uncorroborated** -- the anchor's neighbourhood gives the replacement no
  support, judged from the measured level and the paths: `direct` and
  `chain` fail, `notable` fails unless every path is generic, `ordinary`
  and `none` hold unless a path makes the fact expected;
- **false in the world** -- a separate check: is the corrupted fact
  actually false? `holds: false` means probably TRUE.

The verdict -- `aligned` (all three goals), `partial` (one fails),
`misaligned` -- is derived here, never asked of the model, and "false"
never enters it: it is triage for a Wikidata check.

- **Model and pacing.** `gemini-3.5-flash-lite` (KGMVAD's detector model)
  through `google-genai` with structured JSON output, temperature 0; 4 s
  between calls for the free tier's 15 requests a minute; the client
  retries a refused call with backoff (5 attempts, 10 to 70 s).
- **The key.** `GOOGLE_API_KEY` from `--env-file FILE`, then the
  environment, then a `.env` beside the repository or one level up
  (`../.env.example` is the template). Only the key's source is printed.
- **Resumable.** The record is rewritten atomically after every batch, so a
  quota stop, a crash or Ctrl-C loses at most one batch; the same command
  carries on. A malformed answer is never written -- that case is asked
  again next time. `--redo` re-judges, `--ids c0001,c0002` judges those.
- **Provenance.** Each case's block names its model, prompt version and
  time; the header keeps one session per run (status, calls, tokens,
  retries, errors). `prompts.py` holds the wording and its version history:
  change the instruction, the case block or the schema and the version must
  change too.

## Checks (offline)

```bash
python inference/checks/check_run.py       # recompute every guarantee and measurement of a run
python inference/checks/check_explain.py   # the LLM stage against a scripted fake model
python inference/checks/check_slot.py      # the generator's slot option (pytorch env)
python check_boundaries.py                 # the component rules (repository root)
```

`check_run.py` imports nothing from `inference/` or `kgsage/`: it recomputes
each property from the raw split files, so a bug cannot hide in its own
check. `check_explain.py` works on a temporary 23-case copy.

## Layout

```
corrupt.py        the corruption CLI
explain.py        the LLM CLI
neighbourhood.py  the dataset's graph and every per-case measurement
context.py        CoDEx definition files: labels, descriptions, kinds
prompts.py        the instruction, the case block, the response schema, versioned
formats.py        TSV and JSON readers and writers, atomic record writes
paths.py          where things are
checks/           the offline proofs
```

The imports are flat (`from paths import ...`): each script runs from the
repository root with `inference/` as its own path. `corrupt.py` puts the
repository root on the path ahead of any installed `kgsage`, so it always
runs against this checkout's package.
