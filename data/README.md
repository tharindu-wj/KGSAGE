# kgsage/data — the dataset directories

Knowledge-graph **files** live here. The **code** that reads them lives in
[../kgsage/preprocessing/](../kgsage/preprocessing/) — the two used to share the name `data/`,
which is why they are named apart now.

## Layout

One directory per dataset, each holding tab-separated splits:

```
data/
  FB15K-237/
    train.txt        head<TAB>relation<TAB>tail
    valid.txt
    test.txt
  WN18RR/
  YAGO4.5/
```

`valid.txt` and `test.txt` are optional — `load_kg()` tolerates their absence.

## Adding a dataset

1. Create `data/<NAME>/` and drop the splits in.
2. Optionally register a short name in
   [../kgsage/preprocessing/registry.py](../kgsage/preprocessing/registry.py) so
   `resolve_dataset("<name>")` finds it. Registry paths are relative to the
   directory that *contains* `kgsage/`, so they read `kgsage/data/<NAME>`.
3. Pass the directory to any CLI: `--data kgsage/data/<NAME>`.

Unregistered datasets work fine — pass the path directly.

## YAGO 4.5

Not a plain download; convert the `-tiny` Turtle release first:

```bash
python kgsage/preprocessing/yago_to_tsv.py \
    --in  kgsage/data/YAGO-4.5.0.2-tiny.zip \
    --out kgsage/data/YAGO4.5 \
    --min_degree 5 --max_entities 30000
```

## Git

Large dataset contents are **not** committed — see the `data/` block in
[../.gitignore](../.gitignore). Tracked: this README, the tiny `dummy_kg/`
fixture (used by `smoke_test.py`), and `codex-s/` — its three splits plus the
CoDEx release's four definition files, so the inference and dashboard
components run from a fresh clone:

```
codex-s/
  train.txt valid.txt test.txt     head<TAB>relation<TAB>tail, Wikidata ids
  entities/en/entities.json        entity id -> label, description, wiki link
  relations/en/relations.json      relation id -> label, description
  types/entity2types.json          entity id -> its kinds (type ids)
  types/en/types.json              type id -> label
```

The four JSON files are byte-identical to the copies in KGMVAD's
`data/codex-s/` (which keeps its splits under `triples/`; here they stay flat
because the registry reads `codex-s/train.txt`). Add an ignore entry when you
add a large dataset.
