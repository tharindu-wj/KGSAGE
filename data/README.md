# kgsage/data — the dataset directories

Knowledge-graph **files** live here. The **code** that reads them lives in
[../preprocessing/](../preprocessing/) — the two used to share the name `data/`,
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
   [../preprocessing/registry.py](../preprocessing/registry.py) so
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

Dataset contents are **not** committed — see the `data/` block in
[../.gitignore](../.gitignore). Only this README and the tiny `dummy_kg/`
fixture (used by `smoke_test.py`) are tracked. Add an ignore entry when you add
a large dataset.
