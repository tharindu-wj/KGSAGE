# -*- coding: utf-8 -*-
"""Emit N corruptions as a plain head/relation/tail TSV in the dataset's
own string ids.

The generic handover format. gen_corruptions_csv.py writes a rich CSV for
this repo's own LLM and ego-graph evaluations, and gates rows on a
hand-written relation template list; a downstream DETECTOR wants neither --
it wants every relation and three columns:

    Q7259<TAB>P26<TAB>Q1276

Run from the directory that contains `kgsage/` (pytorch env):
  python -m kgsage.cli.gen_plain_tsv \
      --ckpt outputs/checkpoints/generator_codex-s.pt \
      --data codex-s --n 700 --seed 42 \
      --out outputs/eval/codex_s_kgsage_negatives.tsv

WHAT IS GUARANTEED IN THE FILE. Every row is a corruption the generator
picked, translated back to strings, and it is NOT a fact the dataset holds
in any split. Two things are dropped on the way out and counted in the
summary:

  nulls       rows where every redraw failed and the decoder emitted the
              ORIGINAL triple (generate_negatives reports these in
              stats['null_indices']). A null is a real fact; shipping one
              would hand a detector a true triple labelled false.
  repeats     the same corruption drawn twice, and any row that turns out
              to be a real triple after translation.

The count in the file is therefore <= --n. Ask for more than you need.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from kgsage.corruption_generation import load_checkpoint, generate_negatives
from kgsage.preprocessing.loaders import load_kg
from kgsage.preprocessing.registry import resolve_dataset

#: how much to over-draw per round, since nulls and repeats are dropped
OVERDRAW = 1.5

#: give up after this many rounds rather than loop on a saturated generator
MAX_ROUNDS = 6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="a candidate_v2 .pt")
    parser.add_argument("--data", required=True,
                        help="registry name (e.g. codex-s) or a directory")
    parser.add_argument("--n", type=int, default=700,
                        help="how many corruptions to write (default 700)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True, help="the TSV to write")
    parser.add_argument("--support_max", type=int, default=None,
                        help="corroboration tolerance; omit to leave the "
                             "mask off (generate_negatives' own default)")
    args = parser.parse_args()

    config = resolve_dataset(args.data)
    kg = load_kg(config["path"])
    payload = load_checkpoint(args.ckpt)

    # The checkpoint and the dataset must share a vocabulary: translation
    # goes caller id -> string -> generator id, so a checkpoint trained on
    # another graph fails on the first lookup. Say so here, with the two
    # vocabularies named, rather than in a KeyError a thousand rows later.
    missing = [s for s in list(kg["ent2id"])[:200]
               if s not in payload["ent2id"]]
    if missing:
        raise SystemExit(
            f"{args.ckpt} was not trained on {args.data}: "
            f"{len(missing)} of the first 200 entity ids are absent from the "
            f"checkpoint's vocabulary (e.g. {', '.join(missing[:3])}). The "
            f"checkpoint holds {len(payload['ent2id'])} entities, this "
            f"dataset {kg['n_ent']}. Train on {args.data}, or point --data "
            f"at the graph the checkpoint was trained on.")

    real = (kg["triples_train"] + kg["triples_valid"] + kg["triples_test"])
    real_set = kg["triple_set_all"]
    id2ent, id2rel = kg["id2ent"], kg["id2rel"]
    id_maps = {"id2ent": id2ent, "id2rel": id2rel,
               "ent2id": kg["ent2id"], "rel2id": kg["rel2id"]}

    rng = np.random.default_rng(args.seed)
    kept, seen = [], set()
    nulls = repeats = 0

    for round_number in range(1, MAX_ROUNDS + 1):
        if len(kept) >= args.n:
            break
        want = args.n - len(kept)
        draw = min(len(real), int(math.ceil(want * OVERDRAW)))
        rows = [real[i] for i in rng.choice(len(real), size=draw,
                                            replace=False)]
        corruptions, stats = generate_negatives(
            rows, payload, id_maps, rng=rng, support_max=args.support_max)
        null_at = set(stats.get("null_indices", []))
        nulls += len(null_at)

        for position, triple in enumerate(corruptions):
            if position in null_at or len(kept) >= args.n:
                continue
            if tuple(triple) in real_set:
                repeats += 1        # a real fact: never ship it as false
                continue
            as_strings = (id2ent[triple[0]], id2rel[triple[1]],
                          id2ent[triple[2]])
            if as_strings in seen:
                repeats += 1
                continue
            seen.add(as_strings)
            kept.append(as_strings)
        print(f"  round {round_number}: {len(kept)} of {args.n} kept "
              f"({nulls} null, {repeats} repeated or real, so far)",
              flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for head, relation, tail in kept:
            f.write(f"{head}\t{relation}\t{tail}\n")

    print(f"\nwrote {out}: {len(kept)} corruptions from {args.data} "
          f"(seed {args.seed})")
    print(f"dropped on the way out: {nulls} null, {repeats} repeated or real")
    if len(kept) < args.n:
        print(f"NOTE: asked for {args.n}, got {len(kept)} -- the generator "
              f"ran out of fresh picks. Raise --n or lower what you plant.")


if __name__ == "__main__":
    main()
