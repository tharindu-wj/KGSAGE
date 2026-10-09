"""Corrupt a share of a dataset with a trained KGSAGE checkpoint, and record
every corruption with its neighbourhood.

    python inference/corrupt.py --ckpt outputs/checkpoints/run_codex-s_s0.pt \
        --data codex-s --ratio 0.05 --seed 0

Run it from the repository root, in an environment with torch (the
pytorch env on this machine). It writes one run folder,
outputs/inference/<dataset>_r<ratio>_s<seed>[_<variant>][_<tag>]/:

    corruptions.tsv   the corruptions alone, head/relation/tail -- the
                      handover file a detector plants (KGMVAD
                      --source frozen, ADKGD --anomaly_file)
    kg_corrupted.tsv  the dataset (all splits, deduplicated) with the
                      corruptions mixed in, shuffled by --seed
    labels.tsv        one row per kg_corrupted.tsv row: h r t label kind,
                      1 / kgsage for a corruption, 0 / real otherwise
    corruptions.json  the record: every corruption with its labels,
                      measured structure, facts and neighbourhood
                      (CONTRACTS.md). explain.py adds the LLM's reasoning
                      to it, in place.

WHAT THE FILES GUARANTEE. Every corruption changes exactly one entity slot
of a real triple, keeps the relation, is NOT a fact of the dataset in any
split, and appears once. Nulls (rows where the generator could only return
the original, true triple) and repeats are dropped on the way out and
counted, so the run asks the generator for more rows than it needs.

GUARDS are on by default: kind + support 0.5 + unique, the v2 eval-column
protocol of 24 Sep 2026. --no-guards reproduces every column made before
it. The corroboration MASK (--support_max) stays off unless asked for:
corroboration is always measured and recorded, never enforced by default,
because enforcing it rejects most hand-verified negatives
(docs/KGSAGE_COLUMN.md, option C).
"""
import argparse
import datetime
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

from paths import (CHECKPOINTS, CORRUPTIONS_TSV, INFERENCE, KG_CORRUPTED_TSV,
                   LABELS_TSV, RECORD, ROOT, RUNS_MANIFEST)

# This checkout's kgsage, ahead of any installed copy. Two clones of this
# repository have lived on one machine, and an editable install pointing
# at the other one made every edit here inert without an error.
sys.path.insert(1, str(ROOT))

# Windows without CUDA: MKL / OpenMP crash inside heavy torch ops unless
# they run single-threaded. Set before numpy and torch load; a no-op on
# every other platform.
if sys.platform == "win32":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import kgsage  # noqa: E402
from kgsage import load_kg, resolve_dataset  # noqa: E402
from kgsage.corruption_generation import (generate_negatives,  # noqa: E402
                                          load_checkpoint, render_stats)

from context import DatasetContext  # noqa: E402
from formats import (load_json, load_triples, sha256_of,  # noqa: E402
                     write_json, write_record, write_tsv)
from neighbourhood import Graph  # noqa: E402

SCHEMA = "kgsage-inference/1"

#: the v2 eval-column guards (generate_negatives' `guards`)
GUARDS = {"kind": True, "support": 0.5, "unique": True}

#: rows drawn per round, as a multiple of the rows still missing: nulls
#: and repeats are dropped, so a round asks for more than it needs
OVERDRAW = 1.5
#: give up after this many rounds rather than loop on a saturated generator
MAX_ROUNDS = 8

#: generate_negatives stats summed over the rounds
GENERATOR_STATS = ("processed", "used_original", "resampled", "type_valid",
                   "slot_h", "slot_t", "corroboration_lifted", "guard_null",
                   "unique_masked")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Corrupt a share of a dataset with a KGSAGE checkpoint.")
    parser.add_argument("--ckpt", required=True,
                        help="a candidate_v2 checkpoint (a bare file name "
                             "is looked up in outputs/checkpoints/)")
    parser.add_argument("--data", required=True,
                        help="registry name (codex-s) or a dataset folder")
    size = parser.add_mutually_exclusive_group()
    size.add_argument("--ratio", type=float, default=None,
                      help="corruptions as a share of the dataset's triples, "
                           "all splits, deduplicated (default 0.05)")
    size.add_argument("--n", type=int, default=None,
                      help="an absolute number of corruptions instead")
    parser.add_argument("--seed", type=int, default=0,
                        help="fixes the sources, the generator's draws and "
                             "the shuffle (default 0)")
    parser.add_argument("--sources", default=None,
                        help="a TSV of TRUE triples to corrupt instead of a "
                             "uniform sample of the dataset (paired sources)")
    parser.add_argument("--slot", choices=("both", "head", "tail"),
                        default="both",
                        help="which slot to corrupt (default both, 50/50)")
    parser.add_argument("--guards", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="kind + support 0.5 + unique (default on); "
                             "--no-guards reproduces the pre-24-Sep column")
    parser.add_argument("--types", default=None,
                        help="entity2types.json for the kind guard (default: "
                             "the dataset's types/entity2types.json)")
    parser.add_argument("--support_max", type=int, default=None,
                        help="turn the corroboration MASK on at this "
                             "tolerance (default off: measured, not enforced)")
    parser.add_argument("--max_resample", type=int, default=None,
                        help="redraws before a null (default 32 guarded, "
                             "8 unguarded)")
    parser.add_argument("--hops", type=int, default=2,
                        help="ego depth around the anchor and the true value "
                             "(default 2)")
    parser.add_argument("--candidate_hops", type=int, default=1,
                        help="ego depth around the candidate (default 1)")
    parser.add_argument("--max_neighbours", type=int, default=6,
                        help="context neighbours a focus entity expands, on "
                             "top of the entities a case must show (default 6)")
    parser.add_argument("--outer_neighbours", type=int, default=3,
                        help="context neighbours each node further out "
                             "expands (default 3)")
    parser.add_argument("--anchor_facts", type=int, default=20,
                        help="facts about the anchor kept for the LLM")
    parser.add_argument("--candidate_facts", type=int, default=10,
                        help="facts about the candidate kept for the LLM")
    parser.add_argument("--device", default=None,
                        help="cpu or cuda (default: cuda when available)")
    parser.add_argument("--out", default=None,
                        help="run folder (default outputs/inference/<run>)")
    parser.add_argument("--tag", default=None,
                        help="suffix for a variant run of the same seed")
    parser.add_argument("--force", action="store_true",
                        help="overwrite a record that already has reasoning")
    args = parser.parse_args(argv)
    if (args.hops < 1 or args.candidate_hops < 0 or args.max_neighbours < 1
            or args.outer_neighbours < 0):
        parser.error("--hops >= 1, --candidate_hops >= 0, "
                     "--max_neighbours >= 1, --outer_neighbours >= 0")
    if args.ratio is not None and not 0 < args.ratio <= 1:
        parser.error("--ratio is a share of the dataset: 0 < ratio <= 1")
    if args.n is not None and args.n < 1:
        parser.error("--n must be at least 1")
    return args


class SourceDraw:
    """Source triples in a seeded random order, each used once per pass.

    A second pass (a fresh permutation) starts only if a run needs more
    sources than the pool holds -- never at the default ratio.
    """

    def __init__(self, pool, rng):
        self.pool, self.rng = pool, rng
        self.order = rng.permutation(len(pool))
        self.cursor, self.passes = 0, 1

    def take(self, count):
        rows = []
        while len(rows) < count:
            if self.cursor == len(self.order):
                self.order = self.rng.permutation(len(self.pool))
                self.cursor, self.passes = 0, self.passes + 1
            stop = min(len(self.order), self.cursor + count - len(rows))
            rows.extend(self.pool[i] for i in self.order[self.cursor:stop])
            self.cursor = stop
        return rows


def relative(path):
    """A path as the record names it: relative to the repository root
    when it lives inside it, absolute otherwise."""
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def find_checkpoint(name):
    path = Path(name)
    if path.exists():
        return path
    if (CHECKPOINTS / path.name).exists():
        return CHECKPOINTS / path.name
    held = sorted(p.name for p in CHECKPOINTS.glob("*.pt"))
    raise SystemExit(f"no such checkpoint: {name}. {relative(CHECKPOINTS)}/ "
                     f"holds: {', '.join(held) or 'nothing'}")


def run_name(dataset, args):
    """<dataset>_r<ratio>_s<seed>, plus every non-default choice, so two
    variants of one seed never overwrite each other."""
    size = f"n{args.n}" if args.n is not None else f"r{(args.ratio or 0.05):g}"
    parts = [dataset, size, f"s{args.seed}"]
    if args.slot != "both":
        parts.append(args.slot)
    if not args.guards:
        parts.append("noguards")
    if args.support_max is not None:
        parts.append(f"mask{args.support_max}")
    if args.sources:
        parts.append("paired")
    if args.tag:
        parts.append(args.tag)
    return "_".join(parts)


def refuse_to_erase_reasoning(record_path, force):
    """LLM reasoning is the one thing a rerun cannot reproduce (the model
    varies between identical calls), so overwriting it takes --force."""
    if force or not record_path.exists():
        return
    try:
        old = load_json(record_path)
    except (ValueError, OSError):
        return
    explained = sum(1 for case in old.get("cases", []) if case.get("reasoning"))
    if explained:
        raise SystemExit(
            f"{relative(record_path)} already carries LLM reasoning for "
            f"{explained} case{'' if explained == 1 else 's'}, which a rerun "
            f"cannot reproduce. Pass "
            f"--force to overwrite it, or --tag NAME to write a new run.")


def check_one_slot(source, corruption):
    """The package's contract, held at the boundary: same relation, exactly
    one entity slot changed. A violation is a bug upstream, not data."""
    changed = [i for i in (0, 2) if source[i] != corruption[i]]
    if source[1] != corruption[1] or len(changed) != 1:
        raise RuntimeError(f"generate_negatives broke its contract: "
                           f"{source} -> {corruption}")


def main(argv=None):
    args = parse_args(argv)
    started = time.time()
    if sys.platform == "win32" and not torch.cuda.is_available():
        torch.set_num_threads(1)
        torch.backends.mkldnn.enabled = False

    # ---- the dataset --------------------------------------------------------
    try:
        config = resolve_dataset(args.data)
    except ValueError as unknown:
        raise SystemExit(str(unknown))
    data_dir = Path(config["path"])
    kg = load_kg(str(data_dir))
    if not kg["triples_train"]:
        raise SystemExit(f"{relative(data_dir)} holds no train.txt triples.")
    id2ent, id2rel = kg["id2ent"], kg["id2rel"]

    def as_strings(triple):
        return (id2ent[triple[0]], id2rel[triple[1]], id2ent[triple[2]])

    real_ids = list(dict.fromkeys(kg["triples_train"] + kg["triples_valid"]
                                  + kg["triples_test"]))
    real = [as_strings(triple) for triple in real_ids]
    real_set = set(real)
    context = DatasetContext(data_dir)
    print(f"dataset   {config['name']}: {len(real):,} triples, "
          f"{kg['n_ent']:,} entities, {kg['n_rel']} relations; "
          f"labels: {context.source}")
    if context.missing:
        print(f"  note: no {', '.join(context.missing)} in {relative(data_dir)}"
              f" -- those ids stay unlabelled")

    # ---- the checkpoint -----------------------------------------------------
    checkpoint = find_checkpoint(args.ckpt)
    guards = dict(GUARDS) if args.guards else None
    types_path = None
    if guards:
        types_path = Path(args.types) if args.types else context.types_path
        if types_path is None or not Path(types_path).exists():
            print("  WARNING: no entity2types.json -- the kind guard has no "
                  "kinds to compare and imposes nothing. Pass --types.")
            types_path = None
    device = torch.device(args.device) if args.device else None
    payload = load_checkpoint(str(checkpoint), device=device,
                              types_path=str(types_path) if types_path else None)
    missing = ([e for e in kg["ent2id"] if e not in payload["ent2id"]]
               + [r for r in kg["rel2id"] if r not in payload["rel2id"]])
    if missing:
        raise SystemExit(
            f"{relative(checkpoint)} was not trained on {config['name']}: "
            f"{len(missing)} of its ids are absent from the checkpoint's "
            f"vocabulary (e.g. {', '.join(missing[:3])}). Train on this "
            f"dataset, or point --data at the one the checkpoint knows.")
    print(f"checkpoint {relative(checkpoint)} on {payload['device']}")

    # ---- what to corrupt ----------------------------------------------------
    if args.n is not None:
        wanted = args.n
    else:
        wanted = max(1, round((args.ratio or 0.05) * len(real)))
    if args.sources:
        given = load_triples(args.sources)
        stray = [triple for triple in given if triple not in real_set]
        if stray:
            raise SystemExit(
                f"{len(stray)} of {len(given)} rows in {args.sources} are not "
                f"facts of {config['name']} (e.g. {chr(9).join(stray[0])}). "
                f"--sources must list TRUE triples to corrupt.")
        pool = list(dict.fromkeys(
            (kg["ent2id"][h], kg["rel2id"][r], kg["ent2id"][t])
            for h, r, t in given))
    else:
        pool = real_ids

    name = run_name(config["name"], args)
    out_dir = Path(args.out) if args.out else INFERENCE / name
    refuse_to_erase_reasoning(out_dir / RECORD, args.force)

    # ---- generation ---------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    draw = SourceDraw(pool, rng)
    max_resample = (args.max_resample if args.max_resample is not None
                    else 32 if guards else 8)
    slot = None if args.slot == "both" else args.slot
    kept, seen = [], set()
    counts = Counter()
    generator = Counter()
    round_lines = []
    rounds = 0
    print(f"corrupting {wanted:,} triples (guards {'on' if guards else 'off'},"
          f" slot {args.slot}, seed {args.seed})")
    while len(kept) < wanted and rounds < MAX_ROUNDS:
        rounds += 1
        rows = draw.take(math.ceil((wanted - len(kept)) * OVERDRAW))
        corruptions, stats = generate_negatives(
            rows, payload, kg, rng=rng, max_resample=max_resample,
            support_max=args.support_max, guards=guards, slot=slot)
        for key in GENERATOR_STATS:
            generator[key] += stats.get(key, 0)
        round_lines.append(render_stats(stats))
        null_at = set(stats["null_indices"])
        counts["drawn"] += len(rows)
        for position, (source, corruption) in enumerate(zip(rows, corruptions)):
            if len(kept) >= wanted:
                counts["surplus"] += 1
                continue
            if position in null_at:
                counts["nulls"] += 1
                continue
            source, corruption = as_strings(source), as_strings(corruption)
            check_one_slot(source, corruption)
            if corruption in real_set:
                counts["real"] += 1
                continue
            if corruption in seen:
                counts["repeats"] += 1
                continue
            seen.add(corruption)
            kept.append((source, corruption))
        print(f"  round {rounds}: {len(kept):,} of {wanted:,} kept "
              f"({counts['nulls']} null, {counts['repeats']} repeated so far)")
    if len(kept) < wanted:
        print(f"  NOTE: asked for {wanted:,}, got {len(kept):,} -- the "
              f"generator ran out of fresh picks in {MAX_ROUNDS} rounds.")

    # ---- the cases ----------------------------------------------------------
    graph = Graph(real, context=context)
    width = max(4, len(str(len(kept))))
    cases = []
    for index, (source, corruption) in enumerate(kept, 1):
        head, relation, tail = source
        corrupted_head, _, corrupted_tail = corruption
        if corrupted_head != head:
            slot_name, anchor, true_filler, candidate = (
                "head", tail, head, corrupted_head)
        else:
            slot_name, anchor, true_filler, candidate = (
                "tail", head, tail, corrupted_tail)
        cases.append({
            "id": f"c{index:0{width}d}",
            "slot": slot_name,
            "true": {"h": head, "r": relation, "t": tail},
            "corrupted": {"h": corruption[0], "r": relation, "t": corruption[2]},
            "anchor": anchor,
            "true_filler": true_filler,
            "candidate": candidate,
            "text": {"true": context.triple_text(source),
                     "corrupted": context.triple_text(corruption)},
            "structure": graph.structure(anchor, relation, slot_name,
                                         true_filler, candidate),
            "facts": {
                "anchor": graph.facts_about(anchor, relation,
                                            args.anchor_facts),
                "candidate": graph.facts_about(candidate, relation,
                                               args.candidate_facts)},
            "neighbourhood": graph.neighbourhood(
                anchor, true_filler, candidate, source, corruption,
                args.hops, args.candidate_hops, args.max_neighbours,
                seed=f"{args.seed}|{'|'.join(corruption)}",
                outer_neighbours=args.outer_neighbours),
            "reasoning": None,
        })

    # ---- the tables every case refers into -----------------------------------
    entity_ids, relation_ids = set(), set()
    for case in cases:
        structure = case["structure"]
        entity_ids.update((case["anchor"], case["true_filler"], case["candidate"]))
        entity_ids.update(structure["shared_sample"])
        entity_ids.update(structure["anchor_values"])
        if structure["support"]:
            entity_ids.add(structure["support"]["value"])
        relation_ids.add(case["true"]["r"])
        for node in case["neighbourhood"]["nodes"]:
            entity_ids.add(node["id"])
        for edge in case["neighbourhood"]["edges"]:
            relation_ids.add(edge["r"])
        listed = [*case["facts"]["anchor"], *case["facts"]["candidate"],
                  *structure["direct_facts"]]
        for bridge in structure["bridges"]:
            entity_ids.add(bridge["via"])
            listed += bridge["anchor_side"] + bridge["candidate_side"]
        for head, relation, tail in listed:
            entity_ids.update((head, tail))
            relation_ids.add(relation)
    entities = {}
    for entity in sorted(entity_ids):
        entry = {"label": context.entity_label(entity),
                 "description": context.entity_description(entity),
                 "kinds": context.kind_labels(entity),
                 "degree": graph.degree[entity]}
        if context.entity_wiki(entity):
            entry["wiki"] = context.entity_wiki(entity)
        entities[entity] = entry
    relations = {relation: {"label": context.relation_label(relation),
                            "description": context.relation_description(relation)}
                 for relation in sorted(relation_ids)}

    # ---- the files ----------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    corrupted = [corruption for _, corruption in kept]
    write_tsv(out_dir / CORRUPTIONS_TSV, corrupted)
    mixed = real + corrupted
    random.Random(args.seed).shuffle(mixed)
    write_tsv(out_dir / KG_CORRUPTED_TSV, mixed)
    write_tsv(out_dir / LABELS_TSV,
              [(*triple, 1, "kgsage") if triple in seen
               else (*triple, 0, "real") for triple in mixed])
    files = {}
    for key, file_name, rows in (("corruptions_tsv", CORRUPTIONS_TSV, len(corrupted)),
                                 ("kg_corrupted_tsv", KG_CORRUPTED_TSV, len(mixed)),
                                 ("labels_tsv", LABELS_TSV, len(mixed))):
        files[key] = {"path": file_name, "rows": rows,
                      "sha256": sha256_of(out_dir / file_name)}

    structures = [case["structure"] for case in cases]
    sizes = [(len(case["neighbourhood"]["nodes"]),
              len(case["neighbourhood"]["edges"])) for case in cases]
    record = {
        "schema": SCHEMA,
        "run": name,
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "dataset": {
            "name": config["name"], "path": relative(data_dir),
            "triples": len(real), "entities": kg["n_ent"],
            "relations": kg["n_rel"], "labels": context.source,
            "splits": {split: sha256_of(data_dir / split)
                       for split in ("train.txt", "valid.txt", "test.txt")
                       if (data_dir / split).exists()}},
        "checkpoint": {"path": relative(checkpoint),
                       "sha256": sha256_of(checkpoint),
                       "arch": payload.get("arch"),
                       "entities": payload["n_ent"],
                       "relations": payload["n_rel"]},
        "params": {
            "ratio": None if args.n is not None else (args.ratio or 0.05),
            "n": wanted, "seed": args.seed, "slot": args.slot,
            "sources": relative(args.sources) if args.sources else "uniform",
            "guards": guards, "kind_guard_active": bool(guards and types_path),
            "types": relative(types_path) if types_path else None,
            "support_max": args.support_max, "max_resample": max_resample,
            "hops": args.hops, "candidate_hops": args.candidate_hops,
            "max_neighbours": args.max_neighbours,
            "outer_neighbours": args.outer_neighbours,
            "anchor_facts": args.anchor_facts,
            "candidate_facts": args.candidate_facts},
        "environment": {"python": sys.version.split()[0],
                        "torch": torch.__version__, "numpy": np.__version__,
                        "kgsage": kgsage.__version__,
                        "device": str(payload["device"])},
        "stats": {
            "requested": wanted, "emitted": len(kept),
            "short": wanted - len(kept), "rounds": rounds,
            "drawn": counts["drawn"], "nulls": counts["nulls"],
            "repeats": counts["repeats"], "real": counts["real"],
            "surplus": counts["surplus"], "passes": draw.passes,
            "slot_head": sum(1 for c in cases if c["slot"] == "head"),
            "slot_tail": sum(1 for c in cases if c["slot"] == "tail"),
            "candidate_in_pool": sum(1 for s in structures
                                     if s["candidate_in_pool"]),
            "kind_match": {
                "yes": sum(1 for s in structures if s["kind_match"] is True),
                "no": sum(1 for s in structures if s["kind_match"] is False),
                "unknown": sum(1 for s in structures if s["kind_match"] is None)},
            "uncorroborated": sum(1 for s in structures if s["uncorroborated"]),
            "shared_any": sum(1 for s in structures if s["shared_neighbours"]),
            "direct_edge": sum(1 for s in structures if s["direct_edge"]),
            "support_above_half": sum(1 for s in structures if s["support"]
                                      and s["support"]["share"] > 0.5),
            "corroboration": {level: sum(1 for s in structures
                                         if s["corroboration"] == level)
                              for level in ("direct", "chain", "notable",
                                            "ordinary", "none")},
            "neighbourhood": {
                "nodes_mean": round(sum(n for n, _ in sizes) / max(len(sizes), 1), 1),
                "nodes_max": max((n for n, _ in sizes), default=0),
                "edges_mean": round(sum(e for _, e in sizes) / max(len(sizes), 1), 1),
                "edges_max": max((e for _, e in sizes), default=0)},
            "generator": dict(generator),
            "render_stats": round_lines},
        "files": files,
        "reasoning": None,
        "relations": relations,
        "entities": entities,
        "cases": cases,
    }
    write_record(out_dir / RECORD, record)

    # ---- the run list the dashboard reads ------------------------------------
    if out_dir.resolve().parent == INFERENCE.resolve():
        runs = load_json(RUNS_MANIFEST) if RUNS_MANIFEST.exists() else []
        runs = [entry for entry in runs if entry.get("run") != name]
        runs.insert(0, {"run": name, "folder": out_dir.name,
                        "dataset": config["name"],
                        "created": record["generated"],
                        "cases": len(cases), "explained": 0})
        write_json(RUNS_MANIFEST, runs)

    s = record["stats"]
    print(f"\nwrote {relative(out_dir)}/  ({time.time() - started:.0f} s)")
    print(f"  {CORRUPTIONS_TSV:17s} {len(corrupted):,} corruptions")
    print(f"  {KG_CORRUPTED_TSV:17s} {len(mixed):,} triples "
          f"({len(real):,} real + {len(corrupted):,} corrupted, shuffled)")
    print(f"  {LABELS_TSV:17s} the answer key for {KG_CORRUPTED_TSV}")
    print(f"  {RECORD:17s} {len(cases):,} cases, {len(entities):,} entities")
    print(f"dropped: {s['nulls']} null, {s['repeats']} repeated, {s['real']} real")
    print(f"slots: head {s['slot_head']:,}, tail {s['slot_tail']:,};  "
          f"kind match {s['kind_match']['yes']:,} of {len(cases):,};  "
          f"uncorroborated {s['uncorroborated']:,};  "
          f"direct edge {s['direct_edge']:,}")
    print(f"next: python inference/explain.py --run {relative(out_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
