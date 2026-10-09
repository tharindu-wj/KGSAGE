"""Independent verifier for one inference run folder. Offline, no model.

    python inference/checks/check_run.py                  the newest run
    python inference/checks/check_run.py outputs/inference/codex-s_r0.05_s0

Recomputes every guarantee and every measured field of the run from the
dataset's own split files and definition JSONs -- importing nothing from
inference/ or kgsage/ -- so a bug in the code under test cannot hide in its
own checks: falseness, uniqueness, one slot changed, the three TSVs and
their hashes, every neighbourhood, and every structure field (shared
neighbours, the slot baseline, chains, the corroboration level, bridges,
direct links, kinds, support, facts). Every line must read PASS.
"""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INFERENCE = ROOT / "outputs" / "inference"
if len(sys.argv) > 1:
    run = Path(sys.argv[1])
    run = run if run.is_absolute() or run.exists() else INFERENCE / sys.argv[1]
else:
    runs = json.loads((INFERENCE / "manifest.json").read_text(encoding="utf-8"))
    run = INFERENCE / runs[0]["folder"]
_header = json.loads((run / "corruptions.json").read_text(encoding="utf-8"))
data = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / _header["dataset"]["path"]
print(f"run {run.name}")
problems = []


def check(label, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"   [{detail}]" if detail and not ok else ""))
    if not ok:
        problems.append(label)


def tsv(path, width):
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line:
            parts = line.split("\t")
            assert len(parts) == width, (path, line)
            rows.append(tuple(parts))
    return rows


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---- the dataset, read raw ----------------------------------------------------
real = []
for split in ("train.txt", "valid.txt", "test.txt"):
    if (data / split).exists():
        real.extend(tsv(data / split, 3))
real = list(dict.fromkeys(real))
real_set = set(real)
neighbours = defaultdict(set)
degree = defaultdict(int)
tails_of, heads_of = defaultdict(list), defaultdict(list)
occupants = defaultdict(set)
for h, r, t in real:
    neighbours[h].add(t); neighbours[t].add(h)
    degree[h] += 1; degree[t] += 1
    tails_of[(h, r)].append(t); heads_of[(r, t)].append(h)
    occupants[(r, "head")].add(h); occupants[(r, "tail")].add(t)
types_file = data / "types" / "entity2types.json"
kinds = json.loads(types_file.read_text(encoding="utf-8")) if types_file.exists() else {}
labels_file = data / "entities" / "en" / "entities.json"
entity_defs = json.loads(labels_file.read_text(encoding="utf-8")) if labels_file.exists() else {}

print(f"dataset {data.name}: {len(real)} unique triples")

# ---- the TSVs -----------------------------------------------------------------
print("\nthe triple files")
corruptions = tsv(run / "corruptions.tsv", 3)
mixed = tsv(run / "kg_corrupted.tsv", 3)
labels = tsv(run / "labels.tsv", 5)
check("corruptions are unique", len(set(corruptions)) == len(corruptions))
true_ones = [c for c in corruptions if c in real_set]
check("no corruption is a fact of the dataset", not true_ones, f"{len(true_ones)} real")
check("kg_corrupted = real + corruptions exactly",
      len(mixed) == len(real) + len(corruptions) and set(mixed) == real_set | set(corruptions))
check("kg_corrupted has no duplicate rows", len(set(mixed)) == len(mixed))
check("labels.tsv aligned row by row", [l[:3] for l in labels] == mixed)
corr_set = set(corruptions)
check("label 1/kgsage exactly on corruptions",
      all((l[3], l[4]) == (("1", "kgsage") if l[:3] in corr_set else ("0", "real")) for l in labels))
check("kg_corrupted is shuffled (corruptions not all at the end)",
      sum(1 for row in mixed[-len(corruptions):] if row in corr_set) < len(corruptions))

# ---- the record ---------------------------------------------------------------
print("\nthe record")
record = json.loads((run / "corruptions.json").read_text(encoding="utf-8"))
cases = record["cases"]
check("schema is kgsage-inference/1", record["schema"] == "kgsage-inference/1")
check("one case per corruption, same order",
      [(c["corrupted"]["h"], c["corrupted"]["r"], c["corrupted"]["t"]) for c in cases] == corruptions)
check("case ids unique", len({c["id"] for c in cases}) == len(cases))
for key, meta in record["files"].items():
    check(f"files.{key} sha256 and rows match", sha(run / meta["path"]) == meta["sha256"]
          and meta["rows"] == len(tsv(run / meta["path"], 5 if "labels" in key else 3)))
for split, digest in record["dataset"]["splits"].items():
    check(f"dataset.splits {split} sha256 matches", sha(data / split) == digest)

bad = defaultdict(list)
GROUPS = {}
RECIP = {}
for c in cases:
    T = (c["true"]["h"], c["true"]["r"], c["true"]["t"])
    C = (c["corrupted"]["h"], c["corrupted"]["r"], c["corrupted"]["t"])
    if T not in real_set: bad["source is real"].append(c["id"])
    if C in real_set: bad["corruption is false"].append(c["id"])
    changed = [i for i in (0, 2) if T[i] != C[i]]
    if T[1] != C[1] or len(changed) != 1: bad["one entity slot changed"].append(c["id"])
    slot = "head" if changed == [0] else "tail"
    if c["slot"] != slot: bad["slot field"].append(c["id"])
    anchor, filler, cand = (T[2], T[0], C[0]) if slot == "head" else (T[0], T[2], C[2])
    if (c["anchor"], c["true_filler"], c["candidate"]) != (anchor, filler, cand):
        bad["anchor/true_filler/candidate"].append(c["id"])
    # neighbourhood
    nb = c["neighbourhood"]
    ids = {n["id"] for n in nb["nodes"]}
    roles = {n["id"]: n["role"] for n in nb["nodes"]}
    if len(ids) != len(nb["nodes"]): bad["node ids unique"].append(c["id"])
    if (roles.get(anchor), roles.get(filler), roles.get(cand)) != ("anchor", "true_filler", "candidate"):
        bad["three focus nodes with roles"].append(c["id"])
    edges = [(e["h"], e["r"], e["t"], e["kind"]) for e in nb["edges"]]
    if (T + ("true",)) not in edges: bad["true edge present"].append(c["id"])
    if (C + ("corrupted",)) not in edges: bad["corrupted edge present"].append(c["id"])
    if any(e[0] not in ids or e[2] not in ids for e in edges): bad["edge endpoints are nodes"].append(c["id"])
    if any(e[3] == "context" and e[:3] not in real_set for e in edges): bad["context edges are real"].append(c["id"])
    if len({e[:3] for e in edges}) != len(edges): bad["edges unique"].append(c["id"])
    if any(n["role"] == "context" and n["hop"] not in (1, 2) for n in nb["nodes"]): bad["context hops 1..2"].append(c["id"])
    # structure, recomputed
    s = c["structure"]
    shared = (neighbours[anchor] & neighbours[cand]) - {anchor, cand}
    direct = cand in neighbours[anchor]
    if s["shared_neighbours"] != len(shared): bad["shared_neighbours"].append(c["id"])
    if set(s["shared_sample"]) - shared: bad["shared_sample within shared"].append(c["id"])
    if s["direct_edge"] != direct: bad["direct_edge"].append(c["id"])
    links = lambda f, a, b: tuple(f) in real_set and {f[0], f[2]} == {a, b}
    if direct != bool(s["direct_facts"]) or not all(links(f, anchor, cand) for f in s["direct_facts"]):
        bad["direct_facts real and linking"].append(c["id"])
    if T[1] not in RECIP:
        r_facts = [x for x in real if x[1] == T[1]]
        RECIP[T[1]] = sum(1 for h, r, t in r_facts if (t, r, h) in real_set) / len(r_facts)
    recip = RECIP[T[1]]
    start, end = (anchor, cand) if slot == "tail" else (cand, anchor)
    chain_vias = [] if recip > 0.25 else sorted(
        (v for v in shared if (start, T[1], v) in real_set and (v, T[1], end) in real_set),
        key=lambda e: (degree[e], e))
    if s["chains"] != chain_vias: bad["chains exact and complete"].append(c["id"])
    least = (chain_vias + [v for v in sorted(shared, key=lambda e: (degree[e], e)) if v not in chain_vias])[:3]
    if [b["via"] for b in s["bridges"]] != least: bad["bridges are the least-connected shared"].append(c["id"])
    for b in s["bridges"]:
        if not b["anchor_side"] or not b["candidate_side"] \
                or not all(links(f, anchor, b["via"]) for f in b["anchor_side"]) \
                or not all(links(f, b["via"], cand) for f in b["candidate_side"]):
            bad["bridge facts real, both sides"].append(c["id"])
            break
    base = s["shared_baseline"]
    expected_level = ("direct" if direct else "chain" if chain_vias else "none" if not shared else
                      "notable" if base and len(shared) > base["typical"] and base["as_many"] <= 0.2
                      else "ordinary")
    if s["corroboration"] != expected_level: bad["corroboration category"].append(c["id"])
    nb_ids = {n["id"] for n in c["neighbourhood"]["nodes"]}
    if any(b["via"] not in nb_ids for b in s["bridges"]): bad["bridges drawn in the neighbourhood"].append(c["id"])
    if s["uncorroborated"] != (not direct and not shared): bad["uncorroborated"].append(c["id"])
    if s["candidate_hop"] != (1 if direct else 2 if shared else None): bad["candidate_hop"].append(c["id"])
    if s["candidate_in_pool"] != (cand in occupants[(T[1], slot)]): bad["candidate_in_pool"].append(c["id"])
    ck, fk = set(kinds.get(cand, [])), set(kinds.get(filler, []))
    if s["kind_match"] != ((bool(ck & fk)) if ck and fk else None): bad["kind_match"].append(c["id"])
    values = tails_of[(anchor, T[1])] if slot == "tail" else heads_of[(T[1], anchor)]
    if s["anchor_values"] != values: bad["anchor_values"].append(c["id"])
    if filler not in values: bad["true filler among anchor values"].append(c["id"])
    # support, from scratch: holders of v in this slot, and how many also hold cand
    if (T[1], slot) not in GROUPS:
        groups = defaultdict(set)
        for h, r, t in real:
            if r == T[1]:
                if slot == "tail": groups[h].add(t)
                else: groups[t].add(h)
        GROUPS[(T[1], slot)] = groups
    groups = GROUPS[(T[1], slot)]
    best = None
    for v in values:
        holding = [g for g in groups.values() if v in g]
        if len(holding) < 5:
            continue
        share = sum(1 for g in holding if cand in g) / len(holding)
        if best is None or share > best:
            best = share
    got = s["support"]["share"] if s["support"] else None
    if (best is None) != (got is None) or (best is not None and abs(best - got) > 1e-3):
        bad["support"].append(c["id"])
    # facts lists are the entity's own real triples, priority relation first
    for who, entity in (("anchor", anchor), ("candidate", cand)):
        facts = [tuple(f) for f in c["facts"][who]]
        if any(f not in real_set or entity not in (f[0], f[2]) for f in facts): bad[f"facts.{who} real and incident"].append(c["id"])
        first = [f[1] == T[1] for f in facts]
        if first != sorted(first, reverse=True): bad[f"facts.{who} priority relation first"].append(c["id"])

checks = ["source is real", "corruption is false", "one entity slot changed", "slot field",
          "anchor/true_filler/candidate", "node ids unique", "three focus nodes with roles",
          "true edge present", "corrupted edge present", "edge endpoints are nodes",
          "context edges are real", "edges unique", "context hops 1..2", "shared_neighbours",
          "shared_sample within shared", "direct_edge", "direct_facts real and linking",
          "chains exact and complete", "bridges are the least-connected shared", "bridge facts real, both sides",
          "bridges drawn in the neighbourhood", "corroboration category", "uncorroborated", "candidate_hop",
          "candidate_in_pool", "kind_match", "anchor_values", "true filler among anchor values",
          "support", "facts.anchor real and incident", "facts.anchor priority relation first",
          "facts.candidate real and incident", "facts.candidate priority relation first"]
print(f"\nper-case checks over {len(cases)} cases")
for name in checks:
    check(name, not bad[name], f"{len(bad[name])} cases, e.g. {bad[name][:3]}")

# ---- the tables -----------------------------------------------------------------
print("\nthe entity and relation tables")
referenced = set()
for c in cases:
    referenced.update((c["anchor"], c["true_filler"], c["candidate"]))
    referenced.update(n["id"] for n in c["neighbourhood"]["nodes"])
    for f in c["facts"]["anchor"] + c["facts"]["candidate"]:
        referenced.update((f[0], f[2]))
check("every referenced entity is in the table", referenced <= set(record["entities"]))
wrong_degree = [e for e, v in record["entities"].items() if v["degree"] != degree[e]]
check("entity degrees match the graph", not wrong_degree, f"{len(wrong_degree)}")
wrong_label = [e for e, v in record["entities"].items()
               if entity_defs and v["label"] != (entity_defs.get(e, {}).get("label") or e)]
check("entity labels match entities.json", not wrong_label, f"{len(wrong_label)}")

# ---- the guarantees of a guarded run --------------------------------------------
if record["params"]["guards"]:
    print("\nguarded-run properties")
    over = [c["id"] for c in cases if c["structure"]["support"] and c["structure"]["support"]["share"] > 0.5]
    check("support guard: no candidate the graph predicts above 0.5", not over, f"{len(over)}")
    if record["params"]["kind_guard_active"]:
        mismatch = [c["id"] for c in cases if c["structure"]["kind_match"] is False]
        check("kind guard: no wrong-kind replacement", not mismatch, f"{len(mismatch)}")

st = record["stats"]
print(f"\nstats: emitted {st['emitted']}  nulls {st['nulls']}  repeats {st['repeats']}  "
      f"head {st['slot_head']} tail {st['slot_tail']}  uncorroborated {st['uncorroborated']}  "
      f"direct {st['direct_edge']}  shared_any {st['shared_any']}")
print(f"neighbourhood size: {st['neighbourhood']}")
print(f"record size: {(run / 'corruptions.json').stat().st_size / 1e6:.1f} MB")
print(f"\n{'ALL PASS' if not problems else str(len(problems)) + ' FAILED: ' + ', '.join(problems)}")
sys.exit(1 if problems else 0)
