"""Export one inference run as the dashboard's view model. Nothing is judged here.

    python dashboard/export.py                      the newest run
    python dashboard/export.py --run codex-s_r0.05_s0
    python dashboard/export.py --serve              export, then the Vite dev server
    python dashboard/export.py --build              export, then a static dist/
    python dashboard/export.py --build --single     one self-contained HTML

Reads outputs/inference/<run>/corruptions.json -- the record the inference
component wrote, and the dashboard's only data source -- and writes
outputs/dashboard/data/<run>.json plus manifest.json. The app in
dashboard/app/ serves outputs/dashboard/ as its static root, so what is
written here is what the browser fetches.

The frontend holds no dataset knowledge: every label, kind, description,
measurement and LLM verdict leaves here inside the view model. It computes
two things itself, both by design: the deck filters (pure groupings of the
fields below) and the graph layout (d3-force in the browser, seeded, so a
case draws the same picture every time).

A record that is still being explained exports fine: cases without
reasoning say so on their card, and exporting again later picks up the
rest.
"""
import argparse
import collections
import datetime
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import paths
from formats import load_json, write_json

RECORD_SCHEMA = "kgsage-inference/1"
VIEW_SCHEMA = "kgsage-dashboard/1"
VERDICTS = ("aligned", "partial", "misaligned")
GOALS = ("type_valid", "plausible", "uncorroborated")
#: the measured corroboration levels, weakest first (inference/neighbourhood.py)
CORROBORATION = ("none", "ordinary", "notable", "chain", "direct")
#: relations listed in the overview's table and decks
TOP_RELATIONS = 10


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Export an inference run for the dashboard.")
    parser.add_argument("--run", default=None,
                        help="run folder, run name or corruptions.json "
                             "(default: the newest run)")
    parser.add_argument("--serve", action="store_true",
                        help="after exporting, start the Vite dev server")
    parser.add_argument("--build", action="store_true",
                        help="after exporting, build the static dist/")
    parser.add_argument("--single", action="store_true",
                        help="with --build: one self-contained HTML, data inlined")
    return parser.parse_args(argv)


def resolve_record(name):
    if name is None:
        runs = (load_json(paths.RUNS_MANIFEST)
                if paths.RUNS_MANIFEST.exists() else [])
        if not runs:
            raise SystemExit("no inference runs yet -- make one with "
                             "inference/corrupt.py")
        return paths.INFERENCE / runs[0]["folder"] / paths.RECORD
    path = Path(name)
    if path.is_file():
        return path
    folder = path if path.exists() else paths.INFERENCE / name
    record = folder / paths.RECORD
    if not record.exists():
        held = sorted(p.parent.name for p in paths.INFERENCE.glob(f"*/{paths.RECORD}"))
        raise SystemExit(f"no {paths.RECORD} in {folder}. Runs on disk: "
                         f"{', '.join(held) or 'none'}")
    return record


def relative(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(paths.ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def tally(cases):
    """Verdicts, failed goals and probably-true counts over some cases."""
    explained = [case["reasoning"] for case in cases if case.get("reasoning")]
    return {
        "cases": len(cases),
        "explained": len(explained),
        "verdicts": {verdict: sum(1 for block in explained
                                  if block["verdict"] == verdict)
                     for verdict in VERDICTS},
        "fails": {goal: sum(1 for block in explained if goal in block["fails"])
                  for goal in GOALS},
        "probably_true": sum(1 for block in explained
                             if not block["criteria"]["false"]["holds"]),
    }


def overview_stats(record):
    cases = record["cases"]
    structures = [case["structure"] for case in cases]
    by_relation = collections.defaultdict(list)
    for case in cases:
        by_relation[case["true"]["r"]].append(case)
    ranked = sorted(by_relation.items(), key=lambda item: (-len(item[1]), item[0]))
    probably_true = [case for case in cases if case.get("reasoning")
                     and not case["reasoning"]["criteria"]["false"]["holds"]]
    return {
        **tally(cases),
        "slot": {slot: sum(1 for case in cases if case["slot"] == slot)
                 for slot in ("head", "tail")},
        "by_slot": {slot: tally([case for case in cases if case["slot"] == slot])
                    for slot in ("head", "tail")},
        "measured": {
            "candidate_in_pool": sum(1 for s in structures if s["candidate_in_pool"]),
            "kind_match": sum(1 for s in structures if s["kind_match"] is True),
            "kind_mismatch": sum(1 for s in structures if s["kind_match"] is False),
            "uncorroborated": sum(1 for s in structures if s["uncorroborated"]),
            "shared_any": sum(1 for s in structures if s["shared_neighbours"]),
            "direct_edge": sum(1 for s in structures if s["direct_edge"]),
            "corroboration": {level: sum(1 for s in structures
                                         if s["corroboration"] == level)
                              for level in CORROBORATION},
        },
        "false_confidence": {level: sum(
            1 for case in probably_true
            if case["reasoning"]["criteria"]["false"]["confidence"] == level)
            for level in ("high", "medium", "low")},
        "relations": [{"relation": relation, **tally(group)}
                      for relation, group in ranked],
        "top_relations": [relation for relation, _ in ranked[:TOP_RELATIONS]],
    }


def view_model(record, record_path):
    reasoning = record.get("reasoning") or {}
    sessions = reasoning.get("sessions", [])
    return {
        "schema": VIEW_SCHEMA,
        "run": record["run"],
        "source": relative(record_path),
        "exported": datetime.datetime.now().isoformat(timespec="seconds"),
        "generated": record["generated"],
        "dataset": {key: record["dataset"][key] for key in
                    ("name", "triples", "entities", "relations", "labels")},
        "checkpoint": {"path": record["checkpoint"]["path"],
                       "sha256": record["checkpoint"]["sha256"][:12],
                       "arch": record["checkpoint"]["arch"]},
        "params": record["params"],
        "generation": {key: record["stats"][key] for key in
                       ("requested", "emitted", "short", "nulls", "repeats",
                        "rounds", "drawn")},
        "reasoning": {
            "models": reasoning.get("models", []),
            "prompt_versions": reasoning.get("prompt_versions", []),
            "explained": reasoning.get("explained", 0),
            "total": len(record["cases"]),
            "sessions": [{key: session.get(key) for key in
                          ("started", "finished", "status", "model",
                           "prompt_version", "calls", "explained", "retries")}
                         | {"errors": len(session.get("errors", []))}
                         for session in sessions],
        },
        "stats": overview_stats(record),
        "relations": record["relations"],
        "entities": record["entities"],
        "cases": [{key: case[key] for key in
                   ("id", "slot", "true", "corrupted", "anchor", "true_filler",
                    "candidate", "text", "structure", "facts",
                    "neighbourhood", "reasoning")}
                  for case in record["cases"]],
    }


def export(record_path):
    record = load_json(record_path)
    if record.get("schema") != RECORD_SCHEMA:
        raise SystemExit(f"{relative(record_path)} is not a {RECORD_SCHEMA} "
                         f"record (schema {record.get('schema')!r}) -- this "
                         f"exporter reads only what inference/corrupt.py writes")
    view = view_model(record, record_path)
    out = paths.DATA_OUT / f"{record['run']}.json"
    write_json(out, view, compact=True)

    manifest_path = paths.DATA_OUT / "manifest.json"
    manifest = load_json(manifest_path) if manifest_path.exists() else []
    manifest = [entry for entry in manifest if entry.get("file") != out.name]
    manifest.insert(0, {"file": out.name, "run": record["run"],
                        "dataset": record["dataset"]["name"],
                        "exported": view["exported"],
                        "cases": len(view["cases"]),
                        "explained": view["reasoning"]["explained"]})
    write_json(manifest_path, manifest)

    s = view["stats"]
    print(f"exported {len(view['cases']):,} cases ({s['explained']:,} explained: "
          f"aligned {s['verdicts']['aligned']}, partial {s['verdicts']['partial']}, "
          f"misaligned {s['verdicts']['misaligned']}; probably true "
          f"{s['probably_true']}) -> {relative(out)} "
          f"({out.stat().st_size / 1e6:.1f} MB)")
    return view


def frontend(view, serve, single):
    npm = shutil.which("npm")
    if npm is None:
        raise SystemExit("npm not found on PATH -- install Node.js first.")
    if not (paths.APP / "node_modules").exists():
        print("installing the app's dependencies (first run only) ...")
        subprocess.run([npm, "install"], cwd=paths.APP, check=True)
    if serve:
        subprocess.run([npm, "run", "dev"], cwd=paths.APP, check=True)
        return
    env = dict(os.environ)
    if single:
        env["SINGLE"] = "1"
    subprocess.run([npm, "run", "build"], cwd=paths.APP, check=True, env=env)
    if not single:
        print(f"static build -> {relative(paths.APP / 'dist')} "
              f"(serve it with any static server)")
        return
    # Inline the view model so the one file opens from disk, no server.
    index = paths.APP / "dist" / "index.html"
    html = index.read_text(encoding="utf-8")
    payload = json.dumps(view, ensure_ascii=False,
                         separators=(",", ":")).replace("</", "<\\/")
    html = html.replace("<head>", "<head><script>window.__DASHBOARD_DATA__ = "
                        f"{payload};</script>", 1)
    single_file = paths.APP / "dist" / f"dashboard_{view['run']}.html"
    single_file.write_text(html, encoding="utf-8")
    print(f"self-contained dashboard -> {relative(single_file)} "
          f"({single_file.stat().st_size / 1e6:.1f} MB)")


def main(argv=None):
    args = parse_args(argv)
    view = export(resolve_record(args.run))
    if args.serve or args.build:
        frontend(view, serve=args.serve, single=args.single)
    return 0


if __name__ == "__main__":
    sys.exit(main())
