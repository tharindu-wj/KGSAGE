"""Ask an LLM whether each corruption in a run meets KGSAGE's goals, and
write its reasoning into the run record.

    python inference/explain.py --run outputs/inference/codex-s_r0.05_s0
    python inference/explain.py --run codex-s_r0.05_s0 --limit 20   # try 20 first
    python inference/explain.py --run codex-s_r0.05_s0 --dry-run    # print, call nothing

The criteria are KGSAGE's three goals -- type_valid, plausible,
uncorroborated -- plus a fourth, separate check, false: whether the
corrupted fact is false in the world. The verdict (aligned / partial /
misaligned) is derived here from the three goals, never asked of the
model; "false" stays outside it, as triage for a Wikidata check.

TORCH-FREE. It reads only the run record corrupt.py wrote, so it runs in
any environment with google-genai (the base conda env on this machine).
It needs GOOGLE_API_KEY in the environment, or in a .env beside the
repository or one level up; .env.example is the template.

RESUMABLE BY CONSTRUCTION. Cases are judged in batches, one call each,
and the record is rewritten atomically after every batch, so a quota stop,
a crash or Ctrl-C loses at most the batch in flight. Run the same command
again and it carries on with the cases that have no reasoning yet. Calls
are paced (--pause, 4 s) for the free tier's 15 requests a minute, and the
client retries a refused call with backoff before the run gives up.
"""
import argparse
import datetime
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

from formats import load_json, write_json, write_record
from paths import ENV_FILES, INFERENCE, RECORD, RUNS_MANIFEST
from prompts import (CRITERIA, GOALS, INSTRUCTION, PROMPT_VERSION,
                     RESPONSE_SCHEMA, batch_prompt, verdict_of)

SCHEMA = "kgsage-inference/1"

#: the model KGMVAD's detector uses; any Gemini model name works
DEFAULT_MODEL = "gemini-3.5-flash-lite"

KEY_NAMES = ("GOOGLE_API_KEY", "GEMINI_API_KEY")
CONFIDENCE = ("high", "medium", "low")
VERDICTS = ("aligned", "partial", "misaligned")

#: give up after this many failed calls in a row
MAX_CONSECUTIVE_FAILURES = 3
#: substrings that mean the API refused for quota, not for content
QUOTA_MARKERS = ("resource_exhausted", "429", "quota", "rate limit")
#: the genai client logs one INFO line per backoff before it sleeps; a
#: retry that succeeds is otherwise invisible
GENAI_LOGGER = "google_genai._api_client"


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Explain each corruption of a run with an LLM.")
    parser.add_argument("--run", default=None,
                        help="run folder, run name or corruptions.json "
                             "(default: the newest run)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Gemini model (default {DEFAULT_MODEL})")
    parser.add_argument("--batch", type=int, default=10,
                        help="cases per call (default 10)")
    parser.add_argument("--limit", type=int, default=None,
                        help="judge at most this many cases this session")
    parser.add_argument("--ids", default=None,
                        help="comma-separated case ids to judge (implies "
                             "--redo for them)")
    parser.add_argument("--redo", action="store_true",
                        help="judge cases that already have reasoning again")
    parser.add_argument("--pause", type=float, default=4.0,
                        help="seconds between calls (default 4)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=300,
                        help="seconds before one call is abandoned")
    parser.add_argument("--env-file", default=None,
                        help="read GOOGLE_API_KEY from this file first (for "
                             "a key kept in another repository's .env)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the instruction and the first batch, "
                             "call nothing (no key needed)")
    args = parser.parse_args(argv)
    if args.batch < 1:
        parser.error("--batch must be at least 1")
    return args


def resolve_record(name):
    """--run as a folder, a run name or the record itself; default newest."""
    if name is None:
        runs = load_json(RUNS_MANIFEST) if RUNS_MANIFEST.exists() else []
        if not runs:
            raise SystemExit("no runs yet -- make one with inference/corrupt.py")
        folder = INFERENCE / runs[0]["folder"]
    else:
        path = Path(name)
        if path.is_file():
            return path
        folder = path if path.exists() else INFERENCE / name
    record = folder / RECORD
    if not record.exists():
        held = sorted(p.parent.name for p in INFERENCE.glob(f"*/{RECORD}"))
        raise SystemExit(f"no {RECORD} in {folder}. Runs on disk: "
                         f"{', '.join(held) or 'none'}")
    return record


def _key_in(env_file):
    if not env_file.exists():
        return None
    for line in env_file.read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        if key.strip() in KEY_NAMES and value:
            return value
    return None


def find_key(env_file=None):
    """(key, where it came from): an --env-file first, then the environment,
    then a .env beside the repository or one level up. Only the source is
    ever printed, never the key."""
    if env_file:
        path = Path(env_file)
        if not path.exists():
            raise SystemExit(f"--env-file {env_file}: no such file")
        key = _key_in(path)
        if key:
            return key, path.as_posix()
    for name in KEY_NAMES:
        if os.environ.get(name):
            return os.environ[name], f"${name}"
    for path in ENV_FILES:
        key = _key_in(path)
        if key:
            return key, path.as_posix()
    return None, None


def parse_batch(text, wanted):
    """The model's JSON -> {case id: {criteria, summary}} for every case it
    answered well, plus a list of what was wrong with the rest. Nothing
    malformed is ever written: a case it fumbled stays unexplained and is
    asked again on the next run."""
    try:
        items = json.loads(text)
    except (TypeError, ValueError) as err:
        return {}, [f"unparseable response ({err})"]
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list):
        return {}, ["response is not a JSON array"]
    accepted, problems = {}, []
    for item in items:
        if not isinstance(item, dict):
            problems.append("an answer that is not an object")
            continue
        case_id = str(item.get("id", "")).strip()
        if case_id not in wanted:
            problems.append(f"unrequested id {case_id!r}")
            continue
        if case_id in accepted:
            problems.append(f"{case_id} answered twice")
            continue
        criteria, fault = {}, None
        for name in CRITERIA:
            block = item.get(name)
            why = " ".join(str((block or {}).get("why", "")).split())
            if (not isinstance(block, dict) or not isinstance(block.get("holds"), bool)
                    or not why):
                fault = f"{case_id}: {name} missing or malformed"
                break
            entry = {"holds": block["holds"], "why": why}
            if name == "false":
                confidence = str(block.get("confidence", "")).strip().lower()
                if confidence not in CONFIDENCE:
                    fault = f"{case_id}: false.confidence {confidence!r}"
                    break
                entry["confidence"] = confidence
            criteria[name] = entry
        summary = " ".join(str(item.get("summary", "")).split())
        if fault is None and not summary:
            fault = f"{case_id}: no summary"
        if fault:
            problems.append(fault)
            continue
        accepted[case_id] = {"criteria": criteria, "summary": summary}
    unanswered = [case_id for case_id in wanted if case_id not in accepted]
    if unanswered:
        problems.append(f"{len(unanswered)} not answered "
                        f"({', '.join(unanswered[:5])})")
    return accepted, problems


def classify(err):
    """Why a call failed, which decides whether the run goes on."""
    code = getattr(err, "code", None)
    text = str(err).lower()
    if code == 429 or any(marker in text for marker in QUOTA_MARKERS):
        return "quota"
    if code in (401, 403) or "api key" in text or "permission" in text:
        return "auth"
    if code in (400, 404):
        return "request"
    return "transient"


def summarise(record):
    """The header's reasoning block, rebuilt from the cases every save."""
    previous = record.get("reasoning") or {}
    explained = [case["reasoning"] for case in record["cases"]
                 if case.get("reasoning")]
    return {
        "criteria": list(CRITERIA),
        "goals": list(GOALS),
        "explained": len(explained),
        "total": len(record["cases"]),
        "models": sorted({block["model"] for block in explained}),
        "prompt_versions": sorted({block["prompt_version"] for block in explained}),
        "verdicts": {verdict: sum(1 for block in explained
                                  if block["verdict"] == verdict)
                     for verdict in VERDICTS},
        "fails": {goal: sum(1 for block in explained if goal in block["fails"])
                  for goal in GOALS},
        "probably_true": sum(1 for block in explained
                             if not block["criteria"]["false"]["holds"]),
        "sessions": previous.get("sessions", []),
    }


def save(record_path, record):
    record["reasoning"] = summarise(record)
    write_record(record_path, record)
    # the run list carries the explained count, so the dashboard's run
    # picker can say which runs have reasoning
    if record_path.parent.parent.resolve() == INFERENCE.resolve() \
            and RUNS_MANIFEST.exists():
        runs = load_json(RUNS_MANIFEST)
        for entry in runs:
            if entry.get("folder") == record_path.parent.name:
                entry["explained"] = record["reasoning"]["explained"]
        write_json(RUNS_MANIFEST, runs)


def make_client(key, timeout, temperature):
    """The Gemini client and the call config. Imported here, not at the top,
    so --help and --dry-run work without google-genai installed."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=key, http_options=types.HttpOptions(
        timeout=timeout * 1000,
        retry_options=types.HttpRetryOptions(
            attempts=5, initial_delay=10, max_delay=70, exp_base=2, jitter=1)))
    config = types.GenerateContentConfig(
        system_instruction=INSTRUCTION, temperature=temperature,
        response_mime_type="application/json", response_schema=RESPONSE_SCHEMA)
    return client, config


class RetryWatcher(logging.Handler):
    """Counts the client's backoff retries, which never reach an except."""

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.lines = []

    def emit(self, record):
        message = record.getMessage()
        if "retry" in message.lower():
            self.lines.append(" ".join(message.split())[:200])


def main(argv=None):
    args = parse_args(argv)
    record_path = resolve_record(args.run)
    record = load_json(record_path)
    if record.get("schema") != SCHEMA:
        raise SystemExit(f"{record_path} is not a {SCHEMA} record "
                         f"(schema {record.get('schema')!r})")
    cases = record["cases"]
    by_id = {case["id"]: case for case in cases}

    if args.ids:
        wanted = [part.strip() for part in args.ids.split(",") if part.strip()]
        unknown = [case_id for case_id in wanted if case_id not in by_id]
        if unknown:
            raise SystemExit(f"no such case: {', '.join(unknown)}")
        targets = [by_id[case_id] for case_id in wanted]
    else:
        targets = [case for case in cases
                   if args.redo or case.get("reasoning") is None]
    if args.limit is not None:
        targets = targets[:args.limit]
    done = sum(1 for case in cases if case.get("reasoning"))
    print(f"run       {record['run']}: {len(cases):,} cases, {done:,} explained")
    if not targets:
        print("nothing to judge: every case has reasoning "
              "(--redo judges them again)")
        return 0
    batches = [targets[i:i + args.batch]
               for i in range(0, len(targets), args.batch)]
    print(f"judging   {len(targets):,} case{'' if len(targets) == 1 else 's'} in "
          f"{len(batches)} call{'' if len(batches) == 1 else 's'} of up to "
          f"{args.batch}, model {args.model}, prompt {PROMPT_VERSION}")

    if args.dry_run:
        print("\n=== system instruction " + "=" * 50)
        print(INSTRUCTION)
        print("=== first call " + "=" * 58)
        print(batch_prompt(batches[0], record))
        return 0

    key, source = find_key(args.env_file)
    if not key:
        raise SystemExit(
            "No Gemini key. Put GOOGLE_API_KEY=... in a .env beside the "
            "repository (see .env.example), in the environment, or name a "
            "file holding it with --env-file.")
    print(f"key       from {source}")
    print(f"estimate  about {math.ceil(len(batches) * (args.pause + 6) / 60)} "
          f"min at {args.pause:g} s pacing")

    watcher = RetryWatcher()
    genai_log = logging.getLogger(GENAI_LOGGER)
    genai_log.setLevel(logging.INFO)
    genai_log.addHandler(watcher)
    client, config = make_client(key, args.timeout, args.temperature)

    session = {"started": now(), "finished": None, "status": "running",
               "model": args.model, "prompt_version": PROMPT_VERSION,
               "batch": args.batch, "temperature": args.temperature,
               "targeted": len(targets), "calls": 0, "explained": 0,
               "retries": 0, "tokens": {"prompt": 0, "output": 0, "thoughts": 0},
               "errors": []}
    record["reasoning"] = summarise(record)
    # A session still "running" was killed without a chance to say so (one
    # explain process per record is the rule); name it for what it was.
    for earlier in record["reasoning"]["sessions"]:
        if earlier.get("status") == "running":
            earlier["status"] = "stopped"
    record["reasoning"]["sessions"].append(session)

    started = time.time()
    consecutive = 0
    status = "completed"
    try:
        for number, batch in enumerate(batches, 1):
            if session["calls"] and args.pause > 0:
                time.sleep(args.pause)
            ids = [case["id"] for case in batch]
            span = ids[0] if len(ids) == 1 else f"{ids[0]}..{ids[-1]}"
            session["calls"] += 1
            try:
                response = client.models.generate_content(
                    model=args.model, contents=batch_prompt(batch, record),
                    config=config)
                text = response.text
            except Exception as err:  # noqa: BLE001 -- classified below
                kind = classify(err)
                session["errors"].append({"at": now(), "cases": span,
                                          "kind": kind,
                                          "message": " ".join(str(err).split())[:300]})
                consecutive += 1
                print(f"  call {number}/{len(batches)} {span}: {kind} error: "
                      f"{' '.join(str(err).split())[:160]}")
                if kind in ("quota", "auth", "request"):
                    status = "truncated" if kind == "quota" else "failed"
                    break
                if consecutive >= MAX_CONSECUTIVE_FAILURES:
                    status = "failed"
                    break
                continue
            usage = getattr(response, "usage_metadata", None)
            for field, name in (("prompt", "prompt_token_count"),
                                ("output", "candidates_token_count"),
                                ("thoughts", "thoughts_token_count")):
                session["tokens"][field] += getattr(usage, name, None) or 0

            accepted, problems = parse_batch(text, ids)
            stamp = now()
            for case_id, block in accepted.items():
                verdict, fails = verdict_of(block["criteria"])
                by_id[case_id]["reasoning"] = {
                    "model": args.model, "prompt_version": PROMPT_VERSION,
                    "at": stamp, "verdict": verdict, "fails": fails,
                    "criteria": block["criteria"], "summary": block["summary"]}
            session["explained"] += len(accepted)
            if problems:
                session["errors"].append({"at": stamp, "cases": span,
                                          "kind": "response",
                                          "message": "; ".join(problems)[:300]})
            consecutive = 0 if accepted else consecutive + 1
            session["retries"] = len(watcher.lines)
            save(record_path, record)
            tally = record["reasoning"]["verdicts"]
            print(f"  call {number}/{len(batches)} {span}: {len(accepted)}/"
                  f"{len(batch)} judged   [{record['reasoning']['explained']:,}"
                  f"/{len(cases):,} in the record; aligned {tally['aligned']}, "
                  f"partial {tally['partial']}, misaligned "
                  f"{tally['misaligned']}]  {time.time() - started:.0f} s"
                  + (f"   ({'; '.join(problems)[:120]})" if problems else ""))
            if not accepted and consecutive >= MAX_CONSECUTIVE_FAILURES:
                status = "failed"
                break
    except KeyboardInterrupt:
        status = "interrupted"
        print("\ninterrupted -- saving what was judged")

    if status == "completed" and session["explained"] < len(targets):
        status = "partial"
    session.update(status=status, finished=now(), retries=len(watcher.lines))
    save(record_path, record)

    header = record["reasoning"]
    print(f"\n{status}: {session['explained']:,} of {len(targets):,} cases "
          f"judged in {session['calls']} call(s), {time.time() - started:.0f} s; "
          f"{len(session['errors'])} problem(s), {session['retries']} retries")
    print(f"record    {header['explained']:,} of {header['total']:,} explained: "
          f"aligned {header['verdicts']['aligned']}, partial "
          f"{header['verdicts']['partial']}, misaligned "
          f"{header['verdicts']['misaligned']}; probably true "
          f"{header['probably_true']}")
    tokens = session["tokens"]
    print(f"tokens    prompt {tokens['prompt']:,}, output {tokens['output']:,}, "
          f"thoughts {tokens['thoughts']:,}")
    if header["explained"] < header["total"]:
        print(f"resume    python inference/explain.py --run "
              f"{record_path.parent.as_posix()}")
    return 0 if status in ("completed", "partial") else 1


if __name__ == "__main__":
    sys.exit(main())
