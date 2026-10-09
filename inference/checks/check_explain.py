"""Offline tests for inference/explain.py: a fake model, no network, no key.

    python inference/checks/check_explain.py                 samples the newest run
    python inference/checks/check_explain.py <corruptions.json>

Checks the parser on hand-made answers (malformed ones are never written),
the derived verdict, error classification, and whole sessions against a
scripted fake model: a fumbled batch and a quota stop save what was judged,
a rerun resumes exactly the unanswered cases, --ids re-judges, Ctrl-C is
saved as interrupted. Works on a 23-case COPY in a temporary folder; the
real record is only read. Finally renders every case of the record into a
prompt. Every line must read PASS.
"""
import copy
import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INFERENCE = ROOT / "outputs" / "inference"
if len(sys.argv) > 1:
    SOURCE = Path(sys.argv[1])
else:
    runs = json.loads((INFERENCE / "manifest.json").read_text(encoding="utf-8"))
    SOURCE = INFERENCE / runs[0]["folder"] / "corruptions.json"
WORK = Path(tempfile.mkdtemp(prefix="kgsage_explain_check_"))
sys.path.insert(0, str(ROOT / "inference"))
import explain  # noqa: E402
from prompts import batch_prompt, case_block, support_sentence, verdict_of  # noqa: E402

failures = []


def check(label, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + ("" if ok else f"   [{detail}]"))
    if not ok:
        failures.append(label)


# ---- parse_batch on hand-made responses ----------------------------------
print("parse_batch")


def answer(case_id, holds=(True, True, True, True), confidence="high"):
    return {"id": case_id,
            **{name: {"why": f"reason for {name}", "holds": h}
               for name, h in zip(("type_valid", "plausible", "uncorroborated", "false"), holds)},
            "summary": "a summary"} | {"false": {"why": "world check", "holds": holds[3],
                                                 "confidence": confidence}}


good, problems = explain.parse_batch(json.dumps([answer("c1"), answer("c2")]), ["c1", "c2"])
check("two good answers accepted", set(good) == {"c1", "c2"} and not problems, problems)
good, problems = explain.parse_batch(json.dumps(answer("c1")), ["c1"])
check("a bare object for a batch of one", set(good) == {"c1"} and not problems, problems)
good, problems = explain.parse_batch("not json", ["c1"])
check("unparseable text -> nothing written", not good and problems, problems)
bad = answer("c1")
bad["plausible"]["holds"] = "yes"
good, problems = explain.parse_batch(json.dumps([bad, answer("c2")]), ["c1", "c2"])
check("a string 'holds' is refused, the good case kept", set(good) == {"c2"} and any("c1" in p for p in problems), problems)
bad = answer("c1")
bad["false"]["confidence"] = "certain"
good, problems = explain.parse_batch(json.dumps([bad]), ["c1"])
check("an off-list confidence is refused", not good, problems)
bad = answer("c1")
bad["summary"] = "  "
good, problems = explain.parse_batch(json.dumps([bad]), ["c1"])
check("an empty summary is refused", not good, problems)
good, problems = explain.parse_batch(json.dumps([answer("c1"), answer("c1"), answer("c9")]), ["c1", "c2"])
check("duplicate, unrequested and missing ids reported",
      set(good) == {"c1"} and len(problems) == 3, problems)
good, _ = explain.parse_batch(json.dumps([{**answer("c1"), "type_valid": {"why": "x\n  y", "holds": True}}]), ["c1"])
check("whitespace in a why is normalised", good["c1"]["criteria"]["type_valid"]["why"] == "x y")

print("\nverdict_of")
crit = lambda *h: {name: {"holds": v} for name, v in zip(("type_valid", "plausible", "uncorroborated", "false"), h)}
check("all three goals -> aligned", verdict_of(crit(True, True, True, False)) == ("aligned", []))
check("one fails -> partial", verdict_of(crit(True, False, True, True)) == ("partial", ["plausible"]))
check("two fail -> misaligned", verdict_of(crit(False, True, False, True))[0] == "misaligned")
check("false never enters the verdict", verdict_of(crit(True, True, True, False))[0] == "aligned")

print("\nclassify")


class Err(Exception):
    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code


check("429 -> quota", explain.classify(Err("x", 429)) == "quota")
check("RESOURCE_EXHAUSTED text -> quota", explain.classify(Err("RESOURCE_EXHAUSTED: try later")) == "quota")
check("403 -> auth", explain.classify(Err("denied", 403)) == "auth")
check("404 -> request (bad model name)", explain.classify(Err("model not found", 404)) == "request")
check("503 -> transient", explain.classify(Err("unavailable", 503)) == "transient")

# ---- a whole session against a fake model ---------------------------------
print("\nsessions against a fake model")
record = json.loads(SOURCE.read_text(encoding="utf-8"))
small = copy.deepcopy(record)
small["cases"] = small["cases"][:23]
for case in small["cases"]:
    case["reasoning"] = None
small["reasoning"] = None
WORK.mkdir(exist_ok=True)
path = WORK / "corruptions.json"
path.write_text(json.dumps(small), encoding="utf-8")
untouched = {k: v for k, v in small.items() if k not in ("reasoning", "cases")}


class Usage:
    prompt_token_count, candidates_token_count, thoughts_token_count = 1000, 200, 0


class Response:
    def __init__(self, text):
        self.text, self.usage_metadata = text, Usage()


class FakeModels:
    def __init__(self, script):
        self.script, self.calls = list(script), []

    def generate_content(self, model, contents, config):
        ids = re.findall(r"^CASE (c\d+):", contents, flags=re.M)
        self.calls.append(ids)
        action = self.script.pop(0) if self.script else "good"
        if action == "quota":
            raise Err("429 RESOURCE_EXHAUSTED quota", 429)
        if action == "interrupt":
            raise KeyboardInterrupt
        answers = []
        for n, case_id in enumerate(ids):
            if action == "fumble" and n == 0:
                continue                                   # one missing
            item = answer(case_id, holds=(True, n % 3 != 1, n % 4 != 2, n % 5 != 3),
                          confidence=("high", "medium", "low")[n % 3])
            if action == "fumble" and n == 1:
                item["uncorroborated"]["holds"] = "maybe"   # one malformed
            answers.append(item)
        return Response(json.dumps(answers))


class FakeClient:
    def __init__(self, script):
        self.models = FakeModels(script)


def session(script, *extra):
    client = FakeClient(script)
    explain.make_client = lambda key, timeout, temperature: (client, None)
    code = explain.main(["--run", str(path), "--pause", "0", *extra])
    return code, client.models.calls, json.loads(path.read_text(encoding="utf-8"))


import os  # noqa: E402
os.environ["GOOGLE_API_KEY"] = "test-key-not-real"

code, calls, after = session(["good", "fumble", "quota"])
header = after["reasoning"]
check("quota stop -> exit 1, status truncated", code == 1 and header["sessions"][-1]["status"] == "truncated",
      (code, header["sessions"][-1]["status"]))
check("3 calls of 10, 10, 3", [len(c) for c in calls] == [10, 10, 3], [len(c) for c in calls])
check("18 judged and saved (10 + 8)", header["explained"] == 18, header["explained"])
check("the two fumbled cases stay unexplained",
      [c["id"] for c in after["cases"] if c["reasoning"] is None][:2] == [calls[1][0], calls[1][1]])
errs = header["sessions"][-1]["errors"]
check("both problems recorded (response + quota)", [e["kind"] for e in errs] == ["response", "quota"], errs)
check("the rest of the record is untouched", {k: after[k] for k in untouched} == untouched)
check("non-reasoning case fields untouched",
      all({k: v for k, v in a.items() if k != "reasoning"} == {k: v for k, v in b.items() if k != "reasoning"}
          for a, b in zip(after["cases"], small["cases"])))
block = next(c["reasoning"] for c in after["cases"] if c["reasoning"])
check("a case block carries model, prompt version, verdict, fails",
      {"model", "prompt_version", "at", "verdict", "fails", "criteria", "summary"} <= set(block), block.keys())

first_calls = calls
code, calls, after = session([])
header = after["reasoning"]
expected = first_calls[1][:2] + first_calls[2]
check("resume judges only the 5 left, in one call", calls == [expected], (calls, expected))
check("resume completes: 23 of 23, exit 0", code == 0 and header["explained"] == 23, (code, header["explained"]))
check("two sessions recorded", [s["status"] for s in header["sessions"]] == ["truncated", "completed"])
counted = {v: sum(1 for c in after["cases"] if c["reasoning"]["verdict"] == v) for v in ("aligned", "partial", "misaligned")}
check("header verdict tally matches the cases", header["verdicts"] == counted, (header["verdicts"], counted))
check("probably_true counts false.holds == false",
      header["probably_true"] == sum(1 for c in after["cases"] if not c["reasoning"]["criteria"]["false"]["holds"]))

code, calls, after = session([])
check("nothing left -> no calls, exit 0", code == 0 and calls == [])

first_two = [c["id"] for c in after["cases"][:2]]
code, calls, after = session([], "--ids", ",".join(first_two))
check("--ids re-judges exactly those", calls == [first_two] and code == 0, calls)

code, calls, after = session(["interrupt"], "--redo", "--limit", "4")
check("Ctrl-C -> saved as interrupted", after["reasoning"]["sessions"][-1]["status"] == "interrupted"
      and after["reasoning"]["explained"] == 23)

# ---- prompt rendering on every case of the source record -----------------
print("\nprompt rendering over the whole source record")
broken = []
for case in record["cases"]:
    try:
        text = case_block(case, record)
        assert case["text"]["corrupted"] in text and case["id"] in text
        sentence = support_sentence(case, record)
        assert (sentence is None) == (case["structure"]["support"] is None)
    except Exception as err:  # noqa: BLE001
        broken.append((case["id"], repr(err)))
check(f"all {len(record['cases'])} cases render", not broken, broken[:3])
prompt = batch_prompt(record["cases"][:10], record)
check("a batch of 10 names all ten ids", all(f"CASE {c['id']}:" in prompt for c in record["cases"][:10]))
print(f"  (a 10-case prompt is {len(prompt):,} characters)")

print(f"\n{'ALL PASS' if not failures else str(len(failures)) + ' FAILED'}")
sys.exit(1 if failures else 0)
