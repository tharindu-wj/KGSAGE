"""Prove the components are separate, offline. No model, no API.

    python check_boundaries.py

CONTRACTS.md says what may cross between the package and the two
application components. This turns it from a promise into a check, the way
KGMVAD's own check_boundaries.py does. Every line must end PASS.

  kgsage/      the package          -- knows nothing of the applications
  inference/   corrupt + explain    -- may use the package; writes outputs/inference/
  dashboard/   export + the app     -- reads outputs/inference/; writes outputs/dashboard/
"""
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STDLIB = set(sys.stdlib_module_names)

#: third-party roots each component may import (beyond its own modules and
#: the standard library). google (google-genai) is the LLM stage's alone.
ALLOWED = {
    "inference": {"kgsage", "numpy", "torch", "google"},
    "dashboard": set(),
}
#: modules of inference/ that must stay torch-free: the LLM stage runs in
#: an environment without torch (the base conda env on the dev machine)
TORCH_FREE = ("explain.py",)
#: the paths.py constants each component must never write through
FORBIDDEN_WRITES = {
    "inference": {"DATA", "CHECKPOINTS"},
    "dashboard": {"INFERENCE", "RUNS_MANIFEST", "ROOT"},
}
WRITE_CALLS = {"write_tsv", "write_json", "write_record", "write_text",
               "write_bytes", "mkdir", "unlink", "replace", "rename"}
#: names a package file must never import: the applications, and the
#: detector-side modules (ADKGD's bridge and Reader) -- KGSAGE is standalone
PACKAGE_FORBIDDEN = {"inference", "dashboard", "kgsage_bridge", "dataset",
                     "corrupt", "explain", "prompts", "neighbourhood", "export"}

failures = []


def check(label, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    if not ok:
        failures.append(label)
        if detail:
            print(f"        {detail}")


def python_files(folder):
    return sorted(p for p in (ROOT / folder).rglob("*.py")
                  if "__pycache__" not in p.parts and "node_modules" not in p.parts)


def parse(path):
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as broken:
        failures.append(f"{path} does not parse: {broken}")
        return ast.Module(body=[], type_ignores=[])


def imported_roots(path):
    """Top-level module names a file imports, from the AST (a name in a
    comment or a docstring never counts)."""
    roots = set()
    for node in ast.walk(parse(path)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def names_in(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def write_destinations(path):
    """(line, identifiers) for every call in the file that writes."""
    found = []
    for node in ast.walk(parse(path)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            name, target = func.attr, func.value
            if name in ("replace", "rename") and node.args:
                target = node.args[-1]          # os.replace(src, DESTINATION)
        elif isinstance(func, ast.Name):
            name = func.id
            target = node.args[0] if node.args else None
        else:
            continue
        if name == "open":
            modes = [a.value for a in node.args[1:]
                     if isinstance(a, ast.Constant) and isinstance(a.value, str)]
            modes += [k.value.value for k in node.keywords
                      if k.arg == "mode" and isinstance(k.value, ast.Constant)]
            if not any(set("wax") & set(m) for m in modes):
                continue
        elif name not in WRITE_CALLS:
            continue
        if target is not None:
            found.append((node.lineno, names_in(target)))
    return found


print("\nthe package knows nothing of the applications")
offenders = []
for path in python_files("kgsage"):
    crossed = imported_roots(path) & PACKAGE_FORBIDDEN
    if crossed:
        offenders.append(f"{path.relative_to(ROOT).as_posix()} -> {sorted(crossed)}")
check("kgsage/ imports no application module and no detector module",
      not offenders, "; ".join(offenders))

print("\neach component imports only what it may")
for component, extra in ALLOWED.items():
    own = {p.stem for p in (ROOT / component).glob("*.py")}
    offenders = []
    for path in python_files(component):
        stray = imported_roots(path) - STDLIB - own - extra
        if stray:
            offenders.append(f"{path.relative_to(ROOT).as_posix()} -> {sorted(stray)}")
    allowed = ", ".join(sorted(extra)) or "nothing"
    check(f"{component}/ imports its own modules, the standard library and {allowed}",
          not offenders, "; ".join(offenders))

offenders = []
for path in python_files("inference"):
    if path.name not in ("explain.py",) and "google" in imported_roots(path):
        offenders.append(path.name)
check("google-genai is imported by explain.py alone", not offenders, ", ".join(offenders))

for name in TORCH_FREE:
    own = {p.stem: p for p in (ROOT / "inference").glob("*.py")}
    seen, stack, heavy = set(), [name[:-3]], set()
    while stack:
        module = stack.pop()
        if module in seen or module not in own:
            continue
        seen.add(module)
        roots = imported_roots(own[module])
        heavy |= roots & {"torch", "numpy", "kgsage"}
        stack.extend(roots & set(own))
    check(f"inference/{name} and everything it imports stay torch-free "
          f"({', '.join(sorted(seen))})", not heavy, f"imports {sorted(heavy)}")

print("\nnobody writes another component's folder")
for component, forbidden in FORBIDDEN_WRITES.items():
    offenders = []
    for path in python_files(component):
        for line, identifiers in write_destinations(path):
            trespass = identifiers & forbidden
            if trespass:
                offenders.append(f"{path.relative_to(ROOT).as_posix()}:{line} "
                                 f"writes through {sorted(trespass)}")
    check(f"{component}/ never writes through {', '.join(sorted(forbidden))}",
          not offenders, "; ".join(offenders))

app = ROOT / "dashboard" / "app" / "src"
fetches = []
for path in sorted(app.rglob("*.js*")):
    for match in re.finditer(r"fetch(?:Json)?\(\s*([`'\"])(.*?)\1", path.read_text(encoding="utf-8")):
        if not match.group(2).startswith("./data/"):
            fetches.append(f"{path.name}: {match.group(2)}")
check("the app fetches nothing but ./data/ (outputs/dashboard/data/)",
      not fetches, "; ".join(fetches))

print("\nevery component stands alone")
for component in ("inference", "dashboard"):
    check(f"{component}/ has a README", (ROOT / component / "README.md").exists())
check("CONTRACTS.md exists", (ROOT / "CONTRACTS.md").exists())

print("\nthe duplicated readers have not drifted")
copies = {c: (ROOT / c / "formats.py").read_text(encoding="utf-8")
          for c in ("inference", "dashboard") if (ROOT / c / "formats.py").exists()}
if len(copies) == 2:
    same = len(set(copies.values())) == 1
    print(f"  {'PASS' if same else 'note'}  formats.py "
          f"{'identical in inference/ and dashboard/' if same else 'differs between inference/ and dashboard/ -- deliberate divergence is allowed, accidental drift is not'}")

print("\nthe record's schema version agrees everywhere")
versions = {}
for relative, pattern in (("inference/corrupt.py", r'^SCHEMA = "([^"]+)"'),
                          ("inference/explain.py", r'^SCHEMA = "([^"]+)"'),
                          ("dashboard/export.py", r'^RECORD_SCHEMA = "([^"]+)"')):
    found = re.search(pattern, (ROOT / relative).read_text(encoding="utf-8"), re.M)
    versions[relative] = found.group(1) if found else None
contract = (ROOT / "CONTRACTS.md").read_text(encoding="utf-8") if (ROOT / "CONTRACTS.md").exists() else ""
agreed = len(set(versions.values())) == 1 and None not in versions.values()
check(f"writer, enricher and reader all say {next(iter(versions.values()))}",
      agreed, str(versions))
check("CONTRACTS.md documents that version",
      agreed and next(iter(versions.values())) in contract)

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("all checks pass -- the components are separate, not just tidy.")
