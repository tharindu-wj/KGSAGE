"""KGSAGE command-line tools.

Run every one of them with `python -m` from the directory that CONTAINS
`kgsage/` — invoking a file by path puts `kgsage/` itself on sys.path instead
of its parent, and `import kgsage` then fails.

The current flow, tool by tool:
  python -m kgsage.gan.train
      Phase 2 - Adversarial Generator Training (dual-discriminator
      architecture, one snapshot per epoch).
  python -m kgsage.cli.knockout_eval
      Snapshot SELECTION by knockout J@10: lower = more anchor-specific.

Phase 3 - Corruption Generation - and its evaluation now live beside the
package, in the repository's inference/ and dashboard/ components
(run as scripts from the repository root):
  python inference/corrupt.py     a corrupted dataset + the run record
  python inference/explain.py     the LLM's reasoning, into the record
  python dashboard/export.py      the case-by-case dashboard
They replace five tools retired on 10 Oct 2026 (gen_plain_tsv,
gen_corruptions_csv, ego_from_csv, format_for_llm,
gen_neighbourhood_context); git history keeps them.
"""
import sys


def _configure_utf8_stdout():
    """Force UTF-8 stdout so report Unicode chars print correctly on Windows.

    No-op on Linux/Mac where stdout is already UTF-8, and wherever the
    runtime does not support reconfigure (test runners, redirected pipes).
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError, OSError):
        pass
