"""Make a new version of a finished run from new input (docs/VERSIONS.md).

    uv run --extra cells python scripts/refine.py runs/<run-or-latest-version> inputs.yaml [--label "after IHC"]

The input file lists bench results (by the plan's experiment ids, or new ones) and notes (facts with a source,
questions, steers). It is checked first; nothing is created when it does not pass. Then a new folder runs/<run>.vN
is made beside the original, which is never written, and these stages run there on the parent's outputs: refine
(the input read against the circuits and their plan, steps regraded, the PI's review), cards, experiment, therapy and
the report. Earlier stages are carried over unchanged. The analyses page and the site are built after, as for a run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mechanist import report, versions
from mechanist.runner import STAGES, RunConfig, Runner

AFTER = ["cards", "experiment", "therapy"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("parent", type=Path, help="the run, or its latest version, to build on")
    ap.add_argument("inputs", type=Path, help="the input file (YAML)")
    ap.add_argument("--label", default="", help="a short name for the version, shown in the site's version switch")
    ap.add_argument("--config", type=Path, help="default: the parent's own config.yaml")
    a = ap.parse_args()
    parent = a.parent.resolve()
    from mechanist.present import load_run
    pctx, _ = load_run(parent)
    plan = {e.get("id") for e in (pctx.get("experiment") or {}).get("experiments") or []}
    inputs = versions.load_inputs(a.inputs.resolve(), plan)  # raises with every problem listed; nothing made yet
    child = versions.make(parent, a.inputs.resolve(), inputs, a.label)
    print(f"version {versions.number(child)} at {child}")

    cfg = RunConfig.from_yaml(a.config or parent / "config.yaml")
    r = Runner(cfg, child)
    (child / "config.yaml").write_text((a.config or parent / "config.yaml").read_text())
    order = STAGES["staged"]
    ctx: dict = {"stages": order}
    for name in order:
        if name in pctx:
            ctx[name] = pctx[name]
            r.fps[name] = hashlib.sha256(json.dumps(pctx[name], sort_keys=True, default=str).encode()).hexdigest()
    ctx["inputs"] = {**inputs, "parent_version": versions.number(parent)}
    rf = r._s_refine(ctx)
    ctx["refine"] = rf
    circuits = {**(pctx.get("circuits") or {}), "circuits": rf["circuits"], "ranking": rf["ranking"],
                "refined_from": parent.name, "changes": rf["changes"]}
    text = json.dumps(circuits, indent=1, default=str)
    (child / "circuits.json").write_text(text)
    ctx["circuits"], r.fps["circuits"] = circuits, hashlib.sha256(text.encode()).hexdigest()
    for name in AFTER:
        ctx[name] = getattr(r, f"_s_{name}")(ctx)
    md, _ = report.build(cfg, ctx, child)
    (child / "report.md").write_text(md)
    print(f"wrote {child}: refine, circuits, " + ", ".join(AFTER) + ", report")


if __name__ == "__main__":
    main()
