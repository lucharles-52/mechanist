"""Run ONE stage on a finished run's outputs, in a new run folder: try a changed stage without rerunning the rest.

    uv run --extra cells python scripts/run_stage.py --from runs/<done> --stage circuits --out runs/<new> \
        [--config configs/p033_pg_claude.yaml]

Every stage output of the source run is loaded as it is (it is not rerun or re-checked). Only the chosen stage calls
the model; its calls, transcript and output go to the new folder. The source folder is never written.
For a stage that writes markdown in the report, the new folder also gets <stage>.md. The new folder also gets a full
report.md: stages after the chosen one are carried over from the source run as they are (not redone on the new
output), and the report says which.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mechanist import report
from mechanist.runner import STAGES, RunConfig, Runner


def sources(src: Path) -> list[Path]:
    """The run folder, then the folders it was itself rerun from (a partial rerun records its source), in order."""
    out, cur = [], src
    while cur is not None and cur not in out:
        out.append(cur)
        nxt = None
        ev = cur / "events.jsonl"
        if ev.exists():
            for line in ev.read_text().splitlines():
                d = json.loads(line)
                if d.get("action") == "inputs" and d.get("source"):
                    p = Path(d["source"])
                    nxt = p if p.is_absolute() else cur.parent.parent / p
        cur = nxt
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from", dest="src", type=Path, required=True, help="a finished run folder")
    ap.add_argument("--stage", required=True,
                    help="the stage to run, e.g. circuits; several in order, comma-separated: experiment,therapy")
    ap.add_argument("--out", type=Path, required=True, help="the new run folder")
    ap.add_argument("--report-only", action="store_true",
                    help="do not run the stages: take their outputs already in --out and only write the report")
    ap.add_argument("--config", type=Path, default=Path("configs/p033_pg_claude.yaml"))
    a = ap.parse_args()
    if a.out.resolve() == a.src.resolve():
        raise SystemExit("--out must be a new folder, not the source run")
    cfg = RunConfig.from_yaml(a.config)
    r = Runner(cfg, a.out)
    order = STAGES["staged" if cfg.mode == "multi" and cfg.workflow == "staged" else cfg.mode]
    todo = [x.strip() for x in a.stage.split(",") if x.strip()]
    bad = [x for x in todo if x not in order]
    if bad:
        raise SystemExit(f"unknown stage {bad}; stages: {', '.join(order)}")
    todo.sort(key=order.index)
    chain = sources(a.src)

    def find(name: str) -> Path | None:
        return next((d / f"{name}.json" for d in chain if (d / f"{name}.json").exists()), None)
    ctx: dict = {"stages": order}
    for name in order[: order.index(todo[0])]:
        p = find(name)
        if p is not None:
            text = p.read_text()
            ctx[name] = json.loads(text)
            r.fps[name] = hashlib.sha256(text.encode()).hexdigest()  # stand-in fingerprint: taken as done
    if not a.report_only:
        r.event(stage=todo[0], action="inputs", source=str(a.src), loaded=[k for k in ctx if k != "stages"])
    for name in todo:
        if a.report_only:
            ctx[name] = json.loads((a.out / f"{name}.json").read_text())
        else:
            ctx[name] = getattr(r, f"_s_{name}")(ctx)
        if name == "circuits":
            (a.out / "circuits.md").write_text("\n".join(report.circuits_section(ctx)))
    later = [n for n in order[order.index(todo[-1]) + 1:] if n != "report" and find(n) is not None]
    for name in later:
        ctx[name] = json.loads(find(name).read_text())
    md, _ = report.build(cfg, ctx, a.out)
    head, _, rest = md.partition("\n\n")
    ran = ", ".join(f"`{n}`" for n in todo)
    note = (f"> **Partial rerun.** Only {ran} {'was' if len(todo) == 1 else 'were'} run here, on the outputs of "
            f"`{a.src.name}`. "
            + (f"Carried over unchanged from that run, so not built on the new stages: "
               f"{', '.join(f'`{n}`' for n in later)}. " if later else "")
            + "The model usage below counts this rerun's calls only.")
    (a.out / "report.md").write_text(f"{head}\n\n{note}\n\n{rest}")
    print("wrote " + ", ".join(str(a.out / f"{n}.json") for n in todo))

if __name__ == "__main__":
    main()
