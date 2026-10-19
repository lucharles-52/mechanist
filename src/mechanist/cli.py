"""mechanist run --config configs/p033.yaml [--mode single|multi] [--run-id ID] [--stop-after findings]
mechanist compare runs/A runs/B"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from mechanist.runner import RunConfig, Runner

RUN_ROOT = Path(__file__).resolve().parents[2] / "runs"


def _flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out |= _flat(v, f"{prefix}{k}.")
        else:
            out[f"{prefix}{k}"] = v
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="mechanist")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", type=Path, required=True)
    r.add_argument("--mode", choices=["multi", "single"])
    r.add_argument("--run-id")
    r.add_argument("--run-root", type=Path, default=RUN_ROOT)
    r.add_argument("--stop-after")
    r.add_argument("--base-url", help="override llm.base_url (e.g. the port of a vLLM server started for this job)")
    r.add_argument("--model", help="override llm.model")
    c = sub.add_parser("compare")
    c.add_argument("runs", nargs="+", type=Path)
    a = ap.parse_args(argv)

    if a.cmd == "run":
        cfg = RunConfig.from_yaml(a.config)
        if a.mode:
            cfg.mode = a.mode
        if a.base_url:
            cfg.llm.base_url = a.base_url
        if a.model:
            cfg.llm.model = a.model
        run_id = a.run_id or f"{cfg.study}_{cfg.disease}_{cfg.mode}_{dt.datetime.now():%Y%m%d_%H%M%S}"
        out = Runner(cfg, a.run_root / run_id).run(stop_after=a.stop_after)
        print(out)
    else:
        rows = {}
        for d in a.runs:
            p = d / "report.json"
            rows[d.name] = _flat(json.loads(p.read_text())) if p.exists() else {}
        keys = sorted({k for v in rows.values() for k in v})
        names = list(rows)
        w = max(len(k) for k in keys) if keys else 10
        print(" " * w + " | " + " | ".join(names))
        for k in keys:
            print(k.ljust(w) + " | " + " | ".join(str(rows[n].get(k, "")) for n in names))


if __name__ == "__main__":
    main()
