"""Live status of Mechanist runs: progress (stage, PI gates, questions, spend) and performance (the model so far).

    python3 scripts/status.py [glob ...]          # default: the current Claude (B_api*) and local-model (L_*) runs
    watch -n 30 python3 scripts/status.py         # refreshing board

Reads only the run folders (events.jsonl, llm_calls.jsonl, stage outputs) and squeue; changes nothing.
"""

from __future__ import annotations

import datetime as dt
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1] / "runs"
DEFAULT = ["p033_PG_multi_B*", "p033_PG_multi_L_*"]
STAGE_NAMES = {"dossier": "1 dossier", "check": "2 pathology check", "candidates": "3 candidates",
               "investigate": "4 chase: investigate", "synthesize": "4 chase: synthesis", "critique": "4 chase: PI review",
               "chase": "4 chase: PI rounds", "experiment": "5 experiments", "revise": "6 evidence ledger",
               "circuits": "7 circuits", "linkage": "6a disease linkage",
               "report": "report"}
ORDER = ["load", "findings", "dossier", "check", "panel", "generate", "candidates", "verify_r0", "scout", "investigate",
         "synthesize", "critique", "chase", "linkage", "revise", "circuits", "experiment", "report"]


def read_jsonl(p: Path) -> list[dict]:
    out = []
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def ago(t: str) -> str:
    try:
        s = (dt.datetime.now() - dt.datetime.fromisoformat(t)).total_seconds()
    except (TypeError, ValueError):
        return "?"
    return f"{int(s // 60)}m" if s < 3600 else f"{s / 3600:.1f}h"


def model_summary(d: Path) -> list[str]:
    """Performance: the latest model (circuits > revise > chase interim > first synthesis)."""
    circ = []
    f = d / "circuits.json"
    if f.exists():
        try:
            cs = json.loads(f.read_text()).get("circuits") or []
        except json.JSONDecodeError:
            cs = []
        for c in cs[:3]:
            kinds = {}
            for e in c.get("edges") or []:
                kinds[e.get("status")] = kinds.get(e.get("status"), 0) + 1
            circ.append(f"    circuit {c.get('rank')}: {c.get('title', '')[:100]} | edges observed/likely/analogy/"
                        f"speculative {kinds.get('observed', 0)}/{kinds.get('likely', 0)}/{kinds.get('analogy', 0)}/"
                        f"{kinds.get('speculative', 0)}")
    for name in ("revise", "synthesize"):
        f = d / f"{name}.json"
        if not f.exists():
            continue
        try:
            o = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        syn = o.get("synthesis")
        if not syn:
            return [f"    model ({name}): none parsed"]
        links = syn.get("links") or []
        depths = [lk.get("depth") for lk in links if lk.get("depth")]
        kinds: dict[str, int] = {}
        for lk in links:
            for st in lk.get("steps") or []:
                kinds[st.get("kind", "?")] = kinds.get(st.get("kind", "?"), 0) + 1
        objs = o.get("objections") or []
        bad = sum(x.get("kind") == "invalid_citation" for x in objs)
        relab = sum("relabelled speculative" in x.get("text", "") for x in objs)
        lines = [f"    model ({'final' if name == 'revise' else 'first synthesis'}): {syn.get('title', '')[:110]}",
                 f"      links {len(links)} | mean depth {sum(depths) / len(depths):.1f}/7" if depths else
                 f"      links {len(links)}",
                 (f"      steps tested/literature/speculative {kinds.get('tested', 0)}/{kinds.get('literature', 0)}/"
                  f"{kinds.get('speculative', 0)} | alternatives {len(syn.get('alternatives') or [])} | "
                  f"experiments {len(syn.get('experiments_planned') or [])} | confidence: "
                  f"{(syn.get('confidence') or '-')[:50]}"),
                 f"      verifier: {bad} invalid citations, {relab} steps relabelled speculative"]
        return circ + lines
    return []


def run_status(d: Path) -> list[str]:
    ev = read_jsonl(d / "events.jsonl")
    calls = read_jsonl(d / "llm_calls.jsonl")
    if not ev:
        return [f"{d.name}: not started"]
    started = ev[0].get("time")
    ends = [e["stage"] for e in ev if e.get("action") == "end"]
    starts = [e for e in ev if e.get("action") == "start"]
    current = starts[-1]["stage"] if starts and starts[-1]["stage"] not in ends[-1:] else (ends[-1] if ends else "?")
    done = "report" in ends
    last = ev[-1]
    def billed(c: dict) -> bool:  # subscription calls carry the CLI's estimate but are not billed per call
        u = c.get("usage") or {}
        return u.get("billed") if u.get("billed") is not None else not str(c.get("model", "")).startswith("claude_cli")
    cost = sum((c.get("usage") or {}).get("total_cost_usd") or 0 for c in calls if billed(c))
    est = sum((c.get("usage") or {}).get("total_cost_usd") or 0 for c in calls if not billed(c))
    accts: dict[str, int] = {}
    for c in calls:
        a = (c.get("usage") or {}).get("account")
        if a:
            accts[a] = accts.get(a, 0) + 1
    models = sorted({(c.get("usage") or {}).get("model") or c.get("model", "") for c in calls} - {""})
    q = max([e.get("questions", 0) for e in ev if e.get("action") == "threads"] or [0])
    res = max([e.get("results", 0) for e in ev if e.get("action") == "threads"] or [0])
    try:
        quiet = (dt.datetime.now() - dt.datetime.fromisoformat(last.get("time"))).total_seconds() > 600
    except (TypeError, ValueError):
        quiet = False
    state = "DONE" if done else ("stopped at: " if quiet else "running: ") + STAGE_NAMES.get(current, current)
    head = (f"{d.name}  [{state}]  "
            f"started {ago(started)} ago, last event {ago(last.get('time'))} ago")
    fb = [c for c in calls if (c.get("usage") or {}).get("served_by")]
    lines = [head, f"    calls {len(calls)} | questions {q} | tool results {res}"
             + (f" | spend ${cost:.2f}" if cost else "")
             + (f" | subscription calls not billed (CLI estimate ${est:.2f})" if est else "")
             + (f" | models {', '.join(models)}" if models else "")
             + (f" | answered by fallback after a refusal: {len(fb)} ({', '.join(sorted({c['usage']['served_by'] for c in fb}))})"
                if fb else "")]
    if accts:
        lines.append("    accounts: " + ", ".join(f"{k} {v} calls" for k, v in accts.items()))
    # PI gates per stage: verdict sequence and how the stage closed
    gates: dict[str, list[dict]] = {}
    for e in ev:  # only the latest attempt of each stage (a resumed run restarts an unfinished stage)
        if e.get("action") == "start":
            gates.pop(e["stage"], None)
        if e.get("action") == "pi_gate":
            gates.setdefault(e["stage"], []).append(e)
    pr = [e for e in ev if e.get("action") == "pi_review" and e.get("stage") == "chase"]
    if gates or pr:
        parts = []
        for st, g in gates.items():
            seq = "".join({"pass": "P", "revise": "r", "sign_off": "S"}.get(x.get("verdict"), "?") for x in g)
            closed = g[-1].get("closed")
            parts.append(f"{st} {seq}{' (' + closed + ')' if closed else ''}")
        if pr:
            parts.append(f"chase {len(pr) + 1} PI rounds")
        lines.append("    PI gates: " + " | ".join(parts) + "   (r revise, P pass, S sign-off)")
    lines += model_summary(d)
    return lines


def squeue() -> list[str]:
    env = dict(os.environ, SLURM_CONF="/cm/shared/apps/slurm/etc/slurm/slurm.conf",
               PATH="/cm/local/apps/slurm/current/bin:" + os.environ.get("PATH", ""))
    try:
        out = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-n", "mechanist-p033",
                              "-o", "%i %T %M %R"], check=False, capture_output=True, text=True, env=env, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ["(squeue unavailable)"]
    return [ln for ln in out.splitlines() if ln.strip()] or ["(no Mechanist GPU jobs queued)"]


def main() -> None:
    pats = sys.argv[1:] or DEFAULT
    dirs = sorted({Path(p) for pat in pats for p in glob.glob(str(RUNS / pat))})
    print(f"Mechanist runs — {dt.datetime.now():%Y-%m-%d %H:%M:%S}\n")
    for d in dirs:
        for ln in run_status(d):
            print(ln)
        print()
    print("GPU jobs (local models): " + "; ".join(squeue()))
    q = Path("/vast/projects/thl/understanding-mole/claude-bench/run/claude_queue.log")
    if q.exists():
        print("Claude queue: " + q.read_text().splitlines()[-1][:150])


if __name__ == "__main__":
    main()
