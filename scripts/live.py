"""The relay between a running run and the portal (docs/LIVE.md), through Box.

    python3 scripts/live.py relay runs/<run> [--every 30] [--watch PID]   # beside the run, in its Slurm job
    python3 scripts/live.py status runs/<run>                              # the live picture, printed

Every pass: messages the portal left in box:.../inbox/mechanist_chat/<run>/ are copied into runs/<run>/inbox/ (the run
registers, answers and delivers them) and moved to inbox/processed/; then the live picture (stages, the latest
reasoning, the conversation) goes to box:.../status/mechanist_live/<study>__<run>.json, where the portal reads it.
With --watch the relay ends, after a last pass, when that process (the run) has ended. Standard library and yaml only
(the system python runs it). A failed pass is logged and retried; it never touches the run.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

MECH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MECH / "src"))
from mechanist import live  # noqa: E402

BOX = os.environ.get("BOX_ROOT", "box:CL Analysis Portal")


def rclone_bin() -> str:
    found = shutil.which("rclone") or os.environ.get("RCLONE_BIN")
    if found:
        return found
    hits = sorted(glob.glob("/vast/parcc/spack/sw/apps/*/rclone-*/bin/rclone"))
    return hits[-1] if hits else "rclone"


def rclone(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([rclone_bin(), *args], capture_output=True, text=True, timeout=120)


def study_of(run: Path) -> str:
    import yaml
    try:
        return str((yaml.safe_load((run / "config.yaml").read_text()) or {}).get("study") or "")
    except OSError:
        return ""


def pull(run: Path) -> int:
    """Copy new messages for this run from Box into runs/<run>/inbox/; return how many."""
    remote = f"{BOX}/inbox/mechanist_chat/{run.name}"
    r = rclone("lsjson", "--files-only", remote)
    if r.returncode:
        return 0  # no folder yet: nobody has written
    n = 0
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (run / "inbox").mkdir(exist_ok=True)
    for f in json.loads(r.stdout or "[]"):
        name = f["Name"]
        if not name.endswith(".json") or name.endswith(".portal.json"):
            continue
        dest = run / "inbox" / name
        if not dest.exists():
            tmp = run / "inbox" / f".{name}.part"
            if rclone("copyto", f"{remote}/{name}", str(tmp)).returncode:
                continue
            os.replace(tmp, dest)
            n += 1
        for x in (name, name[:-5] + ".portal.json"):
            rclone("moveto", f"{remote}/{x}", f"{BOX}/inbox/processed/{stamp}/mechanist_chat/{run.name}/{x}")
    return n


def push(run: Path, study: str, running: bool) -> bool:
    data = live.status(run, study, running=running)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(data, fh, ensure_ascii=False)
    try:
        return rclone("copyto", fh.name, f"{BOX}/status/mechanist_live/{study}__{run.name}.json").returncode == 0
    finally:
        os.unlink(fh.name)


def alive(pid: int | None) -> bool:
    if not pid:
        return True
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def relay(run: Path, every: float, watch: int | None) -> None:
    study = study_of(run)
    while True:
        up = alive(watch)
        try:
            got = pull(run)
            ok = push(run, study, running=up)
            if got or not ok:
                print(f"{dt.datetime.now():%H:%M:%S} relay {run.name}: {got} new message(s); status {'sent' if ok else 'NOT sent'}",
                      flush=True)
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            print(f"relay {run.name}: pass failed ({str(exc)[:200]}); retrying", flush=True)
        if not up:
            return
        time.sleep(every)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=["relay", "status"])
    ap.add_argument("run", type=Path)
    ap.add_argument("--every", type=float, default=30)
    ap.add_argument("--watch", type=int, help="end after this process ends")
    a = ap.parse_args()
    run = a.run.resolve()
    if a.what == "status":
        print(json.dumps(live.status(run, study_of(run)), indent=1, ensure_ascii=False))
    else:
        relay(run, a.every, a.watch)


if __name__ == "__main__":
    main()
