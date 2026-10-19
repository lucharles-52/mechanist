"""Follow a Mechanist run as a conversation: who said what, in order, as it happens.

    python3 scripts/transcript_view.py runs/<run>            # whole transcript so far, then follow new turns
    python3 scripts/transcript_view.py runs/<run> --no-follow --role PI --last 40
Roles: planner, analyst Qnn, tool, PI, synthesizer, summariser, disease scholar, panel: <name>, literature scout,
linkage scholar Ln, experimentalist. Replayed turns (served from the cache on a resume) are dimmed.
"""
import argparse
import json
import shutil
import sys
import textwrap
import time
from pathlib import Path

COLOURS = {"PI": "1;31", "planner": "1;34", "synthesizer": "1;35", "experimentalist": "1;32", "disease scholar": "36",
           "summariser": "36", "literature scout": "36", "tool": "2", "PI (candidates)": "1;31",
           "harness": "1;33"}


def colour(role: str) -> str:
    for k, c in COLOURS.items():
        if role == k or role.startswith(k):
            return c
    return "33" if role.startswith("analyst") else "36" if role.startswith(("linkage", "panel")) else "0"


def show(rec: dict, width: int, roles: list[str] | None) -> None:
    role = rec.get("role", "?")
    if roles and not any(role.startswith(r) for r in roles):
        return
    tty = sys.stdout.isatty()
    head = f"{rec.get('time', '')[11:19]} {role}" + (" (replayed)" if rec.get("replayed") else "")
    body = textwrap.fill(rec.get("text", ""), width=width, initial_indent="    ", subsequent_indent="    ")
    if tty:
        c = "2" if rec.get("replayed") else colour(role)
        print(f"\033[{c}m{head}\033[0m\n{body}")
    else:
        print(f"{head}\n{body}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--no-follow", action="store_true")
    ap.add_argument("--role", action="append", help="only these roles (prefix match), e.g. --role PI --role analyst")
    ap.add_argument("--last", type=int, help="start with only the last N lines")
    a = ap.parse_args()
    path = a.run / "transcript.jsonl" if a.run.is_dir() else a.run
    width = max(60, min(shutil.get_terminal_size((120, 40)).columns - 2, 160))
    while not path.exists():
        if a.no_follow:
            sys.exit(f"no transcript at {path}")
        time.sleep(2)
    with path.open() as fh:
        lines = fh.readlines()
        for line in lines[-a.last:] if a.last else lines:
            try:
                show(json.loads(line), width, a.role)
            except json.JSONDecodeError:
                pass
        while not a.no_follow:
            line = fh.readline()
            if not line:
                time.sleep(1)
                continue
            try:
                show(json.loads(line), width, a.role)
                sys.stdout.flush()
            except json.JSONDecodeError:
                pass


if __name__ == "__main__":
    main()
