"""Build the portal's Mechanist interpretation pages for a finished run (display only).

    uv run --extra cells python scripts/build_interpretation.py runs/<run> [--out DIR] [--publish PROJECT_DIR]

Writes interpretation.html (short, figure-led), mechanism.html (the full deep dive) and interpretation.json to
runs/<run>/interpretation/ (or --out). One model call writes the short page's words (cached, checked by code); the
figures and the deep dive are drawn by code from the run's records. --publish PROJECT_DIR (read only: its title and
whether it has a Report tab) also copies the folder to runs/published/<study>/, where the results-report build
(Projects/scripts/build_results_report.py) picks it up as the third study view. Nothing is written under Projects/.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

from mechanist import present


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--reuse-page", action="store_true",
                    help="keep the short page's words from the last build (no model call): for layout changes only")
    ap.add_argument("--publish", type=Path, help="the study folder under Projects/ (e.g. ../Projects/p033), read only")
    a = ap.parse_args()
    out = a.out or a.run / "interpretation"
    project = a.publish
    title, sub = "", ""
    if project and (project / "metadata" / "submission.yaml").exists():
        s = yaml.safe_load((project / "metadata" / "submission.yaml").read_text()) or {}
        st = s.get("project") or s.get("study") or s
        title, sub = st.get("title", ""), f"{project.name} · {st.get('description', '')}"
    tabs = [("Results", "../index.html")]
    if project and (project / "Results" / "report_tab" / "findings.json").exists():
        tabs.append(("Report", "../report.html"))
    present.build(a.run, out, tabs, title or a.run.name, sub or "Mechanist run", reuse_page=a.reuse_page)
    print(f"wrote {out}/interpretation.html and mechanism.html")
    if project:
        dest = Path(__file__).resolve().parents[1] / "runs" / "published" / project.name
        shutil.copytree(out, dest, dirs_exist_ok=True, ignore=shutil.ignore_patterns("plain.json", "plain_calls"))
        print(f"published to {dest}")


if __name__ == "__main__":
    main()
