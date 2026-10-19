"""Publish a finished run's Mechanist site to the study's portal page and to Box.

    python3 scripts/publish_site.py runs/<run> [--no-upload]

1. runs/published/<study>/index.html: the portal's results-report build copies this folder into the study's report
   and switches the Mechanist tab on (Projects/scripts/build_results_report.py), so the site survives every rebuild.
2. The live page now, without waiting for a rebuild: report/mechanist/, the tab switched on in report/data/catalog.js,
   and the Mechanist button in report/index.html as the current template draws it.
3. Those files uploaded to box:CL Analysis Portal/projects/<study>/report/ (the portal serves from Box).

Standard library only (the queue runs it with the system python). Every replaced file is kept under
runs/published/_previous/<study>_<time>/ first.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

MECH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MECH / "src"))
from mechanist import mark  # noqa: E402

PORTAL = MECH.parent / "Interactive" / "output"
BOX = "box:CL Analysis Portal/projects"
RCLONE = "/vast/parcc/spack/sw/apps/linux-zen4/rclone-1.70.0-goyyiowsjornovon55227fml5snz2yu3/bin/rclone"
TITLE = "Mechanist: proposed disease mechanisms, experiments and treatments for this study"
BUTTON = (f'<a href="mechanist/index.html" id="viewMechanist" title="{TITLE}">'
          + mark.svg(small=True, blue="var(--mk)") + "Mechanist ↗</a>")
CSS = ('#viewMechanist{--mk:#2b6cd4;display:inline-flex;align-items:center;gap:6px}#viewMechanist[hidden]{display:none}'
       '#viewMechanist .mk{width:16px;height:16px;flex:none}'
       '@media (prefers-color-scheme:dark){:root:not([data-theme="light"]) #viewMechanist{--mk:#8db4ec}}'
       ':root[data-theme="dark"] #viewMechanist{--mk:#8db4ec}')
ANCHOR = re.compile(r'<a\b[^>]*\bid="viewMechanist"[^>]*>.*?</a>', re.S)
ANALYSES = ('<a href="mechanist/analyses.html" id="viewAnalyses" title="Every data query behind Mechanist\'s results, '
            're-run by the pipeline\'s own code">Analyses</a>')
ANALYSES_RE = re.compile(r'<a\b[^>]*\bid="viewAnalyses"[^>]*>.*?</a>', re.S)


def study_of(run: Path) -> str:
    m = re.search(r"^study:\s*(\S+)", (run / "config.yaml").read_text(), re.M)
    if not m:
        raise SystemExit(f"no study in {run / 'config.yaml'}")
    return m.group(1).strip("'\"")


def keep(path: Path, backup: Path) -> None:
    if path.exists():
        backup.mkdir(parents=True, exist_ok=True)
        (shutil.copytree if path.is_dir() else shutil.copy2)(path, backup / path.name)


def with_tab(catalog: str, analyses: bool = False) -> str:
    """window.CATALOG={...}; with mechanist_tab set to the site (and mechanist_analyses to its analyses page)."""
    head, _, body = catalog.partition("=")
    data = json.loads(body.strip().rstrip(";"))
    data["mechanist_tab"] = "mechanist/index.html"
    if analyses:
        data["mechanist_analyses"] = "mechanist/analyses.html"
    return f"{head}={json.dumps(data, ensure_ascii=False, separators=(',', ':'))};\n"


def with_button(page: str, analyses: bool = False) -> str:
    if not ANCHOR.search(page):
        raise SystemExit("the Results page has no Mechanist tab to switch on")
    page = ANALYSES_RE.sub("", page)
    page = ANCHOR.sub(lambda _m: BUTTON + (ANALYSES if analyses else ""), page, count=1)
    if "#viewMechanist{" not in page:
        page = page.replace("</style>", CSS + "\n</style>", 1)
    return page.replace('id="viewTabs" aria-label="Study views" hidden>', 'id="viewTabs" aria-label="Study views">', 1)


def publish(run: Path, upload: bool = True) -> None:
    site = run / "site" / "index.html"
    if not site.exists():
        raise SystemExit(f"no site at {site}; build it first (scripts/build_site.py {run} --portal)")
    pid = study_of(run)
    backup = MECH / "runs" / "published" / "_previous" / f"{pid}_{time.strftime('%Y%m%d_%H%M%S')}"
    pub = MECH / "runs" / "published" / pid
    keep(pub, backup)
    shutil.rmtree(pub, ignore_errors=True)
    pub.mkdir(parents=True)
    shutil.copy2(site, pub / "index.html")
    analyses = (run / "site" / "analyses.html").exists()  # scripts/build_analyses.py, when it has been run
    if analyses:
        shutil.copy2(run / "site" / "analyses.html", pub / "analyses.html")
    print(f"published {run.name} to {pub}")

    folders = sorted(PORTAL.glob(f"*__{pid}"))
    if len(folders) != 1:
        print(f"no single portal folder for {pid} ({len(folders)} found); the next portal build will add it")
        return
    report = folders[0] / "report"
    for f in ("index.html", "data/catalog.js", "mechanist"):
        keep(report / f, backup / "live" / Path(f).parent)
    shutil.rmtree(report / "mechanist", ignore_errors=True)
    shutil.copytree(pub, report / "mechanist")
    cat = report / "data" / "catalog.js"
    cat.write_text(with_tab(cat.read_text(), analyses))
    page = report / "index.html"
    page.write_text(with_button(page.read_text(), analyses))
    print(f"live page updated: {report}")
    if not upload:
        return
    dest = f"{BOX}/{pid}/report"
    for f in ("index.html", "data/catalog.js", "mechanist/index.html") + (("mechanist/analyses.html",) if analyses else ()):
        subprocess.run([RCLONE, "copyto", str(report / f), f"{dest}/{f}"], check=True)
    print(f"uploaded to {dest}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("--no-upload", action="store_true")
    a = ap.parse_args()
    publish(a.run.resolve(), upload=not a.no_upload)


if __name__ == "__main__":
    main()
