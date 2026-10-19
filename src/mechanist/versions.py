"""Versions of a finished run: new input (bench results, facts, questions, steers) makes a new run folder that points
to its parent and reruns only the stages after the circuits (docs/VERSIONS.md). The parent is never written.

    runs/<run>/        v1, the original
    runs/<run>.v2/     version.json, inputs.yaml, files/, refine.json, circuits.json (refined), cards, experiment, ...
"""

from __future__ import annotations

import datetime as dt
import json
import re
import shutil
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path

import yaml

OUTCOMES = ("positive", "negative", "mixed", "inconclusive")
NOTE_KINDS = ("fact", "question", "steer")
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


class InputError(ValueError):
    pass


def root_of(run: Path) -> Path:
    """The original run folder of any version (runs/x.v3 -> runs/x)."""
    return run.with_name(re.sub(r"\.v\d+$", "", run.name))


def number(run: Path) -> int:
    m = re.search(r"\.v(\d+)$", run.name)
    return int(m.group(1)) if m else 1


def info(run: Path) -> dict:
    """The version's record; for the original run, a record made up on the spot."""
    p = run / "version.json"
    if p.exists():
        return json.loads(p.read_text())
    st = run / "state.json"
    when = ""
    if st.exists():
        fins = [v.get("finished", "") for v in json.loads(st.read_text()).values() if isinstance(v, dict)]
        when = max(fins or [""])[:10]
    return {"version": 1, "parent": None, "label": "original run", "date": when, "items": [], "blinded": None}


def family(run: Path) -> list[dict]:
    """Every version of the run that exists on disk, oldest first: {version, folder, label, date}."""
    root = root_of(run)
    out = [{"version": 1, "folder": root.name, "label": "original run", "date": info(root)["date"]}] if root.exists() else []
    for p in sorted(root.parent.glob(root.name + ".v*"), key=number):
        if (p / "version.json").exists():
            v = info(p)
            out.append({"version": v["version"], "folder": p.name, "label": v.get("label", ""), "date": v.get("date", "")})
    return out


def _pmid(pmid: str, fetch: Callable[[str], dict] | None) -> dict | None:
    """Europe PMC's record of one PMID: title and abstract, or None when it does not resolve."""
    url = EPMC + "?" + urllib.parse.urlencode({"query": f"EXT_ID:{pmid} AND SRC:MED", "format": "json",
                                               "resultType": "core", "pageSize": "1"})
    try:
        body = fetch(url) if fetch else json.load(urllib.request.urlopen(url, timeout=30))
    except OSError:
        return None
    hits = (body.get("resultList") or {}).get("result") or []
    if not hits:
        return None
    r = hits[0]
    return {"pmid": str(r.get("pmid") or pmid), "title": re.sub(r"<[^>]+>", "", r.get("title", "")),
            "year": r.get("pubYear"), "abstract": re.sub(r"<[^>]+>", "", r.get("abstractText", ""))[:3000]}


def load_inputs(path: Path, plan_ids: set[str], fetch: Callable[[str], dict] | None = None) -> dict:
    """The version's input file, checked: X ids in order (results, then notes), outcomes from the list, experiment ids
    that the parent's plan has, files that exist, and each fact's PMID looked up (kept but marked unsourced when it
    does not resolve). Raises InputError listing every problem when the file cannot be used."""
    raw = yaml.safe_load(path.read_text()) or {}
    problems, items = [], []
    for r in raw.get("results") or []:
        x = {"id": f"X{len(items) + 1}", "kind": "result", "experiment": str(r.get("experiment") or "").strip(),
             "title": str(r.get("title") or "").strip(), "outcome": str(r.get("outcome") or "").strip().lower(),
             "what": str(r.get("what") or "").strip(), "found": str(r.get("found") or "").strip(),
             "n": str(r.get("n") or "").strip(), "files": [str(f) for f in r.get("files") or []]}
        if x["outcome"] not in OUTCOMES:
            problems.append(f"{x['id']}: outcome must be one of {', '.join(OUTCOMES)}")
        if x["experiment"] and x["experiment"] not in plan_ids:
            problems.append(f"{x['id']}: experiment {x['experiment']} is not in the plan ({', '.join(sorted(plan_ids))})")
        if not x["experiment"] and not x["title"]:
            problems.append(f"{x['id']}: an experiment outside the plan needs a title")
        if not x["found"]:
            problems.append(f"{x['id']}: say what was found")
        for f in x["files"]:
            if not (path.parent / f).exists():
                problems.append(f"{x['id']}: file {f} not found next to {path.name}")
        items.append(x)
    for n in raw.get("notes") or []:
        x = {"id": f"X{len(items) + 1}", "kind": str(n.get("kind") or "").strip().lower(),
             "text": str(n.get("text") or "").strip(), "source": str(n.get("source") or "").strip()}
        if x["kind"] not in NOTE_KINDS:
            problems.append(f"{x['id']}: kind must be one of {', '.join(NOTE_KINDS)}")
        if not x["text"]:
            problems.append(f"{x['id']}: empty note")
        if x["kind"] == "fact":
            m = re.search(r"\d{5,9}", x["source"])
            x["paper"] = _pmid(m.group(0), fetch) if m else None
            x["sourced"] = x["paper"] is not None
        items.append(x)
    if not items:
        problems.append("no results and no notes")
    if problems:
        raise InputError("; ".join(problems))
    return {"by": str(raw.get("by") or ""), "date": str(raw.get("date") or dt.date.today().isoformat()), "items": items}


def make(parent: Path, inputs_path: Path, inputs: dict, label: str = "") -> Path:
    """A new version folder after `parent`: the inputs and their files copied in, version.json written, and the
    parent named as its source so every loader reads the version as parent plus changes."""
    root = root_of(parent)
    n = max([v["version"] for v in family(root)] + [1]) + 1
    child = root.with_name(f"{root.name}.v{n}")
    child.mkdir()
    shutil.copy2(inputs_path, child / "inputs.yaml")
    files = [f for x in inputs["items"] for f in x.get("files") or []]
    if files:
        (child / "files").mkdir()
        for f in files:
            shutil.copy2(inputs_path.parent / f, child / "files" / Path(f).name)
    kinds = {x["kind"] for x in inputs["items"]}
    has_results, has_notes = "result" in kinds, bool(kinds - {"result"})
    rec = {"version": n, "parent": parent.name, "date": inputs["date"], "by": inputs["by"],
           "label": label or ("after bench results" if has_results and not has_notes else
                              "after notes" if has_notes and not has_results else "after bench results and notes"),
           "kind": [k for k, on in (("results", has_results), ("notes", has_notes)) if on],
           "items": inputs["items"], "blinded": False if has_notes else info(parent).get("blinded")}
    (child / "version.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    with (child / "events.jsonl").open("a") as fh:  # the chain every loader follows (present.load_run, run_stage)
        fh.write(json.dumps({"stage": "refine", "action": "inputs", "source": str(parent.resolve()),
                             "time": dt.datetime.now().isoformat(timespec="seconds")}) + "\n")
    return child


def items_text(items: list[dict]) -> str:
    """The X items as the refine stage reads them."""
    out = []
    for x in items:
        if x["kind"] == "result":
            head = f"{x['id']} RESULT ({x['outcome']})" + (f" of plan experiment {x['experiment']}" if x["experiment"]
                                                             else f": {x['title']} (not in the plan)")
            out.append(f"{head}\n  what was done: {x['what'] or '(not said)'}\n  found: {x['found']}"
                       + (f"\n  n: {x['n']}" if x["n"] else "") + (f"\n  files: {', '.join(x['files'])}" if x["files"] else ""))
        elif x["kind"] == "fact":
            p = x.get("paper")
            out.append(f"{x['id']} FACT (literature, not data): {x['text']}"
                       + (f"\n  source PMID {p['pmid']} ({p['year']}) {p['title']}\n  abstract: {p['abstract'][:1500]}"
                          if p else f"\n  UNSOURCED: '{x['source'] or 'no source'}' did not resolve; weigh it as a claim"))
        else:
            out.append(f"{x['id']} {x['kind'].upper()}: {x['text']}")
    return "\n\n".join(out)


def portal_names(cfg) -> dict:
    """The Results page's names for the comparison a Mechanist figure shows: the run's compartment, and the portal
    labels of its case and reference groups (the config maps portal labels to the run's own names)."""
    opts = cfg.adapter_options or {}
    back = {v: k for k, v in (opts.get("groups") or {}).items()}
    ref = opts.get("reference")
    return {"compartment": opts.get("compartment", ""), "group": back.get(cfg.disease, cfg.disease),
            "reference": back.get(ref, ref or "")}


def diff(old: dict, new: dict) -> dict:
    """What changed from the parent version to this one, by code: circuits, step grades, experiments, drugs."""
    def circs(ctx):
        return {c.get("title", ""): c for c in (ctx.get("circuits") or {}).get("circuits") or []}
    oc, nc = circs(old), circs(new)
    out: dict = {"added": [c["rank"] for t, c in nc.items() if t not in oc],
                 "dropped": [{"rank": c["rank"], "title": t} for t, c in oc.items() if t not in nc],
                 "moved": [{"title": t, "from": oc[t]["rank"], "to": c["rank"]} for t, c in nc.items()
                           if t in oc and oc[t]["rank"] != c["rank"]], "grades": []}
    for t, c in nc.items():
        before = {(e.get("source"), e.get("target")): e.get("status") for e in (oc.get(t) or {}).get("edges") or []}
        for i, e in enumerate(c.get("edges") or [], 1):
            was = before.get((e.get("source"), e.get("target")))
            if was != e.get("status"):
                out["grades"].append({"rank": c["rank"], "step": i, "edge": f"{e.get('source')} → {e.get('target')}",
                                      "from": was or "new step", "to": e.get("status"),
                                      "ids": [x for x in e.get("ids") or [] if str(x).startswith("X")]})
    oe = {e.get("title"): e for e in (old.get("experiment") or {}).get("experiments") or []}
    ne = (new.get("experiment") or {}).get("experiments") or []
    out["new_experiments"] = [e.get("id") for e in ne if e.get("title") not in oe]
    od = {h["drug"]: h.get("rank") for h in (old.get("therapy") or {}).get("hypotheses") or []}
    nd = {h["drug"]: h.get("rank") for h in (new.get("therapy") or {}).get("hypotheses") or []}
    out["drugs"] = {"added": [d for d in nd if d not in od], "dropped": [d for d in od if d not in nd],
                    "moved": [{"drug": d, "from": od[d], "to": r} for d, r in nd.items() if d in od and od[d] != r]}
    return out
