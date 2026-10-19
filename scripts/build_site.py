"""Build the standalone Mechanist site for a run from its outputs alone: one HTML file with four views (overview with
every circuit as an interactive ring figure, experiments, treatments, and the full reasoning record).

    python scripts/build_site.py RUN [--out FILE] [--portal]

RUN is the newest run folder; earlier stages are followed through the folders it was rerun from. --portal writes a
full HTML document with a link back to the study's Results page; without it the file is an artifact fragment."""
import argparse
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mechanist import agents, cellfig, mark  # noqa: E402
from mechanist.present import load_run, panel_data, holds_in  # noqa: E402

TEMPLATE = ROOT / "src" / "mechanist" / "site_template.html"


def lst(x):
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(x) if str(x).startswith("[") else []
    except (ValueError, SyntaxError):
        return []


def short_title(t: str) -> str:
    return re.split(r"[:(]", t)[0].strip()


def run_dirs(run: Path) -> list[Path]:
    """The run and each folder it was rerun from, newest first."""
    out, cur = [], run
    while cur is not None and cur not in out:
        out.append(cur)
        nxt = None
        ev = cur / "events.jsonl"
        if ev.exists():
            for line in ev.read_text().splitlines():
                r = json.loads(line)
                if r.get("action") == "inputs" and r.get("source"):
                    src = Path(r["source"])
                    nxt = src if src.is_absolute() else cur.parent.parent / src
        cur = nxt
    return out


def circuit_block(c: dict, groups: dict, gorder, samples, pdata) -> dict:
    fid = f"c{c['rank']}"
    edges = [e for e in c.get("edges") or [] if isinstance(e, dict)]
    graph = cellfig.build_graph(edges)
    if not graph["ring"]:
        return {"svg": "", "steps": [], "cells": {}, "ends": {}}
    svg = cellfig.figure(graph, fid, short_title(c["title"]))
    arrows = graph["arrows"]
    k = iter(range(len(arrows)))
    svg = re.sub(r'<path d="([^"]+)"( fill="none" stroke="[^"]+"[^>]*marker-end="url\(#' + fid + r'-[a-z]+\)")>(<title>.*?</title>)</path>',
                 lambda m: (lambda i: f'<g class="arw" data-a="{i}"><path class="vis" d="{m.group(1)}"{m.group(2)}>{m.group(3)}</path>'
                            f'<path class="hit" d="{m.group(1)}"/></g>')(next(k)), svg, flags=re.S)
    icon_of = {n["icon"]: n["id"] for n in graph["ring"] if n["icon"]}
    pos = {}
    for m in re.finditer(r'<use href="#bio-([a-z_]+)" x="([\d.]+)" y="([\d.]+)" width="(\d+)"', svg):
        if m.group(1) in icon_of:
            pos[icon_of[m.group(1)]] = (float(m.group(2)) + int(m.group(4)) / 2, float(m.group(3)) + int(m.group(4)) / 2 + 12)
    for n in graph["ring"]:
        if n["id"].startswith("trigger:"):
            m = re.search(r'<rect x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="36"[^>]*/>\s*<text[^>]*>'
                          + re.escape(cellfig.esc(n["label"])[:18]), svg)
            if m:
                pos[n["id"]] = (float(m.group(1)) + float(m.group(3)) / 2, float(m.group(2)) + 18)
    label = {n["id"]: n["label"] for n in graph["ring"]}
    hits = "".join(f'<circle class="chit" data-c="{cid}" tabindex="0" role="button" aria-label="{cellfig.esc(label[cid])}" '
                   f'cx="{x:.1f}" cy="{y:.1f}" r="{40 if cid.startswith("trigger:") else 70}"/>'
                   f'<circle class="ring" data-c="{cid}" cx="{x:.1f}" cy="{y:.1f}" r="{44 if cid.startswith("trigger:") else 74}"/>'
                   for cid, (x, y) in pos.items())
    svg = svg.replace("</svg>", hits + "</svg>")
    steps = []
    for i, raw in enumerate(edges, 1):
        note = f"{raw['source']} → {raw['target']}"
        on_arrows = [j for j, a in enumerate(arrows) if any(n.startswith(note) for n in a["notes"])]
        on_cells = [] if on_arrows else [x for x in cellfig.cells_in(raw["source"]) if x in pos]
        ids = lst(raw.get("ids"))
        text = f"{raw['source']} {raw['target']} {raw.get('via', '')}"
        up = text.upper()
        want = {cellfig.norm(g) for g in cellfig.genes_in(text, 12)} | ({"TNFSF14"} if "LIGHT" in up else set()) | \
               ({"TNFRSF14"} if "HVEM" in up else set())
        cells_txt = [w for w in " ".join(cellfig.cells_in(raw["source"]) + cellfig.cells_in(raw["target"]))
                     .replace("_", " ").split() if len(w) > 2]
        strips, seen = [], set()
        for (rid, g, lin), vals in pdata.items():
            if rid not in ids or cellfig.norm(g) not in want or (g, lin) in seen:
                continue
            if cells_txt and not any(w in lin.lower() for w in cells_txt):
                continue
            seen.add((g, lin))
            h = holds_in(vals, gorder)
            n_case = len([x for x in gorder[0][1] if vals.get(x) is not None])
            other = [v for s, v in vals.items() if s not in gorder[0][1] and v is not None]
            up_ = bool(h) and other and all(vals[x] > max(other) for x in h)
            verdict = ((f"every {gorder[0][0]} sample above every comparator" if up_
                        else f"every {gorder[0][0]} sample below every comparator")
                       if h and len(h) == n_case else f"{len(h)} of {n_case} {gorder[0][0]} samples beyond the comparators"
                       if h else "overlaps the comparators")
            strips.append({"rid": rid, "gene": g, "lin": lin, "vals": vals, "verdict": verdict})
        strips.sort(key=lambda s: (s["verdict"] == "overlaps the comparators", -max(v or 0 for v in s["vals"].values())))
        genes = [g for g in cellfig.genes_in(text, 8) if re.fullmatch(r"[A-Z0-9]{3,}[A-Z0-9-]*", g) and g not in samples]
        steps.append({"i": i, "source": raw["source"], "target": raw["target"], "via": raw.get("via", ""),
                      "status": raw.get("status", "speculative"), "note": raw.get("note", ""), "ids": ids,
                      "genes": genes, "arrows": on_arrows, "cells": on_cells, "strips": strips[:4]})
    return {"svg": svg, "steps": steps, "ends": {i: [a["from"], a["to"]] for i, a in enumerate(arrows)},
            "cells": {n["id"]: {"label": n["label"], "sub": n["sub"]} for n in graph["ring"]}}


def full_record(dirs: list[Path], res: dict) -> list[list]:
    """The whole run in time order, nothing cut: every agent turn with the model's full reply (from the call cache), every
    tool call with its full arguments and result, and the harness's notes. Rows: [time, stage, who, kind, text, full,
    result id]. The transcript itself clips long lines, so the cache and the stage results are the sources of record."""
    def reply(key: str) -> str:
        for d in dirs:
            f = d / "llm_cache" / f"{key}.json"
            if f.exists():
                try:
                    return str(json.loads(f.read_text()).get("response") or "")
                except ValueError:
                    return ""
        return ""
    rows, seen = [], set()
    for d in reversed(dirs):  # the oldest folder first: a rerun folder repeats nothing it carried over
        tp = d / "transcript.jsonl"
        if not tp.exists():
            continue
        for line in tp.read_text().splitlines():
            t = json.loads(line)
            tag, kind = str(t.get("tag") or ""), t.get("kind", "")
            sig = (t.get("time"), t.get("role"), tag, t.get("text", "")[:80])
            if sig in seen:
                continue
            seen.add(sig)
            src = str(t.get("for") or tag) if kind == "result" else tag
            stage = re.sub(r"\d+$", "", src.split(":")[0]) if ":" in src else ("harness" if kind == "note" else "")
            if kind == "result":
                r = res.get(tag) or {}
                args = json.dumps(r.get("args"), ensure_ascii=False) if r else ""
                full = f"{r.get('tool', '')}({args})\n{r.get('summary') or ''}".strip() if r else t.get("text", "")
                rows.append([t["time"], stage, "tool", kind, t.get("text", ""), full, tag])
            else:
                full = reply(t["key"]) if t.get("key") else ""
                rows.append([t["time"], stage, t.get("role", ""), kind, t.get("text", ""),
                             full if full and full.strip() != t.get("text", "").strip() else "", ""])
    rows.sort(key=lambda r: r[0])
    return rows


def reviews(v) -> list[dict]:
    out = []
    for r in v or []:
        if not isinstance(r, dict):
            continue
        text = r.get("assessment") or ""
        if not text and r.get("depth"):
            text = " ".join(f"{d.get('link')}: {d.get('missing')}" for d in r["depth"] if isinstance(d, dict))
        raw = r.get("demands") if isinstance(r.get("demands"), list) else r.get("objections")
        dem = [d.get("text") if isinstance(d, dict) else str(d) for d in (raw if isinstance(raw, list) else [])]
        out.append({"verdict": r.get("verdict", ""), "attempt": r.get("attempt"), "text": text, "demands": dem})
    return out


def questions(qs) -> list[dict]:
    return [{k: q.get(k) for k in ("id", "kind", "text", "answer", "conclusion", "status")} | {"ids": lst(q.get("answer_ids"))}
            for q in qs or []]


def build(run: Path, portal: bool, human: bool = True) -> str:
    ctx, res = load_run(run)
    dirs = run_dirs(run)
    groups = ctx["load"]["groups"]
    from mechanist.runner import RunConfig
    rcfg = RunConfig.from_yaml(run / "config.yaml")
    case, design = rcfg.disease, rcfg.design
    case = case if case in groups else next(iter(groups))
    ref = ctx["load"].get("reference_group")
    order = [case] + [g for g in groups if g not in (case, ref)] + ([ref] if ref in groups else [])
    gorder = [(g, groups[g]) for g in order if g in groups]
    samples = {x for _, v in gorder for x in v}
    circs = ctx["circuits"]["circuits"]
    pdata = panel_data(res)
    blocks = [circuit_block(c, groups, gorder, samples, pdata) for c in circs]
    keys = {n["icon"] for c in circs for n in cellfig.build_graph([e for e in c["edges"] if isinstance(e, dict)])["ring"]
            if n["icon"]}
    circuits = []
    for c, b in zip(circs, blocks):
        cnt = {}
        for e in c.get("edges") or []:
            cnt[e.get("status")] = cnt.get(e.get("status"), 0) + 1
        circuits.append({"rank": c["rank"], "title": c["title"], "short": short_title(c["title"]), "support": c["support"],
                         "summary": c.get("summary", ""), "status": cnt, "merged_from": c.get("merged_from") or [],
                         "steps": b["steps"], "cells": b["cells"], "ends": b["ends"], "has_fig": bool(b["svg"])})
    figs = "".join(f'<template id="fig-{c["rank"]}">{b["svg"]}</template>' for c, b in zip(circs, blocks) if b["svg"])
    cards = {(cc.get("rank"), k["step"]): k for cc in (ctx.get("cards") or {}).get("circuits") or [] for k in cc["cards"]}
    for c in circuits:  # the mechanism card of each step (stage "cards"), when the run has one
        for st_ in c["steps"]:
            st_["card"] = cards.get((c["rank"], st_["i"]))

    ex, th = ctx.get("experiment") or {}, ctx.get("therapy") or {}
    ms = {m["id"]: m for m in ex.get("missing_links") or th.get("missing_links") or []}
    dossier = [{"id": i, "section": sec, "text": t}
               for i, sec, t in agents.run_items(ctx, design)]
    attempts = [{"attempt": a["attempt"], "circuits": [{"rank": c["rank"], "title": c["title"]} for c in a.get("circuits") or []]}
                for a in (ctx.get("circuits") or {}).get("attempts") or []]
    chase_q, check_q = questions((ctx.get("chase") or {}).get("questions")), questions((ctx.get("check") or {}).get("questions"))
    st = lambda k: ctx.get(k) or {}  # noqa: E731
    stages = [
        {"key": "dossier", "n": 1, "name": "Pathology dossier", "count": f"{len(dossier)} items", "reviews": reviews(st("dossier").get("reviews")),
         "asks": "What does the literature say this disease is, and how does it differ from each comparator?"},
        {"key": "check", "n": 2, "name": "Do we see the pathology?", "count": f"{len(check_q)} questions", "reviews": reviews(st("check").get("reviews")),
         "asks": "Is each dossier item present in this data, per sample and against the comparators? What is unexpected?"},
        {"key": "candidates", "n": 3, "name": "Candidate mechanisms", "count": f"{len(st('candidates').get('hypotheses') or [])} candidates",
         "reviews": reviews(st("candidates").get("reviews")),
         "asks": "Which mechanisms could produce the pathology, what does each explain, what would refute it?"},
        {"key": "chase", "n": 4, "name": "Chase", "count": f"{len(chase_q)} questions", "reviews": reviews(st("chase").get("pi_reviews")),
         "asks": "For each candidate: why here, what is it doing, what is upstream, how does it link to the disease?"},
        {"key": "linkage", "n": 5, "name": "Disease linkage", "count": f"{len(st('linkage').get('links') or [])} links",
         "reviews": reviews(st("linkage").get("reviews")), "asks": "What published work links each step to this disease or a related one?"},
        {"key": "circuits", "n": 6, "name": "Circuits", "count": f"{sum(len(a['circuits']) for a in attempts)} proposed, {len(circs)} after merging",
         "reviews": reviews(st("circuits").get("reviews")),
         "asks": "With the facts settled: which complete circuits could produce the disease as seen? Three attempts work blind, then merge."},
        {"key": "experiment", "n": 7, "name": "Experiments", "count": f"{len(ex.get('experiments') or [])} experiments",
         "reviews": reviews(ex.get("reviews")), "asks": "What would show whether each circuit is true?"},
        {"key": "therapy", "n": 8, "name": "Treatments", "count": f"{len(th.get('hypotheses') or [])} drugs", "reviews": reviews(th.get("reviews")),
         "asks": "Which drugs act on the circuits, how sure is each to break them, what does each depend on?"},
    ]
    record = {"dossier": dossier, "check": {"summary": st("check").get("summary"), "questions": check_q},
              "candidates": [{k: h.get(k) for k in ("id", "title", "summary")} for h in st("candidates").get("hypotheses") or []],
              "chase": chase_q, "linkage": [{"link": x.get("link"), "suggests": x.get("suggests")} for x in st("linkage").get("links") or []],
              "attempts": attempts, "ranking": st("circuits").get("ranking", "")}
    results = {rid: {"tool": r.get("tool"), "args": json.dumps(r.get("args"), ensure_ascii=False),
                     "summary": r.get("summary") or ""} for rid, r in res.items()}

    # correspondence of the experiment and treatment stages, from the newest run that made each
    chat, have = [], set()
    for d in dirs:
        tp = d / "transcript.jsonl"
        if not tp.exists():
            continue
        rows = []
        for line in tp.read_text().splitlines():
            t = json.loads(line)
            tag = str(t.get("tag", ""))
            stg = "therapy" if tag.startswith("therapy") else "experiment" if tag.startswith("experiment") else ""
            if t["role"] == "tool" or not stg or stg in have:
                continue
            rows.append({"stage": stg, "role": t["role"], "time": t["time"][11:16], "text": t["text"]})
        have |= {r["stage"] for r in rows}
        chat += rows
    gates = 0
    for d in dirs:
        ev = d / "events.jsonl"
        if ev.exists():
            gates += sum(1 for line in ev.read_text().splitlines() if '"action": "pi_gate"' in line)

    page = {}
    for d in dirs:
        p = d / "interpretation" / "interpretation.json"
        if p.exists():
            page = json.loads(p.read_text())
            break
    data = {"page": {k: page.get(k) for k in ("question", "answer", "points")}, "groups": dict(gorder), "case": case,
            "circuits": circuits, "experiments": ex.get("experiments") or [], "programmes": ex.get("programmes") or [],
            "not_testable": ex.get("not_testable") or [], "therapy": th.get("hypotheses") or [], "missing": ms,
            "considered": th.get("considered") or [], "trace": full_record(dirs, res), "design": design,
            "stages": stages, "record": record, "results": results, "chat": chat,
            "counts": {"tools": len(res), "lit": sum(1 for r in res.values() if r.get("tool") == "literature_search"),
                       "gates": gates, "attempts": len(attempts) or 1},
            "runline": " · ".join(d.name for d in dirs)}
    if human:
        rep = humanize(data, run, run / "site")
        print(f"prose: rewrote {rep['rewritten']} of {rep['passages']} passages; everwrite checker strong findings "
              f"{rep['strong_before']} -> {rep['strong_after']}, weak {rep['weak_before']} -> {rep['weak_after']}")
        data["prose"] = {k: rep[k] for k in ("passages", "rewritten", "strong_before", "strong_after", "weak_before", "weak_after")}
    study = ctx["load"].get("study", "")
    head = (f"{esc(study)} · {esc(', '.join(order))}" + ("" if portal else " · mockup"))
    tpl = TEMPLATE.read_text()
    out = (tpl.replace("<!--SPRITE-->", cellfig.icon_defs(keys)).replace("<!--FIGS-->", figs)
           .replace("<!--RUNNINGHEAD-->", head).replace("<!--MARK-->", mark.svg())
           .replace("<!--FAVICON-->", mark.favicon_link()).replace("<!--STUDY-->", esc(study))
           .replace("<!--BACK-->", ('<a class="back" href="../index.html">Study results</a>' if portal else "")
                    + (' <a class="back" href="analyses.html">Analyses</a>' if (run / "site" / "analyses.html").exists() else ""))
           .replace("/*DATA*/", json.dumps(data, ensure_ascii=False).replace("</", "<\\/")))
    if portal:
        out = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
               '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
               + out.replace("<!--BODY-->", "</head><body>", 1) + "</body></html>")
    else:
        out = out.replace("<!--BODY-->", "", 1)
    return out


# reader-facing prose: rewritten to read as a scientist wrote it (agents.HUMANIZE_SYSTEM), kept only when every number,
# name and id survives (present.plain_check). The reasoning record and the correspondence stay as the agents wrote them.
PROSE = {"page": ("answer", "question"), "points": ("text",), "circuit": ("summary",), "step": ("note",),
         "experiment": ("title", "shows", "adds", "refutes_if", "caveat", "feasibility", "rna_vs_protein"),
         "prediction": ("why",), "programme": ("claim",), "seqstep": ("why_now", "if_yes", "if_no", "adapt"),
         "drug": ("breaks", "rationale"), "not_testable": ("why",),
         "card": ("effect", "follows", "leads_to", "must_hold")}


def _prose_slots(data: dict):
    """(dict, key) for every reader-facing prose field of the site's data."""
    yield from ((data["page"], k) for k in PROSE["page"])
    yield from ((x, k) for x in data["page"].get("points") or [] for k in PROSE["points"])
    for c in data["circuits"]:
        yield from ((c, k) for k in PROSE["circuit"])
        yield from ((st, k) for st in c["steps"] for k in PROSE["step"])
        for st in c["steps"]:
            if st.get("card"):
                yield from ((st["card"], k) for k in PROSE["card"])
                yield from ((x, "claim") for x in st["card"]["literature"])
                if st["card"]["seen_here"]["text"]:
                    yield st["card"]["seen_here"], "text"
    for e in data["experiments"]:
        yield from ((e, k) for k in PROSE["experiment"])
        yield from ((p, k) for p in e.get("predictions") or [] for k in PROSE["prediction"])
    for pr in data["programmes"]:
        yield from ((pr, k) for k in PROSE["programme"])
        yield from ((st, k) for st in pr.get("sequence") or [] for k in PROSE["seqstep"])
    yield from ((d, k) for d in data["therapy"] for k in PROSE["drug"])
    yield from ((x, k) for x in data["not_testable"] for k in PROSE["not_testable"])


def _strong(text: str) -> list[dict]:
    """The everwrite checker's strong findings for one passage (mechanist.tells, vendored from the skill)."""
    from mechanist import tells
    return [h for h in tells.check_text(text)[0] if h["severity"] == "strong"]


def _record_slots(data: dict):
    """(container, key) for the reasoning record and the correspondence: every passage a reader sees there."""
    for st in data["stages"]:
        for r in st["reviews"]:
            yield r, "text"
            yield from ((r["demands"], i) for i in range(len(r["demands"])))
    rec = data["record"]
    yield from ((x, "text") for x in rec["dossier"])
    for k in ("expected_present", "expected_absent", "unexpected"):
        yield from ((x, f) for x in (rec["check"].get("summary") or {}).get(k) or [] for f in ("text", "why", "checks"))
    yield from ((q, f) for q in rec["check"]["questions"] + rec["chase"] for f in ("text", "answer"))
    yield from ((x, f) for x in rec["candidates"] for f in ("title", "summary"))
    yield from ((x, "suggests") for x in rec["linkage"])
    yield rec, "ranking"
    for m in data["chat"]:  # a letter's prose only: not its verdict word, plan line or tool call
        t = m["text"]
        if t.startswith(("ANSWER", "PLAN")):
            continue
        v = re.match(r"VERDICT \w+:\s*", t)
        if v:
            m["head"], m["text"] = t[:v.end()], t[v.end():]
        elif " → " in t:
            m["text"], m["call"] = t.split(" → ", 1)
        yield m, "text"


def rewrite(texts: list[str], run: Path, cache_dir: Path, known=()) -> tuple[dict[str, str], dict]:
    """Rewrite passages so they read as a scientist wrote them, in two passes: the humanize prompt (humanizer +
    everwrite rules), then the everwrite checker (tells.py) on every passage, with a targeted fix for any passage that
    still has a strong finding. A rewrite is kept only when present.plain_check passes against the original.
    Returns {original: rewrite} and a report."""
    import hashlib
    from concurrent.futures import ThreadPoolExecutor

    from mechanist import present, tells
    from mechanist.llm import LLMConfig, LoggedLLM, make_backend, parse_json
    from mechanist.runner import RunConfig
    orig = sorted({t.strip() for t in texts if isinstance(t, str) and len(t.strip()) >= 40})
    cache_dir.mkdir(parents=True, exist_ok=True)
    cfg = RunConfig.from_yaml(run / "config.yaml")  # the run's own backend and accounts (subscription seats)
    lc = LLMConfig.model_validate({**cfg.llm.model_dump(), "model": "sonnet", "thread_model": "", "effort": "medium"})
    (cache_dir / "humanize_calls").mkdir(parents=True, exist_ok=True)
    llm = LoggedLLM(make_backend(lc), cache_dir / "humanize_calls")
    # pass 1: the humanize prompt
    new = present.plain_pass(cache_dir, orig, llm=llm, system=agents.HUMANIZE_SYSTEM,
                             cache="humanized_v3.json", tag="humanize", batch_chars=6000)
    cur = {t: new.get(t, t) for t in orig}
    # refused rewrites: re-check with the study's own names allowed, then one retry that names what went wrong
    known = frozenset(known)
    p1 = json.loads((cache_dir / "humanized_v3.json").read_text())
    hk = lambda t: hashlib.sha256(t.encode()).hexdigest()[:20]  # noqa: E731
    retry = []
    for o in orig:
        r = p1.get(hk(o)) or {}
        if cur[o] == o and r.get("text"):
            why = present.plain_check(o, r["text"], known)
            if not why:
                cur[o] = r["text"]
            elif re.match(r"(numbers|names|ids) changed", why):
                retry.append((o, why))
    rpath = cache_dir / "humanized_retry.json"
    rdone = json.loads(rpath.read_text()) if rpath.exists() else {}
    todo = [(o, w) for o, w in retry if hk(o) not in rdone]
    if todo:
        def again(batch):
            body = "\n\n".join(f"### n={i}\nYour last rewrite was refused ({w}); keep those exactly.\n{o}"
                                 for i, (o, w) in enumerate(batch))
            try:
                items = {int(x.get("n", -1)): str(x.get("text", "")).strip() for x in
                         (parse_json(llm.complete("humanize_retry", agents.HUMANIZE_SYSTEM, body)) or {}).get("items") or []
                         if isinstance(x, dict)}
            except Exception as exc:  # noqa: BLE001
                return [(o, {"ok": False, "why": str(exc)[:200]}) for o, _w in batch]
            return [(o, {"ok": bool(items.get(i)), "text": items.get(i, "")}) for i, (o, _w) in enumerate(batch)]
        with ThreadPoolExecutor(4) as ex:
            for res in ex.map(again, [todo[i:i + 8] for i in range(0, len(todo), 8)]):
                for o, r in res:
                    rdone[hk(o)] = r
        rpath.write_text(json.dumps(rdone, indent=1, ensure_ascii=False))
    for o, _w in retry:
        r = rdone.get(hk(o)) or {}
        if r.get("ok") and not present.plain_check(o, r["text"], known):
            cur[o] = r["text"]
    # pass 2: everything the checker still flags, with its findings, fixed and checked against the original
    path = cache_dir / "humanized_fix.json"
    done = json.loads(path.read_text()) if path.exists() else {}
    todo = [(t, _strong(t)) for t in cur.values() if _strong(t) and hk(t) not in done]
    batches = [todo[i:i + 12] for i in range(0, len(todo), 12)]

    def one(batch):
        body = "\n\n".join(f"### n={i}\nFindings: " + "; ".join(f"{h['category']}: {h['match']}: {h['hint']}" for h in hits)
                           + f"\n{t}" for i, (t, hits) in enumerate(batch))
        try:
            items = {int(x.get("n", -1)): str(x.get("text", "")).strip() for x in
                     (parse_json(llm.complete("humanize_fix", agents.HUMANIZE_FIX_SYSTEM, body)) or {}).get("items") or []
                     if isinstance(x, dict)}
        except Exception as exc:  # noqa: BLE001 - a failed batch keeps the pass-1 text
            return [(t, {"ok": False, "why": str(exc)[:200]}) for t, _h in batch]
        return [(t, {"ok": bool(items.get(i)), "text": items.get(i, "")}) for i, (t, _h) in enumerate(batch)]
    if batches:
        with ThreadPoolExecutor(4) as ex:
            for res in ex.map(one, batches):
                for t, r in res:
                    done[hk(t)] = r
        path.write_text(json.dumps(done, indent=1, ensure_ascii=False))
    for o, t in list(cur.items()):
        r = done.get(hk(t))
        if r and r.get("ok") and not present.plain_check(o, r["text"], known) and len(_strong(r["text"])) < len(_strong(t)):
            cur[o] = r["text"]
    count = lambda ts, sev: sum(1 for t in ts for h in tells.check_text(t)[0] if h["severity"] == sev)  # noqa: E731
    report = {"passages": len(orig), "rewritten": sum(1 for o in orig if cur[o] != o),
              "strong_before": count(orig, "strong"), "strong_after": count(cur.values(), "strong"),
              "weak_before": count(orig, "weak"), "weak_after": count(cur.values(), "weak")}
    return cur, report


def humanize(data: dict, run: Path, cache_dir: Path) -> dict:
    """Every passage a reader sees, on every page, rewritten in place (see rewrite)."""
    slots = [(c, k) for c, k in list(_prose_slots(data)) + list(_record_slots(data))
             if isinstance(c[k] if isinstance(c, list) else c.get(k), str)]
    known = set(data["groups"]) | {w for g in data["groups"] for w in re.split(r"[_\W]+", g) if w}
    m, report = rewrite([c[k] for c, k in slots], run, cache_dir, known)
    for c, k in slots:
        t = c[k].strip()
        if m.get(t):
            c[k] = m[t]
    (cache_dir / "prose_report.json").write_text(json.dumps(report, indent=1))
    return report


def esc(x) -> str:
    return str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out")
    ap.add_argument("--portal", action="store_true")
    ap.add_argument("--no-humanize", action="store_true", help="skip the prose pass (no model calls)")
    a = ap.parse_args()
    run = Path(a.run)
    html = build(run, a.portal, not a.no_humanize)
    out = Path(a.out) if a.out else run / ("site" if a.portal else "site_artifact") / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print("wrote", out, len(html) // 1024, "KB")
