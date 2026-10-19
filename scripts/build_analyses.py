"""Re-run every analysis a finished run made on the study's data, by code, and write a page of them for the portal.

    uv run --extra cells python scripts/build_analyses.py runs/<run>            # re-run, then write the page
    uv run --extra cells python scripts/build_analyses.py runs/<run> --page-only  # the page from the last re-run

Mechanist's agents query the data (gene profiles, co-expression, programme scores, lineage changes, code they wrote)
and quote the numbers. Those numbers should not be taken on trust or only from the agents' text: each call is run
again here through the same tools, with no model involved, and its result compared with the one the run recorded.
The page draws each result (per-sample dot strips, tables, ranked bars) under the circuit step that rests on it, in the
Mechanist site's own design. Writes runs/<run>/site/analyses.json and analyses.html (published beside the site, linked
from the Results page). The re-run needs the study's cell object in memory: run it on a compute node.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
from pathlib import Path

from mechanist import mark
from mechanist.present import load_run
from mechanist.records import ToolCall
from mechanist.runner import RunConfig, Runner

NOT_DATA = {"literature_search", "trial_search", "gene_diseases", "gene_drugs", "find_findings", "known_regulators",
            "receptor_paths", "known_pairs"}
RID = re.compile(r"\bR\d{3,6}\b")
TEMPLATE = Path(__file__).resolve().parents[1] / "src" / "mechanist" / "site_template.html"
KIND = {"gene_profile": "Expression", "gene_by_state": "Expression", "lineage_share": "Composition",
        "state_composition": "Composition", "program_score": "Programmes", "upstream_screen": "Expression",
        "pathway_footprint": "Pathways and regulators", "subset_contrast": "Pathways and regulators",
        "coexpression": "Co-expression", "contamination_check": "Contamination checks",
        "lineage_changes": "What changed most", "compute": "Code the team wrote"}


def cited(ctx: dict) -> dict[str, list[str]]:
    """Result id -> the places that rest on it: circuit steps, mechanism cards, experiments, drugs."""
    out: dict[str, list[str]] = {}

    def add(ids, where):
        for i in ids:
            out.setdefault(i, [])
            if where not in out[i]:
                out[i].append(where)
    for c in (ctx.get("circuits") or {}).get("circuits") or []:
        for k, e in enumerate(c.get("edges") or [], 1):
            if isinstance(e, dict):
                add(e.get("ids") or [], f"circuit {c.get('rank')} step {k}")
    for c in (ctx.get("cards") or {}).get("circuits") or []:
        for k in c.get("cards") or []:
            add(k["seen_here"]["ids"], f"circuit {c.get('rank')} step {k['step']}")
    for e in (ctx.get("experiment") or {}).get("experiments") or []:
        add(RID.findall(json.dumps(e)), f"experiment {e.get('id', '')}".strip())
    for h in (ctx.get("therapy") or {}).get("hypotheses") or []:
        add(RID.findall(json.dumps(h)), f"drug {str(h.get('drug', '')).title()}")
    return out


def _norm(t: str) -> str:
    t = re.sub(r"^\(same call as R\d+; same result\) ", "", str(t or ""))
    return re.sub(r"\s+", " ", t).strip()


def _clean(v):
    """JSON-safe: NaN and infinities become null."""
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


def rerun(run: Path) -> list[dict]:
    ctx, res = load_run(run)
    cfg = RunConfig.from_yaml(run / "config.yaml")
    r = Runner(cfg, run / "analyses_work")
    _ds, fs, _ = r._findings(ctx)
    tb = r.toolbox(fs)
    tb.on_result = None
    where = cited(ctx)
    rows = []
    for rid, x in sorted(res.items(), key=lambda kv: int(kv[0][1:]) if kv[0][1:].isdigit() else 0):
        if x.get("tool") in NOT_DATA or not x.get("ok", True):
            continue
        data = x.get("data")
        try:
            new = tb.run(ToolCall(tool=x["tool"], args=x.get("args") or {}), rid)
            again, ok, data = new.summary, new.ok, (new.data if new.ok else data)
        except Exception as exc:  # noqa: BLE001 - a failed re-run is reported, not hidden
            again, ok = f"re-run failed: {exc}", False
        rows.append({"id": rid, "tool": x["tool"], "args": x.get("args") or {}, "cited_by": where.get(rid, []),
                     "recorded": x.get("summary") or "", "rerun": again, "data": _clean(data),
                     "status": "same" if ok and _norm(again) == _norm(x.get("summary")) else
                               "differs" if ok else "failed"})
    return rows


# ---- figures: each result's data as something to look at --------------------------------------------------------

def short_sample(s: str) -> str:
    return re.sub(r"_(filtered|raw)_feature_bc_matrix$", "", str(s))


def _genes(v) -> list[str]:
    return [str(g) for g in (v if isinstance(v, (list, tuple)) else [v]) if g]


def _strips_from(nested: dict, unit: str = "%", per: int = 4, total: int = 12, floor: float = 5.0) -> tuple[list, int]:
    """{name: {sub: {sample: value}}} -> strip panels, the strongest sub-panels of each name first."""
    panels, left = [], 0
    for name, subs in (nested or {}).items():
        if not isinstance(subs, dict):
            continue
        ranked = sorted(((sub, vals) for sub, vals in subs.items() if isinstance(vals, dict)),
                        key=lambda kv: -max([v for v in kv[1].values() if v is not None] or [0]))
        keep = [kv for kv in ranked if max([v for v in kv[1].values() if v is not None] or [0]) >= floor][:per]
        left += len(ranked) - len(keep)
        panels += [{"title": name, "sub": sub, "unit": unit, "vals": vals} for sub, vals in keep]
    return panels[:total], left + max(0, len(panels) - total)


def figure(row: dict) -> dict:
    """The result as a figure spec the page draws: strips, table, bars or text, with a plain-words title."""
    t, a, d = row["tool"], row["args"], row.get("data") or {}
    g, lin = ", ".join(_genes(a.get("gene"))), a.get("lineage") or a.get("cell") or ""
    try:
        if t == "gene_profile":
            panels, left = _strips_from(d)
            return {"title": f"{g}: share of cells expressing, by cell type", "kind": "strips", "panels": panels,
                    "note": f"{left} weaker cell types (below 5% in every sample) not drawn" if left else ""}
        if t == "gene_by_state":
            panels, left = _strips_from(d, floor=1.0)
            return {"title": f"{g} in {lin}: share of cells expressing, by state", "kind": "strips", "panels": panels,
                    "note": f"{left} states not drawn" if left else ""}
        if t == "lineage_share":
            return {"title": f"{lin} cells as a share of all cells", "kind": "strips",
                    "panels": [{"title": lin, "sub": "of all cells", "unit": "%", "vals": d.get("pct") or {}}]}
        if t == "program_score":
            gs = d.get("genes") or a.get("genes") or []
            return {"title": f"A programme of {len(gs)} genes in {lin}", "kind": "strips", "genes": gs,
                    "panels": [{"title": "programme", "sub": f"mean share of its genes per {lin} cell", "unit": "%",
                                "vals": d.get("pct") or {}}]}
        if t == "state_composition":
            return {"title": f"Which states {lin} cells are in", "kind": "strips",
                    "panels": [{"title": st, "sub": f"% of {lin} cells", "unit": "%", "vals": v}
                               for st, v in (d.get("pct") or {}).items()][:12],
                    "counts": d.get("n") or {}}
        if t == "upstream_screen":  # the tool's own verdict, its first sentence: never a paraphrase of it
            first = re.split(r"(?<=\.)\s", re.sub(r"^PREMISE (FAILS|HOLDS):\s*", "", row.get("rerun") or ""), 1)[0]
            return {"title": f"Is {g} raised in {lin}?", "kind": "strips", "verdict": first,
                    "panels": [{"title": g, "sub": lin, "unit": "%", "vals": d.get("pct") or {}}]}
        if t == "pathway_footprint":
            sets = d.get("sets") or {}
            return {"title": f"Pathway and regulator footprints in {d.get('lineage') or lin}", "kind": "strips",
                    "note": "Scores from each set's target genes, z-scored across samples: a footprint in the "
                            "transcripts, not a measure of activity.",
                    "panels": [{"title": re.sub(r"^TF (.+) targets$", r"\1 targets", n), "sub": "z-score", "unit": "z",
                                "vals": v.get("scores") or {}} for n, v in list(sets.items())[:12]]}
        if t == "coexpression":
            rows = [[short_sample(s), v.get("group", ""), v.get("n"), v.get("pct_a"), v.get("pct_b"), v.get("pct_both"),
                     v.get("expected_both"), v.get("ratio")] for s, v in d.items() if isinstance(v, dict)]
            return {"title": f"Do the same {lin} cells make {', '.join(_genes(a.get('genes_a')))} and "
                             f"{', '.join(_genes(a.get('genes_b')))}?", "kind": "coex", "rows": rows,
                    "a": _genes(a.get("genes_a")), "b": _genes(a.get("genes_b"))}
        if t == "contamination_check":
            m = re.search(r"VERDICT:\s*(.+?)(?:\n|$)", row.get("rerun") or "")
            rows = [[short_sample(s), v.get("group", ""), v.get("pct_all"), v.get("pct_clean"),
                     ", ".join(f"{k} {x:.0%}" for k, x in (v.get("foreign_sources") or {}).items() if x is not None)]
                    for s, v in (d.get("samples") or {}).items() if isinstance(v, dict)]
            return {"title": f"Is {g} in {lin} cells their own, or another cell type's RNA?", "kind": "table",
                    "verdict": (m.group(1).strip() if m else ""),
                    "cols": ["Sample", "Group", f"% of {lin} with {g}", "% without foreign markers", "Foreign markers"],
                    "rows": rows, "pct": [2, 3]}
        if t == "lineage_changes":
            up = [[x.get("gene"), x.get("gap"), ", ".join(x.get("beyond") or [])] for x in (d.get("up_all") or [])][:15]
            down = [[x.get("gene"), x.get("gap"), ", ".join(x.get("beyond") or [])]
                    for x in (d.get("down_all") or [])][:15]
            return {"title": f"What changed most in {lin}", "kind": "bars", "up": up, "down": down,
                    "note": "Genes raised (or lowered) in every patient-group sample beyond every sample of the groups "
                            "named, ranked by the gap in percentage points of cells expressing."}
        if t == "subset_contrast":
            want = [str(x).upper() for x in a.get("sets") or []]
            samples = d.get("samples") or {}
            names = sorted({n for v in samples.values() if isinstance(v, dict) for n in (v.get("scores") or {})
                            if any(re.search(r"\b" + re.escape(w) + r"\b", n.upper()) for w in want)})[:12]
            gname = a.get("gene", "")
            return {"title": f"{gname}-positive against {gname}-negative {lin} cells: pathway footprints",
                    "kind": "strips", "note": f"Within each sample, {gname}+ minus {gname}- cells, in that sample's "
                                              "standard deviations; above zero means higher in the positive cells.",
                    "panels": [{"title": re.sub(r"^TF (.+) targets$", r"\1 targets", n), "sub": "z, within sample",
                                "unit": "z", "vals": {s: (v.get("scores") or {}).get(n) for s, v in samples.items()
                                                      if isinstance(v, dict)}} for n in names]}
        if t == "compute":
            out = (d.get("stdout") if isinstance(d, dict) else "") or row.get("rerun") or ""
            return {"title": a.get("purpose") or "Code the team wrote", "kind": "code", "out": out.strip(),
                    "code": a.get("code") or ""}
    except Exception:  # noqa: BLE001 - an odd result shape falls back to its text
        pass
    return {"title": t.replace("_", " "), "kind": "text", "out": row.get("rerun") or ""}


def page_data(run: Path, rows: list[dict]) -> dict:
    ctx, _res = load_run(run)
    cfg = RunConfig.from_yaml(run / "config.yaml")
    groups = (ctx.get("load") or {}).get("groups") or {}
    ref = (ctx.get("load") or {}).get("reference_group")
    case = cfg.disease if cfg.disease in groups else next(iter(groups), "")
    order = [case] + [g for g in groups if g not in (case, ref)] + ([ref] if ref in groups else [])
    circuits = []
    for c in (ctx.get("circuits") or {}).get("circuits") or []:
        steps = [{"i": k, "source": e.get("source", ""), "target": e.get("target", ""), "via": e.get("via", ""),
                  "status": e.get("status", "")} for k, e in enumerate(c.get("edges") or [], 1) if isinstance(e, dict)]
        circuits.append({"rank": c.get("rank"), "title": c.get("title", ""), "steps": steps})
    an = {}
    for r in rows:
        f = figure(r)
        an[r["id"]] = {"tool": r["tool"], "kind_label": KIND.get(r["tool"], "Other"), "status": r["status"],
                       "cited_by": r["cited_by"], "fig": _clean(f), "recorded": r["recorded"] if r["status"] != "same"
                       else "", "raw": r["rerun"], "args": {k: v for k, v in r["args"].items() if k not in ("code", "purpose")}}
    exps = {e.get("id"): e.get("title", "") for e in (ctx.get("experiment") or {}).get("experiments") or []}
    return {"study": (ctx.get("load") or {}).get("study", ""), "groups": [[g, groups[g]] for g in order],
            "short": {s: short_sample(s) for v in groups.values() for s in v}, "circuits": circuits,
            "experiments": exps, "analyses": an, "design": cfg.design}


def write_page(run: Path, rows: list[dict]) -> Path:
    d = page_data(run, rows)
    tpl = TEMPLATE.read_text()
    style = tpl[tpl.index("<style>") + 7: tpl.index("</style>")]
    fonts = "\n".join(l for l in tpl.splitlines()[:8] if "fonts.g" in l)
    page = (PAGE.replace("/*SITECSS*/", style).replace("<!--FONTS-->", fonts)
            .replace("<!--FAVICON-->", mark.favicon_link()).replace("<!--MARK-->", mark.svg())
            .replace("<!--STUDY-->", html.escape(d["study"]))
            .replace("/*DATA*/", json.dumps(d, ensure_ascii=False).replace("</", "<\\/")))
    out = run / "site" / "analyses.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(page)
    return out


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Mechanist analyses <!--STUDY--></title>
<!--FAVICON-->
<!--FONTS-->
<style>
/*SITECSS*/
/* ---- the analyses page: the site's plate world, one figure per result ---- */
.an-intro{padding-block:38px 6px}
.legend-g{display:flex;flex-wrap:wrap;gap:4px 16px;font:500 13px var(--label);color:var(--ink-2);margin:4px 0 0}
.legend-g i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:-1px}
.tally{display:flex;flex-wrap:wrap;gap:6px 28px;margin:22px 0 0;padding:14px 0;border-top:1px solid var(--ink);border-bottom:1px solid var(--rule)}
.tally div{font:400 13px var(--label);color:var(--muted)}.tally b{display:block;font:600 26px/1.1 var(--serif);color:var(--ink);font-variant-numeric:tabular-nums}
.tools{position:sticky;top:calc(env(safe-area-inset-top,0px) + 58px);z-index:20;background:var(--page);display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center;padding:12px 0;border-bottom:1px solid var(--rule)}
.tools input[type=search],.tools select{font:14px var(--label);padding:6px 10px;border:1px solid var(--rule);background:var(--plate);color:var(--ink);min-width:0;border-radius:0}
.tools input[type=search]{flex:1 1 240px}.tools label{font:500 13.5px var(--label);color:var(--ink-2);display:inline-flex;gap:6px;align-items:center}
.tools .n{font:400 13px var(--label);color:var(--muted);margin-left:auto}
.circ{margin-top:40px}.circ>h2.sec{margin-top:0}
.circ>.ct{font:400 15px/1.45 var(--serif);color:var(--ink-2);max-width:80ch;margin:0 0 10px}
.stp{border-top:1px solid var(--ink);margin-top:18px;padding-top:10px}
.stp>h3{font:600 17px/1.3 var(--serif);margin:0 0 2px}.stp>h3 .k{font:600 12px var(--label);color:var(--muted);margin-right:8px;letter-spacing:.04em;text-transform:uppercase}
.stp>.via{font:400 13.5px var(--label);color:var(--muted);margin:0 0 6px}
.figs{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,430px),1fr));gap:26px 34px;margin-top:10px}
.fig{margin:0;min-width:0;border-top:1px solid var(--rule);padding-top:10px}
.fig>header{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}
.fig h4{font:600 15.5px/1.3 var(--serif);margin:0;flex:1 1 220px;text-wrap:balance}
.fig .id{font:600 12px var(--label);color:var(--ink-2);font-variant-numeric:tabular-nums}
.fig .ok{font:500 12px var(--label);color:var(--muted);display:inline-flex;gap:4px;align-items:center}
.fig .ok svg{width:12px;height:12px;stroke:currentColor;fill:none;stroke-width:2}
.fig .ok.bad{color:var(--sel)}
.fig .vd2{font:500 13.5px/1.45 var(--label);color:var(--ink);margin:8px 0 0}
.fig .nt{font:400 12.5px/1.45 var(--label);color:var(--muted);margin:8px 0 0}
.fig .gl{font:italic 400 13px var(--serif);color:var(--ink-2);margin:4px 0 0}
.fig .also{font:400 12.5px var(--label);color:var(--muted);margin:6px 0 0}
.panels{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:12px 18px;margin-top:10px}
.pn{margin:0;min-width:0}.pn figcaption{font:500 12.5px var(--label);color:var(--ink-2);display:flex;gap:6px;align-items:baseline;flex-wrap:wrap}
.pn figcaption .g{font:italic 600 15px var(--serif);color:var(--ink)}.pn svg{display:block;width:100%;height:auto;margin-top:2px}
.pn .vd{font:500 12px/1.35 var(--label);color:var(--muted);margin-top:2px}.pn .vd.hold{color:var(--ink)}
.tb{overflow-x:auto;margin-top:10px}.tb table{border-collapse:collapse;font:13px/1.35 var(--label);font-variant-numeric:tabular-nums;min-width:100%}
.tb th{font-weight:600;text-align:left;color:var(--ink-2);border-bottom:1px solid var(--ink);padding:5px 10px 5px 0;white-space:nowrap}
.tb td{border-bottom:1px solid var(--hair);padding:5px 10px 5px 0;white-space:nowrap}.tb td.r{text-align:right}
.tb td .sw{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;vertical-align:0}
.bars{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:8px 26px;margin-top:10px}
.bars h5{font:600 12px var(--label);letter-spacing:.04em;text-transform:uppercase;color:var(--muted);margin:0 0 4px}
.br{display:grid;grid-template-columns:86px minmax(0,1fr) 40px;gap:8px;align-items:center;font:13px var(--label);padding:2px 0}
.br .gn{font:italic 600 13.5px var(--serif);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.br .bk{height:7px;background:var(--hair)}.br .bk i{display:block;height:100%;background:var(--g0)}
.br .gp{text-align:right;color:var(--ink-2);font-variant-numeric:tabular-nums}.br small{grid-column:2/-1;color:var(--muted);font-size:11.5px;margin-top:-2px}
.fig pre{white-space:pre-wrap;overflow-wrap:anywhere;overflow-x:auto;font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;color:var(--ink-2);background:var(--plate);border:1px solid var(--hair);padding:9px 11px;margin:8px 0 0;max-height:360px}
.fig details{margin-top:8px}.fig details>summary{cursor:pointer;font:500 12.5px var(--label);color:var(--muted)}
.fig details>summary:hover{color:var(--ink)}.fig .two{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.fig .two b{font:600 12px var(--label);color:var(--muted)}
.rest>details{border-top:1px solid var(--ink);margin-top:14px}.rest>details>summary{cursor:pointer;padding:12px 0;font:600 17px var(--serif)}
.rest>details>summary small{font:400 13px var(--label);color:var(--muted);margin-left:8px}
.empty{font:400 15px var(--serif);color:var(--muted);margin:30px 0}
::selection{background:var(--sel-wash)}
@media (max-width:640px){.bars{grid-template-columns:1fr}.fig .two{grid-template-columns:1fr}.tools{position:static}}
</style></head><body>
<header class="head">
  <div class="wrap">
    <a class="mast" href="index.html#overview"><!--MARK-->Mechanist</a>
    <span class="rh"><!--STUDY--> · the numbers behind the circuits</span>
    <a class="back" href="../index.html">Study results</a>
    <nav class="toc" aria-label="Sections">
      <a href="index.html#overview">Overview<span class="for">for everyone</span></a>
      <a href="index.html#experiments">Experiments<span class="for">for the bench</span></a>
      <a href="index.html#treatments">Treatments<span class="for">for the clinic</span></a>
      <a href="index.html#reasoning">Reasoning<span class="for">every decision</span></a>
      <a href="analyses.html" aria-current="page">Analyses<span class="for">every number</span></a>
    </nav>
    <div class="theme" id="theme" role="group" aria-label="Colour theme">
      <button type="button" data-t="auto" title="Match the system" aria-label="Match the system theme"><svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="5.4"/><path d="M8 2.6a5.4 5.4 0 010 10.8z" fill="currentColor" stroke="none"/></svg></button>
      <button type="button" data-t="light" title="Light" aria-label="Light theme"><svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="2.8"/><path d="M8 1.8v1.4M8 12.8v1.4M1.8 8h1.4M12.8 8h1.4M3.6 3.6l1 1M11.4 11.4l1 1M3.6 12.4l1-1M11.4 4.6l1-1"/></svg></button>
      <button type="button" data-t="dark" title="Dark" aria-label="Dark theme"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M12.8 9.6A5.2 5.2 0 016.4 3.2a5.2 5.2 0 106.4 6.4z"/></svg></button>
    </div>
  </div>
</header>
<main class="wrap">
  <section class="an-intro">
    <h1 class="view">The numbers behind each circuit step</h1>
    <p class="stand">Every query the team ran on this study's data, run again by the pipeline's own code with no model involved. Each result sits under the step that rests on it, drawn per sample, so a claim can be checked against the cells it came from.</p>
    <div class="legend-g" id="legend"></div>
    <div class="tally" id="tally"></div>
  </section>
  <div class="tools">
    <input id="q" type="search" placeholder="Find a gene, a cell type or a result id" aria-label="Find a gene, a cell type or a result id">
    <select id="c" aria-label="Circuit"></select>
    <label><input type="checkbox" id="d"> Only results that differ on re-run</label>
    <span class="n" id="count"></span>
  </div>
  <div id="body"></div>
</main>
<footer class="foot"><div class="wrap"><span>Re-run by Mechanist's analyses script on the study's reviewed cell object. Percentages are shares of cells with at least one read of the gene, per sample.</span></div></footer>
<script>
const D = /*DATA*/;
const $ = s => document.querySelector(s), $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"})[c]);
const GROUPS = D.groups.map(g => g[0]), CASE = GROUPS[0];
const GCOL = Object.fromEntries(GROUPS.map((g, i) => [g, `var(--g${Math.min(i, 3)})`]));
const GROUP_OF = Object.fromEntries(D.groups.flatMap(([g, ss]) => ss.map(s => [s, g])));
const SH = s => D.short[s] || s;
const F = 'font-family="IBM Plex Sans Condensed, Arial Narrow, sans-serif"';
const num = (v, u) => v == null ? "–" : u === "z" ? (v > 0 ? "+" : "") + v.toFixed(2) : v.toFixed(1) + "%";

/* theme: the same stored choice as the site and the study portal */
(function(){ const seg = $$("#theme button");
  const now = () => { try { const v = JSON.parse(localStorage.getItem("clp:theme")); return v === "light" || v === "dark" ? v : "auto"; } catch (e) { return "auto"; } };
  function apply(t, save){ const r = document.documentElement; if (t === "auto") delete r.dataset.theme; else r.dataset.theme = t;
    if (save) { try { t === "auto" ? localStorage.removeItem("clp:theme") : localStorage.setItem("clp:theme", JSON.stringify(t)); } catch (e) {} }
    seg.forEach(b => b.setAttribute("aria-pressed", String(b.dataset.t === t))); }
  seg.forEach(b => b.addEventListener("click", () => apply(b.dataset.t, true)));
  addEventListener("storage", e => { if (e.key === "clp:theme") apply(now(), false); });
  apply(now(), false); })();

/* one dot per sample, one row per group: the site's strip, with a zero line for z-scores */
function strip(p){
  const W = 200, rowH = 15, L = 52, R = 12, H = rowH * GROUPS.length + 20, base = rowH * GROUPS.length + 4;
  const vals = Object.values(p.vals).filter(v => v != null);
  if (!vals.length) return `<p class="nt">No sample has cells of this kind.</p>`;
  let lo = 0, hi;
  if (p.unit === "z") { const m = Math.max(1, ...vals.map(Math.abs)); lo = -Math.ceil(m); hi = Math.ceil(m); }
  else hi = Math.max(10, Math.ceil(Math.max(...vals) / 10) * 10);
  const x = v => L + (W - L - R) * (v - lo) / (hi - lo);
  const ticks = p.unit === "z" ? [lo, 0, hi] : [0, hi / 2, hi];
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(p.title)} ${esc(p.sub || "")}, one dot per sample"><line x1="${L}" y1="${base}" x2="${W-R}" y2="${base}" stroke="var(--rule)"/>`;
  ticks.forEach(t => s += `<line x1="${x(t)}" y1="${base}" x2="${x(t)}" y2="${base+4}" stroke="var(--muted)"/><text x="${x(t)}" y="${base+14}" font-size="9.5" text-anchor="middle" fill="var(--muted)" ${F}>${p.unit === "z" ? (t > 0 ? "+" + t : t) : t + (t === hi ? "%" : "")}</text>`);
  if (p.unit === "z") s += `<line x1="${x(0)}" y1="2" x2="${x(0)}" y2="${base}" stroke="var(--rule)" stroke-dasharray="2 3"/>`;
  GROUPS.forEach((g, r) => { const y = 9 + r * rowH;
    s += `<text x="0" y="${y+4}" font-size="10.5" fill="var(--ink-2)" ${F}>${esc(g.length > 10 ? g.slice(0, 9) + "." : g)}</text><line x1="${L}" y1="${y}" x2="${W-R}" y2="${y}" stroke="var(--hair)"/>`;
    D.groups[r][1].forEach(smp => { const v = p.vals[smp]; if (v == null) return;
      s += `<circle cx="${x(v)}" cy="${y}" r="${r === 0 ? 5 : 4.2}" fill="${GCOL[g]}" stroke="var(--plate)" stroke-width="1"><title>${esc(SH(smp))}: ${num(v, p.unit)}</title></circle>`; }); });
  return s + "</svg>";
}
/* does every patient-group sample sit beyond every other sample? (said, not tested: n is one or two per group) */
function verdict(p){
  const own = D.groups[0][1].map(s => p.vals[s]).filter(v => v != null), rest = D.groups.slice(1).flatMap(([, ss]) => ss.map(s => p.vals[s])).filter(v => v != null);
  if (!own.length || !rest.length) return "";
  if (Math.min(...own) > Math.max(...rest)) return [`every ${CASE} sample above every other sample`, true];
  if (Math.max(...own) < Math.min(...rest)) return [`every ${CASE} sample below every other sample`, true];
  return ["overlaps the other samples", false];
}
const OK = `<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M2 6.5l2.6 2.5L10 3.5"/></svg>`;
function fig(id, where){
  const a = D.analyses[id], f = a.fig;
  const st = a.status === "same" ? `<span class="ok" title="Run again by code: the same result">${OK}same on re-run</span>`
    : `<span class="ok bad">${a.status === "differs" ? "differs on re-run" : "re-run failed"}</span>`;
  let body = "";
  if (f.kind === "strips") body = (f.genes ? `<p class="gl">${esc(f.genes.join(", "))}</p>` : "") +
    `<div class="panels">${f.panels.map(p => { const v = verdict(p); return `<figure class="pn"><figcaption><span class="g">${esc(p.title)}</span>${esc(p.sub || "")}</figcaption>${strip(p)}${v ? `<div class="vd ${v[1] ? "hold" : ""}">${esc(v[0])}</div>` : ""}</figure>`; }).join("")}</div>`;
  else if (f.kind === "coex") body = `<div class="tb"><table><tr><th>Sample</th><th>Cells</th><th>${esc(f.a.join("/"))}</th><th>${esc(f.b.join("/"))}</th><th>Both</th><th>Both if independent</th><th>Ratio</th></tr>${f.rows.map(r => `<tr><td><span class="sw" style="background:${GCOL[r[1]] || "var(--muted)"}"></span>${esc(r[0])}</td><td class="r">${r[2] ?? "–"}</td><td class="r">${num(r[3], "%")}</td><td class="r">${num(r[4], "%")}</td><td class="r"><b>${num(r[5], "%")}</b></td><td class="r">${num(r[6], "%")}</td><td class="r">${r[7] == null ? "–" : "×" + r[7].toFixed(2)}</td></tr>`).join("")}</table></div><p class="nt">A ratio near 1 means the two are made by the same cells only as often as chance would put them together; above 1, the same cells; below 1, different cells.</p>`;
  else if (f.kind === "table") body = `<div class="tb"><table><tr>${f.cols.map(c => `<th>${esc(c)}</th>`).join("")}</tr>${f.rows.map(r => `<tr>${r.map((v, i) => `<td class="${(f.pct || []).includes(i) ? "r" : ""}">${i === 0 ? `<span class="sw" style="background:${GCOL[r[1]] || "var(--muted)"}"></span>` : ""}${(f.pct || []).includes(i) ? num(v, "%") : esc(v ?? "–")}</td>`).join("")}</tr>`).join("")}</table></div>`;
  else if (f.kind === "bars") { const mx = Math.max(1, ...f.up.concat(f.down).map(r => r[1] || 0));
    const col = (h, rs) => `<div><h5>${h}</h5>${rs.length ? rs.map(r => `<div class="br"><span class="gn">${esc(r[0])}</span><span class="bk"><i style="width:${100 * (r[1] || 0) / mx}%"></i></span><span class="gp">${(r[1] || 0).toFixed(0)}</span><small>beyond ${esc(r[2])}</small></div>`).join("") : `<p class="nt">None.</p>`}</div>`;
    body = `<div class="bars">${col("Raised", f.up)}${col("Lowered", f.down)}</div>`; }
  else if (f.kind === "code") body = `<pre>${esc(f.out)}</pre><details><summary>The code</summary><pre>${esc(f.code)}</pre></details>`;
  else body = `<pre>${esc(f.out)}</pre>`;
  const others = a.cited_by.filter(w => w !== where);
  return `<figure class="fig" data-id="${id}" data-text="${esc((id + " " + f.title + " " + JSON.stringify(a.args) + " " + (f.genes || []).join(" ") + " " + (f.panels || []).map(p => p.title + " " + p.sub).join(" ")).toLowerCase())}" data-st="${a.status}">
    <header><h4>${esc(f.title)}</h4><span class="id">${id}</span>${st}</header>${f.verdict ? `<p class="vd2">${esc(f.verdict)}</p>` : ""}${body}${f.note ? `<p class="nt">${esc(f.note)}</p>` : ""}
    ${others.length ? `<p class="also">Also used by ${esc(others.join("; "))}.</p>` : ""}
    ${a.status !== "same" ? `<div class="two"><div><b>Recorded in the run</b><pre>${esc(a.recorded)}</pre></div><div><b>Run again now</b><pre>${esc(a.raw)}</pre></div></div>` : `<details><summary>The tool's own words</summary><pre>${esc(a.raw)}</pre></details>`}</figure>`;
}
/* where each result is used */
const at = {}; Object.entries(D.analyses).forEach(([id, a]) => a.cited_by.forEach(w => (at[w] ||= []).push(id)));
function render(){
  const q = $("#q").value.trim().toLowerCase(), c = $("#c").value, diff = $("#d").checked;
  let html = "";
  D.circuits.filter(cc => !c || String(cc.rank) === c).forEach(cc => {
    const steps = cc.steps.map(s => ({s, ids: (at[`circuit ${cc.rank} step ${s.i}`] || [])})).filter(x => x.ids.length);
    if (!steps.length) return;
    html += `<section class="circ"><h2 class="sec">${cc.rank === 1 ? "Lead circuit" : "Circuit " + cc.rank}</h2><p class="ct">${esc(cc.title)}</p>` +
      steps.map(({s, ids}) => `<div class="stp"><h3><span class="k">Step ${s.i}</span>${esc(s.source)} → ${esc(s.target)}</h3>${s.via ? `<p class="via">via ${esc(s.via)} · ${esc(s.status)}</p>` : ""}<div class="figs">${ids.map(id => fig(id, `circuit ${cc.rank} step ${s.i}`)).join("")}</div></div>`).join("") + `</section>`;
  });
  if (!c) {
    const ex = Object.keys(at).filter(w => /^experiment |^drug /.test(w)).sort();
    if (ex.length) html += `<section class="circ"><h2 class="sec">Behind the experiments and drugs</h2>` + ex.map(w => `<div class="stp"><h3>${esc(w.replace(/^experiment (E\d+)/, (m, e) => `Experiment ${e}: ${D.experiments[e] || ""}`).replace(/^drug /, "Drug: "))}</h3><div class="figs">${at[w].map(id => fig(id, w)).join("")}</div></div>`).join("") + `</section>`;
    const rest = Object.entries(D.analyses).filter(([, a]) => !a.cited_by.length);
    const byk = {}; rest.forEach(([id, a]) => (byk[a.kind_label] ||= []).push(id));
    if (rest.length) html += `<section class="circ rest"><h2 class="sec">Asked along the way, not cited by a step</h2><p class="ct">Questions the team ran while it worked that no final step rests on. Open a kind to read them.</p>` +
      Object.entries(byk).sort((a, b) => b[1].length - a[1].length).map(([k, ids]) => `<details><summary>${esc(k)}<small>${ids.length} results</small></summary><div class="figs">${ids.map(id => fig(id, "")).join("")}</div></details>`).join("") + `</section>`;
  }
  $("#body").innerHTML = html || `<p class="empty">Nothing here.</p>`;
  let shown = 0;
  $$("#body .fig").forEach(f => { const hit = (!q || f.dataset.text.includes(q)) && (!diff || f.dataset.st !== "same"); f.hidden = !hit; shown += hit; });
  $$("#body .stp, #body .rest>details").forEach(s => { s.hidden = !$$(".fig", s).some(f => !f.hidden); if (q && s.tagName === "DETAILS" && !s.hidden) s.open = true; });
  $$("#body .circ").forEach(s => s.hidden = !$$(".fig", s).some(f => !f.hidden));
  $("#count").textContent = `${shown} shown`;
}
$("#legend").innerHTML = D.groups.map(([g, ss]) => `<span><i style="background:${GCOL[g]}"></i>${esc(g)} <small style="color:var(--muted)">${esc(ss.map(SH).join(", "))}</small></span>`).join("");
const all = Object.values(D.analyses), n = s => all.filter(a => a.status === s).length;
$("#tally").innerHTML = `<div><b>${all.length}</b>queries on the data</div><div><b>${all.filter(a => a.cited_by.length).length}</b>that a step, experiment or drug rests on</div><div><b>${n("same")}</b>the same when run again</div><div><b>${n("differs") + n("failed")}</b>different or failed</div>`;
$("#c").innerHTML = `<option value="">Every circuit</option>` + D.circuits.map(c => `<option value="${c.rank}">${c.rank === 1 ? "Lead circuit" : "Circuit " + c.rank}</option>`).join("");
let tmr; $("#q").addEventListener("input", () => { clearTimeout(tmr); tmr = setTimeout(render, 150); });
$("#c").addEventListener("change", render); $("#d").addEventListener("change", render); render();
</script></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("--page-only", action="store_true", help="write the page from the saved re-run (no cells needed)")
    a = ap.parse_args()
    saved = a.run / "site" / "analyses.json"
    if a.page_only:
        rows = json.loads(saved.read_text())
        res = load_run(a.run)[1]  # a re-run saved before figures existed: the run's own records hold the same data
        for r in rows:
            if "data" not in r:
                r["data"] = _clean((res.get(r["id"]) or {}).get("data"))
    else:
        rows = rerun(a.run)
        saved.parent.mkdir(exist_ok=True)
        saved.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    out = write_page(a.run, rows)
    tally = {s: sum(r["status"] == s for r in rows) for s in ("same", "differs", "failed")}
    print(f"{len(rows)} analyses: {tally}; wrote {out}")


if __name__ == "__main__":
    main()
