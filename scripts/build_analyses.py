"""Re-run every analysis a finished run made on the study's data, by code, and write a page of them for the portal.

    uv run --extra cells python scripts/build_analyses.py runs/<run>

Mechanist's agents query the data (gene profiles, co-expression, programme scores, lineage changes, code they wrote)
and quote the numbers. Those numbers should not be taken on trust or only from the agents' text: each call is run
again here through the same tools, with no model involved, and its result compared with the one the run recorded.
Writes runs/<run>/site/analyses.json and runs/<run>/site/analyses.html (published beside the Mechanist site, linked
from the Results page). Needs the study's cell object in memory: run it on a compute node, not the login node.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

from mechanist.present import load_run
from mechanist.records import ToolCall
from mechanist.runner import RunConfig, Runner

NOT_DATA = {"literature_search", "trial_search", "gene_diseases", "gene_drugs", "find_findings", "known_regulators",
            "receptor_paths", "known_pairs"}
RID = re.compile(r"\bR\d{3,6}\b")


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
        try:
            new = tb.run(ToolCall(tool=x["tool"], args=x.get("args") or {}), rid)
            again, ok = new.summary, new.ok
        except Exception as exc:  # noqa: BLE001 - a failed re-run is reported, not hidden
            again, ok = f"re-run failed: {exc}", False
        rows.append({"id": rid, "tool": x["tool"], "args": x.get("args") or {}, "cited_by": where.get(rid, []),
                     "recorded": x.get("summary") or "", "rerun": again,
                     "status": "same" if ok and _norm(again) == _norm(x.get("summary")) else
                               "differs" if ok else "failed"})
    rows.sort(key=lambda r_: (not r_["cited_by"], int(r_["id"][1:]) if r_["id"][1:].isdigit() else 0))
    return rows


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Mechanist analyses {study}</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Condensed:wght@400;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap">
<style>
:root{{--page:#f4f2ed;--plate:#fff;--ink:#161513;--ink-2:#4b4842;--muted:#6f6b62;--hair:#e2ded5;--sel:#8c1d18;--ok:#2f5d34;
  --serif:"Source Serif 4",Georgia,serif;--label:"IBM Plex Sans Condensed","Arial Narrow",sans-serif;color-scheme:light}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--page:#11100e;--plate:#1a1916;--ink:#f1eee6;--ink-2:#c2bdb1;
  --muted:#a19c91;--hair:#2f2c27;--sel:#ef8a7c;--ok:#9fcf9a;color-scheme:dark}}}}
:root[data-theme="dark"]{{--page:#11100e;--plate:#1a1916;--ink:#f1eee6;--ink-2:#c2bdb1;--muted:#a19c91;--hair:#2f2c27;--sel:#ef8a7c;
  --ok:#9fcf9a;color-scheme:dark}}
body{{margin:0;background:var(--page);color:var(--ink);font:16px/1.55 var(--serif)}}
.wrap{{max-width:1080px;margin:0 auto;padding-block:32px 60px;padding-left:max(16px,env(safe-area-inset-left,0px));
  padding-right:max(16px,env(safe-area-inset-right,0px))}}
h1{{font-size:30px;line-height:1.15;margin:0 0 8px;text-wrap:balance}}.stand{{max-width:68ch;color:var(--ink-2);margin:0 0 18px}}
.tally{{font:600 13px var(--label);color:var(--ink-2);margin:0 0 14px}}.tally b{{color:var(--ink)}}
.bar{{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 14px}}.bar input,.bar select{{font:14px var(--label);padding:6px 9px;
  border:1px solid var(--hair);background:var(--plate);color:var(--ink);min-width:0}}.bar input{{flex:1 1 260px}}
.a{{border-top:1px solid var(--hair);padding:10px 0}}.h{{display:flex;flex-wrap:wrap;gap:6px 12px;align-items:baseline;
  font:600 13.5px var(--label)}}.h .t{{color:var(--ink-2);font-weight:400}}.h .s{{margin-left:auto;font-size:12.5px}}
.s.same{{color:var(--ok)}}.s.differs,.s.failed{{color:var(--sel)}}.args{{font:12.5px ui-monospace,Menlo,Consolas,monospace;
  color:var(--muted);overflow-wrap:anywhere;margin:3px 0}}.cb{{font:12.5px var(--label);color:var(--ink-2)}}
details>summary{{cursor:pointer;font:12.5px var(--label);color:var(--muted);margin-top:4px}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:12.5px/1.5 ui-monospace,Menlo,Consolas,monospace;color:var(--ink-2);
  margin:6px 0 0;background:var(--plate);padding:8px 10px;border:1px solid var(--hair)}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}.two b{{font:600 12px var(--label);color:var(--muted)}}
@media (max-width:700px){{.two{{grid-template-columns:1fr}}}}
</style></head><body><div class="wrap">
<h1>The analyses behind Mechanist's {study} results</h1>
<p class="stand">Every query Mechanist's agents ran on this study's data, run again by the pipeline's own code with no model
involved, on the same data. Analyses that a circuit step, experiment or drug rests on come first. Where the re-run
does not give the recorded result, it is flagged; read both before relying on that number.</p>
<p class="tally" id="tally"></p>
<div class="bar"><input id="q" type="search" placeholder="Filter: a gene, a cell type, a step, an id" aria-label="Filter analyses">
<select id="f" aria-label="Show"><option value="">All analyses</option><option value="cited">Cited by a step, experiment or drug</option>
<option value="differs">Re-run differs or failed</option></select></div>
<div id="list"></div></div>
<script>
const A = /*DATA*/;
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}})[c]);
const WORD = {{same: "re-run gives the same result", differs: "re-run differs", failed: "re-run failed"}};
const n = s => A.filter(a => a.status === s).length;
document.getElementById("tally").innerHTML = `<b>${{A.length}}</b> analyses · <b>${{A.filter(a => a.cited_by.length).length}}</b> cited · <b>${{n("same")}}</b> reproduced exactly · <b>${{n("differs")}}</b> differ · <b>${{n("failed")}}</b> failed`;
function draw(){{ const q = document.getElementById("q").value.trim().toLowerCase(), f = document.getElementById("f").value;
  const hits = A.filter(a => (!f || (f === "cited" ? a.cited_by.length : a.status !== "same")) &&
    (!q || (a.id + " " + a.tool + " " + JSON.stringify(a.args) + " " + a.cited_by.join(" ") + " " + a.rerun).toLowerCase().includes(q)));
  document.getElementById("list").innerHTML = hits.map(a => `<div class="a"><div class="h">${{esc(a.id)}}<span class="t">${{esc(a.tool.replace(/_/g, " "))}}</span><span class="s ${{a.status}}">${{WORD[a.status]}}</span></div>
    <div class="args">${{esc(JSON.stringify(a.args))}}</div>${{a.cited_by.length ? `<div class="cb">Used by: ${{esc(a.cited_by.join("; "))}}</div>` : ""}}
    ${{a.status === "same" ? `<details><summary>result</summary><pre>${{esc(a.rerun)}}</pre></details>` :
      `<div class="two"><div><b>Recorded in the run</b><pre>${{esc(a.recorded)}}</pre></div><div><b>Re-run now</b><pre>${{esc(a.rerun)}}</pre></div></div>`}}</div>`).join(""); }}
document.getElementById("q").addEventListener("input", draw); document.getElementById("f").addEventListener("change", draw); draw();
</script></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", type=Path)
    a = ap.parse_args()
    rows = rerun(a.run)
    site = a.run / "site"
    site.mkdir(exist_ok=True)
    (site / "analyses.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    study = (load_run(a.run)[0].get("load") or {}).get("study", "")
    page = PAGE.format(study=html.escape(study)).replace("/*DATA*/", json.dumps(rows, ensure_ascii=False).replace("</", "<\\/"))
    (site / "analyses.html").write_text(page)
    tally = {s: sum(r["status"] == s for r in rows) for s in ("same", "differs", "failed")}
    print(f"{len(rows)} analyses re-run: {tally}; wrote {site / 'analyses.html'}")


if __name__ == "__main__":
    main()
