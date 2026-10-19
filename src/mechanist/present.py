"""The portal's "Mechanist interpretation" pages for a finished run: display only, no stage reads them.

    interpretation.html  short and figure-led: the answer, the lead circuit as a diagram, one evidence card per observed
                         step (per-sample dot plots drawn from the run's own tool results), how each circuit fits the
                         disease, the alternatives and the tests that would decide between them.
    mechanism.html       the full deep dive, written by code from the run's records with nothing condensed: every step
                         of every circuit with its reasoning and figures, loops, phenotype accounting, predictions, the
                         PI's reviews, the evidence ledger and every cited data query.

The short page's words come from one model call (agents.PRESENT_SYSTEM) checked by code: no number or id the circuits
lack, figure panels only from real results, every circuit present, verdicts from a fixed vocabulary. A page that fails
the check is retried once with the reasons, and is not written if it still fails. Its call and cost go to
runs/<id>/present/. Cell illustrations: NIAID NIH BioArt Source (public domain; assets/bioart/CREDITS.json).
"""

from __future__ import annotations

import base64
import html
import json
import re
from pathlib import Path

from mechanist import agents, cellfig

ASSETS = Path(__file__).resolve().parents[2] / "assets" / "bioart"
NUM_RE = re.compile(r"(?<![\w.])-?\d+(?:[.,]\d+)?")
ID_RE = re.compile(r"\b(?:[RFQSD]\d{2,}|PMID[:\s]*\d+)\b")
STATUS = ("best supported", "possible", "not supported", "untested")
VERDICTS = ("explains", "partly", "not explained", "not addressed")
EDGE = ("observed", "likely", "analogy", "speculative")
KINDS = ("root", "observed", "likely", "speculative", "outcome")
# cell type words -> icon (assets/bioart/<key>.png); the first match wins
ICONS = [(r"dying|apoptotic|necro", "dying_neutrophil"), (r"activated neutrophil", "activated_neutrophil"),
         (r"neutrophil|granulocyte", "neutrophil"), (r"macrophage|monocyte|myeloid", "macrophage"),
         (r"fibroblast|stroma", "fibroblast"), (r"plasma", "plasma_cell"), (r"\bb[ -]?cell|lymphoid niche", "b_cell"),
         (r"\bt[ -]?cell|t/nk|\bnk\b", "t_cell"), (r"endotheli|vessel|vascul|pericyte", "blood_vessel"),
         (r"antibod|immunoglobulin", "antibody"), (r"complement", "complement_c1")]


# ---- inputs -------------------------------------------------------------------------------------------------------

def load_run(run: Path) -> tuple[dict, dict]:
    """(ctx: stage name -> output, results: id -> tool result) from the run and, for a one-stage rerun, its source."""
    dirs, cur = [], run  # the run, then each folder it was rerun from in turn
    while cur is not None and cur not in dirs:
        dirs.append(cur)
        nxt = None
        ev = cur / "events.jsonl"
        if ev.exists():
            for line in ev.read_text().splitlines():
                r = json.loads(line)
                if r.get("action") == "inputs" and r.get("source"):
                    src = Path(r["source"])
                    nxt = src if src.is_absolute() else cur.parent.parent / src
        cur = nxt
    ctx, results = {}, {}
    for d in dirs:
        for p in sorted(d.glob("*.json")):
            if p.stem in ("state", "humanized", "interpretation") or p.stem in ctx:
                continue
            try:
                ctx[p.stem] = json.loads(p.read_text())
            except json.JSONDecodeError:
                continue
    for v in ctx.values():
        if isinstance(v, dict):
            for k in ("results", "final_results"):
                for r in v.get(k) or []:
                    if isinstance(r, dict) and r.get("id"):
                        results.setdefault(r["id"], r)
    return ctx, results


def panel_data(results: dict) -> dict[tuple[str, str, str], dict]:
    """(result, gene, lineage) -> {sample: % of cells expressing} for every per-sample expression result."""
    out = {}
    for rid, r in results.items():
        d = r.get("data")
        if not (r.get("ok", True) and isinstance(d, dict)):
            continue
        for g, by in d.items():
            if not isinstance(by, dict):
                continue
            for lin, vals in by.items():
                if isinstance(vals, dict) and vals and all(v is None or isinstance(v, (int, float)) for v in vals.values()):
                    out[(rid, g, lin)] = vals
    return out


def holds_in(vals: dict, gs: list[tuple[str, list[str]]]) -> list[str]:
    """The disease samples that lie beyond every comparator sample, in the direction the disease samples share."""
    case = [x for x in gs[0][1] if vals.get(x) is not None] if gs else []
    other = [vals[x] for _, smp in gs[1:] for x in smp if vals.get(x) is not None]
    if not case or not other:
        return []
    hi, lo = max(other), min(other)
    up = [x for x in case if vals[x] > hi]
    dn = [x for x in case if vals[x] < lo]
    return up if len(up) >= len(dn) else dn


def _pattern(vals: dict, gs) -> str:
    n = len([x for x in gs[0][1] if vals.get(x) is not None]) if gs else 0
    h = holds_in(vals, gs)
    if not h:
        return "overlaps the comparators"
    return "holds in every disease sample" if len(h) == n else f"holds in {len(h)} of {n} disease samples ({', '.join(h)})"


def cited_ids(circuits: list[dict]) -> set[str]:
    return set(ID_RE.findall(json.dumps(circuits)))


def groups(ctx: dict) -> list[tuple[str, list[str]]]:
    g = (ctx.get("load") or {}).get("groups") or {}
    ref = (ctx.get("load") or {}).get("reference_group")
    order = sorted(g, key=lambda k: (k == ref, list(g).index(k)))  # disease first, reference (control) last
    return [(k, list(g[k])) for k in order]


# ---- the model call and its check ---------------------------------------------------------------------------------

def _nums(text: str) -> set[str]:
    return {n.replace(",", ".") for n in NUM_RE.findall(text)}


def check(page: dict, source: str, panels: set, n_circuits: int, item_ids: set[str],
          partial: dict | None = None) -> list[str]:
    errs = []
    partial = partial or {}
    if not isinstance(page, dict):
        return ["the reply is not a JSON object"]
    texts = [page.get("question", ""), page.get("answer", "")]
    texts += [p.get("text", "") for p in page.get("points") or []]
    for c in page.get("circuits") or []:
        texts += [c.get("short_title", ""), c.get("one_line", "")]
        texts += [n.get("label", "") + " " + n.get("sub", "") for n in c.get("nodes") or []]
        texts += [k.get("title", "") + " " + k.get("why", "") for k in c.get("cards") or []]
    texts += [f.get("feature", "") for f in page.get("fit") or []]
    texts += [" ".join(str(t.get(k, "")) for k in ("test", "expects", "decides")) for t in page.get("tests") or []]
    texts += [str(x) for x in page.get("checked") or []]
    new = _nums(" ".join(texts)) - _nums(source)
    if new:
        errs.append(f"numbers the circuits do not have: {', '.join(sorted(new)[:6])}")
    bad_ids = set(ID_RE.findall(" ".join(texts))) - set(ID_RE.findall(source))
    if bad_ids:
        errs.append(f"ids the circuits do not have: {', '.join(sorted(bad_ids)[:6])}")
    cs = page.get("circuits") or []
    if len(cs) != n_circuits:
        errs.append(f"{len(cs)} circuits given; the run has {n_circuits}: give every one, in rank order")
    for c in cs:
        nodes = {n.get("id") for n in c.get("nodes") or []}
        cards = {k.get("id") for k in c.get("cards") or []}
        if c.get("status") not in STATUS:
            errs.append(f"circuit {c.get('rank')}: status must be one of {', '.join(STATUS)}")
        if not 3 <= len(nodes) <= 12:
            errs.append(f"circuit {c.get('rank')}: give 4-10 diagram nodes")
        for n in c.get("nodes") or []:
            if n.get("kind") not in KINDS:
                errs.append(f"circuit {c.get('rank')} node {n.get('id')}: kind must be one of {', '.join(KINDS)}")
            for e in n.get("evidence") or []:
                if e not in cards:
                    errs.append(f"circuit {c.get('rank')} node {n.get('id')}: evidence {e} is not a card id")
        for e in c.get("edges") or []:
            if e.get("from") not in nodes or e.get("to") not in nodes:
                errs.append(f"circuit {c.get('rank')}: edge {e.get('from')}->{e.get('to')} names a missing node")
            if e.get("status") not in EDGE:
                errs.append(f"circuit {c.get('rank')}: edge status must be one of {', '.join(EDGE)}")
        for k in c.get("cards") or []:
            for p in k.get("panels") or []:
                if (p.get("result"), p.get("gene"), p.get("lineage")) not in panels:
                    errs.append(f"card {k.get('id')}: panel {p.get('result')} | {p.get('gene')} | {p.get('lineage')} "
                                "is not in AVAILABLE PANELS")
    for c in cs:
        for k in c.get("cards") or []:
            pats = [partial.get((p.get("result"), p.get("gene"), p.get("lineage"))) for p in k.get("panels") or []]
            if pats and all(x for x in pats) and all(x.startswith("holds in") and "every" not in x for x in pats):
                text = (k.get("title", "") + " " + k.get("why", "")).lower()
                names = {n.lower() for x in pats for n in re.findall(r"\(([^)]*)\)", x)[0].split(", ")}
                if not (re.search(r"\b(one|1) of (two|2|three|3)\b|only in|in one\b", text) or any(re.search(r"\b" + re.escape(n) + r"\b", text) for n in names)):
                    errs.append(f"card {k.get('id')}: its panels hold in only some disease samples ({pats[0]}); say so "
                                "in the title or why")
    for f in page.get("fit") or []:
        if len(f.get("verdicts") or []) != len(cs) or any(v not in VERDICTS for v in f.get("verdicts") or []):
            errs.append(f"fit row '{f.get('feature', '')[:40]}': give one of {', '.join(VERDICTS)} per circuit")
        bad = set(f.get("items") or []) - item_ids
        if bad:
            errs.append(f"fit row '{f.get('feature', '')[:40]}': unknown dossier items {', '.join(sorted(bad))}")
    return errs


def write_page(run: Path, llm=None) -> dict:
    """Ask for the short page (cached), check it, retry once; returns {"page": ..., "errors": [...]}."""
    from mechanist.llm import LLMConfig, LoggedLLM, make_backend, parse_json
    from mechanist.runner import RunConfig

    ctx, results = load_run(run)
    cfg = RunConfig.from_yaml(run / "config.yaml")
    circ = ctx["circuits"]
    circuits = circ["circuits"]
    verdict = ((circ.get("reviews") or [{}])[-1] or {}).get("verdict", "unknown")
    pdata = panel_data(results)
    want = cited_ids(circuits)
    avail = sorted(k for k in pdata if k[0] in want)
    items = agents.phenotype_items((ctx.get("dossier") or {}).get("dossier") or {})
    gs = groups(ctx)
    pats = {k: _pattern(pdata[k], gs) for k in avail}
    source = json.dumps({"circuits": circuits, "ranking": circ.get("ranking", "")}, ensure_ascii=False)
    if llm is None:
        lc = LLMConfig.model_validate({**cfg.llm.model_dump(), "effort": "medium"})
        (run / "present").mkdir(exist_ok=True)
        llm = LoggedLLM(make_backend(lc), run / "present")
    user = agents.PRESENT_USER.format(
        disease=cfg.disease, design=("index patient: one patient with an uncertain diagnosis, compared with the closest "
                                     "known diseases" if cfg.design == "index_patient" else "cohort"),
        comparators=", ".join(g for g, _ in gs[1:]) or "(none)",
        question=((ctx.get("load") or {}).get("question") or "(none given)").replace("\n", " "),
        phenotype=cfg.disease_phenotype or "(not given)", groups="; ".join(f"{g}: {', '.join(s)}" for g, s in gs),
        verdict=verdict, circuits=json.dumps(circuits, ensure_ascii=False, indent=0)[:120000],
        items="\n".join(f"{i} [{sec}] {t[:300]}" for i, sec, t in items),
        panels="\n".join(f"{r} | {g} | {lin} | {pats[(r, g, lin)]}" for r, g, lin in avail[:600]), retry="")
    page, errs = None, ["no reply"]
    for attempt in range(2):
        try:
            page = parse_json(llm.complete(f"present:{attempt}", agents.PRESENT_SYSTEM, user))
        except Exception as exc:  # noqa: BLE001 - a bad reply is retried once, then reported
            page, errs = None, [f"unreadable reply: {str(exc)[:200]}"]
        else:
            errs = check(page, source, set(avail), len(circuits), {i for i, _, _ in items}, pats)
        if not errs:
            break
        user = user.split("\n\nYOUR PREVIOUS PAGE")[0] + ("\n\nYOUR PREVIOUS PAGE WAS NOT USED because: " + "; ".join(errs)
                                                          + ". Fix exactly these and return the whole page again.")
    if isinstance(page, dict):  # the consensus support of each circuit, from the run itself (not the writer)
        by_rank = {int(c.get("rank") or i): c for i, c in enumerate(circuits, 1)}
        for c in page.get("circuits") or []:
            c["support"] = (by_rank.get(int(c.get("rank") or 0)) or {}).get("support", "")
        page["leads"] = [x for x in circ.get("leads") or [] if isinstance(x, dict)]
    return {"page": page, "errors": errs, "panels": {f"{r}|{g}|{lin}": v for (r, g, lin), v in pdata.items()
                                                     if r in want}}


# ---- shared look --------------------------------------------------------------------------------------------------

CSS = """
:root{--bg:#f7f6f2;--surface:#fcfcfb;--surface-2:#f0efec;--border:#e3e1da;--ink:#0b0b0b;--ink-2:#52514e;--muted:#8a887f;
--accent:#3d4f1f;--accent-soft:#e6ebd9;--warn-bg:#fff4d6;--warn-bd:#a8801f;--warn-ink:#8a5a00;--good:#2f6b2f;
--grp0:#c4561a;--grp1:#178a6b;--grp2:#6b4fc9;--grp3:#9a7400;--grp4:#b5487a;--grp5:#4d7a24;--node:#ffffff}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#121211;--surface:#1a1a19;--surface-2:#242422;
--border:#34342f;--ink:#fff;--ink-2:#c3c2b7;--muted:#8f8e86;--accent:#b7c98f;--accent-soft:#2c3320;--warn-bg:#3a3017;
--warn-bd:#a8862f;--warn-ink:#e0b04f;--good:#8fc28a;--grp0:#e07a45;--grp1:#3fbf98;--grp2:#9d86f0;--grp3:#d6a83a;
--grp4:#e07aa6;--grp5:#86b85a;--node:#242422;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#121211;--surface:#1a1a19;--surface-2:#242422;--border:#34342f;--ink:#fff;--ink-2:#c3c2b7;
--muted:#8f8e86;--accent:#b7c98f;--accent-soft:#2c3320;--warn-bg:#3a3017;--warn-bd:#a8862f;--warn-ink:#e0b04f;
--good:#8fc28a;--grp0:#e07a45;--grp1:#3fbf98;--grp2:#9d86f0;--grp3:#d6a83a;--grp4:#e07aa6;--grp5:#86b85a;--node:#242422;
color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
a{color:inherit}
.layout{display:grid;grid-template-columns:230px minmax(0,1fr);min-height:100vh}
nav.side{position:sticky;top:0;height:100vh;overflow-y:auto;padding:16px 12px;border-right:1px solid var(--border);background:var(--bg);display:flex;flex-direction:column;gap:2px}
nav.side a{display:block;padding:7px 10px;border-radius:6px;text-decoration:none;color:var(--ink-2)}
nav.side a:hover,nav.side a.active{background:var(--surface-2);color:var(--ink)}
nav.side a.page{color:var(--ink);font-weight:600}
nav.side a.sub{padding:5px 10px 5px 22px;font-size:13px}
nav.side .sep{height:1px;background:var(--border);margin:10px 4px}
nav.side .themeseg{margin:6px 12px}
.brand{font-weight:700;font-size:15px;padding:4px 8px 2px}
.brand small{display:block;font-weight:400;color:var(--muted);font-size:12px}
main{min-width:0}
.topbar{position:sticky;top:0;z-index:20;background:var(--surface);border-bottom:1px solid var(--border);padding:10px 24px;display:flex;gap:18px;align-items:center;flex-wrap:wrap}
.viewtabs{display:inline-flex;gap:4px;align-self:center;flex-wrap:wrap}
.viewtabs a{padding:4px 12px;border-radius:14px;border:1px solid var(--border);background:var(--surface);text-decoration:none;color:var(--ink);font-size:13px}
.viewtabs a[aria-current="page"]{background:var(--accent);border-color:var(--accent);color:var(--surface)}
.viewtabs a:focus-visible,nav.side a:focus-visible,summary:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
.wrap{max-width:1200px;padding-left:max(24px,env(safe-area-inset-left,0px));padding-right:max(24px,env(safe-area-inset-right,0px));padding-block:0 48px}
h1{font-size:24px;line-height:1.2;margin:26px 0 4px;text-wrap:balance}
h2{font-size:20px;line-height:1.25;margin:0;text-wrap:balance}
h3{font-size:15px;margin:0;text-wrap:balance}
.sub{color:var(--ink-2);margin:4px 0 14px;max-width:72ch}
section{border-top:1px solid var(--border);padding-top:22px;margin-top:26px;scroll-margin-top:70px}
.note{background:var(--warn-bg);border:1px solid var(--warn-bd);border-radius:8px;padding:10px 14px;font-size:13px;line-height:1.5;max-width:860px;margin-top:12px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-top:16px}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:12px 14px}
.tile b{display:block;font-size:24px;font-weight:650;line-height:1.1;font-variant-numeric:tabular-nums}
.tile span{font-size:12px;color:var(--ink-2)}
.key{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px;vertical-align:-1px}
.answer{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:28px;align-items:start;margin-top:4px}
.ahead{font-size:20px;font-weight:700;line-height:1.35;margin:0;text-wrap:balance}
.points{list-style:none;margin:0;padding:0;display:grid;gap:10px}
.points li{line-height:1.5}.points b{font-weight:600;display:block;font-size:12px;color:var(--ink-2)}
.card{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:14px 16px;min-width:0}
.cardhead{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap;margin-bottom:6px}
.badge{display:inline-block;font-size:11px;padding:1px 8px;border-radius:14px;border:1px solid var(--border);color:var(--ink-2);white-space:nowrap}
.badge.good{border-color:var(--good);color:var(--good)}
.badge.warn{background:var(--warn-bg);border-color:var(--warn-bd);color:var(--warn-ink)}
.badge.lead{background:var(--accent-soft);border-color:var(--accent);color:var(--ink)}
.diagram{overflow-x:auto}
.diagram svg{display:block;width:100%;height:auto}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:var(--ink-2);margin-top:8px;align-items:center}
.legend svg{vertical-align:middle;margin-right:4px}
.ev{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}
.evc{scroll-margin-top:80px}.evc:target{outline:2px solid var(--accent);outline-offset:2px}
.num{display:inline-flex;width:20px;height:20px;border-radius:50%;background:var(--ink);color:var(--surface);font-size:11px;font-weight:700;align-items:center;justify-content:center;flex:none}
.evc p{margin:4px 0 8px;font-size:13px;line-height:1.5;color:var(--ink-2)}
.strips{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,220px));gap:8px}
.strip{margin:0}.strip figcaption{font-size:12px;font-weight:600;margin-bottom:2px}.strip figcaption span{font-weight:400;color:var(--muted)}
.strip svg{width:100%;height:auto;display:block}
.strip .vd{font-size:11px;color:var(--ink-2);line-height:1.35;margin-top:2px}.strip .vd b{color:var(--good);font-weight:600}
.chips{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}
.tablewrap{overflow-x:auto;border:1px solid var(--border);border-radius:6px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:7px 10px;border-bottom:1px solid var(--border);text-align:left;vertical-align:top}
th{font-size:12px;font-weight:600;color:var(--ink-2);background:var(--surface-2)}
tr:last-child td{border-bottom:0}
td.c,th.c{text-align:center}
.altlist{display:grid;gap:12px}
table.fit td.fitcell{color:var(--accent);font-size:15px;width:44px}table.fit td.fitcell.on{background:var(--accent-soft)}
table.kv th{width:210px;background:var(--surface-2);vertical-align:top}table.kv ul{margin:0;padding-left:18px}
.forag{display:grid;grid-template-columns:1fr 1fr;gap:14px;font-size:13px}.forag ul{padding-left:18px;margin:4px 0}
.gbar{display:grid;grid-template-columns:minmax(0,1.3fr) minmax(80px,1fr) 52px;gap:8px;align-items:center;font-size:12.5px;margin:4px 0}
.gt{display:block;height:10px;background:var(--surface-2);border-radius:5px;overflow:hidden}.gt i{display:block;height:100%;background:var(--accent)}
.sdot{display:inline-block;width:10px;height:10px;border-radius:50%;margin:0 1px;background:var(--muted);vertical-align:-1px}
.sdot.supported{background:var(--good)}.sdot.partial{background:var(--warn-bd)}.sdot.unsupported,.sdot.contradicted{background:var(--grp0)}
.altrow{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:22px;align-items:center}
.altevs{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px;margin-top:12px}
.altev{background:var(--bg)}.altev p{margin:0 0 6px;font-size:13px}
.altev .strips,.evc .strips{grid-template-columns:repeat(2,minmax(0,1fr))}
.alts{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}
.alts p{margin:6px 0 0;font-size:13px;line-height:1.5;color:var(--ink-2)}
.checked{list-style:none;padding:0;margin:8px 0 0;display:grid;gap:6px;font-size:13px}
.checked li::before{content:"✓ ";color:var(--good);font-weight:700}
details{margin-top:10px;font-size:13px;color:var(--ink-2)}
details summary{cursor:pointer}
.small{font-size:12px;color:var(--muted)}
.more{display:inline-block;margin-top:10px;font-size:13px}
.step{border-top:1px solid var(--border);padding:14px 0;display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:20px}
.step:first-of-type{border-top:0}
.step h4{font-size:14px;margin:0 0 4px;display:flex;gap:8px;align-items:baseline;flex-wrap:wrap}
.step p{margin:4px 0;line-height:1.55;font-size:13px}
.prose{max-width:80ch;line-height:1.6}.prose p{margin:0 0 10px}
.ids{font-size:11px;color:var(--muted);overflow-wrap:anywhere}
.ids a{color:var(--ink-2)}
pre.q{white-space:pre-wrap;font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:var(--surface-2);border-radius:6px;padding:8px 10px;margin:6px 0 0;max-height:320px;overflow:auto}
.res{scroll-margin-top:80px;border-top:1px solid var(--border);padding:10px 0}
.res:target{background:var(--accent-soft)}
.eyebrow{font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:0 0 4px}
.lede{font-size:16px;line-height:1.55;margin:10px 0;max-width:62ch}
.cfig{display:grid;grid-template-columns:minmax(0,3fr) minmax(0,2fr);gap:22px;align-items:start}
.cellfig{display:block;width:100%;height:auto}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}
a.thumb{display:block;text-decoration:none;color:inherit;transition:border-color .15s}
a.thumb:hover{border-color:var(--accent)}a.thumb:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
a.thumb h3{font-size:13.5px;line-height:1.35}
.stepgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:12px;margin-top:16px}
.stepc h3{font-size:13.5px;line-height:1.35}
.stepc .strips{grid-template-columns:repeat(2,minmax(0,1fr))}
.gene{display:inline-block;font:600 11.5px/1.6 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;padding:0 7px;border-radius:5px;background:var(--surface-2);color:var(--ink)}
.leads{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:12px}
.leadc{display:flex;gap:12px;align-items:flex-start}.leadc p{font-size:13.5px;line-height:1.5}
.licon{width:52px;height:52px;flex:none}
.themeseg{display:inline-flex;width:max-content;border:1px solid var(--border);border-radius:6px;overflow:hidden}
.themeseg button{border:0;background:var(--surface);color:var(--ink-2);padding:4px 9px;cursor:pointer;display:grid;place-items:center}
.themeseg button[aria-pressed="true"]{background:var(--accent);color:var(--surface)}
.themeseg button:focus-visible{outline:2px solid var(--accent);outline-offset:-2px}
.themeseg svg{width:14px;height:14px;fill:none;stroke:currentColor;stroke-width:1.6;stroke-linecap:round;stroke-linejoin:round}
@media (max-width:900px){.answer,.step,.cfig,.altrow{grid-template-columns:1fr}.layout{grid-template-columns:1fr}
nav.side{position:static;height:auto;flex-direction:row;flex-wrap:wrap;border-right:0;border-bottom:1px solid var(--border)}
nav.side .sep,.brand small,nav.side a.sub{display:none}.topbar{position:static;padding:10px 16px}
.wrap{padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}}
"""

STRIP_JS = """
function verdict(p,G){const g=G.map(x=>x[1].map(s=>p.v[s]).filter(v=>v!=null));const a=g[0];if(!a||!a.length)return [];
 const out=[];for(let i=1;i<G.length;i++){if(!g[i].length)continue;const n=G[i][0];
  if(Math.min(...a)>Math.max(...g[i]))out.push([1,"every "+G[0][0]+" above every "+n]);
  else if(Math.max(...a)<Math.min(...g[i]))out.push([1,"every "+G[0][0]+" below every "+n]);
  else out.push([0,G[0][0]+" overlaps "+n])}return out}
function strip(p,G){const W=160,H=118,L=26,B=18,T=6;const vals=Object.values(p.v).filter(x=>x!=null);let mx=Math.max(5,...vals);
 const step=mx>40?20:mx>20?10:mx>10?5:2;mx=Math.ceil(mx/step)*step;const y=v=>T+(H-T-B)*(1-v/mx);const cw=(W-L)/G.length;
 let s=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${p.gene} in ${p.lin}: % of cells expressing, one dot per sample">`;
 for(let t=0;t<=mx;t+=step){s+=`<line x1="${L}" x2="${W}" y1="${y(t)}" y2="${y(t)}" stroke="var(--border)"/><text x="${L-4}" y="${y(t)+3}" font-size="9" text-anchor="end" fill="var(--muted)">${t}</text>`}
 G.forEach(([n,ss],i)=>{const cx=L+cw*(i+.5);const vs=ss.map(x=>p.v[x]).filter(x=>x!=null);
  if(vs.length){const m=vs.reduce((a,b)=>a+b,0)/vs.length;s+=`<line x1="${cx-14}" x2="${cx+14}" y1="${y(m)}" y2="${y(m)}" stroke="var(--ink-2)" stroke-width="1.5"/>`}
  ss.forEach((x,j)=>{const v=p.v[x];if(v==null)return;const dx=(j-(ss.length-1)/2)*Math.min(5,24/ss.length);
   s+=`<circle cx="${cx+dx}" cy="${y(v)}" r="3.6" fill="var(--grp${i})" stroke="var(--surface)" stroke-width=".8"><title>${x}: ${v.toFixed(1)}%</title></circle>`});
  s+=`<text x="${cx}" y="${H-5}" font-size="9.5" text-anchor="middle" fill="var(--ink-2)">${n.length>9?n.slice(0,8)+"…":n}</text>`});
 return s+"</svg>"}
function panel(p,G){return `<figure class="strip"><figcaption>${p.gene} <span>${p.lin}</span></figcaption>${strip(p,G)}<div class="vd">${verdict(p,G).map(v=>v[0]?`<b>${v[1]}</b>`:v[1]).join("<br>")}</div></figure>`}
document.querySelectorAll("[data-panels]").forEach(el=>{const ps=JSON.parse(el.dataset.panels);el.innerHTML=ps.map(k=>{const v=PANELS[k];if(!v)return "";const [r,g,l]=k.split("|");return panel({gene:g,lin:l,v},GROUPS)}).join("")});
"""


def esc(x) -> str:
    return html.escape(str(x if x is not None else ""))


def _icon(cell: str) -> str:
    for pat, key in ICONS:
        if cell and re.search(pat, cell, re.IGNORECASE) and (ASSETS / f"{key}.png").exists():
            return "data:image/png;base64," + base64.b64encode((ASSETS / f"{key}.png").read_bytes()).decode()
    return ""


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = str(text).split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    return lines + ([cur] if cur else [])


# ---- the circuit diagram: layered left to right, loops drawn as arcs ------------------------------------------------

def layout(nodes: list[dict], edges: list[dict]) -> dict[str, tuple[int, int]]:
    ids = [n["id"] for n in nodes]
    fwd = [(e["from"], e["to"]) for e in edges if not e.get("loop") and e["from"] in ids and e["to"] in ids]
    layer = {i: 0 for i in ids}
    for _ in range(len(ids)):  # longest path from the roots; a cycle is cut by the bound
        for a, b in fwd:
            if layer[b] < layer[a] + 1:
                layer[b] = min(layer[a] + 1, len(ids) - 1)
    last = max(layer.values(), default=0)
    for n in nodes:
        if n.get("kind") == "outcome":
            layer[n["id"]] = max(last, layer[n["id"]])
    cols: dict[int, list[str]] = {}
    for i in ids:
        cols.setdefault(layer[i], []).append(i)
    keys = sorted(cols)
    pos = {}
    for li in keys:  # order each column by the mean row of its parents
        if li:
            prev = {i: k for k, i in enumerate(cols[keys[keys.index(li) - 1]])} if keys.index(li) else {}
            cols[li].sort(key=lambda i: (sum(prev.get(a, 0) for a, b in fwd if b == i) /
                                         max(1, sum(1 for a, b in fwd if b == i and a in prev))))
        for r, i in enumerate(cols[li]):
            pos[i] = (keys.index(li), r)
    return pos


def diagram(c: dict, cards: dict[str, int]) -> str:
    nodes, edges = c.get("nodes") or [], c.get("edges") or []
    if not nodes:
        return ""
    pos = layout(nodes, edges)
    MAXC = 4  # columns per band; a longer chain wraps onto the next band, read left to right, top to bottom
    rows: dict[int, int] = {}
    for x, y in pos.values():
        rows[x] = max(rows.get(x, 0), y + 1)
    ncols = max(pos[i][0] for i in pos) + 1
    nband = (ncols + MAXC - 1) // MAXC
    NW, NH, GX, GY, TOP, BGAP = 214, 82, 56, 26, 40, 40
    bh = {b: max(rows[c] for c in rows if c // MAXC == b) for b in range(nband)}
    W = min(ncols, MAXC) * NW + (min(ncols, MAXC) - 1) * GX + 20
    btop, y0 = {}, TOP
    for b in range(nband):
        btop[b] = y0
        y0 += bh[b] * NH + (bh[b] - 1) * GY + BGAP + TOP
    H = y0 - BGAP - TOP + 20
    xy = {}
    for n in nodes:
        cx, r = pos[n["id"]]
        b, k = divmod(cx, MAXC)
        off = (bh[b] - rows[cx]) * (NH + GY) / 2  # centre short columns
        xy[n["id"]] = (10 + k * (NW + GX), btop[b] + off + r * (NH + GY))
    style = {"observed": 'stroke-width="3"', "likely": 'stroke-width="1.6"',
             "analogy": 'stroke-width="1.6" stroke-dasharray="2 4"', "speculative": 'stroke-width="1.6" stroke-dasharray="6 4"'}
    s = [f'<svg viewBox="0 0 {W} {H}" style="min-width:{min(W, 760)}px" role="img" aria-label="{esc(c.get("short_title"))}: circuit diagram">',
         ('<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
          'orient="auto-start-reverse"><path d="M0,0L10,5L0,10z" fill="var(--ink-2)"/></marker></defs>')]
    for e in edges:
        if e.get("from") not in xy or e.get("to") not in xy:
            continue
        (ax, ay), (bx, by) = xy[e["from"]], xy[e["to"]]
        st = style.get(e.get("status"), style["likely"])
        fb, tb = pos[e["from"]][0] // MAXC, pos[e["to"]][0] // MAXC
        if tb > fb:  # onto the next band: from the bottom of the source to the top of the target
            x1, y1, x2, y2 = ax + NW / 2, ay + NH, bx + NW / 2, by
            d = f"M{x1},{y1} C{x1},{y1 + 60} {x2},{y2 - 60} {x2},{y2}"
        elif tb < fb:  # back to an earlier band (a loop): up the left margin
            x1, y1, x2, y2 = ax, ay + NH / 2, bx + 24, by + NH
            d = f"M{x1},{y1} C{x1 - 40},{y1} {x2},{y2 + 70} {x2},{y2}"
        elif e.get("loop") or pos[e["to"]][0] <= pos[e["from"]][0]:
            x1, y1, x2, y2 = ax + NW / 2, ay, bx + NW / 2, by
            top = min(y1, y2) - 34
            d = f"M{x1},{y1} C{x1},{top} {x2},{top} {x2},{y2}"
        else:
            x1, y1, x2, y2 = ax + NW, ay + NH / 2, bx, by + NH / 2
            mx = (x1 + x2) / 2
            d = f"M{x1},{y1} C{mx},{y1} {mx},{y2} {x2},{y2}"
        s.append(f'<path d="{d}" fill="none" stroke="var(--ink-2)" {st} marker-end="url(#ah)"><title>{esc(e.get("status"))}</title></path>')
    for n in nodes:
        x, y = xy[n["id"]]
        kind = n.get("kind", "likely")
        fill = "var(--accent-soft)" if kind == "outcome" else "var(--node)"
        strong = kind in ("observed", "outcome")
        dash = ' stroke-dasharray="6 4"' if kind in ("speculative", "root") else ""
        s.append(f'<rect x="{x}" y="{y}" width="{NW}" height="{NH}" rx="8" fill="{fill}" stroke="{"var(--ink)" if strong else "var(--ink-2)"}" '
                 f'stroke-width="{2 if strong else 1}"{dash}/>')
        icon = _icon(n.get("cell", ""))
        tx = x + 12
        if icon:
            s.append(f'<image href="{icon}" x="{x + 8}" y="{y + 10}" width="40" height="40" preserveAspectRatio="xMidYMid meet"/>')
            tx = x + 56
        width = 22 if icon else 28
        for k, line in enumerate(_wrap(n.get("label", ""), width - 4)[:2]):
            s.append(f'<text x="{tx}" y="{y + 22 + k * 16}" font-size="14" font-weight="700" fill="var(--ink)">{esc(line)}</text>')
        nl = min(2, len(_wrap(n.get("label", ""), width - 4)))
        for k, line in enumerate(_wrap(n.get("sub", ""), width - 1)[:2 if nl == 1 else 1]):
            s.append(f'<text x="{tx}" y="{y + 22 + nl * 16 + 2 + k * 14}" font-size="12" fill="var(--ink-2)">{esc(line)}</text>')
        word = {"root": "root · speculative", "outcome": "disease feature"}.get(kind, kind)
        s.append(f'<text x="{x + 12}" y="{y + NH - 9}" font-size="10.5" fill="var(--muted)">{esc(word)}</text>')
        evs = [cards[e] for e in n.get("evidence") or [] if e in cards]
        for k, num in enumerate(reversed(evs)):
            cx, cy = x + NW - 16 - k * 24, y + NH - 14
            s.append(f'<a href="#ev{num}" aria-label="evidence {num}"><circle cx="{cx}" cy="{cy}" r="9" fill="var(--ink)"/>'
                     f'<text x="{cx}" y="{cy + 4}" font-size="11" font-weight="700" text-anchor="middle" fill="var(--surface)">{num}</text></a>')
    return "\n".join(s + ["</svg>"])


LEGEND = """<div class="legend">
<span><svg width="34" height="8" aria-hidden="true"><line x1="0" y1="4" x2="34" y2="4" stroke="currentColor" stroke-width="3"/></svg>observed in this data</span>
<span><svg width="34" height="8" aria-hidden="true"><line x1="0" y1="4" x2="34" y2="4" stroke="currentColor" stroke-width="1.5"/></svg>likely, not tested</span>
<span><svg width="34" height="8" aria-hidden="true"><line x1="0" y1="4" x2="34" y2="4" stroke="currentColor" stroke-width="1.5" stroke-dasharray="2 4"/></svg>published elsewhere</span>
<span><svg width="34" height="8" aria-hidden="true"><line x1="0" y1="4" x2="34" y2="4" stroke="currentColor" stroke-width="1.5" stroke-dasharray="6 4"/></svg>speculative</span>
<span><span class="num" style="width:16px;height:16px;font-size:10px">1</span> evidence below</span></div>"""

NOTICE = ("<b>Mechanist interpretation: a language-model reading of this study.</b> It proposes mechanisms for hypothesis "
          "generation; validate each point against the Results tab before relying on it, and read it as one helpful "
          "viewpoint. {small}")


# The portal's colour theme, shared with the rest of the portal: auto | light | dark, saved as localStorage "clp:theme"
# (the results report and the study form read and write the same key), applied in <head> before the page draws.
THEME_HEAD = ("<script>/* the CL Analysis portal's theme (shared with the portal and the study form): auto | light | dark */ "
              "try{var t=localStorage.getItem(\"clp:theme\");if(t==='\"light\"'||t==='\"dark\"')"
              "document.documentElement.dataset.theme=JSON.parse(t)}catch(e){}</script>")
THEME_SEG = """<div class="themeseg" id="themeSeg" role="group" aria-label="Colour theme">
<button type="button" data-theme="auto" title="Auto theme" aria-label="Auto theme"><svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="5.4"/><path d="M8 2.6a5.4 5.4 0 010 10.8z" fill="currentColor" stroke="none"/></svg></button>
<button type="button" data-theme="light" title="Light theme" aria-label="Light theme"><svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="2.8"/><path d="M8 1.8v1.4M8 12.8v1.4M1.8 8h1.4M12.8 8h1.4M3.6 3.6l1 1M11.4 11.4l1 1M3.6 12.4l1-1M11.4 4.6l1-1"/></svg></button>
<button type="button" data-theme="dark" title="Dark theme" aria-label="Dark theme"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M12.8 9.6A5.2 5.2 0 016.4 3.2a5.2 5.2 0 106.4 6.4z"/></svg></button></div>"""
THEME_JS = """<script>(function(){const seg=[...document.querySelectorAll("#themeSeg button")];
const themeNow=()=>{try{const v=JSON.parse(localStorage.getItem("clp:theme"));return v==="light"||v==="dark"?v:"auto"}catch(e){return "auto"}};
function applyTheme(t,save){const r=document.documentElement;if(t==="auto")delete r.dataset.theme;else r.dataset.theme=t;
 if(save){try{t==="auto"?localStorage.removeItem("clp:theme"):localStorage.setItem("clp:theme",JSON.stringify(t))}catch(e){}}
 seg.forEach(b=>b.setAttribute("aria-pressed",String(b.dataset.theme===t)))}
seg.forEach(b=>b.onclick=()=>applyTheme(b.dataset.theme,true));applyTheme(themeNow(),false);
addEventListener("storage",e=>{if(e.key==="clp:theme")applyTheme(themeNow(),false)})})()</script>"""


def _head(title: str, study: str, sub: str, tabs: list[tuple[str, str, bool]],
          side: list[tuple[str, str]] | None = None, page: str = "summary") -> str:
    """Open the page in the Results tab's shell: a sidebar (study, this page's sections, the other Mechanist page, all
    files, the portal's theme switch) and a main column whose top bar holds the study views."""
    nav = "".join(f'<a href="{esc(h)}"{" aria-current=\"page\"" if cur else ""}>{esc(t)}</a>' for t, h, cur in tabs)
    pid = (sub or "").split(" · ")[0] or study
    subs = "".join(f'<a class="sub" href="{esc(h)}">{esc(t)}</a>' for t, h in side or [])
    pages = "".join(f'<a class="page" href="{h}{"#top" if key == page else ""}">{esc(t)}</a>' + (subs if key == page else "")
                    for key, h, t in PAGES)
    files = next((h for t, h, _ in tabs if t == "Results"), "")
    files = files.rsplit("/", 1)[0] + "/files.html" if files.startswith("..") else ""
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, '
            f'initial-scale=1, viewport-fit=cover"><meta name="color-scheme" content="light dark">{THEME_HEAD}'
            f'<title>{esc(title)}</title><style>{CSS}</style></head><body><div class="layout">'
            f'<nav class="side" aria-label="Sections"><div class="brand">{esc(pid)}<small>{esc(study)}</small></div>'
            f'<div class="sep"></div>{pages}<div class="sep"></div>'
            + (f'<a href="{esc(files)}">All files ↗</a>' if files else "")
            + f'<a href="#" id="dlPage" data-name="{esc(pid)}_mechanist_{"summary" if page == "summary" else "full_mechanism"}.html"'
            f' title="Save this page as one HTML file that opens offline, exactly as shown">Download this page ↓</a>'
            + f'{THEME_SEG}</nav><main><div class="topbar"><nav class="viewtabs" aria-label="Study views">{nav}</nav></div>')


# "Download this page": saves the page's own file (styles, figures, icons and plot data are all inside it), so the copy
# opens offline exactly as shown; if the file cannot be fetched again, the page as it stands now
DL_JS = """<script>(function(){const a=document.getElementById("dlPage");if(!a)return;
a.addEventListener("click",async e=>{e.preventDefault();let blob;
 try{const r=await fetch(location.href.split("#")[0],{cache:"no-store"});if(!r.ok)throw 0;blob=await r.blob()}
 catch(_){blob=new Blob(["<!doctype html>\\n"+document.documentElement.outerHTML],{type:"text/html"})}
 const u=URL.createObjectURL(blob),d=document.createElement("a");d.href=u;d.download=a.dataset.name||"mechanist.html";
 document.body.appendChild(d);d.click();d.remove();setTimeout(()=>URL.revokeObjectURL(u),4000)})})()</script>"""


# the Mechanist pages of this build, in sidebar order (build() drops the ones a run has no records for)
PAGES_ALL = [("summary", "interpretation.html", "Summary"), ("mechanism", "mechanism.html", "Full mechanism"),
             ("experiments", "experiments.html", "Experiments"), ("treatments", "treatments.html", "Treatments")]
PAGES = PAGES_ALL[:2]


# highlights the sidebar link of the section in view, as the Results tab does
SPY_JS = """<script>(function(){const links=[...document.querySelectorAll("nav.side a.sub")];const by={};
links.forEach(a=>{const id=(a.getAttribute("href")||"").replace(/^#/,"");const el=id&&document.getElementById(id);if(el)by[id]=a});
const io=new IntersectionObserver(es=>{es.forEach(e=>{if(e.isIntersecting){links.forEach(a=>a.classList.remove("active"));
const a=by[e.target.id];if(a)a.classList.add("active")}})},{rootMargin:"-30% 0px -60% 0px"});
Object.keys(by).forEach(id=>io.observe(document.getElementById(id)))})()</script>"""


def _scripts(gs, panels) -> str:
    return (f"</main></div><script>const GROUPS={json.dumps(gs)};const PANELS={json.dumps(panels, separators=(',', ':'))};"
            f"{STRIP_JS}</script>{THEME_JS}{SPY_JS}{DL_JS}")


def _credit() -> str:
    try:
        line = json.loads((ASSETS / "CREDITS.json").read_text())["credit_line"]
    except (OSError, KeyError, json.JSONDecodeError):
        return ""
    return f'<p class="small" style="margin-top:28px">{esc(line)}.</p>'


def _small_print(gs, cfg, ctx=None) -> str:
    n = {g: len(s) for g, s in gs}
    first = gs[0][0] if gs else cfg.disease
    out = (f"With {n.get(first, 0)} {esc(first)} sample{'s' if n.get(first, 0) != 1 else ''}, a circuit can be "
           "plausible, not proven.")
    if ((ctx or {}).get("load") or {}).get("review_status") == "suggested":
        out += (" <b>The cell types are the pipeline's suggested labels: this study's review is still pending, so "
                "every cell-type claim is provisional.</b>")
    return out


# ---- page 1: the interpretation -----------------------------------------------------------------------------------

def render_interpretation(run: Path, built: dict, tabs: list[tuple[str, str, bool]], study: str, sub: str) -> str:
    from mechanist.runner import RunConfig
    ctx, _ = load_run(run)
    cfg = RunConfig.from_yaml(run / "config.yaml")
    page, panels, gs = built["page"], built["panels"], groups(ctx)
    cs = page.get("circuits") or []
    lead = cs[0] if cs else {}
    cards = {k["id"]: i + 1 for i, k in enumerate(lead.get("cards") or [])}
    tiles = "".join(f'<div class="tile"><b>{len(s)}</b><span><i class="key" style="background:var(--grp{i})"></i>'
                    f'{esc(g)} samples</span></div>' for i, (g, s) in enumerate(gs))
    tiles += f'<div class="tile"><b>{len(cs)}</b><span>circuits proposed, ranked</span></div>'
    pts = "".join(f'<li><b>{esc(p.get("label"))}</b>{esc(p.get("text"))}</li>' for p in page.get("points") or [])
    badge = lambda st: f'<span class="badge{" lead" if st == "best supported" else ""}">{esc(st)}</span>'
    support = lambda c: (f'<span class="badge">proposed by {esc(c["support"])} attempts</span>'
                         if c.get("support") else "")
    ev = ""
    for k in lead.get("cards") or []:
        keys = [f'{p["result"]}|{p["gene"]}|{p["lineage"]}' for p in k.get("panels") or []]
        rids = sorted({p["result"] for p in k.get("panels") or []})
        ev += (f'<div class="card evc" id="ev{cards[k["id"]]}"><div class="cardhead"><span class="num">{cards[k["id"]]}</span>'
               f'<h3>{esc(k.get("title"))}</h3></div><p>{esc(k.get("why"))}</p>'
               f"<div class=\"strips\" data-panels='{esc(json.dumps(keys))}'></div>"
               f'<div class="chips">{"".join(f"<a class=badge href=mechanism.html#{r}>source {r}</a>" for r in rids)}</div></div>')
    head = '<tr><th>Disease feature</th>' + "".join(f'<th class="c">{c["rank"]} · {esc(c.get("short_title"))}</th>' for c in cs) + "</tr>"
    cls = {"explains": "good", "partly": "warn"}
    body = "".join("<tr><td>" + esc(f.get("feature")) + "</td>" + "".join(
        f'<td class="c">{"<span class=small>not addressed</span>" if v == "not addressed" else f"<span class=\"badge {cls.get(v, str())}\">{esc(v)}</span>"}</td>'
        for v in f.get("verdicts") or []) + "</tr>" for f in page.get("fit") or [])
    alts = ""
    run_c = {int(c.get("rank") or 0): c for c in (ctx.get("circuits") or {}).get("circuits") or []}
    keys = set()
    for c in cs[1:]:  # a wide row per circuit: its cell figure beside its answer, then its evidence across the width
        rc = run_c.get(int(c.get("rank") or 0)) or {}
        g = _graph(rc)
        keys |= {n.get("icon", "") for n in g["ring"]}
        fig = cellfig.figure(g, f"alt{c['rank']}", c.get("short_title", "")) if g["ring"] else diagram(c, {})
        cc = {k["id"]: f"{c['rank']}.{i + 1}" for i, k in enumerate(c.get("cards") or [])}
        ev_html = "".join(
            f'<div class="card altev"><p><b>{cc[k["id"]]}</b> {esc(k.get("title"))}</p>'
            f"<div class=\"strips\" data-panels='{esc(json.dumps([f'{p['result']}|{p['gene']}|{p['lineage']}' for p in k.get('panels') or []]))}'></div>"
            f'<details><summary>Why</summary><p>{esc(k.get("why"))}</p></details></div>' for k in c.get("cards") or [])
        alts += (f'<div class="card"><div class="altrow"><div>{fig}</div><div>'
                 f'<div class="cardhead"><h3>{c["rank"]} · {esc(c.get("short_title"))}</h3></div>'
                 f'<div class="chips" style="margin:0 0 8px">{badge(c.get("status"))}{support(c)}</div>'
                 f'<p class="lede">{esc(c.get("one_line"))}</p>'
                 f'<a class="more" href="mechanism.html#c{c["rank"]}">Full mechanism, step by step →</a></div></div>'
                 + (f'<div class="altevs">{ev_html}</div>' if ev_html else "") + "</div>")
    tests = "".join(f'<tr><td>{esc(t.get("test"))}</td><td>{esc(t.get("expects"))}</td><td>{esc(t.get("decides"))}</td></tr>'
                    for t in page.get("tests") or [])
    checked = "".join(f"<li>{esc(x)}</li>" for x in page.get("checked") or [])
    leads = "".join(f"<li>{esc(x.get('text', ''))} <span class=small>(attempt {esc(x.get('attempt', '?'))})</span></li>"
                    for x in page.get("leads") or [])
    ex, th = ctx.get("experiment") or {}, ctx.get("therapy") or {}
    has_next = any("tests" in e for e in ex.get("experiments") or []) or bool(th.get("hypotheses"))
    shorts = {int(c.get("rank") or 0): c.get("short_title", "") for c in cs}
    nxt = ""
    if has_next:
        nxt = ('<section id="next"><h2>What to test next, and what it could mean for treatment</h2><p class="sub">The '
               'experiments the circuits disagree on most, and drugs acting on their steps.</p>'
               + (_experiments_html(ex, [{"rank": r} for r in sorted(shorts)], shorts, limit=3)
                  + '<a class="more" href="experiments.html">Every experiment, with how to run it and its papers →</a>'
                  if any("tests" in e for e in ex.get("experiments") or []) else "")
               + (f'<p class="small" style="margin-top:8px">Plus {sum(1 for e in ex.get("experiments") or [] if e.get("confirms"))} '
                  'checks that the molecules the circuits rest on are really there.</p>' if ex.get("experiments") else "")
               + (f'<h3 style="margin:18px 0 6px">Therapeutic hypotheses</h3><div class="note" style="margin:0 0 10px">'
                  f'{NOT_ADVICE}</div><p class="small"><b>Acting on a whole cell type</b> the circuits use:</p>'
                  + _therapy_html(th, shorts, limit=2, strategy="cell type")
                  + '<p class="small"><b>Acting on one step</b>:</p>' + _therapy_html(th, shorts, limit=3, strategy="step")
                  + '<a class="more" href="treatments.html">Every hypothesis, the pathways and the drug lookups →</a>'
                  if th.get("hypotheses") else "") + "</section>")
    side = ([("Answer", "#answer"), ("Lead circuit", "#circuit"), ("Evidence", "#evidence"), ("Disease fit", "#disease"),
             ("Other circuits", "#alternatives")] + ([("Leads", "#leads")] if leads else [])
            + ([("Next steps", "#next")] if has_next else []) + [("Deciding tests", "#tests")])
    return (_head(f"{cfg.study} Mechanist interpretation", study, sub, tabs, side, "summary") + cellfig.icon_defs(keys)
            + '<div class="wrap" id="top">'
            f'<h1>{esc(page.get("question"))}</h1>'
            f'<div class="note">{NOTICE.format(small=_small_print(gs, cfg, ctx))}</div><div class="tiles">{tiles}</div>'
            f'<section id="answer"><div class="answer"><p class="ahead">{esc(page.get("answer"))}</p><ul class="points">{pts}</ul></div>'
            f'<div class="card" id="circuit" style="margin-top:18px"><div class="cardhead"><h3>Lead circuit: {esc(lead.get("short_title"))}</h3>'
            f'{badge(lead.get("status"))}{support(lead)}</div><div class="diagram">{diagram(lead, cards)}</div>{LEGEND}'
            f'<p class="small" style="margin:8px 0 0">{esc(lead.get("one_line"))}</p>'
            '<a class="more" href="mechanism.html#c1">Full mechanism, step by step →</a></div></section>'
            '<section id="evidence"><h2>Evidence for each step</h2><p class="sub">Each dot is one sample: the share of that cell '
            'type expressing the gene. Samples with too few of those cells are left out.</p>'
            f'<div class="ev">{ev}</div>'
            + (f'<details><summary>Already checked in this data, and what it rules out</summary><ul class="checked">{checked}</ul></details>' if checked else "")
            + '</section><section id="disease"><h2>How each circuit fits the disease</h2><p class="sub">Features from the '
            'disease background. A circuit has to produce them, not just sit beside them.</p>'
            f'<div class="tablewrap"><table>{head}{body}</table></div></section>'
            '<section id="alternatives"><h2>Other circuits considered</h2><p class="sub">Ranked by plausibility; they are '
            f'not necessarily exclusive.</p><div class="altlist">{alts}</div></section>'
            + (f'<section id="leads"><h2>Leads found by one attempt only</h2><p class="sub">Observations one attempt '
               f'used and the others did not: worth chasing, not yet part of the model.</p><ul>{leads}</ul></section>'
               if leads else "")
            + nxt
            + '<section id="tests"><h2>Tests that would decide between them</h2><div class="tablewrap"><table>'
            f'<tr><th>Test</th><th>Expected if the lead circuit holds</th><th>Decides</th></tr>{tests}</table></div></section>'
            + _credit() + "</div>" + _scripts(gs, panels) + "</body></html>")


# ---- page 2: the full mechanism (deterministic, nothing condensed) -------------------------------------------------

def _gene_words(text: str) -> set[str]:
    return set(re.findall(r"\b[A-Z][A-Z0-9]{1,}(?:-[A-Z0-9]+)?\b", text))


GENERIC = {"cell", "cells", "state", "like", "high", "low", "other", "inflammatory", "activated"}


def _names(lineage: str, text: str) -> bool:
    """Does the text name this cell type (the whole name, or its last distinctive word)?"""
    low = lineage.lower()
    if re.search(r"(?<![a-z])" + re.escape(low) + r"s?(?![a-z])", text):
        return True
    words = [w for w in re.findall(r"[a-z0-9+]+", low) if w not in GENERIC and len(w) > 2]
    return bool(words) and bool(re.search(r"(?<![a-z])" + re.escape(words[-1]) + r"s?(?![a-z])", text))


def _step_panels(edge: dict, pdata: dict, limit: int = 4) -> list[str]:
    """Panels for one step: its cited results, for genes and cell types the step itself names."""
    text = " ".join(str(edge.get(k, "")) for k in ("source", "target", "via", "note"))
    genes, low = _gene_words(text), text.lower()
    picked, seen = [], set()
    for rid in edge.get("ids") or []:
        for k in pdata:
            if k[0] == rid and k[1] in genes and _names(k[2], low) and (k[1], k[2]) not in seen:
                seen.add((k[1], k[2]))
                picked.append("|".join(k))
                if len(picked) >= limit:
                    return picked
    return picked


# ---- page 2: the full mechanism, figure-led (cell figures and dot plots first; the agents' text folded under them) ----

PLAIN: dict[str, str] = {}  # display text -> its plain-language rewrite (set by build when one exists and passed its check)


def say(x) -> str:
    """Escaped display text, in its plain rewrite when there is one."""
    x = "" if x is None else str(x)
    return esc(PLAIN.get(x.strip(), x))


def _sentences(text: str, n: int = 1) -> tuple[str, str]:
    """(the first n sentences, the rest) of a text, split on sentence ends (not inside numbers such as 1.5)."""
    t = PLAIN.get(str(text or "").strip(), str(text or ""))
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", t.strip())
    return " ".join(parts[:n]), " ".join(parts[n:])


def _short_title(title: str, limit: int = 72) -> str:
    t = re.split(r"[:;]\s", str(title or ""), maxsplit=1)[0]
    return t if len(t) <= limit else t[: limit - 1].rsplit(" ", 1)[0] + "…"


def _graph(c: dict) -> dict:
    return cellfig.build_graph([e for e in c.get("edges") or [] if isinstance(e, dict)])


def _thumb(c: dict, fid: str, href: str, title: str, badges: str = "", note: str = "") -> str:
    fig = cellfig.figure(_graph(c), fid, title, labels=False)
    return (f'<a class="thumb card" href="{esc(href)}">{fig}<div class="cardhead" style="margin:6px 0 0">'
            f'<h3>{esc(title)}</h3></div><div class="chips" style="margin-top:4px">{badges}</div>'
            + (f'<p class="small" style="margin:6px 0 0">{note}</p>' if note else "") + "</a>")


def _support(c: dict) -> str:
    return f'<span class="badge lead">proposed by {esc(c["support"])} attempts</span>' if c.get("support") else ""


def _step_card(i: int, e: dict, pdata: dict, results: dict, used: dict, cited: set) -> str:
    keys = _step_panels(e, pdata)
    for k in keys:
        used[k] = pdata[tuple(k.split("|"))]
    cited |= set(e.get("ids") or [])
    genes = cellfig.genes_in(str(e.get("via", "")), 3) or cellfig.genes_in(str(e.get("source", "")), 2)
    ids = " ".join(f'<a href="#{esc(x)}">{esc(x)}</a>' if x in results else esc(x) for x in e.get("ids") or [])
    src, tgt = cellfig._short(str(e.get("source", "")), 6), cellfig._short(str(e.get("target", "")), 6)
    st = e.get("status", "")
    return (f'<div class="card stepc"><div class="cardhead"><span class="num">{i}</span><h3>{esc(src)} → {esc(tgt)}</h3></div>'
            f'<div class="chips" style="margin:0 0 8px">{"".join(f"<span class=gene>{esc(g)}</span>" for g in genes)}'
            f'<span class="badge{" good" if st == "observed" else ""}">{esc(st)}</span></div>'
            + (f"<div class=\"strips\" data-panels='{esc(json.dumps(keys))}'></div>" if keys else
               '<p class="small" style="margin:0">No per-sample plot for this step.</p>')
            + f'<details><summary>Why</summary><p>{say(e.get("via"))}. {say(e.get("note"))}</p>'
            f'<p class="ids">{ids}</p></details></div>')


def _circuit_visual(c: dict, anchor: str, heading: str, short: str, pdata: dict, results: dict, used: dict,
                    cited: set) -> list[str]:
    """One circuit: its cell figure and one-sentence answer, the steps as figure cards, and the agents' full reasoning
    folded underneath."""
    lede, rest = _sentences(c.get("summary"), 1)
    stands = (f'<p class="small">Stands for {esc("; ".join(c["merged_from"]))}'
              f'{" (the first is the version drawn)" if len(c["merged_from"]) > 1 else ""}.</p>'
              if c.get("merged_from") else "")
    out = [(f'<section id="{anchor}"><div class="cfig"><div class="card">'
           f'{cellfig.figure(_graph(c), anchor + "f", short)}{cellfig.LEGEND}</div>'
           f'<div><p class="eyebrow">{esc(heading)}</p><h2>{esc(short)}</h2><div class="chips">{_support(c)}</div>'
           f'<p class="lede">{esc(lede)}</p>{stands}'
           f'<p class="small">{len(c.get("edges") or [])} steps · '
           f'{sum(1 for e in c.get("edges") or [] if e.get("status") == "observed")} observed in this data</p>'
           f'<details><summary>Full title and reasoning</summary>{_reason_table(c, rest)}</details></div></div>')]
    out.append('<div class="stepgrid">' + "".join(_step_card(i, e, pdata, results, used, cited)
                                                  for i, e in enumerate(c.get("edges") or [], 1)) + "</div>")
    if c.get("phenotype"):
        out.append("<details><summary>How it produces the disease</summary><div class='tablewrap'><table><tr><th>Disease "
                   "item</th><th>How</th></tr>" + "".join(f"<tr><td>{''.join(f'<span class=gene>{esc(i)}</span> ' for i in _item_ids(p.get('item'))) or say(p.get('item'))}</td><td>{say(p.get('how'))}</td></tr>"
                                                         for p in c["phenotype"] if isinstance(p, dict))
                   + "</table></div></details>")
    if c.get("predictions"):
        out.append("<details><summary>Predictions that would test it</summary><div class='tablewrap'><table><tr><th>Test"
                   "</th><th>Expects</th><th>In this data</th></tr>"
                   + "".join(f"<tr><td>{say(p.get('test'))}</td><td>{say(p.get('expects'))}</td><td>"
                             f"{'yes' if p.get('in_this_data') else 'needs new data'}</td></tr>"
                             for p in c["predictions"] if isinstance(p, dict)) + "</table></div></details>")
    if c.get("analogue") or c.get("unused"):
        out.append("<details><summary>Closest published circuit, and what it leaves out</summary><div class='prose'>"
                   + (f"<p>{say(c.get('analogue'))}</p>" if c.get("analogue") else "")
                   + "".join(f"<p>{say(x)}</p>" for x in c.get("unused") or []) + "</div></details>")
    out.append("</section>")
    return out


# ---- charts and tables that replace folded prose on the full-mechanism page ------------------------------------------

def _item_ids(text: str) -> list[str]:
    return re.findall(r"\bD\d{2,}\b", str(text or ""))


def _fit_matrix(cs: list[dict], items: list[tuple[str, str, str]], short: dict) -> str:
    """Which disease item each circuit produces (its phenotype accounting), as a matrix; hover a tick for how."""
    how: dict[tuple[str, int], str] = {}
    for c in cs:
        for p in c.get("phenotype") or []:
            if isinstance(p, dict):
                for i in _item_ids(p.get("item")):
                    how[(i, c["rank"])] = str(p.get("how") or "")
    known = {i: (sec, t) for i, sec, t in items}
    rows = [i for i, _, _ in items if any((i, c["rank"]) in how for c in cs)]
    missing = [i for i, _, _ in items if i not in rows]
    head = ("<tr><th>Disease item</th>" + "".join(f'<th class="c" title="{esc(short.get(c["rank"], ""))}">{c["rank"]}</th>'
                                                  for c in cs) + "</tr>")
    body = ""
    for i in rows:
        sec, t = known.get(i, ("", ""))
        label = " ".join(str(t).split()[:9]) + ("…" if len(str(t).split()) > 9 else "")
        body += (f'<tr><td><b>{esc(i)}</b> <span class="small">{esc(sec)}</span><br>{esc(label)}</td>'
                 + "".join(f'<td class="c fitcell{" on" if (i, c["rank"]) in how else ""}"'
                           + (f' title="{esc(PLAIN.get(how[(i, c["rank"])].strip(), how[(i, c["rank"])]))}"'
                              if (i, c["rank"]) in how else "") + ">"
                           + ("●" if (i, c["rank"]) in how else "") + "</td>" for c in cs) + "</tr>")
    total = ("<tr><td><b>Items produced</b></td>" + "".join(
        f'<td class="c"><b>{sum(1 for i in rows if (i, c["rank"]) in how)}</b></td>' for c in cs) + "</tr>")
    return (f'<div class="tablewrap"><table class="fit">{head}{body}{total}</table></div>'
            + (f'<p class="small">Not produced by any circuit: {esc(", ".join(missing))}.</p>' if missing else ""))


def _reason_table(c: dict, rest: str) -> str:
    rows = [("Full title", say(c.get("title"))), ("Why here and not in the comparators", say(c.get("root"))),
            ("Confidence", say(c.get("confidence")))]
    if c.get("loops"):
        rows.append(("Loops", "<ul>" + "".join(f"<li>{say(x)}</li>" for x in c["loops"]) + "</ul>"))
    if rest:
        rows.append(("Rest of the summary", esc(rest)))
    return ('<div class="tablewrap"><table class="kv">'
            + "".join(f"<tr><th>{esc(k)}</th><td>{v}</td></tr>" for k, v in rows if v) + "</table></div>")


def _balance(n_for: int, n_against: int) -> str:
    """A small bar: evidence for (green) against evidence against (warm), one segment per item."""
    tot = max(1, n_for + n_against)
    w = 160
    a = w * n_for / tot
    return (f'<svg viewBox="0 0 {w + 70} 14" width="{w + 70}" height="14" role="img" aria-label="{n_for} for, '
            f'{n_against} against"><rect x="0" y="2" width="{a:.1f}" height="10" rx="2" fill="var(--good)"/>'
            f'<rect x="{a:.1f}" y="2" width="{w - a:.1f}" height="10" rx="2" fill="var(--warn-bd)"/>'
            f'<text x="{w + 6}" y="11" font-size="11" fill="var(--ink-2)">{n_for} / {n_against}</text></svg>')


def _alt_table(alts: list[dict]) -> str:
    rows = ""
    for alt in alts:
        pro = [x for x in alt.get("evidence_for") or [] if x]
        con = [x for x in alt.get("evidence_against") or [] if x]
        conf = str(alt.get("confidence") or "")
        lists = ("<details><summary>The evidence</summary><div class='forag'><div><b>For</b><ul>"
                 + "".join(f"<li>{say(x)}</li>" for x in pro) + "</ul></div><div><b>Against</b><ul>"
                 + "".join(f"<li>{say(x)}</li>" for x in con) + "</ul></div></div></details>") if pro or con else ""
        rows += (f'<tr><td><b>{esc(_short_title(alt.get("title"), 80))}</b><br><span class="small">'
                 f'{esc(_sentences(alt.get("summary"), 1)[0])}</span>{lists}</td>'
                 f'<td class="c"><span class="badge{" warn" if "low" in conf else ""}" title="{esc(conf)}">'
                 f'{esc((conf[:24] + "…") if len(conf) > 26 else (conf or "—"))}</span></td>'
                 f"<td>{_balance(len(pro), len(con))}</td></tr>")
    return ('<div class="tablewrap"><table><tr><th>Explanation</th><th class="c">Confidence</th>'
            f'<th>Evidence for / against</th></tr>{rows}</table></div>')


def _guess_chart(guesses: list[dict]) -> str:
    """For each step the data could not settle, the ranked candidates as bars, length = evidence rung."""
    gs = [g for g in guesses if isinstance(g, dict) and g.get("candidates")]
    if not gs:
        return ""
    top = max([int(x.get("evidence_rung") or 0) for g in gs for x in g["candidates"] if isinstance(x, dict)] + [4])
    out = []
    for g in gs:
        bars = ""
        for x in g["candidates"]:
            if not isinstance(x, dict):
                continue
            r = int(x.get("evidence_rung") or 0)
            text = PLAIN.get(str(x.get("explanation") or "").strip(), str(x.get("explanation") or ""))
            bars += (f'<div class="gbar" title="{esc(text)}"><span class="gl">{esc(" ".join(text.split()[:12]))}'
                     f'{"…" if len(text.split()) > 12 else ""}</span><span class="gt"><i style="width:{100 * (r + .3) / (top + .3):.0f}%">'
                     f"</i></span><span class=\"small\">rung {r}</span></div>")
        out.append(f'<div class="card"><h3 style="font-size:13.5px;margin-bottom:8px">{say(g.get("step"))}</h3>{bars}</div>')
    return ('<h3 style="margin:18px 0 6px">Best guesses for the steps the data could not settle</h3><p class="small">'
            f"Bar length is the evidence rung (0 = inference only, {top} = strongest); hover a bar for the full "
            f'explanation.</p><div class="gallery" style="grid-template-columns:repeat(auto-fill,minmax(340px,1fr))">{"".join(out)}</div>')


def _review_chart(attempts: list[dict], revs: list[dict]) -> str:
    """Per attempt, each PI review's new tests and demands as bars, with the established steps as a line."""
    by = {}
    for r in revs:
        by.setdefault(r.get("attempt") or 1, []).append(r)
    out = []
    for k, rs in sorted(by.items()):
        W, H, L, B = 300, 130, 24, 20
        n = len(rs)
        mx = max([int(r.get("new_tests") or 0) for r in rs] + [len(r.get("demands") or []) for r in rs]
                 + [sum(r.get("established") or [0]) for r in rs] + [4])
        cw = (W - L) / max(n, 1)
        y = lambda v, H=H, B=B, mx=mx: H - B - (H - B - 8) * v / mx
        s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Attempt {k}: PI reviews">']
        pts = []
        for i, r in enumerate(rs):
            x0 = L + i * cw + cw * .2
            t, d = int(r.get("new_tests") or 0), len(r.get("demands") or [])
            e = sum(r.get("established") or [0])
            s.append(f'<rect x="{x0:.1f}" y="{y(t):.1f}" width="{cw * .28:.1f}" height="{H - B - y(t):.1f}" fill="var(--accent)">'
                     f"<title>review {i + 1}: {t} new tests</title></rect>")
            s.append(f'<rect x="{x0 + cw * .3:.1f}" y="{y(d):.1f}" width="{cw * .28:.1f}" height="{H - B - y(d):.1f}" '
                     f'fill="var(--muted)"><title>review {i + 1}: {d} demands</title></rect>')
            pts.append((x0 + cw * .29, y(e), e))
            s.append(f'<text x="{x0 + cw * .29:.1f}" y="{H - 6}" font-size="10" text-anchor="middle" fill="var(--ink-2)">{i + 1}</text>')
        if len(pts) > 1:
            s.append('<polyline fill="none" stroke="var(--grp0)" stroke-width="2" points="'
                     + " ".join(f"{px:.1f},{py:.1f}" for px, py, _ in pts) + '"/>')
        s += [f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3" fill="var(--grp0)"><title>{e} established steps</title></circle>'
              for px, py, e in pts]
        closed = next((r.get("closed_by") or r.get("closed") for r in reversed(rs) if r.get("closed_by") or r.get("closed")), "")
        out.append(f'<div class="card"><h3 style="font-size:13.5px">Attempt {esc(k)}</h3><p class="small" style="margin:2px 0 6px">'
                   f'{n} reviews · last verdict {esc(rs[-1].get("verdict"))}' + (f" · closed by {esc(closed)}" if closed else "")
                   + f'</p>{"".join(s)}</svg></div>')
    return ('<div class="legend" style="margin:0 0 8px"><span><i class="key" style="background:var(--accent);border-radius:2px">'
            '</i>new tests asked</span><span><i class="key" style="background:var(--muted);border-radius:2px"></i>demands</span>'
            '<span><i class="key" style="background:var(--grp0)"></i>established steps (observed, analogies run, disease items)'
            f'</span></div><div class="gallery" style="grid-template-columns:repeat(auto-fill,minmax(300px,1fr))">{"".join(out)}</div>')


# ---- what to test next, and what it could mean for treatment (stages 8-9) -------------------------------------------

GLYPH = {"up": ("▲", "var(--accent)", "up"), "down": ("▼", "var(--grp0)", "down"),
         "no change": ("●", "var(--muted)", "no change"), "not addressed": ("–", "var(--border)", "not addressed")}
LEVEL_CLASS = {"in use for this disease": "good", "approved for another disease": "lead", "in trials": "", "preclinical": "warn"}
NOT_ADVICE = ("<b>Research hypotheses, not treatment recommendations.</b> Drugs come from Open Targets lookups that ran; "
              "clinical stage, action and evidence level are set by code from the lookup, never by the model. A hypothesis "
              "that depends on an untested link should wait for that link's experiment.")


def _mchips(ids, warn: bool = False) -> str:
    return "".join(f'<span class="badge{" warn" if warn else ""}">{esc(m)}</span>' for m in ids)


def _experiments_html(ex: dict, cs: list[dict], short: dict, limit: int | None = None) -> str:
    exps = [e for e in ex.get("experiments") or [] if "tests" in e
            and any(p.get("outcome") != "not addressed" for p in e.get("predictions") or [])]
    if not exps:
        return ""
    ranks = [c["rank"] for c in cs]
    top = max([e.get("discrimination") or 0 for e in exps] + [0.01])
    head = ("<tr><th>#</th><th>Experiment</th>" + "".join(
        f'<th class="c" title="{esc(short.get(r, ""))}">{r}</th>' for r in ranks) + "<th>Separates</th></tr>")
    rows = ""
    for e in exps[:limit] if limit else exps:
        pr = {p["circuit"]: p for p in e.get("predictions") or []}
        cells = ""
        for r in ranks:
            p = pr.get(r) or {"outcome": "not addressed", "why": ""}
            g, col, word = GLYPH.get(p["outcome"], GLYPH["not addressed"])
            cells += f'<td class="c" style="color:{col};font-size:15px" title="{esc(word)}: {esc(p.get("why"))}">{g}</td>'
        w = 100 * (e.get("discrimination") or 0) / top
        how = ('<details><summary>How</summary><table class="kv">'
               + "".join(f"<tr><th>{k}</th><td>{esc(e.get(f))}</td></tr>" for k, f in
                         (("Model", "model"), ("Perturbation", "perturbation"), ("Readout", "readout"),
                          ("Controls", "controls"), ("Counts against the circuit if", "refutes_if"),
                          ("Could mislead because", "caveat"), ("Feasibility", "feasibility")) if e.get(f))
               + (f"<tr><th>Rooted in</th><td>{_precedent_html(e['precedent'])}</td></tr>"
                  if e.get("precedent") else "<tr><th>Rooted in</th><td><span class=small>no published precedent "
                  "cited</span></td></tr>") + "</table></details>") if limit is None else ""
        rows += (f'<tr><td>{e.get("rank")}</td><td><b>{esc(e.get("title") or e.get("perturbation"))}</b> '
                 + (f'<span class="badge lead">{esc(e["kind"])}</span>' if e.get("kind") else "")
                 + f'<span class="badge">{esc(e.get("setting") or e.get("tier") or "")}</span> {_mchips(e.get("tests") or [])}'
                 + f'<br><span class="small">{esc(e.get("readout"))}</span>{how}</td>{cells}'
                 f'<td style="min-width:90px"><span class="gt" title="discrimination {e.get("discrimination")}">'
                 f'<i style="width:{w:.0f}%"></i></span></td></tr>')
    legend = ('<div class="legend" style="margin:6px 0 10px">'
              + "".join(f'<span><b style="color:{c}">{g}</b> {w}</span>' for g, c, w in GLYPH.values())
              + "<span>Hover a mark for why that circuit predicts it.</span></div>")
    return f'{legend}<div class="tablewrap"><table>{head}{rows}</table></div>'


QWORD = {"presence": "Is it there?", "rule_out": "Could rule it out fast", "mechanism": "Does the mechanism work?",
         "in_vivo": "Does it hold in a living system?"}
TIERWORD = {"in_data": "this dataset", "archived_tissue": "archived tissue", "spatial": "spatial", "ex_vivo": "ex vivo",
            "in_vitro": "in vitro", "in_vivo": "in vivo", "cohort": "new cohort"}
PLAN_CSS = """<style>
.prog{border:1px solid var(--border);border-radius:8px;background:var(--surface);padding:14px 16px;margin-top:14px}
.prog h3{font-size:16px}.prog .claim{margin:4px 0 10px;color:var(--ink-2);max-width:80ch}
.seq{list-style:none;margin:0;padding:0;counter-reset:st}
.seq>li{counter-increment:st;display:grid;grid-template-columns:30px minmax(0,1fr);gap:10px;padding:10px 0;
  border-top:1px solid var(--border)}
.seq>li::before{content:counter(st);width:24px;height:24px;border-radius:50%;background:var(--ink);color:var(--surface);
  font-size:12px;font-weight:700;display:grid;place-items:center}
.seq .q{font-size:11.5px;font-weight:600;color:var(--ink-2)}
.seq .t{font-weight:600;margin:1px 0 2px}
.seq .sh{font-size:13px;line-height:1.5;margin:0}
.seq .br{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:4px 16px;font-size:12.5px;margin-top:6px}
.seq .br b{font-weight:600}
.seq .done{margin-top:6px;font-size:12.5px;padding:6px 9px;border:1px dashed var(--warn-bd);border-radius:6px}
@media (max-width:700px){.seq .br{grid-template-columns:1fr}}
</style>"""


def _programmes_html(ex: dict, short: dict) -> str:
    progs = ex.get("programmes") or []
    if not progs:
        return ""
    by = {e.get("id"): e for e in ex.get("experiments") or []}
    out = []
    for pr in progs:
        items = []
        for st in pr.get("sequence") or []:
            e = by.get(st["id"])
            if not e:
                continue
            others = [r for r in e.get("in_programmes") or [] if r != pr["circuit"]]
            shared = (f' <span class="badge">also in {", ".join(str(r) for r in others)}</span>' if others else "")
            done = "".join(f'<div class="done"><b>Done before:</b> {esc(d.get("what") or d.get("title"))} '
                           f'<a href="https://pubmed.ncbi.nlm.nih.gov/{esc(d["pmid"])}/">PMID {esc(d["pmid"])}</a>'
                           f'{" (" + esc(d["year"]) + ")" if d.get("year") else ""}. Replicate in this disease, or cite.</div>'
                           for d in e.get("done_before") or [])
            how = ('<details><summary>Design</summary><table class="kv">'
                   + "".join(f"<tr><th>{k}</th><td>{esc(e.get(f))}</td></tr>" for k, f in
                             (("Model", "model"), ("Perturbation", "perturbation"), ("Readout", "readout"),
                              ("Controls", "controls"), ("Counts against it if", "refutes_if"),
                              ("Protein vs RNA", "rna_vs_protein"), ("Could mislead because", "caveat"),
                              ("Feasibility", "feasibility")) if e.get(f))
                   + (f"<tr><th>Rooted in</th><td>{_precedent_html(e['precedent'])}</td></tr>" if e.get("precedent") else "")
                   + "</table></details>")
            branch = "".join(f"<div><b>{lab}</b> {esc(st.get(k))}</div>" for k, lab in
                             (("if_yes", "If it works:"), ("if_no", "If not:")) if st.get(k))
            items.append(
                f'<li><div><div class="q">{esc(QWORD.get(e.get("question"), "Experiment"))} · '
                f'{esc(TIERWORD.get(e.get("tier"), e.get("setting") or ""))} · {esc(e.get("id"))}{shared}</div>'
                f'<div class="t">{esc(e.get("title"))}</div>'
                + (f'<p class="sh">{esc(e.get("shows"))}' + (f' <span class="small">New: {esc(e.get("adds"))}</span>'
                                                              if e.get("adds") else "") + "</p>" if e.get("shows") else "")
                + (f'<p class="sh small">Adapted here: {esc(st["adapt"])}</p>' if st.get("adapt") else "")
                + (f'<p class="sh small">Why now: {esc(st["why_now"])}</p>' if st.get("why_now") else "")
                + (f'<div class="br">{branch}</div>' if branch else "") + done + how + "</div></li>")
        out.append(f'<div class="prog" id="h{pr["circuit"]}"><h3>Hypothesis {pr["circuit"]}: '
                   f'{esc(short.get(pr["circuit"], ""))}</h3><p class="claim"><b>Sets out to prove:</b> '
                   f'{esc(pr.get("claim"))}</p><ol class="seq">{"".join(items)}</ol></div>')
    return PLAN_CSS + "".join(out)


def _precedent_html(prec: list[dict]) -> str:
    out = []
    for p in prec:
        refs = ", ".join(f'<a href="https://pubmed.ncbi.nlm.nih.gov/{esc(x["pmid"])}/" title="{esc(x.get("title"))}">'
                         f'PMID {esc(x["pmid"])}</a>' + (f' ({esc(x.get("year"))})' if x.get("year") else "")
                         for x in p.get("papers") or [])
        out.append(f"{esc(p.get('what'))}" + (f" <span class='ids'>{refs}</span>" if refs else ""))
    return "<br>".join(out)


def _confirm_html(ex: dict) -> str:
    rows = ""
    for e in [e for e in ex.get("experiments") or [] if e.get("confirms")]:
        rows += (f'<tr><td>{"".join(f"<span class=gene>{esc(g)}</span> " for g in e.get("confirms") or [])}</td>'
                 f'<td><b>{esc(e.get("title"))}</b><br><span class="small">{esc(e.get("readout"))} · '
                 f'{esc(e.get("model"))}</span></td><td><span class="badge">{esc(e.get("setting") or e.get("tier") or "")}'
                 f'</span></td><td class="small">{say(e.get("caveat")) or "—"}</td>'
                 f'<td class="small">{_precedent_html(e.get("precedent") or []) or "—"}</td></tr>')
    if not rows:
        return ""
    miss = ex.get("unconfirmed") or []
    return ('<div class="tablewrap"><table><tr><th>Molecule</th><th>Assay and samples</th><th>Setting</th>'
            f'<th>Could mislead because</th><th>Rooted in</th></tr>{rows}</table></div>'
            + ('<p class="small">Molecules the circuits rest on with no confirmation yet: '
               + "".join(f"<span class='badge warn'>{esc(g)}</span>" for g in miss) + "</p>" if miss else ""))


def _missing_html(ex: dict) -> str:
    ms = ex.get("missing_links") or []
    if not ms:
        return ""
    cover: dict[str, list] = {}
    for e in ex.get("experiments") or []:
        for m in e.get("tests") or []:
            cover.setdefault(m, []).append(str(e.get("rank")))
    nt = {x["id"]: x["why"] for x in ex.get("not_testable") or []}
    rows = "".join(
        f"<tr><td><b>{esc(m['id'])}</b></td><td><span class='badge'>{esc(m['kind'])}</span></td><td>{say(m['text'])}</td>"
        f"<td>{'experiment ' + ', '.join(cover[m['id']]) if m['id'] in cover else ('<span class=small>not testable: ' + esc(nt[m['id']]) + '</span>' if m['id'] in nt else '<span class=badge warn>no experiment</span>')}</td></tr>"
        for m in ms)
    return (f"<details><summary>The {len(ms)} missing links and what covers each</summary><div class='tablewrap'><table>"
            f"<tr><th>Id</th><th>Kind</th><th>Link</th><th>Covered by</th></tr>{rows}</table></div></details>")


def _therapy_html(th: dict, short: dict, limit: int | None = None, strategy: str | None = None) -> str:
    hs = [h for h in th.get("hypotheses") or [] if strategy is None or h.get("strategy", "step") == strategy]
    if not hs:
        return ""
    rows = ""
    for h in hs[:limit] if limit else hs:
        more = (f'<details><summary>Why</summary><p>{say(h.get("rationale"))}</p><p class="small">Mechanism: '
                f'{esc(h.get("mechanism"))}. Listed for: {esc(", ".join(h.get("indications") or []))}.</p></details>'
                if limit is None else "")
        safety = "".join(f'<span class="badge warn" title="{esc(w.get("toxicity"))}">{esc(w.get("type"))}'
                         + (f' · {esc(w.get("toxicity"))}' if w.get("toxicity") else "") + "</span>"
                         for w in h.get("warnings") or [])
        rob = (f'<span class="gt" title="acts in {h.get("robustness", 0) * 100:.0f}% of the circuits\' support">'
               f'<i style="width:{100 * (h.get("robustness") or 0):.0f}%"></i></span>')
        what = (f'removes {esc(h["depletes"])}s' if h.get("depletes") else f'breaks {esc(h.get("breaks"))}')
        rows += (f'<tr><td>{h["rank"]}</td><td><b>{esc(h["drug"].title())}</b> <span class="gene">{esc(h["target"])}</span>'
                 f'<br><span class="small">{what}</span>{more}</td>'
                 f'<td><span class="badge {LEVEL_CLASS.get(h["level"], "")}">{esc(h["level"])}</span>'
                 f'<br><span class="small">{esc((h.get("stage") or "").replace("_", " ").lower())} · '
                 f'{esc(", ".join(h.get("actions") or []))}{"" if h.get("blocks") else " (not a blocker)"}</span></td>'
                 f'<td>{"".join(f"<span class=badge>{r}</span>" for r in h.get("circuits") or [])}{rob}</td>'
                 f'<td>{safety or "<span class=small>no warning listed</span>"}</td>'
                 f'<td>{"".join(f"<span class=badge>{r}</span>" for r in h.get("fails_under") or []) or "—"}</td>'
                 f'<td>{_mchips(h.get("depends_on") or [], warn=True) or "—"}</td></tr>')
    return ('<div class="tablewrap"><table><tr><th>#</th><th>Drug and target</th><th>Evidence</th><th>Acts in circuits '
            '(robustness)</th><th>Safety flags</th><th>Should fail under</th><th>Test first</th></tr>'
            f'{rows}</table></div>')


def _pathway_html(th: dict) -> str:
    ps = th.get("pathways") or []
    if not ps:
        return ""
    top = max([p["weight"] for p in ps] + [0.01])
    bars = "".join(
        f'<div class="gbar"><span class="gl"><b>{esc(p["name"])}</b> <span class="small">{esc(", ".join(p["genes"]))}</span>'
        f'</span><span class="gt"><i style="width:{100 * p["weight"] / top:.0f}%"></i></span><span class="small">'
        f'{esc(", ".join(d.title() for d in p["drugs"][:3])) or "no drug"}</span></div>' for p in ps)
    return ('<h3 style="margin:18px 0 6px">Pathways the circuits converge on</h3><p class="small">Bar: the summed support '
            f'of the circuits that carry the pathway\'s genes. Right: checked drugs acting on it.</p><div class="card">{bars}</div>')


def _lead_card(x: dict, results: dict) -> str:
    text = str(x.get("text", ""))
    cells = cellfig.cells_in(text)
    icon = cellfig.icon_key(cells[0], text) if cells else ""
    head, rest = _sentences(text, 1)
    genes = cellfig.genes_in(text, 4)
    ids = " ".join(f'<a href="#{esc(i)}">{esc(i)}</a>' if i in results else esc(i) for i in x.get("ids") or [])
    return ('<div class="card leadc">' + (f'<svg class="licon" viewBox="0 0 100 100" aria-hidden="true">'
                                          f'<use href="#bio-{icon}"/></svg>' if icon else "")
            + f'<div><div class="chips" style="margin:0 0 6px">{"".join(f"<span class=gene>{esc(g)}</span>" for g in genes)}'
            f'<span class="badge">attempt {esc(x.get("attempt", "?"))}</span></div><p style="margin:0">{esc(head)}</p>'
            + (f"<details><summary>More</summary><p>{esc(rest)}</p><p class='ids'>{ids}</p></details>" if rest or ids else "")
            + "</div></div>")


def render_mechanism(run: Path, tabs: list[tuple[str, str, bool]], study: str, sub: str,
                     shorts: dict | None = None) -> tuple[str, dict]:
    from mechanist.runner import RunConfig
    ctx, results = load_run(run)
    cfg = RunConfig.from_yaml(run / "config.yaml")
    gs = groups(ctx)
    pdata = panel_data(results)
    circ = ctx["circuits"]
    shorts = shorts or {}
    used: dict[str, dict] = {}
    cited: set[str] = set()
    cs = circ["circuits"]
    attempts = circ.get("attempts") or []
    syn = ((ctx.get("revise") or ctx.get("synthesize") or {}).get("synthesis")) or {}
    ledger = [{"source": lk.get("source", ""), "target": lk.get("target", ""), "via": lk.get("mediator", ""),
               "status": "observed" if lk.get("ids") else "likely"} for lk in syn.get("links") or []]
    keys = set()
    for c in cs + [c for a in attempts for c in a.get("circuits") or []] + [{"edges": ledger}]:
        keys |= {n.get("icon", "") for n in _graph(c)["ring"]}
    leads = [x for x in circ.get("leads") or [] if isinstance(x, dict)]
    keys |= {cellfig.icon_key(cellfig.cells_in(str(x.get("text", "")))[0], str(x.get("text", "")))
             for x in leads if cellfig.cells_in(str(x.get("text", "")))}
    short = {c["rank"]: shorts.get(c["rank"]) or _short_title(c.get("title")) for c in cs}
    side = ([("At a glance", "#glance"), ("Evidence ledger", "#ledger"), ("Disease fit", "#fit")]
            + [(f"Circuit {c['rank']}", f"#c{c['rank']}") for c in cs]
            + [("Every attempt", "#attempts"), ("Leads", "#leads"), ("Weaker explanations", "#alternatives"),
               ("Starting hypotheses", "#starting"), ("Run record", "#record")])
    out = [_head(f"{cfg.study} Mechanist full mechanism", study, sub, tabs, side, "mechanism"), cellfig.icon_defs(keys),
           '<div class="wrap" id="top"><h1>The full mechanism, in figures</h1>',
           f'<div class="note">{NOTICE.format(small="Figures are drawn by code from the run record; the agents&#x27; own words are one click away under each figure.")}</div>']
    out.append('<section id="glance"><h2>At a glance</h2><p class="sub">Every circuit the run kept, ranked. Cells are '
               'the ones each circuit names; arrows are its steps.</p>'
               + (f'<details><summary>How they are ranked</summary><div class="prose"><p>{say(circ.get("ranking"))}</p>'
                  "</div></details>" if circ.get("ranking") else "")
               + '<div class="gallery" style="margin-top:12px">'
               + "".join(_thumb(c, f"g{c['rank']}", f"#c{c['rank']}", f"{c['rank']} · {short[c['rank']]}", _support(c))
                         for c in cs) + "</div></section>")
    if ledger:
        for lk in syn.get("links") or []:
            cited |= set(lk.get("ids") or [])
        out.append('<section id="ledger"><div class="cfig"><div class="card">'
                   f'{cellfig.figure(cellfig.build_graph(ledger), "ledf", "Evidence ledger")}{cellfig.LEGEND}</div><div>'
                   '<p class="eyebrow">Evidence ledger</p><h2>What the investigation established first</h2>'
                   '<p class="lede">Before any circuit was drawn, the team investigated the data question by question and '
                   'kept only the cell-to-cell links the data support. This ledger is that grounded model: each arrow is '
                   'one link (a sender cell, the molecule it makes, a receiver cell that carries the receptor), tied to the '
                   'data queries behind it.</p><p class="small">The circuits below are built on top of it: they add the '
                   'order of events, the upstream cause and how the disease follows, which the ledger alone cannot say.</p>'
                   f'<p class="small"><b>The ledger in one line.</b> {esc(_sentences(syn.get("summary"), 1)[0])}</p>'
                   f'<details><summary>The {len(ledger)} links as a table</summary><div class="tablewrap"><table><tr>'
                   '<th>From</th><th>Mediator</th><th>To</th><th>Data</th></tr>'
                   + "".join(f"<tr><td>{esc(cellfig._short(str(lk.get('source', '')), 6))}</td><td>"
                             + "".join(f"<span class=gene>{esc(g)}</span> " for g in cellfig.genes_in(str(lk.get('mediator', '')), 4))
                             + f"</td><td>{esc(cellfig._short(str(lk.get('target', '')), 6))}</td><td class='ids'>"
                             + " ".join(f'<a href="#{esc(x)}">{esc(x)}</a>' if x in results else esc(x) for x in lk.get("ids") or [])
                             + "</td></tr>" for lk in syn.get("links") or [])
                   + "</table></div></details></div></div></section>")
    items = agents.phenotype_items((ctx.get("dossier") or {}).get("dossier") or {})
    if items and any(c.get("phenotype") for c in cs):
        out.append('<section id="fit"><h2>Which disease features each circuit produces</h2><p class="sub">Items from the '
                   'disease background (the dossier). A dot means the circuit says how it produces that item; hover a dot '
                   'for how.</p>' + _fit_matrix(cs, items, short) + "</section>")
    for c in cs:
        out += _circuit_visual(c, f"c{c['rank']}", f"Circuit {c['rank']}", short[c["rank"]], pdata, results, used, cited)
    if len(attempts) > 1:
        where = {}
        for c in cs:
            for m in c.get("merged_from") or []:
                where[m] = c["rank"]
        out.append(f'<section id="attempts"><h2>Every attempt</h2><p class="sub">The circuits above merge '
                   f'{len(attempts)} independent attempts on the same evidence. Each attempt as it stood:</p>')
        for a in attempts:
            k = a.get("attempt")
            out.append(f'<h3 style="margin:16px 0 8px">Attempt {esc(k)} <span class="small">· {len(a.get("reviews") or [])} '
                       'PI reviews</span></h3><div class="gallery">')
            for c in a.get("circuits") or []:
                m = where.get(f"attempt {k}, circuit {c.get('rank')}")
                out.append(_thumb(c, f"a{k}c{c.get('rank')}", f"#a{k}c{c.get('rank')}x",
                                  f"{c.get('rank')} · {_short_title(c.get('title'), 60)}",
                                  f'<span class="badge">in circuit {m}</span>' if m else
                                  '<span class="badge warn">not kept</span>'))
            out.append("</div>")
            for c in a.get("circuits") or []:
                lede, rest = _sentences(c.get("summary"), 1)
                out.append(f'<details id="a{k}c{c.get("rank")}x"><summary>Attempt {esc(k)}, circuit {esc(c.get("rank"))}: '
                           f'{esc(_short_title(c.get("title"), 90))}</summary><p>{esc(lede)}</p>{_reason_table(c, rest)}'
                           '<div class="tablewrap" style="margin-top:8px"><table><tr><th>Step</th><th>Mediator</th>'
                           '<th>Known as</th></tr>' + "".join(
                               f"<tr><td>{esc(cellfig._short(str(e.get('source', '')), 6))} → "
                               f"{esc(cellfig._short(str(e.get('target', '')), 6))}</td><td>"
                               + "".join(f"<span class=gene>{esc(g)}</span> " for g in cellfig.genes_in(str(e.get('via', '')), 3))
                               + f"</td><td><span class='badge{' good' if e.get('status') == 'observed' else ''}'>"
                               f"{esc(e.get('status'))}</span></td></tr>" for e in c.get("edges") or [] if isinstance(e, dict))
                           + "</table></div></details>")
        out.append("</section>")
    if leads:
        out.append('<section id="leads"><h2>Leads found by one attempt only</h2><p class="sub">Observations one attempt '
                   'used and the others did not: worth chasing, not yet part of any circuit.</p><div class="leads">'
                   + "".join(_lead_card(x, results) for x in leads) + "</div></section>")
    alts = [x for x in syn.get("alternatives") or [] if isinstance(x, dict)]
    if alts or syn.get("competing_explanations") or syn.get("best_guesses"):
        out.append('<section id="alternatives"><h2>Weaker and competing explanations</h2><p class="sub">What the '
                   'investigation kept open before the circuits were drawn, with the balance of evidence it found.</p>'
                   + (_alt_table(alts) if alts else "") + _guess_chart(syn.get("best_guesses") or []))
        lists = [("Competing explanations", syn.get("competing_explanations") or []),
                 ("Open questions", syn.get("open_questions") or [])]
        for title, xs in lists:
            if xs:
                out.append(f'<details><summary>{esc(title)} ({len(xs)})</summary><div class="tablewrap"><table>'
                           + "".join(f"<tr><td class='c' style='width:32px'>{i}</td><td>{say(x if isinstance(x, str) else json.dumps(x))}</td></tr>"
                                     for i, x in enumerate(xs, 1)) + "</table></div></details>")
        out.append("</section>")
    hyps = (ctx.get("candidates") or {}).get("hypotheses") or []
    if hyps:
        ver = {v.get("hypothesis_id"): v for v in ctx.get("verify_r0") or [] if isinstance(v, dict)}
        out.append('<section id="starting"><h2>The starting hypotheses</h2><p class="sub">What the specialists proposed '
                   "before the investigation, with the code's check of each step against the findings.</p>"
                   '<div class="tablewrap"><table><tr><th>Id</th><th>Hypothesis</th><th class="c">Steps checked</th>'
                   '<th class="c">Against the findings</th><th class="c">Withdrawn</th></tr>')
        for h in hyps:
            vv = (ver.get(h.get("id")) or {}).get("verification") or {}
            st = [x.get("status") for x in vv.get("steps") or [] if isinstance(x, dict)]
            dots = "".join(f'<i class="sdot {esc(x)}" title="{esc(x)}"></i>' for x in st)
            out.append(f"<tr><td>{esc(h.get('id'))}</td><td><b>{esc(_short_title(h.get('title'), 100))}</b>"
                       f"<details><summary>Summary</summary><p>{say((h.get('summary') or '')[:900])}</p></details></td>"
                       f'<td class="c">{dots}</td><td class="c"><span class="badge">{esc(vv.get("status", ""))}</span></td>'
                       f'<td class="c">{"yes" if h.get("withdrawn") else ""}</td></tr>')
        out.append('</table></div><p class="small">Each dot is one step of the hypothesis: <i class="sdot supported"></i> '
                   'supported, <i class="sdot partial"></i> partly, <i class="sdot unsupported"></i> not supported by the '
                   "findings.</p></section>")
    revs = circ.get("reviews") or []
    cited |= set(ID_RE.findall(json.dumps(circ.get("circuits"))))
    out.append('<section id="record"><h2>Run record</h2><p class="sub">How the PI\'s reviews of the circuits went, and '
               'every data query cited above.</p>')
    if revs:
        out.append(_review_chart(attempts, revs))
        out.append(f"<details><summary>The {len(revs)} reviews in words</summary><div class='tablewrap'><table><tr><th>#</th>"
                   "<th>Attempt</th><th>Verdict</th><th>Assessment</th><th>Demands</th></tr>"
                   + "".join(f"<tr><td>{i}</td><td class='c'>{esc(r.get('attempt', ''))}</td><td><span class='badge'>"
                             f"{esc(r.get('verdict'))}</span></td><td>{say(r.get('assessment') or r.get('text') or '')}</td>"
                             f"<td><ol>" + "".join(f"<li>{say(d.get('text') if isinstance(d, dict) else d)}</li>"
                                                   for d in r.get("demands") or []) + "</ol></td></tr>"
                             for i, r in enumerate(revs, 1)) + "</table></div></details>")
    q = sorted(x for x in cited if x in results)
    out.append(f"<details><summary>The {len(q)} data queries cited</summary><div class='tablewrap'><table><tr><th>Id</th>"
               "<th>Tool</th><th>Asked</th><th>Returned</th></tr>")
    for rid in q:
        r = results[rid]
        out.append(f'<tr class="res" id="{esc(rid)}"><td><b>{esc(rid)}</b></td><td>{esc(r.get("tool"))}</td>'
                   f'<td class="small">{esc(json.dumps(r.get("args"), ensure_ascii=False)[:200])}</td>'
                   f'<td><details><summary>{esc(" ".join(str(r.get("summary") or "").split()[:14]))}…</summary>'
                   f'<pre class="q">{esc(r.get("summary") or "")}</pre></details></td></tr>')
    out.append("</table></div></details></section>" + _credit() + "</div>" + _scripts(gs, used) + DETAILS_JS + "</body></html>")
    return "".join(out), used


# a link to a folded query or attempt opens the fold it sits in
DETAILS_JS = """<script>function openTo(h){const el=h&&document.getElementById(h.slice(1));if(!el)return;
let p=el;while(p){if(p.tagName==="DETAILS")p.open=true;p=p.parentElement}el.scrollIntoView({block:"start"})}
addEventListener("hashchange",()=>openTo(location.hash));document.addEventListener("click",e=>{const a=e.target.closest("a[href^='#']");
if(a)setTimeout(()=>openTo(a.getAttribute("href")),0)});openTo(location.hash)</script>"""


# ---- pages 3 and 4: experiments and treatments ------------------------------------------------------------------------

def render_experiments(run: Path, tabs: list[tuple[str, str, bool]], study: str, sub: str, shorts: dict) -> str:
    from mechanist.runner import RunConfig
    ctx, _ = load_run(run)
    cfg = RunConfig.from_yaml(run / "config.yaml")
    cs = ctx["circuits"]["circuits"]
    short = {c["rank"]: shorts.get(c["rank"]) or _short_title(c.get("title")) for c in cs}
    ex = ctx.get("experiment") or {}
    exps = [e for e in ex.get("experiments") or [] if "tests" in e]
    settings: dict[str, int] = {}
    for e in exps:
        k = e.get("setting") or e.get("tier") or "not stated"
        settings[k] = settings.get(k, 0) + 1
    papers = {p["pmid"] for e in exps for q in e.get("precedent") or [] for p in q.get("papers") or []}
    tiles = "".join(f'<div class="tile"><b>{n}</b><span>{esc(lab)}</span></div>' for n, lab in (
        (sum(1 for e in exps if e.get("confirms")), "check that named molecules are there"),
        (sum(1 for e in exps if (e.get("discrimination") or 0) > 0), "would tell the circuits apart"),
        (len(ex.get("unconfirmed") or []), "molecules no experiment checks yet"),
        (len(papers), "published papers they build on")))
    circs = "".join(f'<span class="badge">{r} · {esc(short[r])}</span>' for r in short)
    side = ([("The plan", "#plan")] if ex.get("programmes") else []) + [
        ("Checks", "#confirm"), ("Telling circuits apart", "#mechanism"), ("Missing links", "#missing"),
        ("By setting", "#tiers")]
    tier_rows = "".join(f"<tr><td>{esc(t)}</td><td class='c'>{n}</td></tr>" for t, n in
                        sorted(settings.items(), key=lambda kv: -kv[1]))
    return (_head(f"{cfg.study} Mechanist experiments", study, sub, tabs, side, "experiments") + '<div class="wrap" id="top">'
            '<h1>Experiments</h1>'
            '<div class="note">Designed by the experimentalist agent on the final circuits and their missing links, with '
            'their key parameters (full protocols come later). Kinds and settings are the agent\'s own. Ranked by code: '
            'first by how much the circuits disagree on the outcome, weighted by how many attempts proposed each, then by '
            'the agent\'s priority. Each builds where it can on papers the team already retrieved.</div>'
            f'<div class="tiles">{tiles}</div>'
            + ('<section id="plan"><h2>The plan, hypothesis by hypothesis</h2><p class="sub">Each hypothesis\'s '
               'experiments in the order the experimentalist would run them: first whether the molecules and cells are '
               'there, then what could rule the hypothesis out quickly, then whether the mechanism works, then whether '
               'it holds in a living system. An experiment marked "also in" serves several hypotheses.</p>'
               + _programmes_html(ex, short) + '</section>' if ex.get("programmes") else "")
            + '<section id="confirm"><h2>Checks on what the circuits rest on</h2><p class="sub">Experiments that check named '
            'molecules are really there, for the molecules the best-supported circuits rest on. The fourth column is the '
            'agent\'s note on what could make the result misleading.</p>'
            + (_confirm_html(ex) or "<p class='small'>No experiment checks a named molecule.</p>")
            + '</section><section id="mechanism"><h2>Experiments that tell the circuits apart</h2><p class="sub">Each row '
            'is an experiment; each column a circuit, with the outcome it predicts. Columns:</p>'
            f'<div class="chips" style="margin-bottom:8px">{circs}</div>{_experiments_html(ex, cs, short)}</section>'
            '<section id="missing"><h2>The missing links, and what covers each</h2><p class="sub">The agenda, built by code '
            'from the circuits: untested steps of circuits at least two attempts proposed, steps the data could not '
            'settle, and leads one attempt alone found.</p>' + _missing_html(ex).replace("<details>", "<details open>")
            + '</section><section id="tiers"><h2>By setting</h2><p class="sub">Where the agent says each experiment is '
            'done.</p><div class="tablewrap" style="max-width:520px">'
            f'<table><tr><th>Setting</th><th class="c">Experiments</th></tr>{tier_rows}</table></div></section>'
            + _credit() + "</div>" + _scripts(groups(ctx), {}) + DETAILS_JS + "</body></html>")


def render_treatments(run: Path, tabs: list[tuple[str, str, bool]], study: str, sub: str, shorts: dict) -> str:
    from mechanist.runner import RunConfig
    ctx, _ = load_run(run)
    cfg = RunConfig.from_yaml(run / "config.yaml")
    cs = ctx["circuits"]["circuits"]
    short = {c["rank"]: shorts.get(c["rank"]) or _short_title(c.get("title")) for c in cs}
    th = ctx.get("therapy") or {}
    hs = th.get("hypotheses") or []
    n_cell = sum(1 for h in hs if h.get("strategy") == "cell type")
    tiles = "".join(f'<div class="tile"><b>{n}</b><span>{esc(lab)}</span></div>' for n, lab in (
        (n_cell, "act on a whole cell type"), (len(hs) - n_cell, "act on one step"),
        (sum(1 for h in hs if h.get("boxed")), "carry a black-box warning"),
        (sum(1 for h in hs if h["level"] == "in use for this disease"), "already used for this disease")))
    targets = th.get("targets") or []
    lookups = {str(r.get("data", {}).get("gene", "")).upper(): r.get("data", {}) for r in th.get("results") or []
               if r.get("tool") == "gene_drugs" and r.get("ok")}
    trows = "".join(
        f"<tr><td><span class='gene'>{esc(t['gene'])}</span></td><td>{''.join(f'<span class=badge>{r}</span>' for r in t['circuits'])}</td>"
        f"<td class='c'>{t['support']}</td><td>{_mchips(t.get('untested') or [], warn=True) or '—'}</td>"
        f"<td class='small'>{esc(', '.join(d['drug'].title() for d in (lookups.get(t['gene']) or {}).get('drugs', [])[:5]) or 'none listed')}</td></tr>"
        for t in targets)
    crows = "".join(
        f"<tr><td><b>{esc(c['lineage'])}</b></td><td>{''.join(f'<span class=badge>{r}</span>' for r in c['circuits'])}</td>"
        f"<td>{''.join(f'<span class=gene>{esc(g)}</span> ' for g in c['genes'])}</td><td class='small'>"
        + esc("; ".join(f"{g}: " + (", ".join(d['drug'].title() for d in (lookups.get(g.upper()) or {}).get('drugs', [])[:3])
                                     or "none") for g in c["genes"])) + "</td></tr>" for c in th.get("cell_targets") or [])
    objs = th.get("objections") or []
    side = [("Whole cell type", "#blunt"), ("One step", "#targeted"), ("Pathways", "#pathways"),
            ("Cell-type targets", "#cells"), ("Step targets", "#targets")]
    return (_head(f"{cfg.study} Mechanist treatments", study, sub, tabs, side, "treatments") + '<div class="wrap" id="top">'
            '<h1>Therapeutic hypotheses</h1>'
            f'<div class="note">{NOT_ADVICE} Proposed by the pharmacologist agent from the drug lookups; whether a '
            f'drug acts on a whole cell type or on one step is set by code (is its target one of the data\'s markers for '
            f'a cell type the circuits use). The clinician weighs certainty of effect against safety.</div>'
            f'<div class="tiles">{tiles}</div>'
            '<section id="blunt"><h2>Acting on a whole cell type the circuits use</h2><p class="sub">The target is one of '
            'the data\'s own markers for that cell type. The robustness bar is the share of the circuits\' support the '
            'drug acts in: one that removes a cell type acts in every circuit that uses it.</p>'
            + (_therapy_html(th, short, strategy="cell type") or "<p class='small'>None proposed.</p>")
            + '</section><section id="targeted"><h2>Acting on one step</h2><p class="sub">The target is a gene on a '
            'circuit step. Works only if that step is right: see "Test first".</p>'
            + (_therapy_html(th, short, strategy="step") or "<p class='small'>None proposed.</p>")
            + (f"<details><summary>{len(objs)} code checks on the model's suggestions</summary><ul class='prose'>"
               + "".join(f"<li>{esc(o.get('text') if isinstance(o, dict) else o)}</li>" for o in objs) + "</ul></details>"
               if objs else "")
            + f'</section><section id="pathways">{_pathway_html(th)}</section>'
            + ('<section id="cells"><h2>Cell-type targets</h2><p class="sub">Each cell type the circuits use, its markers in '
               'this data (genes in most of its cells and few others), and drugs acting on them.</p><div class="tablewrap">'
               f'<table><tr><th>Cell type</th><th>Circuits</th><th>Markers</th><th>Drugs listed</th></tr>{crows}</table></div>'
               '</section>' if crows else "")
            + '<section id="targets"><h2>Step targets: the circuit genes, and their drugs</h2><p class="sub">Every target '
            'was looked up by code before any model call. Support: summed share of attempts behind the circuits that '
            'carry the gene.</p><div class="tablewrap"><table><tr><th>Gene</th><th>Circuits</th><th class="c">'
            f'Support</th><th>Untested links</th><th>Drugs listed</th></tr>{trows}</table></div></section>'
            + _credit() + "</div>" + _scripts(groups(ctx), {}) + DETAILS_JS + "</body></html>")


# ---- the plain-language pass (display only): every rewrite must keep the original's numbers, genes and ids ------------

NOT_NAMES = {"UP", "DOWN", "NOT", "AND", "OR", "VS", "DE", "PI", "RNA", "DNA", "ID", "IDS", "NA", "OK"}


def plain_check(orig: str, new: str, known: frozenset = frozenset()) -> str:
    """'' if the rewrite keeps every number, gene-like name and id of the original and adds none; else why not. known:
    names the rewrite may add because the study already uses them everywhere (its group and disease names)."""
    skip = NOT_NAMES | {cellfig.norm(k) for k in known}

    def toks(t: str) -> tuple[set, set, set]:
        ids = set(ID_RE.findall(t))
        names = {cellfig.norm(g) for g in cellfig.GENE_RE.findall(t) if g not in ids} - NOT_NAMES
        return (set(NUM_RE.findall(t)), names, ids)
    a, b = toks(orig), toks(new)
    b = (b[0], b[1] - (skip - a[1]), b[2])  # adding a known study name is fine; dropping any name is not
    for name, x, y in zip(("numbers", "names", "ids"), a, b):
        if x != y:
            return f"{name} changed: lost {sorted(x - y)[:5]}, added {sorted(y - x)[:5]}"
    if len(new) > len(orig) * 1.4 + 40:
        return "longer than the original"
    return ""


def _page_texts(page: dict) -> list[str]:
    out = [page.get("answer", "")] + [p.get("text", "") for p in page.get("points") or []]
    for c in page.get("circuits") or []:
        out += [c.get("one_line", "")] + [k.get("why", "") for k in c.get("cards") or []]
    out += [t.get(k, "") for t in page.get("tests") or [] for k in ("test", "expects", "decides")]
    return out + list(page.get("checked") or [])


def _run_texts(ctx: dict) -> list[str]:
    circ = ctx.get("circuits") or {}
    out = [circ.get("ranking", "")]
    for c in (circ.get("circuits") or []) + [c for a in circ.get("attempts") or [] for c in a.get("circuits") or []]:
        out += [c.get(k, "") for k in ("summary", "root", "confidence", "analogue")] + list(c.get("loops") or [])
        out += list(c.get("unused") or []) + [e.get(k, "") for e in c.get("edges") or [] for k in ("via", "note")]
        out += [p.get("how", "") for p in c.get("phenotype") or [] if isinstance(p, dict)]
        out += [p.get(k, "") for p in c.get("predictions") or [] if isinstance(p, dict) for k in ("test", "expects")]
    out += [x.get("text", "") for x in circ.get("leads") or [] if isinstance(x, dict)]
    syn = ((ctx.get("revise") or ctx.get("synthesize") or {}).get("synthesis")) or {}
    out += [syn.get("summary", "")] + [lk.get("note", "") for lk in syn.get("links") or []]
    for alt in syn.get("alternatives") or []:
        if isinstance(alt, dict):
            out += [v for k, v in alt.items() if k not in ("rank", "title") and isinstance(v, str)]
    return out


def plain_pass(out: Path, texts: list[str], llm=None, batch_chars: int = 3500, system: str | None = None,
               cache: str = "plain.json", model: str = "haiku", tag: str = "plain") -> dict[str, str]:
    """Rewrite each text in plain prose (agents.PLAIN_SYSTEM, Haiku), keep a rewrite only if plain_check passes. Cached
    in out/plain.json by text, so a rebuild costs nothing."""
    import hashlib
    from concurrent.futures import ThreadPoolExecutor

    from mechanist.llm import LLMConfig, LoggedLLM, make_backend, parse_json
    path = out / cache
    done: dict = json.loads(path.read_text()) if path.exists() else {}
    key = lambda t: hashlib.sha256(t.encode()).hexdigest()[:20]
    todo, seen = [], set()
    for t in texts:
        t = str(t or "").strip()
        prev = done.get(key(t))
        # a failed call (not a rewrite the check refused) is tried again on the next build
        retry = prev and not prev.get("ok") and not re.match(r"(numbers|names|ids) changed|longer than|no rewrite",
                                                             str(prev.get("why"))) and not prev.get("retried")
        if len(t) >= 40 and (prev is None or retry) and t not in seen:
            seen.add(t)
            todo.append(t)
    if todo:
        if llm is None:
            (out / f"{tag}_calls").mkdir(parents=True, exist_ok=True)
            llm = LoggedLLM(make_backend(LLMConfig(backend="anthropic", model=model, external_ok=True, effort="",
                                                   max_tokens=16000)), out / f"{tag}_calls")
        batches, cur, size = [], [], 0
        for t in todo:
            if cur and size + len(t) > batch_chars:
                batches.append(cur)
                cur, size = [], 0
            cur.append(t)
            size += len(t)
        batches += [cur] if cur else []

        def one(batch: list[str]) -> list[tuple[str, dict]]:
            body = "\n\n".join(f"### n={i}\n{t}" for i, t in enumerate(batch))
            try:
                items = {int(x.get("n", -1)): str(x.get("text", "")) for x in
                         (parse_json(llm.complete(tag, system or agents.PLAIN_SYSTEM, body)) or {}).get("items") or []
                         if isinstance(x, dict)}
            except Exception as exc:  # noqa: BLE001 - a failed batch leaves its originals showing
                return [(t, {"ok": False, "why": str(exc)[:200]}) for t in batch]
            res = []
            for i, t in enumerate(batch):
                new = items.get(i, "").strip()
                why = plain_check(t, new) if new else "no rewrite returned"
                res.append((t, {"ok": not why, "text": new, "why": why, "retried": key(t) in done}))
            return res

        with ThreadPoolExecutor(4) as ex:
            for res in ex.map(one, batches):
                for t, r in res:
                    done[key(t)] = r
        path.write_text(json.dumps(done, indent=1, ensure_ascii=False))
    return {str(t).strip(): done[key(str(t).strip())]["text"] for t in texts
            if str(t or "").strip() and done.get(key(str(t).strip()), {}).get("ok")}


def _plain_page(page: dict, plain: dict[str, str]) -> dict:
    """A copy of the short page with its prose in the plain rewrite (headings, ids and panels untouched)."""
    p = json.loads(json.dumps(page))
    f = lambda t: plain.get(str(t or "").strip(), t)
    p["answer"] = f(p.get("answer"))
    for x in p.get("points") or []:
        x["text"] = f(x.get("text"))
    for c in p.get("circuits") or []:
        c["one_line"] = f(c.get("one_line"))
        for k in c.get("cards") or []:
            k["why"] = f(k.get("why"))
    for t in p.get("tests") or []:
        for k in ("test", "expects", "decides"):
            t[k] = f(t.get(k))
    p["checked"] = [f(x) for x in p.get("checked") or []]
    return p


def build(run: Path, out: Path, tabs: list[tuple[str, str]] | None = None, study: str = "", sub: str = "",
          llm=None, plain: bool = True, reuse_page: bool = False) -> dict:
    """Write interpretation.json, interpretation.html and mechanism.html to out; returns the built page record.
    reuse_page: take the short page's words from out/interpretation.json as last written (no model call), for a
    change to the layout only."""
    out.mkdir(parents=True, exist_ok=True)
    if reuse_page and (out / "interpretation.json").exists():
        page = json.loads((out / "interpretation.json").read_text())
        page.pop("errors", None)
        ctx, results = load_run(run)
        want = cited_ids(ctx["circuits"]["circuits"])
        built = {"page": page, "errors": [], "panels": {f"{r}|{g}|{lin}": v for (r, g, lin), v in panel_data(results).items()
                                                         if r in want}}
    else:
        built = write_page(run, llm)
    if not isinstance(built.get("page"), dict):
        raise SystemExit("no page was written: " + "; ".join(built.get("errors") or ["no reply"]))
    (out / "interpretation.json").write_text(json.dumps(built["page"] | {"errors": built["errors"]}, indent=1, ensure_ascii=False))
    if built["errors"]:
        raise SystemExit("the page failed its check twice: " + "; ".join(built["errors"]))
    tabs = tabs or [("Results", "../index.html")]
    PLAIN.clear()
    if plain:
        ctx, _ = load_run(run)
        PLAIN.update(plain_pass(out, _page_texts(built["page"]) + _run_texts(ctx)))
        built = {**built, "page": _plain_page(built["page"], PLAIN)}
    ctx_all, _ = load_run(run)
    has_ex = any("tests" in e for e in (ctx_all.get("experiment") or {}).get("experiments") or [])
    has_th = bool((ctx_all.get("therapy") or {}).get("hypotheses"))
    global PAGES
    PAGES = [p for p in PAGES_ALL if p[0] in ("summary", "mechanism") or (p[0] == "experiments" and has_ex)
             or (p[0] == "treatments" and has_th)]
    t1 = [(t, h, False) for t, h in tabs] + [("Mechanist interpretation", "interpretation.html", True)]
    (out / "interpretation.html").write_text(render_interpretation(run, built, t1, study, sub))
    shorts = {int(c.get("rank") or 0): c.get("short_title") for c in (built["page"] or {}).get("circuits") or []}
    page, _ = render_mechanism(run, [(t, h, False) for t, h in tabs] + [("Mechanist interpretation", "interpretation.html", True)],
                               study, sub, shorts)
    (out / "mechanism.html").write_text(page)
    tabs_m = [(t, h, False) for t, h in tabs] + [("Mechanist interpretation", "interpretation.html", True)]
    for key, fname, render in (("experiments", "experiments.html", render_experiments),
                               ("treatments", "treatments.html", render_treatments)):
        if any(p[0] == key for p in PAGES):
            (out / fname).write_text(render(run, tabs_m, study, sub, shorts))
        elif (out / fname).exists():
            (out / fname).unlink()
    return built
