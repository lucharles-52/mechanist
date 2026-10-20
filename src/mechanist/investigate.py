"""Investigation bookkeeping (no LLM): which questions must be asked, whether they were, and whether answers and
synthesis links cite evidence that exists.

The investigator decides what to ask; code decides what it may not skip. For every hypothesis step the run must ask
why the target cell is there, what it is doing, and why the source cell makes the mediator; for every hypothesis, how
its steps relate, how it explains the disease, whether it is specific, and whether it could be an artifact.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable

from mechanist.records import Hypothesis, Objection, Question, Requirement, Synthesis, ToolResult
from mechanist.verify import CELL_FAMILIES, cell_family

RESOLVED = {"answered", "partial", "unanswerable_with_data", "needs_literature"}
ID_RE = re.compile(r"^([FR])0*(\d+)$")


KIND_SYNONYMS = {
    "expansion": "presence", "abundance": "presence", "recruitment": "presence", "composition": "presence",
    "lineage_share": "presence",
    "output": "output", "secretion": "output", "secreted": "output", "effector": "output", "product": "output",
    "products": "output", "effector_output": "output",
    "state": "function", "phenotype": "function", "program": "function", "markers": "function",
    "state_markers": "function", "state_composition": "function", "gene_by_state": "function",
    "program_score": "function",
    "induction": "upstream", "inducer": "upstream", "source": "upstream", "mechanism": "upstream",
    "incoming_signals": "upstream", "ligand_targets": "upstream",
    "covariation": "relation", "correlation": "relation", "coexpression": "relation", "covary": "relation",
    "co-expression": "relation", "link": "relation",
    "disease": "disease_link", "pathology": "disease_link", "phenotype_link": "disease_link",
    "specific": "specificity", "comparison": "specificity",
    "batch": "artifact", "confound": "artifact", "contamination": "artifact", "artefact": "artifact",
}


def normalize_kind(kind) -> str:
    """Models sometimes put a tool name or a synonym in 'kind'; map it to a question type (else 'other')."""
    k = str(kind or "").strip().lower().replace(" ", "_")
    if k in {"presence", "function", "output", "upstream", "relation", "disease_link", "specificity", "artifact"}:
        return k
    return KIND_SYNONYMS.get(k, "other")


def normalize_id(x: str) -> str:
    """'F4' / 'F004' -> 'F0004' (zero padding only; anything else is returned unchanged)."""
    m = ID_RE.match(str(x).strip())
    return f"{m.group(1)}{int(m.group(2)):04d}" if m else str(x).strip()


OUTPUT_TEXT = ("What do {cell} cells make that acts on other cells or on the tissue (secreted proteins, antibodies, cytokines, "
               "chemokines, proteases, ligands), compared with the comparators, and which "
               "cells carry the receptor or are affected?")


def requirements(hyps: list[Hypothesis], expanded: list[str] | None = None) -> list[Requirement]:
    """Questions the investigation must ask. expanded: cell types expanded in the disease (composition findings);
    each, like every cell type in a hypothesis, must have its output examined (what it makes, who receives it)."""
    reqs: list[Requirement] = []
    seen: set[tuple] = set()

    index: dict[tuple, Requirement] = {}

    def add(kind, text, cell=None, gene=None, hid=None, cells=(), priority=3):
        key = (kind, cell_family(cell) if cell else None, gene.upper() if gene else None, hid)
        if key in index:  # the same question raised again by a more important hypothesis keeps the higher priority
            index[key].priority = min(index[key].priority, priority)
            return
        seen.add(key)
        index[key] = Requirement(kind=kind, cell=cell, gene=gene, hypothesis_id=hid, cells=list(cells), text=text,
                                 priority=priority)
        reqs.append(index[key])

    mech = [h for h in hyps if not h.non_mechanistic and not h.withdrawn]
    for h in mech:
        starting = not h.id.startswith("L")  # literature-scout leads rank below the starting hypotheses
        hc = sorted({c for s in h.steps for c in (s.source_cell, s.target_cell) if c})
        key_step = next((s for s in h.steps if s.genes), h.steps[0] if h.steps else None)
        for s in h.steps:
            gene = s.genes[0] if s.genes else None
            target = s.target_cell or s.source_cell
            pr = (1 if starting else 2) if s is key_step else 3
            add("presence", f"Why are {target} cells present/expanded?", cell=target, priority=pr)
            add("function", f"What are the {target} cells doing there (state, effector program)?", cell=target,
                priority=pr)
            for c in {s.source_cell, s.target_cell} - {None, ""}:
                add("output", OUTPUT_TEXT.format(cell=c), cell=c, priority=1)
            if gene:
                add("upstream", f"Why do {s.source_cell} cells make {gene}? What induces it, and who sends that "
                    f"signal?", cell=s.source_cell, gene=gene, priority=pr)
        hp = 2 if starting else 3
        if len(h.steps) >= 2:
            add("relation", f"{h.id}: are its steps one program (same cells/samples) or separate?", hid=h.id, cells=hc,
                priority=hp)
        add("disease_link", f"{h.id}: how does this explain the disease phenotype?", hid=h.id, cells=hc, priority=hp)
        add("specificity", f"{h.id}: is this specific to {h.disease}, or shared with the other disease?", hid=h.id,
            cells=hc, priority=hp)
        add("artifact", f"{h.id}: could few cells, one dominant sample, contamination or cross-study controls explain "
            f"it?", hid=h.id, cells=hc, priority=hp)
    for c in expanded or []:
        add("output", OUTPUT_TEXT.format(cell=c), cell=c, priority=1)
    if len(mech) >= 2:
        add("relation", "Are the hypotheses connected (shared cells, programs, feed-forward loops) or independent?",
            hid="*", priority=2)
    return sorted(reqs, key=lambda r: r.priority)


def satisfies(q: Question, r: Requirement) -> bool:
    if q.kind != r.kind:
        return False
    if r.cell and not any(cell_family(c) == cell_family(r.cell) for c in q.cells):
        return False
    if r.gene and r.gene.upper() not in {g.upper() for g in q.genes} and r.gene.upper() not in q.text.upper():
        return False
    if r.hypothesis_id == "*":
        return len(set(q.hypothesis_ids)) >= 2 or "*" in q.hypothesis_ids
    if not r.hypothesis_id or r.hypothesis_id in q.hypothesis_ids or r.hypothesis_id in q.text:
        return True
    # a hypothesis-level question about the hypothesis's own cells counts too
    fams = {cell_family(c) for c in r.cells}
    return bool(fams) and any(cell_family(c) in fams for c in q.cells)


def _subtree(q: Question, qs: list[Question]) -> list[Question]:
    kids: dict[str, list[Question]] = {}
    for x in qs:
        if x.parent_id:
            kids.setdefault(x.parent_id, []).append(x)
    out, todo, seen = [], [q], set()
    while todo:
        x = todo.pop()
        if x.id in seen:
            continue
        seen.add(x.id)
        out.append(x)
        todo += kids.get(x.id, [])
    return out


def inducer_tested(q: Question, qs: list[Question], results: dict[str, ToolResult], mediator: str | None) -> bool:
    """An upstream question only counts once a candidate inducer (a gene other than the mediator) was profiled across
    all lineages (gene_profile without a lineage), in this question or a follow-up of it."""
    med = (mediator or "").upper()
    for x in _subtree(q, qs):
        for rid in x.result_ids:
            r = results.get(rid)
            if r is None or not r.ok or r.tool != "gene_profile" or r.args.get("lineage"):
                continue
            genes = r.args.get("gene")
            genes = genes if isinstance(genes, list) else [genes]
            if any(str(g).upper() != med for g in genes if g):
                return True
    return False


def _genes(arg) -> list[str]:
    vals = arg if isinstance(arg, list) else [arg]
    return [str(v).upper() for v in vals if v]


def _profiles(r: ToolResult) -> list[tuple[str, str, dict]]:
    """(gene, lineage, {sample: % expressing}) from a gene_profile result, single gene or list."""
    genes = _genes(r.args.get("gene"))
    data = r.data or {}
    per_gene = data if isinstance(r.args.get("gene"), list) else {genes[0] if genes else "": data}
    out = []
    for g, lins in per_gene.items():
        if not isinstance(lins, dict):
            continue
        for lin, vals in lins.items():
            if isinstance(vals, dict) and any(isinstance(v, (int, float)) for v in vals.values()):
                out.append((str(g).upper(), str(lin), {k: v for k, v in vals.items() if isinstance(v, (int, float))}))
    return out


def _vals(per_sample: dict, samples: list[str]) -> list[float]:
    return [per_sample[s] for s in samples if per_sample.get(s) is not None]


def receptor_ok(per_sample: dict, groups: dict[str, list[str]] | None, case: str | None, reference: str | None,
                min_pct: float) -> bool:
    """A receptor counts on a cell if every disease sample expresses it (>= min_pct) and, given groups, the disease
    average is above the reference-group average. (Full separation would be too strict for receptors: LTBR is on
    46-58% of PG fibroblasts and 15-49% of control fibroblasts.)"""
    if not groups or not case:
        return max(per_sample.values(), default=0) >= min_pct
    cv = _vals(per_sample, groups.get(case, []))
    if len(cv) < 2 or min(cv) < min_pct:
        return False
    rv = _vals(per_sample, groups.get(reference or "", []))
    return not rv or sum(cv) / len(cv) > sum(rv) / len(rv)


# Samples a group needs before "every sample of one group beyond every sample of another" is said. A cohort needs two,
# so the claim is a consistency; an index-patient study is one patient against single reference samples by design, so
# one is enough and every such claim is per sample. The runner sets it from the study design.
MIN_N = 2


def ligand_differs(per_sample: dict, groups: dict[str, list[str]] | None, case: str | None) -> bool:
    """A ligand counts if, in some lineage, every disease sample is above every sample of at least one other group."""
    if not groups or not case:
        return True
    cv = _vals(per_sample, groups.get(case, []))
    if len(cv) < MIN_N:
        return False
    return any(len(rv := _vals(per_sample, ss)) >= MIN_N and min(cv) > max(rv) for g, ss in groups.items() if g != case)


# What counts as a TEST of a candidate inducer: evidence that can discriminate between candidates, in the cell that
# is supposed to respond. Descriptive calls (who makes the ligand, which state) are useful but test nothing.
FOOTPRINT_TOOLS = {"pathway_footprint", "program_score"}
VERDICTS = {"supported", "weakened", "rejected"}
NULL_MAX = 0.25  # a correlation matched by >= 25% of disease-up genes does not discriminate


def _thread(q: Question, qs: list[Question]) -> list[Question]:
    """The question, its ancestors and its follow-ups: where a candidate's test may have been run."""
    by_id = {x.id: x for x in qs}
    up, x = [], q
    while x.parent_id and x.parent_id in by_id and by_id[x.parent_id] not in up:
        x = by_id[x.parent_id]
        up.append(x)
    return [*up, *_subtree(q, qs)]


def _mentions(q: Question, gene: str, cell: str | None) -> bool:
    g = gene.upper()
    if g not in {x.upper() for x in q.genes} and not re.search(rf"\b{re.escape(g)}\b", q.text.upper()):
        return False
    return cell is None or not q.cells or any(cell_family(c) == cell_family(cell) for c in q.cells)


def test_validity(r: ToolResult, keys: set[str], gene: str, cell: str | None) -> tuple[bool, str]:
    """Does this tool result test a candidate (keys = ligand + receptors) for `gene` in `cell`? (ok, why not)."""
    named = set(re.findall(r"[A-Z0-9][A-Z0-9-]*", json.dumps(r.args).upper()))
    in_cell = cell is None or cell_family(str(r.args.get("lineage", ""))) == cell_family(cell)
    if r.tool in FOOTPRINT_TOOLS:
        return (True, "") if in_cell else (False, f"{r.tool} in {r.args.get('lineage')}, not in the receiving cell")
    if r.tool == "coexpression":
        if not keys & named:
            return False, "coexpression does not include the candidate's receptor or ligand"
        where = f"coexpression in {r.args.get('lineage')}, not in the receiving cell {cell}"
        return (True, "") if in_cell else (False, where)
    if r.tool == "covary":
        if not (keys & named and gene.upper() in named):
            return False, "covary does not pair the candidate with the induced gene"
        null = (r.data or {}).get("null") or {}
        if null and null.get("frac", 0) >= NULL_MAX:
            return False, f"covary does not discriminate ({null['frac']:.0%} of disease-up genes correlate as well)"
        within = [v for v in ((r.data or {}).get("rho_within") or {}).values() if isinstance(v, (int, float)) and not math.isnan(v)]
        if within and max(within) <= 0:
            return False, "covary is driven by group differences only (no positive within-group correlation)"
        return True, ""
    return False, f"{r.tool} is descriptive, not a test of induction"


def judged_candidates(qs: list[Question], results: dict[str, ToolResult], candidates: set[str],
                      rec_ligands: dict[str, set[str]], gene: str | None = None,
                      cell: str | None = None) -> tuple[list[dict], list[dict]]:
    """Verdicts on candidate inducers OF THIS GENE IN THIS CELL that rest on a valid test (test_validity) in the
    verdict question's thread. Returns (judged, unbacked): unbacked verdicts carry the reasons their tests failed."""
    judged, unbacked = [], []
    own = [q for q in qs if gene is None or _mentions(q, gene, cell)]
    for q in own:
        for c in q.candidates:
            parts = {p for p in re.split(r"[_/\s,+]+", c.ligand.upper()) if p}
            if c.verdict not in VERDICTS or not parts & candidates:
                continue
            keys = parts | {r for r, ligs in rec_ligands.items() if parts & ligs} | ({c.receptor.upper()} - {""})
            keys |= {p for p in re.split(r"[_/\s,+()]+", c.receptor.upper()) if p}
            tests, why = [], []
            for x in _thread(q, qs):
                for rid in x.result_ids:
                    r = results.get(rid)
                    if r is None or not r.ok or r.tool == "upstream_screen":
                        continue
                    ok, reason = test_validity(r, keys, gene or "", cell)
                    (tests.append(r.id) if ok else why.append(f"{r.id}: {reason}"))
            row = {"ligand": c.ligand, "sender": c.sender, "receptor": c.receptor, "verdict": c.verdict,
                   "reason": c.reason, "question": q.id, "tests": tests}
            if tests:
                judged.append(row)
            else:
                unbacked.append({**row, "invalid_tests": why[:4]})
    return judged, unbacked


def chain_status(gene: str | None, results: list[ToolResult], cell: str | None = None,
                 min_pct: float = 5.0, groups: dict[str, list[str]] | None = None, case: str | None = None,
                 reference: str | None = None, qs: list[Question] | None = None) -> dict:
    """The upstream chain for a mediator gene made by `cell`, from the whole investigation's tool results:
    1) known_regulators(gene) -> activating TFs;
    2) receptor_paths(one of those TFs) -> receptors and their ligands;
    3) one of those receptors shown on `cell` (receptor_ok: in every disease sample, above the reference average);
    4) a ligand OF THAT RECEPTOR profiled across ALL lineages that differs in the disease (ligand_differs: all
       disease samples above every sample of another group, in some lineage) -> who sends the signal;
    5) (given qs) a candidate judged: a verdict on one such ligand, backed by a test beyond the screen.
    upstream_screen(gene, cell) performs steps 1-4 at once. Steps 1-4 find what is possible; step 5 is the thinking.
    Returns which steps are done and what to do next."""
    g = (gene or "").upper()
    ok = [r for r in results if r.ok]
    fam = cell_family(cell) if cell else None
    screens = [r for r in ok if r.tool == "upstream_screen" and str(r.args.get("gene", "")).upper() == g
               and (fam is None or cell_family(str(r.args.get("cell", ""))) == fam)]
    if any(r.data.get("premise") is False for r in screens):
        # the gene is not elevated in this cell: the question is answered by refuting its premise
        return {"complete": True, "premise_failed": True, "steps": {"premise": False}, "tfs": [], "routed_tfs": [],
                "receptors_on_cell": [], "ligands_profiled": [], "ligands_without_difference": [], "judged": [],
                "unbacked": [], "next": ""}
    tfs: set[str] = set()
    for r in ok:
        if r.tool == "known_regulators" and str(r.args.get("gene", "")).upper() == g:
            tfs |= {t.upper() for t in r.data.get("activating_tfs", [])}
    rec_ligands: dict[str, set[str]] = {}
    routed: list[str] = []
    for r in ok:
        if r.tool == "receptor_paths" and str(r.args.get("tf", "")).upper() in tfs:
            routed.append(str(r.args.get("tf")).upper())
            for route in r.data.get("routes", []):
                rec = str(route.get("receptor", "")).upper()
                for lig in route.get("ligands", []):
                    rec_ligands.setdefault(rec, set()).update(str(lig).upper().split("_"))
    screened: dict[str, dict] = {}
    screen_on: set[str] = set()
    screen_flat: set[str] = set()
    for r in screens:
        tfs |= {t.upper() for t in r.data.get("tfs", [])}
        routed.append(f"screen:{r.id}")
        for route in r.data.get("routes", []):
            rec_ligands.setdefault(str(route["receptor"]).upper(), set()).update(x.upper() for x in route["ligands"])
        screen_on |= {x.upper() for x in r.data.get("receptors_on_cell", [])}
        for d in r.data.get("candidates", []):
            screened.setdefault(d["ligand"].upper(), d)
        screen_flat |= {x.upper() for x in r.data.get("ligands_without_difference", [])}
    profiles = [p for r in ok if r.tool == "gene_profile" for p in _profiles(r)]
    on_cell = sorted({gname for gname, lin, vals in profiles if gname in rec_ligands
                      and (fam is None or cell_family(lin) == fam)
                      and receptor_ok(vals, groups, case, reference, min_pct)} | screen_on)
    ligands = set().union(*(rec_ligands[x] for x in on_cell)) if on_cell else set()
    everywhere = [r for r in ok if r.tool == "gene_profile" and not r.args.get("lineage")]
    looked = sorted(({x for r in everywhere for x in _genes(r.args.get("gene"))} | screen_flat) & ligands)
    profiled = sorted({gname for r in everywhere for gname, _lin, vals in _profiles(r)
                       if gname in ligands and ligand_differs(vals, groups, case)} | set(screened))
    steps = {"regulators": bool(tfs), "routes": bool(routed), "receptor_on_cell": bool(on_cell),
             "ligand_profiled": bool(profiled)}
    judged: list[dict] = []
    unbacked: list[dict] = []
    if qs is not None:
        # any ligand shown to differ in the disease may be judged, including one the investigator brought from
        # knowledge outside the curated routes
        differing = set(profiled) | {gname for r in everywhere for gname, _lin, vals in _profiles(r)
                                     if ligand_differs(vals, groups, case)}
        judged, unbacked = judged_candidates(qs, {r.id: r for r in ok}, differing, rec_ligands, gene, cell)
        steps["candidate_judged"] = bool(judged)
    where = f"{cell} cells" if cell else "the producing cells"
    top = [f"{d['ligand']}@{d['sender']}" for d in sorted(screened.values(), key=lambda d: (
        d.get("kind") != "paracrine", not d.get("receptor_on_cell", True), -len(d.get("separates_from", []))))][:5]
    if not tfs:
        nxt = (f"call upstream_screen(gene='{gene}', cell='{cell or ''}') to screen curated inducers against the data "
               f"(or known_regulators -> receptor_paths step by step)")
    elif not routed:
        nxt = (f"call receptor_paths on an activating TF of {gene} ({', '.join(sorted(tfs)[:8])}) to find which "
               f"receptor signals reach it")
    elif not on_cell:
        nxt = (f"check which of the route receptors {where} express (gene_profile on the receptors with "
               f"lineage='{cell or ''}'): {', '.join(sorted(rec_ligands)[:10])}")
    elif not profiled:
        nxt = (f"profile across ALL lineages (gene_profile without a lineage) the ligands of the receptors {where} "
               f"express — " + "; ".join(f"{x}: {', '.join(sorted(rec_ligands[x])[:6])}" for x in on_cell[:4])
               + " — to find which cells send a signal that differs in the disease"
               + (f" (already profiled without a disease difference: {', '.join(looked)})" if looked else ""))
    elif qs is not None and not judged:
        nxt = (("verdicts so far rest on no valid test: " + "; ".join(
                   f"{u['ligand']} ({u['question']}): {', '.join(u['invalid_tests']) or 'no test run'}"
                   for u in unbacked[:3]) + ". ") if unbacked else "") + (f"JUDGE the candidates (do not sweep): pick the 2-4 most plausible inducers of {gene} in {where}"
               + (f" (screen leaders: {', '.join(top)})" if top else f" ({', '.join(profiled[:6])})")
               + f", say why each is plausible or not, and TEST them in a way that can tell them apart: "
                 f"pathway_footprint in {cell or 'the producing cell'} for each candidate's pathway, coexpression in "
                 f"{cell or 'the producing cell'} of the receptor with {gene}, covary only if its null is low; record "
                 f"a verdict for each in 'candidates' (supported / weakened / rejected)")
    else:
        nxt = ""
    return {"complete": all(steps.values()), "steps": steps, "tfs": sorted(tfs), "routed_tfs": routed,
            "receptors_on_cell": on_cell, "ligands_profiled": profiled, "ligands_without_difference":
            sorted(set(looked) - set(profiled)), "judged": judged, "unbacked": unbacked, "next": nxt}


def _upstream_done(q: Question, qs: list[Question], by_id: dict[str, ToolResult], gene: str | None,
                   results: list[ToolResult], strict: bool, cell: str | None = None, ctx: dict | None = None) -> bool:
    return (chain_status(gene, results, cell, **(ctx or {}), qs=qs)["complete"] if strict
            else inducer_tested(q, qs, by_id, gene))


def coverage(reqs: list[Requirement], qs: list[Question], results: list[ToolResult] | None = None,
             strict_chain: bool = False, chain_ctx: dict | None = None) -> list[dict]:
    """Which required questions were asked and resolved. An upstream requirement is resolved only when its chain is
    complete (strict_chain, used when the curated knowledge tools are available) or, without those tools, when a
    candidate inducer was profiled across lineages."""
    res = results or []
    by_id = {r.id: r for r in res}
    out = []
    for r in reqs:
        asked = [q.id for q in qs if satisfies(q, r)]
        resolved = [q.id for q in qs if satisfies(q, r) and q.status in RESOLVED
                    and (r.kind != "upstream" or results is None
                         or _upstream_done(q, qs, by_id, r.gene, res, strict_chain, r.cell, chain_ctx))]
        row = {"requirement": r.model_dump(), "asked_by": asked, "resolved_by": resolved}
        if r.kind == "upstream" and results is not None and strict_chain:
            row["chain"] = chain_status(r.gene, res, r.cell, **(chain_ctx or {}), qs=qs)
        out.append(row)
    return out


def untested_upstream(reqs: list[Requirement], qs: list[Question], results: list[ToolResult],
                      strict_chain: bool = False, chain_ctx: dict | None = None) -> list[tuple[Requirement, str]]:
    """Upstream requirements that were asked but whose chain is not complete, with the next step to take."""
    by_id = {r.id: r for r in results}
    out = []
    for r in reqs:
        if r.kind != "upstream" or not any(satisfies(q, r) for q in qs):
            continue
        if strict_chain:
            st = chain_status(r.gene, results, r.cell, **(chain_ctx or {}), qs=qs)
            if not st["complete"]:
                out.append((r, st["next"]))
        elif not any(satisfies(q, r) and inducer_tested(q, qs, by_id, r.gene) for q in qs):
            out.append((r, "check a candidate inducer across all lineages (gene_profile without a lineage)"))
    return out


_DASHES = str.maketrans({"\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2212": "-", "\u2010": "-"})
_IDS = re.compile(r"\b[FRQ]\d+\b")
_CLAIMED = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?=\s*%)|(?<![\w.])-?\d+\.\d+(?![\w.])")


def untraceable_numbers(text: str, sources: list[str]) -> list[str]:
    """Percentages and decimal numbers in an answer that do not appear (to the stated precision) in the data it
    cites. Catches values the model made up, e.g. when its tool calls failed."""
    src = _IDS.sub(" ", " ".join(sources).translate(_DASHES))
    values = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", src)]
    out = []
    for tok in _CLAIMED.findall(_IDS.sub(" ", text.translate(_DASHES))):
        v = abs(float(tok))
        d = len(tok.split(".")[1]) if "." in tok else 0
        tol = 0.5 * 10 ** -d + 1e-9
        if not any(abs(x - v) <= tol for x in values):
            out.append(tok)
    return list(dict.fromkeys(out))


def missing(reqs: list[Requirement], qs: list[Question]) -> list[Requirement]:
    return [r for r in reqs if not any(satisfies(q, r) for q in qs)]


def answer_matches(q: Question, answer_ids: list[str], text: str, ok_results: set[str]) -> bool:
    """Does an answer belong to this question? If the question's own tool calls returned data, the answer must cite
    at least one of those results; otherwise it must mention one of the question's genes or cell types. Catches
    answers the model filed under the wrong question id."""
    own = [r for r in q.result_ids if r in ok_results]
    if own:
        return bool(set(own) & {normalize_id(i) for i in answer_ids})
    low = text.lower()
    if any(g.lower() in low for g in q.genes):
        return True
    for c in q.cells:
        keys = CELL_FAMILIES.get(cell_family(c) or "", (c.lower(),))
        if any(k in low for k in keys):
            return True
    return not q.genes and not q.cells


def check_answers(qs: list[Question], known: set[str]) -> list[Objection]:
    """Answers may only rest on findings/results that exist and on tool calls that succeeded."""
    out = []
    for q in qs:
        q.answer_ids = [normalize_id(i) for i in q.answer_ids]
        bad = [i for i in q.answer_ids if i not in known]
        if bad:
            out.append(Objection(target=q.id, kind="invalid_citation", source="verifier",
                                 text=f"Answer cites evidence that does not exist: {', '.join(bad)}."))
            q.answer_ids = [i for i in q.answer_ids if i in known]
        if q.conclusion == "observed" and not q.answer_ids:
            out.append(Objection(target=q.id, kind="unsupported_step", source="verifier",
                                 text="Answer is labelled observed but cites no valid finding or result; "
                                      "downgraded to inferred."))
            q.conclusion = "inferred"
    return out


def check_steps(link, known: set[str], lit_ids: set[str], where: str) -> list[Objection]:
    """A step claimed as tested needs a valid non-literature id; one claimed as literature needs the id of a search
    that returned papers. Otherwise it is speculative, and says so."""
    out = []
    for st in link.steps:
        st.ids = [normalize_id(x) for x in st.ids]
        valid = [x for x in st.ids if x in known]
        if st.kind == "tested" and not [x for x in valid if x not in lit_ids]:
            out.append(Objection(target=where, kind="unsupported_step", source="verifier",
                                 text=f"step '{st.claim[:80]}' claimed as tested without a valid result id; "
                                      f"relabelled speculative."))
            st.kind = "speculative"
        elif st.kind == "literature" and not [x for x in valid if x in lit_ids]:
            out.append(Objection(target=where, kind="invalid_citation", source="verifier",
                                 text=f"step '{st.claim[:80]}' claimed as literature-supported without a search "
                                      f"that returned it; relabelled speculative."))
            st.kind = "speculative"
        st.ids = valid
    return out


def check_synthesis(syn: Synthesis, known: set[str], question_ids: set[str] | None = None,
                    lit_ids: set[str] | None = None) -> list[Objection]:
    """Clean citations, refuse 'observed' without evidence, derive each link's status from its two observable ends,
    check each chain step's kind, and flag links the investigation never examined. Alternatives get the same checks."""
    out = []
    every = [(f"link {i}", lk) for i, lk in enumerate(syn.links, 1)]
    every += [(f"alternative {a.rank} link {i}", lk) for a in syn.alternatives for i, lk in enumerate(a.links, 1)]
    for where, link in every:
        out += check_steps(link, known, lit_ids or set(), where)
    for i, link in [(w, lk) for w, lk in every]:
        name = f"{link.source} → {link.target}"
        qids = [str(x).strip().upper() for x in link.ids if re.match(r"^\s*Q\d+\s*$", str(x))]
        if qids:  # models often put question ids among the evidence ids
            link.question_ids = list(dict.fromkeys([*link.question_ids, *[f"Q{int(q[1:]):02d}" for q in qids]]))
            link.ids = [x for x in link.ids if not re.match(r"^\s*Q\d+\s*$", str(x))]
        link.ids = [normalize_id(x) for x in link.ids]
        bad = [x for x in link.ids if x not in known]
        if bad:
            out.append(Objection(target=i, kind="invalid_citation", source="verifier",
                                 text=f"{name}: cites evidence that does not exist: {', '.join(bad)}."))
            link.ids = [x for x in link.ids if x in known]
        if not link.ids and "observed" in (link.mediator_in_source, link.receptor_in_target):
            out.append(Objection(target=i, kind="unsupported_step", source="verifier",
                                 text=f"{name}: expression labelled observed without valid evidence; set to not "
                                      f"checked."))
            link.mediator_in_source = "not_checked" if link.mediator_in_source == "observed" else link.mediator_in_source
            link.receptor_in_target = "not_checked" if link.receptor_in_target == "observed" else link.receptor_in_target
        ends = (link.mediator_in_source, link.receptor_in_target)
        link.evidence = ("contradicted" if "absent" in ends else "expression_observed" if ends == ("observed", "observed")
                         else "partly_observed" if "observed" in ends else "not_observed")
        if question_ids is not None:
            link.question_ids = [q for q in link.question_ids if q in question_ids]
            if not link.question_ids:
                out.append(Objection(target=i, kind="missing_evidence", source="verifier",
                                     text=f"{name}: not examined by any question of the investigation (a lead, not "
                                          f"a finding)."))
    return out


def depth(qs: list[Question]) -> int:
    parent = {q.id: q.parent_id for q in qs}

    def d(qid, seen=()):
        p = parent.get(qid)
        return 1 if not p or p not in parent or p in seen else 1 + d(p, (*seen, qid))
    return max((d(q.id) for q in qs), default=0)


def known_ids(findings_ids: set[str], results: list[ToolResult]) -> set[str]:
    return findings_ids | {r.id for r in results if r.ok}


# ---- causal steps: every question names the step it works on, so a step the data cannot settle is seen as such ------

_STEP_ID = re.compile(r"^S\d+$")
EXHAUSTED_TRIES = 3  # questions on one step that all ended without an answer, or with inferences only


def _label_key(label: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9+]+", " ", label.lower()).split())


def assign_step(raw, qs: list[Question]) -> tuple[str, str]:
    """(step id, label) for a new question: an existing id, an existing label written again, or a new step."""
    s = str(raw or "").strip()
    if not s:
        return "", ""
    labels = {q.step: q.step_label for q in qs if q.step}
    if _STEP_ID.match(s.upper()) and s.upper() in labels:
        return s.upper(), labels[s.upper()]
    lead = re.match(r"^(S\d+)\b", s.upper())
    if lead and lead.group(1) in labels:  # "S06 fibroblast output ..." is step S06 written with its label
        return lead.group(1), labels[lead.group(1)]
    label = re.sub(r"^new\s*:\s*", "", s, flags=re.IGNORECASE).strip()
    if not label or _STEP_ID.match(label.upper()):
        return "", ""
    for sid, lab in labels.items():
        if _label_key(lab) == _label_key(label):
            return sid, lab
    n = max([int(x[1:]) for x in labels if x[1:].isdigit()] + [0]) + 1
    return f"S{n:02d}", label[:160]


def step_registry(qs: list[Question]) -> list[dict]:
    """Each causal step with its questions, the tests tried, and whether the data have run out on it."""
    steps: dict[str, dict] = {}
    for q in qs:
        if not q.step:
            continue
        st = steps.setdefault(q.step, {"id": q.step, "label": q.step_label, "questions": [], "tools": set(),
                                       "decisive": [], "observable": set()})
        st["questions"].append((q.id, q.status))
        st.setdefault("conclusions", []).append(q.conclusion if q.status == "answered" else "")
        st["tools"] |= {c.tool for c in q.calls}
        if q.decisive_observation:
            st["decisive"].append(q.decisive_observation)
            st["observable"].add(q.observable_here or "")
    for st in steps.values():
        done = [s for _, s in st["questions"] if s != "open"]
        # settled: an answer that rests on data or literature; answers that are only inferences do not settle a step
        st["settled"] = settled = any(c in ("observed", "literature") for c in st.get("conclusions", []))
        named = any(s == "unanswerable_with_data" for s in done) and st["decisive"]
        inferred_only = sum(c == "inferred" for c in st.get("conclusions", [])) >= EXHAUSTED_TRIES
        st["exhausted"] = bool(not settled and (named or len(done) >= EXHAUSTED_TRIES or inferred_only))
        st["tools"] = sorted(st["tools"])
        st.pop("conclusions", None)
    return sorted(steps.values(), key=lambda x: x["id"])


def step_progress(before: list[dict], after: list[dict]) -> bool:
    """Did a round move the causal steps: a step newly settled on data or literature (an old one, or one opened this
    round)? Opening a step is new work, not progress: a PI that opens a step every round would otherwise never stop."""
    b = {st["id"]: st for st in before}
    return any(st["settled"] and not (b.get(st["id"]) or {}).get("settled") for st in after)


def is_exhausted(step: str, qs: list[Question]) -> bool:
    return any(st["id"] == step and st["exhausted"] for st in step_registry(qs))


def steps_text(qs: list[Question]) -> str:
    """CAUSAL STEPS block for the planner, the PI and the synthesizer."""
    lines = []
    for st in step_registry(qs):
        qtxt = ", ".join(f"{i} {s}" for i, s in st["questions"])
        line = f"- {st['id']}{' [EXHAUSTED]' if st['exhausted'] else ''} {st['label']} — {qtxt}"
        if st["tools"]:
            line += f"; tests tried: {', '.join(st['tools'])}"
        if st["decisive"]:
            line += f"; would be settled by: {st['decisive'][-1]}"
            obs = st["observable"] - {""}
            if obs:
                line += f" (observable in this dataset: {'/'.join(sorted(obs))})"
        lines.append(line)
    return "\n".join(lines)


def repeated_demands(now: list[dict], before: list[dict], threshold: float = 0.6, share: float = 0.8) -> bool:
    """True when most of this review's demands restate the previous review's (word overlap), i.e. no new ask."""
    def words(d):
        return {w for w in re.findall(r"[a-z0-9]+", str(d.get("text", "") if isinstance(d, dict) else d).lower())
                if len(w) > 2}
    if not now or not before:
        return False
    prev = [words(d) for d in before]
    hits = 0
    for d in now:
        w = words(d)
        if w and any(len(w & p) / len(w | p) >= threshold for p in prev if p):
            hits += 1
    return hits / len(now) >= share


ARTEFACT_RE = re.compile(r"\b(batch|artefacts?|artifacts?|dissociation|ambient|doublets?|technical|mislabel\w*|"
                         r"label(?:ling|ing)? (?:error|artefact|artifact)s?)\b", re.IGNORECASE)


def artefact_circuits(circuits) -> tuple[list, list[Objection]]:
    """The hypothesizer takes the data as real: a circuit whose title or root is an artefact explanation is removed."""
    keep, out = [], []
    for c in circuits:
        if ARTEFACT_RE.search(f"{c.title} {c.root}"):
            out.append(Objection(target=f"circuit {c.rank}", kind="confound", source="verifier",
                                 text=f"Circuit {c.rank} ({c.title[:70]}) explains the data as an artefact; the "
                                      f"circuits take the data as real (the study's caveats already list what could "
                                      f"be technical), so it was removed."))
        else:
            keep.append(c)
    return keep, out


def assemble_consensus(plan: dict, attempts: list[dict]) -> list[dict]:
    """The merged circuits, built by code from the merge plan: each group shows its first member as that attempt wrote
    it, with its support ("n of K", counted here from the members) and the other members named. Every attempt circuit
    the plan left out is kept as its own alternative ("1 of K"), so nothing an attempt proposed is lost."""
    k_all = len(attempts)
    by = {(a.get("attempt", i), int(c.get("rank") or j)): c for i, a in enumerate(attempts, 1)
          for j, c in enumerate(a.get("circuits") or [], 1) if isinstance(c, dict)}
    used, groups = set(), []
    for g in plan.get("groups") or []:
        if not isinstance(g, dict):
            continue
        mem = []
        for m in g.get("members") or []:
            try:
                key = (int(m.get("attempt")), int(m.get("circuit")))
            except (TypeError, ValueError, AttributeError):
                continue
            if key in by and key not in used and key not in mem:
                mem.append(key)
        if mem:
            used.update(mem)
            groups.append((mem, str(g.get("title") or ""), str(g.get("note") or "")))
    groups += [([key], "", "") for key in by if key not in used]
    out = []
    for mem, title, note in groups:
        c = json.loads(json.dumps(by[mem[0]]))
        n = len({a for a, _ in mem})
        others = [f"attempt {a}, circuit {r}" for a, r in mem[1:]]
        c.update(rank=len(out) + 1, title=title or c.get("title", ""), support=f"{n} of {k_all}",
                 merged_from=[f"attempt {a}, circuit {r}" for a, r in mem])
        extra = ((f" Also proposed, in other words, by {'; '.join(others)}." if others else "")
                 + (f" {note}" if note else ""))
        c["summary"] = (str(c.get("summary") or "") + extra).strip()
        out.append(c)
    return out


def check_circuits(circuits, known: set[str], lit_ids: set[str]) -> list[Objection]:
    """Edge tags must be honest: 'observed' needs a real result or finding id, 'analogy' a literature search that ran.
    A tag without its evidence is downgraded (observed -> likely, analogy -> speculative) and said so."""
    out = []
    for c in circuits:
        for e in c.edges:
            e.ids = [normalize_id(i) for i in e.ids]
            good = [i for i in e.ids if i in known]
            if e.status == "observed" and not good:
                e.status = "likely"
                out.append(Objection(target=f"circuit {c.rank}", kind="unsupported_step", source="verifier",
                                     text=f"Edge {e.source} -> {e.target} was tagged observed without a valid id; "
                                          f"relabelled likely."))
            elif e.status == "analogy" and not [i for i in e.ids if i in lit_ids]:
                e.status = "speculative"
                out.append(Objection(target=f"circuit {c.rank}", kind="unsupported_step", source="verifier",
                                     text=f"Edge {e.source} -> {e.target} was tagged analogy without a literature "
                                          f"search that ran; relabelled speculative."))
            e.ids = good if e.status == "observed" else e.ids
        if not c.predictions:
            out.append(Objection(target=f"circuit {c.rank}", kind="missing_evidence", source="verifier",
                                 text=f"Circuit {c.rank} ({c.title[:60]}) makes no prediction that would tell it apart."))
    return out


MAX_STEPS = 12  # longer, a circuit reads as several mechanisms drawn as one (p044 S3's lead had 17, 2026-10-20)
MAX_PLAYERS = 9  # separate cells and signals in its drawing; more and the ring becomes a tangle


def circuit_parts(edges) -> list[list[int]]:
    """The circuit's steps grouped into chains (1-based step numbers): two steps are in one chain when what one step
    reaches (its target: cell families and molecules named) is what another step starts from or acts through. A
    well-formed circuit is one chain; several chains are separate mechanisms drawn together."""
    from mechanist import cellfig

    def get(e, k):
        return str((e.get(k) if isinstance(e, dict) else getattr(e, k, "")) or "")

    def toks(text: str) -> set[str]:
        return set(cellfig.cells_in(text)) | {cellfig.norm(g) for g in cellfig.genes_in(text, limit=8)}
    es = list(edges or [])
    src = [toks(get(e, "source")) | toks(get(e, "via")) for e in es]
    tgt = [toks(get(e, "target")) for e in es]
    parent = list(range(len(es)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(es)):
        for j in range(len(es)):
            if i != j and (tgt[i] & src[j]):
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(es)):
        groups.setdefault(find(i), []).append(i + 1)
    return sorted(groups.values(), key=lambda g: (-len(g), g[0]))


def check_coherence(circuits) -> list[Objection]:
    """Each circuit must read as one mechanism: at most MAX_STEPS steps, no second chain of three or more steps standing
    apart from the first (two mechanisms drawn as one), and at most MAX_PLAYERS cells and signals in its drawing.
    Otherwise the PI hears it, so the circuit is split (each chain its own circuit), trimmed to its core chain, or the
    missing link named. Calibrated on every published run on 2026-10-20: only p044 S3's lead circuit broke all three."""
    from mechanist import cellfig
    out = []
    for c in circuits:
        es = list(getattr(c, "edges", None) or (c.get("edges") if isinstance(c, dict) else []) or [])
        rank = getattr(c, "rank", None) if not isinstance(c, dict) else c.get("rank")
        plain = [e if isinstance(e, dict) else (e.model_dump() if hasattr(e, "model_dump") else vars(e)) for e in es]
        parts = circuit_parts(plain)
        if len(parts) > 1 and len(parts[1]) >= 3:
            out.append(Objection(target=f"circuit {rank}", kind="missing_evidence", source="verifier",
                                 text=f"Circuit {rank} holds separate mechanisms: steps "
                                      + " | ".join(", ".join(map(str, g)) for g in parts if len(g) >= 2)
                                      + " form chains that do not connect. Split it, one circuit per chain, or add the "
                                        "step that links them."))
        if len(es) > MAX_STEPS:
            out.append(Objection(target=f"circuit {rank}", kind="missing_evidence", source="verifier",
                                 text=f"Circuit {rank} has {len(es)} steps (at most {MAX_STEPS}): it reads as several "
                                      f"mechanisms at once. Keep its core chain; move the rest into their own circuits "
                                      f"or into 'unused'."))
        g = cellfig.build_graph(plain)
        players = len(g["ring"]) + len(g["centre"])
        if players > MAX_PLAYERS:
            out.append(Objection(target=f"circuit {rank}", kind="missing_evidence", source="verifier",
                                 text=f"Circuit {rank} has {players} separate cells and signals (at most {MAX_PLAYERS}), "
                                      f"so its drawing becomes a tangle. Name the same cell the same way at both ends of "
                                      f"steps, and move side branches into their own circuits."))
    return out


def check_bench(circuits, items: list[dict], parent: list[dict] | None = None) -> list[Objection]:
    """A version's grades: 'bench' needs a submitted result whose outcome fits (positive or mixed), 'refuted' one that
    argues against (negative or mixed); otherwise the step falls back to 'likely'. A circuit that keeps or improves its
    rank with a refuted step must say why in its summary, or the PI hears of it."""
    outcome = {x["id"]: x.get("outcome") for x in items if x["kind"] == "result"}
    fits = {"bench": ("positive", "mixed"), "refuted": ("negative", "mixed")}
    was = {c.get("title"): c.get("rank") for c in parent or []}
    out = []
    for c in circuits:
        for e in c.edges:
            if e.status in fits and not any(outcome.get(i) in fits[e.status] for i in e.ids):
                out.append(Objection(target=f"circuit {c.rank}", kind="unsupported_step", source="verifier",
                                     text=f"Edge {e.source} -> {e.target} was graded {e.status} without a submitted "
                                          f"result that fits; relabelled likely."))
                e.status = "likely"
        if any(e.status == "refuted" for e in c.edges) and c.rank <= was.get(c.title, 0) \
                and "refut" not in c.summary.lower():
            out.append(Objection(target=f"circuit {c.rank}", kind="missing_evidence", source="verifier",
                                 text=f"Circuit {c.rank} ({c.title[:60]}) has a refuted step but keeps its rank; "
                                      f"lower it, or say in its summary why the refutation does not break it."))
    return out


def check_items(accounting: list[dict], items: list[dict]) -> list[Objection]:
    """Every submitted item must be accounted for: which steps it bears on and how, with a reason."""
    got = {str(a.get("item", "")).upper().strip(): a for a in accounting if isinstance(a, dict)}
    out = [Objection(target="refine", kind="missing_evidence", source="verifier",
                     text=f"{x['id']} ({x['kind']}) is not accounted for: say which steps it bears on and how.")
           for x in items if x["id"] not in got]
    out += [Objection(target="refine", kind="missing_evidence", source="verifier",
                      text=f"{a.get('item')}: give the reason for '{a.get('verdict')}'.")
            for a in got.values() if not str(a.get("reason") or "").strip()]
    return out


def check_accounting(accounting: list[dict], link_ids: list[str], known: set[str]) -> list[Objection]:
    """Every ledger link must be accounted for by the circuits: used (in a circuit), set aside (with a reason) or
    contradicted (with a data id that exists). A link left out, or a verdict without what it needs, is an objection."""
    out, seen = [], {}
    for a in accounting:
        seen[str(a.get("link", "")).upper().strip()] = a
    missing = [i for i in link_ids if i not in seen]
    if missing:
        out.append(Objection(target="circuits", kind="missing_evidence", source="verifier",
                             text=f"{len(missing)} of {len(link_ids)} ledger links are not accounted for: "
                                  f"{', '.join(missing)}. Say for each which circuit uses it, why it is set aside, "
                                  f"or which data id contradicts it."))
    for i in link_ids:
        a = seen.get(i)
        if not a:
            continue
        v = str(a.get("verdict", ""))
        if v == "contradicted" and not [x for x in a.get("ids") or [] if normalize_id(str(x)) in known]:
            out.append(Objection(target="circuits", kind="unsupported_step", source="verifier",
                                 text=f"Ledger link {i} is called contradicted without a data id that exists."))
        elif v == "set_aside" and len(str(a.get("reason", "")).strip()) < 10:
            out.append(Objection(target="circuits", kind="missing_evidence", source="verifier",
                                 text=f"Ledger link {i} is set aside without a reason."))
        elif v not in ("used", "set_aside", "contradicted"):
            out.append(Objection(target="circuits", kind="missing_evidence", source="verifier",
                                 text=f"Ledger link {i} has no verdict (used, set_aside or contradicted)."))
    return out


def check_phenotype(syn: Synthesis, item_ids: list[str]) -> list[Objection]:
    """Every dossier item must be accounted for, and an 'explained' item must name a link of the model."""
    if not item_ids:
        return []
    out = []
    got = {f.item.strip().upper(): f for f in syn.phenotype_fit}
    missing = [i for i in item_ids if i not in got]
    if missing:
        out.append(Objection(target="phenotype accounting", kind="missing_evidence", source="verifier",
                             text=f"{len(missing)} of {len(item_ids)} disease items are not accounted for: "
                                  f"{', '.join(missing[:20])}. Classify every item."))
    ends = {f"{lk.source} {lk.target}".lower() for lk in syn.links}
    bare = [f.item for f in syn.phenotype_fit if f.verdict == "explained" and not any(
        any(w in e for e in ends) for ln in f.links for w in re.findall(r"[a-z0-9+]{4,}", ln.lower()))]
    if bare:
        out.append(Objection(target="phenotype accounting", kind="unsupported_step", source="verifier",
                             text=f"Marked explained without naming a link of the model: {', '.join(bare[:20])}."))
    extra = [f.item for f in syn.phenotype_fit if f.item.strip().upper() not in set(item_ids)]
    if extra:
        out.append(Objection(target="phenotype accounting", kind="invalid_citation", source="verifier",
                             text=f"Items that are not in the dossier: {', '.join(extra[:10])}."))
    return out


# ---- experiments and therapeutic hypotheses (stages 8-9): the agenda, checks and ranking, all deterministic ----------

# the agent names its own kinds and settings; code only checks ids, citations, predictions and controls
OUTCOMES = ("up", "down", "no change", "not addressed")
QUESTIONS = ("presence", "rule_out", "mechanism", "in_vivo")  # what an experiment asks, in the order a lab would ask
TIERS = ("in_data", "archived_tissue", "spatial", "ex_vivo", "in_vitro", "in_vivo", "cohort")
LEVELS = ("in use for this disease", "approved for another disease", "in trials", "preclinical")
# an index patient has no settled disease: the first level means reported in one of the candidate diagnoses
INDEX_LEVELS = ("reported in a candidate disease",) + LEVELS[1:]
_STOP = {"the", "and", "with", "from", "that", "this", "their", "they", "have", "has", "was", "were", "for", "into",
         "over", "under", "patient", "patients", "skin", "lesion", "lesions", "which", "when", "after", "before"}


def said_in(text: str, source: str, share: float = 0.6) -> bool:
    """A paraphrase of the source rather than something new: most of its content words appear in the source."""
    words = [w for w in re.findall(r"[a-z][a-z0-9-]{2,}", str(text).lower()) if w not in _STOP]
    src = str(source).lower()
    return bool(words) and sum(w in src for w in words) / len(words) >= share
BLOCKING = re.compile(r"inhibitor|antagonist|blocker|negative|degrader|antibody|neutraliz|inverse agonist|suppressor",
                      re.IGNORECASE)


def support_n(c: dict) -> tuple[int, int]:
    """'2 of 3' -> (2, 3); a single attempt counts as (1, 1)."""
    m = re.match(r"\s*(\d+)\s*of\s*(\d+)", str(c.get("support") or ""))
    return (int(m.group(1)), int(m.group(2))) if m else (1, 1)


def missing_links(circuits: list[dict], exhausted: list[dict], leads: list[dict]) -> list[dict]:
    """The agenda of experiments and therapy: untested edges of circuits at least two attempts proposed (every circuit
    when there was one attempt), the exhausted causal steps, and the leads one attempt alone found. Ids M1, M2, ..."""
    out: list[dict] = []
    seen: dict[str, dict] = {}
    for c in circuits:
        n, k = support_n(c)
        if k > 1 and n < 2:
            continue
        for i, e in enumerate(c.get("edges") or [], 1):
            if not isinstance(e, dict) or e.get("status") == "observed":
                continue
            key = f"{str(e.get('source', '')).lower()[:60]}>{str(e.get('target', '')).lower()[:60]}"
            if key in seen:
                seen[key]["circuits"].append(int(c.get("rank") or 0))
                continue
            m = {"kind": "edge", "circuits": [int(c.get("rank") or 0)], "edge": i, "status": e.get("status", ""),
                 "text": f"{e.get('source', '')} → {e.get('target', '')}" + (f" ({e.get('via')})" if e.get("via") else "")}
            seen[key] = m
            out.append(m)
    for st in exhausted:
        out.append({"kind": "step", "circuits": [], "step": st.get("id", ""), "status": "exhausted",
                    "text": str(st.get("label", "")) + (f"; would be settled by {st['decisive'][-1]}"
                                                        if st.get("decisive") else "")})
    for x in leads:
        out.append({"kind": "lead", "circuits": [], "status": "lead", "text": str(x.get("text", ""))[:400],
                    "ids": list(x.get("ids") or [])})
    for i, m in enumerate(out, 1):
        m["id"] = f"M{i}"
    return out


def missing_links_text(ms: list[dict]) -> str:
    return "\n".join(f"- {m['id']} [{m['kind']}, {m['status']}"
                     + (f", circuits {', '.join(map(str, sorted(set(m['circuits']))))}" if m["circuits"] else "")
                     + f"] {m['text']}" for m in ms) or "(none)"


def check_experiments(exps: list[dict], circuits: list[dict], ms: list[dict], lit_ok: set[str],
                      papers: dict[str, dict] | None = None) -> tuple[list[dict], list[Objection], list[str]]:
    """Keep only known circuit ranks, M ids, outcomes and tiers; precedent only when its search ran, and PMIDs only
    when a search that ran returned them (papers: PMID -> its record). Returns the cleaned experiments, objections, and
    the M ids no experiment covers."""
    papers = papers or {}
    ranks, mids = {int(c.get("rank") or 0) for c in circuits}, {m["id"] for m in ms}
    out, objs = [], []
    for e in exps:
        if not isinstance(e, dict) or not (e.get("perturbation") or e.get("readout")):
            continue
        preds = {}
        for p in e.get("predictions") or []:
            if not isinstance(p, dict):
                continue
            try:
                r = int(p.get("circuit"))
            except (TypeError, ValueError):
                continue
            o = str(p.get("outcome") or "").strip().lower()
            if r in ranks and o in OUTCOMES:
                preds[r] = {"circuit": r, "outcome": o, "why": str(p.get("why") or "")}
        tests = [m for m in (str(x).strip() for x in e.get("tests") or []) if m in mids]
        bad = [str(x) for x in e.get("tests") or [] if str(x).strip() not in mids]
        if bad:
            objs.append(Objection(target="experiment", kind="unsupported_step", source="verifier",
                                  text=f"Experiment {str(e.get('title') or '')[:60]!r} names unknown links {bad}; dropped."))
        prec = []
        for p in e.get("precedent") or []:
            if not isinstance(p, dict):
                continue
            pm = [x for x in (re.sub(r"\D", "", str(y)) for y in p.get("pmids") or []) if x in papers]
            ids = [i for i in (normalize_id(str(y)) for y in p.get("ids") or []) if i in lit_ok]
            ids += [papers[x]["rid"] for x in pm if papers[x]["rid"] not in ids]
            prec.append({"what": str(p.get("what", "")), "ids": ids,
                         "papers": [{k: papers[x][k] for k in ("pmid", "year", "title")} for x in pm]})
        title = str(e.get("title") or "")[:60]
        if not str(e.get("controls") or "").strip():
            objs.append(Objection(target="experiment", kind="missing_evidence", source="verifier",
                                  text=f"Experiment {title!r} names no controls."))
        if not str(e.get("refutes_if") or "").strip():
            objs.append(Objection(target="experiment", kind="missing_evidence", source="verifier",
                                  text=f"Experiment {title!r} does not say which result would count against the circuit."))
        q = str(e.get("question") or "").strip().lower().replace(" ", "_").replace("-", "_")
        tier = str(e.get("tier") or e.get("setting") or "").strip().lower().replace(" ", "_").replace("-", "_")
        done = []
        for d in e.get("done_before") or []:
            pm = re.sub(r"\D", "", str(d.get("pmid") if isinstance(d, dict) else d))
            if pm in papers:
                done.append({"pmid": pm, "what": str(d.get("what", "") if isinstance(d, dict) else ""),
                             "title": papers[pm].get("title", ""), "year": papers[pm].get("year", "")})
            elif pm:
                objs.append(Objection(target="experiment", kind="unsupported_step", source="verifier",
                                      text=f"Experiment {title!r} says PMID {pm} did it before, but no search returned "
                                           f"that paper; dropped."))
        out.append({**{k: e.get(k) for k in ("title", "model", "perturbation", "readout", "controls", "feasibility")},
                    "id": str(e.get("id") or "").strip().upper(),
                    "shows": str(e.get("shows") or ""), "adds": str(e.get("adds") or ""),
                    "question": q if q in QUESTIONS else "",
                    "kind": str(e.get("kind") or "").strip()[:40],
                    "tier": tier if tier in TIERS else "",
                    "rna_vs_protein": str(e.get("rna_vs_protein") or ""),
                    "perturbation_kind": [str(x) for x in e.get("perturbation_kind") or [] if str(x).strip()][:6],
                    "done_before": done,
                    "setting": str(e.get("setting") or e.get("tier") or "").strip()[:60],
                    "refutes_if": str(e.get("refutes_if") or ""), "caveat": str(e.get("caveat") or ""),
                    "priority": e.get("priority"), "tests": tests,
                    "confirms": [str(g).upper() for g in e.get("confirms") or [] if str(g).strip()],
                    "predictions": [preds[r] for r in sorted(preds)], "precedent": [p for p in prec if p["ids"]]})
    seen = set()  # every experiment gets a unique id; programmes refer to it
    for i, e in enumerate(out, 1):
        if not re.fullmatch(r"E\d+", e["id"]) or e["id"] in seen:
            e["id"] = f"E{i}"
            while e["id"] in seen:
                e["id"] += "b"
        seen.add(e["id"])
    covered = {m for e in out for m in e["tests"]}
    uncovered = [m["id"] for m in ms if m["id"] not in covered]
    return out, objs, uncovered


def check_programmes(progs: list[dict], exps: list[dict], circuits: list[dict]) -> tuple[list[dict], list[Objection]]:
    """Each hypothesis's sequence of experiments. Keeps known circuits and experiment ids in the agent's order; objects
    when a circuit at least two attempts proposed has no sequence, when a sequence has no experiment that tests its
    central claim directly, when presence is checked after the mechanism is built on it, and when an experiment sits
    in no sequence. Marks experiments used by several sequences as shared."""
    ranks = {int(c.get("rank") or 0): c for c in circuits}
    by_id = {e["id"]: e for e in exps}
    out, objs = [], []
    for pr in progs or []:
        if not isinstance(pr, dict):
            continue
        try:
            r = int(pr.get("circuit"))
        except (TypeError, ValueError):
            continue
        if r not in ranks or any(x["circuit"] == r for x in out):
            continue
        seq, used = [], set()
        for st in pr.get("sequence") or []:
            if not isinstance(st, dict):
                continue
            eid = str(st.get("id") or "").strip().upper()
            if eid in by_id and eid not in used:
                used.add(eid)
                seq.append({"id": eid, **{k: str(st.get(k) or "") for k in ("why_now", "if_yes", "if_no", "adapt")}})
        if not seq:
            continue
        out.append({"circuit": r, "claim": str(pr.get("claim") or ""), "sequence": seq})
        qs = [by_id[x["id"]]["question"] for x in seq]
        if "mechanism" not in qs and "in_vivo" not in qs:
            objs.append(Objection(target="experiment", kind="missing_evidence", source="verifier",
                                  text=f"Circuit {r}'s sequence has no experiment that tests its central claim directly "
                                       f"(by giving or blocking the proposed signal)."))
        if "presence" in qs and "mechanism" in qs and qs.index("presence") > qs.index("mechanism"):
            objs.append(Objection(target="experiment", kind="missing_evidence", source="verifier",
                                  text=f"Circuit {r}'s sequence checks presence in tissue only after a mechanism "
                                       f"experiment that relies on it."))
    have = {x["circuit"] for x in out}
    for r, c in ranks.items():
        n, k = support_n(c)
        if (n >= 2 or k == 1) and r not in have:
            objs.append(Objection(target="experiment", kind="missing_evidence", source="verifier",
                                  text=f"Circuit {r} (proposed by {n} of {k} attempts) has no sequence of experiments."))
    use: dict[str, list[int]] = {}
    for x in out:
        for st in x["sequence"]:
            use.setdefault(st["id"], []).append(x["circuit"])
    for e in exps:
        e["in_programmes"] = use.get(e["id"], [])
    loose = [e["id"] for e in exps if not e["in_programmes"]]
    if loose:
        objs.append(Objection(target="experiment", kind="missing_evidence", source="verifier",
                              text=f"Experiments {', '.join(loose)} sit in no hypothesis's sequence: place each, or drop it."))
    return out, objs


def crucial_molecules(circuits: list[dict], ms: list[dict], limit: int = 25, exclude=()) -> list[dict]:
    """The molecules the circuits rest on, to confirm in tissue: genes on the edges of circuits at least two attempts
    proposed (every circuit when there was one attempt), most-supported first."""
    keep = [c for c in circuits if support_n(c)[1] == 1 or support_n(c)[0] >= 2]
    return therapy_targets(keep, ms, limit, exclude)


def uncovered_molecules(exps: list[dict], molecules: list[dict]) -> list[str]:
    done = {g for e in exps for g in e.get("confirms") or []}
    return [m["gene"] for m in molecules if m["gene"] not in done]


def rank_experiments(exps: list[dict], circuits: list[dict]) -> list[dict]:
    """Discrimination: how many distinct outcomes the addressed circuits predict, each outcome weighted by the summed
    support of the circuits predicting it beyond the most common one. Ranked by discrimination, then tier, then the
    agent's priority."""
    sup = {int(c.get("rank") or 0): support_n(c)[0] / max(1, support_n(c)[1]) for c in circuits}
    for e in exps:
        by: dict[str, float] = {}
        for p in e["predictions"]:
            if p["outcome"] != "not addressed":
                by[p["outcome"]] = by.get(p["outcome"], 0.0) + sup.get(p["circuit"], 0.0)
        w = sorted(by.values(), reverse=True)
        e["discrimination"] = round(sum(w[1:]), 3) if len(w) > 1 else 0.0
        e["outcomes"] = len(by)
    exps.sort(key=lambda e: (-e["discrimination"], int(e["priority"]) if str(e.get("priority") or "").isdigit() else 99))
    for i, e in enumerate(exps, 1):
        e["rank"] = i
    return exps


MARKER = re.compile(r"^(?:CD\d+[A-Z]?|HLA-?\w*|MHC-?\w*|MKI67|PTPRC)$")
GENE_TOKEN = re.compile(r"(?<![\w-])(?:IL-?\d+[A-F]?|[A-Z][A-Z0-9]{1,}(?:-[A-Z0-9]+)*)(?![\w])")
NOT_TARGET = {"PG", "DE", "UP", "DOWN", "AND", "OR", "NOT", "TLS", "IBD", "DAMP", "RNA", "DNA", "II", "JAK", "STAT", "ECM",
              "ROS", "NET", "NETS", "FC", "NK", "DC", "UMI", "QC", "HIF", "NF", "TNF-SUPERFAMILY"}


def _target_genes(text: str, exclude=()) -> list[str]:
    """Gene-like tokens in an edge's text; `exclude` holds the dataset's sample and donor ids, which look like genes."""
    out = []
    skip = {str(x).upper() for x in exclude}
    for g in GENE_TOKEN.findall(text.replace("α", "A").replace("β", "B").replace("γ", "G")):
        g = g.replace("-", "") if re.fullmatch(r"IL-\d+[A-F]?", g) else g
        if g in NOT_TARGET or g in skip or MARKER.match(g) or re.fullmatch(r"[A-Z]\d+|[RFQSLMD]\d{2,}", g) or len(g) < 3:
            continue
        if g not in out:
            out.append(g)
    return out


def therapy_targets(circuits: list[dict], ms: list[dict], limit: int = 30, exclude=()) -> list[dict]:
    """Genes on the circuits' edges (mediators, receptors and the genes an edge names; cell markers left out), each with
    the circuits and edges that carry it, their summed support, and the untested M ids among those edges."""
    by: dict[str, dict] = {}
    edge_m = {(c, m["edge"]): m["id"] for m in ms if m["kind"] == "edge" for c in m["circuits"]}
    for c in circuits:
        rank = int(c.get("rank") or 0)
        n, k = support_n(c)
        for i, e in enumerate(c.get("edges") or [], 1):
            if not isinstance(e, dict):
                continue
            genes = _target_genes(f"{e.get('via', '')} {e.get('source', '')} {e.get('target', '')}", exclude)
            for g in genes:
                t = by.setdefault(g, {"gene": g, "circuits": [], "support": 0.0, "edges": [], "untested": []})
                if rank not in t["circuits"]:
                    t["circuits"].append(rank)
                    t["support"] += n / max(1, k)
                t["edges"].append({"circuit": rank, "edge": i, "status": e.get("status", ""),
                                   "text": f"{e.get('source', '')} → {e.get('target', '')}"})
                mid = edge_m.get((rank, i))
                if mid and mid not in t["untested"]:
                    t["untested"].append(mid)
    out = sorted(by.values(), key=lambda t: (-t["support"], -sum(x["status"] == "observed" for x in t["edges"]), t["gene"]))
    for t in out:
        t["support"] = round(t["support"], 3)
    return out[:limit]


def expand_target(name: str, symbols: set[str], receptors_of: Callable[[str], list[str]], members: int = 30,
                  per_ligand: int = 20) -> list[dict]:
    """The genes a therapy target stands for, each with how it was reached: the gene itself; for a family name (no
    gene of that exact symbol, e.g. a ligand family) its members among the known symbols; and for each ligand, its
    receptors' subunits, since many drugs block the receptor rather than the ligand. Order: named, members, receptors.
    No ranking among receptors: the resource has none, so all are looked up (cached) and the pharmacologist judges."""
    g = name.upper()
    if symbols and g not in symbols and "-" in g:  # "IFN-G" is IFNG; "IL-18R" is the IL18R family
        g = g.replace("-", "")
    out = [{"gene": g, "via": "named", "known": True}] if g in symbols or not symbols else []
    if not out:  # "known": a gene symbol here, so an Open Targets name for it is the same gene under another name
        fam = sorted(x for x in symbols if re.fullmatch(re.escape(g) + r"[A-Z0-9]{1,3}", x))[:members]
        out = ([{"gene": x, "via": f"member of {g}", "known": True} for x in fam]
               or [{"gene": g, "via": "named", "known": False}])
    seen = {x["gene"] for x in out}
    of: dict[str, list[str]] = {}  # receptor subunit -> every ligand here it binds, so the label names them all
    for lig in [x["gene"] for x in out]:
        try:
            recs = receptors_of(lig)
        except Exception:  # noqa: BLE001 - not a ligand in the resource
            recs = []
        for r in list(dict.fromkeys(u.upper() for r in recs for u in str(r).split("_") if u))[:per_ligand]:
            if r not in seen:
                of.setdefault(r, []).append(lig)
    return out + [{"gene": r, "via": "receptor of " + ", ".join(ligs), "known": True} for r, ligs in of.items()]


def considered_drugs(targets: list[dict], lookups: dict[str, dict], hyps: list[dict], objs: list,
                     passed: list[dict]) -> list[dict]:
    """Every drug the lookups found for each target, and what became of it: listed (a hypothesis), dropped (by the
    checks), passed over (with the pharmacologist's reason) or not chosen (no reason given). Lookups that found nothing,
    or matched a different gene than asked, are kept so a reader sees why a drug is absent."""
    listed = {(h["drug"].upper(), h["target"].upper()) for h in hyps}
    dropped = {str(o.text).split(" on ")[0].upper() for o in objs if "not in any drug lookup" in str(o.text)}
    why = {str(x.get("drug", "")).upper(): str(x.get("why") or "") for x in passed if isinstance(x, dict)}
    out = []
    for t in targets:
        rows = []
        for lk in t.get("lookups") or [{"gene": t["gene"], "via": "named"}]:
            d = lookups.get(lk["gene"].upper())
            if d is None:
                rows.append({**lk, "matched_as": "", "drugs": [], "note": "lookup failed or not run"})
                continue
            sym = str(d.get("symbol") or lk["gene"]).upper()
            renamed = sym != lk["gene"].upper()
            wrong = renamed and not lk.get("known", True)  # not a gene symbol: Open Targets' top hit may be another gene
            note = (f"Open Targets matched {lk['gene']} as {sym}: check it is the gene meant" if wrong
                    else f"listed in Open Targets as {sym}" if renamed else "")
            drugs = []
            for x in d.get("drugs") or []:
                n = str(x.get("drug", "")).upper()
                st = ("listed" if (n, lk["gene"].upper()) in listed else "dropped" if n in dropped
                      else "passed over" if why.get(n) else "not chosen")
                drugs.append({"drug": x.get("drug"), "stage": x.get("stage", ""),
                              "actions": sorted({a["action"] for a in x.get("actions") or [] if a.get("action")}),
                              "status": st, "why": why.get(n, "")})
            rows.append({**lk, "matched_as": sym if wrong else "", "drugs": drugs,
                         "note": note + ("; no drugs" if note and not drugs else "") if note else
                         ("" if drugs else "no drugs or clinical candidates")})
        out.append({"target": t["gene"], "circuits": t["circuits"], "lookups": rows})
    return out


NUM = re.compile(r"(?<![\w.])(?<![A-Za-z]-)\d+(?:\.\d+)?(?![\w])")  # "IL-20" is a name, not a number
STEP_REF = re.compile(r"\b(?:steps?|circuits?)\s+\d+(?:\s*(?:,|and|or|to|-|–)\s*\d+)*", re.I)


def _numbers(text: str) -> list[str]:
    """The numbers a card states as data: not step or circuit references, not molecule names."""
    return NUM.findall(STEP_REF.sub(" ", text or ""))
CARD_TEXT = ("sender", "signal", "receiver", "receptor", "effect", "follows", "leads_to", "must_hold")


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", str(t or "")).strip().lower().replace("\u2019", "'")


def check_cards(cards: list[dict], edges: list[dict], results: dict) -> tuple[list[dict], list[dict]]:
    """Keep what each mechanism card can show from its sources: numbers only as the cited results give them, quotes
    only as a cited literature result holds them, with that result's PMID. Anything else is dropped and said so."""
    out, dropped = [], []
    for c in cards:
        try:
            k = int(c.get("step"))
        except (TypeError, ValueError):
            continue
        if not 1 <= k <= len(edges) or any(x["step"] == k for x in out):
            continue
        e = edges[k - 1]
        own = [i for i in e.get("ids") or [] if i in results]
        seen = c.get("seen_here") if isinstance(c.get("seen_here"), dict) else {}
        sids = [i for i in seen.get("ids") or [] if i in results]
        hay = " ".join(str(getattr(results[i], "summary", "")) for i in dict.fromkeys(own + sids))
        card = {"step": k}
        for f in CARD_TEXT:
            v = str(c.get(f) or "").strip()
            bad = [n for n in _numbers(v) if n not in hay]
            if bad:
                dropped.append({"step": k, "field": f, "why": f"numbers not in the cited results: {', '.join(bad[:4])}"})
                v = ""
            card[f] = v
        st = str(seen.get("text") or "").strip()
        if st and not sids:
            dropped.append({"step": k, "field": "seen_here", "why": "no cited result"})
            st = ""
        elif st:  # a sentence with a number the cited results do not hold goes; the rest stays
            src, kept = " ".join(str(results[i].summary) for i in sids), []
            for sent in re.split(r"(?<=[.;])\s+(?=[A-Z])", st):
                bad = [n for n in _numbers(sent) if n not in src]
                if bad:
                    dropped.append({"step": k, "field": "seen_here",
                                    "why": f"numbers not in the cited results: {', '.join(bad[:4])}", "sentence": sent[:300]})
                else:
                    kept.append(sent)
            st = " ".join(kept)
        if not st:
            sids = []
        card["seen_here"] = {"text": st, "ids": sids}
        lit = []
        for x in c.get("literature") or []:
            if not isinstance(x, dict):
                continue
            r = results.get(str(x.get("id") or ""))
            q, pmid = str(x.get("quote") or "").strip().strip('"'), re.sub(r"\D", "", str(x.get("pmid") or ""))
            pm = {str(p.get("pmid")) for p in ((getattr(r, "data", None) or {}).get("papers") or [])} if r else set()
            why = ("no such literature result" if r is None or r.tool != "literature_search"
                   else "quote not in that result" if not q or _norm(q) not in _norm(r.summary)
                   else "PMID not in that result" if pmid not in pm else "")
            if why:
                dropped.append({"step": k, "field": "literature", "why": why, "quote": q[:200]})
                continue
            lit.append({"claim": str(x.get("claim") or ""), "quote": q, "pmid": pmid, "id": r.id,
                        "context": str(x.get("context") or "")})
        card["literature"] = lit
        out.append(card)
    out.sort(key=lambda x: x["step"])
    return out, dropped


def drug_level(drug: dict, therapy_texts: str, disease: str) -> str:
    """Evidence level, by code: named in the dossier's therapy items, or an indication matching the disease -> in use;
    approved (or phase 4) elsewhere -> repurposing; any phase -> in trials; otherwise preclinical."""
    name = str(drug.get("drug") or "")
    root = re.sub(r"\s+(SODIUM|PHOSPHATE|HYDROCHLORIDE|DIHYDROCHLORIDE|CITRATE|MALEATE|MESYLATE|SULFATE|"
                  r"MONOHYDRATE|TOSYLATE)\b.*$", "", name.upper()).strip()
    if root and re.search(r"\b" + re.escape(root.lower()) + r"\b", therapy_texts.lower()):
        return LEVELS[0]
    if disease and any(disease.lower() in str(i).lower() for i in drug.get("indications") or []):
        return LEVELS[0]
    st = str(drug.get("stage") or "")
    if st in ("APPROVAL", "PHASE_4"):
        return LEVELS[1]
    if st.startswith(("PHASE", "EARLY_PHASE")):
        return LEVELS[2]
    return LEVELS[3]


def cell_targets(circuits: list[dict], markers: dict[str, list[str]], cells_of) -> list[dict]:
    """Cell-type targets: each lineage the circuits name, with the data's own markers for it, and the circuits that use
    that cell type. cells_of(text) -> the cell families a text names (the figure's reader, shared)."""
    out = []
    for lin, genes in markers.items():
        fam = set(cells_of(lin))
        if not fam or not genes:
            continue
        circs = sorted({int(c.get("rank") or 0) for c in circuits for e in c.get("edges") or [] if isinstance(e, dict)
                        if fam & set(cells_of(f"{e.get('source', '')} {e.get('target', '')}"))})
        if circs:
            out.append({"lineage": lin, "genes": list(genes), "circuits": circs})
    return out


EVIDENCE_SECTIONS = ("therapy response", "difference from comparators")


def is_evidence(section: str) -> bool:
    """An item section (possibly "Sweet: therapy response") that is literature behind a drug, not a feature."""
    return section.split(":")[-1].strip() in EVIDENCE_SECTIONS


def check_therapies(hyps: list[dict], lookups: dict[str, dict], circuits: list[dict], ms: list[dict], items: set[str],
                    therapy_texts: str, disease: str, cell_tg: list[dict] | None = None, design: str = "cohort",
                    reported: dict[str, tuple[str, str]] | None = None,
                    sections: dict[str, str] | None = None) -> tuple[list[dict], list[Objection]]:
    """A therapeutic hypothesis stands only on a lookup that ran: the drug must be listed for that target. Its stage,
    action and evidence level come from the lookup and code, never from the model. A drug that does not block (an
    agonist, an activator) is flagged, since the circuits' steps are activating."""
    ranks, mids = {int(c.get("rank") or 0) for c in circuits}, {m["id"] for m in ms}
    sup = {int(c.get("rank") or 0): support_n(c)[0] / max(1, support_n(c)[1]) for c in circuits}
    total = sum(sup.values()) or 1.0
    marker_of = {g.upper(): t for t in cell_tg or [] for g in t["genes"]}
    out, objs = [], []
    for h in hyps:
        if not isinstance(h, dict):
            continue
        g, name = str(h.get("target") or "").upper().strip(), str(h.get("drug") or "").upper().strip()
        d = next((x for x in (lookups.get(g) or {}).get("drugs") or [] if str(x.get("drug", "")).upper() == name), None)
        if d is None:
            objs.append(Objection(target="therapy", kind="unsupported_step", source="verifier",
                                  text=f"{name or '?'} on {g or '?'} is not in any drug lookup that ran; dropped."))
            continue
        acts = sorted({a["action"] for a in d.get("actions") or [] if a.get("action")})
        cell = marker_of.get(g) or marker_of.get(str((lookups.get(g) or {}).get("symbol") or "").upper())
        blocks = any(BLOCKING.search(a["action"] + " " + a.get("mechanism", "")) for a in d.get("actions") or [])
        blocks = blocks or (cell is not None and any("binding" in a["action"] for a in d.get("actions") or []))
        cs = [r for r in (int(x) for x in h.get("circuits") or [] if str(x).isdigit()) if r in ranks]
        if cell is not None:  # a depleting drug acts in every circuit that uses its cell type
            cs = sorted(set(cs) | set(cell["circuits"]))
        strategy = "cell type" if cell is not None else "step"
        warn = [w for w in d.get("warnings") or [] if w.get("type")]
        level = drug_level(d, therapy_texts, disease)
        root = re.sub(r"\s+(SODIUM|PHOSPHATE|HYDROCHLORIDE|CITRATE|MESYLATE|SULFATE)\b.*$", "", d["drug"].upper()).lower()
        seen_in = sorted({sec.split(":")[0].strip() for sec, t in (reported or {}).values()
                          if root and re.search(r"\b" + re.escape(root) + r"\b", t.lower())})
        improves = [i for i in h.get("improves") or [] if i in items]
        # a treatment response or a comparison of the candidates is literature behind the choice, not a feature the
        # drug improves: sorted out by the item's section (agents.PHENOTYPE_SECTIONS)
        evidence = [i for i in improves if is_evidence((sections or {}).get(i, ""))]
        improves = [i for i in improves if i not in evidence]
        if design == "index_patient":  # the patient's own features; a candidate disease's items are similarities
            similar = [i for i in improves if not i.startswith("P")]
            improves = [i for i in improves if i.startswith("P")]
            if level == LEVELS[0]:
                level = INDEX_LEVELS[0]
        else:
            similar = []
        out.append({"drug": d["drug"], "target": g, "stage": d.get("stage", ""), "actions": acts,
                    "mechanism": "; ".join(a.get("mechanism", "") for a in (d.get("actions") or [])[:2]),
                    "indications": (d.get("indications") or [])[:5], "level": level, "reported_in": seen_in,
                    "similar": similar, "evidence": evidence,
                    "blocks": blocks, "circuits": cs, "support": round(sum(sup.get(r, 0) for r in cs), 3),
                    "breaks": str(h.get("breaks") or ""), "improves": improves,
                    "fails_under": [r for r in (int(x) for x in h.get("fails_under") or [] if str(x).isdigit())
                                    if r in ranks],
                    "depends_on": [m for m in (str(x) for x in h.get("depends_on") or []) if m in mids],
                    "rationale": str(h.get("rationale") or ""), "strategy": strategy,
                    "depletes": cell["lineage"] if cell is not None else "",
                    "robustness": round(sum(sup.get(r, 0) for r in cs) / total, 3),
                    "warnings": warn, "boxed": any("black box" in w["type"].lower() for w in warn),
                    "withdrawn": any(w["type"].lower().startswith("withdrawn") and "least one" not in w["type"].lower()
                                     for w in warn)})
        if design == "index_patient" and not improves and any(i.startswith("P") for i in items):
            objs.append(Objection(target="therapy", kind="unsupported_step", source="verifier",
                                  text=f"{d['drug']} names none of this patient's own features (P items) it should "
                                       f"improve. Name the P items it acts on through the circuit, or say why none."))
        if not blocks:
            objs.append(Objection(target="therapy", kind="unsupported_step", source="verifier",
                                  text=f"{d['drug']} acts on {g} as {', '.join(acts) or 'an unknown action'}, not as a "
                                       f"blocker: check the direction against the circuit step."))
    lv = {x: i for i, x in enumerate(LEVELS)} | {x: i for i, x in enumerate(INDEX_LEVELS)}
    # the drugs are there to break the hypothesised circuits: the circuits' support they act on first, then how few
    # untested steps they depend on; use in this disease (or a candidate one) is supporting evidence and breaks ties
    out.sort(key=lambda h: (-h["support"], len(h["depends_on"]), lv.get(h["level"], 9), not h["blocks"]))
    for i, h in enumerate(out, 1):
        h["rank"] = i
    return out, objs


def pathway_weights(pathways: list[dict], targets: list[dict], hyps: list[dict], circuits: list[dict]) -> list[dict]:
    """Each pathway the agent named: its genes kept when they are therapy targets, its weight the summed support of the
    circuits whose edges carry those genes, and the checked drugs acting on them."""
    by = {t["gene"]: t for t in targets}
    sup = {int(c.get("rank") or 0): support_n(c)[0] / max(1, support_n(c)[1]) for c in circuits}
    out = []
    for p in pathways:
        if not isinstance(p, dict) or not p.get("name"):
            continue
        genes = [g for g in (str(x).upper() for x in p.get("genes") or []) if g in by]
        if not genes:
            continue
        circ = sorted({r for g in genes for r in by[g]["circuits"]})
        out.append({"name": str(p["name"]), "genes": genes, "circuits": circ,
                    "weight": round(sum(sup.get(r, 0.0) for r in circ), 3),
                    "drugs": sorted({h["drug"] for h in hyps if h["target"] in genes}),
                    "note": str(p.get("note") or "")})
    out.sort(key=lambda p: -p["weight"])
    return out


PERTURB = re.compile(r"block|inhibit|antagon|knock|deplet|neutraliz|antibod|explant|organoid|mice|murine|model|trial|"
                     r"treat|therap|blockade|deletion|deficien|ex vivo|in vitro|primary cell|silenc|agonist",
                     re.IGNORECASE)


def retrieved_papers(results: list, terms: set[str], limit: int = 120) -> list[dict]:
    """The papers the run already retrieved (every literature search that ran), once each: PMID, year, title, a quote
    and the search that returned it. Ranked for experiment design: papers describing a perturbation or a model first,
    then by how many of the circuits' genes and cell words they mention."""
    seen: dict[str, dict] = {}
    for r in results:
        r = r.model_dump() if hasattr(r, "model_dump") else r
        if r.get("tool") != "literature_search" or not r.get("ok"):
            continue
        for p in (r.get("data") or {}).get("papers") or []:
            pm = str(p.get("pmid") or "").strip()
            if pm and pm not in seen:
                seen[pm] = {"pmid": pm, "year": str(p.get("year") or ""), "title": str(p.get("title") or ""),
                            "quote": str((p.get("quotes") or [""])[0])[:240], "rid": r["id"]}
    genes = {t for t in terms if len(t) > 2 and t != t.lower()}  # gene symbols, matched as words
    words = {t for t in terms if len(t) > 2 and t == t.lower()}  # cell and tissue words
    for p in seen.values():
        text = f"{p['title']} {p['quote']}"
        g = sum(1 for t in genes if re.search(r"(?<![\w-])" + re.escape(t) + r"(?![\w])", text))
        w = sum(1 for t in words if t in text.lower())
        p["score"] = round(3 * bool(PERTURB.search(text)) + 2 * g + 0.5 * w
                           - (1 if re.search(r"\breview\b", p["title"], re.IGNORECASE) else 0), 2)
    return sorted(seen.values(), key=lambda p: (-p["score"], -int(p["year"] or 0)))[:limit]


def papers_text(papers: list[dict]) -> str:
    return "\n".join(f"- PMID {p['pmid']} ({p['year']}; search {p['rid']}) {p['title']}"
                     + (f' "{p["quote"]}"' if p["quote"] else "") for p in papers) or "(none retrieved)"
