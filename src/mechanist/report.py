"""Research report: assembled by code from records, so every statement traces to a record ID. No LLM.

Order: the integrated model (what the investigation concluded and how sure each link is), the critic's review, the
investigation itself as a question tree (question -> data pulled -> answer), required-question coverage, data limits,
the starting hypotheses, run metrics, and appendices with every tool result and finding referenced.
"""

from __future__ import annotations

import datetime as dt
import json
from collections import Counter
from pathlib import Path

from mechanist import investigate as inv
from mechanist.agents import run_items
from mechanist.findings import digest_line
from mechanist.llm import billed_cost
from mechanist.records import (
    Circuit,
    Dataset,
    Finding,
    Hypothesis,
    Objection,
    Question,
    Synthesis,
    ToolResult,
    Verification,
)

STATUS_ORDER = {"supported": 0, "partial": 1, "rejected": 2, "withdrawn": 3}


def hypothesis_metrics(hyps: list[Hypothesis], ver: dict[str, dict]) -> dict:
    vs = [Verification.model_validate(ver[h.id]["verification"]) for h in hyps if h.id in ver]
    objs = [Objection.model_validate(o) for h in hyps if h.id in ver for o in ver[h.id]["objections"]]
    steps = [s for v in vs for s in v.steps]
    cited = sum(len(s.valid_finding_ids) + len(s.invalid_finding_ids) for s in steps)
    invalid = sum(len(s.invalid_finding_ids) for s in steps)
    step_status = Counter(s.status for s in steps)
    return {
        "n_hypotheses": len(hyps),
        "hypothesis_status": dict(Counter(v.status for v in vs)),
        "n_steps": len(steps),
        "step_status": dict(step_status),
        "citations": cited,
        "invalid_citations": invalid,
        "overreach_steps": sum(o.kind == "overreach_level" for o in objs),
        "not_specific_steps": sum(o.kind == "not_specific" for o in objs),
        "has_non_mechanistic_alternative": any(h.non_mechanistic for h in hyps),
    }


def investigation_metrics(qs: list[Question], results: list[ToolResult], cov: list[dict],
                          answer_objs: list[Objection]) -> dict:
    n_req = len(cov)
    key = [c for c in cov if c["requirement"].get("priority", 3) == 1]
    return {
        "n_questions": len(qs),
        "by_kind": dict(Counter(q.kind for q in qs)),
        "by_status": dict(Counter(q.status for q in qs)),
        "by_asker": dict(Counter(q.asked_by for q in qs)),
        "max_chain_depth": inv.depth(qs),
        "follow_up_questions": sum(bool(q.parent_id) for q in qs),
        "tool_calls": len(results),
        "failed_tool_calls": sum(not r.ok for r in results),
        "required_questions": n_req,
        "required_asked": round(sum(bool(c["asked_by"]) for c in cov) / n_req, 3) if n_req else None,
        "required_resolved": round(sum(bool(c["resolved_by"]) for c in cov) / n_req, 3) if n_req else None,
        "key_required_resolved": round(sum(bool(c["resolved_by"]) for c in key) / len(key), 3) if key else None,
        "upstream_chains_complete": sum(bool(c.get("chain", {}).get("complete")) for c in cov),
        "candidates_judged": len({(j["question"], j["ligand"].upper()) for c in cov
                                  for j in c.get("chain", {}).get("judged", [])}),
        "candidates_by_verdict": dict(Counter(cd.verdict for q in qs for cd in q.candidates)),
        **reasoning_metrics(qs, results, cov),
        **thread_metrics(qs, results),
        "answers_observed": sum(q.conclusion == "observed" for q in qs),
        "answers_with_prior_knowledge": sum(bool(q.prior_knowledge) for q in qs),
        "open_at_end": sum(q.status == "open" for q in qs),
        "answer_citation_problems": len(answer_objs),
    }


def thread_metrics(qs: list[Question], results: list[ToolResult]) -> dict:
    """How the tool loops went: calls per thread, threads that never answered, repeated calls, follow-ups taken."""
    th = [q for q in qs if q.steps]
    if not th:
        return {}
    by_id = {r.id: r for r in results}
    calls = [sum(1 for s in q.steps if s.get("tool")) for q in th]
    return {
        "threads": len(th),
        "thread_calls_mean": round(sum(calls) / len(th), 2),
        "thread_calls_max": max(calls),
        "threads_without_answer": sum(any(s.get("no_answer") for s in q.steps) for q in th),
        "thread_repeat_calls": sum(1 for q in th for s in q.steps if s.get("repeat")),
        "threads_following_up": sum(bool(q.parent_id) for q in th),
        "threads_with_literature": sum(any((r := by_id.get(rid)) is not None and r.tool == "literature_search"
                                           for rid in q.result_ids) for q in th),
        "thread_reviews": sum(1 for q in th for s in q.steps if s.get("review")),
    }


def reasoning_metrics(qs: list[Question], results: list[ToolResult], cov: list[dict]) -> dict:
    """Disease-agnostic signs of thinking rather than sweeping: candidates weighed and rejected, verdicts backed by
    tests and by literature, ideas from outside the screen, and whether 'supported' just copies the screen's top line."""
    by_id = {r.id: r for r in results}
    by_q = {q.id: q for q in qs}
    judged = list({(j["question"], j["ligand"].upper()): j for c in cov
                   for j in c.get("chain", {}).get("judged", [])}.values())
    verdicts = [cd for q in qs for cd in q.candidates if cd.verdict != "untested"]
    screened, tops = set(), set()
    for r in results:
        if r.ok and r.tool == "upstream_screen":
            cands = r.data.get("candidates", [])
            screened |= {d["ligand"].upper() for d in cands}
            para = [d for d in cands if d.get("kind") == "paracrine"]
            if para:
                tops.add(para[0]["ligand"].upper())

    def lit(j):
        q = by_q.get(j["question"])
        return q is not None and any((r := by_id.get(rid)) is not None and r.ok and r.tool == "literature_search"
                                     for x in inv._thread(q, qs) for rid in x.result_ids)
    supported = [j for j in judged if j["verdict"] == "supported"]
    sizes = [len(r.args["gene"]) if isinstance(r.args.get("gene"), list) else 1 for r in results
             if r.tool in ("gene_profile", "gene_by_state")]
    return {
        "candidates_weighed": len(verdicts),
        "candidates_rejected_or_weakened": sum(cd.verdict in ("rejected", "weakened") for cd in verdicts),
        "verdicts_backed_by_test": round(len(judged) / len(verdicts), 3) if verdicts else None,
        "verdicts_with_literature": sum(lit(j) for j in judged),
        "candidates_outside_screen": sum(cd.ligand.upper() not in screened for cd in verdicts) if screened else None,
        "supported_equal_screen_top": (round(sum(j["ligand"].upper() in tops for j in supported) / len(supported), 3)
                                       if supported and tops else None),
        "largest_gene_query": max(sizes, default=0),
    }


def synthesis_metrics(syn: Synthesis | None, objs: list[Objection]) -> dict:
    if syn is None:
        return {"present": False}
    return {"present": True, "links": len(syn.links), "links_by_evidence": dict(Counter(lk.evidence for lk in syn.links)),
            "has_disease_link": bool(syn.disease_link.strip()), "open_questions": len(syn.open_questions),
            "experiments": len(syn.experiments), "competing_explanations": len(syn.competing_explanations),
            "unexamined_links": sum(not lk.question_ids for lk in syn.links),
            "link_depth_mean": (round(sum(lk.depth for lk in syn.links if lk.depth) / max(1, sum(1 for lk in syn.links
                                                                                          if lk.depth)), 2)
                                if any(lk.depth for lk in syn.links) else None),
            "links_with_disease_linkage": sum(bool(lk.disease_linkage and "not searched" not in lk.disease_linkage.lower())
                                              for lk in syn.links),
            "link_citation_problems": sum(o.kind == "invalid_citation" for o in objs)}


def llm_usage(run_dir: Path) -> dict:
    log = run_dir / "llm_calls.jsonl"
    if not log.exists():
        return {"calls": 0}
    rows = [json.loads(x) for x in log.read_text().splitlines() if x.strip()]
    return {"calls": len(rows), "models": sorted({r["model"] for r in rows}),
            "prompt_chars": sum(r.get("prompt_chars", 0) for r in rows),
            "prompt_tokens": sum((r.get("usage") or {}).get("prompt_tokens", 0) for r in rows),
            "completion_tokens": sum((r.get("usage") or {}).get("completion_tokens", 0) for r in rows),
            "seconds": round(sum(r.get("seconds", 0) for r in rows), 1),
            "cost_usd": round(sum(billed_cost(r.get("usage") or {}, r.get("model", "")) for r in rows), 2),
            "accounts": dict(Counter((r.get("usage") or {}).get("account") for r in rows
                                                 if (r.get("usage") or {}).get("account"))),
            "models_used": sorted({(r.get("usage") or {}).get("model") for r in rows} - {None}),
            # calls a safety classifier refused and the API's server-side fallback answered with another model
            "substituted_calls": [{"tag": r["tag"], "requested": (r.get("usage") or {}).get("model"),
                                   "served_by": r["usage"]["served_by"]}
                                  for r in rows if (r.get("usage") or {}).get("served_by")]}


def substitution_lines(run_dir: Path) -> list[str]:
    """Report section: every call answered by a fallback model after a refusal."""
    subs = llm_usage(run_dir).get("substituted_calls") or []
    if not subs:
        return []
    by: dict[str, list[str]] = {}
    for x in subs:
        by.setdefault(x["served_by"], []).append(x["tag"])
    lines = ["## Calls answered by a fallback model", "",
             (f"{len(subs)} calls were refused by the requested model's safety classifier and answered by the API's "
              f"server-side fallback. The answers below came from the model named, not the requested one."), ""]
    for model, tags in by.items():
        lines.append(f"- **{model}** ({len(tags)}): " + ", ".join(f"`{t}`" for t in tags))
    return lines + [""]


def _ids(ids: list[str]) -> str:
    return f" [{', '.join(ids)}]" if ids else ""


def _tree(qs: list[Question], results: dict[str, ToolResult]) -> list[str]:
    kids: dict[str | None, list[Question]] = {}
    ids = {q.id for q in qs}
    for q in qs:
        kids.setdefault(q.parent_id if q.parent_id in ids else None, []).append(q)
    out: list[str] = []

    def walk(q: Question, level: int):
        pad = "  " * level
        who = f"{q.asked_by}, round {q.round}" if q.round else q.asked_by
        about = ", ".join(q.cells + q.genes)
        out.append(f"{pad}- **{q.id}** `{q.kind}`{f' ({about})' if about else ''} — {q.text} _({who})_")
        if q.why:
            out.append(f"{pad}  - _why this thread:_ {q.why}")
        if q.steps:  # a thread: the reasoning before each call, then what the call returned
            for k, st in enumerate(q.steps, 1):
                r = results.get(st.get("result_id") or "")
                got = ("" if r is None else f" → {r.id}: " + (r.summary.splitlines()[0][:200] if r.ok
                                                             else f"FAILED: {r.summary[:140]}"))
                act = f"`{st['tool']}`" if st.get("tool") else "answer"
                out.append(f"{pad}  - step {k}, {act}: _{st.get('thought', '')[:400]}_{got}")
        else:
            for rid in q.result_ids:
                r = results.get(rid)
                if r:
                    first = r.summary.splitlines()[0][:220] if r.ok else f"FAILED: {r.summary[:160]}"
                    out.append(f"{pad}  - data {r.id} `{r.tool}`: {first}")
        if q.status == "open":
            out.append(f"{pad}  - **open** — not answered")
        else:
            out.append(f"{pad}  - **{q.status}** ({q.conclusion}): {q.answer}{_ids(q.answer_ids)}")
        if q.prior_knowledge:
            out.append(f"{pad}  - _background the model brought in (unsourced):_ " + "; ".join(q.prior_knowledge))
        if q.follow_ups:
            out.append(f"{pad}  - _raised:_ " + " | ".join(q.follow_ups[:3]))
        for k in kids.get(q.id, []):
            walk(k, level + 1)

    for q in kids.get(None, []):
        walk(q, 0)
    return out


GATED = (("dossier", "reviews", "1 Pathology dossier"), ("check", "reviews", "2 Do we see the pathology?"),
         ("candidates", "reviews", "3 Candidates"), ("linkage", "reviews", "6a Disease linkage"),
         ("experiment", "reviews", "8 Experiments"), ("therapy", "reviews", "9 Therapeutic hypotheses"),
         ("revise", "final_reviews", "6 Integrated model"), ("circuits", "reviews", "7 Circuits"))


EDGE_ARROW = {"observed": "==>", "likely": "-->", "analogy": "-.->", "speculative": "-.->"}


def _node(name: str, ids: dict[str, str]) -> str:
    if name not in ids:
        ids[name] = f"n{len(ids) + 1}"
    return ids[name]


def circuits_section(ctx: dict) -> list[str]:
    """Stage 7, the final model: each proposed circuit with a diagram (edge style = how it is known)."""
    cs = [Circuit.model_validate(c) for c in (ctx.get("circuits") or {}).get("circuits") or []]
    if not cs:
        return []
    L = ["## Final model: proposed circuits", "", (
        "With the facts settled (the evidence ledger below), the hypothesizer proposed complete circuits that could "
        "produce the disease as seen. Edges: thick = observed in this dataset, solid = likely (the data point to it, "
        "untested), dotted = published analogy or speculation. These are hypotheses for the experiments to decide."), ""]
    circ = ctx.get("circuits") or {}
    if circ.get("consensus"):
        k = len(circ.get("attempts") or [])
        L += [(f"Merged from {k} independent attempts on the same evidence: support says how many proposed each "
               "circuit. What recurs is the model; what one attempt alone proposed is an alternative or a lead to "
               "chase."), ""]
    for c in cs:
        L += [f"### {c.rank}. {c.title}" + (f" (support: {c.support})" if c.support else ""), "", c.summary, ""]
        if c.root:
            L += [f"**Root, and why here and not in the comparators:** {c.root}", ""]
        if c.edges:
            ids: dict[str, str] = {}
            lines = ["```mermaid", "graph LR"]
            for e in c.edges:
                a, b = _node(e.source, ids), _node(e.target, ids)
                label = (e.via or "").replace('"', "'").replace("|", "/")[:40]
                lab = f'|"{label} ({e.status})"|' if label else f'|"{e.status}"|'
                lines.append(f"  {a}{EDGE_ARROW.get(e.status, '-.->')}{lab}{b}")
            lines += [f'  {v}["{k.replace(chr(34), chr(39))[:60]}"]' for k, v in ids.items()] + ["```", ""]
            L += lines
            L += ["| From | Via | To | Known as | Ids | Note |", "|---|---|---|---|---|---|"]
            L += [f"| {e.source} | {e.via} | {e.target} | {e.status} | {', '.join(e.ids) or '—'} | {e.note} |"
                  for e in c.edges] + [""]
        for title, items in (("Loops", c.loops), ("Not used, and why", c.unused)):
            if items:
                L += [f"**{title}:**", ""] + [f"- {x}" for x in items] + [""]
        if c.phenotype:
            L += ["**How it produces the disease:**", ""] + [f"- {p.get('item', '')}: {p.get('how', '')}"
                                                             for p in c.phenotype] + [""]
        if c.predictions:
            L += ["**Predictions that tell it apart:**", ""] + [
                f"- {p.get('test', '')} → {p.get('expects', '')} ({'this dataset' if p.get('in_this_data') else 'needs new data'})"
                for p in c.predictions] + [""]
        if c.analogue:
            L += [f"**Closest published circuit:** {c.analogue}", ""]
        if c.confidence:
            L += [f"Confidence: {c.confidence}", ""]
    rk = (ctx.get("circuits") or {}).get("ranking")
    if rk:
        L += [f"**What would change the ranking:** {rk}", ""]
    objs = (ctx.get("circuits") or {}).get("objections") or []
    if objs:
        L += ["Verifier notes on the edge tags:", ""] + [f"- {o['text'] if isinstance(o, dict) else o.text}"
                                                         for o in objs] + [""]
    attempts = (ctx.get("circuits") or {}).get("attempts") or []
    if len(attempts) > 1:
        L += ["### Every attempt, as it stood (including alternatives that did not recur)", ""]
        for at in attempts:
            L += [f"#### Attempt {at.get('attempt')}", "", f"*Ranking:* {at.get('ranking', '')}", ""]
            for c in at.get("circuits") or []:
                L += [f"**{c.get('rank')}. {c.get('title', '')}**", "", c.get("summary", ""), ""]
                if c.get("root"):
                    L += [f"*Root:* {c['root']}", ""]
                edges = c.get("edges") or []
                if edges:
                    L += ["| From | Via | To | Known as | Ids |", "|---|---|---|---|---|"]
                    L += [f"| {e.get('source', '')} | {e.get('via', '')} | {e.get('target', '')} | {e.get('status', '')} | "
                          f"{', '.join(e.get('ids') or []) or '—'} |" for e in edges] + [""]
                if c.get("confidence"):
                    L += [f"*Confidence:* {c['confidence']}", ""]
    leads = (ctx.get("circuits") or {}).get("leads") or []
    if leads:
        L += ["### Leads found by one attempt only", "", ("Observations a single attempt used and the others did not: "
                                                          "worth chasing, not yet part of the model."), ""]
        L += [f"- {x.get('text', '')} ({', '.join(map(str, x.get('ids') or [])) or 'no id'}; attempt {x.get('attempt', '?')})"
              for x in leads] + [""]
    return L


def staged_gates(ctx: dict) -> dict:
    """Per stage: PI reviews, how the stage closed, and the progress counts the PI gave."""
    out = {}
    for key, field, name in GATED:
        rv = (ctx.get(key) or {}).get(field) or []
        if rv:
            out[name] = {"reviews": len(rv), "closed_by": rv[-1].get("closed_by"),
                         "new_tests": sum(r.get("new_tests", 0) for r in rv),
                         "new_testable_predictions": sum(r.get("new_testable_predictions", 0) for r in rv)}
    pr = (ctx.get("chase") or {}).get("pi_reviews") or []
    if out and pr:
        out["4 Chase"] = {"reviews": len(pr), "closed_by": (ctx.get("chase") or {}).get("stop") or "?",
                          "new_tests": sum(r.get("new_tests", 0) for r in pr),
                          "new_testable_predictions": sum(r.get("new_testable_predictions", 0) for r in pr)}
    return dict(sorted(out.items()))


def _staged_sections(ctx: dict, syn: Synthesis | None, gates: dict, design: str = "cohort") -> list[str]:
    L = ["## The PI's gates", "", "| Stage | Reviews | Closed by | New tests | New testable predictions |",
         "|---|---|---|---|---|"]
    L += [f"| {k} | {g['reviews']} | {g['closed_by']} | {g['new_tests']} | {g['new_testable_predictions']} |"
          for k, g in gates.items()]
    last = [((ctx.get(k) or {}).get(f) or [{}])[-1] for k, f, _ in GATED]
    L += [""] + [f"- **{name}**: {r.get('assessment', '')}" for (_, _, name), r in zip(GATED, last) if r.get("assessment")]
    if syn is not None:
        steps = [(i, st) for i, lk in enumerate(syn.links, 1) for st in lk.steps]
        if steps:
            L += ["", "### Chain steps by kind", "", "| Link | Kind | Step | Ids | Test |", "|---|---|---|---|---|"]
            L += [f"| {i} | {st.kind} | {st.claim} | {', '.join(st.ids) or '—'} | {st.test or '—'} |" for i, st in steps]
        if any(lk.involved or lk.specific for lk in syn.links):
            L += ["", "### Involved vs specific", "", ("Involved: changed against healthy tissue (part of the pathology). "
                  "Specific: different from the related comparator disease. Involved but shared is still a real "
                  "driver."), "", "| Link | Involved (vs healthy) | Specific (vs comparator) |", "|---|---|---|"]
            L += [f"| {lk.source} → {lk.mediator} → {lk.target} | {lk.involved or '—'} | {lk.specific or '—'} |"
                  for lk in syn.links]
        if syn.phenotype_fit:
            items = {i: (sec, t) for i, sec, t in run_items(ctx, design)}
            counts = Counter(f.verdict for f in syn.phenotype_fit)
            L += ["", "### How the model accounts for the disease", "", (
                  "Each item of the stage 1 dossier (from the literature), as the final model accounts for it: "
                  + ", ".join(f"{counts.get(v, 0)} {v.replace('_', ' ')}" for v in
                              ("explained", "consistent", "contradicted", "not_testable"))
                  + (f"; {len(set(items) - {f.item for f in syn.phenotype_fit})} not accounted for." if items else ".")),
                  "", "| Item | Kind | What is known | Verdict | Through | Why |", "|---|---|---|---|---|---|"]
            for f in syn.phenotype_fit:
                sec, txt = items.get(f.item.strip().upper(), ("—", "(not a dossier item)"))
                L.append(f"| {f.item} | {sec} | {txt[:180]} | {f.verdict.replace('_', ' ')} | "
                         f"{'; '.join(f.links) or '—'} | {f.reason} |")
        if syn.best_guesses:
            L += ["", "### Best guesses where the data ran out", "", ("Evidence rungs: 1 a separating test here, 2 an "
                  "association within one sample, 3 a correlation across samples, 4 published work elsewhere, "
                  "5 plausibility alone."), ""]
            for bg in syn.best_guesses:
                L.append(f"**{bg.step}**")
                L += [f"{c.rank}. {c.explanation} (rung {c.evidence_rung or '?'}; {c.confidence or 'confidence not given'})"
                      + (f": {c.evidence}" if c.evidence else "") + (f" [{', '.join(c.ids)}]" if c.ids else "")
                      for c in sorted(bg.candidates, key=lambda c: c.rank)]
                if bg.decisive_test:
                    L.append(f"   - would decide: {bg.decisive_test}")
                L.append("")
        if syn.alternatives:
            L += ["", "### Ranked hypotheses", "", f"Confidence in the lead model: **{syn.confidence or '—'}**", ""]
            for a in sorted(syn.alternatives, key=lambda a: a.rank):
                L += [f"{a.rank}. **{a.title}** ({a.confidence or '—'}): {a.summary}"]
                if a.evidence_for:
                    L.append(f"   - for: {'; '.join(a.evidence_for)}")
                if a.evidence_against:
                    L.append(f"   - against: {'; '.join(a.evidence_against)}")
        if syn.experiments_planned and not any("tests" in e for e in syn.experiments_planned):
            L += experiments_section(ctx)
    lk = (ctx.get("linkage") or {}).get("links") or []
    if lk:
        L += ["", "## Disease linkage (each link across medicine)", "", ("Literature and Open Targets, per link: this "
              "disease, the diseases it occurs with, related diseases, the same state in other tissues, genetics and "
              "therapy. Other studies, not evidence about these samples."), ""]
        for x in lk:
            L += [f"**{x['link']}**: {x['suggests']}", ""]
            L += [f"- *{f['aspect']}*: {f['text']} [{', '.join(f['ids'])}]" for f in x["findings"]] + [""]
    dos = (ctx.get("dossier") or {}).get("dossier")
    if dos:
        from mechanist.agents import check_text, dossier_text
        L += ["", "## Stage 1: pathology dossier", "", dossier_text(dos)]
        if (ctx.get("check") or {}).get("summary"):
            L += ["", "## Stage 2: do we see the pathology?", "", check_text(ctx["check"]["summary"])]
    return L + [""]


def build(cfg, ctx: dict, run_dir: Path) -> tuple[str, dict]:
    ds = Dataset.model_validate(ctx["load"])
    fs = {f["id"]: Finding.model_validate(f) for f in ctx["findings"]["findings"]}
    hsrc = "single" if cfg.mode == "single" else "candidates" if ctx.get("candidates") else "generate"
    hyps = [Hypothesis.model_validate(h) for h in (ctx[hsrc]["hypotheses"] if hsrc == "candidates" else ctx[hsrc])]
    hyps += [Hypothesis.model_validate(h) for h in (ctx.get("scout") or {}).get("new_hypotheses", [])]
    ver = {r["hypothesis_id"]: r for r in ctx["verify_r0"] + (ctx.get("scout") or {}).get("verification", [])}
    inv_d = ctx.get("chase") or ctx["investigate"]
    qs = [Question.model_validate(q) for q in inv_d["questions"]]
    results = {r["id"]: ToolResult.model_validate(r) for r in inv_d["results"]}
    cov = inv_d["coverage"]
    answer_objs = [Objection.model_validate(o) for o in inv_d.get("answer_objections", [])]
    syn_d = ctx.get("revise") or ctx["synthesize"]
    qs += [Question.model_validate(q) for q in syn_d.get("final_questions", [])]  # stage 6 demands chased
    for r in syn_d.get("final_results", []) + (ctx.get("experiment") or {}).get("results", []):
        results.setdefault(r["id"], ToolResult.model_validate(r))
    syn = Synthesis.model_validate(syn_d["synthesis"]) if syn_d.get("synthesis") else None
    if syn is not None and (ctx.get("experiment") or {}).get("experiments"):
        syn.experiments_planned = ctx["experiment"]["experiments"]  # designed on the final ranked hypotheses
    syn_objs = [Objection.model_validate(o) for o in syn_d.get("objections", [])]
    crit = ctx.get("critique")
    scout = ctx.get("scout") or {}

    m = {"study": ds.study, "disease": cfg.disease, "mode": cfg.mode, "model": cfg.llm.model or cfg.llm.backend,
         "findings_total": len(fs), "starting_hypotheses": hypothesis_metrics(hyps, ver),
         "investigation": investigation_metrics(qs, list(results.values()), cov, answer_objs),
         "synthesis": synthesis_metrics(syn, syn_objs), "llm": llm_usage(run_dir)}
    if scout.get("results") or scout.get("assessments"):
        papers = {p["pmid"] for r in scout.get("results", []) for p in (r.get("data") or {}).get("papers", [])}
        m["scout"] = {"searches": len(scout.get("results", [])),
                      "failed_searches": sum(not r["ok"] for r in scout.get("results", [])),
                      "papers": len(papers),
                      "assessments": dict(Counter(a["support"] for a in scout.get("assessments", []))),
                      "unverified_assessments": sum(not a["verified"] for a in scout.get("assessments", [])),
                      "new_hypotheses": len(scout.get("new_hypotheses", []))}
    lit_results = [r for r in results.values() if r.tool == "literature_search" and r.ok]
    m["investigation"]["literature_searches"] = len(lit_results)
    m["investigation"]["answers_from_literature"] = sum(q.conclusion == "literature" for q in qs)
    gates = staged_gates(ctx)
    if gates:
        m["gates"] = gates
        if syn is not None:
            kinds = Counter(st.kind for lk in syn.links for st in lk.steps)
            m["synthesis"]["step_kinds"] = dict(kinds)
            m["synthesis"]["alternatives"] = len(syn.alternatives)
            m["synthesis"]["best_guesses"] = len(syn.best_guesses)
            m["synthesis"]["experiments_planned"] = len(syn.experiments_planned)
        dos = (ctx.get("dossier") or {}).get("dossier") or {}
        m["dossier"] = {"items": sum(len(v) for k, v in dos.items() if k != "novelty_leads"),
                        "novelty_leads": len(dos.get("novelty_leads") or []),
                        "dropped_unbacked": (ctx.get("dossier") or {}).get("dropped", 0),
                        "searches": len((ctx.get("dossier") or {}).get("results", []))}
    if crit:
        m["critic"] = {"verdict": crit.get("verdict"), "objections": len(crit.get("objections", [])),
                       "questions": len(crit.get("questions", []))}

    L: list[str] = []
    L += [f"# Mechanist investigation — {ds.study}, {cfg.disease} ({cfg.mode}-agent)", "",
          ("> **Research hypotheses only.** Not clinical conclusions, not evidence of causality, and not treatment "
          "recommendations. All evidence is single-cell transcript abundance."), "",
          (f"Run `{run_dir.name}` · {dt.date.today().isoformat()} · model `{m['model']}` · "
          f"{len(qs)} questions · {len(results)} data queries"), ""]

    circ = circuits_section(ctx)
    L += circ
    if circ and (ctx.get("experiment") or {}).get("experiments") and any("tests" in e for e in ctx["experiment"]["experiments"]):
        L += experiments_section(ctx) + [""]
    L += therapy_section(ctx) + ([""] if (ctx.get("therapy") or {}).get("hypotheses") else [])
    L += ["## Evidence ledger (the grounded model behind the circuits)" if circ else "## Integrated model", ""]
    if syn is None:
        L += ["_No usable synthesis was produced._", ""]
    else:
        L += [f"### {syn.title}", "", syn.summary, "",
              ("Each arrow is an inference. What transcript data can show is whether the source expresses the mediator "
              "and whether the target expresses its receptor or response genes."), "",
              ("| # | From | Mediator | To | Mediator in source | Receptor/response in target | Status | Depth | Data | "
              "Questions | Note | Disease linkage |"), "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        L += [f"| {i} | {lk.source} | {lk.mediator} | {lk.target} | {lk.mediator_in_source} | "
              f"{lk.receptor_in_target} | **{lk.evidence}** | {lk.depth or '—'}/7 | {', '.join(lk.ids) or '—'} | "
              f"{', '.join(lk.question_ids) or '**unexamined**'} | {lk.note} | {lk.disease_linkage or '—'} |"
              for i, lk in enumerate(syn.links, 1)]
        L += ["", ("Depth is the highest level of the PI ladder the link reached: 1 real, 2 where, 3 regulation, "
                   "4 signal, 5 upstream of the sender, 6 downstream, 7 disease linkage.")]
        L += ["", f"**Link to the disease:** {syn.disease_link or '—'}", ""]
        for title, items in (("Competing explanations", syn.competing_explanations),
                             ("Open questions", syn.open_questions),
                             ("Experiments that would decide", syn.experiments)):
            if items:
                L += [f"**{title}:**", ""] + [f"- {x}" for x in items] + [""]
        if syn_objs:
            L += ["Verifier notes on the model:", ""] + [f"- `{o.kind}` ({o.target}): {o.text}" for o in syn_objs] + [""]

    if gates:
        L += _staged_sections(ctx, syn, gates, cfg.design)
    reviews = (ctx.get("chase") or {}).get("pi_reviews") or []
    if len(reviews) > 1:
        L += ["## PI review rounds", "", ("| Round | Verdict | Signed off | Demands | New tests | New predictions | "
              "Shallowest links |"), "|---|---|---|---|---|---|---|"]
        for rv in reviews:
            shallow = sorted([d for d in rv.get("depth", []) if isinstance(d.get("level"), int)],
                             key=lambda d: d["level"])[:3]
            L.append(f"| {rv['round']} | {rv.get('verdict')} | {'yes' if rv.get('sign_off') else 'no'} | "
                     f"{rv.get('demands')} | {rv.get('new_tests', '—')} | {rv.get('new_testable_predictions', '—')} | "
                     + "; ".join(f"{d.get('link')} ({d.get('level')})" for d in shallow) + " |")
        L.append("")
    if crit:
        L += ["## Critic review (of the first model; its questions were chased before revision)", "",
              f"Verdict: **{crit.get('verdict')}**", ""]
        for o in crit.get("objections", []):
            ob = Objection.model_validate(o)
            L.append(f"- `{ob.kind}` ({ob.target}): {ob.text}{_ids(ob.finding_ids)}"
                     + ("" if ob.valid else " **— cites evidence that does not exist**"))
        L.append("")

    if scout.get("assessments") or scout.get("new_hypotheses"):
        L += ["## Prior work (literature scout)", "",
              ("Searches of PubMed abstracts and curated databases before the data investigation. Literature describes "
               "other studies; it proposes, the data investigation tests."), "",
              "| Hypothesis | Step | Prior work | Results | PMIDs | Quote / note |", "|---|---|---|---|---|---|"]
        for a in scout.get("assessments", []):
            quote = f"“{a['quote']}” " if a.get("quote") else ""
            L.append(f"| {a['hypothesis_id']} | {a['step_id']} | **{a['support']}**"
                     f"{'' if a['verified'] else ' (unverified)'} | {', '.join(a['result_ids']) or '—'} | "
                     f"{', '.join(a['pmids']) or '—'} | {quote}{a['note']} |")
        if scout.get("new_hypotheses"):
            L += ["", "New hypotheses suggested by the reading (tested in the investigation below):", ""]
            for h in scout["new_hypotheses"]:
                chain = "; ".join(f"{st['source_cell']} → {st['mediator']}"
                                  + (f" → {st['target_cell']}" if st.get("target_cell") else "") for st in h["steps"])
                L.append(f"- **{h['id']}** {h['title']} — {chain} (from {', '.join(h['inspired_by'])}; PMID "
                         f"{', '.join(h['pmids']) or '—'})")
        L.append("")

    reg = inv.step_registry(qs)
    if reg:
        L += ["## Causal steps", "", ("Every question names the causal step it works on. Exhausted: the data could not "
              "settle the step; its decisive observation is the experiment."), "",
              "| Step | Questions | Tests tried | Exhausted | Would be settled by |", "|---|---|---|---|---|"]
        L += [f"| {st['id']} {st['label']} | {', '.join(f'{i} ({s})' for i, s in st['questions'])} | "
              f"{', '.join(st['tools']) or '—'} | {'yes' if st['exhausted'] else 'no'} | "
              f"{st['decisive'][-1] if st['decisive'] else '—'} |" for st in reg]
        dfr = (ctx.get("chase") or {}).get("deferred") or []
        if dfr:
            L += ["", "Asks for an exhausted step that named no new kind of test (not run; their best guesses stand):", ""]
            L += [f"- {d['step']} ({d['asked_by']}): {d['text'][:300]}" for d in dfr]
        L.append("")
    L += ["## Investigation", "", ("Each question, the data pulled to answer it, and the answer. Indented questions "
          "follow from the one above."), ""]
    L += _tree(qs, results) or ["_No questions were asked._"]
    if answer_objs:
        L += ["", "Verifier notes on answers:", ""] + [f"- `{o.kind}` ({o.target}): {o.text}" for o in answer_objs]

    L += ["", "## Required questions", "", "Priority 1 = key mediator of a starting hypothesis.", "",
          "| Priority | Kind | Question | Asked | Resolved |", "|---|---|---|---|---|"]
    for c in cov:
        r = c["requirement"]
        chain = ""
        if c.get("chain"):
            st = c["chain"]["steps"]
        if c.get("chain", {}).get("premise_failed"):
            chain = " (premise fails: the gene is not elevated in this cell)"
        elif c.get("chain"):
            chain = (" (chain: regulators " + ("✓" if st["regulators"] else "✗") + " → receptor routes "
                     + ("✓" if st["routes"] else "✗") + " → receptor on the cell "
                     + (f"✓ {len(c['chain']['receptors_on_cell'])}" if st["receptor_on_cell"] else "✗")
                     + " → ligands that differ in the disease "
                     + (f"✓ {len(c['chain']['ligands_profiled'])}" if st["ligand_profiled"] else "✗")
                     + (" → candidates judged and tested " + ("✓ " + "; ".join(
                         f"{j['ligand']}{'@' + j['sender'] if j['sender'] else ''} {j['verdict']} ({j['question']})"
                         for j in c["chain"].get("judged", [])) if st["candidate_judged"] else "✗")
                        if "candidate_judged" in st else "") + ")")
        L.append(f"| {r.get('priority', 3)} | {r['kind']} | {r['text']}{chain} | {', '.join(c['asked_by']) or '**no**'} | "
                 f"{', '.join(c['resolved_by']) or '**no**'} |")

    L += ["", "## Data and its limits", "", "| Group | Samples | n |", "|---|---|---|"]
    L += [f"| {g} | {', '.join(s)} | {len(s)} |" for g, s in ds.groups.items()]
    L += [""] + [f"- {c}" for c in ds.caveats]
    chance = ctx["findings"].get("chance_separations") or {}
    if chance:
        L += ["- chance: full sample separations expected across the scanned genes by chance alone: "
              + "; ".join(f"{k} ≈ {v}" for k, v in chance.items())]
    if cfg.disease_phenotype:
        L += [f"- phenotype given to the agents (configuration, not data): {cfg.disease_phenotype}"]
    L += ["", "Source files:", ""] + [f"- `{f.path}` (sha256 {f.sha256[:12]})" for f in ds.files]

    if ctx.get("panel"):
        L += ["", "## Specialist panel", "", ("Each specialist proposed hypotheses independently (high temperature); "
              "the PI merged them (low temperature). Origin shows whose ideas each starting hypothesis carries."), ""]
        for who, hs in ctx["panel"].items():
            L.append(f"- **{who}**: " + ("; ".join(h["title"] for h in hs) or "no usable proposal"))
    L += ["", "## Starting hypotheses", ""]
    for h in sorted(hyps, key=lambda h: (STATUS_ORDER.get(ver.get(h.id, {}).get("verification", {}).get("status"), 9),
                                         h.id)):
        v = ver.get(h.id)
        st = v["verification"]["status"] if v else "?"
        tag = " · non-mechanistic" if h.non_mechanistic else ""
        chain = "; ".join(f"{s.source_cell} → {s.mediator}" + (f" → {s.target_cell}" if s.target_cell else "")
                          for s in h.steps)
        origin = f" (from {', '.join(h.origin)})" if h.origin else ""
        L.append(f"- **{h.id}** {h.title}{origin} — checked against findings: **{st}**{tag}. {chain}")

    L += [""] + substitution_lines(run_dir)
    L += ["", "## Run metrics", "", "```json", json.dumps(m, indent=1), "```", "",
          "## Appendix A — data queries", ""]
    for r in results.values():
        L += [f"**{r.id}** `{r.tool}({json.dumps(r.args)})`" + ("" if r.ok else " — failed"), "", "```",
              r.summary, "```"] + ([f"_caveats: {'; '.join(r.caveats)}_"] if r.caveats else []) + [""]
    cited = sorted({i for q in qs for i in q.answer_ids if i.startswith("F")}
                   | {i for lk in (syn.links if syn else []) for i in lk.ids if i.startswith("F")}
                   | {fid for r in ver.values() for s in r["verification"]["steps"] for fid in s["valid_finding_ids"]})
    L += ["## Appendix B — findings referenced", ""]
    L += [f"- {digest_line(fs[i])}" + (f"  \n  per sample: {fs[i].per_sample}" if fs[i].per_sample else "")
          for i in cited if i in fs]
    return "\n".join(L) + "\n", m


def experiments_section(ctx: dict) -> list[str]:
    """Experiments ranked by code (how much the circuits disagree on the outcome, then tier), with each circuit's
    predicted outcome; older runs (one prediction text per explanation) are shown as they were written."""
    ex = (ctx.get("experiment") or {})
    exps = ex.get("experiments") or []
    if not exps:
        return []
    if not any("tests" in e for e in exps):  # written before the circuits-based stage
        L = ["", "### Experiments", "", ("| Link | Precedent | Model | Perturbation | Readout | What each explanation "
                                       "predicts |"), "|---|---|---|---|---|---|"]
        for e in sorted(exps, key=lambda e: e.get("priority") or 9):
            prec = "; ".join(f"{p['what']} [{', '.join(p['ids'])}]" for p in e.get("precedent") or []) or "—"
            pred = "; ".join(f"{p.get('explanation')}: {p.get('expects')}" for p in e.get("predictions") or [])
            L.append(f"| {e.get('link')} | {prec} | {e.get('model')} | {e.get('perturbation')} | {e.get('readout')} | "
                     f"{pred or '—'} |")
        return L
    ranks = sorted({p["circuit"] for e in exps for p in e.get("predictions") or []})
    L = ["", "### Experiments that would decide between the circuits", "",
         ("Ranked by code: first by how much the circuits disagree on the outcome (weighted by their support), then by "
          "the cheapest tier that decides. M ids are the missing links each experiment tests."), "",
         "| # | Experiment | Tests | Tier | " + " | ".join(f"Circuit {r}" for r in ranks) + " |",
         "|---|---|---|---|" + "---|" * len(ranks)]
    for e in exps:
        pr = {p["circuit"]: p["outcome"] for p in e.get("predictions") or []}
        L.append(f"| {e.get('rank')} | **{e.get('title') or e.get('perturbation', '')[:60]}**: {e.get('perturbation')}; "
                 f"readout {e.get('readout')} | {', '.join(e.get('tests') or []) or '—'} | {e.get('tier') or '—'} | "
                 + " | ".join(pr.get(r, "—") for r in ranks) + " |")
    if ex.get("not_testable"):
        L += ["", "Not testable with feasible experiments: " + "; ".join(f"{x['id']} ({x['why']})"
                                                                        for x in ex["not_testable"])]
    if ex.get("uncovered"):
        L += ["", f"Missing links left without an experiment: {', '.join(ex['uncovered'])}."]
    return L


def therapy_section(ctx: dict) -> list[str]:
    """Therapeutic hypotheses for research, not recommendations: drugs from Open Targets lookups, with stage, action
    and evidence level set by code."""
    th = ctx.get("therapy") or {}
    hs = th.get("hypotheses") or []
    if not hs and not th.get("pathways"):
        return []
    L = ["", "## Therapeutic hypotheses (research only, not recommendations)", "",
         ("Drugs acting on the circuits' steps, from Open Targets lookups that ran. Clinical stage, action and evidence "
          "level are set by code from the lookup. A hypothesis that depends on an untested link (M id) should wait for "
          "that link's experiment."), "",
         "| # | Drug | Target | Evidence | Action | Circuits | Breaks | Should fail under | Depends on |",
         "|---|---|---|---|---|---|---|---|---|"]
    for h in hs:
        L.append(f"| {h['rank']} | {h['drug']} | {h['target']} | {h['level']} ({h['stage'] or '?'}) | "
                 f"{', '.join(h['actions']) or '?'}{'' if h['blocks'] else ' (not a blocker)'} | "
                 f"{', '.join(map(str, h['circuits'])) or '—'} | {h['breaks']} | "
                 f"{', '.join(map(str, h['fails_under'])) or '—'} | {', '.join(h['depends_on']) or '—'} |")
    if th.get("pathways"):
        L += ["", "**Pathways the circuits converge on** (weight: summed support of the circuits carrying them):", ""]
        L += [f"- {p['name']} ({p['weight']}): {', '.join(p['genes'])}; drugs: {', '.join(p['drugs']) or 'none checked'}"
              + (f". {p['note']}" if p.get("note") else "") for p in th["pathways"]]
    return L
