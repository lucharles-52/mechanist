import json

from conftest import fid, make_cells

from mechanist.cli import main
from mechanist.findings import FindingsConfig
from mechanist.llm import LLMConfig, Replay
from mechanist.runner import RunConfig, Runner


def _replay(findings) -> dict:
    cx = fid(findings, "panel_expression", "Fibroblast|CXCL13|PG|Control")
    comp = fid(findings, "composition", "B cell")
    gen = {"hypotheses": [
        {"title": "Fibroblast CXCL13 recruits B cells", "summary": "s", "disease": "PG",
         "steps": [{"source_cell": "Fibroblast", "target_cell": "B cell", "mediator": "CXCL13", "genes": ["CXCL13"],
                    "direction": "up", "finding_ids": [cx[:1] + cx[2:]]}]},  # 'F004'-style id: repaired
        {"title": "B-cell excess is composition only", "summary": "s", "disease": "PG", "non_mechanistic": True,
         "steps": [{"source_cell": "B cell", "mediator": "proportion", "direction": "up", "finding_ids": [comp]}]},
    ]}
    r1 = {"answers": [], "questions": [
        {"kind": "presence", "text": "Why are B cells expanded?", "cells": ["B cell"], "hypothesis_ids": ["H1"],
         "calls": [{"tool": "lineage_share", "args": {"lineage": "B cell"}}]},
        {"kind": "function", "text": "What are the B cells doing?", "cells": ["B cell"],
         "calls": [{"tool": "state_composition", "args": {"lineage": "B cell"}}]},
        {"kind": "upstream", "text": "Why do fibroblasts make CXCL13?", "cells": ["Fibroblast"], "genes": ["CXCL13"],
         "calls": [{"tool": "incoming_signals", "args": {"lineage": "Fibroblast"}},
                   {"tool": "gene_profile", "args": {"gene": "NOTAGENE"}}]},
    ]}
    r2 = {"answers": [
        {"question_id": "Q01", "status": "answered", "answer": "B cells are 55% of PG cells (R0001).",
         "answer_ids": ["R1"], "conclusion": "observed"},
        {"question_id": "Q02", "status": "answered", "answer": "Activated HLA-II+ state (R0002).",
         "answer_ids": ["R0002", "R0999"], "conclusion": "observed"},
        {"question_id": "Q03", "status": "partial", "answer": "IL1B->IL1R1 rises (R0003).", "answer_ids": ["R0003"],
         "conclusion": "inferred"}],
        "questions": [{"kind": "relation", "text": "Do CXCL13 and CXCL1 come from the same fibroblasts?",
                       "cells": ["Fibroblast"], "genes": ["CXCL13", "CXCL1"], "hypothesis_ids": ["H1"],
                       "parent_id": "Q03", "calls": [{"tool": "coexpression", "args": {
                           "lineage": "Fibroblast", "genes_a": ["CXCL13"], "genes_b": ["CXCL1"]}}]}]}
    r3 = {"answers": [{"question_id": "Q04", "status": "answered", "answer": "Same cells, x2 (R0005).",
                       "answer_ids": ["R0005"], "conclusion": "observed"}], "questions": []}
    syn = {"title": "Inflammatory fibroblast program recruits activated B cells", "summary": "s",
           "links": [{"source": "Fibroblast", "target": "B cell", "mediator": "CXCL13", "mediator_in_source": "observed",
                      "receptor_in_target": "observed", "ids": ["R0005", cx], "question_ids": ["Q04"]},
                     {"source": "B cell", "target": "ulcer", "mediator": "?", "mediator_in_source": "observed",
                      "ids": [], "question_ids": []}],
           "disease_link": "untested", "open_questions": ["what do B cells secrete?"], "experiments": ["IHC"]}
    critic = {"verdict": "weak", "objections": [{"target": "link 2", "kind": "made_up", "text": "No data.",
                                                 "finding_ids": ["F9999"]}],
              "questions": [{"kind": "artifact", "text": "Is PG4L dominating?", "cells": ["B cell"],
                             "hypothesis_ids": ["H1"], "calls": [{"tool": "state_markers",
                                                                   "args": {"state": "Activated HLA-II+ B cell"}}]}]}
    chase = {"answers": [{"question_id": "Q05", "status": "answered", "answer": "HLA-DRA top (R0006).",
                          "answer_ids": ["R0006"], "conclusion": "observed"}], "questions": []}
    return {"generate": json.dumps(gen), "single": json.dumps(gen),
            "investigate:r1": json.dumps(r1), "investigate:r2": json.dumps(r2), "investigate:r3": json.dumps(r3),
            "investigate:r4": json.dumps({"answers": [], "questions": []}),
            "synthesize": json.dumps(syn), "critique_model": json.dumps(critic),
            "chase:r1": json.dumps(chase), "chase:r2": json.dumps({"answers": []}),
            "revise": json.dumps(syn | {"title": "Revised model"})}


def _cfg(projects, mode="multi"):
    # Agent-flow tests run on the synthetic lab panel tables; the genome-wide path is tested in test_expression.py.
    return RunConfig(study="p033", adapter="p033", disease="PG", projects_root=str(projects), mode=mode,
                     disease_phenotype="neutrophilic ulcer",
                     investigation="rounds", investigate_rounds=4, knowledge=False, literature=False, scout=False, panel=[],
                     lab_tables=True, findings=FindingsConfig(expression_source="none"),
                     llm=LLMConfig(backend="replay"))


def test_multi_agent_investigation_end_to_end_and_resume(projects, findings, tmp_path):
    run_dir = tmp_path / "runs" / "m"
    Runner(_cfg(projects), run_dir, llm_backend=Replay(_replay(findings)), cells=make_cells()).run()

    inv = json.loads((run_dir / "chase.json").read_text())
    qs = {q["id"]: q for q in inv["questions"]}
    assert [q["asked_by"] for q in qs.values()] == ["investigator"] * 4 + ["critic"]
    assert qs["Q04"]["parent_id"] == "Q03" and qs["Q01"]["answer_ids"] == ["R0001"]
    assert qs["Q02"]["answer_ids"] == ["R0002"]  # R0999 removed by the answer check
    assert not inv["results"][3]["ok"]  # NOTAGENE failed as a result, not an exception
    assert inv["results"][4]["data"]["P1"]["ratio"] == 2.0

    syn = json.loads((run_dir / "revise.json").read_text())
    assert syn["synthesis"]["title"] == "Revised model"
    assert syn["synthesis"]["links"][0]["evidence"] == "expression_observed"
    assert syn["synthesis"]["links"][1]["evidence"] == "not_observed"  # 'observed' without evidence is reset

    m = json.loads((run_dir / "report.json").read_text())
    i = m["investigation"]
    assert i["n_questions"] == 5 and i["max_chain_depth"] == 2 and i["failed_tool_calls"] == 1
    assert 0 < i["required_asked"] < 1 and i["answer_citation_problems"] == 1
    assert m["critic"]["verdict"] == "weak" and m["synthesis"]["unexamined_links"] == 1
    assert m["synthesis"]["links_by_evidence"] == {"expression_observed": 1, "not_observed": 1}
    assert m["starting_hypotheses"]["invalid_citations"] == 0  # 'F004'-style id was repaired

    report = (run_dir / "report.md").read_text()
    for text in ("## Integrated model", "## Investigation", "  - **Q04** `relation`", "## Required questions",
                 "**no**", "cites evidence that does not exist", "phenotype given to the agents"):
        assert text in report, text

    calls = (run_dir / "llm_calls.jsonl").read_text().count("\n")
    Runner(_cfg(projects), run_dir, llm_backend=Replay({}), cells=make_cells()).run()
    assert (run_dir / "llm_calls.jsonl").read_text().count("\n") == calls


def test_single_agent_baseline_and_compare(projects, findings, tmp_path, capsys):
    root = tmp_path / "runs"
    Runner(_cfg(projects, "single"), root / "s", llm_backend=Replay(_replay(findings)), cells=make_cells()).run()
    Runner(_cfg(projects), root / "m", llm_backend=Replay(_replay(findings)), cells=make_cells()).run()
    s = json.loads((root / "s" / "report.json").read_text())
    assert s["mode"] == "single" and "critic" not in s and s["investigation"]["n_questions"] == 4
    assert not (root / "s" / "chase.json").exists()
    main(["compare", str(root / "m"), str(root / "s")])
    out = capsys.readouterr().out
    assert "investigation.required_asked" in out and "investigation.n_questions" in out


def test_stop_after_findings_makes_no_llm_calls(projects, tmp_path):
    run_dir = tmp_path / "runs" / "f"
    Runner(_cfg(projects), run_dir, llm_backend=Replay({})).run(stop_after="findings")
    assert (run_dir / "findings.json").exists() and not (run_dir / "llm_calls.jsonl").exists()


def test_follow_up_quota_drops_excess_new_topics(projects, tmp_path):
    from mechanist.records import Question
    r = Runner(_cfg(projects), tmp_path / "q", llm_backend=Replay({}))
    qs = [Question(id="Q01", text="a"), Question(id="Q02", text="b")]
    raws = [{"text": "f1", "parent_id": "Q01"}, {"text": "n1"}, {"text": "n2"}, {"text": "n3", "parent_id": "Q77"}]
    keep, dropped = r._admit(raws, qs, 8, rnd=2)
    assert [k["text"] for k in keep] == ["f1", "n1"] and dropped == 2
    keep, dropped = r._admit(raws, qs, 8, rnd=1)  # first round: no quota
    assert len(keep) == 4 and dropped == 0


def test_failed_calls_and_made_up_numbers_are_caught(projects, findings, tmp_path):
    from mechanist.records import Question, ToolResult
    r = Runner(_cfg(projects), tmp_path / "g", llm_backend=Replay({}))
    res = [ToolResult(id="R0001", tool="gene_profile", args={}, ok=False, summary="error"),
           ToolResult(id="R0002", tool="gene_profile", args={}, ok=True, summary="B cell: PG: P1 1.0%")]
    qs = [Question(id="Q01", text="IL6 in B cells?", cells=["B cell"], genes=["IL6"], result_ids=["R0001"]),
          Question(id="Q02", text="IL6 in B cells?", cells=["B cell"], genes=["IL6"], result_ids=["R0002"])]
    objs = r._apply_answers(qs, [
        {"question_id": "Q01", "status": "answered", "answer": "IL6 in 38.4% of B cells", "answer_ids": [],
         "conclusion": "observed"},
        {"question_id": "Q02", "status": "answered", "answer": "IL6 in 1.0% of P1 B cells, 45.1% of P2",
         "answer_ids": ["R0002"], "conclusion": "observed"}], res, findings)
    assert qs[0].status == "unanswerable_with_data" and "38.4" not in qs[0].answer
    assert qs[1].conclusion == "inferred" and "45.1" in qs[1].answer.split("[verifier")[1]
    assert len(objs) == 2


def _kb(tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "lr_consensus.csv").write_text("ligand,receptor\nCXCL13,CXCR5\nLTA_LTB,LTBR\nTNFSF14,LTBR\n")
    (kb / "collectri.csv").write_text("source,target,weight\nRELB,CXCL13,1\nNFKB2,CXCL13,1\n")
    (kb / "progeny.csv").write_text("source,target,weight\nNFkB,CXCL13,2.5\n")
    return kb


EPMC = {"resultList": {"result": [{"pmid": "12345678", "pubYear": "2020", "title": "LTbR drives CXCL13 in stroma",
         "abstractText": "Background text here. Lymphotoxin signalling through LTbR induces CXCL13 in fibroblasts. "
                         "Other sentence about something else entirely."}]}}


def test_literature_scout_checks_citations_and_adds_hypotheses(projects, findings, tmp_path):
    import json as _json

    queries = []
    rep = _replay(findings)
    rep["scout_plan"] = _json.dumps({"searches": [
        {"hypothesis_id": "H1", "tool": "literature_search", "args": {"query": "CXCL13 fibroblast lymphotoxin"}},
        {"hypothesis_id": "H1", "tool": "literature_search", "args": {"query": "CXCL13 P1 fibroblast"}},  # refused
        {"hypothesis_id": "H1", "tool": "known_regulators", "args": {"gene": "CXCL13"}}]})
    rep["scout_read"] = _json.dumps({
        "assessments": [
            {"hypothesis_id": "H1", "step_id": "H1.S1", "support": "supported", "result_ids": ["R1"],
             "pmids": ["12345678"], "quote": "Lymphotoxin signalling through LTbR induces CXCL13 in fibroblasts."},
            {"hypothesis_id": "H1", "step_id": "H1.S1", "support": "supported", "result_ids": ["R1"],
             "pmids": ["99999999"], "quote": "CXCL13 cures everything."}],
        "new_hypotheses": [
            {"title": "B-cell lymphotoxin drives fibroblast CXCL13", "summary": "s", "steps": [
                {"source_cell": "B cell", "target_cell": "Fibroblast", "mediator": "LTA_LTB", "genes": ["LTB"]}],
             "inspired_by": ["R1", "R3"], "pmids": ["12345678", "11111111"]},
            {"title": "made up", "summary": "s", "steps": [], "inspired_by": ["R9"]}]})
    cfg = _cfg(projects)
    cfg.knowledge, cfg.literature, cfg.scout, cfg.kb_dir = True, True, True, str(_kb(tmp_path))
    run_dir = tmp_path / "runs" / "lit"

    def fetch(url):
        queries.append(url)
        return EPMC
    Runner(cfg, run_dir, llm_backend=Replay(rep), cells=make_cells(), lit_fetch=fetch).run(stop_after="scout")
    sc = _json.loads((run_dir / "scout.json").read_text())
    assert [r["ok"] for r in sc["results"]] == [True, False, True]
    assert "study identifiers" in sc["results"][1]["summary"] and len(queries) == 1  # sample code never sent
    assert "RELB" in sc["results"][2]["summary"]
    good, bad = sc["assessments"]
    assert good["verified"] and good["pmids"] == ["12345678"]
    assert not bad["verified"] and "quote not found" in bad["note"] and "PMID not in" in bad["note"]
    assert [h["id"] for h in sc["new_hypotheses"]] == ["L1"]  # the lead citing nothing retrieved is dropped
    assert sc["new_hypotheses"][0]["pmids"] == ["12345678"] and sc["new_hypotheses"][0]["inspired_by"] == ["R0001",
                                                                                                            "R0003"]
    log = (run_dir / "external_queries.jsonl").read_text()
    assert "CXCL13 fibroblast lymphotoxin" in log and "P1" not in log


def test_specialist_panel_merged_by_pi(projects, findings, tmp_path):
    import json as _json

    from mechanist.records import Persona
    cx = fid(findings, "panel_expression", "Fibroblast|CXCL13|PG|Control")
    hyp = {"title": "t", "summary": "s", "disease": "PG", "steps": [
        {"source_cell": "Fibroblast", "target_cell": "B cell", "mediator": "CXCL13", "genes": ["CXCL13"],
         "finding_ids": [cx]}]}
    rep = {"panel:Adaptive immunologist": _json.dumps({"hypotheses": [hyp | {"title": "B-cell idea"}]}),
           "panel:Innate immunologist": _json.dumps({"hypotheses": [hyp | {"title": "Neutrophil idea"}]}),
           "panel:Computational biologist": _json.dumps({"hypotheses": []}),
           "merge": _json.dumps({"hypotheses": [hyp | {"title": "Merged", "origin": ["Adaptive immunologist",
                                                                                      "Innate immunologist"]}]})}
    cfg = _cfg(projects)
    cfg.panel = [Persona(title="Adaptive immunologist", expertise="B cells"),
                 Persona(title="Innate immunologist", expertise="neutrophils")]
    run_dir = tmp_path / "runs" / "panel"
    Runner(cfg, run_dir, llm_backend=Replay(rep)).run(stop_after="generate")
    panel = _json.loads((run_dir / "panel.json").read_text())
    assert [h["title"] for hs in panel.values() for h in hs] == ["B-cell idea", "Neutrophil idea"]
    assert list(panel) == ["Adaptive immunologist", "Innate immunologist", "Computational biologist"]  # sceptic added
    gen = _json.loads((run_dir / "generate.json").read_text())
    assert gen[0]["id"] == "H1" and gen[0]["origin"] == ["Adaptive immunologist", "Innate immunologist"]
    temps = [_json.loads(x)["temperature"] for x in (run_dir / "llm_calls.jsonl").read_text().splitlines()]
    assert temps == [0.8, 0.8, 0.8, 0.2]


def test_panel_prompts_reserve_artifacts_for_the_sceptic(tmp_path):
    from mechanist.agents import panel_propose
    from mechanist.llm import LoggedLLM
    from mechanist.records import Persona
    seen = {}

    class Spy(Replay):
        def complete(self, tag, system, user, temperature=None):
            seen[tag] = user
            return '{"hypotheses": []}', {}
    llm = LoggedLLM(Spy({}), tmp_path)
    panel_propose(llm, Persona(title="Adaptive immunologist", expertise="B cells"), "ctx", "PG", 2, 0.8)
    panel_propose(llm, Persona(title="Computational biologist", expertise="artifacts", artifacts=True), "ctx", "PG",
                  2, 0.8)
    assert "another specialist covers those" in seen["panel:Adaptive immunologist"]
    assert "exactly your field" in seen["panel:Computational biologist"]
    assert "a generalist" in seen["panel:Adaptive immunologist"]


def _thread_replay(findings) -> dict:
    base = _replay(findings)
    call = lambda tool, thought="look", **args: json.dumps({"thought": thought, "tool": tool, "args": args})
    answer = lambda **a: json.dumps({"thought": "enough", "answer": a})
    base.update({
        "investigate:plan1": json.dumps({"threads": [
            {"kind": "presence", "text": "Why are B cells expanded?", "cells": ["B cell"], "hypothesis_ids": ["H1"],
             "why": "the B-cell excess is the largest change"},
            {"kind": "upstream", "text": "Why do fibroblasts make CXCL13?", "cells": ["Fibroblast"],
             "genes": ["CXCL13"], "hypothesis_ids": ["H1"]}]}),
        "investigate:Q01": [call("lineage_share", "is the excess in both PG samples?", lineage="B cell"),
                            answer(status="answered", answer="B cells are 54.5% of P1 cells (R0001).",
                                   answer_ids=["R0001"], conclusion="observed", follow_ups=["What recruits them?"])],
        "investigate:Q02": [call("gene_profile", gene="NOTAGENE"),
                            call("coexpression", "same cells?", lineage="Fibroblast", genes_a=["CXCL13"],
                                 genes_b=["CXCL1"]),
                            call("coexpression", lineage="Fibroblast", genes_a=["CXCL13"], genes_b=["CXCL1"]),
                            answer(status="answered", answer="CXCL13 and CXCL1 share cells (x2.0, 73.2%) (R0007).",
                                   answer_ids=["R0007"], conclusion="observed",
                                   follow_ups=["Does this hold in both PG samples?"])],
        "investigate:plan2": json.dumps({"threads": [
            {"kind": "relation", "text": "Is the CXCL13/CXCL1 program present in both PG samples?",
             "cells": ["Fibroblast"], "genes": ["CXCL13", "CXCL1"], "parent_id": "Q02"}]}),
        "investigate:Q03": ["not json", "still not json"],
        "chase:Q04": [call("state_markers", state="Activated HLA-II+ B cell"),
                      answer(status="answered", answer="HLA-DRA is the top marker (R0099).", answer_ids=["R0099"],
                             conclusion="observed")],
    })
    return base


def test_thread_investigation_loops_checks_and_resumes(projects, findings, tmp_path):
    cfg = _cfg(projects).model_copy(update={"investigation": "threads", "thread_rounds": 2, "threads_per_round": 2,
                                            "thread_steps": 3, "thread_workers": 2, "critic_questions": 1})
    run_dir = tmp_path / "runs" / "t"
    Runner(cfg, run_dir, llm_backend=Replay(_thread_replay(findings)), cells=make_cells()).run()
    inv = json.loads((run_dir / "chase.json").read_text())
    qs = {q["id"]: q for q in inv["questions"]}
    assert list(qs) == ["Q01", "Q02", "Q03", "Q04"] and qs["Q04"]["asked_by"] == "critic"
    # each thread owns a block of result ids (steps + 2), independent of which thread finished first
    assert qs["Q01"]["result_ids"] == ["R0001"] and qs["Q02"]["result_ids"] == ["R0006", "R0007"]
    assert qs["Q01"]["answer_ids"] == ["R0001"] and qs["Q01"]["follow_ups"] == ["What recruits them?"]
    steps = qs["Q02"]["steps"]
    assert steps[0]["result_id"] == "R0006" and steps[2].get("repeat") and steps[-1]["tool"] is None
    assert "73.2" in qs["Q02"]["answer"] and "not found in the cited data" in qs["Q02"]["answer"]
    assert qs["Q03"]["parent_id"] == "Q02" and qs["Q03"]["answer"] == "The thread ended without an answer."
    # the critic's thread cites a result it never got: the answer is rejected and kept visible as such
    assert qs["Q04"]["result_ids"] == ["R0008"] and qs["Q04"]["answer"].startswith("[rejected by the answer checks]")
    assert all(r["id"] == f"R{int(r['id'][1:]):04d}" for r in inv["results"])
    m = json.loads((run_dir / "report.json").read_text())["investigation"]
    assert m["threads"] == 4 and m["threads_without_answer"] == 1 and m["thread_repeat_calls"] == 1
    assert m["threads_following_up"] == 1
    report = (run_dir / "report.md").read_text()
    assert "step 2, `coexpression`: _same cells?_" in report and "_why this thread:_" in report
    calls = (run_dir / "llm_calls.jsonl").read_text().count("\n")
    Runner(cfg, run_dir, llm_backend=Replay({}), cells=make_cells()).run()  # resumes from the cache: no new calls
    assert (run_dir / "llm_calls.jsonl").read_text().count("\n") == calls


def test_review_turn_flags_unbacked_verdicts_and_the_thread_can_revise(projects, tmp_path):
    from mechanist.agents import run_thread
    from mechanist.llm import LoggedLLM
    from mechanist.records import Question, ToolResult
    first = json.dumps({"thought": "done", "answer": {"status": "answered", "answer": "OSM drives CXCL5 (R0001).",
                        "answer_ids": ["R0001"], "candidates": [{"ligand": "OSM", "sender": "Neutrophil",
                                                                  "receptor": "OSMR", "verdict": "supported"}]}})
    final = json.dumps({"thought": "the correlation did not discriminate", "answer": {
        "status": "partial", "answer": "OSM is not supported by a discriminating test (R0001).", "answer_ids": ["R0001"],
        "candidates": [{"ligand": "OSM", "verdict": "untested"}]}})
    llm = LoggedLLM(Replay({"t:Q01": [first, final]}), tmp_path)
    q = Question(id="Q01", kind="upstream", text="Why do fibroblasts make CXCL5?", cells=["Fibroblast"],
                 genes=["CXCL5"])
    own = [ToolResult(id="R0001", tool="covary", ok=True, summary="", args={"a": "OSM@Neutrophil",
                      "b": "CXCL5@Fibroblast"}, data={"rho_all": 0.7, "rho_within": {"Sweet": 0.8},
                                                      "null": {"n": 400, "frac": 0.7}})]
    runner = Runner(_cfg(projects), tmp_path / "r", llm_backend=Replay({}))
    tb = type("TB", (), {"tools": {}})()  # no literature tool
    note = runner._review(json.loads(first)["answer"], q, own, None, tb)
    assert note and "rests on no test" in note and "does not discriminate (70%" in note
    ans, steps = run_thread(llm, "t:Q01", "sys", "user", lambda t, a: None, 3, review=lambda a: note)
    assert ans["candidates"][0]["verdict"] == "untested" and steps[1].get("review")
    # an answer whose verdicts are backed (or that has none) gets no review
    assert runner._review({"candidates": [{"ligand": "OSM", "verdict": "rejected"}]}, q, own, None, tb) is None


def test_pi_rounds_review_chase_and_sign_off(projects, findings, tmp_path):
    base = _thread_replay(findings)
    base["critique_model:pi2"] = json.dumps({
        "verdict": "plausible", "sign_off": True, "depth": [{"link": "Fibroblast -> B cell", "level": 7, "missing": ""}],
        "objections": [{"target": "link 1", "kind": "missing_evidence", "text": "Round-2 objection."}], "questions": []})
    cfg = _cfg(projects).model_copy(update={"investigation": "threads", "thread_rounds": 2, "threads_per_round": 2,
                                            "thread_steps": 3, "thread_workers": 1, "critic_questions": 1,
                                            "pi_rounds": 3})
    run_dir = tmp_path / "runs" / "pi"
    Runner(cfg, run_dir, llm_backend=Replay(base), cells=make_cells()).run()
    chase = json.loads((run_dir / "chase.json").read_text())
    assert [r["round"] for r in chase["pi_reviews"]] == [1, 2] and chase["pi_reviews"][1]["sign_off"]
    assert chase["pi_objections"][0]["text"] == "Round-2 objection."  # the revision answers the latest review
    report = (run_dir / "report.md").read_text()
    assert "## PI review rounds" in report and "| 2 | plausible | yes |" in report and "Depth" in report


def test_staged_workflow_gates_every_stage(projects, findings, tmp_path):
    base = _thread_replay(findings)
    call = lambda tool, purpose="look", **args: json.dumps({"purpose": purpose, "tool": tool, "args": args})
    answer = lambda **a: json.dumps({"purpose": "enough", "answer": a})
    gate = lambda v, demands=(), t=0, p=0: json.dumps({"verdict": v, "assessment": "a", "new_tests": t,
                                                        "new_testable_predictions": p,
                                                        "demands": [{"kind": "presence", "text": d} for d in demands]})
    item = lambda text, ids: {"text": text, "ids": ids}
    dossier = {"defining_features": [item("Neutrophil-rich ulcer", ["R5001"]), item("Unbacked claim", ["R0042"])],
               "cells_states": [item("Neutrophils", ["R5001"]), item("Cited by PMID", ["R12345678"]),
                                item("Seen in the data", ["F1"])],
               "novelty_leads": [{"text": "B cells", "finding_ids": []}]}
    thread_ans = answer(status="answered", answer="B cells are 54.5% of P1 cells (R0001).", answer_ids=["R0001"],
                        conclusion="observed")
    base.update({
        "dossier:r1": [call("literature_search", "pathology", query="neutrophilic ulcer pathology"),
                       call("lineage_share", lineage="B cell"),  # not a knowledge tool: refused in the dossier
                       answer(**dossier)],
        "dossier:gate1": gate("revise", ["Search the comparator disease."]),
        "dossier:r2": answer(**dossier),
        "dossier:gate2": gate("pass"),
        "check:plan": json.dumps({"threads": [{"kind": "presence", "text": "Are B cells expanded?",
                                               "cells": ["B cell"]}]}),
        "check:Q01": [call("lineage_share", lineage="B cell"), thread_ans],
        "check:summary1": json.dumps({"expected_present": [], "expected_absent": [],
                                      "unexpected": [{"text": "B cells", "checks": "both samples",
                                                      "question_ids": ["Q01"]}]}),
        "check:gate1": gate("pass", t=1),
        "candidates:gate1": gate("revise", ["Add a candidate from the unexpected list."], p=1),
        "candidates:r2": base["generate"],
        "candidates:gate2": gate("pass"),
        "investigate:plan1": json.dumps({"threads": [{"kind": "upstream", "text": "Why are B cells there?",
                                                      "cells": ["B cell"]}]}),
        "investigate:Q02": [call("lineage_share", lineage="B cell"), thread_ans],
        "critique_model": json.dumps({"verdict": "weak", "new_tests": 1, "questions": [
            {"kind": "artifact", "text": "Is one sample dominating?", "cells": ["B cell"]}]}),
        "chase:Q03": [call("state_composition", lineage="B cell"), thread_ans],
        "critique_model:pi2": json.dumps({"verdict": "plausible", "sign_off": True, "questions": []}),
        "experiment:r1": [call("literature_search", query="B cell depletion ulcer"),
                          answer(experiments=[{"title": "CXCR5 block", "tests": ["M1", "M9"], "tier": "ex_vivo",
                                               "precedent": [{"what": "depletion study", "ids": ["R9999"]}],
                                               "model": "organoid", "perturbation": "CXCR5 block",
                                               "readout": "neutrophils", "priority": 1,
                                               "predictions": [{"circuit": 1, "outcome": "down", "why": "niche lost"},
                                                               {"circuit": 2, "outcome": "no change", "why": "no niche"},
                                                               {"circuit": 7, "outcome": "up"}]}],
                                 not_testable=[{"id": "M2", "why": "needs fresh ulcer tissue"}])],
        "experiment:gate1": gate("pass"),
        "therapy:r1": answer(hypotheses=[{"drug": "TESTMAB", "target": "CXCL13", "circuits": [1, 2],
                                          "breaks": "Fibroblast → B cell", "improves": ["D99"], "fails_under": [],
                                          "depends_on": ["M1"], "rationale": "blocks the chemokine"},
                                         {"drug": "INVENTED", "target": "CXCL13", "circuits": [1]}],
                             pathways=[{"name": "chemokine niche", "genes": ["CXCL13"], "note": "one antibody"}]),
        "therapy:gate1": gate("pass"),
        "linkage1:L1": [call("literature_search", query="CXCL13 fibroblast tertiary lymphoid structure rheumatoid"),
                        answer(findings=[{"aspect": "other tissue", "text": "CXCL13+ fibroblasts in synovium",
                                          "ids": ["R5006"]},  # the search this thread ran
                                         {"aspect": "therapy", "text": "made up", "ids": ["R9999"]}],
                               suggests="Seen in other tissues.", rung=4)],
        "linkage1": answer(findings=[], suggests="nothing found", rung=5),
        "linkage:gate1": gate("pass"),
        "final:gate1": gate("revise", ["Test the link in both samples."]),
        "final:gate2": gate("revise", ["Test the link in both samples."]),
        "final:gate3": gate("revise", ["Test the link in both samples."]),
        "final2": thread_ans, "final3": thread_ans,
        "circuits:r1": [call("literature_search", query="stromal lymphoid niche ulcer"),
                        answer(circuits=[
                            {"rank": 1, "title": "Stromal niche drives the ulcer", "summary": "s", "root": "B cells only here",
                             "edges": [{"source": "Fibroblast", "target": "B cell", "via": "CXCL13", "status": "observed",
                                        "ids": ["R9999"]},  # not a real id: relabelled likely
                                       {"source": "B cell", "target": "ulcer", "via": "unknown", "status": "analogy",
                                        "ids": ["R5006"]}],  # a search from another stage: still a real search
                             "predictions": [{"test": "block CXCR5", "expects": "fewer B cells", "in_this_data": False}]},
                            {"rank": 2, "title": "One link only", "edges": [
                                {"source": "Fibroblast", "target": "B cell", "via": "CXCL13", "status": "likely"}]}],
                               ranking="a spatial map would")],
        "circuits:gate1": gate("pass"),
    })
    cfg = _cfg(projects).model_copy(update={"investigation": "threads", "thread_rounds": 1, "threads_per_round": 1,
                                            "thread_steps": 3, "thread_workers": 1, "critic_questions": 1,
                                            "workflow": "staged", "circuit_attempts": 1, "check_threads": 1, "dossier_steps": 3,
                                            "experiment_steps": 2})
    cfg.knowledge, cfg.literature, cfg.kb_dir = True, True, str(_kb(tmp_path))
    run_dir = tmp_path / "runs" / "staged"
    def ot(query, v):  # Open Targets, offline: every target exists; CXCL13 has one approved inhibitor
        if "search(" in query:
            return {"data": {"search": {"hits": [{"id": f"ENSG_{v['s']}", "name": v["s"]}]}}}
        rows = ([{"maxClinicalStage": "APPROVAL", "drug": {"id": "D1", "name": "TESTMAB", "drugType": "Antibody",
                  "mechanismsOfAction": {"rows": [{"actionType": "INHIBITOR", "mechanismOfAction": "CXCL13 inhibitor",
                                                   "targets": [{"approvedSymbol": "CXCL13"}]}]}},
                  "diseases": [{"diseaseFromSource": "lupus", "disease": {"name": "lupus"}}]}]
                if v.get("id") == "ENSG_CXCL13" and "mechanismsOfAction" in query else [])
        return {"data": {"target": {"approvedSymbol": v.get("id", "")[5:], "associatedDiseases": {"count": 0, "rows": []},
                                    "drugAndClinicalCandidates": {"count": len(rows), "rows": rows}}}}
    Runner(cfg, run_dir, llm_backend=Replay(base), cells=make_cells(), lit_fetch=lambda url: EPMC, ot_fetch=ot).run()
    d = json.loads((run_dir / "dossier.json").read_text())
    assert [r["verdict"] for r in d["reviews"]] == ["revise", "pass"] and d["reviews"][-1]["closed_by"] == "pass"
    assert [x["text"] for x in d["dossier"]["defining_features"]] == ["Neutrophil-rich ulcer"]  # unbacked: dropped
    assert d["results"][0]["id"] == "R5001" and not d["results"][1]["ok"]
    assert d["dossier"]["cells_states"][1] == {"text": "Cited by PMID", "ids": ["R5001"]}  # PMID mapped to its search
    assert d["dossier"]["data_observations"] == [{"text": "Seen in the data", "ids": ["F0001"]}]
    c = json.loads((run_dir / "check.json").read_text())
    assert c["questions"][0]["asked_by"] == "analyst" and c["summary"]["unexpected"][0]["text"] == "B cells"
    cand = json.loads((run_dir / "candidates.json").read_text())
    assert len(cand["reviews"]) == 2 and cand["hypotheses"][0]["id"] == "H1"
    inv_ = json.loads((run_dir / "investigate.json").read_text())
    assert [q["id"] for q in inv_["questions"]] == ["Q01", "Q02"]  # the chase builds on the pathology check
    ch = json.loads((run_dir / "chase.json").read_text())
    assert [r["round"] for r in ch["pi_reviews"]] == [1, 2] and ch["model"] is not None
    lk = json.loads((run_dir / "linkage.json").read_text())
    assert lk["reviews"][-1]["closed_by"] == "pass" and lk["links"]
    first = lk["links"][0]
    assert [f["text"] for f in first["findings"]] == ["CXCL13+ fibroblasts in synovium"]  # unbacked finding dropped
    ex = json.loads((run_dir / "experiment.json").read_text())
    assert ex["experiments"][0]["precedent"] == []  # cites a search never run
    assert ex["experiments"][0]["tests"] == ["M1"] and [p["circuit"] for p in ex["experiments"][0]["predictions"]] == [1, 2]
    assert ex["experiments"][0]["discrimination"] > 0 and ex["uncovered"] == [] and ex["not_testable"][0]["id"] == "M2"
    assert any("unknown links" in o["text"] for o in ex["objections"])
    th = json.loads((run_dir / "therapy.json").read_text())
    assert [h["drug"] for h in th["hypotheses"]] == ["TESTMAB"]  # the invented drug is dropped
    h = th["hypotheses"][0]
    assert h["level"] == "approved for another disease" and h["blocks"] and h["stage"] == "APPROVAL"
    assert h["depends_on"] == ["M1"] and h["improves"] == []  # D99 is not a dossier item
    assert any("not in any drug lookup" in o["text"] for o in th["objections"])
    assert th["pathways"][0]["drugs"] == ["TESTMAB"] and th["targets"][0]["gene"] == "CXCL13"
    ci = json.loads((run_dir / "circuits.json").read_text())
    assert [c["rank"] for c in ci["circuits"]] == [1, 2] and ci["reviews"][-1]["closed_by"] == "pass"
    assert ci["circuits"][0]["edges"][0]["status"] == "likely"  # observed without a valid id
    assert any("makes no prediction" in o["text"] for o in ci["objections"])  # circuit 2
    assert "DIFFERENCE ACCOUNT" in ci["differences"]
    rev = json.loads((run_dir / "revise.json").read_text())
    fr = rev["final_reviews"]
    assert [r["no_progress"] for r in fr] == [1, 2, 3] and fr[-1]["closed_by"] == "no progress"
    assert rev["synthesis"]["experiments_planned"] == []  # experiments now come after the final model
    assert ex["experiments"][0]["title"] == "CXCR5 block"
    events = [json.loads(x) for x in (run_dir / "events.jsonl").read_text().splitlines()]
    ends = [e["stage"] for e in events if e.get("action") == "end"]
    assert ends.index("revise") < ends.index("experiment")  # designed on the final ranked hypotheses
    assert len(rev["final_questions"]) == 2
    roles = {json.loads(x)["role"] for x in (run_dir / "transcript.jsonl").read_text().splitlines()}
    assert {"PI", "planner", "synthesizer", "tool", "disease scholar", "experimentalist"} <= roles
    report = (run_dir / "report.md").read_text()
    assert "## The PI's gates" in report and "| 6 Integrated model | 3 | no progress |" in report
    assert "## Stage 1: pathology dossier" in report and "### Experiments that would decide between the circuits" in report
    assert "## Therapeutic hypotheses (research only, not recommendations)" in report and "TESTMAB" in report
    assert "## Final model: proposed circuits" in report and "```mermaid" in report
    assert report.index("## Final model: proposed circuits") < report.index("## Evidence ledger")
    assert ends.index("circuits") < ends.index("experiment")
    m = json.loads((run_dir / "report.json").read_text())
    assert m["gates"]["1 Pathology dossier"]["reviews"] == 2 and m["dossier"]["dropped_unbacked"] == 2
    assert m["gates"]["4 Chase"]["closed_by"] == "sign-off"


def test_long_logs_are_trimmed_to_the_budget(projects, tmp_path):
    from mechanist.records import Question
    runner = Runner(_cfg(projects).model_copy(update={"log_budget": 3000}), tmp_path / "r", llm_backend=Replay({}))
    qs = [Question(id=f"Q{i:02d}", kind="presence", text=f"question {i}", status="answered", answer="x" * 400)
          for i in range(1, 41)]
    log = runner._log_with_patterns(qs)
    assert len(log) <= 3000 and "earlier questions not shown" in log and "Q40" in log and "Q01 " not in log
    assert len(runner._short_log(qs)) <= 1500


def test_model_score_counts_tested_steps_and_depth():
    from mechanist.records import Synthesis
    syn = Synthesis.model_validate({"title": "t", "summary": "s", "links": [
        {"source": "A", "target": "B", "mediator": "m", "depth": 4,
         "steps": [{"claim": "x", "kind": "tested", "ids": ["R0001"]}, {"claim": "y", "kind": "speculative"}]},
        {"source": "B", "target": "C", "mediator": "n", "depth": 2,
         "steps": [{"claim": "z", "kind": "tested", "ids": ["R0002"]}]}]})
    assert Runner._model_score(syn) == (2, 6)  # tested steps, total depth: a round counts as progress only if it beats these



def test_best_guesses_and_involved_specific_reach_the_report():
    from mechanist import report
    from mechanist.records import Synthesis
    syn = Synthesis.model_validate({"title": "t", "summary": "s", "links": [
        {"source": "Myeloid", "target": "Fibroblast", "mediator": "TNF", "involved": "yes, 14% vs 1-3% of cells",
         "specific": "no, Sweet's 5-15%"}],
        "best_guesses": [{"step": "What induces the fibroblast programme?", "decisive_test": "explant blockade", "candidates": [
            {"rank": 2, "explanation": "OSM", "evidence_rung": 3}, {"rank": 1, "explanation": "IL-1a", "evidence_rung": 2,
                                                                   "confidence": "low-moderate"}]}]})
    md = "\n".join(report._staged_sections({}, syn, {"6 Integrated model": {"reviews": 1, "closed_by": "pass",
                                                                         "new_tests": 0, "new_testable_predictions": 0}}))
    assert "### Involved vs specific" in md and "yes, 14% vs 1-3% of cells" in md
    assert md.index("1. IL-1a (rung 2; low-moderate)") < md.index("2. OSM (rung 3")  # ranked best guesses
    assert "would decide: explant blockade" in md


def test_transcript_records_who_said_what():
    from mechanist.records import ToolResult
    from mechanist.transcript import Transcript, role_of, summarize
    assert role_of("chase3:Q41:s2") == "analyst Q41" and role_of("check:gate2") == "PI"
    assert role_of("critique_model:pi4") == "PI" and role_of("linkage1:L3:s0") == "linkage scholar L3"
    assert role_of("investigate:plan1") == "planner" and role_of("experiment:r1:s0") == "experimentalist"
    assert summarize('{"purpose": "who makes it", "tool": "gene_profile", "args": {"gene": "CXCL13"}}').startswith(
        "who makes it → gene_profile")
    assert summarize('{"verdict": "revise", "assessment": "thin", "demands": [{"text": "run X"}]}').startswith(
        "VERDICT revise: thin | 1 demands: run X")
    import tempfile
    from pathlib import Path
    p = Path(tempfile.mkdtemp()) / "t.jsonl"
    t = Transcript(p)
    t.turn("check:Q01:s0", '{"purpose": "p", "tool": "lineage_share", "args": {}}')
    t.tool(ToolResult(id="R0001", tool="lineage_share", args={}, ok=True, summary="B cell 54%"))
    rows = [json.loads(x) for x in p.read_text().splitlines()]
    assert [r["role"] for r in rows] == ["analyst Q01", "tool"] and "B cell 54%" in rows[1]["text"]


def test_dossier_and_check_text_tolerate_loose_model_output():
    from mechanist import agents
    t = agents.check_text({"expected_present": [{"text": "neutrophils", "question_ids": [1, "Q02"]}],
                           "unexpected": ["a bare string"]})
    assert "[1, Q02]" in t and "a bare string" in t
    assert "F0001" in agents.dossier_text({"data_observations": [{"text": "x", "ids": "F0001"}, "plain"]})
