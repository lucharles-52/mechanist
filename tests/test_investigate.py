from mechanist import investigate as inv
from mechanist.records import Hypothesis, Link, Question, Step, Synthesis


def _hyps():
    h1 = Hypothesis(id="H1", title="t", disease="PG", steps=[
        Step(source_cell="Fibroblast", target_cell="B cell", mediator="CXCL13", genes=["CXCL13"]),
        Step(source_cell="B cell", target_cell="Plasma cell", mediator="differentiation", genes=["PRDM1"])])
    h2 = Hypothesis(id="H2", title="t", disease="PG", steps=[
        Step(source_cell="Keratinocyte", target_cell="Neutrophil", mediator="CXCL6", genes=["CXCL6"])])
    h3 = Hypothesis(id="H3", title="artifact", disease="PG", non_mechanistic=True)
    return [h1, h2, h3]


def test_requirements_cover_the_users_questions():
    reqs = inv.requirements(_hyps())
    kinds = {(r.kind, r.cell, r.gene, r.hypothesis_id) for r in reqs}
    assert ("presence", "B cell", None, None) in kinds
    assert ("function", "B cell", None, None) in kinds
    assert ("upstream", "Fibroblast", "CXCL13", None) in kinds
    assert ("upstream", "Keratinocyte", "CXCL6", None) in kinds
    assert ("relation", None, None, "H1") in kinds and ("relation", None, None, "*") in kinds
    assert ("disease_link", None, None, "H2") in kinds
    assert not any(r.hypothesis_id == "H3" for r in reqs)  # non-mechanistic alternatives are not interrogated


def test_satisfaction_and_coverage():
    reqs = inv.requirements(_hyps())
    qs = [Question(id="Q01", kind="function", text="What are B cells doing?", cells=["B cells"], status="answered"),
          Question(id="Q02", kind="upstream", text="Why do fibroblasts make CXCL13?", cells=["Fibroblast"],
                   genes=["CXCL13"]),
          Question(id="Q03", kind="relation", text="Are H1 and H2 one program?", hypothesis_ids=["H1", "H2"])]
    cov = {(c["requirement"]["kind"], c["requirement"]["cell"], c["requirement"]["hypothesis_id"]): c
           for c in inv.coverage(reqs, qs)}
    assert cov[("function", "B cell", None)]["resolved_by"] == ["Q01"]
    assert cov[("upstream", "Fibroblast", None)]["asked_by"] == ["Q02"]
    assert cov[("upstream", "Fibroblast", None)]["resolved_by"] == []
    assert cov[("relation", None, "*")]["asked_by"] == ["Q03"]
    assert cov[("relation", None, "H1")]["asked_by"] == ["Q03"]
    assert any(r.kind == "presence" for r in inv.missing(reqs, qs))


def test_normalize_and_check_answers():
    assert inv.normalize_id("F4") == "F0004" and inv.normalize_id("r012") == "r012" and inv.normalize_id("R12") == "R0012"
    q = Question(id="Q01", text="x", status="answered", conclusion="observed", answer_ids=["F4", "R9"])
    objs = inv.check_answers([q], {"F0004"})
    assert q.answer_ids == ["F0004"] and q.conclusion == "observed"
    assert [o.kind for o in objs] == ["invalid_citation"]
    q2 = Question(id="Q02", text="y", status="answered", conclusion="observed", answer_ids=["R9"])
    objs = inv.check_answers([q2], {"F0004"})
    assert q2.conclusion == "inferred" and {o.kind for o in objs} == {"invalid_citation", "unsupported_step"}


def test_check_synthesis_derives_link_status_and_flags_unexamined():
    syn = Synthesis(title="m", summary="s", links=[
        Link(source="Fibroblast", target="B cell", mediator="CXCL13", mediator_in_source="observed",
             receptor_in_target="observed", ids=["R1"], question_ids=["Q01"], evidence="observed"),
        Link(source="B cell", target="ulcer", mediator_in_source="observed", ids=["R99"], question_ids=["Q02"]),
        Link(source="Neutrophil", target="Fibroblast", mediator_in_source="observed", receptor_in_target="absent",
             ids=["R1"], question_ids=["Q09"])])
    objs = inv.check_synthesis(syn, {"R0001"}, {"Q01", "Q02"})
    assert syn.links[0].evidence == "expression_observed" and syn.links[0].ids == ["R0001"]
    assert syn.links[1].mediator_in_source == "not_checked" and syn.links[1].evidence == "not_observed"
    assert syn.links[2].evidence == "contradicted" and syn.links[2].question_ids == []
    assert sorted(o.kind for o in objs) == ["invalid_citation", "missing_evidence", "unsupported_step"]


def test_link_ignores_model_supplied_status():
    lk = Link(source="a", target="b", evidence="observed", mediator_in_source="yes")
    assert lk.evidence == "not_observed" and lk.mediator_in_source == "not_checked"


def test_hypothesis_level_requirement_met_by_question_on_its_cells():
    reqs = [r for r in inv.requirements(_hyps()) if r.kind == "artifact" and r.hypothesis_id == "H1"]
    q = Question(id="Q01", kind="artifact", text="Could one sample drive the B-cell expansion?", cells=["B cell"])
    assert inv.satisfies(q, reqs[0])


def test_depth_follows_parent_chain():
    qs = [Question(id="Q01", text="a"), Question(id="Q02", text="b", parent_id="Q01"),
          Question(id="Q03", text="c", parent_id="Q02"), Question(id="Q04", text="d", parent_id="Q99")]
    assert inv.depth(qs) == 3


def test_open_questions_carry_their_own_results_across_rounds():
    from mechanist.agents import log_text, open_text
    from mechanist.records import ToolResult
    r = ToolResult(id="R0007", tool="state_markers", args={}, ok=True, summary="HLA-DRA top")
    qs = [Question(id="Q01", text="done", status="answered", answer="yes", answer_ids=["R0001"]),
          Question(id="Q02", text="still open", result_ids=["R0007"])]
    block = open_text(qs, {"R0007": r})
    assert "### Q02" in block and "HLA-DRA top" in block and "Q01" not in block
    assert "Q01" in log_text(qs) and "Q02" not in log_text(qs)


def test_normalize_kind_maps_tool_names_and_synonyms():
    assert inv.normalize_kind("state_markers") == "function"
    assert inv.normalize_kind("Covariation") == "relation"
    assert inv.normalize_kind("expansion") == "presence"
    assert inv.normalize_kind("upstream") == "upstream"
    assert inv.normalize_kind("banana") == "other"


def test_synthesis_parser_accepts_dict_lists(tmp_path):
    import json

    from mechanist.agents import synthesize
    from mechanist.llm import LoggedLLM, Replay
    reply = {"title": "t", "summary": "s", "links": [], "experiments": {"IHC": "stain CXCL13"},
             "open_questions": "one", "disease_link": ["a"]}
    syn = synthesize(LoggedLLM(Replay({"revise": json.dumps(reply)}), tmp_path), "revise", "", "", [], "", "")
    assert syn.experiments == ["IHC: stain CXCL13"] and syn.open_questions == ["one"]


def test_answer_must_belong_to_its_question():
    q = Question(id="Q02", text="What fraction of fibroblasts express CXCL13?", cells=["Fibroblast"],
                 genes=["CXCL13"], result_ids=["R0004"])
    ok = {"R0004", "R0009"}
    assert inv.answer_matches(q, ["R4"], "42.5% of fibroblasts", ok)
    assert not inv.answer_matches(q, ["R0009"], "B cells are activated HLA-II+", ok)  # filed under the wrong id
    q2 = Question(id="Q09", text="How do B cells cause ulcers?", cells=["B cell"])
    assert inv.answer_matches(q2, [], "Plasma cells could make antibody", set())
    assert not inv.answer_matches(q2, [], "Neutrophils release proteases", set())


def test_question_ids_in_link_evidence_move_to_question_ids():
    syn = Synthesis(title="m", summary="s", links=[
        Link(source="Fibroblast", target="B cell", mediator_in_source="observed", receptor_in_target="observed",
             ids=["R1", "Q5", "Q12"])])
    objs = inv.check_synthesis(syn, {"R0001"}, {"Q05", "Q12"})
    assert syn.links[0].question_ids == ["Q05", "Q12"] and syn.links[0].ids == ["R0001"] and not objs


def test_untraceable_numbers_catch_made_up_values():
    src = ["B cell: PG: PG4L 1.0%, SweetPat14 4.0% | Control: SSC1 19.0%", "Spearman across 9 samples: -0.20"]
    assert inv.untraceable_numbers("IL6 is in 38.4% of B cells and 1.0% in PG4L", src) == ["38.4"]
    assert inv.untraceable_numbers("about 4% of B cells; rho ‑ 0.20 within Sweet (R0012, F0003)", src) == []
    assert inv.untraceable_numbers("in 2 of 4 samples", src) == []  # plain counts are not checked


def test_upstream_counts_only_after_inducer_profiled_across_lineages():
    from mechanist.records import ToolResult
    reqs = [r for r in inv.requirements(_hyps()) if r.kind == "upstream" and r.gene == "CXCL13"]
    ask = Question(id="Q01", kind="upstream", text="Why do fibroblasts make CXCL13?", cells=["Fibroblast"],
                   genes=["CXCL13"], status="answered", result_ids=["R0001"])
    res = [ToolResult(id="R0001", tool="incoming_signals", args={"lineage": "Fibroblast"}, ok=True, summary="")]
    assert inv.coverage(reqs, [ask], res)[0]["resolved_by"] == []
    assert [r for r, _ in inv.untested_upstream(reqs, [ask], res)] == reqs
    follow = Question(id="Q02", kind="upstream", text="Who makes TNF?", parent_id="Q01", status="answered",
                      result_ids=["R0002"])
    res.append(ToolResult(id="R0002", tool="gene_profile", args={"gene": ["TNF", "LTB"]}, ok=True, summary=""))
    assert inv.coverage(reqs, [ask, follow], res)[0]["resolved_by"] == ["Q01"]
    res[1] = ToolResult(id="R0002", tool="gene_profile", args={"gene": "TNF", "lineage": "Fibroblast"}, ok=True,
                        summary="")
    assert inv.coverage(reqs, [ask, follow], res)[0]["resolved_by"] == []  # checked only in the target cell



def test_upstream_chain_gene_tf_receptor_on_cell_ligand():
    from mechanist.records import ToolResult
    reqs = [r for r in inv.requirements(_hyps()) if r.kind == "upstream" and r.gene == "CXCL13"]
    assert reqs[0].cell == "Fibroblast"
    ask = Question(id="Q01", kind="upstream", text="Why do fibroblasts make CXCL13?", cells=["Fibroblast"],
                   genes=["CXCL13"], status="answered")
    res = [ToolResult(id="R0001", tool="known_regulators", args={"gene": "CXCL13"}, ok=True, summary="",
                      data={"activating_tfs": ["RELB", "NFKB2"]})]
    st = inv.chain_status("CXCL13", res, "Fibroblast")
    assert st["steps"]["regulators"] and not st["steps"]["routes"] and "receptor_paths" in st["next"]
    res.append(ToolResult(id="R0002", tool="receptor_paths", args={"tf": "NFKB2"}, ok=True, summary="", data={
        "routes": [{"receptor": "LTBR", "ligands": ["LTA_LTB", "TNFSF14"]},
                   {"receptor": "TNFRSF1A", "ligands": ["TNF", "LTA"]}]}))
    st = inv.chain_status("CXCL13", res, "Fibroblast")
    assert not st["steps"]["receptor_on_cell"] and "LTBR" in st["next"] and "lineage='Fibroblast'" in st["next"]
    # v9 failure case: TNF profiled everywhere, but no receptor was shown on fibroblasts -> does not count
    res.append(ToolResult(id="R0003", tool="gene_profile", args={"gene": "TNF"}, ok=True, summary="",
                          data={"B cell": {"P1": 18.0}}))
    assert not inv.chain_status("CXCL13", res, "Fibroblast")["complete"]
    # LTBR shown on fibroblasts (>= 5% in some sample); TNFRSF1A only in B cells
    res.append(ToolResult(id="R0004", tool="gene_profile", args={"gene": ["LTBR", "TNFRSF1A"], "lineage": None},
                          ok=True, summary="", data={"LTBR": {"Fibroblast": {"P1": 57.8, "C1": None}},
                                                      "TNFRSF1A": {"B cell": {"P1": 30.0}}}))
    st = inv.chain_status("CXCL13", res, "Fibroblast")
    assert st["receptors_on_cell"] == ["LTBR"] and not st["complete"]  # TNF is not an LTBR ligand
    assert "LTBR: LTA, LTB, TNFSF14" in st["next"]
    assert inv.coverage(reqs, [ask], res, strict_chain=True)[0]["resolved_by"] == []
    res.append(ToolResult(id="R0005", tool="gene_profile", args={"gene": ["LTB", "TNFSF14"]}, ok=True, summary="",
                          data={"LTB": {"B cell": {"P1": 67.6}}, "TNFSF14": {"Neutrophil": {"P1": 8.8}}}))
    st = inv.chain_status("CXCL13", res, "Fibroblast")
    assert st["complete"] and st["ligands_profiled"] == ["LTB", "TNFSF14"]
    # profiling is not thinking: the chain resolves only once a candidate is judged and tested beyond the profile
    cov = inv.coverage(reqs, [ask], res, strict_chain=True)[0]
    assert cov["resolved_by"] == [] and not cov["chain"]["steps"]["candidate_judged"]
    [(_, hint)] = inv.untested_upstream(reqs, [ask], res, strict_chain=True)
    assert "JUDGE" in hint and "verdict" in hint
    from mechanist.records import Candidate
    # a verdict without a test beyond the screen does not count
    ask.candidates = [Candidate(ligand="LTB", sender="B cell", receptor="LTBR", verdict="supported", reason="up")]
    assert inv.coverage(reqs, [ask], res, strict_chain=True)[0]["resolved_by"] == []
    test = Question(id="Q02", kind="relation", text="Are CXCL13+ fibroblasts LTBR+?", cells=["Fibroblast"],
                    genes=["LTBR", "CXCL13"], parent_id="Q01", status="answered", result_ids=["R0006"])
    res.append(ToolResult(id="R0006", tool="coexpression", ok=True, summary="", data={},
                          args={"lineage": "Fibroblast", "genes_a": ["LTBR"], "genes_b": ["CXCL13"]}))
    cov = inv.coverage(reqs, [ask, test], res, strict_chain=True)[0]
    assert cov["resolved_by"] == ["Q01"] and cov["chain"]["complete"]
    assert cov["chain"]["judged"][0]["tests"] == ["R0006"]
    assert inv.untested_upstream(reqs, [ask, test], res, strict_chain=True) == []



def test_chain_requires_disease_difference_when_groups_known():
    from mechanist.records import ToolResult
    groups = {"PG": ["P1", "P2"], "Sweet": ["S1", "S2"], "Control": ["C1", "C2"]}
    ctx = {"groups": groups, "case": "PG", "reference": "Control"}
    res = [ToolResult(id="R1", tool="known_regulators", args={"gene": "CXCL13"}, ok=True, summary="",
                      data={"activating_tfs": ["NFKB2"]}),
           ToolResult(id="R2", tool="receptor_paths", args={"tf": "NFKB2"}, ok=True, summary="", data={
               "routes": [{"receptor": "LTBR", "ligands": ["LTA_LTB", "TNFSF14"]},
                          {"receptor": "ITGB1", "ligands": ["VWF"]}]}),
           # ITGB1 everywhere but lower in PG than controls; LTBR on every PG fibroblast sample, above control mean
           ToolResult(id="R3", tool="gene_profile", args={"gene": ["LTBR", "ITGB1"], "lineage": "Fibroblast"}, ok=True,
                      summary="", data={"LTBR": {"Fibroblast": {"P1": 58, "P2": 46, "C1": 20, "C2": 49}},
                                        "ITGB1": {"Fibroblast": {"P1": 80, "P2": 70, "C1": 90, "C2": 95}}})]
    st = inv.chain_status("CXCL13", res, "Fibroblast", **ctx)
    assert st["receptors_on_cell"] == ["LTBR"]
    # VWF is not an LTBR ligand; TNFSF14 overlaps Sweet in every lineage -> no difference yet
    res.append(ToolResult(id="R4", tool="gene_profile", args={"gene": ["TNFSF14"]}, ok=True, summary="",
                          data={"TNFSF14": {"Neutrophil": {"P1": 9, "P2": 14, "S1": 10, "S2": 2, "C1": 0, "C2": 20}}}))
    st = inv.chain_status("CXCL13", res, "Fibroblast", **ctx)
    assert not st["complete"] and st["ligands_without_difference"] == ["TNFSF14"]
    assert "already profiled without a disease difference: TNFSF14" in st["next"]
    res.append(ToolResult(id="R5", tool="gene_profile", args={"gene": "LTB"}, ok=True, summary="",
                          data={"B cell": {"P1": 67.6, "P2": 68.8, "S1": 30.5, "S2": 28.8, "C1": 0.6, "C2": 17.9}}))
    st = inv.chain_status("CXCL13", res, "Fibroblast", **ctx)
    assert st["complete"] and st["ligands_profiled"] == ["LTB"]


def test_requirements_rank_key_mediators_first():
    reqs = inv.requirements(_hyps())
    assert reqs[0].priority == 1
    key = {(r.kind, r.gene) for r in reqs if r.priority == 1}
    assert ("upstream", "CXCL13") in key and ("upstream", "CXCL6") in key
    assert ("upstream", "PRDM1") not in key  # second step of H1
    assert all(r.priority == 2 for r in reqs if r.kind == "disease_link")


def test_screen_counts_for_steps_but_not_for_judgement():
    from mechanist.records import Candidate, ToolResult
    groups = {"PG": ["P1", "P2"], "Sweet": ["S1", "S2"], "Control": ["C1", "C2"]}
    ctx = {"groups": groups, "case": "PG", "reference": "Control"}
    screen = ToolResult(id="R1", tool="upstream_screen", args={"gene": "CXCL13", "cell": "Fibroblast"}, ok=True,
                        summary="", data={"tfs": ["NFKB2"], "receptors_on_cell": ["LTBR"],
                                          "routes": [{"receptor": "LTBR", "ligands": ["LTB", "TNFSF14"]}],
                                          "candidates": [{"ligand": "LTB", "sender": "B cell", "kind": "paracrine",
                                                          "separates_from": ["Sweet", "Control"]}],
                                          "ligands_without_difference": ["TNFSF14"]})
    q = Question(id="Q01", kind="upstream", text="Why do fibroblasts make CXCL13?", cells=["Fibroblast"],
                 genes=["CXCL13"], status="answered", result_ids=["R1"])
    st = inv.chain_status("CXCL13", [screen], "Fibroblast", **ctx, qs=[q])
    assert all(v for k, v in st["steps"].items() if k != "candidate_judged") and not st["complete"]
    assert "LTB@B cell" in st["next"]
    # a verdict on a ligand the data never showed to differ does not count
    q.candidates = [Candidate(ligand="TNFSF14", verdict="rejected")]
    sweep = ToolResult(id="R2", tool="gene_by_state", ok=True, summary="", data={},
                       args={"gene": "TNFSF14", "lineage": "Macrophage"})
    q.result_ids.append("R2")
    assert not inv.chain_status("CXCL13", [screen, sweep], "Fibroblast", **ctx, qs=[q])["complete"]
    q.candidates.append(Candidate(ligand="LTB", sender="B cell", receptor="LTBR", verdict="supported"))
    q.result_ids += ["R3", "R4", "R5", "R6"]
    # descriptive (which B-cell state makes LTB), receptor checked in the SENDING cell, and a correlation that any
    # disease-up gene shows: none of these is a test
    describe = ToolResult(id="R3", tool="gene_by_state", ok=True, summary="", data={},
                          args={"gene": "LTB", "lineage": "B cell"})
    wrong_cell = ToolResult(id="R4", tool="coexpression", ok=True, summary="", data={},
                            args={"lineage": "B cell", "genes_a": ["LTB"], "genes_b": ["LTBR"]})
    flat = ToolResult(id="R5", tool="covary", ok=True, summary="", args={"a": "LTB@B cell", "b": "CXCL13@Fibroblast"},
                      data={"rho_all": 0.9, "rho_within": {"Sweet": 0.3}, "null": {"n": 400, "frac": 0.7}})
    res = [screen, sweep, describe, wrong_cell, flat]
    st = inv.chain_status("CXCL13", res, "Fibroblast", **ctx, qs=[q])
    assert not st["complete"] and "not in the receiving cell" in st["next"]
    reasons = " ".join(st["unbacked"][-1]["invalid_tests"])
    assert "descriptive" in reasons and "does not discriminate" in reasons
    # the response of the receiving cell's pathway is a test
    res.append(ToolResult(id="R6", tool="pathway_footprint", ok=True, summary="", data={},
                          args={"lineage": "Fibroblast", "sets": ["NFKB2"]}))
    st = inv.chain_status("CXCL13", res, "Fibroblast", **ctx, qs=[q])
    assert st["complete"] and [j["ligand"] for j in st["judged"]] == ["LTB"] and st["judged"][0]["tests"] == ["R6"]


def test_verdicts_count_only_for_their_own_gene():
    """v13 failure: one verdict on LTB for CXCL5 completed the chains of CXCL5, IL7R, THBS1 and IL1A."""
    from mechanist.records import Candidate, ToolResult
    screen = lambda rid, g: ToolResult(id=rid, tool="upstream_screen", args={"gene": g, "cell": "Fibroblast"},
                                       ok=True, summary="", data={"tfs": ["NFKB2"], "receptors_on_cell": ["LTBR"],
                                       "routes": [{"receptor": "LTBR", "ligands": ["LTB"]}],
                                       "candidates": [{"ligand": "LTB", "sender": "B cell", "kind": "paracrine"}]})
    test = ToolResult(id="R3", tool="pathway_footprint", ok=True, summary="", data={},
                      args={"lineage": "Fibroblast", "sets": ["NFKB2"]})
    q = Question(id="Q01", kind="upstream", text="Test LTB as an inducer of fibroblast CXCL5", cells=["Fibroblast"],
                 genes=["CXCL5", "LTB"], status="answered", result_ids=["R3"],
                 candidates=[Candidate(ligand="LTB", sender="B cell", verdict="supported")])
    res = [screen("R1", "CXCL5"), screen("R2", "IL7R"), test]
    assert inv.chain_status("CXCL5", res, "Fibroblast", qs=[q])["complete"]
    assert not inv.chain_status("IL7R", res, "Fibroblast", qs=[q])["complete"]


def test_failed_premise_resolves_the_chain():
    from mechanist.records import ToolResult
    r = ToolResult(id="R1", tool="upstream_screen", args={"gene": "IL1A", "cell": "Fibroblast"}, ok=True, summary="",
                   data={"gene": "IL1A", "cell": "Fibroblast", "premise": False})
    st = inv.chain_status("IL1A", [r], "Fibroblast", qs=[])
    assert st["complete"] and st["premise_failed"]


def test_cross_thread_patterns_and_open_leads():
    from mechanist.agents import cross_patterns, open_leads
    from mechanist.records import Candidate
    qs = [Question(id=f"Q0{i}", kind="upstream", text=f"Why {g}?", genes=[g], status="answered",
                   candidates=[Candidate(ligand="TGFB1", verdict="supported")], follow_ups=[f"next {g}"])
          for i, g in enumerate(["CXCL5", "IL7R", "TDO2"], 1)]
    qs.append(Question(id="Q04", kind="upstream", text="Why CXCL5 again?", genes=["CXCL5"], status="answered",
                       parent_id="Q01", candidates=[Candidate(ligand="TGFB1", verdict="rejected"),
                                                    Candidate(ligand="OSM", verdict="untested")]))
    pats = cross_patterns(qs)
    assert any("3 separate threads" in p and "CXCL5, IL7R, TDO2" in p for p in pats)
    assert any("rejected/weakened in Q04" in p for p in pats)
    leads = open_leads(qs)
    assert "Q02 raised: next IL7R" in leads and not any(x.startswith("Q01 raised") for x in leads)  # Q01 followed
    assert "Q04 left untested: OSM" in leads
