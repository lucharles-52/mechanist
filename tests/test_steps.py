"""Causal-step registry, exhausted steps, the output requirement, tool fixes, synthesis caching, the chat view."""

import json
import types

import pytest

from mechanist import investigate as inv
from mechanist.records import Hypothesis, Question, ToolCall
from mechanist.runner import Runner


def _q(i, step="", label="", status="partial", decisive="", tools=(), conclusion="none"):
    return Question(id=f"Q{i:02d}", kind="upstream", text=f"q{i}", status=status, step=step, step_label=label,
                    decisive_observation=decisive, calls=[ToolCall(tool=t, args={}) for t in tools],
                    conclusion=conclusion)


def test_steps_are_shared_by_id_or_label_and_new_ones_numbered():
    qs = [_q(1, "S01", "What induces the X layer")]
    assert inv.assign_step("S01", qs) == ("S01", "What induces the X layer")
    assert inv.assign_step("new: what induces the X-layer", qs) == ("S01", "What induces the X layer")  # same label
    assert inv.assign_step("new: why are B cells here", qs) == ("S02", "why are B cells here")
    assert inv.assign_step("", qs) == ("", "") and inv.assign_step("S09", qs) == ("", "")  # unknown id: no step


def test_a_step_is_exhausted_when_named_or_after_three_tries_without_an_answer():
    named = [_q(1, "S01", "a", "unanswerable_with_data", decisive="the ligand at protein level")]
    assert inv.is_exhausted("S01", named)
    tries = [_q(i, "S01", "a", tools=("covary",)) for i in (1, 2)]
    assert not inv.is_exhausted("S01", tries)
    assert inv.is_exhausted("S01", tries + [_q(3, "S01", "a", tools=("compute",))])
    settled = tries + [_q(3, "S01", "a", status="answered", conclusion="observed")]
    assert not inv.is_exhausted("S01", settled)
    txt = inv.steps_text(named + [_q(2, "S02", "b", status="answered", conclusion="observed")])
    assert "S01 [EXHAUSTED] a" in txt and "would be settled by: the ligand at protein level" in txt
    assert "S02 b" in txt and "S02 [EXHAUSTED]" not in txt


def test_an_ask_for_an_exhausted_step_needs_a_new_kind_of_test():
    events = []
    stub = types.SimpleNamespace(_deferred=[], event=lambda **kw: events.append(kw))
    qs = [_q(i, "S01", "what induces the layer", tools=("covary",)) for i in (1, 2, 3)]
    again = {"kind": "upstream", "text": "Which ligand induces the layer?", "step": "S01"}
    assert Runner._thread_question(stub, again, 4, "critic", 2, qs) is None
    assert stub._deferred[0]["step"] == "S01" and events[0]["action"] == "deferred"
    new = Runner._thread_question(stub, {**again, "new_approach": "isoform-level reads"}, 4, "critic", 2, qs)
    assert new is not None and new.step == "S01"
    other = Runner._thread_question(stub, {**again, "step": "new: what the layer makes"}, 5, "critic", 2, qs)
    assert other.step == "S02"


def test_repeated_demands_are_not_progress():
    before = [{"text": "Run the per-sample ambient correction for fibroblast CXCL13"}]
    assert inv.repeated_demands([{"text": "Run per-sample ambient correction for fibroblast CXCL13 again"}], before)
    assert not inv.repeated_demands([{"text": "Profile endothelial receptors for the neutrophil chemokines"}], before)
    assert not inv.repeated_demands([], before)


def test_output_is_required_for_hypothesis_cells_and_expanded_ones():
    h = Hypothesis.model_validate({"id": "H1", "title": "t", "summary": "s", "disease": "PG", "steps": [
        {"id": "H1.1", "source_cell": "Fibroblast", "target_cell": "B cell", "mediator": "CXCL13", "genes": ["CXCL13"],
         "direction": "up", "claim_type": "proposed_mechanism", "required_level": "transcript"}]})
    reqs = inv.requirements([h], expanded=["T cell"])
    out = {r.cell for r in reqs if r.kind == "output"}
    assert out == {"Fibroblast", "B cell", "T cell"} and all(r.priority == 1 for r in reqs if r.kind == "output")
    fn = Question(id="Q01", kind="function", text="what state are the B cells in", cells=["B cell"])
    op = Question(id="Q02", kind="output", text="what do B cells make", cells=["B cell"])
    missing = {r.cell for r in inv.missing(reqs, [fn]) if r.kind == "output"}
    assert "B cell" in missing  # knowing the state does not answer what the cells make
    assert "B cell" not in {r.cell for r in inv.missing(reqs, [op]) if r.kind == "output"}
    assert inv.normalize_kind("secretion") == "output"


def test_tool_drops_unknown_arguments_reuses_repeats_and_suggests_lineages(study, findings, cells):
    from mechanist.tools import Toolbox
    tb = Toolbox(study, findings, cells)
    a = tb.run(ToolCall(tool="lineage_share", args={"lineage": "B cell", "purpose": "is it expanded"}), "R0001")
    assert a.ok and any("ignored argument" in c and "purpose" in c for c in a.caveats)
    b = tb.run(ToolCall(tool="lineage_share", args={"lineage": "B cell"}), "R0002")
    assert b.ok and b.id == "R0002" and b.summary.startswith("(same call as R0001; same result)")
    bad = tb.run(ToolCall(tool="lineage_share", args={"lineage": "B cels"}), "R0003")
    assert not bad.ok and "did you mean B cell" in bad.summary


def test_synthesis_passes_read_the_previous_pass_log_from_cache(tmp_path):
    from mechanist.llm import AnthropicAPI, LLMConfig
    key = tmp_path / "k"
    key.write_text("sk-ant-test")
    api = AnthropicAPI(LLMConfig(backend="anthropic", external_ok=True, api_key_file=str(key)), key_file=str(key))
    ctx = "CONTEXT " * 800
    p1 = f"{ctx}\n\nInvestigation log:\nQ01 a\nQ02 b\n\nCAUSAL STEPS x\n\nKey results:\nR1"
    p2 = f"{ctx}\n\nInvestigation log:\nQ01 a\nQ02 b\nQ03 c\n\nPATTERNS ACROSS THREADS y\n\nKey results:\nR1 R2"
    s1 = api._split(p1, False, "synthesize")
    assert [t for _, t in s1] == ["1h", "1h", None] and s1[1][0].endswith("Q02 b")
    s2 = api._split(p2, False, "synthesize:pi2")
    assert s2[0][0] == p1[: p1.index("\n\nCAUSAL")] and s2[1][0] == "\nQ03 c" and s2[0][1] == s2[1][1] == "1h"
    assert api._split("no log here", False, "critique_model:pi2") == [("no log here", None)]


def test_conversation_view_threads_questions_and_their_tool_results(tmp_path):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("cv", Path(__file__).parents[1] / "scripts" / "conversation_view.py")
    cv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cv)
    run = tmp_path / "run"
    run.mkdir()
    rows = [{"time": "2026-10-17T10:00:00", "role": "planner", "tag": "investigate:plan1", "text": "THREADS: a | b"},
            {"time": "2026-10-17T10:00:01", "role": "analyst Q01", "tag": "investigate:Q01:s0",
             "text": "who makes it → gene_profile({\"gene\": \"CXCL13\"})"},
            {"time": "2026-10-17T10:00:02", "role": "tool", "tag": "R0001", "for": "investigate:Q01:s0",
             "text": "[R0001] gene_profile: CXCL13 in fibroblasts"},
            {"time": "2026-10-17T10:00:03", "role": "analyst Q01", "tag": "investigate:Q01:s1",
             "text": "ANSWER (answered): fibroblasts make it [R0001]"},
            {"time": "2026-10-17T10:00:04", "role": "PI", "tag": "critique", "text": "VERDICT weak: thin | 1 demands: run X"}]
    (run / "transcript.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (run / "investigate.json").write_text(json.dumps({"questions": [
        {"id": "Q01", "text": "Who makes CXCL13?", "status": "answered", "result_ids": ["R0001"], "step": "S01",
         "step_label": "source of CXCL13"}]}))
    page = cv.build(run)
    assert "# 4-investigation" in page and "Who makes CXCL13?" in page and "S01 source of CXCL13" in page
    assert "Checking: who makes it" in page and "<b>[R0001] gene_profile</b>" in page and "Opening threads" in page
    assert page.count('class="thread"') == 1 and "1 demands" in page and 'class="tech"' in page
    # with a checked rewrite, the plain text shows and the logged version moves into Details
    (run / "humanized.json").write_text(json.dumps({"critique#0": {"ok": True, "text": "I need one more test:\n- run X",
                                                                   "evidence": ["R0001"]}}))
    page = cv.build(run)
    assert "<li>run X</li>" in page and "<b>As logged</b>" in page and "<span>R0001</span>" in page


def test_rewrites_must_keep_ids_and_numbers_and_add_none():
    from mechanist.humanize import check
    orig = '{"answer": {"answer": "CXCL13 is in 42% of PG4L fibroblasts vs 3.1% in Sweet [R5100, F0012]"}}'
    assert check(orig, "CXCL13 is in 42% of PG4L fibroblasts against 3.1% in Sweet.", ["R5100", "F0012"])[0]
    assert not check(orig, "CXCL13 is in 45% of PG4L fibroblasts against 3.1% in Sweet.", ["R5100", "F0012"])[0]
    assert "drops ids" in check(orig, "CXCL13 is in 42% of fibroblasts vs 3.1%.", ["R5100"])[1]
    assert "names ids" in check(orig, "42% vs 3.1%", ["R5100", "F0012", "R9999"])[1]
    assert "drops numbers" in check(orig, "CXCL13 is higher in PG.", ["R5100", "F0012"])[1]
    long = orig + " " + "x" * 4000  # a long reply may be summarised: dropping is fine, inventing is not
    assert check(long, "CXCL13 is higher in PG fibroblasts.", [])[0]
    assert not check(long, "CXCL13 is in 50% of PG fibroblasts.", [])[0]


def test_lineage_changes_screens_every_gene_without_a_list(study, findings, cells):
    from mechanist.tools import Toolbox
    tb = Toolbox(study, findings, cells)  # no knowledge base: the screen needs none
    r = tb.run(ToolCall(tool="lineage_changes", args={"lineage": "Fibroblast"}), "R0001")
    assert r.ok, r.summary
    up = {d["gene"] for d in r.data["up_all"]}
    assert up == {"CXCL13", "CXCL1"} and "COL1A1" not in up  # expressed everywhere: not a change
    assert "above every" in r.summary and "lineage_changes" in tb.catalog_text()


def test_difference_account_lists_composition_states_and_genes(study, findings, cells):
    from mechanist.tools import Toolbox
    acc = Toolbox(study, findings, cells).difference_account()
    assert "B cell share of all cells up" in acc  # 60 vs 20 B cells per sample
    assert "Activated HLA-II+ B cell (B cell) share of B cell up" in acc
    assert "Fibroblast UP (2 genes): CXCL1" in acc and "CXCL13" in acc


def test_every_dossier_item_must_be_accounted_for_by_the_final_model():
    from mechanist.agents import phenotype_items, phenotype_text
    from mechanist.records import Link, PhenotypeFit, Synthesis
    items = phenotype_items({"defining_features": [{"text": "Undermined ulcer"}, "Pathergy"],
                             "therapies": [{"text": "Responds to anti-TNF"}], "data_observations": [{"text": "x"}]})
    assert [i for i, _, _ in items] == ["D01", "D02", "D03"] and items[2][1] == "therapy response"
    assert "D02 [defining feature] Pathergy" in phenotype_text(items)
    syn = Synthesis(title="t", summary="s", links=[Link(source="Macrophage", target="Fibroblast", mediator="TNF")],
                    phenotype_fit=[PhenotypeFit(item="D01", verdict="explained", links=["Macrophage -> Fibroblast"]),
                                   PhenotypeFit(item="D03", verdict="explained", links=["Keratinocyte -> ulcer"])])
    objs = inv.check_phenotype(syn, [i for i, _, _ in items])
    text = " ".join(o.text for o in objs)
    assert "1 of 3 disease items are not accounted for: D02" in text
    assert "explained without naming a link of the model: D03" in text
    assert inv.check_phenotype(syn, []) == []


def test_difference_account_has_a_baseline_tier(study, findings, cells):
    from mechanist.tools import Toolbox
    acc = Toolbox(study, findings, cells).difference_account()
    assert "PERTURBED FROM CONTROL BUT NOT SPECIFIC" in acc  # the fixture has a healthy reference and a comparator


def test_sweep_plan_dedupes_and_caps():
    from mechanist import agents
    from mechanist.llm import LoggedLLM

    class Fake:
        name = "fake"

        def complete(self, tag, system, user, temperature=None):
            q = [{"query": "product X tissue injury", "kind": "effector", "why": "w"},
                 {"query": "tissue injury product X", "kind": "effector", "why": "dup"},
                 {"query": "product X with partner Y hypoxia", "kind": "interaction", "why": "w"},
                 {"query": "", "kind": "effector"}, "junk", {"query": "third", "kind": "effector"}]
            return '{"queries": ' + __import__("json").dumps(q) + "}", {}
    import tempfile
    from pathlib import Path
    llm = LoggedLLM(Fake(), Path(tempfile.mkdtemp()))
    out = agents.sweep_plan(llm, "circuits:sweep", "p", "l", "d", "i", 2)
    assert [q["query"] for q in out] == ["product X tissue injury", "product X with partner Y hypoxia"]
    assert "at most 2 search queries" in agents.SWEEP_SYSTEM.format(n=2)


def test_index_patient_design_is_generic_and_accepted():
    from pathlib import Path

    from mechanist import agents
    from mechanist.runner import RunConfig
    path = Path(__file__).parents[1] / "configs" / "p044_pg_claude.yaml"
    if not path.exists():  # study configs stay out of the repository
        pytest.skip("needs the p044 study config")
    cfg = RunConfig.from_yaml(path)
    assert cfg.design == "index_patient" and cfg.disease == "Patient"
    text = agents.INDEX_PATIENT.format(patient="Patient")
    assert '"Patient"' in text and "candidate diagnosis" in text
    for word in ("pyoderma", "Sweet", "PG", "neutrophil", "IgM"):  # no disease, cell or gene is written into it
        assert word not in text


def test_a_label_that_starts_with_a_step_id_is_that_step():
    qs = [_q(1, "S06", "fibroblast secreted output and its inducer")]
    assert inv.assign_step("S06 fibroblast secreted output and its inducer", qs) == ("S06", qs[0].step_label)
    assert inv.assign_step("S06: the inducer again", qs)[0] == "S06"
    assert inv.assign_step("S99 something new", qs)[0] == "S07"  # an unknown id is not reused: a new step


def test_a_step_answered_only_by_inference_is_exhausted_after_three():
    inferred = [_q(i, "S01", "a", status="answered", conclusion="inferred") for i in (1, 2)]
    assert not inv.is_exhausted("S01", inferred)
    assert inv.is_exhausted("S01", inferred + [_q(3, "S01", "a", status="answered", conclusion="inferred")])
    on_data = inferred + [_q(3, "S01", "a", status="answered", conclusion="observed")]
    assert not inv.is_exhausted("S01", on_data)


def test_chase_progress_means_a_step_settled_not_a_step_opened():
    a = [_q(1, "S01", "a", status="answered", conclusion="inferred")]
    before = inv.step_registry(a)
    assert not inv.step_progress(before, inv.step_registry(a + [_q(2, "S01", "a", status="answered",
                                                                     conclusion="inferred")]))
    assert inv.step_progress(before, inv.step_registry(a + [_q(2, "S01", "a", status="answered",
                                                                 conclusion="observed")]))
    # a newly opened step is work, not progress (a PI opening one every round never stopped), unless settled at once
    assert not inv.step_progress(before, inv.step_registry(a + [_q(2, "S02", "b")]))
    assert inv.step_progress(before, inv.step_registry(a + [_q(2, "S02", "b", status="answered",
                                                                 conclusion="observed")]))


def test_every_ledger_link_must_be_accounted_for():
    acc = [{"link": "L1", "verdict": "used", "circuit": 1},
           {"link": "L2", "verdict": "contradicted", "ids": ["R9999"]},
           {"link": "L3", "verdict": "set_aside", "reason": "no"}]
    text = " ".join(o.text for o in inv.check_accounting(acc, ["L1", "L2", "L3", "L4"], {"R0001"}))
    assert "1 of 4 ledger links are not accounted for: L4" in text
    assert "L2 is called contradicted without a data id" in text and "L3 is set aside without a reason" in text
    ok = [{"link": "L1", "verdict": "used"}, {"link": "L2", "verdict": "contradicted", "ids": ["R0001"]}]
    assert inv.check_accounting(ok, ["L1", "L2"], {"R0001"}) == []


def test_a_gate_with_a_real_score_stops_when_nothing_is_established(monkeypatch):
    from mechanist import agents
    from mechanist.runner import RunConfig
    reviews_seen = []

    def fake_gate(llm, tag, context, phen, stage, material, reviews):
        reviews_seen.append(tag)
        words = ["alpha beta gamma", "delta epsilon zeta", "eta theta iota", "kappa lambda mu", "nu xi omicron",
                 "pi rho sigma", "tau upsilon phi", "chi psi omega"]
        return {"verdict": "revise", "demands": [{"text": words[(len(reviews_seen) - 1) % 8]}], "new_tests": 5,
                "new_testable_predictions": 3}
    monkeypatch.setattr(agents, "pi_gate", fake_gate)
    stub = types.SimpleNamespace(cfg=RunConfig(study="pX", disease="X", gate_cap=40, no_progress_limit=3), llm=None,
                                 event=lambda **kw: None)
    score = [(1, 0, 2)]
    # the PI keeps naming new tests, but nothing new is established: three reviews and it stops
    out = Runner._gate(stub, "ctx", "7. Circuits", lambda: "m", lambda d, n: None, "circuits", real=lambda: score[0])
    assert len(out) == 3 and out[-1]["closed_by"] == "no progress"
    # something is established every round: the per-gate cap ends it
    reviews_seen.clear()
    ticks = iter(range(100))
    out = Runner._gate(stub, "ctx", "7. Circuits", lambda: "m", lambda d, n: None, "circuits",
                       real=lambda: (next(ticks), 0, 0), cap=6)
    assert len(out) == 6 and out[-1]["closed_by"] == "review cap"


def test_artefact_circuits_are_removed_and_consensus_parses(tmp_path):
    from mechanist import agents
    from mechanist.llm import LoggedLLM
    from mechanist.records import Circuit
    cs = [Circuit(rank=1, title="Myeloid OSM converts the stroma", root="OSM above every comparator"),
          Circuit(rank=2, title="Source-batch and label artefacts explain the signal")]
    keep, objs = inv.artefact_circuits(cs)
    assert [c.rank for c in keep] == [1] and "removed" in objs[0].text

    class Fake:
        name = "fake"

        def complete(self, tag, system, user, temperature=None):
            assert "ATTEMPT 1" in user and "ATTEMPT 2" in user and "Merge these 2 attempts" in user
            return ('{"answer": {"groups": [{"title": "OSM converts the stroma", "members": [{"attempt": 2, "circuit": 1},'
                    ' {"attempt": 1, "circuit": 1}, {"attempt": 9, "circuit": 1}], "note": ""}],'
                    ' "ranking": "support first", "leads": [{"text": "IGF1 lost", "ids": ["R1"], "attempt": 1}]}}'), {}
    llm = LoggedLLM(Fake(), tmp_path)
    attempts = [{"attempt": 1, "circuits": [{"rank": 1, "title": "a", "summary": "A."}, {"rank": 2, "title": "a2"}],
                 "ranking": "r"},
                {"attempt": 2, "circuits": [{"rank": 1, "title": "b", "summary": "B.", "root": "OSM high"}]}]
    plan = agents.circuit_consensus(llm, "circuits:consensus", "p", "i", attempts)
    assert plan["leads"][0]["text"] == "IGF1 lost"
    built = inv.assemble_consensus(plan, attempts)
    # the shown version is copied from attempt 2 as written; support is counted by code; the left-out circuit is kept
    assert [c["support"] for c in built] == ["2 of 2", "1 of 2"]
    assert built[0]["root"] == "OSM high" and built[0]["title"] == "OSM converts the stroma"
    assert built[0]["merged_from"] == ["attempt 2, circuit 1", "attempt 1, circuit 1"]
    assert "attempt 1, circuit 1" in built[0]["summary"] and built[1]["title"] == "a2"
    assert Runner._circuit(built[0], 1).support == "2 of 2"


def test_missing_links_rank_experiments_and_check_therapies():
    circuits = [{"rank": 1, "support": "3 of 3", "edges": [
                    {"source": "Macrophage", "target": "Fibroblast", "via": "OSM", "status": "observed"},
                    {"source": "Fibroblast", "target": "B cell", "via": "CXCL13", "status": "likely"}]},
                {"rank": 2, "support": "2 of 3", "edges": [
                    {"source": "Fibroblast", "target": "B cell", "via": "CXCL13", "status": "speculative"}]},
                {"rank": 3, "support": "1 of 3", "edges": [
                    {"source": "Vessel", "target": "Fibroblast", "via": "HIF", "status": "speculative"}]}]
    ms = inv.missing_links(circuits, [{"id": "S05", "label": "inducer", "decisive": ["a spatial map"]}],
                           [{"text": "MERTK lost", "ids": ["R1"]}])
    # the shared untested edge is one link carried by circuits 1 and 2; circuit 3 (one attempt) adds none
    assert [m["kind"] for m in ms] == ["edge", "step", "lead"] and ms[0]["circuits"] == [1, 2]
    exps = [{"title": "same", "tests": ["M1"], "tier": "in_data", "perturbation": "p",
             "predictions": [{"circuit": 1, "outcome": "up"}, {"circuit": 2, "outcome": "up"}]},
            {"title": "split", "tests": ["M1", "M2"], "tier": "cohort", "perturbation": "p",
             "predictions": [{"circuit": 1, "outcome": "down"}, {"circuit": 2, "outcome": "no change"}]}]
    kept, objs, uncovered = inv.check_experiments(exps, circuits, ms, set())
    ranked = inv.rank_experiments(kept, circuits)
    assert [e["title"] for e in ranked] == ["split", "same"]  # circuits that agree decide nothing
    assert uncovered == ["M3"] and ranked[1]["discrimination"] == 0
    targets = inv.therapy_targets(circuits, ms)
    assert [t["gene"] for t in targets][:2] == ["CXCL13", "OSM"] and targets[0]["untested"] == ["M1"]
    lookups = {"CXCL13": {"drugs": [{"drug": "BLOCKMAB", "stage": "PHASE_2", "indications": ["lupus"],
                                     "actions": [{"action": "inhibitor", "mechanism": "CXCL13 inhibitor"}]},
                                    {"drug": "PUSHMAB", "stage": "APPROVAL", "indications": ["x"],
                                     "actions": [{"action": "agonist", "mechanism": "CXCL13 agonist"}]}]},
               "OSM": {"drugs": [{"drug": "OLDDRUG", "stage": "PRECLINICAL", "indications": [],
                                  "actions": [{"action": "antagonist", "mechanism": "m"}]}]}}
    hyps, objs = inv.check_therapies(
        [{"drug": "blockmab", "target": "CXCL13", "circuits": [1, 2], "depends_on": ["M1", "M9"]},
         {"drug": "PUSHMAB", "target": "CXCL13", "circuits": [2]},
         {"drug": "OLDDRUG", "target": "OSM", "circuits": [1]}, {"drug": "GHOST", "target": "OSM"}],
        lookups, circuits, ms, {"D01"}, "OLDDRUG is reported in case series", "PG")
    assert [h["drug"] for h in hyps] == ["OLDDRUG", "PUSHMAB", "BLOCKMAB"]  # in use, then approved, then trials
    assert hyps[0]["level"] == "in use for this disease" and hyps[2]["depends_on"] == ["M1"]
    assert not hyps[1]["blocks"] and any("not as a blocker" in o.text for o in objs)
    assert any("GHOST" in o.text for o in objs)
    pw = inv.pathway_weights([{"name": "chemokine", "genes": ["CXCL13", "NOTATARGET"]}], targets, hyps, circuits)
    assert pw[0]["genes"] == ["CXCL13"] and pw[0]["weight"] == round(1 + 2 / 3, 3)


def test_agent_labels_pass_through_and_drugs_are_classed_by_facts():
    from mechanist import cellfig
    circuits = [{"rank": 1, "support": "3 of 3", "edges": [
                    {"source": "Fibroblast", "target": "B cell", "via": "CXCL13", "status": "likely"}]},
                {"rank": 2, "support": "2 of 3", "edges": [
                    {"source": "Plasma cell", "target": "Neutrophil", "via": "IgG", "status": "speculative"},
                    {"source": "Macrophage", "target": "B cell", "via": "BAFF", "status": "observed"}]}]
    ms = inv.missing_links(circuits, [], [])
    mols = inv.crucial_molecules(circuits, ms)
    exps, objs, _ = inv.check_experiments([
        {"title": "stain", "kind": "presence check", "confirms": ["CXCL13"], "setting": "archived sections",
         "perturbation": "none", "controls": "isotype", "refutes_if": "no CXCL13 protein in fibroblasts",
         "caveat": "secreted: protein may sit away from the producing cell"},
        {"title": "co-culture", "kind": "a label no list knows", "perturbation": "recombinant CXCL13 on B cells",
         "setting": "primary cells"}],
        circuits, ms, set())
    # the agent's own kinds and settings are kept as written; nothing is mapped onto a fixed list
    assert [e["kind"] for e in exps] == ["presence check", "a label no list knows"]
    assert [e["setting"] for e in exps] == ["archived sections", "primary cells"]
    assert exps[0]["caveat"].startswith("secreted") and exps[0]["refutes_if"]
    assert any("names no controls" in o.text for o in objs)  # every experiment needs controls
    assert any("would count against the circuit" in o.text for o in objs)  # and a result that refutes it
    assert "CXCL13" not in inv.uncovered_molecules(exps, mols) and "BAFF" in inv.uncovered_molecules(exps, mols)
    cells = inv.cell_targets(circuits, {"B cell": ["MS4A1", "CD79A"], "Keratinocyte": ["KRT14"]}, cellfig.cells_in)
    assert cells == [{"lineage": "B cell", "genes": ["MS4A1", "CD79A"], "circuits": [1, 2]}]
    lookups = {"MS4A1": {"symbol": "MS4A1", "drugs": [{"drug": "RITUXIMAB", "stage": "APPROVAL", "indications": ["x"],
               "actions": [{"action": "binding agent", "mechanism": "CD20 binding agent"}],
               "warnings": [{"type": "Black Box Warning", "toxicity": "infectious disease"}]}]},
               "TNFSF13B": {"symbol": "TNFSF13B", "drugs": [{"drug": "BELIMUMAB", "stage": "APPROVAL", "indications": [],
                            "actions": [{"action": "inhibitor", "mechanism": "BAFF inhibitor"}], "warnings": []}]}}
    # a label the model sends is ignored: what a drug acts on is decided from the data's markers
    hyps, objs = inv.check_therapies([{"drug": "RITUXIMAB", "target": "MS4A1", "strategy": "step", "circuits": [1]},
                                      {"drug": "BELIMUMAB", "target": "TNFSF13B", "strategy": "cell type", "circuits": [2]}],
                                     lookups, circuits, ms, set(), "", "PG", cells)
    r = next(h for h in hyps if h["drug"] == "RITUXIMAB")
    b = next(h for h in hyps if h["drug"] == "BELIMUMAB")
    # acting on a cell type's marker: it acts in every circuit that uses the cell type
    assert r["strategy"] == "cell type" and r["depletes"] == "B cell" and r["circuits"] == [1, 2] and r["blocks"]
    assert r["robustness"] == 1.0 and r["boxed"] and b["strategy"] == "step" and b["robustness"] < 1
    assert not any("RITUXIMAB" in o.text for o in objs)


def test_sample_ids_are_not_molecules():
    circuits = [{"rank": 1, "support": "1 of 1", "edges": [
        {"source": "Neutrophil", "target": "Fibroblast", "via": "OSM (more than SSP3)", "status": "likely"}]}]
    genes = [m["gene"] for m in inv.crucial_molecules(circuits, [], exclude={"SSP3"})]
    assert "OSM" in genes and "SSP3" not in genes


def test_programmes_order_experiments_per_hypothesis_and_flag_gaps():
    circuits = [{"rank": 1, "support": "3 of 3", "edges": [{"source": "Macrophage", "target": "Fibroblast", "via": "X",
                                                          "status": "likely"}]},
                {"rank": 2, "support": "2 of 3", "edges": [{"source": "Plasma cell", "target": "Neutrophil", "via": "Y",
                                                          "status": "speculative"}]},
                {"rank": 3, "support": "1 of 3", "edges": []}]
    exps, _, _ = inv.check_experiments([
        {"id": "E1", "title": "X protein is in lesional macrophages", "question": "presence", "tier": "archived tissue",
         "readout": "staining", "controls": "isotype", "refutes_if": "absent", "done_before": [{"pmid": "123"}]},
        {"id": "E2", "title": "X turns fibroblasts into the lesional state", "question": "mechanism", "tier": "in_vitro",
         "perturbation": "recombinant X", "controls": "vehicle", "refutes_if": "no change"},
        {"id": "E2", "title": "a duplicate id", "question": "rule out", "readout": "in data", "controls": "c",
         "refutes_if": "r"}], circuits, [], set())
    assert [e["id"] for e in exps] == ["E1", "E2", "E3"]  # duplicate ids are renumbered
    assert exps[0]["tier"] == "archived_tissue" and exps[2]["question"] == "rule_out"
    assert exps[0]["done_before"] == []  # a PMID no search returned is dropped
    progs, objs = inv.check_programmes([
        {"circuit": 1, "claim": "X converts fibroblasts", "sequence": [{"id": "E1"}, {"id": "E2", "if_no": "drop 1"}]},
        {"circuit": 2, "claim": "Y keeps neutrophils on", "sequence": [{"id": "E1", "adapt": "stain Y"}, {"id": "E9"}]}],
        exps, circuits)
    assert [x["circuit"] for x in progs] == [1, 2] and [s["id"] for s in progs[1]["sequence"]] == ["E1"]
    assert exps[0]["in_programmes"] == [1, 2]  # shared by both hypotheses
    texts = " ".join(o.text for o in objs)
    assert "Circuit 2's sequence has no experiment that tests its central claim" in texts
    assert "E3" in texts and "sit in no hypothesis" in texts
    assert "Circuit 3" not in texts  # one attempt alone: no sequence required


def test_sample_wide_genes_are_those_raised_in_most_lineages():
    from mechanist.tools import sample_wide
    up = {"T cell": ["AMB", "CCR10"], "Fibroblast": ["AMB", "COL"], "Myeloid": ["AMB", "CXCL9"], "Endothelial": ["X"]}
    wide, rose, k = sample_wide(up)
    assert wide == {"AMB"} and k == 3 and rose["AMB"] == 3  # 3 of 4 lineages; CCR10 stays a T-cell change
    assert sample_wide({"A": ["G"], "B": ["G"]})[0] == set()  # two lineages cannot tell sample-wide from shared


def test_trial_search_keeps_trials_that_name_the_condition(tmp_path):
    from mechanist.knowledge import Literature, Trials

    def study(nct, title, conds, phase="PHASE2"):
        return {"protocolSection": {"identificationModule": {"nctId": nct, "briefTitle": title},
                                    "conditionsModule": {"conditions": conds},
                                    "statusModule": {"overallStatus": "RECRUITING", "startDateStruct": {"date": "2026-01"}},
                                    "designModule": {"phases": [phase]},
                                    "armsInterventionsModule": {"interventions": [{"name": "drug x"}]}}}
    body = {"studies": [study("NCT00000001", "Drug x in solid tumours", ["Neoplasms"]),
                        study("NCT00000002", "Drug x in adults", ["Rare ulcer disease"])]}
    urls = []
    lit = Literature(tmp_path / "c", forbidden={"s1"}, log_path=tmp_path / "q.jsonl")
    tr = Trials(lit, fetch=lambda u: urls.append(u) or body)
    text, data = tr.search("rare ulcer disease")
    assert [t["nct"] for t in data["trials"]] == ["NCT00000002"]  # the loose registry match is dropped
    assert "NCT00000002 (PHASE2, recruiting" in text and "drug x" in text
    tr.search("rare ulcer disease")
    assert len(urls) == 1  # cached
    with pytest.raises(Exception, match="study identifiers"):
        tr.search("s1 ulcer")


def test_dossier_lists_what_single_cell_cannot_measure_and_the_check_reports_it():
    from mechanist import agents
    d = {"mediators_genes": [{"text": "m", "ids": ["R5001"]}],
         "beyond_single_cell": [{"text": "a circulating cascade made in another organ", "ids": ["R5002"],
                                 "why": "the protein acts outside the captured cells", "trace": "its receptor on myeloid cells"}]}
    txt = agents.dossier_text(d)
    assert "Known, beyond what single-cell RNA can measure" in txt and "trace to look for: its receptor" in txt
    items = agents.phenotype_items(d)
    assert ("D02", "known, beyond single-cell", "a circulating cascade made in another organ") in items
    assert "beyond_single_cell" not in agents.DOSSIER_SECTIONS
    c = {"beyond_single_cell": [{"text": "a circulating cascade", "trace": "receptor raised", "question_ids": ["Q3"]}]}
    assert "Known but not measurable here" in agents.check_text(c) and "(trace: receptor raised)" in agents.check_text(c)
    for word in ("complement", "C3", "C5"):  # the prompt names no mechanism: the agents find them
        assert word not in agents.DOSSIER_SYSTEM


def test_a_family_target_is_looked_up_through_its_members_and_their_receptors():
    syms = {"FAMA", "FAMB", "FAMRN", "FAMX1", "RECA", "RECB", "OTHER"}
    pairs = {"FAMA": ["RECA_RECB"], "FAMB": ["RECA_RECB"], "FAMRN": ["RECA"]}

    def receptors_of(g):
        if g not in pairs:
            raise KeyError(g)
        return pairs[g]
    got = inv.expand_target("FAM", syms, receptors_of)
    assert [x["gene"] for x in got] == ["FAMA", "FAMB", "FAMRN", "FAMX1", "RECA", "RECB"]
    assert got[0]["via"] == "member of FAM" and got[4]["via"] == "receptor of FAMA"
    assert inv.expand_target("OTHER", syms, receptors_of) == [{"gene": "OTHER", "via": "named"}]
    # an unknown name is still looked up as named (Open Targets may know an alias)
    assert inv.expand_target("ZZZ", syms, receptors_of) == [{"gene": "ZZZ", "via": "named"}]


def test_considered_drugs_say_what_became_of_every_drug_found():
    from mechanist.records import Objection
    targets = [{"gene": "FAM", "circuits": [1], "lookups": [{"gene": "FAMA", "via": "member of FAM"},
                                                             {"gene": "RECA", "via": "receptor of FAMA"},
                                                             {"gene": "FAMZ", "via": "member of FAM"}]}]
    lk = {"FAMA": {"gene": "FAMA", "symbol": "FAMRN", "drugs": []},
          "RECA": {"gene": "RECA", "symbol": "RECA", "drugs": [
              {"drug": "DRUG1", "stage": "APPROVAL", "actions": [{"action": "inhibitor"}]},
              {"drug": "DRUG2", "stage": "PHASE_3", "actions": [{"action": "antagonist"}]},
              {"drug": "DRUG3", "stage": "PHASE_1", "actions": []}]}}
    hyps = [{"drug": "DRUG1", "target": "RECA"}]
    objs = [Objection(target="therapy", kind="unsupported_step", source="verifier",
                      text="DRUG3 on FAMA is not in any drug lookup that ran; dropped.")]
    out = inv.considered_drugs(targets, lk, hyps, objs, [{"drug": "drug2", "why": "a better drug on the same step"}])
    rows = {r["gene"]: r for r in out[0]["lookups"]}
    assert rows["FAMA"]["matched_as"] == "FAMRN" and "matched FAMA as FAMRN; no drugs" in rows["FAMA"]["note"]
    st = {d["drug"]: (d["status"], d["why"]) for d in rows["RECA"]["drugs"]}
    assert st == {"DRUG1": ("listed", ""), "DRUG2": ("passed over", "a better drug on the same step"),
                  "DRUG3": ("dropped", "")}
    assert rows["FAMZ"]["note"] == "lookup failed or not run"


def test_mechanism_cards_keep_only_numbers_and_quotes_their_sources_hold():
    from mechanist.records import ToolResult
    res = {"R1": ToolResult(id="R1", tool="gene_by_state", args={}, ok=True,
                            summary="SIG1 in CellA: case 42.5% of cells vs reference 3.1%"),
           "R2": ToolResult(id="R2", tool="literature_search", args={}, ok=True,
                            summary='PMID 111 (2020) A paper\n  "SIG1 drives CellB proliferation in lesional skin."',
                            data={"papers": [{"pmid": "111"}]})}
    edges = [{"source": "CellA", "target": "CellB", "via": "SIG1", "ids": ["R1", "R2"]}, {"source": "CellB", "target": "X"}]
    cards = [{"step": 1, "sender": "CellA", "signal": "SIG1", "receiver": "CellB", "receptor": "REC1",
              "effect": "Makes CellB divide in 90% of cases.", "follows": "", "leads_to": "Step 2 uses the dividing cells.",
              "seen_here": {"text": "42.5% of case CellA vs 3.1% of reference", "ids": ["R1"]},
              "literature": [{"claim": "drives proliferation", "quote": "SIG1 drives CellB proliferation in lesional skin.",
                              "pmid": "111", "id": "R2", "context": "skin"},
                             {"claim": "made up", "quote": "SIG1 cures everything.", "pmid": "111", "id": "R2"},
                             {"claim": "wrong pmid", "quote": "SIG1 drives CellB proliferation in lesional skin.",
                              "pmid": "999", "id": "R2"}],
              "must_hold": "REC1 on CellB"},
             {"step": 2, "seen_here": {"text": "7 of 8 cells", "ids": []}},
             {"step": 9}]
    out, dropped = inv.check_cards(cards, edges, res)
    assert [c["step"] for c in out] == [1, 2]
    c1 = out[0]
    assert c1["effect"] == "" and c1["seen_here"]["text"].startswith("42.5%") and c1["seen_here"]["ids"] == ["R1"]
    assert [x["pmid"] for x in c1["literature"]] == ["111"] and c1["receptor"] == "REC1"
    assert out[1]["seen_here"] == {"text": "", "ids": []}
    why = sorted((d["step"], d["field"], d["why"]) for d in dropped)
    assert (1, "effect", "numbers not in the cited results: 90") in why
    assert (1, "literature", "quote not in that result") in why and (1, "literature", "PMID not in that result") in why
    assert (2, "seen_here", "no cited result") in why
