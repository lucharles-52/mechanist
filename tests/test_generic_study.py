"""A second, synthetic skinpipe study with other groups and lineages: nothing of p033 (PG, Sweet, Control,
neutrophilic dermatoses) may leak into loading, findings, tools, the panel or the prompts. No real data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from mechanist.adapters import skinpipe
from mechanist.llm import LLMConfig, Replay
from mechanist.records import ToolCall
from mechanist.runner import RunConfig, Runner, derive_panel
from mechanist.tools import CellData, Toolbox

STUDY = "p999"
RAW = {"Lupus · Skin · Lesional": "Lupus", "Psoriasis · Skin · Lesional": "Psoriasis", "Healthy · Skin": "Healthy"}
UNITS = {"Lupus": ["L1", "L2", "L3"], "Psoriasis": ["Q1", "Q2", "Q3"], "Healthy": ["H1", "H2", "H3", "H4"]}
GENES = ["HOUSE", "IFI27", "KIT", "GZMB", "PMEL", "KRT14"]
# lineage -> state names (second state = the disease-expanded one)
LINEAGES = {"Keratinocyte": ["Basal keratinocyte", "IFN-high keratinocyte"], "Melanocyte": ["Melanocyte"],
            "Mast cell": ["Mast cell"], "CD8 T cell": ["Resting CD8 T cell", "GZMB+ CD8 T cell"]}
OPTIONS = {"groups": RAW}


def _cells_for(group: str, sample: str):
    """Rows of raw counts. IFI27: every Lupus keratinocyte sample above every other sample. GZMB: Lupus CD8 T cells.
    KIT: Mast cells everywhere (no difference). Healthy has more melanocytes."""
    rows, obs = [], []
    short = RAW[group]
    lupus = short == "Lupus"
    for lin, states in LINEAGES.items():
        n = {"Melanocyte": 30 if short != "Healthy" else 60}.get(lin, 40)
        for i in range(n):
            state = states[-1] if (len(states) > 1 and lupus and i < n * 3 // 4) else states[0]
            x = dict.fromkeys(GENES, 0)
            x["HOUSE"] = 20
            if lin == "Keratinocyte":
                x["KRT14"] = 8
                x["IFI27"] = (6 if lupus else 0) if i % 2 == 0 else (2 if lupus else 0)
                if short == "Psoriasis" and i % 10 == 0:
                    x["IFI27"] = 1
            if lin == "Mast cell":
                x["KIT"] = 5
            if lin == "Melanocyte":
                x["PMEL"] = 6
            if lin == "CD8 T cell" and state.startswith("GZMB"):
                x["GZMB"] = 4
            rows.append([x[g] for g in GENES])
            obs.append({"replicate_unit": sample, "analysis_group": group, "Reviewed_Detailed_Label": state,
                        "Final_Reviewed_Broad_Group": lin})
    return rows, obs


def write_generic_study(root: Path, *, labels_ok: bool = True, signoff: bool = True,
                        state: str = "complete") -> Path:
    import anndata as ad
    from scipy import sparse

    sd = root / STUDY
    (sd / "metadata").mkdir(parents=True)
    (sd / "metadata" / "run_state.yaml").write_text(yaml.safe_dump({"status": state}))
    if signoff:
        so = sd / "Results" / "compartments" / "all_cells" / "post_review" / "tables"
        so.mkdir(parents=True)
        (so / "all_cells_final_clusterwise_summary.xlsx").write_bytes(b"synthetic sign-off")
    inv = {v: k for k, v in RAW.items()}
    rows, obs = [], []
    for g, units in UNITS.items():
        for u in units:
            r, o = _cells_for(inv[g], u)
            rows += r
            obs += o
    # a group outside the study (dropped by the groups option)
    obs.append({"replicate_unit": "B1", "analysis_group": "Healthy · Blood", "Reviewed_Detailed_Label": "Mast cell",
                "Final_Reviewed_Broad_Group": "Mast cell"})
    rows.append([20, 0, 5, 0, 0, 0])
    if not labels_ok:
        obs[0]["Reviewed_Detailed_Label"] = "C0"
    counts = sparse.csr_matrix(np.array(rows, dtype=np.float32))
    lib = np.asarray(counts.sum(1)).ravel()
    norm = sparse.diags(1e4 / lib) @ counts
    norm.data = np.log1p(norm.data)  # .X normalised, raw counts in the 'counts' layer, as in current studies
    a = ad.AnnData(X=sparse.csr_matrix(norm, dtype=np.float32),
                   obs=pd.DataFrame(obs, index=[f"c{i}" for i in range(len(obs))]).astype("category"),
                   var=pd.DataFrame(index=GENES), layers={"counts": counts})
    obj = sd / "Intermediary" / "compartments" / "all_cells" / "post_review" / "all_cells_final.h5ad"
    obj.parent.mkdir(parents=True)
    a.write_h5ad(obj)

    pair = sd / "Results" / "comparisons" / "global__lupus" / "lupus_skin_lesional_vs_healthy_skin"
    route = pair / "all_cells"
    route.mkdir(parents=True)
    (pair / "run_manifest.yaml").write_text(yaml.safe_dump({
        "comparison_id": "global__lupus", "case": inv["Lupus"], "reference": inv["Healthy"],
        "routes": [{"route": "all_cells", "groupby": "contrast_population", "sample_col": "replicate_unit",
                    "replicates": {"case": 3, "reference": 4}}]}, allow_unicode=True))
    pd.DataFrame([{"contrast_population": "Keratinocyte", "gene": "IFI27", "baseMean": 50, "log2FoldChange": 4.0,
                   "lfcSE": 0.5, "stat": 8.0, "pvalue": 1e-10, "padj": 1e-8}]).to_csv(
        route / "pseudobulk_differential_expression.csv", index=False)
    pd.DataFrame([{"contrast_population": "Melanocyte", "case": inv["Lupus"], "reference": inv["Healthy"], "n_case": 3,
                   "n_reference": 4, "mean_case": 0.15, "mean_reference": 0.25, "delta_mean": -0.10,
                   "test": "mannwhitney_unpaired", "pvalue": 0.06, "n_pairs": 0, "qvalue": 0.2}]).to_csv(
        route / "cell_composition_tests.csv", index=False)
    pd.DataFrame([{"condition": inv["Lupus"], "source": "CD8 T cell", "target": "Keratinocyte", "ligand": "IFNG",
                   "receptor": "IFNGR1_IFNGR2", "n_samples": 3, "mean_magnitude": 0.4, "detection_fraction": 1.0,
                   "ref_n_samples": 4, "ref_mean_magnitude": 0.05, "ref_detection_fraction": 0.5,
                   "reference": inv["Healthy"], "delta_mean": 0.35}]).to_csv(
        route / "communication_condition_summary.csv", index=False)
    return root


@pytest.fixture
def generic(tmp_path) -> Path:
    return write_generic_study(tmp_path / "Projects")


def _cfg(projects: Path, **kw) -> RunConfig:
    return RunConfig(study=STUDY, adapter="skinpipe", disease="Lupus", adapter_options=OPTIONS,
                     projects_root=str(projects), disease_phenotype="photosensitive scarring plaques",
                     investigation="rounds", investigate_rounds=2, knowledge=False, literature=False, scout=False,
                     llm=LLMConfig(backend="replay"), **kw)


def test_adapter_reads_a_reviewed_study_with_its_own_groups(generic):
    s = skinpipe.load(generic, STUDY, primary="Lupus", **OPTIONS)
    ds = s.dataset
    assert ds.groups == UNITS and ds.reference_group == "Healthy" and ds.review_status == "reviewed"
    assert s.standard_contrast == ("Lupus", "Healthy") and s.standard_n == (3, 4)
    assert "broad_group" in s.tables["de"].df and s.tables["communication"].df.condition.tolist() == ["Lupus"]
    assert s.tables["composition"].df.case.tolist() == ["Lupus"]
    assert s.cells.counts_layer == "counts" and s.cells.lineage_col == "Final_Reviewed_Broad_Group"
    assert any(c.startswith("missing: gsea") for c in ds.caveats)


@pytest.mark.parametrize("kw", [{"labels_ok": False}, {"signoff": False}, {"state": "waiting_for_review"}])
def test_adapter_refuses_unreviewed_studies(tmp_path, kw):
    root = write_generic_study(tmp_path / "Projects", **kw)
    with pytest.raises(skinpipe.UnreviewedStudy):
        skinpipe.load(root, STUDY, primary="Lupus", **OPTIONS)


def test_runner_rejects_a_disease_that_is_not_a_group(generic, tmp_path):
    cfg = _cfg(generic)
    cfg.disease = "PG"
    with pytest.raises(ValueError, match="'PG' is not a group"):
        Runner(cfg, tmp_path / "runs" / "bad", llm_backend=Replay({})).run(stop_after="load")


def test_findings_stage_on_a_second_study(generic, tmp_path):
    run_dir = tmp_path / "runs" / "f"
    Runner(_cfg(generic), run_dir, llm_backend=Replay({})).run(stop_after="findings")
    out = json.loads((run_dir / "findings.json").read_text())
    keys = {f["row_key"] for f in out["findings"] if f["kind"] == "expression"}
    # raw counts from the 'counts' layer (.X is normalised): the genome-wide scan ran, every pair derived
    assert "Keratinocyte|IFI27|Lupus|Healthy" in keys and "Keratinocyte|IFI27|Lupus|Psoriasis" in keys
    assert set(out["chance_separations"]) == {"Lupus vs Healthy", "Psoriasis vs Healthy", "Lupus vs Psoriasis"}
    kinds = {f["kind"] for f in out["findings"]}
    assert {"de", "composition", "communication", "expression"} <= kinds
    assert not (run_dir / "llm_calls.jsonl").exists()


def test_tools_order_groups_by_study(generic):
    s = skinpipe.load(generic, STUDY, primary="Lupus", **OPTIONS)
    cells = CellData(s.cells)
    assert cells.is_counts and "Healthy" in set(cells.obs.group) and "B1" not in set(cells.obs["sample"])
    tb = Toolbox(s, [], cells, primary="Lupus")
    assert tb.order == ["Lupus", "Psoriasis", "Healthy"]
    r = tb.run(ToolCall(tool="gene_profile", args={"gene": "IFI27", "lineage": "Keratinocyte"}), "R0001")
    assert r.ok and r.summary.split("\n")[1].startswith("Keratinocyte: Lupus: L1")
    assert "all Lupus > all Psoriasis; all Lupus > all Healthy" in r.summary
    assert "Pipeline DE (Lupus vs Healthy)" in r.summary
    c = tb.run(ToolCall(tool="covary", args={"a": "IFI27@Keratinocyte", "b": "lineage:Melanocyte"}), "R0002")
    assert c.ok and set(c.data["rho_within"]) == {"Healthy"}  # only Healthy has >= 4 replicates
    m = tb.run(ToolCall(tool="state_markers", args={"state": "GZMB+ CD8 T cell", "n": 1}), "R0003")
    assert m.ok and m.data["genes"] == ["GZMB"]
    sig = tb.run(ToolCall(tool="incoming_signals", args={"lineage": "Keratinocyte"}), "R0004")
    assert sig.ok and "Lupus 0.40" in sig.summary and "vs Healthy 0.05" in sig.summary


def test_panel_is_derived_from_the_study():
    panel = derive_panel(["Keratinocyte", "Melanocyte", "Mast cell", "CD8 T cell"], "Lupus", ["Psoriasis"], "Healthy")
    by = {p.title: p for p in panel}
    assert list(by) == ["Tissue biologist", "Adaptive immunologist", "Innate immunologist", "Clinical pathologist",
                        "Computational biologist"]
    assert "Melanocyte" in by["Tissue biologist"].expertise and "Mast cell" in by["Innate immunologist"].expertise
    assert "CD8 T cell" in by["Adaptive immunologist"].expertise
    assert "Lupus compared with Psoriasis, Healthy" in by["Clinical pathologist"].expertise
    assert [p.title for p in panel if p.artifacts] == ["Computational biologist"]
    # a study without immune lineages gets no immunologists, but always the clinician and the sceptic
    assert [p.title for p in derive_panel(["Keratinocyte"], "X", [], "Y")] == [
        "Tissue biologist", "Clinical pathologist", "Computational biologist"]


def test_multi_agent_replay_on_a_second_study(generic, tmp_path):
    run_dir = tmp_path / "runs" / "m"
    Runner(_cfg(generic), run_dir / "pre", llm_backend=Replay({})).run(stop_after="findings")
    fs = json.loads((run_dir / "pre" / "findings.json").read_text())["findings"]
    ifi = next(f["id"] for f in fs if f["row_key"] == "Keratinocyte|IFI27|Lupus|Healthy")
    comp = next(f["id"] for f in fs if f["kind"] == "composition")
    hyp = {"title": "CD8 T-cell interferon drives keratinocyte IFI27", "summary": "s", "disease": "Lupus",
           "steps": [{"source_cell": "CD8 T cell", "target_cell": "Keratinocyte", "mediator": "IFNG",
                      "genes": ["IFI27"], "direction": "up", "finding_ids": [ifi]}]}
    alt = {"title": "Melanocyte loss is composition only", "summary": "s", "disease": "Lupus", "non_mechanistic": True,
           "steps": [{"source_cell": "Melanocyte", "mediator": "proportion", "direction": "down",
                      "finding_ids": [comp]}]}
    r1 = {"answers": [], "questions": [
        {"kind": "upstream", "text": "Why do keratinocytes make IFI27?", "cells": ["Keratinocyte"], "genes": ["IFI27"],
         "hypothesis_ids": ["H1"], "calls": [{"tool": "gene_profile", "args": {"gene": "IFI27"}},
                                             {"tool": "incoming_signals", "args": {"lineage": "Keratinocyte"}}]}]}
    r2 = {"answers": [{"question_id": "Q01", "status": "partial", "answer": "IFI27 in every Lupus keratinocyte "
                       "sample (R0001); IFNG->IFNGR1_IFNGR2 rises (R0002).", "answer_ids": ["R0001", "R0002"],
                       "conclusion": "observed"}], "questions": []}
    syn = {"title": "Interferon keratinocyte program", "summary": "s",
           "links": [{"source": "CD8 T cell", "target": "Keratinocyte", "mediator": "IFNG",
                      "mediator_in_source": "not_checked", "receptor_in_target": "not_checked",
                      "ids": ["R0002"], "question_ids": ["Q01"]}],
           "disease_link": "untested", "open_questions": ["does GZMB+ CD8 kill keratinocytes?"], "experiments": ["IHC"]}
    critic = {"verdict": "weak", "objections": [{"target": "link 1", "kind": "alternative", "text": "Could be IFN-a.",
                                                 "finding_ids": [ifi]}],
              "questions": [{"kind": "function", "text": "What are the GZMB+ CD8 T cells doing?", "cells": ["CD8 T cell"],
                             "hypothesis_ids": ["H1"],
                             "calls": [{"tool": "state_markers", "args": {"state": "GZMB+ CD8 T cell"}}]}]}
    chase = {"answers": [{"question_id": "Q02", "status": "answered", "answer": "GZMB marks them (R0003).",
                          "answer_ids": ["R0003"], "conclusion": "observed"}], "questions": []}
    responses = {"panel": json.dumps({"hypotheses": [hyp]}),
                 "merge": json.dumps({"hypotheses": [hyp | {"origin": ["Adaptive immunologist"]},
                                                     alt | {"origin": ["Computational biologist"]}]}),
                 "investigate:r1": json.dumps(r1), "investigate:r2": json.dumps(r2),
                 "synthesize": json.dumps(syn), "critique_model": json.dumps(critic),
                 "chase:r1": json.dumps(chase), "chase:r2": json.dumps({"answers": []}),
                 "revise": json.dumps(syn | {"title": "Revised interferon model"})}
    prompts: list[str] = []

    class Spy(Replay):
        def complete(self, tag, system, user, temperature=None):
            prompts.append(system + user)
            return super().complete(tag, system, user, temperature)

    Runner(_cfg(generic), run_dir / "run", llm_backend=Spy(responses)).run()
    out = run_dir / "run"
    panel = json.loads((out / "panel.json").read_text())
    assert set(panel) == {"Tissue biologist", "Adaptive immunologist", "Innate immunologist", "Clinical pathologist",
                          "Computational biologist"}
    gen = json.loads((out / "generate.json").read_text())
    assert [h["id"] for h in gen] == ["H1", "H2"] and gen[1]["non_mechanistic"]
    chased = json.loads((out / "chase.json").read_text())
    assert [q["asked_by"] for q in chased["questions"]] == ["investigator", "critic"]
    res = {r["id"]: r for r in chased["results"]}
    assert res["R0001"]["ok"] and "Keratinocyte: Lupus: L1" in res["R0001"]["summary"]
    assert res["R0003"]["ok"] and res["R0003"]["data"]["genes"][0] == "GZMB"
    assert json.loads((out / "revise.json").read_text())["synthesis"]["title"] == "Revised interferon model"
    report = (out / "report.md").read_text()
    assert "p999, Lupus" in report and "## Specialist panel" in report
    text = "\n".join(prompts)
    for leaked in ("Sweet", "PG ", "neutrophilic dermatos", "Control"):
        assert leaked not in text, leaked
    assert "photosensitive scarring plaques" in text  # the phenotype reaches the panel


def test_the_submitters_question_reaches_every_stage(tmp_path):
    from mechanist import agents
    from mechanist.adapters.skinpipe import study_question
    from mechanist.records import Dataset
    (tmp_path / "metadata").mkdir()
    (tmp_path / "metadata" / "submission.yaml").write_text(
        "project:\n  title: Sarcoidosis vs other granulomas\n  description: '- Compare TLS-associated B cells'\n")
    q = study_question(tmp_path)
    assert q == "Title: Sarcoidosis vs other granulomas\n- Compare TLS-associated B cells"
    assert study_question(tmp_path / "none") == ""
    ds = Dataset(study="s", adapter="skinpipe", groups={"A": ["a"]}, files=[], review_status="reviewed", question=q)
    ctx = agents.build_context(ds, [], {})
    assert ctx.startswith("THE SUBMITTER'S QUESTION") and "TLS-associated B cells" in ctx
    assert not agents.build_context(Dataset(study="s", adapter="skinpipe", groups={}, files=[], review_status="reviewed"),
                                    [], {}).startswith("THE SUBMITTER")


def test_the_whole_study_comparison_wins_over_a_deep_dive_that_sorts_first(tmp_path):
    import yaml

    from mechanist.adapters.skinpipe import _pick_comparison
    comp = tmp_path / "Results" / "comparisons"
    for cid, route, extra in (("deep_dive_x__a_vs_b", "focused_compartment", {"compartment": "X"}), ("global__a_vs_b__1", "all_cells", {})):
        pair = comp / cid / "a_vs_b"
        (pair / route).mkdir(parents=True)
        (pair / "run_manifest.yaml").write_text(yaml.safe_dump({"case": "A", "reference": "B", "comparison_id": cid,
                                                                "routes": [{"route": route}], **extra}))
    pair, _d, r = _pick_comparison(tmp_path, "A", "B", None, None)
    assert pair.parent.name == "global__a_vs_b__1" and r["route"] == "all_cells"
