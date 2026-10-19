import pandas as pd

from mechanist.records import ToolCall
from mechanist.tools import Toolbox


def _run(tb, tool, **args):
    return tb.run(ToolCall(tool=tool, args=args), "R0001")


def test_lineage_share_reports_per_sample_agreement(study, findings, cells):
    r = _run(Toolbox(study, findings, cells), "lineage_share", lineage="B cell")
    assert r.ok and "P1 54.5%" in r.summary and "all PG > all Control" in r.summary
    assert set(r.data["pct"]) == {"P1", "P2", "S1", "S2", "S3", "C1", "C2", "C3", "C4"}


def test_state_composition_and_markers(study, findings, cells):
    tb = Toolbox(study, findings, cells)
    r = _run(tb, "state_composition", lineage="B cell")
    assert "Activated HLA-II+ B cell: PG: P1 100.0%" in r.summary
    m = _run(tb, "state_markers", state="Activated HLA-II+ B cell", n=3)
    assert m.ok and m.data["genes"][0] == "HLA-DRA"


def test_coexpression_detects_shared_program(study, findings, cells):
    r = _run(Toolbox(study, findings, cells), "coexpression", lineage="Fibroblast", genes_a=["CXCL13"],
             genes_b=["CXCL1"])
    assert r.ok and abs(r.data["P1"]["ratio"] - 2.0) < 1e-9
    assert "C1" in r.data and r.data["C1"]["pct_both"] == 0


def test_gene_profile_includes_pipeline_de(study, findings, cells):
    r = _run(Toolbox(study, findings, cells), "gene_profile", gene="CXCL13")
    assert "Fibroblast: PG: P1 40.0%" in r.summary and "Pipeline DE" in r.summary


def test_covary_and_table_tools(study, findings, cells):
    tb = Toolbox(study, findings, cells)
    c = _run(tb, "covary", a="CXCL13@Fibroblast", b="lineage:B cell")
    assert c.ok and c.data["rho_all"] > 0.5
    s = _run(tb, "incoming_signals", lineage="Fibroblast")
    assert s.ok and "IL1B->IL1R1" in s.summary
    f = _run(tb, "find_findings", query="CXCL13")
    assert f.ok and f.data["ids"]


def test_tool_errors_are_results_not_exceptions(study, findings, cells):
    tb = Toolbox(study, findings, cells)
    assert not _run(tb, "gene_profile", gene="NOTAGENE").ok
    assert not _run(tb, "state_composition", lineage="Astrocyte").ok
    assert not _run(tb, "no_such_tool").ok
    assert not _run(tb, "gene_profile", wrong_arg=1).ok


def test_catalog_hides_cell_tools_without_cells(study, findings):
    study.cells = None
    text = Toolbox(study, findings).catalog_text()
    assert "incoming_signals" in text and "coexpression" not in text


def test_ligand_targets_checks_receptors_missing_from_liana(study, findings, cells):
    r = _run(Toolbox(study, findings, cells), "ligand_targets", ligand="CXCL1", receptors=["CXCR2"])
    assert r.ok and "no pairs scored" in r.summary and "Not paired with CXCL1" in r.summary
    assert "NOT exhaustive" in r.summary and "WHO EXPRESSES THE LIGAND" in r.summary


def test_incoming_signals_hides_matrix_ligands(study, findings, cells):
    tb = Toolbox(study, findings, cells)
    study.tables["communication"].df.loc[0, "ligand"] = "COL1A1"
    study.tables["communication"].df.loc[0, "target"] = "Fibroblast"
    assert "COL1A1" not in _run(tb, "incoming_signals", lineage="Fibroblast").summary
    assert "COL1A1" in _run(tb, "incoming_signals", lineage="Fibroblast", include_matrix=True).summary


def test_gene_tools_accept_gene_lists(study, findings, cells):
    tb = Toolbox(study, findings, cells)
    r = _run(tb, "gene_profile", gene=["CXCL13", "NOTAGENE", "CXCL1"], lineage="Fibroblast")
    assert r.ok and set(r.data) == {"CXCL13", "CXCL1"} and "NOTAGENE" in r.summary
    assert _run(tb, "gene_by_state", gene=["CXCL13", "CXCL1"], lineage="Fibroblast").ok
    assert not _run(tb, "gene_profile", gene=["NOTAGENE"]).ok


def test_any_tool_exception_becomes_a_failed_result(study, findings, cells):
    tb = Toolbox(study, findings, cells)
    r = _run(tb, "ligand_targets", ligand=["CXCL13", "IL1B"])
    assert r.ok and "CXCL13" in r.summary and "IL1B" in r.summary
    tb.tools["gene_profile"].fn = lambda **kw: 1 / 0
    r = _run(tb, "gene_profile", gene="CXCL13")
    assert not r.ok and "ZeroDivisionError" in r.summary


def _mini_kb(tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "lr_consensus.csv").write_text("ligand,receptor\nCXCL13,CXCR5\nLTA_LTB,LTBR\nTNFSF14,LTBR\nTNF,TRAF2\n")
    (kb / "collectri.csv").write_text("source,target,weight\nRELB,CXCL13,1\nNFKB2,CXCL13,1\n")
    (kb / "progeny.csv").write_text("source,target,weight\nNFkB,CXCL13,2.5\n")
    (kb / "membrane_receptors.csv").write_text("gene,n_resources\nLTBR,20\nCXCR5,20\n")
    (kb / "omnipath_signaling.csv").write_text(
        "source,target,stimulation,inhibition,n_refs\nLTBR,MAP3K14,True,False,5\nMAP3K14,NFKB2,True,False,9\n"
        "TRAF2,MAP3K14,True,False,30\nNFKB2,RELB,True,False,3\n")
    return kb


def test_receptor_paths_and_consensus_receptors(study, findings, cells, tmp_path):
    from mechanist.knowledge import KnowledgeBase
    tb = Toolbox(study, findings, cells, kb=KnowledgeBase(_mini_kb(tmp_path)))
    r = _run(tb, "receptor_paths", tf="RELB")
    assert r.ok and r.data["routes"][0]["path"] == ["LTBR", "MAP3K14", "NFKB2", "RELB"]
    assert r.data["routes"][0]["ligands"] == ["LTA_LTB", "TNFSF14"]
    assert all(x["receptor"] != "TRAF2" for x in r.data["routes"])  # adaptors are not receptors
    t = _run(tb, "ligand_targets", ligand="CXCL13")
    assert t.ok and "Known receptors of CXCL13 (curated consensus resource): CXCR5" in t.summary
    assert "NOT exhaustive" not in t.summary


def test_truncation_is_explicit_and_names_hidden_genes():
    from mechanist.agents import clip
    text = "\n".join(["% of cells expressing DLL1:", "x" * 50, "% of cells expressing LTB:", "B cell: PG: P1 67.6%"])
    out = clip(text, 70)
    assert "TRUNCATED" in out and "LTB" in out.split("TRUNCATED")[1] and "67.6" not in out
    assert clip(text, 10_000) == text


def test_multi_gene_profile_is_compact(study, findings, cells):
    tb = Toolbox(study, findings, cells)
    r = _run(tb, "gene_profile", gene=["CXCL13", "COL1A1"])
    assert r.ok and "COL1A1" in r.summary and "CXCL13" in r.summary
    single = _run(tb, "gene_profile", gene="COL1A1")
    assert "top 3 lineages" not in single.summary  # single-gene calls keep every lineage


def test_gene_lists_are_capped(study, findings, cells):
    r = _run(Toolbox(study, findings, cells), "gene_profile", gene=["CXCL13"] * 7)
    assert not r.ok and "at most 6" in r.summary and "upstream_screen" in r.summary


def test_upstream_screen_keeps_what_the_data_supports(study, findings, tmp_path):
    """CXCL13 <- NFKB2 <- LTBR on fibroblasts; LTB from B cells separates PG; TNFSF14 does not; EGFR is absent."""
    import numpy as np
    from conftest import GROUPS
    from scipy import sparse

    from mechanist.knowledge import KnowledgeBase
    from mechanist.tools import CellData
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "lr_consensus.csv").write_text("ligand,receptor\nLTB,LTBR\nTNFSF14,LTBR\nEGF,EGFR\nCOL1A1,LTBR\n")
    (kb / "collectri.csv").write_text("source,target,weight\nNFKB2,CXCL13,1\nMISSINGTF,CXCL13,1\n")
    (kb / "progeny.csv").write_text("source,target,weight\n")
    (kb / "membrane_receptors.csv").write_text("gene,n_resources\nLTBR,20\nEGFR,20\n")
    (kb / "omnipath_signaling.csv").write_text(
        "source,target,stimulation,inhibition,n_refs\nLTBR,MAP3K14,True,False,5\nMAP3K14,NFKB2,True,False,9\n"
        "EGFR,MAP3K14,True,False,3\n")
    genes = ["CXCL13", "NFKB2", "LTBR", "EGFR", "LTB", "TNFSF14", "COL1A1"]
    obs, rows = [], []
    for g, ss in GROUPS.items():
        for s in ss:
            pg = g == "PG"
            for i in range(40):
                obs.append({"sample": s, "group": g, "lineage": "Fibroblast", "state": "fib"})
                rows.append([int(pg and i < 20), 1, int(i < (24 if pg else 8)), 0, 0, int(i < 4), int(pg or i < 2)])
            for i in range(40):
                obs.append({"sample": s, "group": g, "lineage": "B cell", "state": "b"})
                rows.append([0, 0, 0, 0, int(i < (28 if pg else 2)), int(i < 4), 0])
    cells = CellData.from_frame(pd.DataFrame(obs), sparse.csr_matrix(np.array(rows, dtype=float)), genes)
    tb = Toolbox(study, findings, cells, kb=KnowledgeBase(kb))
    r = _run(tb, "upstream_screen", gene="CXCL13", cell="Fibroblast")
    assert r.ok, r.summary
    d = r.data
    assert d["tfs_expressed"] == ["NFKB2"] and d["receptors_on_cell"] == ["LTBR"] and "EGFR" in d["receptors_off_cell"]
    kinds = {c["ligand"]: (c["sender"], c["kind"]) for c in d["candidates"]}
    assert kinds["LTB"] == ("B cell", "paracrine") and kinds["COL1A1"][1] == "matrix"
    assert "TNFSF14" in d["ligands_without_difference"]
    assert r.summary.index("LTB from B cell") < r.summary.index("COL1A1")
    no = _run(tb, "upstream_screen", gene="NFKB2", cell="Fibroblast")  # same in every group: nothing to explain
    assert no.ok and no.data["premise"] is False and "PREMISE FAILS" in no.summary
    p = _run(tb, "receptor_paths", tf="NFKB2", cell="Fibroblast")
    assert p.ok and "Not expressed on Fibroblast (ruled out): EGFR" in p.summary


def _contam_cells(b_cells_express_g: bool):
    import numpy as np
    from conftest import GROUPS
    from scipy import sparse

    from mechanist.tools import CellData
    genes = ["G", "COL1A1", "DCN", "MS4A1", "CD79A", "BANK1"]
    obs, rows = [], []
    for g, ss in GROUPS.items():
        for s in ss:
            for i in range(60):  # fibroblasts 0-14: G+ and carrying B-cell RNA; 15-19: G+ and clean
                obs.append({"sample": s, "group": g, "lineage": "Fibroblast", "state": "fib",
                            "doublet_score": 0.3 if i < 15 else 0.05})
                b = 1 if i < 15 else 0
                rows.append([int(i < 20), 1, 1, b, b, 0])
            for i in range(40):
                obs.append({"sample": s, "group": g, "lineage": "B cell", "state": "b", "doublet_score": 0.05})
                rows.append([int(b_cells_express_g), 0, 0, 1, 1, 1])
    return CellData.from_frame(pd.DataFrame(obs), sparse.csr_matrix(np.array(rows, dtype=float)), genes)


def test_contamination_explains_a_signal_only_if_the_foreign_cells_make_the_gene(study, findings):
    r = _run(Toolbox(study, findings, _contam_cells(True)), "contamination_check", gene="G", lineage="Fibroblast")
    assert r.ok, r.summary
    assert "CAN EXPLAIN PART OF THE SIGNAL in 2 of 2" in r.summary
    assert "33% of all Fibroblast cells and 11% of those without" in r.summary
    # same foreign markers, but B cells do not express G: the markers cannot explain the G signal
    r = _run(Toolbox(study, findings, _contam_cells(False)), "contamination_check", gene="G", lineage="Fibroblast")
    assert "CANNOT EXPLAIN" in r.summary and not r.data["case_flags"]


def test_coexpression_says_when_a_ratio_is_chance(study, findings, cells):
    r = _run(Toolbox(study, findings, cells), "coexpression", lineage="Fibroblast", genes_a=["COL1A1"],
             genes_b=["CXCL13"])
    assert r.ok and "about chance" in r.summary and "does NOT support a link" in r.summary
    both = _run(Toolbox(study, findings, cells), "coexpression", lineage="Fibroblast", genes_a=["CXCL13"],
                genes_b=["CXCL1"])
    assert "ENRICHED" in both.summary and "above chance in every sample" in both.summary


def test_list_arguments_are_accepted(study, findings, cells):
    tb = Toolbox(study, findings, cells)
    r = _run(tb, "lineage_share", lineage=["B cell", "Fibroblast"])
    assert r.ok and "B cell as %" in r.summary and "Fibroblast as %" in r.summary
    c = _run(tb, "covary", a="CXCL13@Fibroblast", b=["CXCL1@Fibroblast", "MS4A1@B cell"])
    assert c.ok and c.summary.count("Spearman") == 2


def test_subset_contrast_scores_targets_in_expressing_vs_other_cells_of_the_same_sample(study, findings, tmp_path):
    """G+ cells carry the targets of TF 'T' (not of 'U'); the score is higher in G+ cells within every sample."""
    import numpy as np
    from conftest import GROUPS
    from scipy import sparse

    from mechanist.knowledge import KnowledgeBase
    from mechanist.tools import CellData
    kb = tmp_path / "kb"
    kb.mkdir()
    targets = [f"T{i}" for i in range(12)]
    others = [f"U{i}" for i in range(12)]
    (kb / "collectri.csv").write_text("source,target,weight\n" + "".join(f"TF1,{t},1\n" for t in targets)
                                      + "".join(f"TF2,{u},1\n" for u in others))
    (kb / "progeny.csv").write_text("source,target,weight\n")
    (kb / "lr_consensus.csv").write_text("ligand,receptor\n")
    genes = ["G", *targets, *others]
    obs, rows = [], []
    for g, ss in GROUPS.items():
        for s in ss:
            for i in range(60):
                pos = i < 30
                obs.append({"sample": s, "group": g, "lineage": "Fibroblast", "state": "fib" if i % 2 else "fib2"})
                rows.append([5 if pos else 0, *[(8 if pos else 2)] * 12, *[4] * 12])
    cells = CellData.from_frame(pd.DataFrame(obs), sparse.csr_matrix(np.array(rows, dtype=float)), genes)
    tb = Toolbox(study, findings, cells, kb=KnowledgeBase(kb))
    r = _run(tb, "subset_contrast", lineage="Fibroblast", gene="G", sets=["TF1", "TF2"])
    assert r.ok, r.summary
    sc = {s: d["scores"] for s, d in r.data["samples"].items()}
    assert all(v["TF TF1 targets"] > 0 for v in sc.values()) and all(v["TF TF1 targets"] > v["TF TF2 targets"]
                                                                       for v in sc.values())
    st = _run(tb, "subset_contrast", lineage="Fibroblast", gene="G", state="fib", sets="TF1")
    assert st.ok and "state 'fib'" in st.summary



def test_compute_runs_a_snippet_isolated_and_registers_its_result(study, findings, cells):
    import shutil

    import pytest
    if not shutil.which("bwrap"):
        pytest.skip("bubblewrap not installed")
    from mechanist.records import ToolCall
    tb = Toolbox(study, findings, cells)
    code = "m = mask(lineage='B cell')\nresult = round(float(m.mean()) * 100, 1)\nprint('B share', result)"
    ok = tb.run(ToolCall(tool="compute", args={"code": code, "purpose": "B share"}), "R0100")
    assert ok.ok and "B share" in ok.summary and ok.data["code"] == code and isinstance(ok.data["result"], float)
    net = tb.run(ToolCall(tool="compute", args={"code": "import socket\nsocket.create_connection(('8.8.8.8', 53), 3)"}),
                 "R0101")
    assert not net.ok  # no network inside the sandbox
    assert not tb.run(ToolCall(tool="compute", args={"code": "x = 1\n" * 2000}), "R0102").ok  # too long
