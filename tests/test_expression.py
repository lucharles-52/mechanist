"""Genome-wide per-sample expression findings (all genes from raw counts, no curated panel)."""

import gzip
import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from conftest import GROUPS
from scipy import sparse

from mechanist.adapters import p033
from mechanist.findings import (
    FindingsConfig,
    build_findings,
    digest,
    expected_chance_separations,
    expression_from_counts,
    load_expression,
    make_expression,
    relabel_null,
    resolve_pairs,
)
from mechanist.llm import LLMConfig, Replay
from mechanist.runner import RunConfig, Runner


@pytest.fixture
def cstudy(projects_cells):
    return p033.load(projects_cells)


@pytest.fixture
def expr(cstudy):
    return load_expression(cstudy, FindingsConfig())


def _by_key(fs):
    return {f.row_key: f for f in fs}


def test_adapter_without_lab_tables_reads_groups_from_cells(cstudy):
    assert cstudy.dataset.groups == {g: sorted(s) for g, s in GROUPS.items()}
    assert not {"panel", "signatures", "ambient", "fib_programs"} & set(cstudy.tables)
    assert not any("pg_mechanism" in f.path for f in cstudy.dataset.files)
    assert not any(c.startswith("panel:") for c in cstudy.dataset.caveats)


def test_pseudobulk_from_counts(expr):
    assert expr.lineages == ["B cell", "Fibroblast"]  # Unmapped cells and out-of-study groups dropped
    li, si, gi = expr.lineages.index("Fibroblast"), expr.samples.index("P1"), list(expr.genes).index("GENEUP")
    # P1 fibroblasts: 40 cells x (10 HOUSE + 4 GENEUP + 3 MIXED + 2 IGKC) + 1 RARE = 761 counts
    assert expr.cpm[li, si, gi] == pytest.approx(160 / 761 * 1e6)
    assert expr.pct[li, si, list(expr.genes).index("RARE")] == pytest.approx(2.5)
    assert expr.cells[li, si] == 40


def test_normalised_matrix_is_refused():
    obs = pd.DataFrame({"sample": ["a", "b"], "lineage": ["X", "X"]})
    assert expression_from_counts(obs, sparse.csr_matrix([[0.5, 1.2], [0.3, 0.0]]), ["G1", "G2"], "s", "l") is None


def test_genome_wide_separation_findings(cstudy, expr):
    fs = build_findings(cstudy, FindingsConfig(), expr, primary="PG")
    k = _by_key(f for f in fs if f.kind == "expression")
    up = k["Fibroblast|GENEUP|PG|Control"]
    assert up.direction == "up" and up.separated and up.p_separation == 1 / 15
    assert set(up.per_sample) == {"P1", "P2", "C1", "C2", "C3", "C4"} and up.source_file.endswith(".h5ad")
    assert any(c.startswith("genome_wide: one of 3 Fibroblast genes (of 5 scanned)") for c in up.caveats)
    assert k["Fibroblast|GENEDOWN|PG|Control"].direction == "down"
    assert "Fibroblast|GENEUP|Sweet|Control" in k and "Fibroblast|GENEUP|PG|Sweet" in k
    assert "Fibroblast|MIXED|PG|Control" not in k  # the two PG samples disagree
    assert "Fibroblast|RARE|PG|Control" not in k  # one detecting cell per sample
    assert not any(f.cell_type == "B cell" for f in k.values())  # nothing differs in B cells
    igkc = k["Fibroblast|IGKC|PG|Control"]
    assert any(c.startswith("spillover") and "B cell" in c for c in igkc.caveats)
    assert igkc.priority < up.priority
    assert not any(f.kind in ("panel_expression", "signature") for f in fs)
    assert not any(c.startswith("panel") for f in fs for c in f.caveats)


def test_lab_tables_flag_adds_panel_findings(projects_cells):
    s = p033.load(projects_cells, lab_tables=True)
    e = load_expression(s, FindingsConfig())
    kinds = {f.kind for f in build_findings(s, FindingsConfig(), e, lab_tables=True)}
    assert {"expression", "panel_expression"} <= kinds
    assert "panel_expression" not in {f.kind for f in build_findings(s, FindingsConfig(), e)}


def test_default_pairs_and_chance(cstudy, expr):
    pairs = resolve_pairs(FindingsConfig(), cstudy.dataset.groups, "Control", primary="PG")
    assert pairs == [("PG", "Control"), ("Sweet", "Control"), ("PG", "Sweet")]
    assert resolve_pairs(FindingsConfig(pairs=[("PG", "Control"), ("X", "Control")]), cstudy.dataset.groups,
                         "Control") == [("PG", "Control")]
    # No other labelling of P1, P2, C1-C4 separates anything: every planted signal follows the PG label.
    chance = expected_chance_separations(FindingsConfig(), cstudy, expr, primary="PG")
    assert chance["PG vs Control"] == 0.0


def test_relabel_null_counts_other_labellings():
    # One gene high in C1 and C2 only: the true PG labelling finds nothing, 1 of the 14 other labellings of
    # 2 vs 4 samples ({C1, C2} as case) finds it once.
    samples = ["P1", "P2", "C1", "C2", "C3", "C4"]
    groups = {"PG": ["P1", "P2"], "Control": ["C1", "C2", "C3", "C4"]}
    counts = np.array([[1000, 0], [1000, 0], [900, 100], [900, 100], [1000, 0], [1000, 0]], dtype=float)
    e = make_expression(["Fibroblast"], samples, np.array(["HOUSE", "G"]), counts, np.full(6, 50.0),
                         np.where(counts > 0, 50.0, 0.0), "s", "l")
    assert relabel_null(e, FindingsConfig(), groups, ("PG", "Control"))[0] == pytest.approx(1 / 14)


def test_pipeline_pseudobulk_fallback(projects, study):
    rows, obs = [], []
    for g, ss in GROUPS.items():
        for s in ss:
            for lin in ("Fibroblast", "Myeloid"):
                obs.append({"batch": s, "broad_group": lin, "psbulk_cells": "50"})
                rows.append([900, 100 if (g == "PG" and lin == "Fibroblast") else 0])
    obs.append({"batch": "OTHER", "broad_group": "Fibroblast", "psbulk_cells": "5"})  # sample outside the groups
    rows.append([1, 1])
    a = ad.AnnData(X=np.array(rows, dtype=np.float32), obs=pd.DataFrame(obs, index=[str(i) for i in range(len(obs))]),
                   var=pd.DataFrame(index=["HOUSE", "GENEUP"]))
    plain = projects / "pb.h5ad"
    a.write_h5ad(plain)
    gz = projects / "p033" / "Results" / "communication_pipeline" / p033.PSEUDOBULK
    gz.write_bytes(gzip.compress(plain.read_bytes()))
    s = p033.load(projects, lab_tables=True)
    e = load_expression(s, FindingsConfig())
    assert e.label == "pipeline pseudobulk" and e.pct is None and "OTHER" not in e.samples
    assert e.cpm[e.lineages.index("Fibroblast"), e.samples.index("P1"), 1] == pytest.approx(1e5)
    k = _by_key(build_findings(s, FindingsConfig(), e))
    assert k["Fibroblast|GENEUP|PG|Control"].separated


def test_runner_findings_stage_is_genome_wide_by_default(projects_cells, tmp_path):
    cfg = RunConfig(study="p033", adapter="p033", disease="PG", projects_root=str(projects_cells), knowledge=False,
                    literature=False, scout=False, llm=LLMConfig(backend="replay"))
    run_dir = tmp_path / "runs" / "g"
    Runner(cfg, run_dir, llm_backend=Replay({})).run(stop_after="findings")
    out = json.loads((run_dir / "findings.json").read_text())
    kinds = {f["kind"] for f in out["findings"]}
    assert "expression" in kinds and not kinds & {"panel_expression", "signature"}
    assert out["chance_separations"]["PG vs Control"] == 0.0
    assert not (run_dir / "llm_calls.jsonl").exists()


def test_digest_caps_each_cell_type_and_contrast(cstudy, expr):
    fs = build_findings(cstudy, FindingsConfig(), expr, primary="PG")
    cfg = FindingsConfig(digest_quota={"expression": 3}, digest_per_stratum={"expression": 1})
    d = digest(fs, cfg)
    assert len(d) == 3
    assert len({(f.cell_type, f.case, f.reference) for f in d}) == 3  # one per stratum before any repeats
