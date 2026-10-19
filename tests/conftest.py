"""Synthetic p033-shaped study: small, fully known tables. No real data, network or GPU."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mechanist.adapters import p033
from mechanist.findings import FindingsConfig, build_findings

GROUPS = {"PG": ["P1", "P2"], "Sweet": ["S1", "S2", "S3"], "Control": ["C1", "C2", "C3", "C4"]}
LINEAGES = ["Fibroblast", "B cell", "Plasma cell", "Sweat gland", "Keratinocyte"]
GENES = ["CXCL13", "IGKC", "IGHM", "S100A8", "C3"]


def _cpm(group: str, sample: str, lin: str, gene: str) -> float:
    if gene == "CXCL13" and lin == "Fibroblast":
        return {"PG": 1000.0, "Sweet": 20.0, "Control": 5.0}[group]
    if gene == "IGKC":
        if lin == "Plasma cell":
            return 100000.0
        if lin == "Sweat gland" and group == "PG":
            return 5000.0  # spillover: 5% of the plasma-cell level
        return 1.0
    if gene == "IGHM":
        if sample == "C1":
            return 0.0  # undetected in every lineage of C1
        return 500.0 if (lin == "B cell" and group == "PG") else 5.0
    if gene == "S100A8" and lin == "Keratinocyte":
        return {"PG": 800.0, "Sweet": 700.0, "Control": 10.0}[group]
    if gene == "C3" and lin == "Fibroblast":
        return {"PG": 400.0, "Sweet": 30.0, "Control": 20.0}[group]
    return 2.0


def write_study(root: Path) -> Path:
    res = root / "p033" / "Results"
    (res / "pg_mechanism").mkdir(parents=True)
    (res / "communication_pipeline").mkdir(parents=True)
    rows = [{"group": g, "sample": s, "lineage": lin, "gene": gene, "cpm": _cpm(g, s, lin, gene),
             "pct_cells": 1.0, "cells": 10 if (lin == "Keratinocyte" and g == "PG") else 100}
            for g, ss in GROUPS.items() for s in ss for lin in LINEAGES for gene in GENES]
    pd.DataFrame(rows).to_csv(res / "pg_mechanism" / "expression_by_donor_and_type.csv", index=False)
    de = pd.DataFrame([
        {"broad_group": "Fibroblast", "gene": "CXCL13", "baseMean": 100, "log2FoldChange": 5.0, "lfcSE": 0.5,
         "stat": 10.0, "pvalue": 1e-12, "padj": 1e-10},
        {"broad_group": "Fibroblast", "gene": "C3", "baseMean": 80, "log2FoldChange": -2.0, "lfcSE": 0.4,
         "stat": -5.0, "pvalue": 1e-7, "padj": 1e-5},
        {"broad_group": "Fibroblast", "gene": "NOISE", "baseMean": 80, "log2FoldChange": 3.0, "lfcSE": 1,
         "stat": 1.0, "pvalue": 0.3, "padj": 0.5},
    ])
    de.to_csv(res / "communication_pipeline" / "pseudobulk_differential_expression.csv.gz", index=False)
    comp = pd.DataFrame([{"broad_group": "B cell", "case": "PG_Lesional", "reference": "Control_Skin", "n_case": 2,
                          "n_reference": 4, "mean_case": 0.4, "mean_reference": 0.02, "delta_mean": 0.38,
                          "test": "mannwhitney_unpaired", "pvalue": 0.13, "n_pairs": 0, "qvalue": 0.17}])
    comp.to_csv(res / "communication_pipeline" / "cell_composition_tests.csv.gz", index=False)
    comm = pd.DataFrame([
        {"condition": "PG_Lesional", "source": "Fibroblast", "target": "B cell", "ligand": "CXCL13",
         "receptor": "CXCR5", "n_samples": 2, "mean_magnitude": 0.3, "detection_fraction": 1.0, "ref_n_samples": 4,
         "ref_mean_magnitude": 0.0, "ref_detection_fraction": 0.0, "reference": "Control_Skin", "delta_mean": 0.3},
        {"condition": "PG_Lesional", "source": "Myeloid", "target": "Fibroblast", "ligand": "IL1B",
         "receptor": "IL1R1", "n_samples": 2, "mean_magnitude": 0.2, "detection_fraction": 1.0, "ref_n_samples": 4,
         "ref_mean_magnitude": 0.05, "ref_detection_fraction": 1.0, "reference": "Control_Skin", "delta_mean": 0.15},
    ])
    comm.to_csv(res / "communication_pipeline" / "communication_condition_summary.csv.gz", index=False)
    return root


def make_cells():
    """Synthetic single-cell data: PG has many activated B cells and CXCL13+/CXCL1+ fibroblasts (same cells)."""
    import numpy as np
    from scipy import sparse

    from mechanist.tools import CellData

    genes = ["CXCL13", "CXCL1", "MS4A1", "HLA-DRA", "COL1A1"]
    obs, rows = [], []
    for g, ss in GROUPS.items():
        for s in ss:
            nb = 60 if g == "PG" else 20
            for i in range(nb):
                obs.append({"sample": s, "group": g, "lineage": "B cell",
                            "state": "Activated HLA-II+ B cell" if g == "PG" else "Naive B cell"})
                rows.append([0, 0, 1, 1 if g == "PG" else 0, 0])
            for i in range(50):
                obs.append({"sample": s, "group": g, "lineage": "Fibroblast", "state": "CXCL14+ fibroblast"})
                pg = g == "PG"
                rows.append([1 if pg and i < 20 else 0, 1 if pg and i < 25 else 0, 0, 0, 1])
    X = sparse.csr_matrix(np.array(rows, dtype=float))
    return CellData.from_frame(pd.DataFrame(obs), X, genes)


@pytest.fixture
def cells():
    return make_cells()


@pytest.fixture
def projects(tmp_path) -> Path:
    return write_study(tmp_path / "Projects")


def write_cells_h5ad(root: Path) -> Path:
    """Synthetic reviewed object (raw counts) at the p033 CELLS path. Per sample: 40 fibroblasts, 30 B cells.

    GENEUP: fibroblasts, every PG sample above every Control. GENEDOWN: the reverse. MIXED: PG samples disagree.
    RARE: high CPM in PG fibroblasts but detected in one cell per sample (fails the detection filter).
    IGKC: B-cell gene, small PG fibroblast signal (spillover). HOUSE: library bulk."""
    import anndata as ad
    import numpy as np
    from scipy import sparse

    genes = ["HOUSE", "GENEUP", "GENEDOWN", "MIXED", "RARE", "IGKC"]
    raw_group = {"PG": "PG", "Sweet": "Sweet_Skin", "Control": "Control_Skin"}
    obs, rows = [], []
    for g, ss in GROUPS.items():
        for s in ss:
            for i in range(40):
                obs.append({"batch": s, "condition": raw_group[g], "final_annotation": "CXCL14+ fibroblast"})
                pg = g == "PG"
                rows.append([10, 4 if pg else (1 if g == "Sweet" else 0), 0 if pg else 3,
                             (3 if s == "P1" else 0) if pg else 1, 1 if pg and i == 0 else 0, 2 if pg else 0])
            for _ in range(30):
                obs.append({"batch": s, "condition": raw_group[g], "final_annotation": "Naive B cell"})
                rows.append([10, 1, 1, 0, 0, 50])  # background GENEUP/GENEDOWN: detected in every sample
            obs.append({"batch": s, "condition": raw_group[g], "final_annotation": "unreviewed cluster"})
            rows.append([10, 99, 0, 0, 0, 0])  # Unmapped lineage: ignored
    obs.append({"batch": "B1", "condition": "PBMC_Control", "final_annotation": "Naive B cell"})
    rows.append([10, 0, 0, 0, 0, 0])  # group outside the study: dropped
    a = ad.AnnData(X=sparse.csr_matrix(np.array(rows, dtype=np.float32)),
                   obs=pd.DataFrame(obs, index=[f"c{i}" for i in range(len(obs))]).astype("category"),
                   var=pd.DataFrame(index=genes))
    path = root / "p033" / p033.CELLS
    path.parent.mkdir(parents=True, exist_ok=True)
    a.write_h5ad(path)
    return path


@pytest.fixture
def projects_cells(tmp_path) -> Path:
    root = write_study(tmp_path / "Projects")
    write_cells_h5ad(root)
    return root


@pytest.fixture
def study(projects):
    return p033.load(projects, lab_tables=True)


@pytest.fixture
def findings(study):
    return build_findings(study, FindingsConfig(), lab_tables=True, primary="PG")


def fid(findings, kind, row_key):
    hits = [f.id for f in findings if f.kind == kind and f.row_key == row_key]
    assert hits, f"no {kind} finding {row_key}"
    return hits[0]
