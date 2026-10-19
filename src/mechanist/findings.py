"""Molecular analyst: deterministic conversion of study tables into Finding records. No LLM.

Standard tables keep their own statistics (pyDESeq2 padj, Mann-Whitney p/q). Per-sample expression gets an honest
small-sample statistic instead of a test: whether every case sample lies beyond every reference sample, and the exact
chance of that separation if group labels were exchangeable (1 / C(n_case + n_ref, n_case)).

Per-sample expression is scanned genome-wide (every gene, every lineage, every group pair) from raw counts: the
reviewed single-cell object when the adapter provides one, else the pipeline's pseudobulk counts. Hand-made tables
from an earlier investigation (`panel`, `signatures`) are used only when `lab_tables` is on.
"""

from __future__ import annotations

import collections
import gzip
import io
import math
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel

from mechanist.adapters.base import LoadedStudy
from mechanist.records import Finding
from mechanist.verify import cell_family


class FindingsConfig(BaseModel):
    de_padj: float = 0.05
    de_min_abs_lfc: float = 1.0
    de_min_base_mean: float = 10.0
    de_top_per_cell: int = 15
    gsea_collections: list[str] = ["hallmark"]
    gsea_padj: float = 0.05
    gsea_top_per_cell: int = 4
    progeny_min_abs_delta: float = 0.5
    comm_top_up: int = 40
    comm_top_down: int = 15
    panel_min_abs_log2: float = 1.0
    panel_min_cpm: float = 10.0
    # Group pairs for per-sample comparisons. None: every group vs the reference group, then every pair of other
    # groups (the study's primary disease first).
    pairs: list[tuple[str, str]] | None = None
    # Genome-wide per-sample scan. auto: reviewed single-cell object (raw counts) if present, else pipeline pseudobulk.
    expression_source: Literal["auto", "cells", "pseudobulk", "none"] = "auto"
    expr_min_cpm: float = 10.0  # mean CPM in the higher group
    expr_min_pct: float = 5.0  # mean % of cells detecting the gene in the higher group (cell-level source only)
    expr_min_abs_log2: float = 1.0
    expr_min_gap_log2: float = 0.5  # closest case vs reference sample must differ by this much (log2, pseudocount 1)
    ambient_min_ratio: float = 3.0
    spillover_max_ratio: float = 0.3  # flag when, on average over samples, a lineage has <30% of the gene's level
    # in the gene's top lineage
    flagged_priority_factor: float = 0.25  # flagged findings rank lower in the LLM digest
    min_cells: int = 30  # per sample and lineage; fewer cells -> few_cells caveat
    # Per-sample kinds whose digest share is capped per (cell type, contrast), so one lineage cannot fill the digest;
    # spare places go to the next findings by priority.
    digest_per_stratum: dict[str, int] = {"expression": 8, "target_score": 4}
    # A gene separating the same way in at least this many lineages (and this share of scanned lineages) is flagged
    # `broadcast` everywhere but its top lineage.
    broadcast_min_lineages: int = 3
    broadcast_fraction: float = 0.5
    # Per-sample target-gene scores (CollecTRI regulons, PROGENy footprints; needs the knowledge base).
    target_scores: bool = True
    score_min_targets: int = 10  # targets expressed in the lineage
    score_min_delta: float = 1.0  # difference in mean score (score units ~ standard normal per sample under no signal)
    score_min_gap: float = 0.25  # closest case vs reference sample
    score_driver_n: int = 20  # top driver genes compared for redundancy
    score_redundant_share: float = 0.5  # share of driver change already carried by stronger sets' drivers
    # How many findings of each kind the LLM sees (the verifier always uses all findings).
    digest_quota: dict[str, int] = {"expression": 60, "target_score": 25, "panel_expression": 70, "de": 40, "communication": 25, "composition": 10,
                                    "progeny": 10, "pathway": 15, "signature": 10}


def p_separation(n_case: int, n_ref: int) -> float:
    return 1.0 / math.comb(n_case + n_ref, n_case)


def _f(x) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


class _Ambient:
    """Flags case samples whose signal is within `min_ratio` of what ambient RNA alone would give."""

    def __init__(self, df: pd.DataFrame | None, min_ratio: float):
        self.low: dict[tuple[str, str], list[str]] = {}
        if df is None:
            return
        bad = df[df["observed_over_ambient"] < min_ratio]
        for (lin, gene), sub in bad.groupby(["lineage", "gene"]):
            self.low[(lin, gene)] = sorted(sub["sample"])
        self.min_ratio = min_ratio

    def caveat(self, cell: str, genes: list[str], samples: list[str] | None) -> list[str]:
        out = []
        for gene in genes:
            hit = self.low.get((cell, gene), [])
            if samples is not None:
                hit = [s for s in hit if s in samples]
            if hit:
                out.append(f"ambient: {gene} in {cell} is under {self.min_ratio:g}x the ambient-expected level in "
                           f"{', '.join(hit)}")
        return out


def resolve_pairs(cfg: FindingsConfig, groups: dict[str, list[str]], reference: str,
                  primary: str | None = None) -> list[tuple[str, str]]:
    if cfg.pairs is not None:
        return [(c, r) for c, r in cfg.pairs if c in groups and r in groups]
    others = sorted((g for g in groups if g != reference), key=lambda g: (g != primary, g))
    return [(g, reference) for g in others if reference in groups] + list(combinations(others, 2))


def build_findings(study: LoadedStudy, cfg: FindingsConfig, expression: Expression | None = None,
                   lab_tables: bool = False, primary: str | None = None, kb=None) -> list[Finding]:
    """`expression`: genome-wide per-sample pseudobulk (see load_expression). `lab_tables`: also use hand-made
    per-sample tables from an earlier investigation (circular with its hypotheses). `kb`: KnowledgeBase for gene-role
    labels and target-gene scores (skipped without it)."""
    t = study.tables
    ambient = _Ambient(t["ambient"].df if lab_tables and "ambient" in t else None, cfg.ambient_min_ratio)
    pairs = resolve_pairs(cfg, study.dataset.groups, study.dataset.reference_group, primary)
    out: list[Finding] = []
    if study.standard_contrast:
        out += _standard(study, cfg, ambient)
    if expression is not None:
        out += _expression(study, cfg, expression, pairs, ambient, GeneRoles(kb))
        if kb is not None and cfg.target_scores:
            ts = target_scores(expression, gene_sets(kb), cfg.score_min_targets)
            out += _target_findings(study, cfg, expression, ts, pairs)
    if lab_tables and "panel" in t:
        out += _panel(study, cfg, ambient, pairs)
    if lab_tables and "signatures" in t:
        out += _signatures(study, cfg, pairs)
    order = list(cfg.digest_quota)
    out.sort(key=lambda f: (order.index(f.kind) if f.kind in order else 99, -f.priority, f.row_key))
    for i, f in enumerate(out, 1):
        f.id = f"F{i:04d}"
    return out


def _base(study: LoadedStudy, table: str, **kw) -> dict:
    return {"id": "", "study": study.dataset.study, "source_file": str(study.tables[table].path),
            "review_status": study.dataset.review_status, **kw}


def _standard(study: LoadedStudy, cfg: FindingsConfig, ambient: _Ambient) -> list[Finding]:
    t = study.tables
    case, ref = study.standard_contrast
    n_case, n_ref = study.standard_n or (0, 0)
    small = [f"small_n: {case} vs {ref} has {n_case} vs {n_ref} samples"] if min(n_case, n_ref) < 3 else []
    out: list[Finding] = []

    if "de" in t:
        df = t["de"].df.dropna(subset=["padj"])
        df = df[(df.padj < cfg.de_padj) & (df.log2FoldChange.abs() >= cfg.de_min_abs_lfc)
                & (df.baseMean >= cfg.de_min_base_mean)]
        df = df.assign(_s=df.stat.abs()).sort_values("_s", ascending=False).groupby("broad_group").head(cfg.de_top_per_cell)
        for r in df.itertuples():
            out.append(Finding(**_base(study, "de", kind="de", row_key=f"{r.broad_group}|{r.gene}",
                       case=case, reference=ref, cell_type=r.broad_group, feature=r.gene, genes=[r.gene],
                       direction="up" if r.log2FoldChange > 0 else "down", effect=float(r.log2FoldChange),
                       effect_label="log2FC (pyDESeq2, pseudobulk per sample)", p=_f(r.pvalue), q=_f(r.padj),
                       n_case=n_case, n_reference=n_ref,
                       caveats=small + (amb := ambient.caveat(r.broad_group, [r.gene], None)),
                       priority=float(-math.log10(max(r.padj, 1e-300)))
                       * (cfg.flagged_priority_factor if amb else 1))))

    if "gsea" in t:
        df = t["gsea"].df
        df = df[df.collection.isin(cfg.gsea_collections) & (df.padj < cfg.gsea_padj)]
        df = df.assign(_s=df.nes.abs()).sort_values("_s", ascending=False).groupby("broad_group").head(cfg.gsea_top_per_cell)
        for r in df.itertuples():
            out.append(Finding(**_base(study, "gsea", kind="pathway", row_key=f"{r.broad_group}|{r.pathway}",
                       case=case, reference=ref, cell_type=r.broad_group, feature=r.pathway,
                       direction="up" if r.nes > 0 else "down", effect=float(r.nes),
                       effect_label="GSEA NES on DESeq2 stat", q=_f(r.padj), n_case=n_case, n_reference=n_ref,
                       caveats=small, priority=abs(float(r.nes)))))

    if "progeny" in t:
        df = t["progeny"].df
        df = df[df.delta_mean.abs() >= cfg.progeny_min_abs_delta]
        for r in df.itertuples():
            out.append(Finding(**_base(study, "progeny", kind="progeny", row_key=f"{r.broad_group}|{r.pathway}",
                       case=r.case, reference=r.reference, cell_type=r.broad_group,
                       feature=f"PROGENy {r.pathway}", direction="up" if r.delta_mean > 0 else "down",
                       effect=float(r.delta_mean), effect_label="difference in mean PROGENy activity per sample",
                       p=_f(r.pvalue), q=_f(r.qvalue), n_case=int(r.n_case), n_reference=int(r.n_reference),
                       caveats=small, priority=abs(float(r.delta_mean)))))

    if "composition" in t:
        for r in t["composition"].df.itertuples():
            out.append(Finding(**_base(study, "composition", kind="composition", row_key=r.broad_group,
                       case=r.case, reference=r.reference, cell_type=r.broad_group, feature="proportion",
                       direction="up" if r.delta_mean > 0 else "down", effect=float(r.delta_mean),
                       effect_label="difference in mean proportion of cells per sample",
                       p=_f(r.pvalue), q=_f(r.qvalue), n_case=int(r.n_case), n_reference=int(r.n_reference),
                       caveats=small, priority=abs(float(r.delta_mean)))))

    if "communication" in t:
        df = t["communication"].df
        min_q = None
        if "communication_tests" in t:
            min_q = _f(t["communication_tests"].df.qvalue.min())
        note = [f"not_significant: no communication change reached significance (lowest q = {min_q:.2g})"] \
            if min_q is not None and min_q >= 0.1 else []
        up = df[(df.delta_mean > 0) & (df.detection_fraction >= 1)].nlargest(cfg.comm_top_up, "delta_mean")
        down = df[(df.delta_mean < 0) & (df.ref_detection_fraction >= 0.75)].nsmallest(cfg.comm_top_down, "delta_mean")
        for r in pd.concat([up, down]).itertuples():
            genes = [r.ligand, *str(r.receptor).split("_")]
            out.append(Finding(**_base(study, "communication", kind="communication",
                       row_key=f"{r.source}|{r.target}|{r.ligand}|{r.receptor}", case=r.condition,
                       reference=r.reference, cell_type=r.source, target_cell_type=r.target,
                       feature=f"{r.ligand}->{r.receptor}", genes=genes,
                       direction="up" if r.delta_mean > 0 else "down", effect=float(r.delta_mean),
                       effect_label="difference in mean LIANA (cellchat) magnitude per sample",
                       n_case=int(r.n_samples), n_reference=int(r.ref_n_samples),
                       caveats=small + note + ambient.caveat(r.source, [r.ligand], None),
                       priority=abs(float(r.delta_mean)))))
    return out


def _separation(case_vals: np.ndarray, ref_vals: np.ndarray) -> tuple[bool, str | None]:
    if case_vals.min() > ref_vals.max():
        return True, "up"
    if case_vals.max() < ref_vals.min():
        return True, "down"
    return False, None


def _spillover(df: pd.DataFrame, max_ratio: float):
    """Gene level in a lineage relative to the gene's highest-expressing lineage, per sample.

    Immunoglobulin transcripts in sweat-gland cells, or keratinocyte genes in B cells, usually mean ambient RNA or
    doublets rather than expression by that lineage."""
    cpm = df.pivot_table(index=["sample", "gene"], columns="lineage", values="cpm")
    top_lin = cpm.idxmax(axis=1)
    rel = cpm.div(cpm.max(axis=1).replace(0, np.nan), axis=0)

    def check(lin: str, gene: str, samples: list[str]) -> list[str]:
        rows = [(s, gene) for s in samples if (s, gene) in rel.index]
        if not rows or lin not in rel.columns:
            return []
        r = rel.loc[rows, lin].fillna(1.0)
        top = top_lin.loc[rows].mode().iloc[0]
        if r.mean() < max_ratio and top != lin:
            return [(f"spillover: {gene} in {lin} averages {r.mean():.0%} of its level in {top} in the same samples; "
                    f"the {lin} signal may be ambient RNA or doublets")]
        return []
    return check


def _undetected(df: pd.DataFrame):
    """Genes with zero counts in every lineage of a sample: often a gene-annotation or chemistry difference between
    datasets (e.g. public controls processed elsewhere), not biology."""
    tot = df.groupby(["sample", "gene"]).cpm.sum()
    absent = set(tot[tot == 0].index)

    def check(gene: str, samples: list[str]) -> list[str]:
        miss = [s for s in samples if (s, gene) in absent]
        if miss:
            return [(f"detection: {gene} is not detected in any cell of {', '.join(miss)}; a gene-annotation or "
                    f"chemistry difference between datasets could explain the gap")]
        return []
    return check


def _panel(study: LoadedStudy, cfg: FindingsConfig, ambient: _Ambient, pairs: list[tuple[str, str]]) -> list[Finding]:
    df = study.tables["panel"].df
    spill = _spillover(df, cfg.spillover_max_ratio)
    undetected = _undetected(df)
    ncells = df[["lineage", "sample", "cells"]].drop_duplicates().set_index(["lineage", "sample"])["cells"]
    wide = df.pivot_table(index=["lineage", "gene"], columns="sample", values="cpm")
    groups = study.dataset.groups
    out: list[Finding] = []
    for case, ref in pairs:
        for (lin, gene), row in wide.iterrows():
            cs = [s for s in groups[case] if s in row and not pd.isna(row[s])]
            rs = [s for s in groups[ref] if s in row and not pd.isna(row[s])]
            if len(cs) < 2 or len(rs) < 2:
                continue
            cv, rv = row[cs].to_numpy(float), row[rs].to_numpy(float)
            if max(cv.mean(), rv.mean()) < cfg.panel_min_cpm:
                continue
            lfc = math.log2((cv.mean() + 1) / (rv.mean() + 1))
            sep, d = _separation(cv, rv)
            if not sep or abs(lfc) < cfg.panel_min_abs_log2 or (d == "up") != (lfc > 0):
                continue
            ps = p_separation(len(cs), len(rs))
            cav = [f"small_n: {case} vs {ref} has {len(cs)} vs {len(rs)} samples"] if min(len(cs), len(rs)) < 3 else []
            cav += ambient.caveat(lin, [gene], cs) + spill(lin, gene, cs if d == "up" else rs)
            cav += undetected(gene, rs if d == "up" else cs)
            few = {s: int(ncells.get((lin, s), 0)) for s in cs + rs if ncells.get((lin, s), 0) < cfg.min_cells}
            if few:
                cav.append(f"few_cells: {lin} has fewer than {cfg.min_cells} cells in "
                           + ", ".join(f"{s} ({n})" for s, n in few.items()))
            flagged = any(c.startswith(("ambient", "spillover", "detection", "few_cells")) for c in cav)
            out.append(Finding(**_base(study, "panel", kind="panel_expression", row_key=f"{lin}|{gene}|{case}|{ref}",
                       case=case, reference=ref, cell_type=lin, feature=gene, genes=[gene], direction=d,
                       effect=lfc, effect_label="log2 ratio of mean CPM per sample (pseudocount 1)",
                       n_case=len(cs), n_reference=len(rs),
                       per_sample={s: round(float(row[s]), 2) for s in cs + rs}, separated=True, p_separation=ps,
                       caveats=cav + ["panel: curated gene panel from the earlier PG investigation"],
                       priority=abs(lfc) * (1 - ps) * (cfg.flagged_priority_factor if flagged else 1))))
    return out


def _signatures(study: LoadedStudy, cfg: FindingsConfig, pairs: list[tuple[str, str]]) -> list[Finding]:
    df = study.tables["signatures"].df
    sig_cols = [c for c in df.columns if c not in {"group", "sample", "state", "cells"}]
    out: list[Finding] = []
    for case, ref in pairs:
        for state, sub in df.groupby("state"):
            for col in sig_cols:
                cv = sub.loc[sub.group == case, col].dropna()
                rv = sub.loc[sub.group == ref, col].dropna()
                if len(cv) < 2 or len(rv) < 2:
                    continue
                sep, d = _separation(cv.to_numpy(float), rv.to_numpy(float))
                if not sep:
                    continue
                cs = sub.loc[cv.index, "sample"].tolist()
                rs = sub.loc[rv.index, "sample"].tolist()
                ps = p_separation(len(cv), len(rv))
                cav = [f"small_n: {case} vs {ref} has {len(cv)} vs {len(rv)} samples"] if min(len(cv), len(rv)) < 3 else []
                out.append(Finding(**_base(study, "signatures", kind="signature", row_key=f"{state}|{col}|{case}|{ref}",
                           case=case, reference=ref, cell_type=f"Fibroblast ({state})", feature=f"signature: {col}",
                           direction=d, effect=float(cv.mean() - rv.mean()),
                           effect_label="difference in mean signature score (source units)",
                           n_case=len(cv), n_reference=len(rv),
                           per_sample=dict(zip(cs + rs, [round(float(x), 3) for x in [*cv, *rv]])),
                           separated=True, p_separation=ps,
                           caveats=cav + ["panel: signature scores built during the earlier PG investigation"],
                           priority=1 - ps)))
    return out


def expected_chance_separations(findings_cfg: FindingsConfig, study: LoadedStudy,
                                expression: Expression | None = None, lab_tables: bool = False,
                                primary: str | None = None) -> dict[str, float]:
    """How many per-sample findings each contrast would give by chance alone (both directions).

    Genome-wide scan: the same scan, with the same filters, repeated under every other assignment of the pooled
    samples to case and reference (exact relabelling null; random relabellings when there are too many). Lab panel:
    features x exact separation chance (an upper bound, as it ignores the fold-change filter)."""
    g = study.dataset.groups
    pairs = resolve_pairs(findings_cfg, g, study.dataset.reference_group, primary)
    expected = {p: 0.0 for p in pairs}
    if expression is not None:
        for p in pairs:
            expected[p] += float(relabel_null(expression, findings_cfg, g, p).sum())
    if lab_tables and "panel" in study.tables:
        n_features = study.tables["panel"].df[["lineage", "gene"]].drop_duplicates().shape[0]
        for c, r in pairs:
            expected[(c, r)] += 2 * n_features * p_separation(len(g[c]), len(g[r]))
    return {f"{c} vs {r}": round(v, 1) for (c, r), v in expected.items() if v or expression is not None}


# ---- genome-wide per-sample expression --------------------------------------------------------------------------
@dataclass
class Expression:
    """Pseudobulk per sample and lineage over all genes. cpm/pct: (lineage, sample, gene); NaN where a sample has no
    cells of that lineage. pct is None when the source has no cell-level detection (pipeline pseudobulk)."""

    lineages: list[str]
    samples: list[str]
    genes: np.ndarray
    cpm: np.ndarray
    pct: np.ndarray | None
    cells: np.ndarray  # (lineage, sample)
    source: str
    label: str


def expression_from_counts(obs: pd.DataFrame, X, genes, source: str, label: str) -> Expression | None:
    """Aggregate cell-level raw counts (obs: sample, lineage) into per-sample, per-lineage CPM and detection %.
    Returns None when X does not hold raw counts (CPM from normalised values would be meaningless)."""
    from scipy import sparse

    data = X.data[:200_000] if sparse.issparse(X) else np.asarray(X).ravel()[:200_000]
    if len(data) and not np.allclose(data, np.round(data)):
        return None
    keep = (obs["lineage"].astype(str) != "Unmapped").to_numpy()
    lins = sorted(obs.loc[keep, "lineage"].astype(str).unique())
    samples = sorted(obs.loc[keep, "sample"].astype(str).unique())
    li = obs["lineage"].astype(str).map({x: i for i, x in enumerate(lins)}).to_numpy()
    si = obs["sample"].astype(str).map({x: i for i, x in enumerate(samples)}).to_numpy()
    rows = np.flatnonzero(keep)
    code = (li[rows] * len(samples) + si[rows]).astype(int)
    k = len(lins) * len(samples)
    M = sparse.csr_matrix((np.ones(len(rows)), (code, rows)), shape=(k, X.shape[0]))
    Xs = sparse.csr_matrix(X) if not sparse.issparse(X) else X
    counts = np.asarray((M @ Xs).toarray(), dtype=float)
    det = Xs.copy()
    det.data = (det.data > 0).astype(float)
    detected = np.asarray((M @ det).toarray(), dtype=float)
    ncell = np.bincount(code, minlength=k).astype(float)
    return make_expression(lins, samples, np.asarray(genes, dtype=str), counts, ncell, detected, source, label)


def make_expression(lins, samples, genes, counts, ncell, detected, source, label) -> Expression:
    shape = (len(lins), len(samples), len(genes))
    total = counts.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        cpm = np.where((ncell[:, None] > 0) & (total > 0), counts / total * 1e6, np.nan)
        pct = None if detected is None else np.where(ncell[:, None] > 0, detected / ncell[:, None] * 100, np.nan)
    return Expression(lineages=list(lins), samples=list(samples), genes=genes, cpm=cpm.reshape(shape),
                      pct=None if pct is None else pct.reshape(shape),
                      cells=ncell.reshape(shape[:2]).astype(int), source=source, label=label)


def expression_from_pseudobulk(path: Path, sample_col: str, lineage_col: str, cells_col: str | None,
                               samples: set[str]) -> Expression:
    """Pipeline pseudobulk counts (AnnData, rows = sample x lineage, raw summed counts; .h5ad or .h5ad.gz)."""
    import anndata as ad

    raw = Path(path).read_bytes()
    a = ad.read_h5ad(io.BytesIO(gzip.decompress(raw) if str(path).endswith(".gz") else raw))
    obs = a.obs.astype({sample_col: str, lineage_col: str})
    keep = obs[sample_col].isin(samples).to_numpy()
    obs = obs[keep]
    X = a.X[keep]
    X = np.asarray(X.toarray() if hasattr(X, "toarray") else X, dtype=float)
    lins, ss = sorted(obs[lineage_col].unique()), sorted(obs[sample_col].unique())
    counts = np.zeros((len(lins) * len(ss), X.shape[1]))
    ncell = np.zeros(len(lins) * len(ss))
    for row, (lin, s) in enumerate(zip(obs[lineage_col], obs[sample_col])):
        k = lins.index(lin) * len(ss) + ss.index(s)
        counts[k] += X[row]
        ncell[k] += float(obs[cells_col].iloc[row]) if cells_col else 1.0
    return make_expression(lins, ss, np.asarray(a.var_names, dtype=str), counts, ncell, None, str(path),
                            "pipeline pseudobulk")


def load_expression(study: LoadedStudy, cfg: FindingsConfig, cells=None) -> Expression | None:
    """cells: a CellData already in memory (or a zero-argument callable returning one), reused by the tools."""
    src = cfg.expression_source
    if src == "none":
        return None
    if callable(cells):
        cells = cells() if src in ("auto", "cells") else None
    if cells is None and src in ("auto", "cells") and study.cells is not None:
        from mechanist.tools import CellData

        cells = CellData(study.cells)
    if cells is not None and src in ("auto", "cells"):
        e = expression_from_counts(cells.obs, cells.X, cells.gene_names, cells.path, "reviewed single-cell object")
        if e is not None or src == "cells":
            return e
    pb = study.pseudobulk
    if src in ("auto", "pseudobulk") and pb is not None and Path(pb.path).exists():
        samples = {s for ss in study.dataset.groups.values() for s in ss}
        return expression_from_pseudobulk(pb.path, pb.sample_col, pb.lineage_col, pb.cells_col, samples)
    return None


def _sample_idx(e: Expression, groups: dict[str, list[str]], g: str) -> list[int]:
    return [e.samples.index(s) for s in groups.get(g, []) if s in e.samples]


def _scan(e: Expression, cfg: FindingsConfig, ci: list[int], ri: list[int], li: int):
    """Vectorised over genes for one lineage and pair. Returns (case idx, ref idx, tested mask, hit mask, lfc, gap)."""
    ci = [i for i in ci if not np.isnan(e.cpm[li, i, 0])]
    ri = [i for i in ri if not np.isnan(e.cpm[li, i, 0])]
    G = e.cpm.shape[2]
    if len(ci) < 2 or len(ri) < 2:
        z = np.zeros(G, bool)
        return ci, ri, z, z, np.zeros(G), np.zeros(G)
    C, R = e.cpm[li, ci, :], e.cpm[li, ri, :]
    cm, rm = C.mean(0), R.mean(0)
    tested = np.maximum(cm, rm) >= cfg.expr_min_cpm
    if e.pct is not None:
        hi_pct = np.where(cm >= rm, e.pct[li, ci, :].mean(0), e.pct[li, ri, :].mean(0))
        tested &= hi_pct >= cfg.expr_min_pct
    lfc = np.log2((cm + 1) / (rm + 1))
    up_gap = np.log2((C.min(0) + 1) / (R.max(0) + 1))
    down_gap = np.log2((R.min(0) + 1) / (C.max(0) + 1))
    gap = np.where(lfc > 0, up_gap, down_gap)
    hit = tested & (np.abs(lfc) >= cfg.expr_min_abs_log2) & (gap >= cfg.expr_min_gap_log2)
    return ci, ri, tested, hit, lfc, gap


def _labellings(pool: list[int], nc: int, max_labellings: int) -> list[tuple[int, ...]]:
    """Every other assignment of `nc` of the pooled samples to the case group (random ones when there are too many)."""
    actual = frozenset(pool[:nc])
    if math.comb(len(pool), nc) <= max_labellings + 1:
        return [c for c in combinations(pool, nc) if frozenset(c) != actual]
    rng = np.random.default_rng(0)
    out: list[tuple[int, ...]] = []
    while len(out) < max_labellings:
        c = tuple(sorted(rng.choice(pool, nc, replace=False).tolist()))
        if frozenset(c) != actual:
            out.append(c)
    return out


def _relabel(count, n_lineages: int, ci: list[int], ri: list[int], max_labellings: int) -> np.ndarray:
    """Per lineage: mean of count(case idx, ref idx) over the other labellings of the same samples."""
    pool, nc = ci + ri, len(ci)
    if nc < 2 or len(ri) < 2:
        return np.zeros(n_lineages)
    counts = [count(list(c), [i for i in pool if i not in c]) for c in _labellings(pool, nc, max_labellings)]
    return np.mean(counts, axis=0)


def relabel_null(e: Expression, cfg: FindingsConfig, groups, pair, max_labellings: int = 200) -> np.ndarray:
    """Per lineage: mean number of expression findings over the other case/reference labellings of the same samples."""
    def count(ci, ri):
        return np.array([int(_scan(e, cfg, ci, ri, li)[3].sum()) for li in range(len(e.lineages))])
    return _relabel(count, len(e.lineages), _sample_idx(e, groups, pair[0]), _sample_idx(e, groups, pair[1]),
                    max_labellings)


# ---- gene classes and roles ---------------------------------------------------------------------------------------
SEX_GENES = {"XIST", "TSIX", "RPS4Y1", "DDX3Y", "UTY", "KDM5D", "EIF1AY", "USP9Y", "ZFY", "TXLNGY", "NLGN4Y", "PRKY"}
# Immediate-early / heat-shock genes induced by tissue dissociation and handling (van den Brink et al. 2017 core set).
DISSOCIATION_GENES = {"FOS", "FOSB", "JUN", "JUNB", "JUND", "EGR1", "EGR2", "EGR3", "IER2", "IER3", "ATF3", "HSPA1A",
                      "HSPA1B", "HSPA6", "HSPA8", "DNAJB1", "HSPH1", "HSP90AA1", "DUSP1", "NR4A1", "NR4A2", "KLF2",
                      "KLF4", "KLF6", "ZFP36", "PPP1R15A", "CYR61", "CCN1", "SOCS3", "BTG2", "GADD45B"}
_TECHNICAL = [
    ("mitochondrial", re.compile(r"^MT-")),
    ("ribosomal-protein", re.compile(r"^(RPL|RPS|MRPL|MRPS)\d")),
    ("haemoglobin", re.compile(r"^HB[ABDEGMQZ]\d?$")),
    ("unannotated or antisense", re.compile(r"^(ENSG\d|LINC\d|LOC\d)|^A[CLP]\d{6}\.\d+$|^Z\d{5}\.\d+$|-AS\d$")),
]
_IG = re.compile(r"^(IG[HKL][VCJDAGEM]|JCHAIN$)")


def technical_class(gene: str, lineage: str | None = None) -> str | None:
    """Gene classes whose per-sample differences usually reflect technique, sex or contamination, not a program.
    Immunoglobulin genes count only outside B-lineage cells (where they are ambient RNA or doublets)."""
    for name, rx in _TECHNICAL:
        if rx.search(gene):
            return name
    if gene in SEX_GENES:
        return "sex-chromosome (tracks donor sex)"
    if gene in DISSOCIATION_GENES:
        return "dissociation/immediate-early (tracks tissue handling)"
    if _IG.search(gene) and (lineage is None or cell_family(lineage) != "b"):
        return "immunoglobulin (outside B-lineage cells)"
    return None


class GeneRoles:
    """Curated roles from the offline knowledge base (other systems, not this study): ligand, receptor, TF, and the
    PROGENy pathways a gene responds to."""

    def __init__(self, kb=None, progeny_top: int = 100):
        self.ligand: set[str] = set()
        self.receptor: set[str] = set()
        self.tf: set[str] = set()
        self.pathways: dict[str, list[str]] = {}
        if kb is None:
            return
        lr = kb.table("lr_consensus.csv")
        self.ligand = set(lr.ligand.astype(str))
        self.receptor = {x for r in lr.receptor.astype(str) for x in r.split("_")}
        if (kb.dir / "membrane_receptors.csv").exists():
            self.receptor |= set(kb.table("membrane_receptors.csv").gene.astype(str))
        self.tf = set(kb.table("collectri.csv").source.astype(str))
        pr = kb.table("progeny.csv")
        top = pr.assign(a=pr.weight.abs()).sort_values("a", ascending=False).groupby("source").head(progeny_top)
        for src, tgt in zip(top.source, top.target):
            self.pathways.setdefault(str(tgt), []).append(str(src))

    def label(self, gene: str) -> str | None:
        r = [x for x, s in (("ligand", self.ligand), ("receptor", self.receptor), ("TF", self.tf)) if gene in s]
        if gene in self.pathways:
            r.append(f"{'/'.join(sorted(self.pathways[gene]))}-responsive (PROGENy)")
        return f"role: {', '.join(r)}" if r else None


# ---- per-sample expression findings --------------------------------------------------------------------------------
FLAG_PREFIXES = ("ambient", "spillover", "detection", "few_cells", "technical", "broadcast")


def is_flagged(f: Finding) -> bool:
    return any(c.startswith(FLAG_PREFIXES) for c in f.caveats)


def _pattern(hits: dict, lin: str, feat: str, case: str, ref: str, d: str, groups: list[str], reference: str) -> str:
    """How this separation sits among the other contrasts of the same lineage and feature."""
    opp = "down" if d == "up" else "up"

    def sep(a, b, dd):
        return hits.get((lin, feat, a, b)) == dd or hits.get((lin, feat, b, a)) == ("down" if dd == "up" else "up")
    others = [g for g in groups if g not in (case, ref)]
    if ref == reference:
        beyond = [o for o in others if sep(case, o, d)]
        shared = [o for o in others if sep(o, ref, d)]
        if others and len(beyond) == len(others):
            return f"pattern: {case}-specific (also beyond {', '.join(others)})"
        if shared:
            return f"pattern: shared with {', '.join(shared)} (also beyond {ref})"
        return f"pattern: {case} vs {ref} only" + (f" ({', '.join(others)} overlap one or both)" if others else "")
    parts = []
    if reference in groups and sep(case, reference, d):
        parts.append(f"{case} also beyond {reference}")
    if reference in groups and sep(ref, reference, opp):
        parts.append(f"{ref} also beyond {reference} the other way")
    return "pattern: " + ("; ".join(parts) if parts else f"{case} vs {ref} only")


def _expression(study: LoadedStudy, cfg: FindingsConfig, e: Expression, pairs: list[tuple[str, str]],
                ambient: _Ambient, roles: GeneRoles) -> list[Finding]:
    groups = study.dataset.groups
    # Spillover: level in a lineage relative to the gene's top lineage in the same sample.
    filled = np.where(np.isnan(e.cpm), -np.inf, e.cpm)
    top = filled.max(axis=0)  # (sample, gene)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.where(top > 0, e.cpm / top, np.nan)
    top_lin = filled.argmax(axis=0)
    absent = np.nansum(e.cpm, axis=0) == 0  # (sample, gene): no counts in any lineage
    raw, hits, scanned = [], {}, {}
    for case, ref in pairs:
        cidx, ridx = _sample_idx(e, groups, case), _sample_idx(e, groups, ref)
        null = relabel_null(e, cfg, groups, (case, ref))
        scanned[(case, ref)] = 0
        for li, lin in enumerate(e.lineages):
            ci, ri, tested, hit, lfc, gap = _scan(e, cfg, cidx, ridx, li)
            scanned[(case, ref)] += bool(tested.any())
            n_hit = int(hit.sum())
            for gi in np.flatnonzero(hit):
                d = "up" if lfc[gi] > 0 else "down"
                hits[(lin, str(e.genes[gi]), case, ref)] = d
                raw.append((case, ref, li, gi, d, ci, ri, float(lfc[gi]), float(gap[gi]), int(tested.sum()), n_hit,
                            float(null[li])))
    # Broadcast: the same gene separating the same way in many lineages is a sample-wide signal (ambient RNA,
    # doublets, sex, batch) everywhere except, possibly, the lineage that expresses it most.
    spread: dict[tuple, list[int]] = {}
    for case, ref, li, gi, d, *_ in raw:
        spread.setdefault((case, ref, gi, d), []).append(li)
    gnames = [g for g, _ in groups.items()]
    out: list[Finding] = []
    for case, ref, li, gi, d, ci, ri, lfc, gap, n_tested, n_hit, null in raw:
        lin, gene = e.lineages[li], str(e.genes[gi])
        cs, rs = [e.samples[i] for i in ci], [e.samples[i] for i in ri]
        ps = p_separation(len(ci), len(ri))
        excess = max(0.0, 1 - null / n_hit)  # share of this lineage's separations relabelling does not reproduce
        cav = [f"small_n: {case} vs {ref} has {len(cs)} vs {len(rs)} samples"] if min(len(cs), len(rs)) < 3 else []
        cav.append(f"genome_wide: one of {n_hit} {lin} genes (of {n_tested} scanned) separating {case} from {ref}; "
                   f"relabelling the same samples gives {null:.0f} on average")
        few = {e.samples[i]: int(e.cells[li, i]) for i in ci + ri if e.cells[li, i] < cfg.min_cells}
        if few:
            cav.append(f"few_cells: {lin} has fewer than {cfg.min_cells} cells in "
                       + ", ".join(f"{s} ({n})" for s, n in few.items()))
        high, low = (ci, ri) if d == "up" else (ri, ci)
        r = np.nan_to_num(rel[li, high, gi], nan=1.0)
        tl = np.bincount(top_lin[high, gi], minlength=len(e.lineages)).argmax()
        if r.mean() < cfg.spillover_max_ratio and tl != li:
            cav.append(f"spillover: {gene} in {lin} averages {r.mean():.0%} of its level in "
                       f"{e.lineages[tl]} in the same samples; the {lin} signal may be ambient RNA or doublets")
        miss = [e.samples[i] for i in low if absent[i, gi]]
        if miss:
            cav.append(f"detection: {gene} is not detected in any cell of {', '.join(miss)}; a "
                       f"gene-annotation or chemistry difference between datasets could explain the gap")
        cav += ambient.caveat(lin, [gene], cs)
        tech = technical_class(gene, lin)
        if tech:
            cav.append(f"technical: {gene} is in the {tech} gene class")
        lins = spread[(case, ref, gi, d)]
        m = scanned[(case, ref)]
        labels = []
        if len(lins) >= max(cfg.broadcast_min_lineages, math.ceil(cfg.broadcast_fraction * m)):
            home = e.lineages[int(np.nanargmax(np.nan_to_num(e.cpm[:, high, gi], nan=-1).mean(axis=1)))]
            if lin != home:
                cav.append(f"broadcast: {gene} separates {case} from {ref} the same way in {len(lins)} of {m} "
                           f"lineages; outside its top lineage ({home}) this is most likely ambient RNA, doublets or "
                           f"a sample-wide effect")
            else:
                labels.append(f"also in {len(lins) - 1} other lineages (this is its top lineage)")
        labels.insert(0, _pattern(hits, lin, gene, case, ref, d, gnames, study.dataset.reference_group))
        if (role := roles.label(gene)):
            labels.append(role)
        flagged = any(c.startswith(FLAG_PREFIXES) for c in cav)
        per = {e.samples[i]: round(float(e.cpm[li, i, gi]), 2) for i in ci + ri}
        out.append(Finding(id="", study=study.dataset.study, source_file=e.source,
                   review_status=study.dataset.review_status, kind="expression",
                   row_key=f"{lin}|{gene}|{case}|{ref}", case=case, reference=ref, cell_type=lin,
                   feature=gene, genes=[gene], direction=d, effect=lfc,
                   effect_label=f"log2 ratio of mean CPM per sample (pseudocount 1; {e.label}, all genes)",
                   n_case=len(cs), n_reference=len(rs), per_sample=per, separated=True, p_separation=ps,
                   caveats=cav, labels=labels,
                   priority=gap * (1 - ps) * excess * (cfg.flagged_priority_factor if flagged else 1)))
    return out


# ---- per-sample target-gene scores (TF regulons, PROGENy pathways) -------------------------------------------------
def gene_sets(kb, progeny_top: int = 500) -> dict[str, tuple[list[str], np.ndarray, str]]:
    """name -> (target genes, signed weights, source). CollecTRI TF regulons and PROGENy pathway footprints."""
    out = {}
    ct = kb.table("collectri.csv")
    for tf, sub in ct.groupby("source"):
        out[f"TF {tf} targets"] = (sub.target.astype(str).tolist(), sub.weight.to_numpy(float), "CollecTRI")
    pr = kb.table("progeny.csv")
    pr = pr.assign(a=pr.weight.abs()).sort_values("a", ascending=False).groupby("source").head(progeny_top)
    for pw, sub in pr.groupby("source"):
        out[f"PROGENy {pw} targets"] = (sub.target.astype(str).tolist(), sub.weight.to_numpy(float), "PROGENy")
    return out


@dataclass
class TargetScores:
    names: list[str]
    sources: list[str]
    scores: np.ndarray  # (lineage, sample, set); NaN where a sample has no cells of the lineage or too few targets
    n_targets: np.ndarray  # (lineage, set): targets expressed in the lineage
    drivers: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]  # (lineage, set) -> (gene idx, weights)
    z: dict[int, np.ndarray]  # lineage -> (sample, gene) z-scores


def target_scores(e: Expression, sets: dict[str, tuple[list[str], np.ndarray, str]], min_targets: int = 10,
                  min_cpm: float = 1.0) -> TargetScores:
    """Per lineage: z-score log2(CPM+1) of each expressed gene across samples, then score each gene set per sample as
    sum(w * z) / sqrt(sum(w^2)) over its expressed targets. Technical genes (mitochondrial, ribosomal,
    immunoglobulin...) are left out. Labels never enter the score, so relabelling is a valid null."""
    gidx = {g: i for i, g in enumerate(e.genes)}
    keep_gene = np.array([technical_class(str(g)) is None and not _IG.search(str(g)) for g in e.genes])
    names = list(sets)
    L, S, T = len(e.lineages), len(e.samples), len(names)
    scores = np.full((L, S, T), np.nan)
    n_t = np.zeros((L, T), int)
    drivers, zs = {}, {}
    for li in range(L):
        ok = ~np.isnan(e.cpm[li, :, 0])
        if ok.sum() < 3:
            continue
        x = np.log2(e.cpm[li][ok] + 1)
        sd = x.std(axis=0)
        expressed = (e.cpm[li][ok].mean(axis=0) >= min_cpm) & (sd > 0) & keep_gene
        z = np.zeros_like(x)
        z[:, expressed] = (x[:, expressed] - x[:, expressed].mean(axis=0)) / sd[expressed]
        full = np.full((S, len(e.genes)), np.nan)
        full[ok] = z
        zs[li] = full
        for ti, name in enumerate(names):
            tg, w, _ = sets[name]
            idx = np.array([gidx.get(g, -1) for g in tg])
            m = (idx >= 0)
            idx, ww = idx[m], w[m]
            m2 = expressed[idx]
            idx, ww = idx[m2], ww[m2]
            n_t[li, ti] = len(idx)
            if len(idx) < min_targets:
                continue
            scores[li, ok, ti] = z[:, idx] @ ww / math.sqrt(float((ww ** 2).sum()))
            drivers[(li, ti)] = (idx, ww)
    return TargetScores(names, [sets[n][2] for n in names], scores, n_t, drivers, zs)


def _score_scan(ts: TargetScores, cfg: FindingsConfig, ci: list[int], ri: list[int], li: int):
    sc = ts.scores[li]
    ci = [i for i in ci if not np.isnan(sc[i]).all()]
    ri = [i for i in ri if not np.isnan(sc[i]).all()]
    T = sc.shape[1]
    if len(ci) < 2 or len(ri) < 2:
        z = np.zeros(T, bool)
        return ci, ri, z, np.zeros(T), np.zeros(T)
    C, R = sc[ci], sc[ri]
    delta = C.mean(0) - R.mean(0)
    gap = np.where(delta > 0, C.min(0) - R.max(0), R.min(0) - C.max(0))
    hit = ~np.isnan(delta) & (np.abs(delta) >= cfg.score_min_delta) & (gap >= cfg.score_min_gap)
    return ci, ri, np.nan_to_num(hit, nan=False).astype(bool), delta, gap


def _target_findings(study: LoadedStudy, cfg: FindingsConfig, e: Expression, ts: TargetScores,
                     pairs: list[tuple[str, str]]) -> list[Finding]:
    groups = study.dataset.groups
    L = len(e.lineages)
    hits, raw = {}, []
    for case, ref in pairs:
        cidx, ridx = _sample_idx(e, groups, case), _sample_idx(e, groups, ref)

        def count(ci, ri):
            return np.array([int(_score_scan(ts, cfg, ci, ri, li)[2].sum()) for li in range(L)])
        null = _relabel(count, L, cidx, ridx, 200)
        for li in range(L):
            ci, ri, hit, delta, gap = _score_scan(ts, cfg, cidx, ridx, li)
            n_tested = int((~np.isnan(ts.scores[li, ci + ri, :]).any(axis=0)).sum()) if ci and ri else 0
            for ti in np.flatnonzero(hit):
                d = "up" if delta[ti] > 0 else "down"
                hits[(e.lineages[li], ts.names[ti], case, ref)] = d
                raw.append((case, ref, li, ti, d, ci, ri, float(delta[ti]), float(gap[ti]), n_tested, int(hit.sum()),
                            float(null[li])))
    gnames = list(groups)
    # Driver genes (largest weighted target changes) and a scale-free strength: gap over the set's spread across all
    # samples of the lineage (raw scores grow with regulon size).
    # Each driver's level in this lineage relative to its top lineage, in the samples where the score is high.
    filled = np.where(np.isnan(e.cpm), -np.inf, e.cpm)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.where(filled.max(axis=0) > 0, e.cpm / filled.max(axis=0), np.nan)
    top_lin = filled.argmax(axis=0)
    info = []
    for case, ref, li, ti, d, ci, ri, delta, gap, n_tested, n_hit, null in raw:
        idx, w = ts.drivers[(li, ti)]
        dz = ts.z[li][ci][:, idx].mean(0) - ts.z[li][ri][:, idx].mean(0)
        contrib = w * dz * (1 if d == "up" else -1)
        order = [j for j in np.argsort(-contrib)[:cfg.score_driver_n] if contrib[j] > 0]
        spread = float(np.nanstd(ts.scores[li, :, ti])) or 1.0
        high = ci if d == "up" else ri
        foreign, where = 0.0, collections.Counter()
        for j in order:
            g = int(idx[j])
            r = np.nan_to_num(rel[li, high, g], nan=1.0).mean()
            tl = int(np.bincount(top_lin[high, g], minlength=len(e.lineages)).argmax())
            if r < cfg.spillover_max_ratio and tl != li:
                foreign += contrib[j]
                where[e.lineages[tl]] += 1
        total = float(sum(contrib[j] for j in order)) or 1.0
        info.append({"drivers": {int(idx[j]): float(contrib[j]) for j in order},
                     "top": [str(e.genes[idx[j]]) for j in order[:6]], "strength": gap / spread,
                     "foreign": foreign / total, "where": [x for x, _ in where.most_common(3)]})
    # Redundancy: regulons overlap heavily, so one program can light up hundreds of TFs. Within a lineage, contrast and
    # direction, sets are taken strongest first; a set whose driver change is mostly carried by genes that already
    # drive a stronger set is labelled redundant (naming that set) and ranked behind.
    redundant: dict[int, str] = {}
    by_stratum: dict[tuple, list[int]] = {}
    for k, (case, ref, li, ti, d, *_) in enumerate(raw):
        by_stratum.setdefault((li, case, ref, d), []).append(k)
    for ks in by_stratum.values():
        covered: dict[int, int] = {}  # driver gene -> first representative that used it
        for k in sorted(ks, key=lambda k: -info[k]["strength"]):
            drv = info[k]["drivers"]
            tot = sum(drv.values()) or 1.0
            shared = {covered[g] for g in drv if g in covered}
            frac = sum(v for g, v in drv.items() if g in covered) / tot
            if frac >= cfg.score_redundant_share:
                main = collections.Counter(covered[g] for g in drv if g in covered).most_common(1)[0][0]
                redundant[k] = (f"redundant: {frac:.0%} of its driver change comes from genes that drive stronger "
                                f"sets ({ts.names[raw[main][3]]}" + (f" and {len(shared) - 1} more" if len(shared) > 1
                                                                     else "") + ")")
            else:
                for g in drv:
                    covered.setdefault(g, k)
    out = []
    for k, (case, ref, li, ti, d, ci, ri, delta, gap, n_tested, n_hit, null) in enumerate(raw):
        lin, name = e.lineages[li], ts.names[ti]
        cs, rs = [e.samples[i] for i in ci], [e.samples[i] for i in ri]
        ps = p_separation(len(ci), len(ri))
        excess = max(0.0, 1 - null / n_hit)
        top, strength = info[k]["top"], info[k]["strength"]
        cav = [f"small_n: {case} vs {ref} has {len(cs)} vs {len(rs)} samples"] if min(len(cs), len(rs)) < 3 else []
        cav.append(f"genome_wide: one of {n_hit} {lin} gene sets (of {n_tested} scored) separating {case} from {ref}; "
                   f"relabelling the same samples gives {null:.0f} on average")
        cav.append(f"inferred: score from the transcripts of {ts.n_targets[li, ti]} curated {ts.sources[ti]} targets "
                   f"expressed in {lin}; target transcripts, not measured regulator or pathway activity")
        few = {e.samples[i]: int(e.cells[li, i]) for i in ci + ri if e.cells[li, i] < cfg.min_cells}
        if few:
            cav.append(f"few_cells: {lin} has fewer than {cfg.min_cells} cells in "
                       + ", ".join(f"{s} ({n})" for s, n in few.items()))
        if info[k]["foreign"] >= 0.5:
            cav.append(f"spillover: {info[k]['foreign']:.0%} of the score change comes from genes expressed mainly in "
                       f"{', '.join(info[k]['where'])}, not {lin}; ambient RNA or doublets could explain it")
        labels = [_pattern(hits, lin, name, case, ref, d, gnames, study.dataset.reference_group)]
        if top:
            labels.append(f"driven by: {', '.join(top)}")
        if k in redundant:
            labels.append(redundant[k])
        flagged = any(c.startswith(FLAG_PREFIXES) for c in cav)
        regulator = name.split()[1] if name.startswith("TF ") else None
        out.append(Finding(id="", study=study.dataset.study, source_file=e.source,
                   review_status=study.dataset.review_status, kind="target_score",
                   row_key=f"{lin}|{name}|{case}|{ref}", case=case, reference=ref, cell_type=lin, feature=name,
                   genes=[regulator] if regulator else [], direction=d, effect=delta,
                   effect_label=f"difference in mean target-gene score per sample ({ts.sources[ti]} weights, "
                                f"z-scored log CPM; {e.label})",
                   n_case=len(cs), n_reference=len(rs),
                   per_sample={e.samples[i]: round(float(ts.scores[li, i, ti]), 2) for i in ci + ri},
                   separated=True, p_separation=ps, caveats=cav, labels=labels,
                   priority=strength * (1 - ps) * excess * (cfg.flagged_priority_factor if flagged else 1)
                   * (cfg.flagged_priority_factor if k in redundant else 1)))
    return out


def _deprioritised(f: Finding) -> bool:
    return is_flagged(f) or any(x.startswith("redundant") for x in f.labels) or f.priority <= 0


def digest(findings: list[Finding], cfg: FindingsConfig) -> list[Finding]:
    by_kind: dict[str, list[Finding]] = {}
    for f in findings:
        by_kind.setdefault(f.kind, []).append(f)
    keep = []
    for kind, quota in cfg.digest_quota.items():
        ranked = sorted(by_kind.get(kind, []), key=lambda f: -f.priority)
        cap = cfg.digest_per_stratum.get(kind)
        if cap is None:
            keep += ranked[:quota]
            continue
        # Clean findings first, at most `cap` per (cell type, contrast); then other clean ones; flagged or redundant
        # ones last.
        n: dict[tuple[str, str, str], int] = {}
        first, second, last = [], [], []
        for f in ranked:
            if _deprioritised(f):
                last.append(f)
                continue
            key = (f.cell_type, f.case, f.reference)
            n[key] = n.get(key, 0) + 1
            (first if n[key] <= cap else second).append(f)
        keep += (first + second + last)[:quota]
    return sorted(keep, key=lambda f: f.id)


def digest_line(f: Finding) -> str:
    cells = f.cell_type + (f" -> {f.target_cell_type}" if f.target_cell_type else "")
    stats = [f"{f.effect_label.split(' (')[0]}={f.effect:.2f}"]
    if f.q is not None:
        stats.append(f"q={f.q:.2g}")
    elif f.p is not None:
        stats.append(f"p={f.p:.2g}")
    if f.separated:
        stats.append(f"all {f.n_case} {f.case} samples beyond all {f.n_reference} {f.reference} "
                     f"(chance {f.p_separation:.2g})")
    flags = sorted({c.split(":")[0] for c in f.caveats} - {"panel", "small_n", "genome_wide", "inferred"})
    labels = "; ".join(x.removeprefix("pattern: ") for x in f.labels)
    return (f"{f.id} | {f.kind} | {f.case} vs {f.reference} | {cells} | {f.feature} | {f.direction} | "
            f"{'; '.join(stats)} | n={f.n_case}v{f.n_reference}" + (f" | flags: {','.join(flags)}" if flags else "")
            + (f" | {labels}" if labels else ""))
