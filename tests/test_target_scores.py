"""Gene-class and broadcast flags, contrast-pattern and role labels, flagged-last digest, target-gene scores.
Synthetic pseudobulk only."""

import numpy as np
import pandas as pd
import pytest

from mechanist.adapters.base import LoadedStudy
from mechanist.findings import (
    FindingsConfig,
    build_findings,
    digest,
    digest_line,
    gene_sets,
    is_flagged,
    make_expression,
    target_scores,
    technical_class,
)
from mechanist.knowledge import KnowledgeBase
from mechanist.records import Dataset

GROUPS = {"PG": ["P1", "P2"], "Sweet": ["S1", "S2"], "Control": ["C1", "C2", "C3", "C4"]}
SAMPLES = [s for ss in GROUPS.values() for s in ss]
TARGETS = [f"T{i}" for i in range(12)]


def _study() -> LoadedStudy:
    return LoadedStudy(dataset=Dataset(study="syn", adapter="test", groups=GROUPS, reference_group="Control",
                                       files=[], review_status="reviewed"))


def _expr(levels: dict[tuple[str, str], dict[str, float]], lineages: list[str], genes: list[str]):
    """levels[(lineage, gene)][sample] = counts (default 50); every sample also gets 10,000 HOUSE counts."""
    rng = np.random.default_rng(1)
    genes = ["HOUSE", *genes]
    counts = np.zeros((len(lineages) * len(SAMPLES), len(genes)))
    for li, lin in enumerate(lineages):
        for si, s in enumerate(SAMPLES):
            row = counts[li * len(SAMPLES) + si]
            row[0] = 10000
            for gi, g in enumerate(genes[1:], 1):
                row[gi] = levels.get((lin, g), {}).get(s, 50 * rng.uniform(0.8, 1.25))
    detected = np.full_like(counts, 50.0)
    return make_expression(lineages, SAMPLES, np.array(genes), counts, np.full(len(counts), 50.0), detected,
                           "synthetic", "test pseudobulk")


def _kb(tmp_path) -> KnowledgeBase:
    pd.DataFrame({"ligand": ["LIGA"], "receptor": ["RECA_RECB"]}).to_csv(tmp_path / "lr_consensus.csv", index=False)
    pd.DataFrame({"source": ["TFX"] * len(TARGETS) + ["TFSMALL"] * 3,
                  "target": TARGETS + ["T0", "T1", "T2"], "weight": 1.0}).to_csv(tmp_path / "collectri.csv",
                                                                                index=False)
    pd.DataFrame({"source": ["NFkB"] * 3, "target": ["LIGA", "T0", "T1"], "weight": [5.0, 1.0, 1.0]}).to_csv(
        tmp_path / "progeny.csv", index=False)
    return KnowledgeBase(tmp_path)


def _up(case_groups, lo=50.0, hi=400.0):
    return {s: (hi if g in case_groups else lo) for g, ss in GROUPS.items() for s in ss}


def test_technical_gene_classes():
    assert technical_class("MT-CO1").startswith("mitochondrial")
    assert technical_class("RPL13A").startswith("ribosomal")
    assert technical_class("XIST").startswith("sex")
    assert technical_class("FOS").startswith("dissociation")
    assert technical_class("LINC00152").startswith("unannotated")
    assert technical_class("IGKC", "Fibroblast").startswith("immunoglobulin")
    assert technical_class("IGKC", "Plasma cell") is None and technical_class("IGHM", "B cell") is None
    assert technical_class("CXCL13") is None and technical_class("RPLP0") is None


def test_broadcast_flags_all_but_the_top_lineage():
    lins = ["Fibroblast", "Keratinocyte", "Endothelial", "Plasma cell"]
    lv = {(lin, "BCAST"): _up({"PG"}, 5, 40) for lin in lins[:3]} | {("Plasma cell", "BCAST"): _up({"PG"}, 50, 4000)}
    lv[("Fibroblast", "LOCAL")] = _up({"PG"})
    fs = {f.row_key: f for f in build_findings(_study(), FindingsConfig(), _expr(lv, lins, ["BCAST", "LOCAL"]),
                                               primary="PG")}
    for lin in lins[:3]:
        assert any(c.startswith("broadcast") and "Plasma cell" in c for c in fs[f"{lin}|BCAST|PG|Control"].caveats)
    home = fs["Plasma cell|BCAST|PG|Control"]
    assert not is_flagged(home) and "also in 3 other lineages" in " ".join(home.labels)
    assert not is_flagged(fs["Fibroblast|LOCAL|PG|Control"])


def test_contrast_pattern_and_role_labels(tmp_path):
    lv = {("Fibroblast", "PGONLY"): _up({"PG"}), ("Fibroblast", "LIGA"): _up({"PG", "Sweet"})}
    fs = {f.row_key: f for f in build_findings(_study(), FindingsConfig(),
                                               _expr(lv, ["Fibroblast"], ["PGONLY", "LIGA"]), primary="PG",
                                               kb=_kb(tmp_path)) if f.kind == "expression"}
    assert fs["Fibroblast|PGONLY|PG|Control"].labels[0] == "pattern: PG-specific (also beyond Sweet)"
    assert fs["Fibroblast|PGONLY|PG|Sweet"].labels[0] == "pattern: PG also beyond Control"
    shared = fs["Fibroblast|LIGA|PG|Control"]
    assert shared.labels[0] == "pattern: shared with Sweet (also beyond Control)"
    assert "role: ligand, NFkB-responsive (PROGENy)" in shared.labels
    assert "Fibroblast|LIGA|PG|Sweet" not in fs
    line = digest_line(shared)
    assert "shared with Sweet" in line and "role: ligand" in line


def test_digest_puts_flagged_findings_last():
    lv = {("Fibroblast", "CLEAN"): _up({"PG"}, 50, 150), ("Fibroblast", "MT-CO1"): _up({"PG"}, 50, 800)}
    fs = build_findings(_study(), FindingsConfig(), _expr(lv, ["Fibroblast"], ["CLEAN", "MT-CO1"]), primary="PG")
    pg = [f for f in fs if f.case == "PG" and f.reference == "Control"]
    d = digest(pg, FindingsConfig(digest_quota={"expression": 1}))
    assert [f.feature for f in d] == ["CLEAN"]


def test_target_scores_find_a_coherent_regulon(tmp_path):
    lv = {("Fibroblast", t): _up({"PG"}, 50, 200) for t in TARGETS}
    e = _expr(lv, ["Fibroblast"], [*TARGETS, *(f"N{i}" for i in range(30))])
    kb = _kb(tmp_path)
    ts = target_scores(e, gene_sets(kb), min_targets=10)
    ti = ts.names.index("TF TFX targets")
    assert ts.n_targets[0, ti] == 12 and np.isnan(ts.scores[0, :, ts.names.index("TF TFSMALL targets")]).all()
    fs = {f.row_key: f for f in build_findings(_study(), FindingsConfig(), e, primary="PG", kb=kb)
          if f.kind == "target_score"}
    f = fs["Fibroblast|TF TFX targets|PG|Control"]
    assert f.direction == "up" and f.separated and f.genes == ["TFX"] and set(f.per_sample) == {
        "P1", "P2", "C1", "C2", "C3", "C4"}
    assert any(c.startswith("inferred: score from the transcripts of 12 curated CollecTRI targets") for c in f.caveats)
    assert any(x.startswith("driven by: T") for x in f.labels)
    assert f.labels[0] == "pattern: PG-specific (also beyond Sweet)"
    assert not any(k.startswith("Fibroblast|TF TFX targets|Sweet") for k in fs)
    # relabelling: no other split of these samples separates TFX, so nothing is attributed to chance
    assert "gives 0 on average" in next(c for c in f.caveats if c.startswith("genome_wide"))
    assert f.priority > 0


def test_target_scores_skip_technical_genes(tmp_path):
    # A "regulon" made only of mitochondrial genes is never scored.
    mt = [f"MT-X{i}" for i in range(12)]
    pd.DataFrame({"ligand": ["A"], "receptor": ["B"]}).to_csv(tmp_path / "lr_consensus.csv", index=False)
    pd.DataFrame({"source": ["TFM"] * 12, "target": mt, "weight": 1.0}).to_csv(tmp_path / "collectri.csv",
                                                                               index=False)
    pd.DataFrame({"source": ["NFkB"], "target": ["A"], "weight": [1.0]}).to_csv(tmp_path / "progeny.csv", index=False)
    e = _expr({("Fibroblast", g): _up({"PG"}) for g in mt}, ["Fibroblast"], mt)
    ts = target_scores(e, gene_sets(KnowledgeBase(tmp_path)))
    assert ts.n_targets[0, ts.names.index("TF TFM targets")] == 0


@pytest.mark.parametrize("flag", [False, True])
def test_target_scores_can_be_turned_off(tmp_path, flag):
    lv = {("Fibroblast", t): _up({"PG"}, 50, 200) for t in TARGETS}
    e = _expr(lv, ["Fibroblast"], TARGETS)
    kinds = {f.kind for f in build_findings(_study(), FindingsConfig(target_scores=flag), e, kb=_kb(tmp_path))}
    assert ("target_score" in kinds) is flag


def test_overlapping_regulons_are_labelled_redundant(tmp_path):
    # TFY shares 11 of TFX's 12 targets: one program; the weaker copy is labelled and ranked behind.
    kb = _kb(tmp_path)
    ct = pd.read_csv(tmp_path / "collectri.csv")
    extra = pd.DataFrame({"source": "TFY", "target": [*TARGETS[:11], "N0"], "weight": 1.0})
    pd.concat([ct, extra]).to_csv(tmp_path / "collectri.csv", index=False)
    kb = KnowledgeBase(tmp_path)
    lv = {("Fibroblast", t): _up({"PG"}, 50, 200) for t in TARGETS}
    e = _expr(lv, ["Fibroblast"], [*TARGETS, *(f"N{i}" for i in range(30))])
    fs = {f.feature: f for f in build_findings(_study(), FindingsConfig(), e, primary="PG", kb=kb)
          if f.kind == "target_score" and f.reference == "Control"}
    a, b = fs["TF TFX targets"], fs["TF TFY targets"]
    copy, rep = (b, a) if any(x.startswith("redundant") for x in b.labels) else (a, b)
    assert any(x.startswith("redundant: 100% of its driver change") and "TF TFX" in x for x in copy.labels)
    assert not any(x.startswith("redundant") for x in rep.labels) and copy.priority < rep.priority


def test_score_driven_by_another_lineages_genes_is_flagged(tmp_path):
    # TFX targets rise in PG "B cells" at a fraction of their level in macrophages: contamination, not a B-cell program.
    lv = {("B cell", t): _up({"PG"}, 50, 200) for t in TARGETS}
    lv |= {("Macrophage", t): {s: 5000.0 for s in SAMPLES} for t in TARGETS}
    e = _expr(lv, ["B cell", "Macrophage"], [*TARGETS, *(f"N{i}" for i in range(30))])
    f = next(f for f in build_findings(_study(), FindingsConfig(), e, primary="PG", kb=_kb(tmp_path))
             if f.row_key == "B cell|TF TFX targets|PG|Control")
    assert any(c.startswith("spillover:") and "Macrophage" in c for c in f.caveats) and is_flagged(f)
