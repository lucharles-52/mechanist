"""p033 (Sweet's syndrome and PG skin) — legacy study analysed partly outside the pipeline.

Sources:
- Results/communication_pipeline/*.csv.gz: standard cl_analysis tables, PG_Lesional vs Control_Skin only.
- Results/pg_mechanism/*.csv: hand-made per-sample tables for PG, Sweet and Control on a curated gene panel.
  Read only with lab_tables=True: the panel was chosen around the earlier PG hypotheses (circular).
- Intermediary/.../all_cells_final.h5ad: reviewed single-cell object (raw counts); groups and the genome-wide
  per-sample expression scan come from it.
"""

from __future__ import annotations

from pathlib import Path

from mechanist.adapters.base import CellSpec, LoadedStudy, PseudobulkSpec, groups_from_cells, read_tables, sha256_file
from mechanist.records import Dataset, SourceFile

GROUP_NAMES = {"PG_Lesional": "PG", "Control_Skin": "Control", "Sweet_Skin": "Sweet"}
REFERENCE_GROUP = "Control"

STANDARD = {
    "de": "pseudobulk_differential_expression.csv.gz",
    "gsea": "gsea_pathways.csv.gz",
    "progeny": "progeny_condition_tests.csv.gz",
    "composition": "cell_composition_tests.csv.gz",
    "communication": "communication_condition_summary.csv.gz",
    "communication_tests": "communication_condition_tests.csv.gz",
}
MECHANISM = {
    "panel": "expression_by_donor_and_type.csv",
    "signatures": "signatures_by_donor.csv",
    "ambient": "ambient_gauge.csv",
    "states": "expression_by_donor_and_state.csv",
    "fib_programs": "fibroblast_programs.csv",
}

PSEUDOBULK = "pseudobulk_counts.h5ad.gz"  # standard pipeline output, all sample types
CELLS = "Intermediary/PG_chkpt/all_cells/post_review/all_cells_final.h5ad"
# Reviewed state -> lineage. Eight lineages reproduce the cell counts of pg_mechanism/expression_by_donor_and_type
# exactly; B cell and T/NK follow by name. (The object's own General_label puts e.g. Sweet Neutrophils under Myeloid.)
STATE_TO_LINEAGE = {
    "ACKR1+ venular endothelial": "Endothelial", "Arterial/capillary endothelial": "Endothelial",
    "Lymphatic endothelial": "Endothelial",
    "CXCL14+ fibroblast": "Fibroblast", "Collagen-rich fibroblast": "Fibroblast", "Matrix/collagen fibroblast": "Fibroblast",
    "Basal/activated keratinocyte": "Keratinocyte",
    "CD163+ macrophage": "Macrophage",
    "Sweet Neutrophils": "Neutrophil", "Activated inflammatory neutrophil": "Neutrophil",
    "Pericyte / vascular smooth muscle": "Pericyte",
    "IgA plasma cell": "Plasma cell", "Immunoglobulin-high plasma/B cell": "Plasma cell",
    "DCD+ sweat-associated stromal mixture": "Sweat gland", "Sweat duct epithelial": "Sweat gland",
    "Sweat-gland secretory epithelial": "Sweat gland",
    "Naive B cell": "B cell", "Activated/class-switched B cell": "B cell", "Activated HLA-II+ B cell": "B cell",
    "FCRL5+ memory/activated B cell": "B cell", "B cell": "B cell",
    "IL7R+ CD4 memory T cell": "T/NK", "GZMK+ cytotoxic/memory T cell": "T/NK", "Activated/IFN-responsive T-NK": "T/NK",
    "NK/NKT cell": "T/NK", "CD8/NKT with immunoglobulin contamination": "T/NK",
}

# Prefixes map to objection kinds in verify.py.
CAVEATS = [
    "small_n: PG has two lesional samples (PG4L, SweetPat14); no test on PG can reach conventional significance.",
    "classification: SweetPat14 is grouped as PG in the source tables; its clinical classification is unconfirmed.",
    ("confound: the Control skin samples (SSC1/3/5/6) are public (GEO) controls from a Sweet's syndrome study, not "
    "the lab's healthy skin; site, protocol and batch differences are confounded with disease."),
    "legacy: p033 was analysed partly outside the pipeline; broad-group labels come from the legacy review.",
    ("scope: standard pipeline tables compare PG vs Control only; PG vs Sweet comparisons exist only in the "
    "per-sample expression findings."),
    "level: all evidence is single-cell transcript abundance, not protein, activity or function.",
]
LAB_CAVEAT = ("panel: the per-sample lab tables cover a hand-curated gene panel (~160 genes) chosen during the lab's "
              "earlier PG investigation; they are not genome-wide and are circular with respect to those earlier "
              "hypotheses.")


def load(projects_root: Path, study: str = "p033", lab_tables: bool = False, primary: str | None = None) -> LoadedStudy:
    """primary: unused (p033 has one standard contrast, PG vs Control)."""
    results = projects_root / study / "Results"
    paths = {k: results / "communication_pipeline" / v for k, v in STANDARD.items()}
    if lab_tables:
        paths |= {k: results / "pg_mechanism" / v for k, v in MECHANISM.items()}
    tables, files = read_tables(paths)
    for tab in tables.values():  # PG_Lesional -> PG etc., so every table uses the same group names
        for col in ("case", "reference", "condition"):
            if col in tab.df:
                tab.df[col] = tab.df[col].replace(GROUP_NAMES)
    missing = sorted(set(paths) - set(tables))
    if "panel" not in tables and "de" not in tables:
        raise FileNotFoundError(f"{study}: neither standard tables nor pg_mechanism tables found under {results}")
    cells_path = projects_root / study / CELLS
    cells = CellSpec(path=cells_path, sample_col="batch", group_col="condition",
                     group_map={"PG": "PG", "Sweet_Skin": "Sweet", "Control_Skin": "Control"},
                     state_col="final_annotation", state_to_lineage=STATE_TO_LINEAGE) if cells_path.exists() else None

    groups: dict[str, list[str]] = {}
    if cells is not None:
        groups = groups_from_cells(cells)
    elif "panel" in tables:
        df = tables["panel"].df
        for g, s in df[["group", "sample"]].drop_duplicates().itertuples(index=False):
            groups.setdefault(g, []).append(s)
    n = None
    if "composition" in tables:
        row = tables["composition"].df.iloc[0]
        n = (int(row["n_case"]), int(row["n_reference"]))

    caveats = list(CAVEATS) + ([LAB_CAVEAT] if lab_tables else [])
    caveats += [f"missing: {m} table not found" for m in missing]
    dataset = Dataset(study=study, adapter="p033", groups={g: sorted(s) for g, s in groups.items()},
                      files=files, review_status="legacy", caveats=caveats)
    if cells_path.exists():
        dataset.files.append(SourceFile(path=str(cells_path), sha256=sha256_file(cells_path), rows=0))
    pb_path = results / "communication_pipeline" / PSEUDOBULK
    pb = None
    if pb_path.exists():
        dataset.files.append(SourceFile(path=str(pb_path), sha256=sha256_file(pb_path), rows=0))
        pb = PseudobulkSpec(path=pb_path, sample_col="batch", lineage_col="broad_group", cells_col="psbulk_cells")
    if cells is None:
        caveats.append("missing: reviewed single-cell object not found; cell-level tools unavailable")
    return LoadedStudy(dataset=dataset, tables=tables, standard_contrast=("PG", REFERENCE_GROUP), standard_n=n,
                       cells=cells, pseudobulk=pb)
