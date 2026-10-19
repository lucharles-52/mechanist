"""Shared adapter types. Adapters only read files; they never write into the pipeline's project folders."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from mechanist.records import Dataset, SourceFile


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Table:
    name: str
    path: Path
    df: pd.DataFrame


@dataclass
class CellSpec:
    """Where the reviewed single-cell object is and how to read its labels. Loaded only when a tool needs it.

    counts_layer: where the RAW counts are (None = .X). The genome-wide scan needs raw counts (CPM from normalised
    values is meaningless); tools that compare expression levels normalise per cell themselves (log1p CP10k).
    state_to_lineage: reviewed state -> lineage, or lineage_col: an obs column that already holds the lineage."""

    path: Path
    sample_col: str
    group_col: str
    group_map: dict[str, str]  # raw group -> study group; cells of other groups are dropped
    state_col: str
    state_to_lineage: dict[str, str] = field(default_factory=dict)
    lineage_col: str | None = None
    counts_layer: str | None = None
    doublet_col: str | None = "doublet_score"  # per-cell doublet score, used by contamination_check when present


@dataclass
class PseudobulkSpec:
    """Standard pipeline pseudobulk counts (rows = sample x lineage, raw summed counts)."""

    path: Path
    sample_col: str
    lineage_col: str
    cells_col: str | None = None


def groups_from_cells(spec: CellSpec) -> dict[str, list[str]]:
    """Study group -> samples, from the obs columns of the reviewed object (does not load the expression matrix)."""
    import anndata.io as aio
    import h5py

    with h5py.File(spec.path, "r") as f:
        obs = pd.DataFrame({c: aio.read_elem(f["obs"][c]) for c in (spec.sample_col, spec.group_col)}).astype(str)
    obs["g"] = obs[spec.group_col].map(spec.group_map)
    pairs = obs.dropna(subset=["g"])[["g", spec.sample_col]].drop_duplicates()
    return {g: sorted(sub[spec.sample_col]) for g, sub in pairs.groupby("g")}


def lineages_from_cells(spec: CellSpec) -> list[str]:
    """Lineages of the reviewed object (reads obs only)."""
    import anndata.io as aio
    import h5py

    col = spec.lineage_col or spec.state_col
    with h5py.File(spec.path, "r") as f:
        vals = pd.Series(aio.read_elem(f["obs"][col])).astype(str)
    if not spec.lineage_col:
        vals = vals.map(spec.state_to_lineage).dropna()
    return sorted(set(vals) - {"Unmapped"})


def group_order(groups, reference: str | None, primary: str | None = None) -> list[str]:
    """Display order for a study's groups: the primary disease first, other groups by name, the reference last."""
    groups = list(groups)
    others = sorted(g for g in groups if g not in (primary, reference))
    return ([primary] if primary in groups else []) + others + ([reference] if reference in groups else [])


@dataclass
class LoadedStudy:
    dataset: Dataset
    tables: dict[str, Table] = field(default_factory=dict)
    # Standard-pipeline contrast (case, reference) and replicate counts, when the study has one.
    standard_contrast: tuple[str, str] | None = None
    standard_n: tuple[int, int] | None = None
    cells: CellSpec | None = None
    pseudobulk: PseudobulkSpec | None = None


def read_tables(paths: dict[str, Path]) -> tuple[dict[str, Table], list[SourceFile]]:
    tables, files = {}, []
    for name, path in paths.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)  # handles .csv.gz
        tables[name] = Table(name, path, df)
        files.append(SourceFile(path=str(path), sha256=sha256_file(path), rows=len(df)))
    return tables, files
