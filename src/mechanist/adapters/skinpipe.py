"""Any current skinpipe study whose clusters a human has reviewed.

Sources (all read-only, under Projects/<study>/):
- Intermediary/compartments/<comp>/{final/<comp>.h5ad | post_review/<comp>_final.h5ad} (all_cells also
  Intermediary/runner_checkpoints/05_global_reviewed.h5ad): the reviewed object. obs: analysis_group (study group),
  replicate_unit (the replicate: donor under combine_per_donor), Reviewed_Detailed_Label (state),
  Final_Reviewed_Broad_Group (lineage); raw counts in layer 'counts' (.X is normalised).
- Results/comparisons/<comparison_id>/<case_vs_reference>/<route>/: standard condition tables for one planned
  comparison (plain .csv; .csv.gz accepted), pseudobulk_counts.h5ad, and run_manifest.yaml in the pair folder.

Refuses unreviewed studies: a pending review (metadata/run_state.yaml), no review sign-off workbook, no reviewed
object, or reviewed labels that are still cluster ids (C0), blank, or suggestions. Unreviewed clusters are not cell
types, and early_contrasts (which may run on suggested labels) are never read.
Exception, by the researcher's explicit choice only: adapter_options.allow_unreviewed: true reads a study whose review
is still pending, on the labels its object carries (the pipeline's suggestions). Placeholder labels (C0, blank) are
still refused, the dataset is marked review_status "suggested", and every report carries the caveat.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import yaml

from mechanist.adapters.base import CellSpec, LoadedStudy, PseudobulkSpec, read_tables, sha256_file
from mechanist.records import Dataset, SourceFile

STANDARD = {
    "de": "pseudobulk_differential_expression",
    "gsea": "gsea_pathways",
    "progeny": "progeny_condition_tests",
    "composition": "cell_composition_tests",
    "communication": "communication_condition_summary",
    "communication_tests": "communication_condition_tests",
}
ROUTES = ["all_cells", "whole_context", "population_molecular", "focused_compartment"]  # preferred first
SAMPLE_COLS = ["replicate_unit", "study_sample", "batch"]
STATE_COLS = ["Reviewed_Detailed_Label"]
LINEAGE_COLS = ["Final_Reviewed_Broad_Group"]
# Label values that are not reviewed cell types: bare cluster ids, blanks, review placeholders, LLM suggestions.
UNREVIEWED = re.compile(r"^(c\d+|cluster\s*\d+|nan|none|review|unlabelled|unlabeled|suggested.*|llm.*|)$", re.IGNORECASE)


class UnreviewedStudy(ValueError):
    """The study's clusters have not been reviewed by a human; Mechanist does not read it."""


def _obs(path: Path, cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    import anndata.io as aio
    import h5py

    with h5py.File(path, "r") as f:
        have = list(f["obs"].keys())
        df = pd.DataFrame({c: aio.read_elem(f["obs"][c]) for c in cols if c in have}).astype(str)
        layers = list(f["layers"].keys()) if "layers" in f else []
    return df, layers


def _first(options: list[str], have) -> str | None:
    return next((c for c in options if c in have), None)


def reviewed_object(study_dir: Path, comp: str) -> Path | None:
    cands = [study_dir / "Intermediary" / "compartments" / comp / "final" / f"{comp}.h5ad",
             study_dir / "Intermediary" / "compartments" / comp / "post_review" / f"{comp}_final.h5ad"]
    if comp == "all_cells":
        cands.append(study_dir / "Intermediary" / "runner_checkpoints" / "05_global_reviewed.h5ad")
    return next((p for p in cands if p.exists()), None)


def review_signoff(study_dir: Path, comp: str) -> Path | None:
    res = study_dir / "Results" / "compartments" / comp
    cands = [res / "post_review" / "tables" / f"{comp}_final_clusterwise_summary.xlsx",
             res / "review" / f"{comp}_clusterwise_summary_labeled.xlsx"]
    return next((p for p in cands if p.exists()), None)


def check_reviewed(study_dir: Path, comp: str, allow_unreviewed: bool = False) -> tuple[Path, Path]:
    """(reviewed object, sign-off) or UnreviewedStudy. allow_unreviewed: skip the pending-review gate."""
    state = study_dir / "metadata" / "run_state.yaml"
    if state.exists() and not allow_unreviewed:
        st = (yaml.safe_load(state.read_text()) or {}).get("status")
        if st == "waiting_for_review":
            raise UnreviewedStudy(f"{study_dir.name}: run_state is waiting_for_review; review the clusters first")
    obj, signoff = reviewed_object(study_dir, comp), review_signoff(study_dir, comp)
    if obj is None or signoff is None:
        raise UnreviewedStudy(f"{study_dir.name}/{comp}: no reviewed object or review sign-off "
                              f"(object: {obj or 'missing'}, sign-off: {signoff or 'missing'}); pre-review clusters "
                              f"and LLM suggestions are not cell types")
    return obj, signoff


def _platforms(study_dir: Path, gmap: dict[str, str]) -> dict[str, str]:
    """{short group: platform(s)} from the pipeline's design check (early_contrasts/*/design/platform.csv), if any."""
    for f in sorted((study_dir / "Results" / "early_contrasts").glob("*/design/platform.csv")):
        try:
            df = pd.read_csv(f)
        except (OSError, ValueError):
            continue
        if {"group", "platform"} <= set(df.columns):
            out: dict[str, set] = {}
            for g, p in zip(df["group"].astype(str), df["platform"].astype(str)):
                if g in gmap:
                    out.setdefault(gmap[g], set()).add(p)
            return {g: "/".join(sorted(v)) for g, v in out.items()}
    return {}


def _manifests(study_dir: Path) -> list[tuple[Path, dict]]:
    out = []
    for m in sorted((study_dir / "Results" / "comparisons").glob("*/*/run_manifest.yaml")):
        d = yaml.safe_load(m.read_text()) or {}
        if d.get("case") and d.get("reference"):
            out.append((m.parent, d))
    return out


def _pick_comparison(study_dir: Path, case_raw: str | None, ref_raw: str | None, comparison: str | None,
                     route: str | None):
    """(pair folder, manifest, route entry) for the comparison of the primary disease, or None."""
    hits = [(p, d) for p, d in _manifests(study_dir)
            if d["case"] == case_raw and (ref_raw is None or d["reference"] == ref_raw)
            and (comparison is None or d.get("comparison_id") == comparison)]
    if not hits:
        return None
    refs = sorted({d["reference"] for _, d in hits})
    if len(refs) > 1:
        raise ValueError(f"{study_dir.name}: {case_raw!r} is compared with several references ({'; '.join(refs)}); "
                         f"set adapter_options.reference")
    for pair, d in hits:
        routes = {r["route"]: r for r in d.get("routes") or [] if isinstance(r, dict) and r.get("route")}
        names = [route] if route else [r for r in ROUTES if r in routes] + [r for r in routes if r not in ROUTES]
        for name in names:
            if name in routes and (pair / name).is_dir():
                return pair, d, routes[name]
    return None


def _table_paths(route_dir: Path) -> dict[str, Path]:
    out = {}
    for key, stem in STANDARD.items():
        for ext in (".csv", ".csv.gz"):
            if (route_dir / f"{stem}{ext}").exists():
                out[key] = route_dir / f"{stem}{ext}"
                break
        else:
            out[key] = route_dir / f"{stem}.csv"  # missing: reported as a caveat
    return out


def study_question(study_dir: Path) -> str:
    """The submitter's question, from the study's metadata/submission.yaml (its title and description), as written."""
    import yaml
    p = study_dir / "metadata" / "submission.yaml"
    try:
        s = yaml.safe_load(p.read_text()) or {}
    except (OSError, yaml.YAMLError):
        return ""
    st = s.get("project") or s.get("study") or s
    if not isinstance(st, dict):
        return ""
    title, desc = str(st.get("title") or "").strip(), st.get("description") or ""
    if isinstance(desc, list):
        desc = "\n".join(f"- {x}" for x in desc)
    desc = str(desc).strip()
    return "\n".join(x for x in (f"Title: {title}" if title else "", desc) if x)


def load(projects_root: Path, study: str, lab_tables: bool = False, primary: str | None = None,
         compartment: str = "all_cells", reference: str | None = None, groups: dict[str, str] | list[str] | None = None,
         comparison: str | None = None, route: str | None = None, allow_unreviewed: bool = False) -> LoadedStudy:
    """primary: the disease group (study name, after `groups` renaming). reference: the reference group (default:
    the reference of the planned comparison of the primary disease). groups: raw analysis_group values to keep, as a
    list or as {raw: short name}; default every group in the reviewed object. lab_tables: no lab tables exist for
    current studies (ignored). allow_unreviewed: the researcher's choice to read a study whose review is pending, on
    its suggested labels (marked "suggested", with a caveat)."""
    study_dir = projects_root / study
    obj, signoff = check_reviewed(study_dir, compartment, allow_unreviewed)
    obs, layers = _obs(obj, ["analysis_group", *SAMPLE_COLS, *STATE_COLS, *LINEAGE_COLS])
    sample_col, state_col, lineage_col = (_first(SAMPLE_COLS, obs), _first(STATE_COLS, obs),
                                          _first(LINEAGE_COLS, obs))
    if "analysis_group" not in obs or sample_col is None or state_col is None or lineage_col is None:
        raise UnreviewedStudy(f"{obj}: missing reviewed obs columns (need analysis_group, one of {SAMPLE_COLS}, "
                              f"{STATE_COLS[0]}, {LINEAGE_COLS[0]})")

    raw_groups = sorted(obs["analysis_group"].unique())
    if groups is None:
        gmap = {g: g for g in raw_groups}
    elif isinstance(groups, dict):
        gmap = {str(k): str(v) for k, v in groups.items()}
    else:
        gmap = {str(g): str(g) for g in groups}
    unknown = sorted(set(gmap) - set(raw_groups))
    if unknown:
        raise ValueError(f"{study}: groups not in the reviewed object: {unknown}; groups: {raw_groups}")
    inv = {v: k for k, v in gmap.items()}
    if primary is not None and primary not in inv:
        raise ValueError(f"{study}: disease {primary!r} is not a group; groups: {sorted(inv)}")

    kept = obs[obs["analysis_group"].isin(gmap)]
    bad = {col: sorted({x for x in kept[col].unique() if UNREVIEWED.match(x.strip())}) for col in (state_col, lineage_col)}
    bad = {k: v for k, v in bad.items() if v}
    if bad:
        n = int(kept[state_col].str.strip().str.match(UNREVIEWED).sum())
        raise UnreviewedStudy(f"{obj}: {n} cells carry unreviewed labels {bad}; review them before investigating")

    pick = _pick_comparison(study_dir, inv.get(primary) if primary else None,
                            inv.get(reference) if reference else None, comparison, route) if primary else None
    if reference is None:
        if pick is None:
            raise ValueError(f"{study}: no planned comparison for {primary!r}; set adapter_options.reference")
        reference = gmap.get(pick[1]["reference"])
        if reference is None:
            raise ValueError(f"{study}: comparison reference {pick[1]['reference']!r} is not among the kept groups")
    if reference not in inv:
        raise ValueError(f"{study}: reference {reference!r} is not a group; groups: {sorted(inv)}")

    tables, files, caveats = {}, [], []
    standard_contrast = standard_n = pb = None
    if pick is not None:
        pair, man, r = pick
        route_dir = pair / r["route"]
        if str(man.get("labels", "reviewed")) != "reviewed":
            caveats.append(f"missing: comparison tables skipped, they were built on {man.get('labels')} labels")
        else:
            paths = _table_paths(route_dir)
            tables, files = read_tables(paths)
            gcol = r.get("groupby")
            for tab in tables.values():
                df = tab.df
                if gcol and gcol in df and "broad_group" not in df:
                    tab.df = df = df.rename(columns={gcol: "broad_group"})
                for col in ("case", "reference", "condition"):
                    if col in df:
                        df[col] = df[col].astype(str).replace(gmap)
            standard_contrast = (gmap[man["case"]], gmap[man["reference"]])
            rep = r.get("replicates") or {}
            if "case" in rep and "reference" in rep:
                standard_n = (int(rep["case"]), int(rep["reference"]))
            caveats += [f"missing: {k} table not found" for k in sorted(set(paths) - set(tables))]
            caveats.append(f"scope: standard tables come from comparison {man.get('comparison_id')!r} "
                           f"({standard_contrast[0]} vs {standard_contrast[1]}, route {r['route']}, cells grouped by "
                           f"{gcol}); other contrasts exist only in the per-sample expression findings")
            pb_path = route_dir / "pseudobulk_counts.h5ad"
            if pb_path.exists():
                files.append(SourceFile(path=str(pb_path), sha256=sha256_file(pb_path), rows=0))
                pb = PseudobulkSpec(path=pb_path, sample_col=r.get("sample_col") or sample_col,
                                    lineage_col=gcol or "broad_group", cells_col="psbulk_cells")
    else:
        caveats.append(f"missing: no planned comparison of {primary} vs {reference}; standard pipeline tables "
                       f"unavailable, per-sample expression findings only")

    sample_groups = kept.assign(g=kept["analysis_group"].map(gmap))[["g", sample_col]].drop_duplicates()
    if sample_groups[sample_col].duplicated().any():
        dup = sorted(sample_groups[sample_col][sample_groups[sample_col].duplicated()].unique())
        raise ValueError(f"{study}: replicates in more than one group: {dup[:5]}")
    by_group = {g: sorted(sub[sample_col]) for g, sub in sample_groups.groupby("g")}
    small = [f"{g} ({len(s)})" for g, s in by_group.items() if len(s) < 3]
    if small:
        caveats.insert(0, f"small_n: groups with fewer than 3 replicates: {', '.join(small)}; per-sample consistency "
                          f"only, no test can reach conventional significance")
    caveats.append(f"unit: replicates are {sample_col} values (donor-level when repeat samples are combined)")
    caveats.append("level: all evidence is single-cell transcript abundance, not protein, activity or function.")
    plat = _platforms(study_dir, gmap)
    if len(set(plat.values())) > 1:
        caveats.insert(0, "platform: " + "; ".join(f"{g} on {p}" for g, p in sorted(plat.items()))
                       + ". Differences between groups measured on different platforms can be technical (nuclei vs "
                         "whole cells shift many genes); a difference that also holds against a same-platform group "
                         "is the safer one")
    pending = allow_unreviewed and (study_dir / "metadata" / "run_state.yaml").exists() and (yaml.safe_load(
        (study_dir / "metadata" / "run_state.yaml").read_text()) or {}).get("status") == "waiting_for_review"
    if pending:
        caveats.insert(0, "labels: cell types are the pipeline's suggested labels (automatic and language-model "
                          "annotation); the study's review is still pending, so every cell-type claim is provisional")

    cells = CellSpec(path=obj, sample_col=sample_col, group_col="analysis_group", group_map=gmap,
                     state_col=state_col, lineage_col=lineage_col,
                     counts_layer="counts" if "counts" in layers else None)
    if "counts" not in layers:
        caveats.append("missing: no 'counts' layer; the genome-wide scan uses .X only if it holds raw counts")
    files += [SourceFile(path=str(obj), sha256=sha256_file(obj), rows=len(obs)),
              SourceFile(path=str(signoff), sha256=sha256_file(signoff), rows=0)]
    dataset = Dataset(study=study, adapter="skinpipe", groups=by_group, reference_group=reference, files=files,
                      review_status="suggested" if pending else "reviewed", caveats=caveats,
                      question=study_question(study_dir))
    return LoadedStudy(dataset=dataset, tables=tables, standard_contrast=standard_contrast, standard_n=standard_n,
                       cells=cells, pseudobulk=pb)
