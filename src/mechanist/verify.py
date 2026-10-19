"""Evidence verification: deterministic checks of every hypothesis step against the findings. No LLM.

Also produces the verifier's own objections (overreach, small n, confounds, contradictions, lack of specificity),
which are reported whatever the critic says.
"""

from __future__ import annotations

from mechanist.records import (
    LEVEL_RANK,
    Check,
    Dataset,
    Finding,
    Hypothesis,
    Objection,
    Step,
    StepVerification,
    Verification,
)

# Coarse cell families so "Macrophage" (panel) and "Myeloid" (pipeline broad group) can support the same step.
CELL_FAMILIES = {
    "myeloid": ("myeloid", "macrophage", "monocyte", "dendritic", "mono/mac"),
    "neutrophil": ("neutrophil",),
    "b": ("b cell", "plasma", "b-cell", "bcell"),
    "t": ("t cell", "t/nk", "nk", "lymphoid", "t-cell"),
    "fibroblast": ("fibroblast",),
    "endothelial": ("endothelial",),
    "keratinocyte": ("keratinocyte", "epithelial"),
    "mural": ("smooth muscle", "pericyte"),
    "sweat": ("sweat",),
}

CAVEAT_KIND = {"small_n": "small_n", "few_cells": "small_n", "confound": "confound", "ambient": "confound", "spillover": "confound", "detection": "confound", "panel": "confound",
               "technical": "confound", "broadcast": "confound",
               "classification": "confound", "not_significant": "missing_evidence"}


def cell_family(name: str | None) -> str | None:
    if not name:
        return None
    low = name.lower()
    for fam, keys in CELL_FAMILIES.items():
        if any(k in low for k in keys):
            return fam
    return low.strip()


def _step_cells(step: Step) -> set[str]:
    return {c for c in (cell_family(step.source_cell), cell_family(step.target_cell)) if c}


def _relevant(f: Finding, step: Step, disease: str) -> list[Check]:
    """Checks that a cited finding is about this step: disease, cell, gene, direction."""
    cells = _step_cells(step)
    f_cells = {c for c in (cell_family(f.cell_type), cell_family(f.target_cell_type)) if c}
    genes = {g.upper() for g in step.genes}
    checks = [
        Check(rule="disease", passed=f.case == disease, detail=f"{f.id} is {f.case} vs {f.reference}"),
        Check(rule="cell_type", passed=bool(cells & f_cells),
              detail=f"{f.id} is about {', '.join(sorted(f_cells))}; step names {', '.join(sorted(cells))}"),
    ]
    if f.genes:
        checks.append(Check(rule="gene", passed=bool(genes & {g.upper() for g in f.genes}),
                            detail=f"{f.id} genes {','.join(f.genes)}; step genes {','.join(sorted(genes)) or '-'}"))
    checks.append(Check(rule="direction", passed=f.direction == step.direction,
                        detail=f"{f.id} is {f.direction}; step says {step.direction}"))
    return checks


def _contradictions(step: Step, support: list[Finding], findings: list[Finding]) -> list[str]:
    """Findings with the same contrast, cell and gene as a supporting finding but the opposite direction."""
    out = []
    for s in support:
        fam, genes = cell_family(s.cell_type), {g.upper() for g in s.genes}
        for f in findings:
            if (f.id != s.id and f.case == s.case and f.reference == s.reference and cell_family(f.cell_type) == fam
                    and f.direction != step.direction and (genes & {g.upper() for g in f.genes} if genes
                                                          else f.feature == s.feature)):
                out.append(f.id)
    return sorted(set(out))


def verify_step(step: Step, disease: str, by_id: dict[str, Finding], findings: list[Finding]) -> StepVerification:
    checks: list[Check] = []
    invalid = [i for i in step.finding_ids if i not in by_id]
    if invalid:
        checks.append(Check(rule="citation_exists", passed=False, detail=f"no such finding: {', '.join(invalid)}"))
    support, wrong_dir, irrelevant = [], [], []
    for fid in step.finding_ids:
        f = by_id.get(fid)
        if f is None:
            continue
        cs = _relevant(f, step, disease)
        checks += cs
        ok = {c.rule: c.passed for c in cs}
        if all(v for r, v in ok.items() if r != "direction"):
            (support if ok["direction"] else wrong_dir).append(f)
        else:
            irrelevant.append(fid)
    contra = _contradictions(step, support, findings)
    over = LEVEL_RANK[step.required_level] > LEVEL_RANK["transcript"]
    checks.append(Check(rule="evidence_level", passed=not over,
                        detail=f"step needs {step.required_level}; data are transcript level"))
    if contra:
        checks.append(Check(rule="no_contradiction", passed=False, detail=f"opposite-direction findings: {', '.join(contra)}"))

    if wrong_dir or contra:
        status = "contradicted"
    elif not support:
        status = "unsupported"
    elif over or invalid or irrelevant:
        status = "partial"
    else:
        status = "supported"
    valid = [f.id for f in support]
    return StepVerification(step_id=step.id, status=status, valid_finding_ids=valid, invalid_finding_ids=invalid,
                            contradicting_finding_ids=contra + [f.id for f in wrong_dir], checks=checks)


def verify(h: Hypothesis, findings: list[Finding]) -> Verification:
    by_id = {f.id: f for f in findings}
    steps = [verify_step(s, h.disease, by_id, findings) for s in h.steps]
    st = [s.status for s in steps]
    if h.withdrawn:
        status = "withdrawn"
    elif not st:
        status = "partial"  # e.g. a non-mechanistic alternative with nothing the verifier can test
    elif "contradicted" in st or all(s == "unsupported" for s in st):
        status = "rejected"
    elif all(s == "supported" for s in st):
        status = "supported"
    else:
        status = "partial"
    return Verification(hypothesis_id=h.id, status=status, steps=steps)


def verifier_objections(h: Hypothesis, v: Verification, findings: list[Finding], dataset: Dataset) -> list[Objection]:
    by_id = {f.id: f for f in findings}
    out: list[Objection] = []
    for step, sv in zip(h.steps, v.steps):
        if sv.invalid_finding_ids:
            out.append(Objection(target=step.id, kind="invalid_citation", source="verifier",
                                 text=f"Cites findings that do not exist: {', '.join(sv.invalid_finding_ids)}."))
        if sv.status == "contradicted":
            out.append(Objection(target=step.id, kind="contradiction", source="verifier",
                                 finding_ids=sv.contradicting_finding_ids,
                                 text="Findings in the same comparison, cell type and gene point the other way."))
        if sv.status == "unsupported":
            out.append(Objection(target=step.id, kind="unsupported_step", source="verifier",
                                 text="No cited finding matches this step's disease, cell type, gene and direction."))
        if LEVEL_RANK[step.required_level] > 0:
            out.append(Objection(target=step.id, kind="overreach_level", source="verifier",
                                 text=f"Step claims {step.required_level}-level effect ({step.mediator}); scRNA-seq "
                                      f"shows only transcript abundance."))
    # Caveats carried by the evidence the hypothesis actually uses.
    cited = {fid for sv in v.steps for fid in sv.valid_finding_ids}
    seen: dict[str, list[str]] = {}
    for fid in sorted(cited):
        for c in by_id[fid].caveats:
            seen.setdefault(c, []).append(fid)
    for text, fids in seen.items():
        kind = CAVEAT_KIND.get(text.split(":")[0])
        if kind:
            out.append(Objection(target=h.id, kind=kind, source="verifier", text=text, finding_ids=fids[:10]))
    if h.specific_to_disease:
        out += _specificity(h, findings, dataset)
    for i, o in enumerate(out, 1):
        o.id = f"{h.id}.V{i}"
    return out


def _specificity(h: Hypothesis, findings: list[Finding], dataset: Dataset) -> list[Objection]:
    """A disease-specific claim needs the same signal to be absent in the other disease and a direct comparison."""
    others = [g for g in dataset.groups if g not in (h.disease, dataset.reference_group)]
    out = []
    for step in h.steps:
        genes, fams = {g.upper() for g in step.genes}, _step_cells(step)
        if not genes:
            continue
        def hits(f, fams=fams, genes=genes):
            return cell_family(f.cell_type) in fams and genes & {g.upper() for g in f.genes}
        shared = [f.id for f in findings if f.case in others and f.reference == dataset.reference_group
                  and f.direction == step.direction and hits(f)]
        direct = [f.id for f in findings if f.case == h.disease and f.reference in others
                  and f.direction == step.direction and hits(f)]
        if shared:
            out.append(Objection(target=step.id, kind="not_specific", source="verifier", finding_ids=shared,
                                 text=f"The same change is present in {'/'.join(others)} vs "
                                      f"{dataset.reference_group}."))
        if others and not direct:
            out.append(Objection(target=step.id, kind="missing_evidence", source="verifier",
                                 text=f"No direct {h.disease} vs {'/'.join(others)} finding supports this step."))
    return out
