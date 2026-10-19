from conftest import fid

from mechanist.records import Hypothesis, Step
from mechanist.verify import cell_family, verifier_objections, verify


def _hyp(steps, **kw):
    h = Hypothesis(id="H1", title="t", disease="PG", steps=steps, **kw)
    for i, s in enumerate(h.steps, 1):
        s.id = f"H1.S{i}"
    return h


def test_cell_family_aliases():
    assert cell_family("Macrophage") == cell_family("Myeloid")
    assert cell_family("Plasma cell") == cell_family("B cell")
    assert cell_family("Fibroblast (CXCL14+ fibroblast)") == "fibroblast"


def test_supported_step(findings):
    f = fid(findings, "panel_expression", "Fibroblast|CXCL13|PG|Control")
    h = _hyp([Step(source_cell="Fibroblast", target_cell="B cell", mediator="CXCL13", genes=["CXCL13"],
                   direction="up", finding_ids=[f])])
    v = verify(h, findings)
    assert v.status == "supported" and v.steps[0].valid_finding_ids == [f]


def test_wrong_direction_is_contradicted(findings):
    f = fid(findings, "panel_expression", "Fibroblast|CXCL13|PG|Control")
    h = _hyp([Step(source_cell="Fibroblast", mediator="CXCL13", genes=["CXCL13"], direction="down", finding_ids=[f])])
    assert verify(h, findings).status == "rejected"


def test_invalid_citation_and_wrong_cell(findings):
    f = fid(findings, "panel_expression", "Fibroblast|CXCL13|PG|Control")
    h = _hyp([Step(source_cell="Neutrophil", mediator="CXCL13", genes=["CXCL13"], finding_ids=[f, "F9999"])])
    v = verify(h, findings)
    assert v.steps[0].status == "unsupported" and v.steps[0].invalid_finding_ids == ["F9999"]
    kinds = {o.kind for o in verifier_objections(h, v, findings, None)}
    assert {"invalid_citation", "unsupported_step"} <= kinds


def test_overreach_is_partial(study, findings):
    f = fid(findings, "panel_expression", "Fibroblast|CXCL13|PG|Control")
    h = _hyp([Step(source_cell="Fibroblast", mediator="CXCL13 protein secretion", genes=["CXCL13"], direction="up",
                   required_level="protein", finding_ids=[f])])
    v = verify(h, findings)
    assert v.status == "partial"
    objs = verifier_objections(h, v, findings, study.dataset)
    assert any(o.kind == "overreach_level" for o in objs)
    assert any(o.kind == "small_n" for o in objs)


def test_contradiction_found_in_same_contrast(findings):
    # Panel says C3 up in PG fibroblasts; pipeline DE says C3 down in PG vs Control fibroblasts.
    up = fid(findings, "panel_expression", "Fibroblast|C3|PG|Control")
    down = fid(findings, "de", "Fibroblast|C3")
    h = _hyp([Step(source_cell="Fibroblast", mediator="C3", genes=["C3"], direction="up", finding_ids=[up])])
    v = verify(h, findings)
    assert v.steps[0].status == "contradicted" and down in v.steps[0].contradicting_finding_ids


def test_specificity_objections(study, findings):
    f = fid(findings, "panel_expression", "Keratinocyte|S100A8|PG|Control")
    h = _hyp([Step(source_cell="Keratinocyte", mediator="S100A8", genes=["S100A8"], direction="up", finding_ids=[f])],
             specific_to_disease=True)
    objs = verifier_objections(h, verify(h, findings), findings, study.dataset)
    kinds = {o.kind for o in objs}
    assert "not_specific" in kinds  # Sweet vs Control shows the same rise
    assert "missing_evidence" in kinds  # S100A8 is not separated PG vs Sweet
