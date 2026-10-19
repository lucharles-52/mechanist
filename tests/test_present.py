"""The interpretation page's check, diagram layout and figure-panel picking (mechanist.present)."""

from mechanist import present

SOURCE = '{"circuits": [{"title": "Cell A signal X drives cell B", "edges": [{"ids": ["R0001"], "note": "X 12.5% in A"}]}]}'
PANELS = {("R0001", "GENEX", "Cell A")}


def page(**kw):
    p = {"question": "Why?", "answer": "Cell A makes X.", "points": [{"label": "Not", "text": "Not Y."}],
         "circuits": [{"rank": 1, "short_title": "A to B", "status": "best supported", "one_line": "A makes X.",
                       "nodes": [{"id": "a", "label": "Cell A", "sub": "", "cell": "Cell A", "kind": "observed", "evidence": ["c1"]},
                                 {"id": "b", "label": "Cell B", "sub": "", "cell": "", "kind": "likely", "evidence": []},
                                 {"id": "o", "label": "Disease", "sub": "", "cell": "", "kind": "outcome", "evidence": []}],
                       "edges": [{"from": "a", "to": "b", "status": "observed"}, {"from": "b", "to": "o", "status": "likely"}],
                       "cards": [{"id": "c1", "title": "A makes X", "why": "Shown.",
                                  "panels": [{"result": "R0001", "gene": "GENEX", "lineage": "Cell A"}]}]}],
         "fit": [{"feature": "Ulcer", "items": ["D01"], "verdicts": ["explains"]}], "tests": [], "checked": []}
    p.update(kw)
    return p


def test_clean_page_passes():
    assert present.check(page(), SOURCE, PANELS, 1, {"D01"}) == []


def test_new_numbers_ids_and_panels_are_caught():
    p = page(answer="Cell A makes X in 40% of cells (R0099).")
    p["circuits"][0]["cards"][0]["panels"][0]["gene"] = "GENEY"
    errs = " ".join(present.check(p, SOURCE, PANELS, 1, {"D01"}))
    assert "40" in errs and "R0099" in errs and "GENEY" in errs


def test_missing_circuit_bad_vocab_and_unknown_items():
    p = page(fit=[{"feature": "Ulcer", "items": ["D09"], "verdicts": ["maybe"]}])
    p["circuits"][0]["status"] = "proven"
    errs = " ".join(present.check(p, SOURCE, PANELS, 2, {"D01"}))
    assert "2" in errs and "status" in errs and "D09" in errs and "per circuit" in errs


def test_layout_layers_left_to_right_and_puts_outcome_last():
    c = page()["circuits"][0]
    pos = present.layout(c["nodes"], c["edges"] + [{"from": "o", "to": "a", "loop": True}])
    assert pos["a"][0] < pos["b"][0] < pos["o"][0]
    svg = present.diagram(c, {"c1": 1})
    assert svg.startswith("<svg") and 'href="#ev1"' in svg


def test_step_panels_need_the_cell_type_named():
    pdata = {("R1", "GENEX", "B cell"): {"s": 1.0}, ("R1", "GENEX", "Macrophage"): {"s": 2.0}}
    edge = {"source": "Macrophages", "target": "fibroblast", "via": "GENEX", "note": "", "ids": ["R1"]}
    assert present._step_panels(edge, pdata) == ["R1|GENEX|Macrophage"]


def test_panel_data_reads_per_sample_results():
    res = {"R1": {"ok": True, "data": {"G": {"L": {"s1": 1.5, "s2": None}}}}, "R2": {"ok": True, "data": {"x": 1}}}
    assert present.panel_data(res) == {("R1", "G", "L"): {"s1": 1.5, "s2": None}}


def test_single_sample_cards_must_say_so():
    gs = [("PG", ["PGa", "PGb"]), ("Ctrl", ["Ca", "Cb"])]
    assert present.holds_in({"PGa": 9, "PGb": 1, "Ca": 2, "Cb": 3}, gs) == ["PGa"]
    assert present._pattern({"PGa": 9, "PGb": 8, "Ca": 2, "Cb": 3}, gs) == "holds in every disease sample"
    pats = {("R0001", "GENEX", "Cell A"): present._pattern({"PGa": 9, "PGb": 1, "Ca": 2, "Cb": 3}, gs)}
    errs = present.check(page(), SOURCE, PANELS, 1, {"D01"}, pats)
    assert any("only some disease samples" in e for e in errs)
    p = page()
    p["circuits"][0]["cards"][0]["why"] = "Shown in one of two samples."
    assert present.check(p, SOURCE, PANELS, 1, {"D01"}, pats) == []
    p["circuits"][0]["cards"][0]["why"] = "Seen in PGa."
    assert present.check(p, SOURCE, PANELS, 1, {"D01"}, pats) == []


def test_plain_rewrite_must_keep_numbers_names_and_ids():
    orig = "Fibroblast CXCL13 is 12.5% in PG vs 1.2% in Sweet (R0001); IL-1α is not raised."
    assert present.plain_check(orig, "In PG, 12.5% of fibroblasts make CXCL13 against 1.2% in Sweet (R0001). "
                                     "IL-1α is not raised.") == ""
    assert "numbers" in present.plain_check(orig, "Fibroblast CXCL13 is 13% in PG (R0001); IL-1α is not raised, 1.2%.")
    assert "names" in present.plain_check(orig, "Fibroblasts make a chemokine: 12.5% in PG vs 1.2% in Sweet (R0001); "
                                                "IL-1α is not raised.")
    assert "ids" in present.plain_check(orig, "Fibroblast CXCL13 is 12.5% in PG vs 1.2% in Sweet; IL-1α is not raised.")
