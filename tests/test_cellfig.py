from mechanist import cellfig


def test_every_step_is_drawn_even_when_no_cell_family_names_an_end():
    edges = [
        {"source": "copy-number gain (one arm)", "target": "ABC1-high transcript per cell", "via": "gene dosage",
         "status": "observed"},
        {"source": "ABC1-high", "target": "DEF2/GHI3 secretory identity", "via": "co-occurrence", "status": "likely"},
        {"source": "ABC1", "target": "JKL4 heterodimer, then downstream kinase", "via": "pairing",
         "status": "speculative"},
        {"source": "Secretory clone", "target": "Fibroblasts", "via": "MNO5", "status": "likely"},
        {"source": "Fibroblasts", "target": "T cells", "via": "CXCL12", "status": "observed"},
        {"source": "Fibroblasts", "target": "PQR6 output", "via": "transcription", "status": "observed"},
        {"source": "T cells", "target": "tissue IFN-γ", "via": "IFNG", "status": "likely"},
    ]
    g = cellfig.build_graph(edges)
    assert all(g["steps"]) and len(g["steps"]) == len(edges)
    assert sorted(i for a in g["arrows"] for i in a["steps"]) == [1, 2, 3, 4, 5, 7]
    assert g["steps"][5] == "cell"  # a cell's product is written under it
    by = {n["label"]: n for n in g["ring"]}
    # the three ends naming the same thing ("ABC1-high transcript per cell", "ABC1-high", "ABC1") share one node
    assert sum(1 for lab in by if lab.startswith("ABC1")) == 1
    assert by["ABC1-high transcript per cell"]["icon"] == "cell"  # an unknown cell: the generic cell, its own name
    assert by["Secretory clone"]["icon"] == "cell"
    assert by["JKL4 heterodimer"]["icon"] == ""  # not a cell: a named box, cut at the clause break
    assert by["Copy-number gain"]["icon"] == ""
    assert by["Fibroblasts"]["icon"] == "fibroblast" and by["Fibroblasts"]["sub"] == ["PQR6"]
    assert g["centre"] == ["tissue IFN-γ"]  # a signal with no cell stays in the centre
    svg = cellfig.figure(g, "t", "test")
    assert svg.count("marker-end=") == len(g["arrows"]) and 'href="#bio-cell"' in svg
    assert 'id="bio-cell"' in cellfig.icon_defs({n["icon"] for n in g["ring"]})


def test_known_cells_draw_as_before():
    g = cellfig.build_graph([{"source": "Keratinocytes", "target": "IL-1α", "via": "", "status": "observed"},
                             {"source": "IL-1α", "target": "Fibroblasts", "via": "IL1R1", "status": "likely"}])
    assert [n["id"] for n in g["ring"]] == ["skin", "fibroblast"]
    assert [(a["from"], a["to"], a["steps"]) for a in g["arrows"]] == [("skin", "fibroblast", [2])]
    assert g["steps"] == ["cell", "arrow"]


def test_ring_labels_are_short():
    assert cellfig._label("A rather long description of a stromal programme; with detail") == "A rather long description…"
    assert cellfig._label("HAS9/XYZ1 gain and SERPX1/PTGX2/FMDX3 loss") == "HAS9/XYZ1 gain…"
    assert cellfig._label("short thing (with a note)") == "Short thing"
