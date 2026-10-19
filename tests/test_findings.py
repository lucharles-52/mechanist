from conftest import fid

from mechanist.findings import FindingsConfig, digest, digest_line, expected_chance_separations, p_separation


def test_load_normalises_groups_and_records_files(study):
    ds = study.dataset
    assert ds.groups["PG"] == ["P1", "P2"] and len(ds.groups["Control"]) == 4
    assert study.tables["composition"].df.case.iloc[0] == "PG"
    assert all(len(f.sha256) == 64 for f in ds.files)
    assert any(c.startswith("missing: gsea") for c in ds.caveats)
    assert any(c.startswith("confound:") for c in ds.caveats)


def test_p_separation_exact():
    assert p_separation(2, 4) == 1 / 15
    assert p_separation(5, 4) == 1 / 126


def test_panel_separation_finding(findings):
    f = next(x for x in findings if x.row_key == "Fibroblast|CXCL13|PG|Control")
    assert f.direction == "up" and f.separated and f.p_separation == 1 / 15
    assert f.n_case == 2 and f.n_reference == 4 and f.evidence_level == "transcript"
    assert set(f.per_sample) == {"P1", "P2", "C1", "C2", "C3", "C4"}
    assert any(c.startswith("small_n") for c in f.caveats)


def test_spillover_and_detection_flags(findings):
    sweat = next(x for x in findings if x.row_key == "Sweat gland|IGKC|PG|Control")
    assert any(c.startswith("spillover") and "Plasma cell" in c for c in sweat.caveats)
    ighm = next(x for x in findings if x.row_key == "B cell|IGHM|PG|Control")
    assert any(c.startswith("detection") and "C1" in c for c in ighm.caveats)
    plain = next(x for x in findings if x.row_key == "Fibroblast|CXCL13|PG|Control")
    assert ighm.priority < plain.priority


def test_de_thresholds_and_ids(findings):
    de = [f for f in findings if f.kind == "de"]
    assert {f.feature for f in de} == {"CXCL13", "C3"}  # NOISE fails padj
    assert len({f.id for f in findings}) == len(findings)
    assert fid(findings, "de", "Fibroblast|C3").startswith("F")


def test_digest_and_chance(study, findings):
    cfg = FindingsConfig(digest_quota={"panel_expression": 2, "de": 1})
    d = digest(findings, cfg)
    assert len(d) == 3
    assert "chance" in digest_line(next(f for f in d if f.kind == "panel_expression"))
    chance = expected_chance_separations(cfg, study, lab_tables=True, primary="PG")
    assert chance["PG vs Control"] == round(2 * 25 / 15, 1)
