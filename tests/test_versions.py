"""Versions of a finished run: input checks, folders that point to their parent, the diff, the bench grades."""

import json

import pytest

from mechanist import investigate as inv
from mechanist import versions
from mechanist.records import Circuit


def _inputs(tmp_path, body):
    p = tmp_path / "in.yaml"
    p.write_text(body)
    return p


def test_inputs_are_checked_before_anything_is_made(tmp_path):
    good = _inputs(tmp_path, """
by: A. Scientist
results:
  - experiment: E2
    outcome: positive
    found: SIG1 stained in the lesion edge
notes:
  - kind: fact
    text: SIG2 is raised in wound fluid
    source: PMID 1234567
  - kind: question
    text: Could CellB send SIG1?
""")
    paper = {"resultList": {"result": [{"pmid": "1234567", "title": "A paper", "pubYear": "2020", "abstractText": "x"}]}}
    got = versions.load_inputs(good, {"E1", "E2"}, fetch=lambda url: paper)
    assert [x["id"] for x in got["items"]] == ["X1", "X2", "X3"]
    assert got["items"][1]["sourced"] and got["items"][1]["paper"]["title"] == "A paper"
    unsourced = versions.load_inputs(good, {"E2"}, fetch=lambda url: {"resultList": {"result": []}})
    assert unsourced["items"][1]["sourced"] is False
    assert "UNSOURCED" in versions.items_text(unsourced["items"])
    bad = _inputs(tmp_path, """
results:
  - experiment: E9
    outcome: maybe
    files: [missing.png]
""")
    with pytest.raises(versions.InputError) as e:
        versions.load_inputs(bad, {"E1"})
    msg = str(e.value)
    assert "E9 is not in the plan" in msg and "outcome must be" in msg and "missing.png" in msg and "found" in msg


def test_a_version_is_a_new_folder_that_points_to_its_parent(tmp_path):
    run = tmp_path / "runs" / "p1_X_multi_S1"
    run.mkdir(parents=True)
    (run / "state.json").write_text(json.dumps({"report": {"finished": "2026-10-19T10:00:00"}}))
    inp = _inputs(tmp_path, "results:\n  - experiment: E1\n    outcome: negative\n    found: nothing at the edge\n")
    items = versions.load_inputs(inp, {"E1"})
    v2 = versions.make(run, inp, items)
    assert v2.name == "p1_X_multi_S1.v2" and (v2 / "inputs.yaml").exists()
    rec = json.loads((v2 / "version.json").read_text())
    assert rec["version"] == 2 and rec["parent"] == run.name and rec["label"] == "after bench results"
    assert json.loads((v2 / "events.jsonl").read_text().splitlines()[0])["source"] == str(run.resolve())
    notes = _inputs(tmp_path, "notes:\n  - kind: steer\n    text: rank circuit 2 lower\n")
    v3 = versions.make(v2, notes, versions.load_inputs(notes, set()))
    assert v3.name == "p1_X_multi_S1.v3" and json.loads((v3 / "version.json").read_text())["blinded"] is False
    assert [f["version"] for f in versions.family(v3)] == [1, 2, 3] and versions.root_of(v3) == run
    assert versions.info(run)["date"] == "2026-10-19"


def test_the_diff_says_what_moved():
    old = {"circuits": {"circuits": [
        {"rank": 1, "title": "A", "edges": [{"source": "c1", "target": "c2", "status": "likely"}]},
        {"rank": 2, "title": "B", "edges": []}, {"rank": 3, "title": "C", "edges": []}]},
           "therapy": {"hypotheses": [{"drug": "D1", "rank": 1}, {"drug": "D2", "rank": 2}]},
           "experiment": {"experiments": [{"id": "E1", "title": "t1"}]}}
    new = {"circuits": {"circuits": [
        {"rank": 1, "title": "B", "edges": []},
        {"rank": 2, "title": "A", "edges": [{"source": "c1", "target": "c2", "status": "refuted", "ids": ["X1", "R5"]}]},
        {"rank": 3, "title": "N", "edges": []}]},
           "therapy": {"hypotheses": [{"drug": "D2", "rank": 1}, {"drug": "D3", "rank": 2}]},
           "experiment": {"experiments": [{"id": "E1", "title": "t1"}, {"id": "E2", "title": "t2"}]}}
    d = versions.diff(old, new)
    assert d["added"] == [3] and d["dropped"] == [{"rank": 3, "title": "C"}]
    assert {(m["title"], m["from"], m["to"]) for m in d["moved"]} == {("A", 1, 2), ("B", 2, 1)}
    assert d["grades"] == [{"rank": 2, "step": 1, "edge": "c1 → c2", "from": "likely", "to": "refuted", "ids": ["X1"]}]
    assert d["new_experiments"] == ["E2"]
    reworded = {"circuits": {"circuits": [{"rank": 1, "title": "A", "edges": [
        {"source": "c1 cells", "target": "SIG1 protein output", "status": "bench", "ids": ["X1"]}]}]}}
    base = {"circuits": {"circuits": [{"rank": 1, "title": "A", "edges": [
        {"source": "c1 cells", "target": "SIG1 transcript output", "status": "observed"}]}]}}
    assert versions.diff(base, reworded)["grades"][0]["from"] == "observed"  # a reworded step is not a new one
    assert d["drugs"] == {"added": ["D3"], "dropped": ["D1"], "moved": [{"drug": "D2", "from": 2, "to": 1}]}


def test_bench_grades_need_a_result_that_fits_and_a_refuted_step_costs_rank():
    items = [{"id": "X1", "kind": "result", "outcome": "positive"}, {"id": "X2", "kind": "result", "outcome": "negative"},
             {"id": "X3", "kind": "question"}]
    c1 = Circuit.model_validate({"rank": 1, "title": "A", "summary": "s", "edges": [
        {"source": "a", "target": "b", "status": "bench", "ids": ["X1"]},
        {"source": "b", "target": "c", "status": "bench", "ids": ["X2"]},
        {"source": "c", "target": "d", "status": "refuted", "ids": ["X2"]}]})
    objs = inv.check_bench([c1], items, parent=[{"title": "A", "rank": 1}])
    assert [e.status for e in c1.edges] == ["bench", "likely", "refuted"]  # X2 is negative: not a bench result
    assert any("relabelled likely" in o.text for o in objs) and any("keeps its rank" in o.text for o in objs)
    c1.summary = "The refuted step is bypassed by the other arm."
    assert not any("keeps its rank" in o.text for o in inv.check_bench([c1], items, parent=[{"title": "A", "rank": 1}]))
    acc = [{"item": "X1", "verdict": "supports", "reason": "stained"}, {"item": "X2", "verdict": "refutes", "reason": ""}]
    texts = [o.text for o in inv.check_items(acc, items)]
    assert any(t.startswith("X3") for t in texts) and any("X2: give the reason" in t for t in texts)
