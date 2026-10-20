import json

from conftest import make_cells
from test_runner import _cfg, _replay

from mechanist import live
from mechanist.llm import Replay
from mechanist.runner import Runner


def _drop(run, name, **msg):
    (run / "inbox").mkdir(parents=True, exist_ok=True)
    (run / "inbox" / name).write_text(json.dumps(msg))


def test_messages_are_numbered_delivered_once_and_read_from_their_stage(tmp_path):
    run = tmp_path / "r"
    run.mkdir()
    _drop(run, "a.json", kind="question", text="Could the IL-24 come from T cells?", by="x@y")
    _drop(run, "b.json", kind="gossip", text="not a kind")  # ignored
    _drop(run, "c.json", kind="fact", text="C5a is raised in wound fluid", source="PMID 123")
    assert [m["id"] for m in live.arrivals(run)] == ["N1", "N2"]
    assert live.arrivals(run) == []  # registered once
    order = ["load", "findings", "circuits"]
    got = live.take(run, "findings")
    assert [(m["id"], m["from_stage"]) for m in got] == [("N1", "findings"), ("N2", "findings")]
    _drop(run, "d.json", kind="steer", text="Rank the epidermal circuit first")
    got = live.take(run, "circuits")
    assert [m["from_stage"] for m in got] == ["findings", "findings", "circuits"]
    assert [m["id"] for m in live.applied(got, "findings", order)] == ["N1", "N2"]
    assert [m["id"] for m in live.applied(got, "circuits", order)] == ["N1", "N2", "N3"]
    assert live.applied(got, "load", order) == []
    text = live.notes_text(got)
    assert "N2 FACT" in text and "PMID 123" in text and "literature" in text and "weighed" in text


def test_replier_answers_each_message_once_and_never_fails_the_run(tmp_path):
    run = tmp_path / "r"
    run.mkdir()
    (run / "stages.json").write_text(json.dumps(["load", "circuits"]))
    _drop(run, "a.json", kind="question", text="Is the interferon arm real?")

    class Fake:
        name = "fake"
        seen = []

        def complete(self, tag, system, user, temperature=None):
            Fake.seen.append(user)
            return json.dumps({"reply": "Not looked at yet; the circuits stage takes it up."}), {}

    from mechanist.llm import LoggedLLM
    r = live.Replier(run, None, "ulcer")
    r._llm = LoggedLLM(Fake(), run / "chat")
    assert r.once() == 1 and r.once() == 0
    m = live.thread(run)[0]
    assert m["reply"].startswith("Not looked at yet") and "MESSAGE N1 (question)" in Fake.seen[0]
    assert "Building the circuits (next)" in Fake.seen[0]

    class Broken:
        name = "broken"

        def complete(self, *a, **k):
            raise RuntimeError("seat limit")
    _drop(run, "b.json", kind="steer", text="Look at fibroblasts")
    r._llm = LoggedLLM(Broken(), run / "chat2")
    assert r.once() == 1
    assert "could not be written" in live.thread(run)[1]["reply"]


def test_live_status_shows_stages_reasoning_and_chat(tmp_path):
    run = tmp_path / "r"
    run.mkdir()
    (run / "stages.json").write_text(json.dumps(["load", "findings", "circuits"]))
    ev = [{"stage": "load", "action": "start", "time": "t1"}, {"stage": "load", "action": "end", "time": "t2"},
          {"stage": "findings", "action": "start", "time": "t3"}, {"stage": "circuits:k1", "action": "pi_gate", "time": "t4"}]
    (run / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ev))
    (run / "transcript.jsonl").write_text(json.dumps({"time": "t5", "role": "hypothesizer", "tag": "x", "text": "IL36G?"}) + "\n")
    s = live.status(run, "p044")
    assert [x["state"] for x in s["stages"]] == ["done", "now", "next"]
    assert s["stage"] == "findings" and s["stage_label"] == "Scanning the data" and s["running"]
    assert s["reasoning"][-1]["text"] == "IL36G?" and s["chat"] == []


def test_a_message_reaches_later_stages_and_a_resume_rereads_it(projects, findings, tmp_path):
    run_dir = tmp_path / "runs" / "m"
    _drop(run_dir, "a.json", kind="steer", text="Weigh the fibroblast source first", by="pi@lab")

    class Recording(Replay):
        users: list = []

        def complete(self, tag, system, user, temperature=None):
            Recording.users.append(user)
            return super().complete(tag, system, user, temperature)
    Runner(_cfg(projects), run_dir, llm_backend=Recording(_replay(findings)), cells=make_cells()).run()
    m = live.thread(run_dir)[0]
    assert m["id"] == "N1" and m["from_stage"] == "load"  # arrived before the run: every stage reads it
    assert any("N1 STEER from pi@lab: Weigh the fibroblast source first" in u for u in Recording.users)
    assert json.loads((run_dir / "stages.json").read_text())[0] == "load"

    calls = (run_dir / "llm_calls.jsonl").read_text().count("\n")
    Runner(_cfg(projects), run_dir, llm_backend=Replay({}), cells=make_cells()).run()  # same notes, same fingerprints
    assert (run_dir / "llm_calls.jsonl").read_text().count("\n") == calls

    _drop(run_dir, "b.json", kind="question", text="Late question")  # the run is done: no stage left to read it
    Runner(_cfg(projects), run_dir, llm_backend=Replay({}), cells=make_cells()).run()
    assert (run_dir / "llm_calls.jsonl").read_text().count("\n") == calls
    assert [x["from_stage"] for x in live.thread(run_dir)] == ["load", None]


def test_the_pi_cannot_pass_circuits_while_a_coherence_problem_stands(projects, tmp_path, monkeypatch):
    from mechanist import agents
    r = Runner(_cfg(projects), tmp_path / "g", llm_backend=Replay({}))
    monkeypatch.setattr(agents, "pi_gate", lambda *a, **k: {"verdict": "pass", "demands": [], "new_tests": 1,
                                                            "new_testable_predictions": 1, "assessment": "fine"})
    problems = [["Circuit 1 has 17 steps (at most 12)"], []]
    acted = []
    reviews = r._gate("ctx", "7. Circuits", lambda: "m", lambda d, n: (acted.append(d), problems.pop(0)), "circuits:k1",
                      blockers=lambda: problems[0])
    assert reviews[0]["verdict"] == "revise" and reviews[0]["blocked_by_code"] == ["Circuit 1 has 17 steps (at most 12)"]
    assert acted[0][0]["text"].startswith("Circuit 1 has 17 steps")
    assert reviews[1]["verdict"] == "pass" and reviews[1]["closed_by"] == "pass"
