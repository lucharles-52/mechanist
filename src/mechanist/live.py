"""Talking to a run while it goes (docs/LIVE.md): messages in, replies out, notes read from the next stage on, and
the live picture the portal shows.

    runs/<run>/inbox/<time>.json   a message, as the relay copied it from Box (kind, text, source, by, at)
    runs/<run>/chat.jsonl          the conversation: message, reply and delivered records, appended in order
    runs/<run>/stages.json         the stages this run goes through, in order (written when the run starts)

Standard library only at import (the relay runs with the system python). The reply thread imports the model code
when it starts.
"""

from __future__ import annotations

import datetime as dt
import fcntl
import json
import threading
from contextlib import contextmanager
from pathlib import Path

KINDS = ("question", "fact", "steer")
WORDS = {"load": "Loading the study", "findings": "Scanning the data", "dossier": "Reading what is known of the disease",
         "check": "Do we see the pathology?", "panel": "Assembling the panel", "generate": "First explanations",
         "candidates": "Choosing candidate explanations", "verify_r0": "Checking the candidates' claims",
         "scout": "Searching the literature", "investigate": "Testing each candidate", "synthesize": "Weighing the evidence",
         "critique": "The PI's critique", "chase": "Chasing the open steps", "linkage": "Linking the explanations",
         "revise": "Revising", "circuits": "Building the circuits", "experiment": "Planning experiments",
         "cards": "Writing mechanism cards", "therapy": "Treatments aimed at the circuits", "report": "Writing the report",
         "refine": "Reading new input"}


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


@contextmanager
def _locked(run: Path):
    with open(run / ".chat.lock", "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _records(run: Path) -> list[dict]:
    p = run / "chat.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def _append(run: Path, rec: dict) -> None:
    with (run / "chat.jsonl").open("a") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def thread(run: Path) -> list[dict]:
    """The messages in order, each with its reply and the stage that first read it (None until one does)."""
    msgs: dict[str, dict] = {}
    for r in _records(run):
        if r.get("type") == "message":
            msgs[r["id"]] = {k: r.get(k) for k in ("id", "kind", "text", "source", "by", "at", "file")} | {
                "reply": None, "replied_at": None, "from_stage": None}
        elif r.get("id") in msgs and r.get("type") == "reply":
            msgs[r["id"]].update(reply=r.get("text"), replied_at=r.get("at"))
        elif r.get("id") in msgs and r.get("type") == "delivered":
            msgs[r["id"]]["from_stage"] = r.get("stage")
    return list(msgs.values())


def arrivals(run: Path) -> list[dict]:
    """Register messages the relay has dropped into inbox/ since last time; return the new ones, numbered N1, N2, ..."""
    box = run / "inbox"
    if not box.is_dir():
        return []
    new = []
    with _locked(run):
        seen = {m.get("file") for m in thread(run)}
        n = len(seen)
        for f in sorted(box.glob("*.json")):
            if f.name in seen:
                continue
            try:
                d = json.loads(f.read_text())
            except (ValueError, OSError):
                continue
            kind = str(d.get("kind") or "").lower()
            text = str(d.get("text") or "").strip()
            if kind not in KINDS or not text:
                continue
            n += 1
            rec = {"type": "message", "id": f"N{n}", "kind": kind, "text": text[:4000],
                   "source": str(d.get("source") or "").strip()[:500], "by": str(d.get("by") or ""),
                   "at": str(d.get("at") or _now()), "file": f.name}
            _append(run, rec)
            new.append(rec)
    return new


def take(run: Path, stage: str) -> list[dict]:
    """A stage is about to run: deliver every message not yet delivered to it. Returns every delivered message."""
    arrivals(run)
    with _locked(run):
        for m in thread(run):
            if m["from_stage"] is None:
                _append(run, {"type": "delivered", "id": m["id"], "stage": stage, "at": _now()})
    return [m for m in thread(run) if m["from_stage"]]


def applied(msgs: list[dict], stage: str, order: list[str]) -> list[dict]:
    """The messages a stage reads: those delivered to it or to an earlier stage of the run."""
    if stage not in order:
        return []
    i = order.index(stage)
    return [m for m in msgs if m.get("from_stage") in order and order.index(m["from_stage"]) <= i]


def notes_text(msgs: list[dict]) -> str:
    if not msgs:
        return ""
    out = ["Messages from the study's scientists, sent while this run was going (cite them by N id). A fact is literature, "
           "never data; a steer is weighed against the evidence and followed only when the evidence allows; a question is "
           "answered, with the data tools where it bears on your task:"]
    for m in msgs:
        src = f" (source given: {m['source']})" if m.get("source") else (" (no source given)" if m["kind"] == "fact" else "")
        out.append(f"- {m['id']} {m['kind'].upper()} from {m.get('by') or 'a scientist'}: {m['text']}{src}")
    return "\n".join(out)


def fingerprint_part(msgs: list[dict]) -> list:
    return [[m["id"], m["kind"], m["text"], m.get("source") or ""] for m in msgs]


# ---- the live picture ---------------------------------------------------------------------------------------------
def _events(run: Path) -> list[dict]:
    p = run / "events.jsonl"
    out = []
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def _tail(path: Path, n: int, size: int = 400_000) -> list[dict]:
    if not path.exists():
        return []
    with path.open("rb") as fh:
        start = max(0, path.stat().st_size - size)
        fh.seek(start)
        lines = fh.read().decode("utf-8", "replace").splitlines()[1 if start else 0:]  # a cut first line is dropped
    out = []
    for line in lines[-n:]:
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def stages(run: Path) -> list[dict]:
    """Each stage of the run: done, under way or to come, with its times."""
    evs = _events(run)
    if (run / "stages.json").exists():
        order = json.loads((run / "stages.json").read_text())
    else:  # a run started before stages.json: the stages it has begun, in order
        order = list(dict.fromkeys(str(e["stage"]) for e in evs if e.get("action") in ("start", "end", "skip")))
    seen: dict[str, dict] = {}
    for e in evs:
        name = str(e.get("stage") or "")
        if name not in order:
            continue
        s = seen.setdefault(name, {"started": None, "finished": None})
        if e.get("action") == "start":
            s["started"], s["finished"] = e.get("time"), None
        elif e.get("action") in ("end", "skip"):
            s["started"] = s["started"] or e.get("time")
            s["finished"] = e.get("time")
    out = []
    for name in order:
        s = seen.get(name) or {}
        state = "done" if s.get("finished") else "now" if s.get("started") else "next"
        out.append({"name": name, "label": WORDS.get(name, name), "state": state, "started": s.get("started"),
                    "finished": s.get("finished")})
    return out


def reasoning(run: Path, n: int = 14) -> list[dict]:
    """The last turns of the run's conversation (agents and tools), as the Reasoning view shows them."""
    out = []
    for r in _tail(run / "transcript.jsonl", n):
        who = r.get("role") or r.get("kind") or ""
        out.append({"time": r.get("time"), "who": who, "tag": r.get("for") or r.get("tag") or "",
                    "text": str(r.get("text") or "")[:900]})
    return out


def status(run: Path, study: str = "", running: bool = True) -> dict:
    st = stages(run)
    now = next((s for s in st if s["state"] == "now"), None)
    finished = (run / "site" / "index.html").exists() or (bool(st) and all(s["state"] == "done" for s in st))
    return {"run": run.name, "study": study, "updated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "running": running and not finished, "finished": finished,
            "stage": now and now["name"], "stage_label": now and now["label"], "stage_started": now and now["started"],
            "stages": st, "reasoning": reasoning(run), "chat": thread(run)}


# ---- replies ------------------------------------------------------------------------------------------------------
def _json(run: Path, name: str) -> dict:
    try:
        return json.loads((run / f"{name}.json").read_text())
    except (OSError, ValueError):
        return {}


def digest(run: Path) -> str:
    """What the run has established so far, for the reply: stages done, the candidates and circuits as they stand,
    and what the agents are doing now."""
    st = stages(run)
    lines = ["STAGES: " + "; ".join(f"{s['label']} ({s['state']})" for s in st)]
    cand = _json(run, "candidates").get("hypotheses") or []
    if cand:
        lines.append("CANDIDATE EXPLANATIONS:")
        lines += [f"- {h.get('id')}: {h.get('title')}" for h in cand[:12] if isinstance(h, dict)]
    circ = (_json(run, "circuits").get("circuits") or [])
    if circ:
        lines.append("CIRCUITS:")
        lines += [f"- {c.get('rank')}. {c.get('title')}: {str(c.get('summary') or '')[:500]}" for c in circ if isinstance(c, dict)]
    exp = _json(run, "experiment").get("experiments") or []
    if exp:
        lines.append("PLANNED EXPERIMENTS: " + "; ".join(f"{e.get('id')} {e.get('title')}" for e in exp[:12] if isinstance(e, dict)))
    rs = reasoning(run, 10)
    if rs:
        lines.append("LATEST TURNS (newest last):")
        lines += [f"- [{r['tag']}] {r['who']}: {r['text'][:500]}" for r in rs]
    return "\n".join(lines)


class Replier(threading.Thread):
    """Watches inbox/ beside the stages and answers each message within minutes, from what the run has so far."""

    def __init__(self, run: Path, llm_cfg, phenotype: str = "", every: float = 15.0):
        super().__init__(daemon=True, name="mechanist-replier")
        self.run_dir, self.llm_cfg, self.phenotype, self.every = run, llm_cfg, phenotype, every
        self.stop = threading.Event()
        self._llm = None

    def llm(self):
        if self._llm is None:
            from mechanist.llm import LoggedLLM, make_backend
            (self.run_dir / "chat").mkdir(exist_ok=True)
            self._llm = LoggedLLM(make_backend(self.llm_cfg), self.run_dir / "chat")
        return self._llm

    def reply(self, m: dict) -> str:
        from mechanist import agents
        st = stages(self.run_dir)
        nxt = next((s["label"] for s in st if s["state"] == "next"), "the report")
        now = next((s["label"] for s in st if s["state"] == "now"), "")
        user = agents.REPLY_USER.format(phenotype=self.phenotype, digest=digest(self.run_dir), now=now or "between stages",
                                        next=nxt, id=m["id"], kind=m["kind"], by=m.get("by") or "a scientist",
                                        text=m["text"], source=m.get("source") or "(none)")
        text = agents._ask(self.llm(), f"chat:{m['id']}", agents.REPLY_SYSTEM, user,
                           lambda d: str(d["reply"]).strip())
        return text or f"Received as {m['id']}. The next stage ({nxt}) reads it."

    def once(self) -> int:
        arrivals(self.run_dir)
        n = 0
        for m in thread(self.run_dir):
            if m["reply"] is None:
                try:
                    text = self.reply(m)
                except Exception as exc:  # a failed reply must never stop the run
                    text = f"Received as {m['id']}; the reply could not be written ({str(exc)[:200]}). The next stage reads it."
                with _locked(self.run_dir):
                    _append(self.run_dir, {"type": "reply", "id": m["id"], "text": text, "at": _now()})
                n += 1
        return n

    def run(self) -> None:
        while not self.stop.is_set():
            try:
                self.once()
            except Exception:
                pass
            self.stop.wait(self.every)

