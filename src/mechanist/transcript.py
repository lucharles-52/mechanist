"""The run as a conversation: one line per agent turn and per tool result, written as it happens.

runs/<id>/transcript.jsonl holds, in order, who said what: the planner choosing threads, each analyst's steps (its
stated purpose, the tool it called, the result), its answer, the PI's verdicts and demands, the synthesis, the disease
linkage and the experiment plan. It is for reading and auditing; nothing in the run reads it back.
`scripts/transcript_view.py` follows it live.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

# the tag of the agent whose tool calls run on this thread now (set by LoggedLLM; tools run right after the reply)
current = threading.local()

# tag prefix -> role (the first that matches wins); a tool step inside a thread is "analyst Qnn"
ROLES = [
    (r"^(dossier|check|candidates|linkage|experiment|final|circuits):gate\d+", "PI"),
    (r"^critique_model", "PI"),
    (r"^dossier:r\d+", "disease scholar"),
    (r"^(check|investigate|chase\d*|final\d*):plan\d*", "planner"),
    (r"^check:plan", "planner"),
    (r"^check:summary", "summariser"),
    (r"^panel:", "panel"),
    (r"^(generate|merge)", "PI (candidates)"),
    (r"^candidates:r\d+", "PI (candidates)"),
    (r"^(synthesize|revise)", "synthesizer"),
    (r"^scout_", "literature scout"),
    (r"^linkage\d*:L\d+", "linkage scholar"),
    (r"^experiment:r\d+", "experimentalist"),
    (r"^circuits:(k\d+:)?r\d+", "hypothesizer"),
    (r"^circuits:consensus", "hypothesizer (consensus)"),
]


def role_of(tag: str) -> str:
    m = re.search(r":(Q\d+):s\d+$", tag)
    if m:
        return f"analyst {m.group(1)}"
    m = re.match(r"^linkage\d*:(L\d+)", tag)
    if m:
        return f"linkage scholar {m.group(1)}"
    m = re.match(r"^panel:([^:]+)", tag)
    if m:
        return f"panel: {m.group(1)}"
    for pat, role in ROLES:
        if re.match(pat, tag):
            return role
    return "agent"


def _clip(text: str, n: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def summarize(reply: str) -> str:
    """A readable line for one model reply: a tool call, an answer, a verdict, a model, or the text itself."""
    try:
        from mechanist.llm import parse_json
        d = parse_json(reply)
    except (ValueError, TypeError):
        return _clip(reply, 600)
    if not isinstance(d, dict):
        return _clip(json.dumps(d), 400)
    if d.get("tool"):
        args = json.dumps(d.get("args") or {})
        return _clip(f"{d.get('purpose', '')} → {d['tool']}({args})", 600)
    if isinstance(d.get("answer"), dict):
        a = d["answer"]
        if a.get("answer"):
            return _clip(f"ANSWER ({a.get('status', '')}): {a['answer']}", 900)
        if a.get("experiments"):
            return _clip("PLAN: " + " | ".join(f"P{e.get('priority')} {e.get('link', '')}" for e in a["experiments"]), 900)
        if a.get("findings") is not None:
            return _clip(f"LINKAGE: {a.get('suggests', '')} ({len(a.get('findings') or [])} findings)", 900)
        return _clip("ANSWER: " + json.dumps(a), 900)
    if "verdict" in d:
        dem = d.get("demands") or d.get("questions") or []
        text = d.get("assessment") or "; ".join(str(o.get("text", "")) for o in (d.get("objections") or [])[:2])
        first = "; ".join(_clip(x.get("text", ""), 160) for x in dem[:3] if isinstance(x, dict))
        return _clip(f"VERDICT {d['verdict']}: {text} | {len(dem)} demands" + (f": {first}" if first else ""), 1200)
    if d.get("title") and "summary" in d:
        bg = d.get("best_guesses") or []
        return _clip(f"MODEL: {d['title']} — {d.get('summary', '')}" + (f" | {len(bg)} best-guess steps" if bg else ""),
                     1200)
    if d.get("threads") is not None:
        return _clip("THREADS: " + " | ".join(t.get("text", "") for t in d["threads"] if isinstance(t, dict)), 1200)
    if d.get("hypotheses") is not None:
        return _clip("HYPOTHESES: " + " | ".join(h.get("title", "") for h in d["hypotheses"] if isinstance(h, dict)),
                     1200)
    return _clip(json.dumps(d), 600)


class Transcript:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def _write(self, rec: dict) -> None:
        rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **rec}
        with self._lock, self.path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")

    def turn(self, tag: str, reply: str, cached: bool = False, model: str = "", key: str = "") -> None:
        """key: the call's request hash; the full reply is llm_cache/<key>.json (for readers that want all of it)."""
        current.tag = tag
        self._write({"role": role_of(tag), "tag": tag, "kind": "turn", "text": summarize(reply),
                     **({"replayed": True} if cached else {}), **({"model": model} if model else {}),
                     **({"key": key} if key else {})})

    def note(self, text: str) -> None:
        """A harness event worth seeing alongside the conversation (e.g. a switch to another Claude account)."""
        self._write({"role": "harness", "tag": "", "kind": "note", "text": _clip(text, 500)})

    def tool(self, result) -> None:
        first = (result.summary or "").strip().splitlines()[:2]
        self._write({"role": "tool", "tag": result.id, "kind": "result", "for": getattr(current, "tag", ""),
                     "text": _clip(f"[{result.id}] {result.tool}: " + " ".join(first) if result.ok
                                   else f"[{result.id}] {result.tool} FAILED: {result.summary}", 500)})
