"""Display-only rewrite of a run's agent messages into plain team-chat prose (Haiku), for scripts/conversation_view.py.

Nothing in a run reads this: the agents keep their own words, and the chat page shows the rewrite with the original
one click away. A rewrite is shown only if it passes a check against the original reply: it adds no number and no id
that the original lacks, keeps every id of a short reply (in the text or its evidence list), and keeps most of a
short reply's numbers. Otherwise the page shows the message as written and says why.

Its own calls and their cost go to runs/<id>/humanize/ (cache, llm_calls.jsonl); results to runs/<id>/humanized.json.
"""

from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mechanist.agents import HUMANIZE_SYSTEM as SYSTEM

ID_RE = re.compile(r"\b(?:[RFQSLH]\d+(?:\.\d+)?|PMID[:\s]*\d+)\b")
NUM_RE = re.compile(r"(?<![\w.])-?\d+(?:[.,]\d+)?(?=\s*%|\b)")
MAX_TRIES = 2  # a message whose rewrite fails (bad reply, or the check) gets one retry on its own
SHORT = 3000  # a reply this long or shorter must keep all its ids and most of its numbers
KEEP_NUMBERS = 0.9


RANGE_RE = re.compile(r"\b([RFQSLH])(\d+)\s*[-–—]\s*(?:\1)?(\d+)\b")


def _ids(text: str) -> set[str]:
    ids = {re.sub(r"PMID[:\s]*", "PMID ", m) for m in ID_RE.findall(text)}
    for p, a, b in RANGE_RE.findall(text):  # "R5063-R5066" names R5064 and R5065 too
        lo, hi = int(a), int(b)
        if 0 < hi - lo <= 200:
            ids |= {f"{p}{i:0{len(a)}d}" for i in range(lo, hi + 1)}
    return ids


def _nums(text: str) -> set[str]:
    return {n.replace(",", ".") for n in NUM_RE.findall(text) if not re.fullmatch(r"[RFQSLH]?\d", n)}


def is_call(original: str) -> bool:
    """A tool-call step: its numbers are call arguments, which the page shows under Details."""
    head = original[:3000]
    return bool(re.search(r'"tool"\s*:|<invoke name=|<parameter name="(?:args|code)"', head)) \
        and not re.search(r'"answer"\s*:', head)


def check(original: str, text: str, evidence: list[str], known: str = "") -> tuple[bool, str]:
    """Does the rewrite keep the original's ids and numbers, and add none? known: other text the rewriter was shown
    (the speaker's label, e.g. "analyst Q02"), whose ids are not inventions. (ok, reason when not ok)"""
    oi, on = _ids(original) | _ids(known), _nums(original) | _nums(known)
    ri = _ids(text) | _ids(" ".join(evidence))
    rn = _nums(text)
    new_ids = ri - oi
    if new_ids:
        return False, f"the rewrite names ids the original does not: {', '.join(sorted(new_ids)[:5])}"
    new_nums = rn - on - {n for i in oi for n in _nums(i)}
    if new_nums:
        return False, f"the rewrite has numbers the original does not: {', '.join(sorted(new_nums)[:5])}"
    if len(original) <= SHORT:
        lost = _ids(original) - ri
        if lost:
            return False, f"the rewrite drops ids: {', '.join(sorted(lost)[:5])}"
        plain = on - {n for i in oi for n in _nums(i)}
        if plain and not is_call(original) and len(rn & plain) / len(plain) < KEEP_NUMBERS:
            return False, f"the rewrite drops numbers ({len(plain - rn)} of {len(plain)})"
    return True, ""


def turn_key(tag: str, seen: dict[str, int]) -> str:
    """A stable id for a turn: its tag and how many turns with that tag came before it."""
    n = seen.get(tag, 0)
    seen[tag] = n + 1
    return f"{tag}#{n}"


def full_replies(run: Path, rows: list[dict]) -> dict[str, str]:
    """turn key -> the full reply, from the call cache (by the row's request key, else the tag's nth logged call)."""
    by_tag: dict[str, list[str]] = {}
    log = run / "llm_calls.jsonl"
    if log.exists():
        for line in log.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            by_tag.setdefault(r.get("tag", ""), []).append(r.get("request_sha256", ""))
    out, seen = {}, {}
    for r in rows:
        if r.get("role") == "tool":
            continue
        k = turn_key(r.get("tag", ""), seen)
        key = r.get("key") or ""
        if not key and not r.get("replayed"):
            lst = by_tag.get(r.get("tag", ""), [])
            key = lst[int(k.rsplit("#", 1)[1])] if int(k.rsplit("#", 1)[1]) < len(lst) else ""
        p = run / "llm_cache" / f"{key}.json"
        if key and p.exists():
            try:
                out[k] = json.loads(p.read_text())["response"]
            except (json.JSONDecodeError, KeyError):
                pass
    return out


def wanted(r: dict) -> bool:
    """Prose turns only: a tool call already reads as 'Checking: <purpose>'."""
    t = r.get("text", "")
    return r.get("role") not in ("tool", "harness") and not (" → " in t and not t.startswith(
        ("ANSWER", "VERDICT", "MODEL", "THREADS", "PLAN", "LINKAGE")))


class Humanizer:
    def __init__(self, run: Path, workers: int = 4, batch_chars: int = 12000):
        from mechanist.llm import LLMConfig, LoggedLLM, make_backend
        self.run, self.workers, self.batch_chars = run, workers, batch_chars
        self.out_path = run / "humanized.json"
        self.done: dict[str, dict] = json.loads(self.out_path.read_text()) if self.out_path.exists() else {}
        cfg = LLMConfig(backend="anthropic", model="haiku", external_ok=True, effort="", max_tokens=8000)
        (run / "humanize").mkdir(exist_ok=True)
        self.llm = LoggedLLM(make_backend(cfg), run / "humanize")
        self._lock = threading.Lock()

    def pending(self, rows: list[dict]) -> list[tuple[str, dict, str]]:
        full = full_replies(self.run, rows)
        out, seen = [], {}
        for r in rows:
            if r.get("role") == "tool":
                continue
            k = turn_key(r.get("tag", ""), seen)
            prev = self.done.get(k)
            if wanted(r) and (prev is None or (not prev.get("ok") and prev.get("tries", 1) < MAX_TRIES)):
                out.append((k, r, full.get(k) or r.get("text", "")))
        return out

    def _batches(self, items):
        """First tries share a call; a retry goes alone, told why its last rewrite was not used."""
        for it in [x for x in items if x[0] in self.done]:
            yield [it]
        batch, size = [], 0
        for it in [x for x in items if x[0] not in self.done]:
            n = len(it[2])
            if batch and size + n > self.batch_chars:
                yield batch
                batch, size = [], 0
            batch.append(it)
            size += n
        if batch:
            yield batch

    def _one(self, batch) -> None:
        from mechanist.llm import parse_json
        body = "\n\n".join(f"### n={i} · {r.get('role')}\n{orig[:40000]}" for i, (_, r, orig) in enumerate(batch))
        why = (self.done.get(batch[0][0]) or {}).get("why") if len(batch) == 1 else None
        if why:
            body += (f"\n\nYour previous rewrite of this message was not used because {why}. Keep every id (in the "
                     f"text or in evidence) and every number exactly as the original has them.")
        try:
            reply = self.llm.complete("humanize:" + batch[0][0], SYSTEM, body)
            msgs = {int(m.get("n", -1)): m for m in (parse_json(reply) or {}).get("messages", []) if isinstance(m, dict)}
        except Exception as exc:  # noqa: BLE001 - a failed rewrite just leaves the original showing
            msgs, err = {}, str(exc)[:200]
        else:
            err = "no rewrite returned"
        with self._lock:
            for i, (k, _r, orig) in enumerate(batch):
                m = msgs.get(i)
                tries = (self.done.get(k) or {}).get("tries", 0) + 1
                if not m or not str(m.get("text", "")).strip():
                    self.done[k] = {"ok": False, "why": err, "tries": tries}
                    continue
                text, ev = str(m["text"]).strip(), [str(x) for x in m.get("evidence") or []]
                ok, why = check(orig, text, ev, known=str(_r.get("role", "")))
                self.done[k] = {"ok": ok, "text": text, "evidence": ev, "why": why, "tries": tries}

    def update(self, rows: list[dict]) -> int:
        items = self.pending(rows)
        if items:
            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                list(ex.map(self._one, list(self._batches(items))))
            tmp = self.out_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.done))
            tmp.replace(self.out_path)
        return len(items)
