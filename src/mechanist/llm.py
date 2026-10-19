"""LLM backends. Only agents.py calls these.

- openai_compat: any OpenAI-compatible server (the pipeline's local vLLM). Stdlib HTTP, like Projects/scripts/llm_writeup.py.
- claude_cli: Claude through the Claude Code CLI in headless mode (`claude -p`, subscription login); external.
- replay: canned responses by call tag, for tests and deterministic reruns.

Data guard: unless `external_ok` is set, the openai_compat backend refuses any host that is not loopback, so findings
never leave the machine by accident. Every call is cached by request hash (resume) and logged (audit).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse

from pydantic import BaseModel

LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


class LLMConfig(BaseModel):
    backend: str = "openai_compat"  # openai_compat | claude | claude_cli | anthropic | replay
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = ""
    temperature: float = 0.2
    max_tokens: int = 6000
    timeout: int = 3600
    extra: dict = {}  # extra request body, e.g. chat_template_kwargs
    json_mode: bool = True
    external_ok: bool = False
    replay_file: str | None = None
    reuse_sessions: bool = True  # claude_cli: continue one session per thread (prompt caching) instead of re-sending
    # claude_cli: model for thread tool steps (tags ending ":s<n>"); planning, PI reviews, candidates and synthesis
    # keep `model`. Empty: every call uses `model`.
    thread_model: str = ""
    # anthropic (direct Messages API): key file (or ANTHROPIC_API_KEY), thinking effort for judgement calls
    # ("" = no thinking) and for thread steps, and the TTL of the cached study context shared across calls
    api_key_file: str = "~/.config/claude/api_key"
    # server-side fallback (beta): a classifier refusal (e.g. a false-positive "bio" flag on biomedical text) is
    # retried by the API on the model Anthropic recommends for that category; "" disables it
    fallbacks: str = "default"
    effort: str = "medium"
    thread_effort: str = ""
    context_ttl: str = "1h"
    # anthropic: thinking effort for one stage's calls, by tag prefix (the hypothesizer thinks hardest)
    effort_overrides: dict[str, str] = {"circuits": "high"}
    # claude (any Claude account): the accounts to use, in order; a call moves to the next one when the current one
    # is out of usage or credit, and goes back to an earlier one after it has rested. "key:<file>" is an API key file
    # (a glob adds every match, sorted, e.g. key:~/.config/claude/api_key_*); "subscription" is the Claude login
    # (claude -p with ~/.config/claude/oauth_token, not billed per call).
    # "subscription:<glob>" adds pooled member seats (Team plan), one token file each, e.g.
    # subscription:~/.config/claude/oauth_token_*
    accounts: list[str] = ["key:~/.config/claude/api_key", "key:~/.config/claude/api_key_*", "subscription"]


class Backend(Protocol):
    name: str

    def complete(self, tag: str, system: str, user: str, temperature: float | None = None) -> tuple[str, dict]: ...

    def chat(self, tag: str, messages: list[dict], temperature: float | None = None) -> tuple[str, dict]: ...


class DataGuardError(RuntimeError):
    pass


class OpenAICompat:
    def __init__(self, cfg: LLMConfig):
        host = urlparse(cfg.base_url).hostname or ""
        if host not in LOCAL_HOSTS and not cfg.external_ok:
            raise DataGuardError(f"LLM host {host!r} is not local; set llm.external_ok only if findings may leave Betty")
        if not cfg.model:
            raise ValueError("llm.model is required")
        self.cfg = cfg
        self.name = cfg.model

    def complete(self, tag: str, system: str, user: str, temperature: float | None = None) -> tuple[str, dict]:
        return self.chat(tag, [{"role": "system", "content": system}, {"role": "user", "content": user}], temperature)

    def chat(self, tag: str, messages: list[dict], temperature: float | None = None) -> tuple[str, dict]:
        cfg = self.cfg
        payload = {"model": cfg.model, "messages": messages,
                   "max_tokens": cfg.max_tokens,
                   "temperature": cfg.temperature if temperature is None else temperature, **cfg.extra}
        if cfg.json_mode:
            payload["response_format"] = {"type": "json_object"}
        for attempt in range(2):
            req = urllib.request.Request(cfg.base_url.rstrip("/") + "/chat/completions",
                                         data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=cfg.timeout) as r:
                    body = json.load(r)
                return body["choices"][0]["message"].get("content") or "", body.get("usage") or {}
            except urllib.error.HTTPError as exc:
                if attempt == 0 and "response_format" in payload:  # some servers refuse JSON mode
                    payload.pop("response_format")
                    continue
                raise RuntimeError(f"{cfg.model} HTTP {exc.code}: {exc.read().decode(errors='replace')[-600:]}") from None
        return "", {}


class Replay:
    """Responses keyed by tag; a list is consumed in order. Tags: generate, critique:H1, revise:H1, single."""

    name = "replay"

    def __init__(self, responses: dict[str, str | list[str]]):
        self.responses = {k: list(v) if isinstance(v, list) else [v] for k, v in responses.items()}

    def complete(self, tag: str, system: str, user: str, temperature: float | None = None) -> tuple[str, dict]:
        return self.chat(tag, [], temperature)

    def chat(self, tag: str, messages: list[dict], temperature: float | None = None) -> tuple[str, dict]:
        """Longest matching tag prefix wins: 'thread:Q01:s2' -> 'thread:Q01:s2', 'thread:Q01', 'thread'."""
        parts = tag.split(":")
        queue = next((self.responses[k] for k in (":".join(parts[:i]) for i in range(len(parts), 0, -1))
                      if self.responses.get(k)), None)
        if not queue:
            raise KeyError(f"replay has no response for {tag!r}")
        return (queue.pop(0) if len(queue) > 1 else queue[0]), {}


class ClaudeCLI:
    """Claude through the Claude Code CLI in headless mode (`claude -p`), using the account's subscription login
    (CLAUDE_CODE_OAUTH_TOKEN or `claude /login`), with every tool, MCP server and settings file disabled, so it is a
    plain text model like the local backend. Data leaves the cluster: requires external_ok.

    Multi-turn calls (investigation threads) continue one CLI session per thread (`--resume`): each step sends only
    the new message, and Claude Code caches the earlier turns, instead of re-sending (and re-writing to the prompt
    cache) the whole transcript every step. A step with no session to continue (the first one, or one after cached
    steps were replayed) is sent as one transcript and starts the session."""

    def __init__(self, cfg: LLMConfig, auth: str = "", token_file: str = ""):
        import shutil
        if not cfg.external_ok:
            raise DataGuardError("claude_cli sends prompts to Anthropic; set llm.external_ok: true to allow it")
        self.cfg = cfg
        self.auth = auth  # "subscription": always the Claude login; "": as the launcher chose (MECHANIST_USE_API_KEY)
        self.token_file = token_file  # a pooled member seat: its token, used whatever the environment holds
        self.exe = shutil.which("claude") or str(Path.home() / ".local" / "bin" / "claude")
        if not Path(self.exe).exists():
            raise ValueError("claude CLI not found (install: curl -fsSL https://claude.ai/install.sh | bash)")
        self.name = f"claude_cli:{cfg.model or 'default'}" + (f"+threads:{cfg.thread_model}" if cfg.thread_model else "")
        self._sessions: dict[str, str] = {}  # hash of the messages so far (ending in our reply) -> session id
        self._spent: dict[str, float] = {}  # session id -> its cumulative cost (the CLI reports a running total)
        self._slock = __import__("threading").Lock()
        self._home: Path | None = None  # one empty working folder for all thread sessions (resume needs the same)

    def complete(self, tag: str, system: str, user: str, temperature: float | None = None) -> tuple[str, dict]:
        return self.chat(tag, [{"role": "system", "content": system}, {"role": "user", "content": user}], temperature)

    @staticmethod
    def render(messages: list[dict]) -> tuple[str, str]:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        turns = [m for m in messages if m["role"] != "system"]
        if len(turns) == 1:
            return system, turns[0]["content"]
        body = "\n\n".join(f"<{m['role']}>\n{m['content']}\n</{m['role']}>" for m in turns)
        return system, (f"The conversation so far:\n\n{body}\n\nYou are the assistant. Reply to the last user message "
                        f"only, in exactly the format the instructions ask for.")

    @staticmethod
    def _key(messages: list[dict]) -> str:
        return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()

    def _session_home(self) -> Path:
        import atexit
        import shutil
        import tempfile
        with self._slock:
            if self._home is None:
                self._home = Path(tempfile.mkdtemp(prefix="mechanist-claude-threads-"))
                key = re.sub(r"[^A-Za-z0-9]", "-", str(self._home))
                store = Path.home() / ".claude" / "projects" / key  # the sessions' transcripts (home has an inode limit)
                atexit.register(lambda: (shutil.rmtree(store, ignore_errors=True),
                                         shutil.rmtree(self._home, ignore_errors=True)))
            return self._home

    def chat(self, tag: str, messages: list[dict], temperature: float | None = None) -> tuple[str, dict]:
        step = bool(re.search(r":s\d+$", tag))  # a tool step inside a thread
        model = self.cfg.thread_model if step and self.cfg.thread_model and not _judgement(tag) else self.cfg.model
        # the same thinking effort as the API route: a stage's override (circuits: high) wins over the default
        effort = (self.cfg.thread_effort if step and not _judgement(tag) else
                  self.cfg.effort_overrides.get(tag.split(":")[0], self.cfg.effort))
        if not self.cfg.reuse_sessions or not step:  # not a thread step: one-off call
            text, usage = self._call(messages, None, None, model=model, effort=effort,
                                     max_output=LONG_OUTPUT.get(tag, 0))
            usage["model"] = model
            return text, usage
        home = self._session_home()
        sid = self._sessions.get(self._key(messages[:-1])) if messages[-1]["role"] == "user" else None
        text = None
        if sid:
            try:
                text, usage, new_sid = self._call(messages, sid, home, model=model, effort=effort)
            except RuntimeError as exc:
                if re.search(r"credit balance|limit.*resets", str(exc), re.IGNORECASE):
                    raise
                sid = None  # the session is gone or unusable: start again from the whole transcript
        if text is None:
            text, usage, new_sid = self._call(messages, None, home, persist=True, model=model, effort=effort)
        if new_sid:
            with self._slock:
                self._sessions[self._key(messages + [{"role": "assistant", "content": text}])] = new_sid
                total = usage.get("total_cost_usd") or 0.0
                if sid:  # this step's own cost, not the session's running total
                    usage["session_cost_usd"] = total
                    usage["total_cost_usd"] = round(total - self._spent.get(sid, 0.0), 6)
                self._spent[new_sid] = total
        usage["resumed"], usage["model"] = bool(sid), model
        return text, usage

    def _call(self, messages: list[dict], sid: str | None, home: Path | None, persist: bool = False,
              model: str | None = None, effort: str = "", max_output: int = 0):
        """One CLI call. sid: continue that session with only the last message. persist: keep the new session so the
        next step can continue it. Returns (text, usage) for a one-off call, else (text, usage, session_id)."""
        import os
        import subprocess
        import tempfile
        system, prompt = self.render(messages)
        if sid:
            prompt = messages[-1]["content"]
        session = home is not None
        tmpdir = None if session else tempfile.TemporaryDirectory(prefix="mechanist-claude-")
        cwd = home if session else Path(tmpdir.name)  # empty cwd: no CLAUDE.md, no memory
        try:
            sp = Path(tempfile.mkstemp(prefix="system-", suffix=".txt", dir=cwd)[1])
            sp.write_text(system)
            cmd = [self.exe, "-p", "--output-format", "json", "--system-prompt-file", str(sp), "--tools", "",
                   "--strict-mcp-config", "--setting-sources", ""]
            if sid:
                cmd += ["--resume", sid]
            elif not (session and persist):
                cmd += ["--no-session-persistence"]
            if model or self.cfg.model:
                cmd += ["--model", model or self.cfg.model]
            if effort:
                cmd += ["--effort", effort]
            env = {**os.environ, "DISABLE_AUTOUPDATER": "1", "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1"}
            env.pop("CLAUDE_EFFORT", None)  # a parent session's effort (e.g. the desktop app's) must not leak in
            if self.auth == "subscription":
                env.pop("ANTHROPIC_API_KEY", None)
                tok = Path.home() / ".config" / "claude" / "oauth_token"
                if self.token_file:
                    env["CLAUDE_CODE_OAUTH_TOKEN"] = Path(self.token_file).read_text().strip()
                elif not env.get("CLAUDE_CODE_OAUTH_TOKEN") and tok.exists():
                    env["CLAUDE_CODE_OAUTH_TOKEN"] = tok.read_text().strip()
                for k in [k for k in env if k in ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "CLAUDECODE")
                          or (k.startswith("CLAUDE_CODE_") and k not in ("CLAUDE_CODE_DISABLE_AUTO_MEMORY",
                                                                       "CLAUDE_CODE_OAUTH_TOKEN"))]:
                    env.pop(k, None)  # never route through a parent Claude Code session
            elif os.environ.get("MECHANIST_USE_API_KEY") == "1":  # an API key chosen on purpose by the launcher
                env.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
                # a parent Claude Code session (e.g. the desktop app) exports its own endpoint and account; inherited,
                # they would route these calls through that session instead of to the API with this key
                for k in [k for k in env if k == "ANTHROPIC_BASE_URL" or k == "ANTHROPIC_AUTH_TOKEN" or k == "CLAUDECODE"
                          or (k.startswith("CLAUDE_CODE_") and k not in ("CLAUDE_CODE_DISABLE_AUTO_MEMORY",))]:
                    env.pop(k, None)
            else:
                env.pop("ANTHROPIC_API_KEY", None)  # use the subscription login, never a stray key
            if max_output:  # set after the clean-up above, which drops every inherited CLAUDE_CODE_* setting
                env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(max_output)
            for attempt in range(3):
                p = subprocess.run(cmd, check=False, input=prompt, capture_output=True, text=True, cwd=cwd, env=env,
                                   timeout=self.cfg.timeout)
                try:
                    out = json.loads(p.stdout)
                except json.JSONDecodeError:
                    out = {"is_error": True, "result": (p.stderr or p.stdout)[-600:]}
                if not out.get("is_error") and p.returncode == 0:
                    usage = dict(out.get("usage") or {})
                    usage["total_cost_usd"] = out.get("total_cost_usd")
                    if session:
                        return str(out.get("result") or ""), usage, out.get("session_id")
                    return str(out.get("result") or ""), usage
                if "not logged in" in str(out.get("result", "")).lower():
                    raise RuntimeError("claude CLI is not logged in: run `claude setup-token` and export "
                                       "CLAUDE_CODE_OAUTH_TOKEN, or run `claude` and /login")
                if sid or re.search(r"credit balance|limit.*resets", str(out.get("result", "")), re.IGNORECASE):
                    break  # a resume failure falls back to a fresh call; billing and usage limits do not retry
                if attempt < 2:
                    time.sleep(10 * (attempt + 1))
            raise RuntimeError(f"claude CLI failed ({p.returncode}): {str(out.get('result'))[-600:]}")
        finally:
            if tmpdir is not None:
                tmpdir.cleanup()
            else:
                sp.unlink(missing_ok=True)


# Calls whose reply is longer than the CLI's default output cap (32k tokens): the circuits merge restates every
# circuit of every attempt, and a reply cut off at the cap is unparseable JSON.
LONG_OUTPUT = {"circuits:consensus": 64000}


def _judgement(tag: str) -> bool:
    """Tool loops whose answer is a judgement (designing experiments) stay on the main model even though they are
    multi-step; the investigation's tool steps go to thread_model."""
    return tag.startswith(("experiment", "circuits"))


# USD per million tokens (platform.claude.com/docs/en/about-claude/pricing, read 2026-10-17):
# base input, 5-minute cache write, 1-hour cache write, cache read, output. Haiku: the price for prompts <= 100k tokens.
ANTHROPIC_PRICES = {"claude-opus-5-5": (4.0, 5.0, 8.0, 0.20, 20.0), "claude-sonnet-5-5": (2.0, 2.5, 4.0, 0.10, 10.0),
                    "claude-haiku-5-5": (0.10, 0.125, 0.20, 0.01, 0.50),
                    # models a refused request may fall back to (server-side fallback)
                    "claude-opus-5": (5.0, 6.25, 10.0, 0.50, 25.0), "claude-opus-4-8": (5.0, 6.25, 10.0, 0.50, 25.0),
                    "claude-opus-4-7": (5.0, 6.25, 10.0, 0.50, 25.0), "claude-opus-4-6": (5.0, 6.25, 10.0, 0.50, 25.0),
                    "claude-sonnet-5": (2.0, 2.5, 4.0, 0.20, 10.0), "claude-sonnet-4-6": (3.0, 3.75, 6.0, 0.30, 15.0)}
ANTHROPIC_MODELS = {"opus": "claude-opus-5-5", "sonnet": "claude-sonnet-5-5", "haiku": "claude-haiku-5-5"}
# Where a thread's first message splits into parts shared by many calls (agents.THREAD_USER): everything before the
# first marker (study context, hypotheses, tool catalog) is shared by every thread of a run; the log up to the
# second marker by every thread of one round; the rest is the thread's own question.
THREAD_CACHE_SPLITS = ("\n\nWhat the investigation has found so far", "\n\nYOUR QUESTION —")
# Synthesis passes (synthesize, revise) repeat one layout: the study context, then the investigation log, which only
# grows by appending; the computed tail blocks and the results that follow change every pass. Each pass marks the end
# of the log as a cache breakpoint, and the next pass reads up to the previous pass's mark from the cache.
SYNTH_FAMILY = ("synthesize", "revise")
SYNTH_LOG_START = "\n\nInvestigation log:"
SYNTH_LOG_ENDS = ("\n\nPATTERNS ACROSS THREADS", "\n\nCAUSAL STEPS", "\n\nKey results:")


def anthropic_cost(model: str, usage: dict) -> float:
    """Cost of one response. With server-side fallback, usage.iterations lists each attempt with its own model
    (the refused one is billed too), so the cost is their sum."""
    its = [it for it in usage.get("iterations") or [] if isinstance(it, dict)]
    if its:
        return sum(anthropic_cost(it.get("model") or model, {k: v for k, v in it.items() if k != "iterations"})
                   for it in its)
    pin, w5, w1, rd, out = ANTHROPIC_PRICES.get(model, (5.0, 6.25, 10.0, 0.50, 25.0))
    cc = usage.get("cache_creation") or {}
    w1h = cc.get("ephemeral_1h_input_tokens") or 0
    w5m = cc.get("ephemeral_5m_input_tokens")
    w5m = (usage.get("cache_creation_input_tokens") or 0) - w1h if w5m is None else w5m
    return ((usage.get("input_tokens") or 0) * pin + w5m * w5 + w1h * w1 + (usage.get("cache_read_input_tokens") or 0)
            * rd + (usage.get("output_tokens") or 0) * out) / 1e6


class AnthropicAPI:
    """Claude through the Messages API with an API key: no CLI, so every saving is explicit.
    - Each call's system prompt and the study context shared by many calls are cached for an hour (explicit
      breakpoints); a thread's growing conversation is cached by automatic caching (top-level cache_control), so
      each step pays only for the new tool result.
    - The first call to carry a new shared prefix runs alone; calls with the same prefix wait for it, because a cache
      entry is not visible to requests sent before the first one starts responding.
    - Thread tool steps (tags ending ":s<n>") go to thread_model without thinking; judgement calls (planning, PI
      reviews, candidates, synthesis) go to model with adaptive thinking at `effort`.
    Prompts and findings leave the cluster: requires external_ok."""

    URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, cfg: LLMConfig, key_file: str = ""):
        import collections
        import os
        if not cfg.external_ok:
            raise DataGuardError("the anthropic backend sends prompts to Anthropic; set llm.external_ok: true")
        self.cfg = cfg
        key = "" if key_file else os.environ.get("ANTHROPIC_API_KEY") or ""  # an explicit file wins over the env
        kf = Path(os.path.expanduser(key_file or cfg.api_key_file))
        if not key and kf.exists():
            key = kf.read_text().strip()
        if not key:
            raise ValueError(f"no API key: set ANTHROPIC_API_KEY or save one to {cfg.api_key_file}")
        self._key = key
        self.model = ANTHROPIC_MODELS.get(cfg.model, cfg.model or "claude-opus-5-5")
        self.thread_model = ANTHROPIC_MODELS.get(cfg.thread_model, cfg.thread_model) or self.model
        self.name = f"anthropic:{self.model}" + (f"+threads:{self.thread_model}" if cfg.thread_model else "")
        self._recent = collections.deque(maxlen=40)  # recent first messages, to find long shared prefixes
        self._warm: dict[str, threading.Event] = {}
        self.refusals = 0  # refused attempts that a retry recovered or that ended the call
        self._synth_mark = ""  # the previous synthesis pass's cached prefix (study context + log)
        self._lock = threading.Lock()

    def complete(self, tag: str, system: str, user: str, temperature: float | None = None) -> tuple[str, dict]:
        return self.chat(tag, [{"role": "system", "content": system}, {"role": "user", "content": user}], temperature)

    def _split(self, text: str, step: bool, tag: str = "") -> list[tuple[str, str | None]]:
        """The first user message as cacheable parts [(text, ttl or None)]."""
        if not step and tag.split(":")[0] in SYNTH_FAMILY:
            parts = self._split_synth(text)
            if parts:
                return parts
        if step:
            cuts = [text.find(m) for m in THREAD_CACHE_SPLITS]
            if all(c > 0 for c in cuts) and cuts[0] < cuts[1]:
                with self._lock:  # other calls (PI reviews, synthesis) start with the same study context
                    self._recent.append(text[:cuts[0]])
                return [(text[:cuts[0]], self.cfg.context_ttl), (text[cuts[0]:cuts[1]], "5m"), (text[cuts[1]:], None)]
        best = 0
        with self._lock:
            for other in self._recent:
                best = max(best, len(os.path.commonprefix([text, other])))
            self._recent.append(text)
        if best >= 6000:  # a long prefix another call already sent (usually the study context)
            cut = text.rfind("\n\n", 0, best)
            if cut > 4000:
                return [(text[:cut], self.cfg.context_ttl), (text[cut:], None)]
        return [(text, None)]

    def _split_synth(self, text: str) -> list[tuple[str, str | None]] | None:
        start = text.find(SYNTH_LOG_START)
        ends = [i for i in (text.find(m, start + 1) for m in SYNTH_LOG_ENDS) if i > start] if start > 0 else []
        if not ends:
            return None
        end = min(ends)
        with self._lock:
            prev = self._synth_mark
            self._synth_mark = text[:end]
        if prev and len(prev) > 4000 and text.startswith(prev) and len(prev) < end:
            return [(prev, self.cfg.context_ttl), (text[len(prev):end], self.cfg.context_ttl), (text[end:], None)]
        return [(text[:start], self.cfg.context_ttl), (text[start:end], self.cfg.context_ttl), (text[end:], None)]

    def chat(self, tag: str, messages: list[dict], temperature: float | None = None) -> tuple[str, dict]:
        step = bool(re.search(r":s\d+$", tag)) and not _judgement(tag)
        model = self.thread_model if step else self.model
        effort = self.cfg.thread_effort if step else self.cfg.effort_overrides.get(tag.split(":")[0], self.cfg.effort)
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        turns = [m for m in messages if m["role"] != "system"]
        out_msgs, prefix_key = [], None
        for i, m in enumerate(turns):
            if i == 0 and m["role"] == "user":
                parts = self._split(m["content"], step, tag)
                blocks = []
                for text, ttl in parts:
                    b = {"type": "text", "text": text}
                    if ttl:
                        b["cache_control"] = {"type": "ephemeral", "ttl": ttl}
                    blocks.append(b)
                if len(parts) > 1:
                    prefix_key = hashlib.sha256(json.dumps([model, effort, system, parts[0][0]]).encode()).hexdigest()
                out_msgs.append({"role": "user", "content": blocks})
            else:
                out_msgs.append({"role": m["role"], "content": [{"type": "text", "text": m["content"]}]})
        marks = sum(1 for m in out_msgs for b in m["content"] if "cache_control" in b)
        body = {"model": model, "max_tokens": self.cfg.max_tokens, "messages": out_msgs,
                "cache_control": {"type": "ephemeral"},  # automatic: the conversation so far
                "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral",
                                                                              "ttl": self.cfg.context_ttl}}]}
        if marks >= 3:  # at most 4 breakpoints per request: the system prompt and these take them all
            body.pop("cache_control")
        if self.cfg.fallbacks:
            body["fallbacks"] = self.cfg.fallbacks
        if effort:  # (these models take no temperature: the API rejects it)
            body["thinking"] = {"type": "adaptive"}
            body["output_config"] = {"effort": effort}
        # the first call with a new shared prefix writes the cache; calls with the same prefix wait for it
        waiter, owner = None, False
        if prefix_key and len(turns) == 1:
            with self._lock:
                waiter = self._warm.get(prefix_key)
                if waiter is None:
                    waiter = self._warm[prefix_key] = threading.Event()
                    owner = True
            if not owner:
                waiter.wait(timeout=300)
        try:
            data = self._post(body)
            for _ in range(1):  # a refusal the fallback could not serve: one more try (refusals are billed)
                if data.get("stop_reason") != "refusal":
                    break
                self.refusals += 1
                data = self._post(body)
        finally:
            if owner:
                waiter.set()
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        usage = dict(data.get("usage") or {})
        usage["model"], usage["total_cost_usd"] = model, round(anthropic_cost(model, usage), 6)
        usage["stop_reason"] = data.get("stop_reason")
        served = data.get("model") or model
        if any(it.get("type") == "fallback_message" for it in usage.get("iterations") or [] if isinstance(it, dict)):
            usage["served_by"] = served  # a refusal was answered by the fallback model: recorded for the report
            usage["refusal_category"] = (data.get("stop_details") or {}).get("category") or "see iterations"
        if data.get("stop_reason") == "refusal":  # never pass a refusal on as an empty answer
            raise RuntimeError(f"Anthropic API: the model refused this request twice, fallback included ({tag}); "
                               f"{len(json.dumps(body)) // 1000}k characters sent")
        return text, usage

    def _post(self, body: dict) -> dict:
        req_body = json.dumps(body).encode()
        delay = 5.0
        for attempt in range(8):
            headers = {"x-api-key": self._key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
            if body.get("fallbacks"):
                headers["anthropic-beta"] = "server-side-fallback-2026-07-01"
            req = urllib.request.Request(self.URL, data=req_body, method="POST", headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.cfg.timeout) as r:
                    return json.load(r)
            except urllib.error.HTTPError as exc:
                msg = exc.read().decode(errors="replace")[-800:]
                if exc.code in (429, 500, 502, 503, 504, 529) and attempt < 7:
                    wait = float(exc.headers.get("retry-after") or delay)
                    time.sleep(min(wait, 120))
                    delay = min(delay * 2, 120)
                    continue
                raise RuntimeError(f"Anthropic API HTTP {exc.code}: {msg}") from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                if attempt < 7:
                    time.sleep(delay)
                    delay = min(delay * 2, 120)
                    continue
                raise RuntimeError(f"Anthropic API unreachable: {exc}") from None
        raise RuntimeError("Anthropic API: retries exhausted")


# an account that cannot serve more calls for now: subscription window or weekly limit, no credit, an org spend
# limit, or a rate limit that outlasted every retry (accounts are separate organisations with separate limits)
EXHAUSTED = re.compile(r"credit balance|usage limits?|spend(ing)? limit|limit.{0,80}resets|HTTP 429",
                       re.IGNORECASE | re.DOTALL)


class ClaudeAccounts:
    """Claude through an ordered list of accounts (API keys and the subscription login), one backend per account.
    Each call goes to the first account that is not resting; one that runs out (EXHAUSTED) rests for REST_S and the
    call moves on to the next. When every account is out, the last account's error is raised unchanged, so a queue can
    still read its reset time or pause for credit. Every call records which account answered it and whether it was
    billed. Cached replies are shared across accounts: the cache key names the models, not the account."""

    REST_S = 1800

    def __init__(self, cfg: LLMConfig):
        if not cfg.external_ok:
            raise DataGuardError("the claude backend sends prompts to Anthropic; set llm.external_ok: true")
        self.cfg = cfg
        self.accounts: list[str] = []
        self._discover()
        if not self.accounts:
            raise ValueError("no Claude account found: save an API key (bench/save_api_key.sh) or the subscription "
                             "token (~/.config/claude/oauth_token)")
        self.name = f"claude:{cfg.model or 'default'}" + (f"+threads:{cfg.thread_model}" if cfg.thread_model else "")
        self._backends: dict[str, Backend] = {}
        self._resting: dict[str, float] = {}  # account -> until when it rests
        self._lock = threading.Lock()
        self.on_switch = None  # called with each move to another account (LoggedLLM puts it in the transcript)

    def _discover(self) -> None:
        """Add the accounts the configured patterns match now: a seat saved while a run is going joins it."""
        import glob
        import os
        for a in self.cfg.accounts:
            a = a.strip()
            if a.startswith("key:"):
                files = sorted(glob.glob(os.path.expanduser(a[4:])))
                self.accounts += [f"key:{f}" for f in files if Path(f).is_file() and f"key:{f}" not in self.accounts]
            elif a == "subscription":
                if a not in self.accounts and (Path.home() / ".config" / "claude" / "oauth_token").exists() or os.environ.get(
                        "CLAUDE_CODE_OAUTH_TOKEN"):
                    self.accounts.append(a)
            elif a.startswith("subscription:"):  # pooled Team seats: one saved token per member (bench/save_token.sh)
                files = sorted(glob.glob(os.path.expanduser(a[len("subscription:"):])))
                self.accounts += [f"subscription:{f}" for f in files
                                  if Path(f).is_file() and f"subscription:{f}" not in self.accounts]
            else:
                raise ValueError(f"unknown Claude account {a!r} (use key:<file>, subscription or subscription:<file>)")

    @staticmethod
    def label(account: str) -> str:
        if account == "subscription":
            return "subscription"
        if account.startswith("subscription:"):
            return "subscription " + Path(account[len("subscription:"):]).name.removeprefix("oauth_token_")
        return "API key " + Path(account[4:]).name

    @staticmethod
    def is_subscription(account: str) -> bool:
        return account == "subscription" or account.startswith("subscription:")

    def _backend(self, account: str) -> Backend:
        with self._lock:
            if account not in self._backends:
                self._backends[account] = (
                    ClaudeCLI(self.cfg, auth="subscription") if account == "subscription"
                    else ClaudeCLI(self.cfg, auth="subscription", token_file=account[len("subscription:"):])
                    if account.startswith("subscription:") else AnthropicAPI(self.cfg, key_file=account[4:]))
            return self._backends[account]

    def complete(self, tag: str, system: str, user: str, temperature: float | None = None) -> tuple[str, dict]:
        return self.chat(tag, [{"role": "system", "content": system}, {"role": "user", "content": user}], temperature)

    def chat(self, tag: str, messages: list[dict], temperature: float | None = None) -> tuple[str, dict]:
        from mechanist import pool
        last: Exception | None = None
        now = time.time()
        with self._lock:
            self._discover()
        chain = pool.ordered(self.accounts)  # the shared pool's order (pool.json), re-read every call
        with self._lock:
            ready = [a for a in chain if now >= self._resting.get(a, 0.0) and not pool.resting(a, now)]
        for account in ready or chain:  # all resting: try them all again, in order
            try:
                text, usage = self._backend(account).chat(tag, messages, temperature)
            except RuntimeError as exc:
                if not EXHAUSTED.search(str(exc)):
                    raise
                until = pool.reset_at(str(exc)) or time.time() + self.REST_S
                with self._lock:
                    fresh = now >= self._resting.get(account, 0.0)
                    self._resting[account] = until
                pool.mark(account, "ran_out", pool.reset_at(str(exc)))  # every other run rests it too
                if fresh and self.on_switch:
                    self.on_switch(f"{self.label(account)} is out of usage or credit; resting it until "
                                   f"{time.strftime('%H:%M', time.localtime(until))} and moving on ({str(exc)[-160:]})")
                last = exc
                continue
            usage["account"], usage["billed"] = self.label(account), not self.is_subscription(account)
            pool.mark(account, "used")
            return text, usage
        raise last if last else RuntimeError("no Claude account is available")


def billed_cost(usage: dict, model: str = "") -> float:
    """What a call cost in money: subscription calls carry the CLI's estimate but are not billed per call."""
    if usage.get("billed") is False or (usage.get("billed") is None and str(model).startswith("claude_cli")):
        return 0.0
    return float(usage.get("total_cost_usd") or 0.0)


def make_backend(cfg: LLMConfig) -> Backend:
    if cfg.backend == "claude":
        return ClaudeAccounts(cfg)
    if cfg.backend == "openai_compat":
        return OpenAICompat(cfg)
    if cfg.backend == "claude_cli":
        return ClaudeCLI(cfg)
    if cfg.backend == "anthropic":
        return AnthropicAPI(cfg)
    if cfg.backend == "replay":
        if not cfg.replay_file:
            raise ValueError("replay backend needs llm.replay_file")
        return Replay(json.loads(Path(cfg.replay_file).read_text()))
    raise ValueError(f"unknown LLM backend {cfg.backend!r}")


class LoggedLLM:
    """Caches each call under runs/<id>/llm_cache by request hash and appends it to llm_calls.jsonl."""

    def __init__(self, backend: Backend, run_dir: Path):
        self.backend = backend
        self.cache = run_dir / "llm_cache"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.log = run_dir / "llm_calls.jsonl"
        from mechanist.transcript import Transcript
        self.transcript = Transcript(run_dir / "transcript.jsonl")  # the run as a conversation, for reading
        if hasattr(backend, "on_switch"):
            backend.on_switch = self.transcript.note
        self.calls = 0
        self.cached = 0
        self._lock = threading.Lock()  # investigation threads call in parallel

    def complete(self, tag: str, system: str, user: str, temperature: float | None = None) -> str:
        key_parts = [self.backend.name, system, user] + ([temperature] if temperature is not None else [])
        key = hashlib.sha256(json.dumps(key_parts).encode()).hexdigest()
        hit = self.cache / f"{key}.json"
        if hit.exists():
            self.cached += 1
            text = json.loads(hit.read_text())["response"]
            self.transcript.turn(tag, text, cached=True, key=key)
            return text
        t0 = time.time()
        text, usage = self.backend.complete(tag, system, user, temperature)
        self.transcript.turn(tag, text, model=str(usage.get("model", "")), key=key)
        rec = {"tag": tag, "model": self.backend.name, "temperature": temperature, "request_sha256": key, "seconds": round(time.time() - t0, 1),
               "usage": usage, "prompt_chars": len(system) + len(user), "response": text}
        self._record(hit, rec)
        return text

    def chat(self, tag: str, messages: list[dict], temperature: float | None = None) -> str:
        """Multi-turn call (an investigation thread). Cached by the whole message list, so a resumed thread replays."""
        key_parts = [self.backend.name, "chat", messages] + ([temperature] if temperature is not None else [])
        key = hashlib.sha256(json.dumps(key_parts).encode()).hexdigest()
        hit = self.cache / f"{key}.json"
        if hit.exists():
            with self._lock:
                self.cached += 1
            text = json.loads(hit.read_text())["response"]
            self.transcript.turn(tag, text, cached=True, key=key)
            return text
        t0 = time.time()
        text, usage = self.backend.chat(tag, messages, temperature)
        self.transcript.turn(tag, text, model=str(usage.get("model", "")), key=key)
        rec = {"tag": tag, "model": self.backend.name, "temperature": temperature, "request_sha256": key,
               "seconds": round(time.time() - t0, 1), "usage": usage,
               "prompt_chars": sum(len(str(m.get("content", ""))) for m in messages), "response": text}
        self._record(hit, rec)
        return text

    def _record(self, hit: Path, rec: dict) -> None:
        hit.write_text(json.dumps(rec))
        with self._lock:
            with self.log.open("a") as fh:
                fh.write(json.dumps({k: v for k, v in rec.items() if k != "response"}) + "\n")
            self.calls += 1


def parse_json(text: str):
    """First complete JSON object/array in a reply (tolerates code fences and reasoning preambles)."""
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE)
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
    if start < 0:
        raise ValueError("no JSON in reply")
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("unterminated JSON in reply")
