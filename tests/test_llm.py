import pytest

from mechanist.llm import DataGuardError, LLMConfig, LoggedLLM, Replay, make_backend, parse_json


def test_data_guard_refuses_remote_host():
    with pytest.raises(DataGuardError):
        make_backend(LLMConfig(base_url="https://api.example.com/v1", model="m"))
    assert make_backend(LLMConfig(base_url="https://api.example.com/v1", model="m", external_ok=True))
    assert make_backend(LLMConfig(base_url="http://127.0.0.1:9/v1", model="m"))


def test_parse_json_tolerates_fences_and_braces_in_strings():
    assert parse_json('reasoning...\n```json\n{"a": "x}{y", "b": [1, {"c": "\\"q\\""}]}\n```') == \
        {"a": "x}{y", "b": [1, {"c": '"q"'}]}
    with pytest.raises(ValueError):
        parse_json("no json here")


def test_logged_llm_caches_by_request(tmp_path):
    llm = LoggedLLM(Replay({"generate": ["one", "two"]}), tmp_path)
    assert llm.complete("generate", "s", "u") == "one"
    assert llm.complete("generate", "s", "u") == "one"  # cache hit, replay not consumed
    assert llm.calls == 1 and llm.cached == 1
    assert (tmp_path / "llm_calls.jsonl").read_text().count("\n") == 1


def test_claude_cli_backend_needs_external_ok_and_renders_threads(tmp_path, monkeypatch):
    import json
    import os
    import stat

    import pytest

    from mechanist.llm import ClaudeCLI, DataGuardError, LLMConfig
    with pytest.raises(DataGuardError):
        ClaudeCLI(LLMConfig(backend="claude_cli", model="opus"))
    fake = tmp_path / "claude"  # echoes the prompt it received and the flags, as the CLI's JSON result
    fake.write_text("#!/usr/bin/env python3\nimport json, sys\nprompt = sys.stdin.read()\n"
                    "print(json.dumps({'type': 'result', 'is_error': False, 'result': json.dumps("
                    "{'argv': sys.argv[1:], 'prompt': prompt}), 'usage': {'output_tokens': 3}, 'total_cost_usd': 0.01}))\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    b = ClaudeCLI(LLMConfig(backend="claude_cli", model="opus", external_ok=True))
    text, usage = b.chat("t", [{"role": "system", "content": "SYS"}, {"role": "user", "content": "Q1"},
                               {"role": "assistant", "content": "A1"}, {"role": "user", "content": "R1"}])
    out = json.loads(text)
    assert usage["total_cost_usd"] == 0.01
    assert "--tools" in out["argv"] and out["argv"][out["argv"].index("--tools") + 1] == ""
    assert "--model" in out["argv"] and "--no-session-persistence" in out["argv"]
    assert "<assistant>\nA1\n</assistant>" in out["prompt"] and out["prompt"].index("Q1") < out["prompt"].index("R1")


def test_claude_cli_thread_steps_resume_one_session(tmp_path, monkeypatch):
    """Thread steps continue one CLI session and send only the new message; one-off calls stay unsaved."""
    import json as _json

    from mechanist.agents import run_thread
    from mechanist.llm import ClaudeCLI, LLMConfig, LoggedLLM
    from mechanist.records import ToolResult
    log = tmp_path / "calls.jsonl"
    fake = tmp_path / "claude"
    fake.write_text("#!/usr/bin/env python3\n"
                    "import json, os, sys, uuid\n"
                    "a = sys.argv[1:]; p = sys.stdin.read(); log = os.environ['FAKE_LOG']\n"
                    "n = open(log).read().count('\\n') if os.path.exists(log) else 0\n"
                    "sid = a[a.index('--resume') + 1] if '--resume' in a else str(uuid.uuid4())\n"
                    "open(log, 'a').write(json.dumps({'resume': '--resume' in a, 'saved': "
                    "'--no-session-persistence' not in a, 'chars': len(p), "
                    "'model': a[a.index('--model') + 1] if '--model' in a else None}) + '\\n')\n"
                    "r = json.dumps({'tool': 'lineage_share', 'args': {'lineage': str(n)}}) if n < 2 else "
                    "json.dumps({'answer': {'status': 'answered', 'answer': 'ok'}})\n"
                    "print(json.dumps({'result': r, 'session_id': sid, 'usage': {}, 'total_cost_usd': 0}))\n")
    fake.chmod(0o755)
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.setattr("shutil.which", lambda name: str(fake))
    b = ClaudeCLI(LLMConfig(backend="claude_cli", model="opus", thread_model="sonnet", external_ok=True))
    llm = LoggedLLM(b, tmp_path / "run")
    k = [0]

    def execute(tool, args):
        k[0] += 1
        return ToolResult(id=f"R{k[0]:04d}", tool=tool, args=args, ok=True, summary="result")
    ans, _ = run_thread(llm, "investigate:Q01", "system", "context " * 500, execute, 4)
    b.complete("plan", "system", "one-off")
    calls = [_json.loads(x) for x in log.read_text().splitlines()]
    assert ans == {"status": "answered", "answer": "ok"}
    assert [c["resume"] for c in calls] == [False, True, True, False]
    assert [c["saved"] for c in calls] == [True, True, True, False]
    assert calls[1]["chars"] < calls[0]["chars"] / 10  # a resumed step sends only the new tool result
    assert [c["model"] for c in calls] == ["sonnet", "sonnet", "sonnet", "opus"]  # judgement calls stay on opus


def test_anthropic_backend_caches_shared_context_and_routes_models(tmp_path, monkeypatch):
    import io
    import json as _json
    import threading as _threading
    import time as _time

    from mechanist.agents import THREAD_USER
    from mechanist.llm import THREAD_CACHE_SPLITS, AnthropicAPI, LLMConfig, anthropic_cost
    cuts = [THREAD_USER.find(m) for m in THREAD_CACHE_SPLITS]  # the split markers track the thread prompt
    assert all(c > 0 for c in cuts) and cuts[0] < cuts[1]
    sent, order = [], []

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def urlopen(req, timeout=None):
        body = _json.loads(req.data)
        sent.append(body)
        order.append(("start", len(sent)))
        _time.sleep(0.2)
        order.append(("end", len(sent)))
        usage = {"input_tokens": 10, "cache_creation_input_tokens": 1000, "cache_read_input_tokens": 0,
                 "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 1000},
                 "output_tokens": 50}
        return Resp(_json.dumps({"content": [{"type": "thinking", "thinking": "x"}, {"type": "text", "text": "{}"}],
                                 "usage": usage, "stop_reason": "end_turn"}).encode())
    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    b = AnthropicAPI(LLMConfig(backend="anthropic", model="opus", thread_model="sonnet", external_ok=True,
                               max_tokens=1000))
    shared = "CONTEXT " * 2000
    user = lambda q: f"{shared}{THREAD_CACHE_SPLITS[0]} log{THREAD_CACHE_SPLITS[1]} {q}"
    # two threads start together with the same shared prefix: the second waits for the first to respond
    ts = [_threading.Thread(target=b.chat, args=(f"investigate:Q0{i}:s0", [{"role": "system", "content": "SYS"},
                                                                           {"role": "user", "content": user(i)}]))
          for i in (1, 2)]
    for t in ts:
        t.start()
        _time.sleep(0.05)
    for t in ts:
        t.join()
    assert order[:2] == [("start", 1), ("end", 1)]  # no overlap: the cache was written before the second call
    step = sent[0]
    assert step["model"] == "claude-sonnet-5-5" and "thinking" not in step and "temperature" not in step
    blocks = step["messages"][0]["content"]
    assert [b_.get("cache_control", {}).get("ttl") for b_ in blocks] == ["1h", "5m", None]
    assert blocks[0]["text"] == shared and step["system"][0]["cache_control"]["ttl"] == "1h"
    assert step["cache_control"] == {"type": "ephemeral"}  # automatic caching of the growing conversation
    # a judgement call: opus with adaptive thinking; a second call sharing a long prefix caches that prefix
    text, usage = b.chat("synthesize", [{"role": "system", "content": "S2"}, {"role": "user", "content": shared + "\n\nA"}])
    b.chat("critique_model", [{"role": "system", "content": "S3"}, {"role": "user", "content": shared + "\n\nB"}])
    judge, second = sent[-2], sent[-1]
    assert judge["model"] == "claude-opus-5-5" and judge["thinking"] == {"type": "adaptive"}
    assert judge["output_config"] == {"effort": "medium"} and "temperature" not in judge and text == "{}"
    assert second["messages"][0]["content"][0]["cache_control"]["ttl"] == "1h"
    assert abs(usage["total_cost_usd"] - anthropic_cost("claude-opus-5-5", usage)) < 1e-9
    assert abs(anthropic_cost("claude-opus-5-5", {"input_tokens": 1e6}) - 4.0) < 1e-9
    assert abs(anthropic_cost("claude-sonnet-5-5", {"cache_read_input_tokens": 1e6}) - 0.10) < 1e-9


def test_claude_accounts_move_on_when_one_runs_out(tmp_path, monkeypatch):
    import pytest

    from mechanist import llm
    monkeypatch.setattr(llm.Path, "home", lambda: tmp_path)
    (tmp_path / ".config" / "claude").mkdir(parents=True)
    for f in ("api_key", "api_key_2", "oauth_token"):
        (tmp_path / ".config" / "claude" / f).write_text("x")
    cfg = llm.LLMConfig(backend="claude", model="opus", external_ok=True,
                        accounts=[f"key:{tmp_path}/.config/claude/api_key", f"key:{tmp_path}/.config/claude/api_key_*",
                                  "key:/nonexistent/key", "subscription"])
    b = llm.make_backend(cfg)
    assert [b.label(a) for a in b.accounts] == ["API key api_key", "API key api_key_2", "subscription"]
    seen = []

    class Fake:
        def __init__(self, account, error=None):
            self.account, self.error = account, error

        def chat(self, tag, messages, temperature=None):
            seen.append(self.account)
            if self.error:
                raise RuntimeError(self.error)
            return "ok", {"total_cost_usd": 0.5}
    b._backends = {b.accounts[0]: Fake("k1", "Anthropic API HTTP 400: Your credit balance is too low"),
                   b.accounts[1]: Fake("k2", "Anthropic API HTTP 400: You have reached your specified API usage limits"),
                   "subscription": Fake("sub")}
    notes = []
    b.on_switch = notes.append
    text, usage = b.complete("x", "s", "u")
    assert (text, seen, usage["account"], usage["billed"]) == ("ok", ["k1", "k2", "sub"], "subscription", False)
    assert len(notes) == 2 and llm.billed_cost(usage) == 0.0
    b.chat("y", [{"role": "user", "content": "u"}])
    assert seen[3:] == ["sub"]  # the spent keys rest; no second note for them
    assert len(notes) == 2
    b._backends["subscription"] = Fake("sub", "claude CLI failed (1): bad JSON")  # not a limit: raised, not skipped
    with pytest.raises(RuntimeError, match="bad JSON"):
        b.chat("z", [{"role": "user", "content": "u2"}])
    b._backends["subscription"] = Fake("sub", "claude CLI failed (1): You've hit your limit · resets 7:45pm")
    with pytest.raises(RuntimeError, match="resets 7:45pm"):  # all out: the last error reaches the queue unchanged
        b.chat("w", [{"role": "user", "content": "u3"}])
    assert llm.billed_cost({"total_cost_usd": 0.5, "billed": True}) == 0.5
    assert llm.billed_cost({"total_cost_usd": 0.5}, "claude_cli:opus") == 0.0


def test_pooled_member_seats_come_between_own_seat_and_keys(tmp_path, monkeypatch):
    from mechanist import llm
    monkeypatch.setattr(llm.Path, "home", lambda: tmp_path)
    d = tmp_path / ".config" / "claude"
    d.mkdir(parents=True)
    for f in ("oauth_token", "oauth_token_jane", "oauth_token_ben", "api_key"):
        (d / f).write_text("sk-ant-oat01-x" if f.startswith("oauth") else "sk-ant-api03-x")
    cfg = llm.LLMConfig(backend="claude", model="opus", external_ok=True,
                        accounts=["subscription", f"subscription:{d}/oauth_token_*", f"key:{d}/api_key"])
    b = llm.make_backend(cfg)
    assert [b.label(a) for a in b.accounts] == ["subscription", "subscription ben", "subscription jane", "API key api_key"]
    assert [b.is_subscription(a) for a in b.accounts] == [True, True, True, False]
    seat = b._backend(b.accounts[2])
    assert isinstance(seat, llm.ClaudeCLI) and seat.token_file.endswith("oauth_token_jane")


def test_pool_order_off_and_shared_resting(tmp_path, monkeypatch):
    import json

    from mechanist import llm, pool
    monkeypatch.setenv("MECHANIST_POOL_DIR", str(tmp_path))
    for f in ("oauth_token", "oauth_token_jane", "oauth_token_ben", "api_key"):
        (tmp_path / f).write_text("x")
    accts = ["subscription", f"subscription:{tmp_path}/oauth_token_ben", f"subscription:{tmp_path}/oauth_token_jane",
             f"key:{tmp_path}/api_key"]
    assert [pool.account_id(a) for a in pool.ordered(accts)] == ["me", "ben", "jane", "key:api_key"]
    pool.write_order(["jane", "me"], ["ben"], "test")
    assert [pool.account_id(a) for a in pool.ordered(accts)] == ["jane", "me", "key:api_key"]
    pool.mark(accts[2], "ran_out")
    assert pool.resting(accts[2]) and not pool.resting(accts[0])
    rows = {r["id"]: r for r in pool.status(accts)}
    assert rows["ben"]["on"] is False and rows["jane"]["resting_until"] and rows["jane"]["position"] == 1
    assert json.loads((tmp_path / "pool.json").read_text())["updated_by"] == "test"
    # the chain skips the resting seat and the switched-off one
    cfg = llm.LLMConfig(backend="claude", model="opus", external_ok=True, accounts=accts)
    b = llm.ClaudeAccounts(cfg)
    used = []

    class Fake:
        def __init__(self, name):
            self.name = name

        def chat(self, tag, messages, temperature=None):
            used.append(self.name)
            return "ok", {}
    b._backends = {a: Fake(pool.account_id(a)) for a in accts}
    _, usage = b.chat("t", [{"role": "user", "content": "x"}])
    assert used == ["me"] and usage["account"] == "subscription" and usage["billed"] is False


def test_cli_passes_the_stage_effort_and_drops_an_inherited_one(tmp_path, monkeypatch):
    import json as _json

    from mechanist.llm import ClaudeCLI, LLMConfig
    log = tmp_path / "calls.jsonl"
    fake = tmp_path / "claude"
    fake.write_text("#!/usr/bin/env python3\n"
                    "import json, os, sys\n"
                    "a = sys.argv[1:]; sys.stdin.read()\n"
                    "open(os.environ['FAKE_LOG'], 'a').write(json.dumps({'effort': a[a.index('--effort') + 1] "
                    "if '--effort' in a else None, 'env': os.environ.get('CLAUDE_EFFORT')}) + '\\n')\n"
                    "print(json.dumps({'result': 'ok', 'session_id': 's', 'usage': {}, 'total_cost_usd': 0}))\n")
    fake.chmod(0o755)
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.setenv("CLAUDE_EFFORT", "low")
    monkeypatch.setattr("shutil.which", lambda name: str(fake))
    b = ClaudeCLI(LLMConfig(backend="claude_cli", model="opus", external_ok=True, effort="medium",
                            effort_overrides={"circuits": "high"}))
    b.complete("circuits:r1", "system", "x")
    b.complete("plan", "system", "x")
    calls = [_json.loads(x) for x in log.read_text().splitlines()]
    assert [c["effort"] for c in calls] == ["high", "medium"] and all(c["env"] is None for c in calls)


def test_a_new_seat_goes_before_the_lab_key(tmp_path, monkeypatch):
    from mechanist import pool
    monkeypatch.setenv("MECHANIST_POOL_DIR", str(tmp_path))
    accts = ["subscription", f"subscription:{tmp_path}/oauth_token_a", f"subscription:{tmp_path}/oauth_token_new",
             f"key:{tmp_path}/api_key"]
    pool.write_order(["a", "me", "key:api_key"], [], "t")
    assert [pool.account_id(x) for x in pool.ordered(accts)] == ["a", "me", "new", "key:api_key"]


def test_seats_rest_until_their_reset_and_new_seats_join_a_running_pool(tmp_path, monkeypatch):
    import datetime as dt
    import time

    from mechanist import llm, pool
    monkeypatch.setenv("MECHANIST_POOL_DIR", str(tmp_path))
    monkeypatch.setattr(llm.Path, "home", lambda: tmp_path / "home")
    for f in ("oauth_token_ann", "oauth_token_bo"):
        (tmp_path / f).write_text("x")
    now = time.time()
    r = pool.reset_at("You've hit your session limit · resets 10:40am (America/New_York)", now)
    t = dt.datetime.fromtimestamp(r)
    assert (t.hour, t.minute) == (10, 42) and now < r <= now + 86400
    ann, bo = (f"subscription:{tmp_path}/oauth_token_{n}" for n in ("ann", "bo"))
    pool.mark(ann, "ran_out", now + 600)  # known reset: rests until then, not the default half hour
    assert pool.resting(ann, now + 300) and not pool.resting(ann, now + 700)
    pool.mark(bo, "ran_out")  # no reset in the message: the default rest
    assert pool.ready_at(bo) >= now + pool.REST_S - 1
    assert abs(pool.next_ready() - (now + 600)) < 2  # the queue wakes for the first seat back, not the last
    (tmp_path / "oauth_token_cy").write_text("x")  # saved while everything rests: ready at once
    assert pool.next_ready() == 0.0

    cfg = llm.LLMConfig(backend="claude", model="opus", external_ok=True, accounts=[f"subscription:{tmp_path}/oauth_token_*"])
    b = llm.ClaudeAccounts(cfg)
    used = []

    class Fake:
        def __init__(self, name):
            self.name = name

        def chat(self, tag, messages, temperature=None):
            if self.name != "dee":
                raise RuntimeError("claude CLI failed (1): You've hit your session limit · resets 1pm")
            used.append(self.name)
            return "ok", {}

    (tmp_path / "oauth_token_cy").unlink()
    b.accounts = [a for a in b.accounts if not a.endswith("_cy")]
    (tmp_path / "oauth_token_dee").write_text("x")  # a seat saved after the run started
    b._backend = lambda a: Fake(pool.account_id(a))
    _, usage = b.chat("t", [{"role": "user", "content": "x"}])
    assert used == ["dee"] and usage["account"] == "subscription dee"
