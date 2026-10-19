"""The shared Claude account pool: which accounts calls go to, in what order, and which are resting.

Files in ~/.config/claude/ (the folder that holds the tokens and keys; only their names appear here):
    pool.json        {"order": [ids], "off": [ids]}, set by hand (scripts/claude_pool.py) or from the portal's admin
                     page. Ids: "me" (your own seat, oauth_token), a member's seat by name (oauth_token_<name>),
                     "key:<file>" for an API key. Accounts it does not list keep their configured order, after the
                     listed ones; "off" accounts are skipped.
    pool_state.json  per id: when it last answered, and when it last ran out (shared by every run, so a seat that one
                     run found exhausted is rested by the others too).

Read before every call (a changed order takes effect at once, even mid-run). Stdlib only: the Betty cycle and the
portal bridge import it without Mechanist's dependencies.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

REST_S = 1800  # an account that ran out rests this long before it is tried again


def folder() -> Path:
    return Path(os.environ.get("MECHANIST_POOL_DIR") or Path.home() / ".config" / "claude")


def account_id(account: str) -> str:
    """'subscription' -> 'me'; 'subscription:<.../oauth_token_jane>' -> 'jane'; 'key:<.../api_key>' -> 'key:api_key'."""
    if account == "subscription":
        return "me"
    if account.startswith("subscription:"):
        return Path(account.split(":", 1)[1]).name.removeprefix("oauth_token_")
    return "key:" + Path(account.split(":", 1)[1]).name


def kind(account: str) -> str:
    return "API key (lab, billed)" if account.startswith("key:") else "seat (subscription)"


def read_order() -> dict:
    p = folder() / "pool.json"
    try:
        d = json.loads(p.read_text())
    except (OSError, ValueError):
        return {"order": [], "off": []}
    return {"order": [str(x) for x in d.get("order") or []], "off": [str(x) for x in d.get("off") or []],
            "updated_at": d.get("updated_at"), "updated_by": d.get("updated_by")}


def write_order(order: list[str], off: list[str], by: str = "") -> None:
    p = folder() / "pool.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({"order": order, "off": off, "updated_by": by,
                               "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=1))
    os.chmod(tmp, 0o600)
    tmp.replace(p)


def ordered(accounts: list[str]) -> list[str]:
    """The configured accounts in the pool's order, without the ones switched off (all of them if every one is off).
    A seat the order does not list yet (newly added) goes just before the first API key, so the lab key, which is
    billed, is never drawn on ahead of a seat."""
    o = read_order()
    ids = {a: account_id(a) for a in accounts}
    if not o["order"]:  # no order set yet: the configured order
        return [a for a in accounts if ids[a] not in set(o["off"])] or list(accounts)
    listed = [a for i in o["order"] for a in accounts if ids[a] == i]
    new_seats = [a for a in accounts if a not in listed and not a.startswith("key:")]
    new_keys = [a for a in accounts if a not in listed and a.startswith("key:")]
    first_key = next((k for k, a in enumerate(listed) if a.startswith("key:")), len(listed))
    chain = listed[:first_key] + new_seats + listed[first_key:] + new_keys
    off = set(o["off"])
    return [a for a in chain if ids[a] not in off] or chain


@contextmanager
def _state():
    p = folder() / "pool_state.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p.with_suffix(".lock"), "a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            st = json.loads(p.read_text()) if p.exists() else {}
        except ValueError:
            st = {}
        yield st
        tmp = p.with_suffix(".tmpstate")
        tmp.write_text(json.dumps(st, indent=1))
        os.chmod(tmp, 0o600)
        tmp.replace(p)


def read_state() -> dict:
    p = folder() / "pool_state.json"
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


def resting(account: str, now: float | None = None) -> bool:
    t = (read_state().get(account_id(account)) or {}).get("ran_out_at") or 0
    return (now or time.time()) - float(t) < REST_S


def mark(account: str, event: str) -> None:
    """event: 'used' (it answered) or 'ran_out' (out of usage or credit). 'used' is written at most once a minute."""
    i, now = account_id(account), time.time()
    if event == "used" and now - float((read_state().get(i) or {}).get("used_at") or 0) < 60:
        return
    with _state() as st:
        e = st.setdefault(i, {})
        if event == "used":
            e["used_at"] = now
        else:
            e["ran_out_at"] = now


def status(accounts: list[str]) -> list[dict]:
    """The chain as the pool will use it now: position, id, kind, on/off, resting until, last answered."""
    o, st, now = read_order(), read_state(), time.time()
    chain = ordered(accounts)
    off = set(o["off"])
    rows = []
    for a in chain + [a for a in accounts if a not in chain]:
        i = account_id(a)
        e = st.get(i) or {}
        out = float(e.get("ran_out_at") or 0)
        rows.append({"id": i, "kind": kind(a), "on": i not in off, "position": chain.index(a) + 1 if a in chain else None,
                     "resting_until": out + REST_S if now - out < REST_S else None,
                     "used_at": e.get("used_at"), "ran_out_at": e.get("ran_out_at")})
    return rows


def accounts_from(patterns: list[str]) -> list[str]:
    """The accounts a config's llm.accounts list names (files that exist), as ClaudeAccounts expands them."""
    import glob
    out: list[str] = []
    for a in patterns:
        a = a.strip()
        if a == "subscription":
            if (folder() / "oauth_token").exists():
                out.append(a)
        elif a.startswith(("subscription:", "key:")):
            pre, pat = a.split(":", 1)
            for f in sorted(glob.glob(os.path.expanduser(pat))):
                if Path(f).is_file() and f"{pre}:{f}" not in out:
                    out.append(f"{pre}:{f}")
    return out
