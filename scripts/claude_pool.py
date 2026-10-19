"""See and arrange the shared Claude account pool (the order calls draw on seats and the lab key).

    python3 scripts/claude_pool.py                      # show the chain
    python3 scripts/claude_pool.py order jane,me,ben    # these first, in this order; the rest after, as configured
    python3 scripts/claude_pool.py off ben              # skip a seat (on ben to bring it back)
    python3 scripts/claude_pool.py status-json OUT      # the chain as JSON for the portal (no secrets: names only)
    python3 scripts/claude_pool.py apply FILE           # an order sent from the portal's admin page, checked first

Ids: "me" (your seat), a member's name (their saved oauth_token_<name>), key:<file> for an API key.
Stdlib only, so the Betty cycle can run it with any python3.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mechanist import pool

PATTERNS = ["subscription", "subscription:~/.config/claude/oauth_token_*", "key:~/.config/claude/api_key",
            "key:~/.config/claude/api_key_*"]


def accounts() -> list[str]:
    return pool.accounts_from(PATTERNS)


def ids() -> list[str]:
    return [pool.account_id(a) for a in accounts()]


def _t(x) -> str:
    return dt.datetime.fromtimestamp(float(x)).strftime("%a %H:%M") if x else "-"


def show() -> None:
    o = pool.read_order()
    print(f"Claude pool (order set {o.get('updated_at') or 'never'}{' by ' + o['updated_by'] if o.get('updated_by') else ''})")
    for r in pool.status(accounts()):
        state = "off" if not r["on"] else f"resting until {_t(r['resting_until'])}" if r["resting_until"] else "ready"
        pos = f"{r['position']:>2}" if r["position"] else " -"
        print(f"{pos}  {r['id']:<20} {r['kind']:<24} {state:<22} last answered {_t(r['used_at'])}")


def set_order(new: list[str], off: list[str], by: str) -> None:
    known = set(ids())
    bad = [x for x in new + off if x not in known]
    if bad:
        raise SystemExit(f"unknown account(s): {', '.join(bad)}; known: {', '.join(sorted(known))}")
    pool.write_order(new, off, by)


def main() -> None:
    a = sys.argv[1:]
    if not a or a[0] == "show":
        show()
        return
    cmd = a[0]
    o = pool.read_order()
    if cmd == "order" and len(a) == 2:
        new = [x.strip() for x in a[1].split(",") if x.strip()]
        set_order(new + [x for x in o["order"] if x not in new], o["off"], "betty")
    elif cmd in ("on", "off") and len(a) == 2:
        off = [x for x in o["off"] if x != a[1]] + ([a[1]] if cmd == "off" else [])
        set_order(o["order"], off, "betty")
    elif cmd == "status-json" and len(a) == 2:
        Path(a[1]).write_text(json.dumps({"generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                                          "order": o, "chain": pool.status(accounts())}, indent=1))
        return
    elif cmd == "apply" and len(a) == 2:
        d = json.loads(Path(a[1]).read_text())
        order = [str(x) for x in d.get("order") or []]
        off = [str(x) for x in d.get("off") or []]
        set_order(order, off, str(d.get("by") or "portal"))
    else:
        raise SystemExit(__doc__)
    show()


if __name__ == "__main__":
    main()
