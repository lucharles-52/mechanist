"""A run as a team chat: one channel per stage, one thread per question, tool results folded under the step that
asked for them. Reads runs/<id>/transcript.jsonl and the stage outputs; writes runs/<id>/conversation.html.

    python3 scripts/conversation_view.py runs/<run>            # write conversation.html once
    python3 scripts/conversation_view.py runs/<run> --watch    # rewrite it every 30 s while the run goes

    uv run python scripts/conversation_view.py runs/<run> --watch --humanize   # plus plain-chat rewrites (Haiku)

Changes nothing in the run. Plain Python, no dependencies (except --humanize, which calls Haiku 5.5 on the API key).
"""

from __future__ import annotations

import argparse
import html
import json
import re
import time
from pathlib import Path

# (channel, title) by tag prefix, in workflow order; the first match wins
CHANNELS = [
    (r"^dossier", "1-dossier", "Stage 1 · what is known about the disease"),
    (r"^check", "2-pathology-check", "Stage 2 · do we see the pathology, what is unusual"),
    (r"^(panel|generate|merge|candidates|verify)", "3-candidates", "Stage 3 · candidate mechanisms"),
    (r"^scout", "3-candidates", "Stage 3 · candidate mechanisms"),
    (r"^(investigate|synthesize$|critique$)", "4-investigation", "Stage 4 · first investigation and model"),
    (r"^(chase|synthesize:pi|critique_model)", "4-chase", "Stage 4 · the PI's chase rounds"),
    (r"^linkage", "linkage", "Disease linkage · each link across medicine"),
    (r"^(revise|final)", "6-final-model", "Stage 6 · integrated model"),
    (r"^circuits", "7-circuits", "Stage 7 · circuits, the final model"),
    (r"^experiment", "5-experiments", "Stage 5 · experiments"),
]
ROLE_STYLE = {"PI": ("PI", "pi"), "planner": ("PL", "plan"), "synthesizer": ("SY", "syn"), "summariser": ("SU", "syn"),
              "experimentalist": ("EX", "exp"), "disease scholar": ("DS", "lit"), "literature scout": ("LS", "lit"),
              "PI (candidates)": ("PI", "pi"), "harness": ("⚙", "sys"),
              "hypothesizer": ("HY", "syn")}


def channel(tag: str) -> tuple[str, str]:
    for pat, ch, title in CHANNELS:
        if re.match(pat, tag):
            return ch, title
    return "other", "Other"


def thread_key(tag: str) -> str | None:
    """Turns that belong to one thread: an analyst's question, a linkage scholar's link, one experiment round."""
    m = (re.search(r":(Q\d+):s\d+$", tag) or re.match(r"^linkage\d*:(L\d+)", tag)
         or re.match(r"^((?:experiment|circuits):r\d+):s", tag))
    return m.group(1) if m else None


def questions(run: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for f in ("check", "investigate", "chase", "revise"):
        p = run / f"{f}.json"
        if not p.exists():
            continue
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        for q in d.get("questions", []) + d.get("final_questions", []):
            out[q["id"]] = q
    p = run / "linkage.json"
    if p.exists():
        try:
            for i, lk in enumerate(json.loads(p.read_text()).get("links") or [], 1):
                out[f"L{i}"] = {"text": lk.get("link", ""), "status": ""}
        except json.JSONDecodeError:
            pass
    return out


def _turn_key(tag: str, seen: dict[str, int]) -> str:
    n = seen.get(tag, 0)
    seen[tag] = n + 1
    return f"{tag}#{n}"


def _full_replies(run: Path, rows: list[dict]) -> dict[str, str]:
    try:
        from mechanist.humanize import full_replies
    except ImportError:
        return {}
    return full_replies(run, rows)


def esc(x) -> str:
    return html.escape(str(x))


def render_turn(r: dict) -> str:
    """Plain words for one turn, from the transcript's one-line summary."""
    t = r.get("text", "")
    if " → " in t and not t.startswith(("ANSWER", "VERDICT", "MODEL", "THREADS", "PLAN", "LINKAGE")):
        purpose, call = t.split(" → ", 1)
        return f'<p>Checking: {esc(purpose) or "(no purpose given)"}</p><code class="call">{esc(call)}</code>'
    m = re.match(r"ANSWER \(([^)]*)\): (.*)", t, re.DOTALL)
    if m:
        return (f'<p class="answer"><span class="badge {esc(m.group(1))}">{esc(m.group(1) or "answer")}</span> '
                f'{esc(m.group(2))}</p>')
    m = re.match(r"VERDICT (\w+): (.*?)(?: \| (\d+) demands(?:: (.*))?)?$", t, re.DOTALL)
    if m:
        dem = [d.strip() for d in (m.group(4) or "").split("; ") if d.strip()]
        out = f'<p><span class="badge {esc(m.group(1))}">{esc(m.group(1))}</span> {esc(m.group(2))}</p>'
        if m.group(3):
            out += f'<p class="sub">{esc(m.group(3))} demands' + (":" if dem else "") + "</p>"
            out += "<ul>" + "".join(f"<li>{esc(d)}</li>" for d in dem) + "</ul>" if dem else ""
        return out
    if t.startswith("THREADS: "):
        items = [x.strip() for x in t[9:].split(" | ") if x.strip()]
        return "<p>Opening threads:</p><ol>" + "".join(f"<li>{esc(x)}</li>" for x in items) + "</ol>"
    if t.startswith("MODEL: "):
        title, _, rest = t[7:].partition(" — ")
        return f'<p class="model"><b>{esc(title)}</b></p><p>{esc(rest)}</p>'
    if t.startswith("HYPOTHESES: "):
        return "<p>Candidate mechanisms:</p><ol>" + "".join(
            f"<li>{esc(x)}</li>" for x in t[12:].split(" | ") if x.strip()) + "</ol>"
    for head in ("PLAN: ", "LINKAGE: "):
        if t.startswith(head):
            return f"<p><b>{esc(head[:-2].title())}:</b> {esc(t[len(head):])}</p>"
    return f"<p>{esc(t)}</p>"


def avatar(role: str) -> tuple[str, str]:
    if role.startswith("analyst"):
        return role.split()[-1], "ana"
    if role.startswith("linkage scholar"):
        return role.split()[-1], "lit"
    if role.startswith("panel: "):
        return "".join(w[0] for w in role[7:].split()[:2]).upper(), "panel"
    return ROLE_STYLE.get(role, (role[:2].upper(), "sys"))


def build(run: Path, live: bool = False, artifact: bool = False) -> str:
    """The page; artifact=True leaves out the outer document (the artifact host wraps it)."""
    rows = []
    p = run / "transcript.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    qs = questions(run)
    rid_q = {rid: qid for qid, q in qs.items() for rid in q.get("result_ids") or []}
    hum = {}
    if (run / "humanized.json").exists():
        try:
            hum = json.loads((run / "humanized.json").read_text())
        except json.JSONDecodeError:
            hum = {}
    full = _full_replies(run, rows)
    seen: dict[str, int] = {}
    chans: dict[str, dict] = {}
    pending: dict[str, list] = {}  # tool name -> open calls waiting for their result (older transcripts)
    last_ch = ("other", "Other")
    for r in rows:
        if r.get("role") == "tool":
            tag = r.get("for") or ""
            qid = rid_q.get(r.get("tag", ""))
            tool = (re.match(r"\[R\d+\] (\w+)", r.get("text", "")) or [None, ""])[1]
            if not tag and pending.get(tool):
                tag = pending[tool].pop(0)
            ch = channel(tag)[0] if tag else last_ch[0]
            key = qid or (thread_key(tag) if tag else None)
            msg = {"tool": True, "text": r.get("text", ""), "time": r.get("time", ""), "toolname": tool}
        else:
            tag = r.get("tag", "")
            ch, title = channel(tag) if r.get("role") != "harness" else last_ch
            last_ch = (ch, title)
            chans.setdefault(ch, {"title": title, "posts": [], "threads": {}})
            key = thread_key(tag)
            k = _turn_key(tag, seen)
            msg = {"role": r.get("role", ""), "tag": tag, "text": r.get("text", ""), "time": r.get("time", ""),
                   "replayed": r.get("replayed", False), "model": r.get("model", ""), "hum": hum.get(k),
                   "full": full.get(k, ""), "results": []}
            m = re.search(r" → (\w+)\(", r.get("text", ""))
            if m:
                pending.setdefault(m.group(1), []).append(tag)
        c = chans.setdefault(ch, {"title": channel(tag)[1] if tag else "Other", "posts": [], "threads": {}})
        if key:
            th = c["threads"].get(key)
            if th is None:
                th = c["threads"][key] = {"key": key, "msgs": [], "time": msg["time"]}
                c["posts"].append(("thread", key))
            if msg.get("tool"):  # a result goes under the step that asked for it
                caller = next((m for m in reversed(th["msgs"]) if not m.get("tool")
                               and f" → {msg['toolname']}(" in m["text"] and len(m["results"]) == 0), None)
                if caller is not None:
                    caller["results"].append(msg["text"])
                    continue
            th["msgs"].append(msg)
        else:
            c["posts"].append(("msg", msg))
    order = [ch for _, ch, _ in CHANNELS] + ["other"]
    order = list(dict.fromkeys(order))
    nav, body = [], []
    for ch in order:
        if ch not in chans:
            continue
        c = chans[ch]
        n = sum(1 for k, _ in c["posts"])
        nav.append(f'<a href="#{ch}"># {esc(ch)} <span>{n}</span></a>')
        body.append(f'<section id="{ch}"><h2># {esc(ch)}</h2><p class="chdesc">{esc(c["title"])}</p>')
        for kind, item in c["posts"]:
            body.append(render_thread(c["threads"][item], qs) if kind == "thread" else render_msg(item))
        body.append("</section>")
    name = esc(re.sub(r"^p\d+_[A-Za-z]+_(multi|single)_", "", run.name))
    return (HEAD + BODY if artifact else PAGE).replace("{{name}}", name).replace("{{nav}}", "\n".join(nav)).replace("{{body}}", "\n".join(body)) \
        .replace("{{updated}}", time.strftime("%Y-%m-%d %H:%M:%S")) \
        .replace("{{live}}", "true" if live else "false") \
        .replace("{{livebadge}}", '<label class="live"><input type="checkbox" id="follow" checked> live · follow '
                 'newest</label>' if live else "")


def _prose(text: str) -> str:
    out, items = [], []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(("- ", "* ", "• ")):
            items.append(f"<li>{esc(line[2:])}</li>")
            continue
        if items:
            out.append("<ul>" + "".join(items) + "</ul>")
            items = []
        if line:
            out.append(f"<p>{esc(line)}</p>")
    if items:
        out.append("<ul>" + "".join(items) + "</ul>")
    return "".join(out)


def _original(full: str) -> str:
    txt = full
    try:
        txt = json.dumps(json.loads(re.sub(r"^```(json)?|```$", "", full.strip(), flags=re.MULTILINE)), indent=1)
    except (json.JSONDecodeError, ValueError):
        pass
    return txt[:30000] + ("\n… (cut for length)" if len(txt) > 30000 else "")


def _details(m: dict, shown_as_written: bool) -> str:
    """Everything technical, folded: the call and its results, evidence ids, the message as logged, the full reply."""
    parts = []
    h = m.get("hum") or {}
    if h.get("ok") and h.get("evidence"):
        parts.append('<div class="ev">' + "".join(f"<span>{esc(e)}</span>" for e in h["evidence"]) + "</div>")
    if " → " in m["text"] and not m["text"].startswith(("ANSWER", "VERDICT", "MODEL", "THREADS", "PLAN", "LINKAGE")):
        parts.append(f'<code class="call">{esc(m["text"].split(" → ", 1)[1])}</code>')
    for res in m.get("results") or []:
        head, _, rest = res.partition(": ")
        parts.append(f'<div class="res"><b>{esc(head)}</b><pre>{esc(rest)}</pre></div>')
    if not shown_as_written:
        parts.append(f'<div class="asw"><b>As logged</b>{render_turn(m)}</div>')
    if h and not h.get("ok") and h.get("why"):
        parts.append(f'<p class="sub">Shown as written: {esc(h["why"])}</p>')
    if m.get("full"):
        parts.append(f'<div class="res"><b>Full reply</b><pre>{esc(_original(m["full"]))}</pre></div>')
    return f'<details class="tech"><summary>Details</summary>{"".join(parts)}</details>' if parts else ""


def render_msg(m: dict, inner: bool = False) -> str:
    if m.get("tool"):
        head, _, rest = m["text"].partition(": ")
        return (f'<details class="tool"><summary>{esc(head)} result</summary><pre>{esc(rest)}</pre></details>')
    ini, cls = avatar(m["role"])
    meta = esc(m["time"][11:19]) + (' · <i>replayed</i>' if m.get("replayed") else "") + \
        (f' · {esc(m["model"])}' if m.get("model") else "")
    return (f'<div class="msg{" inner" if inner else ""}" data-role="{esc(m["role"])}">'
            f'<div class="av {cls}">{esc(ini)}</div><div class="bubble"><div class="who">{esc(m["role"])}'
            f'<span class="meta">{meta}</span></div>{_body(m)}</div></div>')


def _body(m: dict) -> str:
    h = m.get("hum") or {}
    t = m["text"]
    if " → " in t and not t.startswith(("ANSWER", "VERDICT", "MODEL", "THREADS", "PLAN", "LINKAGE")):
        purpose = t.split(" → ", 1)[0]
        return f'<p>Checking: {esc(purpose) or "(no purpose given)"}</p>' + _details(m, False)
    if h.get("ok"):
        badge = ""
        mm = re.match(r"(?:ANSWER \(([^)]*)\)|VERDICT (\w+))", t)
        if mm:
            b = mm.group(1) or mm.group(2)
            badge = f'<span class="badge {esc(b)}">{esc(b)}</span>'
        return badge + _prose(h["text"]) + _details(m, False)
    return render_turn(m) + _details(m, True)


def render_thread(th: dict, qs: dict) -> str:
    q = qs.get(th["key"], {})
    analysts = [m for m in th["msgs"] if not m.get("tool")]
    answer = next((m for m in reversed(analysts) if m["text"].startswith(("ANSWER", "PLAN", "LINKAGE"))), None)
    role = analysts[0]["role"] if analysts else "analyst"
    head = q.get("text") or (analysts[0]["text"] if analysts else th["key"])
    step = f'<span class="step">{esc(q["step"])} {esc(q.get("step_label", ""))}</span>' if q.get("step") else ""
    status = q.get("status") or ""
    replies = "".join(render_msg(m, inner=True) for m in th["msgs"] if m is not answer)
    n_tools = sum(1 for m in th["msgs"] if m.get("tool"))
    summary = render_msg(answer) if answer else '<p class="sub">(no answer yet)</p>'
    return (f'<div class="thread" data-role="{esc(role)}"><div class="qhead"><span class="qid">{esc(th["key"])}</span>'
            f'{f"<span class=badge {esc(status)}>{esc(status)}</span>" if status else ""}{step}'
            f'<p>{esc(head[:600])}</p></div>{summary}'
            f'<details class="replies" id="r-{esc(th["key"])}"><summary>{len(analysts)} steps · {n_tools} tool results</summary>'
            f'{replies}</details></div>')


HEAD = """<title>{{name}} Team Chat</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap">
<style>
/* Layout: a chat client. Channel rail on the left (a strip on phones), one column of posts; each question is a card
   whose steps and raw data fold away. */
:root{--bg:#f6f6f3;--panel:#ffffff;--ink:#1e1f1c;--mute:#666861;--line:#e2e2dc;--rail:#262a2e;--railink:#d5d8da;
--railhi:#3a4045;--accent:#2b6f8f;--ok:#2f7a4f;--warn:#a5741e;--bad:#a8432f;--exh:#7b5a96;--code:#eef0ec;
--pi:#a8432f;--plan:#2b6f8f;--syn:#6e4d93;--exp:#2f7a4f;--lit:#22767a;--ana:#545c66;--panel2:#87692a;--sys:#80827c;
--sans:"IBM Plex Sans",-apple-system,"Segoe UI",Roboto,sans-serif;--mono:"IBM Plex Mono",ui-monospace,Menlo,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#17191b;--panel:#202326;--ink:#e9eae6;
--mute:#a0a39c;--line:#33373b;--rail:#101214;--railink:#c6c9cc;--railhi:#262a2e;--accent:#6fb4d4;--ok:#5fb383;
--warn:#d1a24e;--bad:#e07a62;--exh:#ad8fd0;--code:#2a2e31;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#17191b;--panel:#202326;--ink:#e9eae6;--mute:#a0a39c;--line:#33373b;--rail:#101214;
--railink:#c6c9cc;--railhi:#262a2e;--accent:#6fb4d4;--ok:#5fb383;--warn:#d1a24e;--bad:#e07a62;--exh:#ad8fd0;
--code:#2a2e31;color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 var(--sans);display:flex;min-height:100%}
nav{width:236px;flex:none;background:var(--rail);color:var(--railink);padding:16px 10px;position:sticky;
top:env(safe-area-inset-top,0px);height:100vh;overflow:auto}
nav h1{font-size:15px;margin:0 6px 2px;color:var(--railink);font-weight:700;text-wrap:balance}
nav .upd{font-size:11px;margin:0 6px 14px;opacity:.75;font-family:var(--mono)}
nav a{display:flex;justify-content:space-between;gap:8px;color:inherit;text-decoration:none;padding:5px 8px;
border-radius:6px;font-size:13px}nav a:hover,nav a:focus-visible{background:var(--railhi);outline:none}
nav a span{opacity:.6;font-variant-numeric:tabular-nums}
nav input{width:100%;margin:12px 0 0;padding:6px 8px;border-radius:6px;border:1px solid var(--railhi);
background:var(--railhi);color:var(--railink);font:inherit}
nav .live{display:block;font-size:12px;margin:0 6px 12px;color:var(--ok)}nav .live input{width:auto;margin:0 4px 0 0}
main{flex:1;min-width:0;max-width:980px;padding-block:8px 80px;padding-left:max(24px,env(safe-area-inset-left,0px));
padding-right:max(24px,env(safe-area-inset-right,0px))}
section{margin-top:26px}h2{font-size:18px;margin:0;font-family:var(--mono);font-weight:500}
.chdesc{color:var(--mute);margin:2px 0 12px;font-size:13px}
.msg{display:flex;gap:10px;padding:8px 0}.msg.inner{padding:6px 0 6px 4px}
.av{width:34px;height:34px;flex:none;border-radius:8px;color:#fff;font-weight:700;font-size:11px;display:flex;
align-items:center;justify-content:center;font-family:var(--mono)}
.pi{background:var(--pi)}.plan{background:var(--plan)}.syn{background:var(--syn)}.exp{background:var(--exp)}
.lit{background:var(--lit)}.ana{background:var(--ana)}.panel{background:var(--panel2)}.sys{background:var(--sys)}
.bubble{min-width:0;flex:1}.bubble p{margin:2px 0;max-width:72ch}.who{font-weight:600}
.meta{color:var(--mute);font-weight:400;font-size:12px;margin-left:8px;font-family:var(--mono)}
.sub{color:var(--mute)}
code.call{display:inline-block;font:12px/1.4 var(--mono);background:var(--code);padding:2px 6px;border-radius:4px;
word-break:break-word}
.thread{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px 14px;margin:10px 0}
.qhead p{margin:4px 0 6px;font-weight:600;max-width:80ch}.qid{font-weight:600;margin-right:8px;font-family:var(--mono)}
.step{font-size:12px;color:var(--mute);margin-left:8px}
.badge{font-size:11px;padding:1px 7px;border-radius:10px;background:var(--line);color:var(--ink);margin-right:6px;
font-weight:600;letter-spacing:.02em}
.badge.answered,.badge.pass{background:var(--ok);color:var(--panel)}
.badge.partial,.badge.revise{background:var(--warn);color:var(--panel)}
.badge.unanswerable_with_data{background:var(--exh);color:var(--panel)}
.badge.sign_off{background:var(--accent);color:var(--panel)}
details>summary{cursor:pointer;color:var(--mute);font-size:12px}
details>summary:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
details.replies{margin-top:6px}details.tool{margin:2px 0 2px 44px}
details.tech{margin-top:4px}details.tech>summary{list-style:none}details.tech>summary::-webkit-details-marker{display:none}
details.tech>summary::before{content:"\\25B8  "}details.tech[open]>summary::before{content:"\\25BE  "}
details.tech>*:not(summary){margin-top:6px}
pre{white-space:pre-wrap;word-break:break-word;font:12px/1.45 var(--mono);background:var(--code);color:var(--ink);
padding:8px;border-radius:6px;margin:4px 0;max-height:420px;overflow:auto}
.res b,.asw b{font-size:12px;color:var(--mute);font-weight:500}
.ev span{display:inline-block;font:11px var(--mono);background:var(--code);border-radius:4px;padding:0 5px;
margin:0 4px 4px 0}
ul,ol{margin:4px 0 4px 18px;padding:0;max-width:72ch}.model b{font-size:15px}
@media (max-width:760px){body{display:block}nav{position:sticky;top:env(safe-area-inset-top,0px);width:auto;
height:auto;display:flex;gap:4px;overflow-x:auto;padding:8px max(16px,env(safe-area-inset-left,0px)) 8px
max(16px,env(safe-area-inset-right,0px));z-index:2}nav h1,nav .upd,nav .live,nav input{display:none}
nav a{flex:none}main{padding-left:max(16px,env(safe-area-inset-left,0px));
padding-right:max(16px,env(safe-area-inset-right,0px))}details.tool{margin-left:0}}
@media (prefers-reduced-motion:reduce){*{scroll-behavior:auto!important}}
</style>"""

BODY = """<nav aria-label="Channels"><h1>{{name}}</h1><div class="upd">updated {{updated}}</div>{{livebadge}}{{nav}}
<input id="q" aria-label="Filter messages" placeholder="Filter (role, gene, Q id)…"></nav><main>{{body}}</main><script>
const LIVE={{live}}, K='mechanist-chat:'+document.title;
function save(){try{sessionStorage.setItem(K,JSON.stringify({y:scrollY,
 open:[...document.querySelectorAll('details[open][id]')].map(d=>d.id),q:document.getElementById('q').value,
 follow:document.getElementById('follow')?.checked}))}catch(e){}}
function restore(){let st=null;try{st=JSON.parse(sessionStorage.getItem(K)||'null')}catch(e){}
 if(!st)return;(st.open||[]).forEach(id=>{const d=document.getElementById(id);if(d)d.open=true});
 const f=document.getElementById('follow');if(f&&st.follow===false)f.checked=false;
 const q=document.getElementById('q');if(st.q){q.value=st.q;q.dispatchEvent(new Event('input'))}
 if(f&&f.checked)scrollTo(0,document.body.scrollHeight);else scrollTo(0,st.y||0)}
window.addEventListener('load',()=>{if(LIVE)restore()});
if(LIVE)setInterval(()=>{const busy=document.activeElement&&document.activeElement.id==='q'&&document
 .getElementById('q').value;if(!busy){save();location.reload()}},20000);
document.getElementById('q').addEventListener('input',e=>{const v=e.target.value.toLowerCase();
document.querySelectorAll('.thread,main>section>.msg').forEach(el=>{el.hidden=!!v&&!el.textContent.toLowerCase()
.includes(v)})});</script>"""

PAGE = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">' + HEAD + "</head><body>" + BODY
        + "</body></html>")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("--watch", action="store_true", help="rewrite the page every 15 s")
    ap.add_argument("--artifact", type=Path, help="also write the page for a claude.ai artifact to this file")
    ap.add_argument("--humanize", action="store_true", help="rewrite new messages as plain chat (Haiku 5.5, API key; "
                    "display only, checked against the originals); needs the mechanist environment (uv run)")
    a = ap.parse_args()
    out = a.run / "conversation.html"
    hz = None
    if a.humanize:
        from mechanist.humanize import Humanizer
        hz = Humanizer(a.run)
    while True:
        if hz is not None:
            rows = [json.loads(x) for x in (a.run / "transcript.jsonl").read_text().splitlines() if x.strip()] \
                if (a.run / "transcript.jsonl").exists() else []
            n = hz.update(rows)
            if n:
                ok = sum(1 for v in hz.done.values() if v.get("ok"))
                print(f"{time.strftime('%H:%M:%S')} rewrote {n} messages ({ok}/{len(hz.done)} pass the id/number check)",
                      flush=True)
        tmp = out.with_suffix(".tmp")
        tmp.write_text(build(a.run, live=a.watch))
        tmp.replace(out)  # a reload never sees half a page
        if a.artifact:
            a.artifact.write_text(build(a.run, artifact=True))
        if not a.watch:
            print(f"wrote {out}")
            return
        time.sleep(15)


if __name__ == "__main__":
    main()
