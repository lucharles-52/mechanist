"""Cell-circuit figures for the interpretation pages (display only; no stage reads them).

Cells sit around a ring as NIH BioArt icons, arrows between them are labelled with the mediator, the line style says how
each step is known (observed, likely, analogy, speculative) and the disease features the circuit produces sit in the
centre. The figure is drawn by code from a circuit's edges, or from the evidence ledger's links:

- each end of a step is read for the cell types it names (outside brackets first); a step whose two ends are the same
  cell type is a state of that cell, and its genes are written under the cell;
- a step from a cell to molecules records which cell makes them; a later step that starts at a molecule is drawn from
  the cells that make it, labelled with that molecule;
- an end that names no cell is a disease feature (centre) when it reads like one, or a trigger (a ring node without a
  cell) otherwise.

Icons are written once per page as an SVG sprite (icon_defs) and each figure points at them, so a page with many figures
stays small.
"""

from __future__ import annotations

import base64
import html
import math
import re
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[2] / "assets" / "bioart"

# family key, ring label, words that name it (searched in order of position in the text)
CELLS = [
    ("neutrophil", "Neutrophils", r"neutrophil|granulocyte"),
    ("macrophage", "Macrophages", r"macrophage|\bcd163\+?|\bmacs?\b"),
    ("monocyte", "Monocytes", r"monocyte"),
    ("langerhans_cell", "Langerhans cells", r"langerhans"),
    ("dendritic_cell", "Dendritic cells", r"dendritic|\bm?dcs?\b|\bpdcs?\b"),
    ("mast_cell", "Mast cells", r"\bmast[ -]cell"),
    ("eosinophil", "Eosinophils", r"eosinophil"),
    ("fibroblast", "Fibroblasts", r"fibroblast|strom(a|al)\b"),
    ("plasma_cell", "Plasma cells", r"plasma[ -]?cell|plasmablast|antibody-secreting|\bb/plasma"),
    ("b_cell", "B cells", r"\bb[ -]?cells?\b|\bb/plasma|lymphoid niche"),
    ("nk_cell", "NK cells", r"\bnk\b|natural killer"),
    ("t_cell", "T cells", r"\bt[ -]?cells?\b|\bcd[48]\+?(?=\s|$)|\btregs?\b|\bth(1|2|17)\b|lymphocyte"),
    ("blood_vessel", "Vessels", r"endotheli|vessels?\b|vascul|pericyte|capillar"),
    ("skin", "Keratinocytes", r"keratinocyte|epiderm|epitheli"),
    ("antibody", "Antibody", r"antibod|immunoglobulin|immune[ -]complex"),
    ("complement_c1", "Complement", r"complement"),
]
FALLBACK = [("macrophage", r"myeloid (cells?|niche|compartment|lineage)")]  # only when nothing more specific is named
ICON_OF = {"neutrophil": ("activated_neutrophil", r"activat|primed|inflammatory"),
           "macrophage": ("activated_macrophage", r"activat|inflammatory|recruited|sender")}
OUTCOME_RE = re.compile(r"ulcer|undermin|matrix|tissue|destruct|damage|necro|pathergy|wound|re-?epitheli|granulation|"
                        r"lesion|burden|progress|disease|scar|border|loss of|breakdown|injury|blister|detach", re.IGNORECASE)
# words that make an end a cell even when no family above names it (drawn as a generic cell)
CELLISH_RE = re.compile(r"\bcells?\b|\bclon(e|es|al)\b|lineage|population|subset|subtype|\w{3,}cytes?\b|\w{3,}blasts?\b|"
                        r"\btumou?r\b", re.IGNORECASE)
GENE_RE = re.compile(r"(?<![\w-])(?:IL-?\d+[αβA-F]?|TNF-?α|IFN-?[γαβ]|TGF-?β\d?|NF-[κk]B|[A-Z][A-Z0-9]{1,}(?:-[A-Z0-9]+)*)"
                     r"(?![\w])")
MARKER_RE = re.compile(r"CD\d+[A-Z]?|HLA-?\w*|MHC-?\w*")  # cell markers name the cell, not the mediator
NOT_GENE = {"PG", "DE", "UP", "DOWN", "AND", "OR", "NOT", "TLS", "IBD", "DAMP", "DAMPS", "RNA", "DNA", "II", "NB", "MHC",
            "HLA", "JAK", "STAT", "ECM", "ROS", "NET", "NETS", "AS", "IS", "OF", "THE", "FC", "CD", "NK", "DC", "T",
            "B", "PMID", "SSP", "UMAP", "QC"}
STRENGTH = {"bench": 4, "observed": 3, "analogy": 2, "likely": 1, "speculative": 0, "refuted": -1}
# bench / refuted: a version's submitted result shows the step or argues against it (docs/VERSIONS.md)
STYLE = {"bench": ("var(--ink)", 4.4, ""), "observed": ("var(--accent)", 3.2, ""), "likely": ("var(--ink-2)", 1.8, ""),
         "analogy": ("var(--ink-2)", 1.8, "2 4"), "speculative": ("var(--muted)", 1.6, "6 4"),
         "refuted": ("var(--sel)", 1.6, "1 3")}


def esc(x) -> str:
    return html.escape(str(x if x is not None else ""))


def _outside(text: str) -> str:
    return re.sub(r"\([^)]*\)", " ", text)


def cells_in(text: str) -> list[str]:
    """Cell families an end of a step names, in order; outside brackets first, then inside."""
    for part in (_outside(text), text):
        found = []
        for key, _label, pat in CELLS:
            m = re.search(pat, part, re.IGNORECASE)
            if m:
                found.append((m.start(), key))
        if not found:
            for key, pat in FALLBACK:
                m = re.search(pat, part, re.IGNORECASE)
                if m:
                    found.append((m.start(), key))
        if found:
            out = []
            for _, k in sorted(found):
                if k not in out:
                    out.append(k)
            return out
    return []


def norm(g: str) -> str:
    """One spelling per molecule: IL-1α -> IL1A, TNF-α -> TNFA."""
    return g.upper().replace("-", "").replace("Α", "A").replace("Β", "B").replace("Γ", "G").replace("Κ", "K")


def genes_in(text: str, limit: int = 4, markers: bool = False) -> list[str]:
    out = []
    for g in GENE_RE.findall(text or ""):
        g = g.strip("-")
        if g.upper() in NOT_GENE or len(g) < 2 or re.fullmatch(r"[A-Z]\d+|R\d{3,}|F\d{3,}|S\d+|L\d+|Q\d+", g):
            continue
        if not markers and MARKER_RE.fullmatch(g):
            continue
        if g not in out:
            out.append(g)
        if len(out) >= limit:
            break
    return out


def _short(text: str, words: int = 4) -> str:
    w = re.sub(r"\s+", " ", _outside(text)).strip().split(" ")
    return " ".join(w[:words]) + ("…" if len(w) > words else "")


def target_cells(text: str) -> list[str]:
    """The cells an end names, unless it reads as a disease feature first ("perivascular matrix loss")."""
    cs = cells_in(text)
    o = OUTCOME_RE.search(_outside(text))
    if cs and o:
        first = min((m.start() for _k, _l, pat in CELLS for m in [re.search(pat, _outside(text), re.IGNORECASE)] if m),
                    default=len(text))
        if o.start() < first:
            return []
    return cs


def _label(text: str, words: int = 5, chars: int = 34) -> str:
    """A ring name for an end no cell family covers: outside brackets, up to the first clause break, a few words."""
    t = re.sub(r"\s+", " ", _outside(text)).strip(" .;,:")
    head = re.split(r"[;,:]\s| [–—-] ", t, maxsplit=1)[0]
    if len(head.split()) >= 2:
        t = head
    w = t.split(" ")
    out = " ".join(w[:words])
    while len(out) > chars and " " in out:
        out = out.rsplit(" ", 1)[0]
    if len(out) > chars:
        out = out[: chars - 1]
    if out != t:
        out = re.sub(r"(\s+(and|or|of|the|in|to|with|by|vs|per|a|an|from|on|at))+$", "", out, flags=re.IGNORECASE) + "…"
    return out[:1].upper() + out[1:]


def _words(text: str) -> list[str]:
    return re.sub(r"[^0-9a-zα-ω]+", " ", _outside(text).lower()).split()


def build_graph(edges: list[dict]) -> dict:
    """edges: [{source, target, via, status}] -> {"ring": [node], "centre": [str], "arrows": [arrow], "steps": [str]}.

    Every step is shown: as an arrow, as genes under a cell (a cell's product or state), or as a disease feature in the
    centre ("steps" says which, per edge). An end that names no known cell family and is no disease feature is its own
    ring node, named from its text; ends that name the same thing (one's words lead the other's) share the node."""
    nodes: dict[str, dict] = {}
    centre: list[str] = []
    makes: dict[str, set[str]] = {}
    shown: list[str] = [""] * len(edges)

    def node(key: str, mention: str = "", trigger: bool = False) -> str:
        if key not in nodes:
            label = next((lab for k, lab, _ in CELLS if k == key), key.split(":", 1)[-1])
            nodes[key] = {"id": key, "label": label, "sub": [], "mentions": [], "trigger": trigger}
        if mention:
            nodes[key]["mentions"].append(mention)
        return key

    def named(text: str, trigger: bool = False) -> str:
        """The ring node for an end no cell family covers; reused when another end names the same thing."""
        w = _words(text)
        for k, nd in nodes.items():
            if k.startswith("trigger:") and w and (w[: len(nd["words"])] == nd["words"]
                                                    or nd["words"][: len(w)] == w):
                if not trigger:
                    nd["trigger"] = False
                return node(k, text)
        k = node("trigger:" + _label(text), text, trigger=trigger)
        nodes[k]["words"] = w
        return k

    def add_sub(key: str, genes: list[str]) -> None:
        for g in genes:
            if g not in nodes[key]["sub"]:
                nodes[key]["sub"].append(g)

    def outcome(text: str) -> str:
        t = _short(text, 5)
        if t not in centre:
            centre.append(t)
        return t

    for i, e in enumerate(edges):  # pass 1: who makes what, and cell states
        s, t = cells_in(e.get("source", "")), target_cells(e.get("target", ""))
        tg = genes_in(e.get("target", ""), 8)
        if s and not t and not OUTCOME_RE.search(e.get("target", "")) and tg:
            for g in tg:
                makes.setdefault(norm(g), set()).update(s)
            for k in s:
                add_sub(node(k, e.get("source", "")), genes_in(e.get("target", ""), 3))
            shown[i] = "cell"
        elif s and t and set(s) == set(t):
            for k in s:
                add_sub(node(k, e.get("source", "")), genes_in(e.get("target", ""), 3))
                node(k, e.get("target", ""))
            shown[i] = "cell"
    arrows: dict[tuple[str, str], dict] = {}
    for i, e in enumerate(edges):  # pass 2: arrows
        if shown[i]:
            continue
        src, tgt, via = e.get("source", ""), e.get("target", ""), e.get("via", "")
        status = e.get("status") if e.get("status") in STYLE else "speculative"
        s, t = cells_in(src), target_cells(tgt)
        label = (genes_in(via, 2) or genes_in(_outside(src), 2)) if s else []
        if not t and OUTCOME_RE.search(tgt):
            t = ["centre:" + outcome(tgt)]
        if not s:
            mol = genes_in(src)
            s = sorted({k for g in mol for k in makes.get(norm(g), ())})
            label = mol[:3] or genes_in(via, 2)
            if not s:
                if OUTCOME_RE.search(src) and not mol and t and t[0].startswith("centre:"):
                    shown[i] = "centre"  # an outcome feeding an outcome: shown by the centre list
                    continue
                s = [named(src, trigger=True)]
        for k in s:
            node(k, src)
        if not t:
            t = [named(tgt)]
        else:
            for k in t:
                if not k.startswith("centre:"):
                    node(k, tgt)
        pairs = [(a, b) for a in s for b in t if a != b]
        if not pairs:  # a cell acting on itself: a state of that cell
            for k in s:
                add_sub(k, genes_in(tgt, 3) or label)
            shown[i] = "cell"
            continue
        for a, b in pairs:
            arr = arrows.setdefault((a, b), {"from": a, "to": b, "labels": [], "status": status, "notes": [],
                                             "steps": []})
            for g in label or ([_short(via, 3)] if via else []):
                if g and g not in arr["labels"] and len(arr["labels"]) < 3:
                    arr["labels"].append(g)
            if STRENGTH[status] > STRENGTH[arr["status"]]:
                arr["status"] = status
            arr["notes"].append(f"{src} → {tgt}" + (f" ({via})" if via else ""))
            if i + 1 not in arr["steps"]:
                arr["steps"].append(i + 1)
        shown[i] = "arrow"
    used = {a["from"] for a in arrows.values()} | {a["to"] for a in arrows.values()}
    ring = [n for k, n in nodes.items() if k in used or n["sub"]]
    for n in ring:
        n.pop("words", None)
        n["icon"] = icon_key(n["id"], " ".join(n["mentions"]))
    return {"ring": ring, "centre": centre, "arrows": list(arrows.values()), "steps": shown}


def icon_key(family: str, mentions: str = "") -> str:
    if family.startswith("trigger:"):
        if CELLISH_RE.search(family.split(":", 1)[1]):
            return "cell"  # a cell no family covers: the generic cell, under its own name
        return "cytokines" if re.search(r"systemic|serum|circulat|unmeasured|signal", family, re.IGNORECASE) else ""
    alt = ICON_OF.get(family)
    if alt and re.search(alt[1], mentions, re.IGNORECASE) and (ASSETS / f"{alt[0]}.png").exists():
        return alt[0]
    return family if (ASSETS / f"{family}.png").exists() else ""


# the generic cell (drawn here, not a BioArt icon): membrane, cytoplasm and nucleus in the page's own colours
GENERIC_CELL = ('<symbol id="bio-cell" viewBox="0 0 100 100"><ellipse cx="50" cy="52" rx="38" ry="33" '
                'fill="var(--accent-soft)" stroke="var(--ink-2)" stroke-width="3"/><ellipse cx="57" cy="47" rx="14" '
                'ry="12" fill="var(--surface)" stroke="var(--ink-2)" stroke-width="2.5"/><circle cx="60" cy="45" r="4" '
                'fill="var(--ink-2)"/><circle cx="32" cy="62" r="3" fill="var(--ink-2)" opacity=".5"/><circle cx="40" '
                'cy="72" r="2.5" fill="var(--ink-2)" opacity=".5"/></symbol>')


def icon_defs(keys) -> str:
    """The page's icon sprite: one hidden SVG with a symbol per icon; figures point at #bio-<key>."""
    out = []
    for k in sorted({k for k in keys if k}):
        p = ASSETS / f"{k}.png"
        if k == "cell":
            out.append(GENERIC_CELL)
        elif p.exists():
            uri = "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()
            out.append(f'<symbol id="bio-{k}" viewBox="0 0 100 100"><image href="{uri}" x="0" y="0" width="100" '
                       f'height="100" preserveAspectRatio="xMidYMid meet"/></symbol>')
    return f'<svg width="0" height="0" style="position:absolute" aria-hidden="true"><defs>{"".join(out)}</defs></svg>'


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = str(text).split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    return lines + ([cur] if cur else [])


def ring_order(ring: list[dict], arrows: list[dict]) -> list[dict]:
    """Seat connected cells next to each other: the order (first cell fixed) with the least squared ring distance
    summed over the arrows; tried exhaustively up to 8 cells, else the order of first mention."""
    n = len(ring)
    if n <= 3 or n > 8:
        return ring
    import itertools
    idx = {nd["id"]: i for i, nd in enumerate(ring)}
    links = [(idx[a["from"]], idx[a["to"]]) for a in arrows if a["from"] in idx and a["to"] in idx]
    best, best_cost = list(range(n)), None
    for perm in itertools.permutations(range(1, n)):
        order = (0, *perm)
        seat = {c: k for k, c in enumerate(order)}
        cost = sum(min(abs(seat[a] - seat[b]), n - abs(seat[a] - seat[b])) ** 2 for a, b in links)
        if best_cost is None or cost < best_cost:
            best, best_cost = list(order), cost
    return [ring[i] for i in best]


def figure(graph: dict, fid: str, title: str = "", labels: bool = True) -> str:
    """The ring figure as inline SVG. labels=False draws a thumbnail (cells and arrows only)."""
    ring, centre, arrows = graph["ring"], graph["centre"], graph["arrows"]
    if not ring:
        return ""
    W, H = 900, 760
    cx, cy = W / 2, H / 2 - 4
    ring = ring_order(ring, arrows)
    n = len(ring)
    R = 268 if n >= 6 else 250 if n >= 4 else 215 if n == 3 else 180
    pos = {}
    for i, nd in enumerate(ring):
        tilt = math.pi / n if (n % 2 == 0 and centre and n > 2) else 0.0  # keep cells off the centre box's sides
        a = -math.pi / 2 + tilt + 2 * math.pi * i / n if n > 1 else -math.pi / 2
        if n == 2:
            a = math.pi if i == 0 else 0.0
        pos[nd["id"]] = (cx + R * math.cos(a), cy + R * math.sin(a))
    lines = [_wrap(c, 22)[:2] for c in centre[:4]]
    bh = (24 + sum(len(x) for x in lines) * 15 + 8) if centre else 0
    bw = 176
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)}: cell circuit" class="cellfig">',
         "<defs>" + "".join(
             f'<marker id="{fid}-{k}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="{6 if k == "observed" else 7}" '
             f'markerHeight="{6 if k == "observed" else 7}" orient="auto-start-reverse"><path d="M0,0L10,5L0,10z" '
             f'fill="{col}"/></marker>' for k, (col, _w, _d) in STYLE.items()) + "</defs>"]
    if centre:
        s.append(f'<rect x="{cx - bw / 2}" y="{cy - bh / 2}" width="{bw}" height="{bh}" rx="12" fill="var(--accent-soft)" '
                 f'stroke="var(--accent)" stroke-width="1"/>')
        s.append(f'<text x="{cx}" y="{cy - bh / 2 + 17}" font-size="10.5" font-weight="700" letter-spacing=".08em" '
                 f'text-anchor="middle" fill="var(--ink-2)">DISEASE</text>')
        y = cy - bh / 2 + 34
        for ls in lines:
            for ln in ls:
                s.append(f'<text x="{cx}" y="{y}" font-size="12.5" text-anchor="middle" fill="var(--ink)">{esc(ln)}</text>')
                y += 15

    def endpoint(k: str) -> tuple[float, float]:
        return (cx, cy) if k.startswith("centre:") else pos[k]

    pairs = {(a["from"], a["to"]) for a in arrows}
    # half-width of each cell's footprint: its icon, or its name and the genes under it when wider
    names = {nd["id"]: _wrap(nd["label"], 20)[:2] or [""] for nd in ring}  # a long name under an icon takes two lines
    namew = {nd["id"]: max(46, 4.9 * max(len(x_) for x_ in names[nd["id"]]) + 8, 3.9 * len(", ".join(nd["sub"][:4])) + 6)
             for nd in ring}
    # what each cell covers, for placing arrow labels: (half-width, top, bottom) around its seat
    foot = {}
    for nd in ring:
        if nd.get("icon"):
            foot[nd["id"]] = (namew[nd["id"]], -54, 86 + 18 * (len(names[nd["id"]]) - 1))
        else:
            box = _wrap(nd["label"], 22)[:2] or [""]
            foot[nd["id"]] = (min(170, 16 + 7.2 * max(len(x_) for x_ in box)) / 2, -24 - 8 * len(box), 16 + 8 * len(box))
    def edge(k: str, ux: float, uy: float, r: float) -> float:
        """How far from a cell's seat an arrow starts or stops: r for an icon, the border (plus a gap) for a named box."""
        hw, top, bot = foot[k]
        if r and k in nodes_by and not nodes_by[k].get("icon"):
            hh = (bot - top) / 2 - 12
            return min(hw / abs(ux) if ux else 1e9, hh / abs(uy) if uy else 1e9) + 8
        return r

    nodes_by = {nd["id"]: nd for nd in ring}
    label_svg = []
    placed: list[tuple[float, float, float]] = []
    for a in arrows:
        if a["from"] not in pos or (a["to"] not in pos and not a["to"].startswith("centre:")):
            continue
        (x1, y1), (x2, y2) = endpoint(a["from"]), endpoint(a["to"])
        dx, dy = x2 - x1, y2 - y1
        d = math.hypot(dx, dy) or 1
        ux, uy = dx / d, dy / d
        r1 = edge(a["from"], ux, uy, 56)
        r2 = ((min(bw / 2, bh / 2) + 14 if centre else 30) if a["to"].startswith("centre:")
              else edge(a["to"], ux, uy, 58))
        sx, sy, ex, ey = x1 + ux * r1, y1 + uy * r1, x2 - ux * r2, y2 - uy * r2
        hx, hy = (sx + ex) / 2, (sy + ey) / 2
        ox, oy = hx - cx, hy - cy
        on = math.hypot(ox, oy) or 1
        ox, oy = ox / on, oy / on  # outward, from the centre through the chord's middle
        to_centre = a["to"].startswith("centre:")
        seats = [nd["id"] for nd in ring]
        gap = (min(abs(seats.index(a["from"]) - seats.index(a["to"])), n - abs(seats.index(a["from"]) - seats.index(a["to"])))
               if not to_centre and a["to"] in seats else 0)
        reverse = (a["to"], a["from"]) in pairs
        if gap == 1 or n <= 2:  # neighbours: arc outside the ring (a reverse arrow takes the inside)
            inward = reverse and a["from"] > a["to"]
            off = d * (0.12 if inward else 0.30) * (-1 if inward else 1)
            mx, my = hx + ox * off, hy + oy * off
        elif to_centre:
            mx, my = hx + uy * d * 0.08, hy - ux * d * 0.08
        else:  # across the ring: bend until the middle clears the centre box
            nx, ny = uy, -ux
            side = 1.0 if reverse or (hx - cx) * nx + (hy - cy) * ny >= 0 else -1.0
            clear = (math.hypot(bw / 2, bh / 2) + 18 if centre else 40)
            off = side * d * (0.16 if reverse else 0.08)
            for _ in range(40):
                if math.hypot(hx + nx * off / 2 - cx, hy + ny * off / 2 - cy) >= clear:
                    break
                off += side * 12
            mx, my = hx + nx * off, hy + ny * off
        col, w, dash = STYLE[a["status"]]
        s.append(f'<path d="M{sx:.1f},{sy:.1f} Q{mx:.1f},{my:.1f} {ex:.1f},{ey:.1f}" fill="none" stroke="{col}" '
                 f'stroke-width="{w}"{f" stroke-dasharray=\"{dash}\"" if dash else ""} stroke-linecap="round" '
                 f'marker-end="url(#{fid}-{a["status"]})"><title>{esc(a["status"])}: {esc("; ".join(a["notes"])[:600])}'
                 f"</title></path>")
        shown = [x for x in a["labels"] if GENE_RE.fullmatch(x)]
        if labels and shown:
            text = ", ".join(shown)
            half = 4.6 * len(text) + 6

            def at(t: float, sx=sx, sy=sy, mx=mx, my=my, ex=ex, ey=ey) -> tuple[float, float]:
                return ((1 - t) ** 2 * sx + 2 * (1 - t) * t * mx + t * t * ex,
                        (1 - t) ** 2 * sy + 2 * (1 - t) * t * my + t * t * ey)

            def clash(x: float, y: float, half=half) -> int:
                n_ = sum(1 for k_, (px, py) in pos.items()
                         if abs(x - px) < half + foot[k_][0] and foot[k_][1] < y - py < foot[k_][2])
                if centre and abs(x - cx) < bw / 2 + half and abs(y - cy) < bh / 2 + 12:
                    n_ += 1
                return n_ + sum(1 for qx, qy, qh in placed if abs(x - qx) < half + qh and abs(y - qy) < 18)

            ts = [0.35, 0.25, 0.45, 0.18] if to_centre else [0.5, 0.4, 0.6, 0.3, 0.7, 0.22, 0.78]
            nx_, ny_ = uy, -ux
            cands = []
            for k, t in enumerate(ts):
                x_, y_ = at(t)
                if gap == 1:
                    x_, y_ = x_ + ox * 12 * (1 if off > 0 else -1), y_ + oy * 12 * (1 if off > 0 else -1)
                for j, sh in enumerate((0, 16, -16, 30, -30, 44, -44)):
                    cands.append((clash(x_ + nx_ * sh, y_ + ny_ * sh), j, k, x_ + nx_ * sh, y_ + ny_ * sh))
            _, _, _, lx, ly = min(cands)
            placed.append((lx, ly, half))
            label_svg.append(f'<text x="{lx:.1f}" y="{ly + 4:.1f}" font-size="15" font-weight="650" text-anchor="middle" '
                             f'fill="var(--ink)" stroke="var(--surface)" stroke-width="5" paint-order="stroke" '
                             f'stroke-linejoin="round">{esc(text)}</text>')
    for nd in ring:
        x, y = pos[nd["id"]]
        if nd.get("icon"):
            s.append(f'<use href="#bio-{nd["icon"]}" x="{x - 44:.1f}" y="{y - 46:.1f}" width="88" height="88"/>')
            ty = y + 58
        else:
            label = nd["label"]
            tw = min(170, 16 + 7.2 * max(len(x_) for x_ in _wrap(label, 22)[:2]))
            nl = len(_wrap(label, 22)[:2])
            s.append(f'<rect x="{x - tw / 2:.1f}" y="{y - 14 - 8 * nl:.1f}" width="{tw:.1f}" height="{20 + 16 * nl}" rx="10" '
                     f'fill="var(--surface)" stroke="var(--ink-2)" stroke-dasharray="5 4"/>')
            for k, ln in enumerate(_wrap(label, 22)[:2]):
                s.append(f'<text x="{x:.1f}" y="{y - 2 - 8 * (nl - 1) + 16 * k:.1f}" font-size="12.5" font-weight="650" '
                         f'text-anchor="middle" fill="var(--ink)">{esc(ln)}</text>')
            continue
        for k, ln in enumerate(names[nd["id"]]):
            s.append(f'<text x="{x:.1f}" y="{ty + 18 * k:.1f}" font-size="16" font-weight="700" text-anchor="middle" '
                     f'fill="var(--ink)" stroke="var(--surface)" stroke-width="4" paint-order="stroke">{esc(ln)}</text>')
        ty += 18 * (len(names[nd["id"]]) - 1)
        if labels and nd["sub"]:
            s.append(f'<text x="{x:.1f}" y="{ty + 17:.1f}" font-size="13" text-anchor="middle" fill="var(--ink-2)" '
                     f'stroke="var(--surface)" stroke-width="4" paint-order="stroke">{esc(", ".join(nd["sub"][:4]))}</text>')
    # crop the canvas to what is drawn: the cells with their names and genes, the centre box, and the arrows' labels
    xs = ([x + d for k, (x, _) in pos.items() for d in (-namew[k] - 14, namew[k] + 14)]
          + ([cx - bw / 2 - 10, cx + bw / 2 + 10] if centre else []) + [x + d for x, _, h in placed for d in (-h - 6, h + 6)])
    ys = [y for _, y in pos.values()] + ([cy - bh / 2, cy + bh / 2] if centre else []) + [y for _, y, _ in placed]
    x0, x1 = max(0.0, min(xs)), min(float(W), max(xs))
    y0, y1 = max(0.0, min(ys) - 64), min(float(H), max(ys) + 96)
    MW, MH = 640, 540  # a floor on the canvas, so a two-cell figure is drawn at the same scale as a busy one
    if x1 - x0 < MW:
        x0, x1 = max(0.0, (x0 + x1 - MW) / 2), min(float(W), (x0 + x1 + MW) / 2)
    if y1 - y0 < MH:
        y0, y1 = max(0.0, (y0 + y1 - MH) / 2), min(float(H), (y0 + y1 + MH) / 2)
    s[0] = s[0].replace(f'viewBox="0 0 {W} {H}"', f'viewBox="{x0:.0f} {y0:.0f} {x1 - x0:.0f} {y1 - y0:.0f}"')
    return "\n".join(s + label_svg + ["</svg>"])


LEGEND = """<div class="legend">
<span><svg width="34" height="8" aria-hidden="true"><line x1="0" y1="4" x2="34" y2="4" stroke="var(--accent)" stroke-width="3.2"/></svg>observed in this data</span>
<span><svg width="34" height="8" aria-hidden="true"><line x1="0" y1="4" x2="34" y2="4" stroke="currentColor" stroke-width="1.8"/></svg>likely, not tested</span>
<span><svg width="34" height="8" aria-hidden="true"><line x1="0" y1="4" x2="34" y2="4" stroke="currentColor" stroke-width="1.8" stroke-dasharray="2 4"/></svg>published elsewhere</span>
<span><svg width="34" height="8" aria-hidden="true"><line x1="0" y1="4" x2="34" y2="4" stroke="currentColor" stroke-width="1.6" stroke-dasharray="6 4"/></svg>speculative</span>
<span>Labels on arrows: the mediator. Under each cell: what it makes or turns on.</span></div>"""
