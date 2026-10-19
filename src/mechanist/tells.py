#!/usr/bin/env python3
# Vendored unchanged from the everwrite skill (scripts/tells.py) on 2026-10-19; refresh by copying the skill's current file.
"""tells.py: flag the mechanical signs of AI-written prose in markdown or plain text.

Part of the everwrite skill. Standard library only, Python 3.9 or later.

    python tells.py FILE_OR_DIR [...]     check files (directories: every .md and .txt inside)
    python tells.py -                     read stdin
    python tells.py --list                print every rule and the stock-word list
    python tells.py --fix-hidden FILE     strip invisible characters and odd spaces in place, then check

Invisible characters are checked on every line, code included; they are file hygiene, not a
watermark (Claude's and Gemini's text watermarks are statistical and add no characters).

Skips YAML frontmatter, fenced code, inline code, HTML comments, URLs and link
targets, and any region between lines holding only <!-- tells: off --> and <!-- tells: on -->.
Exit 0: no strong tells. Exit 1: at least one strong tell. Exit 2: bad input.
"""

import argparse
import bisect
import json
import re
import sys
import unicodedata
from pathlib import Path

# Everything the checker knows lives in the tables below; a refresh edits them here
# and logs the change in CHANGELOG.md with the RESEARCH.md finding behind it.

# Stock AI vocabulary. Two or more distinct words in one paragraph is a strong tell; one alone is weak.
STOCK_WORDS = [
    "align(?:s|ed)? with", "bolster(?:s|ed|ing)?", "crucial(?:ly)?", "cutting-edge", "deep dive",
    "delve(?:s|d)?", "delving", "diverse array", "elevate(?:s|d)?", "embark(?:s|ed)?", "empower(?:s|ed|ing)?",
    "emphasizing", "enduring", "enhance(?:s|d)?", "enhancing", "ever-evolving", "foster(?:s|ed|ing)?",
    "game[- ]chang(?:er|ing)", "garner(?:s|ed)?", "groundbreaking", "harness(?:es|ed|ing)?", "holistic",
    "interplay", "intricate", "intricacies", "landscape", "leverag(?:e|es|ed|ing)", "meticulous(?:ly)?",
    "multifaceted", "paramount", "pivotal", "profound", "realm", "renowned", "robust", "seamless",
    "showcas(?:e|es|ed|ing)", "streamlin(?:e|es|ed|ing)", "supercharg(?:e|es|ed|ing)", "tapestry",
    "testament", "transformative", "underscor(?:e|es|ed|ing)", "utiliz(?:e|es|ed|ing)", "vibrant",
    "beacon", "commitment to",
]

# (category, severity, regex, hint[, flags]). Matched per paragraph, case-insensitive, over prose with
# code and URLs blanked. Flags: raw (also sees URLs), nocond (skip when the sentence opens a condition:
# "If the value is not set, it is zero" is plain technical prose).
RULES = [
    ("dash", "strong", r"[\u2014\u2013]", "use a period, comma, colon or parentheses"),
    ("dash", "strong", r"(?<=\s)--(?=\s)", "use a period, comma, colon or parentheses"),

    ("residue", "strong", r"\bI hope (?:this|that) helps\b", "delete the chat wrapper"),
    ("residue", "strong", r"\blet me know if\b", "delete the chat wrapper"),
    ("residue", "strong", r"\b(?:would|do) you like me to\b|\bwant me to\b|\bshould I continue\b", "delete the offer"),
    ("residue", "strong", r"\b(?:great|excellent|good) question\b|\byou'?re absolutely right\b", "delete the praise"),
    ("residue", "strong", r"(?:^|[.!?]\s+)(?:certainly|of course|absolutely|sure thing)!", "delete the chat wrapper"),
    ("residue", "strong", r"\bhere'?s (?:a |an )?(?:quick |brief |detailed |short )?(?:breakdown|overview|summary|rundown)\b", "start with the content"),
    ("residue", "strong", r"\blet'?s (?:dive|delve|jump) in(?:to)?\b|\bwithout further ado\b", "start with the content"),
    ("residue", "strong", r"\b(?:in conclusion|in summary|to sum up|to summarize)\b", "end on the last concrete point"),
    ("residue", "strong", r"\bit'?s (?:important|worth|crucial|critical) (?:to note|noting|to remember|to mention)\b|\bworth noting\b", "state the point"),
    ("residue", "strong", r"\b(?:happy|glad) to help\b|\bfeel free to\b|\bdon'?t hesitate to\b|\b(?:excited|thrilled) to (?:share|announce)\b", "say it directly"),
    ("residue", "strong", r"\bhope (?:this|my) (?:message |email |note )?finds you well\b|\bhope you'?re (?:doing )?well\b", "open with the reason for writing"),
    ("residue", "strong", r"\bas of my (?:last|latest) (?:knowledge |training )?(?:update|cutoff)\b|\bas an AI(?: language model)?\b", "delete the disclaimer"),
    ("residue", "strong", r"\bbased on (?:the )?available information\b|\bnot (?:widely|publicly|readily) (?:available|documented|disclosed)\b|\bmaintains a low profile\b", "say what the source does not show, or cut"),

    ("markup", "strong", r"contentReference\[|oaicite|oai_citation|turn\d+(?:search|image|news|file)\d+", "chatbot citation residue"),
    ("markup", "strong", r"\[cite:\s*\d|\(start_span\)|\(end_span\)|grok_card|grok_render|\u3010\d+\u2020|:::writing\{|\[attached_file:\d+\]|\[web:\d+\]", "chatbot citation residue"),
    ("markup", "strong", r"utm_source=(?:chatgpt\.com|openai|copilot\.com)|referrer=grok\.com", "strip the tracking parameter", "raw"),

    ("placeholder", "strong", r"\[(?:insert|your|add|enter) [^\]]{1,40}\]|\(add your [^)]{1,40}\)|\b20\d\d-xx-xx\b", "fill it in or cut it"),

    ("contrast", "strong", r"\bnot (?:just|only|merely|simply)\b[^.!?]{0,120}?\bbut\b", "state the second half directly"),
    ("contrast", "strong", r"\bmore than just\b", "state what it is"),
    ("contrast", "strong", r"\b(?:it|this|that)(?:'s| is| was) not (?:about )?[^.!?,;\n]{1,60}[,;] (?:it|this|that)(?:'s| is| was)\b", "state the second half directly", "nocond"),
    ("contrast", "strong", r"\b(?:is|was|are)(?:n'?t| not)\s+[^.!?,;\n]{1,60}[,;]\s+(?:it|this|that|they)(?:'s|'re| is| was| are)\b", "state the second half directly", "nocond"),
    ("contrast", "strong", r"\b(?:does|do|did)(?:n'?t| not) mean\b[^.!?]{1,80}\.\s+(?:it|this|that|they) means?\b", "state the second half directly"),
    ("contrast", "weak", r"\b(?:does|do|did)(?:n'?t| not) (?!mean\b)(\w+)\b[^.!?]{1,80}\.\s+(?:it|this|that|they) \1s?\b", "if the first sentence only sets up the second, state the second"),
    ("contrast", "weak", r"\b(?:is|was)(?:n'?t| not)\b[^.!?]{1,80}\.\s+(?:it|this|that)(?:'s| is| was)\b", "if the first sentence only sets up the second, state the second"),
    ("contrast", "strong", r"\bno [a-z]+, no [a-z]+, (?:just|only)\b", "say what it is"),
    ("contrast", "strong", r"(?:^|[.!?]\s+)(?:not|no) [^.!?]{1,30}\.\s+(?:not|no) [^.!?]{1,30}\.", "negative listing: say what it is"),
    ("contrast", "weak", r"\brather than\b", "fine when both halves carry information"),

    ("setup", "strong", r"\bhere'?s the thing\b|\bthe thing is,|\blet me be (?:clear|honest)\b|\bto be clear\b|\bdon'?t get me wrong\b", "make the point"),
    ("setup", "strong", r"\bthe (?:honest |uncomfortable |simple |hard )?(?:truth|reality) is\b|\bthe (?:real|deeper) (?:question|issue|problem) is\b", "make the point"),
    ("setup", "strong", r"\bwhat (?:most|many) people (?:get wrong|miss|don'?t realize)\b|\bhere'?s what (?:nobody|no one) tells you\b|\bthe part (?:everyone|most people) miss(?:es)?\b", "make the claim stand alone"),
    ("setup", "strong", r"\bwhat if I told you\b|\blet that sink in\b|\bread that again\b|\bfull stop\.|\bplot twist:", "cut the staging"),
    ("setup", "strong", r"\b(?:the|one) (?:best|key|real|crazy|wild|surprising) (?:part|thing|takeaway|detail)(?: is)?:", "write it as a plain sentence"),
    ("setup", "strong", r"\bthat last part matters\b|\bthis (?:distinction|part|point) matters\b|\bthe key point is\b|\bas you can see\b", "show why instead of saying it matters"),
    ("setup", "weak", r"\bthe (?:whole |main |real )?point (?:here )?is that\b", "make the point"),
    ("setup", "strong", r"\ba tempting approach would be\b|\bone might be tempted to\b|\byou might think\b", "drop the option nobody raised"),
    ("setup", "strong", r"(?:^|[.!?]\s+)(?:honestly|look|real talk)[?,!]", "cut the staged candor"),
    ("setup", "strong", r"\bat its core\b|\bat the end of the day\b|\bin today'?s (?:fast[- ]paced |digital |modern )?world\b|\bin the age of\b", "delete"),

    ("inflation", "strong", r"\b(?:stands|serves) as an? (?:testament|reminder)\b|\bis a testament to\b|\bindelible mark\b", "state the fact"),
    ("inflation", "strong", r"\bplay(?:s|ed|ing)? an? (?:vital|pivotal|crucial|key|significant|central|critical) role\b", "say what it does"),
    ("inflation", "strong", r"\bmark(?:s|ed|ing)? an? (?:pivotal|significant|major|key|new) (?:moment|shift|milestone|turning point|era)\b", "state the fact"),
    ("inflation", "strong", r"\bsetting the stage for\b|\breflect(?:s|ing)? (?:a )?broader\b|\bevolving landscape\b|\bdespite (?:these|its|their) challenges\b", "state the fact"),
    ("inflation", "strong", r"\bthe future looks bright\b|\bexciting times (?:lie )?ahead\b|\ba step in the right direction\b", "end on the last concrete fact"),
    ("inflation", "strong", r"\bnestled\b|\bin the heart of\b|\bactive social media presence\b|\bnatural beauty\b|\bbreathtaking\b", "sales language: say what it is"),

    ("rider", "strong", r",\s+(?:highlighting|underscoring|emphasizing|showcasing|reflecting|symbolizing|ensuring|fostering|cultivating|contributing to|solidifying)\b", "cut the -ing tail or make it a real claim"),

    ("authority", "weak", r"\b(?:experts|observers|critics|analysts|researchers) (?:argue|say|agree|believe|note|suggest|have cited)\b|\bindustry reports\b|\bstudies (?:show|suggest)\b|\bwidely regarded as\b", "name the source or cut"),
    ("copula", "weak", r"\b(?:serves|stands|functions|acts|operates) as (?:a|an|the)\b|\bboasts\b", "use is, are or has"),
    ("association", "weak", r"\b(?:is|was|are|were|been|be) (?:closely |widely |particularly )?(?:associated|connected) with\b|\bin (?:connection|association) with\b", "name the relationship"),
    ("qualifier", "weak", r"\b(?:could|might|may) (?:potentially|possibly)\b|\bit could be argued\b|\barguably\b", "one hedge, only for real doubt"),
    ("adverb", "weak", r"\b(?:really|truly|actually|literally|genuinely|simply|fundamentally|incredibly|quietly|seamlessly|importantly)\b", "cut unless it carries meaning"),
    ("transition", "weak", r"(?:^|[.!?]\s+)(?:additionally|moreover|furthermore|notably),", "use also or and, or merge the sentences"),
]

SMALL_WORDS = {
    "a", "an", "the", "and", "or", "but", "nor", "for", "so", "yet", "of", "in", "on", "at", "to", "by",
    "up", "as", "is", "if", "vs", "via", "per", "with", "from", "into", "onto", "over", "than",
}
# Invisible characters: covert-mark carriers (zero-width, word joiner, soft hyphen, bidi controls, tag and
# variation-selector runs) are strong; odd spaces (no-break, narrow no-break, figure) are weak because some
# typography wants them. Scanned over every line, code included. --fix-hidden deletes the first set and turns
# the second into plain spaces. U+FE0F stays with EMOJI. Research: R-20260928-6.
HIDDEN = re.compile("[\u00AD\u180E\u200B-\u200D\u2060-\u2064\u202A-\u202E\u2066-\u2069\uFEFF\uFE00-\uFE0E"
                    "\U000E0000-\U000E007F\U000E0100-\U000E01EF]")
ODD_SPACE = re.compile("[\u00A0\u2007\u202F]")
EMOJI = re.compile("[\U0001F300-\U0001FAFF\U0001F000-\U0001F2FF\u2600-\u2712\u2715-\u27BF\u2B50\u2B55\uFE0F]")
ARROWS = re.compile("[\u2190-\u21FF\u27F0-\u27FF]")
CURLY = re.compile("[\u201C\u201D\u2018\u2019]")
BOLD_LABEL = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(?:\*\*|__)[^*_]{1,60}?(?::(?:\*\*|__)|(?:\*\*|__)\s*[:\u2014\u2013-])")
HEADING = re.compile(r"^\s{0,3}(#{1,6})(?:[ \t]+(.*))?$")
LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
RULE_LINE = re.compile(r"^\s{0,3}([-*_])(?:\s*\1){2,}\s*$")
FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
OFF = re.compile(r"<!--\s*tells:\s*off\s*-->", re.I)
ON = re.compile(r"<!--\s*tells:\s*on\s*-->", re.I)
WORD = re.compile(r"[A-Za-z0-9][\w'\u2019-]*")
SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]\u201D\u2019]*\s+")
CONDITIONAL = re.compile(r"\b(?:if|when|unless|whether|once|until|because|since|although|though|while|whereas)\b", re.I)

COMPILED = [(r[0], r[1], re.compile(r[2], re.I | re.M), r[3], r[4] if len(r) > 4 else "") for r in RULES]
STOCK = re.compile(r"\b(?:%s)\b" % "|".join(STOCK_WORDS), re.I)


def _blank(match):
    return " " * len(match.group(0))


def blank_code_spans(line):
    """Blank inline code: a run of k backticks up to the next run of exactly k (CommonMark)."""
    out, i, n = list(line), 0, len(line)
    while i < n:
        if line[i] != "`":
            i += 1
            continue
        j = i
        while j < n and line[j] == "`":
            j += 1
        k, p, close = j - i, j, -1
        while True:
            q = line.find("`" * k, p)
            if q < 0:
                break
            r = q
            while r < n and line[r] == "`":
                r += 1
            if r - q == k:
                close = q
                break
            p = r
        if close < 0:
            i = j
            continue
        out[i:close + k] = " " * (close + k - i)
        i = close + k
    return "".join(out)


def blank_comments(line, in_comment):
    """Blank HTML comments; return the line and whether a comment is still open at its end."""
    out, i = [], 0
    while i < len(line):
        if in_comment:
            end = line.find("-->", i)
            if end < 0:
                out.append(" " * (len(line) - i))
                return "".join(out), True
            out.append(" " * (end + 3 - i))
            i, in_comment = end + 3, False
        else:
            start = line.find("<!--", i)
            if start < 0:
                out.append(line[i:])
                break
            out.append(line[i:start])
            i, in_comment = start, True
    return "".join(out), in_comment


def prepare(text, notes=None):
    """Yield (lineno, prose, raw, kind) for every line the checker should read.

    prose has code, comments, URLs and link targets blanked; raw keeps URLs.
    kind is one of text, heading, item, table, rule, blank. Problems with the
    input itself (an unclosed fence) are appended to notes as (lineno, message).
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    i = 0
    if lines and lines[0].strip() == "---":
        for j in range(1, len(lines)):
            if lines[j].strip() in ("---", "..."):
                i = j + 1
                break
    fence, fence_line = None, 0
    in_comment = False
    off = False
    prev_kind = "blank"
    for n in range(i, len(lines)):
        line = lines[n]
        lineno = n + 1
        if fence:
            m = FENCE.match(line)
            if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence) and not line.strip()[len(m.group(1)):].strip():
                fence = None
            continue
        if OFF.fullmatch(line.strip()):
            off = True
            continue
        if ON.fullmatch(line.strip()):
            off = False
            continue
        if off:
            continue
        m = FENCE.match(line)
        if m and not in_comment:
            fence, fence_line = m.group(1), lineno
            continue
        line, in_comment = blank_comments(line, in_comment)
        line = blank_code_spans(line)
        raw = line
        prose = re.sub(r"<https?://[^>\s]*>|https?://\S+|www\.\S+", _blank, line)
        prose = re.sub(r"\]\([^)\n]{0,500}\)", lambda mm: "]" + " " * (len(mm.group(0)) - 1), prose)
        prose = re.sub(r"^\s*\[[^\]]{1,200}\]:\s*\S+.*$", _blank, prose)
        if not prose.strip():
            kind = "blank"
        elif HEADING.match(prose):
            kind = "heading"
        elif RULE_LINE.match(prose):
            # a rule right under a text line is a setext heading underline, not a divider
            kind = "blank" if prev_kind == "text" else "rule"
        elif prose.lstrip().startswith("|"):
            kind = "table"
        elif LIST_ITEM.match(prose):
            kind = "item"
        else:
            kind = "text"
        prev_kind = kind
        yield lineno, prose, raw, kind
    if fence and notes is not None:
        notes.append((fence_line, "code fence never closed; the rest of the file was not checked"))


def paragraphs(prepared):
    """Group prepared lines into paragraphs: headings, list items and table rows start new ones."""
    current = []
    for lineno, prose, raw, kind in prepared:
        if kind in ("blank", "rule"):
            if current:
                yield current
            current = []
            continue
        if kind in ("heading", "item", "table") and current:
            yield current
            current = []
        current.append((lineno, prose, raw, kind))
        if kind in ("heading", "table"):
            yield current
            current = []
    if current:
        yield current


def describe_chars(found):
    """'U+200B ZERO WIDTH SPACE x3, U+00AD SOFT HYPHEN' for the distinct characters in found."""
    counts = {}
    for ch in found:
        counts[ch] = counts.get(ch, 0) + 1
    parts = []
    for ch, k in counts.items():
        name = unicodedata.name(ch, "") or ("TAG" if 0xE0000 <= ord(ch) <= 0xE007F else "UNNAMED")
        parts.append("U+%04X %s%s" % (ord(ch), name, " x%d" % k if k > 1 else ""))
    return ", ".join(parts)


def fix_hidden(text):
    """Delete covert-mark characters and turn odd spaces into plain ones; return (text, changes)."""
    text, a = HIDDEN.subn("", text)
    text, b = ODD_SPACE.subn(" ", text)
    return text, a + b


def same_words(a, b):
    """Two texts that say the same thing word for word, ignoring case, markup and punctuation."""
    key = lambda s: " ".join(re.findall(r"\w+", s.lower()))
    return bool(key(a)) and key(a) == key(b)


def check_text(text, max_words=30, allow_dashes=False, page_title=None):
    """page_title: the title the host prints above the page (a GitHub wiki prints the file name), which the
    first heading must not repeat."""
    hits = []
    words = 0
    notes = []
    prepared = list(prepare(text, notes))

    def add(lineno, severity, category, match, hint):
        hits.append({"line": lineno, "severity": severity, "category": category,
                     "match": " ".join(match.split())[:70], "hint": hint})

    for n, line in enumerate(text.replace("\r\n", "\n").replace("\r", "\n").split("\n"), 1):
        for regex, severity, category, hint in (
                (HIDDEN, "strong", "hidden-char", "invisible character: delete it (--fix-hidden)"),
                (ODD_SPACE, "weak", "odd-space", "plain space unless the typography needs it (--fix-hidden)")):
            found = regex.findall(line)
            if found:
                add(n, severity, category, describe_chars(found), hint)

    for lineno, message in notes:
        add(lineno, "weak", "input", "```", message)

    rules = [ln for ln in prepared if ln[3] == "rule"]
    if len(rules) >= 2:
        for lineno, _, _, _ in rules:
            add(lineno, "weak", "divider", "---", "one rule between every section is decoration")

    # The same thing said twice in a row: a heading straight under a heading with the same words (the title a
    # host prints and the page's own `# Title`, or `## Install` then `### Install`), or a paragraph repeated.
    previous = None
    first_heading = True
    for para in paragraphs(prepared):
        lineno, prose, _, kind = para[0]
        text_now = " ".join(p[1] for p in para)
        if kind == "heading":
            title_now = (HEADING.match(prose).group(2) or "").rstrip(" \t#")
            if first_heading and page_title and same_words(title_now, page_title) and not previous:
                add(lineno, "strong", "repeated-heading", title_now,
                    "the host already prints the page title; start with the first paragraph")
            elif previous and previous[0] == "heading" and same_words(title_now, previous[1]):
                add(lineno, "strong", "repeated-heading", title_now, "the same heading twice in a row; cut one")
            first_heading = False
            previous = ("heading", title_now)
        else:
            if previous and previous[0] == kind and kind in ("text", "item") and same_words(text_now, previous[1]):
                add(lineno, "strong", "repeated-text", text_now, "said twice in a row; cut one")
            previous = (kind, text_now)

    for para in paragraphs(prepared):
        starts, pos = [], 0
        prose_parts, raw_parts = [], []
        for lineno, prose, raw, kind in para:
            starts.append(pos)
            prose_parts.append(prose)
            raw_parts.append(raw)
            pos += len(prose) + 1
        prose_text = "\n".join(prose_parts)
        raw_text = "\n".join(raw_parts)
        first_kind = para[0][3]

        def line_at(offset):
            return para[bisect.bisect_right(starts, offset) - 1][0]

        for category, severity, regex, hint, flags in COMPILED:
            if category == "dash" and allow_dashes:
                continue
            source = raw_text if "raw" in flags else prose_text
            for m in regex.finditer(source):
                if "nocond" in flags:
                    boundary = max(source.rfind(c, 0, m.start()) for c in ".!?\n")
                    if CONDITIONAL.search(source, boundary + 1, m.start()):
                        continue
                add(line_at(m.start()), severity, category, m.group(0), hint)

        found = list(STOCK.finditer(prose_text))
        distinct = {m.group(0).lower() for m in found}
        for m in found:
            add(line_at(m.start()), "strong" if len(distinct) >= 2 else "weak", "vocabulary", m.group(0),
                "plain word or cut; never swap for another listed word")

        for lineno, prose, raw, kind in para:
            if EMOJI.search(prose):
                add(lineno, "strong", "emoji", EMOJI.search(prose).group(0), "remove")
            if kind in ("heading", "item") and ARROWS.search(prose):
                add(lineno, "weak", "decoration", ARROWS.search(prose).group(0), "arrows as decoration")
            if CURLY.search(prose):
                add(lineno, "weak", "curly-quote", CURLY.search(prose).group(0), "use straight quotes if the target does")
            if kind == "item" and BOLD_LABEL.match(prose):
                add(lineno, "strong", "bold-label", BOLD_LABEL.match(prose).group(0).strip(), "plain list or prose")
            if kind == "heading":
                title = (HEADING.match(prose).group(2) or "").rstrip(" \t#")
                significant = [w for w in re.findall(r"[A-Za-z][A-Za-z'\u2019]*", title) if w.lower() not in SMALL_WORDS]
                if len(significant) >= 3 and all(w[0].isupper() for w in significant) and not all(w.isupper() for w in significant):
                    add(lineno, "weak", "title-case", title, "sentence case")

        if first_kind in ("heading", "table"):
            continue
        body = LIST_ITEM.sub(_blank, prose_text, count=1)
        spans, start = [], 0
        for m in SENTENCE_END.finditer(body):
            spans.append((start, m.start()))
            start = m.end()
        spans.append((start, len(body)))
        openings = []
        for s, e in spans:
            sentence = body[s:e]
            count = len(WORD.findall(sentence))
            if not count:
                continue
            words += count
            lineno = line_at(s + len(sentence) - len(sentence.lstrip()))
            if count > max_words:
                add(lineno, "weak", "long-sentence", "%d words: %s" % (count, sentence.strip()[:40]), "split it; one idea per sentence")
            first = WORD.findall(sentence)[:1]
            openings.append((first[0].lower() if first else "", lineno))
            if len(openings) >= 3 and openings[-1][0] and openings[-1][0] == openings[-2][0] == openings[-3][0]:
                add(lineno, "weak", "repeated-opening", openings[-1][0], "merge or vary the openings")

    seen, unique = set(), []
    for h in sorted(hits, key=lambda h: (h["line"], h["severity"] != "strong", h["category"])):
        key = (h["line"], h["category"], h["match"].lower())
        if key not in seen:
            seen.add(key)
            unique.append(h)
    return unique, words


def gather(paths):
    files = []
    for p in paths:
        if p == "-":
            files.append("-")
            continue
        path = Path(p)
        if path.is_dir():
            files.extend(sorted(str(f) for f in path.rglob("*") if f.suffix.lower() in (".md", ".txt") and f.is_file()))
        elif path.is_file():
            files.append(str(path))
        else:
            raise FileNotFoundError(p)
    return files


def read(name):
    if name == "-":
        data = sys.stdin.buffer.read()
    else:
        data = Path(name).read_bytes()
    return data.decode("utf-8-sig", errors="replace")


def print_rules():
    print("Rules (category, severity, hint):")
    for category, severity, _, hint, _ in COMPILED:
        print("  %-16s %-6s %s" % (category, severity, hint))
    print("  %-16s %-6s %s" % ("vocabulary", "both", "strong when two or more distinct words share a paragraph"))
    print("  also: emoji, bold-label, title-case, divider, curly-quote, decoration, long-sentence, repeated-opening")
    print("  also (strong): repeated-heading (a heading straight under one with the same words, or with --wiki the")
    print("  first heading repeating the page title the host prints from the file name) and repeated-text (the same")
    print("  paragraph or list item twice in a row)")
    print("  also, over every line including code: hidden-char (strong: zero-width, soft hyphen, bidi, tag and")
    print("  variation-selector characters) and odd-space (weak: no-break, narrow no-break, figure space);")
    print("  --fix-hidden deletes the first and turns the second into plain spaces")
    print("Stock words (regex): " + ", ".join(STOCK_WORDS))


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Flag the mechanical signs of AI-written prose.")
    ap.add_argument("paths", nargs="*", help="files or directories (.md, .txt), or - for stdin")
    ap.add_argument("--max-words", type=int, default=30, help="flag sentences longer than this (default 30)")
    ap.add_argument("--allow-dashes", action="store_true", help="the author's own sample uses dashes")
    ap.add_argument("--wiki", action="store_true",
                    help="GitHub wiki pages: the host prints the file name as the page title, so a first heading "
                         "that repeats it is flagged")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--list", action="store_true", help="print the rules and the stock-word list")
    ap.add_argument("--fix-hidden", action="store_true",
                    help="rewrite the files in place without invisible characters and odd spaces, then check them")
    args = ap.parse_args(argv)
    if args.list:
        print_rules()
        return 0
    if not args.paths:
        ap.print_usage(sys.stderr)
        return 2
    try:
        files = gather(args.paths)
    except FileNotFoundError as exc:
        print("tells: no such file: %s" % exc, file=sys.stderr)
        return 2
    if args.fix_hidden:
        if "-" in files:
            print("tells: --fix-hidden rewrites files; it cannot take stdin", file=sys.stderr)
            return 2
        for name in files:
            data = Path(name).read_bytes()
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                print("tells: %s is not UTF-8; left unchanged" % name, file=sys.stderr)
                continue
            fixed, changes = fix_hidden(text)
            if changes:
                Path(name).write_bytes(fixed.encode("utf-8"))
                print("fixed %s: %d invisible character(s) or odd space(s)" % (name, changes))
    report, total_words = [], 0
    for name in files:
        title = None
        if args.wiki and name != "-" and not Path(name).name.startswith("_"):
            title = Path(name).stem.replace("-", " ")
        hits, words = check_text(read(name), args.max_words, args.allow_dashes, title)
        total_words += words
        for h in hits:
            h["file"] = "<stdin>" if name == "-" else name
            report.append(h)
    strong = sum(1 for h in report if h["severity"] == "strong")
    weak = len(report) - strong
    summary = {"strong": strong, "weak": weak, "words": total_words, "files": len(files)}
    if args.json:
        print(json.dumps({"hits": report, "summary": summary}, ensure_ascii=False, indent=1))
    else:
        for h in report:
            print("%s:%d: %s %s: %r (%s)" % (h["file"], h["line"], h["severity"], h["category"], h["match"], h["hint"]))
        print("tells: %d strong, %d weak, %d words, %d file(s)" % (strong, weak, total_words, len(files)))
    return 1 if strong else 0


if __name__ == "__main__":
    sys.exit(main())
