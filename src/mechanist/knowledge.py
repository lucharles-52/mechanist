"""Outside knowledge the investigator may consult — curated databases (offline, kb/) and the literature (Europe PMC).

Both only ever *propose*: a regulator or a paper is a lead that must then be tested in this study's data.

Data guard for the literature: a query may contain only gene symbols, cell-type and pathway words. It is refused if it
contains a sample or donor code, the study id, or anything that is not a plain word, and every query is logged.
Responses are cached by query so reruns are reproducible. Quotes are copied verbatim from the abstract; PMIDs come
from the API, never from the model.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path

import pandas as pd

KB_DIR = Path(__file__).resolve().parents[2] / "kb"
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
CTGOV = "https://clinicaltrials.gov/api/v2/studies"
STOP = {"and", "or", "the", "of", "in", "by", "a", "an", "to", "for", "with", "on", "is", "are", "cells", "cell"}


class KnowledgeError(ValueError):
    pass


class KnowledgeBase:
    """CollecTRI (TF -> target), PROGENy (pathway -> responsive genes), LIANA consensus ligand-receptor pairs."""

    def __init__(self, kb_dir: Path = KB_DIR):
        self.dir = kb_dir
        self._t: dict[str, pd.DataFrame] = {}

    def table(self, name: str) -> pd.DataFrame:
        return self._table(name)

    def _table(self, name: str) -> pd.DataFrame:
        if name not in self._t:
            path = self.dir / name
            if not path.exists():
                raise KnowledgeError(f"knowledge base file {path} missing; run scripts/build_kb.py")
            self._t[name] = pd.read_csv(path)
        return self._t[name]

    @property
    def available(self) -> bool:
        return all((self.dir / f).exists() for f in ("lr_consensus.csv", "collectri.csv", "progeny.csv"))

    @property
    def has_signalling(self) -> bool:
        return (self.dir / "omnipath_signaling.csv").exists() and (self.dir / "membrane_receptors.csv").exists()

    def receptor_paths(self, tf: str, max_steps: int = 3, limit: int = 12) -> tuple[str, dict]:
        """Signalling routes from cell-surface receptors to a transcription factor (OmniPath, directed and signed),
        with the ligands of each receptor — the link from 'which TF induces this gene' to 'which signal, from which
        cell'."""
        target = tf.upper()
        sig = self._table("omnipath_signaling.csv")
        lr = self._table("lr_consensus.csv")
        membrane = set(self._table("membrane_receptors.csv").gene.str.upper())
        rec_ligands: dict[str, set[str]] = {}
        for lig, rec in zip(lr.ligand, lr.receptor):
            for part in str(rec).upper().split("_"):
                if part in membrane:  # adaptors such as TRAF2 appear in some LR complexes; they are not receptors
                    rec_ligands.setdefault(part, set()).add(lig)
        into: dict[str, list[tuple[str, int, int]]] = {}
        for src, tgt, stim, inh, refs in zip(sig.source.str.upper(), sig.target.str.upper(), sig.stimulation,
                                             sig.inhibition, sig.n_refs):
            sign = 1 if stim and not inh else -1 if inh and not stim else 0
            into.setdefault(tgt, []).append((src, sign, int(refs)))
        if target not in into:
            raise KnowledgeError(f"{tf} has no upstream signalling interactions in OmniPath")

        def receptor_of(node: str) -> str | None:
            return next((p for p in node.split("_") if p in rec_ligands), None)

        paths, frontier = [], [([target], 1, 0)]
        for _ in range(max(1, min(int(max_steps), 4))):
            nxt = []
            for nodes, sign, refs in frontier:
                for src, s, r in into.get(nodes[0], []):
                    if src in nodes:
                        continue
                    item = ([src, *nodes], sign * s, refs + r)
                    if receptor_of(src) and len(nodes) > 0:
                        paths.append(item)
                    elif src in into:
                        nxt.append(item)
            frontier = sorted(nxt, key=lambda x: -x[2])[:400]
        if not paths:
            raise KnowledgeError(f"no receptor-to-{tf} route within {max_steps} steps in OmniPath")
        paths.sort(key=lambda x: (x[1] <= 0, len(x[0]), -x[2]))
        seen, rows, lines = set(), [], []
        for nodes, sign, refs in paths:
            rec = receptor_of(nodes[0])
            key = (rec, tuple(nodes[1:]))
            if key in seen:
                continue
            seen.add(key)
            kind = "activating" if sign > 0 else "inhibiting" if sign < 0 else "sign unknown"
            ligs = sorted(rec_ligands.get(rec, set()))
            rows.append({"receptor": rec, "path": nodes, "sign": kind, "references": refs, "ligands": ligs})
            lines.append(f"- {' → '.join(nodes)} ({kind}; {len(nodes) - 1} steps; {refs} refs) — ligands of {rec}: "
                         f"{', '.join(ligs[:8]) or '—'}")
            if len(rows) >= limit:
                break
        text = (f"Signalling routes from receptors to {tf} (OmniPath, curated; not this study). Activating, short, "
                f"well-referenced routes first:\n" + "\n".join(lines)
                + "\nNext: is the receptor on the cell that expresses the target gene, and which cells express its "
                  "ligands, in both disease samples?")
        return text, {"routes": rows}

    def regulators(self, gene: str) -> tuple[str, dict]:
        g = gene.upper()
        tf = self._table("collectri.csv")
        tf = tf[tf.target.str.upper() == g]
        pg = self._table("progeny.csv")
        pg = pg[pg.target.str.upper() == g].sort_values("weight", key=abs, ascending=False)
        up = sorted(tf[tf.weight > 0].source)
        down = sorted(tf[tf.weight < 0].source)
        lines = [f"Curated regulators of {gene} (databases, not this study):",
                 f"- transcription factors that ACTIVATE it (CollecTRI): {', '.join(up) or 'none listed'}",
                 f"- transcription factors that REPRESS it (CollecTRI): {', '.join(down) or 'none listed'}",
                 "- signalling pathways whose activity raises/lowers it (PROGENy weight): "
                 + (", ".join(f"{r.source} {r.weight:+.1f}" for r in pg.itertuples()) or "none listed")]
        lines.append("Next: which signals activate these TFs/pathways, which receptors carry them, and which cells "
                     "express the ligands — then test each in the data.")
        return "\n".join(lines), {"activating_tfs": up, "repressing_tfs": down,
                                  "pathways": pg[["source", "weight"]].to_dict("records")}

    def pairs(self, gene: str) -> tuple[str, dict]:
        g = gene.upper()
        lr = self._table("lr_consensus.csv")

        def has(col):
            return lr[col].str.upper().str.split("_").apply(lambda parts: g in parts)
        as_lig, as_rec = lr[has("ligand")], lr[has("receptor")]
        if as_lig.empty and as_rec.empty:
            raise KnowledgeError(f"{gene} is in no ligand-receptor pair of the LIANA consensus resource")
        lines = [f"Curated ligand-receptor pairs for {gene} (LIANA consensus resource, not this study; '_' = complex):"]
        if len(as_lig):
            lines.append(f"- as ligand, binds: {', '.join(sorted(set(as_lig.receptor)))}")
        if len(as_rec):
            lines.append(f"- as receptor (or subunit), bound by: {', '.join(sorted(set(as_rec.ligand)))}")
        return "\n".join(lines), {"receptors": sorted(set(as_lig.receptor)), "ligands": sorted(set(as_rec.ligand))}


def _sentences(text: str) -> list[str]:
    text = re.sub(r"<[^>]+>", "", text or "")
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text) if len(s.strip()) > 20]


OT_GRAPHQL = "https://api.platform.opentargets.org/api/v4/graphql"
OT_SEARCH = """query($s:String!){ search(queryString:$s, entityNames:["target"], page:{index:0,size:5}){ hits { id name } } }"""
OT_TARGET = """query($id:String!,$n:Int!){ target(ensemblId:$id){ approvedSymbol
  associatedDiseases(page:{index:0,size:$n}){ count rows { score disease { name therapeuticAreas { name } }
  datatypeScores { id score } } }
  drugAndClinicalCandidates { count rows { maxClinicalStage drug { name drugType }
  diseases { diseaseFromSource disease { name } } } } } }"""


OT_DRUGS = """query($id:String!){ target(ensemblId:$id){ approvedSymbol drugAndClinicalCandidates { count rows {
  maxClinicalStage drug { id name drugType maximumClinicalStage mechanismsOfAction { rows { mechanismOfAction actionType
  targets { approvedSymbol } } } drugWarnings { warningType toxicityClass } } diseases { diseaseFromSource
  disease { name } } } } } }"""
STAGE_ORDER = {"APPROVAL": 5, "PHASE_4": 5, "WITHDRAWAL": 0.5, "PHASE_3": 4, "PHASE_2_3": 3.5, "PHASE_2": 3, "PHASE_1_2": 2.5, "PHASE_1": 2,
               "EARLY_PHASE_1": 1.5, "PRECLINICAL": 1}


class OpenTargets:
    """Open Targets Platform: diseases associated with a gene (by evidence type) and drugs acting on it. Only the gene
    symbol leaves the cluster, after the same identifier check as the literature search; every query is logged."""

    def __init__(self, cache_dir: Path, guard: Literature, log_path: Path | None = None,
                 fetch: Callable[[str, dict], dict] | None = None):
        self.cache = cache_dir
        self.cache.mkdir(parents=True, exist_ok=True)
        self.guard = guard
        self.log = log_path
        self._fetch = fetch or self._http

    @staticmethod
    def _http(query: str, variables: dict) -> dict:
        req = urllib.request.Request(OT_GRAPHQL, data=json.dumps({"query": query, "variables": variables}).encode(),
                                     headers={"content-type": "application/json"})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=40) as r:
                    return json.load(r)
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
        return {}

    def _cached(self, query: str, variables: dict) -> dict:
        key = hashlib.sha256(json.dumps([query, variables], sort_keys=True).encode()).hexdigest()
        hit = self.cache / f"ot_{key}.json"
        cached = hit.exists()
        body = json.loads(hit.read_text()) if cached else self._fetch(query, variables)
        if not cached:
            hit.write_text(json.dumps(body))
        if self.log is not None:
            with self.log.open("a") as fh:
                fh.write(json.dumps({"opentargets": variables, "cached": cached, "time": time.time()}) + "\n")
        return body

    def gene(self, gene: str, n: int = 15) -> tuple[str, dict]:
        g = self.guard.check_query(gene)
        if len(g.split()) != 1:
            raise KnowledgeError("give one gene symbol")
        hits = ((self._cached(OT_SEARCH, {"s": g}).get("data") or {}).get("search") or {}).get("hits") or []
        hit = next((h for h in hits if str(h.get("name", "")).upper() == g.upper()), None)
        if hit is None:
            raise KnowledgeError(f"Open Targets has no target named {g}")
        t = ((self._cached(OT_TARGET, {"id": hit["id"], "n": max(5, min(int(n), 25))}).get("data") or {})
             .get("target") or {})
        rows = (t.get("associatedDiseases") or {}).get("rows") or []
        diseases = [{"disease": r["disease"]["name"], "score": round(r["score"], 3),
                     "areas": [a["name"] for a in r["disease"].get("therapeuticAreas") or []][:2],
                     "evidence": {x["id"]: round(x["score"], 2) for x in r.get("datatypeScores") or []}} for r in rows]
        drugs = []
        for r in ((t.get("drugAndClinicalCandidates") or {}).get("rows") or [])[:15]:
            ds = [((x.get("disease") or {}).get("name") or x.get("diseaseFromSource")) for x in r.get("diseases") or []]
            drugs.append({"drug": (r.get("drug") or {}).get("name"), "stage": r.get("maxClinicalStage"),
                          "type": (r.get("drug") or {}).get("drugType"), "diseases": [d for d in ds if d][:4]})
        lines = [(f"Open Targets, {g} ({hit['id']}): {(t.get('associatedDiseases') or {}).get('count', 0)} associated "
                 f"diseases; top {len(diseases)} by overall score (evidence types: genetic_association, literature, "
                 f"rna_expression, animal_model, affected_pathway, known_drug, somatic_mutation):")]
        lines += [f"- {d['disease']} {d['score']} [" + ", ".join(f"{k} {v}" for k, v in d["evidence"].items()) + "]"
                  for d in diseases]
        lines.append(f"Drugs acting on {g}: " + ("; ".join(f"{d['drug']} ({d['stage']}; {', '.join(d['diseases'][:3])})"
                                                        for d in drugs) if drugs else "none in clinical development"))
        return "\n".join(lines), {"target": hit["id"], "diseases": diseases, "drugs": drugs}


    def drugs(self, gene: str, n: int = 25) -> tuple[str, dict]:
        """Drugs and clinical candidates acting on one gene: highest clinical stage, action type and mechanism (from
        the drug's own mechanisms of action, so an agonist is never mistaken for an inhibitor) and indications."""
        g = self.guard.check_query(gene)
        if len(g.split()) != 1:
            raise KnowledgeError("give one gene symbol")
        hits = ((self._cached(OT_SEARCH, {"s": g}).get("data") or {}).get("search") or {}).get("hits") or []
        hit = next((h for h in hits if str(h.get("name", "")).upper() == g.upper()), None)
        alias = None
        if hit is None and hits:  # a common name (e.g. a ligand's trivial name): Open Targets' top hit, said so
            hit, alias = hits[0], str(hits[0].get("name", ""))
        if hit is None:
            raise KnowledgeError(f"Open Targets has no target named {g}")
        t = ((self._cached(OT_DRUGS, {"id": hit["id"]}).get("data") or {}).get("target") or {})
        rows = ((t.get("drugAndClinicalCandidates") or {}).get("rows") or [])
        drugs: dict[str, dict] = {}
        for r in rows:
            d = r.get("drug") or {}
            name = str(d.get("name") or "").strip()
            if not name:
                continue
            acts = [{"action": str(m.get("actionType") or "").lower(), "mechanism": str(m.get("mechanismOfAction") or ""),
                     "targets": [x.get("approvedSymbol") for x in m.get("targets") or [] if x.get("approvedSymbol")]}
                    for m in ((d.get("mechanismsOfAction") or {}).get("rows") or [])]
            acts = [a for a in acts if g.upper() in {x.upper() for x in a["targets"]}] or acts
            ind = [((x.get("disease") or {}).get("name") or x.get("diseaseFromSource")) for x in r.get("diseases") or []]
            e = drugs.setdefault(name, {"drug": name, "id": d.get("id"), "type": d.get("drugType"), "stage": "",
                                        "actions": [], "indications": [], "warnings": []})
            for w in d.get("drugWarnings") or []:
                x = {"type": str(w.get("warningType") or ""), "toxicity": str(w.get("toxicityClass") or "")}
                if x["type"] and x not in e["warnings"]:
                    e["warnings"].append(x)
            for st in (str(d.get("maximumClinicalStage") or ""), str(r.get("maxClinicalStage") or "")):
                if STAGE_ORDER.get(st, 0) > STAGE_ORDER.get(e["stage"], 0):  # the drug's own record first
                    e["stage"] = st
            if str(r.get("maxClinicalStage")) == "WITHDRAWAL":
                w = {"type": "Withdrawn for at least one indication or market", "toxicity": ""}
                if w not in e["warnings"]:
                    e["warnings"].append(w)
            e["actions"] += [a for a in acts if a not in e["actions"]]
            e["indications"] += [x for x in ind if x and x.lower() not in {y.lower() for y in e["indications"]}]
        out = sorted(drugs.values(), key=lambda e: -STAGE_ORDER.get(e["stage"], 0))[:n]
        lines = [f"Open Targets drugs acting on {g}" + (f" (matched as {alias})" if alias else "") + f" ({hit['id']}): "
                 f"{len(drugs)} drugs or candidates"
                 + (f"; top {len(out)} by clinical stage:" if out else ".")]
        lines += [f"- {e['drug']} ({e['stage'] or 'stage unknown'}; "
                  + ", ".join(sorted({a['action'] for a in e['actions'] if a['action']}) or {"action unknown"})
                  + f"; {'; '.join(a['mechanism'] for a in e['actions'][:2])}"
                  + "".join(f"; {w['type']}" + (f" ({w['toxicity']})" if w["toxicity"] else "") for w in e["warnings"][:2])
                  + ") for "
                  + (", ".join(e["indications"][:4]) or "no indication listed") for e in out]
        return "\n".join(lines), {"target": hit["id"], "gene": g, "symbol": alias or g, "drugs": out}


class Literature:
    """Europe PMC search returning PMIDs, titles and verbatim abstract sentences that match the query."""

    def __init__(self, cache_dir: Path, forbidden: set[str], log_path: Path | None = None,
                 fetch: Callable[[str], dict] | None = None):
        self.cache = cache_dir
        self.cache.mkdir(parents=True, exist_ok=True)
        self.forbidden = {f.lower() for f in forbidden if f}
        self.log = log_path
        self._fetch = fetch or self._http

    def check_query(self, query: str) -> str:
        q = " ".join(str(query).split())
        if not q or len(q) > 200 or len(q.split()) > 16:
            raise KnowledgeError("query must be 1-16 plain words")
        if not re.fullmatch(r"[A-Za-z0-9 \-+/αβγκ]+", q):
            raise KnowledgeError("query may contain only gene symbols, cell-type and pathway words")
        hit = [w for w in re.split(r"[\s/]+", q.lower()) if w in self.forbidden]
        if hit:
            raise KnowledgeError(f"query refused: contains study identifiers ({', '.join(hit)})")
        return q

    @staticmethod
    def _http(url: str) -> dict:
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=30) as r:
                    return json.load(r)
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
        return {}

    def search(self, query: str, n: int = 5) -> tuple[str, dict]:
        q = self.check_query(query)
        n = max(1, min(int(n), 12))
        params = {"query": f"({q}) AND SRC:MED AND HAS_ABSTRACT:y", "format": "json", "resultType": "core",
                  "pageSize": str(n)}
        url = EPMC + "?" + urllib.parse.urlencode(params)
        key = hashlib.sha256(url.encode()).hexdigest()
        hit = self.cache / f"{key}.json"
        if hit.exists():
            body = json.loads(hit.read_text())
        else:
            body = self._fetch(url)
            hit.write_text(json.dumps(body))
        if self.log is not None:
            with self.log.open("a") as fh:
                fh.write(json.dumps({"query": q, "url": url, "cached": hit.exists(), "time": time.time()}) + "\n")
        terms = [t.lower() for t in re.split(r"[\s/]+", q) if t.lower() not in STOP and len(t) > 1]
        papers, lines = [], []
        for r in (body.get("resultList") or {}).get("result", [])[:n]:
            pmid = str(r.get("pmid") or "")
            if not pmid:
                continue
            sents = _sentences(r.get("abstractText", ""))
            scored = sorted(sents, key=lambda s: -sum(t in s.lower() for t in terms))
            quotes = [s for s in scored[:2] if any(t in s.lower() for t in terms)]
            paper = {"pmid": pmid, "year": r.get("pubYear"), "title": re.sub(r"<[^>]+>", "", r.get("title", "")),
                     "journal": (r.get("journalInfo") or {}).get("journal", {}).get("title", ""), "quotes": quotes}
            papers.append(paper)
            lines.append(f"PMID {pmid} ({paper['year']}) {paper['title']}"
                         + "".join(f'\n  "{s}"' for s in quotes))
        text = (f"Europe PMC, query '{q}' — {len(papers)} papers (other studies, not this one). Quotes are verbatim "
                f"from abstracts:\n" + ("\n".join(lines) or "no matching papers"))
        return text, {"query": q, "papers": papers}


class Trials:
    """ClinicalTrials.gov registry search by condition. Uses the literature tool's query rules, cache and log."""

    FIELDS = "NCTId,BriefTitle,Condition,Phase,OverallStatus,InterventionName,StartDate,HasResults"

    def __init__(self, lit: Literature, fetch: Callable[[str], dict] | None = None):
        self.lit = lit
        self._fetch = fetch or Literature._http

    def search(self, condition: str, n: int = 10) -> tuple[str, dict]:
        q = self.lit.check_query(condition)
        n = max(1, min(int(n), 20))
        url = CTGOV + "?" + urllib.parse.urlencode({"query.cond": q, "pageSize": str(min(3 * n, 60)),
                                                    "fields": self.FIELDS, "sort": "StartDate:desc"})
        key = hashlib.sha256(url.encode()).hexdigest()
        hit = self.lit.cache / f"{key}.json"
        if hit.exists():
            body = json.loads(hit.read_text())
        else:
            body = self._fetch(url)
            hit.write_text(json.dumps(body))
        if self.lit.log is not None:
            with self.lit.log.open("a") as fh:
                fh.write(json.dumps({"query": q, "url": url, "cached": hit.exists(), "time": time.time()}) + "\n")
        # the registry matches loosely (synonyms, related terms): keep trials that name every query word
        words = [w for w in re.split(r"[\s/]+", q.lower()) if w not in STOP and len(w) > 1]
        trials, lines = [], []
        for st in body.get("studies") or []:
            ps = st.get("protocolSection") or {}
            conds = (ps.get("conditionsModule") or {}).get("conditions") or []
            title = (ps.get("identificationModule") or {}).get("briefTitle", "")
            hay = " ".join([title, *conds]).lower()
            if not all(w in hay for w in words):
                continue
            t = {"nct": (ps.get("identificationModule") or {}).get("nctId", ""), "title": title,
                 "conditions": conds[:4], "phase": "/".join((ps.get("designModule") or {}).get("phases") or []) or "n/a",
                 "status": (ps.get("statusModule") or {}).get("overallStatus", ""),
                 "start": ((ps.get("statusModule") or {}).get("startDateStruct") or {}).get("date", ""),
                 "interventions": [i.get("name", "") for i in (ps.get("armsInterventionsModule") or {})
                                   .get("interventions") or []][:4],
                 "has_results": bool(st.get("hasResults"))}
            trials.append(t)
            lines.append(f"{t['nct']} ({t['phase']}, {t['status'].lower().replace('_', ' ')}, started {t['start'] or '?'}"
                         f"{', results posted' if t['has_results'] else ''}) {t['title']}: "
                         f"{', '.join(t['interventions']) or 'no intervention listed'}")
            if len(trials) >= n:
                break
        text = (f"ClinicalTrials.gov, condition '{q}': {len(trials)} trials, newest first (registered plans and "
                f"status, not outcomes):\n" + ("\n".join(lines) or "no registered trials name this condition"))
        return text, {"query": q, "trials": trials}


PMID_RE = re.compile(r"PMID[:\s]*?(\d{5,9})", re.IGNORECASE)


def cited_pmids(text: str) -> list[str]:
    return PMID_RE.findall(text or "")
