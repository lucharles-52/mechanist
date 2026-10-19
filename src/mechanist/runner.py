"""Orchestrator: a fixed state machine with fingerprinted, resumable stages.

multi:  load -> findings -> generate -> verify_r0 -> investigate -> synthesize -> critique -> chase -> revise -> report
single: load -> findings -> single   -> verify_r0 -> investigate -> synthesize -> report

The investigation is a bounded loop: the investigator asks typed questions with tool calls, code runs the tools, the
investigator answers and asks follow-ups. Code keeps a list of questions that may not be skipped (investigate.py).
In multi mode an independent critic reviews the integrated model and poses new questions, which are chased with the
same tools before the model is revised. Single mode has the same tools, rounds and required questions but no critic.

Each stage writes <stage>.json and records a fingerprint of (stage, code/prompt version, config, upstream
fingerprints) in state.json. A rerun skips stages whose fingerprint is unchanged; LLM calls are additionally cached
by request hash, so an interrupted stage resumes without repeating finished calls.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal, get_args

import yaml
from pydantic import BaseModel

from mechanist import agents, cellfig, report
from mechanist import findings as fnd
from mechanist import investigate as inv
from mechanist.adapters import p033, skinpipe
from mechanist.adapters.base import LoadedStudy, group_order, lineages_from_cells
from mechanist.findings import digest_line
from mechanist.knowledge import KB_DIR, KnowledgeBase, Literature, OpenTargets, cited_pmids
from mechanist.llm import LLMConfig, LoggedLLM, make_backend
from mechanist.records import (
    Candidate,
    Circuit,
    Dataset,
    Finding,
    Hypothesis,
    LitAssessment,
    Objection,
    Persona,
    Question,
    QuestionStatus,
    Synthesis,
    ToolCall,
    ToolResult,
)
from mechanist.tools import CellData, Toolbox
from mechanist.verify import verifier_objections, verify

CODE_VERSION = "0.32.0"
# (projects_root, study, lab_tables=, primary=, **adapter_options)
ADAPTERS: dict[str, Callable[..., LoadedStudy]] = {"p033": p033.load, "skinpipe": skinpipe.load}
STAGES = {"multi": ["load", "findings", "panel", "generate", "verify_r0", "scout", "investigate", "synthesize", "critique",
                    "chase", "revise", "report"],
          "single": ["load", "findings", "single", "verify_r0", "scout", "investigate", "synthesize", "report"],
          # the team workflow: dossier -> do we see it -> candidates -> chase -> experiments -> model, a PI gate after each
          "staged": ["load", "findings", "dossier", "check", "panel", "generate", "candidates", "verify_r0", "scout",
                     "investigate", "synthesize", "critique", "chase", "linkage", "revise", "circuits", "experiment",
                     "therapy", "report"]}
LIT_TOOLS = {"literature_search", "gene_diseases", "find_findings", "known_regulators", "known_pairs", "receptor_paths"}
# fields added after runs began: left out of stage fingerprints while at these defaults, so older runs keep their cache
FP_DEFAULTS = {"study_question": False, "therapy_steps": 6, "therapy_targets": 30, "experiment_papers": 120}
DRUG_TOOLS = {"literature_search", "gene_diseases", "gene_drugs", "find_findings"}  # experiments and therapy
# the hypothesizer checks its speculation with these (published analogues, a needed receptor or co-occurrence)
CIRCUIT_TOOLS = {"literature_search", "gene_profile", "coexpression", "lineage_changes", "gene_by_state", "find_findings"}
SCOUT_TOOLS = {"literature_search", "known_regulators", "known_pairs", "receptor_paths"}


# The artifact sceptic sits on every panel, derived or configured: no panel goes without the non-biological view.
SCEPTIC = Persona(title="Computational biologist", expertise="single-cell data artifacts, batch effects, ambient RNA "
                  "and doublets, composition effects and small-sample statistics",
                  role="propose the non-biological explanations a careful analyst would rule out first", artifacts=True)
# Lineage name keywords -> panel field. Anything unmatched is tissue (structural/parenchymal) biology.
ADAPTIVE = ("b cell", "plasma", "t cell", "t/nk", "cd4", "cd8", "treg", "lymphocyte", "nkt", "b-cell", "t-cell")
INNATE = ("myeloid", "macrophage", "monocyte", "dendritic", "langerhans", "neutrophil", "granulocyte", "mast",
          "eosinophil", "basophil", "nk", "ilc", "innate", "mono/mac", "pdc", "cdc")
MAX_NAMED = 8  # lineages named per persona


def _has_word(text: str, keys) -> bool:
    """Keyword as a whole word ("t cell" must not match "mast cell")."""
    return any(re.search(rf"(^|[^a-z0-9]){re.escape(k)}($|[^a-z0-9])", text) for k in keys)


def derive_panel(cell_types: list[str], disease: str, others: list[str], reference: str) -> list[Persona]:
    """Specialists for THIS study: one per field that has lineages in the data (named in the persona), a clinician
    for the disease and its comparators, and the artifact sceptic. Lineages, not a fixed list, decide who sits."""
    fields: dict[str, list[str]] = {"tissue": [], "adaptive": [], "innate": []}
    for ct in sorted(set(cell_types)):
        low = ct.lower()
        if _has_word(low, ADAPTIVE):
            fields["adaptive"].append(ct)
        elif _has_word(low, INNATE):
            fields["innate"].append(ct)
        else:
            fields["tissue"].append(ct)

    def named(lins):
        return ", ".join(lins[:MAX_NAMED]) + (f" and {len(lins) - MAX_NAMED} more" if len(lins) > MAX_NAMED else "")
    panel = []
    if fields["tissue"]:
        panel.append(Persona(title="Tissue biologist", expertise=f"the structural and parenchymal cells of this "
                             f"tissue ({named(fields['tissue'])}): their states, the signals they make and receive, "
                             f"barrier function, vasculature and tissue remodelling"))
    if fields["adaptive"]:
        panel.append(Persona(title="Adaptive immunologist", expertise=f"lymphocytes ({named(fields['adaptive'])}): "
                             f"activation, antigen presentation, lymphoid organisation, cytokine and antibody "
                             f"responses"))
    if fields["innate"]:
        panel.append(Persona(title="Innate immunologist", expertise=f"innate immune cells "
                             f"({named(fields['innate'])}): recruitment, inflammatory cytokines and chemokines, "
                             f"effector programs and tissue damage"))
    comp = ", ".join([*others, reference]) or "the comparison groups"
    panel.append(Persona(title="Clinical pathologist", expertise=f"the clinical and histological phenotype of {disease} "
                         f"compared with {comp}, its known pathogenesis, and the diseases it is mistaken for"))
    return panel + [SCEPTIC]


def study_panel(cfg_panel: list[Persona] | None, cell_types: list[str], disease: str, groups: list[str],
                reference: str) -> list[Persona]:
    """The configured panel (with the sceptic added if it lacks one), or one derived from the study. [] = no panel."""
    if cfg_panel is None:
        return derive_panel(cell_types, disease, [g for g in groups if g not in (disease, reference)], reference)
    if cfg_panel and not any(p.artifacts for p in cfg_panel):
        return [*cfg_panel, SCEPTIC]
    return list(cfg_panel)


class RunConfig(BaseModel):
    study: str  # pNNN
    adapter: str = "skinpipe"  # skinpipe (any current reviewed study) | p033 (legacy)
    projects_root: str = "/vast/projects/thl/understanding-mole/Projects"
    disease: str  # the primary disease group, named as in the study's groups
    # Adapter settings, e.g. skinpipe: compartment, reference, groups ({raw analysis_group: short name}), comparison,
    # route. See the adapter's load().
    adapter_options: dict = {}
    disease_phenotype: str = ""
    mode: Literal["multi", "single"] = "multi"
    n_hypotheses: int = 3
    # threads: each question is one investigator's tool loop (sees every result before the next call); a planner
    # picks a few deep threads per round. rounds: the older batch protocol (all questions answered in one call).
    investigation: Literal["threads", "rounds"] = "threads"
    # staged: the team workflow with a PI gate after every stage (multi mode); classic: the earlier stage list
    workflow: Literal["staged", "classic"] = "classic"
    dossier_steps: int = 12  # literature calls per dossier round
    check_threads: int = 5  # stage 2 threads
    experiment_steps: int = 14  # literature calls per experimentalist round
    experiment_papers: int = 120  # papers already retrieved, shown to the experimentalist (perturbations first)
    therapy_steps: int = 6  # the pharmacologist's own checks per round (drug lookups and searches beyond the targets')
    therapy_targets: int = 30  # circuit genes whose drugs are looked up by code before the pharmacologist writes
    linkage_steps: int = 10  # literature / Open Targets calls per link in the disease-linkage stage
    circuit_steps: int = 8  # the hypothesizer's checks per round (literature, a receptor, a co-occurrence)
    # 2026-10-18: the circuits stage is back to v29 by default (the sweep handed the hypothesizer framings to chase,
    # and the baseline tier doubled the difference account with noise); both stay available as switches
    circuit_sweep: int = 0  # targeted literature queries after each first draft, aimed at its open steps (off by default)
    circuit_attempts: int = 3  # independent circuit attempts, merged by a consensus pass (1: a single attempt)
    circuit_review_cap: int = 6  # PI reviews of the circuits at most (progress counts only established steps)
    difference_baseline: bool = False  # the difference account's PERTURBED FROM <healthy> BUT NOT SPECIFIC tier
    linkage_links: int = 8  # links of the model placed in the wider disease context
    no_progress_limit: int = 3  # PI reviews in a row with no new test and no new testable prediction
    gate_cap: int = 40  # reviews per gate: a guard against a broken loop, not a design limit
    # chase (staged): what counts as progress for the stop rule. "model": the integrated model gained tested steps or
    # depth over its best so far (a PI that always finds more to ask cannot keep the loop alive); "pi": the PI's own
    # counts of new tests and predictions. chase_max_rounds: 0 = no cap.
    # what counts as progress in a chase round: "steps" (a causal step newly settled on data or literature, or a new
    # step opened), "model" (the interim model gained tested steps or depth), "pi" (the PI named new tests)
    chase_progress: Literal["steps", "model", "pi"] = "steps"
    chase_max_rounds: int = 0
    # prompt budgets (characters) for the investigation log and the cited results shown to the synthesizer, the PI
    # and the critic, so a long investigation never overflows the model's context (the synthesis then fails)
    log_budget: int = 120_000
    results_budget: int = 90_000
    thread_rounds: int = 3
    threads_per_round: int = 4
    thread_steps: int = 8  # tool calls per thread
    thread_workers: int = 4  # threads run in parallel against the same model server
    thread_result_cap: int = 3500  # characters of each tool result shown to the thread
    chase_thread_rounds: int = 1  # after the critique: its questions as threads, then planner rounds
    pi_rounds: int = 1  # PI review rounds: critique -> chase its demands -> re-synthesise -> next review
    investigate_rounds: int = 8  # rounds protocol: the last round only answers
    missing_shown: int = 12  # required topics shown per round, key mediators first
    # Hand-made tables from an earlier investigation of the same hypotheses (circular): its curated gene panel and
    # signature findings, ambient gauge and the fibroblast_programs tool. Off: per-sample findings are genome-wide.
    lab_tables: bool = False
    questions_per_round: int = 8
    calls_per_question: int = 3
    critic_questions: int = 6
    chase_rounds: int = 2
    max_revisions: int = 1  # 0 skips critique/chase/revise
    investigate_digest: int = 60  # findings shown to the investigator (it can search the rest)
    knowledge: bool = True  # curated databases in kb/ (offline)
    literature: bool = True  # Europe PMC search; only gene/cell/pathway words leave the cluster
    kb_dir: str | None = None
    scout: bool = True  # check hypotheses against prior work and add literature-inspired hypotheses
    # Multi mode: specialists propose hypotheses in parallel, the PI merges. None: derived from the study's lineages,
    # disease and groups (derive_panel); a list: these specialists (the artifact sceptic is added if missing);
    # []: no panel (a single generator proposes, still with a non-mechanistic alternative).
    panel: list[Persona] | None = None
    panel_hypotheses_each: int = 2
    panel_temperature: float = 0.8  # creative, independent proposals (Virtual Lab: parallel meetings)
    merge_temperature: float = 0.2  # consistent merge by the PI
    guidance: str = ""  # high-level direction from the human researcher, shown to every agent
    # cohort: the disease group is a disease. index_patient: it is one patient (or a few) whose diagnosis is uncertain,
    # the comparators are the closest known diseases, and the aim is that patient's own mechanism (agents.INDEX_PATIENT)
    design: Literal["cohort", "index_patient"] = "cohort"
    # give every stage the submitter's question (metadata/submission.yaml). Off by default: a blinded run's submission
    # can name the answer. While off it is left out of the stage fingerprints, so runs that predate it keep their cache.
    study_question: bool = False
    scout_searches_per_hypothesis: int = 3
    scout_searches_total: int = 10
    scout_new_hypotheses: int = 2
    findings: fnd.FindingsConfig = fnd.FindingsConfig()
    llm: LLMConfig = LLMConfig()

    @classmethod
    def from_yaml(cls, path: Path) -> RunConfig:
        return cls.model_validate(yaml.safe_load(path.read_text()) or {})


def _dump(obj):
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if isinstance(obj, list):
        return [_dump(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _dump(v) for k, v in obj.items()}
    return obj


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


class Runner:
    def __init__(self, cfg: RunConfig, run_dir: Path, llm_backend=None, cells: CellData | None = None,
                 lit_fetch=None, ot_fetch=None):
        self.cfg = cfg
        self.dir = run_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        self.fps: dict[str, str] = {}
        self._study: LoadedStudy | None = None
        self._backend = llm_backend
        self._llm: LoggedLLM | None = None
        self._cells = cells
        self._toolbox: Toolbox | None = None
        self._lit_fetch = lit_fetch
        self._ot_fetch = ot_fetch
        self._deferred: list[dict] = []  # asks for an exhausted step that named no new kind of test
        (self.dir / "config.yaml").write_text(yaml.safe_dump(cfg.model_dump(), sort_keys=False))

    # ---- plumbing -------------------------------------------------------------------------------------------
    def event(self, **kw):
        kw["time"] = dt.datetime.now().isoformat(timespec="seconds")
        with (self.dir / "events.jsonl").open("a") as fh:
            fh.write(json.dumps(kw) + "\n")

    @property
    def llm(self) -> LoggedLLM:
        if self._llm is None:
            self._llm = LoggedLLM(self._backend or make_backend(self.cfg.llm), self.dir)
        return self._llm

    def study(self) -> LoadedStudy:
        if self._study is None:
            if self.cfg.adapter not in ADAPTERS:
                raise ValueError(f"unknown adapter {self.cfg.adapter!r}; adapters: {', '.join(ADAPTERS)}")
            s = ADAPTERS[self.cfg.adapter](Path(self.cfg.projects_root), self.cfg.study, lab_tables=self.cfg.lab_tables,
                                           primary=self.cfg.disease, **self.cfg.adapter_options)
            ds = s.dataset
            if self.cfg.disease not in ds.groups or self.cfg.disease == ds.reference_group:
                raise ValueError(f"{ds.study}: disease {self.cfg.disease!r} must be a non-reference group; groups: "
                                 f"{', '.join(ds.groups)} (reference {ds.reference_group})")
            self._study = s
        return self._study

    def cells(self) -> CellData | None:
        """The reviewed single-cell object, loaded once and shared by the findings scan and the tools."""
        if self._cells is None and self.study().cells is not None:
            self._cells = CellData(self.study().cells)
        return self._cells

    def toolbox(self, fs: list[Finding]) -> Toolbox:
        if self._toolbox is None:
            s = self.study()
            kb_dir = Path(self.cfg.kb_dir) if self.cfg.kb_dir else KB_DIR
            kb = KnowledgeBase(kb_dir) if self.cfg.knowledge else None
            if kb is not None and not kb.available:
                self.event(stage="toolbox", action="knowledge base missing", path=str(kb_dir))
                kb = None
            lit = None
            if self.cfg.literature:
                forbidden = {self.cfg.study, *(x for ss in s.dataset.groups.values() for x in ss)}
                lit = Literature(kb_dir / "lit_cache", forbidden, self.dir / "external_queries.jsonl",
                                 fetch=self._lit_fetch)
            ot = (OpenTargets(kb_dir / "lit_cache", lit, self.dir / "external_queries.jsonl", fetch=self._ot_fetch)
                  if lit is not None else None)
            self._toolbox = Toolbox(s, fs, cells=self._cells, kb=kb, literature=lit, lab_tables=self.cfg.lab_tables,
                                    opentargets=ot,
                                    primary=self.cfg.disease,
                                    expression_loader=lambda: fnd.load_expression(s, self.cfg.findings, cells=self.cells))
            self._toolbox.on_result = self.llm.transcript.tool
        return self._toolbox

    def stage(self, name: str, deps: list[str], fn: Callable[[], object], always: bool = False):
        cfg = self.cfg.model_dump(exclude={"llm": {"timeout"}, **{k: True for k, v in FP_DEFAULTS.items()
                                                                     if getattr(self.cfg, k) == v}})
        fp = _sha({"stage": name, "code": CODE_VERSION, "prompts": agents.PROMPT_VERSION, "config": cfg,
                   "deps": [self.fps[d] for d in deps]})
        out = self.dir / f"{name}.json"
        prev = self.state.get(name, {})
        if not always and prev.get("fingerprint") == fp and prev.get("status") == "complete" and out.exists():
            self.fps[name] = prev["output_sha256"]
            self.event(stage=name, action="skip", fingerprint=fp[:12])
            return json.loads(out.read_text())
        self.event(stage=name, action="start", fingerprint=fp[:12])
        data = _dump(fn())
        text = json.dumps(data, indent=1, default=str)
        tmp = out.with_suffix(".tmp")
        tmp.write_text(text)
        os.replace(tmp, out)
        self.fps[name] = _sha(text)
        self.state[name] = {"fingerprint": fp, "status": "complete", "output_sha256": self.fps[name],
                            "finished": dt.datetime.now().isoformat(timespec="seconds")}
        self.state_path.write_text(json.dumps(self.state, indent=1))
        self.event(stage=name, action="end", output=out.name)
        return data

    def run(self, stop_after: str | None = None) -> Path:
        stages = STAGES["staged" if self.cfg.mode == "multi" and self.cfg.workflow == "staged" else self.cfg.mode]
        if self.cfg.max_revisions == 0:
            stages = [s for s in stages if s not in ("critique", "chase", "revise")]
        ctx: dict = {"stages": stages}
        for name in stages:
            ctx[name] = getattr(self, f"_s_{name}")(ctx)
            if name == stop_after:
                break
        return self.dir

    # ---- helpers --------------------------------------------------------------------------------------------
    def _findings(self, ctx, n: int | None = None) -> tuple[Dataset, list[Finding], str]:
        ds = Dataset.model_validate(ctx["load"])
        fs = [Finding.model_validate(f) for f in ctx["findings"]["findings"]]
        ids = ctx["findings"]["digest_ids"]
        shown = [f for f in fs if f.id in set(ids)]
        if n is not None:
            shown = sorted(shown, key=lambda f: -f.priority)[:n]
            shown.sort(key=lambda f: f.id)
        context = agents.build_context(ds, shown, ctx["findings"]["chance_separations"])
        if ctx.get("dossier"):
            context += "\n\nPathology dossier (stage 1, from the literature):\n" + agents.dossier_text(
                ctx["dossier"].get("dossier") or {})
        if ctx.get("check"):
            context += "\n\nDo we see the pathology? (stage 2):\n" + agents.check_text(ctx["check"].get("summary") or {})
        if self.cfg.design == "index_patient":
            context += "\n\n" + agents.INDEX_PATIENT.format(patient=self.cfg.disease)
        if self.cfg.guidance.strip():
            context += f"\n\nGuidance from the human researcher: {self.cfg.guidance.strip()}"
        return ds, fs, context

    def _base_hyps(self, ctx) -> list[Hypothesis]:
        if ctx.get("candidates"):  # stage 3 after the PI's gate
            return [Hypothesis.model_validate(h) for h in ctx["candidates"]["hypotheses"]]
        key = "single" if self.cfg.mode == "single" else "generate"
        return [Hypothesis.model_validate(h) for h in ctx.get(key) or []]

    def _hyps(self, ctx) -> list[Hypothesis]:
        """Starting hypotheses plus any the literature scout proposed."""
        extra = [Hypothesis.model_validate(h) for h in (ctx.get("scout") or {}).get("new_hypotheses", [])]
        return self._base_hyps(ctx) + extra

    def _prior_work_text(self, ctx) -> str:
        sc = ctx.get("scout") or {}
        if not sc.get("assessments") and not sc.get("new_hypotheses"):
            return ""
        lines = ["", "", "Prior work found by the literature scout (other studies; results can be cited by id):"]
        for a in sc.get("assessments", []):
            lines.append(f"- {a['hypothesis_id']} {a['step_id']}: {a['support']}{'' if a['verified'] else ' (unverified)'}"
                         f" [{', '.join(a['result_ids'])}; PMID {', '.join(a['pmids']) or '-'}] {a['note'][:200]}")
        for h in sc.get("new_hypotheses", []):
            lines.append(f"- new lead {h['id']} from the literature: {h['title']} [{', '.join(h['inspired_by'])}]")
        return "\n".join(lines)

    def _s_scout(self, ctx):
        _ds, fs, _ = self._findings(ctx)
        hyps = self._base_hyps(ctx)

        def fn():
            empty = {"results": [], "assessments": [], "new_hypotheses": [], "verification": []}
            tb = self.toolbox(fs)
            if not self.cfg.scout or not (set(tb.tools) & SCOUT_TOOLS) or not hyps:
                return empty
            results: list[ToolResult] = []
            plan = agents.scout_plan(self.llm, self.cfg.disease_phenotype or "(not given)", hyps,
                                     self.cfg.scout_searches_per_hypothesis, self.cfg.scout_searches_total)
            for p in plan:
                tool = str(p.get("tool", ""))
                if tool in SCOUT_TOOLS and tool in tb.tools:
                    results.append(tb.run(ToolCall(tool=tool, args=dict(p.get("args") or {})),
                                          f"R{len(results) + 1:04d}"))
            ok = {r.id: r for r in results if r.ok}
            read = agents.scout_read(self.llm, hyps, list(ok.values()), self.cfg.scout_new_hypotheses)
            assessments = [self._check_assessment(a, ok) for a in read["assessments"]]
            assessments = [a for a in assessments if a is not None]
            new, ver = [], []
            for raw in read["new_hypotheses"]:
                try:
                    h = Hypothesis.model_validate({**raw, "disease": raw.get("disease") or self.cfg.disease})
                except (ValueError, TypeError):
                    continue
                h.inspired_by = [i for i in (inv.normalize_id(x) for x in h.inspired_by) if i in ok]
                if not h.inspired_by:
                    continue  # a literature-inspired lead must point at something actually retrieved
                retrieved = {p["pmid"] for i in h.inspired_by for p in ok[i].data.get("papers", [])}
                h.pmids = [p for p in h.pmids if p in retrieved]
                h.id, h.author = f"L{len(new) + 1}", "literature_scout"
                for j, st in enumerate(h.steps, 1):
                    st.id = f"{h.id}.S{j}"
                    st.finding_ids = [inv.normalize_id(x) for x in st.finding_ids]
                v = verify(h, fs)
                ver.append({"hypothesis_id": h.id, "verification": v,
                            "objections": verifier_objections(h, v, fs, self.study().dataset)})
                new.append(h)
            return {"results": results, "assessments": assessments, "new_hypotheses": new, "verification": ver}
        return self.stage("scout", ["verify_r0"], fn)

    def _check_assessment(self, a: dict, ok: dict[str, ToolResult]):
        try:
            la = LitAssessment.model_validate({**a, "result_ids": [inv.normalize_id(x) for x in a.get("result_ids")
                                                                  or []],
                                               "pmids": [str(x).replace("PMID", "").strip() for x in a.get("pmids")
                                                         or []]})
        except (ValueError, TypeError):
            return None
        cited = [i for i in la.result_ids if i in ok]
        retrieved = {p["pmid"] for i in cited for p in ok[i].data.get("papers", [])}
        text = " ".join(ok[i].summary for i in cited).lower()
        quote = " ".join(la.quote.split()).strip(' "\'').lower()
        problems = []
        if la.support != "not_found" and not cited:
            problems.append("cites no retrieved result")
        if any(p not in retrieved for p in la.pmids) and any(ok[i].tool == "literature_search" for i in cited):
            problems.append("PMID not in the cited search")
        if quote and quote not in " ".join(text.split()):
            problems.append("quote not found verbatim")
        if problems:
            la.verified = False
            la.note = f"[unverified: {'; '.join(problems)}] {la.note}"
            la.support = "not_found" if "cites no retrieved result" in problems else la.support
        la.result_ids = cited
        return la

    def _new_question(self, raw: dict, n: int, asked_by: str, rnd: int, results: list[ToolResult],
                      tb: Toolbox) -> Question | None:
        try:
            kind = inv.normalize_kind(raw.get("kind"))
            q = Question(id=f"Q{n:02d}", kind=kind, text=str(raw.get("text", "")).strip(),
                         cells=[str(x) for x in raw.get("cells") or []], genes=[str(x) for x in raw.get("genes") or []],
                         hypothesis_ids=[str(x) for x in raw.get("hypothesis_ids") or []],
                         parent_id=raw.get("parent_id"), asked_by=asked_by, round=rnd,
                         calls=[ToolCall(tool=str(c.get("tool", "")), args=dict(c.get("args") or {}))
                                for c in (raw.get("calls") or [])[: self.cfg.calls_per_question] if isinstance(c, dict)])
        except (TypeError, ValueError, AttributeError):
            return None
        if not q.text:
            return None
        for call in q.calls:
            r = tb.run(call, f"R{len(results) + 1:04d}")
            results.append(r)
            q.result_ids.append(r.id)
        return q

    def _apply_answers(self, qs: list[Question], answers: list[dict], results: list[ToolResult],
                       fs: list[Finding] | None = None) -> list[Objection]:
        by_id = {q.id: q for q in qs}
        ok = {r.id for r in results if r.ok}
        res = {r.id: r for r in results}
        fby = {f.id: f for f in fs or []}
        rejected: list[Objection] = []
        for a in answers:
            if not isinstance(a, dict):
                continue
            q = by_id.get(str(a.get("question_id", "")).strip())
            if q is None or q.status != "open":
                continue
            if not inv.answer_matches(q, list(a.get("answer_ids") or []), str(a.get("answer", "")), ok):
                rejected.append(Objection(target=q.id, kind="unsupported_step", source="verifier",
                                          text=f"Answer rejected: it does not use this question's data or name its "
                                               f"genes/cells (likely filed under the wrong id): "
                                               f"{str(a.get('answer', ''))[:160]}"))
                continue
            text = str(a.get("answer", "")).strip()
            ids = [inv.normalize_id(x) for x in a.get("answer_ids") or []]
            cited = [i for i in ids if i in ok or i in fby]
            if q.result_ids and not any(r in ok for r in q.result_ids) and not cited:
                # every tool call failed and nothing else is cited: whatever the model wrote is not from data
                q.status, q.conclusion, q.answer_ids = "unanswerable_with_data", "none", []
                q.answer = "All tool calls for this question failed; no answer from data."
                rejected.append(Objection(target=q.id, kind="unsupported_step", source="verifier",
                                          text=f"Answer discarded: its tool calls failed, so it cannot rest on data: "
                                               f"{text[:160]}"))
                continue
            st = a.get("status")
            q.status = st if st in get_args(QuestionStatus) and st != "open" else "partial"
            q.decisive_observation = str(a.get("decisive_observation") or "").strip()[:400]
            ob = str(a.get("observable_here") or "").strip().lower()
            q.observable_here = ob if ob in ("yes", "no", "other_data") else ""
            q.answer = text
            q.answer_ids = ids
            c = a.get("conclusion")
            q.conclusion = c if c in ("observed", "literature", "inferred", "none") else "inferred"
            q.prior_knowledge = [str(x) for x in a.get("prior_knowledge") or []]
            q.candidates = []
            for c in a.get("candidates") or []:
                if not isinstance(c, dict) or not str(c.get("ligand", "")).strip():
                    continue
                v = str(c.get("verdict", "untested")).strip().lower()
                q.candidates.append(Candidate(
                    ligand=str(c["ligand"]).strip(), sender=str(c.get("sender") or ""),
                    receptor=str(c.get("receptor") or ""), reason=str(c.get("reason") or ""),
                    verdict=v if v in ("supported", "weakened", "rejected", "untested") else "untested"))
            sources = [res[i].summary for i in {*cited, *q.result_ids} if i in res and res[i].ok]
            sources += [digest_line(fby[i]) for i in cited if i in fby]
            known_pmids = {p["pmid"] for i in {*cited, *q.result_ids} if i in res and res[i].ok
                           for p in res[i].data.get("papers", [])}
            fake = [p for p in cited_pmids(text) if p not in known_pmids]
            if fake:
                q.answer += f" [verifier: PMIDs not among the retrieved papers it cites: {', '.join(fake)}]"
                rejected.append(Objection(target=q.id, kind="invalid_citation", source="verifier",
                                          text=f"PMIDs not retrieved by a cited literature search: {', '.join(fake)}."))
            bad = inv.untraceable_numbers(re.sub(r"PMID[:\s]*\d+", " ", text), sources)
            if bad:
                q.conclusion = "inferred" if q.conclusion == "observed" else q.conclusion
                q.answer += f" [verifier: not found in the cited data: {', '.join(bad)}]"
                rejected.append(Objection(target=q.id, kind="unsupported_step", source="verifier",
                                          text=f"Numbers not found in the cited data: {', '.join(bad)}."))
        return rejected

    def _sample_ids(self) -> set[str]:
        return {x for ss in self.study().dataset.groups.values() for x in ss}

    def _chain_ctx(self) -> dict:
        ds = self.study().dataset
        return {"groups": ds.groups, "case": self.cfg.disease, "reference": ds.reference_group}

    def _missing_text(self, reqs, qs, results, tb) -> list[str]:
        """Required topics not yet asked and upstream chains not yet complete, most important first."""
        label = {1: "[KEY] ", 2: "", 3: "[lower priority] "}
        items = [(r.priority, f"{label[r.priority]}{r.text}") for r in inv.missing(reqs, qs)]
        items += [(r.priority, (f"{label[r.priority]}{r.text} — asked, but the upstream chain is incomplete: next, "
                               f"{hint} (as a follow-up)"))
                  for r, hint in inv.untested_upstream(reqs, qs, results, self._strict_chain(tb), self._chain_ctx())]
        return [t for _, t in sorted(items, key=lambda x: x[0])][: self.cfg.missing_shown]

    @staticmethod
    def _strict_chain(tb: Toolbox) -> bool:
        """Require the full gene -> TF -> receptor -> ligand chain when the curated tools exist."""
        return {"known_regulators", "receptor_paths"} <= set(tb.tools)

    def _admit(self, raws: list, qs: list[Question], max_q: int, rnd: int) -> tuple[list, int]:
        """Chasing quota: from the second round on, at least half of the new questions must follow up an existing
        question (parent_id). New-topic questions beyond the number of follow-ups are dropped."""
        raws = [r for r in raws if isinstance(r, dict)][:max_q]
        if rnd <= 1 or not qs:
            return raws, 0
        ids = {q.id for q in qs}
        follow = [r for r in raws if str(r.get("parent_id") or "").strip() in ids]
        new = [r for r in raws if str(r.get("parent_id") or "").strip() not in ids]
        keep = follow + new[: len(follow)]
        return keep, len(new) - min(len(new), len(follow))

    def _loop(self, ctx, qs: list[Question], results: list[ToolResult], rounds: int, max_q: int, asked_by: str,
              tag: str) -> tuple[list[Question], list[ToolResult], list[Objection]]:
        _ds, fs, context = self._findings(ctx, self.cfg.investigate_digest)
        context += self._prior_work_text(ctx)
        tb = self.toolbox(fs)
        hyps = self._hyps(ctx)
        reqs = self._reqs(ctx, hyps)
        rejected: list[Objection] = []
        for rnd in range(1, rounds + 1):
            missing = self._missing_text(reqs, qs, results, tb)
            by_id = {r.id: r for r in results}
            reply = agents.investigate_round(
                self.llm, f"{tag}:r{rnd}", context, self.cfg.disease_phenotype or "(not given)", hyps,
                tb.catalog_text(), agents.log_text(qs), agents.open_text(qs, by_id), missing, rnd, rounds, max_q)
            rejected += self._apply_answers(qs, reply["answers"], results, fs)
            accepted, dropped = self._admit(reply["questions"], qs, max_q, rnd)
            for raw in accepted:
                q = self._new_question(raw, len(qs) + 1, asked_by, rnd, results, tb)
                if q is not None:
                    qs.append(q)
            self.event(stage=tag, action="round", round=rnd, questions=len(qs), results=len(results),
                       dropped_new_topics=dropped)
        known = inv.known_ids({f.id for f in fs}, results)
        return qs, results, rejected + inv.check_answers(qs, known)

    # ---- investigation threads -----------------------------------------------------------------------------
    def _thread_question(self, raw: dict, n: int, asked_by: str, rnd: int, qs: list[Question]) -> Question | None:
        try:
            parent = str(raw.get("parent_id") or "").strip() or None
            q = Question(id=f"Q{n:02d}", kind=inv.normalize_kind(raw.get("kind")), text=str(raw.get("text", "")).strip(),
                         cells=[str(x) for x in raw.get("cells") or []], genes=[str(x) for x in raw.get("genes") or []],
                         hypothesis_ids=[str(x) for x in raw.get("hypothesis_ids") or []],
                         parent_id=parent if parent in {x.id for x in qs} else None, asked_by=asked_by, round=rnd,
                         why=str(raw.get("why") or ""))
        except (TypeError, ValueError, AttributeError):
            return None
        if not q.text:
            return None
        q.step, q.step_label = inv.assign_step(raw.get("step"), qs)
        if q.step and inv.is_exhausted(q.step, qs) and not str(raw.get("new_approach") or "").strip():
            # the data already ran out on this step and the ask names no new kind of test: its best guess stands
            self._deferred.append({"step": q.step, "label": q.step_label, "text": q.text, "asked_by": asked_by})
            self.event(stage="threads", action="deferred", step=q.step, asked_by=asked_by, text=q.text[:200])
            return None
        return q

    def _review(self, ans: dict, q: Question, own: list[ToolResult], execute, tb: Toolbox) -> str | None:
        """System review of a thread's answer before it is final: 'supported' verdicts that rest on no
        discriminating test (inv.test_validity), and a literature search for each supported link if the thread ran
        none. Facts the thread must weigh, not a verdict."""
        sup = [c for c in ans.get("candidates") or [] if isinstance(c, dict)
               and str(c.get("verdict", "")).lower() == "supported" and str(c.get("ligand", "")).strip()][:3]
        if not sup:
            return None
        gene = q.genes[0] if q.genes else ""
        cell = q.cells[0] if q.cells else None
        notes = []
        for c in sup:
            keys = {p for p in re.split(r"[_/\s,+()]+", f"{c.get('ligand', '')} {c.get('receptor', '')}".upper()) if p}
            ok, why = [], []
            for r in own:
                if r.ok and r.tool not in ("upstream_screen", "literature_search"):
                    v, reason = inv.test_validity(r, keys, gene, cell)
                    (ok if v else why).append(r.id if v else f"{r.id}: {reason}")
            if not ok:
                notes.append(f"- Your 'supported' verdict for {c['ligand']} rests on no test that can tell it from its "
                             f"rivals in this thread ({'; '.join(why[:3]) or 'no test was run'}). Downgrade it to "
                             f"'untested' or 'weakened', or say what discriminating evidence supports it.")
        if "literature_search" in tb.tools and not any(r.tool == "literature_search" for r in own):
            word = re.sub(r"[^A-Za-z ]", " ", cell or "").strip()
            for c in sup[:2]:
                lig = (re.findall(r"[A-Za-z0-9-]+", str(c["ligand"])) or [""])[0]
                query = " ".join([lig] + [x for x in (gene, word) if x and x.upper() != lig.upper()])
                r = execute("literature_search", {"query": query, "n": 4})
                notes.append(f"- Literature for {lig} -> {gene or 'this program'}" + (f" in {cell}" if cell else "")
                             + f": [{r.id}] " + (agents.clip(r.summary, 1500) if r.ok else f"search failed: {r.summary[:200]}"))
        if not notes:
            return None
        return ("REVIEW from the system before your answer is final (checks, not the user):\n" + "\n".join(notes)
                + "\n\nGive your final answer in the answer form. Keep, revise or downgrade each verdict in light of "
                  "this; if the literature does not show the link in this cell type, say so. Cite result ids you use.")

    def _thread_loop(self, ctx, qs: list[Question], results: list[ToolResult], rounds: int, per_round: int,
                     asked_by: str, tag: str, seeds: list[dict] | None = None, hyps: list[Hypothesis] | None = None
                     ) -> tuple[list[Question], list[ToolResult], list[Objection]]:
        """Plan a few threads, run each as its own tool loop (in parallel), check every answer, repeat. Coverage of the
        required topics is shown to the planner as information; nothing is forced."""
        _ds, fs, context = self._findings(ctx, self.cfg.investigate_digest)
        context += self._prior_work_text(ctx)
        tb = self.toolbox(fs)
        if tb.has_cells:
            _ = tb.cells  # load once, before threads share the toolbox
        hyps = self._hyps(ctx) if hyps is None else hyps
        reqs = self._reqs(ctx, hyps)
        phen = self.cfg.disease_phenotype or "(not given)"
        rejected: list[Objection] = []
        for rnd in range(1, rounds + 1):
            if rnd == 1 and seeds:
                raws = seeds[:per_round]
            else:
                miss = sorted((r for r in inv.missing(reqs, qs) if r.priority <= 2), key=lambda r: r.kind != "output")
                gaps = [("REQUIRED: " if r.kind == "output" else "") + r.text for r in miss][: self.cfg.missing_shown]
                raws = agents.plan_threads(self.llm, f"{tag}:plan{rnd}", context, phen, hyps, self._short_log(qs),
                                           gaps, per_round, rnd, rounds, agents.open_leads(qs),
                                           agents.cross_patterns(qs))
            new = []
            for raw in raws:
                q = self._thread_question(raw, len(qs) + 1, asked_by, rnd, qs)
                if q is not None:
                    qs.append(q)
                    new.append(q)
            if not new:
                break
            # each thread gets its own block of result ids, so ids (and the request cache) do not depend on timing
            base = max([int(r.id[1:]) for r in results if r.id[1:].isdigit()] + [0]) + 1
            block = self.cfg.thread_steps + 2
            log = self._short_log(qs)
            by_id = {q.id: q for q in qs}
            catalog = tb.catalog_text()

            def work(item, base=base, block=block, log=log, by_id=by_id, catalog=catalog):
                k, q = item
                own: list[ToolResult] = []

                def execute(tool, args):
                    r = tb.run(ToolCall(tool=tool, args=args), f"R{base + k * block + len(own):04d}")
                    own.append(r)
                    return r
                system, user = agents.thread_prompt(context, phen, hyps, catalog, log, q, by_id.get(q.parent_id or ""),
                                                    self.cfg.thread_steps)
                ans, steps = agents.run_thread(self.llm, f"{tag}:{q.id}", system, user, execute,
                                               self.cfg.thread_steps, self.cfg.thread_result_cap,
                                               review=lambda a: self._review(a, q, own, execute, tb))
                return q, ans, steps, own
            with ThreadPoolExecutor(max_workers=max(1, self.cfg.thread_workers)) as ex:
                done = list(ex.map(work, enumerate(new)))
            for q, ans, steps, own in done:
                results.extend(own)
                q.result_ids = [r.id for r in own]
                q.calls = [ToolCall(tool=r.tool, args=r.args) for r in own]
                q.steps = steps
                if ans is None:
                    q.status, q.answer, q.conclusion = "partial", "The thread ended without an answer.", "none"
                    continue
                q.follow_ups = [str(x) for x in ans.get("follow_ups") or []][:5]
                rejected += self._apply_answers(qs, [{**ans, "question_id": q.id}], results, fs)
                if q.status == "open":  # the answer failed the checks; keep it visible, marked
                    q.status, q.conclusion = "partial", "none"
                    q.answer = "[rejected by the answer checks] " + str(ans.get("answer", ""))[:800]
            results.sort(key=lambda r: int(r.id[1:]) if r.id[1:].isdigit() else 0)
            self.event(stage=tag, action="threads", round=rnd, threads=[q.id for q in new], questions=len(qs),
                       results=len(results))
        known = inv.known_ids({f.id for f in fs}, results)
        return qs, results, rejected + inv.check_answers(qs, known)

    # ---- stages ---------------------------------------------------------------------------------------------
    def _s_load(self, ctx):
        # Always re-read: the dataset fingerprint (file hashes) decides whether anything downstream reruns.
        def fn():
            ds = self.study().dataset
            if not self.cfg.study_question:  # left out entirely, so the load record (and every fingerprint) is unchanged
                return ds.model_dump(exclude={"question"})
            return ds
        return self.stage("load", [], fn, always=True)

    def _s_findings(self, ctx):
        def fn():
            s, fc, lab = self.study(), self.cfg.findings, self.cfg.lab_tables
            expr = fnd.load_expression(s, fc, cells=self.cells)
            self.event(stage="findings", action="expression", source=expr.label if expr else None,
                       genes=len(expr.genes) if expr else 0, lineages=expr.lineages if expr else [])
            kb = KnowledgeBase(Path(self.cfg.kb_dir) if self.cfg.kb_dir else KB_DIR) if self.cfg.knowledge else None
            kb = kb if kb is not None and kb.available else None  # gene roles and target-gene scores need it
            fs = fnd.build_findings(s, fc, expr, lab_tables=lab, primary=self.cfg.disease, kb=kb)
            dg = fnd.digest(fs, fc)
            return {"findings": fs, "digest_ids": [f.id for f in dg],
                    "chance_separations": fnd.expected_chance_separations(fc, s, expr, lab, self.cfg.disease)}
        return self.stage("findings", ["load"], fn)

    def panel(self, ctx) -> list[Persona]:
        ds, fs, _ = self._findings(ctx)
        cells = [c for f in fs for c in (f.cell_type, f.target_cell_type) if c]
        spec = self.study().cells
        if self.cfg.panel is None and spec is not None and Path(spec.path).exists():
            cells += lineages_from_cells(spec)  # every lineage of the study, not only those with findings
        groups = group_order(ds.groups, ds.reference_group, self.cfg.disease)
        return study_panel(self.cfg.panel, cells, self.cfg.disease, groups, ds.reference_group)

    def _s_panel(self, ctx):
        _, _, context = self._findings(ctx)
        panel = self.panel(ctx)

        def fn():
            self.event(stage="panel", action="personas", panel=[p.model_dump() for p in panel])
            return {p.title: agents.panel_propose(self.llm, p, context, self.cfg.disease, self.cfg.panel_hypotheses_each,
                                                  self.cfg.panel_temperature, self.cfg.disease_phenotype or "(not given)")
                    for p in panel}
        return self.stage("panel", ["findings"], fn)

    def _s_generate(self, ctx):
        _, _, context = self._findings(ctx)
        if not ctx.get("panel"):
            return self.stage("generate", ["findings"],
                              lambda: agents.generate(self.llm, context, self.cfg.disease, self.cfg.n_hypotheses))
        proposals = {k: [Hypothesis.model_validate(h) for h in v] for k, v in ctx["panel"].items()}
        return self.stage("generate", ["panel"], lambda: agents.merge_proposals(
            self.llm, context, self.cfg.disease, proposals, self.cfg.n_hypotheses, self.cfg.merge_temperature,
            self.cfg.disease_phenotype or "(not given)"))

    def _s_single(self, ctx):
        _, _, context = self._findings(ctx)
        return self.stage("single", ["findings"],
                          lambda: agents.single_agent(self.llm, context, self.cfg.disease, self.cfg.n_hypotheses))

    def _s_verify_r0(self, ctx):
        src = "single" if self.cfg.mode == "single" else "candidates" if ctx.get("candidates") else "generate"
        ds, fs, _ = self._findings(ctx)
        hyps = self._base_hyps(ctx)

        def fn():
            out = []
            for h in hyps:
                v = verify(h, fs)
                out.append({"hypothesis_id": h.id, "verification": v,
                            "objections": verifier_objections(h, v, fs, ds)})
            return out
        return self.stage("verify_r0", [src, "findings"], fn)

    def _s_investigate(self, ctx):
        who = "single_agent" if self.cfg.mode == "single" else "investigator"

        def fn():
            start = [ToolResult.model_validate(r) for r in (ctx.get("scout") or {}).get("results", [])]
            first: list[Question] = []
            if ctx.get("check"):  # the chase builds on stage 2: its questions are in the log, its results citable
                first = [Question.model_validate(q) for q in ctx["check"]["questions"]]
                start += [ToolResult.model_validate(r) for r in ctx["check"]["results"]]
            if self.cfg.investigation == "threads":
                qs, results, objs = self._thread_loop(ctx, first, start, self.cfg.thread_rounds,
                                                      self.cfg.threads_per_round, who, "investigate")
            else:
                qs, results, objs = self._loop(ctx, [], start, self.cfg.investigate_rounds,
                                               self.cfg.questions_per_round, who, "investigate")
            reqs = self._reqs(ctx)
            return {"questions": qs, "results": results, "coverage": inv.coverage(reqs, qs, results, self._strict_chain(self._toolbox),
                                                                          self._chain_ctx()),
                    "answer_objections": objs}
        return self.stage("investigate", ["scout"], fn)

    def _missing_texts(self, ctx, qs: list[Question]) -> list[str]:
        """Required questions never asked, the output questions first and marked REQUIRED."""
        miss = sorted(inv.missing(self._reqs(ctx), qs), key=lambda r: (r.kind != "output", r.priority))
        return [("REQUIRED: " if r.kind == "output" else "") + r.text for r in miss][: self.cfg.missing_shown]

    EXPANDED_MIN = 0.05  # a cell type whose share of the disease samples is at least 5 points above the reference

    def _reqs(self, ctx, hyps: list[Hypothesis] | None = None) -> list:
        """The required questions: from the hypotheses, plus the output of every cell type the disease expands."""
        if getattr(self, "_expanded", None) is None:
            fs = [Finding.model_validate(f) for f in ctx["findings"]["findings"] if f.get("kind") == "composition"]
            self._expanded = sorted({f.cell_type for f in fs if f.direction == "up" and f.case == self.cfg.disease
                                     and abs(f.effect) >= self.EXPANDED_MIN})
        return inv.requirements(self._hyps(ctx) if hyps is None else hyps, self._expanded)

    def _investigation(self, ctx, name: str) -> tuple[list[Question], list[ToolResult]]:
        d = ctx[name]
        return [Question.model_validate(q) for q in d["questions"]], [ToolResult.model_validate(r) for r in d["results"]]

    def _log_with_patterns(self, qs: list[Question]) -> str:
        """The investigation log with cross-thread patterns, within log_budget: full answers if they fit, else
        shortened answers, else the most recent questions (earlier ones are named, and summarised in stage 2)."""
        pats = agents.cross_patterns(qs)
        tail = ("\n\nPATTERNS ACROSS THREADS (computed from the verdicts):\n" + "\n".join(f"- {p}" for p in pats)
                if pats else "") + self._steps_tail(qs)
        budget = self.cfg.log_budget - len(tail)
        log = agents.log_text(qs)
        if len(log) > budget:
            log = agents.thread_log(qs, cap=300)
        return self._trim(log, qs, budget) + tail

    def _short_log(self, qs: list[Question]) -> str:
        """The log the planner and each thread see (shortened answers), within half the log budget."""
        tail = self._steps_tail(qs)
        return self._trim(agents.thread_log(qs), qs, self.cfg.log_budget // 2 - len(tail)) + tail

    @staticmethod
    def _steps_tail(qs: list[Question]) -> str:
        txt = inv.steps_text(qs)
        return f"\n\nCAUSAL STEPS (each question's step; EXHAUSTED = the data could not settle it):\n{txt}" if txt else ""

    @staticmethod
    def _trim(log: str, qs: list[Question], budget: int) -> str:
        if len(log) > budget:
            lines, kept, size = log.splitlines(), [], 0
            for ln in reversed(lines):
                if size + len(ln) + 1 > budget - 400:
                    break
                kept.append(ln)
                size += len(ln) + 1
            dropped = [q.id for q in qs if q.status != "open"][: len(lines) - len(kept)]
            log = (f"[{len(dropped)} earlier questions not shown for length ({dropped[0]}-{dropped[-1]}); the stage 2 "
                   f"summary in the context covers the pathology check]\n" + "\n".join(reversed(kept)))
        return log

    def _budget_results(self, results: list[ToolResult]) -> list[ToolResult]:
        """Cited results within results_budget, most recent first (the deepest work), shown in id order."""
        out, size = [], 0
        for r in sorted(results, key=lambda r: r.id, reverse=True):
            n = min(len(r.summary), 1500) + 200
            if size + n > self.cfg.results_budget:
                continue
            out.append(r)
            size += n
        return sorted(out, key=lambda r: r.id)

    def _synth(self, ctx, qs, results, objections: str, tag: str, author: str):
        _ds, fs, context = self._findings(ctx, self.cfg.investigate_digest)
        cited = {i for q in qs for i in q.answer_ids}
        shown = self._budget_results([r for r in results if r.id in cited and r.ok])
        syn = agents.synthesize(self.llm, tag, context, self.cfg.disease_phenotype or "(not given)", self._hyps(ctx),
                                self._log_with_patterns(qs), agents.results_text(shown, cap=1500), objections, author)
        if syn is None:
            return {"synthesis": None, "objections": []}
        lit_ids = {r.id for r in results if r.ok and r.tool == "literature_search" and r.data.get("papers")}
        objs = inv.check_synthesis(syn, inv.known_ids({f.id for f in fs}, results), {q.id for q in qs}, lit_ids)
        if tag.startswith("revise") and self.staged:
            objs += inv.check_phenotype(syn, [i for i, _, _ in self._phenotype_items(ctx)])
        retrieved = {p["pmid"] for r in results if r.ok for p in r.data.get("papers", [])}
        text = json.dumps(syn.model_dump())
        fake = sorted({p for p in cited_pmids(text) if p not in retrieved})
        if fake:
            objs.append(Objection(target="model", kind="invalid_citation", source="verifier",
                                  text=f"PMIDs never retrieved in this investigation: {', '.join(fake)}."))
        return {"synthesis": syn, "objections": objs}

    def _s_synthesize(self, ctx):
        qs, results = self._investigation(ctx, "investigate")
        author = "single_agent" if self.cfg.mode == "single" else "synthesizer"
        return self.stage("synthesize", ["investigate"], lambda: self._synth(ctx, qs, results, "", "synthesize", author))

    def _s_critique(self, ctx):
        _ds, fs, context = self._findings(ctx, self.cfg.investigate_digest)
        qs, results = self._investigation(ctx, "investigate")
        syn = ctx["synthesize"]["synthesis"]

        def fn():
            if syn is None:
                return {"verdict": "unsupported", "objections": [], "questions": []}
            missing = self._missing_texts(ctx, qs)
            return agents.critique_model(self.llm, context, Synthesis.model_validate(syn), self._log_with_patterns(qs),
                                         missing, self.toolbox(fs).catalog_text(), self.cfg.critic_questions,
                                         inv.known_ids({f.id for f in fs}, results))
        return self.stage("critique", ["synthesize"], fn)

    def _s_chase(self, ctx):
        _, fs, _ = self._findings(ctx)

        def fn():
            qs, results = self._investigation(ctx, "investigate")
            c0 = ctx["critique"]
            streak = 0 if (c0.get("new_tests") or c0.get("new_testable_predictions")) else 1
            if self.staged and self.cfg.chase_progress in ("model", "steps"):
                streak = 0
            first = ctx["synthesize"].get("synthesis")
            best = self._model_score(Synthesis.model_validate(first)) if first else (0, 0)
            reviews = [{"round": 1, "verdict": c0.get("verdict"), "sign_off": bool(c0.get("sign_off")),
                        "depth": c0.get("depth", []), "demands": len(c0.get("questions", [])),
                        "new_tests": c0.get("new_tests", 0),
                        "new_testable_predictions": c0.get("new_testable_predictions", 0), "no_progress": streak}]
            model, stop, seeds = None, "first review only", []
            # staged: no fixed number of PI rounds; stop on sign-off, no demands, or three reviews without progress
            limit = self.cfg.gate_cap if self.staged else self.cfg.pi_rounds
            if self.staged and self.cfg.chase_max_rounds:
                limit = min(limit, self.cfg.chase_max_rounds)
            last_objs = [Objection.model_validate(o) for o in ctx["critique"].get("objections", [])]
            if self.cfg.investigation == "threads":
                seeds = [r for r in ctx["critique"].get("questions", []) if isinstance(r, dict)]
                qs, results, objs = self._thread_loop(ctx, qs, results, self.cfg.chase_thread_rounds,
                                                      max(self.cfg.critic_questions, 1), "critic", "chase", seeds)
                # further PI rounds: re-synthesise what is now known, let the PI review it, chase its demands
                _ds, fs2, context = self._findings(ctx, self.cfg.investigate_digest)
                stop = "round limit"
                steps_before = inv.step_registry(qs)
                for r in range(2, limit + 1):
                    if reviews[-1]["sign_off"] or not reviews[-1]["demands"]:
                        stop = "sign-off" if reviews[-1]["sign_off"] else "no demands"
                        break
                    if self.staged and streak >= self.cfg.no_progress_limit:
                        stop = "no progress"
                        break
                    interim = self._synth(ctx, qs, results, "", f"synthesize:pi{r}", "synthesizer")["synthesis"]
                    if interim is None:
                        stop = "interim synthesis unusable"
                        break
                    model = interim
                    score = self._model_score(interim)
                    improved = score[0] > best[0] or score[1] > best[1]
                    best = (max(best[0], score[0]), max(best[1], score[1]))
                    missing = self._missing_texts(ctx, qs)
                    crit = agents.critique_model(self.llm, context, interim, self._log_with_patterns(qs), missing,
                                                 self.toolbox(fs2).catalog_text(), self.cfg.critic_questions,
                                                 inv.known_ids({f.id for f in fs2}, results), tag=f"critique_model:pi{r}")
                    if self.staged and self.cfg.chase_progress == "steps":
                        steps_now = inv.step_registry(qs)
                        streak = 0 if inv.step_progress(steps_before, steps_now) else streak + 1
                        steps_before = steps_now
                    elif self.staged and self.cfg.chase_progress == "model":
                        streak = 0 if improved else streak + 1
                    else:
                        streak = 0 if (crit.get("new_tests") or crit.get("new_testable_predictions")) else streak + 1
                    reviews.append({"round": r, "verdict": crit.get("verdict"), "sign_off": crit.get("sign_off"),
                                    "depth": crit.get("depth", []), "demands": len(crit.get("questions", [])),
                                    "new_tests": crit.get("new_tests", 0),
                                    "new_testable_predictions": crit.get("new_testable_predictions", 0),
                                    "no_progress": streak})
                    self.event(stage="chase", action="pi_review", round=r, sign_off=crit.get("sign_off"),
                               demands=len(crit.get("questions", [])))
                    last_objs = crit.get("objections", [])
                    seeds = [x for x in crit.get("questions", []) if isinstance(x, dict)]
                    if crit.get("sign_off") or not seeds or (self.staged and streak >= self.cfg.no_progress_limit):
                        stop = ("sign-off" if crit.get("sign_off") else "no demands" if not seeds else "no progress")
                        break
                    qs, results, more = self._thread_loop(ctx, qs, results, 1, max(self.cfg.critic_questions, 1),
                                                          "critic", f"chase{r}", seeds)
                    objs += more
            else:
                tb = self.toolbox(fs)
                for raw in ctx["critique"].get("questions", []):
                    q = self._new_question(raw, len(qs) + 1, "critic", 0, results, tb)
                    if q is not None:
                        qs.append(q)
                qs, results, objs = self._loop(ctx, qs, results, self.cfg.chase_rounds, 4, "investigator", "chase")
            objs = [Objection.model_validate(o) for o in ctx["investigate"].get("answer_objections", [])] + objs
            reqs = self._reqs(ctx)
            return {"questions": qs, "results": results, "coverage": inv.coverage(reqs, qs, results, self._strict_chain(self._toolbox),
                                                                          self._chain_ctx()),
                    "answer_objections": objs, "pi_reviews": reviews, "pi_objections": last_objs, "model": model,
                    "stop": stop, "open_demands": [str(x.get("text", "")) for x in (seeds or []) if isinstance(x, dict)],
                    "deferred": list(self._deferred)}
        return self.stage("chase", ["critique"], fn)

    @staticmethod
    def _model_score(syn: Synthesis) -> tuple[int, int]:
        """How far the integrated model has got: tested chain steps, and total link depth."""
        return (sum(st.kind == "tested" for lk in syn.links for st in lk.steps),
                sum(lk.depth or 0 for lk in syn.links))

    def _s_revise(self, ctx):
        qs, results = self._investigation(ctx, "chase")
        # answer the latest PI review (the first one when there was only one round)
        objs = [Objection.model_validate(o) for o in (ctx["chase"].get("pi_objections")
                                                      or ctx["critique"].get("objections", []))]
        text = "\n".join(f"- [{o.kind}] {o.target}: {o.text}" for o in objs if o.valid)
        if self.staged and ctx["chase"].get("stop") in ("no progress", "round limit", "gate cap"):
            # the chase stopped improving the model: the data have reached their limit on what is still open
            left = ctx["chase"].get("open_demands") or []
            text += ("\n\nSTUCK: the PI's chase stopped because further rounds no longer improved the model. These "
                     "demands were still open:\n" + "\n".join(f"- {d}" for d in left[:8]) + "\nThe data have reached "
                     "their limit on these points. For every open step give your ranked best guesses now (best_guesses), "
                     "each on the strongest rung of evidence it has, and say plainly in the summary that this is where "
                     "the data ran out.")
        deferred = ctx["chase"].get("deferred") or []
        if self.staged and deferred:
            text += ("\n\nEXHAUSTED STEPS asked again without a new kind of test (not run; give each its ranked best "
                     "guess and its decisive observation as the test):\n"
                     + "\n".join(f"- {d['step']} {d['label']}: {d['text'][:300]}" for d in deferred[:10]))
        results = results + [ToolResult.model_validate(r) for r in (ctx.get("linkage") or {}).get("results", [])]
        text += self._linkage_text(ctx)
        items = self._phenotype_items(ctx) if self.staged else []
        if items:
            text += agents.phenotype_text(items)

        def fn():
            out = self._synth(ctx, qs, results, text, "revise", "reviser")
            if self.staged and out["synthesis"] is not None:
                out = self._final_gate(ctx, qs, results, text, out, [])
            if out["synthesis"] is None and ctx["synthesize"].get("synthesis"):
                # keep the first model rather than losing it; say so in the report
                out = {"synthesis": ctx["synthesize"]["synthesis"], "objections": list(ctx["synthesize"]["objections"])
                       + [Objection(target="model", kind="missing_evidence", source="verifier",
                                    text="The revision could not be parsed; this is the unrevised model.")]}
            return out
        dep = "linkage" if ctx.get("linkage") else "experiment" if ctx.get("experiment") else "chase"
        return self.stage("revise", [dep], fn)

    def _phenotype_items(self, ctx) -> list[tuple[str, str, str]]:
        return agents.phenotype_items((ctx.get("dossier") or {}).get("dossier") or {})

    def _final_gate(self, ctx, qs, results, text, out, experiments):
        """Stage 6: the PI reviews the integrated model; its demands are chased in the data and the model rebuilt,
        until it signs off or three reviews bring no progress."""
        _ds, _fs, context = self._findings(ctx, self.cfg.investigate_digest)
        st = {"qs": list(qs), "results": list(results), "out": out}  # the thread loop appends in place

        def material():
            syn = st["out"]["synthesis"]
            syn.experiments_planned = experiments
            objs = "\n".join(f"- [{o.kind}] {o.text}" for o in st["out"]["objections"]) or "(none)"
            items = {i: (sec, t) for i, sec, t in self._phenotype_items(ctx)}
            fit = {f.item.strip().upper(): f for f in syn.phenotype_fit}
            table = "\n".join(f"- {i} [{sec}] {t[:160]} => " + (f"{fit[i].verdict}: {fit[i].reason}"
                               + (f" (links: {'; '.join(fit[i].links)})" if fit[i].links else "") if i in fit
                               else "NOT ACCOUNTED FOR") for i, (sec, t) in items.items())
            return (("PHENOTYPE ACCOUNTING (how the model ties into the disease):\n" + table + "\n\n" if items else "")
                    + "Integrated model (JSON):\n" + json.dumps(syn.model_dump(), indent=1)[:24000]
                    + "\n\nAutomatic checks on it:\n" + objs)

        def act(demands, n):
            qs2, res2, _more = self._thread_loop(ctx, st["qs"], st["results"], 1, max(self.cfg.critic_questions, 1),
                                                 "critic", f"final{n + 1}", demands)
            asks = "\n".join(f"- [PI] {d['text']}" for d in demands)
            new = self._synth(ctx, qs2, res2, text + "\n" + asks, f"revise:r{n + 1}", "reviser")
            st.update(qs=qs2, results=res2)
            if new["synthesis"] is not None:
                st["out"] = new
        reviews = self._gate(context, "6. Integrated model", material, act, "final", final=True)
        st["out"]["synthesis"].experiments_planned = experiments
        return {**st["out"], "final_reviews": reviews, "final_questions": st["qs"][len(qs):],
                "final_results": st["results"][len(results):]}

    # ---- the staged workflow: a PI gate after every stage -------------------------------------------------------
    @property
    def staged(self) -> bool:
        return self.cfg.mode == "multi" and self.cfg.workflow == "staged"

    def _gate(self, context: str, stage: str, material: Callable[[], str], act: Callable[[list[dict], int], None],
              tag: str, final: bool = False, progress: Callable[[], int] | None = None,
              real: Callable[[], tuple] | None = None, cap: int | None = None) -> list[dict]:
        """The PI reviews a stage until it passes, signs off (final stage only), has no demands, or three reviews in a
        row bring no new test and no new testable prediction. No fixed number of rounds; gate_cap guards a broken
        loop. act(demands, n) does the stage's work on the demands. progress() (literature stages) counts new
        evidence the stage now cites, so a dossier that improves through searches is not closed as stalled.
        real() (circuits): a score of what the stage has actually established (e.g. observed edges); when given, a
        review counts as progress only if the score rose since the best so far, whatever new tests the PI names.
        cap: this gate's own review limit (below gate_cap)."""
        reviews: list[dict] = []
        streak = 0
        best = real() if real is not None else None
        phen = self.cfg.disease_phenotype or "(not given)"
        for n in range(1, min(self.cfg.gate_cap, cap or self.cfg.gate_cap) + 1):
            r = agents.pi_gate(self.llm, f"{tag}:gate{n}", context, phen, stage, material(), reviews)
            if r["verdict"] == "sign_off" and not final:
                r["verdict"] = "pass"
            r["new_cited"] = progress() if progress is not None else 0
            r["repeated"] = bool(reviews) and inv.repeated_demands(r["demands"], reviews[-1]["demands"])
            if real is not None:  # progress means the stage established something, not that the PI asked for more
                now = real()
                moved = any(a > b for a, b in zip(now, best)) and not r["repeated"]
                best = tuple(max(a, b) for a, b in zip(now, best))
                r["established"] = list(now)
            else:
                moved = (r["new_tests"] > 0 or r["new_testable_predictions"] > 0 or r["new_cited"] > 0) and not r["repeated"]
            streak = 0 if moved else streak + 1
            r.update(review=n, no_progress=streak)
            reviews.append(r)
            closed = ("pass" if r["verdict"] in ("pass", "sign_off") else "no demands" if not r["demands"]
                      else "no progress" if streak >= self.cfg.no_progress_limit else None)
            self.event(stage=tag, action="pi_gate", review=n, verdict=r["verdict"], new_tests=r["new_tests"],
                       new_predictions=r["new_testable_predictions"], demands=len(r["demands"]), closed=closed)
            if closed:
                r["closed_by"] = closed
                break
            act(r["demands"], n)
        else:
            reviews[-1]["closed_by"] = "review cap" if cap and cap < self.cfg.gate_cap else "gate cap"
        return reviews

    def _restricted(self, tb: Toolbox, allowed: set[str], results: list[ToolResult], start: int):
        """A tool executor limited to the knowledge tools (dossier, experiments) and its catalog text."""
        allowed = allowed & set(tb.tools)
        catalog = "\n".join(line for line in tb.catalog_text(staged=True).splitlines()
                            if any(line.startswith(f"- {t}(") for t in allowed))

        def execute(tool, args):
            rid = f"R{start + len(results):04d}"
            if tool not in allowed:
                r = ToolResult(id=rid, tool=tool, args=args, ok=False,
                               summary=f"{tool} is not available in this stage; tools: {', '.join(sorted(allowed))}")
            else:
                r = tb.run(ToolCall(tool=tool, args=args), rid)
            results.append(r)
            return r
        return execute, catalog

    @staticmethod
    def _lit_ok(results: list[ToolResult]) -> set[str]:
        return {r.id for r in results if r.ok and r.tool in LIT_TOOLS | DRUG_TOOLS}

    def _s_dossier(self, ctx):
        _ds, fs, context = self._findings(ctx, self.cfg.investigate_digest)
        phen = self.cfg.disease_phenotype or "(not given)"

        def fn():
            tb = self.toolbox(fs)
            results: list[ToolResult] = []
            execute, catalog = self._restricted(tb, LIT_TOOLS, results, 5001)
            st: dict = {"dossier": {}, "steps": [], "dropped": 0, "dropped_items": [], "seen": set()}

            def draft(previous: str, tag: str):
                system = agents.DOSSIER_SYSTEM.format(steps=self.cfg.dossier_steps)
                user = agents.DOSSIER_USER.format(context=context, phenotype=phen, catalog=catalog, previous=previous)
                ans, steps = agents.run_thread(self.llm, tag, system, user, execute, self.cfg.dossier_steps,
                                               self.cfg.thread_result_cap)
                st["steps"] += steps
                if ans is None:
                    return
                ok, out = self._lit_ok(results), {"data_observations": []}
                by_pmid: dict[str, list[str]] = {}
                for r in results:
                    for p in (r.data or {}).get("papers", []) if r.ok else []:
                        by_pmid.setdefault(str(p.get("pmid")), []).append(r.id)
                known_f = {f.id for f in fs}

                def resolve(y):  # a search id, or a PMID a search here returned (models often cite those)
                    i = inv.normalize_id(str(y))
                    if i in ok:
                        return [i]
                    return by_pmid.get(re.sub(r"\D", "", str(y)), [])
                for key in ("defining_features", "cells_states", "mediators_genes", "genetics", "therapies",
                            "comparator_differences"):
                    items = [x for x in ans.get(key) or [] if isinstance(x, dict) and x.get("text")]
                    out[key] = []
                    for x in items:
                        ids = list(dict.fromkeys(i for y in x.get("ids") or [] for i in resolve(y)))
                        fids = [i for i in (inv.normalize_id(str(y)) for y in x.get("ids") or []) if i in known_f]
                        if ids:  # every literature item rests on a search run here
                            out[key].append({"text": str(x["text"]), "ids": ids})
                        elif fids:  # a data observation, not literature: kept, labelled as such
                            out["data_observations"].append({"text": str(x["text"]), "ids": fids})
                        else:
                            st["dropped"] += 1
                            st["dropped_items"].append({"section": key, "text": x["text"], "ids": x.get("ids")})
                known = {f.id for f in fs}
                out["novelty_leads"] = [{"text": str(x["text"]),
                                         "finding_ids": [i for i in x.get("finding_ids") or [] if i in known]}
                                        for x in ans.get("novelty_leads") or [] if isinstance(x, dict) and x.get("text")]
                st["dossier"] = out
            draft("", "dossier:r1")

            def material():
                return ("Dossier:\n" + agents.dossier_text(st["dossier"]) + f"\n\n(items dropped for lacking a search "
                        f"id: {st['dropped']})\n\nSearches run:\n"
                        + "\n".join(f"[{r.id}] {r.tool}({json.dumps(r.args)}){'' if r.ok else ' FAILED'}"
                                    for r in results))

            def act(demands, n):
                draft("Your previous dossier (JSON):\n" + json.dumps(st["dossier"])[:12000]
                      + "\n\nThe PI's demands (address each with searches, then give the full revised dossier):\n"
                      + "\n".join(f"- {d['text']}" for d in demands), f"dossier:r{n + 1}")
            def cited():  # search ids the dossier cites that it did not cite at the last review
                now = {i for k, v in st["dossier"].items() if k != "data_observations" for x in v
                       for i in x.get("ids") or []}
                new, st["seen"] = len(now - st["seen"]), st["seen"] | now
                return new
            reviews = self._gate(context, "1. Pathology dossier", material, act, "dossier", progress=cited)
            return {"dossier": st["dossier"], "results": results, "steps": st["steps"], "dropped": st["dropped"],
                    "dropped_items": st["dropped_items"][-40:], "reviews": reviews}
        return self.stage("dossier", ["findings"], fn)

    def _s_check(self, ctx):
        phen = self.cfg.disease_phenotype or "(not given)"

        def fn():
            _ds, _fs, context = self._findings(ctx, self.cfg.investigate_digest)
            dossier = ctx["dossier"].get("dossier") or {}
            seeds = agents.plan_check(self.llm, "check:plan", context, phen, dossier, self.cfg.check_threads)
            results = [ToolResult.model_validate(r) for r in ctx["dossier"]["results"]]
            qs, results, objs = self._thread_loop(ctx, [], results, 1, self.cfg.check_threads, "analyst", "check",
                                                  seeds, hyps=[])
            st = {"qs": qs, "results": results, "objs": objs,
                  "summary": agents.summarize_check(self.llm, "check:summary1", dossier, self._log_with_patterns(qs))}

            def material():
                return ("Summary:\n" + agents.check_text(st["summary"]) + "\n\nAnalysts' log:\n"
                        + self._log_with_patterns(st["qs"]))

            def act(demands, n):
                qs2, res2, more = self._thread_loop(ctx, st["qs"], st["results"], 1, self.cfg.check_threads, "critic",
                                                    f"check{n + 1}", demands, hyps=[])
                st.update(qs=qs2, results=res2, objs=st["objs"] + more,
                          summary=agents.summarize_check(self.llm, f"check:summary{n + 1}", dossier,
                                                         self._log_with_patterns(qs2)))
            reviews = self._gate(context, "2. Do we see the pathology?", material, act, "check")
            return {"summary": st["summary"], "questions": st["qs"], "results": st["results"],
                    "answer_objections": st["objs"], "reviews": reviews}
        return self.stage("check", ["dossier"], fn)

    def _s_candidates(self, ctx):
        _ds, _fs, context = self._findings(ctx)

        def fn():
            st = {"hyps": [Hypothesis.model_validate(h) for h in ctx["generate"]]}

            def material():
                return "Candidate mechanisms:\n" + "\n".join(
                    f"{h.id}: {h.title}\n  {h.summary}\n" + "\n".join(
                        f"  {s.id}: {s.source_cell} -[{s.mediator}]-> {s.target_cell or '·'} ({s.direction}; "
                        f"{', '.join(s.genes)}; findings {', '.join(s.finding_ids) or '-'})" for s in h.steps)
                    + (f"\n  predictions: {'; '.join(h.predictions)}" if h.predictions else "") for h in st["hyps"])

            def act(demands, n):
                st["hyps"] = agents.revise_candidates(self.llm, f"candidates:r{n + 1}", context, self.cfg.disease,
                                                      st["hyps"], demands)
            reviews = self._gate(context, "3. Candidates", material, act, "candidates")
            return {"hypotheses": st["hyps"], "reviews": reviews}
        return self.stage("candidates", ["generate", "check"], fn)

    def _s_linkage(self, ctx):
        """Each link of the current model, placed across medicine: this disease, the diseases the dossier says it
        occurs with, related diseases, the same state in other tissues, genetics, therapy. One thread per link."""
        _ds, fs, context = self._findings(ctx, self.cfg.investigate_digest)
        phen = self.cfg.disease_phenotype or "(not given)"

        def fn():
            model = ctx["chase"].get("model") or ctx["synthesize"].get("synthesis") or {}
            syn = Synthesis.model_validate(model) if model else None
            links = (syn.links if syn else [])[: self.cfg.linkage_links]
            tb = self.toolbox(fs)
            prior = [ToolResult.model_validate(r) for r in ctx["chase"]["results"]]
            base = max([int(r.id[1:]) for r in prior if r.id[1:].isdigit()] + [0]) + 1
            block = self.cfg.linkage_steps + 4
            out: dict[int, dict] = {}
            results: list[ToolResult] = []
            steps_log: list[dict] = []

            def run(round_no, demands):
                def one(item):
                    i, lk = item
                    own: list[ToolResult] = []
                    execute, catalog = self._restricted(tb, LIT_TOOLS, own, base + (round_no - 1) * 1000 + i * block)
                    desc = (f"{lk.source} -[{lk.mediator}]-> {lk.target} (depth {lk.depth or '?'}). {lk.note[:600]}")
                    prev = ""
                    if out.get(i):
                        prev = ("Your earlier findings for this link (JSON):\n" + json.dumps(out[i])[:6000])
                    if demands:
                        prev += "\n\nThe PI's demands (address those that concern this link):\n" + "\n".join(
                            f"- {d['text']}" for d in demands)
                    system = agents.LINKAGE_SYSTEM.format(steps=self.cfg.linkage_steps)
                    user = agents.LINKAGE_USER.format(context=context, phenotype=phen, title=syn.title, summary=syn.summary,
                                                      link=desc, catalog=catalog, previous=prev)
                    ans, steps = agents.run_thread(self.llm, f"linkage{round_no}:L{i + 1}", system, user, execute,
                                                   self.cfg.linkage_steps, self.cfg.thread_result_cap)
                    return i, lk, ans, steps, own
                with ThreadPoolExecutor(max_workers=max(1, self.cfg.thread_workers)) as ex:
                    done = list(ex.map(one, enumerate(links)))
                for i, lk, ans, steps, own in done:
                    results.extend(own)
                    steps_log.extend(steps)
                    ok = self._lit_ok(results) | self._lit_ok(prior)
                    if ans is None:
                        continue
                    found = [{"aspect": str(f.get("aspect", "")), "text": str(f.get("text", "")),
                              "ids": [x for x in (inv.normalize_id(str(y)) for y in f.get("ids") or []) if x in ok]}
                             for f in ans.get("findings") or [] if isinstance(f, dict) and f.get("text")]
                    out[i] = {"link": f"{lk.source} -[{lk.mediator}]-> {lk.target}",
                              "findings": [f for f in found if f["ids"]], "suggests": str(ans.get("suggests", "")),
                              "rung": ans.get("rung")}
            run(1, [])
            seen: set[str] = set()

            def material():
                return "Disease linkage per link (JSON):\n" + json.dumps([out[k] for k in sorted(out)], indent=1)[:24000]

            def cited():
                now = {x for v in out.values() for f in v["findings"] for x in f["ids"]}
                new = len(now - seen)
                seen.update(now)
                return new
            rounds = [1]

            def act(demands, n):
                rounds.append(rounds[-1] + 1)
                run(rounds[-1], demands)
            reviews = self._gate(context, "6a. Disease linkage", material, act, "linkage", progress=cited)
            return {"links": [out[k] for k in sorted(out)], "results": results, "steps": steps_log, "reviews": reviews}
        return self.stage("linkage", ["chase"], fn)

    def _linkage_text(self, ctx) -> str:
        lk = (ctx.get("linkage") or {}).get("links") or []
        if not lk:
            return ""
        lines = ["", "Disease linkage per link (stage 6a; cite these R ids in each link's disease_linkage):"]
        for x in lk:
            lines.append(f"- {x['link']}: {x['suggests']}")
            lines += [f"    [{f['aspect']}] {f['text'][:300]} [{', '.join(f['ids'])}]" for f in x["findings"][:8]]
        return "\n".join(lines)

    @staticmethod
    def _circuit(c, i: int) -> Circuit | None:
        """One circuit from a model reply, tidied: rank, edges with a source, a known status tag, string ids."""
        if not isinstance(c, dict) or not c.get("title"):
            return None
        c = {**c, "rank": c.get("rank") or i,
             "edges": [e for e in c.get("edges") or [] if isinstance(e, dict) and e.get("source")],
             "phenotype": [p for p in c.get("phenotype") or [] if isinstance(p, dict)],
             "predictions": [p for p in c.get("predictions") or [] if isinstance(p, dict)],
             "loops": [str(x) for x in c.get("loops") or []], "unused": [str(x) for x in c.get("unused") or []],
             "support": str(c.get("support") or "")}
        for e in c["edges"]:
            if e.get("status") not in ("observed", "likely", "analogy", "speculative"):
                e["status"] = "speculative"
            e["ids"] = [str(x) for x in e.get("ids") or []]
        try:
            return Circuit.model_validate(c)
        except ValueError:
            return None

    def _s_circuits(self, ctx):
        """Stage 7: with the facts settled, the hypothesizer proposes complete circuits; the PI reviews plausibility."""
        _ds, fs, context = self._findings(ctx, self.cfg.investigate_digest)
        phen = self.cfg.disease_phenotype or "(not given)"

        def fn():
            ledger = (ctx.get("revise") or {}).get("synthesis") or ctx["chase"].get("model") or {}
            tb = self.toolbox(fs)
            diffs = (tb.difference_account(baseline=self.cfg.difference_baseline) if tb.has_cells
                     else "DIFFERENCE ACCOUNT: (no single-cell object)")
            qs = [Question.model_validate(q) for q in ctx["chase"]["questions"]]
            qs += [Question.model_validate(q) for q in (ctx.get("revise") or {}).get("final_questions", [])]
            exhausted = "\n".join(f"- {st['id']} {st['label']}" + (f": would be settled by {st['decisive'][-1]}"
                                    if st["decisive"] else "") for st in inv.step_registry(qs) if st["exhausted"])
            items = "\n".join(f"{i} [{sec}] {t[:300]}" for i, sec, t in self._phenotype_items(ctx)) or "(none)"
            prior = [ToolResult.model_validate(r) for r in ctx["chase"]["results"]]
            prior += [ToolResult.model_validate(r) for r in (ctx.get("linkage") or {}).get("results", [])]
            prior += [ToolResult.model_validate(r) for r in (ctx.get("revise") or {}).get("final_results", [])]
            start = max([int(r.id[1:]) for r in prior if r.id[1:].isdigit()] + [0]) + 1
            results: list[ToolResult] = []  # shared by every attempt, so result ids stay unique
            execute, catalog = self._restricted(tb, CIRCUIT_TOOLS, results, start)
            led = json.dumps(ledger, indent=1)[:30000]
            # every ledger link, numbered, for the mandatory accounting (used / set aside / contradicted)
            link_rows = [lk for lk in (ledger.get("links") or []) if isinstance(lk, dict)]
            link_ids = [f"L{i}" for i in range(1, len(link_rows) + 1)]
            links = "\n".join(f"{i} {lk.get('source', '')} -> {lk.get('target', '')} via {lk.get('mediator', '')}"
                               for i, lk in zip(link_ids, link_rows)) or "(none)"
            k_all = max(1, self.cfg.circuit_attempts)

            def attempt(k: int) -> dict:
                """One independent set of circuits: a draft from the evidence, an optional targeted search on its
                open steps, then the PI's review until the circuits stop establishing anything new."""
                base = "circuits" if k_all == 1 else f"circuits:k{k}"
                st: dict = {"circuits": [], "ranking": "", "objections": [], "steps": [], "sweep": [], "accounting": [],
                            "sweep_text": ""}
                note = (f"\n(Independent attempt {k} of {k_all}: reason from the evidence yourself.)" if k_all > 1
                        else "")

                def targeted_search() -> None:
                    if not (self.cfg.literature and self.cfg.circuit_sweep > 0 and "literature_search" in tb.tools):
                        return
                    drafted = json.dumps([c.model_dump() for c in st["circuits"]], indent=1)[:20000]
                    plan = agents.sweep_plan(self.llm, f"{base}:sweep", phen, drafted, diffs, items,
                                             self.cfg.circuit_sweep)
                    parts = []
                    for q in plan:
                        r = execute("literature_search", {"query": q["query"]})
                        st["sweep"].append({**q, "result_id": r.id, "ok": r.ok})
                        parts.append(f"[{r.id}] ({q['kind']}) {q['query']}: {q['why']}\n{r.summary[:2500]}")
                    st["sweep_text"] = ("\nTARGETED LITERATURE (searched for your first draft's unproven steps; weigh "
                                        "it, it does not replace the data):\n" + ("\n\n".join(parts) or "(no queries)")
                                        + "\n")

                def draft(previous: str, tag: str):
                    user = agents.CIRCUIT_USER.format(context=context, phenotype=phen, ledger=led, differences=diffs,
                                                      exhausted=exhausted or "(none)", items=items, links=links,
                                                      sweep=st["sweep_text"], catalog=catalog,
                                                      previous=previous + note)
                    ans, steps = agents.run_thread(self.llm, tag,
                                                   agents.CIRCUIT_SYSTEM.format(steps=self.cfg.circuit_steps), user,
                                                   execute, self.cfg.circuit_steps, self.cfg.thread_result_cap)
                    st["steps"] += steps
                    if ans is None:
                        return
                    cs = []
                    for i, c in enumerate(ans.get("circuits") or [], 1):
                        c = self._circuit(c, i)
                        if c is not None:
                            cs.append(c)
                    cs, art = inv.artefact_circuits(cs)
                    known = inv.known_ids({f.id for f in fs}, prior + results)
                    st["objections"] = art + inv.check_circuits(cs, known, self._lit_ok(prior) | self._lit_ok(results))
                    st["circuits"], st["ranking"] = sorted(cs, key=lambda c: c.rank), str(ans.get("ranking") or "")
                    st["accounting"] = [x for x in ans.get("accounting") or [] if isinstance(x, dict)]
                    st["objections"] += inv.check_accounting(st["accounting"], link_ids, known)

                draft("", f"{base}:r1")
                targeted_search()

                def material():
                    objs = "\n".join(f"- [{o.kind}] {o.text}" for o in st["objections"]) or "(none)"
                    return ("Proposed circuits (JSON):\n"
                            + json.dumps([c.model_dump() for c in st["circuits"]], indent=1)[:24000]
                            + f"\n\nWhat would change the ranking: {st['ranking'] or '(not said)'}"
                            + "\n\nAutomatic checks on the edge tags:\n" + objs + "\n\n" + diffs
                            + ("\n\nTargeted literature searched for the draft:\n"
                               + "\n".join(f"- [{q['result_id']}] ({q['kind']}) {q['query']}" for q in st["sweep"])
                               if st["sweep"] else ""))

                def act(demands, n):
                    draft("Your previous circuits (JSON):\n"
                          + json.dumps([c.model_dump() for c in st["circuits"]])[:14000]
                          + "\n\nThe PI's demands (address each, then give the full revised set of circuits):\n"
                          + "\n".join(f"- {d['text']}" for d in demands), f"{base}:r{n + 1}")

                def established() -> tuple:
                    """What the circuits have actually pinned down: observed edges, published analogies that ran,
                    and disease items the circuits produce."""
                    cs = st["circuits"]
                    return (sum(e.status == "observed" for c in cs for e in c.edges),
                            sum(e.status == "analogy" for c in cs for e in c.edges),
                            len({p.get("item", "") for c in cs for p in c.phenotype if isinstance(p, dict)}))
                reviews = self._gate(context, "7. Circuits", material, act, base, real=established,
                                     cap=self.cfg.circuit_review_cap)
                return {"attempt": k, "circuits": [c.model_dump() for c in st["circuits"]], "ranking": st["ranking"],
                        "objections": [o.model_dump() if hasattr(o, "model_dump") else o for o in st["objections"]],
                        "steps": st["steps"], "reviews": reviews, "sweep": st["sweep"], "accounting": st["accounting"]}

            attempts = [attempt(k) for k in range(1, k_all + 1)]
            out = {"circuits": attempts[0]["circuits"], "ranking": attempts[0]["ranking"],
                   "objections": attempts[0]["objections"], "leads": [], "consensus": False}
            if k_all > 1:
                merged = agents.circuit_consensus(self.llm, "circuits:consensus", phen, items, attempts)
                if merged:
                    built = inv.assemble_consensus(merged, attempts)
                    cs = [c for c in (self._circuit(c, i) for i, c in enumerate(built, 1)) if c]
                    cs, art = inv.artefact_circuits(cs)
                    known = inv.known_ids({f.id for f in fs}, prior + results)
                    objs = art + inv.check_circuits(cs, known, self._lit_ok(prior) | self._lit_ok(results))
                    # which attempt circuits each merged one stands for (kept off Circuit, whose dump feeds prompts)
                    origin = {c.get("rank"): c.get("merged_from") or [] for c in built}
                    out = {"circuits": [c.model_dump() | {"merged_from": origin.get(c.rank, [])}
                                        for c in sorted(cs, key=lambda c: c.rank)],
                           "ranking": str(merged.get("ranking") or ""), "objections": objs,
                           "leads": [x for x in merged.get("leads") or [] if isinstance(x, dict)], "consensus": True}
            return {**out, "results": results, "attempts": attempts,
                    "steps": [s_ for a_ in attempts for s_ in a_["steps"]],
                    "reviews": [{**r, "attempt": a_["attempt"]} for a_ in attempts for r in a_["reviews"]],
                    "differences": diffs, "sweep": [q for a_ in attempts for q in a_["sweep"]],
                    "accounting": attempts[0]["accounting"] if k_all == 1 else
                    {a_["attempt"]: a_["accounting"] for a_ in attempts}}
        return self.stage("circuits", ["revise"], fn)

    def _agenda(self, ctx) -> tuple[list[dict], list[dict]]:
        """The final circuits and their missing links (untested edges, exhausted steps, leads), the agenda of the
        experiment and therapy stages."""
        circs = [c for c in (ctx.get("circuits") or {}).get("circuits") or [] if isinstance(c, dict)]
        qs = [Question.model_validate(q) for q in (ctx.get("chase") or {}).get("questions") or []]
        exhausted = [st for st in inv.step_registry(qs) if st["exhausted"]]
        leads = [x for x in (ctx.get("circuits") or {}).get("leads") or [] if isinstance(x, dict)]
        return circs, inv.missing_links(circs, exhausted, leads)

    def _prior_results(self, ctx) -> list[ToolResult]:
        out = []
        for name, key in (("dossier", "results"), ("check", "results"), ("chase", "results"), ("linkage", "results"),
                          ("revise", "final_results"), ("circuits", "results"), ("experiment", "results")):
            out += [ToolResult.model_validate(r) for r in (ctx.get(name) or {}).get(key, []) or []]
        return out

    def _s_experiment(self, ctx):
        _ds, fs, context = self._findings(ctx, self.cfg.investigate_digest)
        phen = self.cfg.disease_phenotype or "(not given)"

        def fn():
            final = (ctx.get("revise") or {}).get("synthesis") if "revise" in ctx else None
            ledger = final or (ctx.get("chase") or {}).get("model") or (ctx.get("synthesize") or {}).get("synthesis") or {}
            circs, ms = self._agenda(ctx)
            model = ({"circuits": circs, "ranking": (ctx.get("circuits") or {}).get("ranking", ""),
                      "evidence_ledger": ledger} if circs else ledger)
            tb = self.toolbox(fs)
            prior = self._prior_results(ctx)
            start = max([int(r.id[1:]) for r in prior if r.id[1:].isdigit()] + [0]) + 1
            results: list[ToolResult] = []
            execute, catalog = self._restricted(tb, DRUG_TOOLS, results, start)
            st: dict = {"experiments": [], "not_testable": [], "uncovered": [m["id"] for m in ms], "objections": [],
                        "steps": [], "seen": set(), "unconfirmed": [], "programmes": []}
            terms = {g for c in circs for e in c.get("edges") or [] if isinstance(e, dict)
                     for g in inv._target_genes(f"{e.get('via', '')} {e.get('source', '')} {e.get('target', '')}")}
            terms |= {w for c in circs for e in c.get("edges") or [] if isinstance(e, dict)
                      for w in re.findall(r"[a-z]{5,}", f"{e.get('source', '')} {e.get('target', '')}".lower())}
            shortlist = inv.papers_text(inv.retrieved_papers(prior, terms, self.cfg.experiment_papers))
            molecules = inv.crucial_molecules(circs, ms, exclude=self._sample_ids())
            mol_text = "\n".join(f"- {m['gene']} | circuits {', '.join(map(str, m['circuits']))} | {m['support']}"
                                 for m in molecules) or "(none)"

            def draft(previous: str, tag: str):
                system = agents.EXPERIMENT_SYSTEM.format(steps=self.cfg.experiment_steps)
                user = agents.EXPERIMENT_USER.format(context=context, phenotype=phen,
                                                     model=json.dumps(model, indent=1)[:24000] + self._linkage_text(ctx),
                                                     missing=inv.missing_links_text(ms), molecules=mol_text,
                                                     papers=shortlist, catalog=catalog, previous=previous)
                ans, steps = agents.run_thread(self.llm, tag, system, user, execute, self.cfg.experiment_steps,
                                               self.cfg.thread_result_cap)
                st["steps"] += steps
                if ans is None:
                    return
                papers = {p["pmid"]: p for p in inv.retrieved_papers(prior + results, set(), limit=100000)}
                exps, objs, uncovered = inv.check_experiments(ans.get("experiments") or [], circs, ms,
                                                              self._lit_ok(results) | self._lit_ok(prior), papers)
                mids = {m["id"] for m in ms}
                nt = [{"id": str(x.get("id")), "why": str(x.get("why") or "")} for x in ans.get("not_testable") or []
                      if isinstance(x, dict) and str(x.get("id")) in mids]
                ranked = inv.rank_experiments(exps, circs)
                progs, pobjs = inv.check_programmes(ans.get("programmes") or [], ranked, circs)
                objs = objs + pobjs
                st.update(experiments=ranked, not_testable=nt, objections=objs, programmes=progs,
                          uncovered=[m for m in uncovered if m not in {x["id"] for x in nt}],
                          unconfirmed=inv.uncovered_molecules(ranked, molecules))

            draft("", "experiment:r1")

            def material():
                return ("Each hypothesis's sequence of experiments (JSON, in the experimentalist's order):\n"
                        + json.dumps(st["programmes"], indent=1)[:9000]
                        + "\n\nThe experiments (JSON):\n" + json.dumps(st["experiments"], indent=1)[:20000]
                        + "\n\nCode checks on the plan:\n" + ("\n".join(f"- {o.text}" for o in st["objections"]) or "none")
                        + "\n\nMissing links no experiment covers and none marked not testable: "
                        + (", ".join(st["uncovered"]) or "none")
                        + "\nMolecules the circuits rest on that no experiment checks: "
                        + (", ".join(st["unconfirmed"]) or "none") + "\nMissing links:\n" + inv.missing_links_text(ms))

            def act(demands, n):
                draft("Your previous plan (JSON):\n" + json.dumps({"experiments": st["experiments"],
                                                                  "programmes": st["programmes"],
                                                                  "not_testable": st["not_testable"]})[:18000]
                      + "\n\nThe PI's demands (address each, then give the full revised plan):\n"
                      + "\n".join(f"- {d['text']}" for d in demands), f"experiment:r{n + 1}")

            def progress():
                now = {i for e in st["experiments"] for p in e["precedent"] for i in p["ids"]}
                now |= {f"cover:{m}" for e in st["experiments"] for m in e["tests"]}
                new, st["seen"] = len(now - st["seen"]), st["seen"] | now
                return new
            reviews = self._gate(context, "8. Experiments", material, act, "experiment", progress=progress)
            return {"experiments": st["experiments"], "programmes": st["programmes"],
                    "not_testable": st["not_testable"], "uncovered": st["uncovered"],
                    "molecules": molecules, "unconfirmed": st["unconfirmed"], "missing_links": ms,
                    "objections": st["objections"], "results": results, "steps": st["steps"], "reviews": reviews}
        dep = ("circuits" if "circuits" in ctx else "revise" if "revise" in ctx else "linkage" if ctx.get("linkage")
               else "chase")
        return self.stage("experiment", [dep], fn)

    def _s_therapy(self, ctx):
        _ds, fs, context = self._findings(ctx, self.cfg.investigate_digest)
        phen = self.cfg.disease_phenotype or "(not given)"

        def fn():
            circs, ms = self._agenda(ctx)
            targets = inv.therapy_targets(circs, ms, self.cfg.therapy_targets, self._sample_ids())
            tb = self.toolbox(fs)
            prior = self._prior_results(ctx)
            start = max([int(r.id[1:]) for r in prior if r.id[1:].isdigit()] + [0]) + 1
            results: list[ToolResult] = []
            execute, catalog = self._restricted(tb, DRUG_TOOLS, results, start)
            for t in targets:  # every target's drugs are looked up by code, before any model call
                execute("gene_drugs", {"gene": t["gene"]})
            cell_tg: list[dict] = []
            if tb.has_cells:  # cell-type targets: the data's own markers of each lineage the circuits use
                names = tb.cells.gene_names
                markers = {lin: [names[j] for j in idx][:6] for lin, idx in tb._lineage_markers().items()}
                cell_tg = inv.cell_targets(circs, markers, cellfig.cells_in)
                for ct in cell_tg:
                    for g in ct["genes"]:
                        execute("gene_drugs", {"gene": g})
            items = agents.phenotype_items((ctx.get("dossier") or {}).get("dossier") or {})
            therapy_texts = " ".join(t for _i, sec, t in items if "therap" in sec)

            def lookups() -> dict[str, dict]:
                return {str(r.data.get("gene", "")).upper(): r.data for r in results if r.tool == "gene_drugs" and r.ok}

            def targets_text() -> str:
                lk = lookups()
                lines = []
                for t in targets:
                    ds = (lk.get(t["gene"]) or {}).get("drugs") or []
                    drugs = "; ".join(f"{d['drug']} ({d.get('stage') or '?'}; "
                                      + ", ".join(sorted({a['action'] for a in d.get('actions') or [] if a.get('action')}))
                                      + f"; {', '.join((d.get('indications') or [])[:3])})" for d in ds[:6])
                    lines.append(f"- {t['gene']} | circuits {', '.join(map(str, t['circuits']))} | {t['support']} | "
                                 f"{', '.join(t['untested']) or '-'} | {drugs or 'no drugs listed'}")
                return "\n".join(lines) or "(none)"

            def cells_text() -> str:
                lk = lookups()
                lines = []
                for ct in cell_tg:
                    drugs = "; ".join(f"{g}: " + (", ".join(f"{d['drug']} ({d.get('stage') or '?'})"
                                                            for d in (lk.get(g.upper()) or {}).get("drugs", [])[:4]) or "none")
                                      for g in ct["genes"])
                    lines.append(f"- {ct['lineage']} | circuits {', '.join(map(str, ct['circuits']))} | {drugs}")
                return "\n".join(lines) or "(no cell data)"
            st: dict = {"hypotheses": [], "pathways": [], "objections": [], "steps": [], "seen": set()}

            def draft(previous: str, tag: str):
                system = agents.THERAPY_SYSTEM.format(steps=self.cfg.therapy_steps)
                user = agents.THERAPY_USER.format(
                    context=context, phenotype=phen, circuits=json.dumps(circs, indent=1)[:20000],
                    missing=inv.missing_links_text(ms),
                    items="\n".join(f"{i} [{sec}] {t[:300]}" for i, sec, t in items), targets=targets_text(),
                    cells=cells_text(), catalog=catalog, previous=previous)
                ans, steps = agents.run_thread(self.llm, tag, system, user, execute, self.cfg.therapy_steps,
                                               self.cfg.thread_result_cap)
                st["steps"] += steps
                if ans is None:
                    return
                hyps, objs = inv.check_therapies(ans.get("hypotheses") or [], lookups(), circs, ms,
                                                 {i for i, _s, _t in items}, therapy_texts, self.cfg.disease, cell_tg)
                st.update(hypotheses=hyps, objections=objs,
                          pathways=inv.pathway_weights(ans.get("pathways") or [], targets, hyps, circs))

            draft("", "therapy:r1")

            def material():
                return ("Therapeutic hypotheses (JSON; stage, action and evidence level set by code from the lookups):\n"
                        + json.dumps(st["hypotheses"], indent=1)[:16000] + "\n\nPathways:\n"
                        + json.dumps(st["pathways"], indent=1)[:4000] + "\n\nCode objections:\n"
                        + "\n".join(f"- {o.text}" for o in st["objections"]))

            def act(demands, n):
                draft("Your previous answer (JSON):\n" + json.dumps({"hypotheses": st["hypotheses"],
                                                                    "pathways": st["pathways"]})[:12000]
                      + "\n\nThe PI's demands (address each, then give the full revised answer):\n"
                      + "\n".join(f"- {d['text']}" for d in demands), f"therapy:r{n + 1}")

            def progress():
                now = {(h["drug"], h["target"]) for h in st["hypotheses"]}
                new, st["seen"] = len(now - st["seen"]), st["seen"] | now
                return new
            reviews = self._gate(context, "9. Therapeutic hypotheses", material, act, "therapy", progress=progress)
            return {"hypotheses": st["hypotheses"], "pathways": st["pathways"], "targets": targets, "cell_targets": cell_tg,
                    "missing_links": ms, "objections": st["objections"], "results": results, "steps": st["steps"],
                    "reviews": reviews}
        return self.stage("therapy", ["experiment" if "experiment" in ctx else "circuits"], fn)

    def _s_report(self, ctx):
        def fn():
            md, metrics = report.build(self.cfg, ctx, self.dir)
            (self.dir / "report.md").write_text(md)
            return metrics
        deps = [s for s in ctx["stages"] if s in ctx and s != "report"]
        return self.stage("report", deps, fn)
