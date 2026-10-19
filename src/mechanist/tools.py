"""Deterministic data-query tools the investigator calls to chase a question. No LLM.

Every call returns a ToolResult with an ID (R0001…), a compact per-sample summary the LLM reads, the structured
values, the source files and caveats. Values are always shown per sample and grouped (primary disease first, the
reference group last), because with two disease samples the question is "do both disease samples agree, and against
which comparators", not a p-value.

Cell-level tools read the reviewed single-cell object (loaded once, on first use). Table tools read the pipeline tables
(and the lab's pg_mechanism tables when lab_tables is on).
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mechanist.adapters.base import CellSpec, LoadedStudy, group_order
from mechanist.findings import digest_line
from mechanist.investigate import ligand_differs, receptor_ok
from mechanist.knowledge import KnowledgeBase, KnowledgeError, Literature
from mechanist.records import Finding, ToolCall, ToolResult
from mechanist.verify import cell_family

MIN_CELLS = 20  # per sample and lineage/state, below which a percentage is not reported
MIN_PCT = 5.0  # % of cells for a TF, receptor or ligand to count as expressed
MAX_GENES = 6  # per gene_profile / gene_by_state call: ask about candidates you have a reason for, not sweeps
EXCLUDE_GENE = re.compile(r"^(MT-|RPL|RPS|MRPL|MRPS)")
# Structural / housekeeping ligands that dominate LIANA rankings but say little about what induces a program.
MATRIX_LIGAND = re.compile(r"^(COL\d|LAM[ABC]|FN1|LUM|DCN|BGN|LGALS|HSPG2|ADAM\d|APOE|APP|VIM|SPARC|THBS|TNC|MDK|PTN|"
                           r"HMGB1|CD99|MIF|NID|VCAN|FBN|ELN|CALR|HSP|PSAP|SEMA|EFN|NAMPT|RPS|RPL)")
# Cell-contact, adhesion and antigen-presentation ligands: real interactions, but not soluble inducers of a program.
CONTACT_LIGAND = re.compile(r"^(HLA-|CD22$|CD24$|CD47$|CD58$|CD80$|CD86$|ICAM|VCAM|CDH\d|JAM\d|NECTIN|PVR|CEACAM|"
                            r"SELP|SELE|SELL|CLEC|SIGLEC|TLN1|VCL|CALM|YBX1|PDAP1)")



def _did_you_mean(name: str, options) -> str:
    """' (did you mean X?)' for a misspelt or near-miss name; never a silent substitution."""
    import difflib
    opts = [str(o) for o in options]
    low = {o.lower(): o for o in opts}
    hits = difflib.get_close_matches(str(name).lower(), list(low), n=3, cutoff=0.5)
    hits += [o.lower() for o in opts if str(name).lower() in o.lower() and o.lower() not in hits]
    return f" (did you mean {', '.join(low[h] for h in hits[:3])}?)" if hits else ""

class ToolError(ValueError):
    pass


def _norm(name: str) -> str:
    """Case-, dash- and spacing-insensitive key (models often write U+2011 hyphens or drop punctuation)."""
    return re.sub(r"[^a-z0-9+]", "", str(name).lower())


@dataclass
class Tool:
    name: str
    args: str
    description: str
    fn: Callable[..., tuple[str, dict, list[str]]]
    needs_cells: bool = False
    staged: bool = False  # offered only where a stage asks for it (catalog_text(staged=True)), not in the general catalog


def _is_counts(X) -> bool:
    data = X.data[:200_000] if hasattr(X, "data") and not isinstance(X, np.ndarray) else np.asarray(X).ravel()[:200_000]
    return bool(len(data) == 0 or np.allclose(data, np.round(data)))


class CellData:
    """Reviewed single-cell object restricted to the study groups, CSC for gene access.

    X holds the spec's counts layer (raw counts in every current study; .X of p033's object is raw counts too).
    Detection (% of cells > 0) is the same for raw and normalised values; level comparisons use `normalised`, which
    is log1p CP10k when X holds counts and X itself otherwise (`is_counts`)."""

    def __init__(self, spec: CellSpec):
        import anndata as ad  # optional dependency (extra "cells")

        a = ad.read_h5ad(spec.path)
        grp = a.obs[spec.group_col].astype(str).map(spec.group_map)
        keep = grp.notna().to_numpy()
        state = a.obs[spec.state_col].astype(str)
        lineage = (a.obs[spec.lineage_col].astype(str) if spec.lineage_col
                   else state.map(spec.state_to_lineage).fillna("Unmapped"))
        if spec.counts_layer and spec.counts_layer not in a.layers:
            raise ValueError(f"{spec.path}: no layer {spec.counts_layer!r} (layers: {', '.join(a.layers) or 'none'})")
        X = (a.layers[spec.counts_layer] if spec.counts_layer else a.X)[keep]
        obs = pd.DataFrame({
            "sample": a.obs[spec.sample_col].astype(str).to_numpy()[keep],
            "group": grp.to_numpy()[keep],
            "state": state.to_numpy()[keep],
            "lineage": lineage.to_numpy()[keep],
        })
        if spec.doublet_col and spec.doublet_col in a.obs:
            obs["doublet_score"] = pd.to_numeric(a.obs[spec.doublet_col], errors="coerce").to_numpy()[keep]
        self._init(obs, X, list(a.var_names.astype(str)), str(spec.path))

    @classmethod
    def from_frame(cls, obs: pd.DataFrame, X, genes: list[str], path: str = "synthetic") -> CellData:
        self = cls.__new__(cls)
        self._init(obs.reset_index(drop=True), X, genes, path)
        return self

    def _init(self, obs: pd.DataFrame, X, genes: list[str], path: str):
        self.obs = obs
        self.X = X.tocsc() if hasattr(X, "tocsc") else np.asarray(X)
        self.genes = {g: i for i, g in enumerate(genes)}
        self.gene_names = np.array(genes)
        # Neutral order here; the toolbox orders groups for display (primary disease first, reference last).
        self.samples = self.obs[["group", "sample"]].drop_duplicates().sort_values(["group", "sample"])
        self.path = path
        self.is_counts = _is_counts(self.X)
        self.libsize = np.asarray(self.X.sum(1)).ravel().astype(float) if self.is_counts else None

    def normalised(self, rows: np.ndarray):
        """Rows of X on a comparable scale: log1p CP10k from counts, X as stored otherwise."""
        Xr = self.X[rows]
        if not self.is_counts:
            return Xr
        from scipy import sparse

        lib = np.where(self.libsize[rows] > 0, self.libsize[rows], 1.0)
        Xn = sparse.diags(1e4 / lib) @ sparse.csr_matrix(Xr)
        Xn.data = np.log1p(Xn.data)
        return Xn

    def gene(self, g: str) -> np.ndarray:
        if g not in self.genes:
            alt = [x for x in self.genes if x.upper() == g.upper()]
            if not alt:
                raise ToolError(f"gene {g!r} is not in the single-cell object")
            g = alt[0]
        col = self.X[:, self.genes[g]]
        return np.asarray(col.toarray()).ravel() if hasattr(col, "toarray") else np.asarray(col).ravel()

    def lineage_mask(self, lineage: str) -> np.ndarray:
        lins = self.obs.lineage.unique()
        hit = [x for x in lins if x.lower() == lineage.lower()] or \
              [x for x in lins if cell_family(x) == cell_family(lineage)]
        if not hit:
            raise ToolError(f"unknown lineage {lineage!r}{_did_you_mean(lineage, lins)}; lineages: "
                            f"{', '.join(sorted(lins))}")
        return self.obs.lineage.isin(hit).to_numpy()

    def state_mask(self, state: str) -> np.ndarray:
        states = self.obs.state.unique()
        key = _norm(state)
        hit = [x for x in states if _norm(x) == key] or [x for x in states if key and key in _norm(x)]
        if not hit:
            raise ToolError(f"unknown state {state!r}; use one of: {', '.join(sorted(states))}")
        return self.obs.state.isin(hit[:1]).to_numpy()


def _fmt(values: dict[str, float | None], samples: pd.DataFrame, order: list[str], unit: str = "",
         nd: int = 1) -> str:
    """'Disease: D1 42.5, D2 25.7 | Other: … | Control: …' in `order` (— where a sample has too few cells)."""
    parts = []
    for g in order:
        ss = samples[samples.group == g]["sample"].tolist()
        if not ss:
            continue
        vals = []
        for s in ss:
            v = values.get(s)
            vals.append(f"{s} {'—' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{v:.{nd}f}{unit}'}")
        parts.append(f"{g}: " + ", ".join(vals))
    return " | ".join(parts)


def _agreement(values: dict[str, float | None], samples: pd.DataFrame, order: list[str]) -> str:
    """Do all case samples lie above / below every sample of each other group?"""
    def vals(g):
        return [values[s] for s in samples[samples.group == g]["sample"] if values.get(s) is not None
                and not np.isnan(values[s])]
    case = order[0]
    cv = vals(case)
    if len(cv) < 2:
        return ""
    out = []
    for g in order:
        if g == case:
            continue
        rv = vals(g)
        if len(rv) < 2:
            continue
        if min(cv) > max(rv):
            out.append(f"all {case} > all {g}")
        elif max(cv) < min(rv):
            out.append(f"all {case} < all {g}")
        else:
            out.append(f"{case} overlaps {g}")
    return "; ".join(out)


def sample_wide(up: dict[str, list[str]]) -> tuple[set[str], dict[str, int], int]:
    """Genes raised in at least 60% (and at least 3) of the lineages that have changes: a sample-wide signal. Needs 3+
    lineages. Returns (genes, lineage count per gene, the threshold)."""
    rose: dict[str, int] = {}
    for genes in up.values():
        for g in set(genes):
            rose[g] = rose.get(g, 0) + 1
    k = max(3, int(np.ceil(0.6 * len(up))))
    return ({g for g, n in rose.items() if n >= k} if len(up) >= 3 else set()), rose, k


class Toolbox:
    def __init__(self, study: LoadedStudy, findings: list[Finding], cells: CellData | None = None,
                 kb: KnowledgeBase | None = None, literature: Literature | None = None, lab_tables: bool = True,
                 primary: str | None = None, expression_loader: Callable | None = None, opentargets=None):
        self.study = study
        self.ot = opentargets
        ds = study.dataset
        if primary is None:  # the study's standard contrast, else the first non-reference group
            primary = study.standard_contrast[0] if study.standard_contrast else None
        self.order = group_order(ds.groups, ds.reference_group, primary)
        if not self.order:
            raise ValueError(f"{ds.study}: no groups")
        self.case, self.ref = self.order[0], ds.reference_group
        self.contrast = (f"{study.standard_contrast[0]} vs {study.standard_contrast[1]}" if study.standard_contrast
                         else f"{self.case} vs {self.ref}")
        self.findings = findings
        self._cells = cells
        self.kb = kb
        self.lit = literature
        self._idx = None
        self._lock = threading.RLock()
        self._expression_loader = expression_loader
        self._sandbox = None
        self.on_result = None
        self._done: dict[str, ToolResult] = {}  # call (tool + args) -> its first successful result  # called with every result (the run's transcript)
        self._expression = None
        self._scores = None
        self.tools: dict[str, Tool] = {t.name: t for t in self._catalog()
                                       if (t.name not in ("known_regulators", "known_pairs", "receptor_paths")
                                           or kb is not None)
                                       and (t.name not in ("receptor_paths", "upstream_screen")
                                            or (kb is not None and kb.has_signalling))
                                       and (t.name != "pathway_footprint" or kb is not None)
                                       and (t.name != "literature_search" or literature is not None)
                                       and (t.name not in ("gene_diseases", "gene_drugs") or opentargets is not None)
                                       and (t.name != "fibroblast_programs" or lab_tables)}

    # ---- plumbing -------------------------------------------------------------------------------------------
    @property
    def cells(self) -> CellData:
        if self._cells is None:
            if self.study.cells is None:
                raise ToolError("no single-cell object for this study")
            self._cells = CellData(self.study.cells)
        return self._cells

    @property
    def has_cells(self) -> bool:
        return self._cells is not None or self.study.cells is not None

    def catalog_text(self, staged: bool = False) -> str:
        return "\n".join(f"- {t.name}({t.args}): {t.description}" for t in self.tools.values()
                         if (self.has_cells or not t.needs_cells) and (staged or not t.staged))

    def run(self, call: ToolCall, rid: str) -> ToolResult:
        if call.tool == "compute":  # runs in its own process for up to minutes: never hold the other threads up
            with self._lock:
                _ = self.cells
            r = self._run(call, rid)
        else:
            with self._lock:  # threads share the toolbox; tools are fast and their lazy caches are not thread-safe
                r = self._run(call, rid)
        if self.on_result is not None:
            try:
                self.on_result(r)
            except Exception:  # noqa: BLE001, S110 - the transcript must never break a run
                pass
        return r

    def _run(self, call: ToolCall, rid: str) -> ToolResult:
        import inspect
        tool = self.tools.get(call.tool)
        if tool is None:
            return ToolResult(id=rid, tool=call.tool, args=call.args, ok=False,
                              summary=f"unknown tool {call.tool!r}; tools: {', '.join(self.tools)}")
        # an argument the tool does not take (often the step's "purpose" put inside args) is dropped, and said so
        params = inspect.signature(tool.fn).parameters
        takes_any = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
        extra = [] if takes_any else [k for k in call.args if k not in params]
        args = {k: v for k, v in call.args.items() if k not in extra}
        note = [f"ignored argument(s) this tool does not take: {', '.join(extra)}"] if extra else []
        key = json.dumps([call.tool, {k: v for k, v in args.items() if k != "purpose"}], sort_keys=True, default=str)
        prev = self._done.get(key)
        if prev is not None:  # the same call again: the same result, under its new id, pointing to the first
            return ToolResult(id=rid, tool=call.tool, args=call.args, ok=True,
                              summary=f"(same call as {prev.id}; same result) " + prev.summary, data=prev.data,
                              sources=prev.sources, caveats=prev.caveats + note)
        try:
            summary, data, caveats = tool.fn(**args)
            src = [self.cells.path] if tool.needs_cells else []
            r = ToolResult(id=rid, tool=call.tool, args=call.args, ok=True, summary=summary, data=data,
                           sources=src, caveats=list(caveats) + note)
            self._done.setdefault(key, r)
            return r
        except Exception as exc:  # noqa: BLE001 - a tool failure is a result the model sees, never a crash
            return ToolResult(id=rid, tool=call.tool, args=call.args, ok=False,
                              summary=f"error: {type(exc).__name__}: {exc}"[:600])

    def _catalog(self) -> list[Tool]:
        return [
            Tool("upstream_screen", "gene, cell", "FIRST step for any 'why does this cell make this gene' question. "
                 "Walks the curated chain (activating TFs of the gene -> receptors with a signalling route to them -> "
                 "their ligands) and keeps only what THIS study's data supports: TFs and receptors expressed on the "
                 "cell in every disease sample, and ligands whose expression separates disease from another group, "
                 "with the lineage that sends each. Returns ranked candidate inducers and what the data ruled out. "
                 "A screen, not an answer: judge and test the candidates next.", self.upstream_screen, True),
            Tool("gene_profile", f"gene (one gene or a list of up to {MAX_GENES}), lineage=None",
                 "% of cells expressing a gene per sample in each lineage (or one lineage), plus pipeline DE rows. "
                 f"Answers: who makes this gene, and is it changed in {self.case}? Omit lineage to see every lineage (do this "
                 "for candidate inducers, which are usually made by other cells).", self.gene_profile, True),
            Tool("lineage_share", "lineage", "the lineage's share of all cells per sample. Answers: is this "
                 "cell type expanded?", self.lineage_share, True),
            Tool("state_composition", "lineage", "cell states within a lineage, % per sample. Answers: what kind "
                 "of cells are these — which state is expanded?", self.state_composition, True),
            Tool("gene_by_state", f"gene (one gene or a list of up to {MAX_GENES}), lineage", "% expressing a gene in each state of a lineage, per sample. "
                 "Answers: which state makes this gene?", self.gene_by_state, True),
            Tool("lineage_changes", "lineage, n=20", "genome-wide screen of ONE lineage: every gene that separates the "
                 f"{self.case} samples from each other group (above or below every sample), ranked by the gap, with genes "
                 "flagged when another lineage makes them far more (possible ambient RNA). Answers: what has changed in "
                 "these cells, including what they now make or have lost? No gene list is needed: judge the hits.",
                 self.lineage_changes, True),
            Tool("state_markers", "state, n=20", "genes that distinguish a state from the rest of its lineage. "
                 "Answers: what are these cells doing (programs, effector genes)?", self.state_markers, True),
            Tool("coexpression", "lineage, genes_a, genes_b", "within ONE lineage's cells, per sample: % expressing "
                 "any of genes_a, any of genes_b, both, and the % expected if independent. Answers: one program in "
                 "the same cells, or separate? (For two different cell types use covary.)", self.coexpression, True),
            Tool("program_score", "lineage, genes", "mean % of the given genes expressed per cell, per sample. "
                 "Answers: is a whole program (gene set) up, not just one gene?", self.program_score, True),
            Tool("covary", "a, b", "Spearman correlation across samples between two per-sample features, written "
                 "'gene@lineage' (% expressing), 'score:<TF or PROGENy pathway>@lineage' (target-gene score), "
                 "'state:<state>' (% of all cells) or 'lineage:<lineage>' (% of all cells); b may be a list of up to "
                 "4. Also within each group of ≥4 samples, and a NULL: how many genes that are simply up in the "
                 "disease in b's lineage correlate with a as strongly. Answers: do two observations rise and fall "
                 "together more than any two disease-up features would?", self.covary, True),
            Tool("compute", "code, purpose=''", "LAST RESORT, when no other tool can answer: run a short Python "
                 "snippet on this study's cells, e.g. a lineage's share x its per-cell rate (which cells contribute "
                 "most of a gene), gating cells, a custom gene set, a per-state statistic, a small regression. Inside: "
                 "obs (DataFrame per cell: sample, group, state, lineage, umi, doublet_score), X (CSC counts, cells x "
                 "genes), genes (gene -> column), np, pd, stats (scipy.stats), col(gene), expr(gene) (log1p per "
                 "10k), mask(**column=value), pct(gene, mask), kb(name) (curated tables). Assign the answer to "
                 "`result` and print a short summary. Isolated: no network, no files, 180 s. Its output is a result "
                 "with an id like any other; numbers worked out by hand are not results.", self.compute, True),
            Tool("subset_contrast", "lineage, gene, sets=None, state=None", "WITHIN each sample: target-gene scores "
                 "(TF regulons, PROGENy pathways) in the lineage's cells that express a gene vs those that do not "
                 "(optionally within one state). Answers: what regulation is active in the very cells that make this "
                 "gene? Comparisons inside a sample cannot be explained by batch, lab or between-sample composition.",
                 self.subset_contrast, True),
            Tool("contamination_check", "gene, lineage", "is a gene's signal in a lineage real, or ambient RNA / "
                 "doublets from another cell type? Among the lineage's cells that express the gene vs those that "
                 "do not: how often they also express marker genes of each other lineage, and their doublet score. "
                 "Run it before building on an unexpected gene in an unexpected cell type.",
                 self.contamination_check, True),
            Tool("pathway_footprint", "lineage, sets=None", "target-gene scores per sample in ONE lineage for TF "
                 "regulons (CollecTRI, e.g. 'STAT3') or PROGENy pathways (e.g. 'JAK-STAT', 'NFkB', 'TGFb', 'TNFa'): "
                 "is the downstream response of a receptor's pathway present in the RECEIVING cell? Without sets, "
                 "lists the sets that best separate the disease. Discriminates candidate inducers that use different "
                 "pathways. Inferred from target transcripts, not measured activity.", self.pathway_footprint, True),
            Tool("incoming_signals", "lineage, top=12, include_matrix=False", "ligand→receptor signals received by a "
                 f"lineage, ranked by {self.contrast} change (LIANA, per-sample means; collagen/integrin-type matrix "
                 "ligands hidden unless include_matrix). Answers: what could be inducing this cell's program?",
                 self.incoming_signals),
            Tool("ligand_targets", "ligand, receptors=None", "receptors and target cells for a ligand (LIANA) and "
                 "which cells express the receptors. Pass receptors=[...] to check receptors the LIANA resource "
                 "lacks. Answers: who can respond to this mediator?", self.ligand_targets),
            Tool("fibroblast_programs", "", "lab table: per-sample fibroblast CXCL13+/CXCR2-ligand+ co-expression "
                 "and receptor-conditional CXCL13 (LTBR, IL1R1, OSMR) plus pathway signature scores.",
                 self.fibroblast_programs),
            Tool("find_findings", "query", "search all findings (not only those shown); every word must match a gene, "
                 "cell type, feature or group, e.g. '<gene> <cell type> <group>'.", self.find_findings),
            Tool("known_regulators", "gene", "curated DATABASE knowledge (CollecTRI, PROGENy), not this study: which "
                 "transcription factors and signalling pathways are known to induce or repress a gene. Answers: what "
                 "could make this cell express it? Then test the TFs, receptors and ligands in the data.",
                 self.known_regulators),
            Tool("receptor_paths", "tf, cell=None, max_steps=3", "curated DATABASE signalling routes (OmniPath) from "
                 "cell-surface receptors to a transcription factor, with each receptor's ligands; with cell=, each "
                 "receptor is marked expressed or not on that cell (expressed first). Answers: which signal, through "
                 "which receptor, could switch on this TF? upstream_screen does this for every TF of a gene at once.",
                 self.receptor_paths),
            Tool("known_pairs", "gene", "curated DATABASE ligand-receptor pairs (LIANA consensus, more complete than "
                 "the study's LIANA run): receptors of a ligand or ligands of a receptor.", self.known_pairs),
            Tool("literature_search", "query, n=8", "search PubMed abstracts (Europe PMC) with a few plain words — "
                 "gene symbols, cell types, pathways and disease names, never sample names (up to 12 papers; OR "
                 "between alternatives is allowed). Returns PMIDs and verbatim sentences about OTHER studies: use "
                 "them to propose mechanisms, then test them in this data. Search related diseases and other "
                 "tissues too, not only this disease.", self.literature_search),
            Tool("gene_diseases", "gene", "Open Targets: the diseases a gene is associated with across all of "
                 "medicine, with the evidence type behind each (genetic association, literature, expression, animal "
                 "model, known drug), and the drugs acting on it with their clinical stage. Answers: where else is "
                 "this mediator implicated, and is it already a drug target?", self.gene_diseases),
            Tool("gene_drugs", "gene", "Open Targets: the drugs and clinical candidates acting on one gene, each with its "
                 "highest clinical stage, its action (inhibitor, antagonist, agonist, modulator) and mechanism, and its "
                 "indications. Answers: what drug would block (or activate) this step, and how far has it got?",
                 self.gene_drugs, staged=True),
        ]

    # ---- cell-level tools -----------------------------------------------------------------------------------
    def _per_sample(self, mask: np.ndarray, values: np.ndarray, stat: str = "pct") -> dict[str, float | None]:
        c = self.cells
        out: dict[str, float | None] = {}
        for s in c.samples["sample"]:
            m = mask & (c.obs["sample"].to_numpy() == s)
            n = int(m.sum())
            if n < MIN_CELLS:
                out[s] = None
            elif stat == "pct":
                out[s] = float((values[m] > 0).mean() * 100)
            else:
                out[s] = float(values[m].mean())
        return out

    def _each_gene(self, fn, gene, **kw):
        """Run a single-gene tool for each gene of a list; a gene missing from the object is reported, not fatal.
        With several genes, each gene is shown compactly (its top lineages) so no gene is pushed out of view.
        Lists are capped: a long list is a sweep, not a question."""
        if len(gene) > MAX_GENES:
            raise ToolError(f"{len(gene)} genes asked; at most {MAX_GENES} per call. Pick the candidates you have a "
                            f"reason to expect and say why. To find candidate inducers of a gene in a cell, use "
                            f"upstream_screen(gene, cell), which filters curated routes by this data.")
        texts, data, cav = [], {}, []
        if len(gene) > 1 and fn == self.gene_profile and not kw.get("lineage"):
            kw = {**kw, "top": 3}
        for g in gene:
            try:
                t, d, c = fn(str(g), **kw)
                texts.append(t)
                data[str(g)] = d
                cav += c
            except ToolError as exc:
                texts.append(f"{g}: {exc}")
        if not data:
            raise ToolError("; ".join(texts))
        return "\n\n".join(texts), data, list(dict.fromkeys(cav))

    def gene_profile(self, gene, lineage=None, top: int | None = None):
        if isinstance(lineage, (list, tuple)):
            parts = [self.gene_profile(gene, str(lin), top) for lin in lineage[:4]]
            return "\n\n".join(p[0] for p in parts), {str(lin): p[1] for lin, p in zip(lineage, parts)}, parts[0][2]
        if isinstance(gene, (list, tuple)):
            return self._each_gene(self.gene_profile, gene, lineage=lineage)
        c = self.cells
        x = c.gene(gene)
        lins = [lineage] if lineage else sorted(c.obs.lineage.unique())
        rows, data = [], {}
        case = c.samples[c.samples.group == self.case]["sample"].tolist()
        for lin in lins:
            vals = self._per_sample(c.lineage_mask(lin), x)
            if not lineage and max([v for v in vals.values() if v is not None] or [0]) < 5:
                continue
            data[lin] = vals
            cv = [vals[s] for s in case if vals.get(s) is not None]
            rank = sum(cv) / len(cv) if cv else -1.0
            line = f"{lin}: {_fmt(vals, c.samples, self.order, '%')}" + (f" [{a}]" if (a := _agreement(vals, c.samples, self.order)) else "")
            rows.append((rank, line))
        shown = rows
        hidden = 0
        if top and len(rows) > top:  # compact view: lineages with the highest disease-sample level first
            shown = sorted(rows, key=lambda r: -r[0])[:top]
            hidden = len(rows) - top
        text = (f"% of cells expressing {gene}" + (f" (top {top} lineages by {self.case} level; {hidden} more "
                                                     f"lineages ≥5% not shown)" if hidden else "") + ":\n"
                + ("\n".join(r[1] for r in shown) or "expressed in <5% of cells of every lineage"))
        de = self.study.tables.get("de")
        if de is not None:
            d = de.df[de.df.gene.str.upper() == gene.upper()].dropna(subset=["padj"])
            if len(d):
                text += f"\nPipeline DE ({self.contrast}): " + "; ".join(
                    f"{r.broad_group} log2FC {r.log2FoldChange:.1f} padj {r.padj:.1g}" for r in d.itertuples())
        return text, data, [f"cells with < {MIN_CELLS} in a sample shown as —"]

    def compute(self, code, purpose=""):
        from mechanist.compute import MAX_CODE, MAX_OUT, ComputeSandbox
        if not isinstance(code, str) or not code.strip():
            raise ToolError("give the Python snippet as code")
        if len(code) > MAX_CODE:
            raise ToolError(f"keep the snippet under {MAX_CODE} characters")
        with self._lock:
            if self._sandbox is None:
                self._sandbox = ComputeSandbox(self.cells, str(self.kb.dir) if self.kb is not None else None)
            sandbox = self._sandbox
        out = sandbox.run(code)
        if not out.get("ok"):
            raise ToolError(f"the snippet failed: {out.get('error', '')[-700:]}")
        text = (out.get("stdout") or "").strip()[:MAX_OUT]
        res = json.dumps(out.get("result"))[:3000] if out.get("result") is not None else ""
        head = "compute" + (f" ({str(purpose)[:120]})" if purpose else "")
        summary = f"{head}:\n{text}" + (f"\nresult: {res}" if res else "")
        caveat = "computed by a model-written snippet; its code is in this call's args: check the code before relying on it"
        return summary, {"code": code, "stdout": text, "result": out.get("result")}, [caveat]

    def lineage_share(self, lineage):
        if isinstance(lineage, (list, tuple)):
            parts = [self.lineage_share(str(x)) for x in lineage[:MAX_GENES]]
            return "\n".join(p[0] for p in parts), {str(x): p[1] for x, p in zip(lineage, parts)}, parts[0][2]
        c = self.cells
        m = c.lineage_mask(lineage)
        vals = {}
        for s in c.samples["sample"]:
            sm = c.obs["sample"].to_numpy() == s
            vals[s] = float(m[sm].mean() * 100)
        return (f"{lineage} as % of all cells: {_fmt(vals, c.samples, self.order, '%')} [{_agreement(vals, c.samples, self.order)}]",
                {"pct": vals}, ["composition depends on dissociation and sampling; share is relative to other cells"])

    def state_composition(self, lineage: str):
        c = self.cells
        m = c.lineage_mask(lineage)
        sub = c.obs[m]
        tab = pd.crosstab(sub.state, sub["sample"])
        pct = tab / tab.sum() * 100
        rows, data = [], {}
        for st in pct.index:
            vals = {s: float(pct.loc[st, s]) if s in pct.columns and tab[s].sum() >= MIN_CELLS else None
                    for s in c.samples["sample"]}
            if max([v for v in vals.values() if v is not None] or [0]) < 2:
                continue
            data[st] = vals
            rows.append(f"{st}: {_fmt(vals, c.samples, self.order, '%')}" + (f" [{a}]" if (a := _agreement(vals, c.samples, self.order)) else ""))
        n = {s: int(tab[s].sum()) if s in tab.columns else 0 for s in c.samples["sample"]}
        return (f"States as % of {lineage} cells (cells per sample: {n}):\n" + "\n".join(rows), {"pct": data, "n": n},
                [])

    def gene_by_state(self, gene, lineage: str):
        if isinstance(gene, (list, tuple)):
            return self._each_gene(self.gene_by_state, gene, lineage=lineage)
        c = self.cells
        x = c.gene(gene)
        m = c.lineage_mask(lineage)
        rows, data = [], {}
        for st in sorted(c.obs.state[m].unique()):
            vals = self._per_sample(m & (c.obs.state.to_numpy() == st), x)
            if max([v for v in vals.values() if v is not None] or [0]) < 5:
                continue
            data[st] = vals
            rows.append(f"{st}: {_fmt(vals, c.samples, self.order, '%')}")
        return (f"% of cells expressing {gene} by {lineage} state:\n" + ("\n".join(rows) or "<5% in every state"),
                data, [])

    def state_markers(self, state: str, n: int = 20):
        c = self.cells
        sm = c.state_mask(state)
        st = c.obs.state[sm].iloc[0]
        lin = c.obs.lineage[sm].iloc[0]
        rest = (c.obs.lineage.to_numpy() == lin) & ~sm
        if rest.sum() < MIN_CELLS:
            rest = ~sm
        Xi, Xr = c.normalised(np.where(sm)[0]), c.normalised(np.where(rest)[0])  # rank on log1p CP10k, not counts
        mi, mr = np.asarray(Xi.mean(0)).ravel(), np.asarray(Xr.mean(0)).ravel()
        pi = np.asarray((Xi > 0).mean(0)).ravel()
        pr = np.asarray((Xr > 0).mean(0)).ravel()
        ok = (pi >= 0.25) & ~np.array([bool(EXCLUDE_GENE.match(g)) for g in c.gene_names])
        score = np.where(ok, mi - mr, -np.inf)
        top = np.argsort(-score)[: int(n)]
        genes = [f"{c.gene_names[i]} ({pi[i]:.0%} vs {pr[i]:.0%})" for i in top if np.isfinite(score[i])]
        by_sample = c.obs[sm].groupby("sample").size().to_dict()
        return ((f"{st} ({lin}) vs other {lin} cells — top genes (% expressing in state vs rest): {', '.join(genes)}.\n"
                f"Cells in this state per sample: {by_sample}"),
                {"genes": [c.gene_names[i] for i in top], "cells_per_sample": by_sample},
                ["pooled over samples: a state dominated by one sample reflects that sample"]
                if by_sample and max(by_sample.values()) > 0.6 * sum(by_sample.values()) else [])

    def _any(self, genes) -> np.ndarray:
        genes = [genes] if isinstance(genes, str) else list(genes)
        out = None
        for g in genes:
            x = self.cells.gene(g) > 0
            out = x if out is None else (out | x)
        return out

    def coexpression(self, lineage: str, genes_a, genes_b):
        c = self.cells
        m = c.lineage_mask(lineage)
        a, b = self._any(genes_a), self._any(genes_b)
        rows, data = [], {}
        for s in c.samples["sample"]:
            sm = m & (c.obs["sample"].to_numpy() == s)
            n = int(sm.sum())
            if n < MIN_CELLS:
                continue
            pa, pb, pab = a[sm].mean(), b[sm].mean(), (a & b)[sm].mean()
            exp = pa * pb
            ratio = pab / exp if exp > 0 else float("nan")
            g = c.samples.set_index("sample").loc[s, "group"]
            data[s] = {"group": g, "n": n, "pct_a": pa * 100, "pct_b": pb * 100, "pct_both": pab * 100,
                       "expected_both": exp * 100, "ratio": ratio}
            word = ("ENRICHED: the same cells tend to express both" if ratio >= 1.5 else
                    "DEPLETED: they tend to be in different cells" if ratio <= 0.67 else
                    "about chance: no evidence that the same cells express both")
            rows.append(f"{g} {s} (n={n}): A {pa:.1%}, B {pb:.1%}, both {pab:.1%} vs {exp:.1%} expected "
                        f"(x{ratio:.1f}; {word})")
        case = [d["ratio"] for s_, d in data.items() if d["group"] == self.case and d["ratio"] == d["ratio"]]
        verdict = ""
        if case:
            verdict = (f"\nIn the {self.case} samples the ratio is {', '.join(f'x{r:.1f}' for r in case)}: "
                       + ("co-expression above chance in every sample." if min(case) >= 1.5 else
                          "no more co-expression than chance in at least one sample; a ratio near 1 does NOT "
                          "support a link between A and B." if max(case) < 1.5 else
                          "the samples disagree."))
        return (f"Co-expression in {lineage}: A = any of {genes_a}, B = any of {genes_b}\n" + "\n".join(rows)
                + verdict, data,
                [("ratio = observed double-positive / expected if independent; with few double-positive cells it is "
                 "noisy")])

    def program_score(self, lineage: str, genes):
        c = self.cells
        genes = [genes] if isinstance(genes, str) else list(genes)
        present = [g for g in genes if g in c.genes or any(x.upper() == g.upper() for x in c.genes)]
        if not present:
            raise ToolError("none of the genes are in the object")
        m = c.lineage_mask(lineage)
        frac = np.mean([c.gene(g) > 0 for g in present], axis=0)
        vals = self._per_sample(m, frac, stat="mean")
        vals = {k: (v * 100 if v is not None else None) for k, v in vals.items()}
        return ((f"Mean % of program genes ({', '.join(present)}) expressed per {lineage} cell: "
                f"{_fmt(vals, c.samples, self.order, '%')} [{_agreement(vals, c.samples, self.order)}]"), {"pct": vals, "genes": present},
                [f"missing genes: {sorted(set(genes) - set(present))}"] if len(present) < len(genes) else [])

    def _feature(self, spec: str) -> dict[str, float | None]:
        c = self.cells
        spec = spec.strip()
        if spec.startswith("state:"):
            sm = c.state_mask(spec[6:].strip())
            return {s: float(sm[c.obs["sample"].to_numpy() == s].mean() * 100) for s in c.samples["sample"]}
        if spec.startswith("lineage:"):
            lm = c.lineage_mask(spec[8:].strip())
            return {s: float(lm[c.obs["sample"].to_numpy() == s].mean() * 100) for s in c.samples["sample"]}
        if spec.startswith("score:") and "@" in spec:
            name, lin = spec[6:].split("@", 1)
            return self._score_feature(name.strip(), lin.strip())
        if "@" in spec:
            gene, lin = spec.split("@", 1)
            return self._per_sample(c.lineage_mask(lin.strip()), c.gene(gene.strip()))
        raise ToolError(f"feature {spec!r} must be gene@lineage, score:<TF or pathway>@lineage, state:<name> or "
                        f"lineage:<name>")

    def _score_feature(self, name: str, lineage: str) -> dict[str, float | None]:
        from mechanist import findings as fnd
        if self.kb is None:
            raise ToolError("target-gene scores need the curated knowledge base")
        if self._scores is None:
            self._scores = fnd.target_scores(self._expr(), fnd.gene_sets(self.kb))
        e, ts = self._expr(), self._scores
        n = self._resolve_set(name, ts.names)
        if n is None:
            raise ToolError(f"no regulon or pathway {name!r} (TF symbol or PROGENy name)")
        li, ti = self._lineage_index(e, lineage), ts.names.index(n)
        return {e.samples[i]: (None if np.isnan(v := ts.scores[li, i, ti]) else float(v)) for i in range(len(e.samples))}

    def covary(self, a: str, b):
        if isinstance(b, (list, tuple)):
            parts = [self.covary(a, str(x)) for x in b[:4]]
            return "\n\n".join(p[0] for p in parts), {str(x): p[1] for x, p in zip(b, parts)}, parts[0][2]
        c = self.cells
        fa, fb = self._feature(a), self._feature(b)
        df = pd.DataFrame({"a": fa, "b": fb}).dropna()
        df["group"] = c.samples.set_index("sample").loc[df.index, "group"]
        rho_all = df.a.rank().corr(df.b.rank()) if len(df) >= 4 else float("nan")
        within = {g: (len(sub), sub.a.rank().corr(sub.b.rank())) for g in self.order
                  if len(sub := df[df.group == g]) >= 4}
        wtext = "; ".join(f"within {g} only (n={n}): {r:.2f}" for g, (n, r) in within.items()) or \
            "no group has ≥4 samples for a within-group value"
        null = self._covary_null(fa, b, rho_all)
        ntext, head = "", ""
        if null:
            ntext = (f"\nNULL: of {null['n']} genes up in {self.case} in {null['lineage']} (every {self.case} sample above "
                     f"every {self.ref} sample), {null['frac']:.0%} correlate with A at least this strongly.")
            pos_within = [r for r in (v for _, v in within.values()) if not np.isnan(r) and r > 0]
            head = ("VERDICT: NOT DISCRIMINATING — most genes that are simply up in the disease correlate with A this "
                    "well, so this correlation is no evidence for a specific link. Do not cite it as support.\n"
                    if null["frac"] >= 0.25 else
                    "VERDICT: the correlation is stronger than for most disease-up genes"
                    + (" and holds within a group." if pos_within else
                       ", but it is driven by the group contrast (no positive within-group value).") + "\n")
        return ((f"{head}Spearman across {len(df)} samples: {rho_all:.2f}; {wtext}.{ntext}\n"
                f"A={a}: {_fmt(fa, c.samples, self.order)}\nB={b}: {_fmt(fb, c.samples, self.order)}"),
                {"rho_all": rho_all, "rho_within": {g: r for g, (_, r) in within.items()}, "a": fa, "b": fb,
                 "null": null},
                [("across-sample correlation is driven by group differences; within-group values are the less "
                 "confounded ones, and n is tiny")])

    # ---- discriminating tests -------------------------------------------------------------------------------
    def _expr(self):
        if self._expression is None:
            from mechanist import findings as fnd
            self._expression = (self._expression_loader() if self._expression_loader else
                                fnd.load_expression(self.study, fnd.FindingsConfig(), cells=self.cells))
            if self._expression is None:
                raise ToolError("no per-sample expression for this study (raw counts or pipeline pseudobulk)")
        return self._expression

    def _lineage_index(self, e, lineage: str) -> int:
        hit = [i for i, x in enumerate(e.lineages) if x.lower() == lineage.lower()] or \
              [i for i, x in enumerate(e.lineages) if cell_family(x) == cell_family(lineage)]
        if not hit:
            raise ToolError(f"unknown lineage {lineage!r}{_did_you_mean(lineage, e.lineages)}; lineages: "
                            f"{', '.join(e.lineages)}")
        return hit[0]

    def _covary_null(self, fa: dict, b: str, rho: float, n_max: int = 400) -> dict | None:
        """Spearman of A against genes that separate case from reference in B's lineage: is rho special?"""
        if "@" not in b or not np.isfinite(rho):
            return None
        try:
            e = self._expr()
            li = self._lineage_index(e, b.split("@", 1)[1].strip())
        except ToolError:
            return None
        groups = {g: [e.samples.index(s) for s in ss if s in e.samples] for g, ss in self.study.dataset.groups.items()}
        ci, ri = groups.get(self.case, []), groups.get(self.ref, [])
        m = e.pct[li] if e.pct is not None else e.cpm[li]  # (sample, gene)
        if len(ci) < 2 or len(ri) < 2:
            return None
        up = np.where(np.nanmin(m[ci], axis=0) > np.nanmax(m[ri], axis=0))[0]
        if len(up) < 20:
            return None
        rng = np.random.default_rng(0)
        up = rng.choice(up, size=min(n_max, len(up)), replace=False)
        keep = [i for i, smp in enumerate(e.samples) if fa.get(smp) is not None and not np.isnan(m[i, up]).any()]
        if len(keep) < 4:
            return None
        av = pd.Series([fa[e.samples[i]] for i in keep]).rank().to_numpy()
        bm = pd.DataFrame(m[keep][:, up]).rank().to_numpy()
        av, bm = av - av.mean(), bm - bm.mean(axis=0)
        r = (av @ bm) / np.sqrt((av ** 2).sum() * (bm ** 2).sum(axis=0))
        return {"n": len(up), "frac": float(np.mean(r >= rho - 1e-9)), "lineage": e.lineages[li]}

    def _state_shift(self, lineage: str, min_points: float = 25.0) -> str:
        """Warn when one state's share of the lineage differs strongly between the disease and the other groups:
        pseudobulk scores then compare different mixtures of cells, not the same cells responding."""
        c = self.cells
        sub = c.obs[c.lineage_mask(lineage)]
        if sub.state.nunique() < 2:
            return ""
        tab = pd.crosstab(sub.state, sub["sample"])
        pct = tab / tab.sum() * 100
        case = [x for x in c.samples[c.samples.group == self.case]["sample"] if x in pct.columns]
        rest = [x for x in pct.columns if x not in case]
        if not case or not rest:
            return ""
        diff = pct[case].mean(axis=1) - pct[rest].mean(axis=1)
        st = diff.abs().idxmax()
        if abs(diff[st]) < min_points:
            return ""
        return (f"\nCOMPOSITION WARNING: '{st}' is {pct.loc[st, case].mean():.0f}% of {lineage} cells in {self.case} vs "
                f"{pct.loc[st, rest].mean():.0f}% in the other samples. These scores compare different mixtures of "
                f"cells: a state shift alone moves them. Before reading a score as signalling, check the program within "
                f"one state (gene_by_state, state_markers) or ask what made that state expand.")

    def _lineage_markers(self, k: int = 8) -> dict[str, list[int]]:
        """Per lineage, k genes detected in many of its cells and few cells of any other lineage (gene indices)."""
        if getattr(self, "_markers", None) is None:
            from scipy import sparse
            c = self.cells
            lins = sorted(c.obs.lineage.unique())
            code = pd.Categorical(c.obs.lineage, categories=lins).codes
            M = sparse.csr_matrix((np.ones(len(code)), (code, np.arange(len(code)))), shape=(len(lins), len(code)))
            B = c.X.copy()
            B.data = (B.data > 0).astype(float)
            frac = np.asarray((M @ B).todense()) / np.bincount(code, minlength=len(lins))[:, None]
            ok = ~np.array([bool(EXCLUDE_GENE.match(g)) for g in c.gene_names])
            out = {}
            for i, lin in enumerate(lins):
                other = np.delete(frac, i, axis=0).max(axis=0)
                score = np.where(ok & (frac[i] >= 0.3), frac[i] - other, -np.inf)
                top = [j for j in np.argsort(-score)[:k] if np.isfinite(score[j]) and score[j] > 0.2]
                out[lin] = top
            self._markers = out
        return self._markers

    def subset_contrast(self, lineage: str, gene: str, sets=None, state: str | None = None, min_cells: int = 10):
        """Per sample: pseudobulk of gene+ and gene- cells (same lineage, optionally same state), difference of
        log2 CPM, scored on each gene set (signed weights, the gene itself removed), in units of the SD of that
        sample's per-gene differences. Ranked within the sample so a score can be read against all other sets."""
        from mechanist import findings as fnd
        c = self.cells
        if not c.is_counts:
            raise ToolError("subset_contrast needs raw counts in the single-cell object")
        if self.kb is None:
            raise ToolError("subset_contrast needs the curated knowledge base")
        x = c.gene(gene) > 0
        m = c.lineage_mask(lineage)
        if state:
            m = m & c.state_mask(state)
        gsets = fnd.gene_sets(self.kb)
        gidx = c.genes
        g_up = gene.upper()
        prepared = {}
        for name, (targets, w, _src) in gsets.items():
            pairs = [(gidx[t], wt) for t, wt in zip(targets, w) if t in gidx and t.upper() != g_up]
            if len(pairs) >= 10:
                prepared[name] = (np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs], dtype=float))
        smp = c.obs["sample"].to_numpy()
        per, lines = {}, []
        for s_ in c.samples["sample"]:
            pos, neg = m & (smp == s_) & x, m & (smp == s_) & ~x
            if pos.sum() < min_cells or neg.sum() < min_cells:
                continue
            cp = np.asarray(c.X[np.where(pos)[0]].sum(0)).ravel() / c.libsize[pos].sum() * 1e6
            cn = np.asarray(c.X[np.where(neg)[0]].sum(0)).ravel() / c.libsize[neg].sum() * 1e6
            expressed = (cp + cn) / 2 >= 1.0
            d = np.log2(cp + 1) - np.log2(cn + 1)
            sd = float(d[expressed].std()) or 1.0
            sc = {}
            for name, (idx, w) in prepared.items():
                keep = expressed[idx]
                if keep.sum() < 10:
                    continue
                ww = w[keep]
                sc[name] = float((d[idx[keep]] * ww).sum() / np.sqrt((ww ** 2).sum()) / sd)
            order = sorted(sc, key=lambda k: -sc[k])
            per[s_] = {"n_pos": int(pos.sum()), "n_neg": int(neg.sum()), "scores": sc,
                       "rank": {k: i + 1 for i, k in enumerate(order)}, "n_sets": len(order)}
        if not per:
            raise ToolError(f"no sample has {min_cells} {gene}+ and {min_cells} {gene}- {lineage} cells")
        groups = c.samples.set_index("sample")["group"]
        case = [s_ for s_ in per if groups[s_] == self.case]
        others = [s_ for s_ in per if groups[s_] != self.case]

        def row(name):
            def cell(s_):
                d = per[s_]
                v = d["scores"].get(name)
                return "—" if v is None else f"{v:+.2f} (rank {d['rank'][name]}/{d['n_sets']})"
            return (f"- {name}: " + "; ".join(f"{s_} {cell(s_)}" for s_ in case)
                    + (" | other groups: " + "; ".join(f"{s_} {cell(s_)}" for s_ in others[:5]) if others else ""))
        if sets:
            asked = [sets] if isinstance(sets, str) else list(sets)
            names = [n for a in asked[:8] if (n := self._resolve_set(str(a), list(prepared)))]
            missing = [a for a in asked[:8] if not self._resolve_set(str(a), list(prepared))]
        else:
            ref = case or list(per)
            mean = {n: np.mean([per[s_]["scores"].get(n, 0.0) for s_ in ref]) for n in prepared}
            names = sorted(mean, key=lambda n: -mean[n])[:8] + sorted(mean, key=lambda n: mean[n])[:3]
            missing = []
        lines = [row(n) for n in names] + [f"- {a}: no such regulon or pathway" for a in missing]
        where = f"{lineage}" + (f" state '{state}'" if state else "")
        head = (f"Within-sample contrast in {where}: {gene}+ vs {gene}- cells, target-gene scores in SD units of that "
                f"sample's gene-wise differences (positive = higher in {gene}+ cells; {gene} removed from every set). "
                f"Cells per sample (+/-): " + ", ".join(f"{s_} {per[s_]['n_pos']}/{per[s_]['n_neg']}" for s_ in per)
                + (". No set asked: the 8 highest and 3 lowest in the disease samples." if not sets else "."))
        return (head + "\n" + "\n".join(lines), {"lineage": lineage, "gene": gene, "state": state, "samples": per},
                ["within one sample, so batch, lab and between-sample composition cannot explain a difference",
                 "states within the lineage can: pass state= to hold the state fixed",
                 "inferred from target transcripts, not measured activity; small subsets are noisy"])

    def contamination_check(self, gene: str, lineage: str):
        """Can another cell type's RNA (doublets, co-captured RNA) EXPLAIN a gene's signal in a lineage? Foreign RNA
        explains the signal only if the gene is still much rarer once cells carrying another lineage's markers are set
        aside, and the foreign lineage itself expresses the gene. Markers alone do not make a signal artefactual."""
        c = self.cells
        x = c.gene(gene) > 0
        lm = c.lineage_mask(lineage)
        own = {str(v) for v in c.obs.lineage[lm].unique()}
        markers = {k: v for k, v in self._lineage_markers().items() if k not in own and v}
        smp = c.obs["sample"].to_numpy()
        lin_all = c.obs.lineage.to_numpy()
        has_db = "doublet_score" in c.obs
        hits = {lin: np.asarray((c.X[:, idx] > 0).sum(1)).ravel() >= 2 for lin, idx in markers.items()}
        foreign = np.zeros(len(x), dtype=bool)
        for h in hits.values():
            foreign |= h
        lines, data, present, explains, minors = [], {}, [], [], []
        case_samples = set(c.samples[c.samples.group == self.case]["sample"])
        for s_ in c.samples["sample"]:
            ls = lm & (smp == s_)
            pos, neg = ls & x, ls & ~x
            if pos.sum() < 10 or neg.sum() < 10:
                continue
            g_all = float(x[ls].mean())
            clean = ls & ~foreign
            g_clean = float(x[clean].mean()) if clean.sum() >= 10 else float("nan")
            rates = {lin: (float(h[pos].mean()), float(h[neg].mean())) for lin, h in hits.items()}
            top = sorted(rates.items(), key=lambda kv: -(kv[1][0] - kv[1][1]))[:3]
            db = (float(np.nanmean(c.obs.doublet_score.to_numpy()[pos])),
                  float(np.nanmean(c.obs.doublet_score.to_numpy()[neg]))) if has_db else None
            g = c.samples.set_index("sample").loc[s_, "group"]
            src = {}
            for lin, (a, b) in top:
                if a >= 0.10 and a >= 2 * max(b, 0.01):
                    ym = (lin_all == lin) & (smp == s_)
                    src[lin] = float(x[ym].mean()) if ym.sum() >= 10 else float("nan")
            drop = (g_all - g_clean) / g_all if g_all > 0 and not np.isnan(g_clean) else 0.0
            makes = any(not np.isnan(v) and v >= 0.5 * g_all for v in src.values())
            can = bool(src) and drop >= 0.3 and makes
            minor = bool(src) and 0.1 <= drop < 0.3 and makes
            data[s_] = {"group": g, "pct_all": g_all * 100, "pct_clean": g_clean * 100, "markers": rates,
                        "foreign_sources": src, "doublet": db, "explains": can, "minor": minor, "drop": drop}
            if src:
                note = (f"{s_}: {gene}+ {lineage} cells carry " + ", ".join(f"{lin} markers {rates[lin][0]:.0%} vs "
                        f"{rates[lin][1]:.0%} (and {lin} cells express {gene} in {v:.0%})" for lin, v in src.items()))
                note += f" ({gene} {drop:.0%} rarer without those cells)"
                (explains if can else minors if minor else present).append(note)
            lines.append(f"{g} {s_}: {gene} in {g_all:.0%} of all {lineage} cells and {g_clean:.0%} of those without "
                         f"another lineage's markers; markers in {gene}+/- cells: "
                         + ", ".join(f"{lin} {a:.0%}/{b:.0%}" for lin, (a, b) in top)
                         + (f"; doublet score {db[0]:.2f}/{db[1]:.2f}" if db else ""))
        if not lines:
            raise ToolError(f"too few {gene}+ or {gene}- {lineage} cells per sample (need 10 each)")
        case_explains = [e for e in explains if e.split(":")[0] in case_samples]
        case_present = [e for e in present if e.split(":")[0] in case_samples]
        case_minor = [e for e in minors if e.split(":")[0] in case_samples]
        n_case = sum(1 for s_ in data if s_ in case_samples)
        hit = {e.split(":")[0] for e in case_explains}
        if case_explains:
            verdict = (f"VERDICT: CONTAMINATION CAN EXPLAIN PART OF THE SIGNAL in {len(hit)} of {n_case} {self.case} "
                       f"samples: {'; '.join(case_explains[:3])}. Without those cells {gene} is much rarer in {lineage}, "
                       f"and the foreign cell type expresses {gene} itself. Treat {gene} in {lineage} as partly another "
                       f"cell type's RNA there.")
        elif case_minor:
            verdict = (f"VERDICT: A MINOR PART of the {gene} signal may be another cell type's RNA: {'; '.join(case_minor[:3])}. "
                       f"Most of it remains in {lineage} cells without foreign markers, so {gene} is mainly {lineage} "
                       f"expression; quantities are slightly inflated.")
        elif case_present:
            verdict = (f"VERDICT: foreign markers present but they CANNOT EXPLAIN the {gene} signal: {gene} stays about "
                       f"as common in {lineage} cells without another lineage's markers, or the foreign cell type does "
                       f"not express {gene}. ({'; '.join(case_present[:2])}.) The signal is {lineage} expression.")
        else:
            verdict = f"VERDICT: no sign that another cell type's RNA explains {gene} in {lineage} in {self.case}."
        case_lines = [ln for ln in lines if ln.startswith(self.case + " ")]
        return (f"Contamination check: {gene} in {lineage}.\n{verdict}\n" + "\n".join(case_lines)
                + f"\n(other samples: {len(lines) - len(case_lines)} checked)\nMarkers used: "
                + "; ".join(f"{lin}: {', '.join(str(c.gene_names[j]) for j in idx[:4])}" for lin, idx in markers.items()),
                {"samples": data, "flags": explains + minors + present, "case_flags": case_explains, "case_minor": case_minor,
                 "case_markers_only": case_present},
                [("ambient RNA spread evenly over all cells is not detected by this comparison; it detects co-captured "
                  "RNA and doublets"), "doublet scores come from the pipeline"])

    def _resolve_set(self, name: str, names: list[str]) -> str | None:
        name = name.translate(str.maketrans({"α": "a", "β": "b", "κ": "k", "‑": "-"}))  # TGF-β, NF-κB, TNF-α
        m = re.match(r"^(PROGENY|TF)[\s_:-]+", name.strip(), flags=re.IGNORECASE)
        source = m.group(1).upper() if m else None
        name = name.strip()[m.end():] if m else name.strip()
        name = name.replace("_", "-")
        name = re.sub(r"[\s_-]+targets$", "", name, flags=re.IGNORECASE)
        name = {"TGF-B": "TGFb", "NF-KB": "NFkB", "TNF-A": "TNFa"}.get(name.upper().replace(" ", ""), name)
        key = _norm(name)
        order = ((f"PROGENy {name} targets", name, f"TF {name} targets") if source == "PROGENY" else
                 (f"TF {name} targets", name, f"PROGENy {name} targets"))
        for cand in order:
            hit = [n for n in names if _norm(n) == _norm(cand)]
            if hit:
                return hit[0]
        hit = [n for n in names if key and key == _norm(n.replace("TF ", "").replace("PROGENy ", "")
                                                          .replace(" targets", ""))]
        return hit[0] if hit else None

    def pathway_footprint(self, lineage: str, sets=None):
        from mechanist import findings as fnd
        if self._scores is None:
            self._scores = fnd.target_scores(self._expr(), fnd.gene_sets(self.kb))
        e, ts = self._expr(), self._scores
        li = self._lineage_index(e, lineage)
        samples = self.cells.samples if self.has_cells else pd.DataFrame(
            [{"group": g, "sample": x} for g, ss in self.study.dataset.groups.items() for x in ss])
        groups = {g: [e.samples.index(x) for x in ss if x in e.samples] for g, ss in self.study.dataset.groups.items()}
        ci = groups.get(self.case, [])

        def vals(ti):
            return {e.samples[i]: (None if np.isnan(v := ts.scores[li, i, ti]) else float(v)) for i in range(len(e.samples))}

        def separates(ti):
            cv = ts.scores[li, ci, ti]
            if len(cv) < 2 or np.isnan(cv).any():
                return [], 0
            up = [g for g, ix in groups.items() if g != self.case and len(ix) >= 2
                  and not np.isnan(rv := ts.scores[li, ix, ti]).all() and cv.min() > np.nanmax(rv)]
            dn = [g for g, ix in groups.items() if g != self.case and len(ix) >= 2
                  and not np.isnan(rv := ts.scores[li, ix, ti]).all() and cv.max() < np.nanmin(rv)]
            return (up, 1) if up else (dn, -1) if dn else ([], 0)

        def drivers(ti, k=4):
            d = ts.drivers.get((li, ti))
            if d is None or li not in ts.z:
                return ""
            idx, w = d
            contrib = np.nanmean(ts.z[li][ci][:, idx], axis=0) * w
            top = np.argsort(-contrib)[:k]
            return ", ".join(str(e.genes[idx[j]]) for j in top if contrib[j] > 0)
        lines, data = [], {}
        if sets:
            asked = [sets] if isinstance(sets, str) else list(sets)
            asked = [p.strip() for x in asked for p in re.split(r"[/,;]| and ", str(x)) if p.strip()]
            progeny = sorted({n[8:-8] for n in ts.names if n.startswith("PROGENy ")})
            for name in asked[:8]:
                n = self._resolve_set(str(name), ts.names)
                if n is None:
                    lines.append(f"- {name}: no such regulon or pathway. Use a TF symbol (its CollecTRI regulon) or "
                                 f"a PROGENy pathway: {', '.join(progeny)}")
                    continue
                ti = ts.names.index(n)
                v = vals(ti)
                if all(x is None for x in v.values()):
                    lines.append(f"- {n}: fewer than the minimum targets expressed in {e.lineages[li]}")
                    continue
                sep, sign = separates(ti)
                data[n] = {"scores": v, "separates_from": sep, "direction": sign}
                lines.append(f"- {n} ({ts.n_targets[li, ti]} targets): {_fmt(v, samples, self.order, nd=2)}"
                             + (f" [{self.case} {'above' if sign > 0 else 'below'} every {', '.join(sep)} sample]"
                                if sep else " [no separation]")
                             + (f"; drivers in {self.case}: {drivers(ti)}" if sign > 0 else ""))
        sep_all = [(ti, *separates(ti)) for ti in range(len(ts.names)) if not np.isnan(ts.scores[li, :, ti]).all()]
        n_up = sum(1 for _, g, s_ in sep_all if s_ > 0)
        n_dn = sum(1 for _, g, s_ in sep_all if s_ < 0)
        if not sets:
            best = sorted([x for x in sep_all if x[2] > 0], key=lambda x: (-len(x[1]),
                          -float(np.nanmean(ts.scores[li, ci, x[0]]))))[:10]
            for ti, sep, _ in best:
                v = vals(ti)
                data[ts.names[ti]] = {"scores": v, "separates_from": sep, "direction": 1}
                lines.append(f"- {ts.names[ti]}: {_fmt(v, samples, self.order, nd=2)} [{self.case} above every "
                             f"{', '.join(sep)} sample]; drivers: {drivers(ti)}")
        shift = self._state_shift(e.lineages[li]) if self.has_cells else ""
        head = (f"Target-gene scores in {e.lineages[li]} (z-scored across samples; CollecTRI regulons, PROGENy "
                f"pathways; inferred from target transcripts, not activity). Context: {n_up} of {len(sep_all)} sets are "
                f"higher and {n_dn} lower in every {self.case} sample than in every sample of some other group, so a "
                f"single separating set is common; compare the sets of competing candidates.")
        return head + "\n" + ("\n".join(lines) or "none") + shift, {"lineage": e.lineages[li], "sets": data,
                                                               "n_up": n_up, "n_down": n_dn}, [
            "scores are relative to the other samples of this lineage; a shift in cell states also moves them",
            "overlapping regulons share targets: check the drivers"]

    # ---- table tools ----------------------------------------------------------------------------------------
    def _comm(self) -> pd.DataFrame:
        """Communication summary rows of the primary contrast (all rows when the primary group has none)."""
        t = self.study.tables.get("communication")
        if t is None:
            raise ToolError("no communication table for this study")
        df = t.df
        if "condition" in df and (df.condition == self.case).any():
            df = df[df.condition == self.case]
            if "reference" in df and (df.reference == self.ref).any():
                df = df[df.reference == self.ref]
        return df

    def _comm_groups(self, d: pd.DataFrame) -> tuple[str, str]:
        case = str(d.condition.iloc[0]) if "condition" in d and len(d) else self.case
        ref = str(d.reference.iloc[0]) if "reference" in d and len(d) else self.ref
        return case, ref

    def incoming_signals(self, lineage: str, top: int = 12, include_matrix: bool = False):
        df = self._comm()
        fam = cell_family(lineage)
        d = df[df.target.map(cell_family) == fam].copy()
        if not include_matrix:
            d = d[~d.ligand.astype(str).str.match(MATRIX_LIGAND)]
        if d.empty:
            raise ToolError(f"no signals to {lineage}; targets: {', '.join(sorted(df.target.unique()))}")
        d = d.sort_values("delta_mean", ascending=False).head(int(top))
        case, ref = self._comm_groups(d)
        rows = [f"{r.source}→{r.target} {r.ligand}->{r.receptor}: {case} {r.mean_magnitude:.2f} "
                f"(in {r.detection_fraction:.0%} of {case} samples) vs {ref} {r.ref_mean_magnitude:.2f}"
                for r in d.itertuples()]
        return (f"Signals received by {lineage}, largest {case}-vs-{ref} increase (LIANA magnitude, per-sample mean):\n"
                + "\n".join(rows), {"rows": d[["source", "target", "ligand", "receptor", "delta_mean"]].to_dict("records")},
                ["LIANA infers signals from ligand and receptor transcripts; no change reached significance",
                 "the ranking omits pairs missing from the LIANA resource; test known inducers directly",
                 f"communication tables compare {case} vs {ref}; other groups are not in this ranking"])

    def ligand_targets(self, ligand, receptors=None):
        if isinstance(ligand, (list, tuple)):
            parts = [self.ligand_targets(str(x), receptors) for x in ligand]
            return "\n\n".join(p[0] for p in parts), {str(x): p[1] for x, p in zip(ligand, parts)}, parts[0][2]
        df = self._comm()
        d = df[df.ligand.str.upper() == ligand.upper()].sort_values("delta_mean", ascending=False).head(15)
        case, ref = self._comm_groups(d)
        rows = [f"{r.source}→{r.target} via {r.receptor}: {case} {r.mean_magnitude:.2f} vs {ref} "
                f"{r.ref_mean_magnitude:.2f}" for r in d.itertuples()]
        liana = sorted({g for r in d.receptor for g in str(r).split("_")})
        curated: list[str] = []
        if self.kb is not None:
            try:
                curated = sorted({g for r in self.kb.pairs(ligand)[1]["receptors"] for g in str(r).split("_")})
            except KnowledgeError:
                curated = []
        asked = [receptors] if isinstance(receptors, str) else list(receptors or [])
        if d.empty and not asked and not curated:
            raise ToolError(f"{ligand} has no ligand-receptor pairs in the communication table; pass receptors=[...]")
        text = ""
        if self.has_cells:
            try:
                src, _, _ = self.gene_profile(ligand, top=4)
                text = "WHO EXPRESSES THE LIGAND — " + src.split("\nPipeline")[0] + "\n\n"
            except ToolError:
                text = f"{ligand} is not in the single-cell object.\n\n"
        if curated:
            text += f"Known receptors of {ligand} (curated consensus resource): {', '.join(curated)}.\n\n"
        text += (f"{ligand} signalling in this study's LIANA run (a narrower resource):\n"
                 + ("\n".join(rows) or "no pairs scored"))
        if not curated:
            text += ("\nThe LIANA list is NOT exhaustive: a receptor missing here may still be expressed. Check known "
                     "receptors directly (receptors=[...] or gene_profile).")
        missing = [r for r in asked if r.upper() not in {x.upper() for x in liana + curated}]
        if missing:
            text += f"\nNot paired with {ligand} in the LIANA resource: {', '.join(missing)}"
        data = {"rows": d[["source", "target", "receptor", "delta_mean"]].to_dict("records") if len(d) else []}
        if self.has_cells:
            order = asked + [r for r in curated if r not in asked] + [r for r in liana if r not in asked + curated]
            for rec in order[:6]:
                try:
                    s, _, _ = self.gene_profile(rec, top=4)
                    text += f"\n\nRECEPTOR {rec} — " + s.split("\nPipeline")[0]
                except ToolError:
                    text += f"\n\nRECEPTOR {rec} — not in the single-cell object"
        return text, data, ["LIANA pairs are transcript-based predictions"]

    def fibroblast_programs(self):
        t = self.study.tables.get("fib_programs")
        if t is None:
            raise ToolError("no fibroblast_programs table")
        df = t.df
        cols = ["CXCL13+_%", "CXCR2-ligand+_%", "of_those_also_CXCR2-ligand+_%", "expected_if_independent_%",
                "LTBR+_%", "IL1R1+_%", "OSMR+_%", "CXCL13+_%_in_LTBR+", "CXCL13+_%_in_LTBR-", "CXCL13+_%_in_IL1R1+",
                "CXCL13+_%_in_IL1R1-", "CXCL13+_%_in_OSMR+", "CXCL13+_%_in_OSMR-"]
        cols = [c for c in cols if c in df.columns]
        lines = [f"{r['group']} {r['sample']}: " + ", ".join(f"{c} {r[c]:.1f}" for c in cols if pd.notna(r[c]))
                 for _, r in df.iterrows()]
        sig = self.study.tables.get("signatures")
        if sig is not None:
            s = sig.df[sig.df.state == "all fibroblasts"]
            sc = [c for c in s.columns if c not in {"group", "sample", "state", "cells"}]
            lines.append("Signature scores (all fibroblasts): " + "; ".join(
                f"{r['sample']} " + ", ".join(f"{c} {r[c]:.1f}" for c in sc) for _, r in s.iterrows()))
        return ("Fibroblast programs (lab table):\n" + "\n".join(lines), {"rows": df.to_dict("records")},
                ["hand-made table from the earlier PG investigation (circular with its hypotheses)",
                 "CXCR2 ligands as defined by the lab table"])

    # ---- outside knowledge -----------------------------------------------------------------------------------
    def known_regulators(self, gene):
        if isinstance(gene, (list, tuple)):
            parts = [self.known_regulators(str(g)) for g in gene[:MAX_GENES]]
            return "\n\n".join(p[0] for p in parts), {str(g): p[1] for g, p in zip(gene, parts)}, parts[0][2]
        try:
            text, data = self.kb.regulators(gene)
        except KnowledgeError as exc:
            raise ToolError(str(exc)) from None
        return text, data, ["curated database knowledge from other systems, not evidence from this study"]

    def receptor_paths(self, tf: str, cell: str | None = None, max_steps: int = 3):
        try:
            text, data = self.kb.receptor_paths(tf, max_steps, limit=30 if cell else 12)
        except KnowledgeError as exc:
            raise ToolError(str(exc)) from None
        cav = ["curated signalling knowledge from other systems, not evidence from this study"]
        if not cell or not self.has_cells:
            return text, data, cav
        c, groups = self.cells, self._groups()
        cm = c.lineage_mask(cell)
        on, off, lines = [], [], []
        for route in data["routes"]:
            vals = self._safe_per_sample(cm, route["receptor"])
            ok = vals is not None and receptor_ok(vals, groups, self.case, self.ref, MIN_PCT)
            route["on_cell"] = ok
            (on if ok else off).append(route)
        for route in on[:10]:
            vals = self._safe_per_sample(cm, route["receptor"])
            lines.append(f"- {' → '.join(route['path'])} ({route['sign']}) — {route['receptor']} on {cell}: "
                         f"{self._group_means(vals, groups)}; ligands: {', '.join(route['ligands'][:8]) or '—'}")
        data["routes"] = on + off
        text = (f"Signalling routes from receptors to {tf} (OmniPath, curated; not this study), restricted to "
                f"receptors expressed on {cell} (every {self.case} sample ≥{MIN_PCT:.0f}%, {self.case} mean "
                f"above {self.ref}):\n" + ("\n".join(lines) or "none")
                + f"\nNot expressed on {cell} (ruled out): "
                + (", ".join(dict.fromkeys(r["receptor"] for r in off)) or "none"))
        return text, data, cav

    # ---- the upstream screen ---------------------------------------------------------------------------------
    def _groups(self) -> dict[str, list[str]]:
        c = self.cells
        return {g: c.samples[c.samples.group == g]["sample"].tolist() for g in self.order
                if (c.samples.group == g).any()}

    def _safe_per_sample(self, mask, gene):
        try:
            return self._per_sample(mask, self.cells.gene(gene))
        except ToolError:
            return None

    def _group_means(self, vals, groups) -> str:
        parts = []
        for g, ss in groups.items():
            xs = [vals[s] for s in ss if vals.get(s) is not None]
            if xs:
                parts.append(f"{g} " + (", ".join(f"{x:.0f}" for x in xs) + "%" if g == self.case
                                        else f"mean {sum(xs) / len(xs):.0f}%"))
        return " | ".join(parts)

    def _pct_table(self, x: np.ndarray) -> dict[str, dict[str, float | None]]:
        """% of cells expressing, per lineage and sample, for one gene vector (one pass over the cells)."""
        c = self.cells
        if self._idx is None:
            key = c.obs.lineage.astype(str) + "\t" + c.obs["sample"].astype(str)
            codes, uniq = pd.factorize(key)
            self._idx = (codes, [u.split("\t") for u in uniq], np.bincount(codes))
        codes, keys, n = self._idx
        pos = np.bincount(codes, weights=(x > 0).astype(float), minlength=len(keys))
        out: dict[str, dict[str, float | None]] = {}
        for (lin, smp), k, p in zip(keys, n, pos):
            out.setdefault(lin, {})[smp] = float(p / k * 100) if k >= MIN_CELLS else None
        return out

    def _detected(self, vals, groups) -> bool:
        """Detected (≥1%) in every disease sample: too low to count as expressed, too high to call absent."""
        cv = [vals[s] for s in groups[self.case] if vals.get(s) is not None]
        return len(cv) >= 2 and min(cv) >= 1.0

    def _share(self, lineage: str) -> dict[str, float]:
        c = self.cells
        m = c.obs.lineage.to_numpy() == lineage
        return {s: float(m[c.obs["sample"].to_numpy() == s].mean() * 100) for s in c.samples["sample"]}

    def upstream_screen(self, gene: str, cell: str, top: int = 8):
        """gene -> activating TFs expressed on the cell -> receptors with an activating route to them that are on the
        cell -> ligands that differ in the disease, with the sending lineage. Deterministic; the agent judges."""
        c, groups = self.cells, self._groups()
        case, ref = self.case, self.ref
        cm = c.lineage_mask(cell)
        own = self._safe_per_sample(cm, gene)
        if own is None:
            raise ToolError(f"{gene} is not in the single-cell object")
        cv = [own[x] for x in groups.get(case, []) if own.get(x) is not None]
        if len(cv) < 2 or min(cv) < MIN_PCT or not ligand_differs(own, groups, case):
            # premise check: do not explain an induction the data does not show
            return ((f"PREMISE FAILS: {gene} is not elevated in {cell} in this data, so there is no induction to "
                     f"explain. % of {cell} cells expressing {gene}: {_fmt(own, c.samples, self.order, '%')}. Every "
                     f"{case} sample must be ≥{MIN_PCT:.0f}% and above every sample of another group. Answer the "
                     f"question with this, then ask which cell DOES make {gene} (gene_profile without a lineage)."),
                    {"gene": gene, "cell": cell, "premise": False, "pct": own}, [])
        try:
            _, reg = self.kb.regulators(gene)
        except KnowledgeError as exc:
            raise ToolError(str(exc)) from None
        tfs = reg["activating_tfs"]
        if not tfs:
            raise ToolError(f"no curated activating TFs for {gene}; reason from the gene's biology instead")
        tf_vals = {t: self._safe_per_sample(cm, t) for t in tfs}
        tf_on = [t for t in tfs if tf_vals[t] is not None and receptor_ok(tf_vals[t], groups, case, None, MIN_PCT)]
        tf_low = [t for t in tfs if t not in tf_on and tf_vals[t] is not None and self._detected(tf_vals[t], groups)]
        tf_off = [t for t in tfs if t not in tf_on and t not in tf_low]
        receptors: dict[str, dict] = {}
        for tf in tf_on:
            try:
                _, rd = self.kb.receptor_paths(tf, 3, limit=40)
            except KnowledgeError:
                continue
            for route in rd["routes"]:
                if route["sign"] != "activating":
                    continue
                r = receptors.setdefault(route["receptor"], {"path": route["path"], "ligands": set(), "tfs": set()})
                r["ligands"] |= {p.upper() for lig in route["ligands"] for p in str(lig).split("_")}
                r["tfs"].add(tf)
        rec_on, rec_low, rec_off, rec_vals = [], [], [], {}
        for rec in receptors:
            v = self._safe_per_sample(cm, rec)
            if v is not None and receptor_ok(v, groups, case, ref, MIN_PCT):
                rec_on.append(rec)
                rec_vals[rec] = v
            elif v is not None and self._detected(v, groups):
                rec_low.append(rec)
                rec_vals[rec] = v
            else:
                rec_off.append(rec)
        lig_recs: dict[str, list[str]] = {}
        for rec in rec_on + rec_low:  # low receptors carry candidates too (low-RNA cells), ranked below
            for lig in receptors[rec]["ligands"]:
                lig_recs.setdefault(lig, []).append(rec)
        cands, flat, missing_genes = [], [], []
        for lig, recs in sorted(lig_recs.items()):
            try:
                x = c.gene(lig)
            except ToolError:
                missing_genes.append(lig)
                continue
            best = None
            for lin, vals in self._pct_table(x).items():
                cv = [vals[s] for s in groups[case] if vals.get(s) is not None]
                if len(cv) < 2 or sum(cv) / len(cv) < MIN_PCT or not ligand_differs(vals, groups, case):
                    continue
                sep = [g for g, ss in groups.items() if g != case
                       and len(rv := [vals[s] for s in ss if vals.get(s) is not None]) >= 2 and min(cv) > max(rv)]
                others = [vals[s] for g, ss in groups.items() if g != case for s in ss if vals.get(s) is not None]
                score = (len(sep), sum(cv) / len(cv) - (sum(others) / len(others) if others else 0))
                if best is None or score > best[0]:
                    best = (score, lin, vals, sep)
            if best is None:
                flat.append(lig)
                continue
            score, lin, vals, sep = best
            kind = ("matrix" if MATRIX_LIGAND.match(lig) or CONTACT_LIGAND.match(lig) else
                    "autocrine" if cell_family(lin) == cell_family(cell) else "paracrine")
            cands.append({"ligand": lig, "sender": lin, "kind": kind, "pct": vals, "separates_from": sep,
                          "receptors": [r for r in recs if r in rec_on] + [f"{r} (low)" for r in recs if r in rec_low],
                          "receptor_on_cell": any(r in rec_on for r in recs), "score": score[1],
                          "sender_share": self._share(lin)})
        cands.sort(key=lambda d: (not d["receptor_on_cell"], -len(d["separates_from"]), -d["score"]))
        para = [d for d in cands if d["kind"] == "paracrine"]
        auto = [d for d in cands if d["kind"] == "autocrine"]
        matrix = [d for d in cands if d["kind"] == "matrix"]
        shown = para[:6] + auto[:3]
        carrying = list(dict.fromkeys(r.split(" ")[0] for d in shown for r in d["receptors"]))
        n_groups = len(groups) - 1

        def cand_line(d):
            both = "" if len(d["separates_from"]) == n_groups else f" (overlaps {', '.join(g for g in groups if g not in d['separates_from'] and g != case)})"
            return (f"- {d['ligand']} from {d['sender']} → {'/'.join(d['receptors'])}: {_fmt(d['pct'], c.samples, self.order, '%')} "
                    f"[every {case} sample > every {', '.join(d['separates_from'])} sample]{both}; {d['sender']} as "
                    f"% of all cells: {self._group_means(d['sender_share'], groups)}")
        contamination: list[str] = []
        try:
            contamination = self.contamination_check(gene, cell)[1].get("case_flags", [])
        except ToolError:
            pass
        hit = sorted({f.split(":")[0] for f in contamination})
        n_case = int((c.samples.group == case).sum())
        lines = [(f"PREMISE QUESTIONABLE: another cell type's RNA can explain part of {gene} in {cell} in {len(hit)} "
                  f"of {n_case} {case} samples ({'; '.join(contamination[:3])}). "
                  + ("Part of what this screen explains may not be " + cell + " expression at all: resolve that first "
                     "(contamination_check, state markers)." if len(hit) == n_case else
                     "Check whether the signal holds in the clean sample(s) before building on it."))] \
            if contamination else []
        lines += [(f"UPSTREAM SCREEN: why do {cell} cells make {gene}? Curated routes (CollecTRI TFs, OmniPath "
                  f"signalling, LIANA-consensus ligands) filtered by this study's data. A screen of what is possible, "
                  f"not evidence that any signal acts."),
                 (f"1. TFs that activate {gene}: expressed in {cell} in every {case} sample (≥{MIN_PCT:.0f}%): "
                  f"{', '.join(tf_on) or 'none'}; low (detected but <{MIN_PCT:.0f}% in some {case} sample — may be "
                  f"detection, not absence): {', '.join(tf_low) or 'none'}; not detected: {', '.join(tf_off) or 'none'}."),
                 (f"2. Receptors with an activating route to those TFs: {len(receptors)}. On {cell} (every {case} sample "
                  f"≥{MIN_PCT:.0f}%, {case} mean above {ref}): {len(rec_on)}. Those carrying the candidates below: "
                  + ("; ".join(f"{r}{' (low)' if r in rec_low else ''} [{self._group_means(rec_vals[r], groups)}] "
                               f"via {' → '.join(receptors[r]['path'])}" for r in carrying) or "none")
                  + f". Other receptors on {cell}: {', '.join(r for r in rec_on if r not in carrying) or 'none'}."
                  + f" Low on {cell} (may be detection, not absence): {', '.join(rec_low[:25]) or 'none'}"
                  + (f" (+{len(rec_low) - 25})" if len(rec_low) > 25 else "")
                  + f". Not detected on {cell}: {len(rec_off)} receptors."),
                 (f"3. Ligands of the receptors on {cell} (or low on it, marked '(low)', listed after) whose expression "
                  f"separates {case} from at least one group (every {case} sample above every sample of that group; "
                  f"the lineage that separates best):"),
                 f"A. Signals from OTHER cell types ({len(para)}):"]
        lines += [cand_line(d) for d in para[:6]] or ["- none"]
        if len(para) > 6:
            lines.append(f"  more: {', '.join(d['ligand'] + '@' + d['sender'] for d in para[6:18])}"
                         + (f" (+{len(para) - 18})" if len(para) > 18 else ""))
        lines.append(f"B. Made by {cell} itself ({len(auto)}; autocrine — may be part of the same program rather than "
                     f"upstream of it):")
        lines += [cand_line(d) for d in auto[:3]] or ["- none"]
        if len(auto) > 3:
            lines.append(f"  more: {', '.join(d['ligand'] for d in auto[3:15])}")
        lines.append(f"C. Matrix, adhesion and antigen-presentation ligands that also differ ({len(matrix)}; usually a "
                     f"readout of remodelling or contact, rarely the inducer of a program): {', '.join(d['ligand'] + '@' + d['sender'] for d in matrix[:10])}"
                     + (" …" if len(matrix) > 10 else ""))
        sig_flat = [x for x in flat if not MATRIX_LIGAND.match(x)]
        lines.append(f"No disease difference in any lineage ({len(flat)}), signalling ligands among them: "
                     f"{', '.join(sig_flat[:40])}" + (f" (+{len(sig_flat) - 40})" if len(sig_flat) > 40 else "") + ".")
        lines.append(f"Next: judge these as a scientist; the order reflects expression differences only, not "
                     f"biology. Which are plausible inducers of {gene} in {cell}, and why (known to induce it? the "
                     f"literature can check; is the sender abundant in the lesion? does it separate {case} from BOTH "
                     f"groups? does it explain other observations)? Which are unlikely? Could {gene} have a "
                     f"non-ligand cause? Test the 2-4 that best discriminate between explanations beyond this screen (coexpression in {cell} of the receptor "
                     f"and {gene}; gene_by_state of the ligand in the sender; covary ligand@sender with "
                     f"{gene}@{cell}) and record a verdict for each.")
        data = {"gene": gene, "cell": cell, "premise": True, "contamination": contamination, "tfs": tfs, "tfs_expressed": tf_on, "tfs_low": tf_low,
                "routes": [{"receptor": r, "path": receptors[r]["path"], "ligands": sorted(receptors[r]["ligands"]),
                            "tfs": sorted(receptors[r]["tfs"]), "on_cell": r in rec_on} for r in receptors],
                "receptors_on_cell": rec_on, "receptors_low": rec_low, "receptors_off_cell": rec_off,
                "candidates": [{k: v for k, v in d.items() if k != "sender_share"} for d in cands],
                "ligands_without_difference": flat}
        return "\n".join(lines), data, ["curated routes are from other systems; expression is transcript level",
                                          "a ligand that separates the groups is a candidate, not a cause",
                                          "low-RNA cells (e.g. neutrophils) detect fewer receptors and TFs"]

    def _changes(self, lineage: str) -> dict:
        """Every gene of one lineage, % of cells expressing per sample, against each other group: which genes are above
        (or below) every sample of every other group, and which of only some groups. No gene list: the data decide."""
        c = self.cells
        lin_mask = c.lineage_mask(str(lineage))
        rows = np.where(lin_mask)[0]
        from scipy import sparse
        Xl = sparse.csr_matrix(c.X[rows]) if not isinstance(c.X, np.ndarray) else c.X[rows]
        smp = c.obs["sample"].to_numpy()[rows]
        pct: dict[str, np.ndarray] = {}
        for s_ in c.samples["sample"]:
            m = smp == s_
            if int(m.sum()) >= MIN_CELLS:
                pct[s_] = np.asarray((Xl[m] > 0).sum(0)).ravel() / int(m.sum()) * 100
        everyone = {g: c.samples[c.samples.group == g]["sample"].tolist() for g in self.order}
        groups = {g: [x for x in everyone[g] if x in pct] for g in self.order}
        case = groups.get(self.case, [])
        if len(case) < len(everyone[self.case]):
            raise ToolError(f"only {len(case)} of {len(everyone[self.case])} {self.case} samples have {MIN_CELLS}+ "
                            f"{lineage} cells: too few to compare")
        # a comparison group counts only with enough samples that have this lineage (two, or all it has)
        others = [g for g in self.order if g != self.case and groups.get(g)
                  and len(groups[g]) >= min(2, len(everyone[g]))]
        if not others:
            raise ToolError(f"no other group has {MIN_CELLS}+ {lineage} cells in enough samples to compare")
        col = {x: i for i, x in enumerate(pct)}
        P = np.vstack([pct[x] for x in pct])
        ci = [col[x] for x in case]
        cmin, cmax = P[ci].min(0), P[ci].max(0)
        gap_up = {g: cmin - P[[col[x] for x in groups[g]]].max(0) for g in others}
        gap_dn = {g: P[[col[x] for x in groups[g]]].min(0) - cmax for g in others}
        keep = P.max(0) >= 5
        # fold, scale-free: a gene in few cells (a cytokine, a growth factor) can separate as cleanly as an abundant
        # one with a far smaller gap in points; +1 keeps genes absent from a group finite
        omax = np.max([P[[col[x] for x in groups[g]]].max(0) for g in others], axis=0)
        omin = np.min([P[[col[x] for x in groups[g]]].min(0) for g in others], axis=0)
        fold_up, fold_dn = (cmin + 1) / (omax + 1), (omin + 1) / (cmax + 1)
        rows_out = {"up_all": [], "down_all": [], "up_some": [], "down_some": []}
        names = c.gene_names
        for i in np.where(keep)[0]:
            ups = [o for o in others if gap_up[o][i] > 0]
            dns = [o for o in others if gap_dn[o][i] > 0]
            if ups:
                rows_out["up_all" if len(ups) == len(others) else "up_some"].append(
                    (float(min(gap_up[o][i] for o in ups)), str(names[i]), ups))
            elif dns:
                rows_out["down_all" if len(dns) == len(others) else "down_some"].append(
                    (float(min(gap_dn[o][i] for o in dns)), str(names[i]), dns))
        for v in rows_out.values():
            v.sort(reverse=True)
        depth = {}
        if c.libsize is not None:
            lib = c.libsize[rows]
            depth = {g: float(np.median(lib[np.isin(smp, groups[g])])) for g in [self.case] + others}
        idx = {str(g): i for i, g in enumerate(names)}
        fold = {k: {g: float((fold_up if k == "up_all" else fold_dn)[idx[g]]) for _, g, _ in v}
                for k, v in rows_out.items() if k in ("up_all", "down_all")}
        return {"pct": pct, "groups": groups, "case": case, "others": others, "rows": rows_out, "depth": depth,
                "index": idx, "fold": fold}

    def _depth_note(self, ch: dict) -> str:
        """Fewer reads per cell lower detection of every gene: name it when the disease cells are much shallower."""
        d = ch.get("depth") or {}
        if not d or self.case not in d:
            return ""
        lo = min(v for g, v in d.items() if g != self.case) if len(d) > 1 else None
        ratio = d[self.case] / lo if lo else 1.0
        txt = ", ".join(f"{g} {v:,.0f}" for g, v in d.items())
        return (f" (median reads per cell: {txt}" + ("; the disease cells are shallower, so many losses may be depth"
                                                     if ratio < 0.6 else "") + ")")

    def _made_elsewhere(self, gene: str, lineage: str, case: list[str]) -> str:
        """Another lineage that makes this gene far more: its RNA here may be ambient or doublets."""
        c = self.cells
        x = c.gene(gene) > 0
        cm = c.obs["sample"].isin(case).to_numpy()
        lv = {}
        for ln in c.obs.lineage.unique():
            m = cm & c.lineage_mask(ln)
            if m.sum() >= MIN_CELLS:
                lv[ln] = float(x[m].mean() * 100)
        own = lv.get(str(lineage), 0.0)
        best = max(((v, ln) for ln, v in lv.items() if ln != str(lineage)), default=(0.0, ""))
        return (f" [made mostly by {best[1]} ({best[0]:.0f}% vs {own:.0f}% here): check ambient RNA]"
                if best[0] >= 50 and best[0] >= 2 * own else "")

    def lineage_changes(self, lineage, n: int = 20):
        """Genome-wide: the genes of one lineage that separate the disease samples from each other group."""
        if isinstance(lineage, (list, tuple)):
            lineage = lineage[0]
        ch = self._changes(str(lineage))
        pct, groups, case, others, idx = ch["pct"], ch["groups"], ch["case"], ch["others"], ch["index"]

        def rng(g, smps):
            v = [pct[x][idx[g]] for x in smps]
            return f"{min(v):.0f}-{max(v):.0f}%" if len(v) > 1 else f"{v[0]:.0f}%"
        heads = {"up_all": f"UP in every {self.case} sample, above every sample of every other group",
                 "down_all": f"DOWN in every {self.case} sample, below every sample of every other group",
                 "up_some": "UP beyond some groups only (involved, but shared with the others)",
                 "down_some": "DOWN below some groups only"}
        out = [(f"{lineage}: every gene reaching 5% of cells in some sample, % expressing per sample, ranked by the "
                f"gap between the {self.case} samples and each other group (compared with {', '.join(others)})"
                f"{self._depth_note(ch)}.")]
        for k in ("up_all", "down_all", "up_some", "down_some"):
            v = ch["rows"][k]
            out.append(f"\n{heads[k]} ({len(v)}):")
            for gap, g, grps in v[:n]:
                out.append(f"- {g}: {self.case} {rng(g, case)}; " + "; ".join(f"{o} {rng(g, groups[o])}" for o in others)
                           + f" [{'above' if k.startswith('up') else 'below'} every {', '.join(grps)} sample by "
                             f"{gap:.0f} points]" + (self._made_elsewhere(g, str(lineage), case) if k == "up_all"
                                                     else ""))
            if len(v) > n:
                out.append(f"- … {len(v) - n} more: {', '.join(g for _, g, _ in v[n:n + 60])}")
            if k in ch["fold"] and len(v) > n:
                shown = {g for _, g, _ in v[:n]}
                byf = sorted(((f, g) for g, f in ch["fold"][k].items() if g not in shown), reverse=True)[:n]
                if byf:
                    out.append("  ranked by fold instead (few-cell genes the point gap hides): "
                               + ", ".join(f"{g} (x{f:.1f})" for f, g in byf))
        data = {k: [{"gene": g, "beyond": grps, "gap": round(gap, 1)} for gap, g, grps in v] for k, v in ch["rows"].items()}
        return "\n".join(out), data, ["transcripts: a gene's mRNA, not its protein, secretion or activity",
                                       f"samples with < {MIN_CELLS} {lineage} cells are left out"]

    def difference_account(self, n: int = 15, baseline: bool = True) -> str:
        """Everything the data show differs between the disease and its comparators, for the hypothesizer: lineage
        shares, state shares within each lineage, and each lineage's clearest gene changes. Computed, not chosen."""
        c = self.cells
        groups = {g: c.samples[c.samples.group == g]["sample"].tolist() for g in self.order}
        case, others = groups[self.case], [g for g in self.order if g != self.case]
        tot = c.obs.groupby("sample").size()

        def sep(vals: dict[str, float]) -> str | None:
            cv = [vals[x] for x in case if x in vals]
            if not cv:
                return None
            up = all(min(cv) > max([vals[x] for x in groups[o] if x in vals] or [np.inf]) for o in others)
            dn = all(max(cv) < min([vals[x] for x in groups[o] if x in vals] or [-np.inf]) for o in others)
            return "up" if up else "down" if dn else None

        def show(vals, unit="%"):
            return "; ".join(f"{g} " + "-".join(f"{v:.0f}" for v in sorted({min(x), max(x)}))
                             + unit for g in self.order if (x := [vals[s_] for s_ in groups[g] if s_ in vals]))
        # the healthy baseline, when the study has one and another (disease) comparator besides it
        ref = self.ref if baseline and self.ref in others and len(others) > 1 else None

        def sep_ref(vals: dict[str, float]) -> str | None:
            """Beyond every healthy sample but not beyond every comparator: perturbed, not specific."""
            cv = [vals[x] for x in case if x in vals]
            rv = [vals[x] for x in groups[ref] if x in vals]
            if not cv or not rv or sep(vals):
                return None
            return "up" if min(cv) > max(rv) else "down" if max(cv) < min(rv) else None
        out = [(f"DIFFERENCE ACCOUNT ({self.case} against {', '.join(others)}; every {self.case} sample beyond every "
                f"sample of every other group; computed from the cells, not chosen).")]
        base: list[str] = []
        comp = []
        for ln in sorted(c.obs.lineage.unique()):
            share = (c.obs[c.obs.lineage == ln].groupby("sample").size() / tot * 100).reindex(tot.index).fillna(0)
            d = sep(share.to_dict())
            if d:
                comp.append(f"- {ln} share of all cells {d}: {show(share.to_dict())}")
            elif ref and (b := sep_ref(share.to_dict())):
                base.append(f"- {ln} share of all cells {b} vs {ref}: {show(share.to_dict())}")
        out += ["\nComposition:"] + (comp or ["- (no lineage share separates the groups)"])
        states = []
        for ln in sorted(c.obs.lineage.unique()):
            sub = c.obs[c.obs.lineage == ln]
            n_ln = sub.groupby("sample").size()
            for st, part in sub.groupby("state"):
                share = (part.groupby("sample").size() / n_ln * 100).reindex(n_ln.index).fillna(0)
                share = share[n_ln >= MIN_CELLS]
                d = sep(share.to_dict())
                if d:
                    states.append(f"- {st} ({ln}) share of {ln} {d}: {show(share.to_dict())}")
                elif ref and (b := sep_ref(share.to_dict())):
                    base.append(f"- {st} ({ln}) share of {ln} {b} vs {ref}: {show(share.to_dict())}")
        out += ["\nStates within lineages:"] + (states or ["- (none)"])
        changes = {}
        for ln in sorted(c.obs.lineage.unique()):
            try:
                changes[ln] = self._changes(ln)
            except ToolError:
                continue
        # a gene raised in most lineages of the disease samples is a sample-wide signal (ambient RNA from damaged
        # cells, or a systemic state), not a change of any one cell type: name it once, and rank each lineage's own
        # changes without it, so cell-type-specific changes are not pushed off the list
        wide, rose, k_wide = sample_wide({ln: [g for _gap, g, _ in ch["rows"]["up_all"]] for ln, ch in changes.items()})
        if wide:
            order = sorted(wide, key=lambda g: -rose[g])
            out.append(f"\nSample-wide (raised in at least {k_wide} of {len(changes)} lineages; ambient RNA or a "
                       f"systemic state, not a change of one cell type; {len(wide)} genes): {', '.join(order[:40])}")
        out.append("\nGene changes per lineage (top by gap, sample-wide genes left out; the rest named):")
        for ln, ch in changes.items():
            vs = "" if len(ch["others"]) == len(others) else f" [compared with {', '.join(ch['others'])} only]"
            for k, word in (("up_all", "UP"), ("down_all", "DOWN")):
                v = [r for r in ch["rows"][k] if not (k == "up_all" and r[1] in wide)]
                if not v:
                    continue
                top = []
                for gap, g, _ in v[:n]:
                    flag = self._made_elsewhere(g, ln, ch["case"]) if k == "up_all" else ""
                    top.append(f"{g} ({'+' if k == 'up_all' else '-'}{gap:.0f}){' [likely ambient]' if flag else ''}")
                more = f"; also {', '.join(g for _, g, _ in v[n:n + 25])}" if len(v) > n else ""
                shown = {g for _, g, _ in v[:n + 25]}
                byf = sorted(((f, g) for g, f in ch["fold"][k].items() if g not in shown and g not in wide), reverse=True)[:10]
                more += ("; by fold: " + ", ".join(f"{g} (x{f:.1f})" for f, g in byf)) if byf else ""
                note = self._depth_note(ch) if k == "down_all" else ""
                out.append(f"- {ln} {word} ({len(v)} genes){vs}{note}: {', '.join(top)}{more}")
            if ref and ref in ch["others"]:
                pct, idx = ch["pct"], ch["index"]
                for k, word in (("up_some", "UP"), ("down_some", "DOWN")):
                    v = [(gap, g) for gap, g, grps in ch["rows"][k] if ref in grps]
                    if not v:
                        continue
                    top = []
                    for _gap, g in v[:n]:
                        cv = [pct[x][idx[g]] for x in ch["case"]]
                        rv = [pct[x][idx[g]] for x in ch["groups"][ref]]
                        gap = min(cv) - max(rv) if k == "up_some" else min(rv) - max(cv)
                        top.append(f"{g} ({'+' if k == 'up_some' else '-'}{gap:.0f})")
                    more = f"; also {', '.join(g for _, g in v[n:n + 25])}" if len(v) > n else ""
                    base.append(f"- {ln} {word} vs {ref} ({len(v)} genes): {', '.join(top)}{more}")
        if ref:
            out.append(f"\nPERTURBED FROM {ref.upper()} BUT NOT SPECIFIC (every {self.case} sample beyond every {ref} "
                       f"sample, not beyond every sample of the other comparators; still suspect: such a factor can act "
                       f"through an interaction with a specific one):")
            out += base or ["- (none)"]
        return "\n".join(out)

    def known_pairs(self, gene):
        if isinstance(gene, (list, tuple)):
            parts = [self.known_pairs(str(g)) for g in gene[:MAX_GENES]]
            return "\n\n".join(p[0] for p in parts), {str(g): p[1] for g, p in zip(gene, parts)}, parts[0][2]
        try:
            text, data = self.kb.pairs(gene)
        except KnowledgeError as exc:
            raise ToolError(str(exc)) from None
        return text, data, ["curated database knowledge, not evidence from this study"]

    def literature_search(self, query: str, n: int = 5):
        try:
            text, data = self.lit.search(query, n)
        except KnowledgeError as exc:
            raise ToolError(str(exc)) from None
        except OSError as exc:
            raise ToolError(f"literature service unreachable: {exc}") from None
        return text, data, ["published findings from other studies; not evidence about these samples",
                            "abstract sentences only; the full paper was not read"]

    def gene_diseases(self, gene: str):
        try:
            text, data = self.ot.gene(gene)
        except KnowledgeError as exc:
            raise ToolError(str(exc)) from None
        except OSError as exc:
            raise ToolError(f"Open Targets unreachable: {exc}") from None
        return text, data, [("association scores aggregate published evidence across studies; not evidence about "
                            "these samples"), "a drug acting on the gene is not evidence that it helps this disease"]

    def gene_drugs(self, gene: str):
        try:
            text, data = self.ot.drugs(gene)
        except KnowledgeError as exc:
            raise ToolError(str(exc)) from None
        except OSError as exc:
            raise ToolError(f"Open Targets unreachable: {exc}") from None
        return text, data, ["drugs from other indications; not evidence that any helps this disease",
                            "clinical stage and action as Open Targets records them"]

    def find_findings(self, query: str):
        words = [w for w in re.split(r"[\s,;]+", query.upper()) if w and w not in {"VS", "AND", "IN", "THE"}]

        def fields(f):
            return [f.feature.upper(), f.cell_type.upper(), (f.target_cell_type or "").upper(), f.case.upper(),
                    f.reference.upper(), *[g.upper() for g in f.genes]]
        hits = [f for f in self.findings if words and all(any(w in x for x in fields(f)) for w in words)]
        hits.sort(key=lambda f: -f.priority)
        return (f"{len(hits)} findings match {query!r}" + (" (top 15)" if len(hits) > 15 else "") + ":\n"
                + "\n".join(digest_line(f) for f in hits[:15]), {"ids": [f.id for f in hits[:15]]}, [])
