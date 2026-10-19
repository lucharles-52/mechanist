"""Typed records passed between stages. Every record has an ID; LLM output is accepted only through these models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

FindingKind = Literal["de", "pathway", "progeny", "composition", "communication", "expression", "target_score", "panel_expression",
                      "signature"]
Direction = Literal["up", "down"]
EvidenceLevel = Literal["transcript", "protein", "activity", "function"]
LEVEL_RANK = {"transcript": 0, "protein": 1, "activity": 2, "function": 3}


class SourceFile(BaseModel):
    path: str
    sha256: str
    rows: int


class Dataset(BaseModel):
    """What was loaded, from where, and what limits every conclusion drawn from it."""

    study: str
    adapter: str
    groups: dict[str, list[str]]  # group -> samples
    reference_group: str = "Control"
    files: list[SourceFile]
    review_status: Literal["reviewed", "suggested", "legacy"]
    caveats: list[str] = []
    question: str = ""  # the submitter's own question (the study's metadata/submission.yaml): what the run must answer


class Finding(BaseModel):
    id: str
    kind: FindingKind
    study: str
    source_file: str
    row_key: str
    case: str
    reference: str
    cell_type: str
    target_cell_type: str | None = None  # communication only: cell_type is the sender
    feature: str  # gene, pathway, "proportion", or "LIGAND->RECEPTOR"
    genes: list[str] = []  # gene symbols behind the feature (empty for pathways/proportions)
    direction: Direction
    effect: float
    effect_label: str
    p: float | None = None
    q: float | None = None
    n_case: int
    n_reference: int
    unit: Literal["sample", "donor"] = "sample"
    per_sample: dict[str, float] | None = None
    separated: bool | None = None  # every case sample beyond every reference sample
    p_separation: float | None = None  # exact chance of that separation under exchangeable labels
    evidence_level: Literal["transcript"] = "transcript"
    review_status: Literal["reviewed", "suggested", "legacy"]
    caveats: list[str] = []
    labels: list[str] = []  # descriptive context (contrast pattern, curated gene role, drivers); not caveats
    priority: float = 0.0  # ranking for the LLM digest only


class Step(BaseModel):
    id: str = ""
    source_cell: str
    target_cell: str | None = None
    mediator: str
    genes: list[str] = []
    direction: Direction = "up"
    effect: str = ""
    claim_type: Literal["association", "proposed_mechanism"] = "association"
    required_level: EvidenceLevel = "transcript"
    finding_ids: list[str] = []


class Hypothesis(BaseModel):
    id: str = ""
    title: str
    summary: str = ""
    disease: str
    specific_to_disease: bool = False
    non_mechanistic: bool = False  # artifact / confound / composition alternative
    steps: list[Step] = []
    predictions: list[str] = []
    revised_from: str | None = None
    withdrawn: bool = False
    author: str = ""  # generator | reviser | single_agent | literature_scout
    inspired_by: list[str] = []  # literature/database result ids (R…) a scout hypothesis came from
    origin: list[str] = []  # panel specialists whose proposals this hypothesis merges
    pmids: list[str] = []


class Check(BaseModel):
    rule: str
    passed: bool
    detail: str


class StepVerification(BaseModel):
    step_id: str
    status: Literal["supported", "partial", "contradicted", "unsupported"]
    valid_finding_ids: list[str] = []
    invalid_finding_ids: list[str] = []
    contradicting_finding_ids: list[str] = []
    checks: list[Check] = []


class Verification(BaseModel):
    hypothesis_id: str
    status: Literal["supported", "partial", "rejected", "withdrawn"]
    steps: list[StepVerification]


ObjectionKind = Literal["unsupported_step", "overreach_level", "confound", "alternative", "missing_evidence",
                        "small_n", "contradiction", "not_specific", "invalid_citation"]


class Objection(BaseModel):
    id: str = ""
    target: str  # hypothesis or step id
    kind: ObjectionKind
    text: str
    finding_ids: list[str] = []
    source: Literal["verifier", "critic"]
    valid: bool = True  # False when a critic objection cites findings that do not exist


class Critique(BaseModel):
    hypothesis_id: str
    verdict: Literal["plausible", "weak", "unsupported"]
    objections: list[Objection] = Field(default_factory=list)


# ---- Investigation -------------------------------------------------------------------------------------------------

QuestionKind = Literal["presence", "function", "output", "upstream", "relation", "disease_link", "specificity",
                       "artifact", "other"]
QuestionStatus = Literal["open", "answered", "partial", "unanswerable_with_data", "needs_literature"]


class ToolCall(BaseModel):
    tool: str
    args: dict = {}


class ToolResult(BaseModel):
    id: str  # "R0001"
    tool: str
    args: dict
    ok: bool
    summary: str  # compact text the LLM reads
    data: dict = {}
    sources: list[str] = []
    caveats: list[str] = []


class Candidate(BaseModel):
    """A candidate inducer the investigator judged: which signal, from which cell, through which receptor, and its
    verdict after testing it in the data."""

    ligand: str
    sender: str = ""
    receptor: str = ""
    verdict: Literal["supported", "weakened", "rejected", "untested"] = "untested"
    reason: str = ""


class Question(BaseModel):
    id: str = ""  # "Q01"
    kind: QuestionKind = "other"
    text: str
    cells: list[str] = []
    genes: list[str] = []
    hypothesis_ids: list[str] = []
    parent_id: str | None = None
    asked_by: str = ""  # investigator | critic | single_agent
    round: int = 0
    calls: list[ToolCall] = []
    result_ids: list[str] = []
    status: QuestionStatus = "open"
    answer: str = ""
    answer_ids: list[str] = []  # findings / results the answer rests on
    conclusion: Literal["observed", "literature", "inferred", "none"] = "none"
    prior_knowledge: list[str] = []  # background claims the model brought in (unsourced; reported as such)
    candidates: list[Candidate] = []  # upstream questions: candidate inducers and the verdict on each
    why: str = ""  # why the planner chose this thread
    steps: list[dict] = []  # thread trail: {"thought", "tool", "args", "result_id"} per step
    follow_ups: list[str] = []  # questions the answer raised, most important first
    # the causal step this question works on (S01, ...; shared by every question on the same step) and its label
    step: str = ""
    step_label: str = ""
    # when the data cannot settle the step: the observation that would, and whether this dataset can make it
    decisive_observation: str = ""
    observable_here: Literal["yes", "no", "other_data", ""] = ""


class Requirement(BaseModel):
    """A question the investigation must ask (derived by code from the hypotheses)."""

    kind: QuestionKind
    cell: str | None = None
    gene: str | None = None
    hypothesis_id: str | None = None
    cells: list[str] = []  # for hypothesis-level requirements: the hypothesis's cell types
    priority: int = 3  # 1 = key mediator of a starting hypothesis; 2 = hypothesis-level; 3 = everything else
    text: str


Presence = Literal["observed", "absent", "not_checked"]


class ChainStep(BaseModel):
    """One step of a link's chain and its kind: tested here, supported by literature retrieved here, or speculative."""

    claim: str
    kind: Literal["tested", "literature", "speculative"] = "speculative"
    ids: list[str] = []  # tested: result/finding ids; literature: ids of the searches that returned the PMIDs
    test: str = ""  # speculative: the analysis or experiment that would test it

    @field_validator("kind", mode="before")
    @classmethod
    def _kind(cls, v):
        v = str(v or "").lower()
        return "literature" if v.startswith("lit") else "tested" if v.startswith("test") else "speculative"


class Link(BaseModel):
    """One arrow of the model. Transcript data can show that the source expresses the mediator and that the target
    expresses its receptor (or a response); the effect itself is always an inference."""

    source: str
    target: str
    mediator: str = ""
    mediator_in_source: Presence = "not_checked"
    receptor_in_target: Presence = "not_checked"
    evidence: Literal["expression_observed", "partly_observed", "not_observed", "contradicted"] = "not_observed"
    ids: list[str] = []  # finding (F) and result (R) ids
    question_ids: list[str] = []  # questions of the investigation that examined this link
    note: str = ""
    depth: int | None = None  # highest level of the PI depth ladder reached (1 real ... 7 disease linkage)
    steps: list[ChainStep] = []  # the link's chain, each step labelled by kind
    disease_linkage: str = ""  # what the literature searches say about this link in this or related diseases
    involved: str = ""  # vs healthy tissue: is it part of the pathology?
    specific: str = ""  # vs the related comparator disease: is it what makes this disease different?

    @field_validator("depth", mode="before")
    @classmethod
    def _depth(cls, v):
        try:
            return max(1, min(7, int(v))) if v not in (None, "") else None
        except (TypeError, ValueError):
            return None

    @field_validator("mediator_in_source", "receptor_in_target", mode="before")
    @classmethod
    def _presence(cls, v):
        return v if v in ("observed", "absent", "not_checked") else "not_checked"

    @field_validator("evidence", mode="before")
    @classmethod
    def _evidence(cls, v):  # derived by code (investigate.check_synthesis); model-supplied values are ignored
        return v if v in ("expression_observed", "partly_observed", "not_observed", "contradicted") else "not_observed"


class Alternative(BaseModel):
    """A ranked competing hypothesis kept beside the lead model."""

    rank: int = 2
    title: str
    summary: str = ""
    confidence: str = ""  # e.g. low / moderate, with the reason
    links: list[Link] = []
    evidence_for: list[str] = []
    evidence_against: list[str] = []


class GuessCandidate(BaseModel):
    """One ranked explanation for a step the data could not settle."""

    rank: int = 1
    explanation: str
    evidence_rung: int | None = None  # 1 separating test in this dataset ... 5 plausibility alone
    evidence: str = ""
    ids: list[str] = []
    confidence: str = ""


class BestGuess(BaseModel):
    """An open step with its ranked best guesses and the test that would decide."""

    step: str
    candidates: list[GuessCandidate] = []
    decisive_test: str = ""


class CircuitEdge(BaseModel):
    """One arrow of a proposed circuit, tagged by how it is known."""

    source: str
    target: str
    via: str = ""  # mediator, contact, product
    # bench / refuted: a version's submitted result (X id) shows it or argues against it (docs/VERSIONS.md)
    status: Literal["bench", "refuted", "observed", "likely", "analogy", "speculative"] = "speculative"
    ids: list[str] = []  # observed: result/finding ids; analogy: ids of the searches that returned the precedent
    note: str = ""


class Circuit(BaseModel):
    """A complete proposed mechanism: what starts it, how it sustains itself, how it produces the disease."""

    rank: int = 1
    title: str
    summary: str = ""
    root: str = ""  # the observed difference that starts it, and why it runs here and not in the comparators
    edges: list[CircuitEdge] = []
    loops: list[str] = []
    phenotype: list[dict] = []  # {"item": dossier id or feature, "how": str}
    predictions: list[dict] = []  # {"test": str, "expects": str, "in_this_data": bool}
    analogue: str = ""
    unused: list[str] = []  # striking differences left out of this circuit, and why
    confidence: str = ""
    support: str = ""  # consensus: how many independent attempts proposed it ("3 of 3"); empty for a single attempt


class PhenotypeFit(BaseModel):
    """How the model accounts for one item of the disease dossier (a feature, gene, therapy response, association)."""

    item: str  # D01, ... (the dossier item's id)
    verdict: Literal["explained", "consistent", "contradicted", "not_testable"] = "not_testable"
    links: list[str] = []  # the model's links that produce it (source -> target), for "explained"
    reason: str = ""


class Synthesis(BaseModel):
    title: str
    summary: str
    links: list[Link] = []
    confidence: str = ""
    alternatives: list[Alternative] = []
    experiments_planned: list[dict] = []  # stage 5 plan, attached by the runner
    disease_link: str = ""
    competing_explanations: list[str] = []
    open_questions: list[str] = []
    experiments: list[str] = []
    best_guesses: list[BestGuess] = []
    phenotype_fit: list[PhenotypeFit] = []  # final model only: every dossier item accounted for
    author: str = ""


class LitAssessment(BaseModel):
    """How prior work (literature / curated databases) bears on one hypothesis step."""

    hypothesis_id: str
    step_id: str = ""
    support: Literal["supported", "contradicted", "mixed", "not_found"] = "not_found"
    result_ids: list[str] = []
    pmids: list[str] = []
    quote: str = ""
    note: str = ""
    verified: bool = True  # False when the PMIDs or quote were not found in the retrieved results


class Persona(BaseModel):
    """A specialist on the hypothesis panel (Virtual Lab style: title, expertise, goal, role)."""

    title: str
    expertise: str
    goal: str = "propose mechanisms for the disease that the data can test"
    role: str = "propose hypotheses from your field's perspective and say what in the data would refute them"
    artifacts: bool = False  # this specialist (and only this one) proposes non-biological explanations

    def prompt(self) -> str:
        return (f"You are a {self.title}. Your expertise is in {self.expertise}. Your goal is to {self.goal}. "
                f"Your role is to {self.role}.")
