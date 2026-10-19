# Mechanist — architecture

Status: v1 slice implemented 2026-10-14 (p033 adapter, findings, verifier, generator/critic/reviser, single-agent
baseline, runner, report, 20 tests). Literature, targets and experiments are not implemented.

## Principles

1. **Code does what code can do.** Loading data, computing findings, checking citations
   and checking claims against data are deterministic Python. LLMs are used only to
   *propose* hypotheses and *criticise* them.
2. **Everything is a typed record with an ID.** LLM output is accepted only if it parses
   into the schema and every ID it cites exists. A claim with no cited ID is dropped.
3. **A state machine, not a conversation.** Stages run in a fixed order with bounded
   loops. Agents never message each other; they read and write records.
4. **Read-only on the pipeline.** Adapters read `Projects/`; Mechanist writes only `runs/`.
5. **Auditable and resumable.** Every stage writes its output plus the sha256 of its
   inputs, prompt and model; a rerun skips stages whose fingerprint is unchanged.

## v3 additions (2026-10-15/16)

- **Outside knowledge** (`knowledge.py`, `kb/` built by `scripts/build_kb.py`): CollecTRI (TF → target), PROGENy,
  LIANA consensus ligand–receptor pairs, OmniPath directed signalling + intercell membrane receptors. Tools:
  `known_regulators`, `receptor_paths` (receptor → … → TF, with ligands), `known_pairs`, and `literature_search`
  (Europe PMC; PMIDs and verbatim abstract sentences; queries limited to gene/cell/pathway words, sample codes
  refused, every query logged to `runs/<id>/external_queries.jsonl`, responses cached).
- **Literature scout** (stage `scout`): plans searches per hypothesis link (general mechanism first, then the
  disease), grades each link supported / contradicted / mixed / not found with PMIDs and a quote (both verified
  against what was retrieved), and adds up to 2 literature-inspired hypotheses (`L1`, `L2`) that must point at a
  retrieved result. They enter the investigation like any hypothesis.
- **Upstream chain**: with the curated tools available, an upstream question counts as resolved only when the
  chain gene → activating TF (`known_regulators`) → receptor route (`receptor_paths`) → a ligand profiled across all
  lineages is complete; the next missing step is fed back each round.
- **Specialist panel** (multi mode, after the Virtual Lab, Swanson et al., Nature 2025): five specialists (stromal,
  adaptive, innate, dermatopathology, computational sceptic) propose hypotheses independently at temperature 0.8;
  the PI merges them at 0.2 and records each hypothesis's origin. Single mode keeps one generator.
- **Guards**: answers whose tool calls all failed are discarded; numbers and PMIDs in answers must appear in the
  cited results; tool exceptions are results, not crashes; each GPU job runs from a frozen code snapshot.
- `guidance` (config): optional high-level direction from the human researcher, shown to every agent.

## v2: question-driven investigation (2026-10-14)

After the first PG run the user's critique was that the agents listed puzzle pieces (CXCL13→CXCR5, CXCL6→CXCR2)
without asking why B cells are there, what they are doing, why fibroblasts make CXCL13, whether the pieces are one
program, and how any of it produces the disease. v2 makes that the core loop:

```
generate -> verify -> INVESTIGATE (rounds: ask typed questions + tool calls -> code runs tools -> answer + follow up)
         -> SYNTHESIZE one model (links: observed | inferred | untested)
         -> [multi] CRITIC reviews model + poses questions -> CHASE them with the same tools -> REVISE model -> report
```

- **Questions** are records (`Question`): kind ∈ presence, function, upstream, relation, disease_link, specificity,
  artifact; parent_id makes the follow-up chain explicit.
- **Required questions** (`investigate.requirements`): for every hypothesis step, why the target cell is there, what
  it is doing, and why the source cell makes the mediator; per hypothesis, relation/disease link/specificity/
  artifact; across hypotheses, whether they connect. Unasked ones are fed back each round and reported.
- **Tools** (`tools.py`) answer from the data, per sample, against both comparators: gene_profile, lineage_share,
  state_composition, gene_by_state, state_markers, coexpression, program_score, covary (single-cell object);
  incoming_signals, ligand_targets, fibroblast_programs, find_findings (tables). Failed calls are results, not crashes.
- **Checks** (`investigate.py`): answers and links may only cite F/R ids that exist and tool calls that succeeded;
  "observed" without valid evidence is downgraded to "inferred". IDs like `F4` are normalised to `F0004`.
- **Evidence grading** does not use p-values (n = 2): tools report whether all PG samples sit above/below every
  sample of each comparator, plus cell numbers. Findings with <30 cells in a sample are flagged `few_cells`.
- **Single-agent baseline**: same tools, rounds and required questions, no independent critic/chase/revise.
- **Metrics**: required questions asked/resolved, chain depth, follow-ups, failed calls, citation problems, links by
  evidence class, critic verdict, tokens.

## Vertical slice (v1)

```
LOAD ──> FINDINGS ──> HYPOTHESES ──> VERIFY ──> CRITIQUE ──> REVISE (≤1) ──> VERIFY ──> REPORT
code      code        LLM (k drafts)  code       LLM          LLM                code       code
```

| Stage | Kind | Input → output |
|---|---|---|
| LOAD | adapter | study tables → `Dataset` (tables + file sha256 + review status + n per group) |
| FINDINGS | code | `Dataset` → `Finding[]` (ranked, capped, e.g. top 200) |
| HYPOTHESES | LLM | `Finding[]` digest → `Hypothesis[]` (several competing, incl. a non-mechanistic/null alternative) |
| VERIFY | code | `Hypothesis[]` × `Finding[]` → `Verification[]` per step |
| CRITIQUE | LLM | hypothesis + its verification + findings → `Objection[]` |
| REVISE | LLM | hypothesis + objections → revised hypothesis (max 1 round) |
| REPORT | code | all records → `report.md` + `report.json`; every sentence carries record IDs |

Literature retrieval (Europe PMC / PubMed with PMID verification), target nomination and
experiment design are slice 2+. In v1 a step needing outside knowledge is labelled
`literature: not retrieved`; the LLM may not cite papers.

## Records (pydantic)

```python
Finding:
  id                      # "F012"
  kind                    # de | pathway | composition | communication | sample_expression | signature
  study, source_file, source_sha256, row_key
  contrast                # case, reference  (or "per-sample profile")
  cell_type, feature      # e.g. "Fibroblast", "CXCL13" / "CXCL13->CXCR5"
  effect, direction, p, q # q may be None
  n_case, n_reference, unit   # unit: sample | donor
  per_sample: {sample: value} | None
  consistency             # e.g. "2/2 case samples above max reference"
  evidence_level          # always "transcript" for scRNA-seq
  review_status           # reviewed | suggested
  caveats: [str]          # n<3, cross-study control, ambient risk, broad-group level…

Hypothesis:
  id, title, summary
  steps: [Step]           # ordered causal chain
  alternative_to: [hypothesis_id]
  predictions: [str]      # what would distinguish it from the alternatives

Step:
  id, source_cell, mediator, target_cell, effect
  claim_type              # association | proposed_mechanism   (causal only with literature, slice 2)
  required_level          # transcript | protein | activity | function
  finding_ids: [str]

Verification:   step_id, status (supported | partial | contradicted | unsupported), checks: [Check]
Objection:      id, target (hypothesis/step id), kind, text, finding_ids
                kind ∈ unsupported_step | overreach_level | confound | alternative | missing_evidence | small_n
```

## Verification rules (deterministic)

Per step:
- every `finding_id` exists and belongs to the same cell type and molecule the step names;
- the direction of each finding matches the step (an "up" claim cited with a "down" finding → contradicted);
- `required_level` above `transcript` with only transcript findings → `partial` and an automatic `overreach_level` note
  (e.g. IGHM transcript ≠ IgM deposition, C3 expression ≠ complement activation);
- `n_case < 3` → `small_n` caveat carried into the report;
- searches the finding table for **contradicting** findings (same cell type and feature, opposite direction) and lists them.

Hypotheses with any `contradicted` or all-`unsupported` steps are reported as rejected, not hidden.

Per-sample expression findings (`kind: expression`) are genome-wide: raw counts of the reviewed single-cell object
(adapter `CellSpec`; else the pipeline's `pseudobulk_counts.h5ad.gz`, adapter `PseudobulkSpec`) are summed per sample
and lineage into CPM for every gene, and every group pair (each group vs the reference, then the other groups, primary
disease first) is scanned for full per-sample separation (mean CPM >= 10 and >= 5% of cells detecting in the higher
group, |log2 ratio| >= 1, closest case vs reference sample >= 0.5 log2 apart). Chance is estimated by repeating the
same scan under every other labelling of the pooled samples (`genome_wide` caveat, `chance_separations`); priority is
scaled by the share of a lineage's separations relabelling does not reproduce, and the digest takes at most 8 per
(cell type, contrast) before repeats. The lab's hand-made `panel`/`signatures`/`ambient` tables are read only with
`lab_tables: true`.

Relevance (all generic): `technical` flags ribosomal, mitochondrial, haemoglobin, unannotated/antisense, sex-chromosome,
dissociation/immediate-early, and immunoglobulin genes outside B-lineage cells; `broadcast` flags a gene that separates
the same way in at least half the scanned lineages, everywhere except its top lineage. Findings carry `labels`:
`pattern` (e.g. PG-specific = also beyond the other disease group; shared with another group) and the curated `role`
from `kb/` (ligand, receptor, TF, PROGENy-responsive). Labels are context, not ranking. `target_score` findings score
each sample and lineage on CollecTRI regulons and PROGENy footprints (z-scored log CPM, technical genes excluded) with
the same separation rule and relabelling null; overlapping regulons are labelled `redundant` (>= 50% of the driver
change already carried by a stronger set) and scores driven mostly by other lineages' genes are flagged `spillover`.
The digest takes clean findings first (per-stratum cap), then flagged or redundant ones.

Data-quality flags set in FINDINGS (and turned into verifier objections when a hypothesis cites the finding):
`small_n`, `ambient` (lab ambient gauge), `spillover` (gene's level in this lineage averages <30% of its top lineage
in the same samples), `detection` (gene zero in every lineage of a sample — dataset/annotation gap), `panel`
(curated gene panel, circular with earlier PG work; only with `lab_tables: true`), `not_significant`. Flagged findings rank lower in the digest.

## Agents (v1)

| Role | Implementation | Prompt input |
|---|---|---|
| Orchestrator | Python state machine (`runner.py`) | — |
| Molecular analyst | Python (`findings.py`) | — |
| Mechanism generator | LLM, k samples (default 3), JSON-schema output | findings digest |
| Scientific critic | LLM; v1 uses the same served model as the generator (one model per GPU job) | one hypothesis + verification + relevant findings |
| Literature researcher, translational agent | slice 2+ | — |

## Single-agent baseline

The same findings digest, the same verifier and the same report builder. One LLM call
(or a single self-revising call) produces hypotheses; no separate critic or revision.
Token budget matched to the multi-agent run. Selected with `--mode single|multi`.

## Evaluation protocol (to fix before the first PG run)

1. **No-data control:** shuffle the group labels (or remove the findings) and rerun. If the
   hypotheses do not change, they come from prior knowledge, not the data.
2. **Planted errors:** inject fabricated findings and wrong-direction citations; measure how
   many the verifier and critic catch (multi vs single).
3. **Traceability:** fraction of report claims backed by a valid finding ID; rate of unsupported claims.
4. **Known biology (secondary):** recovery of established Sweet's / PG axes (neutrophils, IL-1, CXCL8, G-CSF, TNF, IL-17).
5. **Blinded expert rating** of single-agent vs multi-agent reports.

PG is a development case; none of this is validation on new data.

## LLM backend

`llm.py` exposes one function, `complete(messages, schema) -> dict`, with three backends:
- `openai_compat`: any OpenAI-compatible server (local vLLM default, base URL from config); stdlib HTTP.
- (not implemented) hosted APIs: the openai_compat backend refuses any non-loopback host unless `external_ok: true`.
- `replay`: returns recorded responses from a fixture file (tests, deterministic reruns).

Every call is logged to `runs/<id>/llm_calls.jsonl` (prompt hash, model, params, raw response).
The data guard sends only findings (aggregated values with internal sample codes) and
refuses external backends unless the config explicitly allows them.

On Betty the local model is served the same way as the pipeline's LLM workers
(`vllm serve` on loopback inside a GPU allocation, one model at a time).

## Run layout

```
runs/<run_id>/
  config.yaml            # resolved config, incl. backend and mode
  events.jsonl           # append-only stage start/end, fingerprints
  dataset.json           # source files + sha256 + review status
  findings.json
  hypotheses.r0.json  verification.r0.json  objections.json  hypotheses.r1.json  verification.r1.json
  llm_calls.jsonl
  report.md  report.json
```

## Package layout

```
mechanist/
  pyproject.toml          # uv; deps: pandas, pyyaml, pydantic (anndata optional)
  src/mechanist/
    records.py            # pydantic models
    adapters/skinpipe.py  # comparison tables; p033 legacy + pg_mechanism tables
    findings.py
    verify.py
    agents.py             # generator, critic, reviser, single-agent baseline (prompts + parsing)
    llm.py                # backends + data guard
    runner.py             # state machine, fingerprints, resume
    report.py
    cli.py                # mechanist run --study p033 --config ... --mode multi
  tests/                  # tiny synthetic tables + replay LLM fixtures; no network, no GPU
```

## Staged workflow (multi mode, `workflow: staged`; code 0.30, 2026-10-17)

```
load → findings → dossier ─PI→ check ─PI→ panel/generate → candidates ─PI→ scout → investigate → synthesize
     → critique → chase (PI rounds until the model stops improving) → linkage ─PI→ revise = evidence ledger ─PI
     → circuits ─PI→ experiment ─PI→ report
```

- **Evidence ledger** (`revise`): the grounded model. Each link has its evidence, depth, involved-vs-specific
  verdicts and best guesses. Phenotype accounting classifies every dossier item (D01, ...) as explained, consistent,
  contradicted or not testable. `check_phenotype` objects to skipped items.
- **Causal steps**: every question names a step (S01, ...). A step is exhausted when an answer says the data cannot
  settle it, naming the decisive observation, or after three tries. Asking again needs a `new_approach`.
- **Circuits** (stage 7, the final model): the hypothesizer (Opus, high effort, 8 checking calls per round) receives
  the ledger, a **difference account** computed by code (`Toolbox.difference_account`: lineage and state shares and
  genome-wide gene changes per lineage, with ambient and depth flags; no curated gene lists), the exhausted steps and
  the dossier items.
  - It proposes 2-4 complete circuits: root and why here and not in the comparators, edges tagged
    observed/likely/analogy/speculative, loops, how each produces the phenotype, discriminating predictions, and the
    closest analogue.
  - `check_circuits` downgrades tags without evidence.
  - The PI judges plausibility, not proof.
  - Experiments are designed to tell the circuits apart.
- Prompts are disease-agnostic. Nothing names a disease, gene or pathway. Groups, lineages and comparators come from
  the study.

## Experiments and therapeutic hypotheses (code 0.33, 2026-10-18)

```
… → circuits ─PI→ experiment ─PI→ therapy ─PI→ report
```

Both stages work from the final circuits, not the ledger alone. Both are research hypotheses: nothing here is a
recommendation for a patient (hard rule), and every drug fact comes from a lookup that ran.

**Missing links (code, `investigate.missing_links`).** From the circuits: every edge not tagged observed, in a circuit
proposed by at least 2 of K attempts (any edge when there is one attempt); every exhausted causal step; every lead one
attempt alone found. Each gets an id (M1, …) with the circuits that use it and their support. This list is the agenda
of both stages.

**Experiment stage (stage 8).**
- The experimentalist (tool loop: literature_search, gene_diseases, gene_drugs, find_findings) designs experiments that
  decide between the circuits and settle the missing links. It must account for every M id: test it, or say why no
  feasible experiment can.
- Each experiment names the M ids it tests, a tier (`in_data`, `archived_tissue`, `spatial`, `ex_vivo`, `in_vivo`,
  `cohort`), the perturbation, readout, controls, feasibility, precedent (ids of searches that ran), and **what each
  circuit predicts**: `up`, `down`, `no change` or `not addressed`, with a one-line reason.
- Code (`investigate.rank_experiments`): discrimination = the number of distinct outcomes the addressed circuits
  predict, weighted by their support; ranked by discrimination, then by tier (cheapest first), then the agent's
  priority. `check_experiments` drops unknown circuit ranks and M ids, keeps precedent only when its search ran, and
  lists every M id no experiment covers.
- The page shows a decision matrix: experiments × circuits, each cell the predicted outcome.

**Drug lookup (`gene_drugs`, Open Targets).** For one gene: drugs and clinical candidates with their highest clinical
stage, action type (inhibitor, antagonist, agonist, modulator…), mechanism, targets and indications. Only the gene
symbol leaves the cluster (the literature guard's check); responses are cached and logged like every external query.

**Therapy stage (stage 9).**
- Code (`investigate.therapy_targets`) collects the genes on each circuit's edges (mediators, receptors, the genes the
  edge names; cell markers left out), with the circuits that carry them, their support and whether the edge is observed
  or untested. It looks up drugs for each (`gene_drugs`, top genes by support, capped), before any model call.
- The therapy agent (tool loop: literature_search, gene_diseases, gene_drugs) writes therapeutic hypotheses from that
  table: the drug, its target, the circuit edge it would break, the disease items (D ids) it should improve, the
  circuits under which it should fail, and the untested M ids it depends on (test first). It also groups the targets
  into pathways (its own names; prompts stay disease-agnostic) with the drug classes acting on each.
- Code checks (`investigate.check_therapies`): a drug must be in a lookup that ran, with that target; its stage and
  action come from the lookup, never from the model; an agonist on an edge the circuit needs active is flagged; the
  evidence level is set by code: **in use for this disease** (named in the dossier's therapy items), **approved for
  another disease** (repurposing), **in trials**, **preclinical**.
- Ranking (code): evidence level, then the support of the circuits it acts on, then fewest untested dependencies.
  Pathways are weighted by the summed support of the circuits whose edges carry their genes.
- The page shows the therapeutic hypotheses (drug × circuit, evidence badges, "depends on M3: test first") and a
  pathway chart.

**Cache safety.** New config fields are left out of stage fingerprints while at their defaults, so runs that predate
them keep their saved stages. Prompt edits change only the calls of the stages they belong to.

### Revision (2026-10-18, same day): experiment types and treatment strategies

**Two kinds of experiment** (full SOPs are the end goal, not yet; for now each is listed with its key parameters):
- **Confirmatory.** Code lists the *molecules the circuits rest on*: the genes on the edges of circuits that at least two
  attempts proposed. Each needs a check in tissue or at protein level (staining, flow, ELISA, spatial) in disease vs
  comparator samples. The agent notes where the protein may differ from the RNA (secreted, cleaved, stored or
  short-lived proteins; literature or proteomics that says so). Code lists molecules no confirmatory experiment covers.
- **Mechanistic.** Claimed interactions reproduced in vitro (primary cells or co-culture of the circuit's cell types),
  ex vivo, or in vivo: the perturbation kind (knockout, knockdown, blocking antibody, small molecule, recombinant
  ligand, dose response, time course) and controls are required (code objects to a mechanistic experiment without
  controls). These carry each circuit's predicted outcome and are ranked by how much the circuits disagree.

**Two treatment strategies** (the clinician weighs certainty of effect against safety; Mechanist only lays it out):
- **Blunt**: removes a whole cell type of the circuits (depleting antibodies against its surface markers) or suppresses
  broadly. Likely to break the circuits whichever is right, with more side effects.
- **Targeted**: blocks one mediator or receptor. Fewer off-target effects, but works only if that step is right.
- Code supplies **cell-type targets**: the data's own lineage markers (genes in at least 30% of a lineage's cells and few
  elsewhere) for each cell type the circuits name, with their drugs looked up like the step genes. A drug on a lineage
  marker is classed blunt (cell depletion) by code.
- **Robustness** (code): the summed support of the circuits the drug acts in, as a share of all circuit support; a
  depleting drug acts in every circuit that uses its cell type.
- **Safety flags** (code, from Open Targets drug warnings): black-box warnings and withdrawals, with their toxicity
  class. Nothing about dose or patient selection.

### Revision (2026-10-18, late): experiments as a plan per hypothesis

The flat ranked list of experiments read without logic. Experiments are now planned **hypothesis by hypothesis**, as a
sequence in which each experiment builds on the ones before it. The prompts state the principles; the agent fills them.

- **Four questions, in the order a lab asks them.** Each experiment answers one: *is it there* (the molecules and cells
  a hypothesis needs, at protein level in the right cells, in disease vs comparators, before anything is built on
  them); *what could rule it out fast* (cheap, in this dataset or archived tissue); *does the mechanism work* (the
  hypothesis's central causal claim tested directly: give the proposed signal to the receiving cells, and block it);
  *does it hold in a living system* (the most important claim in vivo, where a model exists). The general experiment
  types (confirmatory, mechanistic; perturbation kinds; controls; tiers) are back in the prompt as guidance.
- **Each experiment says what it shows** (the claim a positive result establishes) and **what it adds** beyond
  published work, and is titled by that claim in plain words, not by method or team labels.
- **Done before.** If a paper already did essentially this experiment, it is marked with its PMID (checked by code
  against papers a search returned); the human decides whether to replicate or cite.
- **Shared experiments.** One experiment can sit in several hypotheses' sequences, as it is or adapted; the page marks
  it "also in".
- **Code checks** (`check_programmes`): every circuit at least two attempts proposed has a sequence; each sequence has a
  mechanism or in-vivo experiment; presence is not checked after a mechanism experiment that relies on it; every
  experiment sits in some sequence.
- **The PI reviews the plan with the same questions**: what does each experiment show, what does it add, which
  mechanism does it prove; is the central claim tested directly rather than by correlation; do the experiments build
  on each other, and does the sequence say what happens if one fails?

### Reader-facing prose (2026-10-19)

The site's reader-facing text (the answer and brief, circuit summaries, step notes, experiments, their sequences, drug
reasoning) gets a prose pass on every build (`scripts/build_site.py`, cached under `runs/<run>/site/`):
1. `agents.HUMANIZE_SYSTEM`: the rules of the humanizer and everwrite skills, adapted for a science report (no dashes,
   no "not X but Y", no inflation or -ing riders, the team's shorthand written out, arrow chains turned into sentences).
2. `mechanist.tells` (everwrite's `tells.py`, vendored unchanged) checks every passage; any with a strong finding goes
   back with its findings to `agents.HUMANIZE_FIX_SYSTEM`.
3. A rewrite is kept only if `present.plain_check` passes against the original (same numbers, names and ids, not much
   longer). The checker's before/after counts go to `site/prose_report.json` and the page footer.
The calls use the run's own backend and accounts (subscription seats). The reasoning record and the agent
correspondence stay verbatim: they are the record of what the agents wrote.
