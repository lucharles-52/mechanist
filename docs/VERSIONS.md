# Versions and refining after experiments

Status: design, 2026-10-19. Builds on the staged workflow in ARCHITECTURE.md.

## The problem

A Mechanist run ends with ranked circuits, a plan of experiments per hypothesis and drugs aimed at the circuits'
steps. Scientists then do some of those experiments, or know something the run did not, and want to ask: given what
we found, how do the hypotheses change? Today the only answer is a new run from scratch, which throws away the
reasoning that led to the plan and cannot say what moved and why. The original must also stay readable as it was.

## The idea in one paragraph

A finished run is version 1 and never changes. New input (bench results, a fact, a question, a steer) makes a new
version: a new run folder that points to its parent, holds the input, and reruns only the stages after the
circuits. A new stage, `refine`, reads each input against the circuits and the experiment plan, regrades the steps
it bears on, can drop, add or re-rank circuits, and passes the PI's review like every other stage. Then mechanism
cards, experiments, treatments and the report are redone on the refined circuits. The site shows every version,
switches between them, and says what changed from the parent.

## Input

One YAML file per version, filled by the scientist (later from a portal form). Every item gets an id, X1, X2, ...,
that the stages cite like a result id.

```yaml
by: J. Huang
date: 2026-10-24
results:                     # experiments from the plan, or not
  - experiment: E5           # the plan's id; omit for one the plan did not have
    title: IL-36γ and hBD-2 staining of the archived lesion   # needed when there is no plan id
    outcome: positive        # positive | negative | mixed | inconclusive
    what: IHC on two sections from the ulcer edge
    found: IL-36γ strong in suprabasal keratinocytes at the edge, absent 2 mm away; hBD-2 in the same cells
    n: 1 patient, 2 sections
    files: [ihc_il36g_edge.png]
notes:
  - kind: fact               # fact | question | steer
    text: Complement C5a is raised in PG wound fluid
    source: PMID 12345678    # a fact needs a source; it is looked up and the abstract kept
  - kind: question
    text: Could the IL-24 come from T cells rather than fibroblasts?
  - kind: steer
    text: Rank the interferon circuit below the epidermal one unless the data force otherwise
```

Checks by code before anything runs: every `experiment` id exists in the parent's plan; outcomes are from the list;
a fact's PMID resolves (Europe PMC) or the fact is kept as "unsourced" and said so; files exist and are copied into
the version folder.

## Evidence grades

Two grades join the four a step can carry (observed, likely, analogy, speculative):

- `bench`: shown at the bench by a submitted result (X id). The strongest grade; drawn as a double line.
- `refuted`: a submitted result argues against the step. Kept in the drawing, struck through, so a reader sees what
  was ruled out rather than a circuit that silently changed shape.

A result can only move a step's grade through an X id that the refine stage cites, and code checks it: a step marked
`bench` or `refuted` must cite an X result whose outcome fits (positive/mixed for `bench`, negative/mixed for
`refuted`), and a result that refutes a step the circuit needs must lower that circuit's support or rank, or the PI
must accept a written reason why not.

## The refine stage

Inputs: the parent's final circuits, its experiment plan (each experiment's `shows`, `refutes_if` and predictions),
its mechanism cards, and the X items.

The refiner (the circuits' hypothesizer) answers, for every X item:

- which steps and which predictions it bears on, and how (supports, refutes, neither, with the reason);
- for a question: the answer, using the data tools (gene_profile, coexpression, lineage_changes, literature) when
  needed, with result ids;
- for a fact: which steps it supports or contradicts, labelled as literature, never as data;
- for a steer: whether the evidence allows it; a steer is weighed, never obeyed against the data.

Then it returns the refined circuits in the same form as the circuits stage, each step with its grade, and a short
account of what changed and why. The PI reviews this with the same gate (three reviews without progress stop it).

## What is rerun, what is carried over

Carried over from the parent, unchanged and not recomputed: load, findings, dossier, check, candidates, the chase,
linkage and the parent's circuits (kept as the parent's). Rerun in the version: refine, cards, experiment (done
experiments marked done with their result; the next experiments build on them), therapy (drugs re-ranked on the
refined circuits), report, analyses and site. A refine costs a small fraction of a run: only the stages above call
the model, and the parent's tool results are reused.

## Folders and versions

```
runs/p044_Patient_multi_S3/            v1, the original; never written again
runs/p044_Patient_multi_S3.v2/         version.json, inputs.yaml, files/, refine.json, circuits.json (refined), ...
runs/p044_Patient_multi_S3.v3/         parent: .v2
```

`version.json`: version number, parent folder, kind (`results`, `notes` or both), who and when, the X items, the
stages rerun, and `blinded: false` once any note is added (a blinded run stays comparable only as v1).
The existing chain (`events.jsonl` "inputs" source) lets every loader read a version as parent plus changes.

## The site

- A version switch in the header: "v1 · original run, 19 Oct", "v2 · after bench results, 24 Oct". Published as
  `published/<study>/index.html` (the latest), `v1/index.html`, `v2/index.html`, and `versions.json`.
- In a version after the first, a "What changed" section first on the overview, computed by code from the two
  versions: circuits added, dropped or re-ranked; steps whose grade moved, with the X item that moved them;
  experiments now done and new ones; drugs that moved.
- Every X item is shown in full on the Reasoning view, with its files, and the refine stage's correspondence with
  the PI like any other stage.

## Not in this step

- A portal form for the inputs (the YAML file is the interface first).
- Messages to a run that is still going (the live view and nudge inbox come after this, and will write the same X
  items into the next version rather than into the running one).
