# Talking to a run, and bench results from the portal

Status: design, 2026-10-19. Builds on VERSIONS.md (versions and the refine stage) and the portal request flow
(Projects/scripts/mechanist_requests.py).

Two things scientists asked for:

1. **While a run is going**: see where it is and what it is reasoning about, and send it a note (a question, a fact
   with its source, a steer) that it reads and answers.
2. **After a run**: hand in bench results and notes from the portal, which makes a new version (VERSIONS.md). Today
   that takes a YAML file and a command.

## How messages travel

Nothing connects into Betty, so everything goes through Box, as the study form and the reviews already do.

```
portal ──▶ Box inbox/mechanist_chat/<run>/<time>.json ──▶ relay ──▶ runs/<run>/inbox/
portal ◀── Box status/mechanist_live/<study>__<run>.json ◀── relay ◀── runs/<run>/ (stages, transcript, chat)
```

The relay is a small loop that runs beside each run in the run's own Slurm job, every 30 seconds, and stops with
the run. It needs no new service. The portal reads the live file straight from Box while someone has the panel open,
so a reply shows within about a minute of being written.

## What the run does with a message

Every message gets an id, N1, N2, and so on. Stages cite these ids the way they cite result ids.

- **A reply within minutes.** A thread inside the run process watches `inbox/`. For each message it writes the
  message to `chat.jsonl` and asks the lead investigator for a reply. The reply is built from what the run has
  established so far (stages done, current candidates or circuits, the stage in progress). It answers a question
  from those results with their ids, or says it will put the question to the next stage. For a fact, the reply
  says how it will be weighed. For a steer, it says whether the data allow it. The reply makes no data queries:
  it runs beside the stage, and the stage owns the data tools.
- **Read from the next stage on.** When a stage is about to start (not when it is skipped from the cache), it takes
  delivery of every message not yet delivered. From then on every agent sees them in its context, under the same
  rules as refine notes:
  - a fact is literature, not data;
  - a steer is weighed, never obeyed against the data;
  - a question is answered with data tools where it bears on the stage's task.

  Each message records the stage it was delivered to (`from_stage`). That stage and every later one carries it in
  its fingerprint, so a resumed run rereads exactly the same notes at the same places.
- **Blinding.** A message to a blinded run unblinds it. The run record says so, as a refine note does.
- **The long stages.** A note reaches agents at the next stage boundary. The circuits and chase stages can take an
  hour, so the reply says which stage will read the note first. Interrupting a stage mid-way is not in this step: it
  would make resumes rerun work.

## The live view (portal, study page)

A Mechanist panel with the blue mark, shown while a run is going:

- the stages as a row of steps: done, under way (with its elapsed time), to come;
- the latest reasoning: the last turns of the transcript (agent turns and tool results), as the Reasoning view
  shows them;
- the conversation: each message with who sent it and when, the run's reply, and the stage that first read it;
- a box to write in, with the kind (question, fact, steer) and a source field for facts.

Only the study's owners and editors and the admins can write. Everyone who can see the study can read.

## Bench results from the portal

On a study with a published interpretation, an "Add bench results" form builds the same input file as VERSIONS.md:

- results, each with:
  - the plan's experiment, picked from the published plan (or "not in the plan" with a title);
  - the outcome (positive, negative, mixed or inconclusive);
  - what was done, what was found, n;
  - files (images or PDFs, up to 5 files of 20 MB each);
- notes (fact with source, question, steer).

The portal checks what it can (required fields, outcome, experiment ids against the plan, file types and sizes)
and sends the input to Box at `inbox/mechanist_refine/<study>/<time>/`.

On Betty, `box_bridge pull` hands it to `mechanist_requests.py refine`. That job:

1. checks the input with the version loader (`refine.py --check`), which also looks up each fact's PMID;
2. queues the refine job on Slurm against the newest published version;
3. builds and publishes the new version automatically, as every finished run is published.

If the check fails, nothing is made and the problems go back to the portal in `status/mechanist_runs.json`.

A version builds on a run an admin already approved and costs a fraction of a run. So it needs no second approval:
admins see every submission and its state on the admin Mechanist page. This is easy to change if admins want to
approve versions too.

`publish_site.py` writes `plan.json` (the newest version's experiments: id, title, hypothesis, done or not) next to
the site, so the form can offer the plan's experiments.

## Not in this step

- Interrupting a stage mid-way for a note.
- Data queries in the immediate reply.
- Editing or withdrawing a message once sent.
