# AGENTS.md — Mechanist

Agentic hypothesis-discovery workflow over finished skinpipe study outputs.
Read `docs/PROJECT_CONTEXT.md` (what exists) and `docs/ARCHITECTURE.md` (design) first.

## Hard rules
- **Read-only** on `../Projects/`. Write only under `runs/`. Never write into `pNNN/Results/` or review folders.
- **Independent of skin-fm.** Do not import from or call `../skin-fm/`.
- **Never invent data or citations.** Every claim cites a `Finding` or (slice 2+) a retrieved, verified PMID. Uncited claims are dropped, not repaired.
- scRNA-seq evidence is **transcript level**. Never state protein, activity or function from it.
- Association ≠ causation. `claim_type` is `association` or `proposed_mechanism` unless experimental literature supports more.
- Always carry `n_case`/`n_reference` and the unit (sample/donor). n < 3 → per-sample consistency, not p-values.
- Unreviewed clusters (`C0`, LLM suggestions) are not cell types. Adapters set `review_status`; `adapters/skinpipe.py` (any current study) refuses unreviewed ones (pending review, no sign-off workbook, no reviewed object, C0/blank labels). `adapters/p033.py` is the legacy case.
- Nothing study-specific in code: group order comes from the study (disease first, reference last), the specialist panel from its lineages (`runner.derive_panel`; the artifact sceptic is always added), adapter settings from `adapter_options`.
- Default LLM is local (OpenAI-compatible vLLM). External backends need `external_ok: true` and receive findings only — no cell-level data, no identifiers.
- Therapeutic text is research-only and never a recommendation for a patient.

## Commands
- `uv run pytest -q` · `uv run ruff check src tests`
- Deterministic stages only (no LLM): `uv run mechanist run --config configs/p033_pg.yaml --stop-after findings`
  (genome-wide scan loads the reviewed object: ~1 min and ~5 GB RAM on p033; fine on a login node)
- Any reviewed study: copy `configs/skinpipe_template.yaml` (set study, disease, adapter_options.groups/reference)
- Full PG run, multi + single on one local model: `sbatch slurm/run_p033.sbatch` (GPU job; serves vLLM on loopback)
- Compare runs: `uv run mechanist compare runs/A runs/B`
- uv lives at `../Projects/.cl_analysis/llm/bin/uv`; set `UV_CACHE_DIR` under `.cl_analysis/llm/runtime/cache/uv` (home quota is small).

## Layout
`records.py` (schemas) · `adapters/` (read-only loaders) · `findings.py` (molecular analyst) · `verify.py` ·
`tools.py` (data queries) · `investigate.py` (required questions, citation checks) · `agents.py` (all prompts) · `llm.py` (backends, data guard, cache) · `runner.py` (state machine) · `report.py` · `cli.py`

## Engineering
- Python 3.12, uv. `uv sync`, `uv run pytest`, `uv run ruff check`.
- Minimal deps: pandas, pyyaml, pydantic; extra `cells` = anndata + scipy for single-cell tools. No agent frameworks.
- Agents must chase questions (why here / what doing / upstream / relation / disease link); never weaken the required-question list to raise coverage.
- LLMs only in `agents.py`; everything else is deterministic and unit-tested.
- Tests use synthetic tables and the `replay` backend: no network, no GPU, no real study data.
- Each stage writes its output + input fingerprint; reruns must resume, not recompute.
- Reuse pipeline helpers by reading their outputs; import `Projects/scripts` only via the adapter, never copy logic.
