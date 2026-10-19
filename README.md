# Mechanist

Mechanist reads the outputs of a single-cell skin study (skinpipe) and works out the most likely disease mechanisms
behind them. A team of model agents builds a disease dossier from the literature, checks it against the data, proposes
and tests hypotheses, assembles cell-to-cell circuits, and plans experiments and treatments for each one. A principal
investigator agent reviews every stage before the run moves on. Every claim cites the data query or paper it rests on.

The result is a static site with four views: an overview with the circuits, an experiment plan for the bench, the
treatments for the clinic, and the full reasoning record.

## Layout

- `src/mechanist/`: the runner, agents and prompts, data tools, adapters for study outputs, and the site template
- `scripts/`: site and report builders, the knowledge-table builder, status and transcript views
- `tests/`: synthetic studies only, no real data, network or GPU
- `configs/`: templates; per-study configs are kept out of the repository
- `docs/ARCHITECTURE.md`: how the stages fit together

## Running

```bash
uv sync
uv run python scripts/build_kb.py
uv run mechanist run --config configs/your_study.yaml --mode multi
uv run python scripts/build_site.py runs/<run> --portal
uv run pytest -q
```

Start a study config from `configs/skinpipe_template.yaml`.
