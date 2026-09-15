# Artefact Contract

## Run ID

Use a stable ID:

`<date>__<model>__<condition>__<shift_proxy>__<taskset>__r<replicate>`

Use lowercase ASCII, hyphens inside fields, and double underscores between fields.
Never put API keys, raw prompts, or arbitrary filesystem paths in an ID.

## Required files per accepted run

- `run_manifest.json`: immutable configuration, versions, hashes, and status;
- `prompt_original.txt`: original task prompt;
- `prompt_shifted.txt`: shifted task prompt, if applicable;
- `prompt_grounded.txt`: selected DocGround prompt, if mitigation is active;
- `response.txt`: model response after secret redaction;
- `evaluation.json`: correctness, security, hallucination, syntax, and error fields;
- `interpretation.txt`: restrained interpretation mapped to RQ and limitations.

## Aggregated artefacts

- `results/<run_id>/metrics.csv`: machine-readable per-output metrics;
- `results/tables/table_<number>_<slug>.csv` and `.md`;
- `results/figures/figure_<number>_<slug>.png` and source data;
- `artefacts/screenshots/<run_id>__<slug>.png` with viewport and capture metadata;
- `analysis/<run_id>__interpretation.txt` linking every claim to source files.

Raw responses, screenshots, and provider metadata must be reviewed for secrets before
being committed. Ignored directories are for local runs until a deliberate evidence
selection is made.
