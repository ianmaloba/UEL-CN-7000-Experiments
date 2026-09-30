# Artefact Contract

## Run ID

Use a stable ID:

`<date>__<model>__<condition>__<shift_proxy>__<taskset>-<task>__r<replicate>`

Use lowercase ASCII, hyphens inside fields, and double underscores between fields.
Never put API keys, raw prompts, or arbitrary filesystem paths in an ID. The task
suffix prevents per-task responses in the same task set from colliding.
Attempt 1 uses the base ID; later append-only attempts add `__a2`, `__a3`, and so on.
Never overwrite a failed, truncated, or interrupted attempt.

## Required files per accepted run

- `run_manifest.json`: immutable configuration, versions, hashes, and status;
- `prompt_original.txt`: original task prompt;
- `prompt_shifted.txt`: shifted task prompt, if applicable;
- `prompt_grounded.txt`: selected DocGround prompt, if mitigation is active;
- `response.txt`: model response after secret redaction;
- `evaluation.json`: correctness, security, hallucination, syntax, and error fields;
- `interpretation.txt`: restrained interpretation mapped to RQ and limitations.

The manifest records the actual route, per-model reasoning effort, output-token cap,
and attempt number. Evaluation metadata records HTTP status, request ID, provider usage,
finish status, retryability, and an explicit unknown-billing marker when an in-flight
request is interrupted.

## Aggregated artefacts

- `results/<run_id>/metrics.csv`: machine-readable per-output metrics;
- `results/tables/table_<number>_<slug>.csv` and `.md`;
- `results/figures/figure_<number>_<slug>.png` (or `figure_<slug>.png`) and source data;
- `artefacts/screenshots/<run_id>__<slug>.png` with viewport and capture metadata;
- `analysis/<run_id>__interpretation.txt` linking every claim to source files.

The earlier one-task coverage audit is regenerated with
`python -m analysis.aggregate_model_coverage --config configs/model_variant_coverage_v1_1.json`.
It writes `table_model_coverage_provider`, `table_model_coverage_pairs`, and
`table_model_coverage_attempts` under `results/tables/`, plus three descriptive
provider-level figures under `results/figures/`. These execution-audit outputs include
partial/error outcomes and are not accepted dissertation evidence; final inferential
figures must be regenerated from the accepted multi-task dataset.

The six-task API grounding-core pilot uses
`configs/docground_api_pilot_v1_recovery_512.json`. Its tables are under
`results/tables/docground_api_pilot_v1/`, and four descriptive figures are under
`results/figures/docground_api_pilot_v1/`, each in PNG and PDF:

- `figure_model_coverage_generation_status`: completion, truncation, and request errors;
- `figure_model_coverage_functional_status`: original functional evaluation outcomes;
- `figure_model_coverage_paired_status`: original/shifted generation availability;
- `figure_model_coverage_docground_paired_status`: shifted/grounding-core generation availability.

These figures are execution summaries, not evidence of full live-wrapper efficacy.
`docground_reuse_audit.json` preserves the initial static audit.
`docground_reuse_audit_0_2_1.json` preserves the first parallel replay, including
timeouts and cleanup warnings that were not confirmed at run time. The later
`docground_reuse_audit_0_2_1_serial.json` completed all available replays with zero
infrastructure errors, zero cleanup uncertainties, and zero network calls. It
reports 317 pass, 37 fail, and seven not-run results, with one functional status
differing from the original: a Voxtral pandas-merge response is a reproducible
failure, where the original evaluation timed out. All 408 grounded source-record
hashes match the earlier audit and raw records remain unchanged. The comparison
status is `requires_review`, not equivalence. Preserve both replay attempts and
their settings. Never replace historical model responses or evaluations to make
the replay agree.

Raw responses, screenshots, and provider metadata must be reviewed for secrets before
being committed. Ignored directories are for local runs until a deliberate evidence
selection is made.
