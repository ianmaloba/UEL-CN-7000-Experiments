# Recorded API constraint and grounding results

This package preserves the supplied run records, evaluations, summaries and figures for six model targets, six tasks and three conditions. It contains 108 selected slots and 116 attempts. The recorded functional totals are 31/33 at baseline, 22/36 with API constraints and 31/35 with documentation grounding.

## Evidence status

These are **supplied recorded evaluations, not independently authenticated live measurements**. The original source inventory references a missing `dataset_provenance.json`. The chart registry calls the records fixtures, while `validation.json` describes live training. The validation and replay-audit files differ from their original inventory hashes. Original records and inventories are preserved unchanged so the discrepancies remain inspectable.

The two recorded security flags are B101 `assert_used` findings. Neither corresponding saved response contains an assert statement. They are unverified scanner labels, **not proven vulnerabilities**. The supplied replay audit records static verification, with no newly executed functional tests.

## Interpretation

Grounding versus API constraints has 13 rescues and three regressions among 35 complete pairs. The pooled paired effect is +28.57 percentage points. The supplied equal-task effect is +28.33 points; its reported 95% task-cluster bootstrap interval is [17.22, 45.00]. These estimators weight tasks differently and should not be conflated. The API-conformance change is less certain, and no broad vulnerability or hallucination reduction is established.

API constraints add implementation requirements and are not a pure syntax-preserving distribution-shift test. The saved campaign configuration describes a larger historical catalogue; only the six identifiers present in this package are analysed. No model-weight training is established by the folder name.

## Navigation and reconciliation

- `raw/`: preserved prompts, responses, manifests and evaluations for every attempt.
- `taskset/`, `config/`, `protocol_review/`: task contracts, settings and grounding prompts.
- `tables/`, `figures/`, `chart_registry.json`: supplied derived results and their mappings.
- `docground_replay/`: supplied static replay record, not a new generation cohort.
- `manifest.json`, `source_inventory.json`: original inventories; retain their historical discrepancies.
- `review/record_reconciliation.json`: separate arithmetic, prompt-hash and static-source review.

Run `python analysis/audit_recorded_dissertation_results.py` from the repository root to regenerate the separate review. It uses the standard library, makes no requests and executes no candidate code. The original package validation is not silently rewritten. OS metadata such as `.DS_Store` is omitted from publication; original inventories may continue to list it.

The earlier API pilot and synthetic teaching data are separate cohorts and are not pooled here. Dissertation files are intentionally excluded from this publication.
