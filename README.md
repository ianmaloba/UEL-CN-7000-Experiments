# UEL-CN-7000 Experiments

This repository is the dissertation experiment and evidence project for **Robustness
of LLMs for Code Generation Under Distribution Shift**. It is intentionally separate
from [DocGround](https://github.com/ianmaloba/DocGround), the reusable grounding and
verification wrapper, and from [Draft-DocGround](https://github.com/ianmaloba/Draft-DocGround),
the proof-of-concept baseline.

## Responsibilities

This repository owns benchmark task manifests, perturbations, model-run records,
paired baseline/mitigated outputs, correctness/security/hallucination metrics,
statistical analysis, tables, graphs, screenshots, and dissertation interpretation
notes. DocGround is developed and released separately. The API-task pilot used its
0.2.0 grounding core; the experiment environment now has the released 0.2.1 wheel
installed for offline compatibility and verification checks.

## Research questions

- **RQ1:** How does functional correctness differ between in-distribution and shifted tasks?
- **RQ2:** Do models differ in security-vulnerability rate under shifted conditions?
- **RQ3:** Does documentation-grounded prompting reduce hallucination and vulnerability
  rates relative to baseline prompting under shift?
- **RQ4:** Which model/task characteristics associate with robustness?

## Distribution-shift proxies

Because closed-model training data is not published, the experiment must not claim to
know whether an individual task was literally unseen during training. Shift is instead
operationalised through three declared proxies:

1. **Recency cutoff:** features or APIs released after the model's stated training
   cutoff, with the cutoff and evidence recorded per task.
2. **Rarity:** uncommon libraries, APIs, or usage patterns, with the rarity source and
   threshold recorded rather than asserted informally.
3. **Controlled syntactic perturbation:** semantics-preserving rewording, identifier
   renaming, and format changes based on the ReCode methodology.

Each run records the proxy, transformation, source task, expected invariant, and
review status. These are evidence-relative shift proxies, not proof of training-data
absence.

## Planned run design

The dissertation's later paired study will hold task, decoding settings, seed policy,
and verification configuration constant while comparing:

- baseline: original task prompt;
- shifted baseline: transformed task prompt;
- shifted + DocGround: the same shifted task passed through the reviewed mitigation
  workflow.

The earlier model-coverage stage ran the first two conditions on HumanEval/0 to
check model/endpoint availability and pipeline behavior. The subsequent six-task
API pilot includes all three conditions, with the third using DocGround's approved
grounding proposal and the experiment runner's transport and evaluator. It is a
grounding-core pilot, not a live end-to-end test of DocGround's provider adapters,
interactive CLI, or verification/export workflow. No pilot becomes an accepted
dissertation comparison merely because execution has finished. Manifest integrity,
preserved outputs, evaluation completeness, selection policy, and scope limitations
must be reviewed. Failed runs remain labelled and are never silently overwritten.

## Artefacts and interpretation

See `docs/PLAN.md`, `docs/ARTEFACTS.md`, and `docs/INTERPRETATION_INDEX.txt` before
creating results. Every accepted result gets a stable run ID and mapped files for raw
outputs, metrics, tables, figures, screenshots, and a plain-text interpretation note.
Interpretations describe what the data supports and what it does not support; they do
not turn anecdotal smoke tests into dissertation findings.

## Initial setup

```bash
python3 -m venv .venv
./.venv/bin/pip install -e '.[dev]'
# Obtain the wheel from the private DocGround v0.2.1 release first.
./.venv/bin/pip install ../DocGround/dist/docground-0.2.1-py3-none-any.whl
./.venv/bin/python -m pytest -q
```

## Current implementation status

The framework validates task manifests and shift metadata, writes append-only run
artefacts, redacts configured secret values, and separates syntax, functional,
security, hallucination, provider, and evaluator outcomes. Historical calibration
and HumanEval coverage remain distinct from the API pilot.

The API pilot's frozen configuration is
`configs/docground_api_pilot_v1_recovery_512.json`: 68 model IDs/aliases across GLM,
DeepSeek, xAI, and Mistral, six tasks, and three conditions, giving 1,224 planned
slots. It preserves 2,479 attempts across its original and recovery configurations.
At the selected settings there are 1,032 complete generations, 117 incomplete
responses, 72 provider errors, three transport errors, and no missing slots.
Original functional outcomes are 873 pass, 159 fail, two timeout, and 190 not run.
These are execution counts, not 1,224 successful experiments or independent models.
See `results/tables/docground_api_pilot_v1/` and
`results/figures/docground_api_pilot_v1/` for derived outputs.

DocGround [v0.2.1](https://github.com/ianmaloba/DocGround/releases/tag/v0.2.1)
provides a tested wheel and source archive. Its grounding prompts and documentation
snapshot match the pilot's historical 0.2.0 records. The reuse audit verifies all
1,224 selected prompt/manifests and identifies 361 nonempty grounded responses that
can be replayed locally. The other 47 grounded slots have no response to replay.
Recorded-response replay is a new local verification of saved observations, not a
new model sample or proof that different live adapter payloads would behave alike.

All new paid model requests are paused because API credits are unavailable. Existing
outputs, including failures and truncations, are retained unchanged. To check reuse
without model requests:

```bash
./.venv/bin/python -m analysis.audit_docground_reuse \
  --output results/tables/docground_api_pilot_v1/docground_reuse_audit_0_2_1_static.json
# Optional functional replay requires a running Docker daemon and the original
# local evaluator image uel-cn7000/docground-api-eval:py3.14.7-v1.
./.venv/bin/python -m analysis.audit_docground_reuse --functional --workers 1 \
  --output results/tables/docground_api_pilot_v1/docground_reuse_audit_0_2_1_new.json
```

The audit keeps original evaluations separate from current verification and records
extraction differences, fixture results, infrastructure failures, and missing
responses. Choose a new output filename for each run; existing audits cannot be
overwritten. The initial parallel replay is retained as a diagnostic attempt, not
as a final functional result: it recorded 78 timeouts, including 31 cleanup warnings
that were not confirmed at run time. A later single-worker replay completed all 354
available grounded responses, with zero infrastructure errors, zero cleanup
uncertainties, and zero network calls. Its 317 pass, 37 fail, and seven not-run
outcomes include one status that differs from the original evaluation: a Mistral
Voxtral pandas-merge response failed the frozen test when replayed, whereas its
original evaluation had timed out. All 408 grounded source-record hashes match the
earlier audit and raw records remain unchanged. The functional comparison therefore
remains `requires_review`, not equivalent or accepted evidence. The audit's
top-level `audit_status` describes integrity processing only. The parallel attempt
and its diagnostics are preserved separately. See `docs/STUDY_SCOPE.md` and
`docs/PILOT_PROTOCOL.md` for interpretation and acceptance rules.
