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
notes. It will install DocGround as a dependency rather than copy its implementation.

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

For every eligible task and model, hold task, decoding settings, seed policy, and
verification configuration constant while comparing:

- baseline: original task prompt;
- shifted baseline: transformed task prompt;
- shifted + DocGround: the same shifted task passed through the reviewed mitigation
  workflow.

No result is accepted into the dissertation dataset until the run manifest is valid,
the output is preserved, the checks complete, and the paired comparison passes its
quality gates. Failed runs remain labelled and are never silently overwritten.

## Artefacts and interpretation

See `docs/PLAN.md`, `docs/ARTEFACTS.md`, and `docs/INTERPRETATION_INDEX.txt` before
creating results. Every accepted result gets a stable run ID and mapped files for raw
outputs, metrics, tables, figures, screenshots, and a plain-text interpretation note.
Interpretations describe what the data supports and what it does not support; they do
not turn anecdotal smoke tests into dissertation findings.

## Initial setup

```bash
python3 -m venv venv
./venv/bin/pip install -e .
./venv/bin/python -m pytest -q
```

The actual benchmark harness will be added incrementally after the experiment schema
and quality gates are reviewed. No API key is required for this initial scaffold.
