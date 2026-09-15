# Dissertation Experiment Plan

## Gate 0: research contract

Before collecting results, freeze the task sources, model list, provider/model IDs,
decoding parameters, run count, shift transformations, evaluation tools, and analysis
plan. Store the contract in a versioned manifest. Changes after data collection need a
new manifest version and an explicit rationale.

## Gate 1: task and shift validity

Start with a pilot subset from HumanEval/MBPP or another approved secondary source.
For each task, preserve the original prompt, reference solution/tests, shifted prompt,
shift proxy, transformation, and a human-readable invariant check. Reject a shifted
task if its semantics change, its test becomes ambiguous, or the shift label cannot be
defended from recorded evidence.

Shift proxies:

- `recency`: post-cutoff library/API feature, documented release date and model cutoff;
- `rarity`: uncommon library/API, documented frequency/source and threshold;
- `syntax`: controlled rewording, identifier renaming, or formatting perturbation based
  on ReCode, with semantic equivalence checked.

## Gate 2: generation comparability

For each task/model/replicate, run paired conditions with the same task content and
settings: in-distribution baseline, shifted baseline, and shifted + DocGround. Save
provider, model, timestamp, temperature/top-p/max tokens, seed if supported, prompt
hashes, documentation snapshot, and response metadata without secrets.

A transient provider failure is a failed run, not a missing zero. Retry policy and all
attempts are recorded. No manual cherry-picking of outputs is allowed.

## Gate 3: evaluation

Evaluate every output with:

- functional correctness: isolated tests and pass/fail, later aggregated as pass@k;
- security: Bandit first, CodeQL/Semgrep integration where approved, severity and rule;
- hallucination: import/package and documented API cross-reference with provenance;
- validity: parse status, timeout, missing output, and evaluator errors.

The evaluator must distinguish model failure, provider failure, evaluator failure, and
unsupported task. It must never silently convert an error into a pass or fail.

## Gate 4: quality and stopping rules

A pilot is only accepted when paired runs are complete, outputs are reproducible under
the run manifest, evaluator fixtures detect known positive/negative cases, and manual
review of a sample finds no unresolved schema or metric errors. Do not advance to the
next RQ while a quality gate fails; revise the metric or protocol and rerun the pilot.

The final sample size and repeated-run count will be justified from cost, power, and
supervisor-approved scope. The pilot is not a dissertation result.

## Gate 5: analysis and evidence

Produce paired contingency tables, confidence intervals, effect sizes, and uncertainty
for correctness, security, and hallucination. Use a predeclared paired test and report
multiple-comparison handling where applicable. Figures and tables are generated from
versioned result data, never hand-edited.

Every accepted output must map to `docs/INTERPRETATION_INDEX.txt` with a restrained
statement of what it supports, limitations, and the relevant research question.
