# Study scope and research workflow

## Research aim

Evaluate how LLM-generated code changes across controlled distribution-shift proxies
in functional correctness, static security findings, and API/package hallucination;
then test whether DocGround's reviewed documentation-grounding workflow improves
those outcomes on the same shifted tasks. Relate robustness to model and task
characteristics where public metadata supports a defensible comparison.

The proposal's theory combines covariate shift (Shimodaira; Ben-David et al.) with
shortcut learning (Geirhos et al.). In practical terms: hold the task's required
behavior fixed, alter declared surface or API-distribution cues, then measure paired
outcome changes. A recency, rarity, or syntax proxy is evidence of a controlled
distributional difference; it cannot establish that an item was absent from a
provider's training data.

## Research questions and outcomes

| RQ | Contrast | Primary outcome | Supporting evidence |
| --- | --- | --- | --- |
| RQ1 | Original task vs semantics-preserving shifted task | Functional test pass; paired correctness change and pass@k | Task manifest, prompt hashes, isolated evaluation, paired tables and confidence intervals |
| RQ2 | Model and condition under shift | Bandit/CodeQL rule and severity rates; findings are not proof of exploitability | Tool versions, rule IDs, source lines, reviewed false-positive sample |
| RQ3 | Shifted baseline vs the same shifted task through DocGround | Verified package/API hallucination and security outcomes, with correctness guardrail | Documentation snapshot, prompt decision/revision, model response, API/package provenance |
| RQ4 | Robustness vs model/task characteristics | Associations, with uncertainty | Public model metadata (release/cutoff, size, weights status) and task features |

## Experimental phases

1. **Pipeline calibration.** Validate manifests, provider handling, immutable run
   records, secret scrubbing, Bandit, and the network-isolated correctness runner on
   known fixtures and HumanEval reference solutions. Calibration is method evidence,
   not a dissertation finding.
2. **Model-variant coverage.** Snapshot every accessible model ID and alias returned
   by the five active provider accounts. The current frozen registry includes 149
   primary text/code targets and preserves six search-specialized targets for a
   separate exploratory cohort. Nine catalog-listed OpenAI IDs officially shut
   down before this snapshot are inventoried but excluded from new calls. Coverage
   v1.1 uses one paired HumanEval task per primary target (298 responses, 149 pairs)
   to establish cost and request compatibility across the catalog. An earlier two-task configuration was stopped
   after it exposed output-cap and retired-model issues; its append-only records remain
   an engineering audit, not part of the clean coverage cohort.
3. **Baseline and shift study.** Freeze a reviewed HumanEval/MBPP task set and
   controlled ReCode-style syntax variants. Expand paired baseline/shift runs across
   every eligible model variant, keeping task, provider/model ID, decoding settings,
   evaluator, and replicate policy fixed within each pair. Add rare-library and
   recency tasks as separate reviewed task families; do not imply HumanEval alone
   tests those proxies.
4. **Security and hallucination.** Analyze the same preserved responses with
   versioned static tools and package/API checks. Manually review a sample of
   findings, retain false positives, and report uncertainty and denominators.
5. **DocGround.** After the baseline/shift protocol and task set are stable, implement
   the reusable wrapper in its own repository, pin that installed version here, and
   add `shifted_docground` as a third paired condition. The experiment must capture
   retrieved documentation, proposed and selected prompt, approval state, and
   verification provenance.
6. **Analysis and figures.** Generate all tables, graphs, and study diagrams from
   accepted source data with scripts. Keep calibration, incomplete, failed-provider,
   rejected, and accepted evidence visibly distinct.

## Current execution position

The phases above describe the study design, not a claim that the full planned study
has been completed. After HumanEval coverage, a separately reviewed six-task API
pilot used 68 targets from GLM, DeepSeek, xAI, and Mistral across `baseline`,
`shifted_baseline`, and `shifted_docground`. Its 1,224 configured slots include
incomplete responses and infrastructure failures. The third condition used
DocGround 0.2.0's grounding core and approved partial documentation snapshot, with
the experiment runner handling requests and evaluation. It did not execute the
product's complete live adapter, CLI, verification, and export path.

DocGround 0.2.1 is now released and installed as a wheel. Offline replay checks its
compatibility with the preserved prompts and responses, and can run the original
task tests through the released verifier. Original observations and evaluations
remain unchanged; new verification records are separate. Replay does not add
independent samples or establish equivalence between differing request payloads.
All new model requests are paused because further API credits are unavailable.
The first parallel functional replay had 78 timeouts and 31 unconfirmed cleanup
warnings, so it is retained only as a diagnostic attempt. A later single-worker
replay processed 361 grounded responses, including 354 complete outputs, with zero
network calls, infrastructure errors, or cleanup uncertainties. The new functional
statuses are 317 pass, 37 fail, and seven not run. One replayed Voxtral pandas-merge
response fails the frozen test, where its original run timed out. The 408 grounded
source-record hashes match the prior audit, raw records were not modified, and the
comparison remains `requires_review`. This replay verifies saved outputs locally;
it is not a new model sample or a completed RQ-level evaluation. Counts, commands,
and provenance are in `README.md` and
`results/tables/docground_api_pilot_v1/docground_reuse_audit_0_2_1_serial.json`.

## Model coverage rule

The live registry is account-scoped and time-stamped. It contains provider-advertised
model IDs and aliases; it is not a claim to cover every model that exists globally.
The primary code-generation cohort includes text input/text output models (plus
text-completion models with a suitable endpoint). Audio, image-generation, video,
embedding, OCR/ASR, and moderation-only systems are inventoried with an exclusion
reason because their output task differs. Aliases and pinned snapshots are retained
as separate call targets when the provider advertises them; analysis groups them by
underlying family so they are not mistaken for independent architectures. MiniMax is
excluded from new runs after the account returned HTTP 402; those attempts remain in
the calibration audit trail.

The 2026-09-28 snapshot inventories 234 distinct IDs/aliases. The frozen primary
coverage targets are OpenAI 68, GLM 11, DeepSeek 2, xAI 43, and Mistral 25. Search-
specialized OpenAI variants stay visible in the registry but are not mixed into the
no-tools code-generation cohort; modality-specialized, archived, and non-chat
entries retain explicit exclusion reasons.

The intended active set is all eligible targets in the frozen model registry, not
only one representative per provider. Provider/model IDs, endpoint type, advertised
capabilities, catalog timestamp, pricing source, and model version are recorded.
Unknown size, weights status, or training cutoff stays `unknown`; do not infer it from
model name.

## Failure, retry, and acceptance policy

- A valid model response that fails parsing, tests, security review, or API checks is
  an observed outcome. Preserve it; do not regenerate it until it passes.
- Provider/network/rate-limit errors and evaluator/infrastructure errors are not model
  outcomes. Record each attempt and error class. Retry only under the frozen policy,
  uniformly across model conditions; never overwrite the earlier attempt.
- The coverage runner makes one automatic retry only after an explicit HTTP 429.
  Timeouts and interrupted transports can have uncertain billing, so they are never
  automatically resent. A user-authorized, exact-provider/model repair may append a
  new attempt for an ambiguous transport result only after the budget gate reserves
  its worst-case prior charge; 5xx and access-denied errors remain blocked.
- `--retry-recorded-errors` is an explicit engineering-recovery path, not permission
  to resample a valid but incorrect answer. It preserves every attempt, requires the
  provider and exact model ID, and uses the per-provider ceiling and holdback.
- If a fixture/reference solution or evaluator gate fails, fix the pipeline and rerun
  the affected fixture gate until it passes before using that evaluator version.
- A result is accepted only when the manifest is valid, both paired outputs are
  available or an explicitly allowed error is recorded, evaluation completes, and
  the sample review gate passes. Partial runs remain pilot/incomplete.
- Hard provider spend limits are budget ceilings, not a goal to exhaust credits. Use
  published prices and response token usage for estimates; stop before the ceiling
  and preserve a reserve for retries or reconciliation.

## Interpretation and ethics note

The benchmark task text and test suites are secondary, published data. Outputs made
by sending those tasks to live model APIs are newly generated experimental
observations. The researcher confirmed on 2026-09-28 that the supervisor and UEL
allow the planned experiments. Retain the underlying approval/exemption evidence
with the research records; do not present generated API responses as secondary data.
This distinction belongs in the final methodology.

Generated code runs only in the Docker evaluator configured with no network, no
capabilities, a non-root user, resource limits, and a read-only bind mount. This is
the project's safety control; static-analysis findings are reported as findings,
not as proof that code is exploitable.
