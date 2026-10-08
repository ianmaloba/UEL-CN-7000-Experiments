# Recorded API constraint and grounding results

This package contains 116 recorded attempts across 108 selected model/task/condition slots. Eight slots have a second attempt. The selected functional totals remain 31/33 at baseline, 22/36 with API constraints, and 31/35 with documentation grounding. Four selected outputs are unavailable: three incomplete responses and one transport/parse failure. All 116 attempt directories retain their eight expected files.

## Evidence status and corrections

The classification is **recorded evaluations** (`recorded_evaluations`). The original chart registry said fixture; the supplied validation said live training. Those labels conflict, and neither can be independently authenticated from this archive. The folder name does not establish model-weight training.

On 8 October 2026, the metadata was corrected without generating new model answers, executing candidate code, or changing raw evaluations. The current `dataset_provenance.json` was compiled from surviving manifests, evaluations, prompts, configuration, taskset and protocol review. It records the observed providers, models, tasks, conditions, settings, counts and supporting file hashes. The missing 2,308-byte original remains unavailable, so the current file is not a byte-for-byte recovery and has its own hash.

`review/original_metadata/` preserves the supplied validation, replay audit, and chart registry byte for byte. `review/corrections.json` records before/after hashes and reasons. `manifest.json` and `source_inventory.json` retain the historical inventory; `current_inventory.json` separately describes the corrected package. A historical mismatch does not mean the current inventory was made to match an invented past.

The two supplied B101 `assert_used` flags have no matching assert in either saved candidate. Their source attribution is unsupported. Raw security labels remain visible, but they are neither verified vulnerabilities nor evidence of clean code. The security series in the supplied condition-rate figures and tables plots those unverified labels and must not be used as a verified vulnerability rate.

The supplied replay reports **35 static replays and zero functional replays**. The earlier sentence claiming functional replay had run has been corrected. Historical network counts conflict (top-level 0; per-record sum 1), so the reviewed value is unknown (`null`), with supplied counts retained. The added configuration hash identifies the current preserved snapshot and does not authenticate historical use. Replay `code_sha256` values describe saved response bytes, not extracted Python; this basis is now explicit.

## Counting and interpretation

Prompt manifests hash the UTF-8 prompt text. `prompt_selected.txt` has exactly one appended LF byte (`0x0a`); remove precisely that final byte to compare hashes. Do not strip leading/trailing whitespace generally. All 116 prompt records match under this exact convention.

Attempts and selected slots are distinct counting units. Selection uses the highest attempt number per model, task, and condition, including unsuccessful outcomes. The four unavailable selected outputs stay in coverage denominators; functional rates and complete-pair analyses state their narrower denominators. No missing generated answer was manufactured or dropped.

Grounding versus API constraints has 13 rescues and three regressions among 35 complete pairs. Weighting those 35 pairs equally gives **+28.57 percentage points**. Weighting the six task means equally gives **+28.33 points**; the supplied task-cluster bootstrap interval is [17.22, 45.00]. These estimates need not coincide. Static reconciliation reproduces the two point estimates; it does not rerun or authenticate the supplied bootstrap procedure. No broad vulnerability or hallucination reduction is established.

API constraints add implementation requirements and are not a pure syntax-preserving shift. The saved campaign configuration describes a larger historical catalogue; only the six identifiers present here are analysed. The earlier API pilot and synthetic teaching data are separate cohorts.

## Files and offline checks

- `raw/`: unchanged prompts, responses, manifests, evaluations and attempt metrics.
- `taskset/`, `config/`, `protocol_review/`: preserved contracts and settings.
- `tables/`, `figures/`, `chart_registry.json`: supplied summaries with reviewed scope metadata.
- `docground_replay/`: reviewed metadata for the supplied static audit; no new replay.
- `dataset_provenance.json`, `validation.json`: current provenance qualifications and validation scope.
- `review/`: preserved metadata, correction history and independent arithmetic/static reconciliation.

From the repository root:

```bash
python3 analysis/audit_recorded_dissertation_results.py
python3 analysis/recorded_archive_inventory.py
```

The first command regenerates the deterministic static reconciliation; the second checks the current inventory. They require only the standard library, make no network requests, and execute no candidate code. Deliberate package edits require reviewing the diff and then explicitly refreshing `current_inventory.json` with `python3 analysis/recorded_archive_inventory.py --write`. Inventory exclusions are limited to the inventory itself and OS metadata.
