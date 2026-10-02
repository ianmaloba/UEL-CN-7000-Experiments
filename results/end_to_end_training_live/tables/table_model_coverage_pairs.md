# Paired model outputs

Pair status mirrors analysis/aggregate_model_coverage.py.

| provider | model_id | task_id | pair_generation_status | grounding_core_pair_generation_status | baseline_functional_status | shifted_baseline_functional_status | shifted_docground_functional_status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| deepseek | deepseek-flash | api-numpy-norm | complete_pair | complete_pair | pass | pass | pass |
| deepseek | deepseek-flash | api-numpy-reshape | complete_pair | complete_pair | pass | pass | pass |
| deepseek | deepseek-flash | api-pandas-groupby | complete_pair | complete_pair | pass | fail | pass |
| deepseek | deepseek-flash | api-pandas-merge | complete_pair | complete_pair | pass | pass | pass |
| deepseek | deepseek-flash | api-requests-get-json | complete_pair | complete_pair | pass | pass | pass |
| deepseek | deepseek-flash | api-requests-post-json | complete_pair | complete_pair | pass | pass | pass |
| glm | glm-4.5 | api-numpy-norm | complete_pair | complete_pair | pass | pass | fail |
| glm | glm-4.5 | api-numpy-reshape | complete_pair | complete_pair | pass | pass | pass |
| glm | glm-4.5 | api-pandas-groupby | complete_pair | complete_pair | pass | fail | pass |
| glm | glm-4.5 | api-pandas-merge | complete_pair | complete_pair | pass | pass | fail |
| glm | glm-4.5 | api-requests-get-json | complete_pair | complete_pair | pass | pass | pass |
| glm | glm-4.5 | api-requests-post-json | complete_pair | complete_pair | pass | pass | pass |
| glm | glm-4.5-air | api-numpy-norm | complete_pair | complete_pair | pass | fail | pass |
| glm | glm-4.5-air | api-numpy-reshape | complete_pair | complete_pair | pass | pass | pass |
| glm | glm-4.5-air | api-pandas-groupby | complete_pair | complete_pair | pass | fail | fail |
| glm | glm-4.5-air | api-pandas-merge | complete_pair | complete_pair | pass | fail | pass |
| glm | glm-4.5-air | api-requests-get-json | complete_pair | complete_pair | pass | fail | pass |
| glm | glm-4.5-air | api-requests-post-json | mixed_generation_pair | complete_pair | not_run | pass | pass |
| mistral | codestral-2508 | api-numpy-norm | complete_pair | complete_pair | fail | fail | pass |
| mistral | codestral-2508 | api-numpy-reshape | complete_pair | complete_pair | pass | pass | pass |
| mistral | codestral-2508 | api-pandas-groupby | complete_pair | complete_pair | pass | fail | pass |
| mistral | codestral-2508 | api-pandas-merge | mixed_generation_pair | complete_pair | not_run | pass | pass |
| mistral | codestral-2508 | api-requests-get-json | complete_pair | complete_pair | pass | fail | pass |
| mistral | codestral-2508 | api-requests-post-json | complete_pair | complete_pair | pass | pass | pass |
| xai | grok-4.20-0309-non-reasoning | api-numpy-norm | complete_pair | complete_pair | pass | pass | pass |
| xai | grok-4.20-0309-non-reasoning | api-numpy-reshape | complete_pair | complete_pair | pass | fail | pass |
| xai | grok-4.20-0309-non-reasoning | api-pandas-groupby | complete_pair | complete_pair | pass | pass | pass |
| xai | grok-4.20-0309-non-reasoning | api-pandas-merge | complete_pair | complete_pair | pass | fail | pass |
| xai | grok-4.20-0309-non-reasoning | api-requests-get-json | complete_pair | complete_pair | pass | fail | pass |
| xai | grok-4.20-0309-non-reasoning | api-requests-post-json | complete_pair | complete_pair | pass | fail | pass |
| xai | grok-4.20-0309-reasoning | api-numpy-norm | infrastructure_error_pair | complete_pair | not_run | pass | pass |
| xai | grok-4.20-0309-reasoning | api-numpy-reshape | complete_pair | mixed_generation_pair | pass | pass | not_run |
| xai | grok-4.20-0309-reasoning | api-pandas-groupby | complete_pair | complete_pair | pass | fail | pass |
| xai | grok-4.20-0309-reasoning | api-pandas-merge | complete_pair | complete_pair | pass | pass | pass |
| xai | grok-4.20-0309-reasoning | api-requests-get-json | complete_pair | complete_pair | pass | pass | fail |
| xai | grok-4.20-0309-reasoning | api-requests-post-json | complete_pair | complete_pair | fail | pass | pass |
