"""Apply the isolated evaluator to existing local calibration artefacts."""
from __future__ import annotations
import json
from pathlib import Path
from .evaluate import evaluate_python_isolated
from .artefacts import write_metrics
from .schema import RunManifest, TaskSet

def main() -> None:
    tasksets = (
        TaskSet.from_path(Path('configs/pilot_tasks.json')),
        TaskSet.from_path(Path('configs/humaneval_syntax_pilot_v1.json')),
    )
    tasks = {t.task_id: t for taskset in tasksets for t in taskset.tasks}
    updated = 0
    for evaluation_path in Path('results/raw').rglob('evaluation.json'):
        data = json.loads(evaluation_path.read_text())
        if data['details'].get('provider_error'):
            continue
        manifest = json.loads((evaluation_path.parent / 'run_manifest.json').read_text())
        response = (evaluation_path.parent / 'response.txt').read_text()
        result = evaluate_python_isolated(tasks[manifest['task_id']], response)
        preserved = {
            k: v for k, v in data['details'].items()
            if k not in {
                'functional_output', 'functional_returncode', 'candidate_test_failure',
                'candidate_test_timeout', 'bandit_findings', 'evaluation_mode',
                'syntax_error', 'expected_symbol_present',
            }
        }
        result.details.update(preserved)
        evaluation_path.write_text(result.to_json(), encoding='utf-8')
        write_metrics(evaluation_path.parent, RunManifest.model_validate(manifest), result)
        updated += 1
    print(f'updated {updated} isolated evaluations')

if __name__ == '__main__':
    main()
