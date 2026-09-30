"""Check that preserved HumanEval canonical solutions pass our wrapped tests."""
from __future__ import annotations
import json
import shutil
import subprocess
from pathlib import Path
from experiments.evaluate import evaluate_python_isolated
from experiments.schema import TaskSet

def main() -> None:
    docker = shutil.which("docker")
    if docker is None:
        raise SystemExit("evaluator infrastructure unavailable: Docker CLI is missing")
    try:
        subprocess.run([docker, "info"], check=True, capture_output=True, timeout=20)
    except (OSError, subprocess.SubprocessError) as error:
        raise SystemExit(f"evaluator infrastructure unavailable: Docker daemon is not ready ({type(error).__name__})") from error
    taskset = TaskSet.from_path(Path("configs/humaneval_syntax_pilot_v1.json"))
    results = []
    for task in taskset.tasks:
        evaluation = evaluate_python_isolated(task, task.original_prompt.split("\n\n", 1)[1] + task.reference_solution)
        results.append({"task_id": task.task_id, "syntax": evaluation.syntax_status, "functional": evaluation.functional_status, "security": evaluation.security_status})
    output = Path("artefacts/generated/humaneval-reference-check.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for result in results:
        print(f"{result['task_id']}: syntax={result['syntax']} functional={result['functional']} security={result['security']}")
    if any(result["functional"] != "pass" for result in results):
        raise SystemExit("canonical reference gate failed")

if __name__ == "__main__":
    main()
