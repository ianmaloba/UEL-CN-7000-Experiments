"""Create a reviewed six-task, syntax-shift HumanEval pilot manifest."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

from experiments.schema import TaskSet

SOURCE = Path("artefacts/generated/HumanEval.jsonl.gz")
DESTINATION = Path("configs/humaneval_syntax_pilot_v1.json")
SOURCE_URL = "https://github.com/openai/human-eval/blob/master/data/HumanEval.jsonl.gz"

SHIFTS = {
    "HumanEval/0": "Provide a complete Python definition of `has_close_elements(numbers, threshold)`. Determine whether any two distinct values in the input list have an absolute difference strictly smaller than the given threshold. Return a boolean. Preserve the callable name and parameter order.",
    "HumanEval/1": "Provide a complete Python definition of `separate_paren_groups(paren_string)`. Ignore spaces, then split a balanced-parentheses sequence into its top-level balanced groups in left-to-right order. Preserve the callable name and parameter order.",
    "HumanEval/2": "Provide a complete Python definition of `truncate_number(number)`. For a positive floating-point input, return only the fractional remainder after its integer part is removed. Preserve the callable name and parameter order.",
    "HumanEval/3": "Provide a complete Python definition of `below_zero(operations)`. Start a running account total at zero; return True exactly when processing the ordered integer operations makes that total negative at any point. Otherwise return False. Preserve the callable name and parameter order.",
    "HumanEval/4": "Provide a complete Python definition of `mean_absolute_deviation(numbers)`. Compute the mean of the numeric list, then return the average absolute distance of its values from that mean. Preserve the callable name and parameter order.",
    "HumanEval/5": "Provide a complete Python definition of `intersperse(numbers, delimeter)`. Return a list with the supplied delimiter placed between each adjacent pair of input elements, without a leading or trailing delimiter. Preserve the callable name and parameter order.",
}


def symbol_from_prompt(prompt: str) -> str:
    for line in prompt.splitlines():
        line = line.strip()
        if line.startswith("def "):
            return line.removeprefix("def ").split("(", 1)[0]
    raise ValueError("no function definition in HumanEval prompt")


def main() -> None:
    selected = {}
    with gzip.open(SOURCE, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["task_id"] in SHIFTS:
                selected[record["task_id"]] = record
    if set(selected) != set(SHIFTS):
        raise RuntimeError("HumanEval source did not contain every selected task")
    tasks = []
    for source_id, record in selected.items():
        symbol = symbol_from_prompt(record["prompt"])
        tasks.append({
            "task_id": source_id.lower().replace("/", "-"), "language": "python",
            "source_reference": f"{SOURCE_URL}#{source_id}", "source_license": "MIT; OpenAI HumanEval",
            "original_prompt": "Provide complete, self-contained Python code for this task:\n\n" + record["prompt"],
            "shifted_prompt": SHIFTS[source_id], "shift_proxy": "syntax",
            "transformation": "reviewed semantic-preserving rewording; callable name and parameter order retained",
            "semantic_invariant": f"Both prompts must define `{symbol}` with the original HumanEval behavior and callable interface.",
            "expected_symbol": symbol,
            "reference_solution": record["canonical_solution"],
            "test_code": "from candidate import " + symbol + "\n\n" + record["test"] + f"\n\ndef test_humaneval_reference_suite():\n    check({symbol})\n",
            "status": "pilot",
        })
    manifest = {"schema_version": "1.0", "taskset_id": "humaneval-syntax-pilot-v1", "source": "OpenAI HumanEval MIT; source hash recorded in artefacts/generated/HumanEval.source.txt", "tasks": tasks}
    TaskSet.model_validate(manifest)
    DESTINATION.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(tasks)} curated tasks to {DESTINATION}")


if __name__ == "__main__":
    main()
