"""Create a human-reviewable, hash-locked preview of DocGround prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from docground.grounding.doc_store import DocStore
from docground.grounding.wrapper import ground
from experiments.schema import TaskSet


def build_review(taskset_path: Path) -> dict[str, object]:
    source_bytes = taskset_path.read_bytes()
    taskset = TaskSet.model_validate_json(source_bytes)
    store = DocStore.load()
    tasks = []
    for task in taskset.tasks:
        proposal = ground(task.shifted_prompt, store=store)
        tasks.append({
            "task_id": task.task_id,
            "required_apis": task.required_apis,
            "libraries": list(proposal.libraries),
            "original_prompt": task.original_prompt,
            "shifted_prompt": task.shifted_prompt,
            "suggested_prompt": proposal.text,
            "prompt_hashes": {
                "original": _sha256(task.original_prompt),
                "shifted": _sha256(task.shifted_prompt),
                "suggested": _sha256(proposal.text),
            },
            "documentation_snapshot_sha256": proposal.snapshot_sha256,
            "approval": "pending",
        })
    return {
        "schema_version": "1.0",
        "status": "review_pending",
        "created_at": datetime.now(UTC).isoformat(),
        "taskset_id": taskset.taskset_id,
        "taskset_path": str(taskset_path),
        "taskset_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "docground_version": version("docground"),
        "documentation_snapshot_date": store.snapshot_date,
        "documentation_snapshot_sha256": store.snapshot_sha256,
        "tasks": tasks,
    }


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--taskset", type=Path,
        default=Path("configs/docground_api_pilot_v1_candidate.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/protocol_review/docground_api_pilot_v1_candidate.json"),
    )
    args = parser.parse_args()
    record = build_review(args.taskset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Review preview written to {args.output}")
    print(f"Task set SHA256: {record['taskset_sha256']}")
    for task in record["tasks"]:
        print(f"{task['task_id']}: libraries={','.join(task['libraries']) or 'none'}")


if __name__ == "__main__":
    main()
