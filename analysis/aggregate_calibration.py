"""Create clearly labelled, local aggregate tables from calibration artefacts."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ACTIVE_PROVIDERS = {"openai", "glm", "deepseek", "xai", "mistral"}
RAW_ROOT = Path("results/raw")
TABLE_ROOT = Path("results/tables")


def load_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for evaluation_path in RAW_ROOT.glob("*/evaluation.json"):
        manifest = json.loads((evaluation_path.parent / "run_manifest.json").read_text())
        evaluation = json.loads(evaluation_path.read_text())
        rows.append({
            "run_id": manifest["run_id"], "provider": manifest["model_provider"],
            "model_id": manifest["model_id"], "task_id": manifest["task_id"],
            "condition": manifest["condition"], "replicate": manifest["replicate"],
            "syntax_status": evaluation["syntax_status"],
            "functional_status": evaluation["functional_status"],
            "security_status": evaluation["security_status"],
            "provider_error": evaluation["details"].get("provider_error"),
        })
    return pd.DataFrame(rows)


def write_table(name: str, frame: pd.DataFrame, note: str) -> None:
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TABLE_ROOT / f"{name}.csv", index=False)
    headers = list(frame.columns)
    markdown = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        markdown.append("| " + " | ".join(str(value) for value in row) + " |")
    (TABLE_ROOT / f"{name}.md").write_text(
        f"# Calibration only — not dissertation evidence\n\n{note}\n\n" + "\n".join(markdown) + "\n", encoding="utf-8")


def main() -> None:
    all_rows = load_rows()
    active = all_rows[all_rows.provider.isin(ACTIVE_PROVIDERS)].copy()
    completion = (all_rows.assign(completed=lambda x: x.provider_error.isna())
                  .groupby("provider", as_index=False)
                  .agg(outputs=("run_id", "size"), completed=("completed", "sum"),
                       provider_errors=("provider_error", lambda x: x.notna().sum())))
    write_table("table_1_calibration_completion", completion,
                "Includes MiniMax to preserve the documented provider failures.")
    correctness = (active.assign(functional_pass=lambda x: x.functional_status.eq("pass"))
                   .groupby(["provider", "model_id", "condition"], as_index=False)
                   .agg(outputs=("run_id", "size"), functional_passes=("functional_pass", "sum")))
    correctness["functional_pass_rate"] = correctness.functional_passes / correctness.outputs
    write_table("table_2_calibration_correctness", correctness,
                "Five active providers only; two repository fixtures per condition, so no inference is warranted.")
    paired = active.pivot(index=["provider", "model_id", "task_id", "replicate"], columns="condition", values="functional_status").reset_index()
    paired["transition"] = paired.apply(lambda row: f"{row.get('baseline', 'missing')}→{row.get('shifted_baseline', 'missing')}", axis=1)
    transitions = paired.groupby(["provider", "model_id", "transition"], as_index=False).size().rename(columns={"size": "paired_tasks"})
    write_table("table_3_calibration_paired_transitions", transitions,
                "Baseline-to-shifted functional transitions for completed active-provider pairs only.")
    print(f"wrote 3 calibration tables from {len(all_rows)} outputs ({len(active)} active-provider outputs)")


if __name__ == "__main__":
    main()
