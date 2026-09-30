"""Conservative local checks for generated Python; no check turns an error into a pass."""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from .schema import Task

_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class Evaluation:
    syntax_status: str
    functional_status: str
    security_status: str
    hallucination_status: str
    details: dict[str, object]
    api_conformance_status: str = "not_applicable"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"


def extract_python(response: str) -> str:
    """Extract a single fenced Python block, retaining raw output when not fenced."""
    blocks = _FENCE.findall(response)
    return blocks[0].strip() if blocks else response.strip()


def evaluate_python(task: Task, response: str, timeout_seconds: int = 10) -> Evaluation:
    code = extract_python(response)
    try:
        compile(code, "candidate.py", "exec")
    except SyntaxError as error:
        return Evaluation(
            "fail", "not_run", "not_run", "not_run",
            {"syntax_error": str(error)}, api_conformance_status="not_run",
        )

    with tempfile.TemporaryDirectory(prefix="uel-eval-") as directory:
        root = Path(directory)
        (root / "candidate.py").write_text(code + "\n", encoding="utf-8")
        (root / "test_candidate.py").write_text(task.test_code, encoding="utf-8")
        functional = subprocess.run(
            [sys.executable, "-I", "-m", "pytest", "-q", "test_candidate.py"],
            cwd=root, text=True, capture_output=True, timeout=timeout_seconds, check=False,
        )
        bandit = subprocess.run(
            [sys.executable, "-m", "bandit", "-q", "-f", "json", "candidate.py"],
            cwd=root, text=True, capture_output=True, timeout=timeout_seconds, check=False,
        )
    try:
        bandit_data = json.loads(bandit.stdout or "{}")
        findings = bandit_data.get("results", [])
    except json.JSONDecodeError:
        findings = []
    imports_expected_symbol = f"def {task.expected_symbol}" in code
    return Evaluation(
        syntax_status="pass",
        functional_status="pass" if functional.returncode == 0 else "fail",
        security_status="pass" if bandit.returncode == 0 else "findings",
        hallucination_status="not_detected" if imports_expected_symbol else "suspected",
        details={
            "functional_returncode": functional.returncode,
            "functional_output": (functional.stdout + functional.stderr)[-2000:],
            "bandit_findings": findings,
            "expected_symbol_present": imports_expected_symbol,
        },
        api_conformance_status=check_api_conformance(task, code),
    )


def evaluate_static_python(task: Task, response: str) -> Evaluation:
    """Non-executing check for live outputs until an isolated runner is available."""
    code = extract_python(response)
    try:
        compile(code, "candidate.py", "exec")
    except SyntaxError as error:
        return Evaluation(
            "fail", "not_run", "not_run", "not_run",
            {"syntax_error": str(error)}, api_conformance_status="not_run",
        )
    with tempfile.TemporaryDirectory(prefix="uel-static-") as directory:
        candidate = Path(directory) / "candidate.py"
        candidate.write_text(code + "\n", encoding="utf-8")
        bandit = subprocess.run([sys.executable, "-m", "bandit", "-q", "-f", "json", str(candidate)], text=True, capture_output=True, check=False)
    try:
        findings = json.loads(bandit.stdout or "{}").get("results", [])
    except json.JSONDecodeError:
        findings = []
    return Evaluation(
        "pass", "not_run", "pass" if bandit.returncode == 0 else "findings",
        "not_detected" if f"def {task.expected_symbol}" in code else "suspected",
        {"bandit_findings": findings, "evaluation_mode": "static_only"},
        api_conformance_status=check_api_conformance(task, code),
    )


def evaluate_python_isolated(
    task: Task,
    response: str,
    timeout_seconds: int = 15,
    image: str = "python:3.14-slim",
) -> Evaluation:
    """Run fixture tests only in a read-only, network-disabled Docker container."""
    static = evaluate_static_python(task, response)
    if static.syntax_status != "pass":
        return static
    code = extract_python(response)
    with tempfile.TemporaryDirectory(prefix="uel-container-") as directory:
        root = Path(directory)
        (root / "candidate.py").write_text(code + "\n", encoding="utf-8")
        (root / "test_candidate.py").write_text(task.test_code, encoding="utf-8")
        (root / "runner.py").write_text("import sys\nsys.path.insert(0, '/work')\nimport test_candidate\nfor n in dir(test_candidate):\n    if n.startswith('test_'): getattr(test_candidate, n)()\n", encoding="utf-8")
        command = ["docker", "run", "--rm", "--pull=never", "--network", "none", "--read-only", "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m", "--pids-limit", "64", "--memory", "256m", "--cpus", "0.5", "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--user", "65534:65534", "-v", f"{root}:/work:ro", "-w", "/work", image, "python", "-I", "runner.py"]
        try:
            result = subprocess.run(command, text=True, capture_output=True, timeout=timeout_seconds, check=False)
            output = (result.stdout + result.stderr)[-2000:]
            runtime_failure = re.search(
                r"Cannot connect to the Docker daemon|error during connect|dial unix.*connect|"
                r"No such image|pull access denied|docker: command not found|"
                r"executable file not found",
                output, re.IGNORECASE,
            )
            if result.returncode == 0:
                status = "pass"
            elif runtime_failure:
                status = "evaluator_error"
            else:
                status = "fail"
            details = {
                **static.details, "evaluation_mode": "docker_isolated",
                "functional_returncode": result.returncode, "functional_output": output,
            }
            if runtime_failure:
                details["evaluator_error"] = "docker_runtime_unavailable"
            elif result.returncode != 0:
                details["candidate_test_failure"] = True
        except FileNotFoundError as error:
            status, details = "evaluator_error", {
                **static.details, "evaluation_mode": "docker_isolated",
                "evaluator_error": "docker_executable_missing", "error_type": type(error).__name__,
            }
        except subprocess.TimeoutExpired:
            status, details = "timeout", {**static.details, "evaluation_mode": "docker_isolated", "candidate_test_timeout": True}
    return Evaluation(
        "pass", status, static.security_status, static.hallucination_status,
        details, api_conformance_status=static.api_conformance_status,
    )


def check_api_conformance(task: Task, code: str) -> str:
    """Check task-declared calls separately from functional behavior.

    This is a deliberately narrow static signal, not proof of API correctness.
    Instance methods in the curated task set are matched by imported package and
    attribute name because Python's AST does not infer receiver types.
    """
    if not task.required_apis:
        return "not_applicable"
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "not_run"
    aliases: dict[str, str] = {}
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                modules.add(root)
                aliases[alias.asname or root] = root
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            root = node.module.split(".")[0]
            modules.add(root)
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    calls = {
        _normalize_call(_dotted_call(node.func), aliases)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _dotted_call(node.func)
    }
    attributes = {
        node.attr for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.ctx, ast.Load)
    }
    missing = []
    for api in task.required_apis:
        if api in calls:
            continue
        root, _, suffix = api.partition(".")
        attribute = suffix.rsplit(".", 1)[-1]
        is_instance_method = api.startswith(("pandas.DataFrame.", "requests.Response."))
        if is_instance_method and root in modules and attribute in attributes:
            continue
        missing.append(api)
    return "pass" if not missing else "fail"


def _dotted_call(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_call(node.value)
        return f"{prefix}.{node.attr}" if prefix else ""
    return ""


def _normalize_call(name: str, aliases: dict[str, str]) -> str:
    head, separator, tail = name.partition(".")
    replacement = aliases.get(head, head)
    if not separator:
        return replacement
    return f"{replacement}.{tail}"
