"""Inventory current archive bytes without claiming historical authentication.

Run with --write to create or refresh current_inventory.json after reviewed
changes. Without --write, verify the saved inventory and report differences.
Historical source_inventory.json and manifest.json remain ordinary input files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile


INVENTORY_NAME = "current_inventory.json"
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "results/end_to_end_training_live"
EXCLUSION_POLICY = {
    "root_relative_paths": [INVENTORY_NAME],
    "file_basenames_at_any_depth": [".DS_Store", "Thumbs.db"],
    "symlinks": "Rejected; symlink files and directories are never followed.",
}
SCOPE = (
    "Current archive file bytes only. Matching hashes establish consistency "
    "with this inventory, not historical authenticity or provider execution."
)


class InventoryError(ValueError):
    """The archive or inventory cannot be inspected safely."""


def _root(path: Path) -> Path:
    path = Path(path)
    if path.is_symlink():
        raise InventoryError(f"Archive root is a symlink: {path}")
    if not path.is_dir():
        raise InventoryError(f"Archive root is not a directory: {path}")
    return path.resolve()


def _excluded(relative: str) -> bool:
    return relative == INVENTORY_NAME or PurePosixPath(relative).name in (
        EXCLUSION_POLICY["file_basenames_at_any_depth"]
    )


def _read_regular(path: Path) -> bytes:
    # O_NOFOLLOW closes the final-component symlink race on supported systems.
    if path.is_symlink():
        raise InventoryError(f"Symlink is not allowed: {path}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise InventoryError(f"Not a regular file: {path}")
        return stream.read()


def _file_record(path: Path, relative: str) -> dict:
    data = _read_regular(path)
    return {
        "path": relative,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _scan(root: Path) -> list[dict]:
    records = []

    def visit(directory: Path) -> None:
        with os.scandir(directory) as entries:
            ordered = sorted(entries, key=lambda entry: entry.name)
        for entry in ordered:
            path = Path(entry.path)
            relative = path.relative_to(root).as_posix()
            if entry.is_symlink():
                raise InventoryError(f"Symlink is not allowed: {relative}")
            if entry.is_dir(follow_symlinks=False):
                visit(path)
            elif entry.is_file(follow_symlinks=False):
                if not _excluded(relative):
                    records.append(_file_record(path, relative))
            else:
                raise InventoryError(f"Not a regular file or directory: {relative}")

    visit(root)
    return sorted(records, key=lambda record: record["path"])


def build_inventory(root: Path) -> dict:
    return {
        "schema_version": 1,
        "scope": SCOPE,
        "exclusion_policy": EXCLUSION_POLICY,
        "files": _scan(_root(root)),
    }


def write_inventory(root: Path) -> dict:
    root = _root(root)
    inventory = build_inventory(root)
    # Atomic replacement prevents a partially written inventory on interruption.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=root, delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(inventory, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        os.replace(temporary, root / INVENTORY_NAME)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return inventory


def _validate_inventory(inventory: dict) -> dict[str, dict]:
    if not isinstance(inventory, dict) or inventory.get("schema_version") != 1:
        raise InventoryError("Unsupported inventory schema")
    if inventory.get("exclusion_policy") != EXCLUSION_POLICY:
        raise InventoryError("Inventory exclusion policy does not match this verifier")
    if not isinstance(inventory.get("files"), list):
        raise InventoryError("Inventory files must be a list")
    expected = {}
    for record in inventory["files"]:
        if not isinstance(record, dict):
            raise InventoryError("Inventory file records must be objects")
        relative = record.get("path")
        if not isinstance(relative, str) or not relative:
            raise InventoryError("Inventory path must be a nonempty string")
        path = PurePosixPath(relative)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in relative
            or "\x00" in relative
            or path.as_posix() != relative
            or relative == "."
        ):
            raise InventoryError(f"Unsafe or noncanonical inventory path: {relative!r}")
        if _excluded(relative):
            raise InventoryError(f"Inventory lists an excluded file: {relative}")
        if relative in expected:
            raise InventoryError(f"Duplicate inventory path: {relative}")
        size, digest = record.get("bytes"), record.get("sha256")
        if type(size) is not int or size < 0:
            raise InventoryError(f"Invalid byte count: {relative}")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise InventoryError(f"Invalid SHA-256: {relative}")
        expected[relative] = record
    return expected


def verify_inventory(root: Path) -> dict:
    root = _root(root)
    inventory_path = root / INVENTORY_NAME
    if inventory_path.is_symlink():
        raise InventoryError(f"Inventory is a symlink: {inventory_path}")
    expected = _validate_inventory(json.loads(_read_regular(inventory_path)))
    observed = {record["path"]: record for record in _scan(root)}
    missing = sorted(expected.keys() - observed.keys())
    extra = sorted(observed.keys() - expected.keys())
    changed = [
        {"path": path, "expected": expected[path], "observed": observed[path]}
        for path in sorted(expected.keys() & observed.keys())
        if any(expected[path][key] != observed[path][key] for key in ("bytes", "sha256"))
    ]
    return {
        "status": "failed" if missing or extra or changed else "passed",
        "scope": SCOPE,
        "files_checked": len(observed),
        "missing": missing,
        "changed": changed,
        "extra": extra,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--write", action="store_true", help="explicitly create or refresh the inventory"
    )
    args = parser.parse_args(argv)
    try:
        if args.write:
            inventory = write_inventory(args.root)
            report = {
                "status": "written",
                "inventory": str(args.root / INVENTORY_NAME),
                "files": len(inventory["files"]),
                "scope": SCOPE,
            }
        else:
            report = verify_inventory(args.root)
    except (OSError, ValueError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, indent=2))
        return 2
    print(json.dumps(report, indent=2))
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
