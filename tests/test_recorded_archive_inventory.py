import hashlib
import json

import pytest

from analysis.recorded_archive_inventory import (
    INVENTORY_NAME,
    InventoryError,
    build_inventory,
    main,
    verify_inventory,
    write_inventory,
)


def test_inventory_is_deterministic_and_preserves_historical_inventory(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "response.txt").write_bytes(b"answer\n")
    (tmp_path / "source_inventory.json").write_bytes(b"historical bytes")
    for parent in (tmp_path, nested):
        for name in (".DS_Store", "Thumbs.db"):
            (parent / name).write_bytes(b"irrelevant OS metadata")

    inventory = write_inventory(tmp_path)
    first = (tmp_path / INVENTORY_NAME).read_bytes()
    write_inventory(tmp_path)
    assert (tmp_path / INVENTORY_NAME).read_bytes() == first
    assert [record["path"] for record in inventory["files"]] == [
        "nested/response.txt", "source_inventory.json"
    ]
    assert inventory["files"][0] == {
        "path": "nested/response.txt",
        "bytes": 7,
        "sha256": hashlib.sha256(b"answer\n").hexdigest(),
    }
    assert (tmp_path / "source_inventory.json").read_bytes() == b"historical bytes"
    assert verify_inventory(tmp_path)["status"] == "passed"


def test_verification_detects_same_size_mutation_missing_and_extra_files(tmp_path):
    (tmp_path / "changed.txt").write_bytes(b"one")
    (tmp_path / "missing.txt").write_bytes(b"missing")
    write_inventory(tmp_path)
    original = (tmp_path / INVENTORY_NAME).read_bytes()
    (tmp_path / "changed.txt").write_bytes(b"two")
    (tmp_path / "missing.txt").unlink()
    (tmp_path / "extra.txt").write_bytes(b"extra")

    report = verify_inventory(tmp_path)
    assert report["status"] == "failed"
    assert report["missing"] == ["missing.txt"]
    assert report["extra"] == ["extra.txt"]
    assert [record["path"] for record in report["changed"]] == ["changed.txt"]
    assert report["changed"][0]["expected"]["bytes"] == 3
    assert report["changed"][0]["observed"]["bytes"] == 3
    assert (tmp_path / INVENTORY_NAME).read_bytes() == original


@pytest.mark.parametrize("path", ["../outside.txt", "/outside.txt", "a/../../outside.txt", "a\\..\\outside.txt", "./file.txt", "a//file.txt"])
def test_verification_rejects_escaping_or_noncanonical_inventory_paths(tmp_path, path):
    inventory = build_inventory(tmp_path)
    inventory["files"] = [{"path": path, "bytes": 0, "sha256": "0" * 64}]
    (tmp_path / INVENTORY_NAME).write_text(json.dumps(inventory))
    with pytest.raises(InventoryError, match="Unsafe or noncanonical"):
        verify_inventory(tmp_path)


@pytest.mark.parametrize("kind", ["file", "directory", "inventory"])
def test_symlinks_are_rejected_without_following_targets(tmp_path, kind):
    root = tmp_path / "archive"
    root.mkdir()
    write_inventory(root)
    outside = tmp_path / "outside"
    if kind == "directory":
        outside.mkdir()
    else:
        outside.write_bytes(b"outside data")
    link = root / (INVENTORY_NAME if kind == "inventory" else "link")
    if kind == "inventory":
        link.unlink()
    link.symlink_to(outside, target_is_directory=kind == "directory")
    with pytest.raises(InventoryError, match="[Ss]ymlink"):
        verify_inventory(root)
    with pytest.raises(InventoryError, match="[Ss]ymlink"):
        write_inventory(root)


def test_cli_requires_explicit_write_and_reports_failure(tmp_path, capsys):
    (tmp_path / "file.txt").write_text("original")
    assert main(["--root", str(tmp_path)]) == 2
    assert not (tmp_path / INVENTORY_NAME).exists()
    assert main(["--root", str(tmp_path), "--write"]) == 0
    assert main(["--root", str(tmp_path)]) == 0
    original = (tmp_path / INVENTORY_NAME).read_bytes()
    (tmp_path / "file.txt").write_text("changed length")
    assert main(["--root", str(tmp_path)]) == 1
    assert (tmp_path / INVENTORY_NAME).read_bytes() == original
    assert '"status": "failed"' in capsys.readouterr().out
