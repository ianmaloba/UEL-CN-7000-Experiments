"""Fetch the official HumanEval data locally and record a content hash for curation."""
from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen

URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
OUT = Path("artefacts/generated/HumanEval.jsonl.gz")
META = Path("artefacts/generated/HumanEval.source.txt")


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(URL, timeout=30) as response:  # noqa: S310
        content = response.read()
    OUT.write_bytes(content)
    META.write_text(
        f"source_url={URL}\nretrieved_at={datetime.now(UTC).isoformat()}\nsha256={sha256(content).hexdigest()}\nlicense=MIT (OpenAI human-eval repository)\n",
        encoding="utf-8",
    )
    print(f"fetched {len(content)} bytes; sha256={sha256(content).hexdigest()}")


if __name__ == "__main__":
    main()
