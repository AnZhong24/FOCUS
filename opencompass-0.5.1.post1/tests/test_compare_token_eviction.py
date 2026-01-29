import subprocess
import sys
from pathlib import Path


def test_compare_token_eviction_cli():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "opencompass/models/SDAR-8B-Chat-b32/compare_token_eviction.py"

    result = subprocess.run(
        [sys.executable, str(script), "--num-tests", "200", "--seq-len", "64"],
        check=False,
    )

    assert result.returncode == 0

