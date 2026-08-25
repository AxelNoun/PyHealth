"""AMP wiring: labs-only matches the notes scripts, default off, bf16 error is actionable."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = [
    REPO / "scripts" / "will" / "condor" / "labs_only" / "run_labs_only_rnn.sh",
    REPO / "scripts" / "will" / "condor" / "labs_notes" / "run_labs_notes_rnn.sh",
    REPO
    / "scripts"
    / "will"
    / "condor"
    / "labs_notes_cxr"
    / "run_labs_notes_cxr_rnn.sh",
]


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.parent.name)
def test_amp_env_defaults_off(script: Path):
    text = script.read_text(encoding="utf-8")
    assert 'USE_AMP="${USE_AMP:-0}"' in text
    assert "AMP_DTYPE=" in text
    assert "--use-amp" in text
    assert "--amp-dtype" in text


def _bash_executable() -> str:
    git_bash = Path(r"C:\Program Files\Git\bin\bash.exe")
    if git_bash.is_file():
        return str(git_bash)
    found = shutil.which("bash")
    assert found, "bash is required to syntax-check the Condor launchers"
    return found


def _bash_path(path: Path) -> str:
    """Path form that Git Bash / MSYS accepts (``/c/Users/...``)."""
    resolved = path.resolve()
    posix = resolved.as_posix()
    if resolved.drive:
        return f"/{resolved.drive[0].lower()}{posix[len(resolved.drive):]}"
    return posix


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.parent.name)
def test_amp_scripts_bash_n(script: Path):
    subprocess.check_call([_bash_executable(), "-n", _bash_path(script)])


def test_bf16_error_names_the_gpu_and_fp16_fallback(monkeypatch):
    from pyhealth.trainer import resolve_amp_dtype

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a, **k: "Tesla V100-SXM2-32GB")
    with pytest.raises(RuntimeError, match="Tesla V100") as caught:
        resolve_amp_dtype("bf16", use_amp=True)
    assert "fp16" in str(caught.value)
