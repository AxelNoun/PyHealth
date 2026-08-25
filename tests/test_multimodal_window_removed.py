"""The observation-window kwargs are gone from the multimodal MIMIC-IV tasks."""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path
from unittest import mock

import pytest

from pyhealth.tasks.multimodal_mimic4 import (
    CXRMIMIC4,
    LabsMIMIC4,
    NotesLabsCXRMIMIC4,
    NotesLabsMIMIC4,
)
from tests.conftest import h72_lab_patient

TASKS = (LabsMIMIC4, NotesLabsMIMIC4, NotesLabsCXRMIMIC4, CXRMIMIC4)
_WINDOW_RE = re.compile(r"(?<![\w])window_hours")
_TASK_FILE = (
    Path(__file__).resolve().parents[1] / "pyhealth" / "tasks" / "multimodal_mimic4.py"
)
_RUNNER = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "mortality_prediction"
    / "unified_embedding_e2e_mimic4.py"
)
# Distinct 48h early-warning protocol on the MIMIC-IV mortality task; not the
# multimodal observation-window kwargs removed here.
_ALLOWED_INPUT_WINDOW = "pyhealth/tasks/in_hospital_mortality_mimic4.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("e2e_window_removed", _RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _parse(mod, *argv):
    with mock.patch.object(sys, "argv", ["e2e.py", "--ehr-root", "/tmp", *argv]):
        return mod.parse_args()


@pytest.mark.parametrize("cls", TASKS)
def test_window_hours_kwarg_is_rejected(cls):
    with pytest.raises(TypeError):
        cls(window_hours=24)


def test_h72_lab_is_kept_on_full_stay():
    record = LabsMIMIC4()(h72_lab_patient())[0]
    assert record["labs"][0], "a lab at H+72 must be kept once the window is gone"
    assert record["labs"][0][0] == pytest.approx(72.0)


def test_multimodal_task_file_has_no_window_hours():
    text = _TASK_FILE.read_text(encoding="utf-8")
    assert _WINDOW_RE.search(text) is None


def test_e2e_cli_has_no_window_flag():
    captured: list[argparse.ArgumentParser] = []
    real_parse = argparse.ArgumentParser.parse_args

    def capture_parse(self, *args, **kwargs):
        captured.append(self)
        return real_parse(self, *args, **kwargs)

    with mock.patch.object(argparse.ArgumentParser, "parse_args", capture_parse):
        _parse(_load_runner())
    assert captured, "parse_args did not construct an ArgumentParser"
    flags = {
        flag
        for parser in captured
        for action in parser._actions
        for flag in action.option_strings
    }
    assert "--observation-window-hours" not in flags
    assert "--window-hours" not in flags
    with pytest.raises(SystemExit):
        _parse(_load_runner(), "--observation-window-hours", "24")
    with pytest.raises(SystemExit):
        _parse(_load_runner(), "--window-hours", "24")
