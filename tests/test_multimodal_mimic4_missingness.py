"""Missingness / empty-patient behaviour for multimodal MIMIC-IV tasks."""

from __future__ import annotations

import inspect
import logging

import torch
from torch.utils.data import DataLoader

from pyhealth.datasets.utils import collate_fn_dict_with_padding
from pyhealth.processors.stagenet_processor import StageNetTensorProcessor
from pyhealth.processors.tuple_time_text_processor import TupleTimeTextProcessor
from pyhealth.tasks.multimodal_mimic4 import (
    CXRMIMIC4,
    LabsMIMIC4,
    NotesLabsCXRMIMIC4,
    NotesLabsMIMIC4,
)
from tests.conftest import (
    empty_patient,
    labs_only_patient,
    notes_only_patient,
    real_zero_lab_patient,
)

TASKS = (NotesLabsMIMIC4, NotesLabsCXRMIMIC4, LabsMIMIC4, CXRMIMIC4)


def test_empty_patient_is_rejected_by_all_four_tasks(caplog):
    patient = empty_patient()
    caplog.set_level(logging.INFO, logger="pyhealth.tasks.multimodal_mimic4")
    for cls in TASKS:
        assert cls()(patient) == [], cls.__name__
    assert "no data in any modality" in caplog.text


def test_labs_only_patient_survives_collate():
    task = NotesLabsMIMIC4()
    records = [
        task(labs_only_patient(patient_id="a"))[0],
        task(labs_only_patient(patient_id="b"))[0],
    ]
    for record in records:
        assert record["labs"][0], "labs-only patient must keep lab events"
        assert record["admission_note_times"][0] == []

    lab_proc = StageNetTensorProcessor()
    lab_proc.fit(records, "labs")
    note_proc = TupleTimeTextProcessor(tokenizer_model=None)
    processed = [
        {
            "labs": lab_proc.process(record["labs"]),
            "notes": note_proc.process(record["admission_note_times"]),
            "mortality": record["mortality"],
        }
        for record in records
    ]
    batch = next(
        iter(DataLoader(processed, batch_size=2, collate_fn=collate_fn_dict_with_padding))
    )
    times, values = batch["labs"]
    assert torch.is_tensor(times)
    assert torch.is_tensor(values)


def test_notes_only_patient_survives_notes_labs_task():
    records = NotesLabsMIMIC4()(notes_only_patient())
    assert len(records) == 1
    assert records[0]["admission_note_times"][0]
    assert records[0]["labs"][0] == []


def test_real_zero_lab_is_observed():
    record = LabsMIMIC4()(real_zero_lab_patient())[0]
    values = record["labs"][1]
    masks = record["labs_mask"][1]
    assert values[0][0] == 0.0
    assert masks[0][0] is True
    assert any(not flag for flag in masks[0][1:])


def test_dead_text_and_code_tokens_are_gone():
    from pyhealth.tasks import multimodal_mimic4 as module

    src = inspect.getsource(module)
    assert "MISSING_TEXT_TOKEN" not in src
    assert "MISSING_CODE_TOKEN" not in src
    assert "MISSING_FLOAT_TOKEN" in src
