"""Synthetic MIMIC-IV patients for multimodal task tests. No PhysioNet I/O."""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any, Optional

import polars as pl
import pytest

ADMIT = datetime(2180, 5, 6, 8, 0, 0)
DISCHARGE = ADMIT + timedelta(days=5)
H72 = ADMIT + timedelta(hours=72)
SODIUM_ITEMID = "50983"
HADM_ID = 1001

_LAB_SCHEMA = {
    "timestamp": pl.Datetime,
    "labevents/itemid": pl.Utf8,
    "labevents/valuenum": pl.Float64,
    "labevents/storetime": pl.Utf8,
}


def _empty_labs() -> pl.DataFrame:
    return pl.DataFrame(schema=_LAB_SCHEMA)


def _labs_df(timestamp: datetime, valuenum: float) -> pl.DataFrame:
    store = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    return pl.DataFrame(
        {
            "timestamp": [timestamp],
            "labevents/itemid": [SODIUM_ITEMID],
            "labevents/valuenum": [valuenum],
            "labevents/storetime": [store],
        },
        schema=_LAB_SCHEMA,
    )


def _admission(expire_flag: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        timestamp=ADMIT,
        dischtime=DISCHARGE.strftime("%Y-%m-%d %H:%M:%S"),
        hadm_id=HADM_ID,
        hospital_expire_flag=expire_flag,
    )


def _discharge_note() -> SimpleNamespace:
    return SimpleNamespace(
        text="Chief Complaint:\n\nchest pain\n\n",
        timestamp=ADMIT + timedelta(days=4),
        hadm_id=HADM_ID,
    )


def _cxr_event() -> SimpleNamespace:
    return SimpleNamespace(
        image_path="/tmp/fake_cxr.jpg",
        timestamp=ADMIT + timedelta(hours=12),
    )


class FakePatient:
    """Minimal patient standing in for ``patient.get_events``."""

    def __init__(
        self,
        patient_id: str = "p0",
        *,
        has_patients: bool = True,
        admissions: Optional[list] = None,
        labs: Optional[pl.DataFrame] = None,
        notes: Optional[list] = None,
        cxr: Optional[list] = None,
    ) -> None:
        self.patient_id = patient_id
        self._patients = [SimpleNamespace()] if has_patients else []
        self._admissions = admissions if admissions is not None else [_admission()]
        self._labs = labs if labs is not None else _empty_labs()
        self._notes = notes or []
        self._cxr = cxr or []

    def get_events(
        self,
        event_type: str,
        start: Any = None,
        end: Any = None,
        filters: Any = None,
        return_df: bool = False,
    ):
        if event_type == "patients":
            return self._patients
        if event_type == "admissions":
            return self._admissions
        if event_type == "labevents":
            return self._labs
        if event_type in ("discharge", "radiology"):
            events = [n for n in self._notes if event_type == "discharge"]
            if event_type == "radiology":
                events = []
            if filters:
                for field, _op, value in filters:
                    events = [e for e in events if getattr(e, field, None) == value]
            if start is not None:
                events = [e for e in events if e.timestamp >= start]
            if end is not None:
                events = [e for e in events if e.timestamp <= end]
            return events
        if event_type == "metadata":
            events = list(self._cxr)
            if start is not None:
                events = [e for e in events if e.timestamp >= start]
            if end is not None:
                events = [e for e in events if e.timestamp <= end]
            return events
        if event_type in ("diagnoses_icd", "procedures_icd"):
            return []
        return _empty_labs() if return_df else []


def empty_patient() -> FakePatient:
    return FakePatient(patient_id="empty")


def labs_only_patient(
    patient_id: str = "labs_only", *, valuenum: float = 140.0, at: datetime = ADMIT
) -> FakePatient:
    return FakePatient(
        patient_id=patient_id,
        labs=_labs_df(at, valuenum),
    )


def notes_only_patient() -> FakePatient:
    return FakePatient(patient_id="notes_only", notes=[_discharge_note()])


def full_patient() -> FakePatient:
    return FakePatient(
        patient_id="full",
        labs=_labs_df(ADMIT, 140.0),
        notes=[_discharge_note()],
        cxr=[_cxr_event()],
    )


def h72_lab_patient() -> FakePatient:
    return FakePatient(patient_id="h72", labs=_labs_df(H72, 141.0))


def real_zero_lab_patient() -> FakePatient:
    return FakePatient(patient_id="zero", labs=_labs_df(ADMIT, 0.0))


@pytest.fixture
def patient_empty():
    return empty_patient()


@pytest.fixture
def patient_labs_only():
    return labs_only_patient()


@pytest.fixture
def patient_notes_only():
    return notes_only_patient()


@pytest.fixture
def patient_full():
    return full_patient()
