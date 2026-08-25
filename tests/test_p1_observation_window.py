"""The observation window is gone from the multimodal MIMIC-IV tasks.

``window_hours`` used to cap (or claim to cap) collection per admission.
It is no longer a constructor argument. Collection is the full stay.
``emitted_data_version`` is 5 so caches from versions 1-4 cannot be reused.
CXR arms must still not skip later stays against the first admission's clock.

Repro::

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. \\
      python -m pytest tests/test_p1_observation_window.py -q
"""

from __future__ import annotations

import inspect
import json
import unittest
import uuid


LAB_TASKS = [
    "LabsMIMIC4",
    "NotesLabsMIMIC4",
    "NotesLabsCXRMIMIC4",
    "CXRMIMIC4",
]


class TestP1ObservationWindow(unittest.TestCase):
    def test_window_hours_kwarg_is_rejected(self):
        from pyhealth.tasks import multimodal_mimic4 as m

        for name in LAB_TASKS:
            with self.assertRaises(TypeError):
                getattr(m, name)(window_hours=24)

    def test_cxr_arms_do_not_skip_later_stays_on_the_first_admit_clock(self):
        from pyhealth.tasks.multimodal_mimic4 import CXRMIMIC4, NotesLabsCXRMIMIC4

        for cls in (NotesLabsCXRMIMIC4, CXRMIMIC4):
            src = inspect.getsource(cls.__call__)
            self.assertNotIn(
                "admission_time >= effective_end",
                src,
                msg=f"{cls.__name__} still drops later stays against first admit + window",
            )

    def test_empty_drop_invalidates_the_cache(self):
        from pyhealth.tasks.multimodal_mimic4 import LabsMIMIC4

        task = LabsMIMIC4()
        self.assertIsNotNone(vars(task).get("emitted_data_version"))
        self.assertGreaterEqual(task.emitted_data_version, 5)

        def cache_key(t, drop_version=False):
            v = dict(vars(t))
            if drop_version:
                v.pop("emitted_data_version", None)
            params = json.dumps(
                {
                    **v,
                    "input_schema": t.input_schema,
                    "output_schema": t.output_schema,
                },
                sort_keys=True,
                default=str,
            )
            return str(uuid.uuid5(uuid.NAMESPACE_DNS, params))

        self.assertNotEqual(cache_key(task), cache_key(task, drop_version=True))

    def test_protocol_default_has_no_window_hours(self):
        from pyhealth.tasks.multimodal_mimic4 import (
            CXRMIMIC4,
            LabsMIMIC4,
            NotesLabsCXRMIMIC4,
            NotesLabsMIMIC4,
        )

        for task in (
            NotesLabsMIMIC4(),
            NotesLabsCXRMIMIC4(),
            LabsMIMIC4(),
            CXRMIMIC4(),
        ):
            self.assertFalse(hasattr(task, "window_hours"))

    def test_discharge_coded_icd_is_not_a_mortality_task(self):
        from pyhealth.tasks import multimodal_mimic4 as m
        from pyhealth.tasks.multimodal_mimic4 import NotesLabsMIMIC4

        self.assertFalse(hasattr(m, "ICDLabsMIMIC4"))
        self.assertFalse(NotesLabsMIMIC4().include_icd)
