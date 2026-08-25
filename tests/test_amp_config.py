"""AMP wiring: bf16 error names the GPU and suggests fp16."""

from __future__ import annotations

import pytest
import torch


def test_bf16_error_names_the_gpu_and_fp16_fallback(monkeypatch):
    from pyhealth.trainer import resolve_amp_dtype

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a, **k: "Tesla V100-SXM2-32GB")
    with pytest.raises(RuntimeError, match="Tesla V100") as caught:
        resolve_amp_dtype("bf16", use_amp=True)
    assert "fp16" in str(caught.value)
