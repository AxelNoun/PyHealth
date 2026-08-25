"""Decision 1b gate: an empty modality must not leak into the unified sequence.

A padded slot (``pad_mask=False``) is zeroed after encoding. Garbage in that
slot must therefore be indistinguishable from zeros, and a masked mean of the
unified sequence must match a labs-only forward. If this fails, Decision 1b
(keep the cohort, empty sequences + masks, no filler tokens) is invalid.

This test does not download weights. It uses two numeric
``StageNetTensorProcessor`` fields as a stand-in for labs + notes.
"""

from __future__ import annotations

import torch

from pyhealth.models.embedding.unified import UnifiedMultimodalEmbeddingModel
from pyhealth.processors.stagenet_processor import StageNetTensorProcessor


def _masked_mean(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(sequence.dtype)
    denom = weights.sum(dim=1).clamp(min=1.0)
    return (sequence * weights).sum(dim=1) / denom


def _two_field_model() -> UnifiedMultimodalEmbeddingModel:
    labs = StageNetTensorProcessor()
    notes = StageNetTensorProcessor()
    labs.fit([{"labs": ([0.0], [[1.0, 2.0]])}], "labs")
    notes.fit([{"notes": ([0.0], [[1.0, 2.0]])}], "notes")
    model = UnifiedMultimodalEmbeddingModel(
        processors={"labs": labs, "notes": notes},
        embedding_dim=8,
        normalize_content=True,
    )
    model.eval()
    return model


def test_empty_modality_is_neutralised_by_mask():
    model = _two_field_model()

    labs_value = torch.tensor([[[3.0, 4.0]]])
    labs_time = torch.tensor([[1.0]])
    labs_pad = torch.tensor([[True]])

    notes_time = torch.tensor([[2.0]])
    notes_pad = torch.tensor([[False]])
    notes_garbage = torch.tensor([[[999.0, -999.0]]])
    notes_zeros = torch.tensor([[[0.0, 0.0]]])

    labs_field = {"value": labs_value, "time": labs_time, "pad_mask": labs_pad}

    with torch.no_grad():
        out_garbage = model(
            {
                "labs": labs_field,
                "notes": {
                    "value": notes_garbage,
                    "time": notes_time,
                    "pad_mask": notes_pad,
                },
            }
        )
        out_zeros = model(
            {
                "labs": labs_field,
                "notes": {
                    "value": notes_zeros,
                    "time": notes_time,
                    "pad_mask": notes_pad,
                },
            }
        )
        out_labs = model({"labs": labs_field})

    pad_slots_g = out_garbage["sequence"][0, out_garbage["mask"][0] < 0.5]
    pad_slots_z = out_zeros["sequence"][0, out_zeros["mask"][0] < 0.5]
    assert pad_slots_g.numel() > 0
    assert torch.allclose(pad_slots_g, torch.zeros_like(pad_slots_g), atol=1e-5)
    assert torch.allclose(pad_slots_z, torch.zeros_like(pad_slots_z), atol=1e-5)

    mean_g = _masked_mean(out_garbage["sequence"], out_garbage["mask"])
    mean_z = _masked_mean(out_zeros["sequence"], out_zeros["mask"])
    mean_labs = _masked_mean(out_labs["sequence"], out_labs["mask"])
    assert torch.allclose(mean_g, mean_z, atol=1e-5)
    assert torch.allclose(mean_g, mean_labs, atol=1e-5)
    assert torch.isfinite(mean_g).all()
