"""Tests for the multi-LoRA additions to rl_collate_fn.

These exercise the all-or-nothing adapter_id → adapter_indices wiring
added in this checkout. Other collate_fn behavior is covered by the
existing test_collate_fn.py.
"""

from __future__ import annotations

import pytest
import torch

from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.interfaces import DatumSpec


def _make_datum(idx: int, adapter_id: int | None = None) -> DatumSpec:
    base: DatumSpec = {
        "message_log": [
            {
                "role": "user",
                "content": "hi",
                "token_ids": torch.tensor([1, 2, 3]),
            }
        ],
        "length": 3,
        "extra_env_info": None,
        "loss_multiplier": 1.0,
        "idx": idx,
        "task_name": "smoke",
    }
    if adapter_id is not None:
        base["adapter_id"] = adapter_id
    return base


def test_no_adapter_id_omits_adapter_indices():
    batch = [_make_datum(i) for i in range(3)]
    out = rl_collate_fn(batch)
    assert "adapter_indices" not in out


def test_all_adapter_ids_stacks_into_indices():
    batch = [_make_datum(0, 0), _make_datum(1, 1), _make_datum(2, 0)]
    out = rl_collate_fn(batch)
    assert "adapter_indices" in out
    assert isinstance(out["adapter_indices"], torch.Tensor)
    assert out["adapter_indices"].dtype == torch.int64
    assert out["adapter_indices"].tolist() == [0, 1, 0]


def test_partial_adapter_id_presence_raises():
    batch = [_make_datum(0, 0), _make_datum(1)]  # second is missing
    with pytest.raises(ValueError, match="all-or-nothing"):
        rl_collate_fn(batch)


def test_single_sample_with_adapter_id():
    batch = [_make_datum(0, 7)]
    out = rl_collate_fn(batch)
    assert out["adapter_indices"].tolist() == [7]
