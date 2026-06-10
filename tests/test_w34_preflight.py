from types import SimpleNamespace

import pytest
import torch

from mode_dispatch import _assert_tube_lead_shape


def test_tube_shape_assert_accepts_fourteen_leads():
    model = SimpleNamespace(prediction_leads=tuple(range(15, 29)))
    pred = torch.zeros(1, 14, 2, 3)
    target = torch.zeros(1, 14, 2, 3)
    _assert_tube_lead_shape(model, pred, target)


def test_tube_shape_assert_rejects_wrong_lead_dimension():
    model = SimpleNamespace(prediction_leads=tuple(range(15, 29)))
    pred = torch.zeros(1, 13, 2, 3)
    target = torch.zeros(1, 14, 2, 3)
    with pytest.raises(RuntimeError, match="expected L=14"):
        _assert_tube_lead_shape(model, pred, target)
