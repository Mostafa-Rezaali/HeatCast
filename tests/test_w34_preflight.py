from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from mesh_backbone import MeshFlowNet
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


class ToyTubeDecoder(MeshFlowNet):
    def __init__(self, chunk_size):
        nn.Module.__init__(self)
        self.tube_decode_chunk_size = chunk_size
        self.tube_num_leads = 4
        self.img_channels = 2
        self.distributional_head = True

    def _decode_grid_output(self, mesh_features, grid_skip, t_emb, mesh, height, width):
        del mesh, height, width
        values = mesh_features.mean(dim=1) + grid_skip.mean(dim=1) + t_emb
        return values[:, :, None, None]


def test_chunked_tube_decode_matches_full_decode_and_gradients():
    tube_h = torch.randn(1, 4, 3, 2, requires_grad=True)
    grid_skip = torch.randn(1, 5, 2, requires_grad=True)
    t_emb = torch.randn(1, 2, requires_grad=True)
    lead_t_emb = torch.randn(4, 2, requires_grad=True)

    full = ToyTubeDecoder(chunk_size=0)
    full.train()
    full_out = full._decode_tube_output(
        tube_h, grid_skip, t_emb, lead_t_emb, object(), 1, 1
    )
    full_out.sum().backward()
    full_grads = tuple(value.grad.detach().clone() for value in (tube_h, grid_skip, t_emb, lead_t_emb))

    for value in (tube_h, grid_skip, t_emb, lead_t_emb):
        value.grad = None
    chunked = ToyTubeDecoder(chunk_size=2)
    chunked.train()
    chunked_out = chunked._decode_tube_output(
        tube_h, grid_skip, t_emb, lead_t_emb, object(), 1, 1
    )
    chunked_out.sum().backward()
    chunked_grads = tuple(value.grad.detach().clone() for value in (tube_h, grid_skip, t_emb, lead_t_emb))

    assert torch.equal(full_out, chunked_out)
    for full_grad, chunked_grad in zip(full_grads, chunked_grads):
        assert torch.equal(full_grad, chunked_grad)
