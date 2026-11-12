import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.ops  # noqa: E402
import comfy.rmsnorm  # noqa: E402


def _patch_torch_npu_module_cast(monkeypatch):
    """Mirror torch_npu/utils/_module.py: Module.cast_weight(device) exists and Module.to() calls it before moving weights."""
    calls = []

    def cast_weight(self, device):
        calls.append((self, device))

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        with torch.no_grad():
            self.cast_weight(device)

        def convert(t):
            return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None, non_blocking)

        return self._apply(convert)

    monkeypatch.setattr(torch.nn.Module, "cast_weight", cast_weight, raising=False)
    monkeypatch.setattr(torch.nn.Module, "to", to)
    return calls


@pytest.mark.parametrize("ops_module", [comfy.ops.disable_weight_init, comfy.ops.manual_cast])
def test_rmsnorm_to_dispatches_to_module_cast_weight(monkeypatch, ops_module):
    # torch_npu moves modules through Module.cast_weight(device); the weight-cast context helper
    # must not absorb those calls (it expects the activation tensor, see #17327).
    calls = _patch_torch_npu_module_cast(monkeypatch)
    norm = ops_module.RMSNorm(16)
    norm.to("cpu")
    assert calls == [(norm, torch.device("cpu"))]


def test_rmsnorm_forward_comfy_cast_weights_still_casts():
    norm = comfy.ops.manual_cast.RMSNorm(16)
    x = torch.randn(2, 16)
    expected = comfy.rmsnorm.rms_norm(x, norm.weight, norm.eps)
    assert torch.allclose(norm.forward_comfy_cast_weights(x), expected)


def test_rmsnorm_fused_input_act_still_casts():
    norm = comfy.ops.manual_cast.RMSNorm(16)
    x = torch.randn(2, 16)
    expected = comfy.rmsnorm.rms_norm(x, norm.weight, norm.eps)
    assert torch.allclose(comfy.ops._eager_input_act(x, "rms_norm", act_weight=norm), expected)
