"""Full-sequence Mamba3 SISO with a local BF16 execution boundary.

The upstream kernels require CUDA. Parameters stay in FP32, and outputs are
returned in the caller's dtype so existing heads/losses need no AMP changes.
"""

import torch
from mamba_ssm.modules.mamba3 import Mamba3


class Mamba3SISO(Mamba3):
    def __init__(self, *args, is_mimo=False, **kwargs):
        if is_mimo is not False:
            raise ValueError("Only Mamba3 SISO is supported; set is_mimo=false")
        super().__init__(*args, is_mimo=False, **kwargs)
        # Legacy trainer initialization must not overwrite SSM-specific values.
        for param in self.parameters():
            param._preserve_mamba_init = True

    def forward(self, u, **kwargs):
        if u.device.type != "cuda":
            raise RuntimeError(
                "Mamba3 SISO requires CUDA for forward/backward execution; "
                "there is no CPU fallback. CPU construction/checkpoint loading is supported."
            )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            result = super().forward(u, **kwargs)
        return result.to(dtype=u.dtype)
