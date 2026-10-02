"""CPU-only Mamba mixer stack, vendored from mamba-ssm.

Source: https://github.com/state-spaces/mamba at tag v1.2.0 (commit
2845255a95f7e4a76fe649ec0532178fe3015dd0), files ``mamba_ssm/modules/mamba_simple.py``,
``mamba_ssm/models/mixer_seq_simple.py`` and ``mamba_ssm/ops/selective_scan_interface.py``.
Copyright (c) 2023, Tri Dao, Albert Gu. Licensed under Apache-2.0 (see
THIRD_PARTY_NOTICES.md).

Why this exists. The Chemical Dice model runs through ``mamba_ssm.MixerModel``. The
``mamba-ssm`` package cannot be imported on a machine without a GPU (its Triton
layernorm module autotunes at import time) and has to be compiled with nvcc, and Ersilia
images are CPU-only. This file carries only the code path the model uses:

* ``rms_norm=False`` and ``fused_add_norm=False``, so plain ``nn.LayerNorm`` is used;
* ``causal_conv1d`` is not installed in the authors' deployment either, so the short
  convolution is a plain ``nn.Conv1d``;
* the CUDA ``selective_scan_fn`` kernel is replaced by mamba-ssm's own pure-PyTorch
  reference implementation, ``selective_scan_ref``, copied below.

Changes from upstream, all of them deletions or the substitution above: the generation
cache and ``step`` decoding code, the fused CUDA/Triton paths, the weight
initialisation (every tensor is overwritten by the checkpoint, which is loaded with
``strict=True``), complex-valued and grouped ``B``/``C`` branches of the scan. Module and
parameter names are unchanged, so upstream state dicts load as they are.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False):
    """Pure-PyTorch selective scan (real ``A``, input-dependent ``B`` and ``C``).

    u: (B D L), delta: (B D L), A: (D N), B: (B N L), C: (B N L), D: (D),
    z: (B D L), delta_bias: (D). Returns (B D L).
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    B = B.float()
    C = C.float()
    x = A.new_zeros((batch, dim, dstate))
    ys = []
    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        y = torch.einsum("bdn,bn->bd", x, C[:, :, i])
        ys.append(y)
    y = torch.stack(ys, dim=2)  # (batch dim L)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    return out.to(dtype=dtype_in)


class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        layer_idx=None,
    ):
        super().__init__()
        # dt_min, dt_max, dt_init, dt_scale and dt_init_floor only matter for
        # initialisation, which the checkpoint overwrites; they are accepted so the
        # IBM configuration can be passed through unchanged.
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.activation = "silu"
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.zeros(self.d_inner, self.d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

    def forward(self, hidden_states):
        """hidden_states: (B, L, D) -> (B, L, D)."""
        batch, seqlen, dim = hidden_states.shape

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        x, z = xz.chunk(2, dim=1)
        x = self.act(self.conv1d(x)[..., :seqlen])

        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_ref(
            x,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=z,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )
        y = rearrange(y, "b d l -> b l d")
        return self.out_proj(y)


class Block(nn.Module):
    """Add -> LayerNorm -> Mamba mixer, returning (hidden_states, residual)."""

    def __init__(self, dim, mixer_cls, norm_cls=nn.LayerNorm):
        super().__init__()
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)

    def forward(self, hidden_states, residual=None):
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        hidden_states = self.mixer(hidden_states)
        return hidden_states, residual


class MixerModel(nn.Module):
    def __init__(self, d_model, n_layer, vocab_size, ssm_cfg=None, norm_epsilon=1e-5):
        super().__init__()
        ssm_cfg = {} if ssm_cfg is None else ssm_cfg
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                Block(
                    d_model,
                    lambda dim, i=i: Mamba(dim, layer_idx=i, **ssm_cfg),
                    norm_cls=lambda dim: nn.LayerNorm(dim, eps=norm_epsilon),
                )
                for i in range(n_layer)
            ]
        )
        self.norm_f = nn.LayerNorm(d_model, eps=norm_epsilon)

    def forward(self, input_ids):
        hidden_states = self.embedding(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)
        residual = (hidden_states + residual) if residual is not None else hidden_states
        return self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
