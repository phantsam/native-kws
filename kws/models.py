"""NativeKWS: broadcast-residual conv front-end + selective state-space (Mamba) temporal back-end.

Design goals
  * accuracy competitive with large SOTA KWS models at a few hundred K parameters;
  * every block has an embedded-friendly inference form:
      - conv part uses only (depthwise) Conv2d, BatchNorm (foldable), Add, AvgPool,
        SubSpectralNorm (per-sub-band affine at inference);
      - the SSM is a diagonal recurrence h_t = a_t * h_{t-1} + b_t, i.e. constant
        memory and O(1) work per new frame when run causally (streaming).

The bidirectional variant is used for offline 1 s benchmarks; causal=True gives the
streaming model that will run on the microcontroller.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- conv front-end
class SubSpectralNorm(nn.Module):
    """BatchNorm with separate statistics for S frequency sub-bands (Chang et al. 2021)."""

    def __init__(self, channels, sub_bands):
        super().__init__()
        self.s = sub_bands
        self.bn = nn.BatchNorm2d(channels * sub_bands)

    def forward(self, x):
        b, c, f, t = x.shape
        x = x.reshape(b, c * self.s, f // self.s, t)
        return self.bn(x).reshape(b, c, f, t)


def conv_bn(cin, cout, k, stride=1, groups=1, dilation=1, ssn_bands=0, act=True, causal=False):
    if isinstance(k, int):
        k = (k, k)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    pad = tuple(d * (kk - 1) // 2 for kk, d in zip(k, dilation))
    layers = []
    if causal and k[1] > 1:  # pad time (last axis) on the left only: no look-ahead
        layers.append(nn.ZeroPad2d((dilation[1] * (k[1] - 1), 0, 0, 0)))
        pad = (pad[0], 0)
    layers.append(nn.Conv2d(cin, cout, k, stride, pad, dilation, groups, bias=False))
    layers.append(SubSpectralNorm(cout, ssn_bands) if ssn_bands else nn.BatchNorm2d(cout))
    if act:
        layers.append(nn.SiLU() if act == "silu" else nn.ReLU())
    return nn.Sequential(*layers)


class BCBlock(nn.Module):
    """Broadcasted residual block: 2-D frequency-depthwise conv, frequency pooling,
    1-D temporal depthwise conv, broadcast back onto the frequency axis."""

    def __init__(self, cin, cout, freq_stride, dilation, ssn_bands, dropout):
        super().__init__()
        self.transition = cin != cout
        f2 = []
        if self.transition:
            f2.append(conv_bn(cin, cout, 1))
        f2.append(
            conv_bn(cout, cout, (3, 1), (freq_stride, 1), groups=cout, ssn_bands=ssn_bands, act=False)
        )
        self.f2 = nn.Sequential(*f2)
        self.f1 = nn.Sequential(
            conv_bn(cout, cout, (1, 3), groups=cout, dilation=(1, dilation), act="silu"),
            nn.Conv2d(cout, cout, 1, bias=False),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        shortcut = x
        x = self.f2(x)
        aux = x
        x = self.f1(x.mean(dim=2, keepdim=True))
        x = x + aux
        if not self.transition:
            x = x + shortcut
        return F.relu(x)


# ---------------------------------------------------------------- state-space back-end
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


def selective_scan(log_a, bx, c, chunk=8):
    """Parallel evaluation of h_t = exp(log_a_t) * h_{t-1} + bx_t ;  y_t = <h_t, c_t>.

    log_a, bx: (B, T, E, N);  c: (B, T, N).  Exact, computed in chunks with a
    log-space cumulative-sum so the running exponent stays within float32 range.
    """
    b, t, e, n = bx.shape
    h = bx.new_zeros(b, e, n)
    ys = []
    for s in range(0, t, chunk):
        la = log_a[:, s : s + chunk]
        cum = torch.cumsum(la, dim=1)  # (B, L, E, N), <= 0
        hs = torch.exp(cum) * (h[:, None] + torch.cumsum(bx[:, s : s + chunk] * torch.exp(-cum), dim=1))
        ys.append(torch.einsum("blen,bln->ble", hs, c[:, s : s + chunk]))
        h = hs[:, -1]
    return torch.cat(ys, dim=1)


def selective_scan_ref(log_a, bx, c):
    """Sequential reference implementation (used in tests and for streaming)."""
    b, t, e, n = bx.shape
    h = bx.new_zeros(b, e, n)
    ys = []
    for i in range(t):
        h = torch.exp(log_a[:, i]) * h + bx[:, i]
        ys.append(torch.einsum("ben,bn->be", h, c[:, i]))
    return torch.stack(ys, dim=1)


class SSMBranch(nn.Module):
    """One direction of a Mamba mixer: causal depthwise conv + selective SSM."""

    def __init__(self, inner, state, dt_rank, conv_k, dt_min=1e-3, dt_max=1e-1, max_decay=8.0,
                 noise_aware=False):
        super().__init__()
        self.state, self.dt_rank, self.max_decay = state, dt_rank, max_decay
        # Noise-aware selective scan: per-channel strength w_e = sigmoid(gate_logit_e) with which
        # the speech-presence gate g_t scales the step size: dt' = dt * (1 - w_e * (1 - g_t)).
        self.gate_logit = nn.Parameter(torch.zeros(inner)) if noise_aware else None
        self.conv = nn.Conv1d(inner, inner, conv_k, groups=inner, padding=conv_k - 1)
        self.x_proj = nn.Linear(inner, dt_rank + 2 * state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, inner)
        dt = torch.exp(torch.rand(inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus
        self.A_log = nn.Parameter(torch.log(torch.arange(1, state + 1).float()).repeat(inner, 1))
        self.D = nn.Parameter(torch.ones(inner))
        self.A_log._no_weight_decay = True
        self.D._no_weight_decay = True

    def forward(self, x, gate=None):  # x: (B, T, E); gate: (B, T) in (0, 1) or None
        t = x.shape[1]
        x = F.silu(self.conv(x.transpose(1, 2))[..., :t].transpose(1, 2))
        dt, bm, cm = self.x_proj(x).split([self.dt_rank, self.state, self.state], dim=-1)
        with torch.autocast(device_type=x.device.type, enabled=False):
            x32, bm, cm = x.float(), bm.float(), cm.float()
            dt = F.softplus(self.dt_proj(dt.float()))  # (B, T, E)
            if self.gate_logit is not None and gate is not None:
                w = torch.sigmoid(self.gate_logit.float())
                dt = dt * (1 - w * (1 - gate.float()[..., None]))
            A = -torch.exp(self.A_log.float())  # (E, N)
            log_a = (dt[..., None] * A).clamp(min=-self.max_decay)
            bx = dt[..., None] * bm[:, :, None, :] * x32[..., None]
            y = selective_scan(log_a, bx, cm) + self.D * x32
        return y.to(x.dtype)


class MambaBlock(nn.Module):
    """Pre-norm residual Mamba block; bidirectional (offline) or causal (streaming)."""

    def __init__(self, dim, expand=2, state=16, conv_k=4, causal=False, drop_path=0.0, noise_aware=False):
        super().__init__()
        inner = int(expand * dim)
        dt_rank = math.ceil(dim / 16)
        self.norm = RMSNorm(dim)
        self.in_proj = nn.Linear(dim, 2 * inner, bias=False)
        self.fwd = SSMBranch(inner, state, dt_rank, conv_k, noise_aware=noise_aware)
        self.bwd = None if causal else SSMBranch(inner, state, dt_rank, conv_k, noise_aware=noise_aware)
        self.out_proj = nn.Linear(inner, dim, bias=False)
        self.drop_path = drop_path

    def forward(self, x, gate=None):
        res = x
        xs, z = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        y = self.fwd(xs, gate)
        if self.bwd is not None:
            y = y + self.bwd(xs.flip(1), None if gate is None else gate.flip(1)).flip(1)
        y = self.out_proj(y * F.silu(z))
        if self.training and self.drop_path > 0:
            keep = torch.rand(x.shape[0], 1, 1, device=x.device) >= self.drop_path
            y = y * keep / (1 - self.drop_path)
        return res + y


# ---------------------------------------------------------------- full model
class NativeKWS(nn.Module):
    def __init__(
        self,
        num_classes=12,
        n_mels=40,
        stem=32,
        channels=(32, 48, 64),
        blocks=(2, 2, 2),
        freq_strides=(1, 2, 2),
        dim=96,
        ssm_layers=3,
        expand=2,
        state=16,
        conv_k=4,
        time_stride=1,
        causal=False,
        dropout=0.1,
        drop_path=0.1,
        ssn_bands=5,
        noise_aware=False,
    ):
        super().__init__()
        assert not (noise_aware and time_stride != 1), "NASS gate is frame-aligned; needs time_stride=1"
        self.noise_aware = noise_aware
        self.gate = SpeechPresenceGate(n_mels, causal=causal) if noise_aware else None
        self.stem = conv_bn(1, stem, 5, stride=(2, 1))
        layers, cin, f = [], stem, n_mels // 2
        for si, (c, n, fs) in enumerate(zip(channels, blocks, freq_strides)):
            for bi in range(n):
                s = fs if bi == 0 else 1
                layers.append(BCBlock(cin, c, s, dilation=2**si, ssn_bands=ssn_bands, dropout=dropout))
                cin, f = c, f // s
        self.body = nn.Sequential(*layers)
        # Collapse frequency with a depthwise conv spanning all remaining bands.
        self.collapse = nn.Sequential(
            nn.Conv2d(cin, cin, (f, 3), (1, time_stride), (0, 1), groups=cin, bias=False),
            nn.Conv2d(cin, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
        )
        dpr = [drop_path * i / max(ssm_layers - 1, 1) for i in range(ssm_layers)]
        self.ssm = nn.ModuleList(
            MambaBlock(dim, expand, state, conv_k, causal, dpr[i], noise_aware) for i in range(ssm_layers)
        )
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def features(self, x, gate=None):
        x = self.body(self.stem(x))
        x = self.collapse(x).squeeze(2).transpose(1, 2)  # (B, T, D)
        for blk in self.ssm:
            x = blk(x, gate)
        return self.norm(x)

    def forward(self, x, return_gate=False):
        gate_logits = self.gate(x) if self.gate is not None else None
        g = torch.sigmoid(gate_logits) if gate_logits is not None else None
        logits = self.head(self.features(x, g).mean(dim=1))
        return (logits, gate_logits) if return_gate else logits


# ---------------------------------------------------------------- reference baseline
class BCResNet(nn.Module):
    """Port of the official BC-ResNet (Kim et al. 2021) for pipeline validation."""

    def __init__(self, base_c=8, num_classes=12):
        super().__init__()
        c = [base_c * 2, base_c, int(base_c * 1.5), base_c * 2, int(base_c * 2.5), base_c * 4]
        n, strided = [2, 2, 4, 4], (1, 2)
        self.stem = conv_bn(1, c[0], 5, stride=(2, 1))
        layers = []
        for idx, reps in enumerate(n):
            for i in range(reps):
                cin = c[idx] if i == 0 else c[idx + 1]
                s = 2 if (idx in strided and i == 0) else 1
                layers.append(BCBlock(cin, c[idx + 1], s, dilation=2**idx, ssn_bands=5, dropout=0.1))
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Conv2d(c[-2], c[-2], 5, groups=c[-2], padding=(0, 2), bias=False),
            nn.Conv2d(c[-2], c[-1], 1, bias=False),
            nn.BatchNorm2d(c[-1]),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c[-1], num_classes, 1),
        )

    def forward(self, x):
        return self.head(self.body(self.stem(x))).flatten(1)


# ---------------------------------------------------------------- BC-SSM network (ideas 1 + 2)
class SpeechPresenceGate(nn.Module):
    """Idea 2 front half: a tiny learned speech-vs-noise detector on the log-mel input.

    Returns per-frame logits (B, T); sigmoid(logits) = g_t is the probability that frame t
    is dominated by speech rather than noise. Trained with free frame-level SNR targets.
    """

    def __init__(self, n_mels=40, hidden=24, causal=False):
        super().__init__()
        self.causal = causal
        self.norm = nn.BatchNorm1d(n_mels)
        self.conv1 = nn.Conv1d(n_mels, hidden, 5, padding=0)
        self.conv2 = nn.Conv1d(hidden, hidden, 3, padding=0, groups=hidden)
        self.out = nn.Conv1d(hidden, 1, 1)

    def _pad(self, x, k):
        return F.pad(x, (k - 1, 0) if self.causal else ((k - 1) // 2, (k - 1) // 2))

    def forward(self, feats):  # feats: (B, 1, F, T)
        x = self.norm(feats.squeeze(1))
        x = F.silu(self.conv1(self._pad(x, 5)))
        x = F.silu(self.conv2(self._pad(x, 3)))
        return self.out(x).squeeze(1)


class BCSSMBlock(nn.Module):
    """Idea 1: broadcasted state-space residual block.

    Same as a BC-ResNet block, except the 1-D temporal path applied to the frequency-pooled
    features is a selective SSM (whole-utterance context, streamable recurrence) instead of
    a 3-tap temporal convolution. With temporal="conv" it reduces exactly to BC-ResNet.
    """

    def __init__(self, cin, cout, freq_stride, dilation, ssn_bands, dropout, temporal="ssm",
                 state=8, causal=False, noise_aware=False):
        super().__init__()
        self.transition = cin != cout
        self.temporal = temporal
        f2 = []
        if self.transition:
            f2.append(conv_bn(cin, cout, 1))
        f2.append(conv_bn(cout, cout, (3, 1), (freq_stride, 1), groups=cout, ssn_bands=ssn_bands, act=False))
        self.f2 = nn.Sequential(*f2)
        if temporal == "ssm":
            dt_rank = math.ceil(cout / 16)
            self.fwd = SSMBranch(cout, state, dt_rank, conv_k=4, noise_aware=noise_aware)
            self.bwd = None if causal else SSMBranch(cout, state, dt_rank, conv_k=4, noise_aware=noise_aware)
            self.f1 = nn.Sequential(nn.BatchNorm2d(cout), nn.SiLU(), nn.Conv2d(cout, cout, 1, bias=False),
                                    nn.Dropout2d(dropout))
        else:
            self.f1 = nn.Sequential(
                conv_bn(cout, cout, (1, 3), groups=cout, dilation=(1, dilation), act="silu", causal=causal),
                nn.Conv2d(cout, cout, 1, bias=False),
                nn.Dropout2d(dropout),
            )

    def forward(self, x, gate=None):
        shortcut = x
        x = self.f2(x)
        aux = x
        x = x.mean(dim=2, keepdim=True)  # (B, C, 1, T)
        if self.temporal == "ssm":
            seq = x.squeeze(2).transpose(1, 2)  # (B, T, C)
            y = self.fwd(seq, gate)
            if self.bwd is not None:
                y = y + self.bwd(seq.flip(1), None if gate is None else gate.flip(1)).flip(1)
            x = y.transpose(1, 2).unsqueeze(2)
        x = self.f1(x) + aux
        if not self.transition:
            x = x + shortcut
        return F.relu(x)


class BCSSMNet(nn.Module):
    """BC-ResNet topology with BC-SSM blocks in selected stages and an optional
    noise-aware selective scan driven by a shared speech-presence gate."""

    def __init__(self, base_c=64, num_classes=12, ssm_stages=(2, 3), state=8, causal=False,
                 noise_aware=False, dropout=0.1):
        super().__init__()
        c = [base_c * 2, base_c, int(base_c * 1.5), base_c * 2, int(base_c * 2.5), base_c * 4]
        n, strided = [2, 2, 4, 4], (1, 2)
        self.noise_aware = noise_aware
        self.gate = SpeechPresenceGate(causal=causal) if noise_aware else None
        self.stem = conv_bn(1, c[0], 5, stride=(2, 1), causal=causal)
        self.blocks = nn.ModuleList()
        for idx, reps in enumerate(n):
            for i in range(reps):
                cin = c[idx] if i == 0 else c[idx + 1]
                s = 2 if (idx in strided and i == 0) else 1
                self.blocks.append(BCSSMBlock(
                    cin, c[idx + 1], s, dilation=2**idx, ssn_bands=5, dropout=dropout,
                    temporal="ssm" if idx in ssm_stages else "conv", state=state, causal=causal,
                    noise_aware=noise_aware))
        self.head = nn.Sequential(
            *([nn.ZeroPad2d((4, 0, 0, 0))] if causal else []),
            nn.Conv2d(c[-2], c[-2], 5, groups=c[-2], padding=(0, 0 if causal else 2), bias=False),
            nn.Conv2d(c[-2], c[-1], 1, bias=False),
            nn.BatchNorm2d(c[-1]),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c[-1], num_classes, 1),
        )

    def forward(self, x, return_gate=False):
        gate_logits = self.gate(x) if self.gate is not None else None
        g = torch.sigmoid(gate_logits) if gate_logits is not None else None
        h = self.stem(x)
        for blk in self.blocks:
            h = blk(h, g)
        logits = self.head(h).flatten(1)
        return (logits, gate_logits) if return_gate else logits


def build_model(name, num_classes, **kw):
    if name.startswith("bcresnet"):
        tau = float(name.split("-")[1])
        return BCResNet(int(tau * 8), num_classes)
    if name.startswith("bcssm"):
        tau = float(name.split("-")[1])
        keys = ("ssm_stages", "state", "causal", "noise_aware", "dropout")
        return BCSSMNet(int(tau * 8), num_classes, **{k: kw[k] for k in keys if k in kw})
    return NativeKWS(num_classes=num_classes, **kw)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
