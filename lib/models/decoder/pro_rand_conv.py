#!/usr/bin/env python
import math
import random
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.ops import deform_conv2d as tv_deform_conv2d
    _HAS_TV_DEFORM = True
except Exception:
    _HAS_TV_DEFORM = False


def _make_gaussian_kernel_2d(ks, sigma):
    """Returns 2D Gaussian weights of shape [ks, ks] without normalization."""
    ax = torch.arange(ks, dtype=torch.float32) - (ks - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    g = torch.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    return g  # shape [ks, ks]


class ProRandConvModule(nn.Module):
    """
    Progressive Random Convolutions (Pro-RandConv)
    Block per mini-batch: DeformableConv -> Standardize -> Affine -> tanh
    Repeat the same block L times, L ~ Uniform(1..L_max), with shared params within the mini-batch.

    Key defaults follow the paper:
      - k = 3 (fixed)
      - He-style sigma_w = 1/sqrt(k^2 * C_in), then element-wise Gaussian smoothing with sigma_g ~ U(eps, 1)
      - Offsets ~ N(0, sigma_delta^2), sigma_delta ~ U(eps, b_delta)
      - γ, β ~ N(0, 0.5^2)
      - L ~ Uniform{1..L_max}

    Notes:
      - If torchvision deform_conv2d is unavailable, falls back to regular conv2d.
      - If clamp_output=True and data_mean/std are given, output is clamped to the allowed normalized range.
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        kernel_size=3,
        mixing=False,
        identity_prob=0.0,
        data_mean=None,
        data_std=None,
        clamp_output=False,
        use_deformable=True,
        L_max=10,
        b_delta=0.5,      # upper bound for offset std (0.2 for 32x32, 0.5 for 224x224+ in paper)
        eps=1e-3,
        sigma_gamma=0.5,
        sigma_beta=0.5,
    ):
        super().__init__()

        assert kernel_size == 3, "Paper fixes k=3. Change at your own risk."
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = kernel_size
        self.padding = kernel_size // 2
        self.mixing = mixing
        self.identity_prob = identity_prob
        self.clamp_output = clamp_output
        self.use_deformable = use_deformable and _HAS_TV_DEFORM
        self.L_max = int(L_max)
        self.b_delta = float(b_delta)
        self.eps = float(eps)
        self.sigma_gamma = float(sigma_gamma)
        self.sigma_beta = float(sigma_beta)

        if self.mixing:
            assert in_channels == out_channels or out_channels == 1, "Mixing mode requires in==out or out==1."

        # For output clamping in normalized space
        self.register_buffer('data_mean', None if data_mean is None else torch.tensor(data_mean).reshape(1, -1, 1, 1))
        self.register_buffer('data_std', None if data_std is None else torch.tensor(data_std).reshape(1, -1, 1, 1))
        if self.clamp_output:
            assert (self.data_mean is not None) and (self.data_std is not None), "Need data mean/std for clamping."
            range_up = (torch.ones(1, 3, 1, 1) - self.data_mean) / self.data_std
            range_low = (torch.zeros(1, 3, 1, 1) - self.data_mean) / self.data_std
            self.register_buffer('range_up', range_up)   # broadcastable
            self.register_buffer('range_low', range_low) # broadcastable
        else:
            self.register_buffer('range_up', None)
            self.register_buffer('range_low', None)

        # Placeholder parameters to satisfy .parameters() without adding learnables
        # All block params are sampled per mini-batch inside forward
        self.register_parameter('_dummy', None)

    @torch.no_grad()
    def _sample_conv_weight(self, device, dtype):
        # sigma_w = 1/sqrt(k^2 * C_in)
        sigma_w = 1.0 / math.sqrt(self.k * self.k * self.in_channels)
        w = torch.randn(self.out_channels, self.in_channels, self.k, self.k, device=device, dtype=dtype) * sigma_w

        # Element-wise Gaussian smoothing g[i,j], sigma_g ~ U(eps, 1)
        sigma_g = torch.empty(1, device=device, dtype=dtype).uniform_(self.eps, 1.0).item()
        g2d = _make_gaussian_kernel_2d(self.k, sigma_g).to(device=device, dtype=dtype)  # [k, k]
        w = w * g2d.view(1, 1, self.k, self.k)
        return w  # shape [C_out, C_in, k, k]

    @torch.no_grad()
    def _sample_offsets(self, N, H, W, device, dtype):
        # sigma_delta ~ U(eps, b_delta), offsets ~ N(0, sigma_delta^2)
        sigma_delta = torch.empty(1, device=device, dtype=dtype).uniform_(self.eps, self.b_delta).item()
        num_offset_ch = 2 * self.k * self.k
        offsets = torch.randn(N, num_offset_ch, H, W, device=device, dtype=dtype) * sigma_delta

        # Optional spatial correlation by light Gaussian blur to mimic GRF feel
        if H >= 3 and W >= 3:
            kernel = torch.tensor([[1., 2., 1.],
                                   [2., 4., 2.],
                                   [1., 2., 1.]], device=device, dtype=dtype)
            kernel = kernel / kernel.sum()
            kernel = kernel.view(1, 1, 3, 3)
            # depthwise conv over each offset channel
            offsets = F.conv2d(offsets, kernel.repeat(num_offset_ch, 1, 1, 1),
                               bias=None, stride=1, padding=1, groups=num_offset_ch)
        return offsets  # [N, 2*k*k, H, W]

    @torch.no_grad()
    def _sample_affine(self, device, dtype):
        gamma = torch.randn(1, self.out_channels, 1, 1, device=device, dtype=dtype) * self.sigma_gamma
        beta = torch.randn(1, self.out_channels, 1, 1, device=device, dtype=dtype) * self.sigma_beta
        return gamma, beta

    def _apply_block_once(self, x, weight, offsets, gamma, beta):
        # Deformable or regular conv
        if self.use_deformable:
            # torchvision deform_conv2d expects offsets of shape [N, 2*k*k, H, W]
            x = tv_deform_conv2d(
                input=x,
                weight=weight,
                offset=offsets,
                bias=None,
                stride=1,
                padding=self.padding,
                dilation=1,
                mask=None
            )
        else:
            x = F.conv2d(x, weight, bias=None, stride=1, padding=self.padding, dilation=1)

        # Per-image per-channel standardization
        B, C, H, W = x.shape
        x_resh = x.view(B, C, -1)
        mu = x_resh.mean(dim=2).view(B, C, 1, 1)
        var = x_resh.var(dim=2, unbiased=False).view(B, C, 1, 1)
        x = (x - mu) / torch.sqrt(var + 1e-6)

        # Affine and tanh
        x = gamma * x + beta
        x = torch.tanh(x)
        return x

    def forward(self, x):
        # Identity shortcut with some probability
        if self.identity_prob > 0 and torch.rand(1, device=x.device) < self.identity_prob:
            return x

        # Sample the block params per mini-batch
        B, C, H, W = x.shape
        dtype = x.dtype
        device = x.device

        weight = self._sample_conv_weight(device, dtype)
        gamma, beta = self._sample_affine(device, dtype)

        # Repeat count L ~ Uniform{1..L_max}
        L = random.randint(1, max(1, self.L_max))

        # Offsets can depend on spatial size, so sample after shapes are known
        offsets = None
        if self.use_deformable:
            offsets = self._sample_offsets(B, H, W, device, dtype)

        # Apply the same sampled params L times progressively
        out = x
        for _ in range(L):
            out = self._apply_block_once(out, weight, offsets, gamma, beta)

        # Optional residual mixing
        if self.mixing:
            alpha = random.random()
            out = alpha * out + (1.0 - alpha) * x

        # Optional clamp back to normalized data range
        if self.clamp_output:
            out = torch.max(torch.min(out, self.range_up), self.range_low)

        return out

    # Compatibility helpers
    def randomize(self):
        # No persistent state to randomize. Params are sampled inside forward per mini-batch.
        return

    def whiten(self, input):
        if (self.data_mean is None) or (self.data_std is None):
            return input
        return (input - self.data_mean) / self.data_std

    def dewhiten(self, input):
        if (self.data_mean is None) or (self.data_std is None):
            return input
        return input * self.data_std + self.data_mean