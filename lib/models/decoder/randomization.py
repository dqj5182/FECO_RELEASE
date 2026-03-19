import torch
import torch.nn as nn


class StyleRandomization(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x, style_source, alpha=None):
        N, C, H, W = x.size()
        x_ = x.view(N, C, -1)
        s_ = style_source.view(N, C, -1)

        mean_x = x_.mean(-1, keepdim=True)
        std_x = x_.var(-1, keepdim=True).add(self.eps).sqrt()

        mean_s = s_.mean(-1, keepdim=True)
        std_s = s_.var(-1, keepdim=True).add(self.eps).sqrt()

        if alpha is None:
            alpha = torch.rand(N, 1, 1, device=x.device)

        mean_mix = (1 - alpha) * mean_x + alpha * mean_s
        std_mix = (1 - alpha) * std_x + alpha * std_s

        x_norm = (x_ - mean_x) / std_x
        x_stylized = x_norm * std_mix + mean_mix
        return x_stylized.view(N, C, H, W)


class ContentRandomization(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x, content_source, mask=None):
        N, C, H, W = x.shape

        if content_source.shape[-2:] != (H, W):
            content_source = F.interpolate(content_source, size=(H, W), mode="bilinear", align_corners=False)
        if mask is None:
            mask = x.new_ones(N, 1, H, W)
        elif mask.shape[-2:] != (H, W):
            mask = F.interpolate(mask, size=(H, W), mode="nearest")
        mask = mask.to(dtype=x.dtype)

        x = x.reshape(N, C, -1)
        content_source = content_source.reshape(N, C, -1)
        mask_flat = mask.reshape(N, 1, -1)

        denom = mask_flat.sum(-1, keepdim=True).clamp_min(1e-6)

        mean_x = (x * mask_flat).sum(-1, keepdim=True) / denom
        var_x  = ((x - mean_x) ** 2 * mask_flat).sum(-1, keepdim=True) / denom

        mean_cs = (content_source * mask_flat).sum(-1, keepdim=True) / denom
        var_cs  = ((content_source - mean_cs) ** 2 * mask_flat).sum(-1, keepdim=True) / denom

        content_source = (content_source - mean_cs) / (var_cs + self.eps).sqrt()
        x = content_source * (var_x + self.eps).sqrt() + mean_x

        x = x.view(N, C, H, W)
        return x * mask