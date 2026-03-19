import torch
import torch.nn as nn
import torch.nn.functional as F


class ClsLoss(nn.Module):
    def __init__(self):
        super(ClsLoss, self).__init__()
        self.criterion = nn.BCELoss(reduction='mean')

    def forward(self, pred, gt, valid=None):
        gt = gt.to(pred.device).float()
        pred = pred.sigmoid()

        if valid is not None:
            valid = valid.to(pred.device).bool().squeeze(-1)
            if not valid.any():
                return torch.tensor(0.0, device=pred.device, dtype=torch.float32)
            
            pred = pred[valid]
            gt = gt[valid]

        return self.criterion(pred, gt)


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, gt, valid=None):
        device = pred.device
        gt = gt.to(device).float()
        prob = pred.sigmoid()

        if valid is not None:
            valid = valid.to(device).bool().squeeze(-1)
            if not valid.any():
                return prob.sum() * 0.0
            prob = prob[valid]
            gt = gt[valid]

        intersection = (prob * gt).sum(dim=1)
        union = prob.sum(dim=1) + gt.sum(dim=1)
        dice = (2.0 * intersection + self.eps) / (union + self.eps)
        return 1.0 - dice.mean()


class AdvLoss(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, logits, valid=None):
        x = logits
        x = torch.stack([-x, x], dim=1)

        logp = F.log_softmax(x, dim=1)
        loss_per = -logp.mean(dim=1).flatten(1).mean(dim=1)

        if valid is not None:
            v = valid.view(valid.size(0), -1)[:, 0].bool()
            if not v.any():
                return logits.sum() * 0.0
            return loss_per[v].mean()

        return loss_per.mean()


class EntropyLoss(nn.Module):
    def __init__(self, eps=1e-5):
        super(EntropyLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, valid=None):
        pred = pred.sigmoid()

        entropy = - (pred * torch.log(pred + self.eps) + (1 - pred) * torch.log(1 - pred + self.eps))

        if valid is not None:
            valid = valid.to(pred.device).bool().squeeze(-1)
            if not valid.any():
                return torch.tensor(0.0, device=pred.device, dtype=torch.float32)

            entropy = entropy[valid]

        return - entropy.mean()


class CosineLoss(nn.Module):
    def __init__(self):
        super(CosineLoss, self).__init__()

    def forward(self, pred, gt, valid=None):
        pred = F.normalize(pred, dim=-1)
        gt = F.normalize(gt.to(pred.device), dim=-1)

        if valid is not None:
            valid = valid.bool().squeeze(-1)
            if not valid.any():
                return torch.tensor(0.0, device=pred.device, dtype=torch.float32)
            pred, gt = pred[valid], gt[valid]

        cos_sim = F.cosine_similarity(pred, gt, dim=-1)
        loss = 1 - cos_sim

        return loss.mean()


# This loss is re-implementation of norm_loss from https://github.com/ShengCN/PixHtLab-Src
class NormLoss(nn.Module):
    def __init__(self, p=1, eps=1e-8):
        super().__init__()
        self.p = p
        self.eps = eps

    def forward(self, pred, target, valid=None):
        pred   = pred.float()
        target = target.to(pred.device).float()
        assert pred.shape == target.shape and pred.ndim == 3, "Expect (B,256,256)"

        if valid is None:
            mask = torch.ones(pred.size(0), 1, 1, device=pred.device, dtype=pred.dtype)
        else:
            v = valid.to(pred.device).float()
            if v.ndim == 1:
                v = v.view(-1, 1)
            elif v.ndim == 2 and v.shape[1] == 1:
                pass
            else:
                raise ValueError("valid must be (B,) or (B,1)")
            mask = v.view(-1, 1, 1)

        mask_full = mask.expand_as(pred)

        diff = (pred - target) * mask_full
        if self.p == 1:
            per_elem = diff.abs()
        elif self.p == 2:
            per_elem = diff.pow(2)
        else:
            per_elem = diff.abs().pow(self.p)

        denom = mask_full.sum().clamp_min(self.eps)
        return per_elem.sum() / denom