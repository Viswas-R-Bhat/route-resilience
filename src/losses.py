"""Combined loss for thin, class-imbalanced road masks (operates on LOGITS).

Roads are ~2-5% of pixels, so plain BCE under-segments. We combine:
  - Dice   : overlap-based, robust to class imbalance
  - BCE    : per-pixel calibration (BCEWithLogits, stable)
  - Connectivity : dilated-prob vs dilated-target BCE — rewards continuous road structures,
                   a lightweight proxy for topological correctness (helps occlusion recovery).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        prob = torch.sigmoid(logits)
        p = prob.reshape(-1)
        t = target.reshape(-1)
        inter = (p * t).sum()
        return 1 - (2 * inter + self.smooth) / (p.sum() + t.sum() + self.smooth)


class ConnectivityLoss(nn.Module):
    """Soft-Dice between *dilated* prediction and *dilated* target.

    Dilating both maps makes the term tolerant to small misalignments, so it rewards
    predictions that form continuous road structures (a lightweight topological proxy).
    Uses only mul/sum/div -> safe under AMP autocast (unlike raw binary_cross_entropy).
    """
    def __init__(self, k=5, smooth=1.0):
        super().__init__()
        self.k = k
        self.smooth = smooth

    def forward(self, logits, target):
        prob = torch.sigmoid(logits)
        pad = self.k // 2
        pd = F.max_pool2d(prob, self.k, 1, pad)
        td = F.max_pool2d(target, self.k, 1, pad)
        pd = pd.reshape(-1)
        td = td.reshape(-1)
        inter = (pd * td).sum()
        return 1 - (2 * inter + self.smooth) / (pd.sum() + td.sum() + self.smooth)


class CombinedLoss(nn.Module):
    def __init__(self, dice=0.5, bce=0.3, connectivity=0.2):
        super().__init__()
        self.wd, self.wb, self.wc = dice, bce, connectivity
        self.dice = DiceLoss()
        self.conn = ConnectivityLoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target):
        return (self.wd * self.dice(logits, target)
                + self.wb * self.bce(logits, target)
                + self.wc * self.conn(logits, target))


class LovaszDiceLoss(nn.Module):
    """Stronger loss for the heavyweight model.

    Lovasz-hinge directly optimizes the IoU surrogate and is strong on thin, imbalanced
    structures (recovers missed thin roads); Dice stabilizes; the connectivity term keeps
    roads continuous. All operate on logits and are AMP-safe.
    """
    def __init__(self, lovasz=0.5, dice=0.4, connectivity=0.1):
        super().__init__()
        self.wl, self.wd, self.wc = lovasz, dice, connectivity
        self.lov = smp.losses.LovaszLoss(mode="binary", from_logits=True)
        self.dic = smp.losses.DiceLoss(mode="binary", from_logits=True)
        self.conn = ConnectivityLoss()

    def forward(self, logits, target):
        return (self.wl * self.lov(logits, target)
                + self.wd * self.dic(logits, target)
                + self.wc * self.conn(logits, target))
