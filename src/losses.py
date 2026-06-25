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


# ----------------------------------------------------------------- clDice (topology loss)
def _soft_erode(x):
    p1 = -F.max_pool2d(-x, (3, 1), (1, 1), (1, 0))
    p2 = -F.max_pool2d(-x, (1, 3), (1, 1), (0, 1))
    return torch.min(p1, p2)


def _soft_open(x):
    return F.max_pool2d(_soft_erode(x), (3, 3), (1, 1), (1, 1))      # erode then dilate


def soft_skeletonize(x, iters=10):
    """Differentiable morphological skeleton (Shit et al., clDice). Repeated soft erosion
    peels the mask to its 1-px centerline; AMP-safe (only max_pool2d / relu / min)."""
    sk = F.relu(x - _soft_open(x))
    for _ in range(iters):
        x = _soft_erode(x)
        delta = F.relu(x - _soft_open(x))
        sk = sk + F.relu(delta - sk * delta)
    return sk


class ClDiceDiceLoss(nn.Module):
    """clDice (centerline Dice) + soft-Dice + BCE — directly optimizes *connectivity*.

    clDice measures overlap between the soft skeletons of prediction and target, so a single
    broken pixel (a road snapped by tree canopy) is penalized far more than by area Dice. It's
    unstable alone, so Dice + BCE anchor it. All terms operate on logits and are AMP-safe.
    Recommended for the connectivity-critical road task; swap in via `--loss cldice`.
    """
    def __init__(self, cldice=0.4, dice=0.4, bce=0.2, iters=10, smooth=1.0):
        super().__init__()
        self.wc, self.wd, self.wb, self.iters, self.smooth = cldice, dice, bce, iters, smooth
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()

    def _cldice(self, prob, target):
        sp = soft_skeletonize(prob, self.iters).float()
        st = soft_skeletonize(target, self.iters).float()
        t, p = target.float(), prob.float()
        tprec = (torch.sum(sp * t) + self.smooth) / (torch.sum(sp) + self.smooth)   # skel_pred on target
        tsens = (torch.sum(st * p) + self.smooth) / (torch.sum(st) + self.smooth)   # skel_target on pred
        return 1.0 - 2.0 * tprec * tsens / (tprec + tsens)

    def forward(self, logits, target):
        prob = torch.sigmoid(logits)
        return (self.wc * self._cldice(prob, target)
                + self.wd * self.dice(logits, target)
                + self.wb * self.bce(logits, target))


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
