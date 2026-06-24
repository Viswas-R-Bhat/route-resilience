"""Augmentations for occlusion-robust segmentation (albumentations 2.0.x API).

Signature technique: CoarseDropout blacks out random patches of the REAL satellite image
while leaving the REAL mask untouched (fill_mask=None) — teaching the model that roads
persist under canopy/shadow/cloud. This *augments* real data; it does not fabricate it.
"""
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# normalized value of a pure-black pixel, per channel (used for controlled occlusion at eval)
_BLACK = [(0.0 - m) / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)]


def occlusion_aug(crop, max_holes=8, max_frac=0.125, p=0.5):
    max_sz = max(8, int(max_frac * crop))
    min_sz = max(8, max_sz // 4)
    return A.CoarseDropout(
        num_holes_range=(2, max_holes),
        hole_height_range=(min_sz, max_sz),
        hole_width_range=(min_sz, max_sz),
        fill=0,           # black out the image patch (simulates occlusion / deep shadow)
        fill_mask=None,   # CRITICAL: leave mask intact — the road still exists underneath
        p=p,
    )


def train_tf(crop=512, occ_p=0.5, occ_max_holes=8, occ_max_frac=0.125):
    return A.Compose([
        A.RandomCrop(crop, crop, pad_if_needed=True),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Affine(scale=(0.85, 1.2), translate_percent=(0.0, 0.08),
                 rotate=(-45, 45), border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
            A.CLAHE(clip_limit=4.0),
        ], p=0.7),
        A.GaussNoise(p=0.3),
        occlusion_aug(crop, occ_max_holes, occ_max_frac, occ_p),
        A.OneOf([A.GaussianBlur(blur_limit=(3, 7)), A.MotionBlur(blur_limit=7)], p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def val_tf(crop=512):
    return A.Compose([
        A.CenterCrop(crop, crop, pad_if_needed=True),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# --- post-crop variants (cropping is done in the dataset, e.g. white-nodata-aware) ---
def train_aug(crop=640, occ_p=0.5, occ_max_holes=8, occ_max_frac=0.125):
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Affine(scale=(0.85, 1.2), translate_percent=(0.0, 0.06),
                 rotate=(-30, 30), border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
            A.CLAHE(clip_limit=4.0),
        ], p=0.7),
        A.GaussNoise(p=0.3),
        occlusion_aug(crop, occ_max_holes, occ_max_frac, occ_p),
        A.OneOf([A.GaussianBlur(blur_limit=(3, 7)), A.MotionBlur(blur_limit=7)], p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def val_aug():
    return A.Compose([A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])


def occlude_tensor(img_t, frac=0.18, holes=4, rng=None):
    """Controlled occlusion for the occlusion-recall metric.

    Takes a normalized (C,H,W) tensor, blacks out `holes` random rectangles (recording the
    exact occluded region), and returns (occluded_tensor, occ_mask[H,W]). Because WE generate
    the mask, we know precisely which pixels were hidden — a controlled, reproducible
    perturbation of real images (not fabricated data).
    """
    rng = rng or np.random.default_rng()
    C, H, W = img_t.shape
    occ = img_t.clone()
    mask = torch.zeros(H, W)
    sz = max(8, int(frac * min(H, W)))
    for _ in range(holes):
        h = int(rng.integers(sz // 2, sz + 1))
        w = int(rng.integers(sz // 2, sz + 1))
        y = int(rng.integers(0, H - h + 1))
        x = int(rng.integers(0, W - w + 1))
        for c in range(C):
            occ[c, y:y + h, x:x + w] = _BLACK[c]
        mask[y:y + h, x:x + w] = 1.0
    return occ, mask
