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


# ------------------------------------------------------------ configurable input channels
# RGB is the base. Extra channels let the model see beyond colour:
#   hue/sat/val : HSV decomposition (separates colour from brightness — gray asphalt has low sat)
#   exg         : Excess-Green vegetation index (2G-R-B) — vegetation/canopy cue for occlusion
#   exr         : Excess-Red index (1.4R-G) — bare-soil / unpaved-track cue (complements exg)
#   nir         : near-infrared IF supplied (LISS-IV / Sentinel-2 / Cartosat MS). DeepGlobe and
#                 Esri are RGB-only, so 'nir' is zero-filled unless a real NIR band is passed in.
RGB = ["r", "g", "b"]
_RGB_IDX = {"r": 0, "g": 1, "b": 2}


def _exg_u8(rgb):
    r, g, b = (rgb[..., i].astype(np.int32) for i in range(3))
    return np.clip((2 * g - r - b + 510) * (255.0 / 1020.0), 0, 255).astype(np.uint8)


def _exr_u8(rgb):
    # ponytail: ExR = 1.4R - G (Meyer 2008), raw form to mirror _exg_u8. Span [-255,357]->[0,255].
    #           1.4 is the standard coefficient; bump it if red-soil tracks need more contrast.
    r, g = rgb[..., 0].astype(np.float32), rgb[..., 1].astype(np.float32)
    return np.clip((1.4 * r - g + 255.0) * (255.0 / 612.0), 0, 255).astype(np.uint8)


def build_channels(rgb, spec, nir=None):
    """Stack the configured channels from an RGB uint8 image -> HxWxlen(spec) uint8.

    rgb: HxWx3 uint8 (RGB); nir: optional HxW uint8. Every channel is emitted as uint8 [0,255]
    so it drops straight into A.Normalize (see `channel_norm`)."""
    hsv, out = None, []
    for ch in spec:
        if ch in _RGB_IDX:
            out.append(rgb[..., _RGB_IDX[ch]])
        elif ch in ("hue", "sat", "val"):
            if hsv is None:
                hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            if ch == "hue":
                out.append((hsv[..., 0].astype(np.uint16) * 255 // 179).astype(np.uint8))  # [0,179]->[0,255]
            else:
                out.append(hsv[..., 1 if ch == "sat" else 2])
        elif ch == "exg":
            out.append(_exg_u8(rgb))
        elif ch == "exr":
            out.append(_exr_u8(rgb))
        elif ch == "nir":
            out.append(nir if nir is not None else np.zeros(rgb.shape[:2], np.uint8))
        else:
            raise ValueError(f"unknown channel '{ch}'")
    return np.stack(out, axis=-1)


def channel_norm(spec):
    """(mean, std) per channel for A.Normalize, matching build_channels order: RGB -> ImageNet
    stats, derived/extra channels -> map [0,1] to ~[-1,1]."""
    mean = [IMAGENET_MEAN[_RGB_IDX[c]] if c in _RGB_IDX else 0.5 for c in spec]
    std = [IMAGENET_STD[_RGB_IDX[c]] if c in _RGB_IDX else 0.5 for c in spec]
    return mean, std


class StackChannels(A.ImageOnlyTransform):
    """Expand the (augmented) RGB image to the configured channels — placed just before
    Normalize so geometric/photometric augs stay on RGB and hue/exg reflect the augmented colour."""
    def __init__(self, spec, p=1.0):
        super().__init__(p=p)
        self.spec = spec

    def apply(self, img, **params):
        return build_channels(img, self.spec)

    def get_transform_init_args_names(self):
        return ("spec",)


def _finalize(augs, channels):
    """Append (StackChannels +) per-channel Normalize + ToTensor. RGB stays byte-identical."""
    mean, std = channel_norm(channels)
    if channels != RGB:
        augs = augs + [StackChannels(channels)]
    return A.Compose(augs + [A.Normalize(mean=mean, std=std), ToTensorV2()])


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


def train_tf(crop=512, occ_p=0.5, occ_max_holes=8, occ_max_frac=0.125, channels=RGB):
    return _finalize([
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
    ], channels)


def val_tf(crop=512, channels=RGB):
    return _finalize([A.CenterCrop(crop, crop, pad_if_needed=True)], channels)


# --- post-crop variants (cropping is done in the dataset, e.g. white-nodata-aware) ---
def train_aug(crop=640, occ_p=0.5, occ_max_holes=8, occ_max_frac=0.125, channels=RGB):
    return _finalize([
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
    ], channels)


def val_aug(channels=RGB):
    return _finalize([], channels)


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
            occ[c, y:y + h, x:x + w] = _BLACK[c] if c < len(_BLACK) else -1.0   # extra channels: normalized 0
        mask[y:y + h, x:x + w] = 1.0
    return occ, mask
