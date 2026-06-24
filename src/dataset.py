"""DeepGlobe road dataset — REAL paired satellite tiles + binary road masks.

Layout (balraj98/deepglobe-road-extraction-dataset):
    data/deepglobe/train/{id}_sat.jpg     # real RGB satellite tile (1024x1024, ~0.5 m/px)
    data/deepglobe/train/{id}_mask.png     # real binary road mask (road = white)
The official valid/ and test/ folders have NO public masks, so we split the labelled
train/ folder into our own train / held-out-val sets for honest, real metrics.
"""
import os
os.environ.setdefault("OPENCV_LOG_LEVEL", "OFF")  # silence GeoTIFF tag warnings on Massachusetts .tiff
import glob
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def list_pairs(train_dir):
    """Return (sat_paths, mask_paths) for every tile that has BOTH a sat image and a mask."""
    sats = sorted(glob.glob(os.path.join(train_dir, "*_sat.jpg")))
    sat_paths, mask_paths = [], []
    for s in sats:
        m = s[:-len("_sat.jpg")] + "_mask.png"
        if os.path.exists(m):
            sat_paths.append(s)
            mask_paths.append(m)
    return sat_paths, mask_paths


class DeepGlobeRoads(Dataset):
    def __init__(self, sat_paths, mask_paths, transform=None, road_thresh=127):
        assert len(sat_paths) == len(mask_paths)
        self.sat_paths = sat_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.road_thresh = road_thresh

    def __len__(self):
        return len(self.sat_paths)

    def __getitem__(self, i):
        img = cv2.imread(self.sat_paths[i], cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(self.sat_paths[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        m = cv2.imread(self.mask_paths[i], cv2.IMREAD_GRAYSCALE)
        mask = (m > self.road_thresh).astype(np.uint8)

        if self.transform is not None:
            out = self.transform(image=img, mask=mask)
            img, mask = out["image"], out["mask"]

        if not torch.is_tensor(img):  # fallback if no ToTensorV2 in pipeline
            img = torch.from_numpy(img.transpose(2, 0, 1).copy()).float() / 255.0
        if not torch.is_tensor(mask):
            mask = torch.from_numpy(np.ascontiguousarray(mask))
        mask = mask.unsqueeze(0).float()  # (1, H, W)
        return img, mask


# ============================================================================
# Combined multi-dataset loader (DeepGlobe + Massachusetts), with white-nodata
# -aware cropping. Cropping happens HERE so we can reject Massachusetts' large
# white no-data regions; the transform then does only photometric/geometric aug.
# ============================================================================
def list_massachusetts(root):
    """Massachusetts: tiff/train/{id}.tiff  +  tiff/train_labels/{id}.tif (road=white)."""
    tr = os.path.join(root, "tiff", "train")
    lb = os.path.join(root, "tiff", "train_labels")
    out = []
    for s in sorted(glob.glob(os.path.join(tr, "*.tiff"))):
        stem = os.path.splitext(os.path.basename(s))[0]
        m = os.path.join(lb, stem + ".tif")
        if os.path.exists(m):
            out.append((s, m))
    return out


def list_combined(which, deepglobe_dir, mass_root):
    """which: 'deepglobe' | 'deepglobe+mass' | 'mass'. Returns list of (img_path, mask_path)."""
    samples = []
    if "deepglobe" in which:
        ds, dm = list_pairs(deepglobe_dir)
        samples += list(zip(ds, dm))
    if "mass" in which:
        samples += list_massachusetts(mass_root)
    return samples


def _read_pair(img_path, mask_path, road_thresh):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mk = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return img, (mk > road_thresh).astype(np.uint8)


def _pad_to(img, mask, crop):
    H, W = img.shape[:2]
    if H >= crop and W >= crop:
        return img, mask
    ph, pw = max(0, crop - H), max(0, crop - W)
    img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
    mask = cv2.copyMakeBorder(mask, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
    return img, mask


def _content_crop(img, mask, crop, tries=10, white_thr=238, max_white=0.6):
    """Random crop that rejects windows >max_white pure-white (Massachusetts no-data)."""
    img, mask = _pad_to(img, mask, crop)
    H, W = img.shape[:2]
    fallback = None
    for _ in range(tries):
        y = random.randint(0, H - crop); x = random.randint(0, W - crop)
        ic = img[y:y + crop, x:x + crop]; mc = mask[y:y + crop, x:x + crop]
        if fallback is None:
            fallback = (ic, mc)
        if (ic > white_thr).all(axis=2).mean() <= max_white:
            return ic, mc
    return fallback


def _center_crop(img, mask, crop):
    img, mask = _pad_to(img, mask, crop)
    H, W = img.shape[:2]
    y, x = (H - crop) // 2, (W - crop) // 2
    return img[y:y + crop, x:x + crop], mask[y:y + crop, x:x + crop]


class RoadSegDataset(Dataset):
    """Unified dataset for DeepGlobe + Massachusetts. Crops in __getitem__ (nodata-aware),
    then applies a post-crop transform (augment.train_aug / val_aug)."""
    def __init__(self, samples, transform, crop=640, mode="train", road_thresh=127):
        self.samples = samples
        self.transform = transform
        self.crop = crop
        self.mode = mode
        self.road_thresh = road_thresh

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        ip, mp = self.samples[i]
        img, mask = _read_pair(ip, mp, self.road_thresh)
        if self.mode == "train":
            img, mask = _content_crop(img, mask, self.crop)
        else:
            img, mask = _center_crop(img, mask, self.crop)
        out = self.transform(image=img, mask=mask)
        img, mask = out["image"], out["mask"]
        if not torch.is_tensor(mask):
            mask = torch.from_numpy(np.ascontiguousarray(mask))
        return img, mask.unsqueeze(0).float()
