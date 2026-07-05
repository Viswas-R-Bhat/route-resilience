"""Phase 1 training — occlusion-robust road segmentation on REAL DeepGlobe data.

Every reported number is computed on a real held-out split (never seen in training).
Outputs (in out_dir): best.pt, last.pt, metrics.csv, summary.json, curves.png,
prediction grids (input | GT | prediction | occluded-input).

Usage:
  python src/train.py                          # full run from config
  python src/train.py --epochs 3 --limit 600   # fast first loop to sanity-check metrics+viz
"""
import os
import sys
import json
import time
import argparse
import random
import csv

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
import augment
import dataset as DS
import model as M
import losses
import metrics as ME
import viz


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def split_pairs(sats, masks, train_split, seed, limit=None):
    idx = list(range(len(sats)))
    random.Random(seed).shuffle(idx)
    if limit:
        idx = idx[:limit]
    n_tr = int(len(idx) * train_split)
    tr, va = idx[:n_tr], idx[n_tr:]
    pick = lambda I, L: [L[i] for i in I]
    return (pick(tr, sats), pick(tr, masks)), (pick(va, sats), pick(va, masks))


def split_samples(samples, train_split, seed, limit=None):
    """Split a list of (img_path, mask_path) tuples into train/val (seeded)."""
    idx = list(range(len(samples)))
    random.Random(seed).shuffle(idx)
    if limit:
        idx = idx[:limit]
    n_tr = int(len(idx) * train_split)
    tr = [samples[i] for i in idx[:n_tr]]
    va = [samples[i] for i in idx[n_tr:]]
    return tr, va


@torch.no_grad()
def validate(net, loader, crit, device, buffer_px, occ_frac):
    net.eval()
    loss_m = ME.RunningMean()
    tp = fp = fn = 0.0                 # aggregate over the whole val set
    rtp = rfp = rfn = 0.0             # relaxed (buffered)
    occ_tp = occ_fn = 0.0            # occlusion-recall
    rng = np.random.default_rng(0)
    vis = None
    for bi, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        with autocast("cuda"):
            logit = net(x)
            loss = crit(logit, y)
        loss_m.update(loss.item(), x.size(0))
        pred = ME.binarize(logit)
        tp += (pred * y).sum().item()
        fp += (pred * (1 - y)).sum().item()
        fn += ((1 - pred) * y).sum().item()
        for b in range(x.size(0)):
            p = pred[b, 0].cpu().numpy().astype(np.uint8)
            g = y[b, 0].cpu().numpy().astype(np.uint8)
            import cv2
            k = 2 * buffer_px + 1
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            gd, pd_ = cv2.dilate(g, ker), cv2.dilate(p, ker)
            rtp += np.logical_and(p, gd).sum()
            rfp += np.logical_and(p, np.logical_not(gd)).sum()
            rfn += np.logical_and(g, np.logical_not(pd_)).sum()

        # controlled occlusion -> recall of road inside hidden region
        occ_imgs, occ_masks = [], []
        for b in range(x.size(0)):
            oi, om = augment.occlude_tensor(x[b].cpu(), frac=occ_frac, holes=4, rng=rng)
            occ_imgs.append(oi); occ_masks.append(om)
        occ_x = torch.stack(occ_imgs).to(device)
        occ_region = torch.stack(occ_masks).unsqueeze(1).to(device)
        with autocast("cuda"):
            occ_pred = ME.binarize(net(occ_x))
        occ_tp += (occ_pred * y * occ_region).sum().item()
        occ_fn += ((1 - occ_pred) * y * occ_region).sum().item()

        if bi == 0:
            vis = (x[:6].cpu(), y[:6].cpu(), pred[:6].cpu(), occ_x[:6].cpu())

    eps = 1e-6
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    riou = rtp / (rtp + rfp + rfn + eps)
    occ_recall = occ_tp / (occ_tp + occ_fn + eps)
    return dict(loss=loss_m.avg, iou=iou, dice=dice, relaxed_iou=riou,
                occlusion_recall=occ_recall), vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--crop", type=int)
    ap.add_argument("--out")
    ap.add_argument("--arch")                 # unet | unetpp | deeplabv3plus
    ap.add_argument("--encoder")              # e.g. resnet34 | resnet50 | efficientnet-b4
    ap.add_argument("--loss", default="cldice")  # combined | lovasz | cldice
    ap.add_argument("--warmup-dice", type=int)   # cldice only: epochs of pure Dice before swapping to clDice
    ap.add_argument("--datasets", default="deepglobe")  # deepglobe | deepglobe+mass
    ap.add_argument("--val-every", type=int, default=1)  # validate every N epochs (saves ~40% time at N=3)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    seed_all(cfg["seed"])
    torch.backends.cudnn.benchmark = True

    epochs = args.epochs or cfg["train"]["epochs"]
    bs = args.batch_size or cfg["train"]["batch_size"]
    crop = args.crop or cfg["data"]["crop_size"]
    limit = args.limit if args.limit is not None else cfg["data"]["limit"]
    out_dir = args.out or cfg["train"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    device = cfg["train"]["device"] if torch.cuda.is_available() else "cpu"

    samples = DS.list_combined(args.datasets, cfg["data"]["train_dir"], cfg["data"].get("mass_root"))
    if not samples:
        sys.exit(f"[ERROR] No samples for datasets='{args.datasets}'. Check data paths in config.yaml.")
    tr_samples, va_samples = split_samples(samples, cfg["data"]["train_split"], cfg["seed"], limit)
    print(f"Datasets [{args.datasets}]: {len(samples)} real tiles | train {len(tr_samples)} | "
          f"val {len(va_samples)} | crop {crop} | bs {bs} | epochs {epochs} | device {device}")

    a = cfg["augment"]
    channels = cfg["model"].get("channels", augment.RGB)        # input bands (RGB + optional hue/sat/val/exg/nir)
    cfg["model"]["channels"] = channels                          # ensure it's saved in the checkpoint cfg
    cfg["model"]["in_channels"] = len(channels)                 # keep in_channels consistent with the band count
    tr_ds = DS.RoadSegDataset(tr_samples, augment.train_aug(crop, a["occlusion_prob"],
                              a["occlusion_max_holes"], a["occlusion_max_frac"], channels), crop, "train", cfg["data"]["road_thresh"])
    va_ds = DS.RoadSegDataset(va_samples, augment.val_aug(channels), crop, "val", cfg["data"]["road_thresh"])
    nw = cfg["data"]["num_workers"]
    tr_dl = DataLoader(tr_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True, drop_last=True)
    va_dl = DataLoader(va_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

    arch = args.arch or cfg["model"]["arch"]
    encoder = args.encoder or cfg["model"]["encoder"]
    print(f"  input channels ({len(channels)}): {channels}")
    net = M.build_model(arch, encoder, cfg["model"]["encoder_weights"],
                        len(channels), cfg["model"]["classes"]).to(device)
    if args.loss == "lovasz":
        main_crit = losses.LovaszDiceLoss().to(device)
    elif args.loss == "cldice":
        main_crit = losses.ClDiceDiceLoss().to(device)
    else:
        main_crit = losses.CombinedLoss(cfg["loss"]["dice"], cfg["loss"]["bce"], cfg["loss"]["connectivity"]).to(device)
    # clDice curriculum: warm up with pure Dice, then swap to clDice (skeletonizing early junk is noisy)
    warmup = (args.warmup_dice if args.warmup_dice is not None
              else cfg["loss"].get("warmup_dice_epochs", 0)) if args.loss == "cldice" else 0
    warmup_crit = losses.DiceLoss().to(device) if warmup else None
    print(f"  model: {arch} / {encoder} | loss: {args.loss}"
          f"{f' (Dice warmup {warmup}ep -> clDice)' if warmup else ''} | datasets: {args.datasets}")
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = GradScaler("cuda")

    hist = {k: [] for k in ["train_loss", "val_loss", "val_iou", "val_dice", "val_relaxed_iou", "val_occ_recall"]}
    best_iou, best_ep, patience = -1.0, -1, cfg["train"]["early_stop_patience"]
    csv_path = os.path.join(out_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_iou", "val_dice", "val_relaxed_iou", "val_occ_recall", "lr", "sec"])

    val_every = args.val_every
    for ep in range(1, epochs + 1):
        crit = warmup_crit if (warmup_crit is not None and ep <= warmup) else main_crit
        if warmup_crit is not None and ep == warmup + 1:
            print(f"  >> epoch {ep}: swapping Dice warmup -> clDice loss")
        net.train(); t0 = time.time(); tl = ME.RunningMean()
        for x, y in tqdm(tr_dl, desc=f"epoch {ep}/{epochs}", ncols=90):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with autocast("cuda"):
                loss = crit(net(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tl.update(loss.item(), x.size(0))
        sched.step()

        do_val = (ep % val_every == 0) or ep == epochs or ep == 1
        if do_val:
            val, vis = validate(net, va_dl, crit, device, cfg["eval"]["relaxed_buffer_px"], cfg["eval"]["occlusion_eval_frac"])
            dt = time.time() - t0
            hist["train_loss"].append(tl.avg); hist["val_loss"].append(val["loss"])
            hist["val_iou"].append(val["iou"]); hist["val_dice"].append(val["dice"])
            hist["val_relaxed_iou"].append(val["relaxed_iou"]); hist["val_occ_recall"].append(val["occlusion_recall"])
            print(f"  ep{ep}: train_loss {tl.avg:.4f} | val_loss {val['loss']:.4f} | IoU {val['iou']:.4f} | "
                  f"Dice {val['dice']:.4f} | RelaxedIoU {val['relaxed_iou']:.4f} | OccRecall {val['occlusion_recall']:.4f} | {dt:.0f}s")
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([ep, f"{tl.avg:.5f}", f"{val['loss']:.5f}", f"{val['iou']:.5f}",
                                        f"{val['dice']:.5f}", f"{val['relaxed_iou']:.5f}", f"{val['occlusion_recall']:.5f}",
                                        f"{opt.param_groups[0]['lr']:.2e}", f"{dt:.0f}"])
            viz.save_curves(hist, os.path.join(out_dir, "curves.png"))
            if vis is not None:
                viz.save_prediction_grid(vis[0], vis[1], vis[2], os.path.join(out_dir, f"preds_ep{ep:02d}.png"), occluded=vis[3])
            meta = {"arch": arch, "encoder": encoder, "loss": args.loss, "datasets": args.datasets}
            torch.save({"model": net.state_dict(), "cfg": cfg, "epoch": ep, **meta}, os.path.join(out_dir, "last.pt"))
            if val["iou"] > best_iou:
                best_iou, best_ep = val["iou"], ep
                torch.save({"model": net.state_dict(), "cfg": cfg, "epoch": ep, "val": val, **meta}, os.path.join(out_dir, "best.pt"))
            elif ep - best_ep >= patience * val_every:
                print(f"  early stop (no IoU improvement for {patience * val_every} epochs)"); break
        else:
            dt = time.time() - t0
            hist["train_loss"].append(tl.avg)
            for k in ["val_loss", "val_iou", "val_dice", "val_relaxed_iou", "val_occ_recall"]:
                hist[k].append(hist[k][-1] if hist[k] else 0.0)
            print(f"  ep{ep}: train_loss {tl.avg:.4f} | {dt:.0f}s (skip val)")
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([ep, f"{tl.avg:.5f}", "", "", "", "", "",
                                        f"{opt.param_groups[0]['lr']:.2e}", f"{dt:.0f}"])

    summary = dict(best_epoch=best_ep, best_val_iou=best_iou,
                   final=({k: hist[k][-1] for k in hist}), n_train=len(tr_samples), n_val=len(va_samples),
                   epochs_run=len(hist["train_loss"]))
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=2)
    print(f"\nDONE. Best IoU {best_iou:.4f} @ epoch {best_ep}. Artifacts in {out_dir}/")


if __name__ == "__main__":
    main()
