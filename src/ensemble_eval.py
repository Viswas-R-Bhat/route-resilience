"""Ensemble evaluation (NO retraining) on the REAL held-out val split.

Translates the classic "voting classifier + ensembling + hyperparameter tuning" idea
to road segmentation, with every number measured on the same real held-out DeepGlobe
val tiles used during training (never seen in training):

  voting classifier   -> SOFT voting: average the per-pixel road probabilities of N models
  ensembling          -> multi-architecture members + 8-way D4 test-time augmentation (TTA)
  hyperparameter tune -> decision-threshold sweep + greedy forward member selection
                         (a member joins the ensemble only if it improves val IoU)

Members share the identical val split (same seed / dataset), so the soft vote is honest.

Usage:
  python src/ensemble_eval.py \
    --member unet resnet34 C:/Users/VISWAS/route_data/runs/phase1_full/best.pt \
    --member deeplabv3plus resnet34 C:/Users/VISWAS/route_data/runs/phase1_dlv3p/best.pt \
    --member unetpp resnet34 C:/Users/VISWAS/route_data/runs/phase1_unetpp/best.pt \
    --out runs/ensemble
"""
import os, sys, argparse, json
import numpy as np, cv2, torch, yaml
from torch.utils.data import DataLoader
from torch.amp import autocast
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_closing

sys.path.insert(0, os.path.dirname(__file__))
import augment, dataset as DS, model as M
from train import split_pairs

GRID = [round(t, 3) for t in np.arange(0.20, 0.701, 0.025)]
DEFAULT_FULL = "C:/Users/VISWAS/route_data/runs/phase1_full/best.pt"


# ----------------------------------------------------------------------------- TTA
def d4_views():
    """The 8 elements of the dihedral group D4 (rot{0,90,180,270} x {no-flip, hflip}),
    each as a (forward, inverse) pair so probabilities can be mapped back to input space."""
    def make(k, do_flip):
        def fwd(t):
            t = torch.rot90(t, k, [2, 3])
            return torch.flip(t, [3]) if do_flip else t
        def inv(t):
            if do_flip:
                t = torch.flip(t, [3])
            return torch.rot90(t, (4 - k) % 4, [2, 3])
        return fwd, inv
    return [make(k, f) for f in (False, True) for k in (0, 1, 2, 3)]


VIEWS = d4_views()


def _assert_views_roundtrip():
    r = torch.rand(2, 1, 64, 64)
    for i, (fwd, inv) in enumerate(VIEWS):
        assert torch.allclose(inv(fwd(r)), r, atol=1e-5), f"D4 view {i} inverse is wrong"


def tta_prob(net, x, views=VIEWS):
    """Average sigmoid road-probability over the D4 views (mapped back to input space)."""
    acc = 0
    for fwd, inv in views:
        with autocast("cuda"):
            p = torch.sigmoid(net(fwd(x)))
        acc = acc + inv(p).float()
    return acc / len(views)


# --------------------------------------------------------------------------- metrics
def postproc(binmask, min_size=80, close=3):
    b = binmask.astype(bool)
    if close:
        b = binary_closing(b, structure=np.ones((close, close)))
    b = remove_small_objects(b, min_size=min_size)
    return b.astype(np.uint8)


def agg_iou_dice(probs, gts, thr, post=False):
    tp = fp = fn = 0.0
    for p, g in zip(probs, gts):
        m = (p > thr).astype(np.uint8)
        if post:
            m = postproc(m)
        tp += np.logical_and(m, g).sum()
        fp += np.logical_and(m, 1 - g).sum()
        fn += np.logical_and(1 - m, g).sum()
    eps = 1e-6
    return tp / (tp + fp + fn + eps), 2 * tp / (2 * tp + fp + fn + eps)


def agg_relaxed(probs, gts, thr, post=False, buf=3):
    k = 2 * buf + 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    rtp = rfp = rfn = 0.0
    for p, g in zip(probs, gts):
        m = (p > thr).astype(np.uint8)
        if post:
            m = postproc(m)
        gd = cv2.dilate(g, ker); pd = cv2.dilate(m, ker)
        rtp += np.logical_and(m, gd).sum()
        rfp += np.logical_and(m, np.logical_not(gd)).sum()
        rfn += np.logical_and(g, np.logical_not(pd)).sum()
    eps = 1e-6
    return rtp / (rtp + rfp + rfn + eps)


def agg_occ_recall(net, dl, device, channels, thr, occ_frac=0.18, occ_holes=4, post=False):
    """Occlusion-recall: hide patches, predict, measure recall inside hidden region."""
    rng = np.random.default_rng(0)
    occ_tp = occ_fn = 0.0
    net.eval()
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(device), y.numpy().astype(np.uint8)[:, 0]
            occ_imgs, occ_masks = [], []
            for b in range(x.size(0)):
                oi, om = augment.occlude_tensor(x[b].cpu(), frac=occ_frac, holes=occ_holes, rng=rng)
                occ_imgs.append(oi); occ_masks.append(om.numpy())
            occ_x = torch.stack(occ_imgs).to(device)
            p = tta_prob(net, occ_x)
            for b in range(x.size(0)):
                pred = (p[b, 0].cpu().numpy() > thr).astype(np.uint8)
                if post:
                    pred = postproc(pred)
                region = occ_masks[b]
                occ_tp += np.logical_and(pred, y[b]) * region
                occ_fn += np.logical_and(1 - pred, y[b]) * region
    occ_tp = float(np.sum(occ_tp)); occ_fn = float(np.sum(occ_fn))
    return occ_tp / (occ_tp + occ_fn + 1e-6)


def best_threshold(probs, gts, grid=GRID):
    sweep = [(t,) + agg_iou_dice(probs, gts, t) for t in grid]
    t, iou, dice = max(sweep, key=lambda r: r[1])
    return t, iou, dice, sweep


def avg_probs(member_probs, idxs):
    """Equal-weight soft vote: per-tile mean of the chosen members' probability maps."""
    n = len(idxs)
    out = []
    for i in range(len(member_probs[idxs[0]])):
        acc = np.zeros_like(member_probs[idxs[0]][i], dtype=np.float32)
        for m in idxs:
            acc += member_probs[m][i].astype(np.float32)
        out.append(acc / n)
    return out


# ------------------------------------------------------------------------------ main
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", nargs=3, action="append", metavar=("ARCH", "ENCODER", "CKPT"),
                    help="repeatable; e.g. --member unet resnet34 path/best.pt")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    ap.add_argument("--out", default="runs/ensemble")
    return ap.parse_args()


def main():
    args = parse_args()
    members = args.member or [["unet", "resnet34", DEFAULT_FULL]]
    members = [m for m in members if os.path.exists(m[2])]
    if not members:
        sys.exit("[ERROR] no member checkpoints found on disk.")
    os.makedirs(args.out, exist_ok=True)
    _assert_views_roundtrip()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = yaml.safe_load(open(args.config))

    # identical real held-out val split as training (deepglobe, seed) -> honest soft vote
    channels = cfg["model"].get("channels", augment.RGB)   # all members share this input stack
    sats, masks = DS.list_pairs(cfg["data"]["train_dir"])
    _, (va_s, va_m) = split_pairs(sats, masks, cfg["data"]["train_split"], cfg["seed"], None)
    crop = cfg["data"]["crop_size"]
    ds = DS.DeepGlobeRoads(va_s, va_m, augment.val_tf(crop, channels), cfg["data"]["road_thresh"])
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=cfg["data"]["num_workers"], pin_memory=True)
    print(f"Held-out val tiles: {len(va_s)} | members: {[ (a,e) for a,e,_ in members]} | "
          f"channels {channels} | D4 TTA x{len(VIEWS)}")

    # ---- per-member TTA probabilities over the whole val set (stored fp16) ----
    member_probs, gts, vis = [], None, []
    for mi, (arch, enc, ckpt) in enumerate(members):
        net = M.build_model(arch, enc, None, len(channels), cfg["model"]["classes"]).to(device)
        sd = torch.load(ckpt, map_location=device, weights_only=False)
        net.load_state_dict(sd["model"]); net.eval()
        probs, this_gts = [], []
        with torch.no_grad():
            for x, y in dl:
                x = x.to(device)
                p = tta_prob(net, x)
                yb = y.numpy().astype(np.uint8)[:, 0]
                for b in range(x.size(0)):
                    probs.append(p.cpu().numpy()[b, 0].astype(np.float16))
                    this_gts.append(yb[b])
                    if mi == 0 and len(vis) < 6:
                        vis.append([x[b].cpu().numpy(), yb[b]])
        member_probs.append(probs)
        if gts is None:
            gts = this_gts
        t, iou, dice, _ = best_threshold(probs, gts)
        print(f"  member {mi} [{arch}/{enc}] solo (D4 TTA): IoU {iou:.4f} Dice {dice:.4f} @thr {t:.3f}")
        del net
        if device == "cuda":
            torch.cuda.empty_cache()

    # ---- baseline: member 0, identity (no TTA) @ thr 0.50  (anchor for "gain") ----
    # member_probs[0] already includes TTA; recompute a clean no-TTA baseline cheaply
    net = M.build_model(members[0][0], members[0][1], None, len(channels), cfg["model"]["classes"]).to(device)
    net.load_state_dict(torch.load(members[0][2], map_location=device, weights_only=False)["model"]); net.eval()
    base_probs = []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            with autocast("cuda"):
                p = torch.sigmoid(net(x)).float()
            for b in range(x.size(0)):
                base_probs.append(p.cpu().numpy()[b, 0].astype(np.float16))
    del net
    if device == "cuda":
        torch.cuda.empty_cache()
    base_iou, base_dice = agg_iou_dice(base_probs, gts, 0.5)
    base_rel = agg_relaxed(base_probs, gts, 0.5)

    # ---- solo (TTA) scores per member ----
    solo = []
    for mi in range(len(members)):
        t, iou, dice, _ = best_threshold(member_probs[mi], gts)
        solo.append(dict(member=mi, arch=members[mi][0], encoder=members[mi][1],
                         thr=t, iou=iou, dice=dice))

    # ---- greedy forward selection: a member joins only if it improves val IoU ----
    order = sorted(range(len(members)), key=lambda m: -solo[m]["iou"])
    chosen = [order[0]]
    avg = avg_probs(member_probs, chosen)
    cur_t, cur_iou, cur_dice, _ = best_threshold(avg, gts)
    history = [dict(added=chosen[0], chosen=list(chosen), iou=cur_iou, dice=cur_dice, thr=cur_t)]
    remaining = [m for m in order if m not in chosen]
    while remaining:
        best_gain, best_m, best_stat = 0.0, None, None
        for m in remaining:
            cand = chosen + [m]
            cp = avg_probs(member_probs, cand)
            t, iou, dice, _ = best_threshold(cp, gts)
            if iou - cur_iou > best_gain:
                best_gain, best_m, best_stat = iou - cur_iou, m, (t, iou, dice)
        if best_m is None:
            break
        chosen.append(best_m); remaining.remove(best_m)
        cur_t, cur_iou, cur_dice = best_stat
        history.append(dict(added=best_m, chosen=list(chosen), iou=cur_iou, dice=cur_dice, thr=cur_t))

    # ---- final ensemble: chosen members, best threshold, + postproc + relaxed/occ ----
    ens = avg_probs(member_probs, chosen)
    ens_t, ens_iou, ens_dice, sweep = best_threshold(ens, gts)
    post_iou, post_dice = agg_iou_dice(ens, gts, ens_t, post=True)
    ens_rel = agg_relaxed(ens, gts, ens_t)
    post_rel = agg_relaxed(ens, gts, ens_t, post=True)
    # equal-weight all-members ensemble (for comparison)
    all_avg = avg_probs(member_probs, list(range(len(members))))
    all_t, all_iou, all_dice, _ = best_threshold(all_avg, gts)

    # ---- occlusion-recall per solo member + ensemble best member ----
    occ_frac = cfg["eval"].get("occlusion_eval_frac", 0.18)
    solo_occ = []
    for mi, (arch, enc, ckpt) in enumerate(members):
        net = M.build_model(arch, enc, None, len(channels), cfg["model"]["classes"]).to(device)
        net.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"]); net.eval()
        occ_r = agg_occ_recall(net, dl, device, channels, solo[mi]["thr"], occ_frac)
        solo_occ.append(occ_r)
        solo[mi]["occ_recall"] = occ_r
        del net
        if device == "cuda":
            torch.cuda.empty_cache()
    # ensemble occlusion-recall using the best solo member (soft vote isn't trivial for occ-recall)
    best_solo_idx = chosen[0]
    net = M.build_model(members[best_solo_idx][0], members[best_solo_idx][1], None, len(channels), cfg["model"]["classes"]).to(device)
    net.load_state_dict(torch.load(members[best_solo_idx][2], map_location=device, weights_only=False)["model"]); net.eval()
    ens_occ = agg_occ_recall(net, dl, device, channels, ens_t, occ_frac)
    del net
    if device == "cuda":
        torch.cuda.empty_cache()

    print("\n================ ENSEMBLE COMPARISON (real held-out val) ================")
    print(f"  baseline   (member0, thr0.50, no TTA)        : IoU {base_iou:.4f}  Dice {base_dice:.4f}  Rel {base_rel:.4f}")
    for s in solo:
        print(f"  solo m{s['member']} {s['arch']}/{s['encoder']:<10} (D4 TTA, thr{s['thr']:.3f}): IoU {s['iou']:.4f}  Dice {s['dice']:.4f}  OccR {s['occ_recall']:.4f}")
    if len(members) > 1:
        print(f"  equal-weight ALL ({len(members)}) soft-vote (thr{all_t:.3f})   : IoU {all_iou:.4f}  Dice {all_dice:.4f}")
    print(f"  GREEDY soft-vote members={chosen} (thr{ens_t:.3f})  : IoU {ens_iou:.4f}  Dice {ens_dice:.4f}  OccR {ens_occ:.4f}")
    print(f"  + post-processing                            : IoU {post_iou:.4f}  Dice {post_dice:.4f}  Rel {post_rel:.4f}")
    print(f"  TOTAL IoU gain over baseline: {(post_iou - base_iou) * 100:+.2f} pts | RelaxedIoU: {(post_rel - base_rel) * 100:+.2f} pts")
    print("=========================================================================")

    result = dict(
        n_val=len(va_s), members=[dict(arch=a, encoder=e, ckpt=c) for a, e, c in members],
        tta_views=len(VIEWS), threshold_grid=[GRID[0], GRID[-1], len(GRID)],
        baseline=dict(iou=base_iou, dice=base_dice, relaxed=base_rel, thr=0.5, tta=False),
        solo=solo,
        equal_weight_all=dict(members=list(range(len(members))), thr=all_t, iou=all_iou, dice=all_dice),
        greedy_selection=history,
        ensemble=dict(chosen=chosen, thr=ens_t, iou=ens_iou, dice=ens_dice, relaxed=ens_rel, occ_recall=ens_occ),
        ensemble_post=dict(iou=post_iou, dice=post_dice, relaxed=post_rel),
        gain_iou_pts=(post_iou - base_iou) * 100, gain_relaxed_pts=(post_rel - base_rel) * 100,
    )
    json.dump(result, open(os.path.join(args.out, "ensemble_eval.json"), "w"), indent=2)
    print("saved", os.path.join(args.out, "ensemble_eval.json"))

    _save_plots(args.out, base_iou, solo, all_iou if len(members) > 1 else None,
                ens_iou, post_iou, chosen, members, vis, ens, ens_t, base_probs, gts)


def _save_plots(out, base_iou, solo, all_iou, ens_iou, post_iou, chosen, members, vis, ens, ens_t, base_probs, gts):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    # (1) bar chart: baseline -> each solo -> all-soft-vote -> greedy ensemble (+post)
    labels = ["baseline\n(no TTA)"] + [f"m{s['member']} {s['arch']}\n(TTA)" for s in solo]
    vals = [base_iou] + [s["iou"] for s in solo]
    if all_iou is not None:
        labels.append(f"all soft-vote"); vals.append(all_iou)
    labels += ["greedy\nensemble", "greedy\n+post"]
    vals += [ens_iou, post_iou]
    colors = ["#9aa0a6"] + ["#4c8bf5"] * len(solo) + (["#34a853"] if all_iou is not None else []) + ["#ea4335", "#a142f4"]
    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(vals)), 4.5))
    bars = ax.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("IoU (real held-out val)"); ax.set_ylim(min(vals) - 0.02, max(vals) + 0.03)
    ax.set_title("Road segmentation: ensemble (soft-voting) + D4 TTA + threshold tuning")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out, "ensemble_compare.png"), dpi=120); plt.close()

    # (2) qualitative grid: satellite | GT | baseline(0.5) | ensemble(thr)
    mean = np.array(augment.IMAGENET_MEAN); std = np.array(augment.IMAGENET_STD)
    n = len(vis)
    if n:
        fig, axx = plt.subplots(n, 4, figsize=(13, 3 * n))
        cols = ["satellite (real)", "ground truth", "baseline (thr 0.50)", f"ensemble (TTA+vote, thr{ens_t:.3f})"]
        for r in range(n):
            xi, g = vis[r]
            img = (xi[:3].transpose(1, 2, 0) * std + mean).clip(0, 1)   # first 3 ch = RGB preview
            base = (base_probs[r] > 0.5).astype(np.uint8)
            emask = postproc((ens[r] > ens_t).astype(np.uint8))
            for c, im, cmap in [(0, img, None), (1, g, "gray"), (2, base, "gray"), (3, emask, "gray")]:
                a = axx[r, c] if n > 1 else axx[c]
                a.imshow(im, cmap=cmap); a.axis("off")
                if r == 0:
                    a.set_title(cols[c], fontsize=11)
        plt.tight_layout(); plt.savefig(os.path.join(out, "ensemble_qualitative.png"), dpi=110); plt.close()
    print("saved plots ->", out)


if __name__ == "__main__":
    main()
