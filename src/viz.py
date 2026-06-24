"""Visualisation helpers — every training run produces real, inspectable figures."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_MEAN = np.array([0.485, 0.456, 0.406])
_STD = np.array([0.229, 0.224, 0.225])


def denorm(img_t):
    x = img_t.detach().cpu().numpy().transpose(1, 2, 0)
    return (x * _STD + _MEAN).clip(0, 1)


def save_prediction_grid(images, gts, preds, path, occluded=None, n=6):
    """Columns: input | ground truth | prediction [| occluded input]."""
    n = min(n, len(images))
    cols = 4 if occluded is not None else 3
    titles = ["Satellite (real)", "Ground truth", "Prediction"] + (["Occluded input"] if cols == 4 else [])
    fig, axes = plt.subplots(n, cols, figsize=(cols * 3.0, n * 3.0))
    if n == 1:
        axes = axes[None, :]
    for r in range(n):
        cells = [denorm(images[r]), gts[r].squeeze().cpu().numpy(), preds[r].squeeze().cpu().numpy()]
        if cols == 4:
            cells.append(denorm(occluded[r]))
        for c in range(cols):
            ax = axes[r, c]
            if c in (1, 2):
                ax.imshow(cells[c], cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(cells[c])
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_curves(history, path):
    """history: dict of lists -> train_loss, val_loss, val_iou, val_dice (per epoch)."""
    ep = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, history["train_loss"], label="train loss")
    ax[0].plot(ep, history["val_loss"], label="val loss")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend(); ax[0].set_title("Loss")
    ax[1].plot(ep, history["val_iou"], label="val IoU")
    ax[1].plot(ep, history["val_dice"], label="val Dice")
    if "val_occ_recall" in history:
        ax[1].plot(ep, history["val_occ_recall"], label="occlusion-recall")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("score"); ax[1].legend(); ax[1].set_title("Validation metrics")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
