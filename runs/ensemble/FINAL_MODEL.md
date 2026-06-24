# Final model — locked in (Phase 1, road segmentation)

**Production model:** `route_data/runs/phase1_full/best.pt`
**Architecture:** U-Net / ResNet34 (ImageNet-pretrained), single binary road class.
**Trained on:** DeepGlobe only, 0.85/0.15 split, seed 42 (5292 train / 934 held-out val tiles).
**Loss:** Dice + BCE + Connectivity. **Schedule:** AdamW, lr 3e-4, cosine, AMP. Best epoch 22 / 25.

## Inference recipe (this is the "config" that ships)
1. **8-way D4 test-time augmentation** — average sigmoid probabilities over the 8 dihedral views (identity, 3 rotations, ×{no-flip, h-flip}).
2. **Decision threshold 0.475** — IoU-optimal on the real held-out val (swept 0.20–0.70).
3. *(Optional)* light post-processing (remove blobs < 80 px, close 3 px) — trades a hair of strict IoU for better connectivity/Relaxed-IoU.

Run it: `python src/ensemble_eval.py --member unet resnet34 <best.pt> --out runs/ensemble`

## Metrics — real held-out DeepGlobe val (934 tiles, never seen in training)

| Config | IoU | Dice | Relaxed-IoU |
|---|---|---|---|
| Baseline (thr 0.50, no TTA) | 0.6004 | 0.7503 | 0.7652 |
| **+ D4 TTA + thr 0.475  ← SHIPPED** | **0.6054** | **0.7542** | **0.7697** |
| + post-processing | 0.6027 | 0.7521 | 0.7683 |

Occlusion-recall (road correctly recovered inside synthetically hidden regions): **0.63**.
Net real gain from the free levers: **+0.50 pt IoU**, **+0.45 pt Relaxed-IoU** — no retraining.

## Why no ensemble (honest negative result)
A 3-model soft-voting ensemble was evaluated (U-Net 0.605, DeepLabV3+ 0.598, U-Net++ ep11 0.587, all ResNet34). The equal-weight vote scored 0.604 — **below** the best single model — and the greedy selector (adds a member only if it raises val IoU) **kept U-Net alone**. Cause: members share the same encoder + training data, so their errors are correlated, and two were weaker. Voting helps only with comparably-strong, decorrelated members.

**To make an ensemble pay off later:** add a genuinely diverse encoder (EfficientNet / transformer) and train U-Net++ to full convergence, then weighted soft-vote.

_Artifacts: [ensemble_eval.json](ensemble_eval.json), ensemble_compare.png, ensemble_qualitative.png._
