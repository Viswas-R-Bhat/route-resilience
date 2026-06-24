"""Phase 3 healing on REAL model predictions (the honest case).

Runs the trained baseline model on real tiles (CPU — does NOT touch the GPU that's
training), where occlusion leaves genuine small gaps, then heals those gaps and reports
the Connectivity Ratio improvement. This is the real input Phase 3 was designed for.
"""
import os, sys, glob
import numpy as np, cv2, torch, networkx as nx
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import model as M
from augment import IMAGENET_MEAN, IMAGENET_STD
from phase2_skeleton import mask_to_skeleton
from phase2_graph import skeleton_to_graph
from phase3_heal import heal_graph, connectivity_report

torch.set_num_threads(4)                      # be polite to the training's CPU dataloaders
DG = "C:/Users/VISWAS/route_data/deepglobe/train"
CKPT = "C:/Users/VISWAS/route_data/runs/phase1_full/best.pt"
OUT = "runs/phase3"; os.makedirs(OUT, exist_ok=True)
CROP, MAX_GAP, ANG_TOL = 512, 50, 35


def predict_mask(net, img):
    H, W = img.shape[:2]
    y, x = (H - CROP) // 2, (W - CROP) // 2
    crop = img[y:y + CROP, x:x + CROP]
    mean = np.array(IMAGENET_MEAN, np.float32); std = np.array(IMAGENET_STD, np.float32)
    t = ((crop.astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1).astype(np.float32)
    with torch.no_grad():
        prob = torch.sigmoid(net(torch.from_numpy(t).unsqueeze(0)))[0, 0].numpy()
    return crop, (prob > 0.5).astype(np.uint8)


def main():
    net = M.build_model("unet", "resnet34", None).to("cpu").eval()
    net.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=False)["model"])
    print("baseline model loaded on CPU")

    scored = []
    for s in sorted(glob.glob(os.path.join(DG, "*_sat.jpg")))[:40]:
        img = cv2.cvtColor(cv2.imread(s), cv2.COLOR_BGR2RGB)
        crop, mask = predict_mask(net, img)
        if mask.mean() < 0.01:
            continue
        G = skeleton_to_graph(mask_to_skeleton(mask))
        c = nx.number_connected_components(G)
        if c >= 3 and G.number_of_nodes() > 8:
            scored.append((c, crop, mask, G))
    scored.sort(key=lambda r: r[0], reverse=True)
    tiles = scored[:4]
    print(f"selected {len(tiles)} fragmented predictions to heal")

    fig, ax = plt.subplots(len(tiles), 3, figsize=(13, 4.3 * len(tiles)))
    ab = aa = 0
    for i, (c, crop, mask, G) in enumerate(tiles):
        H, healed = heal_graph(G, MAX_GAP, ANG_TOL)
        rep = connectivity_report(G, H)
        ab += rep["lcc_frac_before"]; aa += rep["lcc_frac_after"]
        print(f"  tile {i+1}: components {rep['components_before']}->{rep['components_after']} | "
              f"LCC {rep['lcc_frac_before']*100:.0f}%->{rep['lcc_frac_after']*100:.0f}% | +{healed} bridges")
        ax[i, 0].imshow(crop); ax[i, 0].axis("off")
        ax[i, 1].imshow(mask, cmap="gray"); ax[i, 1].axis("off")
        ax[i, 2].imshow(crop)
        for u, v, d in H.edges(data=True):
            if d.get("healed"):
                p1 = H.nodes[u]["pos"]; p2 = H.nodes[v]["pos"]
                ax[i, 2].plot([p1[0], p2[0]], [p1[1], p2[1]], "--", color="#ffcc00", lw=2.2)
            else:
                p = d["pts"]; ax[i, 2].plot(p[:, 1], p[:, 0], "-", color="#00e6bd", lw=1.2)
        ax[i, 2].axis("off")
        if i == 0:
            ax[i, 0].set_title("real satellite"); ax[i, 1].set_title("model prediction (gappy)")
            ax[i, 2].set_title("healed graph (gold = bridged gaps)")
    n = max(1, len(tiles))
    print(f"\n  Connectivity Ratio: mean LCC {ab/n*100:.0f}% -> {aa/n*100:.0f}% after healing")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "phase3_pred_healing.png"), dpi=110, bbox_inches="tight")
    print("saved", os.path.join(OUT, "phase3_pred_healing.png"))


if __name__ == "__main__":
    main()
