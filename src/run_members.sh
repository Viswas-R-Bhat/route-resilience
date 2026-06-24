#!/usr/bin/env bash
# Sequentially train two diverse ensemble members (proven combined-loss recipe,
# datasets=deepglobe seed=42 so they share phase1_full's exact held-out val split).
# Each writes best.pt/last.pt/metrics.csv/summary.json to its own out_dir.
PY="/c/Users/VISWAS/anaconda3/envs/route/python.exe"
ROOT="C:/Users/VISWAS/OneDrive/Desktop/Claude/ISRO Hackathon"
RUNS="C:/Users/VISWAS/route_data/runs"
cd "$ROOT"
EPOCHS=20

echo "[$(date +%H:%M:%S)] START member B: deeplabv3plus/resnet34 ($EPOCHS ep)"
"$PY" src/train.py --arch deeplabv3plus --encoder resnet34 --epochs $EPOCHS \
    --out "$RUNS/phase1_dlv3p" > "$RUNS/phase1_dlv3p_train.log" 2>&1
echo "[$(date +%H:%M:%S)] member B exit=$?"

echo "[$(date +%H:%M:%S)] START member C: unetpp/resnet34 ($EPOCHS ep)"
"$PY" src/train.py --arch unetpp --encoder resnet34 --epochs $EPOCHS \
    --out "$RUNS/phase1_unetpp" > "$RUNS/phase1_unetpp_train.log" 2>&1
echo "[$(date +%H:%M:%S)] member C exit=$?"

echo "DONE $(date +%H:%M:%S)" > "$RUNS/_members_DONE.txt"
echo "[$(date +%H:%M:%S)] ALL MEMBERS DONE"
