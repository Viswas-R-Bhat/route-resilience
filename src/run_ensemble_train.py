"""Train ensemble members sequentially: UNet++ → DeepLabV3+ → LinkNet (6ch clDice).

Launch detached — survives terminal close. Waits for an existing training PID if given.
Usage:
  python src/run_ensemble_train.py                    # start immediately
  python src/run_ensemble_train.py --wait-pid 17280   # wait for U-Net to finish first
"""
import subprocess, sys, os, time, argparse

PY = sys.executable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = "C:/Users/VISWAS/route_data/runs"

MEMBERS = [
    ("unetpp",        "resnet34", f"{RUNS}/phase1_6ch_unetpp"),
    ("deeplabv3plus", "resnet34", f"{RUNS}/phase1_6ch_dlv3p"),
    ("linknet",       "resnet34", f"{RUNS}/phase1_6ch_linknet"),
]


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def train_member(arch, encoder, out_dir):
    cmd = [PY, "src/train.py",
           "--arch", arch, "--encoder", encoder,
           "--loss", "cldice", "--out", out_dir, "--val-every", "3"]
    print(f"\n{'='*60}")
    print(f"  STARTING {arch}/{encoder}  ->  {out_dir}")
    print(f"{'='*60}\n", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT)
    dt = time.time() - t0
    status = "OK" if r.returncode == 0 else f"FAILED (rc={r.returncode})"
    print(f"\n  {arch} finished in {dt/60:.0f}min — {status}\n", flush=True)
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-pid", type=int, help="wait for this PID to exit before starting")
    args = ap.parse_args()

    if args.wait_pid:
        print(f"Waiting for PID {args.wait_pid} to finish...", flush=True)
        while pid_alive(args.wait_pid):
            time.sleep(30)
        print(f"PID {args.wait_pid} done.\n", flush=True)

    results = []
    for arch, enc, out in MEMBERS:
        ok = train_member(arch, enc, out)
        results.append((arch, ok))

    print(f"\n{'='*60}")
    print("  ENSEMBLE TRAINING COMPLETE")
    for arch, ok in results:
        print(f"    {arch:20s} {'OK' if ok else 'FAILED'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
