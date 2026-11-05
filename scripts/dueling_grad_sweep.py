#!/usr/bin/env python3
"""Run the dueling gradient probe across several Atari envs and seeds.

``dueling_instability_probe.py`` handles one env and one seed. The notes need
the per-stream gradient norms averaged over seeds, for both aggregation rules,
on environments that are hard enough for the split to matter. This drives that
script over a grid and leaves one ``instability.json`` per cell.

    .venv/bin/python scripts/dueling_grad_sweep.py --workers 5

Output layout:
    runs/dueling_grad_sweep/<env_slug>/seed<seed>/instability.json
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROBE = ROOT / "scripts" / "dueling_instability_probe.py"
OUT = ROOT / "runs" / "dueling_grad_sweep"

# Deliberately not toy control tasks: every one of these has a large,
# sparse reward signal and a policy that keeps changing late in training,
# which is where the aggregation rule has something to act on.
ENVS = [
    "ALE/Breakout-v5",
    "ALE/Seaquest-v5",
    "ALE/Qbert-v5",
    "ALE/BeamRider-v5",
    "ALE/Asterix-v5",
]
SEEDS = [0, 1, 2, 3, 4]


def slug(env_id: str) -> str:
    return env_id.replace("ALE/", "").replace("-v5", "").lower()


def cell(env_id: str, seed: int, steps: int, log_every: int, device: str) -> str:
    d = OUT / slug(env_id) / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    done = d / "instability.json"
    if done.exists():
        try:
            if json.loads(done.read_text()).get("steps") == steps:
                return f"skip  {slug(env_id):10s} seed{seed} (cached)"
        except Exception:
            pass
    cmd = [
        sys.executable, str(PROBE),
        "--env-id", env_id,
        "--steps", str(steps),
        "--log-every", str(log_every),
        "--seed", str(seed),
        "--device", device,
        "--out-dir", str(d),
    ]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    if r.returncode != 0:
        return f"FAIL  {slug(env_id):10s} seed{seed} ({dt:.0f}s)\n{r.stderr[-800:]}"
    return f"ok    {slug(env_id):10s} seed{seed} ({dt:.0f}s)"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--device", default="mps")
    p.add_argument("--envs", nargs="*", default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--out", default=None,
                   help="Output root (default runs/dueling_grad_sweep). Point "
                        "this elsewhere to re-measure into a fresh cache.")
    args = p.parse_args()

    global OUT
    if args.out:
        OUT = Path(args.out).resolve()

    envs = args.envs or ENVS
    seeds = args.seeds or SEEDS
    grid = [(e, s) for e in envs for s in seeds]
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"{len(grid)} cells, {args.workers} workers, "
          f"{args.steps} steps each, device={args.device}", flush=True)
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(cell, e, s, args.steps, args.log_every, args.device): (e, s)
            for e, s in grid
        }
        for i, fut in enumerate(cf.as_completed(futs), 1):
            print(f"[{i}/{len(grid)}] {fut.result()}", flush=True)
    print(f"\ntotal {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
