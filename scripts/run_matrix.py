"""Run the full 2x2x2 (+ ancillary) experiment matrix across seeds.

This is the script that produces the data behind the paper's main table. It iterates
over all experiment YAMLs in configs/experiments/ and N seeds, writes one row per
(cell, seed) to results/matrix.csv.
"""
from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs-dir", type=Path, default=Path("configs/experiments"))
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    configs = sorted(args.configs_dir.glob("*.yaml"))

    for cfg, seed in itertools.product(configs, seeds):
        cmd = [
            sys.executable,
            "scripts/train_single.py",
            "--config", str(cfg),
            "--seed", str(seed),
            "--output-dir", str(args.output_dir),
        ]
        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
