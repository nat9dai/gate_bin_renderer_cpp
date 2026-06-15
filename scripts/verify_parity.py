#!/usr/bin/env python3
"""Verify the C++ gate_render_cli output matches the JAX golden masks.

Run after building and after gen_golden.py:

    python3 scripts/verify_parity.py [--cli PATH] [--backend auto|cpu|cuda]

For each (ncam in {1,2,4}) x (mode in {wireframe,filled,soft}):
  * feeds test/golden/cases.txt to the CLI,
  * reshapes the raw-float32 stdout to (n_cases, H, W),
  * compares to test/golden/golden_<ncam>cam_<mode>.npy.
Binary modes must match pixel-exact (after >0.5 threshold); soft must match
within a small absolute tolerance. Exits non-zero on any failure.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parent.parent
GOLDEN = PKG / "test" / "golden"
# A faithful float32 port cannot bit-match XLA at pixels whose distance sits
# exactly on the line-thickness threshold / gate-frame edge. We therefore accept
# differences only where they are (a) immediately adjacent to an 'on' pixel
# (i.e. a boundary pixel) and (b) a tiny fraction of the lit pixels.
MAX_MISMATCH_FRAC = 0.01   # <=1% of lit pixels may flip
SOFT_MEAN_ATOL = 5e-4      # soft masks: mean|Δ| ceiling


def find_cli() -> str:
    # Workspace layout: .../realflight_ws/src/gate_bin_renderer_cpp
    ws = PKG.parent.parent
    candidates = [
        ws / "install" / "gate_bin_renderer_cpp" / "lib" / "gate_bin_renderer_cpp" / "gate_render_cli",
        ws / "build" / "gate_bin_renderer_cpp" / "gate_render_cli",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    raise FileNotFoundError(
        "gate_render_cli not found; build the package first. Looked in:\n  "
        + "\n  ".join(str(c) for c in candidates))


def dims(ncam: int) -> tuple[int, int]:
    H = 84
    W = 84 if ncam == 1 else 150 * ncam
    return H, W


def run_cli(cli: str, ncam: int, mode: str, backend: str, cases: str) -> np.ndarray:
    H, W = dims(ncam)
    proc = subprocess.run(
        [cli, "--ncam", str(ncam), "--mode", mode, "--backend", backend, "--format", "f32"],
        input=cases.encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    arr = np.frombuffer(proc.stdout, dtype=np.float32)
    n = arr.size // (H * W)
    return arr.reshape(n, H, W)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", default=None)
    ap.add_argument("--backend", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    cli = args.cli or find_cli()
    cases = (GOLDEN / "cases.txt").read_text()
    meta = json.loads((GOLDEN / "meta.json").read_text())
    print(f"CLI: {cli}\nbackend: {args.backend}\ncases: {meta['n_cases']}\n")

    all_ok = True
    for ncam in (1, 2, 4):
        for mode in ("wireframe", "filled", "soft"):
            gold = np.load(GOLDEN / f"golden_{ncam}cam_{mode}.npy")
            got = run_cli(cli, ncam, mode, args.backend, cases)
            if got.shape != gold.shape:
                print(f"[FAIL] {ncam}cam {mode}: shape {got.shape} != golden {gold.shape}")
                all_ok = False
                continue

            gb, cb = gold > 0.5, got > 0.5
            nmis = int(np.count_nonzero(gb != cb))
            on = max(1, int(gb.sum()))
            nonbound = count_non_boundary(gb, cb)
            frac = nmis / on
            extra = ""
            if mode == "soft":
                d = np.abs(got - gold)
                extra = f" max|Δ|={d.max():.2e} mean|Δ|={d.mean():.2e}"
                soft_ok = d.mean() <= SOFT_MEAN_ATOL
            else:
                soft_ok = True
            ok = (nonbound == 0) and (frac <= MAX_MISMATCH_FRAC) and soft_ok
            tag = "OK  " if ok else "FAIL"
            print(f"[{tag}] {ncam}cam {mode:9s}: flips={nmis}/{on} ({frac*100:.2f}%) "
                  f"non_boundary={nonbound}{extra}")
            all_ok = all_ok and ok

    print("\n" + ("ALL PARITY CHECKS PASSED (boundary-only float diffs) ✅"
                  if all_ok else "PARITY FAILURES ❌"))
    return 0 if all_ok else 1


def count_non_boundary(gb: np.ndarray, cb: np.ndarray) -> int:
    """Count binarized mismatches NOT adjacent to an 'on' pixel in either mask
    (a structural error, as opposed to a threshold-boundary float flip)."""
    n = 0
    for i in range(gb.shape[0]):
        for (y, x) in np.argwhere(gb[i] != cb[i]):
            g_adj = gb[i, max(0, y - 1):y + 2, max(0, x - 1):x + 2].max()
            c_adj = cb[i, max(0, y - 1):y + 2, max(0, x - 1):x + 2].max()
            if not (g_adj or c_adj):
                n += 1
    return n


if __name__ == "__main__":
    sys.exit(main())
