#!/usr/bin/env python3
"""Generate golden gate-mask references from the dva-quad-jax JAX reference.

These goldens are the *correctness oracle* for the C++ renderer: the C++
`gate_render_cli` is fed the exact same (drone_pose, gate_pose) cases and its
output is compared pixel-for-pixel (binary) / within tolerance (soft) against
the masks produced here by the original training-time JAX rasteriser.

Run from anywhere; set DVA_ROOT if the reference repo lives elsewhere.

    python3 gen_golden.py            # writes test/golden/*.npy + cases.txt + meta.json

Reference functions:
  envs.tasks.gate_traversal_binary_mask        (1-cam: CameraConfig)
  envs.tasks.gate_traversal_binary_mask_2_cam  (2-cam: default_front_two_cam_configs)
  envs.tasks.gate_traversal_binary_mask_4_cam  (4-cam: default_four_cam_configs)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

DVA_ROOT = os.environ.get("DVA_ROOT", "/home/nat-nus/apg/dva-quad-jax")
sys.path.insert(0, DVA_ROOT)

import jax.numpy as jnp  # noqa: E402

from envs.tasks.gate_traversal_binary_mask import (  # noqa: E402
    CameraConfig,
    _compute_vis_obs_jax,
)
from envs.tasks.gate_traversal_binary_mask_2_cam import (  # noqa: E402
    default_front_two_cam_configs,
    _compute_vis_obs_2cam_jax,
)
from envs.tasks.gate_traversal_binary_mask_4_cam import (  # noqa: E402
    default_four_cam_configs,
    _compute_vis_obs_4cam_jax,
)

TAU = 1.5  # soft-mask sharpness (matches CameraConfig.line_thickness default)


def euler_to_quat_xyzw(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX Tait-Bryan -> [x,y,z,w]  (matches gate/task.euler_to_quat_xyzw)."""
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    q = np.array([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ], dtype=np.float64)
    return q / max(np.linalg.norm(q), 1e-9)


# (drone_pos, drone_rpy, gate_pos, gate_rpy) — sim world ENU / body FLU.
# Spans: centred / lateral / yawed / rolled-gate / close / behind (near-clip) /
# moved-and-rotated gate.
CASES = [
    ((-1.80, 0.00, 0.00), (0.0, 0.0, 0.0),    (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ((-0.80, 0.00, 0.00), (0.0, 0.0, 0.0),    (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ((-1.50, 0.25, 0.10), (0.0, 0.0, 0.0),    (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ((-1.50, 0.40, 0.00), (0.0, 0.0, -0.20),  (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ((-1.20, 0.00, 0.20), (0.15, 0.25, 0.05), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ((-1.50, 0.00, 0.00), (0.0, 0.0, 0.0),    (0.0, 0.0, 0.0), (0.60, 0.0, 0.0)),
    ((-1.50, 0.00, 0.00), (0.0, 0.0, 0.0),    (0.0, 0.0, 0.0), (0.0, 0.0, 0.30)),
    ((-0.40, 0.10, 0.00), (0.0, 0.30, 0.0),   (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    (( 0.50, 0.00, 0.00), (0.0, 0.0, 0.0),    (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ((-1.00, 0.80, 0.30), (0.1, 0.1, -0.40),  (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    (( 0.30, 1.00, 1.50), (0.0, 0.0, 1.00),   (2.0, 1.0, 1.5), (0.20, 0.0, 1.00)),
    ((-1.30, -0.30, -0.20), (-0.1, 0.2, 0.15),(0.0, 0.0, 0.0), (0.0, 0.10, -0.20)),
]


def build_state(cases):
    dpos = np.array([c[0] for c in cases], dtype=np.float32)
    dquat = np.array([euler_to_quat_xyzw(*c[1]) for c in cases], dtype=np.float32)
    gpos = np.array([c[2] for c in cases], dtype=np.float32)
    gquat = np.array([euler_to_quat_xyzw(*c[3]) for c in cases], dtype=np.float32)
    state = SimpleNamespace(
        pos=jnp.asarray(dpos), quat=jnp.asarray(dquat),
        gate_pos=jnp.asarray(gpos), gate_quat=jnp.asarray(gquat),
    )
    return state, dpos, dquat, gpos, gquat


def main():
    out = Path(__file__).resolve().parent.parent / "test" / "golden"
    out.mkdir(parents=True, exist_ok=True)

    state, dpos, dquat, gpos, gquat = build_state(CASES)
    n = len(CASES)

    cam1 = CameraConfig()
    cams2 = default_front_two_cam_configs()
    cams4 = default_four_cam_configs()

    H1, W1 = cam1.height, cam1.width
    H2, W2 = cams2[0].height, cams2[0].width
    H4, W4 = cams4[0].height, cams4[0].width

    specs = {
        1: (H1, W1 * 1),
        2: (H2, W2 * 2),
        4: (H4, W4 * 4),
    }

    def render(ncam, mode):
        tau = TAU if mode == "soft" else None
        if ncam == 1:
            m = _compute_vis_obs_jax(state, cam1, mode, TAU if mode == "soft" else 1.5)
            H, W = H1, W1 * 1
        elif ncam == 2:
            m = _compute_vis_obs_2cam_jax(state, cams2[0], cams2[1], mode, tau)
            H, W = H2, W2 * 2
        else:
            m = _compute_vis_obs_4cam_jax(state, cams4, mode, tau)
            H, W = H4, W4 * 4
        return np.asarray(m, dtype=np.float32).reshape(n, H, W)

    for ncam in (1, 2, 4):
        for mode in ("wireframe", "filled", "soft"):
            arr = render(ncam, mode)
            np.save(out / f"golden_{ncam}cam_{mode}.npy", arr)
            nz = float((arr > 0.5).mean())
            print(f"  {ncam}cam {mode:9s} -> {arr.shape}  frac_on={nz:.4f}")

    # cases.txt: 14 floats/line  drone(px py pz qx qy qz qw) gate(px py pz qx qy qz qw)
    with open(out / "cases.txt", "w") as f:
        for i in range(n):
            row = list(dpos[i]) + list(dquat[i]) + list(gpos[i]) + list(gquat[i])
            f.write(" ".join(f"{v:.9g}" for v in row) + "\n")

    meta = {
        "tau": TAU,
        "n_cases": n,
        "dims": {str(k): {"H": v[0], "W": v[1]} for k, v in specs.items()},
        "cam1": dict(width=cam1.width, height=cam1.height, fx=cam1.fx, fy=cam1.fy,
                     cx=cam1.cx, cy=cam1.cy, near=cam1.near,
                     line_thickness=cam1.line_thickness, pitch_deg=cam1.pitch_deg,
                     offset_b=list(cam1.offset_b)),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print("wrote goldens to", out)


if __name__ == "__main__":
    main()
