#!/usr/bin/env python3
"""End-to-end ROS test for gate_bin_renderer_node.

For a few golden cases it:
  1. inverse-converts the sim-frame (ENU/FLU) drone pose to PX4 inputs using
     rl_infer's own frame math (enu_to_ned_vec, enu_flu_quat_to_ned_frd),
  2. publishes them on /fmu/out/vehicle_local_position + /fmu/out/vehicle_attitude
     and the gate on /vrpn_mocap/gate/pose,
  3. captures the node's /gate/binary_mask Image,
  4. checks it matches test/golden/golden_2cam_wireframe.npy[case] (boundary-only
     float diffs) — validating the ROS plumbing AND the PX4->policy frame
     conversion together.

Assumes the package is built + sourced. Starts the node itself as a subprocess.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from px4_msgs.msg import VehicleLocalPosition, VehicleAttitude

PKG = Path(__file__).resolve().parent.parent
GOLDEN = PKG / "test" / "golden"
sys.path.insert(0, "/home/nat-nus/realflight_ws/src/rl_infer")
from rl_infer.util.math import enu_to_ned_vec, enu_flu_quat_to_ned_frd  # noqa: E402

CASES = [0, 4, 11]   # level, tilted, yawed+moved-gate
NCAM, MODE = 2, "wireframe"


def sensor_qos():
    return QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                      history=HistoryPolicy.KEEP_LAST, depth=5)


class Tester(Node):
    def __init__(self):
        super().__init__("gbr_itest")
        self.pub_lpos = self.create_publisher(
            VehicleLocalPosition, "/fmu/out/vehicle_local_position", sensor_qos())
        self.pub_att = self.create_publisher(
            VehicleAttitude, "/fmu/out/vehicle_attitude", sensor_qos())
        self.pub_gate = self.create_publisher(
            PoseStamped, "/vrpn_mocap/gate/pose", sensor_qos())
        self.last_img = None
        self.create_subscription(Image, "/gate/binary_mask", self._img_cb,
                                 QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                            history=HistoryPolicy.KEEP_LAST, depth=1))
        self.scenario = None
        self.create_timer(0.02, self._tick)  # 50 Hz

    def set_scenario(self, dpos_enu, dquat_flu_xyzw, gpos_enu, gquat_xyzw):
        self.scenario = (dpos_enu, dquat_flu_xyzw, gpos_enu, gquat_xyzw)
        self.last_img = None

    def _img_cb(self, msg: Image):
        self.last_img = msg

    def _tick(self):
        if self.scenario is None:
            return
        dpos, dquat, gpos, gquat = self.scenario
        ned = enu_to_ned_vec(np.asarray(dpos, dtype=np.float64))
        # enu_flu_quat_to_ned_frd expects [w,x,y,z]; dquat is [x,y,z,w].
        q_wxyz = np.array([dquat[3], dquat[0], dquat[1], dquat[2]])
        q_ned = enu_flu_quat_to_ned_frd(q_wxyz)  # [w,x,y,z]

        lp = VehicleLocalPosition()
        lp.xy_valid = True
        lp.z_valid = True
        lp.x, lp.y, lp.z = float(ned[0]), float(ned[1]), float(ned[2])
        self.pub_lpos.publish(lp)

        at = VehicleAttitude()
        at.q = [float(v) for v in q_ned]
        self.pub_att.publish(at)

        ps = PoseStamped()
        ps.header.frame_id = "world"
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, gpos)
        ps.pose.orientation.x = float(gquat[0])
        ps.pose.orientation.y = float(gquat[1])
        ps.pose.orientation.z = float(gquat[2])
        ps.pose.orientation.w = float(gquat[3])
        self.pub_gate.publish(ps)


def load_cases():
    rows = [list(map(float, l.split()))
            for l in (GOLDEN / "cases.txt").read_text().splitlines() if l.strip()]
    return np.array(rows)


def main() -> int:
    node_proc = subprocess.Popen(
        ["ros2", "run", "gate_bin_renderer_cpp", "gate_bin_renderer_node",
         "--ros-args", "-p", f"ncam:={NCAM}", "-p", f"mode:={MODE}",
         "-p", "pose_source:=px4", "-p", "use_gpu:=true", "-p", "rate:=50.0"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3.0)  # let the node come up + discover

    rclpy.init()
    t = Tester()
    rows = load_cases()
    gold = np.load(GOLDEN / f"golden_{NCAM}cam_{MODE}.npy")
    H, W = gold.shape[1], gold.shape[2]

    all_ok = True
    try:
        for ci in CASES:
            r = rows[ci]
            t.set_scenario(r[0:3], r[3:7], r[7:10], r[10:14])
            # spin ~2 s; keep the latest image
            deadline = time.time() + 4.0
            got = None
            while time.time() < deadline:
                rclpy.spin_once(t, timeout_sec=0.05)
                if t.last_img is not None and time.time() > deadline - 2.5:
                    got = t.last_img
            if got is None:
                print(f"[FAIL] case {ci}: no mask received")
                all_ok = False
                continue
            img = np.frombuffer(bytes(got.data), dtype=np.uint8).reshape(got.height, got.width)
            if (got.height, got.width) != (H, W):
                print(f"[FAIL] case {ci}: image {got.height}x{got.width} != {H}x{W}")
                all_ok = False
                continue
            gb = gold[ci] > 0.5
            cb = img > 127
            nmis = int(np.count_nonzero(gb != cb))
            on = max(1, int(gb.sum()))
            # boundary-adjacency check
            nb = 0
            for (y, x) in np.argwhere(gb != cb):
                ga = gb[max(0, y-1):y+2, max(0, x-1):x+2].max()
                ca = cb[max(0, y-1):y+2, max(0, x-1):x+2].max()
                if not (ga or ca):
                    nb += 1
            ok = nb == 0 and nmis <= max(5, int(0.01 * on))
            print(f"[{'OK  ' if ok else 'FAIL'}] case {ci}: flips={nmis}/{on} non_boundary={nb}")
            all_ok = all_ok and ok
    finally:
        t.destroy_node()
        rclpy.shutdown()
        node_proc.terminate()
        try:
            node_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            node_proc.kill()

    print("\n" + ("NODE INTEGRATION TEST PASSED ✅" if all_ok else "NODE INTEGRATION TEST FAILED ❌"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
