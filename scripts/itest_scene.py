#!/usr/bin/env python3
"""Integration test for the renderer's RViz scene outputs (TF + clean-pass box).

Starts gate_bin_renderer_node with publish_tf + publish_clean_pass_box, feeds it
a synthetic PX4 drone pose + gate pose (golden case 11), and checks:
  * TF map->base_footprint exists and recovers the drone CG pose,
  * TF map->gate exists,
  * the /gate/clean_pass_box marker is a 12-edge LINE_LIST (24 points).
"""
from __future__ import annotations
import subprocess, sys, time
from pathlib import Path
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import tf2_ros
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from px4_msgs.msg import VehicleLocalPosition, VehicleAttitude

PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, "/home/nat-nus/realflight_ws/src/rl_infer")
from rl_infer.util.math import enu_to_ned_vec, enu_flu_quat_to_ned_frd  # noqa: E402

# Golden case 11: drone (0.3,1.0,1.5) yaw 1.0; gate (2.0,1.0,1.5).
DPOS = np.array([0.30, 1.00, 1.50]); GPOS = np.array([2.0, 1.0, 1.5])


def euler_xyzw(r, p, y):
    cr, sr = np.cos(r/2), np.sin(r/2); cp, sp = np.cos(p/2), np.sin(p/2); cy, sy = np.cos(y/2), np.sin(y/2)
    return np.array([sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy])


def be():
    return QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5)


def main():
    proc = subprocess.Popen(
        ["ros2", "run", "gate_bin_renderer_cpp", "gate_bin_renderer_node", "--ros-args",
         "-p", "pose_source:=px4", "-p", "publish_tf:=true", "-p", "publish_clean_pass_box:=true",
         "-p", "publish_gate_marker:=true",
         "-p", "world_frame:=map", "-p", "drone_frame:=base_footprint", "-p", "gate_frame:=gate"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3.0)
    rclpy.init()
    n = Node("gbr_scene_itest")
    pub_lp = n.create_publisher(VehicleLocalPosition, "/fmu/out/vehicle_local_position", be())
    pub_at = n.create_publisher(VehicleAttitude, "/fmu/out/vehicle_attitude", be())
    pub_g = n.create_publisher(PoseStamped, "/vrpn_mocap/gate/pose", be())
    marker = {}
    rel = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1)
    n.create_subscription(Marker, "/gate/clean_pass_box", lambda m: marker.update(m=m), rel)
    n.create_subscription(MarkerArray, "/gate/gate_marker", lambda m: marker.update(gate=m), rel)
    buf = tf2_ros.Buffer(); tf2_ros.TransformListener(buf, n)

    dq = euler_xyzw(0, 0, 1.0); gq = euler_xyzw(0.2, 0, 1.0)
    ned = enu_to_ned_vec(DPOS); qn = enu_flu_quat_to_ned_frd(np.array([dq[3], dq[0], dq[1], dq[2]]))

    def publish():
        lp = VehicleLocalPosition(); lp.xy_valid = True; lp.z_valid = True
        lp.x, lp.y, lp.z = map(float, ned); pub_lp.publish(lp)
        at = VehicleAttitude(); at.q = [float(v) for v in qn]; pub_at.publish(at)
        ps = PoseStamped(); ps.header.frame_id = "map"; ps.header.stamp = n.get_clock().now().to_msg()
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, GPOS)
        ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = map(float, gq)
        pub_g.publish(ps)

    ok = True
    try:
        deadline = time.time() + 6
        while time.time() < deadline:
            publish(); rclpy.spin_once(n, timeout_sec=0.05)
        # check TF
        for child in ("base_footprint", "gate"):
            try:
                tr = buf.lookup_transform("map", child, rclpy.time.Time())
                t = tr.transform.translation
                print(f"[OK  ] TF map->{child}: ({t.x:.3f},{t.y:.3f},{t.z:.3f})")
                if child == "base_footprint":
                    err = np.linalg.norm([t.x-DPOS[0], t.y-DPOS[1], t.z-DPOS[2]])
                    if err > 1e-2:
                        print(f"[FAIL] base_footprint off by {err:.3f} m (expected {DPOS})"); ok = False
                if child == "gate":
                    err = np.linalg.norm([t.x-GPOS[0], t.y-GPOS[1], t.z-GPOS[2]])
                    if err > 1e-2:
                        print(f"[FAIL] gate off by {err:.3f} m"); ok = False
            except Exception as e:
                print(f"[FAIL] no TF map->{child}: {e}"); ok = False
        # check marker
        m = marker.get("m")
        if m is None:
            print("[FAIL] no clean_pass_box marker"); ok = False
        else:
            npts = len(m.points)
            print(f"[OK  ] marker type={m.type} (LINE_LIST=5) frame={m.header.frame_id} points={npts}")
            if m.type != Marker.LINE_LIST or npts != 24:
                print("[FAIL] marker should be LINE_LIST with 24 points (12 edges)"); ok = False
        # check gate body markers
        gm = marker.get("gate")
        if gm is None:
            print("[FAIL] no gate_marker MarkerArray"); ok = False
        else:
            cubes = [mm for mm in gm.markers if mm.type == Marker.CUBE]
            frames = {mm.header.frame_id for mm in gm.markers}
            print(f"[OK  ] gate body: {len(gm.markers)} markers ({len(cubes)} CUBE) frames={frames}")
            if len(cubes) != 4 or frames != {"gate"}:
                print("[FAIL] gate body should be 4 CUBE markers in the gate frame"); ok = False
    finally:
        n.destroy_node(); rclpy.shutdown(); proc.terminate()
        try: proc.wait(timeout=5)
        except subprocess.TimeoutExpired: proc.kill()
    print("\n" + ("SCENE TEST PASSED ✅" if ok else "SCENE TEST FAILED ❌"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
