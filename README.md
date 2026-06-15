# gate_bin_renderer_cpp

GPU/CPU **gate-traversal binary-mask renderer** (C++ / ROS 2). It reconstructs,
live on the drone, the exact binary camera mask the visual gate-traversal policy
was trained on — by projecting the physical gate through a pinhole camera model
co-located with the drone — and publishes it as a `sensor_msgs/Image` (viewable
in RViz) and an optional `std_msgs/Float32MultiArray` (for the policy).

The geometry is a faithful port of the `dva-quad-jax` training rasterisers:

| ncam | resolution (HxW) | intrinsics                | reference (`envs/tasks/…`) |
|------|------------------|---------------------------|----------------------------|
| 1    | 84 x 84          | fx=fy=40, cx=cy=42, 45° tilt | `gate_traversal_binary_mask.py` |
| 2    | 84 x 300         | fx=fy=10.1, cx=75, cy=42   | `gate_traversal_binary_mask_2_cam.py` |
| 4    | 84 x 600         | fx=fy=10.1, cx=75, cy=42   | `gate_traversal_binary_mask_4_cam.py` |

Multi-cam masks are concatenated horizontally in the reference order
(2-cam = `[FL | FR]`, 4-cam = `[FR | BR | BL | FL]`).

Modes (`mode` param): **`wireframe`** (4-edge opening outline — default),
`filled` (ray-cast gate-frame band), `soft` (`exp(-d²/2τ²)` distance field).

## Inputs / outputs

Subscribes:
* **drone pose** — `pose_source: px4` (default): `/fmu/out/vehicle_local_position`
  (NED) + `/fmu/out/vehicle_attitude` (FRD→NED), converted to the policy ENU/FLU
  frame exactly as `rl_infer`'s `gate_jax_node`. Also supports `pose`/`vrpn`
  (`geometry_msgs/PoseStamped`) and `odom` (`nav_msgs/Odometry`).
* **gate pose** — `/vrpn_mocap/gate/pose` (`geometry_msgs/PoseStamped`, ENU);
  `gate_frame_offset_rpy` rotates the mocap rigid-body frame to the policy gate
  frame (opening normal +x, up +z).

Publishes:
* `/gate/binary_mask` — `sensor_msgs/Image`, `mono8` (0/255). **RViz-viewable.**
* `/gate/binary_mask_vec` — `std_msgs/Float32MultiArray`, flattened `[H, ncam*W]`
  in `[0,1]` (set `publish_float:=false` to disable). Row-major, matching the
  training obs layout — a visual policy reshapes to `(H, ncam*W)` and feeds it.

Optional **RViz scene** (`publish_tf:=true`, `publish_clean_pass_box:=true`):
* `/tf` — `world_frame -> drone_frame` (drone CG) and `world_frame -> gate_frame`.
  `drone_frame` defaults to `base_footprint` (the charpi URDF root, a zero-offset
  FLU frame at the CG) so a `robot_state_publisher` mesh attaches to this pose.
* `/gate/clean_pass_box` — `visualization_msgs/Marker` LINE_LIST: the 12 edges of
  the Charpi body box (`clean_pass_box_size`, default 0.327x0.327x0.127 m) used by
  the clean-pass collision check, centred at the CG (edges-only so the mesh shows).
  Wired into `offboard_state_machine`'s `interactive_test.launch.py` via `rviz:=true`.

## Build

CUDA is auto-detected (standard `/usr/local/cuda` installs work without putting
`nvcc` on `PATH`); otherwise it builds CPU-only (OpenMP). Both backends run the
*identical* per-pixel code, so they agree to float rounding.

```bash
colcon build --packages-select gate_bin_renderer_cpp
# force CPU-only:        --cmake-args -DGBR_DISABLE_CUDA=ON
# pin the GPU arch:      --cmake-args -DCMAKE_CUDA_ARCHITECTURES=87   # Orin
```

Runs on x86 (Humble) and Jetson Orin (sm_87). `use_gpu:=true` uses CUDA when
present and silently falls back to CPU otherwise.

## Run

```bash
ros2 launch gate_bin_renderer_cpp gate_bin_renderer.launch.py \
    ncam:=2 mode:=wireframe pose_source:=px4
# visualise:
rviz2 -d $(ros2 pkg prefix gate_bin_renderer_cpp)/share/gate_bin_renderer_cpp/rviz/gate_mask.rviz
# or quickly:  ros2 run rqt_image_view rqt_image_view /gate/binary_mask
```

### SITL smoke test
```bash
# 1) PX4 + Gazebo (gate world, drone spawned)
GATE_ROLL_DEG=0 PX4_GZ_MODEL_POSE="0,-1.5,0,0,0,1.5708" PX4_GZ_WORLD=charpi_gate \
    make px4_sitl gz_charpi_vision        # (run in PX4-Autopilot)
# 2) uXRCE-DDS bridge
MicroXRCEAgent udp4 -p 8888
# 3) gate pose (dummy mocap)
ros2 launch rl_infer gate_pose_dummy.launch.py angle:=0
# 4) the renderer
ros2 launch gate_bin_renderer_cpp gate_bin_renderer.launch.py ncam:=2 mode:=wireframe
# (optional) the state-based controller:  ros2 launch offboard_state_machine \
#     interactive_test.launch.py task:=gate enable_rl:=true   -> type ttt0
```

## Verify against the JAX reference

```bash
python3 scripts/gen_golden.py                 # JAX -> test/golden/*.npy (oracle)
python3 scripts/verify_parity.py --backend cpu    # then --backend cuda
```
The C++ output matches the JAX masks except at threshold-boundary pixels
(irreducible float32 rounding); the check asserts every diff is boundary-adjacent.

## Layout
```
include/gate_bin_renderer_cpp/
  math3.hpp            Vec3/Quat, host+device (__host__ __device__)
  gate_render.hpp      gate geometry + per-camera solve + per-pixel evaluators
  camera_presets.hpp   1/2/4-cam configs (.cpp has the OAK-FFC extrinsics)
  frame_transforms.hpp PX4 NED/FRD <-> policy ENU/FLU (mirrors rl_infer util/math)
  renderer.hpp         backend-agnostic interface + factory
src/
  render_cpu.cpp       CPU/OpenMP backend + make_renderer() factory
  render_cuda.cu       CUDA backend (built only when CUDA is available)
  gate_bin_renderer_node.cpp   the ROS 2 node
  gate_render_cli.cpp  offline renderer (parity test / debugging)
```
