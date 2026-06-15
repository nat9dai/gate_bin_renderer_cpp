"""Launch the gate binary-mask renderer.

Example (matches the SITL test in the README):
    ros2 launch gate_bin_renderer_cpp gate_bin_renderer.launch.py \
        ncam:=2 mode:=wireframe pose_source:=px4

The node loads config/gate_bin_renderer.yaml, then applies the launch-arg
overrides below so the common knobs are settable from the command line.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("gate_bin_renderer_cpp")
    params = os.path.join(pkg_share, "config", "gate_bin_renderer.yaml")

    args = [
        DeclareLaunchArgument("ncam", default_value="2", description="1 | 2 | 4"),
        DeclareLaunchArgument("mode", default_value="wireframe",
                              description="wireframe | filled | soft"),
        DeclareLaunchArgument("use_gpu", default_value="true"),
        DeclareLaunchArgument("pose_source", default_value="px4",
                              description="px4 | pose | vrpn | odom"),
        DeclareLaunchArgument("drone_id", default_value="0"),
        DeclareLaunchArgument("gate_pose_topic", default_value="/vrpn_mocap/gate/pose"),
        DeclareLaunchArgument("mask_topic", default_value="/gate/binary_mask"),
    ]

    node = Node(
        package="gate_bin_renderer_cpp",
        executable="gate_bin_renderer_node",
        name="gate_bin_renderer_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            params,
            {
                "ncam": LaunchConfiguration("ncam"),
                "mode": LaunchConfiguration("mode"),
                "use_gpu": LaunchConfiguration("use_gpu"),
                "pose_source": LaunchConfiguration("pose_source"),
                "drone_id": LaunchConfiguration("drone_id"),
                "gate_pose_topic": LaunchConfiguration("gate_pose_topic"),
                "mask_topic": LaunchConfiguration("mask_topic"),
            },
        ],
    )
    return LaunchDescription(args + [node])
