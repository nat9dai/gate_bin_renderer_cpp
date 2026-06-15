// ROS 2 node: render the gate-traversal binary mask live from the drone pose
// and the mocap gate pose, and publish it as a sensor_msgs/Image (mono8,
// directly viewable in RViz) and an optional std_msgs/Float32MultiArray (the
// flattened [0,1] mask for a visual policy).
//
// Geometry matches the dva-quad-jax gate_traversal_binary_mask{,_2_cam,_4_cam}
// rasterisers; pose plumbing matches rl_infer's gate_jax_node (px4 EKF source).
#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <px4_msgs/msg/vehicle_local_position.hpp>
#include <px4_msgs/msg/vehicle_attitude.hpp>

#include "gate_bin_renderer_cpp/frame_transforms.hpp"
#include "gate_bin_renderer_cpp/renderer.hpp"

using namespace std::chrono_literals;

namespace {
rclcpp::QoS sensor_qos() {
  // best_effort + keep_last: compatible with PX4 (best_effort) publishers and
  // with reliable mocap/dummy publishers alike.
  return rclcpp::QoS(rclcpp::KeepLast(5)).best_effort();
}
}  // namespace

class GateBinRendererNode : public rclcpp::Node {
 public:
  GateBinRendererNode() : rclcpp::Node("gate_bin_renderer_node") {
    ncam_ = declare_parameter<int>("ncam", 2);
    mode_str_ = declare_parameter<std::string>("mode", "wireframe");
    tau_ = static_cast<float>(declare_parameter<double>("tau", 1.5));
    use_gpu_ = declare_parameter<bool>("use_gpu", true);

    mask_topic_ = declare_parameter<std::string>("mask_topic", "/gate/binary_mask");
    publish_float_ = declare_parameter<bool>("publish_float", true);
    float_topic_ = declare_parameter<std::string>("float_topic", "/gate/binary_mask_vec");
    frame_id_ = declare_parameter<std::string>("frame_id", "gate_cam");

    pose_source_ = declare_parameter<std::string>("pose_source", "px4");
    drone_id_ = declare_parameter<int>("drone_id", 0);
    gt_pose_topic_ = declare_parameter<std::string>(
        "gt_pose_topic", "/model/charpi_vision_0/odometry");
    auto off = require3("gt_pos_offset_enu",
                        declare_parameter<std::vector<double>>(
                            "gt_pos_offset_enu", std::vector<double>{0.0, 0.0, 0.0}));
    pos_offset_ = {static_cast<float>(off[0]), static_cast<float>(off[1]),
                   static_cast<float>(off[2])};

    gate_pose_topic_ = declare_parameter<std::string>("gate_pose_topic",
                                                      "/vrpn_mocap/gate/pose");
    auto grpy = require3("gate_frame_offset_rpy",
                         declare_parameter<std::vector<double>>(
                             "gate_frame_offset_rpy", std::vector<double>{0.0, 0.0, 0.0}));
    gate_offset_q_ = gbr::euler_zyx_to_quat_xyzw(grpy[0], grpy[1], grpy[2]);

    rate_ = declare_parameter<double>("rate", 30.0);
    pose_timeout_ = declare_parameter<double>("pose_timeout", 0.5);

    // ── RViz scene viz (TF + clean-pass box), default off ───────────────────
    publish_tf_ = declare_parameter<bool>("publish_tf", false);
    world_frame_ = declare_parameter<std::string>("world_frame", "map");
    // drone_frame MUST equal the charpi URDF root (base_footprint, a zero-offset
    // FLU frame at the CG) for the RViz RobotModel mesh to attach to this pose.
    drone_frame_ = declare_parameter<std::string>("drone_frame", "base_footprint");
    gate_frame_ = declare_parameter<std::string>("gate_frame", "gate");
    publish_box_ = declare_parameter<bool>("publish_clean_pass_box", false);
    box_topic_ = declare_parameter<std::string>("clean_pass_box_topic", "/gate/clean_pass_box");
    publish_gate_marker_ = declare_parameter<bool>("publish_gate_marker", false);
    gate_marker_topic_ = declare_parameter<std::string>("gate_marker_topic", "/gate/gate_marker");
    // Charpi body box used by the clean-pass collision check (327x327x127 mm).
    auto bsz = require3("clean_pass_box_size",
                        declare_parameter<std::vector<double>>(
                            "clean_pass_box_size", std::vector<double>{0.327, 0.327, 0.127}));
    box_half_ = {static_cast<float>(bsz[0] * 0.5), static_cast<float>(bsz[1] * 0.5),
                 static_cast<float>(bsz[2] * 0.5)};

    // ── renderer ──────────────────────────────────────────────────────────
    std::vector<gbr::CameraSpec> cams;
    gbr::Mode mode;
    try {
      cams = gbr::make_cameras(ncam_);
      mode = gbr::parse_mode(mode_str_);
    } catch (const std::exception &e) {
      RCLCPP_FATAL(get_logger(), "config error: %s", e.what());
      throw;
    }
    bool using_gpu = false;
    renderer_ = gbr::make_renderer(cams, mode, tau_, use_gpu_, &using_gpu);
    H_ = renderer_->height();
    W_ = renderer_->total_width();
    mask_.assign(static_cast<size_t>(H_) * W_, 0.0f);
    RCLCPP_INFO(get_logger(),
                "renderer: ncam=%d mode=%s backend=%s dims=%dx%d gpu_requested=%s",
                ncam_, mode_str_.c_str(), renderer_->backend_name(), H_, W_,
                use_gpu_ ? "true" : "false");
    if (use_gpu_ && !using_gpu) {
      RCLCPP_WARN(get_logger(),
                  "use_gpu=true but no CUDA backend available — using CPU.");
    }

    // ── publishers ────────────────────────────────────────────────────────
    pub_img_ = create_publisher<sensor_msgs::msg::Image>(mask_topic_, 1);
    if (publish_float_) {
      pub_vec_ = create_publisher<std_msgs::msg::Float32MultiArray>(float_topic_, 1);
    }
    if (publish_tf_) {
      tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    }
    if (publish_box_) {
      pub_box_ = create_publisher<visualization_msgs::msg::Marker>(box_topic_, 1);
    }
    if (publish_gate_marker_) {
      pub_gate_ = create_publisher<visualization_msgs::msg::MarkerArray>(gate_marker_topic_, 1);
    }

    // ── subscriptions ─────────────────────────────────────────────────────
    sub_gate_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        gate_pose_topic_, sensor_qos(),
        std::bind(&GateBinRendererNode::gate_cb, this, std::placeholders::_1));

    if (pose_source_ == "px4") {
      const std::string ns = drone_id_ == 0 ? "/fmu/" : "/px4_" + std::to_string(drone_id_) + "/fmu/";
      sub_lpos_ = create_subscription<px4_msgs::msg::VehicleLocalPosition>(
          ns + "out/vehicle_local_position", sensor_qos(),
          std::bind(&GateBinRendererNode::lpos_cb, this, std::placeholders::_1));
      sub_att_ = create_subscription<px4_msgs::msg::VehicleAttitude>(
          ns + "out/vehicle_attitude", sensor_qos(),
          std::bind(&GateBinRendererNode::att_cb, this, std::placeholders::_1));
      RCLCPP_INFO(get_logger(), "drone pose: px4 %sout/{vehicle_local_position,vehicle_attitude}",
                  ns.c_str());
    } else if (pose_source_ == "odom") {
      sub_odom_ = create_subscription<nav_msgs::msg::Odometry>(
          gt_pose_topic_, sensor_qos(),
          std::bind(&GateBinRendererNode::odom_cb, this, std::placeholders::_1));
      RCLCPP_INFO(get_logger(), "drone pose: odom %s", gt_pose_topic_.c_str());
    } else {  // "pose" / "vrpn"
      sub_pose_ = create_subscription<geometry_msgs::msg::PoseStamped>(
          gt_pose_topic_, sensor_qos(),
          std::bind(&GateBinRendererNode::pose_cb, this, std::placeholders::_1));
      RCLCPP_INFO(get_logger(), "drone pose: pose %s", gt_pose_topic_.c_str());
    }
    RCLCPP_INFO(get_logger(), "gate pose: %s; publishing mask on %s%s",
                gate_pose_topic_.c_str(), mask_topic_.c_str(),
                publish_float_ ? (" and " + float_topic_).c_str() : "");

    const auto period = std::chrono::duration<double>(1.0 / std::max(1.0, rate_));
    timer_ = create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(period),
        std::bind(&GateBinRendererNode::tick, this));
  }

 private:
  // Fail loudly on a misconfigured 3-vector param rather than reading OOB.
  std::vector<double> require3(const std::string &name, std::vector<double> v) {
    if (v.size() != 3) {
      RCLCPP_FATAL(get_logger(), "param '%s' must have 3 elements, got %zu",
                   name.c_str(), v.size());
      throw std::invalid_argument(name + " must have 3 elements");
    }
    return v;
  }

  // ── pose callbacks ────────────────────────────────────────────────────────
  void lpos_cb(const px4_msgs::msg::VehicleLocalPosition::SharedPtr m) {
    if (!(m->xy_valid && m->z_valid)) return;
    const gbr::Vec3 enu = gbr::ned_to_enu({m->x, m->y, m->z});
    drone_pos_ = {enu.x + pos_offset_[0], enu.y + pos_offset_[1], enu.z + pos_offset_[2]};
    drone_pos_ready_ = true;
    drone_pos_stamp_ = now_sec();
  }
  void att_cb(const px4_msgs::msg::VehicleAttitude::SharedPtr m) {
    // VehicleAttitude.q is [w,x,y,z] (FRD body -> NED world).
    drone_quat_ = gbr::ned_frd_quat_to_enu_flu(m->q[0], m->q[1], m->q[2], m->q[3]);
    drone_att_ready_ = true;
  }
  void pose_cb(const geometry_msgs::msg::PoseStamped::SharedPtr m) {
    const auto &p = m->pose.position;
    const auto &q = m->pose.orientation;
    drone_pos_ = {static_cast<float>(p.x), static_cast<float>(p.y), static_cast<float>(p.z)};
    drone_quat_ = {static_cast<float>(q.x), static_cast<float>(q.y),
                   static_cast<float>(q.z), static_cast<float>(q.w)};
    drone_pos_ready_ = drone_att_ready_ = true;
    drone_pos_stamp_ = now_sec();
  }
  void odom_cb(const nav_msgs::msg::Odometry::SharedPtr m) {
    const auto &p = m->pose.pose.position;
    const auto &q = m->pose.pose.orientation;
    drone_pos_ = {static_cast<float>(p.x), static_cast<float>(p.y), static_cast<float>(p.z)};
    drone_quat_ = {static_cast<float>(q.x), static_cast<float>(q.y),
                   static_cast<float>(q.z), static_cast<float>(q.w)};
    drone_pos_ready_ = drone_att_ready_ = true;
    drone_pos_stamp_ = now_sec();
  }
  void gate_cb(const geometry_msgs::msg::PoseStamped::SharedPtr m) {
    const auto &p = m->pose.position;
    const auto &q = m->pose.orientation;
    gate_pos_ = {static_cast<float>(p.x), static_cast<float>(p.y), static_cast<float>(p.z)};
    const gbr::Quat mocap_q{static_cast<float>(q.x), static_cast<float>(q.y),
                            static_cast<float>(q.z), static_cast<float>(q.w)};
    gate_quat_ = gbr::quat_mul_xyzw(mocap_q, gate_offset_q_);
    gate_ready_ = true;
    gate_stamp_ = now_sec();
  }

  double now_sec() { return this->get_clock()->now().seconds(); }

  // ── render + publish ──────────────────────────────────────────────────────
  void tick() {
    if (!(drone_pos_ready_ && drone_att_ready_ && gate_ready_)) {
      warn_throttled("waiting for drone pose / attitude / gate pose...");
      return;
    }
    const double t = now_sec();
    if ((t - drone_pos_stamp_) > pose_timeout_ || (t - gate_stamp_) > pose_timeout_) {
      warn_throttled("pose stale (>timeout) — not publishing mask.");
      return;
    }

    renderer_->render(drone_pos_, drone_quat_, gate_pos_, gate_quat_, mask_.data());

    auto stamp = this->get_clock()->now();

    sensor_msgs::msg::Image img;
    img.header.stamp = stamp;
    img.header.frame_id = frame_id_;
    img.height = static_cast<uint32_t>(H_);
    img.width = static_cast<uint32_t>(W_);
    img.encoding = "mono8";
    img.is_bigendian = 0;
    img.step = static_cast<uint32_t>(W_);
    img.data.resize(static_cast<size_t>(H_) * W_);
    for (size_t i = 0; i < mask_.size(); ++i) {
      const float v = mask_[i] * 255.0f + 0.5f;
      img.data[i] = static_cast<uint8_t>(v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v));
    }
    pub_img_->publish(std::move(img));

    if (pub_vec_) {
      std_msgs::msg::Float32MultiArray arr;
      arr.layout.data_offset = 0;
      arr.layout.dim.resize(2);
      // std_msgs/MultiArrayLayout convention: dim[n].stride = size of the array
      // spanned by dim[n] and all inner dims, so the OUTER dim's stride is the
      // full element count. Data is row-major; a consumer reshapes to
      // (dim[0].size, dim[1].size) = (H, ncam*W).
      arr.layout.dim[0].label = "height";
      arr.layout.dim[0].size = static_cast<uint32_t>(H_);
      arr.layout.dim[0].stride = static_cast<uint32_t>(H_ * W_);
      arr.layout.dim[1].label = "width";
      arr.layout.dim[1].size = static_cast<uint32_t>(W_);
      arr.layout.dim[1].stride = static_cast<uint32_t>(W_);
      arr.data = mask_;
      pub_vec_->publish(std::move(arr));
    }

    publish_scene(stamp);
  }

  // Broadcast world->drone (CG) and world->gate TF, and the clean-pass body-box
  // edge marker, for the RViz scene. No-ops unless the corresponding params are on.
  void publish_scene(const rclcpp::Time &stamp) {
    if (tf_broadcaster_) {
      tf_broadcaster_->sendTransform(make_tf(stamp, world_frame_, drone_frame_, drone_pos_, drone_quat_));
      tf_broadcaster_->sendTransform(make_tf(stamp, world_frame_, gate_frame_, gate_pos_, gate_quat_));
    }
    if (pub_box_) pub_box_->publish(make_box_marker(stamp));
    if (pub_gate_) pub_gate_->publish(make_gate_markers(stamp));
  }

  // Picture-frame gate body as 4 CUBE bars (outer rect minus inner opening),
  // from the dva-quad-jax dims, in the gate frame (x=normal/depth, y=width, z=up).
  visualization_msgs::msg::MarkerArray make_gate_markers(const rclcpp::Time &stamp) const {
    const float hwi = gbr::GATE_HW, hhi = gbr::GATE_HH;            // inner half-extents
    const float hwo = gbr::GATE_FRAME_HW_OUTER, hho = gbr::GATE_FRAME_HH_OUTER;  // outer
    const float depth = 0.03f;                                    // visible bar depth (m)
    // bar = {center_y, center_z, size_y, size_z}
    const float bars[4][4] = {
        {0.0f, 0.5f * (hhi + hho), 2.0f * hwo, hho - hhi},        // top
        {0.0f, -0.5f * (hhi + hho), 2.0f * hwo, hho - hhi},       // bottom
        {-0.5f * (hwi + hwo), 0.0f, hwo - hwi, 2.0f * hhi},       // left
        {0.5f * (hwi + hwo), 0.0f, hwo - hwi, 2.0f * hhi},        // right
    };
    visualization_msgs::msg::MarkerArray arr;
    for (int i = 0; i < 4; ++i) {
      visualization_msgs::msg::Marker m;
      m.header.stamp = stamp;
      m.header.frame_id = gate_frame_;
      m.ns = "gate";
      m.id = i;
      m.type = visualization_msgs::msg::Marker::CUBE;
      m.action = visualization_msgs::msg::Marker::ADD;
      m.pose.position.x = 0.0;
      m.pose.position.y = bars[i][0];
      m.pose.position.z = bars[i][1];
      m.pose.orientation.w = 1.0;
      m.scale.x = depth;
      m.scale.y = bars[i][2];
      m.scale.z = bars[i][3];
      m.color.r = 1.0f; m.color.g = 0.55f; m.color.b = 0.0f; m.color.a = 0.85f;
      arr.markers.push_back(m);
    }
    return arr;
  }

  geometry_msgs::msg::TransformStamped make_tf(const rclcpp::Time &stamp,
                                               const std::string &parent, const std::string &child,
                                               const gbr::Vec3 &p, const gbr::Quat &q) const {
    geometry_msgs::msg::TransformStamped t;
    t.header.stamp = stamp;
    t.header.frame_id = parent;
    t.child_frame_id = child;
    t.transform.translation.x = p.x;
    t.transform.translation.y = p.y;
    t.transform.translation.z = p.z;
    t.transform.rotation.x = q.x;
    t.transform.rotation.y = q.y;
    t.transform.rotation.z = q.z;
    t.transform.rotation.w = q.w;
    return t;
  }

  // 12-edge wireframe of the body box (LINE_LIST = edges only, so the drone mesh
  // shows through), centred at the CG in the drone (FLU) frame.
  visualization_msgs::msg::Marker make_box_marker(const rclcpp::Time &stamp) const {
    visualization_msgs::msg::Marker m;
    m.header.stamp = stamp;
    m.header.frame_id = drone_frame_;
    m.ns = "clean_pass_box";
    m.id = 0;
    m.type = visualization_msgs::msg::Marker::LINE_LIST;
    m.action = visualization_msgs::msg::Marker::ADD;
    m.pose.orientation.w = 1.0;
    m.scale.x = 0.006;  // line width (m)
    m.color.r = 0.1f; m.color.g = 1.0f; m.color.b = 0.3f; m.color.a = 0.9f;
    const float hx = box_half_[0], hy = box_half_[1], hz = box_half_[2];
    const float cx[8] = {-hx, hx, hx, -hx, -hx, hx, hx, -hx};
    const float cy[8] = {-hy, -hy, hy, hy, -hy, -hy, hy, hy};
    const float cz[8] = {-hz, -hz, -hz, -hz, hz, hz, hz, hz};
    const int edges[12][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0},   // bottom
                              {4, 5}, {5, 6}, {6, 7}, {7, 4},   // top
                              {0, 4}, {1, 5}, {2, 6}, {3, 7}};  // verticals
    for (const auto &e : edges) {
      for (int k = 0; k < 2; ++k) {
        geometry_msgs::msg::Point pt;
        pt.x = cx[e[k]]; pt.y = cy[e[k]]; pt.z = cz[e[k]];
        m.points.push_back(pt);
      }
    }
    return m;
  }

  void warn_throttled(const char *msg) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "%s", msg);
  }

  // params
  int ncam_;
  std::string mode_str_;
  float tau_;
  bool use_gpu_;
  std::string mask_topic_, float_topic_, frame_id_;
  bool publish_float_;
  std::string pose_source_, gt_pose_topic_, gate_pose_topic_;
  int drone_id_;
  std::array<float, 3> pos_offset_{};
  gbr::Quat gate_offset_q_{0, 0, 0, 1};
  double rate_, pose_timeout_;
  // scene viz
  bool publish_tf_ = false, publish_box_ = false, publish_gate_marker_ = false;
  std::string world_frame_, drone_frame_, gate_frame_, box_topic_, gate_marker_topic_;
  std::array<float, 3> box_half_{};

  // renderer
  std::unique_ptr<gbr::IGateRenderer> renderer_;
  int H_ = 0, W_ = 0;
  std::vector<float> mask_;

  // state
  gbr::Vec3 drone_pos_{0, 0, 0};
  gbr::Quat drone_quat_{0, 0, 0, 1};
  gbr::Vec3 gate_pos_{0, 0, 0};
  gbr::Quat gate_quat_{0, 0, 0, 1};
  bool drone_pos_ready_ = false, drone_att_ready_ = false, gate_ready_ = false;
  double drone_pos_stamp_ = 0.0, gate_stamp_ = 0.0;

  // pubs/subs
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_img_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr pub_vec_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_gate_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_pose_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
  rclcpp::Subscription<px4_msgs::msg::VehicleLocalPosition>::SharedPtr sub_lpos_;
  rclcpp::Subscription<px4_msgs::msg::VehicleAttitude>::SharedPtr sub_att_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_box_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_gate_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<GateBinRendererNode>());
  rclcpp::shutdown();
  return 0;
}
