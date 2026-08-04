// Copyright 2026 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "bag_index_scanner.hpp"

#include "topic_config.hpp"

#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>
#include <rclcpp/time.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_storage/storage_filter.hpp>

#include <autoware_perception_msgs/msg/tracked_objects.hpp>
#include <autoware_perception_msgs/msg/traffic_light_group_array.hpp>
#include <autoware_planning_msgs/msg/lanelet_route.hpp>
#include <autoware_vehicle_msgs/msg/turn_indicators_report.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace autoware::diffusion_planner::data_tools {

namespace {

// Latest sample with stamp <= t; `cursor` carries forward across increasing t.
template <typename SampleT>
const SampleT *
latest_at_or_before(const std::vector<std::pair<double, SampleT>> &samples,
                    const double t, size_t &cursor) {
  while (cursor + 1 < samples.size() && samples[cursor + 1].first <= t) {
    ++cursor;
  }
  if (samples.empty() || samples[cursor].first > t) {
    return nullptr;
  }
  return &samples[cursor].second;
}

} // namespace

BagFrameIndex scan_bag_index(const std::string &bag_path,
                             const TopicConfig &topics,
                             const double time_step_s,
                             const double history_window_s,
                             const double future_horizon_s,
                             const double traffic_light_timeout_s) {
  using autoware_perception_msgs::msg::TrackedObjects;
  using autoware_perception_msgs::msg::TrafficLightGroupArray;
  using autoware_planning_msgs::msg::LaneletRoute;
  using autoware_vehicle_msgs::msg::TurnIndicatorsReport;
  using nav_msgs::msg::Odometry;

  if (!std::isfinite(time_step_s) || time_step_s <= 0.0) {
    throw std::invalid_argument(
        "time_step_s must be finite and greater than zero");
  }
  if (!std::isfinite(history_window_s) || history_window_s < 0.0) {
    throw std::invalid_argument(
        "history_window_s must be finite and non-negative");
  }
  if (!std::isfinite(future_horizon_s) || future_horizon_s < 0.0) {
    throw std::invalid_argument(
        "future_horizon_s must be finite and non-negative");
  }
  if (!std::isfinite(traffic_light_timeout_s) ||
      traffic_light_timeout_s < 0.0) {
    throw std::invalid_argument(
        "traffic_light_timeout_s must be finite and non-negative");
  }

  struct EgoSample {
    float speed_mps;
    float yaw_rate_rps;
  };
  std::vector<std::pair<double, EgoSample>> ego_samples;
  std::vector<std::pair<double, uint8_t>> turn_samples;
  std::vector<std::pair<double, int32_t>> objects_samples;
  std::vector<double> traffic_stamps;
  std::vector<double> route_stamps;

  // Single sequential pass over the whole bag.
  {
    rosbag2_cpp::Reader reader;
    reader.open(bag_path);
    rosbag2_storage::StorageFilter filter;
    filter.topics = {topics.kinematic_state, topics.tracked_objects,
                     topics.turn_indicators, topics.traffic_signals,
                     topics.route};
    reader.set_filter(filter);

    rclcpp::Serialization<Odometry> odom_serializer;
    rclcpp::Serialization<TrackedObjects> objects_serializer;
    rclcpp::Serialization<TurnIndicatorsReport> turn_serializer;
    rclcpp::Serialization<TrafficLightGroupArray> traffic_serializer;
    rclcpp::Serialization<LaneletRoute> route_serializer;

    while (reader.has_next()) {
      const auto bag_msg = reader.read_next();
      rclcpp::SerializedMessage raw(*bag_msg->serialized_data);
      const std::string &topic = bag_msg->topic_name;

      if (topic == topics.kinematic_state) {
        Odometry msg;
        odom_serializer.deserialize_message(&raw, &msg);
        ego_samples.emplace_back(
            rclcpp::Time(msg.header.stamp).seconds(),
            EgoSample{static_cast<float>(msg.twist.twist.linear.x),
                      static_cast<float>(msg.twist.twist.angular.z)});
      } else if (topic == topics.tracked_objects) {
        TrackedObjects msg;
        objects_serializer.deserialize_message(&raw, &msg);
        objects_samples.emplace_back(rclcpp::Time(msg.header.stamp).seconds(),
                                     static_cast<int32_t>(msg.objects.size()));
      } else if (topic == topics.turn_indicators) {
        TurnIndicatorsReport msg;
        turn_serializer.deserialize_message(&raw, &msg);
        turn_samples.emplace_back(rclcpp::Time(msg.stamp).seconds(),
                                  msg.report);
      } else if (topic == topics.traffic_signals) {
        TrafficLightGroupArray msg;
        traffic_serializer.deserialize_message(&raw, &msg);
        traffic_stamps.push_back(rclcpp::Time(msg.stamp).seconds());
      } else if (topic == topics.route) {
        LaneletRoute msg;
        route_serializer.deserialize_message(&raw, &msg);
        if (!msg.segments.empty()) {
          route_stamps.push_back(rclcpp::Time(msg.header.stamp).seconds());
        }
      }
    }
  }

  BagFrameIndex index;
  if (ego_samples.empty()) {
    return index; // no ego odometry: no frames
  }

  // Stamps may not be strictly monotonic across topics; sort defensively.
  std::sort(ego_samples.begin(), ego_samples.end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });
  std::sort(turn_samples.begin(), turn_samples.end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });
  std::sort(objects_samples.begin(), objects_samples.end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });
  std::sort(traffic_stamps.begin(), traffic_stamps.end());
  std::sort(route_stamps.begin(), route_stamps.end());

  const double ego_first_sec = ego_samples.front().first;
  const double ego_last_sec = ego_samples.back().first;
  const double turn_first_sec =
      turn_samples.empty() ? ego_last_sec : turn_samples.front().first;
  const double objects_first_sec =
      objects_samples.empty() ? ego_last_sec : objects_samples.front().first;

  const auto num_frames = static_cast<size_t>(std::floor(
                              (ego_last_sec - ego_first_sec) / time_step_s)) +
                          1;

  index.frame_time_ns.reserve(num_frames);
  size_t ego_cursor = 0;
  size_t turn_cursor = 0;
  size_t objects_cursor = 0;
  size_t traffic_cursor = 0;

  for (size_t i = 0; i < num_frames; ++i) {
    const double t = ego_first_sec + static_cast<double>(i) * time_step_s;

    // Advance the ZOH cursors even for skipped frames so that they stay
    // consistent across the whole grid.
    const EgoSample *ego = latest_at_or_before(ego_samples, t, ego_cursor);
    const uint8_t *turn = latest_at_or_before(turn_samples, t, turn_cursor);
    const int32_t *objects =
        latest_at_or_before(objects_samples, t, objects_cursor);
    while (traffic_cursor + 1 < traffic_stamps.size() &&
           traffic_stamps[traffic_cursor + 1] <= t) {
      ++traffic_cursor;
    }
    const bool traffic_fresh =
        !traffic_stamps.empty() && traffic_stamps[traffic_cursor] <= t &&
        t - traffic_stamps[traffic_cursor] <= traffic_light_timeout_s;

    // Only valid frames are emitted: full past window coverage, full ego
    // future horizon coverage, and an available route.
    const bool past_covered = (t - history_window_s >= ego_first_sec) &&
                              (t - history_window_s >= turn_first_sec) &&
                              (t - history_window_s >= objects_first_sec);
    const bool future_covered = ego_last_sec >= t + future_horizon_s;
    const bool route_available =
        !route_stamps.empty() && route_stamps.front() <= t + 1e-6;
    if (!(past_covered && future_covered && route_available)) {
      continue;
    }

    index.frame_time_ns.push_back(static_cast<int64_t>(std::llround(t * 1e9)));
    index.ego_speed_mps.push_back(ego != nullptr ? ego->speed_mps : 0.0f);
    index.ego_yaw_rate_rps.push_back(ego != nullptr ? ego->yaw_rate_rps : 0.0f);
    index.turn_indicator.push_back(turn != nullptr ? *turn : 0);
    index.num_objects.push_back(objects != nullptr ? *objects : 0);
    index.traffic_signal_fresh.push_back(traffic_fresh);
  }

  return index;
}

} // namespace autoware::diffusion_planner::data_tools
