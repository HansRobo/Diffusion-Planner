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

#include "skip_index.hpp"
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

template <typename SampleT>
std::vector<double>
stamps_of(const std::vector<std::pair<double, SampleT>> &samples) {
  std::vector<double> stamps;
  stamps.reserve(samples.size());
  for (const auto &[stamp, _] : samples) {
    stamps.push_back(stamp);
  }
  return stamps;
}

} // namespace

BagFrameIndex scan_bag_index(const std::string &bag_path,
                             const TopicConfig &topics,
                             const IndexerParam &param) {
  using autoware_perception_msgs::msg::TrackedObjects;
  using autoware_perception_msgs::msg::TrafficLightGroupArray;
  using autoware_planning_msgs::msg::LaneletRoute;
  using autoware_vehicle_msgs::msg::TurnIndicatorsReport;
  using nav_msgs::msg::Odometry;

  if (!std::isfinite(param.time_step_s) || param.time_step_s <= 0.0) {
    throw std::invalid_argument(
        "time_step_s must be finite and greater than zero");
  }
  if (!std::isfinite(param.min_travel_distance) ||
      param.min_travel_distance < 0.0) {
    throw std::invalid_argument(
        "min_travel_distance must be finite and non-negative");
  }
  struct EgoSample {
    float speed_mps;
    float yaw_rate_rps;
    double x;
    double y;
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
                      static_cast<float>(msg.twist.twist.angular.z),
                      msg.pose.pose.position.x, msg.pose.pose.position.y});
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

  std::vector<std::pair<double, double>> ego_positions;
  ego_positions.reserve(ego_samples.size());
  for (const auto &ego_sample : ego_samples) {
    ego_positions.emplace_back(ego_sample.second.x, ego_sample.second.y);
  }
  if (const auto warning = check_min_travel_distance(
          ego_positions, param.min_travel_distance)) {
    index.warnings.push_back(*warning);
    index.skipped = true;
    return index;
  }

  const double ego_first_sec = ego_samples.front().first;
  const double ego_last_sec = ego_samples.back().first;
  const auto num_frames = static_cast<size_t>(std::floor(
                              (ego_last_sec - ego_first_sec) / param.time_step_s)) +
                          1;

  // Gap checking works on stamps alone, so extract them once.
  const std::vector<double> ego_stamps = stamps_of(ego_samples);
  const std::vector<double> turn_stamps = stamps_of(turn_samples);
  const std::vector<double> objects_stamps = stamps_of(objects_samples);

  index.frame_time_ns.reserve(num_frames);
  size_t ego_cursor = 0;
  size_t turn_cursor = 0;
  size_t objects_cursor = 0;
  size_t invalid_range_cursor = 0;
  const FrameRange frame_range =
      calculate_frame_range(topics, param, ego_stamps, turn_stamps,
                            objects_stamps, traffic_stamps, route_stamps,
                            num_frames);
  index.all_frames = num_frames;
  index.usable_frames = frame_range.usable_frames;
  index.warnings = frame_range.warnings;

  for (size_t i = 0; i < num_frames; ++i) {
    const double t = ego_first_sec + static_cast<double>(i) * param.time_step_s;

    // Advance the ZOH cursors even for skipped frames so that they stay
    // consistent across the whole grid.
    const EgoSample *ego = latest_at_or_before(ego_samples, t, ego_cursor);
    const uint8_t *turn = latest_at_or_before(turn_samples, t, turn_cursor);
    const int32_t *objects =
        latest_at_or_before(objects_samples, t, objects_cursor);

    if (t > frame_range.last_valid_t) {
      break;
    }
    if (t < frame_range.first_valid_t ||
        is_frame_invalid(frame_range.invalid_ranges, t,
                         invalid_range_cursor)) {
      continue;
    }

    index.frame_time_ns.push_back(static_cast<int64_t>(std::llround(t * 1e9)));
    index.ego_speed_mps.push_back(ego != nullptr ? ego->speed_mps : 0.0f);
    index.ego_yaw_rate_rps.push_back(ego != nullptr ? ego->yaw_rate_rps : 0.0f);
    index.turn_indicator.push_back(turn != nullptr ? *turn : 0);
    index.num_objects.push_back(objects != nullptr ? *objects : 0);
  }

  return index;
}

} // namespace autoware::diffusion_planner::data_tools
