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

#include "frame_data_cache.hpp"

#include "label_builder.hpp"

#include <autoware/vehicle_info_utils/vehicle_info.hpp>
#include <autoware_lanelet2_extension/projection/mgrs_projector.hpp>

#include <lanelet2_io/Io.h>

#include <memory>
#include <string>
#include <utility>

namespace autoware::diffusion_planner::data_tools {

VehicleSpec
make_vehicle_spec(const double wheel_base_m, const double wheel_tread_m,
                  const double front_overhang_m, const double rear_overhang_m,
                  const double left_overhang_m, const double right_overhang_m) {
  autoware::vehicle_info_utils::VehicleInfo info{};
  info.wheel_base_m = wheel_base_m;
  info.wheel_tread_m = wheel_tread_m;
  info.front_overhang_m = front_overhang_m;
  info.rear_overhang_m = rear_overhang_m;
  info.left_overhang_m = left_overhang_m;
  info.right_overhang_m = right_overhang_m;
  return VehicleSpec(info);
}

FrameDataCache::FrameDataCache(const size_t reader_capacity,
                               const size_t map_capacity,
                               const TopicConfig &topics,
                               const double line_string_max_step_m)
    : readers_(reader_capacity), map_contexts_(map_capacity), topics_(topics),
      line_string_max_step_m_(line_string_max_step_m) {}

BagFrameReader &FrameDataCache::reader_for(const std::string &bag_path) {
  return *readers_.get_or_create(bag_path, [&]() {
    return std::make_shared<BagFrameReader>(bag_path, topics_);
  });
}

preprocess::InputDataResult FrameDataCache::create_frame_data(
    const std::string &bag_path, const std::string &map_path,
    const int64_t frame_time_ns, const VehicleSpec &vehicle_spec,
    const double traffic_light_timeout_s, const int64_t num_future_steps,
    const double neighbor_observation_timeout_s) {
  auto &map_context = map_contexts_.get_or_create(map_path, [&]() {
    lanelet::projection::MGRSProjector projector;
    lanelet::ErrorMessages errors;
    const lanelet::LaneletMapPtr map =
        lanelet::load(map_path, projector, &errors);
    return std::shared_ptr<const preprocess::LaneSegmentContext>(
        preprocess::build_map_context(map, line_string_max_step_m_));
  });

  BagFrameReader &reader = reader_for(bag_path);
  const rclcpp::Time frame_time(frame_time_ns, RCL_ROS_TIME);

  preprocess::InputBuilderParams input_params;
  input_params.traffic_light_group_msg_timeout_seconds =
      traffic_light_timeout_s;
  preprocess::InputDataResult result = reader.create_input_data(
      frame_time, *map_context, vehicle_spec, input_params);
  if (!result) {
    return result;
  }

  LabelBuilderParams label_params;
  label_params.num_future_steps = num_future_steps;
  label_params.neighbor_observation_timeout_s = neighbor_observation_timeout_s;
  label_params.traffic_light_timeout_s = traffic_light_timeout_s;
  try {
    preprocess::InputDataMap label_data_map =
        reader.create_label_data(frame_time, *map_context, label_params);
    for (auto &[key, value] : label_data_map) {
      result.value()[key] = std::move(value);
    }
  } catch (const std::exception &e) {
    return tl::unexpected(std::string{e.what()});
  }

  return result;
}

} // namespace autoware::diffusion_planner::data_tools
