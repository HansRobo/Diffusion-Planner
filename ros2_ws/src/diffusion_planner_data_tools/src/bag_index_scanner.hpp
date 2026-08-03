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

#ifndef BAG_INDEX_SCANNER_HPP_
#define BAG_INDEX_SCANNER_HPP_

#include <cstdint>
#include <string>
#include <vector>

namespace autoware::diffusion_planner::data_tools {

struct TopicConfig;

/**
 * @brief Per-frame index of one bag, sampled on a fixed time grid anchored at
 * the first ego odometry stamp.
 *
 * All vectors have the same length. Only valid frames are included: full past
 * window coverage on ego/turn/objects, full ego future horizon coverage, and
 * an available route. The curation stats are zero-order-hold samples at each
 * frame time.
 */
struct BagFrameIndex {
  std::vector<int64_t> frame_time_ns;
  std::vector<float> ego_speed_mps;
  std::vector<float> ego_yaw_rate_rps;
  std::vector<uint8_t> turn_indicator;
  std::vector<int32_t> num_objects;
  std::vector<bool> traffic_signal_fresh;
};

/**
 * @brief Scan a bag once and build its frame index (no map required).
 *
 * @param bag_path Path to the rosbag directory.
 * @param topics Topic name configuration.
 * @param time_step_s Frame grid interval.
 * @param history_window_s Required past coverage of ego/turn/objects.
 * @param future_horizon_s Required ego future coverage.
 * @param traffic_light_timeout_s Freshness threshold for traffic_signal_fresh.
 */
BagFrameIndex scan_bag_index(const std::string &bag_path,
                             const TopicConfig &topics, double time_step_s,
                             double history_window_s, double future_horizon_s,
                             double traffic_light_timeout_s);

} // namespace autoware::diffusion_planner::data_tools

#endif // BAG_INDEX_SCANNER_HPP_
