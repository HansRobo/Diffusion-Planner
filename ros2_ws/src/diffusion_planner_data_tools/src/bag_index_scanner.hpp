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
 * @brief Longest publication gap tolerated per topic, in seconds.
 *
 * Some bags miss messages, either because a node was down for a while or
 * because the topic was never recorded. A frame is only emitted if every topic
 * below has a message at or before the start of the frame window and no gap
 * longer than its threshold anywhere in that window, so dropouts never reach
 * the frame data. A non-finite or non-positive value disables the check for
 * that topic.
 *
 * The window is derived from the diffusion planner input/output dimensions
 * because all four topics feed both the past inputs and the future labels.
 */
struct TopicDropThresholds {
  double kinematic_state{0.5};
  double tracked_objects{0.5};
  double turn_indicators{0.5};
  double traffic_signals{0.5};
};

/**
 * @brief Configurable parameters that shape the frame grid and decide which
 * frames survive.
 */
struct IndexerParam {
  /// Frame grid interval [s].
  double time_step_s{0.1};
  /// Minimum cumulative ego travel distance required to keep a bag [m].
  double min_travel_distance{0.0};
  /// Longest publication gap tolerated per topic.
  TopicDropThresholds topic_drop_thresholds{};
};

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
  std::vector<std::string> warnings;
  size_t all_frames{0};
  size_t usable_frames{0};
  bool skipped{false};
};

/**
 * @brief Scan a bag once and build its frame index (no map required).
 *
 * @param bag_path Path to the rosbag directory.
 * @param topics Topic name configuration.
 * @param param Frame grid and validity parameters.
 */
BagFrameIndex scan_bag_index(const std::string &bag_path,
                             const TopicConfig &topics,
                             const IndexerParam &param);

} // namespace autoware::diffusion_planner::data_tools

#endif // BAG_INDEX_SCANNER_HPP_
