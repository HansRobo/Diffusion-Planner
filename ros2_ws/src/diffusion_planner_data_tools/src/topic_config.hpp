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

#ifndef TOPIC_CONFIG_HPP_
#define TOPIC_CONFIG_HPP_

#include <string>

namespace autoware::diffusion_planner::data_tools {

struct TopicConfig {
  std::string kinematic_state;
  std::string tracked_objects;
  std::string turn_indicators;
  std::string traffic_signals;
  std::string route;
};

/**
 * @brief Load the topic configuration from the installed package share
 * directory (config/data_tools.param.yaml). Throws on missing file or keys.
 */
TopicConfig load_topic_config();

} // namespace autoware::diffusion_planner::data_tools

#endif // TOPIC_CONFIG_HPP_
