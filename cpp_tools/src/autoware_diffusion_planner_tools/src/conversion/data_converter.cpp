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

#include "conversion/data_converter.hpp"

#include "io/bag_metadata.hpp"
#include "io/frame_writer.hpp"
#include "io/projector_factory.hpp"
#include "processing/frame_processor.hpp"
#include "processing/sequence_builder.hpp"
#include "rosbag/parsed_bag_data.hpp"
#include "types/override_segment.hpp"

#include <autoware/diffusion_planner/preprocessing/lane_segments.hpp>

#include <lanelet2_core/LaneletMap.h>
#include <lanelet2_io/Io.h>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace
{

std::vector<OverrideSegment> build_override_segments(
  const std::vector<ControlModeSample> & control_modes)
{
  std::vector<OverrideSegment> segments;
  if (control_modes.empty()) {
    return segments;
  }

  int32_t current_mode = control_modes.front().mode;
  int64_t start_timestamp = control_modes.front().rosbag_time;
  int64_t previous_timestamp = control_modes.front().rosbag_time;
  for (size_t index = 1; index < control_modes.size(); ++index) {
    const auto & sample = control_modes[index];
    if (sample.mode != current_mode) {
      if (current_mode == 4) {
        // Use the first non-4 sample as the exclusive end boundary. This keeps
        // a single-sample override visible to half-open interval readers.
        segments.push_back({start_timestamp, sample.rosbag_time});
      }
      current_mode = sample.mode;
      start_timestamp = sample.rosbag_time;
    }
    previous_timestamp = sample.rosbag_time;
  }
  if (current_mode == 4) {
    segments.push_back({start_timestamp, previous_timestamp});
  }
  return segments;
}

void save_override_segments_json(
  const std::string & output_dir, const std::vector<OverrideSegment> & segments,
  const std::size_t control_mode_sample_count)
{
  nlohmann::json payload;
  payload["control_mode_sample_count"] = control_mode_sample_count;
  payload["override_segments"] = nlohmann::json::array();
  for (const auto & segment : segments) {
    payload["override_segments"].push_back(
      {{"start_timestamp_ns", segment.start_timestamp_ns},
       {"end_timestamp_ns", segment.end_timestamp_ns}});
  }
  std::filesystem::create_directories(output_dir);
  const std::filesystem::path output_path =
    std::filesystem::path(output_dir) / "control_mode_4_intervals.json";
  std::ofstream output_file(output_path);
  if (!output_file.is_open()) {
    std::cerr << "Failed to open override segment output: " << output_path << std::endl;
    return;
  }
  output_file << std::setw(2) << payload << std::endl;
}

}  // namespace

int run_data_converter(const ConverterPaths & paths, const ConverterOptions & converter)
{
  lanelet::ErrorMessages errors{};
  const std::unique_ptr<lanelet::Projector> projector =
    create_projector_from_yaml(paths.vector_map_path);
  const std::shared_ptr<lanelet::LaneletMap> lanelet_map_ptr =
    lanelet::load(paths.vector_map_path, *projector, &errors);

  std::cout << "Loaded lanelet2 map with " << lanelet_map_ptr->laneletLayer.size() << " lanelets"
            << std::endl;

  const autoware::diffusion_planner::preprocess::LaneSegmentContext lane_segment_context(
    lanelet_map_ptr);
  const std::string rosbag_dir_name = paths.get_rosbag_dir_name();
  const BagMetadata bag_metadata = load_bag_metadata(paths.rosbag_path);

  ParsedBagData bag_data = load_rosbag(
    paths.rosbag_path, converter.limit, converter.extract_override_segments);
  const std::vector<OverrideSegment> override_segments =
    converter.extract_override_segments ? build_override_segments(bag_data.control_modes)
                                        : std::vector<OverrideSegment>{};
  if (converter.extract_override_segments) {
    save_override_segments_json(
      paths.save_dir, override_segments, bag_data.control_modes.size());
  }

  const auto missing_topics_skip = check_missing_topics(bag_data);
  if (missing_topics_skip) {
    std::cout << "Skipping rosbag due to missing required topics:" << std::endl;
    for (const auto & t : missing_topics_skip->missing_topic_types) {
      std::cout << "  - " << to_topic_name(t) << std::endl;
    }
    std::cout << "No training samples will be generated from this rosbag." << std::endl;
    save_route_json(
      paths.save_dir, rosbag_dir_name, "missing_topics", 0, 0.0, 0, 0, missing_topics_skip.value(),
      bag_data.timestamp_stats_map, false, bag_metadata);
    return 0;
  }

  std::vector<SequenceData> sequences = build_sequences(bag_data, converter.search_nearest_route);

  std::cout << "Total " << sequences.size() << " sequences" << std::endl;

  for (int64_t seq_id = 0; seq_id < static_cast<int64_t>(sequences.size()); ++seq_id) {
    process_sequence(
      sequences[seq_id], seq_id, paths, converter, lane_segment_context,
      bag_data.timestamp_stats_map, bag_metadata);
  }

  std::cout << "Data conversion completed!" << std::endl;
  return 0;
}
