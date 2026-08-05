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

// Python bindings to build diffusion planner model inputs directly from
// rosbags, sharing the exact preprocessing code used at inference time
// (autoware::diffusion_planner::preprocess::create_input_data_map).

#include "bag_index_scanner.hpp"
#include "frame_data_cache.hpp"
#include "topic_config.hpp"

#include "autoware/diffusion_planner/dimensions.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <rcutils/logging.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

namespace ddt = autoware::diffusion_planner::data_tools;
using autoware::diffusion_planner::VehicleSpec;
namespace preprocess = autoware::diffusion_planner::preprocess;

py::dict to_numpy_dict(const preprocess::InputDataMap &input_data_map) {
  py::dict result;
  for (const auto &[key, value] : input_data_map) {
    std::vector<py::ssize_t> shape;
    shape.reserve(value.dimension());
    for (const size_t dimension : value.shape()) {
      shape.push_back(static_cast<py::ssize_t>(dimension));
    }
    py::array_t<float> array(shape);
    std::copy(value.cbegin(), value.cend(), array.mutable_data());
    result[py::str(key)] = std::move(array);
  }
  return result;
}

// Convert the frame index into a dict of numpy arrays (one entry per column).
py::dict to_numpy_dict(const ddt::BagFrameIndex &index) {
  py::dict result;
  const auto n = static_cast<py::ssize_t>(index.frame_time_ns.size());
  const auto add = [&](const char *key, const auto &values, auto element) {
    using ElementT = decltype(element);
    // With the pybind11 version shipped by ROS 2 Humble, the scalar-size
    // constructor creates a zero-stride array. Use an explicit shape container
    // so that every index row owns a distinct, contiguous element.
    const std::vector<py::ssize_t> shape{n};
    py::array_t<ElementT> array(shape);
    for (py::ssize_t i = 0; i < n; ++i) {
      array.mutable_data()[i] =
          static_cast<ElementT>(values[static_cast<size_t>(i)]);
    }
    result[key] = std::move(array);
  };
  add("frame_time_ns", index.frame_time_ns, int64_t{});
  add("ego_speed_mps", index.ego_speed_mps, float{});
  add("ego_yaw_rate_rps", index.ego_yaw_rate_rps, float{});
  add("turn_indicator", index.turn_indicator, uint8_t{});
  add("num_objects", index.num_objects, int32_t{});
  return result;
}

py::object create_frame_data(ddt::FrameDataCache &cache,
                             const std::string &bag_path,
                             const std::string &map_path,
                             const int64_t frame_time_ns,
                             const VehicleSpec &vehicle_spec,
                             const double traffic_light_timeout_s,
                             const int64_t num_future_steps,
                             const double neighbor_observation_timeout_s) {
  preprocess::InputDataResult result =
      tl::unexpected(std::string{"not created"});
  {
    py::gil_scoped_release release;
    result =
        cache.create_frame_data(bag_path, map_path, frame_time_ns, vehicle_spec,
                                traffic_light_timeout_s, num_future_steps,
                                neighbor_observation_timeout_s);
  }
  if (!result) {
    return py::none();
  }
  return to_numpy_dict(result.value());
}

py::tuple scan_bag_index(const std::string &bag_path,
                         const ddt::TopicConfig &topics,
                         const ddt::IndexerParam &param) {
  ddt::BagFrameIndex index;
  {
    py::gil_scoped_release release;
    index = ddt::scan_bag_index(bag_path, topics, param);
  }
  py::dict stats;
  stats["all_frames"] = index.all_frames;
  stats["usable_frames"] = index.usable_frames;
  stats["kept_frames"] = index.frame_time_ns.size();
  stats["skipped"] = index.skipped;
  return py::make_tuple(to_numpy_dict(index), py::cast(index.warnings), stats);
}

} // namespace

PYBIND11_MODULE(_diffusion_planner_data_tools, m) {
  using autoware::diffusion_planner::HISTORY_WINDOW_S;
  using autoware::diffusion_planner::OUTPUT_T;

  m.doc() = "Build diffusion planner model inputs directly from rosbags";

  // Silence the per-file "Opened database ..." INFO logs from rosbag2.
  (void)rcutils_logging_initialize();
  (void)rcutils_logging_set_logger_level("rosbag2_storage",
                                         RCUTILS_LOG_SEVERITY_WARN);

  m.attr("HISTORY_WINDOW_S") = HISTORY_WINDOW_S;

  py::class_<ddt::TopicConfig>(m, "TopicConfig")
      .def(py::init<>())
      .def_readwrite("kinematic_state", &ddt::TopicConfig::kinematic_state)
      .def_readwrite("tracked_objects", &ddt::TopicConfig::tracked_objects)
      .def_readwrite("turn_indicators", &ddt::TopicConfig::turn_indicators)
      .def_readwrite("traffic_signals", &ddt::TopicConfig::traffic_signals)
      .def_readwrite("route", &ddt::TopicConfig::route);

  py::class_<ddt::TopicDropThresholds>(m, "TopicDropThresholds")
      .def(py::init<>())
      .def_readwrite("kinematic_state",
                     &ddt::TopicDropThresholds::kinematic_state)
      .def_readwrite("tracked_objects",
                     &ddt::TopicDropThresholds::tracked_objects)
      .def_readwrite("turn_indicators",
                     &ddt::TopicDropThresholds::turn_indicators)
      .def_readwrite("traffic_signals",
                     &ddt::TopicDropThresholds::traffic_signals);

  py::class_<ddt::IndexerParam>(m, "IndexerParam")
      .def(py::init<>())
      .def_readwrite("time_step_s", &ddt::IndexerParam::time_step_s)
      .def_readwrite("min_travel_distance",
                     &ddt::IndexerParam::min_travel_distance)
      .def_readwrite("topic_drop_thresholds", &ddt::IndexerParam::topic_drop_thresholds);

  py::class_<VehicleSpec>(m, "VehicleSpec")
      .def(py::init<double, double, double>(), py::arg("base_link_to_front"),
           py::arg("vehicle_length"), py::arg("vehicle_width"))
      .def_readonly("base_link_to_front", &VehicleSpec::base_link_to_front)
      .def_readonly("vehicle_length", &VehicleSpec::vehicle_length)
      .def_readonly("vehicle_width", &VehicleSpec::vehicle_width);

  m.def("scan_bag_index", &scan_bag_index, py::arg("bag_path"),
        py::arg("topics") = ddt::load_topic_config(),
        py::arg("param") = ddt::IndexerParam{});

  py::class_<ddt::FrameDataCache>(m, "FrameDataCache")
      .def(py::init<size_t, size_t, ddt::TopicConfig, double>(),
           py::arg("reader_capacity") = 16, py::arg("map_capacity") = 4,
           py::arg("topics") = ddt::load_topic_config(),
           py::arg("line_string_max_step_m") = 5.0)
      .def("create_frame_data", &create_frame_data, py::arg("bag_path"),
           py::arg("map_path"), py::arg("frame_time_ns"),
           py::arg("vehicle_spec"), py::arg("traffic_light_timeout_s") = 0.2,
           py::arg("num_future_steps") = OUTPUT_T,
           py::arg("neighbor_observation_timeout_s") = 0.3,
           "Build the single-batch model inputs and training labels for one "
           "frame as one dict (None "
           "if the frame is not usable)");
}
