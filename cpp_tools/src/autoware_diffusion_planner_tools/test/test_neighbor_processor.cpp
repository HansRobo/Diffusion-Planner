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

#include "processing/neighbor_processor.hpp"

#include <autoware/diffusion_planner/dimensions.hpp>

#include <autoware_perception_msgs/msg/object_classification.hpp>
#include <autoware_perception_msgs/msg/tracked_object.hpp>

#include <gtest/gtest.h>

#include <vector>

namespace
{
using autoware::diffusion_planner::OUTPUT_T;

constexpr int64_t NEIGHBOR_FUTURE_DIM = 4;  // x, y, cos(yaw), sin(yaw)

// Build a single tracked object (a car) at position (x, y) with a fixed UUID so
// it is treated as the same agent across frames.
autoware_perception_msgs::msg::TrackedObject make_car(double x, double y)
{
  autoware_perception_msgs::msg::TrackedObject obj;
  // Fixed, non-zero UUID -> stable hex id across frames.
  for (size_t i = 0; i < obj.object_id.uuid.size(); ++i) {
    obj.object_id.uuid[i] = static_cast<uint8_t>(i + 1);
  }
  autoware_perception_msgs::msg::ObjectClassification cls;
  cls.label = autoware_perception_msgs::msg::ObjectClassification::CAR;
  cls.probability = 1.0f;
  obj.classification.push_back(cls);

  obj.kinematics.pose_with_covariance.pose.position.x = x;
  obj.kinematics.pose_with_covariance.pose.position.y = y;
  obj.kinematics.pose_with_covariance.pose.orientation.w = 1.0;  // identity yaw
  obj.shape.dimensions.x = 4.0;                                  // length
  obj.shape.dimensions.y = 2.0;                                  // width
  return obj;
}

FrameData make_frame(int64_t stamp_ns, double x, double y)
{
  FrameData frame;
  frame.timestamp = stamp_ns;
  frame.tracked_objects.header.stamp.sec = static_cast<int32_t>(stamp_ns / 1000000000LL);
  frame.tracked_objects.header.stamp.nanosec = static_cast<uint32_t>(stamp_ns % 1000000000LL);
  frame.tracked_objects.objects.push_back(make_car(x, y));
  frame.kinematic_state.header.stamp = frame.tracked_objects.header.stamp;
  return frame;
}
}  // namespace

// The neighbor_future array must contain OUTPUT_T *future* states (the current
// state excluded). When every future frame is available, the last future
// timestep must be populated with the corresponding future position, not left
// as the zero-fill default.
TEST(NeighborProcessorTest, LastFutureTimestepIsPopulated)
{
  // Frame 0 is "current"; frames 1..OUTPUT_T are the future. Give the agent a
  // distinct, non-zero x at every frame so a missing slot is detectable.
  std::vector<FrameData> data_list;
  const int64_t dt_ns = 100000000LL;  // 0.1 s
  for (int64_t i = 0; i <= OUTPUT_T; ++i) {
    const double x = 100.0 + static_cast<double>(i);  // strictly increasing, non-zero
    data_list.push_back(make_frame(i * dt_ns, x, 0.0));
  }

  const int64_t current_idx = 0;
  const Eigen::Matrix4d map2bl = Eigen::Matrix4d::Identity();

  const NeighborResult result = process_neighbor_agents_and_future(data_list, current_idx, map2bl);

  ASSERT_EQ(
    static_cast<int64_t>(result.neighbor_future.size()),
    autoware::diffusion_planner::MAX_NUM_NEIGHBORS * OUTPUT_T * NEIGHBOR_FUTURE_DIM);
  ASSERT_FALSE(result.neighbor_ids.empty()) << "the agent should be kept as a neighbor";

  // Agent 0, last future timestep (t = OUTPUT_T - 1), x component (dim 0).
  const int64_t last_t = OUTPUT_T - 1;
  const int64_t base_idx = 0 * OUTPUT_T * NEIGHBOR_FUTURE_DIM + last_t * NEIGHBOR_FUTURE_DIM;

  // Future frame for t = OUTPUT_T is data_list[OUTPUT_T], whose x = 100 + OUTPUT_T.
  const float expected_x = static_cast<float>(100.0 + static_cast<double>(OUTPUT_T));
  EXPECT_NEAR(result.neighbor_future[base_idx + 0], expected_x, 1e-3f)
    << "last future timestep x was left as zero-fill (future buffer is one slot short)";
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
