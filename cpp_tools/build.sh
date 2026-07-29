#!/bin/bash
set -eux

cd $(dirname $0)

if [[ "${1:-}" == "--test" ]]; then
  echo "[INFO] Build and run unit tests with colcon test"
  colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DBUILD_TESTING=1 --packages-up-to autoware_diffusion_planner_tools
  colcon test --packages-select autoware_diffusion_planner_tools
  colcon test-result --all
  exit 0
fi

if [[ "${1:-}" == "--validator" ]]; then
  # Headless closed-loop validator overlay: builds the SSV2_HEADLESS_EGO ego + the pybind facade
  # (openscenario_python). Excludes scenario_test_runner / simple_sensor_simulator (see plan). The
  # -DSSV2_HEADLESS_EGO=ON flag is global so traffic_simulator exports the headless EgoEntity and
  # the define propagates (PUBLIC) to every downstream TU consistently.
  # behavior_tree_plugin: loaded through pluginlib at runtime, so it does NOT appear in any
  # package's build dependency graph and --packages-up-to will not pull it in. Omitting it produces
  # an overlay that builds and imports fine but CANNOT SPAWN ANY ENTITY -- activate() fails with
  # "the class behavior_tree_plugin/VehicleBehaviorTree ... does not exist. Declared types are []".
  # (Verified 2026-07-29 on sakurab with a clean overlay: every scenario failed at activate.)
  # autoware_lanelet2_extension_python: the scenario_sim Python rollout
  # (scenario_generation.scenario_sim_rollout / LaneletSceneBuilder + road-border
  # metrics) imports its MGRSProjector. Not pulled in by the C++ deps, so name it
  # explicitly. It does not include ego_entity.hpp, so the global SSV2_HEADLESS_EGO
  # flag is a harmless no-op for it (no ODR concern).
  # -DCMAKE_CXX_FLAGS=-DSSV2_HEADLESS_EGO is REQUIRED, not belt-and-braces: the CMake *variable*
  # -DSSV2_HEADLESS_EGO=ON only drives if() branches. traffic_simulator exports the preprocessor
  # define with target_compile_definitions(... PUBLIC), but ament_auto's dependency wiring passes
  # <pkg>_DEFINITIONS variables rather than imported targets, so INTERFACE_COMPILE_DEFINITIONS does
  # NOT cross the package boundary. openscenario_interpreter's own CMakeLists has no
  # target_compile_definitions, so without a global flag its headless_interpreter_drive_check target
  # compiles against a header whose StepOutcome/step()/resultKind() are #ifdef'd out and the build
  # fails. A global define is also what the ODR constraint wants: ego_entity.hpp branches its class
  # declaration on this macro, so every TU that includes a traffic_simulator header must agree.
  # (Verified 2026-07-29: a clean container build without this flag fails in openscenario_interpreter.)
  echo "[INFO] Build the headless closed-loop validator overlay (SSV2_HEADLESS_EGO)"
  colcon build \
    --symlink-install \
    --cmake-args \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    -DBUILD_TESTING=0 \
    -DSSV2_HEADLESS_EGO=ON \
    -DCMAKE_CXX_FLAGS=-DSSV2_HEADLESS_EGO \
    --packages-up-to openscenario_python autoware_lanelet2_extension_python behavior_tree_plugin
  exit 0
fi

colcon build \
  --symlink-install \
  --cmake-args \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
  -DBUILD_TESTING=0 \
  --packages-up-to autoware_diffusion_planner_tools
