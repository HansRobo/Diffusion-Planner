# Scenario generation utilities

This workspace package supports the HDP branch's closed-loop validation and NPZ route utilities.
It is intentionally kept independent from legacy RL research code.

Current supported paths:

- `scenario_generation.closed_loop_eval`: route-level closed-loop evaluation used by Diffusion Planner training validation hooks.
- `scenario_generation.simulate`: model loading and step simulation helpers.
- `scenario_generation.reproducer_rollout`: reproduced-neighbor closed-loop rollout and rendering.
- `scenario_generation.tensor_converter`: conversion between scene objects and Diffusion Planner tensors.
- `scenario_generation.tools`: lightweight route/NPZ inspection and selection utilities that do not depend on legacy RL packages.

Legacy exploration/reward-mining tools were removed from this HDP-specialized branch to keep the dependency graph and training path explicit.
