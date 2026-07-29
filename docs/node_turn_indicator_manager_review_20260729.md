# Turn-indicator state machine (node C++) — review and required changes

**Target repo:** `tier4/autoware_universe`, branch `feat/hdp-turn-indicator-debounce`
**Reviewed at commit:** `2d57dfd14c1d` (2026-07-29T07:00:12Z)
**Scope:** the deployment C++ only. Nothing in this document asks for a change to the
Python training code or to the model.

Files in scope (paths relative to the repo root):

| Path | Role |
| --- | --- |
| `planning/autoware_diffusion_planner/include/autoware/diffusion_planner/postprocessing/turn_indicator_manager.hpp` | class + design docstring |
| `planning/autoware_diffusion_planner/src/postprocessing/turn_indicator_manager.cpp` | `evaluate()` — the state machine |
| `planning/autoware_diffusion_planner/test/turn_indicator_manager_test.cpp` | gtest, 9 cases |
| `planning/autoware_diffusion_planner/src/diffusion_planner_node.cpp` | parameter declaration + dynamic reconfigure |
| `planning/autoware_diffusion_planner/src/diffusion_planner_core.cpp` | `sync_turn_indicator_managers()`, the per-batch call site |
| `planning/autoware_diffusion_planner/config/diffusion_planner.param.yaml` | shipped values |
| `planning/autoware_diffusion_planner/schema/diffusion_planner.schema.json` | declared defaults |

Line numbers below are as of `2d57dfd14c1d`. Re-verify against the branch head before
editing; the branch moves.

---

## Design premises — read before changing anything

1. **This is not the Python state machine.** `diffusion_planner/utils/turn_indicator.py`
   (training repo) is softmax → EMA(α=0.5) → argmax → probability thresholds
   (0.60/0.60/0.70) → consecutive-frame debounce → 1.0 s minimum-active lockout. The C++
   manager is **per-cycle argmax plus a time window**, with no softmax, no EMA and no
   probability threshold. Defects found in the Python thresholds **do not transfer** and
   must not be "ported".
2. **Do not add a logit-margin / confidence gate.** The header docstring rejects it
   explicitly: *"No logit-margin gate: it would add a calibration-sensitive knob, while
   consecutive agreement filters the same single-frame glitches."* That is a deliberate
   decision; every fix below stays inside the temporal-consistency design.
3. **DISABLE is the fail-safe state.** The shape-error branch already resets straight to
   DISABLE with no debounce. Several fixes below extend that same principle; keep them
   consistent with it.
4. **Do not invent numeric constants.** Any new threshold must either be derived from an
   existing parameter (e.g. `planning_frequency_hz`) or be a new parameter whose default
   preserves today's behaviour exactly. Constants are to be calibrated offline on dumped
   logits, not guessed here.

---

## P0-1 — No finiteness check: NaN/Inf logits silently produce a normal-looking command

**Where:** `src/postprocessing/turn_indicator_manager.cpp:76`

```cpp
const auto max_it = std::max_element(turn_indicator_logit.begin(), turn_indicator_logit.end());
const uint8_t observed = raw_state_to_command(
  static_cast<std::size_t>(std::distance(turn_indicator_logit.begin(), max_it)));
```

**Defect.** `std::max_element` advances only on `*largest < *i`. Every comparison against
NaN is false, therefore:

- NaN at index 0 → the loop never advances → returns index 0 → silently emits **DISABLE**;
- NaN at index 1 or 2 → that class can never win the argmax → the decision is silently
  skewed toward the remaining classes;
- all three NaN → index 0 → DISABLE;
- `+Inf` wins outright, so a single overflowed class silently becomes a confident command.

The only input validation in the whole C++ path is the **size** check at
`turn_indicator_manager.cpp:58`. There is no `std::isfinite` anywhere on this data — not in
the manager, not at the call site (`diffusion_planner_core.cpp:641-651` checks size only).
The header's claim *"A malformed logit vector immediately resets to DISABLE"* covers shape,
not values. The Python side by contrast raises
`ValueError("Turn-indicator probabilities must be finite")`.

The failure mode is worse than a crash: the garbage argmax enters the debounce as a
legitimate observation, accumulates a normal confirmation window, and publishes a normal
`TurnIndicatorsCommand`. Nothing is logged.

**Why this is not hypothetical.** On 2026-07-29 the same turn-indicator head produced
non-finite logits during training validation on 6/6 attempts (jobs 1538/1539,
`turn_logit_finite=False`, under bf16 at local batch 128). The shipped node config is
`precision: "fp32"` (`config/diffusion_planner.param.yaml:7`), so the default launch is not
currently exposed — but that precision is a one-line config change, fp16 TensorRT
reintroduces exactly the same mantissa-underflow class, and the manager's contract should
not silently depend on engine precision.

**Required change.** Merge finiteness into the existing malformed-input branch, so bad
input takes the fail-safe path that already exists:

```cpp
  const bool malformed_shape =
    turn_indicator_logit.size() != static_cast<std::size_t>(TURN_INDICATOR_OUTPUT_DIM);
  // std::max_element compares with operator<, so a NaN either pins the argmax to index 0
  // or silently excludes its own class: a non-finite logit must be treated as malformed
  // rather than allowed to enter the debounce as a legitimate observation.
  const bool non_finite =
    !malformed_shape &&
    std::any_of(turn_indicator_logit.begin(), turn_indicator_logit.end(), [](const float v) {
      return !std::isfinite(v);
    });
  if (malformed_shape || non_finite) {
    stable_command_ = TurnIndicatorsCommand::DISABLE;
    has_candidate_ = false;
    has_last_stamp_ = false;
    command_msg.command = TurnIndicatorsCommand::DISABLE;
    return command_msg;
  }
```

Add `#include <cmath>`; `<algorithm>` is already included.

**Do not** throw on non-finite values — neither in the manager nor at
`diffusion_planner_core.cpp:646`. Throwing aborts the whole planning cycle, which is a
strictly worse outcome than degrading the auxiliary blinker output. Instead add a throttled
warning at the call site (`RCLCPP_WARN_THROTTLE`, ≥5 s period) so the condition is
diagnosable from a vehicle log; a silent degradation is untraceable.

**Tests to add** (`test/turn_indicator_manager_test.cpp`):

- NaN at each of the three indices, with an active command established first → the command
  releases to DISABLE and does not adopt the NaN-skewed argmax.
- `+Inf`/`-Inf` in each position → same.
- A finite frame after a non-finite one re-arms normally (state was fully reset).

---

## P0-2 — `turn_indicator_hold_duration` declares a code default of `0.0`, which voids all release hysteresis

**Where:** `src/diffusion_planner_node.cpp:205-208`

```cpp
params_.turn_indicator_hold_duration =
  this->declare_parameter<double>("turn_indicator_hold_duration", 0.0);
params_.turn_indicator_on_confirmation_duration =
  this->declare_parameter<double>("turn_indicator_on_confirmation_duration", 0.2);
```

**Defect.** The two defaults are asymmetric: activation falls back to a safe `0.2`, release
falls back to `0.0`. With `hold_duration_ == 0`, the commit test
`(stamp - candidate_since_) >= required_duration` is **true on the first contrary frame**,
so `LEFT → DISABLE` and `LEFT → RIGHT` both become single-frame passthrough. The header's
guarantee — *"an active blinker never drops or flips unless the model insists for that
long, which also guarantees a minimum on-time of hold_duration"* — is void in that
configuration.

Three sources disagree with the code:

- `config/diffusion_planner.param.yaml:33` → `1.0`
- `schema/diffusion_planner.schema.json:152-156` → `"default": 1.0`, and the key is in the
  schema's `required` list
- the header docstring describes a long release window

The schema being `required` does not protect the runtime: `declare_parameter`'s fallback
fires for any launch whose overrides omit the key (integration tests, ad-hoc launches,
a trimmed param file). The schema is a documentation/CI artifact.

**Required change.** Set the code default to `1.0` so all four sources agree.

**Also:** neither duration is validated anywhere. This file already validates
`ego_history_reset_gap_s` (`diffusion_planner_node.cpp:216`) and `planning_frequency_hz`
(`diffusion_planner_core.cpp:149`) for finiteness and sign; these two have no such check,
while the dynamic-parameter callback at `diffusion_planner_node.cpp:344-347` can set them
to a negative or NaN value at runtime. `rclcpp::Duration::from_seconds(-1.0)` is legal and
makes `required_duration` negative, i.e. every contrary frame commits immediately. Add,
following the existing precedent, at the declaration **and** in the `on_set_parameters`
callback:

```cpp
if (!std::isfinite(params_.turn_indicator_hold_duration) ||
    params_.turn_indicator_hold_duration < 0.0 ||
    !std::isfinite(params_.turn_indicator_on_confirmation_duration) ||
    params_.turn_indicator_on_confirmation_duration < 0.0) {
  throw std::invalid_argument(
    "turn indicator debounce durations must be finite and non-negative");
}
```

In the dynamic-reconfigure callback, reject the update (return `successful = false` with a
reason) rather than throwing — match whatever the surrounding cases in that callback do.

Add `"minimum": 0.0` to both properties in `schema/diffusion_planner.schema.json`.

---

## P1-1 — Alternating contrary observations stick the signal on indefinitely (and a test pins the bug as a feature)

**Where:** `src/postprocessing/turn_indicator_manager.cpp:83-99`

```cpp
  } else {
    if (!has_candidate_ || candidate_command_ != observed) {
      has_candidate_ = true;
      candidate_command_ = observed;
      candidate_since_ = stamp;
    }
    const rclcpp::Duration & required_duration =
      (stable_command_ == TurnIndicatorsCommand::DISABLE) ? on_confirmation_duration_
                                                          : hold_duration_;
    if ((stamp - candidate_since_) >= required_duration) {
      stable_command_ = candidate_command_;
      has_candidate_ = false;
    }
  }
```

**Defect.** The window restarts whenever the contrary observation *changes identity*. With
`stable_command_ == ENABLE_LEFT` and an observation stream `R, D, R, D, R, D, …`, neither
RIGHT nor DISABLE ever accumulates `hold_duration`, so the blinker stays **LEFT forever**
while the model has not said LEFT once.

`test/turn_indicator_manager_test.cpp:121-139`
(`MixedContraryEvidenceKeepsRestartingTheWindow`) asserts exactly this: 30 frames over 3 s
with no LEFT observation, and the expectation is `ENABLE_LEFT` throughout. **That test
encodes the defect and must be rewritten as part of this fix.**

**Why the current behaviour is wrong.** Two different kinds of evidence are being measured
with one counter:

- *"the model no longer believes LEFT"* — **release** evidence. DISABLE is the fail-safe
  state; losing confidence in an active direction is sufficient to stop asserting it, and
  requires no agreement on a specific replacement.
- *"the model believes RIGHT"* — **flip** evidence. A direct reversal is a stronger action
  and should keep requiring a consistent window on RIGHT itself.

Today the flip standard gates the turn-off, so "the model is oscillating" is converted into
"lock the current lamp on" — and oscillation is precisely the state in which a definite
turn intent should not keep being asserted.

**Required change.** Track contrary-to-active evidence separately from candidate identity.

New members in the header:

```cpp
  /// Any-non-active contrary evidence, used to release an active command to DISABLE.
  bool has_contrary_{false};
  rclcpp::Time contrary_since_{};
```

Body:

```cpp
  if (observed == stable_command_) {
    // Agreement with the published command discards all pending contrary evidence.
    has_candidate_ = false;
    has_contrary_ = false;
  } else {
    if (!has_contrary_) {
      has_contrary_ = true;
      contrary_since_ = stamp;
    }
    if (!has_candidate_ || candidate_command_ != observed) {
      has_candidate_ = true;
      candidate_command_ = observed;
      candidate_since_ = stamp;
    }
    const bool active = stable_command_ != TurnIndicatorsCommand::DISABLE;
    const rclcpp::Duration & required_duration =
      active ? hold_duration_ : on_confirmation_duration_;
    if ((stamp - candidate_since_) >= required_duration) {
      // A specific consistent observation was confirmed: activate, or flip direction.
      stable_command_ = candidate_command_;
      has_candidate_ = false;
      has_contrary_ = false;
    } else if (active && (stamp - contrary_since_) >= required_duration) {
      // Sustained disagreement with the active command, without agreement on any single
      // replacement: fall back to DISABLE rather than keeping a lamp the model rejects.
      stable_command_ = TurnIndicatorsCommand::DISABLE;
      has_candidate_ = false;
      has_contrary_ = false;
    }
  }
```

Note the ordering: the specific-candidate branch is checked first, so a genuine sustained
flip still reaches `ENABLE_RIGHT` and does not detour through DISABLE.

**Behaviour preserved.** Single-frame glitch rejection is unaffected: one agreeing frame
clears `has_contrary_`. `ActivationRequiresSustainedEvidence`,
`SingleFrameGlitchIsRejected`, `ReleaseRequiresSustainedContraryEvidence` and
`DirectionFlipUsesReleaseWindow` must all still pass unmodified — treat any change in those
four as a regression in this patch, not as an expected update.

**Tests.** Rewrite `MixedContraryEvidenceKeepsRestartingTheWindow` →
`SustainedDisagreementReleasesToDisable`: activate LEFT, then alternate DISABLE/RIGHT at
10 Hz; assert the command stays `ENABLE_LEFT` for the first `hold_duration` and becomes
`DISABLE` once contrary evidence has spanned `hold_duration`. Add a case where an agreeing
LEFT frame lands mid-window and the release clock restarts. Update the header docstring —
the "any interruption restarts the window" sentence becomes inaccurate.

---

## P1-2 — The window is duration-only, so dropped cycles degrade it to a 2-sample filter

**Where:** `src/postprocessing/turn_indicator_manager.cpp:96`

**Defect.** The commit test is a pure timestamp difference. There is no minimum observation
count and no check that the cycles *inside* the window were contrary — an interval with no
frames at all counts fully. Concretely: `stable_command_ == ENABLE_LEFT`; at t=1.3 one
DISABLE frame arrives, `candidate_since_ = 1.3`; the planner then stalls for 1.2 s (GPU
hiccup, rosbag gap, dropped cycles); at t=2.5 the next DISABLE frame gives
`2.5 - 1.3 = 1.2 ≥ 1.0` → the signal releases on **two** observations.

The header promises *"the same contrary observation has **persisted** for a confirmation
window"*. A duration test only measures persistence under an implicit assumption that
cycles never drop. Every existing test uses uniform 0.1 s spacing, so none can catch this.

**Required change — both parts:**

1. **Minimum observation count.** Add `int candidate_count_` (and `contrary_count_` for
   P1-1), incremented per contrary frame and reset with the corresponding flag. Require
   `count >= static_cast<int>(std::ceil(required_duration.seconds() * cycle_hz))` in
   addition to the duration test. `cycle_hz` must come from the existing
   `planning_frequency_hz` parameter — pass it through `set_durations` (rename to
   `set_parameters`, or add a separate setter) from
   `diffusion_planner_core.cpp:84-102`. Do not hardcode 10 Hz.
2. **Staleness reset.** Before evaluating the window, drop stale evidence:

   ```cpp
   if (has_last_stamp_ && (stamp - last_stamp_) > max_evidence_gap_) {
     has_candidate_ = false;
     has_contrary_ = false;
   }
   ```

   with `max_evidence_gap_` derived as ~3 cycle periods from `planning_frequency_hz`. This
   node already has the same idea for a different signal: `ego_history_reset_gap_s = 0.5`
   (`diffusion_planner_node.cpp:215`).

This changes the `set_durations` signature, so update `sync_turn_indicator_managers()`
(both call sites: `diffusion_planner_core.cpp:81` and `:235`) and the test helper
`make_manager` (`turn_indicator_manager_test.cpp:45-49`).

**Tests.** Establish LEFT, deliver one contrary frame, jump the stamp past
`hold_duration`, deliver a second contrary frame → assert the command is still
`ENABLE_LEFT` (the gap must not buy confirmation). Also assert a full-rate contrary run
still releases on schedule.

---

## P2-1 — No maximum on-time failsafe

**Where:** the manager as a whole — there is no upper bound on how long
`stable_command_` may stay non-DISABLE.

**Defect.** If the model latches on one class, the blinker stays on indefinitely. A
permanently lit indicator is actively misread by other road users, so this deserves a
bound independent of what the model says.

**Required change.** Add a `turn_indicator_max_active_duration` parameter, **default
`0.0` meaning disabled**, so today's behaviour is bit-for-bit unchanged unless configured.
When enabled and exceeded, force `stable_command_ = DISABLE` and require a fresh
`on_confirmation_duration` before the signal can light again (i.e. clear the candidate
state too, so the timeout cannot be immediately undone by the next frame).

Do **not** pick a non-zero default in this patch. The value has to be chosen from dumped
logits over the 46,262-scene validation set (longest legitimate continuous active run per
class) plus whatever the applicable regulation requires; guessing it here would silently
truncate legitimate long turns. Declare it in the yaml and schema as `0.0` with a comment
saying it is pending offline calibration.

---

## P2-2 — A backwards timestamp keeps the active command (judgment call)

**Where:** `src/postprocessing/turn_indicator_manager.cpp:68-74`

```cpp
  // A backwards timestamp (simulation reset, bag loop) invalidates any evidence
  // window in progress; keep the stable command and restart confirmation.
  if (has_last_stamp_ && stamp < last_stamp_) {
    has_candidate_ = false;
  }
```

**Observation.** The candidate is cleared but `stable_command_` survives, so a sim reset or
bag loop that happened while the blinker was on carries that command across the
discontinuity. The malformed-input branch treats an equally discontinuous event by resetting
to DISABLE; applying the same fail-safe here would be more self-consistent:

```cpp
  if (has_last_stamp_ && stamp < last_stamp_) {
    stable_command_ = TurnIndicatorsCommand::DISABLE;
    has_candidate_ = false;
    has_contrary_ = false;
  }
```

**This one is a decision, not a defect** — the current comment says the retention is
deliberate, and if bag replay is intended to preserve state, leaving it alone is
defensible. Raise it in the PR description and let the reviewer choose; do not change it
silently. If it is changed, `BackwardsTimeRestartsEvidenceWindow`
(`turn_indicator_manager_test.cpp:149-161`) needs a case starting from an *active* command —
today it only starts from DISABLE, which is exactly why it cannot tell the two behaviours
apart.

---

## Minor

- `raw_state_to_command`'s `default:` branch (`turn_indicator_manager.cpp:33-34`) is
  unreachable once the size check has passed. Defensive, correct as-is, leave it.
- Inconsistent batch-size floors: `diffusion_planner_core.cpp:642` uses
  `std::max(params_.batch_size, 0)` while `sync_turn_indicator_managers()` at `:89` uses
  `std::max<int>(params_.batch_size, 1)`. With `batch_size <= 0` the expected logit size
  becomes 0, an empty vector passes the equality check at `:645`, the loop at `:652` never
  runs, and nothing is published — silently. Harmless under the shipped yaml; unify the two
  floors to `1` and let the size check fail loudly instead.
- No clock-source hazard: `candidate_since_` / `last_stamp_` are only read on paths where a
  prior call assigned them from `stamp`, so `rclcpp::Time`'s
  "can't subtract times with different time sources" throw is not reachable today. Any new
  code that compares against a default-constructed `rclcpp::Time{}` (which is
  `RCL_SYSTEM_TIME`) would introduce it — keep new time members assignment-initialised from
  `stamp`.

---

## Suggested PR split

1. **PR A (P0-1 + P0-2 + minor batch-size floor).** No behaviour-contract change, no
   existing test modified, only additions. Ship first.
2. **PR B (P1-1).** Changes the release contract and rewrites
   `MixedContraryEvidenceKeepsRestartingTheWindow`; needs the docstring updated. Discuss in
   the PR body that the rewritten test previously asserted the old behaviour.
3. **PR C (P1-2).** Changes the `set_durations` signature and threads
   `planning_frequency_hz` into the manager.
4. **PR D (P2-1), and P2-2 as a question in the PR body.**

## Verification

```bash
cd <autoware_workspace>
colcon build --packages-select autoware_diffusion_planner \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
colcon test --packages-select autoware_diffusion_planner
colcon test-result --verbose
```

Also run `pre-commit run --files <changed files>` — this repo enforces clang-format and
cpplint, and the surrounding style is 2-space indent with a 100-column limit.

Regression bar for every PR: the four cases `ActivationRequiresSustainedEvidence`,
`SingleFrameGlitchIsRejected`, `ReleaseRequiresSustainedContraryEvidence`,
`DirectionFlipUsesReleaseWindow` must pass **unmodified**. Any patch that needs to edit one
of them has changed a behaviour it was not supposed to change.
