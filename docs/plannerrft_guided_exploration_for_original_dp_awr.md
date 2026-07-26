# PlannerRFT-style guided exploration for Original-DP AWR

Updated: 2026-07-20

## Decision

PlannerRFT's policy-guided denoising is relevant to our AWR pipeline and has
now passed the real-model candidate-discovery audit. It remains a
**training-time exploration mechanism**: guidance is never present in the
zero-noise deployment planner and must not be described as a safety controller.

The candidate-generation contract that has passed the real-model audit is:

1. generate one zero-noise deployment-behavior trajectory and nine native
   stochastic trajectories;
2. score the native group first;
3. only when native best reward is below 0.90, generate ten bounded
   fixed-symmetric-Beta guided candidates around a globally frozen v5
   reference;
4. keep the zero-noise behavior trajectory unconditionally, then fill the
   remaining nine cache slots by reward from the native/guided union;
5. persist the selected trajectories and their audited reward breakdowns;
   deployment and every checkpoint comparison remain zero-noise and unguided.

This is the correct way to reuse PlannerRFT as data augmentation for the
current Original-DP AWR system.  It is **distribution augmentation before
reward**, not unconditional target augmentation after reward.  A commanded
left/right or slow/fast trajectory is allowed into the candidate pool, but it
becomes a regression target only when the unchanged vehicle-level reward says
that it is better than the deployable candidate by the configured margin.

There are three distinct transfer levels and they must not be conflated:

| Transfer level | Current status | Admission rule |
|---|---|---|
| bounded fixed-Beta guided candidates | active in full Cycle 1 | candidate 0 is always retained; reward ranks the native/guided union |
| paper-style stochastic frozen reference | paired shadow ablation prepared | may enter a later refresh only if reward-relevant recovery improves without more selected hard events |
| learned scene-conditioned Beta policy | research follow-up, not active | first mix with fixed-Beta fallback in shadow mode; requires closed-loop/context-adaptive evidence |

The third level is the main difference between the published full PlannerRFT
and the current formal run.  The paper's guidance head predicts two Beta
distributions from scene and frozen-reference features, while a value head and
closed-loop PPO teach it which directions are useful over time.  No official
implementation is linked from the arXiv record or project page as of
2026-07-20, so the paper equations and supplementary algorithm are the primary
source.  Rebuilding that branch would be a new optimizer and simulator
experiment, not a zero-risk configuration switch inside AWR.

For an AWR-only system, the safest way to absorb part of PlannerRFT's
adaptivity is a contextual-bandit proposal head rather than pretending that
open-loop groups contain PPO returns.  On a future refresh, persist every
generated guided command and reward before top-K censoring, form the per-scene
advantage `A_k = R_k - R_deploy`, and fit a scene-conditioned Beta proposal
with reward-weighted maximum likelihood:

```text
p_k proportional to 1[A_k > margin] * exp(A_k / temperature)
L_proposal = -sum_k p_k log q_phi(eta_k | scene, frozen_reference)
             + lambda_KL * KL(q_phi || q_fixed)
```

The next refresh would split the same conditional guided budget between fixed
stratified-Beta and learned-Beta samples.  It would still generate the native
group, reserve deployment candidate 0 and the native best, and let the
unchanged reward fill the remaining slots.  A bad proposal head can therefore
waste some exploration slots but cannot delete the known native alternatives.
This is not active in Cycle 1: the current cache does not persist
per-candidate guidance commands, and learning from only the retained top-K
members would introduce selection bias.

This AWR-only head matches the information available now: one contextual scene
and one open-loop score per candidate.  PlannerRFT's PPO/value branch becomes
justified only after genuine multi-step closed-loop state transitions and
future returns are available.  Until then, adding a critic would add model risk
without adding temporal information.

The repository already contains an `ExplorationPolicy` architecture, but its
existing trainers are **not** a drop-in implementation of this follow-up.  The
deprecated open-loop `GRPOExplorationTrainer` still builds the legacy literal
longitudinal guidance and uses a different reference/sampling contract.  The
supervised `train_explorer_regression` path learns one grid-search argmax as a
point target and is calibrated to the older v1 guidance envelope.  Neither
path consumes the uncensored `(eta, reward)` bank from the formal frozen-v5,
quintic-ramp Original-DP sampler, and neither can represent two equally useful
guidance modes for one scene.  The reusable pieces are therefore the scene
encoder/reference mixer/cross-attention/Beta-head architecture—not the old
training entry points or their guidance envelope.  A formal proposal head must
first persist all pre-top-K commands and rewards and train against their
reward-weighted distribution under the exact current sampler contract.

The optimizer applied to those candidates is a separate question. Plain
group-relative MSE replay is now rejected by experiment: it improves neither
the random selector nor later replay epochs. The active ablation retains only
candidates whose reward exceeds the deployment anchor by more than 0.01, uses
beta=2 AWR weights, and adds independently hard-gated logged-expert targets as
an anti-forgetting prior. This is still one denoising-regression objective, not
a second SFT branch.

This is deliberately not a claim of zero risk. Reserving one slot changes the
group distribution and the behavior member participates in group reward
normalization. Its advantages are narrower: the actual deployed trajectory is
always represented in the training group, cannot be evicted by top-K selection,
and provides a per-scene baseline against which exploratory modes are learned.

This ordering is supported by PlannerRFT's own ablation: fixed Beta guidance
already improves the reactive score by `+2.47`, while scene-adaptive guidance
improves it by `+4.03`. Uniform guidance has the highest diversity but degrades
the score by `-2.36`. More geometric spread is therefore not the objective;
**reward-relevant multimodality** is.

This proposal borrows PlannerRFT's exploration mechanism; it is **not** a
reproduction of PlannerRFT's complete optimizer. The paper combines guided
sampling with closed-loop execution, PPO for the Exploration Policy, GRPO over
diffusion transitions and a survival reward. Our pilot intentionally keeps the
existing open-loop reward and AWR regression objective so that the exploration
change can be isolated. The paper's end-to-end gain is evidence for the idea,
not a guaranteed gain for this adaptation.

The optimization mismatch is material. PlannerRFT treats every one of five
stochastic DDIM transitions as a Gaussian policy, uses group-relative GRPO with
denoising discount 0.8, and adds a behavior-cloning term with published weight
0.4. It trains with DDIM stochasticity `eta=1` and evaluates with `eta=0`.
Original-DP AWR instead performs reward-weighted denoising regression on cached
final trajectories. Therefore guidance can be transferred as a data-generation
idea, but its published checkpoint gain cannot be attributed to data
augmentation alone or assumed to survive our different projection loss.
Likewise, PlannerRFT's published BC coefficient `c_b=0.4` multiplies a separate
diffusion-transition log-policy term. It does not establish that appending one
logged-expert x-start target with scalar AWR weight `0.4` to every T4 scene is
equivalent. The matching number in the local expert-anchor recipe is an
engineering prior whose scope and effect require local ablation.

### Why the formal replay uses direct x-start MSE

This choice is faithful to the relevant HDP code path, not a generic
post-hoc trajectory MSE.  Released HDP first noises the selected model action,
runs the denoiser, and applies `exp(group_normalized_reward)` to the
per-sample diffusion prediction MSE.  Its additional waypoint/hybrid term is
enabled only when `kinematic_type == "diff"`: that branch integrates a
velocity-like representation back to waypoints.  Original DP already predicts
absolute `x_start` waypoints, so enabling the same velocity-integration
auxiliary would not be a literal transfer.  The formal run therefore uses the
direct normalized x-start diffusion loss and leaves the local hybrid surrogate
off.

The one deliberate adaptation is sampling diffusion training times from
`[0.001, 0.2]` rather than the full interval.  Deployment is a zero-noise
x-start solve, and previous target geometry showed that full-range regression
could improve stochastic targets while moving that deterministic projection in
the opposite direction.  Matched 153,600-scene one-pass pilots reduced the
train-selector regression from `-0.000300` with full-range direct MSE to
`-0.000189` with low-time direct MSE.  Output-only tuning reduced it further,
and the final `beta=2`, `margin=0.01`, expert-weight `0.4` recipe reached
`-0.000034`.  These pilots did not produce a promotable checkpoint; they justify
the formal full-corpus causal test and its immutable starting-checkpoint
fallback, not a positive performance claim.

### What is transferred from PlannerRFT, exactly

PlannerRFT does not smooth a completed trajectory. At every diffusion
denoising step it differentiates two energies around a trajectory from a
frozen reference planner. In compact form, its lateral equation and printed
longitudinal equation are:

```text
lateral target     = normal(ref) · (x - x_ref)  -> lambda_lat * eta_lat
printed lon. error = tangent(ref) · (v - lambda_lon * eta_lon * v_ref) -> 0
eta_lat, eta_lon in [-1, 1]
```

The printed longitudinal expression would target zero speed when
`eta_lon=0`, which conflicts with the paper's description of unbiased,
zero-mean exploration around the reference. We therefore do not claim that a
literal implementation of that line is the intended one.

Different command pairs create left/right and slow/fast modes before reward is
known. They are exploration commands, not labels: infeasible candidates are
allowed to exist, receive a low or gated reward, and are not promoted by AWR.

One reference-sampling difference must remain explicit. PlannerRFT's
supplementary Algorithm 1 initializes the frozen Reference DiT from Gaussian
noise and produces one stochastic reference trajectory for the scene before
sampling the guided group. The current T4 implementation instead uses the
frozen v5 planner's zero-noise trajectory as the common reference. This is a
deliberate stability adaptation, not a line-for-line reproduction: it keeps
the guidance coordinate frame tied to the deployable IL behavior and makes the
effect of `(eta_lat, eta_lon)` easier to audit, but it may leave some reference-
mode diversity unused.

The safe follow-up is a candidate-generation ablation, not an in-place recipe
change. On the same scenes and current-policy latent tensors, compare (a) the
zero-noise frozen reference and (b) one paper-style stochastic frozen
reference. In both arms preserve current-policy candidate 0 and the native
best, rank the native/guided union with the identical reward, and keep the
`gain > 0.01` replay overlay unchanged. A stochastic reference can be admitted
to a later refresh only if it improves best-of-union reward and all-zero-group
recovery without increasing selected hard-gate events. This preserves the
current deployment and training contracts even if the ablation fails.

The paper's full Exploration Policy is a second, more invasive step. It fuses
scene and reference tokens, predicts scene-conditioned lateral/longitudinal
Beta parameters, and learns them with closed-loop PPO. The current formal run
uses the paper's non-learned fixed-Beta ablation instead. This choice is
evidence-based rather than merely convenient: the paper reports `+2.47`
reactive-score improvement for fixed Beta versus `+4.03` for the learned
policy, while uniform exploration regresses by `-2.36`. Fixed Beta therefore
captures a large part of the demonstrated exploration gain without adding a
second policy, critic, simulator-return estimator or deployment dependency.

If a learned proposal is tested later, it should first run in shadow mode and
must not replace the fixed support. A conservative group would reserve native
candidate 0 and the native best, then split guided proposals between fixed
stratified Beta and a learned scene-conditioned Beta head. Reward ranking can
only add proposals that beat the retained native alternatives. This tests the
paper's adaptivity claim while keeping fixed exploration available when the
learned head is wrong or overconfident.

Original-DP stores absolute future positions beginning 0.1 s after the current
state. Applying PlannerRFT's constant lateral target literally would demand the
full offset at the first waypoint and create a position discontinuity. The
formal T4 adaptation therefore uses the same signed normal-frame intent but
ramps the target from zero with a quintic smootherstep over 20 waypoints
(2 seconds), then holds it. Longitudinal exploration uses the repository's
audited speed-stretch surrogate, `stretch = 1 + 0.25 * eta_lon`, which
operationalizes a signed +/-25% relative-speed command and changes
successive displacement rather than translating the whole path. This preserves
the current-state boundary and was selected by real-scene response sweeps; it
is not presented as a line-for-line reproduction of the paper's printed
velocity energy.

For each scene, AWR consumes the result as follows:

```text
current policy: 1 deterministic + 9 native stochastic plans
        |
        +-- native best >= 0.90 --> keep native group
        |
        `-- native best < 0.90  --> add 10 guided plans around frozen v5
                                     |
                              OBB/map/kinematic reward
                                     |
                         preserve deterministic behavior;
                         top-9 from native/guided union
                                     |
                        positive gain > 0.01, beta=2 AWR
                              + gated expert weight 0.4
                                     |
                         ordinary Original-DP checkpoint
```

The frozen reference, Beta commands and guidance gradients are absent from
checkpoint selection, validation and deployment. What the deployed model can
retain is only the behavior distilled into its normal denoising weights by
AWR. This is the reason the mechanism is useful for post-training without
creating a second inference stack.

### Transfer decision for Original-DP AWR

The useful transfer is not “add geometric offsets to every training label.”
It is “use bounded guidance to discover alternatives, let the unchanged
vehicle reward admit them, and distil only admitted behavior into the ordinary
planner.”  The AWR-specific boundary matters because AWR weights are positive:
a low-reward candidate is imitated less, but it does not receive the signed
repulsive gradient that PlannerRFT's GRPO objective can provide.  Therefore
blindly retaining infeasible samples as “negative feedback” would reproduce
neither the paper's optimizer nor a useful AWR signal.

| PlannerRFT idea | Original-DP decision | Reason |
|---|---|---|
| frozen IL reference | use now | anchors exploration to a stable, non-drifting behavior distribution |
| lateral and longitudinal energy at every denoising step | use now | creates modes before reward evaluation rather than fabricating labels afterward |
| fixed symmetric Beta commands | use now | paper ablation is positive and the T4 paired probe shows reward-relevant recovery |
| constant lateral offset from the first future point | replace with a 2 s quintic onset | Original DP predicts absolute future positions; a full 0.1 s offset is a discontinuity |
| guide every scene | guide only when native best is below 0.90 | solved scenes gain little and unnecessary exploration increases projection noise |
| keep every bad guided sample as negative feedback | do not copy into AWR | positive AWR has no negative-policy gradient; preserve deployment/native candidates and reward-rank the union |
| stochastic frozen-reference trajectory | paired shadow ablation | closer to Algorithm 1, but can move the guidance frame away from the deployable mode |
| learned scene-conditioned Beta policy | later contextual proposal experiment | the paper trains it from genuine closed-loop PPO returns; our current open-loop cache cannot honestly supply those returns |
| guidance at deployment | reject | the paper's stronger and faster deployment result is the unguided fine-tuned planner |

Two opt-in candidate-generation experiments are consequently higher priority
than increasing random noise or globally enlarging the lateral range.  First,
compare the current zero-noise frozen reference with one paper-style stochastic
reference under exactly paired policy latents and reward.  Second, compare the
current bounded speed-stretch surrogate with a reference-tangent,
reference-speed longitudinal energy; the paper's printed longitudinal equation
is internally ambiguous at `eta_lon=0`, so this must be judged by trajectory
response and reward rather than assumed correct.  Neither experiment may alter
an active cache or checkpoint.  A later refresh may adopt an arm only after it
improves recovery/best-of-union reward without increasing the hard-event rate
of retained candidates.

The post-Cycle-1 supervisor now executes this as a twelve-arm factorial audit on
384 fixed, failure-stratified real scenes using the selected Cycle-1 policy:
`zero/stochastic reference` crossed with
`candidate-stretch/reference-speed-push` and lateral targets
`1.0/1.5/2.5 m`. The last value is the paper default; the smaller values test
whether Original DP can obtain the useful lane-level branch without an
over-aggressive absolute-position response. Every arm reuses identical policy
latents and stratified-Beta commands. Selection measures the non-negative
best-of-native-guided-union gain, because production always retains native K;
an alternative is adopted for later refreshes only when its mean union gain is
at least 0.002 above the Cycle-1 arm, it loses no more than one all-zero
recovery, its hard-safe candidate count drops by no more than 0.25 out of 10,
it retains at least 80% of the baseline endpoint pairwise diversity, and its
maximum first-waypoint deviation remains at most 0.30 m.
Otherwise the supervisor keeps `zero_noise + candidate_stretch + 1.0 m`. This audit
changes neither the completed Cycle-1 cache nor deployment inference.

### Current experimental conclusion (2026-07-20 18:35 JST)

The candidate-generation part works, but the globally deployable AWR result is
not yet established. The behavior-anchored replay branch continued through
epoch 10 and did not recover: deterministic train-selector reward fell by
`0.002301`, progress by `0.1105`, and path length by `0.422 m`; ADE/FDE rose by
`0.154/0.401 m`. The incumbent checkpoint was therefore retained.

Restricting the update to low diffusion time and the decoder output projection
reduces, but does not remove, global contraction. Matched one-pass results on
the unchanged random 8,192-scene train selector are:

| Replay weighting, low-t/output-only | Reward delta | Progress delta | Path-length delta |
|---|---:|---:|---:|
| positive candidates, beta=1 | -0.000148 | -0.00643 | -0.0358 m |
| positive candidates, beta=2 | -0.000105 (CI crosses 0) | -0.00526 | -0.0374 m |
| gain >0.01, beta=2, no behavior self-distillation | -0.000194 | -0.00523 | -0.0347 m |

The last row nevertheless improves the *actual scenes that supplied its RL
gradient*. On a fixed 8,192-scene sample from the 30,067 active groups,
zero-noise K=1 reward rises from `0.902722` to `0.904101`: delta `+0.001379`,
paired 95% bootstrap CI `[+0.000806,+0.001984]`, with 55.7% of scenes improved.
Road-border events improve by two net scenes, while one new kinematic event is
introduced. This separates the mechanisms: AWR can internalize the selected
guided targets on their source distribution, but the sparse hard-scene update
does not retain ordinary-scene behavior.

The first retention ablation adds safe logged-expert denoising targets at
weight `0.4`. The expert is independently checked by the exact configured hard
gates; 96.44% are admitted, with mean reward 0.9720. On the random 8,192-scene
selector this reduces the significant-target branch's reward regression from
`-0.000194` to `-0.000034` (95% CI crosses zero), with no new collision or
kinematic event and one net road-border recovery. On the fixed active hard
sample, reward improves from `0.902722` to `0.904420`: delta `+0.001698`, paired
95% CI `[+0.001287,+0.002204]`. Thus expert retention and hard-scene AWR are
complementary, but weight 0.4 is not yet enough to establish a globally
positive checkpoint.

Increasing the same expert weight to `1.0` is worse, not better. Random-selector
reward falls by `0.000419`, with paired 95% CI
`[-0.000816,-0.000137]`; one road-border and one kinematic event are added.
This rejects monotonic "more BC is safer" reasoning and supports keeping the
paper's published 0.4 retention scale while changing the hard-scene AWR data
distribution.

A strict `Lt90` overlay was also tested. It preserves the original
group-relative AWR weights only when candidate-0 reward is below 0.90 and
selects 25,106/153,600 groups (16.35%), close to PlannerRFT's published
24,691/144,494 (17.09%) data ratio. This faithful data-distribution ablation is
negative on the random selector: reward changes by `-0.000228`, with paired
95% CI `[-0.000518,-0.000041]`; progress changes by `-0.00569`, ADE/FDE by
`+0.00726/+0.01753 m`, and one additional zero-reward scene appears. It is
therefore rejected in favor of the significant-positive target rule.

Reducing the update by checkpoint interpolation does not repair the remaining
global gap. Mixing 10%, 25%, 50% or 75% of the expert-0.4 proposal into the
incumbent produces random-selector rewards `0.932265`, `0.932268`, `0.932261`
and `0.932304`, respectively, versus `0.932339` for the incumbent under the
same run. None is promoted. The next retention test keeps the deterministic
behavior target only when that target is safe, assigns unsafe behavior anchors
zero weight, and combines this with expert weight 0.4. This changes projection
targets, not candidate mining, reward or deployment inference.

That safe-behavior-anchor test is also negative. On the random 8,192-scene
train selector, reward changes from `0.932354` to `0.932254`: delta
`-0.000100`, paired 95% CI `[-0.000194,-0.000006]`. Collision, road-border and
kinematic event counts are unchanged; ADE/FDE increase by
`0.00726/0.01609 m`. On the fixed 8,192 active hard-scene selector it still
raises reward by `0.001341`, CI `[0.000870,0.001852]`, but this is weaker than
the no-behavior-anchor expert-0.4 result (`+0.001698`) and it introduces one
collision while recovering two road-border events and one kinematic event.
The behavior anchor is therefore rejected. The selected projection recipe for
full-data testing remains significant-positive guided targets (`margin=0.01`,
`beta=2`) plus independently hard-gated logged expert weight `0.4`, with no
deterministic behavior target.

Full-corpus cycle 1 is now running from the retained incumbent on all
5,446,154 filtered training scenes. Mining remains group-relative so the
PlannerRFT native/guided top-union contract is unchanged. Logged-expert
trajectory, reward and hard-safe status are computed once during mining and
committed as a strict rank-local sidecar. After all eight manifests close, a
read-only overlay applies the selected `margin=0.01`, `beta=2`, no-behavior-
anchor weights, and epochs 2--10 replay that cache with expert weight 0.4.
Checkpoint selection uses a fixed 65,536-scene train selector; all 46,262
validation scenes are report-only evaluated every replay epoch. Guidance is
absent from every selector, validation and deployment rollout.

Before the first replay update, the complete 46,262-scene unguided K=1
validation baseline is reward `0.933522`, zero-reward rate `1.8979%`, collision
rate `0.1556%`, road-border crossing rate `1.6601%`, kinematic-failure rate
`0.1038%`, ADE `2.576 m` and FDE `5.677 m`. These are comparison anchors, not
post-training results. The first projection recipe was tracked as W&B run
`1fjgj2d0`; it has now been stopped after the safely closed epoch 3 because the
full validation and fixed train selector both showed statistically resolved
regression. Its `best_train.pth` remains the unmodified incumbent rather than
either negative proposal. The score-aligned replay ablation described below
was launched automatically from the same incumbent and immutable cache. Its
W&B run is `7dn5cx11`.

The score-aligned epoch-2 result is the first full-data positive checkpoint
from this PlannerRFT-guided branch. On the fixed 65,536-scene train selector,
deterministic DP10 reward improves by `+0.00022045`, with paired 95% bootstrap
interval `[+0.00007897, +0.00036839]` and 99.89% bootstrap probability of a
positive mean. Progress improves by `+0.00519`; ADE/FDE improve by
`0.00443/0.00943 m`, all with intervals excluding zero. The selector recovers
three collision scenes and introduces one (net `-2` events), recovers eleven
road-border scenes and introduces five (net `-6`), and reduces zero-reward
scenes by nine net. The declared train-set selector therefore promotes epoch 2
to `best_train.pth` rather than retaining the starting incumbent.

On all 46,262 validation scenes, reward also moves in the positive direction
by `+0.00010865`, but its interval `[-0.00013203, +0.00035011]` crosses zero.
Progress and ADE/FDE improvements are individually resolved; road-border,
kinematic and zero-reward counts improve slightly, while two new collision
events appear without a recovered collision in that population. This is
positive report-only evidence, not a claim that every safety slice improved.
The automatic continuation has started epochs 3--10 from the selected epoch-2
checkpoint. It preserves best-so-far selection and uses the objective-equivalent
active-only context decoder described below.

### Separate the exploration solver from the deployment solver

PlannerRFT candidate discovery and Original-DP checkpoint deployment are two
different protocols and must not accidentally share one `sample_steps` knob.
The formal run now makes the split explicit:

| Purpose | K | Denoising steps | Noise/guidance |
|---|---:|---:|---|
| refresh candidate mining | 10 native + conditional 10 guided | 5 | stochastic training exploration |
| fixed train-set checkpoint selector | 1 | 10 | zero-noise, no guidance |
| full validation every replay epoch | 1 | 10 | zero-noise, no guidance |
| final deployed Original-DP checkpoint | 1 | 10 | zero-noise, no guidance |

Five steps are retained for mining because that is the tested PlannerRFT
exploration sampler and it dominates rollout cost. Ten steps are required for
selection because the ordinary Original-DP decoder uses ten denoising steps at
inference.

An earlier diagnostic incorrectly compared a freshly evaluated DP10 candidate
(`0.932452`) with the run's stored DP5 incumbent (`0.932337`) and reported
`+0.000115`. That is a cross-protocol comparison and is invalid. Re-evaluating
the incumbent with the same DP10 protocol gives `0.932448`, so the valid DP10
paired delta is only `+0.000004`, with 95% CI
`[-0.000273,+0.000331]`. Under a matched DP5 protocol the delta is
`-0.000102`, CI `[-0.000188,-0.000016]`. Thus solver choice can change the
strict-mean selection decision, but the historical DP10 arm does **not**
establish a real improvement. Deployment alignment—not that faulty historical
comparison—is the reason DP10 is the formal selector.

Full validation is K=1 because the replaceable artifact is one ordinary
zero-noise trajectory, not a best-of-10 oracle. Candidate diversity remains a
training-data diagnostic measured from every refresh cache and visualized from
real scene groups. It must not be mixed into the deployment score. The code
records `training.eval_sample_steps` independently, verifies it when any
baseline is reused, and the epoch-100 supervisor rejects a replay run unless
`eval_k=1` and `eval_sample_steps=10` are both present in its immutable
effective configuration.

Guided exploration is not disabled after Cycle 1. Global epochs 11, 21, ...,
91 each refresh the replay cache with the latest selected policy under the same
conditional PlannerRFT candidate contract, while the frozen v5 reference stays
fixed. Epochs between refreshes reuse that cycle's cache; this is exploitation
of already scored candidates, not a claim that augmentation has been switched
off. Every refresh preserves the current policy's unguided deterministic
candidate 0, then reward-ranks the other nine slots over `native[1:]` and all
guided candidates. This guarantees that the final group's best reward cannot
be lower than the native group's best reward. It does **not** preserve the
identity of the native-best trajectory when at least nine guided alternatives
score above it; in that case evicting it is intentional rather than a loss of
the native reward ceiling.

### What “epoch 100” means in this method

The epoch index counts the complete HDP policy-iteration schedule, not only
gradient passes. Refresh epochs `1, 11, ..., 91` are rollout-only: the frozen
behavior policy generates and scores candidates, the replay buffer is replaced,
and `optimizer_steps=0`. The remaining epochs consume that frozen buffer and
update the policy. Consequently the planned run through global epoch 100 is
exactly **10 full-corpus refreshes plus 90 full replay updates**.

This is not an accidental T4 shortcut. The official HDP NAVSIM implementation
uses the same branch: when `current_epoch % replay_buffer_update_epoch == 0`,
`compute_loss()` calls `_rl_rollout()` and returns metrics without a `loss`;
Lightning therefore performs no backward or optimizer step. All other epochs
call `_rl_train_step()`. With `replay_buffer_update_epoch=10`, HDP's own epoch
number therefore also includes one collection-only epoch per ten schedule
epochs. Reports must state both counts and must not describe global epoch 100
as 100 gradient traversals.

### Autonomy contract: no human checkpoint gate between epochs

The production protocol must not wait for a person to read each audit. At the
end of every replay epoch, the same frozen 65,536-scene train selector evaluates
the deployable zero-noise EMA policy. A checkpoint replaces the immutable
cycle best if and only if its mean deterministic reward is strictly higher;
the last epoch has no special status. Full validation, paired confidence
intervals and component/event counts are recorded automatically for diagnosis
but do not become an undeclared second selector.

At every ten-epoch boundary, the next refresh starts from that immutable best.
If no replay checkpoint beats the cycle incumbent, the system automatically
tries the pre-declared smaller commit fractions on the same train selector;
if none wins, it carries the incumbent forward unchanged. The refresh,
replay, selection, deployment audit and next refresh then proceed without a
manual approval step. A process-level interruption is resumed from strict
manifests and committed checkpoints by a bounded autonomous owner. It retries
only resumable process failures; repeated exits that create no new committed
artifact still fail closed instead of silently changing data or reward.

The schedule target is parameterized rather than embedded in the selection
logic. `TARGET_EPOCH=100` means ten refresh cycles and 90 replay passes;
`TARGET_EPOCH=1000` means 100 refresh cycles and 900 replay passes under the
same rule. Extending the budget cannot overwrite a better earlier exported
model because the best checkpoint is immutable and propagated by measured
train reward. A larger budget may still waste computation after saturation,
so 100 versus 1000 is a resource choice—not a different algorithm and not a
reason to introduce human per-epoch cherry-picking.

### The RL-specific acceleration: score once, replay nine times

The expensive RL operation is not ordinary backpropagation. It is generating
`native K + conditional guided K` trajectories and evaluating every trajectory
against actors, traffic lights, road borders and vehicle kinematics. A refresh
therefore commits the selected final trajectories, exact rewards/AWR weights,
frozen scene representation and independently gated expert target as one
immutable replay generation. The following nine update epochs reuse those
same policy-iteration facts; they do not reopen the raw scene to regenerate K
plans or recompute reward.

Within each replay epoch, scene groups are drawn with replacement for one
buffer-sized pass. This matches the released HDP `ReplayBuffer.get(B)`, which
uses `random.choices`; it is not an accidental failure to shuffle the disk
cache. A single pass therefore contains duplicates and omissions, while the
nine replay passes provide repeated stochastic coverage before the next
on-policy refresh.

The remaining full-data bottleneck is specific to replay: random sampling
re-reads the frozen scene encoding and joint-neighbor diffusion context on
every pass.  The strict Cycle-1 cache stores about 4.6 TiB of such float32
arrays.  The completed lossless sidecar is 763 GiB, about 16.7% of the raw
context. It keeps the original cache untouched, binds every rank to
the exact scene-order hash, checksums each frame, atomically publishes only a
complete manifest, and reconstructs float32 tensors byte-for-byte before the
existing model path.  Plain-memmap and compressed readers are regression-tested
to return `torch.equal` trajectories, context, encoding and expert anchors,
including duplicated and out-of-order replay indices.  This reduces random
I/O without changing candidates, weights, RNG, batch order or gradients. In
the first stable full-data window it sustained 64 replay updates in 31 seconds
(`2.06 updates/s`), projecting about 29 minutes for 3,546 updates; the older raw
context path required roughly 50 minutes for the same replay body. This is an
online throughput observation, not a reward result.

The score-aligned recipe makes another exact optimization possible. Only
19.45% of groups have a non-zero AWR target, and its expert anchor is explicitly
gated to those same groups. Future replay processes first read the small weight
row and decompress scene encoding, neighbor context and expert trajectory only
for groups with non-zero target weight. Zero-weight rows retain zero host
placeholders and cannot be selected by the compacted loss graph; their exact
expert reward/safe scalars are still mmap-read for unchanged diagnostics. A
parallel and active-only regression suite verifies byte equality for every
materialized row, duplicate index and sample order. This removes dead context
I/O; it does not resample hard scenes or change the number, weight, noise, time
or denominator of any gradient-bearing target.

A low-priority audit on the real rank-0 full cache sampled the exact epoch-2
RNG batch of 192 groups: 42 carried non-zero weight. Every gradient-bearing
encoding, neighbor tensor and expert trajectory was byte-identical to the full
decoder; trajectories, noise scales, weights, paths and expert reward/safe
scalars were identical for the complete batch. Context materialization fell
from `0.511 s` to `0.126 s` (`4.04x`) for that batch. End-to-end replay speedup
will be measured separately because decoder compute and DDP synchronization do
not scale by the same factor.

This is method-level acceleration, not a change to the AWR objective. The next
refresh still uses the latest selected policy and fresh diffusion noise, so the
data distribution follows policy improvement. A strict eight-rank manifest is
the transaction boundary: replay starts only after every rank has written its
declared scene count and schema. An incomplete generation is never mixed with
the previous one. This is how large-scale rollout--replay remains both fast and
auditable.

The full-data replay forms one 5% policy proposal per complete replay pass:
train a live output-projection proposal, form
`candidate = 0.95 * old + 0.05 * proposal`, and evaluate that deployable EMA
on the fixed DP10 train selector. The committed 5% policy becomes the starting
policy for the next replay epoch whether or not it is a new best; only a strict
mean-reward improvement replaces `best_train.pth`. Adam state is cleared at
the 5% commit boundary, so each pass proposes a fresh local step from the
previous committed policy.

Training continuity and checkpoint promotion are deliberately separate. An
earlier same-cache branch showed that cumulative epochs can decline almost
monotonically, which is why the immutable best-so-far file is required. But an
early negative epoch can also be part of a multi-pass recovery, and restoring
`best_train.pth` after every non-promotion would reduce epochs 2--10 to nine
independent one-pass attempts rather than one training trajectory. The final
artifact remains protected because deployment and the next refresh consume
only `best_train.pth`; continuing a non-best intermediate spends training
budget but cannot overwrite the selected checkpoint. W&B therefore reports
“selected as best” and “used as next start” as separate binary facts.
This also matches the released HDP training lifecycle: replay epochs update
continuously and `on_train_epoch_end()` performs no selector-driven rollback;
checkpoint selection is an evaluation concern rather than an optimizer-state
transition.

The 5% boundary is important for scale equivalence. A 153,600-scene pilot has
about 100 effective optimizer updates, so minibatch EMA decay 0.999 exposes
roughly 9.5% new policy; one 5.45-million-scene pass has 3,546 updates and would
expose roughly 97.1%. Keeping the same minibatch decay would therefore turn a
conservative pilot into an almost-raw full-data update. The fixed boundary
commit and independent best-so-far selection change no candidate, reward,
target or replay count. They are an explicit Original-DP policy-improvement
schedule,
not claimed as an unambiguous public-HDP EMA convention. Validation remains
report-only and does not accept or reject updates.

At 368,832 fully written prefix groups (6.77% of the padded full-corpus pass),
the formal mine remains consistent with the 153,600-scene audit: deterministic
deployment reward is 0.93201, final native-guided-union best reward is 0.93973,
and mean best-minus-deployment reward is 0.00771. 19.44% of groups contain a
candidate exceeding deployment by more than 0.01, 1.96% are all-zero, and
96.37% of logged-expert targets pass the independent hard gates. These are
union-level training-signal measurements and are not misreported as the
isolated causal contribution of guidance; that attribution comes from the
matched-scene native-vs-guided probes below.
The phase-2 overlay has mean ESS 1.30 and mean top-1 share 0.893 over update
groups, so it is intentionally close to best-of-K regression rather than
mode averaging. Those concentration metrics remain explicit diagnostics; they
are candidate-weight evidence, not checkpoint-performance evidence.

A later rewards-only sweep over the same 4,241,472 committed groups exposed an
important optimizer sensitivity that is independent of candidate generation.
With the positive margin fixed at 0.01 and the expert weight fixed at 0.4,
`beta=0.5` gives mean active-candidate ESS fraction 0.913, mean top-1 share
0.693 and 40.3% global expert weight share. The heaviest 10% of active groups
carry 29.2% of candidate weight and candidate-weight p99 is 2.83.
`beta=1` gives mean active-candidate ESS fraction 0.781, mean top-1 share 0.783
and 34.5% global expert weight share. At `beta=2` those become 0.666, 0.892 and
16.9%; the heaviest 10% of active groups carry 47.1% of candidate weight, and
the candidate-weight p99 rises from 8.01 to 64.24. Some top-1 concentration is
structural because 39.9% of active groups contain only one positive candidate,
but the additional cross-scene concentration at beta 2 is real under
`hdp_mean`, which divides by the full target count rather than by weight sum.

This does not invalidate the guided cache: trajectories, rewards and frozen
decoder context are independent of replay beta, so the same immutable cache can
evaluate all settings. It does mean that beta 2 remains an experimental arm,
not a settled property of PlannerRFT or HDP. The active beta-2 Cycle-1 replay is
allowed to produce its epoch proposal and is protected by the immutable
train-selector incumbent. Before Cycle 2, the supervisor now runs a bounded
paired probe: beta 0.5, beta 1 and beta 2 each start from the identical
incumbent, consume the same full-cache replay indices for 256 updates, use
expert weight 0.4, and are measured on the exact fixed 65,536-scene train
selector. The higher post-update reward delta selects the later-cycle beta;
differences within `1e-5` select the lower beta because it has higher measured
ESS and lower cross-scene concentration on this cache.
Full validation remains report-only in the formal replay; ESS is a stability
diagnostic, not a checkpoint-selection substitute.

This candidate signal remains stable deep into the corpus. A later read-only
audit at 3,947,712 committed groups (72.48% of the padded pass) divided every
rank-local stream into four equal chronological quarters. The fraction with
`best - deployment > 0.01` was respectively 19.453%, 19.463%, 19.384% and
19.453%; the all-zero fraction was 1.949%, 1.954%, 1.973% and 1.959%.
Rank-level active fractions stayed between 19.336% and 19.520%. Thus the
153,600-scene candidate-discovery result is not an early-file or single-rank
artifact. It still does not establish post-update checkpoint improvement;
that requires the fixed train-selector and full-validation measurements after
replay.

The full mine is now strictly complete at 5,446,656 padded groups: all eight
ranks contain exactly 680,832 groups and publish replay, expert-anchor and
decoder-context completion manifests. The original close stopped with 6,720
tail slots missing on ranks 2 and 7. Recovery verified the existing canonical
path prefix, regenerated only those slots with the same frozen policy,
reference, reward and K=10 candidate contract, and retained strict close. No
scene was skipped or replaced. The decoder-context attachment audit then
reported `candidate_arrays_unchanged=true` and
`reward_weight_expert_rng_unchanged=true`. Epochs 2--10 replay began from this
immutable cache at 15:15 JST; post-update checkpoint improvement is therefore
still pending.

The complete target-geometry audit over all 5,446,656 groups applies
the exact planned overlay (`margin=0.01`, `beta=2`, no behavior anchor) and the
safe-expert weight `0.4`. It finds 19.4468% active AWR groups with 2.712 candidate
targets per active group. Their reward-weighted target improves reward by
`+0.02617` over the deployment anchor and is `+0.498 m` longer in arc length;
its endpoint is `+0.432 m` farther longitudinally. Adding the expert target
makes the active-scene combined target `+0.675 m` longer. On ordinary scenes,
77.30% of all groups receive expert-only retention and those expert paths are
`+0.662 m` longer than the incumbent output. Globally the expert accounts for
16.90% of target weight, so it supplies broad coverage without numerically
overwriting the concentrated hard-scene AWR signal. These measurements reject
the hypothesis that the current target data itself rewards global stopping;
they do not prove that nonlinear diffusion regression will internalize the
same direction.

Multimodality is measured again *after* the positive-advantage filter, rather
than counting every noisy sample. Of active groups, 60.15% retain at least two
positive-weight candidate targets; 27.20% have positive-target endpoints
separated by at least 1 m and 8.61% by at least 2 m. Direction is not inferred
from ego-frame x/y: each endpoint is projected to the closest point and local
tangent of candidate 0's 10 Hz path. In that route frame, 0.934% of active
groups have at least 1 m lateral (`d`) separation and 7.93% have at least 2 m
longitudinal (`s`) separation. Multi-target groups average 1.143 m `s` range
and 0.230 m `d` range. Thus the current exploration contributes mainly
go/slow/stop coverage; genuine lateral branching is present but rare. These
remain geometric branches, not automatically named lane-change, go, or yield
modes; the reproducible selector and OBB/GIF scene renderer provide the
required semantic audit.

### Reward horizon and the guided tail

The formal T4 reward scores the first 40 waypoints (4 seconds), while Original
DP predicts and regresses 80 waypoints (8 seconds). This deserves explicit
audit, but it is not automatically an implementation bug. PlannerRFT itself
stores 150 future frames, reports 4 seconds as its best reward horizon (6
seconds is similar), and applies the resulting trajectory-level reward to the
diffusion trajectory update. A shorter reward horizon than the available
future is therefore part of the published design, not evidence that the loss
must be truncated to 4 seconds.

The T4 cache nevertheless shows why this choice must remain visible. In an
8,192-active-scene paired rescore, the best candidate changes in 67.25% of
groups when the reward horizon moves from 4 to 8 seconds, and only 32.48% of
4-second positive slots remain positive under the 8-second margin. Most of
the selected 4-second targets are still better than deployment over the full
horizon (80.73% of positive slots), while 2.31% acquire a later hard event;
only 0.188% acquire a hard-event type not already present in the deterministic
deployment trajectory. This says that the 4-second objective is genuinely
different, not that its selected candidates are generally invalid.

Accordingly, Cycle 1 is not modified in place. Truncating candidate regression
to 40 steps would leave the deployment tail under-supervised, while blindly
switching reward selection to 80 steps would abandon the paper-supported
4-second objective. If the completed Cycle-1 update does not transfer its
positive source-scene signal to the fixed train selector, the next causal
ablation is a paired same-cache comparison of 4-second and 8-second reward
selection (with the expert anchor still supervising all 80 steps). It is not a
silent loss-mask change during an active full-corpus run.

The same audit quantifies hard-gate cliffs instead of hiding them. A
deterministic-zero candidate with another candidate above reward 0.9 occurs in
0.151% of prefix groups. Within that small recovery stratum, 2.51% have at most
5 cm maximum trajectory displacement and 0.93% at most 1 cm; the former is
only 0.00380% of all groups. One rendered example is a stationary scene where
road-border clearance changes from 0.1973 m to 0.2006 m across the 0.20 m hard
threshold. It is a boundary diagnostic, not a multimodality success example.
Across all hard recoveries, mean per-step, maximum, and endpoint displacement
are 0.290 m, 0.628 m, and 0.549 m respectively, so millimetre-scale flips do
not dominate the formal AWR signal.

The remaining zero-reward stratum needs a different treatment. On a separate
1,045,056-group committed prefix, 1.948% of groups have all ten final rewards
equal to zero. Only 0.84% of that stratum has a logged expert that passes the
same hard gates, leaving 1.932% of all scenes with neither a current nonzero
candidate nor a safe expert target. The formal replay correctly drops those
groups. They become a recoverability pool for later re-sampling or R2LPL-style
repair; globally enabling survival reward would otherwise train against scenes
whose logged expert is also judged unsafe. Any recovered plan must first pass
the unchanged hard reward checks before it can re-enter the same AWR objective.

A later 512-scene audit tested whether PlannerRFT's terminal-prefix survival
signal could safely recover part of this pool.  Unconstrained “later failure is
better” distinguished 21.09% of sampled all-zero groups, but its winner reduced
path length by 0.034 m and progress by 0.014 on average.  We therefore added a
strict diagnostic gate: the candidate must delay failure by at least one frame,
stay within 0.02 progress and 0.2 m path length of deployment, and introduce no
new hard-event type.  Only 3.52% of all-zero groups passed—about 0.069% of the
committed corpus—and every accepted case delayed failure by exactly one frame.
Their mean progress delta was positive (+0.0277), but the coverage and delay are
too small to justify changing the main replay.  These scenes remain a targeted
repair/re-sampling pool rather than a global survival-reward branch.

## Measured T4 evidence and current status

The first production-size candidate mine used 153,600 unique filtered training
scenes, native `K=10`, conditional guided `K=10`, and retained a final `K=10`
reward-top union. It completed on all eight ranks with strict manifests:

| Mining measurement | Result |
|---|---:|
| Unique scenes | 153,600 |
| Throughput | 245.0 scenes/s |
| Scenes triggering guidance (`native best < 0.90`) | 15.16% |
| Mean final-best gain from guided enrichment, all scenes | +0.001076 |
| Mean gain per triggered scene | +0.00710 |
| Guided members retained per triggered final group | 4.05 / 10 |
| All-zero final groups | 1.93% |

This proves that bounded guidance discovers reward-relevant candidates. It does
**not** prove that AWR internalizes them. The first all-stochastic replay pilots
made that distinction explicit:

A separate geometry audit on 65,536 real training scenes asks what those
reward-weighted targets actually change relative to the zero-noise deployment
trajectory.  A scene is active only when the unchanged AWR overlay assigns
positive weight to at least one sampled target; 19.30% of scenes are active.
The table below uses deliberately strict route-frame thresholds: a mode is
called longitudinal only above 1.0 m endpoint progress difference and lateral
only above 0.5 m endpoint offset.  These are target-distribution measurements
before any optimizer update, not checkpoint results.

| Strict target mode among active scenes | Fraction | Mean reward gain vs deployment |
|---|---:|---:|
| near deployment anchor | 71.57% | +0.02588 |
| faster only | 21.44% | +0.02347 |
| slower only | 4.57% | +0.02349 |
| any marked lateral mode | 2.42% | +0.03099 overall |

The threshold sensitivity is important.  With looser 0.5 m longitudinal and
0.25 m lateral thresholds, 11.45% of active scenes contain a lateral mode; at
the strict thresholds only 2.42% do.  Guidance is therefore not globally
turning the planner into a lane-change generator.  Most accepted supervision
remains close to the deployed policy, while a small tail supplies genuinely
different speed or lateral decisions.  This supports a future
scene-conditioned proposal head: spend lateral samples where their measured
advantage is high, rather than increasing the lateral range for every scene.
The source audit files are `target_mode_prefix_65536.json` and
`target_mode_prefix_65536_strict_thresholds.json` in the active Cycle-1 run.

Before the production mine, a paired 192-scene stratified probe reused the
same K latent tensors across native and guided arms. With K=10, stratified
fixed-Beta guidance raised mean best reward from 0.5522 to 0.7315, recovered
35/72 native all-zero groups and lost no native nonzero group. Endpoint
pairwise spread rose from 1.19 m to 2.11 m. The effect is scene-specific rather
than cosmetic: on 24 road-border scenes best reward rose from 0.163 to 0.529;
on 24 collision scenes, from 0.477 to 0.772; on already-solved native
mode-collapse scenes (mean best 0.991), it was unchanged. This is why formal
mining guides only low-reward groups and keeps a native+guided reward-top
union. Stratification preserves the fixed Beta marginal while reducing K=10
coverage variance; it is an implementation adaptation, not attributed to the
paper's literal IID sampler.

The scene-level evidence must distinguish maneuver diversity from mere endpoint
spread.  Formal replay group `rank 1 / local index 100415` is a clean turning-
progress example: all ten candidates follow the same right-turn route and all
ten pass the collision, road-border and kinematic hard checks, but their 8 s
path lengths split into three endpoint components (roughly 24--28 m versus
44--45 m, plus one intermediate member).  None is stopped.  This is evidence
of meaningful longitudinal/progress multimodality through a turn; it is **not**
called a lane-change example.  The figure and full 80-frame animation read the
exact frozen cache group, use the formal `+1` neighbour-future alignment, draw
the true ego and neighbour OBBs, and recompute the replay overlay weights:

- [turning-progress reward/geometry figure](../../Diffusion-Planner-hyper-diffusion-planner/docs/t4_conference_assets/formal_plannerrft_cache/turning_progress_rank1_100415/scene_000.png)
- [turning-progress 80-frame OBB animation](../../Diffusion-Planner-hyper-diffusion-planner/docs/t4_conference_assets/formal_plannerrft_cache/turning_progress_rank1_100415/scene_000.gif)
- [machine-readable scene metrics](../../Diffusion-Planner-hyper-diffusion-planner/docs/t4_conference_assets/formal_plannerrft_cache/turning_progress_rank1_100415/scene_000.json)
- [rendering and provenance manifest](../../Diffusion-Planner-hyper-diffusion-planner/docs/t4_conference_assets/formal_plannerrft_cache/turning_progress_rank1_100415/manifest.json)

This visualization proves that the replay targets contain separated feasible
modes.  Like the aggregate candidate probe, it does not prove that an unguided
post-training checkpoint has internalized the modes; that claim remains gated
on paired checkpoint evaluation after replay.

| One-pass update from the same incumbent | Train-selector reward delta |
|---|---:|
| HDP-style sampling with replacement | -0.000287; 95% paired CI [-0.000640, +0.000074] |
| One exhaustive shuffled pass | -0.000121 |
| 50% low-reward-priority replay | -0.000130 |

Across these branches, centerline/road-border behavior was flat or slightly
better, but zero-noise progress fell by about 0.013--0.014. Therefore neither
sampling variance nor ordinary-scene dilution explains the failure by itself.
The leading issue is conversion of an all-stochastic, multimodal target group
into the single zero-noise deployment mode: the deployment behavior was absent
from every replay group, and MSE projection contracted progress.

A previous EMA line search was accidentally evaluated with ten solver steps,
while the declared train selector uses five. That apparent positive result is
discarded. The corrected five-step line search was negative at proposal mixing
fractions 0.10, 0.25, 0.50 and 0.75; no checkpoint was promoted.

The behavior-anchored ablation above is implemented, unit-tested and mined on
the same 153,600 scenes. All eight manifests contain 19,200 groups;
`candidate0_deterministic` and `behavior_anchor_preserved` are both 1.0. It
retains 0.614 guided candidates per group on average, raises final best over
the deployment anchor by 0.007591, has a 1.93% all-zero rate, and an ESS of
5.53/10.

Its first exhaustive replay pass is nevertheless significantly negative on the
unchanged five-step, zero-noise 8,192-scene train selector:

| Metric | Delta after one 153,600-scene pass |
|---|---:|
| Deterministic reward | -0.000321; paired 95% CI [-0.000422, -0.000220] |
| Progress component | -0.01312 |
| Output path length | -0.0843 m |
| Collision / road-border / kinematic events | 0 / 0 / 0 |
| ADE / FDE | +0.0170 m / +0.0413 m |

This is not evidence that the reward targets favor stopping. Across 150,635
nonzero groups, the normalized reward-weighted trajectory centroid is on
average 0.0957 m *longer* than the zero-noise anchor, and its endpoint distance
is 0.1036 m larger. The post-update deployment trajectory moves in the opposite
direction. The current failure is therefore a diffusion projection/alignment
problem: weighted final-trajectory regression does not guarantee that the
zero-latent deterministic solver moves toward the weighted target centroid.

One pass is only 100 optimizer steps, or 2.8% of the approximately 3,547 steps
in one complete 5.45M-scene replay epoch. The exact live policy, EMA and AdamW
state are being continued through replay epoch 10, selecting every pass by the
same train selector. This is necessary because the earlier positive full-data
branch also dipped on its first replay epoch and recovered from the next one.
No anchored checkpoint is promoted unless that trajectory actually turns
positive.

## What PlannerRFT actually adds

PlannerRFT freezes a copy of the IL planner as a reference model. For every
scene it first samples a common reference trajectory `x_ref`. It then samples
candidate-specific lateral and longitudinal commands from Beta distributions
and applies their energy gradients at every diffusion denoising step.

The two commands are conceptually:

- lateral: move to a bounded signed offset from `x_ref` in its Frenet normal
  direction;
- longitudinal: move faster or slower than the reference in its Frenet tangent
  direction.

The guidance itself has no map or vehicle collision constraints. In the
PlannerRFT GRPO optimizer, unsafe candidates can create negative-advantage
feedback. In our AWR adaptation they do **not** create an explicit repulsive
gradient: they serve as screening evidence and receive low or zero regression
weight after reward ranking. The reference model and guidance machinery are
removed for deployment in both designs.

PlannerRFT's published defaults are five stochastic DDIM steps, group size 8,
maximum lateral offset 2.5 m and maximum relative speed change 25%. Its learned
policy predicts two Beta distributions conditioned on scene context and the
reference trajectory. The final linear layer is zero-initialized so the initial
distributions are symmetric around zero command.

Those paper offsets are not numerically conservative relative to our current
augmentation.  The paper does not publish the fixed Beta concentration or its
logit-to-concentration function.  Our local exploration head uses
`alpha = beta = softplus(0) + 1 = 1.693` at zero initialization; the mapped
command then has standard deviation `0.477`.
At `lambda_lat = 2.5 m`, the lateral target therefore has standard deviation
`1.19 m`; 50%, 90% and 95% of samples have absolute targets below roughly
`0.95 m`, `1.93 m` and `2.13 m`. By comparison, the current Gaussian offset
with `std = 0.5 m` has a central-95% absolute range of about `0.98 m`.
Therefore the paper's 2.5 m setting must first pass a real-scene response sweep;
it is not a zero-risk drop-in constant for Original DP. The formal supervisor
now performs that paired sweep at `1.0/1.5/2.5 m` after Cycle 1 and admits a
larger value only when it improves union reward while retaining at least 80%
of baseline endpoint diversity, nearly all all-zero recoveries and the
hard-safe candidate count, with at most 0.30 m first-waypoint deviation.

Sources:

- paper method: `reference/papers/2601.12901/src/sec/3_method.tex`;
- supplementary algorithm and hyperparameters:
  `reference/papers/2601.12901/src/sec/X_suppl.tex`;
- official project page: <https://opendrivelab.com/PlannerRFT/>.

## Evidence that determines our design

| Exploration during RL | Reactive score | Change from IL | Diversity | Group reward mean | Group reward std |
|---|---:|---:|---:|---:|---:|
| IL pretrained model | 68.18 | — | — | — | — |
| Diffusion noise only | 68.83 | +0.65 | 5.65 | 69.06 | 0.02 |
| Uniform guidance | 65.82 | -2.36 | **39.78** | 60.44 | 0.12 |
| Fixed symmetric Beta | 70.65 | +2.47 | 27.73 | 71.50 | 0.07 |
| Learned PlannerRFT policy | **72.21** | **+4.03** | 25.34 | **73.88** | 0.06 |

Two conclusions are important for our AWR implementation.

First, native diffusion noise alone is often not enough to expose different
driving decisions. Second, maximizing diversity is actively harmful when the
sampler leaves the useful behavior distribution. A bounded Beta prior is a
better first experiment than Gaussian or uniform commands.

The component ablation also shows that the two directions are complementary:
lateral exploration mainly helps drivable-area compliance and comfort;
longitudinal exploration mainly helps collision, TTC, progress and speed. The
combined sampler performs best. It is not sufficient to add only a lateral
lane-change perturbation.

### Data distribution matters as much as the guidance

PlannerRFT also provides a directly relevant warning about replay composition.
Its RL ablation reports:

| RL training scenes | Val14 NR | Val14 R | Test14-hard NR | Test14-hard R |
|---|---:|---:|---:|---:|
| failures only (`Fail`) | 82.97 | 77.48 | 69.26 | 63.75 |
| all scenes (`All`) | 89.93 | **84.88** | 75.50 | 70.43 |
| score below 90 (`Lt90`) | **89.96** | 84.46 | **77.16** | **72.21** |

Training only failures is destructive: it forgets ordinary driving. Training
uniformly on all scenes preserves general performance but dilutes the hard-case
signal. The score-threshold pool gives the best hard-set result while retaining
the general NR result; its Val14 reactive result is, however, 0.42 points below
the all-scene run. Therefore the paper does **not** justify replacing our full
dataset with hard scenes only.

For our full-data requirement, the faithful adaptation is a matched replay
sampling ablation in which every filtered scene remains eligible but low-score
groups are oversampled and ordinary groups remain as an anti-forgetting
component. The actual Cycle-2 cache contains 38.43% groups whose stochastic
group-mean reward is below 0.90, so quotas of 50% and 65% are meaningful
oversampling arms; a 25% quota would accidentally *under-sample* the measured
hard pool. “Hard” must be defined by our own audited group reward rather than
assuming that a local 0.90 threshold is semantically identical to nuPlan's
score. Compare against uniform full-data replay under the same number of
optimizer updates. Selection still uses the unchanged, uniformly sampled
65,536-scene train selector, so a hard-case gain cannot win merely by changing
the evaluation distribution.

This sampling ablation is not active in Cycle 2 and must not be conflated with
guided denoising: one changes which scene groups are replayed; the other changes
which trajectories exist inside each group.

## Difference from the legacy HDP output-space augmentation

The optional legacy-compatible implementation in
`rlvr/awr.py::_apply_hdp_trajectory_augmentation` is a post-denoising
transformation:

- every candidate is first generated normally by Original DP;
- two unbounded Gaussian position offsets are sampled;
- the offsets are applied in each candidate's own heading frame;
- a 20-step quintic onset makes the Original-DP absolute `x_start`
  representation continuous;
- heading is preserved.

PlannerRFT is structurally different:

- all candidates are organized around one frozen common reference trajectory;
- commands are bounded and sampled from Beta distributions;
- longitudinal exploration changes velocity, not merely longitudinal position;
- the guidance gradient is injected throughout denoising, so the resulting
  plan can remain on the model's learned trajectory manifold;
- an optional learned policy changes the command distribution by scene.

The smooth ramp fixed the fatal first-waypoint jump in a direct HDP offset
port, but it does not turn a post-hoc offset into guided denoising. The formal
PlannerRFT full-corpus run sets `hdp_trajectory_augmentation=false`; it uses
guided denoising instead, and the two mechanisms are rejected if enabled
together. Historical ramp experiments remain useful evidence, but they are
not the current candidate-generation method.

The retained pre-PlannerRFT ramp experiment improved the complete
46,262-scene, ten-step deterministic deployment validation by `+0.000499`
mean reward. Its replay groups contained much larger candidate-level ranking
signal (about `+0.045` reward for the weighted candidate over the group mean in
the full-cache audit). These figures come from different scene distributions
and must not be divided into a claimed conversion ratio, but they show that
candidate discovery and policy internalization are distinct bottlenecks.
Guided denoising may create more decision-level alternatives and targets closer
to the model manifold; it does not by itself solve learning rate, EMA or
AWR-loss conversion. Both pre-update candidate quality and post-update
unguided deterministic reward must improve.

## Representation and implementation traps

### 1. Do not use the legacy longitudinal module literally

The printed paper equation and the local legacy module use

`target_speed = lambda_lon * eta_lon * reference_speed`.

With `eta_lon = 0`, that target is zero speed. This conflicts with the paper's
description of zero-mean exploration around a reference and with
`lambda_lon = 25%` being a *relative speed deviation*. It would make the
nominal command a braking/stopping command.

For our probe, the intended operational meaning must be implemented as

`target_speed_scale = 1 + lambda_lon * eta_lon`,

with `lambda_lon = 0.25`, so `eta_lon = -1, 0, +1` means approximately
`0.75x, 1.00x, 1.25x` reference progress. The repository's
`speed_stretch_batched` head has this neutral-at-one behavior. The legacy
`longitudinal` head is retained only for reproduction and must not be selected
for the new AWR sampler.

### 2. Original DP needs a feasible lateral onset

The stock lateral energy requests its complete offset at the first 0.1 s
waypoint. That is incompatible with the absolute Original-DP trajectory state
and can bend the plan head or create an instantaneous lateral jump. Use the
`lateral_ramp_batched` target with a 20-step quintic smootherstep onset, and
audit the full vehicle OBB, yaw rate and acceleration—not just centerline
points. The desired offset begins near zero and reaches the command at 2 s
without a target-velocity or target-acceleration discontinuity.

For a unit command, the 2 s quintic target is only `1.16 mm` laterally at the
first 0.1 s waypoint.  Its analytic peak lateral acceleration is
`5.7735 * A / T^2`: `1.44 m/s^2` for the active `A=1 m, T=2 s`, but
`3.61 m/s^2` for the paper's `A=2.5 m` under the same onset.  The latter
already exceeds the configured `2 m/s^2` lateral-acceleration threshold before
adding road curvature; it would need at least about `2.69 s` of onset even in
the idealized straight-road case.  This is a concrete reason to sweep
amplitude and onset jointly instead of copying `2.5 m` alone.

The smooth *energy target* is not a proof that the denoiser's resulting plan
head is equally smooth.  In the paired 192-scene probe, the maximum actual
first-waypoint displacement between a guided and paired native sample is
`0.218 m`, because a denoising gradient can couple all waypoints.  Every
candidate therefore still passes the full kinematic/OBB reward; the ramp is a
well-posed exploration prior, not a replacement for feasibility scoring.

The active longitudinal surrogate preserves the current-state boundary and
uses the audited `0.75--1.25x` speed-scale semantics, but its displacement
correction is not itself quintic-ramped.  A ramped reference-velocity energy is
a useful paired candidate-generation ablation for a later refresh: compare it
against the current stretch head under identical latent tensors, and admit it
only if it reduces early acceleration/jerk or kinematic rejects without losing
best reward and go/slow separation.  It must not be switched inside the
ongoing Cycle-1 cache.

The ramp is an adaptation for Original DP. It is intentionally not claimed to
be bit-for-bit PlannerRFT.

### 3. `eta = 0` is not the same as guidance off

A quadratic lateral energy with zero target still pulls a noisy candidate
toward `x_ref`. A true unguided control requires `guidance_strength = 0` or no
composer at all. Tests must not use `eta = 0` as an alleged unguided identity
check.

### 4. The reference must remain fixed within an experiment

PlannerRFT attributes stability to guidance around a globally frozen IL
reference rather than the continually changing fine-tuned model. For a matched
AWR pilot, `x_ref` should come from the untouched Original-DP v5 source model,
even if the policy being improved starts from a later selected AWR checkpoint.
The reference checkpoint hash must be stored in the cache manifest and remain
fixed across refresh cycles. Using each new EMA as its own moving reference is
a separate method and removes the stabilizing anchor demonstrated by the
paper. Changing the reference at replay time would also invalidate a cached
group.

### 5. Guided sampling is substantially more expensive

The guidance composer performs an additional model correction and energy
gradient during active denoising steps, and a reference trajectory must also be
generated. PlannerRFT reports 75.48 ms for guided inference versus 34.27 ms for
unguided inference; it removes guidance at deployment both for speed and because
unguided deployment scores slightly better.

Our exact cost depends on Original DP's DPM solver and batched K=10 path. It
must be benchmarked on real X2 scenes before a full 5.45M-scene refresh. A full
guided refresh should not be launched from the paper's latency ratio alone.

The local implementation already contains an important exact optimization:
`rlvr.guidance_batched.dit_memo` reuses the composer's detached x0-correction
DiT result for the solver's same-value DiT call. Its unit tests verify exact
matching, mutation safety and scoped restoration. The real-model tools
`verify_dit_memo.py` and `bench_explorer_latency.py` must be run with our source
checkpoint before estimating full-corpus time. This can make our overhead much
smaller than the paper's unoptimized guided/unguided ratio, but the gain must be
measured rather than assumed.

The real-scene response tool also executes the nine guided commands as one
GPU batch with one shared initial latent. It keeps the unguided control in a
separate `composer=None` call and generates the frozen reference once, reducing
the scene-level model-call groups from roughly 11 sequential calls to 3 without
conflating `eta=0` with guidance-off. The tool fails closed unless the legacy
neighbor offset is `+1`, and records/checks the effective X2 ego width used by
reward and OBB rendering.

### 6. Guidance is exploration, not reward

Collision, TTC, road-border, progress, lane and kinematic semantics stay in the
existing audited reward. Guidance only changes which candidates the reward can
compare. It must not be reported as a safety constraint, and a guided candidate
must not bypass any OBB or legacy neighbor-future `+1` calculation.

### 7. The reward horizon and regression horizon are different in Original DP

The public HDP NAVSIM recipe has a 4 s target and a 4 s PDM reward horizon, so
every waypoint regressed by its planner is covered by the reward. PlannerRFT's
own ablation reports that 4 s and 6 s reward horizons are similar and both beat
2 s, but that experiment does not contain an unscored 8 s target tail.

Original DP predicts 80 waypoints at 10 Hz. The current formal T4 recipe uses a
40-step reward but regresses the complete 80-step candidate. This is therefore
an adaptation mismatch, not a property copied faithfully from HDP. It must be
measured rather than hidden.

On 1,507,968 committed Cycle-1 groups, the positive-overlay candidate target is
on average 0.227 m from candidate 0 during the scored first 4 s and 0.666 m away
during the unscored 4--8 s tail, a ratio of 2.93. The boundary and 8 s endpoint
displacements are 0.465 m and 0.858 m. Some growth is expected because a
go/slow decision accumulates longitudinal separation with time; displacement
alone does not imply that the tail is unsafe.

The paired reward-horizon audit therefore reloads the exact same NPZ, preserves
the `+1` neighbor alignment and legacy X2 width, and re-scores the same cached
K=10 trajectories at both 40 and 80 steps.  The final audit uses a fixed
8,192-group sample drawn only from groups that actually supply positive AWR
targets.  At 8 s:

- 79.11% of all 4 s positive candidate slots remain strictly better than the
  deterministic candidate and 80.73% are non-worse; mean full-horizon
  advantage is `+0.01261`;
- 79.68% of each group's 4 s winner remains strictly better and 81.20% is
  non-worse; mean full-horizon advantage is `+0.01272`;
- only 32.48% of positive slots and 33.78% of 4 s winners retain an advantage
  larger than the original `0.01` margin;
- the identity of the best candidate changes in 67.25% of groups;
- 2.31% of positive slots and 2.15% of 4 s winners acquire at least one newly
  visible hard event in the second half, while candidate 0 itself does so in
  1.84% of the same scenes;
- for the 4 s winner, the newly visible event rates are collision `0.256%`,
  road-border `1.501%`, red-light `0.439%`, and kinematic `0.012%`;
- relative to candidate 0, events uniquely introduced by that winner are much
  rarer: collision `0.098%`, road-border `0.146%`, red-light `0.037%`, and no
  kinematic event.  Across every positive slot, the corresponding unique rates
  are collision `0.072%`, road-border `0.188%`, red-light `0.027%`, and no
  kinematic event.

The CPU audit reproduces stored 4 s totals with mean absolute error `1.65e-7`.
One of 8,192 groups contains a coherent difference across its ten totals, with
a maximum of `0.001095`; every other sampled group's maximum is at most
`2.98e-7`.  This discrepancy is now explained: the formal miner enables
high-matmul TF32, while the first audit used CPU/full-FP32 reward arithmetic.
Re-scoring the exact group on CUDA with the miner's TF32 settings reproduces
all ten stored totals bit-for-bit; disabling TF32 reproduces the CPU values.
It is therefore not a cache, `+1` alignment, X2-width or batch-slicing error.
The audit tool now records its device/matmul protocol and supports exact CUDA
TF32 replay.  The horizon rates above remain the conservative CPU/full-FP32
analysis and are unaffected by one numerically sensitive group.

Dataset strata show why a universal horizon change is not justified.  The 4 s
winner remains non-worse at 8 s in 83.68% of `j6`, 83.07% of `x2_dev`, 80.20%
of `xx1_real`, and 81.07% of `xx1_psim` groups.  `xx1_psim` has the largest new
hard-event rate for positive slots (`5.35%`, versus `0.84--1.43%` in the other
three strata), so a tail guard should be logged per dataset rather than hidden
inside a single corpus mean.  This is small but non-zero, which is exactly why
“native-best cannot decrease during candidate union” is not the same guarantee
as “every selected regression target has a safe 8 s tail.”

A real-cache counterexample makes the mismatch concrete. In formal replay group
`rank 6 / local index 14678`, candidate 1 is the 4 s winner: reward `0.834`
versus deterministic `0.811`, and its stored AWR weight is `11.23`. All ten
candidates are hard-feasible over 4 s. The exact same candidate crosses the
road-border margin at 6.3 s when scored over 8 s (`RB min = 0.090 m < 0.200 m`)
and its total becomes zero, while candidate 0 remains feasible (`RB min =
0.462 m`, reward `0.785`). The paired figures and full 80-frame OBB animations
use the same frozen trajectories and differ only in reward horizon:

- [4 s formal reward view](../outputs/awr_t4_full_sequence_filtered/plannerrft_full_cycle01_horizon_visual_audit/r6_i14678_h40/scene_000.png) · [4 s GIF](../outputs/awr_t4_full_sequence_filtered/plannerrft_full_cycle01_horizon_visual_audit/r6_i14678_h40/scene_000.gif)
- [8 s tail audit](../outputs/awr_t4_full_sequence_filtered/plannerrft_full_cycle01_horizon_visual_audit/r6_i14678_h80/scene_000.png) · [8 s GIF](../outputs/awr_t4_full_sequence_filtered/plannerrft_full_cycle01_horizon_visual_audit/r6_i14678_h80/scene_000.gif)

This is intentionally a failure-case visualization, not evidence that guidance
improves the scene. It shows exactly what a future tail guard is meant to remove.

Thus the current 4 s reward usually selects a direction that remains useful at
8 s and does not systematically create unsafe tails, so Cycle 1 should not be
discarded. It is nevertheless noisy as an 8 s ranking rule. Before changing a
later refresh, expand this audit and compare three matched choices: 4 s reward,
8 s reward, and a conservative dual-horizon rule that requires positive 4 s
advantage while using 8 s reward only as a tail-feasibility/tie-break signal.
Do not silently switch the ongoing cache, and do not call `awr_use_prefix_mask`
a horizon fix: that option controls SFT-style random delay conditioning, not
which future waypoints contribute to the diffusion loss.

The dual-horizon option is preferred to blindly replacing 4 s by 8 s. The
deployed planner replans at 10 Hz, so the 4--8 s log-replay future is not
executed as one open-loop commitment and becomes less reliable as an
interaction model. At the same time, AWR currently regresses that complete
tail, so ignoring it altogether is also unjustified. A 4 s primary advantage
plus an 8 s “no newly introduced hard event” guard preserves the paper-backed
moderate planning horizon while preventing an unscored tail from entering the
target unchecked. It remains an ablation until a matched train-selector result
beats the current recipe.

### 7a. Full-cache diagnosis and the score-aligned replay ablation

The completed Cycle-1 cache contains all `5,446,656` padded replay groups. It
confirms that candidate discovery is useful but sparse: `1,059,200` groups
(`19.4468%`) contain at least one candidate above deterministic reward by the
declared `0.01` margin, with `2.712` positive candidates per active group on
average. Within active groups, `60.15%` contain multiple positive targets;
`27.20%` have at least one positive endpoint separated by 1 m and `8.61%` by
2 m. This is evidence of usable candidate support, not yet evidence that the
deployed zero-noise checkpoint improves.

The same full-cache audit exposes two separate projection mismatches in the
first formal replay recipe:

1. The 40-step reward-selected candidate is regressed over 80 steps. Its mean
   displacement from deterministic is `0.2268 m` in the scored prefix but
   `0.6655 m` in the unscored tail (`2.934x`); the first unscored point is
   already `0.4724 m` away on average.
2. A safe expert target is injected in `96.38%` of all groups, although only
   `19.45%` have an AWR candidate. Consequently `77.30%` of the corpus is
   expert-only training, and the expert takes `82.71%` of per-scene target
   share on average when any target exists. That is broad low-noise SFT, not a
   local retention prior around an AWR correction.

The first full replay epoch is a small but statistically resolved negative
result. Epoch 2 changes full deterministic validation reward by
`-0.00051829` on 46,262 paired scenes (bootstrap 95% interval
`[-0.00077384, -0.00025773]`) and fixed train-selector reward by
`-0.00022490` on 65,536 paired scenes (95% interval
`[-0.00035732, -0.00009101]`). Because the earlier group-relative full run
dipped at epoch 2 and recovered at epoch 3, the current process is allowed to
complete epoch 3 before an experiment switch; epoch 2 alone is not used to
declare the entire method invalid.

Epoch 3 did not recover and instead strengthened the diagnosis. Full
validation reward changed by `-0.00069438`, with paired 95% interval entirely
below zero; progress/path length changed by `-0.01326/-0.09528 m` and ADE/FDE
by `+0.01624/+0.04300 m`. The fixed train selector changed by `-0.00029394`,
95% interval `[-0.00044985, -0.00013320]`, while progress/path length changed
by `-0.00972/-0.05615 m` and ADE/FDE by `+0.01474/+0.04040 m`. Rare hard-event
changes do not provide a consistent compensating gain: full validation
recovers one net collision but adds five net road-border and three net
kinematic events; the train selector recovers one collision and six
road-border events with no net kinematic change. The predeclared automatic
decision therefore stopped this branch after epoch 3 and started the matched
40-step-candidate/active-expert replay. This rejects the first **projection
recipe**, not the already measured candidate-discovery benefit of bounded
guidance.

The matched follow-up changes only the two diagnosed projection variables:

- sampled AWR candidates receive diffusion loss on the reward-supported first
  40 steps;
- deterministic candidate 0 and every safe expert anchor keep the complete
  80-step horizon;
- the expert anchor is injected only when the same scene already has a
  non-zero AWR candidate target.

On this exact overlay, active-only expert retention covers `1,039,384` safe
active groups. It contributes only `3.87%` of global target weight; on an
equal-active-scene view its mean share is `13.99%`, median `8.42%`. Therefore
the existing expert weight `0.4` remains fixed for the first test: changing its
scope is isolated from changing its magnitude.

The selected candidate distribution is not equivalent to either of
PlannerRFT's rejected extremes. Although `65.37%` of active *scene counts* have
deterministic reward at least 0.9, candidate AWR weight is concentrated lower:
`56.24%` of total candidate weight comes from `Rdet < 0.9`, and the `0.5--0.9`
range alone contributes `55.08%`. Terminal/very-low scenes (`Rdet < 0.5`)
contribute only `1.16%`, while very easy scenes (`Rdet >= 0.98`) contribute
`0.56%`. The cache therefore behaves like a soft middle-difficulty curriculum,
not All-data easy-scene domination and not Fail-only hard-case replay. A hard
`Rdet < 0.9` filter remains a separate later ablation; adding it to the first
score-aligned test would discard `43.76%` of current candidate weight and
confound the diagnosed horizon/anchor changes.

The implementation is default-off in `rlvr/train_awr.py` through
`awr_candidate_loss_horizon` and `expert_anchor_active_groups_only`. The
historical full-horizon behavior is unchanged unless both are explicitly
selected. Per-target horizon reduction keeps each target's loss normalized by
its own horizon, so shortening a candidate from 80 to 40 points does not halve
its AWR weight. Candidate/expert shape alignment, replacement semantics and
zero tail gradient are covered by `rlvr/test_awr_loss_horizons.py`. The exact
full-data launcher is
`rlvr/autoresearch/run_plannerrft_scored_prefix_active_expert_ablation.sh`;
it starts from the same incumbent, immutable cache, seed/order, beta, margin,
optimizer, EMA commit and evaluation sets as Cycle 1. It also fails closed on
an incomplete lossless compressed-context sidecar.

The replay context has been converted once into eight rank-local, independently
checksummed zstd streams (`763 GB`, from a `4.6 TB` raw context). Random byte
probes verify exact float32 reconstruction before a manifest is published.
This is RL-specific acceleration: candidate scores and scene encodings remain
fixed within one replay cycle, so replay epochs decode the immutable context
instead of rereading the raw per-scene cache. The raw cache is deliberately
retained because the next policy refresh still needs it to regenerate candidate
groups. A first score-aligned launch was rejected before training because the
valid manifest was unreadable under the process's stale supplementary groups;
the empty startup directory was removed, and the unchanged run was restarted
under the existing `ubuntu` group. All eight ranks now see all `5,446,154`
train scenes and `46,262` valid scenes.

This ablation still does not reproduce PlannerRFT's learned exploration policy
or its PPO/GRPO negative feedback. In AWR, infeasible guidance has zero weight
and cannot push the policy away from an action. PlannerRFT guidance is therefore
used only to propose diverse candidates; the AWR objective distills a guided
candidate only after the ordinary vehicle reward verifies it.

There is also a measured solver-transfer mismatch that this cheapest replay
ablation intentionally does not change. Cycle-1 mining uses five denoising
steps, following PlannerRFT, while the declared Original-DP checkpoint selector
and deployment evaluation use ten. On the exact fixed 65,536 train scenes, the
same incumbent's cached five-step candidate 0 has mean reward `0.93184991`
versus `0.93178066` at ten steps; the mean difference is not resolved
(`-0.00006925`, 95% CI crossing zero). The distributions are nevertheless not
interchangeable: mean absolute per-scene reward difference is `0.004172`,
`66.28%` differ by more than `0.001`, and the ten-step output is `0.5220 m`
shorter on average; `82.44%` differ in path length by more than `0.1 m`.
Therefore a negative result after the score-aligned replay must not trigger
further blind tuning on the same five-step cache. The next matched experiment
must compare five- and ten-step candidate mining against the declared ten-step
deployment projection (or explicitly change and validate the deployment
solver); PlannerRFT's paper-backed result uses a matched five-step train/test
solver and does not validate this cross-solver transfer.

### 8. Survival reward is promising in principle, but the current implementation is rejected

PlannerRFT introduces survival reward because a terminal collision/off-road
gate makes every candidate zero in the hardest groups. Its published equation
accumulates only valid, non-terminal reward prefixes. In the current Cycle-1
prefix, `1.951%` of scene groups are all-zero under the formal gate reward, so
this is a real—but narrow—missing-signal population.

A same-cache 512-group audit initially appeared strongly positive: the
repository's historical `reward_mode=survival` made `97.85%` of all-zero groups
rankable. Scene rendering exposed why that number is invalid evidence. The
implementation multiplies first-failure survival fraction by quality computed
over the complete 4 s trajectory. Therefore progress, path length and comfort
*after the first terminal event* can choose the winner. In formal group
`r2:122188`, all ten candidates cross the road-border at exactly `0.3 s`; gate
correctly assigns all ten zero, while legacy survival ranks the longest 4 s
path highest even though it does not survive one frame longer. Group
`r3:52919` is the contraction counterexample: all candidates fail at `0.1 s`,
but legacy survival prefers a path about `0.94 m` shorter than candidate 0.

- [same-failure-time gate view](../outputs/awr_t4_full_sequence_filtered/plannerrft_full_cycle01_mine/20260720-050941_plannerrft_full_cycle01_mine/survival_signal_visual_audit/useful_gate/scene_000.png) · [80-frame GIF](../outputs/awr_t4_full_sequence_filtered/plannerrft_full_cycle01_mine/20260720-050941_plannerrft_full_cycle01_mine/survival_signal_visual_audit/useful_gate/scene_000.gif)
- [legacy-survival false positive](../outputs/awr_t4_full_sequence_filtered/plannerrft_full_cycle01_mine/20260720-050941_plannerrft_full_cycle01_mine/survival_signal_visual_audit/useful_survival/scene_000.png) · [80-frame GIF](../outputs/awr_t4_full_sequence_filtered/plannerrft_full_cycle01_mine/20260720-050941_plannerrft_full_cycle01_mine/survival_signal_visual_audit/useful_survival/scene_000.gif)
- [contraction-risk gate view](../outputs/awr_t4_full_sequence_filtered/plannerrft_full_cycle01_mine/20260720-050941_plannerrft_full_cycle01_mine/survival_signal_visual_audit/stop_risk_gate/scene_000.png) · [80-frame GIF](../outputs/awr_t4_full_sequence_filtered/plannerrft_full_cycle01_mine/20260720-050941_plannerrft_full_cycle01_mine/survival_signal_visual_audit/stop_risk_gate/scene_000.gif)
- [contraction-risk legacy survival](../outputs/awr_t4_full_sequence_filtered/plannerrft_full_cycle01_mine/20260720-050941_plannerrft_full_cycle01_mine/survival_signal_visual_audit/stop_risk_survival/scene_000.png) · [80-frame GIF](../outputs/awr_t4_full_sequence_filtered/plannerrft_full_cycle01_mine/20260720-050941_plannerrft_full_cycle01_mine/survival_signal_visual_audit/stop_risk_survival/scene_000.gif)

Re-scoring the same 512 groups with a faithful terminal-prefix signal—reward is
only the fraction of steps before the first collision or road-border event—
changes the conclusion. Only `26.95%` of all-zero groups become rankable, or
about `0.53%` of the complete replay population. Candidate 0 remains a tied or
strict winner in `83.79%`; among the rankable subset, the best candidate delays
failure by only about `0.68` planning step (`0.068 s`) on average. This narrow
signal may still be useful, but it does not justify changing the formal reward.

Decision: do **not** enable the existing survival mode. A future matched
ablation must compute first-terminal time exactly, ignore every waypoint after
failure, leave equal-failure candidates tied, and handle kinematic-only failures
separately because the current metric exposes no first-violation frame. Given
the small recoverable fraction, R2LPL-style recoverability mining is likely a
better second-stage treatment for the remaining all-zero scenes than forcing
them into ordinary AWR.

## Formal decision and remaining exploration work

### Adopted role in the AWR system

The matched full-data pilot is positive, so PlannerRFT-style guidance now
**replaces** the old post-hoc Gaussian trajectory translation during refresh.
The mechanisms are never stacked: stacking would obscure attribution and could
reintroduce the off-manifold, first-waypoint and mode-averaging failures already
observed in the direct HDP-output-space port.

The adopted generator is deliberately conservative:

1. Generate the complete native `K=10` bank, including the exact zero-noise
   deployment trajectory at index 0.
2. Only when native best reward is below `0.90`, generate a second `K=10`
   guided bank around the globally frozen Original-DP IL reference.
3. Keep deployment candidate 0 and fill the remaining nine slots from the
   reward-ranked native–guided union. A guided sample can add an option but
   cannot remove the deployment control.
4. Enter a sampled trajectory into AWR only if it exceeds candidate-0 reward
   by more than `0.01`. Unsafe or merely different samples are observations,
   not regression targets.
5. Score and regress the sampled candidate over the same first 40 steps;
   candidate 0 and the safe expert retention target remain full 80-step
   anchors. The expert target is present only in an already-active group.
6. Remove the frozen reference and all guidance modules at evaluation and
   deployment. The replaceable checkpoint remains an ordinary Original DP.

This is a transfer of PlannerRFT's guided-denoising *data augmentation*, not a
claim to reproduce its complete algorithm. PlannerRFT additionally learns a
scene-conditioned Exploration Policy with PPO/GAE in closed-loop simulation;
our current version uses a fixed symmetric Beta distribution and optimizes the
ordinary planner with group-relative AWR.

### Completed evidence chain

The following phases are complete rather than proposed:

- Real-model response curves verified left/right and slow/fast direction,
  bounded response, finite OBB reward geometry, and a continuous plan head.
  The lateral command reaches its target through a 20-step quintic onset; the
  longitudinal stretch leaves the current-state boundary unchanged.
- A matched 192-scene probe increased endpoint pairwise spread from `1.19 m`
  to `2.11 m`, raised best reward from `0.5522` to `0.7315`, and recovered
  `35/72` native all-zero groups without losing a native nonzero group.
- The strict full cache contains `5,446,154` real scenes. `1,059,200`
  (`19.4468%`) have at least one sampled target above deployment by the
  declared `0.01` margin; active groups contain `2.712` positive candidates on
  average, and `60.15%` contain more than one accepted target.
- The score-aligned epoch-2 checkpoint improves the fixed 65,536-scene train
  selector by `+0.00022045` reward, paired 95% interval
  `[+0.00007897,+0.00036839]`. Progress, ADE and FDE all improve with intervals
  excluding zero. Full 46,262-scene validation is directionally positive by
  `+0.00010865`, but its reward interval crosses zero; this is reported as
  unresolved rather than significant.
- Continued replay is not monotone. Epoch 3 regressed against the re-evaluated
  epoch-2 start and was not promoted (continuous training still proceeded from
  that proposal). Epoch 4 then improved the same fixed selector
  by `+0.00018473`, paired 95% interval
  `[+0.00004214,+0.00033499]` and was promoted. Epoch 5 then became
  `best_train.pth` (`54e182b0...`): against the exact same epoch-2
  starting policy it improves selector reward by `+0.00025719`, paired 95%
  interval `[+0.00009639,+0.00041903]`. Selector progress improves by
  `+0.01026 m`, ADE/FDE improve by `0.00793/0.01301 m`, and smoothness improves
  by `+0.00208`; all five intervals exclude zero. The rare-event counts are
  mixed rather than hidden: epoch 5 introduces three selector collision scenes,
  recovers two net road-border scenes, leaves net kinematic and all-zero counts
  unchanged, and recovers one net lane-crossing scene. On full 46,262-scene
  validation, reward is directionally positive by `+0.00012011`, ADE/FDE and
  progress improve significantly, and smoothness is directionally positive;
  reward and smoothness intervals still cross zero. Full validation introduces
  two net collision, five net road-border and seven net all-zero scenes while
  recovering one net kinematic scene. This is evidence for immutable
  best-checkpoint selection and refresh/early-stopping, not for treating every
  replay epoch as useful or declaring rare-event safety solved from open-loop
  validation. Epoch 6 subsequently became the current formal best
  (`83ae1647...`). Selector reward improves by `+0.00034759`, paired 95%
  interval `[+0.00017393,+0.00052011]`; progress improves by `+0.01174 m` and
  ADE/FDE by `0.00859/0.01482 m`, all significant. Crucially, this is the
  first continuation checkpoint whose full 46,262-scene reward interval is
  entirely positive: `+0.00038282`, interval
  `[+0.00005612,+0.00070995]`. Its trade-off is also significant: smoothness
  changes by `-0.00415` on the selector and `-0.00292` on full validation.
  Relative to the cycle start, epoch 6 has one net new selector collision,
  four net fewer road-border events, two net fewer kinematic events and five
  net fewer all-zero scenes; full validation has five net new collisions,
  eight net fewer road-border events, two net new kinematic events and one net
  fewer all-zero scene. The declared reward selector therefore promotes epoch
  6, while epoch 5 remains the observed smoother alternative—not a hidden
  replacement selected with validation leakage.
- Epoch 7 verifies that the immutable selector is doing real work rather than
  merely saving the last proposal. Relative to the cycle-start policy, epoch 7
  is still significantly positive on the fixed train selector
  (`+0.00020061`, paired 95% interval
  `[+0.00001079,+0.00039186]`), improves progress by `+0.01590 m`, and improves
  ADE/FDE by `0.01008/0.02007 m`. However, its selector reward is
  `0.93207789`, which is `-0.00014698` below epoch 6's `0.93222487`; it is
  therefore not promoted. Full-validation reward is `+0.00032055` versus the
  cycle start but its interval narrowly crosses zero, and smoothness regresses
  by `-0.01268/-0.01279` on selector/full validation. Epoch 7 also adds six net
  full-validation collision scenes while recovering seven net road-border
  scenes. `best_train.pth` remains byte-identical to epoch 6
  (`83ae1647...`). This is the intended behavior: training may explore through
  a proposal, while the single exported candidate advances only when the fixed
  train selector improves.
- Epoch 8 shows the same saturation more strongly. It remains significantly
  above the cycle start on the train selector (`+0.00023476`, paired 95%
  interval `[+0.00004250,+0.00042046]`) and improves progress and ADE/FDE, but
  its selector reward `0.93211205` is `-0.00011282` below epoch 6, so it is not
  promoted. Full-validation reward is `+0.00034560` versus the cycle start,
  with an interval that still narrowly crosses zero. Smoothness regresses by
  `-0.01636/-0.01224` on selector/full validation; full validation also has
  eight net new collision scenes, four net recovered road-border scenes and
  four net new kinematic scenes. `best_train.pth` therefore remains epoch 6.
- Epoch 9 reverses the epoch-7/8 selector dip and becomes the new immutable
  best (`ca3ae34...`). Its fixed train-selector reward is `0.93232654`, which
  is `+0.00010166` above epoch 6 and `+0.00044925` above the exact cycle start;
  the paired 95% interval versus the start is
  `[+0.00025413,+0.00064819]`. This promotion is independently supported by
  all 46,262 validation scenes: reward rises from `0.93340250` to
  `0.93393482`, a paired gain of `+0.00053232` with interval
  `[+0.00017893,+0.00088164]`. Selector/full progress improve by
  `+0.02389/+0.01866 m`, ADE by `0.01120/0.01199 m`, and FDE by
  `0.02443/0.01800 m`; every interval excludes zero. The trade-off remains
  visible: smoothness changes by `-0.01258/-0.00608`. On the selector, epoch 9
  has one net new collision, unchanged net road-border, one net new kinematic,
  15 net new lane-crossing and three net new all-zero scenes. On full
  validation it has nine net new collision, three net recovered road-border,
  two net new kinematic, 34 net new lane-crossing and eight net new all-zero
  scenes. These paired counts are diagnostics, not an undeclared promotion
  veto: the pre-registered selector remains train-set mean deterministic
  reward, while the immutable checkpoint prevents later proposals from
  erasing epoch 9.
- Epoch 10 closes Cycle 1 without replacing epoch 9. It is still significantly
  better than the exact cycle start on the fixed train selector
  (`+0.00029930`, paired 95% interval
  `[+0.00009359,+0.00050264]`) and on all 46,262 validation scenes
  (`+0.00038437`, interval `[+0.00000795,+0.00075710]`). Progress and ADE/FDE
  again improve significantly. Nevertheless, its selector reward
  `0.93217658` is `-0.00014995` below epoch 9, while smoothness deteriorates
  further to `-0.02311/-0.01725` versus the start on selector/full validation.
  Full validation has six net new collision, unchanged net road-border, five
  net new kinematic, 40 net new lane-crossing and ten net new all-zero scenes.
  The completed Cycle-1 artifact therefore exports epoch 9, with
  `best_train.pth` byte-identical to `epoch_009.pth`
  (`ca3ae34c4a3b6ecb55913e12be9ce7cc42eaae6046d8c0844872672a87217110`).
  This observed late-cycle drift is the reason each refresh starts from the
  selected immutable checkpoint and why replay temperature and candidate
  generation are chosen with matched, bounded probes before epoch 11.
- A systematic audit of `16,384` real active replay groups recomputed the
  production trajectory metrics with each scene's effective vehicle shape,
  including the legacy X2 width override. On the exact 40-step candidate-loss
  prefix, every positive target passes the kinematic hard gate. Relative to
  deterministic candidate 0, reward-weighted positive candidates improve
  smoothness by `+0.36685` on average (`80.90%` of groups), HDP comfort by
  `+0.06263` (`86.59%`) and feasibility by `+0.04433`. Adding the active safe
  expert anchor raises the expected improvements to `+0.43200` smoothness and
  `+0.07455` comfort. A second audit explicitly constructs the raw-coordinate
  centroids implied by plain MSE: the positive-candidate centroid improves
  smoothness/comfort by `+0.43375/+0.07021`, and the candidate-plus-expert
  centroid by `+0.55198/+0.08043`; all `16,384` prefix centroids pass the
  kinematic gate. The checkpoint's occasional smoothness regression is
  therefore explained by neither low-comfort labels nor static geometric mode
  averaging. The remaining suspects are denoising-dynamics projection,
  repeated fixed-cache optimizer drift and the five-step-mining to
  ten-step-deployment solver transfer. Those require matched training
  ablations; blindly filtering guidance would remove good data.

These results establish that the augmentation creates useful candidates and
that the matched AWR projection can internalize part of their gain. They do not
yet establish the optimum number of replay epochs or that every rare event
improves; epoch-by-epoch full validation and immutable best-train selection
remain active through the 100-epoch campaign.

### Why fixed Beta comes before a learned Exploration Policy

PlannerRFT's published ablation gives the relevant ordering. From an IL
R-score of `68.18`, unguided RL reaches `68.83` (`+0.65`), fixed Beta reaches
`70.65` (`+2.47`), and the learned scene-conditioned explorer reaches `72.21`
(`+4.03`). Uniform guidance produces the highest diversity (`39.78%`) but
reduces R-score to `65.82` (`-2.36`); fixed Beta and the learned explorer have
lower, more useful diversity (`27.73%` and `25.34%`). This directly rejects the
idea that maximizing geometric spread is itself the objective. The generator
must expose reward-improving modes while staying near the frozen IL prior.

The paper's learned policy is attractive because it can spend lateral samples
on scenes where lateral alternatives help and longitudinal samples on scenes
where yielding/progress alternatives help. It is not a zero-risk first change:
it adds a second trainable policy, value estimation, temporal credit assignment
and a possible source of exploration collapse. Fixed stratified Beta isolates
the causal value of guidance while retaining symmetric zero-mean coverage.

The next safe extension is therefore a **shadow explorer**, not an immediate
mainline replacement. Log `(scene, reference, eta_lat, eta_lon, reward)` during
future refreshes; train a proposal head offline to increase the probability of
measured high-advantage commands; then compare it against fixed Beta using the
same scenes, latent noise, candidate budget and unchanged AWR objective. Admit
it only if it improves best/weighted candidate reward and useful mode coverage
without increasing all-zero groups or concentrating all probability on one
command. Until that paired evidence exists, fixed Beta remains the formal
generator.

Before Cycle 2, replay temperature is selected separately from candidate
generation. The `beta=0.5/1/2` arms use the same immutable cache, replay seed,
fixed 65,536-scene train selector and one complete `3,546`-update replay pass.
This full-depth comparison replaces the earlier 256-update direction probe:
selector evaluation dominates the probe cost, while choosing the temperature
for nine later cycles from only `7.2%` of a replay epoch would be unnecessarily
noisy. Reward delta is the selector; when deltas are within `1e-5`, the lower
beta is the tie-break because it gives less concentrated AWR weights.

Candidate-generation parameters must also be selected under the same diffusion
solver that will write the next cache. Cycle 1 mined candidates with five
denoising steps, while deployment and checkpoint selection use ten. A previous
draft of the parameter-sensitivity runner silently defaulted to ten steps even
though the subsequent refresh would have returned to five; that mismatch has
been removed. Before epoch 11, a paired 384-scene audit now compares five and
ten steps using identical scene IDs, latent noise, stratified Beta commands,
reference policy, K and reward. Ten steps is adopted for future
*training-time* generation only if it improves native-guided union reward by at
least `0.002` while retaining all-zero recovery, hard-safe candidate count and
at least `80%` of five-step diversity; otherwise the less expensive five-step
solver remains. The subsequent twelve-arm guidance search inherits the chosen
step count, and every refreshed cache records and validates it. Deployment
evaluation remains the fixed ten-step deterministic protocol in either case.

Two reproducibility limits are explicit. First, PlannerRFT does not publish the
fixed Beta concentration mapping; our symmetric
`alpha=beta=softplus(0)+1` prior and CDF stratification are local,
deterministically tested variance-reduction choices. Second, the paper's
printed longitudinal energy is not inert at `eta_lon=0` if read literally.
The Original-DP adapter uses the physically testable form
`speed_scale = 1 + lambda_lon * eta_lon`, for which zero command exactly
preserves the candidate. As of 2026-07-20 the official project page's Code
button is still commented out, so neither detail should be attributed to an
unreleased reference implementation.

ADE/FDE remains diagnostic rather than a hard veto. Checkpoint selection uses
the declared fixed train-set deterministic reward; full validation is an
independent report, not the selector.

## Visualization and observability contract

Every compared scene should produce an animated top-down visualization based on
the existing colleague tools, extended rather than replaced:

- map and road borders;
- ego and neighbor OBBs at synchronized time steps;
- frozen reference trajectory;
- all K candidates colored by reward rank;
- selected high-weight AWR targets;
- collision/border/TTC event frame and footprint clearance;
- lateral and longitudinal command values;
- per-candidate total reward and component table.

Group-level charts should include:

- footprint-mIoU diversity, matching PlannerRFT's stated diversity concept;
- pairwise trajectory ADE and endpoint lateral/longitudinal spread;
- discrete maneuver occupancy (hold/yield/proceed and left/center/right), so
  speed jitter is not mistaken for a new mode;
- reward mean, best, standard deviation and best-minus-mean;
- all-zero/tied rate, valid-group rate, ESS fraction and top-1 weight share;
- collision, TTC, road-border, progress, lane and kinematic component rates.

The decisive visualization is not the widest bundle. It is a scene where the
baseline candidates all make the same poor decision and bounded guidance
creates a feasible alternative that receives higher reward and is subsequently
internalized by the unguided AWR checkpoint.

## Current code readiness

Reusable local components already exist:

- `rlvr/guidance_batched.py`: batched lateral ramp, speed stretch and optimized
  guidance composer;
- `exploration_policy/`: symmetric Beta heads and frozen-reference utilities;
- Original-DP decoder classifier-guidance hook;
- `rlvr/autoresearch/tools/viz_explorer_trajectories.py` and existing AWR/GRPO
  visualization tools.

The colleague visualization code remains the required base.
`viz_explorer_trajectories.py` has now been routed through
`build_head_composer` with `lateral_ramp_batched + speed_stretch_batched`;
it no longer invokes the legacy `longitudinal` head with the zero-command
stopping trap above. `grpo_viz.py` already contains
the stronger synchronized OBB, reward-geometry and GIF rendering primitives;
the guided sampler should feed those primitives so figures remain comparable
with the existing AWR assets.

The focused guidance, AWR observability, sidecar, overlay and enrichment suite
now passes `74` tests. Real-model response curves, matched-noise audits,
production-size throughput, OBB scene visualization and a 153,600-scene strict
mine have also completed. These establish candidate-generation and engineering
readiness. The 5.45-million-scene cycle is the remaining test of whether the
selected candidates produce a globally better unguided checkpoint; no final
performance claim is made before its replay and full validation complete.

The official PlannerRFT project page currently exposes the paper and results;
no official reference-code link was found there. The local implementation must
therefore be audited against the paper semantics rather than treated as
upstream-certified code.
