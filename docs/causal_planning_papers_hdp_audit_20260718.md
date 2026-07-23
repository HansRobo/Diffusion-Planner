# Causal Planning Papers: HDP Performance Audit

Date: 2026-07-18

This audit compares the current ego-only Hyper Diffusion Planner (HDP) with:

1. **CausalVAD: De-confounding End-to-End Autonomous Driving via Causal
   Intervention**, arXiv:2603.18561 v2 (CVPR 2026 Highlight).
2. **CausalPlanner: A Causality-Enhanced Planning Framework for Generalizable
   Autonomous Driving**, IEEE Robotics and Automation Letters, vol. 11, no. 5,
   2026. The supplied local PDF is the source for this review.

The local copies are under:

- `reference/papers/arxiv_2603.18561_causalvad/`
- `reference/papers/causalplanner_2026_lra/`

The papers are related, but neither is a drop-in replacement for HDP. The
performance question is whether their causal training ideas can make the
current model safer and more robust without destroying the high-capacity
scene representation, route conditioning, or diffusion objective.

## Executive Verdict

The strongest conclusion is **not** that HDP should become CausalVAD or
CausalPlanner. The highest-value path is a targeted, data-grounded
counterfactual training branch on top of the current HDP:

1. Keep the current temporal diffusion decoder, velocity representation,
   hybrid waypoint loss, full route/lane tokens, and road-border geometry.
2. Generate high-confidence causal-element labels offline using the existing
   route-aware traffic-light and collision geometry.
3. Add a small positive-view consistency objective only when an input element
   is proven irrelevant to the expert trajectory.
4. Use the resulting causal ranking to suppress copycat shortcuts (especially
   stopped neighbors) rather than subtracting arbitrary scene features.

The two papers provide evidence for this direction, but not a reason to add a
generic feature-subtraction module to the main model immediately. Both papers
use approximations to causal inference and neither evaluates the current Tier
IV data contract, red-light labels, road-border geometry, or HDP diffusion
score.

## Paper 1: CausalVAD

### Method

CausalVAD starts from a VAD-style image/BEV model and identifies three claimed
confounding paths: co-occurrence in perception, BEV features acting as a common
cause for prediction, and correlation between agents and map features in
planning. It introduces SCIS (Sparse Causal Intervention Scheme):

- A frozen offline dictionary is built by running a pre-trained VAD once and
  clustering object, map, and agent query embeddings with K-means++.
- The reported prototype counts are `(k_object, k_map, k_agent) = (10, 3, 6)`.
- PDM estimates a prototype-dependent logit bias and subtracts a learned
  class-wise amount from perception logits.
- IDM uses cross-attention from a query to a prototype dictionary to estimate a
  spurious feature component, predicts a per-feature gate, and subtracts the
  gated component before later interactions.
- The dictionary is frozen during the second-stage end-to-end training.
- The total supervised VAD loss is otherwise retained; there is no diffusion
  consistency objective.

The supplement explicitly describes the derivation as a tractable
approximation. It assumes an additive decomposition of a feature into causal
and spurious parts and uses a softmax/expectation approximation. This is useful
engineering intuition, not proof that the learned prototypes are identifiable
causal variables.

### Results relevant to HDP

On the paper's nuScenes setup, CausalVAD reports an average L2 of 0.54 m and
collision rate 0.11%, versus 0.74 m and 0.44% for its VAD-tiny baseline. It also
reports robustness when ego velocity is zeroed or perturbed and transfers to
NAVSIM and Bench2Drive. These numbers are **not directly comparable** to our
Tier IV PDMS/DAC or closed-loop data: the sensor representation, dataset,
horizon, planner, simulator, and metric definitions differ.

The most relevant observation is the reported failure case: a stopped ego and
stopped neighboring vehicle frequently co-occur, so a model can learn to stop
because the neighbor stops rather than because the route-bound red light is
red. That is the same class of copycat failure that matters for our red-light
and intersection behavior.

### What is and is not transferable

Transferable:

- Treat causal intervention as a **training-time counterfactual**, not as a
  post-processing rule.
- Use a frozen, offline artifact for any causal prototype/label computation;
  do not let an auxiliary dictionary move with the model and collapse the
  subtraction target.
- Measure robustness by controlled perturbations of ego state, neighbors,
  route signals, and map context rather than relying on nominal open-loop loss.

Not transferable without evidence:

- Subtracting a learned prototype from HDP scene tokens. HDP cross-attends to
  lane, route, boundary, polygon, goal, neighbor, and turn-indicator tokens;
  a generic subtraction can remove genuinely causal road geometry and change
  the diffusion conditional mean.
- Building the dictionary from a VAD embedding and applying it to HDP. The
  representation spaces and token semantics are different.
- Removing ego history. The paper demonstrates a shortcut risk, not that
  history is useless. HDP currently uses a 21-frame ego history, a current
  velocity condition, and history dropout; this needs a controlled ablation.

## Paper 2: CausalPlanner

### Method

CausalPlanner is a Pluto-style NuPlan planner. It uses 2 seconds of history and
an 8-second future. Its key contribution is a causal-element pseudo-label and
counterfactual contrastive pipeline:

1. Sample diverse lattice trajectories in the drivable area.
2. Represent vehicles, static obstacles, and route-relevant red lights as
   time-varying or fixed boxes.
3. For each candidate trajectory, find elements that cause a collision and
   retain the element with the earliest collision time as a direct-causal
   pseudo-label.
4. Train a causal decoder to produce an element-level causal confidence.
5. Construct a positive scene by removing low-confidence elements. The
   positive scene is required to preserve the original planning target.
6. Construct a negative scene by removing high-confidence elements. The
   negative scene is pushed away from the original in feature space; the paper
   does not require it to produce the original plan.
7. Use both a feature contrastive loss and an output-space constraint, together
   with causal-element focal loss, agent prediction loss, and planning loss.

The PDF's text around the anti-causal mask contains a notation/wording
ambiguity (it defines an anti-causal mask and then refers to the causal mask in
one sentence). The intended behavior is clear from the surrounding description
and figures, but this is another reason not to copy the equations blindly.

The paper reports that the output-space constraint is more important than the
feature-only constraint, that moderate perturbations work best, and that a
two-epoch warm-up before contrastive training improves stability. It also
reports 22.8 hours for pseudo-label generation with 40 CPU threads and 52 hours
for training on its one-million-sample NuPlan setup.

### Why this is especially relevant to our red-light problem

The paper's motivating example is exactly the distinction we need:

- current red route signal + stopped ego is causal;
- a stopped neighbor that merely happens to co-occur is not necessarily causal;
- removing the neighbor while retaining the route-bound red signal should leave
  the stop target unchanged;
- removing the red signal should not be treated as a label-preserving positive
  intervention.

This gives us a principled way to test whether HDP has learned the signal or a
copycat neighbor shortcut.

## Current HDP versus the papers

| Aspect | Current HDP | CausalVAD | CausalPlanner |
|---|---|---|---|
| Input | Tier IV vectorized ego, neighbor, lane, route, boundary, polygon, goal, signal features | camera/BEV VAD sparse queries | vectorized NuPlan scene and Pluto queries |
| Action model | 80 temporal ego action tokens, diffusion, velocity latent | direct trajectory decoder | multimodal lane-conditioned direct decoder |
| Scene conditioning | full scene cross-attention plus global route AdaLN | VAD interaction queries plus SCIS | context attention plus causal decoder |
| Training target | VPSDE x-start/velocity MSE plus small hybrid waypoint loss | supervised VAD losses | supervised planning/prediction plus contrastive causal losses |
| Safety geometry | road-border and neighbor geometry available; coefficients are configurable | collision metric and VAD auxiliary losses | lattice collision pseudo-labels and lane/collision auxiliaries |
| Main causal mechanism | existing history dropout, targeted data filtering/oversampling, route-aware features | prototype bias subtraction | element-level pseudo-labels and counterfactual masking |
| Direct evidence for our data | current Tier IV experiments | nuScenes/NAVSIM/Bench2Drive | NuPlan/InterPlan |

The papers do not invalidate the current HDP architecture. They suggest an
additional **training protocol** around it.

## Highest-value improvements for HDP

### 1. Build a route-aware causal-element sidecar

This is the most promising borrow. Do it offline, not inside every diffusion
step. For each training sample, record element-level labels or soft scores for:

- route-bound red light / stop-line constraints;
- neighbor trajectories that intersect or gate the expert trajectory;
- road-border constraints that the expert or a candidate trajectory actually
  activates;
- static obstacles and other route-relevant constraints;
- clearly non-interacting, distant elements.

The label should be based on a **counterfactual difference**, not merely on
distance or co-occurrence. A practical high-confidence direct-causal test is:

1. run the expert or a small set of expert-near candidate trajectories with an
   element present;
2. remove only that element;
3. recompute route-bound collision/stop feasibility;
4. label it causal only when the feasible action set changes, preferably at the
   earliest affected time.

The existing route-aware red-light filtering must remain strict: only the light
controlling the ego route and stop line counts. A generic nearby red light must
not become a causal label.

This sidecar can be generated once with vectorized geometry and many CPU
threads. It does not modify shared NPZ data or the original lists.

### 2. Add label-preserving positive-view consistency

For a sample with a high-confidence non-causal element, create a second input
by masking only that element to the same padding representation. Feed the
original and masked inputs with the same diffusion time and noise. Keep the
normal HDP loss as the dominant objective and add a small consistency term on
the predicted x-start velocity/waypoint output:

```text
L = L_HDP(original) + L_HDP(positive) + lambda_pos *
    distance(stop_gradient(x0_original), x0_positive)
```

The expert target remains the same only for the label-preserving positive view.
The original branch should be detached in the consistency target (or the two
branches should be symmetrized while retaining the supervised loss) to avoid a
trivial collapse. Start after the existing warm-up, use a small coefficient,
and gate it to high-confidence labels.

This directly targets the stopped-neighbor copycat problem without changing the
decoder or the deployment contract.

### 3. Train a causal ranking head only if the sidecar is reliable

A small head over the existing scene tokens can predict the causal score. It
can be used for the positive-view sampler and for diagnostics. It should not
replace route/lane tokens or subtract features from the main path. The head can
be detached from the main diffusion gradient initially, as the current turn
head is, then enabled only if its labels are stable.

The ranking head should be evaluated with precision/recall on high-confidence
geometric labels and with downstream ablations, not just classification
accuracy. A high causal-head accuracy that does not improve red-light,
road-border, right-turn, or closed-loop metrics is not useful.

### 4. Use controlled counterfactual evaluation before changing training

Before any retraining, run the same checkpoint under a perturbation matrix:

- zero or perturb ego velocity while keeping position history consistent;
- remove stopped neighbors while retaining route and red-light features;
- remove the route-bound red-light feature while retaining neighbors;
- remove distant/non-interacting agents;
- mask non-route lanes versus the route lane;
- perturb road-border features only in scenes where the expert has clearance.

Measure action/velocity change, red-light violation, road-border clearance,
right-turn success, DAC/PDMS, and closed-loop safety. This identifies whether a
shortcut exists before a causal loss is allowed to alter the policy.

### 5. Keep the existing targeted data balancing

CausalVAD's motivation about straight-driving bias supports targeted balancing,
but it does not justify global random masking. Our filtered base list and three
`is_skipped`-filtered unprotected-right-turn lists already provide a more
domain-specific intervention. Preserve them, and stratify diagnostics by
straight/left/right, red-light state, stopped-neighbor state, and road-border
clearance. Do not use the papers as a reason to remove the route or road-border
signals.

## Ideas that should not be adopted directly

1. **Generic SCIS subtraction in HDP.** It could remove a true road-border or
   route constraint and bias the diffusion score.
2. **Random masking of route/lane/signal tokens.** It creates invalid scenes and
   can teach the model to ignore the exact information needed for red-light
   compliance and unprotected turns.
3. **Forcing the negative counterfactual to imitate the expert.** When a causal
   element is removed, the original expert target may no longer be physically
   feasible. Only the label-preserving positive view should receive a strict
   same-target constraint.
4. **Replacing the velocity/hybrid diffusion loss with Smooth-L1.** The papers
   use direct trajectory decoders; this would discard the current HDP score
   parameterization and its temporal denoising behavior.
5. **Removing all ego history.** The shortcut experiment should be measured;
   current speed and recent motion are still causal information for a vehicle
   planner.
6. **Using a pre-trained VAD dictionary for HDP.** The embedding spaces are not
   interchangeable.

## Priority and acceptance criteria

### P0: diagnostic, no model retraining

- Implement/validate the perturbation matrix on a fixed validation subset.
- Produce per-category statistics for stopped-neighbor/red-light scenes,
  right turns, road-border clearance, and ego-velocity perturbations.
- Confirm that route-light masking only refers to the ego-controlling route
  light and stop line.

### P1: low-risk SFT experiment

- Generate a frozen causal sidecar for high-confidence elements.
- Add positive-view consistency with a small coefficient and same `t,z` pair.
- Keep the current HDP loss, full scene tokens, road-border configuration,
  and checkpoint/evaluation contract unchanged.
- Compare against the same seed/data/checkpoint protocol.

### P2: optional causal ranking ablation

- Add a detached causal-element head and test whether its scores improve the
  positive-view sampler.
- Only consider feature intervention or prototype subtraction if this ranking
  branch demonstrates a repeatable downstream gain.

An experiment is a win only if it improves or preserves PDMS/DAC and open-loop
quality while reducing the targeted closed-loop failures. A lower training loss
or higher causal-head accuracy alone is insufficient.

## Final Decision

These papers reveal a credible performance opportunity for HDP, especially for
the red-light/stopped-neighbor copycat failure and other context shortcuts.
They do **not** justify replacing the current HDP decoder or globally removing
scene information. The recommended implementation is an offline, route-aware
causal sidecar plus a small label-preserving positive-view consistency loss,
validated first by counterfactual diagnostics. No main HDP code or active
training configuration was changed in this paper audit.
