# PlAnTF Head Integration in Diffusion Planner

This document records the current design and implementation status of the
PlAnTF-style regression head in Diffusion Planner (DP). For the production
command and parameter reference, see [PlantF Usage and Production Training
Guide](plantf_usage.md).

## Status

The PlantF decoder is implemented on `feat/plantf-decoder-head` and is selected
with `decoder_type=plantf`. The implementation is covered by
`diffusion_planner/tests/test_plantf_decoder.py`.

Implemented:

- a PlAnTF-style winner-takes-all ego trajectory head and direct neighbor
  predictor;
- `mlp`, `basis`, `gru`, and `cross_attn` ego-head variants;
- DP-compatible `prediction`, turn-indicator, normalization, and safety-loss
  interfaces;
- agent-relative XY output support (`plantf_relative_xy`);
- tail-weighted regression, second-difference smoothness, and masked neighbor
  losses;
- full and split ONNX export wrappers; and
- open-loop validation, replan-consistency evaluation, and focused unit tests.

Validated so far:

- unit tests for PlantF decoding, loss paths, and ONNX wrappers;
- local small-subset training and validation smoke tests; and
- open-loop experiments with the current mode-one production recipe.

Still required before production deployment:

- a controlled EMA-versus-regular-weight evaluation;
- closed-loop comparison against the DP head on matched scenarios; and
- deployment validation in the target Autoware runtime.

## Goal

PlantF retains DP's data contract and encoder, while replacing the iterative
diffusion decoder with a one-shot PlAnTF-style trajectory regression decoder.
The objective is to compare a direct regression head and the diffusion head
under the same DP input representation, encoder, data, and output interface.

The adaptation deliberately keeps the DP ecosystem intact:

- input features such as `ego_agent_past`, `neighbor_agents_past`, lanes,
  routes, polygons, and line strings are unchanged;
- the DP encoder and its token order are unchanged;
- downstream consumers continue to receive
  `prediction: [B, 1 + P_n, T, 4]`; and
- the decoder is selected by a configuration switch rather than by a separate
  model family.

The output channels are `(x, y, cos(heading), sin(heading))`. The PlantF
implementation uses the DP horizon (`T=80` at 0.1 s intervals), whereas the
original planTF implementation predicts a much shorter set of sparse future
poses. This temporal-grid difference is important when interpreting the first
predicted point: the PlantF head does not explicitly pin it to the current
state.

## Architecture

| Component | DP diffusion head | PlantF head |
| --- | --- | --- |
| Encoder | DP MLP-Mixer and fusion encoder | Shared DP encoder |
| Ego decoder | DiT-like denoising model | One-shot multimodal regression head |
| Neighbor decoder | Joint diffusion output | Direct per-neighbor regression head |
| Candidate selection | Sampling result | Highest-probability trajectory mode |
| Inference | Iterative solver / denoising schedule | One decoder forward pass |
| Guidance and delay prefix | Supported by the diffusion path | Not part of the one-shot decoder |

The first encoder token is the ego token. PlantF uses it to generate `K`
candidate ego trajectories and `K` mode logits. Neighbor tokens are passed to
the direct neighbor predictor. At inference, the argmax-probability ego mode is
concatenated with the neighbor predictions so that the final tensor remains
compatible with DP consumers.

During training, the decoder additionally returns:

- `trajectory: [B, K, T, 4]` in state-normalized coordinates;
- `probability: [B, K]` mode logits; and
- `neighbor_prediction: [B, P_n, T, 4]` in state-normalized coordinates.

The deployed `prediction` is converted back to the DP ego-centric metric
coordinate system.

## Training objective

For each sample, the ego candidate with the lowest XY ADE is selected without
gradient. The selected trajectory receives Smooth L1 supervision on all four
output channels. Valid neighbor futures receive masked Smooth L1 supervision.

The shared training loop also supports:

- turn-indicator objectives inherited from DP;
- road-border penalty;
- optional neighbor-collision penalty;
- tail weighting of ego regression and smoothness terms; and
- a second-difference XY smoothness penalty.

For `num_modes > 1`, a cross-entropy objective trains the mode probabilities.
For `num_modes=1`, this loss is mathematically zero and is intentionally not
computed or logged.

The current production baseline uses `num_modes=1`. This is a deliberate
choice: current data did not provide sufficient evidence that a learned
mode-selection distribution improves the selected trajectory. Multi-mode
training remains supported and retains its mode diagnostics.

## Coordinates and normalization

PlantF uses the same `StateNormalizer` contract as DP. The normalization file
is therefore part of the model contract and must match across:

1. pretrained encoder loading;
2. PlantF training and validation;
3. checkpoint resume; and
4. inference and ONNX export.

With `plantf_relative_xy=True`, ego and neighbor futures are regressed relative
to their own observed current positions. The decoder restores those positions
before exposing the normal DP-shaped prediction. This is the recommended
representation for a newly trained PlantF head. Do not resume an absolute-XY
PlantF head with this flag enabled, or vice versa.

## DP interface and feature support

`Diffusion_Planner` chooses the decoder through `build_decoder`:

```python
if decoder_type == "diffusion":
    decoder = Decoder(config)
elif decoder_type == "plantf":
    decoder = PlanTFDecoder(config)
```

This keeps training, validation, checkpoint visualization, and most deployment
call sites shared. PlantF does not use diffusion samples, diffusion time, or
the denoising delay internally. Some export wrappers retain these inputs only
to preserve an existing full-graph interface.

The following features are diffusion-specific and should not be assumed to
affect a PlantF prediction:

- iterative denoising steps;
- diffusion guidance;
- delay/prefix constraints applied inside the denoising loop; and
- intermediate denoising trajectory visualizations.

## ONNX and Autoware considerations

The recommended integration path is the full PlantF ONNX graph with a
single-step runtime. The full wrapper retains the legacy input names required
by the existing DP interface, including inputs that the one-shot decoder does
not use directly.

The split multi-step DP runtime is not compatible with PlantF: it expects a
diffusion decoder inside a DPM-Solver loop and, in some configurations, a
separate turn-indicator graph. A dedicated one-shot split runtime would be
required to use split PlantF graphs in that path.

Before deployment, verify all of the following with the actual target runtime:

1. `predicted_neighbor_num=320`, yielding the expected 321-agent output axis;
2. full-graph input and output names and dynamic batch dimensions;
3. numerical agreement between PyTorch and ONNX Runtime; and
4. the intended behavior without diffusion guidance and delay-prefix support.

## Validation and model selection

For `num_modes=1`, PlantF logs top-1 ADE, FDE, and miss rate, plus trajectory
progress/length, speed MAE, second-difference smoothness, stop/slow/move
stratification, and replan consistency when consecutive validation frames are
available.

Mode-only diagnostics that become constants with one mode are omitted. In
multi-mode runs, min-over-mode metrics, oracle gap, mode accuracy, entropy,
mode usage, and the classification loss remain available.

The current global best checkpoint is selected by validation lateral error. It
is not a full driving-quality ranking. FDE, miss rate, moving-scene metrics,
smoothness, road-border behavior, neighbor clearance, and replan consistency
must be reviewed before selecting a deployment candidate.

## Tests

Run the PlantF test suite from the repository root:

```bash
cd diffusion_planner
PYTHONPATH=. ../.venv/bin/python -m pytest tests/test_plantf_decoder.py -q
```

For a training-path smoke test, use a small local train/validation subset and
run one epoch with the same representation flags as the intended production
run. Confirm that `train_log.tsv` contains the expected mode-one metrics and
that trajectory overlays have the correct coordinate anchoring.

## References

- J. Cheng et al., [planTF](https://github.com/jchengai/planTF),
  *Rethinking Imitation-based Planner for Autonomous Driving*, ICRA 2024.
- `diffusion_planner/model/module/plantf_decoder.py`
- `diffusion_planner/diffusion_planner/model/diffusion_planner.py`
- `diffusion_planner/diffusion_planner/validate_model.py`
- [PlantF Usage and Production Training Guide](plantf_usage.md)
