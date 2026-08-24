"""Training / validation loop for the DrivoR predictor head.

Kept apart from ``train_epoch.py`` and ``validate_model.py`` because almost
nothing in those is shared: the diffusion head's losses (neighbour prediction,
turn indicator, road-border and neighbour-collision penalties, the DiT sampler)
have no counterpart here, and this head's objective needs the PDM oracle labels
for its own proposals every step.

Throughput notes -- the whole step is written to avoid host synchronisation:
autocast + TF32 for the model, the oracle on un-normalized tensors under
``no_grad``, per-step scalars accumulated as GPU tensors (one reduction per
epoch instead of one ``.item()`` per metric per step), gradient statistics only
on the logging cadence, and no ``torch.cuda.synchronize()`` in the loop.
"""

import os
from typing import Any, Mapping, Optional

import torch
import torch.distributed as dist
from torch import nn
from tqdm import tqdm

from diffusion_planner.model.module.drivor_loss import DrivoRLoss
from diffusion_planner.utils import ddp
from diffusion_planner.utils.drivor_lr import scheduler_base_lrs
from diffusion_planner.utils.drivor_metrics import (
    epoch_metrics,
    global_progress_means,
    selection_metrics,
    step_metric,
    step_metrics,
    trajectory_metrics,
)
from diffusion_planner.utils.drivor_oracle import TTC_UNDEFINED, DrivoROracle
from diffusion_planner.utils.drivor_sampling import (
    resample_expert_future,
    scoring_horizon_slice,
)
from diffusion_planner.utils.train_utils import compute_grad_stats


def heading_to_cos_sin(x: torch.Tensor) -> torch.Tensor:
    """(x, y, heading) -> (x, y, cos, sin); idempotent on 4-column input."""
    if x.shape[-1] == 4:
        return x
    return torch.cat([x[..., :2], x[..., 2:3].cos(), x[..., 2:3].sin()], dim=-1)


def build_drivor_loss(args) -> DrivoRLoss:
    return DrivoRLoss(
        trajectory_weight=args.drivor_trajectory_weight,
        final_score_weight=args.drivor_final_score_weight,
        prev_weight=args.drivor_prev_weight,
        label_smoothing=args.drivor_label_smoothing,
    )


def build_drivor_oracle(args) -> DrivoROracle:
    return DrivoROracle(
        dt=args.drivor_oracle_dt,
        pose_dt=args.drivor_pose_dt,
        scoring_num_poses=args.drivor_scoring_num_poses,
        collision_stride=args.drivor_oracle_collision_stride,
        ttc_stride=args.drivor_oracle_ttc_stride,
        border_stride=args.drivor_oracle_border_stride,
        route_stride=args.drivor_oracle_route_stride,
        max_neighbours=args.drivor_oracle_max_neighbours,
        max_border_segments=args.drivor_oracle_max_border_segments,
        max_route_segments=args.drivor_oracle_max_route_segments,
        score_weights=tuple(
            float(getattr(args, f"drivor_weight_{name}"))
            for name in (
                "no_at_fault_collisions",
                "drivable_area_compliance",
                "driving_direction_compliance",
                "time_to_collision_within_bound",
                "ego_progress",
                "history_comfort",
            )
        ),
    )


def amp_dtype(args) -> torch.dtype:
    return torch.float16 if str(args.amp_dtype).lower() in ("fp16", "float16") else torch.bfloat16


def resolve_device(args) -> torch.device:
    """The device this rank owns.

    ``ddp_setup_universal`` has already called ``torch.cuda.set_device(local_rank)``,
    so the current device is the right one -- the *global* rank is not, as soon as
    there is more than one node.
    """
    if args.device != "cuda":
        return torch.device(args.device)
    return torch.device("cuda", torch.cuda.current_device())


# --------------------------------------------------------------------------
# batch preparation
# --------------------------------------------------------------------------
def prepare_batch(inputs: dict, args, aug=None):
    """Augment, then split the batch into model, oracle and target views.

    The oracle needs metres: ``ObservationNormalizer`` returns a shallow copy, so
    the pre-normalization dict is handed to the oracle while the model consumes
    the normalized one.  Neighbour futures stay in their (x, y, heading) layout
    for the oracle -- this head predicts no neighbours, so nothing else needs the
    (cos, sin) form.

    The three time axes of :mod:`drivor_sampling` are separated here, once:

    * ``ego_future`` is sub-sampled to the head's own ``drivor_num_poses`` at
      ``drivor_pose_dt`` -- by default 40 @ 0.1 s, i.e. the first 4 s of the
      stored rows verbatim -- and is the imitation target.
    * ``ego_reference`` and the oracle's neighbour futures are *clipped* to the
      scorer's ``drivor_scoring_num_poses`` (40 @ 0.1 s, the same 4 s).  Clipping
      keeps the dataset's 0.1 s, which already is the scoring step, so it is
      exact; up-sampling a coarser grid back to it would not be.

    Augmentation runs first, on the full stored horizon, so every derived view
    stays consistent with the perturbed scene.
    """
    inputs["ego_agent_past"] = heading_to_cos_sin(inputs["ego_agent_past"])
    inputs["goal_pose"] = heading_to_cos_sin(inputs["goal_pose"])

    ego_future = inputs["ego_agent_future"]
    neighbors_future = inputs["neighbor_agents_future"]
    if aug is not None:
        inputs, ego_future, neighbors_future = aug(inputs, ego_future, neighbors_future)

    ego_future = heading_to_cos_sin(ego_future)
    scoring_steps = int(args.drivor_scoring_num_poses)
    horizon = scoring_horizon_slice(scoring_steps, ego_future.shape[1])
    ego_reference = ego_future[:, horizon]
    ego_future = resample_expert_future(
        ego_future, int(args.drivor_num_poses), float(args.drivor_pose_dt)
    )

    oracle_inputs = dict(inputs)
    oracle_inputs["neighbor_agents_future"] = neighbors_future[:, :, horizon]
    model_inputs = args.observation_normalizer(inputs)
    return model_inputs, oracle_inputs, ego_future, ego_reference


# --------------------------------------------------------------------------
# epoch aggregation without per-step host syncs
# --------------------------------------------------------------------------
class EpochAccumulator:
    """Running sums of per-step scalars, reduced once at the end of the epoch."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._sums: dict[str, torch.Tensor] = {}
        self._counts: dict[str, torch.Tensor] = {}

    def add(self, values: Mapping[str, Any]) -> None:
        for key, value in values.items():
            if str(key).startswith("_"):
                continue
            if torch.is_tensor(value):
                if value.numel() != 1:
                    continue
                scalar = value.detach().float().reshape(())
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                scalar = torch.as_tensor(float(value), device=self.device)
            else:
                continue
            if key in self._sums:
                self._sums[key] += scalar
                self._counts[key] += 1
            else:
                self._sums[key] = scalar.clone()
                self._counts[key] = torch.ones((), device=self.device)

    def mean(self) -> dict[str, float]:
        """One collective and one host sync for the whole epoch."""
        if not self._sums:
            return {}
        names = sorted(self._sums)
        stacked = torch.stack(
            [self._sums[name] for name in names] + [self._counts[name] for name in names]
        ).to(self.device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(stacked, op=dist.ReduceOp.SUM)
        half = len(names)
        sums, counts = stacked[:half], stacked[half:]
        values = (sums / counts.clamp(min=1.0)).tolist()
        return dict(zip(names, values))


# --------------------------------------------------------------------------
# divergence guard (DrivoR's circuit breaker)
# --------------------------------------------------------------------------
class DivergenceGuard:
    """Skip a step whose loss or logits would poison the weights.

    Breach = non-finite loss, ``|logit|`` beyond the saturation cliff, or a loss
    several times the running EMA.  The flag is MAX-all-reduced so every rank
    skips together: a one-rank skip desynchronises DDP, and one rank's exploding
    batch reaches every rank through the gradient all-reduce anyway.  Repeated
    breaches inside one window halve the learning rate -- if the operating point
    keeps producing catastrophic batches, the peak LR is what has to give.

    Two policies, selected by ``sync_every``:

    ``sync_every == 1`` reads the verdict back to the host every step and truly
    skips ``optimizer.step()``.  Exact, and what this guard has always done --
    but ``.item()`` is an implicit ``cuda.synchronize()``, so the CPU cannot run
    ahead into the next iteration while the GPU drains this one.  Measured on
    8xH100 that serialisation left the GPUs at ~58 % SM and 32 % of their power
    budget, with each rank's Python thread pegged at one core.

    ``sync_every > 1`` keeps the per-step decision entirely on device: the
    MAX-all-reduce stays on the NCCL stream (a collective is not a host sync --
    only the readback is) and the verdict is applied by multiplying the
    gradients by a 0/1 device scalar.  ``optimizer.step()`` then always runs, so
    a breached step contributes zero gradient but still takes AdamW's
    stale-momentum and weight-decay motion; that is a far smaller error than the
    step it replaces.  The host readback drops to once every ``sync_every``
    steps and carries only the decisions that need a Python branch: the LR cut
    and the log line.

    The one thing the device mask cannot do is clear NaN: ``torch 2.11`` has no
    ``_foreach_nan_to_num_`` and ``nan * 0 == nan``, so a non-finite gradient
    survives the mask.  With ``sync_every > 1`` a NaN batch is therefore applied
    and caught within ``sync_every`` steps rather than skipped outright.  Finite
    breaches -- the exploding-logit and loss-spike cases this guard has actually
    fired on -- are handled exactly as before.
    """

    def __init__(self, enabled: bool, logit_bound: float = 0.0, sync_every: int = 1) -> None:
        self.enabled = bool(enabled)
        self.scheduler = None
        self.absmax_limit = 15.0
        self.factor = 4.0
        self.grace = 200
        self.window = 200
        self.max_skips = 10
        self.lr_cut_cooldown = 500
        self.absmax_ceiling = min(0.85 * logit_bound, 8.0) if logit_bound > 0.0 else 8.0
        self.step_count = 0
        self.loss_ema: Optional[torch.Tensor] = None
        self.absmax_ema: Optional[torch.Tensor] = None
        self.last_lr_cut = -(10**9)
        self.recent_skips: list[int] = []
        self.sync_every = max(1, int(sync_every))
        # [breach_seen, drift_seen], MAX-accumulated on device across the window.
        self._pending: Optional[torch.Tensor] = None
        self._keep: Optional[torch.Tensor] = None

    def _drift_flag(self, absmax: torch.Tensor) -> torch.Tensor:
        """Slow-drift flag, kept on device so the decision costs no host sync."""
        zero = torch.zeros((), device=absmax.device, dtype=torch.float32)
        if self.absmax_ema is None:
            self.absmax_ema = absmax.clone()
            return zero
        self.absmax_ema = 0.99 * self.absmax_ema + 0.01 * absmax
        if self.step_count == self.grace:
            # Calibrate the ceiling to *this* model's operating point once the
            # grace window has established it: drift is a departure from where
            # the model runs, not an absolute number.  The one host sync this
            # costs happens exactly once per run.
            self.absmax_ceiling = max(self.absmax_ceiling, 1.25 * float(self.absmax_ema))
        if self.step_count < self.grace:
            return zero
        if self.step_count - self.last_lr_cut < self.lr_cut_cooldown:
            return zero
        return (self.absmax_ema > self.absmax_ceiling).to(dtype=torch.float32).reshape(())

    def check(self, loss_dict: Mapping[str, Any], optimizer) -> bool:
        if not self.enabled:
            return False
        self.step_count += 1
        loss_value = loss_dict["loss"].detach().float()
        absmax = loss_dict.get("logit_absmax")
        drift = (
            self._drift_flag(absmax.detach().float())
            if torch.is_tensor(absmax)
            else torch.zeros((), device=loss_value.device, dtype=torch.float32)
        )

        breach = ~torch.isfinite(loss_value)
        if torch.is_tensor(absmax):
            breach = breach | (absmax.detach().float() > self.absmax_limit)
        if self.loss_ema is not None and self.step_count > self.grace:
            breach = breach | (loss_value > self.factor * self.loss_ema)

        # ONE collective per step, executed unconditionally on every rank and
        # carrying both decisions: a collective inside a conditional deadlocks.
        # The all-reduce itself is not a host sync -- it is enqueued on the NCCL
        # stream like any other kernel.  Only reading ``flags`` back is.
        flags = torch.stack([breach.reshape(()).to(dtype=torch.float32), drift])
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(flags, op=dist.ReduceOp.MAX)

        if self.sync_every > 1:
            # Stay on device: the verdict becomes a 0/1 multiplier that
            # ``mask_grads`` applies to the gradients, and the host learns about
            # it at the next ``resolve``.
            self._keep = 1.0 - flags[0]
            self._pending = flags if self._pending is None else torch.maximum(self._pending, flags)
            keep = self._keep
            updated = (
                0.98 * self.loss_ema + 0.02 * loss_value
                if self.loss_ema is not None
                else loss_value
            )
            # Only a non-breached step may move the baseline the breach test uses.
            self.loss_ema = (
                updated if self.loss_ema is None else torch.lerp(self.loss_ema, updated, keep)
            )
            return False

        breached = bool(flags[0].item() > 0.0)
        if bool(flags[1].item() > 0.0):
            self._cut_lr(optimizer, floor=1e-7, reason="logit drift")
            self.last_lr_cut = self.step_count

        if not breached:
            self.loss_ema = (
                loss_value if self.loss_ema is None else 0.98 * self.loss_ema + 0.02 * loss_value
            )
            return False

        self.recent_skips = [
            step for step in self.recent_skips if step > self.step_count - self.window
        ]
        self.recent_skips.append(self.step_count)
        rank = os.environ.get("LOCAL_RANK", "0")
        print(
            f"[drivor-guard][rank={rank}] step {self.step_count} skipped "
            f"(loss={float(loss_value):.3f}, "
            f"absmax={float(absmax) if torch.is_tensor(absmax) else -1:.2f}, "
            f"skips_in_window={len(self.recent_skips)})",
            flush=True,
        )
        if len(self.recent_skips) >= self.max_skips:
            self._cut_lr(optimizer, floor=1e-6, reason="repeated breaches")
            self.recent_skips.clear()
        return True

    def mask_grads(self, grads: list) -> None:
        """Zero this step's gradients if the device-side verdict says breach.

        One fused ``_foreach_mul_`` over the whole gradient list, so the cost is
        a single kernel launch regardless of parameter count, and no host sync.
        Called on the already-all-reduced gradients, with a ``keep`` that came
        from a MAX-all-reduce -- so every rank multiplies by the same value and
        the ranks cannot drift apart.

        Note this cannot clear NaN (``nan * 0 == nan``); see the class docstring.
        """
        if self._keep is None or not grads:
            return
        torch._foreach_mul_(grads, self._keep)

    def resolve(self, optimizer) -> bool:
        """Periodic host readback: the LR cut and the log line need a branch.

        Returns whether a breach was seen anywhere in the window.  A no-op until
        ``sync_every`` steps have accumulated, which is where the throughput win
        comes from.
        """
        if self.sync_every <= 1 or self._pending is None:
            return False
        if self.step_count % self.sync_every != 0:
            return False
        pending = self._pending
        self._pending = None
        breached = bool(pending[0].item() > 0.0)
        if bool(pending[1].item() > 0.0):
            self._cut_lr(optimizer, floor=1e-7, reason="logit drift")
            self.last_lr_cut = self.step_count
        if not breached:
            return False
        self.recent_skips = [
            step for step in self.recent_skips if step > self.step_count - self.window
        ]
        # The window is only resolved in blocks, so attribute the breach to the
        # block rather than claiming a step number the readback cannot know.
        self.recent_skips.append(self.step_count)
        rank = os.environ.get("LOCAL_RANK", "0")
        print(
            f"[drivor-guard][rank={rank}] breach in steps "
            f"{self.step_count - self.sync_every + 1}-{self.step_count}: gradients zeroed "
            f"(breaches_in_window={len(self.recent_skips)})",
            flush=True,
        )
        if len(self.recent_skips) >= self.max_skips:
            self._cut_lr(optimizer, floor=1e-6, reason="repeated breaches")
            self.recent_skips.clear()
        return True

    def attach_scheduler(self, scheduler) -> None:
        """Register a per-step scheduler so LR cuts are not undone next step.

        A step-interval scheduler recomputes every ``param_group["lr"]`` from its
        ``base_lrs`` on each ``step()``, so halving ``param_groups`` alone would
        survive exactly one iteration.  Cutting ``base_lrs`` instead makes the
        reduction permanent and keeps the shape of the ramp/cosine intact.
        """
        self.scheduler = scheduler

    def _cut_lr(self, optimizer, floor: float, reason: str) -> None:
        for group in optimizer.param_groups:
            group["lr"] = max(float(group["lr"]) * 0.5, floor)
        for base_lrs in scheduler_base_lrs(self.scheduler) if self.scheduler else []:
            base_lrs[:] = [max(float(lr) * 0.5, floor) for lr in base_lrs]
        rank = os.environ.get("LOCAL_RANK", "0")
        print(
            f"[drivor-guard][rank={rank}] {reason}: learning rate halved to "
            f"{optimizer.param_groups[0]['lr']:.2e}",
            flush=True,
        )


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
def train_epoch_drivor(
    data_loader,
    model,
    optimizer,
    args,
    ema,
    aug=None,
    *,
    loss_fn: DrivoRLoss,
    oracle: DrivoROracle,
    guard: DivergenceGuard,
    scaler: Optional[torch.amp.GradScaler] = None,
    wandb_run=None,
    global_step: int = 0,
    scheduler=None,
    max_steps: Optional[int] = None,
) -> tuple[dict[str, float], float, int]:
    """One training epoch.  Returns (epoch metric dict, total loss, global step).

    ``scheduler`` is DrivoR's step-interval scheduler (drivor_agent.py:483) and
    is advanced once per iteration here; pass ``None`` for the repo's legacy
    per-epoch schedule, which ``drivor_train_loop`` steps itself.  ``max_steps``
    cuts the epoch short -- used by the LR range test, which only needs its sweep.
    """

    model.train()
    device = resolve_device(args)
    accumulator = EpochAccumulator(device)
    traj_accumulator = EpochAccumulator(device)
    dtype = amp_dtype(args)
    log_every = max(1, int(args.drivor_log_every_n_steps))
    is_main = ddp.get_rank() == 0
    # Materialise once: ``clip_grad_norm_`` and the gradient statistics both
    # consumed a fresh ``model.parameters()`` generator every step, and the
    # gradient mask needs a list anyway.
    params = list(model.parameters())

    iterator = tqdm(data_loader, desc="Training", unit="batch") if is_main else data_loader
    for step, inputs in enumerate(iterator):
        inputs = {key: value.to(device, non_blocking=True) for key, value in inputs.items()}
        model_inputs, oracle_inputs, ego_future, ego_reference = prepare_batch(inputs, args, aug)

        detailed = (step % log_every) == 0

        with torch.autocast(
            device_type="cuda", dtype=dtype, enabled=bool(args.use_amp) and device.type == "cuda"
        ):
            _, pred = model(model_inputs)

        # The oracle is pure geometry on metric tensors -- fp32, no grad, and
        # deliberately outside autocast so label boundaries cannot move with the
        # compute dtype.
        labels = oracle(pred["proposals"], oracle_inputs, ego_reference)
        loss_dict = loss_fn(pred, ego_future, labels)

        optimizer.zero_grad(set_to_none=True)
        total = loss_dict["loss"]
        if scaler is not None:
            scaler.scale(total).backward()
        else:
            total.backward()

        # The guard's verdict needs a host readback, so it is issued *after*
        # backward is enqueued: the GPU then has a whole backward pass in flight
        # while the CPU waits, and the sync costs almost nothing.  Nothing is
        # committed before ``optimizer.step()``, so a skipped step just drops its
        # gradients -- the next iteration's ``zero_grad`` clears them.
        # Read the LR before the guard can cut it and before the scheduler
        # advances, so the logged value is the one this step actually applied.
        step_lr = float(optimizer.param_groups[0]["lr"])
        skip = guard.check(loss_dict, optimizer)
        if not skip:
            if scaler is not None:
                scaler.unscale_(optimizer)
            # Apply the guard's device-side verdict, if it is using that policy.
            # ``set_to_none=True`` above means the grad tensors are recreated
            # each step, so the list is rebuilt -- pure Python, no launches.
            guard.mask_grads([p.grad for p in params if p.grad is not None])
            if detailed:
                # Before clipping, so an exploding gradient is not hidden by it.
                loss_dict.update(compute_grad_stats(params))
            nn.utils.clip_grad_norm_(params, args.drivor_grad_clip)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            if ema is not None:
                ema.update(model)
        # Host readback of the accumulated verdict; a no-op except every
        # ``--drivor_guard_sync_every`` steps.
        guard.resolve(optimizer)

        # DrivoR advances the schedule per optimizer step (drivor_agent.py:483).
        # Skipped steps advance it too: Lightning's "step" interval counts
        # batches, and holding the LR back on a skipped batch would desynchronize
        # the ramp/cosine phase across ranks that skipped different batches.
        if scheduler is not None:
            scheduler.step()

        traj = trajectory_metrics(pred, ego_future)
        accumulator.add(loss_dict)
        traj_accumulator.add(traj)
        traj_accumulator.add(selection_metrics(loss_dict, "train"))
        global_step += 1

        if detailed:
            live = global_progress_means(step_metrics(loss_dict, traj))
            if is_main and wandb_run is not None and live:
                wandb_run.log(
                    {step_metric(name): float(value) for name, value in live.items()}
                    | {step_metric("learning_rate"): step_lr},
                    step=global_step,
                )
            if is_main and isinstance(iterator, tqdm):
                iterator.set_postfix(
                    lr=f"{step_lr:.2e}",
                    loss=f"{float(live.get('loss_total', loss_dict['loss'])):.3f}",
                    pdms=f"{float(live.get('oracle_selected', 0.0)):.3f}",
                )
            if is_main and getattr(args, "drivor_lr_schedule", "") == "probe":
                # The range test is read off the log, so emit (lr, loss) as plain
                # text too -- it must not depend on wandb being reachable.
                print(
                    "[lr-probe] step={} lr={:.6e} loss={:.4f} pdms={:.4f}".format(
                        global_step,
                        step_lr,
                        float(live.get("loss_total", loss_dict["loss"])),
                        float(live.get("oracle_selected", 0.0)),
                    ),
                    flush=True,
                )

        if max_steps is not None and global_step >= max_steps:
            if is_main:
                print(f"Reached max_steps={max_steps}; ending epoch early.", flush=True)
            break

    means = accumulator.mean()
    extra = traj_accumulator.mean()
    metrics = epoch_metrics(means, "train", extra)
    if ddp.get_rank() == 0:
        print(
            "train loss={:.4f} trajectory={:.4f} scorer={:.4f} oracle_selected={:.4f}".format(
                means.get("loss", float("nan")),
                means.get("trajectory_loss", float("nan")),
                means.get("final_score_loss", float("nan")),
                extra.get("train/selection/oracle_selected", float("nan")),
            )
        )
    return metrics, means.get("loss", float("nan")), global_step


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------
@torch.no_grad()
def validate_drivor(
    data_loader,
    model,
    args,
    *,
    loss_fn: DrivoRLoss,
    oracle: DrivoROracle,
) -> tuple[dict[str, float], dict[str, float]]:
    """Validate the DrivoR head.

    Returns ``(epoch metric dict, devkit score report)``.  The report is the
    PDM panel of the model's *selected* trajectory -- scored on its own, so
    ``ego_progress`` is measured against the demonstration rather than against
    the best of the model's own proposals, which is what makes the number
    comparable across runs.
    """
    model.eval()
    device = resolve_device(args)
    accumulator = EpochAccumulator(device)
    traj_accumulator = EpochAccumulator(device)
    panel = EpochAccumulator(device)
    dtype = amp_dtype(args)

    iterator = (
        tqdm(data_loader, desc="Validation", unit="batch") if ddp.get_rank() == 0 else data_loader
    )
    for inputs in iterator:
        inputs = {key: value.to(device, non_blocking=True) for key, value in inputs.items()}
        # Validation is never augmented: the metric has to describe the data.
        model_inputs, oracle_inputs, ego_future, ego_reference = prepare_batch(
            inputs, args, aug=None
        )

        with torch.autocast(
            device_type="cuda", dtype=dtype, enabled=bool(args.use_amp) and device.type == "cuda"
        ):
            _, pred = model(model_inputs)

        labels = oracle(pred["proposals"], oracle_inputs, ego_reference)
        loss_dict = loss_fn(pred, ego_future, labels)
        accumulator.add(loss_dict)
        traj_accumulator.add(trajectory_metrics(pred, ego_future))
        traj_accumulator.add(selection_metrics(loss_dict, "val"))

        selected = oracle(pred["trajectory"][:, None], oracle_inputs, ego_reference)[:, 0]
        # The TTC sentinel means "no evaluable step", not "score 2"; the panel
        # reports it the way the aggregate treats it -- as no infraction.
        ttc_column = ORACLE_PANEL_KEYS.index("time_to_collision_within_bound")
        selected[:, ttc_column] = torch.where(
            selected[:, ttc_column] == TTC_UNDEFINED,
            torch.ones_like(selected[:, ttc_column]),
            selected[:, ttc_column],
        )
        panel.add({name: selected[:, index].mean() for index, name in enumerate(ORACLE_PANEL_KEYS)})

    means = accumulator.mean()
    extra = traj_accumulator.mean()
    return epoch_metrics(means, "val", extra), panel.mean()


# Panel key order == DrivoROracle output order; these names are the ones the
# devkit's REPORT_KEY_TO_WANDB_KEY table maps onto pdms/nc/dac/ddc/ttc/ep/comfort.
ORACLE_PANEL_KEYS: tuple[str, ...] = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "time_to_collision_within_bound",
    "ego_progress",
    "history_comfort",
    "score",
)


__all__ = [
    "DivergenceGuard",
    "EpochAccumulator",
    "ORACLE_PANEL_KEYS",
    "amp_dtype",
    "build_drivor_loss",
    "build_drivor_oracle",
    "heading_to_cos_sin",
    "prepare_batch",
    "resolve_device",
    "train_epoch_drivor",
    "validate_drivor",
]
