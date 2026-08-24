"""Training entry point for the DrivoR predictor head.

``train.py::model_training`` dispatches here when ``predictor_head == "drivor"``.
The encoder, dataset, augmentation, EMA and LR schedule are the repository's;
everything downstream of the encoder -- objective, metrics, validation -- is
DrivoR's.

Throughput is a first-class requirement here, so the loop enables every
acceleration DrivoR uses: TF32 for fp32 matmuls, autocast (bf16 by default),
``torch.compile``, fused AdamW, persistent DataLoader workers with a filled
prefetch queue, ``find_unused_parameters=False`` plus bucket-view gradients for
DDP, and no per-step ``cuda.synchronize()`` or host readback.
"""

import json
import os

import pandas as pd
import torch
import wandb
from timm.utils import ModelEma
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.train_config import TrainConfig
from diffusion_planner.utils import ddp
from diffusion_planner.utils.data_augmentation import StatePerturbation
from diffusion_planner.utils.data_augmentation_bridge import (
    StatePerturbation as BridgeStatePerturbation,
)
from diffusion_planner.utils.dataset import DiffusionPlannerData
from diffusion_planner.utils.devkit_wandb import (
    define_wandb_score_metrics,
    score_report_to_wandb,
)
from diffusion_planner.utils.drivor_ema import FusedModelEma
from diffusion_planner.utils.drivor_lr import (
    build_drivor_scheduler,
    build_lr_probe,
    scaled_peak_lr,
)
from diffusion_planner.utils.drivor_train import (
    DivergenceGuard,
    build_drivor_loss,
    build_drivor_oracle,
    train_epoch_drivor,
    validate_drivor,
)
from diffusion_planner.utils.lr_schedule import CosineAnnealingWarmUpRestarts
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer
from diffusion_planner.utils.train_utils import resume_model, set_seed

# The checkpoint-selection metric: the PDM score of the trajectory the scorer
# actually selects, which is the quantity the whole head exists to maximise.
CHECKPOINT_METRIC = "val/selection/oracle_selected"

# Where DrivoR-head runs go unless overridden.  Deliberately not an org name in
# source: set ``DRIVOR_WANDB_PATH=<entity>/<project>`` in the environment, or
# pass ``--wandb_project_name <entity>/<project>`` on the command line.
DEFAULT_WANDB_PATH = os.environ.get("DRIVOR_WANDB_PATH", "DrivoR-Head")


def _wandb_target(args) -> tuple[str, str | None]:
    """Resolve ``(project, entity)``, accepting an inline ``entity/project`` path."""
    project = args.wandb_project_name
    entity = args.wandb_entity or None
    if project in ("", None, "Diffusion-Planner"):
        # The diffusion default means "not set for this head".
        project = DEFAULT_WANDB_PATH
    if "/" in project:
        inline_entity, project = project.split("/", 1)
        entity = entity or inline_entity
    return project, entity


def _enable_tf32() -> None:
    """TF32 tensor cores for every fp32 matmul/conv (Ampere+)."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def _build_optimizer(model, args) -> optim.AdamW:
    params = [{"params": ddp.get_model(model, args.ddp).parameters(), "lr": args.learning_rate}]
    if args.fused_optimizer and args.device == "cuda":
        try:
            return optim.AdamW(params, fused=True)
        except (RuntimeError, ValueError) as error:  # unsupported dtype/device mix
            print(f"[drivor] fused AdamW unavailable ({error}); falling back to foreach")
    return optim.AdamW(params, foreach=True)


def _loader(dataset, args, *, global_rank: int, shuffle: bool, drop_last: bool):
    sampler = DistributedSampler(
        dataset, num_replicas=ddp.get_world_size(), rank=global_rank, shuffle=shuffle
    )
    workers = int(args.num_workers)
    kwargs = {}
    if workers > 0:
        # Restarting workers every epoch costs seconds per epoch and empties the
        # prefetch queue exactly when the first batches are needed.
        kwargs["persistent_workers"] = bool(args.persistent_workers)
        kwargs["prefetch_factor"] = int(args.prefetch_factor)
    return sampler, DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size // ddp.get_world_size(),
        num_workers=workers,
        pin_memory=args.pin_mem,
        drop_last=drop_last,
        **kwargs,
    )


def model_training_drivor(args: TrainConfig):
    global_rank, rank, _ = ddp.ddp_setup_universal(True, args)
    print(f"{global_rank=}, {rank=}")

    args_dict = {}
    if global_rank == 0:
        print("------------- {} (DrivoR head) -------------".format(args.exp_name))
        print("Batch size: {}".format(args.batch_size))
        print("Learning rate: {}".format(args.learning_rate))
        print("Use device: {}".format(args.device))
        print("Use AMP: {} ({})".format(args.use_amp, args.amp_dtype))
        print(
            "Proposals: {}  refinement stages: {}".format(
                args.drivor_proposal_num, args.drivor_ref_num
            )
        )

        save_path = args.save_dir
        os.makedirs(save_path, exist_ok=True)
        args_dict = {
            key: value
            if not isinstance(value, (StateNormalizer, ObservationNormalizer))
            else value.to_dict()
            for key, value in vars(args).items()
        }
        args_dict["major_version"] = 5
        with open(os.path.join(save_path, "args.json"), "w", encoding="utf-8") as handle:
            json.dump(args_dict, handle, indent=4)
    else:
        save_path = None

    set_seed(args.seed + global_rank)
    _enable_tf32()
    if args.deterministic:
        # Deterministic kernels cost throughput and the DrivoR objective is not
        # bit-reproducible anyway (the oracle's topk ties are arbitrary), so this
        # is opt-in rather than the default it is on the diffusion path.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False

    if args.use_data_augment:
        if args.augment_type == "bridge":
            aug = BridgeStatePerturbation(augment_prob=args.augment_prob, device=args.device)
        else:
            aug = StatePerturbation(
                augment_prob=args.augment_prob,
                num_refine=args.num_refine,
                device=args.device,
                ego_past_noise_std=args.ego_past_noise_std,
                use_smoothing_future_trajectory=args.use_smoothing_future_trajectory,
            )
    else:
        aug = None

    train_set = DiffusionPlannerData(args.train_set_list)
    valid_set = DiffusionPlannerData(args.valid_set_list)
    train_set.data_list = train_set.data_list[:: args.train_subsample_step]
    valid_set.data_list = valid_set.data_list[:: getattr(args, "valid_subsample_step", 1) or 1]

    train_sampler, train_loader = _loader(
        train_set, args, global_rank=global_rank, shuffle=True, drop_last=True
    )
    _, valid_loader = _loader(
        valid_set, args, global_rank=global_rank, shuffle=False, drop_last=False
    )

    if global_rank == 0:
        print("Dataset Prepared: {} train / {} valid".format(len(train_set), len(valid_set)))
    if args.ddp:
        torch.distributed.barrier()

    model = Diffusion_Planner(args)
    model = model.to(rank if args.device == "cuda" else args.device)
    if args.ddp:
        model = DDP(
            model,
            device_ids=[rank],
            # Every parameter of this head participates in every step, so the
            # unused-parameter graph traversal is pure overhead; bucket views
            # remove one full-size gradient copy per step.
            find_unused_parameters=bool(args.drivor_ddp_find_unused),
            gradient_as_bucket_view=True,
        )

    # timm's ModelEma carries its own deprecation notice and rebuilds two
    # state_dict OrderedDicts per step, then issues three kernels per tensor --
    # several hundred launches every iteration on the one thread that feeds the
    # GPU.  FusedModelEma does the identical arithmetic in two _foreach_ calls
    # and keeps the same state_dict keys, so checkpoints stay interchangeable.
    ema_class = FusedModelEma if args.drivor_fused_ema else ModelEma
    model_ema = ema_class(model, decay=0.999, device=args.device) if args.use_ema else None

    if global_rank == 0:
        print(
            "Model Params: {}".format(
                sum(p.numel() for p in ddp.get_model(model, args.ddp).parameters())
            )
        )

    # ---- learning rate ---------------------------------------------------
    # DrivoR scales the peak LR by sqrt(global_batch / base_batch_size)
    # (drivor_agent.py:415) rather than taking it literally.  Opt in with
    # --drivor_lr_base_batch_size 64 and --learning_rate becomes DrivoR's
    # base_lr; leave it at 0 and --learning_rate is the peak as before.
    if args.drivor_lr_base_batch_size > 0:
        peak_lr = scaled_peak_lr(
            args.learning_rate, args.batch_size, args.drivor_lr_base_batch_size
        )
        if global_rank == 0:
            print(
                "DrivoR sqrt LR scaling: base_lr={:.2e} @ batch {} -> peak {:.3e} @ global batch {}".format(
                    args.learning_rate, args.drivor_lr_base_batch_size, peak_lr, args.batch_size
                )
            )
        args.learning_rate = peak_lr

    optimizer = _build_optimizer(model, args)

    # ``len(train_loader)`` is already the per-rank optimizer-step count, i.e.
    # DrivoR's ``batches_per_epoch`` (drivor_agent.py:441-445) under the same
    # drop_last=True policy.
    steps_per_epoch = len(train_loader)
    max_steps = None
    if args.drivor_lr_schedule == "drivor":
        total_steps = steps_per_epoch * args.train_epochs
        scheduler = build_drivor_scheduler(
            optimizer, total_steps=total_steps, warmup_ratio=args.drivor_warmup_ratio
        )
        step_scheduler = True
        if global_rank == 0:
            ramp = int(total_steps * args.drivor_warmup_ratio)
            print(
                "DrivoR step schedule: {} steps/epoch x {} epochs = {} steps; "
                "linear ramp 0 -> {:.3e} over {} steps, then cosine -> 0".format(
                    steps_per_epoch,
                    args.train_epochs,
                    total_steps,
                    args.learning_rate,
                    ramp,
                )
            )
    elif args.drivor_lr_schedule == "probe":
        # LR range test: sweep geometrically and stop.  Nothing is learned from
        # it beyond the loss-vs-LR curve, so it deliberately never reaches
        # validation.
        max_steps = args.drivor_lr_probe_steps
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
        scheduler = build_lr_probe(
            optimizer,
            steps=max_steps,
            lr_min=args.learning_rate,
            lr_max=args.drivor_lr_probe_max,
        )
        step_scheduler = True
        if global_rank == 0:
            print(
                "LR range test: {:.3e} -> {:.3e} over {} steps (global batch {})".format(
                    args.learning_rate, args.drivor_lr_probe_max, max_steps, args.batch_size
                )
            )
    else:
        scheduler = CosineAnnealingWarmUpRestarts(optimizer, args.train_epochs, args.warm_up_epoch)
        step_scheduler = False

    if args.resume_model_path is not None:
        print(f"Model loaded from {args.resume_model_path}")
        model, optimizer, scheduler, init_epoch, _, model_ema = resume_model(
            args.resume_model_path, model, optimizer, scheduler, model_ema, args.device
        )
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
        print(f"Learning rate reset to {args.learning_rate}")
    else:
        init_epoch = 0

    if global_rank == 0:
        # Self-documenting throughput config: every knob the throughput A/B
        # measured, so a log alone is enough to tell which code path ran.
        print(
            "[throughput] "
            f"global_batch={args.batch_size} "
            f"per_rank={args.batch_size // ddp.get_world_size()} "
            f"amp={args.use_amp}/{args.amp_dtype} "
            f"compile={args.compile_model}/{args.compile_mode} "
            f"ema={args.use_ema}/{'fused' if args.drivor_fused_ema else 'timm'} "
            f"guard_sync_every={args.drivor_guard_sync_every} "
            f"workers={args.num_workers} prefetch={args.prefetch_factor}"
        )

    if args.compile_model:
        if global_rank == 0:
            print(
                f"Compiling model with torch.compile (mode={args.compile_mode}; "
                "first steps will be slow)"
            )
        # In-place compile keeps state_dict keys unchanged, so checkpoint
        # save/resume and the EMA copy stay compatible.
        model.compile(mode=args.compile_mode)

    loss_fn = build_drivor_loss(args)
    oracle = build_drivor_oracle(args)
    guard = DivergenceGuard(
        args.drivor_divergence_guard,
        logit_bound=args.drivor_logit_bound,
        sync_every=args.drivor_guard_sync_every,
    )
    # A step-interval scheduler rewrites param_groups["lr"] every iteration, so
    # the guard has to cut ``base_lrs`` for its halving to survive.
    guard.attach_scheduler(scheduler if step_scheduler else None)
    scaler = (
        torch.amp.GradScaler("cuda")
        if args.use_amp and str(args.amp_dtype).lower() in ("fp16", "float16")
        else None
    )

    wandb_run = None
    if global_rank == 0:
        os.environ["WANDB_MODE"] = "online" if args.use_wandb else "offline"
        project, entity = _wandb_target(args)
        print(f"wandb: {entity or '<default>'}/{project} ({os.environ['WANDB_MODE']})")
        wandb.init(
            project=project,
            entity=entity,
            name=args.exp_name,
            notes=args.notes,
            resume="allow",
            id=args.wandb_run_id,
            dir=f"{save_path}",
        )
        wandb.config.update(args_dict)
        wandb_run = wandb.run
        # The devkit's own score panel, used exactly as the devkit defines it.
        define_wandb_score_metrics(wandb_run, prefix="devkit", step_metric="epoch")
        # Epoch aggregates plot against the epoch; the live perf/* series keep the
        # default per-step x-axis.
        for pattern in ("train/*", "val/*", "lr/*"):
            wandb_run.define_metric(pattern, step_metric="epoch")

    if args.ddp:
        torch.distributed.barrier()

    data_list = []
    best_score = float("-inf")
    global_step = 0

    for epoch in range(init_epoch, args.train_epochs):
        if args.ddp:
            torch.distributed.barrier()

        # The repo's diffusion path hard-writes a 0.1x / 0.01x taper over the
        # last 10 epochs.  DrivoR's cosine already anneals to eta_min=0 across
        # the whole run, and this would overwrite it every epoch, so it only
        # applies to the legacy per-epoch schedule.
        final_epoch_count = 10
        if not step_scheduler and epoch >= args.train_epochs - final_epoch_count:
            factor = 0.01 if epoch >= args.train_epochs - final_epoch_count // 2 else 0.1
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate * factor
            if global_rank == 0:
                print(
                    f"Final phase: Epoch {epoch + 1}, LR adjusted to {args.learning_rate * factor}"
                )

        train_metrics, train_total_loss, global_step = train_epoch_drivor(
            train_loader,
            model,
            optimizer,
            args,
            model_ema,
            aug,
            loss_fn=loss_fn,
            oracle=oracle,
            guard=guard,
            scaler=scaler,
            wandb_run=wandb_run,
            global_step=global_step,
            scheduler=scheduler if step_scheduler else None,
            max_steps=max_steps,
        )

        if max_steps is not None and global_step >= max_steps:
            if global_rank == 0:
                print("LR range test complete.", flush=True)
                if wandb.run is not None:
                    wandb.finish()
            return

        valid_metrics, panel = validate_drivor(
            valid_loader, model, args, loss_fn=loss_fn, oracle=oracle
        )

        if global_rank == 0:
            selected = valid_metrics.get(CHECKPOINT_METRIC, float("nan"))
            print(
                "Epoch {}/{}  val ADE={:.3f} FDE={:.3f} oracle_selected={:.4f} "
                "oracle_best={:.4f} top1={:.3f}".format(
                    epoch + 1,
                    args.train_epochs,
                    valid_metrics.get("val/trajectory/selected_ADE", float("nan")),
                    valid_metrics.get("val/trajectory/selected_FDE", float("nan")),
                    selected,
                    valid_metrics.get("val/selection/oracle_best", float("nan")),
                    valid_metrics.get("val/selection/top1_hit", float("nan")),
                )
            )
            print(
                "devkit panel: PDMS={:.4f} NC={:.4f} DAC={:.4f} DDC={:.4f} TTC={:.4f} "
                "EP={:.4f} Comfort={:.4f}".format(
                    panel.get("score", float("nan")),
                    panel.get("no_at_fault_collisions", float("nan")),
                    panel.get("drivable_area_compliance", float("nan")),
                    panel.get("driving_direction_compliance", float("nan")),
                    panel.get("time_to_collision_within_bound", float("nan")),
                    panel.get("ego_progress", float("nan")),
                    panel.get("history_comfort", float("nan")),
                )
            )

            # Everything shares one monotone wandb step (the global training
            # step); ``epoch`` rides along as the x-axis for the epoch series,
            # which is what ``define_wandb_score_metrics`` declares it as. Logging
            # epoch aggregates at step=epoch instead would move the step counter
            # backwards and wandb would silently drop them.
            wandb.log(
                {
                    **train_metrics,
                    **valid_metrics,
                    "lr/lr": optimizer.param_groups[0]["lr"],
                    "epoch": epoch + 1,
                },
                step=global_step,
            )
            # The devkit panel goes through the devkit's own logger, unchanged.
            score_report_to_wandb(
                panel, prefix="devkit", epoch=epoch + 1, step=global_step, run=wandb_run
            )

            curr_data = {
                "epoch": epoch + 1,
                "train_loss": train_total_loss,
                **{key.replace("/", "_"): value for key, value in valid_metrics.items()},
                **{f"devkit_{key}": value for key, value in panel.items()},
            }
            data_list.append(curr_data)
            pd.DataFrame(data_list).to_csv(
                os.path.join(save_path, "train_log.tsv"), index=False, sep="\t"
            )

            model_dict = {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "ema_state_dict": model_ema.ema.state_dict() if model_ema is not None else None,
                "optimizer": optimizer.state_dict(),
                "schedule": scheduler.state_dict(),
                "loss": valid_metrics.get("val/loss/total", float("nan")),
                "wandb_id": None,
            }
            torch.save(model_dict, f"{save_path}/latest.pth")

            if (epoch + 1 - init_epoch) % args.save_utd == 0:
                curr_dir = os.path.join(save_path, f"epoch{epoch + 1:04d}")
                os.makedirs(curr_dir, exist_ok=True)
                torch.save(model_dict, f"{curr_dir}/best_model.pth")
                with open(os.path.join(curr_dir, "best_model_info.json"), "w") as handle:
                    json.dump(curr_data, handle, indent=4)
                with open(os.path.join(curr_dir, "args.json"), "w", encoding="utf-8") as handle:
                    json.dump(args_dict, handle, indent=4)

            # Selection quality, not a regression loss, is what this head is
            # trained for, so the best checkpoint tracks the oracle score of the
            # selected trajectory.
            if selected == selected and selected > best_score:
                best_score = selected
                curr_dir = os.path.join(save_path, "best_model")
                os.makedirs(curr_dir, exist_ok=True)
                torch.save(model_dict, f"{curr_dir}/best_model.pth")
                curr_data["best_score"] = best_score
                with open(os.path.join(curr_dir, "best_model_info.json"), "w") as handle:
                    json.dump(curr_data, handle, indent=4)
                with open(os.path.join(curr_dir, "args.json"), "w", encoding="utf-8") as handle:
                    json.dump(args_dict, handle, indent=4)

        if not step_scheduler:
            scheduler.step()
        train_sampler.set_epoch(epoch + 1)

    if global_rank == 0 and wandb.run is not None:
        wandb.finish()


__all__ = ["model_training_drivor"]
