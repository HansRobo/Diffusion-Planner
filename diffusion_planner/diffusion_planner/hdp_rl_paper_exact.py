"""Paper-exact HDP-RL configuration.

``--rl_paper_exact`` pins every knob that shapes the RL objective to the value
published in the Hyper Diffusion Planner paper, so a run can be certified as
reproducing Eq. (14)-(17) and Table 3 rather than this repository's real-vehicle
extensions.

Authoritative sources, both vendored under ``reference/``:

- ``reference/papers/hyper_diffusion_planner_paper/src/neurips_2026.tex``
  -- Section "KL-Regularized RL" / "RL-Hybrid Loss" (Eq. ``eq:awr``,
  ``eq:awr_hybrid``), Appendix ``app:rewards`` (reward definitions and the total
  training reward), and Table ``tab:param`` (hyperparameters).
- ``reference/papers/hyper_diffusion_planner_paper/src/code_rl.tex``
  -- Algorithm 2, ``rl_hybrid_loss``.
- ``reference/external/Hyper-Diffusion-Planner/HDP-navsim/hdp_navsim/agent/
  dp_vla/dp_vla_rl_agent.py`` -- the authors' released implementation
  (``_rl_rollout``, ``_rl_train_step``, ``compute_loss``,
  ``on_train_epoch_start``) plus its config
  ``hdp_navsim/config/agent/dp_vla_rl_agent.yaml``.

``docs/hdp_rl_paper_fidelity.md`` records the full line-by-line audit behind the
table below, including the three places where the paper text and the released
code contradict each other and the one mechanism that cannot be reproduced
verbatim in this action space.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

PAPER_REWARD_VARIANTS = ("multi", "single")


@dataclass(frozen=True)
class PaperExactSetting:
    """One pinned field, with the citation that fixes its value.

    ``value`` is the published constant, or a callable ``args -> value`` for the
    handful of settings the paper states relative to the planning horizon L.
    """

    field: str
    value: Any
    source: str


def _horizon_minus_one(args: Any) -> int:
    """W = L - 1 for this run's planning horizon L."""
    return max(1, int(args.future_len) - 1)


def resolve_paper_exact_value(setting: PaperExactSetting, args: Any) -> Any:
    """The published value of ``setting`` for this run's horizon."""
    if callable(setting.value):
        return setting.value(args)
    return setting.value


# Fields shared by both published reward settings.
_COMMON: tuple[PaperExactSetting, ...] = (
    # ---- Table 3 (tab:param), RL block -------------------------------------
    PaperExactSetting(
        "num_generations",
        32,
        "neurips_2026.tex tab:param -- RL / Group size = 32",
    ),
    PaperExactSetting(
        "rl_reward_beta",
        1.0,
        "neurips_2026.tex tab:param -- RL / Temperature beta = 1.0; "
        "matches dp_vla_rl_agent.py torch.exp(rewards) with no explicit beta",
    ),
    PaperExactSetting(
        "rl_ema_update_rate",
        0.05,
        "neurips_2026.tex tab:param -- RL / EMA = 0.05; Sec. RL 'we employ "
        "Exponential Moving Average (EMA) for policy updates'",
    ),
    # ---- Eq. (awr_hybrid) + Algorithm 2 + _rl_train_step -------------------
    PaperExactSetting(
        "rl_reward_normalize",
        "group",
        "neurips_2026.tex Sec. RL 'we apply reward group normalization'; "
        "dp_vla_rl_agent.py _rl_train_step group-relative normalisation",
    ),
    PaperExactSetting(
        "advantage_eps",
        1e-6,
        "code_rl.tex Algorithm 2 r.std() + 1e-6; dp_vla_rl_agent.py "
        "reward_abs.std(dim=1, keepdim=True) + 1e-6",
    ),
    PaperExactSetting(
        "rl_bc_weight",
        0.0,
        "neurips_2026.tex eq:awr_hybrid -- the RL loss is the weighted "
        "regression alone; no expert-anchor term",
    ),
    PaperExactSetting(
        "rl_diffusion_t_min",
        0.0,
        "neurips_2026.tex eq:awr_hybrid -- expectation over the full t range",
    ),
    PaperExactSetting(
        "rl_diffusion_t_max",
        1.0,
        "neurips_2026.tex eq:awr_hybrid -- expectation over the full t range",
    ),
    # ---- The hybrid norm inside eq:awr_hybrid ------------------------------
    # Eq. (awr_hybrid) weights the *hybrid* distance, so omega and W are part of
    # the RL objective, not only of imitation pretraining. Both published
    # implementations disagree with Table 3 here; see docs/hdp_rl_paper_fidelity.md.
    PaperExactSetting(
        "planning_hybrid_loss",
        0.1,
        "neurips_2026.tex tab:param -- Hybrid loss weight omega = 0.1; "
        "code_rl.tex Algorithm 2 forwards omega into hybrid_loss()",
    ),
    PaperExactSetting(
        "hybrid_loss_window",
        _horizon_minus_one,
        "neurips_2026.tex Appendix Hybrid Loss -- 'In practice, we set W=L-1'",
    ),
    # ---- Appendix app:rewards, "Total Training Reward" ---------------------
    PaperExactSetting(
        "rl_reward_source",
        "native",
        "neurips_2026.tex app:rewards -- the published reward set, not the "
        "ported original-DP PDM objective",
    ),
    PaperExactSetting(
        "rl_reward_aggregation",
        "weighted_sum",
        "neurips_2026.tex app:rewards -- r is a weighted sum of the components",
    ),
    PaperExactSetting(
        "rl_behavior_gate",
        "none",
        "neurips_2026.tex app:rewards -- the published sum applies no gate to "
        "the car-following / lane-keeping terms",
    ),
    PaperExactSetting(
        "rl_reward_w_progress",
        0.0,
        "neurips_2026.tex app:rewards -- the published reward has no progress "
        "term (real-vehicle extension of this repository)",
    ),
    PaperExactSetting(
        "rl_reward_w_road_border",
        0.0,
        "neurips_2026.tex app:rewards -- the published reward has no "
        "road-border term (real-vehicle extension of this repository)",
    ),
    PaperExactSetting(
        "rl_red_light_constraint",
        False,
        "neurips_2026.tex app:rewards -- r_safety and r_risk carry no "
        "traffic-light factor; this repository folds one into both",
    ),
    PaperExactSetting(
        "rl_occupancy_use_road_border",
        False,
        "neurips_2026.tex app:rewards -- OCC is 'occupancy distance to "
        "static/uncertain regions', not HD-map border geometry",
    ),
    PaperExactSetting(
        "rl_reward_horizon_steps",
        0,
        "neurips_2026.tex app:rewards -- every reward is 'evaluated on the "
        "candidate trajectory over the planning horizon of L steps'",
    ),
    PaperExactSetting(
        "rl_candidate_loss_horizon",
        0,
        "neurips_2026.tex eq:awr_hybrid -- the regression covers the whole action tau^v_0",
    ),
    # ---- Released implementation: rollout / replay schedule ----------------
    PaperExactSetting(
        "rl_rollout_steps",
        5,
        "dp_vla_rl_agent.yaml rl_config.rollout_steps = 5",
    ),
    PaperExactSetting(
        "rl_rollout_interval",
        10,
        "dp_vla_rl_agent.yaml rl_config.replay_buffer_update_epoch = 10; "
        "dp_vla_rl_agent.py compute_loss / on_train_epoch_start epoch state "
        "machine",
    ),
    PaperExactSetting(
        "rl_updates_per_rollout",
        1,
        "dp_vla_rl_agent.yaml rl_config.diffusion_repeat_size = 1 -- one noising per replay draw",
    ),
    PaperExactSetting(
        "rl_noise_scale",
        1.0,
        "dp_vla_rl_agent.py _rl_rollout -- model.generate() samples the prior at unit temperature",
    ),
    PaperExactSetting(
        "rl_init_use_ema",
        True,
        "neurips_2026.tex Sec. RL -- pi^0 is the imitation model pretrained with the hybrid loss",
    ),
    # ---- Mechanisms absent from both sources -------------------------------
    PaperExactSetting(
        "rl_candidate_aug_prob",
        0.0,
        "not in neurips_2026.tex; the released augment_trajectory_batch adds a "
        "constant waypoint offset, which is a first-step impulse under this "
        "repository's velocity actions -- see docs/hdp_rl_paper_fidelity.md",
    ),
    PaperExactSetting(
        "rl_first_waypoint_gate",
        False,
        "not in neurips_2026.tex nor dp_vla_rl_agent.py -- real-vehicle "
        "candidate filter of this repository",
    ),
)

# Appendix app:rewards, "Total Training Reward":
#   r = r_safety                                                (single-reward)
#   r = lambda_risk r_risk + lambda_follow r_follow
#       + lambda_lane r_lane                                     (multi-reward)
_REWARD_VARIANTS: Mapping[str, tuple[PaperExactSetting, ...]] = {
    "multi": (
        PaperExactSetting(
            "rl_reward_w_safety",
            0.0,
            "neurips_2026.tex app:rewards -- the multi-reward setting replaces "
            "r_safety with r_risk",
        ),
        PaperExactSetting(
            "rl_reward_w_risk",
            1.0,
            "neurips_2026.tex tab:param -- RL / lambda_risk = 1.0",
        ),
        PaperExactSetting(
            "rl_reward_w_follow",
            3.0,
            "neurips_2026.tex tab:param -- RL / lambda_follow = 3.0",
        ),
        PaperExactSetting(
            "rl_reward_w_lane",
            2.5,
            "neurips_2026.tex tab:param -- RL / lambda_lane = 2.5",
        ),
    ),
    "single": (
        PaperExactSetting(
            "rl_reward_w_safety",
            1.0,
            "neurips_2026.tex app:rewards -- 'The single-reward baseline uses "
            "r_safety alone' (HDP-RL dagger)",
        ),
        PaperExactSetting(
            "rl_reward_w_risk",
            0.0,
            "neurips_2026.tex app:rewards -- single-reward baseline",
        ),
        PaperExactSetting(
            "rl_reward_w_follow",
            0.0,
            "neurips_2026.tex app:rewards -- single-reward baseline",
        ),
        PaperExactSetting(
            "rl_reward_w_lane",
            0.0,
            "neurips_2026.tex app:rewards -- single-reward baseline",
        ),
    ),
}


def paper_exact_settings(reward_variant: str) -> tuple[PaperExactSetting, ...]:
    """Every field pinned by ``--rl_paper_exact`` for one reward variant."""
    if reward_variant not in _REWARD_VARIANTS:
        raise ValueError(
            f"Unsupported rl_paper_reward={reward_variant!r}; "
            f"expected one of {PAPER_REWARD_VARIANTS}"
        )
    return _COMMON + _REWARD_VARIANTS[reward_variant]


def paper_exact_values(reward_variant: str, args: Any = None) -> dict[str, Any]:
    """``field -> published value`` mapping for one reward variant.

    ``args`` supplies the planning horizon for the settings the paper states
    relative to L, and is required whenever the variant contains one.
    """
    values: dict[str, Any] = {}
    for setting in paper_exact_settings(reward_variant):
        if callable(setting.value) and args is None:
            raise ValueError(
                f"{setting.field!r} is published relative to the planning horizon L; "
                "pass args to resolve it"
            )
        values[setting.field] = resolve_paper_exact_value(setting, args)
    return values


def paper_exact_fields() -> tuple[str, ...]:
    """Every field this mode controls, across all reward variants."""
    fields: list[str] = [setting.field for setting in _COMMON]
    for variant in PAPER_REWARD_VARIANTS:
        for setting in _REWARD_VARIANTS[variant]:
            if setting.field not in fields:
                fields.append(setting.field)
    return tuple(fields)


def _same(current: Any, target: Any) -> bool:
    if isinstance(target, bool) or isinstance(current, bool):
        return bool(current) is bool(target)
    if isinstance(target, float):
        try:
            return math.isclose(float(current), target, rel_tol=0.0, abs_tol=0.0)
        except (TypeError, ValueError):
            return False
    return current == target


def explicit_paper_exact_dests(
    parser: argparse.ArgumentParser,
    argv: list[str] | None,
    fields: Iterable[str] | None = None,
) -> set[str]:
    """Which paper-exact fields the caller passed on the command line.

    argparse only writes a default when the destination is absent from the
    namespace, so pre-seeding every controlled destination with a sentinel makes
    explicitly-passed options -- in any spelling, including ``--flag=value`` and
    the ``--official_*`` aliases -- the only ones that get overwritten.
    """
    controlled = tuple(fields) if fields is not None else paper_exact_fields()
    sentinel = object()
    namespace = argparse.Namespace(**dict.fromkeys(controlled, sentinel))
    parsed = parser.parse_args(argv, namespace=namespace)
    return {name for name in controlled if getattr(parsed, name) is not sentinel}


def apply_paper_exact_settings(
    args,
    reward_variant: str,
    explicit_fields: Iterable[str] = (),
) -> list[str]:
    """Pin ``args`` to the published RL configuration.

    Returns one human-readable line per field that had to change, for the run
    log. Raises when the caller explicitly asked for a value the paper
    contradicts: silently overriding an explicit flag would make the run log
    lie about what was trained.
    """
    settings = paper_exact_settings(reward_variant)
    explicit = set(explicit_fields)

    conflicts = [
        f"--{setting.field}={getattr(args, setting.field)!r} conflicts with the "
        f"published {resolve_paper_exact_value(setting, args)!r} ({setting.source})"
        for setting in settings
        if setting.field in explicit
        and not _same(getattr(args, setting.field), resolve_paper_exact_value(setting, args))
    ]
    if conflicts:
        raise ValueError(
            "--rl_paper_exact cannot be combined with these explicit overrides:\n  "
            + "\n  ".join(conflicts)
            + "\nDrop the flags, or drop --rl_paper_exact and document the deviation."
        )

    changed: list[str] = []
    for setting in settings:
        if not hasattr(args, setting.field):
            raise AttributeError(
                f"--rl_paper_exact controls unknown field {setting.field!r}; "
                "the pinned table and the RL trainer have drifted apart"
            )
        current = getattr(args, setting.field)
        target = resolve_paper_exact_value(setting, args)
        if _same(current, target):
            continue
        setattr(args, setting.field, target)
        changed.append(f"{setting.field}: {current!r} -> {target!r}  [{setting.source}]")

    if getattr(args, "rl_rollout_interval", 0) > 0 and not getattr(args, "rl_replay_dir", None):
        raise ValueError(
            "--rl_paper_exact reproduces the released mine/replay epoch state "
            "machine (replay_buffer_update_epoch = 10) and therefore requires "
            "--rl_replay_dir"
        )
    return changed


def assert_paper_exact(args, reward_variant: str | None = None) -> None:
    """Raise unless ``args`` still matches the published configuration."""
    variant = reward_variant if reward_variant is not None else args.rl_paper_reward
    deviations = [
        f"{setting.field}={getattr(args, setting.field, None)!r} != "
        f"{resolve_paper_exact_value(setting, args)!r} ({setting.source})"
        for setting in paper_exact_settings(variant)
        if not _same(getattr(args, setting.field, None), resolve_paper_exact_value(setting, args))
    ]
    if deviations:
        raise ValueError("Run is not paper-exact:\n  " + "\n  ".join(deviations))
