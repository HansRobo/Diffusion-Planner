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


# ---------------------------------------------------------------------------
# The hybrid norm: inherited from the frozen IL base, not pinned to the paper
# ---------------------------------------------------------------------------
#
# Eq. (awr_hybrid) weights the *hybrid* distance, so omega and W really do
# belong to the RL objective and not only to imitation pretraining. That is
# precisely why they are not pinned to Table 3 here. This pipeline does not
# retrain IL, and omega/W define the geometry of the norm the frozen base was
# fitted under: post-training a base fitted at omega=0.01 / W=10 under
# omega=0.1 / W=L-1 rescales the waypoint term 10x and widens the detach
# window, so every RL gradient would be measured in a norm the policy has
# never been optimized for.
#
# Paper-exact therefore inherits both fields from the checkpoint the run
# starts from, and refuses a conflicting explicit value. Which pair is better
# *for RL* is an empirical question, and ``--rl_hybrid_ablation`` is how it
# gets asked: it releases the two fields and stamps the run as an ablation
# arm, so a sweep reads as a sweep in args.json rather than as a paper-exact
# run that quietly disagrees with its own base.

HYBRID_FIELDS: tuple[str, ...] = ("planning_hybrid_loss", "hybrid_loss_window")

_PAPER_HYBRID_SOURCES: Mapping[str, str] = {
    "planning_hybrid_loss": "neurips_2026.tex tab:param -- Hybrid loss weight omega = 0.1",
    "hybrid_loss_window": ("neurips_2026.tex Appendix Hybrid Loss -- 'In practice, we set W=L-1'"),
}


def paper_hybrid_values(args: Any) -> dict[str, Any]:
    """What Table 3 and the Hybrid Loss appendix publish, for the run log.

    Reported alongside the inherited values so a paper-exact run states plainly
    where it departs from the paper and why.
    """
    return {
        "planning_hybrid_loss": 0.1,
        "hybrid_loss_window": _horizon_minus_one(args),
    }


def apply_frozen_base_hybrid(
    args: Any,
    base_args: Mapping[str, Any] | None,
    base_label: str,
    explicit_fields: Iterable[str],
    *,
    ablation: bool = False,
) -> list[str]:
    """Inherit the hybrid norm from the frozen IL base this run starts from.

    ``base_args`` is the ``args.json`` written beside that checkpoint. Returns
    one human-readable line per inherited field, for the run log and args.json.
    Raises when an explicit flag contradicts the base and ``ablation`` is off.
    """
    if ablation:
        published = paper_hybrid_values(args)
        return [
            "hybrid norm released for ablation: "
            + ", ".join(f"{f}={getattr(args, f)!r}" for f in HYBRID_FIELDS)
            + " (frozen base "
            + (
                ", ".join(f"{f}={base_args.get(f, '?')!r}" for f in HYBRID_FIELDS)
                if base_args is not None
                else "unknown"
            )
            + "; paper "
            + ", ".join(f"{f}={published[f]!r}" for f in HYBRID_FIELDS)
            + ")"
        ]

    if base_args is None:
        raise ValueError(
            "--rl_paper_exact inherits the hybrid norm (omega, W) from the checkpoint the run "
            "starts from, but no args.json was found beside it. Point --init_weights_path at a "
            "run directory that contains args.json, or pass --rl_hybrid_ablation True to set "
            "omega and W deliberately as an ablation arm."
        )

    explicit = set(explicit_fields)
    published = paper_hybrid_values(args)
    conflicts: list[str] = []
    changes: list[str] = []
    for field in HYBRID_FIELDS:
        if field not in base_args:
            raise ValueError(
                f"{base_label} does not record {field!r}, so the frozen IL base's hybrid norm "
                "cannot be reproduced. Pass --rl_hybrid_ablation True to set it deliberately."
            )
        base_value = base_args[field]
        current = getattr(args, field)
        if field in explicit and current != base_value:
            conflicts.append(
                f"  --{field} {current!r} != {base_value!r} in {base_label}\n"
                f"      [frozen IL base; the paper publishes {published[field]!r} "
                f"-- {_PAPER_HYBRID_SOURCES[field]}]"
            )
            continue
        # Both branches carry the same citation, including the published value
        # this run does not use: a departure from the paper has to be readable
        # in the run log whether or not the launcher happened to pass it already.
        provenance = (
            f"\n      [inherited from the frozen IL base {base_label}; "
            f"the paper publishes {published[field]!r} -- {_PAPER_HYBRID_SOURCES[field]}]"
        )
        if current != base_value:
            changes.append(f"{field}: {current!r} -> {base_value!r}{provenance}")
            setattr(args, field, base_value)
        else:
            changes.append(f"{field}: {base_value!r} (unchanged){provenance}")
    if conflicts:
        raise ValueError(
            "--rl_paper_exact keeps the hybrid norm of the IL base it post-trains, because this "
            "pipeline does not retrain IL. These explicit overrides contradict that base:\n"
            + "\n".join(conflicts)
            + "\n  Drop them, or pass --rl_hybrid_ablation True to run this as a deliberate "
            "omega/W ablation arm."
        )
    return changes


# ---------------------------------------------------------------------------
# The corpus: identical to the run that produced the frozen IL base
# ---------------------------------------------------------------------------
#
# RL post-training moves the same policy on the same distribution; a different
# corpus or a different perturbation makes the RL delta unattributable, because
# any change could be the objective or could be the data. These fields are
# therefore compared against the base run's args.json rather than trusted to
# match by convention.
#
# The extra manifests are compared by basename: they are addressed as
# ``${REPO}/artifacts/...``, so the same file has a different absolute path in
# each checkout. Contents are verified separately by the launcher's input
# fingerprint.

_BASE_DATA_FIELDS: tuple[str, ...] = (
    "train_set_list",
    "valid_set_list",
    "extra_train_set_repeat",
    "filter_skipped",
    "train_subsample_step",
    "align_legacy_neighbor_futures",
)

_BASE_AUGMENT_FIELDS: tuple[str, ...] = (
    "use_data_augment",
    "augment_prob",
    "augment_type",
    "num_refine",
    "ego_past_noise_std",
    "use_smoothing_future_trajectory",
)


def _basenames(paths: Any) -> list[str]:
    if paths is None:
        return []
    if isinstance(paths, str):
        paths = [paths]
    return sorted(str(p).rsplit("/", 1)[-1] for p in paths)


def base_corpus_mismatches(args: Any, base_args: Mapping[str, Any]) -> list[str]:
    """Every way this run's corpus or perturbation differs from the IL base's."""
    mismatches: list[str] = []
    for field in _BASE_DATA_FIELDS + _BASE_AUGMENT_FIELDS:
        if field not in base_args:
            mismatches.append(f"  {field}: the base args.json does not record it")
            continue
        current = getattr(args, field, None)
        base_value = base_args[field]
        if current != base_value:
            mismatches.append(f"  --{field} {current!r} != {base_value!r} in the IL base")
    current_extra = _basenames(getattr(args, "extra_train_set_list", None))
    base_extra = _basenames(base_args.get("extra_train_set_list"))
    if current_extra != base_extra:
        mismatches.append(
            f"  --extra_train_set_list {current_extra} != {base_extra} in the IL base"
        )
    # The normalizer is compared by resolved content, not by path: the same file
    # lives at a different --normalization_file_path in each checkout.
    for field in ("observation_normalizer", "state_normalizer"):
        if field not in base_args:
            continue
        current = getattr(args, field, None)
        current = current if isinstance(current, dict) else getattr(current, "config", None)
        if current is not None and current != base_args[field]:
            mismatches.append(f"  {field} differs from the IL base's resolved normalizer")
    return mismatches


def assert_base_corpus_identical(args: Any, base_args: Mapping[str, Any], base_label: str) -> None:
    """Refuse to post-train on a corpus the frozen IL base was not fitted on."""
    mismatches = base_corpus_mismatches(args, base_args)
    if mismatches:
        raise ValueError(
            "--rl_paper_exact post-trains a frozen IL base, so the corpus and the input "
            f"perturbation must be the ones that base was trained on ({base_label}). "
            "These differ:\n"
            + "\n".join(mismatches)
            + "\n  Match the base run's launcher, or pass --rl_base_corpus_check False to "
            "post-train on a deliberately different distribution."
        )


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
