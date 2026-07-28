# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Registry and per-batch force-mask dispatch for augmentation on repeat draws."""

# Canonical application order. NOT user-configurable: "state_perturbation" must run
# last because it terminates with centric_transform(), which every other augmentation
# assumes has not yet happened. This order reproduces the sequence the individual
# augmentation PRs established in train_epoch.
AUG_ORDER: tuple[str, ...] = (
    "flip",
    "neighbor_dropout",
    "neighbor_noise",
    "turn_indicator",
    "traffic_light",
    "state_perturbation",
)

# Registry name -> the CLI flag that enables it, for error messages that tell the
# operator what to actually type.
_USE_FLAG: dict[str, str] = {
    "flip": "--use_flip_augment",
    "neighbor_dropout": "--use_neighbor_dropout",
    "neighbor_noise": "--use_neighbor_noise",
    "turn_indicator": "--use_turn_indicator_dropout",
    "traffic_light": "--use_traffic_light_dropout",
    "state_perturbation": "--use_data_augment",
}


def build_aug_pipeline(augmentations: dict) -> list[tuple[str, object]]:
    """Order enabled augmentations canonically.

    ``augmentations`` maps registry name -> augmentation object or None. Entries that
    are None are treated as disabled and omitted. Returns ``[(name, aug), ...]`` in
    AUG_ORDER.
    """
    unknown = sorted(set(augmentations) - set(AUG_ORDER))
    if unknown:
        raise ValueError(
            f"unknown augmentation name(s) {unknown}; valid names are {list(AUG_ORDER)}"
        )
    return [
        (name, augmentations[name]) for name in AUG_ORDER if augmentations.get(name) is not None
    ]


def resolve_repeat_aug_pool(
    pool_spec: str,
    pipeline: list[tuple[str, object]],
    augment_type: str,
) -> tuple[list[str], list[str]]:
    """Resolve --repeat_aug_pool into concrete registry names.

    Returns ``(pool_names, warnings)``. ``pool_names`` is in AUG_ORDER. Raises
    ValueError for anything that would make the run silently do the wrong thing --
    these are surfaced at launch, before the dataset scan.
    """
    enabled = [name for name, _ in pipeline]
    if not enabled:
        raise ValueError(
            "--force_aug_on_repeat was requested but no augmentations are enabled; "
            "enable at least one (e.g. --use_data_augment True)"
        )

    warnings: list[str] = []
    requested = [part.strip() for part in pool_spec.split(",")]
    requested = [part for part in requested if part]

    if requested:
        unknown = [name for name in requested if name not in AUG_ORDER]
        if unknown:
            raise ValueError(
                f"--repeat_aug_pool names unknown augmentation(s) {unknown}; "
                f"valid names are {list(AUG_ORDER)}"
            )
        not_enabled = [name for name in requested if name not in enabled]
        if not_enabled:
            flags = [_USE_FLAG[name] for name in not_enabled]
            raise ValueError(
                f"--repeat_aug_pool names {not_enabled}, which are not enabled. "
                f"Enable them with {flags} or drop them from the pool."
            )
        pool = [name for name in AUG_ORDER if name in set(requested)]
        if augment_type == "bridge" and "state_perturbation" in pool:
            warnings.append(
                "--repeat_aug_pool names state_perturbation while --augment_type bridge. "
                "Bridge forcing is NOT cost-neutral: its augment() searches per flagged row "
                "at roughly 912 ms/sample, so forcing adds proportional wall-time."
            )
    else:
        pool = list(enabled)
        if augment_type == "bridge" and "state_perturbation" in pool:
            # Excluded by default rather than silently doubling an already-prohibitive
            # cost. Nameable explicitly (above) for anyone who wants it anyway.
            pool = [name for name in pool if name != "state_perturbation"]
        if not pool:
            raise ValueError(
                "--repeat_aug_pool resolved to an empty pool. state_perturbation is "
                "excluded from the default pool under --augment_type bridge, and it is "
                "the only enabled augmentation. Either enable another augmentation, "
                "switch to --augment_type quintic, or name it explicitly with "
                "--repeat_aug_pool state_perturbation."
            )

    if pool == ["flip"]:
        warnings.append(
            "--repeat_aug_pool resolves to flip alone. Forced flip is deterministic, so "
            "two forced occurrences of the same sample mirror identically and remain "
            "duplicates of each other. Add another augmentation to the pool."
        )

    return pool, warnings
