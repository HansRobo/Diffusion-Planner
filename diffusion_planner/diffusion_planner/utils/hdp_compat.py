def has_velocity_normalizer(state_normalizer) -> bool:
    return (
        isinstance(state_normalizer, dict)
        and "ego_velocity_mean" in state_normalizer
        and "ego_velocity_std" in state_normalizer
    )


def require_velocity_normalizer(state_normalizer, source: str) -> None:
    if has_velocity_normalizer(state_normalizer):
        return
    raise RuntimeError(
        "This HDP velocity checkpoint was saved without independent ego_velocity "
        f"normalization stats in {source}. Refusing to load a legacy fused-scaling "
        "velocity checkpoint."
    )


def require_waypoint_diffusion(model_args, caller: str) -> None:
    if getattr(model_args, "use_velocity_representation", False):
        raise NotImplementedError(
            f"{caller} is not HDP velocity-aware. It builds waypoint-space diffusion "
            "targets and losses by hand, so using a velocity-latent checkpoint would "
            "silently optimize the wrong objective. Use the supervised HDP trainer or "
            "add a velocity-aware implementation for this path."
        )
