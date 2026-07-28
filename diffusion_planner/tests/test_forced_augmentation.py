import pytest
from diffusion_planner.utils.forced_augmentation import (
    AUG_ORDER,
    build_aug_pipeline,
    resolve_repeat_aug_pool,
)


class _FakeAug:
    """Stand-in for an augmentation object; identity is all these tests need."""

    def __init__(self, name):
        self.name = name


class TestAugOrder:
    def test_state_perturbation_is_last(self):
        """It terminates with centric_transform, which the others assume has not run."""
        assert AUG_ORDER[-1] == "state_perturbation"

    def test_order_matches_todays_train_epoch_sequence(self):
        assert AUG_ORDER == (
            "flip",
            "neighbor_dropout",
            "neighbor_noise",
            "turn_indicator",
            "traffic_light",
            "state_perturbation",
        )


class TestBuildAugPipeline:
    def test_orders_canonically_regardless_of_dict_order(self):
        augs = {
            "state_perturbation": _FakeAug("sp"),
            "flip": _FakeAug("flip"),
            "neighbor_noise": _FakeAug("nn"),
        }
        pipeline = build_aug_pipeline(augs)
        assert [name for name, _ in pipeline] == ["flip", "neighbor_noise", "state_perturbation"]

    def test_skips_none_entries(self):
        augs = {"flip": None, "state_perturbation": _FakeAug("sp")}
        pipeline = build_aug_pipeline(augs)
        assert [name for name, _ in pipeline] == ["state_perturbation"]

    def test_empty_dict_yields_empty_pipeline(self):
        assert build_aug_pipeline({}) == []

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="unknown augmentation"):
            build_aug_pipeline({"route_input": _FakeAug("r")})


class TestResolveRepeatAugPool:
    def _pipeline(self, *names):
        return [(n, _FakeAug(n)) for n in AUG_ORDER if n in names]

    def test_empty_spec_defaults_to_all_enabled(self):
        pipeline = self._pipeline("flip", "neighbor_noise")
        pool, warnings = resolve_repeat_aug_pool("", pipeline, augment_type="quintic")
        assert pool == ["flip", "neighbor_noise"]
        assert warnings == []

    def test_explicit_spec_restricts_pool(self):
        pipeline = self._pipeline("flip", "neighbor_noise", "state_perturbation")
        pool, _ = resolve_repeat_aug_pool("neighbor_noise", pipeline, augment_type="quintic")
        assert pool == ["neighbor_noise"]

    def test_explicit_spec_is_reordered_canonically(self):
        pipeline = self._pipeline("flip", "neighbor_noise")
        pool, _ = resolve_repeat_aug_pool("neighbor_noise,flip", pipeline, augment_type="quintic")
        assert pool == ["flip", "neighbor_noise"]

    def test_whitespace_and_empty_entries_tolerated(self):
        pipeline = self._pipeline("flip", "neighbor_noise")
        pool, _ = resolve_repeat_aug_pool(" flip , ,neighbor_noise ", pipeline, "quintic")
        assert pool == ["flip", "neighbor_noise"]

    def test_unknown_name_raises_listing_valid_names(self):
        pipeline = self._pipeline("flip")
        with pytest.raises(ValueError) as exc:
            resolve_repeat_aug_pool("nonsense", pipeline, augment_type="quintic")
        assert "nonsense" in str(exc.value)
        assert "flip" in str(exc.value)

    def test_name_not_enabled_raises_naming_the_use_flag(self):
        pipeline = self._pipeline("flip")
        with pytest.raises(ValueError) as exc:
            resolve_repeat_aug_pool("neighbor_noise", pipeline, augment_type="quintic")
        assert "--use_neighbor_noise" in str(exc.value)

    def test_empty_pipeline_raises(self):
        with pytest.raises(ValueError, match="no augmentations are enabled"):
            resolve_repeat_aug_pool("", [], augment_type="quintic")

    def test_bridge_excludes_state_perturbation_from_default_pool(self):
        """912 ms/sample per-row search: forcing it is not cost-neutral."""
        pipeline = self._pipeline("flip", "state_perturbation")
        pool, warnings = resolve_repeat_aug_pool("", pipeline, augment_type="bridge")
        assert pool == ["flip"]
        assert not any("bridge" in w for w in warnings)

    def test_bridge_default_pool_collapsing_to_flip_still_warns(self):
        """Operator did not ask for flip-alone; bridge reduced it. Warn anyway."""
        pipeline = self._pipeline("flip", "state_perturbation")
        pool, warnings = resolve_repeat_aug_pool("", pipeline, augment_type="bridge")
        assert pool == ["flip"]
        assert any("deterministic" in w for w in warnings)

    def test_bridge_allows_explicit_state_perturbation_with_warning(self):
        pipeline = self._pipeline("flip", "state_perturbation")
        pool, warnings = resolve_repeat_aug_pool(
            "state_perturbation", pipeline, augment_type="bridge"
        )
        assert pool == ["state_perturbation"]
        assert any("bridge" in w for w in warnings)

    def test_bridge_only_state_perturbation_enabled_raises(self):
        """Default pool would be empty, which is a misconfiguration, not a silent no-op."""
        pipeline = self._pipeline("state_perturbation")
        with pytest.raises(ValueError, match="empty"):
            resolve_repeat_aug_pool("", pipeline, augment_type="bridge")

    def test_quintic_keeps_state_perturbation_in_default_pool(self):
        pipeline = self._pipeline("flip", "state_perturbation")
        pool, _ = resolve_repeat_aug_pool("", pipeline, augment_type="quintic")
        assert pool == ["flip", "state_perturbation"]

    def test_flip_only_pool_warns_about_determinism(self):
        """Forced flip is deterministic: two forced occurrences mirror identically."""
        pipeline = self._pipeline("flip")
        pool, warnings = resolve_repeat_aug_pool("flip", pipeline, augment_type="quintic")
        assert pool == ["flip"]
        assert any("deterministic" in w for w in warnings)
