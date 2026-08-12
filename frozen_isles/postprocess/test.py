import numpy as np

from frozen_isles.postprocess import (
    close_gaps,
    component_count,
    drop_small_components,
    postprocess,
)


def _two_blobs_with_a_gap(gap: int = 1) -> np.ndarray:
    """One lesion severed into two halves `gap` voxels apart — our measured failure mode."""
    volume = np.zeros((12, 12, 12), dtype=np.uint8)
    volume[2:6, 4:8, 4:8] = 1
    volume[6 + gap : 10 + gap, 4:8, 4:8] = 1
    return volume


def test_closing_rejoins_a_severed_lesion():
    severed = _two_blobs_with_a_gap(gap=1)
    assert component_count(severed) == 2, "fixture should start as two pieces"
    assert component_count(close_gaps(severed, radius=1)) == 1


def test_closing_radius_zero_is_a_no_op():
    volume = _two_blobs_with_a_gap()
    assert np.array_equal(close_gaps(volume, radius=0), volume)


def test_drop_small_removes_debris_but_keeps_the_lesion():
    volume = np.zeros((12, 12, 12), dtype=np.uint8)
    volume[2:8, 2:8, 2:8] = 1          # the lesion, 216 voxels
    volume[10, 10, 10] = 1             # a single-voxel speck
    assert component_count(volume) == 2
    cleaned = drop_small_components(volume, min_voxels=10)
    assert component_count(cleaned) == 1
    assert cleaned.sum() == 216, "the real lesion must be untouched"


def test_min_voxels_of_one_is_a_no_op():
    volume = _two_blobs_with_a_gap()
    assert np.array_equal(drop_small_components(volume, min_voxels=1), volume)


def test_dropping_everything_is_allowed_and_yields_an_empty_mask():
    """An empty prediction is a legitimate output here — a lesion-free case pays four of five metrics."""
    volume = np.zeros((8, 8, 8), dtype=np.uint8)
    volume[1, 1, 1] = 1
    assert drop_small_components(volume, min_voxels=100).sum() == 0


def test_postprocess_closes_before_dropping():
    """Order matters: closing can rescue a fragment into a component large enough to survive.

    Two 4x4x4 halves (64 voxels each) split by a 1-voxel gap. With min_voxels=100 neither half survives
    alone, but closed first they form one 100+ voxel component that does.
    """
    severed = _two_blobs_with_a_gap(gap=1)
    kept = postprocess(severed, min_voxels=100, closing_radius=1)
    assert kept.sum() > 0, "closing should have rescued the lesion before the size filter ran"
    assert component_count(kept) == 1

    # The opposite order would have deleted both halves first.
    dropped_first = close_gaps(drop_small_components(severed, min_voxels=100), radius=1)
    assert dropped_first.sum() == 0


def test_identity_settings_leave_the_mask_untouched():
    volume = _two_blobs_with_a_gap()
    assert np.array_equal(postprocess(volume, min_voxels=1, closing_radius=0), volume)


def test_closing_bridges_a_gap_it_can_span_and_leaves_a_wider_one_alone():
    """The contract closing is tuned on: radius r rejoins halves separated by up to 2r voxels.

    Pins BEHAVIOUR rather than the implementation. `close_gaps` was reimplemented as iterated
    dilation for speed, and the property that must survive is which gaps close — not how the
    structuring element is built.
    """
    for radius in (1, 2, 3):
        for gap in range(1, 2 * radius + 3):
            mask = np.zeros((12, 12, 8 + gap + 8), dtype=np.uint8)
            mask[4:8, 4:8, :8] = 1
            mask[4:8, 4:8, 8 + gap:] = 1
            joined = component_count(close_gaps(mask, radius=radius)) == 1
            assert joined == (gap <= 2 * radius), (
                f"radius {radius}, gap {gap}: joined={joined}, expected {gap <= 2 * radius}"
            )


def test_closing_adds_voxels_everywhere_except_the_volume_border():
    """Closing is extensive in the INTERIOR, and erodes at the FOV faces. Both halves are the contract.

    Textbook closing satisfies X ⊆ closing(X). `scipy.ndimage`'s erosion pads the array with 0, so
    foreground within `radius` of a face gets eaten — measured 2026-08-02 on a dense random mask:
    410 voxels lost at radius 1 and 1,280 at radius 4, **every one of them less than `radius` from a
    face of the volume, none deeper**. The affected shell grows with the radius, which is why the
    bound is stated in terms of it rather than as "the outermost layer".

    This predates the iterated-dilation rewrite and is unchanged by it. It is pinned rather than
    fixed because `border_value=1` would change every post-processing number, and a sweep was in
    flight. See `FIXME.md`.
    """
    rng = np.random.default_rng(0)
    mask = (rng.random((20, 22, 18)) < 0.2).astype(np.uint8)
    for radius in (1, 2, 3, 4):
        closed = close_gaps(mask, radius=radius)
        lost = np.argwhere(mask.astype(bool) & ~closed.astype(bool))
        assert len(lost), f"radius {radius}: expected the known border erosion, saw none"
        depth = np.minimum(lost, np.array(mask.shape) - 1 - lost).min(axis=1)
        assert (depth < radius).all(), (
            f"radius {radius}: closing removed voxels {int(depth.max())} deep, beyond the {radius}-voxel "
            f"border shell — that would be a real defect rather than the known padding effect"
        )
