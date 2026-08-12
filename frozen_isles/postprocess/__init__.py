"""Post-processing the thresholded mask — aimed at fragmentation, which is our measured failure.

**Why this exists.** Measured 2026-07-30: the reference has a median of **2** lesions (40% of subjects
have exactly one), while our predictions are off by a median of 2 and by as much as 78. We
**over-segment few large lesions into pieces**; we do not miss small ones. Two of the challenge's five
ranked metrics — lesion-F1 and lesion-count difference — read that fragmentation directly, so it costs
twice.

Two operations, deliberately the two that address fragmentation and nothing else:

- `drop_small_components` removes specks below a volume, which is what turns one lesion plus debris
  back into one lesion.
- `close_gaps` dilates then erodes, bridging a severed lesion whose halves sit a voxel or two apart —
  the signature of per-slice boundary noise in a slice-wise encoder.

Both operate on the BINARY mask, in 3D, with 26-connectivity, matching how the organizers' instance
metrics label components.

**Neither can raise voxel Dice much, and that is fine.** They target instance structure. Judge them
by `rank_then_aggregate` over all five metrics, never by Dice alone.
"""

import numpy as np
import numpy.typing as npt
from scipy import ndimage

# 26-connectivity in 3D: the organizers' instance metrics label components this way, so post-processing
# must agree or it would merge/split differently from the thing scoring us.
CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=bool)


def drop_small_components(mask: npt.NDArray, *, min_voxels: int) -> npt.NDArray:
    """Remove connected components smaller than `min_voxels`.

    `min_voxels <= 1` returns the mask unchanged, so a sweep can include "no filtering" without a
    special case at the call site.
    """
    if min_voxels <= 1:
        return mask
    labelled, count = ndimage.label(mask.astype(bool), structure=CONNECTIVITY_26)  # type: ignore[misc]
    if count == 0:
        return mask
    # index 0 is background; bincount over labels gives each component's size in one pass.
    sizes = np.bincount(labelled.ravel())
    too_small = np.flatnonzero(sizes < min_voxels)
    if too_small.size == 0:
        return mask
    return np.where(np.isin(labelled, too_small), 0, mask).astype(mask.dtype)


def close_gaps(mask: npt.NDArray, *, radius: int) -> npt.NDArray:
    """Binary closing (dilate then erode) with a ball of `radius`, to rejoin severed components.

    `radius <= 0` returns the mask unchanged. Radius is in VOXELS, not mm — native voxel size varies
    across this cohort (74 distinct spacings), so a mm radius would mean different things per subject
    and make the sweep uninterpretable.

    **Closing is extensive in the interior but ERODES within `radius` of the volume's faces**,
    because `scipy.ndimage`'s erosion pads with 0. A lesion reaching the FOV boundary loses voxels in
    that shell, and the shell grows with the radius.
    Measured 2026-08-02 and pinned by a test; `border_value=1` removes the effect and changes every
    post-processing number, so it is a deliberate open decision rather than an oversight.
    """
    if radius <= 0:
        return mask
    # `iterations=radius` dilates radius times then erodes radius times, which is closing with the
    # radius-times-iterated structure -- morphological dilation is associative, so this is the SAME
    # operation, not an approximation of it. Verified byte-identical on lesion-like volumes and on
    # border-touching, dense, all-foreground and all-background masks at radius 1-5.
    #
    # It matters because building the structure explicitly makes a (2r+1)^3 kernel and the cost grows
    # with it: on a 181x213x173 volume, 0.47 s -> 0.12 s at radius 2 and 2.31 s -> 0.15 s at radius 4
    # (3.8x and 15x). The post-processing sweep pays this per subject per variant, and the grid has to
    # widen in exactly the direction that was most expensive.
    return ndimage.binary_closing(mask.astype(bool), structure=CONNECTIVITY_26,
                                  iterations=radius).astype(mask.dtype)


def postprocess(mask: npt.NDArray, *, min_voxels: int, closing_radius: int) -> npt.NDArray:
    """Close first, then drop: closing can rejoin a fragment into a component large enough to survive.

    Dropping first would delete fragments that closing would have rescued, which is the opposite of
    what we want when the failure is fragmentation.
    """
    return drop_small_components(close_gaps(mask, radius=closing_radius), min_voxels=min_voxels)


def component_count(mask: npt.NDArray) -> int:
    """Components under the same 26-connectivity the metrics use — for reporting count error directly."""
    labelled, count = ndimage.label(mask.astype(bool), structure=CONNECTIVITY_26)  # type: ignore[misc]
    del labelled
    return int(count)
