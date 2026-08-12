"""Reading ATLAS v3.0 volumes in native space, where nothing about geometry or dtype is uniform.

The organizers warn that native/RAW data carries "varying orientations and resolutions". The released
files are worse than that suggests, and the corrected r032 masks alone show it:

- **dtypes differ between subjects' masks** — float64, uint8 and uint16 across three files.
- **`sub-r032s013`'s float64 mask stores its foreground as `1.00000006`, not `1.0`.** A loader written
  as `mask == 1` returns an EMPTY mask for that subject, with no error, and the subject silently
  becomes a lesion-free case. `load_mask` binarises on a threshold and refuses anything that is not
  two-valued, so a multi-label mask crashes instead of being flattened.
- **spacing is strongly anisotropic** — those four files are 0.9375 × 3.0 × 0.9375 mm. The thick axis
  is not a constant across the dataset, so any slice-based feature extractor has to choose its slicing
  axis PER SUBJECT (`thick_axis`) rather than hardcoding one.
"""

from functools import cache
from pathlib import Path

import nibabel as nib
import numpy as np
import numpy.typing as npt

# A mask value counts as foreground above this. Not 1.0-equality: see the module docstring.
_FOREGROUND_THRESHOLD = 0.5

# How far a stored foreground value may sit from 1 before we treat the file as something other than a
# binary mask. 1.00000006 is float32's representation error promoted to float64; a genuine multi-label
# map (2, 3, ...) is far outside this and must fail loudly.
_BINARY_TOLERANCE = 1e-3


def _load(path: Path) -> nib.Nifti1Image:
    """nibabel types `load` as returning any FileBasedImage; we require a NIfTI and say so."""
    img = nib.load(path)
    assert isinstance(img, nib.Nifti1Image), f"{path.name} is {type(img).__name__}, not a NIfTI"
    return img


def load_soft(path: Path) -> npt.NDArray[np.float32]:
    """A probability map as float32 — the counterpart to `load_mask`, for continuous volumes.

    Exists because three scripts had each written
    `np.asanyarray(nib.load(str(p)).dataobj).astype(np.float32)` inline. Besides being the same
    incantation three times, `nib.load` is typed as returning `FileBasedImage`, which does not declare
    `dataobj` — so every copy raised the same type-checker warning, and each was individually harmless
    enough to leave. `_load` asserts the concrete `Nifti1Image`, so the access is typed here once.

    **No thresholding and no validation.** A probability map is legitimately continuous, which is
    exactly what `load_mask` refuses.
    """
    return np.asanyarray(_load(path).dataobj).astype(np.float32)


def load_mask(path: Path) -> npt.NDArray[np.bool_]:
    """Binarise on a threshold, refusing anything that is not two-valued. See the module docstring.

    The binary check is O(n) with no sort. It used to call `np.unique`, which sorts the whole volume —
    ~8.7M voxels — and `predict_from_store` calls this on EVERY training step, where the mask is the
    target. `np.unique` is now reached only on the failure path, where its sorted output makes the
    error message readable and its cost does not matter.
    """
    values = np.asanyarray(_load(path).dataobj)
    foreground = values > _FOREGROUND_THRESHOLD
    # Identical predicate to the old `all(|unique_foreground - 1| < tol)`, evaluated per voxel instead
    # of per distinct value: a file passes iff EVERY foreground voxel is within tolerance of 1.
    if not np.all(np.abs(values[foreground] - 1.0) < _BINARY_TOLERANCE):
        offending = np.unique(values[foreground])
        offending = offending[np.abs(offending - 1.0) >= _BINARY_TOLERANCE]
        raise AssertionError(
            f"{path.name} is not a binary mask — foreground values {offending[:8]}. "
            "A multi-label map must not be silently flattened."
        )
    return foreground


def load_volume_and_mask(
    *, t1w: Path, lesion_mask: Path,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_], tuple[float, float, float], float]:
    """One subject's image, reference mask, voxel spacing and voxel volume — the four things every
    consumer of a subject needs, read once.

    Lives here rather than in `frozen_isles.train` (where it was until 2026-08-01) because it reads
    NIfTIs and knows nothing about training. Importing it from there dragged the whole training
    module — and through it `panoptica` — into `scripts/extract_features.py`, which needs no metrics,
    and into any figure script run under an interpreter that lacks them.

    Paths rather than an `AtlasSubject`, so the lowest layer of the package does not import the
    dataset layer that sits above it.
    """
    image = _load(t1w)
    volume = np.asanyarray(image.dataobj).astype(np.float32)
    mask = load_mask(lesion_mask)
    assert volume.shape == mask.shape, (
        f"{t1w.name}: image {volume.shape} vs mask {mask.shape} — the QC scan should have caught this"
    )
    zooms = tuple(float(z) for z in image.header.get_zooms()[:3])
    return volume, mask, (zooms[0], zooms[1], zooms[2]), voxel_volume_ml(t1w)


@cache
def voxel_volume_ml(path: Path) -> float:
    """One voxel's volume in mL — what the challenge's absolute-volume-difference metric needs.

    Read per subject from that subject's own header. In native space this is never a constant, and the
    four corrected r032 files (0.9375 × 3.0 × 0.9375 mm) are 3× a 1 mm isotropic voxel.

    CACHED because it is a pure function of a path whose file does not change during a run, and
    `predict_from_store` called it on every training step — measured at 196 ms per call against a
    1.36 s step, for a value the training loop then DISCARDS. The cache holds one float per subject.
    """
    zooms = _load(path).header.get_zooms()[:3]
    assert len(zooms) == 3, f"{path.name}: expected a 3-D volume, got zooms {zooms}"
    assert all(z > 0 for z in zooms), f"{path.name}: non-positive voxel spacing {zooms}"
    return float(np.prod(zooms)) / 1000.0                      # mm³ → mL


def thick_axis(path: Path) -> int:
    """Index of the most coarsely-sampled axis — the one a 2-D extractor should slice ALONG.

    Slicing along the thick axis keeps each slice at the fine in-plane resolution. Slicing across it
    would build every slice out of 3 mm-apart samples and throw away the resolution we have. Which axis
    that is varies by subject, so it is computed, never assumed.
    """
    zooms = _load(path).header.get_zooms()[:3]
    return int(np.argmax(zooms))


def orientation(path: Path) -> str:
    """Axis codes such as "LAS" — which anatomical direction each array axis runs in."""
    return "".join(nib.aff2axcodes(_load(path).affine))
