"""The Grand Challenge case loop, with the model left open.

Everything here is about surviving a case rather than segmenting one: find the input, read it, hand
it to a model, and write SOMETHING scoreable no matter what went wrong on the way. Two submission
containers share it — the frozen DINOv3 probe and nnU-Net — because this is the part that took the
incidents to get right, and a second copy of it would rot out of step with the first.

The contract, taken from the organizers' own template rather than guessed:

    in : /input/images/<slug>/<uuid>.mha
    out: /output/images/stroke-lesion-segmentation/<THE SAME uuid>.mha
         /output/results.json

A model plugs in as a `Predictor`: `load(device)` once, then `predict(...)` per case, returning a
probability map indexed [x, y, z] to match `sitk.Image.GetSize()`. Both hooks may raise — a raised
exception degrades the case to an empty map, which is what the guarantee below is built on.

**Nothing here raises.** A case that crashes takes the WORST rank on every metric, so a degraded
answer beats no answer. The failure path emits an exactly-constant zero map, which is also the correct
answer for a genuinely lesion-free case — five such subjects exist in ATLAS training, at 2 to 86 days
post stroke, where a T1w lesion is not yet visible.
"""

import sys
import time
import traceback
from pathlib import Path
from typing import Protocol

import numpy as np
import SimpleITK as sitk
import torch

from frozen_isles.gcio import (
    CANDIDATE_OUTPUT_SOCKETS,
    CANDIDATE_PROBABILITY_SOCKETS,
    binarise,
    describe_tree,
    find_input_image,
    input_root,
    output_root,
    read_metadata,
    write_prediction,
    write_results_json,
)
from frozen_isles.postprocess import component_count, postprocess

THRESHOLD = 0.5

# Post-processing, selected on the 500-epoch model over ALL 291 held-out subjects (`docs/RESULTS.md`,
# jobs 66982393 and 66981170). `min_voxels` 25 and 50 tie for first and both beat doing nothing;
# **25 is chosen over the equally-ranked 50 because the failure mode is DELETING a real lesion** — of
# two settings that score the same, take the one that removes less.
#
# **That selection describes a model we no longer ship.** At the shipped TopK10 @ 1,000 epochs the
# filter is INERT: identical Dice, lesion-F1 and PR-AUC raw against filtered, and |Δvol| 0.0042 mL
# worse — the longer schedule leaves it nothing to remove. The value stays at 25 because a filter with
# nothing to remove costs nothing, and because dropping it would be an unmeasured change to a shipped
# container. Read the paragraph above as how 25 was chosen, never as a claim about the shipped system.
#
# Do NOT raise this to 100. That value won an earlier sweep on a 100-subject SELECTION SUBSET and
# then LOST to no post-processing on the full fold: it gives an empty prediction to 16 subjects who
# have a lesion, against 2, and each of those takes the worst rank on every metric at once.
#
# `closing_radius=0` because closing bought lesion-F1 and gave back more elsewhere under the five-metric
# objective — and because it erodes foreground within `radius` of the FOV faces (`frozen_isles.postprocess`).
MIN_VOXELS = 25
CLOSING_RADIUS = 0

# **420, not 600.** The Sanity Check phase page states "a time limit of 7 minutes per case"
# [read from the phase's own submission page, 2026-08-02]. Every earlier note in this repository said
# 10 minutes, from a forum quote ("Dockers are killed after 10 mins"); the phase page is newer, more
# specific, and is what actually kills the job.
BUDGET_SECONDS = 420


class Predictor(Protocol):
    def load(self, device: torch.device) -> None:
        ...

    def predict(self, *, image: sitk.Image, image_path: Path, device: torch.device) -> np.ndarray:
        """Probability map indexed [x, y, z], matching `image.GetSize()`.

        Both the read image and its path are passed because models disagree about what they want:
        the probe resamples the array it is given, while nnU-Net must read through its OWN
        `image_reader_writer` to guarantee the axis ordering it was trained on.
        """
        ...


def _geometry_from_header(image_path: Path) -> sitk.Image:
    """An all-zero image with the input's geometry, read from the HEADER ALONE.

    The last line of defence when the pixel buffer cannot be read: a truncated or corrupt volume
    usually still has an intact header, and the challenge requires the output to match the input's
    dimensions, spacing and orientation. Reading only the header is far more likely to succeed than
    reading the data, and it is the difference between a scoreable empty answer and no answer.
    """
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(image_path))
    reader.ReadImageInformation()
    image = sitk.Image(reader.GetSize(), sitk.sitkUInt8)
    image.SetSpacing(reader.GetSpacing())
    image.SetOrigin(reader.GetOrigin())
    image.SetDirection(reader.GetDirection())
    return image


def _any_input_file(inputs: Path) -> Path | None:
    """Any plausible input volume, so a failed lookup can still NAME an output.

    The output must be named after the input file, so without a name nothing can be written at all.
    """
    for candidate in sorted(inputs.rglob("*")):
        if candidate.is_file() and candidate.suffix.lower() in {".mha", ".mhd", ".gz", ".nii", ".nrrd"}:
            return candidate
    return None


def run_case(predictor: Predictor) -> int:
    started = time.perf_counter()
    inputs, outputs = input_root(), output_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device} torch={torch.__version__}", flush=True)

    # EVERY step below is guarded, and that is the point of this function's shape. Until
    # 2026-08-02 the input lookup, the image read and the metadata read all ran BEFORE the try, so a
    # malformed input raised out of main and the container wrote NO mask, NO probability map and NO
    # results.json — the exact outcome this design exists to prevent, in the one place where it is
    # unrecoverable. A missing result takes the WORST rank on that case; a degraded one does not.
    image_path: Path | None = None
    volume_image: sitk.Image | None = None
    try:
        image_path = find_input_image(inputs)
        volume_image = sitk.ReadImage(str(image_path))
        print(f"[input] {image_path.name} size={volume_image.GetSize()} "
              f"spacing={volume_image.GetSpacing()}", flush=True)
    except Exception:
        print("[FATAL] could not read the input image — degrading rather than failing", flush=True)
        traceback.print_exc(file=sys.stdout)
        if image_path is None:
            image_path = _any_input_file(inputs)
            print(f"[recover] naming the output after {image_path}", flush=True)
        if image_path is not None:
            try:
                volume_image = _geometry_from_header(image_path)
                print(f"[recover] geometry from the header alone: {volume_image.GetSize()}",
                      flush=True)
            except Exception:
                print("[FATAL] the header is unreadable too", flush=True)
                traceback.print_exc(file=sys.stdout)

    if image_path is None:
        # The one unrecoverable rung: with no input filename there is nothing to name an output after,
        # so this case is scored as a missing result and takes the worst rank on every metric
        # (`docs/CHALLENGE.md`, *Ranking*). That happens whatever this function returns, because the
        # organizers key on the absent file rather than on the exit status — so the printed tree below
        # is the only thing that will explain WHY, and it has to be loud.
        #
        # **Whether to exit non-zero here is OPEN.** Returning 0 reports success for a case that
        # produced nothing; exiting non-zero would surface it in Grand Challenge's own job view. Which
        # is right depends on whether a failed job costs only its own case, and nobody has verified
        # that. Do not flip it on intuition — it is a shipped container.
        print("[FATAL] no input file found at all — cannot name an output", flush=True)
        print(describe_tree(inputs), flush=True)
        return 0
    if volume_image is None:
        # Reached only when the pixel data AND the header are both unreadable, so the true dimensions
        # are unknowable here. **A 1×1×1 output cannot match the input geometry the challenge
        # requires**, so whether the evaluator scores it or rejects it as missing is UNVERIFIED — this
        # rung may buy nothing over writing no file at all. It is kept because it costs one voxel and
        # the alternative is raising out of `run_case`, which loses the results.json for the case too.
        print("[recover] emitting a minimal 1x1x1 output so the case is answered at all", flush=True)
        volume_image = sitk.Image((1, 1, 1), sitk.sitkUInt8)

    try:
        print(f"[metadata] {read_metadata(inputs) or 'none'}", flush=True)
    except Exception:                                  # metadata is informational, never fatal
        print("[warn] metadata unreadable; continuing", flush=True)
        traceback.print_exc(file=sys.stdout)

    try:
        predictor.load(device)
        print(f"[model] loaded at {time.perf_counter() - started:.1f}s", flush=True)
        probability = predictor.predict(image=volume_image, image_path=image_path, device=device)
        assert probability.shape == tuple(volume_image.GetSize()), (
            f"{type(predictor).__name__} returned {probability.shape} for an input of "
            f"{volume_image.GetSize()} — the map must be indexed [x, y, z]"
        )
    except Exception:
        print("[FATAL] prediction failed — emitting an empty map so the case still scores",
              flush=True)
        traceback.print_exc(file=sys.stdout)
        probability = np.zeros(volume_image.GetSize(), dtype=np.float32)

    mask = binarise(probability, THRESHOLD)
    raw_voxels, raw_components = int(mask.sum()), component_count(mask)
    # The MASK is post-processed and the PROBABILITY MAP is written untouched. PR-AUC is computed on
    # the soft map, so filtering it would change a metric the filter was never evaluated against — and
    # the sweep that chose these settings scored exactly this way: filtered mask, raw soft map.
    if MIN_VOXELS or CLOSING_RADIUS:
        try:
            mask = postprocess(mask, min_voxels=MIN_VOXELS, closing_radius=CLOSING_RADIUS)
        except Exception:                       # a post-processing failure must not lose the mask
            print("[warn] post-processing failed; writing the unfiltered mask", flush=True)
            traceback.print_exc(file=sys.stdout)
    print(f"[predict] foreground={int(mask.sum()):,} voxels "
          f"({100 * mask.mean():.3f}% of the volume), {component_count(mask)} lesion(s) — "
          f"before post-processing: {raw_voxels:,} voxels, {raw_components} lesion(s) "
          f"[min_voxels={MIN_VOXELS}, closing_radius={CLOSING_RADIUS}]", flush=True)

    write_prediction(array=mask, reference=volume_image, sockets=CANDIDATE_OUTPUT_SOCKETS,
                     filename=image_path.name, outputs=outputs)
    write_prediction(array=probability, reference=volume_image,
                     sockets=CANDIDATE_PROBABILITY_SOCKETS,
                     filename=image_path.name, outputs=outputs)
    write_results_json(output_filename=image_path.name, input_filename=image_path.name,
                       input_socket=image_path.parent.name, outputs=outputs)

    elapsed = time.perf_counter() - started
    print(f"[done] {elapsed:.1f}s of the {BUDGET_SECONDS} s budget "
          f"({elapsed / BUDGET_SECONDS:.1%})", flush=True)
    print(describe_tree(outputs), flush=True)
    return 0
