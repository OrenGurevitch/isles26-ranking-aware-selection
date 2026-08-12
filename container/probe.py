"""ISLES'26 sanity-phase entrypoint: an instrument, not a model.

The test phase allows ONE submission and we do not know the platform's socket names, whether metadata
arrives, what format the image is in, or what hardware we get. The sanity phase allows more than one,
so this container's job is to come back with those answers.

This said "unlimited" until 2026-08-01 and that contradicted the challenge record: `FIXME.md` has
**2 attempts per day** (relayed by the user) and `docs/CHALLENGE.md` has "multiple attempts allowed".
The difference is not cosmetic — the reason to spend the FIRST attempt on this probe rather than on
the real container is precisely that attempts are rationed. `docs/CHALLENGE.md` owns the number; do
not restate it here.

It prints everything observable, writes a valid prediction to every candidate socket, and — critically
— exits 0 whatever happens. A crash would tell us only that it crashed, and a crashed case takes the
worst rank; a container that survives and reports is worth far more than one that is right.

The "prediction" is a deliberately trivial intensity rule. It is not meant to score. Anything that
scores would confound the one question this container exists to answer.
"""

import sys
import traceback

import numpy as np
import SimpleITK as sitk

from frozen_isles.gcio import (
    CANDIDATE_OUTPUT_SOCKETS,
    CANDIDATE_PROBABILITY_SOCKETS,
    binarise,
    describe_environment,
    describe_tree,
    find_input_image,
    input_root,
    output_root,
    read_metadata,
    write_prediction,
    write_results_json,
)

# Stroke lesions are HYPOintense on T1w — dark against brain. This picks the darkest tissue inside the
# head as a placeholder so the output is a plausible shape rather than a constant, without pretending
# to be a method.
_DARK_PERCENTILE = 2.0


def placeholder_probability(volume: sitk.Image) -> np.ndarray:
    array = sitk.GetArrayFromImage(volume).transpose(2, 1, 0).astype(np.float32)   # -> x, y, z
    inside_head = array > 0                                                        # already skull-stripped
    if not inside_head.any():
        print("[predict] the volume is entirely zero — emitting an empty map", flush=True)
        return np.zeros_like(array)
    cutoff = float(np.percentile(array[inside_head], _DARK_PERCENTILE))
    print(f"[predict] {_DARK_PERCENTILE}th percentile inside head = {cutoff:.3f}", flush=True)
    return (inside_head & (array <= cutoff)).astype(np.float32)


def main() -> int:
    inputs, outputs = input_root(), output_root()
    print(describe_environment(inputs, outputs), flush=True)

    try:
        image_path = find_input_image(inputs)
        print(f"\n[input] {image_path}", flush=True)
        volume = sitk.ReadImage(str(image_path))
        print(f"[input] size={volume.GetSize()} spacing={volume.GetSpacing()} "
              f"origin={volume.GetOrigin()} pixel={volume.GetPixelIDTypeAsString()}", flush=True)

        metadata = read_metadata(inputs)
        print(f"[metadata] {metadata or 'NONE FOUND'}", flush=True)

        probability = placeholder_probability(volume)
        mask = binarise(probability)
        print(f"[predict] foreground voxels = {int(mask.sum()):,}", flush=True)

        # The output is named after the INPUT file, per the organizers' template — not "output.mha".
        write_prediction(array=mask, reference=volume, sockets=CANDIDATE_OUTPUT_SOCKETS,
                         filename=image_path.name, outputs=outputs)
        write_prediction(array=probability, reference=volume,
                         sockets=CANDIDATE_PROBABILITY_SOCKETS,
                         filename=image_path.name, outputs=outputs)
        # Without results.json the platform never learns an output exists. The input socket is the
        # directory the image was mounted in, which is what the probe is here to discover.
        write_results_json(output_filename=image_path.name, input_filename=image_path.name,
                           input_socket=image_path.parent.name, outputs=outputs)
    except Exception:
        # Deliberate blanket catch, and the one place it is right: this container's product is its
        # log. Re-raising would lose the diagnosis we came for.
        print("\n[FATAL] the probe failed — traceback follows, but exiting 0 so the log survives",
              flush=True)
        traceback.print_exc(file=sys.stdout)

    print("\n=== OUTPUT TREE (after writing) ===", flush=True)
    print(describe_tree(outputs), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
