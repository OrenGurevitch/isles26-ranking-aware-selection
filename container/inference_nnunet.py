"""ISLES'26 submission entrypoint for the nnU-Net baseline.

The case loop and the degradation ladder are shared with the probe entrypoint (`container/run.py`).
What is here is the part nnU-Net insists on owning: reading the volume through its OWN
`image_reader_writer`, and sliding-window prediction with the plans it was trained under.

**The reader is not interchangeable.** `predict_single_npy_array` documents that a disturbed axis
ordering silently produces bad results — no crash, just a worse score. `nnUNetPlans.json` names
`SimpleITKIO` for this dataset, so that class reads the input here, and the array it returns was
confirmed on 2026-08-02 to match `sitk.GetArrayFromImage` exactly, i.e. [z, y, x]. The one transpose
below converts that to the [x, y, z] the writer expects, and `run_case` asserts the result against
`GetSize()` rather than trusting it.

**Test-time mirroring is 8 forward passes per tile.** It is what nnU-Net's own reported numbers
include, so it is on by default; `ISLES_NNUNET_TTA=0` turns it off if the T4 budget turns out not to
hold. Which of those is true is a MEASUREMENT — the sanity-check phase reports the per-case time on
the real hardware, and this file prints its own.
"""

import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

from container.run import run_case

# The baked-in path, overridable so this exact class can be verified OUTSIDE the image against a
# trained fold on the cluster — the check that proves the reader and the transpose, which is the
# whole risk here. In the image the variable is unset and the default applies.
MODEL_DIR = os.environ.get("ISLES_NNUNET_DIR", "/opt/app/resources/nnunet")
# `"all"` is nnU-Net's own name for a model trained on EVERY subject with nothing held out — the
# mode the shipped model uses, because the hidden test set is our held-out set. `use_folds` accepts
# strings as well as ints, and the directory it looks for is literally `fold_all`.
#
# **Do NOT stage all-data weights into a `fold_0` directory to avoid changing this.** The path
# would then claim a held-out model while holding one trained on everything, and the only symptom
# would be a number nobody can reproduce.
FOLDS: tuple[int | str, ...] = ("all",)
FOREGROUND_CLASS = 1


class NnunetPredictor:
    def load(self, device: torch.device) -> None:
        # nnunetv2 lives in the IMAGE, not in this project's environment -- the container is the
        # only place it is installed, so the checker cannot see it here.
        from nnunetv2.inference.predict_from_raw_data import (  # type: ignore[reportMissingImports]
            nnUNetPredictor,
        )

        use_mirroring = os.environ.get("ISLES_NNUNET_TTA", "1") != "0"
        self.predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=use_mirroring,
            perform_everything_on_device=device.type == "cuda",
            device=device,
            allow_tqdm=False,
        )
        self.predictor.initialize_from_trained_model_folder(
            MODEL_DIR, use_folds=FOLDS, checkpoint_name="checkpoint_final.pth"
        )
        print(f"[model] nnU-Net from {MODEL_DIR} folds={FOLDS} mirroring={use_mirroring}", flush=True)

    def predict(self, *, image: sitk.Image, image_path: Path, device: torch.device) -> np.ndarray:
        from nnunetv2.imageio.simpleitk_reader_writer import (  # type: ignore[reportMissingImports]
            SimpleITKIO,
        )

        data, properties = SimpleITKIO().read_images((str(image_path),))
        _, probabilities = self.predictor.predict_single_npy_array(
            data, properties, save_or_return_probabilities=True
        )
        probabilities = np.asarray(probabilities, dtype=np.float32)
        assert probabilities.shape[0] > FOREGROUND_CLASS, (
            f"nnU-Net returned {probabilities.shape[0]} classes; expected a foreground channel at "
            f"index {FOREGROUND_CLASS}"
        )
        return probabilities[FOREGROUND_CLASS].transpose(2, 1, 0)


if __name__ == "__main__":
    raise SystemExit(run_case(NnunetPredictor()))
