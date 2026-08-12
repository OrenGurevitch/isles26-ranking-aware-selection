"""Contracts for the container boundary, exercised against a simulated Grand Challenge filesystem.

None of this needs Docker. The platform's contract is a directory layout, so a directory layout is
what the tests build — which means the entrypoint is verified end to end before it is ever submitted,
and the sanity phase is spent learning the socket names rather than finding our own bugs.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from frozen_isles.gcio import (
    PROBABILITY_SOCKET,
    SEGMENTATION_SOCKET,
    binarise,
    describe_environment,
    describe_tree,
    empty_prediction,
    find_input_image,
    is_exactly_constant,
    read_metadata,
    write_prediction,
    write_results_json,
)


def _mount(root: Path, *, socket: str = "t1-brain-mri", suffix: str = ".mha") -> Path:
    """Build what Grand Challenge mounts: /input/images/<socket>/<uuid><suffix>."""
    directory = root / "images" / socket
    directory.mkdir(parents=True)
    volume = sitk.GetImageFromArray(np.zeros((4, 5, 6), np.float32))   # z, y, x
    volume.SetSpacing((0.9375, 3.0, 0.9375))
    path = directory / f"6f0a1b2c-dead-beef{suffix}"
    sitk.WriteImage(volume, str(path))
    return path


@pytest.mark.parametrize("suffix", [".mha", ".nii.gz", ".nrrd"])
def test_the_input_is_found_whatever_the_socket_is_called(tmp_path: Path, suffix: str) -> None:
    """The socket name is the unknown, so the search must not depend on it."""
    expected = _mount(tmp_path, socket="some-name-we-did-not-predict", suffix=suffix)
    assert find_input_image(tmp_path) == expected


def test_a_missing_input_reports_what_it_did_see(tmp_path: Path) -> None:
    """A container that dies silently teaches the sanity run nothing."""
    (tmp_path / "images" / "empty-socket").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="empty-socket"):
        find_input_image(tmp_path)


def test_two_input_volumes_are_refused_rather_than_picked_between(tmp_path: Path) -> None:
    _mount(tmp_path, socket="a")
    _mount(tmp_path, socket="b")
    with pytest.raises(AssertionError, match="expected one input volume"):
        find_input_image(tmp_path)


def test_metadata_is_read_and_merged(tmp_path: Path) -> None:
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "case.json").write_text(
        json.dumps({"DAYS_POST_STROKE": 42, "CHRONICITY": 1, "CENTER": "r032"})
    )
    assert read_metadata(tmp_path) == {"DAYS_POST_STROKE": 42, "CHRONICITY": 1, "CENTER": "r032"}


def test_absent_or_broken_metadata_yields_empty_rather_than_crashing(tmp_path: Path) -> None:
    """Missing results take the WORST rank, so an optional file must never kill the case."""
    assert read_metadata(tmp_path) == {}
    (tmp_path / "broken.json").write_text("{not json")
    assert read_metadata(tmp_path) == {}


def test_the_prediction_keeps_the_inputs_geometry(tmp_path: Path) -> None:
    """Scoring happens in the input's native space — spacing and origin must survive."""
    source = _mount(tmp_path / "in")
    reference = sitk.ReadImage(str(source))
    array = np.zeros(reference.GetSize(), np.uint8)               # x, y, z
    array[1, 2, 3] = 1

    written = write_prediction(
        array=array, reference=reference, sockets=("socket-a", "socket-b"),
        filename="6f0a1b2c-dead-beef.mha", outputs=tmp_path / "out",
    )
    assert [p.parent.name for p in written] == ["socket-a", "socket-b"]

    result = sitk.ReadImage(str(written[0]))
    assert result.GetSize() == reference.GetSize()
    assert result.GetSpacing() == pytest.approx(reference.GetSpacing())
    assert sitk.GetArrayFromImage(result).transpose(2, 1, 0)[1, 2, 3] == 1


def test_a_prediction_of_the_wrong_shape_is_refused(tmp_path: Path) -> None:
    reference = sitk.ReadImage(str(_mount(tmp_path / "in")))
    with pytest.raises(AssertionError, match="native space"):
        write_prediction(
            array=np.zeros((2, 2, 2), np.uint8), reference=reference,
            sockets=("socket-a",), filename="x.mha", outputs=tmp_path / "out",
        )


def test_binarise_uses_a_threshold_not_an_equality() -> None:
    """Same defect class as the float mask in the released labels: never compare a float to 1."""
    assert binarise(np.array([0.0, 0.49, 0.5, 1.00000006])).tolist() == [0, 0, 1, 1]


def test_the_environment_dump_names_variables_without_printing_values(tmp_path: Path) -> None:
    """The log goes to a service we do not control; a value could be a credential."""
    import os

    os.environ["ISLES_TEST_SECRET_VALUE"] = "do-not-log-me"
    try:
        dump = describe_environment(tmp_path, tmp_path)
        assert "ISLES_TEST_SECRET_VALUE" in dump
        assert "do-not-log-me" not in dump
    finally:
        del os.environ["ISLES_TEST_SECRET_VALUE"]


def test_the_tree_description_lists_files_with_sizes(tmp_path: Path) -> None:
    _mount(tmp_path)
    described = describe_tree(tmp_path)
    assert "images" in described and "6f0a1b2c-dead-beef.mha" in described and " B]" in described


def test_the_output_is_named_after_the_input_not_a_fixed_name(tmp_path: Path) -> None:
    """The organizers' template names the output after the input's UUID; results.json refers to it."""
    reference = sitk.ReadImage(str(_mount(tmp_path / "in")))
    written = write_prediction(
        array=np.zeros(reference.GetSize(), np.uint8), reference=reference,
        sockets=(SEGMENTATION_SOCKET,), filename="6f0a1b2c-dead-beef.mha",
        outputs=tmp_path / "out",
    )
    assert written[0].name == "6f0a1b2c-dead-beef.mha"


def test_results_json_declares_the_output_the_platform_should_ingest(tmp_path: Path) -> None:
    """Omitting results.json is not a degraded submission — the platform never sees the output."""
    path = write_results_json(
        output_filename="6f0a1b2c.mha", input_filename="6f0a1b2c.mha",
        input_socket="t1-brain-mri", outputs=tmp_path,
    )
    payload = json.loads(path.read_text())
    assert path.name == "results.json"
    # BOTH outputs: the phase requires the mask AND the probability map, and declaring one while
    # writing two tells the platform about half of what we produced.
    assert payload[0]["outputs"] == [
        {"type": "Image", "slug": SEGMENTATION_SOCKET, "filename": "6f0a1b2c.mha"},
        {"type": "Image", "slug": PROBABILITY_SOCKET, "filename": "6f0a1b2c.mha"},
    ]
    assert payload[0]["inputs"][0]["slug"] == "t1-brain-mri"


def test_an_empty_prediction_is_exactly_constant(tmp_path: Path) -> None:
    """Three of five ranked metrics turn on bit-identical voxels, so this cannot be approximate."""
    reference = sitk.ReadImage(str(_mount(tmp_path / "in")))
    empty = empty_prediction(reference)
    assert empty.shape == reference.GetSize()
    assert is_exactly_constant(empty)
    assert empty.dtype == np.float32


def test_near_zero_noise_is_not_constant_and_would_forfeit_the_point() -> None:
    noisy = np.full((4, 4, 4), 1e-12, np.float32)
    noisy[0, 0, 0] = 2e-12
    assert not is_exactly_constant(noisy)


def test_the_probe_container_entrypoint_runs_end_to_end(tmp_path, monkeypatch) -> None:
    """The sanity-phase container, exercised against a simulated Grand Challenge mount.

    We get ONE test-phase submission and a rationed number of sanity attempts, so the entrypoint's
    contract is pinned here rather than discovered on the platform: it must find the input by SUFFIX
    (the ISLES'26 input slug is still unknown), write every candidate socket, name each output after
    the INPUT file, emit `results.json`, and exit 0 whatever happens.

    Run manually on 2026-08-02 against a synthetic volume before any upload; this is that check made
    repeatable.
    """
    import importlib.util
    import json

    import numpy as np
    import SimpleITK as sitk

    inputs, outputs = tmp_path / "input", tmp_path / "output"
    (inputs / "images" / "some-unknown-slug").mkdir(parents=True)
    outputs.mkdir()

    volume = np.zeros((30, 32, 24), dtype=np.float32)
    volume[5:25, 6:26, 4:20] = 120.0
    volume[12:16, 14:18, 9:13] = 60.0                     # a hypointense blob
    image = sitk.GetImageFromArray(volume.transpose(2, 1, 0))
    image.SetSpacing((1.0, 1.0, 1.0))
    uuid = "00000000-0000-0000-0000-00000000abcd"
    sitk.WriteImage(image, str(inputs / "images" / "some-unknown-slug" / f"{uuid}.mha"))
    (inputs / "isles26-metadata.json").write_text(json.dumps({"CHRONICITY": 1, "CENTER": "R001"}))

    monkeypatch.setenv("ISLES_INPUT_ROOT", str(inputs))
    monkeypatch.setenv("ISLES_OUTPUT_ROOT", str(outputs))

    probe_path = Path(__file__).resolve().parents[2] / "container" / "probe.py"
    spec = importlib.util.spec_from_file_location("isles_probe", probe_path)
    assert spec and spec.loader
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)

    assert probe.main() == 0, "the probe must exit 0 whatever happens — a crash tells us nothing"

    written = sorted(outputs.rglob(f"*/{uuid}.mha"))
    assert written, "no socket was written; the output must be named after the INPUT file"
    manifest = json.loads((outputs / "results.json").read_text())
    assert manifest[0]["outputs"][0]["filename"] == f"{uuid}.mha"
    assert manifest[0]["inputs"][0]["slug"] == "some-unknown-slug", (
        "the input slug must be DISCOVERED, not assumed — we do not know ISLES'26's yet"
    )

    mask = sitk.GetArrayFromImage(sitk.ReadImage(str(written[0])))
    assert set(np.unique(mask)) <= {0, 1}, f"a mask socket is not binary: {np.unique(mask)[:5]}"


def test_the_submission_container_writes_an_output_even_when_the_input_is_unreadable(
    tmp_path, monkeypatch
) -> None:
    """Finding A12: a malformed input must NOT produce a silent no-output.

    `container/inference.py` used to do its input lookup, image read and metadata read BEFORE its
    `try:`, so a corrupt volume raised out of `main` and the container wrote no mask, no probability
    map and no `results.json`. Under this challenge a missing result takes the WORST rank on that
    case, while a degraded one merely scores badly — and we get ONE test-phase submission, so the
    difference is unrecoverable.

    This writes a file with a valid header and a truncated pixel buffer, which is what a partial
    upload looks like, and asserts the container still names and writes an output.
    """
    import SimpleITK as sitk

    inputs, outputs = tmp_path / "input", tmp_path / "output"
    (inputs / "images" / "some-slug").mkdir(parents=True)
    outputs.mkdir()

    uuid = "11111111-2222-3333-4444-555555555555"
    target = inputs / "images" / "some-slug" / f"{uuid}.mha"
    reference = sitk.Image((8, 9, 10), sitk.sitkFloat32)
    reference.SetSpacing((1.0, 1.0, 2.0))
    sitk.WriteImage(reference, str(target))
    intact = target.read_bytes()
    target.write_bytes(intact[: len(intact) // 2])          # header survives, pixel data does not

    monkeypatch.setenv("ISLES_INPUT_ROOT", str(inputs))
    monkeypatch.setenv("ISLES_OUTPUT_ROOT", str(outputs))

    # A predictor that raises, which is what a REAL model does when handed the header-only image this
    # path recovers. Testing the guarantee against a stub rather than against DINOv3 or nnU-Net is
    # deliberate: the promise belongs to `run_case` and holds for every arm, and a test that needed
    # baked-in weights could not run here at all.
    assert _run_with(_Raises()) == 0, "the container must not raise out of run_case"

    written = sorted(outputs.rglob(f"*/{uuid}.mha"))
    assert written, "a corrupt input still has to produce a named output — A12"
    assert (outputs / "results.json").is_file(), "results.json is how the platform learns we answered"
    recovered = sitk.ReadImage(str(written[0]))
    assert recovered.GetSize() == (8, 9, 10), (
        f"the header carries the geometry the output must match; got {recovered.GetSize()}"
    )


class _Raises:
    def load(self, device):
        raise RuntimeError("no weights here")

    def predict(self, *, image, image_path, device):
        raise AssertionError("unreachable: load raises first")


class _WrongShape:
    """Returns a map indexed [z, y, x] — the transpose bug the nnU-Net path is one line away from."""

    def load(self, device):
        pass

    def predict(self, *, image, image_path, device):
        import numpy as np

        return np.zeros(tuple(reversed(image.GetSize())), dtype=np.float32)


def _run_with(predictor) -> int:
    try:
        from container.run import run_case
    except ImportError as missing:                          # torch is absent from some environments
        import pytest

        pytest.skip(f"container/run.py needs {missing.name}")
    return run_case(predictor)


def test_a_transposed_probability_map_is_caught_rather_than_written(tmp_path, monkeypatch):
    """A map indexed [z, y, x] must degrade the case, not ship as a silently wrong segmentation.

    This is the failure mode `predict_single_npy_array` warns about in its own docstring: nnU-Net
    returns [z, y, x] and the writer wants [x, y, z], so one missing transpose produces a mask that
    is the right dtype, the right size in voxels and completely wrong. Nothing downstream would
    notice — on a non-cubic volume SimpleITK would build an image of the wrong dimensions, and on a
    cubic one it would build a plausible image of scrambled anatomy.
    """
    import SimpleITK as sitk

    inputs, outputs = tmp_path / "input", tmp_path / "output"
    (inputs / "images" / "some-slug").mkdir(parents=True)
    outputs.mkdir()
    uuid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    sitk.WriteImage(sitk.Image((8, 9, 10), sitk.sitkFloat32),
                    str(inputs / "images" / "some-slug" / f"{uuid}.mha"))

    monkeypatch.setenv("ISLES_INPUT_ROOT", str(inputs))
    monkeypatch.setenv("ISLES_OUTPUT_ROOT", str(outputs))

    assert _run_with(_WrongShape()) == 0
    written = sorted(outputs.rglob(f"*/{uuid}.mha"))
    assert written, "the case must still be answered, with an empty map"
    recovered = sitk.ReadImage(str(written[0]))
    assert recovered.GetSize() == (8, 9, 10), (
        f"the degraded output still has to match the input geometry; got {recovered.GetSize()}"
    )


class _BlobAndSpeck:
    """A big lesion plus a 8-voxel speck — what `min_voxels` exists to remove."""

    def load(self, device):
        pass

    def predict(self, *, image, image_path, device):
        import numpy as np

        probability = np.zeros(image.GetSize(), dtype=np.float32)
        probability[5:11, 5:11, 5:11] = 0.9       # 216 voxels, comfortably above the filter
        # Kept 4 voxels clear of the blob. Components are labelled with 26-CONNECTIVITY, so a speck
        # one voxel diagonally from the blob is part of it and the filter would never see it.
        probability[0:2, 0:2, 0:2] = 0.9          # 8 voxels, below the filter
        return probability


def test_the_container_applies_the_measured_post_processing_to_the_mask_only(tmp_path, monkeypatch):
    """The speck is dropped from the MASK; the probability map is written untouched.

    Both halves are the contract. PR-AUC is computed on the soft map, so filtering it would change a
    metric the filter was never evaluated against — and the sweep that chose `min_voxels=25` scored
    exactly this way: filtered mask, raw soft map (`docs/RESULTS.md`).
    """
    import SimpleITK as sitk

    inputs, outputs = tmp_path / "input", tmp_path / "output"
    (inputs / "images" / "some-slug").mkdir(parents=True)
    outputs.mkdir()
    uuid = "cccccccc-dddd-eeee-ffff-000000000000"
    sitk.WriteImage(sitk.Image((16, 16, 16), sitk.sitkFloat32),
                    str(inputs / "images" / "some-slug" / f"{uuid}.mha"))
    monkeypatch.setenv("ISLES_INPUT_ROOT", str(inputs))
    monkeypatch.setenv("ISLES_OUTPUT_ROOT", str(outputs))

    assert _run_with(_BlobAndSpeck()) == 0

    mask = sitk.GetArrayFromImage(sitk.ReadImage(
        str(outputs / "images" / "stroke-lesion-segmentation" / f"{uuid}.mha")))
    soft = sitk.GetArrayFromImage(sitk.ReadImage(
        str(outputs / "images" / "lesion-probability-map" / f"{uuid}.mha")))

    assert int(mask.sum()) == 216, (
        f"expected the 216-voxel blob to survive and the 8-voxel speck to be dropped; got "
        f"{int(mask.sum())} foreground voxels"
    )
    assert float(soft.max()) > 0.5 and int((soft > 0.5).sum()) == 224, (
        "the probability map must be written UNFILTERED — PR-AUC is scored on it"
    )
