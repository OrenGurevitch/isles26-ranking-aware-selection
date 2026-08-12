"""Reading and writing across the Grand Challenge container boundary, where the contract is unknown.

The platform hands a container `/input` and collects `/output`. The exact socket names — the directory
level under `images/` — are set per challenge, and ISLES'26 has not published its Docker template. The
one public ISLES'26 container guesses three output names, which is what someone does when they cannot
find the spec either. **We get exactly one test-phase submission**, so guessing is not a strategy.

The way out is that the sanity-check phase allows unlimited submissions. So the first container we
send is not a model — it is an INSTRUMENT. `describe_environment` dumps everything the container can
observe, and `write_prediction` writes to every candidate socket at once. The sanity run's logs then
tell us the real input socket, whether metadata arrives, the image format, and which output the
platform accepted. After that we build the real container against facts.
"""

import json
import os
import platform
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import SimpleITK as sitk

# Where Grand Challenge mounts the case. Overridable by environment variable so the container can be
# exercised against a simulated mount locally — the same code path, not a parallel one. These are read
# through `input_root()` / `output_root()` rather than bound as default arguments, because a default is
# captured at import time and then cannot be redirected, which is how the first local run silently
# wrote to the real /output.
_DEFAULT_INPUT = Path("/input")
_DEFAULT_OUTPUT = Path("/output")


def input_root() -> Path:
    return Path(os.environ.get("ISLES_INPUT_ROOT", _DEFAULT_INPUT))


def output_root() -> Path:
    return Path(os.environ.get("ISLES_OUTPUT_ROOT", _DEFAULT_OUTPUT))

# CONFIRMED from the organizers' own ISLES'22 template (github.com/ezequieldlrosa/isles22-docker-template,
# which they named on the forum as what the ISLES'26 template will look like): the segmentation slug is
# `stroke-lesion-segmentation`, the file must be named after the INPUT file, and /output/results.json
# must declare what was written.
SEGMENTATION_SOCKET = "stroke-lesion-segmentation"

# The remaining names are written by BOTH entrypoints — `container/run.py` is shared by the probe and
# the nnU-Net submission — harmlessly, in case ISLES'26 renamed the slug.
CANDIDATE_OUTPUT_SOCKETS: tuple[str, ...] = (
    SEGMENTATION_SOCKET,
    "lesion-segmentation",
    "brain-lesion-segmentation",
    "ischemic-stroke-lesion-segmentation",
    "infarct-segmentation",
)

# ✅ **RESOLVED 2026-08-02 from the Sanity Check phase's own submission page**, which lists the
# required outputs as **"Stroke Lesion Segmentation"** and **"Lesion Probability Map"** — and says an
# algorithm "need[s] to implement ALL of the following input-output combinations". So the probability
# map is REQUIRED, not optional, and PR-AUC really is one of the five scored metrics. The design
# document was right and the Info page's mask-only listing was incomplete.
#
# `PROBABILITY_SOCKET` is the confirmed one and leads the list. The others stay because the phase page
# shows DISPLAY names and the slug is inferred from them; writing the alternates costs a few MB and
# removes the only way this could still be wrong. Trim once a real submission confirms the slug.
PROBABILITY_SOCKET = "lesion-probability-map"
CANDIDATE_PROBABILITY_SOCKETS: tuple[str, ...] = (
    PROBABILITY_SOCKET,
    "stroke-lesion-probability",
    "stroke-lesion-segmentation-probability",
)

_IMAGE_SUFFIXES = (".mha", ".mhd", ".nii.gz", ".nii", ".nrrd")


def find_input_image(root: Path | None = None) -> Path:
    """The single input volume, wherever the platform mounted it.

    Searched by suffix across the whole tree rather than at a fixed path, because the input socket
    name is exactly what we do not know yet. Fails loudly with the tree listing attached — a container
    that cannot find its input must say what it DID see, or the sanity run teaches us nothing.
    """
    root = root if root is not None else input_root()
    # Collected across ALL suffixes before choosing. Looping and returning the first hit would take a
    # .mha and never notice a .nii.gz beside it, while the assert below reads as if it had.
    found = {suffix: sorted(p for p in root.rglob(f"*{suffix}") if p.is_file())
             for suffix in _IMAGE_SUFFIXES}
    matches = [p for paths in found.values() for p in paths]
    if matches:
        assert len(matches) == 1, f"expected one input volume, found {[str(m) for m in matches]}"
        return matches[0]
    raise FileNotFoundError(
        f"no volume with suffix {_IMAGE_SUFFIXES} under {root}. Tree:\n" + describe_tree(root)
    )


def read_metadata(root: Path | None = None) -> dict[str, object]:
    """The per-case JSON — DAYS_POST_STROKE, CHRONICITY, CENTER.

    ✅ **CONFIRMED from the sanity-phase log, 2026-08-02**: the file is `/input/stroke-metadata.json`,
    at the TOP level and not under `images/`. Observed contents, one line each:

        {'CENTER': 'SOOP', 'CHRONICITY': 1,    'DAYS_POST_STROKE': None}
        {'CENTER': 'R001', 'CHRONICITY': None, 'DAYS_POST_STROKE': 471.0}

    ⚠️ **A case may carry CHRONICITY or DAYS_POST_STROKE and not both** — one of the two sanity cases
    had each. Any consumer must handle either being None.

    ⚠️ **`/input` also holds `inputs.json`, which is the platform's INTERFACE descriptor, not ours.**
    This used to merge every `*.json` it found, and was saved from absorbing that file only by the
    `isinstance(dict)` guard below — GC ships it as an array. That is luck, not design, so the named
    file is preferred now and the merge is the fallback.

    Returns {} when absent rather than raising: the metadata is a documented input, but a container
    that dies because an optional file is missing scores the worst rank on that case.
    """
    root = root if root is not None else input_root()
    named = sorted(root.rglob("*metadata*.json"))
    # One file is documented. Two would mean the platform changed something, and a silent
    # last-wins merge would hide it behind whichever name sorts last.
    if len(named) > 1:
        print(f"[metadata] ⚠️ {len(named)} metadata files, expected 1: {[str(p) for p in named]}. "
              "Merging in sorted order; later files win.", flush=True)
    candidates = named or sorted(root.rglob("*.json"))
    if not named:
        print("[metadata] no *metadata*.json found; merging every JSON under the input root, which "
              "may include the platform's own descriptors", flush=True)
    merged: dict[str, object] = {}
    for path in candidates:
        try:
            content = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[metadata] unreadable {path}: {exc}", flush=True)
            continue
        if isinstance(content, dict):
            merged.update(content)
    return merged


def describe_tree(root: Path) -> str:
    if not root.exists():
        return f"  {root} does not exist"
    lines = [f"  {root} (root)"]
    for path in sorted(root.rglob("*")):
        size = f"{path.stat().st_size:,} B" if path.is_file() else "dir"
        lines.append(f"  {path}  [{size}]")
    return "\n".join(lines) if len(lines) > 1 else f"  {root} is empty"


def describe_environment(inputs: Path | None = None, outputs: Path | None = None) -> str:
    """Everything the container can observe about the platform. The point of the sanity submission."""
    inputs = inputs if inputs is not None else input_root()
    outputs = outputs if outputs is not None else output_root()
    sections = [
        "=== PLATFORM ===",
        f"  python      {platform.python_version()} on {platform.platform()}",
        f"  cpus        {os.cpu_count()}",
        f"  cwd / user  {Path.cwd()} / uid={os.getuid()} gid={os.getgid()}",
        f"  disk free   {shutil.disk_usage('/').free / 1e9:.1f} GB",
        "=== GPU ===",
        f"  {_describe_gpu()}",
        "=== INPUT TREE ===",
        describe_tree(inputs),
        "=== OUTPUT TREE (before writing) ===",
        describe_tree(outputs),
        "=== ENVIRONMENT VARIABLES ===",
    ]
    # Values are not printed: a platform may pass credentials through the environment, and this log is
    # written to a service we do not control. Names alone answer what we need.
    sections.append("  " + ", ".join(sorted(os.environ)))
    return "\n".join(sections)


def _describe_gpu() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        return (out.stdout or out.stderr).strip() or "nvidia-smi returned nothing"
    except (OSError, subprocess.SubprocessError) as exc:
        return f"no nvidia-smi ({exc})"


def write_prediction(
    *,
    array: npt.NDArray,
    reference: sitk.Image,
    sockets: Sequence[str],
    filename: str,
    outputs: Path | None = None,
) -> list[Path]:
    """Write one volume to every candidate socket, preserving the reference's spatial metadata.

    `filename` MUST be the input image's own filename. The organizers' template names the output after
    the input's UUID, not something fixed like "output.mha", and `results.json` refers to it by that
    name — a mismatch is a failed case.

    `array` is indexed [x, y, z] to match `sitk.Image.GetSize()`; SimpleITK's buffer order is the
    reverse, so it is transposed here rather than at each call site.
    """
    assert tuple(array.shape) == tuple(reference.GetSize()), (
        f"prediction shape {array.shape} does not match the input's {reference.GetSize()} — "
        "the challenge scores in the input's native space"
    )
    image = sitk.GetImageFromArray(array.transpose(2, 1, 0))
    image.CopyInformation(reference)

    outputs = outputs if outputs is not None else output_root()
    written = []
    for socket in sockets:
        directory = outputs / "images" / socket
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename
        # Compressed: eight candidate sockets of an uncompressed float volume is ~150 MB of
        # near-empty output per case, and the platform copies all of it.
        sitk.WriteImage(image, str(path), useCompression=True)
        written.append(path)
        print(f"[out] {path}", flush=True)
    return written


def binarise(probability: npt.NDArray, threshold: float = 0.5) -> npt.NDArray[np.uint8]:
    return (probability >= threshold).astype(np.uint8)


def write_results_json(
    *, output_filename: str, input_filename: str, input_socket: str,
    outputs: Path | None = None,
) -> Path:
    """`/output/results.json` — the manifest Grand Challenge reads to find what the container produced.

    Omitting it is not a degraded submission, it is a failed one: the platform learns about outputs
    from this file, not by scanning directories. Shape copied from the organizers' template.

    ⚠️ **BOTH outputs are declared, because the phase requires both.** The Sanity Check submission page
    states an algorithm "need[s] to implement ALL of the following input-output combinations" and
    lists **Stroke Lesion Segmentation** *and* **Lesion Probability Map** [read 2026-08-02]. This
    function used to announce the mask alone while the container wrote both files — the platform would
    have been told about half of what we produced, on a phase that demands both.
    """
    outputs = outputs if outputs is not None else output_root()
    outputs.mkdir(parents=True, exist_ok=True)
    path = outputs / "results.json"
    path.write_text(json.dumps([{
        "outputs": [
            {"type": "Image", "slug": SEGMENTATION_SOCKET, "filename": output_filename},
            {"type": "Image", "slug": PROBABILITY_SOCKET, "filename": output_filename},
        ],
        "inputs": [{"type": "Image", "slug": input_socket, "filename": input_filename}],
    }]))
    print(f"[out] {path}", flush=True)
    return path


def empty_prediction(reference: sitk.Image) -> npt.NDArray[np.float32]:
    """An EXACTLY constant zero map — how to say "there is no lesion here" and be paid for it.

    The organizers confirmed an empty ground truth scores 1 rather than being excluded. Measured in
    `metrics/test.py` against their own code, that splits into two separate wins, and conflating them
    overstates the second:

    - **Predicting nothing at all** on a lesion-free case takes Dice, lesion-F1, count difference and
      volume difference — four metrics — because those are computed from the THRESHOLDED mask, and an
      empty mask matches an empty reference exactly. Any soft map that thresholds to nothing gets this.
    - **Making the map exactly constant** adds only PR-AUC, the fifth. `compute_pr_auc` grants its 1.0
      only when `np.all(pred == pred[0])`; near-zero varying noise scores 0.0 there while still
      collecting the other four.

    So the large lever is recognising "no lesion here" at all; the constant map is one further metric
    on top. One of five is still worth having under rank-then-aggregate, where every case weighs the
    same — hence a dedicated constructor rather than `probability * 0` or a threshold that happens to
    zero everything, since those are constant only by accident.
    """
    return np.zeros(reference.GetSize(), dtype=np.float32)


def is_exactly_constant(probability: npt.NDArray) -> bool:
    """Whether the shipped PR-AUC would treat this as a clean "nothing here" on an empty reference."""
    flat = np.asarray(probability).ravel()
    return bool(flat.size and np.all(flat == flat[0]))
