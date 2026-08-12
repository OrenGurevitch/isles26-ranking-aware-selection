"""Does DAYS_POST_STROKE carry information about difficulty BEYOND lesion size?

**The screen that gates a metadata retrain.** The challenge hands us `DAYS_POST_STROKE` at inference
and every model we have trained ignores it. Conditioning the POST-PROCESSING on it was tested and
closed negative, but a component filter can only delete what the network already found, so that result
says nothing about using the timing as a model INPUT.

Before paying for a retrain — new plans, a fold-0 train, an all-1,453 train, a container rebuild — this
asks the cheap prior question. `docs/RESULTS.md` establishes that acute cases fail most (median Dice
0.455, 29.6% zero) AND that their lesions are ~14x smaller in median volume, so timing is largely
**lesion size wearing a clock**. If days predicts nothing once true lesion size is known, an input
channel carrying days has nothing to teach the network.

⚠️ **This is a SCREEN, not a proof.** It can kill the idea cheaply. Passing it would NOT establish that
a conditioning channel works: a network could exploit an association this correlation cannot see, and a
scalar broadcast as a constant channel is a weak conditioning mechanism regardless.

⚠️ **Reference volume is the control on purpose.** Controlling for PREDICTED volume would answer a
different question — whether timing tells us anything the model's own output does not already imply —
which is the right control for a post-hoc rule and the wrong one for an input channel.

    uv run python scripts/timing_beyond_size.py \
        --scores /scratch/orengur2/nnunet_isles/scored/scores_shipped_min25.json \
        --inventory /scratch/orengur2/isles26/inventory.csv \
        --tree /scratch/orengur2/isles26/ATLAS3_Training_Raw
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from frozen_isles.data import discover

ACUTE_DAYS = 14.0
BOOTSTRAP_DRAWS = 2000
SEED = 0


class DegenerateInput(ValueError):
    """A variable is constant, so a rank correlation involving it is undefined rather than zero."""


def _ranks(values: np.ndarray) -> np.ndarray:
    # Average ties: Dice has a spike at exactly 0 (29 of 291 on the 250-epoch arm), and an ordinal
    # ranking inside that tie makes the correlation depend on the order subjects leave the JSON.
    return rankdata(values, method="average")


def _correlate(a: np.ndarray, b: np.ndarray, *, what: str) -> float:
    a, b = a - a.mean(), b - b.mean()
    denominator = math.sqrt(float((a**2).sum()) * float((b**2).sum()))
    if denominator == 0.0:
        raise DegenerateInput(f"{what}: a variable is constant, so the correlation is undefined")
    return float((a * b).sum() / denominator)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return _correlate(_ranks(a), _ranks(b), what="spearman")


def _residual(target: np.ndarray, design: np.ndarray) -> np.ndarray:
    # lstsq rather than a normal-equation solve: site one-hots go rank-deficient the moment a
    # bootstrap draw misses a site, and lstsq returns the least-norm solution instead of raising.
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    return target - design @ coefficients


def partial_spearman(
    x: np.ndarray, y: np.ndarray, controls: np.ndarray, *, rank_controls: np.ndarray | None = None
) -> float:
    """Rank association of x with y holding one or more controls fixed.

    `controls` is (n,) or (n, k). Continuous columns are rank-transformed here so that a caller cannot
    mix a raw control with ranked variables — doing so silently disagrees with the two-variable closed
    form by ~2e-3, and it made this script's own site-controlled bootstrap inconsistent with its
    volume-only one. `rank_controls` is a boolean mask over the columns; indicator columns pass through.
    """
    controls = np.atleast_2d(controls.T).T if controls.ndim == 1 else controls
    mask = np.ones(controls.shape[1], dtype=bool) if rank_controls is None else rank_controls
    assert mask.shape == (controls.shape[1],), "rank_controls must have one entry per control column"
    ranked = np.column_stack([
        _ranks(controls[:, j]) if mask[j] else controls[:, j] for j in range(controls.shape[1])
    ])
    design = np.column_stack([np.ones(len(x)), ranked])
    return _correlate(_residual(_ranks(x), design), _residual(_ranks(y), design),
                      what="partial_spearman")


def one_hot(labels: list[str]) -> np.ndarray:
    levels = sorted(set(labels))
    # Drop the first level; the intercept carries it and the design stays identifiable.
    return np.array([[1.0 if label == level else 0.0 for level in levels[1:]] for label in labels])


def reference_volumes_ml(inventory: Path) -> dict[str, float]:
    volumes: dict[str, float] = {}
    for row in csv.DictReader(inventory.open()):
        zooms = [float(z) for z in row["zooms"].split("x")]
        assert len(zooms) == 3, f"{row['subject_id']}: expected 3 zooms, got {row['zooms']}"
        voxel_mm3 = zooms[0] * zooms[1] * zooms[2]
        volumes[row["subject_id"]] = int(row["lesion_voxels"]) * voxel_mm3 / 1000.0
    return volumes


def centres(inventory: Path) -> dict[str, str]:
    return {row["subject_id"]: row["center"] for row in csv.DictReader(inventory.open())}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--model", default=None, help="key in the scores JSON; default is the first")
    parser.add_argument("--metric", default="dice")
    args = parser.parse_args()

    scored = json.loads(args.scores.read_text())
    model = args.model or next(iter(scored))
    per_subject = scored[model]
    volumes = reference_volumes_ml(args.inventory)
    centre_of = centres(args.inventory)
    days_by_subject = {s.subject_id: s.days_post_stroke for s in discover(args.tree)}

    subjects, metric, days, volume = [], [], [], []
    missing_days = 0
    for subject_id, row in per_subject.items():
        day = days_by_subject.get(subject_id)
        if day is None:
            missing_days += 1
            continue
        assert subject_id in volumes, f"{subject_id} scored but absent from the inventory"
        subjects.append(subject_id)
        metric.append(float(row[args.metric]))
        days.append(float(day))
        volume.append(volumes[subject_id])

    metric_array = np.array(metric)
    days_array = np.array(days)
    # log volume, because the size effect this controls for spans four orders of magnitude and a rank
    # correlation on the raw value would be identical anyway -- the log is for the reported medians.
    log_volume = np.log10(np.maximum(np.array(volume), 1e-3))

    print(f"=== {model}, metric '{args.metric}' ===")
    print(f"scored subjects: {len(per_subject)}; usable: {len(subjects)}; "
          f"dropped for missing DAYS_POST_STROKE: {missing_days}")
    print("⚠️  the dropped subjects are NOT missing at random -- missingness is site-structured "
          "(docs/RESULTS.md, 'The missingness is site-structured')\n")

    r_days = spearman(days_array, metric_array)
    r_volume = spearman(log_volume, metric_array)
    r_days_volume = spearman(days_array, log_volume)
    r_partial = partial_spearman(days_array, metric_array, log_volume)

    print(f"Spearman  days      vs {args.metric:<10} : {r_days:+.4f}")
    print(f"Spearman  log volume vs {args.metric:<10} : {r_volume:+.4f}")
    print(f"Spearman  days      vs log volume    : {r_days_volume:+.4f}")
    print(f"PARTIAL   days      vs {args.metric} | log volume : {r_partial:+.4f}")

    rng = np.random.default_rng(SEED)
    draws = np.empty(BOOTSTRAP_DRAWS)
    for i in range(BOOTSTRAP_DRAWS):
        pick = rng.integers(0, len(subjects), len(subjects))
        draws[i] = partial_spearman(days_array[pick], metric_array[pick], log_volume[pick])
    low, high = np.nanpercentile(draws, [2.5, 97.5])
    print(f"          95% CI over {BOOTSTRAP_DRAWS} subject bootstraps: [{low:+.4f}, {high:+.4f}]")
    print("          → CI containing 0 means timing adds nothing measurable beyond lesion size here\n")

    # 🔴 Acute subjects concentrate in particular centres, so the association above could be site
    # difficulty rather than lesion age. Adding site as a control separates them.
    site_labels = [centre_of[s] for s in subjects]
    site_columns = one_hot(site_labels)
    # log_volume goes in RAW and is rank-transformed inside partial_spearman, so a bootstrap draw
    # ranks it within the draw exactly as the volume-only loop above does. Passing pre-computed
    # full-sample ranks here made the two loops disagree, since a resample is not a subset.
    controls = np.column_stack([log_volume, site_columns])
    rank_controls = np.array([True] + [False] * site_columns.shape[1])
    r_site = partial_spearman(days_array, metric_array, controls, rank_controls=rank_controls)
    site_draws = np.empty(BOOTSTRAP_DRAWS)
    for i in range(BOOTSTRAP_DRAWS):
        pick = rng.integers(0, len(subjects), len(subjects))
        site_draws[i] = partial_spearman(
            days_array[pick], metric_array[pick], controls[pick], rank_controls=rank_controls
        )
    site_low, site_high = np.nanpercentile(site_draws, [2.5, 97.5])
    print(f"CONTROLLING FOR SITE AS WELL ({len(set(site_labels))} centres, "
          f"{site_columns.shape[1]} indicator columns)")
    print(f"PARTIAL   days vs {args.metric} | log volume + site : {r_site:+.4f}")
    print(f"          95% CI over {BOOTSTRAP_DRAWS} subject bootstraps: "
          f"[{site_low:+.4f}, {site_high:+.4f}]")
    print("          → if this collapses toward 0, the timing effect was site difficulty\n")

    # 🔴 A collapse has TWO explanations and they are not the same finding: either site difficulty
    # really explains the timing effect, or acute subjects sit in so few centres that site indicators
    # absorb them by construction. Only centres holding BOTH acute and later subjects can identify a
    # within-site timing effect at all, so count them before reading the number above.
    acute_mask = days_array < ACUTE_DAYS
    by_site: dict[str, list[bool]] = {}
    for label, is_acute in zip(site_labels, acute_mask, strict=True):
        by_site.setdefault(label, []).append(bool(is_acute))
    both = {s: v for s, v in by_site.items() if any(v) and not all(v)}
    acute_total = int(acute_mask.sum())
    acute_inside = sum(sum(v) for v in both.values())
    print(f"identifiability: {acute_total} acute subjects across "
          f"{sum(1 for v in by_site.values() if any(v))} centres")
    print(f"  centres holding BOTH acute and later subjects: {len(both)} of {len(by_site)}")
    print(f"  acute subjects inside those centres: {acute_inside} of {acute_total} "
          f"({100 * acute_inside / max(acute_total, 1):.0f}%)")
    print("  → the site-controlled estimate rests ONLY on these; few of them means the collapse is\n"
          "    non-identifiability rather than evidence that site explains the effect\n")

    # Stratify, because a single coefficient hides whether the acute deficit survives inside a band.
    tertiles = np.quantile(log_volume, [1 / 3, 2 / 3])
    band = np.digitize(log_volume, tertiles)
    print(f"{'volume tertile':<18} {'n':>4} {'n acute':>8} {'median ' + args.metric + ' acute':>22} "
          f"{'median later':>14}")
    for b, name in enumerate(("small", "medium", "large")):
        inside = band == b
        acute = inside & (days_array < ACUTE_DAYS)
        later = inside & (days_array >= ACUTE_DAYS)
        acute_median = np.median(metric_array[acute]) if acute.sum() else float("nan")
        later_median = np.median(metric_array[later]) if later.sum() else float("nan")
        print(f"{name:<18} {inside.sum():>4} {acute.sum():>8} {acute_median:>22.4f} "
              f"{later_median:>14.4f}")
    print("\n⚠️  a band with very few acute subjects cannot separate them; read n acute before the medians.")


if __name__ == "__main__":
    main()
