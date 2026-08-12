"""Bootstrap confidence intervals — ported from `frozen_wmh.stats`, which this project lacked.

frozen-isles could report a median and rank two runs against each other, but had no way to say
whether a difference between two runs was distinguishable from noise. So every A/B here — 8 vs 12
epochs, upscale x2 vs x4, and now 468K vs 5.5M parameters — has been decided on a bare difference of
two point estimates.

**The rule this restores (user, 2026-07-14): one run plus a paired subject-level bootstrap comes
FIRST, before any seed fan-out.** The paired Δ is free once both runs exist, and it is what says
whether a lead is worth spending more compute on.

🚨 **The resampling unit is the SUBJECT.** The independence hierarchy is subject → volume → slices →
voxels; slices from one subject are not independent, so resampling below the subject inflates the
effective n and produces an artificially narrow interval. Every function here takes one whole-volume
metric per held-out subject.

**Pair by subject id, never by position.** `paired_delta_by_subject` builds the pairing from the
`SubjectScore` keys, so two runs that scored slightly different subject sets cannot be silently
mis-paired — the failure where shapes match and the delta lies. That trap has already cost this
group once, in a frozen-wmh loader that stacked mixed experimental conditions because the shapes
agreed.

`rank_then_aggregate` stays the rule for choosing a CHECKPOINT (it is the challenge's own objective).
This module answers a different question: is run A better than run B, and by more than noise.
"""

from collections.abc import Iterable, Mapping
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
from scipy.stats import bootstrap

from frozen_isles.metrics import SubjectScore

_N_RESAMPLES = 10_000
_CONFIDENCE = 0.95


class CI(NamedTuple):
    """A point estimate with its bootstrap confidence interval."""

    mean: float
    lo: float
    hi: float

    def excludes_zero(self) -> bool:
        """True iff the whole interval sits on one side of 0 — 'significant' for a Δ."""
        return self.lo > 0.0 or self.hi < 0.0

    def __str__(self) -> str:
        return f"{self.mean:.4f} [{self.lo:.4f}, {self.hi:.4f}]"


def mean_ci(values: npt.ArrayLike, *, confidence: float = _CONFIDENCE,
            n_resamples: int = _N_RESAMPLES, seed: int = 0) -> CI:
    """95% percentile-bootstrap CI on the mean of one metric per SUBJECT."""
    x = np.asarray(values, dtype=float)
    assert x.ndim == 1 and x.size >= 2, f"need a 1-D sample of >=2 values, got shape {x.shape}"
    assert np.isfinite(x).all(), "mean_ci: non-finite value(s) — filter inf/nan upstream"
    mean = float(np.mean(x))
    # EXACTLY-constant guard, deliberately not np.allclose: a near-constant sample must keep its real
    # width, or it collapses to a zero-width interval that reads as significant. Reachable here —
    # 24% of ISLES subjects score exactly 0.000 Dice, so a bad run's sample really can be constant.
    if np.all(x == x[0]):
        return CI(mean, mean, mean)
    result = bootstrap((x,), np.mean, confidence_level=confidence, n_resamples=n_resamples,
                       method="percentile", rng=np.random.default_rng(seed))
    return CI(mean, float(result.confidence_interval.low), float(result.confidence_interval.high))


def paired_delta(a: npt.ArrayLike, b: npt.ArrayLike, *, confidence: float = _CONFIDENCE,
                 n_resamples: int = _N_RESAMPLES, seed: int = 0) -> CI:
    """Paired bootstrap CI on the mean per-subject difference (a - b), caller-aligned.

    Δ>0 means `a` beats `b`. Prefer `paired_delta_by_subject`, which cannot be mis-aligned.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape and a.ndim == 1, f"a,b must be paired 1-D; got {a.shape},{b.shape}"
    return mean_ci(a - b, confidence=confidence, n_resamples=n_resamples, seed=seed)


def paired_delta_by_subject(
    a: Mapping[str, SubjectScore], b: Mapping[str, SubjectScore], metric: str, *,
    subjects: Iterable[str] | None = None, confidence: float = _CONFIDENCE,
    n_resamples: int = _N_RESAMPLES, seed: int = 0,
) -> CI:
    """Alignment-safe paired Δ-CI on `metric`, pairing by SUBJECT ID.

    Defaults to the subjects present in both runs, dropping any pair non-finite on either side.
    This is the canonical entry point for comparing two ISLES runs.
    """
    assert metric in SubjectScore.__dataclass_fields__, (
        f"{metric!r} is not a SubjectScore field; expected one of "
        f"{sorted(SubjectScore.__dataclass_fields__)}"
    )
    keys = sorted(set(a) & set(b)) if subjects is None else [s for s in subjects if s in a and s in b]
    pairs = [(getattr(a[s], metric), getattr(b[s], metric)) for s in keys]
    finite = [(x, y) for x, y in pairs
              if x is not None and y is not None and np.isfinite(x) and np.isfinite(y)]
    assert len(finite) >= 2, (
        f"paired Δ {metric!r}: only {len(finite)} finite common-subject pair(s); need >=2"
    )
    return paired_delta(np.array([x for x, _ in finite], dtype=float),
                        np.array([y for _, y in finite], dtype=float),
                        confidence=confidence, n_resamples=n_resamples, seed=seed)
