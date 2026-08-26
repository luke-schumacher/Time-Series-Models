"""The pause between two scans of the same examination.

WHY THIS EXISTS
---------------
Measured on the ten real exam exports (41,106 scans, 4,747 visits, Jan 2024),
an MRI examination is 21.7 min of wall clock but only 15.1 min of scanning.
The other 30.4% is the gap between consecutive sequences: planning the next
slab off the localizer, repositioning, coaching a breath-hold, injecting
contrast.

`07_generate_synthetic_data.py` had no such gap. `_generate_exam_rows` returns
`t = msr_start_t + duration_sec`, and the per-patient scan loop feeds that
straight back as the next scan's start, so scan N+1 began the instant scan N
ended — 100% of generated intra-visit gaps were exactly 0.0s against a real 3%.
Per-scan durations can be perfectly calibrated and the EXAMINATION still comes
out ~30% short, which is exactly what happened.

This is generator work, not model work: the examination model is trained on, and
responsible for, the MRI_MSR_100 -> MRI_MSR_104 span alone. The patient handover
between visits is already modelled by the exchange block; only the pause *inside*
a visit was missing.

WHY AN EMPIRICAL QUANTILE GRID
------------------------------
The real distribution is heavy-tailed and bimodal-ish by scan type:

    all intra-visit gaps   n=36,325  mean 47.1s  median 19s
                           p10 1  p25 4  p75 47  p90 97  p99 468  max 3500
                           3.0% are exactly zero

    before a scout/localizer   mean 319s  median 168s   <- 5-10x everything else
    before a tse               mean  42s  median  15s
    before a vibe              mean  34s  median  12s

The pre-scout pause is large because a scout mid-visit is really a stage change
(new region, new coil), not a pause. A single mean would erase that; a fitted
lognormal would erase both the 3% zeros and the p99. So the prior is stored as a
101-point quantile grid per sequence type and sampled by inverse-CDF, which
reproduces the source distribution's shape by construction.

The gap is attributed to the scan it PRECEDES, because physically it is the
operator preparing that scan.
"""

import json
import os
from collections import defaultdict

import numpy as np

from AlternatingPipeline.config import SEQUENCE_TYPE_VOCAB, classify_sequence_type

# A type needs at least this many observed gaps to earn its own grid; below it
# the global grid is used. 200 keeps the p99 of a per-type grid meaningful
# (a 101-point grid over 40 samples is mostly interpolation between duplicates).
MIN_GAP_SAMPLES = 200

# Nothing longer than this is treated as an intra-visit pause. Real p99 is 468s
# and the observed max is 3500s; past ~30 min the scanner is idle, on a service
# interruption, or the visit segmentation is wrong, and letting a draw like that
# through would silently eat an hour of a generated day.
GAP_MAX_SEC = 1800.0

# Gaps at or beyond this are dropped at BUILD time for the same reason, plus the
# overnight case (end of one day, start of the next, StepCount not reset).
_GAP_BUILD_MAX_SEC = 3600.0

_QUANTILES = np.linspace(0.0, 100.0, 101)


def _f(v, default=None):
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _epoch(v):
    """Seconds for a timestamp that may be a number, a datetime, or a string."""
    n = _f(v)
    if n is not None:
        return n
    try:
        import pandas as pd
        ts = pd.to_datetime(v, errors='coerce')
        if ts is None or (hasattr(ts, '__len__') is False and pd.isna(ts)):
            return None
        return float(ts.timestamp())
    except Exception:
        return None


def gaps_from_exam_rows(rows):
    """Extract (gap_seconds, sequence_type_id) for every intra-visit gap.

    `rows` are exam-CSV records needing SN, PatientID, StepCount, startTime,
    duration and Sequence. A visit boundary is a StepCount that does not
    increase — the same rule used to measure the real reference figures, and the
    only one available without a patient-visit id in the CSV.
    """
    parsed = []
    for r in rows:
        start = _epoch(r.get('startTime'))
        step = _f(r.get('StepCount'))
        dur = _f(r.get('duration'), 0.0)
        if start is None or step is None or dur is None:
            continue
        parsed.append((str(r.get('SN', '')), start, step, dur,
                       classify_sequence_type(r.get('Sequence'))))

    out = []
    by_scanner = defaultdict(list)
    for p in parsed:
        by_scanner[p[0]].append(p)

    for _sn, items in by_scanner.items():
        items.sort(key=lambda p: p[1])
        for i in range(len(items) - 1):
            _, start_i, step_i, dur_i, _seq_i = items[i]
            _, start_j, step_j, _dur_j, seq_j = items[i + 1]
            # A new visit starts where StepCount stops increasing. The pause
            # across that boundary is the patient handover, which the exchange
            # model already generates; counting it here would inflate the
            # intra-visit prior by an order of magnitude (real handover median
            # 355s vs intra-visit median 19s).
            if step_j <= step_i:
                continue
            gap = start_j - (start_i + dur_i)
            # Negative means the rows overlap (bad timestamps); beyond the build
            # cap means a day boundary or an idle scanner, not a pause.
            if gap < 0.0 or gap >= _GAP_BUILD_MAX_SEC:
                continue
            out.append((float(gap), int(seq_j)))
    return out


def build_gap_quantiles(rows):
    """Build the serialisable prior from exam-CSV rows.

    Returns {'n', 'global': [101 quantiles], 'by_type': {seq_type_id: [...]},
    'stats': {...}} — plain JSON, so it can be written next to the models and
    read back by the generator without recomputing.
    """
    gaps = rows if (rows and isinstance(rows[0], tuple)) else gaps_from_exam_rows(rows)
    if not gaps:
        return {'n': 0, 'global': [], 'by_type': {}, 'stats': {}}

    allv = np.asarray([g for g, _ in gaps], dtype=float)
    by_type = defaultdict(list)
    for g, t in gaps:
        by_type[t].append(g)

    grids = {}
    for t, vals in by_type.items():
        if len(vals) >= MIN_GAP_SAMPLES:
            grids[str(int(t))] = [round(float(v), 3)
                                  for v in np.percentile(np.asarray(vals), _QUANTILES)]

    return {
        'n': int(len(allv)),
        'global': [round(float(v), 3) for v in np.percentile(allv, _QUANTILES)],
        'by_type': grids,
        'stats': {
            'mean': round(float(allv.mean()), 2),
            'median': round(float(np.median(allv)), 2),
            'p90': round(float(np.percentile(allv, 90)), 2),
            'p99': round(float(np.percentile(allv, 99)), 2),
            'zero_pct': round(100.0 * float((allv < 1.0).mean()), 2),
            'types_with_own_grid': sorted(int(k) for k in grids),
        },
    }


class IntraVisitGapSampler:
    """Inverse-CDF sampler over the per-sequence-type quantile grids."""

    def __init__(self, prior):
        self._prior = prior or {}
        self._global = list(self._prior.get('global') or [])
        self._by_type = {int(k): list(v)
                         for k, v in (self._prior.get('by_type') or {}).items()}

    @property
    def prior(self):
        """The serialisable dict this sampler was built from."""
        return self._prior

    @property
    def is_live(self):
        """False when no prior could be built — the generator then behaves
        exactly as it did before this module existed, rather than crashing."""
        return bool(self._global)

    @property
    def stats(self):
        return dict(self._prior.get('stats') or {})

    @property
    def n(self):
        return int(self._prior.get('n', 0))

    def sample(self, sequence_type_id, rng):
        """One gap in seconds, for the scan this pause precedes."""
        if not self._global:
            return 0.0
        grid = self._by_type.get(int(sequence_type_id)) or self._global
        # Interpolate between the two bracketing quantiles so the draw is
        # continuous rather than snapping to one of 101 stored values.
        u = float(rng.random()) * (len(grid) - 1)
        lo = int(u)
        hi = min(lo + 1, len(grid) - 1)
        val = grid[lo] + (u - lo) * (grid[hi] - grid[lo])
        return float(min(max(val, 0.0), GAP_MAX_SEC))

    def describe(self):
        s = self.stats
        if not self.is_live:
            return ('intra-visit gap prior: EMPTY — scans will be generated '
                    'back-to-back, as before')
        return (f"intra-visit gap prior: {self.n:,} real gaps  "
                f"mean {s.get('mean')}s  median {s.get('median')}s  "
                f"p99 {s.get('p99')}s  zero {s.get('zero_pct')}%  "
                f"own grid for {len(self._by_type)} sequence type(s)")


def build_gap_sampler_from_csvs(exam_csv_dir, pattern='DATA_*.csv', read_csv=None):
    """Build the prior straight from the real exam CSVs.

    Those are the same files the ±15s benchmark and the Qlik dashboard are
    measured on (`/dbfs/FileStore/csv_pipeline/exam/DATA_{serial}.csv`), so no
    Spark run and no pkl rebuild is needed to pick this up. Returns a dead
    sampler if the directory is absent, so a missing mount degrades to the old
    back-to-back behaviour instead of failing a long generation run.
    """
    import glob
    if read_csv is None:
        import pandas as pd
        read_csv = pd.read_csv

    paths = sorted(glob.glob(os.path.join(exam_csv_dir, pattern)))
    if not paths:
        return IntraVisitGapSampler(build_gap_quantiles([])), paths

    want = ['SN', 'PatientID', 'StepCount', 'startTime', 'duration', 'Sequence']
    gaps = []
    for p in paths:
        try:
            df = read_csv(p, usecols=lambda c: c in want, low_memory=False)
        except (OSError, ValueError):
            continue
        if df.empty:
            continue
        if 'SN' not in df.columns:
            # One CSV per serial, and the serial is in the filename.
            df = df.assign(SN=os.path.basename(p).replace('DATA_', '').replace('.csv', ''))
        gaps.extend(gaps_from_exam_rows(df.to_dict('records')))

    return IntraVisitGapSampler(build_gap_quantiles(gaps)), paths


def save_gap_prior(prior, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fh:
        json.dump(prior, fh, indent=2)
    return path


def load_gap_sampler(path):
    try:
        with open(path) as fh:
            return IntraVisitGapSampler(json.load(fh))
    except (OSError, ValueError):
        return IntraVisitGapSampler(build_gap_quantiles([]))
