"""The pause between two scans of the SAME examination must exist.

Georg, 2026-08-26: "our mean duration is 2 mins, in real life it would be closer
to 20." Measured on the ten real exam exports (41,106 scans, 4,747 visits), a
real examination is 21.7 min of wall clock, of which only 15.1 min is scanning —
30.4% is the gap between consecutive sequences (planning, positioning,
breath-hold coaching, contrast). The generator had no such gap at all: in
`_generate_exam_rows` the clock advances by `duration_sec` alone, so scan N+1
began the instant scan N ended and 100% of intra-visit gaps were exactly 0.0s.
Per-scan durations were right and the EXAMINATION still came out ~30% short.

The gap is heavy-tailed (median 19s, mean 47s, p99 468s) and is NOT flat across
scan types: the pause before a scout/localizer is 5-10x the rest (median 168s vs
19s) because it is really a stage change inside the visit. So it is sampled from
an empirical per-sequence-type quantile grid rather than a mean or a fitted
lognormal, either of which would erase both facts.
"""

import json
import math
import os
import tempfile
import unittest

import numpy as np

from AlternatingPipeline.config import SEQUENCE_TYPE_VOCAB
from AlternatingPipeline.data.intra_visit_gap import (
    GAP_MAX_SEC, MIN_GAP_SAMPLES, IntraVisitGapSampler,
    build_gap_quantiles, gaps_from_exam_rows, load_gap_sampler,
)


def _row(sn, pid, step, start, dur, seq='tse'):
    """One exam CSV row, in the columns build_gap_quantiles actually reads."""
    return {'SN': sn, 'PatientID': pid, 'StepCount': step, 'Sequence': seq,
            'startTime': start, 'duration': dur}


def _visit(sn, pid, t0, scans, gap, seq='tse', dur=100):
    """`scans` scans of `dur` seconds separated by `gap` seconds."""
    rows, t = [], t0
    for i in range(scans):
        rows.append(_row(sn, pid, i + 1, t, dur, seq))
        t += dur + gap
    return rows


class TestGapExtraction(unittest.TestCase):

    def test_gap_is_the_gap_between_scans_of_one_visit(self):
        rows = _visit('A', 'p1', 0, scans=4, gap=30)
        got = gaps_from_exam_rows(rows)
        # 4 scans -> 3 intra-visit gaps, all 30s
        self.assertEqual([g for g, _ in got], [30.0, 30.0, 30.0])

    def test_gap_is_attributed_to_the_scan_it_precedes(self):
        # The pause is the operator setting up the NEXT sequence, so it belongs
        # to the next scan's type. Attributing it backwards would put the big
        # pre-scout pause on whatever happened to run before the scout.
        rows = (_visit('A', 'p1', 0, scans=1, gap=0, seq='tse')
                + [_row('A', 'p1', 2, 200, 100, 'AAHScout')])
        rows[0]['duration'] = 100
        got = gaps_from_exam_rows(rows)
        self.assertEqual(len(got), 1)
        _, seq_id = got[0]
        self.assertEqual(seq_id, SEQUENCE_TYPE_VOCAB['scout'])

    def test_patient_handover_is_not_an_intra_visit_gap(self):
        # StepCount resetting to 1 starts a new visit; the pause across that
        # boundary is the exchange block, which the generator already models.
        rows = _visit('A', 'p1', 0, scans=2, gap=10) + _visit('A', 'p2', 5000, scans=2, gap=10)
        got = gaps_from_exam_rows(rows)
        self.assertEqual([g for g, _ in got], [10.0, 10.0])

    def test_scanners_are_segmented_independently(self):
        rows = _visit('A', 'p1', 0, scans=2, gap=10) + _visit('B', 'p9', 0, scans=2, gap=10)
        self.assertEqual(len(gaps_from_exam_rows(rows)), 2)

    def test_negative_and_absurd_gaps_are_dropped(self):
        rows = [_row('A', 'p1', 1, 0, 100), _row('A', 'p1', 2, 50, 100)]      # overlapping
        self.assertEqual(gaps_from_exam_rows(rows), [])
        rows = [_row('A', 'p1', 1, 0, 100), _row('A', 'p1', 2, 99999, 100)]   # overnight
        self.assertEqual(gaps_from_exam_rows(rows), [])

    def test_zero_gaps_are_kept(self):
        # 3.0% of real intra-visit gaps are genuinely 0s. Dropping them would
        # bias the sampled distribution upward.
        rows = _visit('A', 'p1', 0, scans=3, gap=0)
        self.assertEqual([g for g, _ in gaps_from_exam_rows(rows)], [0.0, 0.0])


class TestQuantileBuild(unittest.TestCase):

    def test_per_type_grid_when_there_is_enough_data(self):
        rows = _visit('A', 'p1', 0, scans=MIN_GAP_SAMPLES + 2, gap=25, seq='tse')
        q = build_gap_quantiles(rows)
        self.assertIn(str(SEQUENCE_TYPE_VOCAB['tse']), q['by_type'])
        self.assertTrue(all(abs(v - 25) < 1e-6 for v in q['by_type'][str(SEQUENCE_TYPE_VOCAB['tse'])]))

    def test_thin_types_fall_back_to_global_rather_than_inventing_a_grid(self):
        rows = (_visit('A', 'p1', 0, scans=MIN_GAP_SAMPLES + 2, gap=25, seq='tse')
                + _visit('A', 'p2', 900000, scans=3, gap=300, seq='space'))
        q = build_gap_quantiles(rows)
        self.assertNotIn(str(SEQUENCE_TYPE_VOCAB['space']), q['by_type'])
        self.assertIn('global', q)

    def test_empty_input_produces_an_empty_prior_not_a_crash(self):
        q = build_gap_quantiles([])
        self.assertEqual(q['n'], 0)
        self.assertEqual(q['by_type'], {})


class TestSampling(unittest.TestCase):

    def _sampler(self, gaps, seq='tse'):
        rows, t = [], 0
        for i, g in enumerate(gaps):
            rows.append(_row('A', 'p1', i + 1, t, 10, seq))
            t += 10 + g
        rows.append(_row('A', 'p1', len(gaps) + 1, t, 10, seq))
        return IntraVisitGapSampler(build_gap_quantiles(rows))

    def test_reproduces_the_median_of_its_source(self):
        src = list(np.random.default_rng(1).lognormal(3.0, 1.2, 4000))
        s = self._sampler(src)
        rng = np.random.default_rng(0)
        drawn = [s.sample(SEQUENCE_TYPE_VOCAB['tse'], rng) for _ in range(6000)]
        self.assertAlmostEqual(np.median(drawn), np.median(src), delta=0.12 * np.median(src))

    def test_keeps_the_heavy_tail(self):
        # A mean or a normal fit would flatten p99; the whole point of the
        # quantile grid is that it does not.
        src = list(np.random.default_rng(2).lognormal(3.0, 1.4, 4000))
        s = self._sampler(src)
        rng = np.random.default_rng(0)
        drawn = [s.sample(SEQUENCE_TYPE_VOCAB['tse'], rng) for _ in range(8000)]
        self.assertGreater(np.percentile(drawn, 99), 0.6 * np.percentile(src, 99))

    def test_never_negative_and_never_past_the_cap(self):
        s = self._sampler([0.0] * 50 + [10_000.0] * 50)
        rng = np.random.default_rng(0)
        for _ in range(500):
            g = s.sample(SEQUENCE_TYPE_VOCAB['tse'], rng)
            self.assertGreaterEqual(g, 0.0)
            self.assertLessEqual(g, GAP_MAX_SEC)

    def test_unknown_type_falls_back_to_global_instead_of_zero(self):
        s = self._sampler([40.0] * (MIN_GAP_SAMPLES + 5))
        rng = np.random.default_rng(0)
        drawn = [s.sample(SEQUENCE_TYPE_VOCAB['medic'], rng) for _ in range(200)]
        self.assertAlmostEqual(float(np.mean(drawn)), 40.0, delta=2.0)

    def test_an_empty_prior_samples_zero_and_says_so(self):
        # A missing exam-CSV directory must degrade to today's behaviour, not
        # crash a six-hour generation run.
        s = IntraVisitGapSampler(build_gap_quantiles([]))
        self.assertFalse(s.is_live)
        rng = np.random.default_rng(0)
        self.assertEqual(s.sample(SEQUENCE_TYPE_VOCAB['tse'], rng), 0.0)

    def test_sampling_is_deterministic_for_a_seeded_rng(self):
        s = self._sampler([5.0, 10.0, 15.0, 20.0] * 30)
        a = [s.sample(SEQUENCE_TYPE_VOCAB['tse'], np.random.default_rng(7)) for _ in range(5)]
        b = [s.sample(SEQUENCE_TYPE_VOCAB['tse'], np.random.default_rng(7)) for _ in range(5)]
        self.assertEqual(a, b)


class TestRoundTrip(unittest.TestCase):

    def test_prior_survives_json(self):
        rows = _visit('A', 'p1', 0, scans=MIN_GAP_SAMPLES + 2, gap=33, seq='tse')
        q = build_gap_quantiles(rows)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 'gap.json')
            with open(p, 'w') as fh:
                json.dump(q, fh)
            s = load_gap_sampler(p)
        self.assertTrue(s.is_live)
        rng = np.random.default_rng(0)
        self.assertAlmostEqual(s.sample(SEQUENCE_TYPE_VOCAB['tse'], rng), 33.0, delta=1e-6)

    def test_missing_file_returns_a_dead_sampler(self):
        s = load_gap_sampler('/nonexistent/gap.json')
        self.assertFalse(s.is_live)


if __name__ == '__main__':
    unittest.main()
