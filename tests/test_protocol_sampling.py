"""Synthetic scans must be conditioned on a REAL protocol, drawn per site.

The protocol is a model input and the strongest one this model has: step 04's
gate measures a per-protocol group mean at held-out R2 76.3% / MAE 16.2s against
sequence_type's 18.7% / 59.9s. Until 2026-08-21, step 07 fed the model no
protocol at all — `sequence_generator` falls back to RARE_PROTOCOL_ID when the
key is absent, so doing nothing silently pinned every synthetic scan to the one
embedding row trained on the 3.4% of sequences too infrequent to earn their own.
Meanwhile the CSV's Protocol column carried one of eight invented names that
appear nowhere in the real corpus, so it could not be used as a shared Qlik
dimension either.
"""

import unittest

import numpy as np

from AlternatingPipeline.data.protocol_sampling import (
    MIN_POOL, ProtocolSampler, build_protocol_sampler,
)
from AlternatingPipeline.data.protocol_vocab import (
    RARE_PROTOCOL_ID, build_protocol_vocab,
)


def _seqs(name, serial, region, seq_type, n):
    return [{'protocol_name': name, 'serial_idx': serial,
             'body_region': region, 'sequence_type': seq_type} for _ in range(n)]


def _vocab(sequences, min_count=3):
    return build_protocol_vocab([s['protocol_name'] for s in sequences], min_count)


class ProtocolSamplerTests(unittest.TestCase):

    def setUp(self):
        # Two sites running disjoint protocol catalogues for the same
        # (region, type) — the real shape of this corpus, where only 4.1% of
        # names appear on more than one scanner.
        self.sequences = (
            _seqs('t1_tse_sag', serial=0, region=1, seq_type=2, n=20)
            + _seqs('t2_tse_tra', serial=1, region=1, seq_type=2, n=20)
        )
        self.vocab = _vocab(self.sequences)
        self.rng = np.random.default_rng(0)

    def _sampler(self):
        return ProtocolSampler(self.sequences, self.vocab, rng=self.rng)

    def test_the_draw_respects_the_scanner(self):
        """Serial is the FIRST backoff key, not an afterthought.

        A protocol drawn from the wrong site is a name that scanner has never
        run. That breaks the one thing the synthetic Protocol column is for.
        """
        sampler = self._sampler()
        self.assertEqual({sampler.sample(0, 1, 2)[1] for _ in range(30)},
                         {'t1_tse_sag'})
        self.assertEqual({sampler.sample(1, 1, 2)[1] for _ in range(30)},
                         {'t2_tse_tra'})

    def test_id_and_name_come_from_the_same_draw(self):
        """The id conditions the model, the name goes into the CSV. Drawing
        them separately would let the two disagree about what was generated."""
        sampler = self._sampler()
        for _ in range(30):
            pid, name = sampler.sample(0, 1, 2)
            self.assertEqual(pid, self.vocab[name.casefold()])

    def test_an_unseen_scanner_backs_off_instead_of_failing(self):
        sampler = self._sampler()
        drawn = {sampler.sample(9, 1, 2)[1] for _ in range(40)}
        self.assertTrue(drawn <= {'t1_tse_sag', 't2_tse_tra'})
        self.assertTrue(drawn)

    def test_a_thin_pool_is_skipped_rather_than_trusted(self):
        """MIN_POOL, same contract as SUTParameterSampler.

        One observation is not a distribution; drawing from it would make a
        single historical scan the protocol for every synthetic scan in that
        context.
        """
        thin = _seqs('rare_one_off', serial=5, region=3, seq_type=4, n=1)
        sequences = self.sequences + thin
        sampler = ProtocolSampler(sequences, _vocab(sequences, min_count=1),
                                  rng=np.random.default_rng(0))
        self.assertLess(1, MIN_POOL)
        drawn = {sampler.sample(5, 3, 4)[1] for _ in range(40)}
        self.assertNotIn('rare_one_off', drawn)

    def test_no_observations_at_all_yields_the_rare_bucket(self):
        """RARE_PROTOCOL_ID is what the model already substitutes for an absent
        key, so this stays in distribution; the empty name keeps the CSV honest
        about having invented nothing."""
        self.assertEqual(ProtocolSampler([], {}).sample(0, 0, 0),
                         (RARE_PROTOCOL_ID, ''))

    def test_unnamed_rows_never_become_a_drawable_outcome(self):
        """Rows with no protocol carry no protocol information.

        Pooling them would make "unknown" drawable at its corpus frequency
        rather than only where the pools genuinely run out.
        """
        sequences = self.sequences + [
            {'protocol_name': '', 'serial_idx': 0, 'body_region': 1, 'sequence_type': 2}
            for _ in range(50)
        ]
        sampler = ProtocolSampler(sequences, self.vocab, rng=np.random.default_rng(0))
        self.assertEqual(sampler.observations, 40)
        self.assertNotIn('', {sampler.sample(0, 1, 2)[1] for _ in range(40)})

    def test_ids_are_plain_ints_so_the_model_can_batch_them(self):
        """SequenceGeneratorModel._ensure_batched promotes scalars via
        isinstance(val, int), which a numpy integer fails — it would reach
        nn.Embedding unbatched."""
        pid, _ = self._sampler().sample(0, 1, 2)
        self.assertIs(type(pid), int)

    def test_describe_names_the_collapse_it_is_there_to_catch(self):
        self.assertIn('NO observations', ProtocolSampler([], {}).describe())
        described = self._sampler().describe()
        self.assertIn('2 distinct protocol names', described)

    def test_build_from_a_pkl_dict(self):
        self.assertIsNotNone(
            build_protocol_sampler({'examination': self.sequences}, self.vocab)
        )
        self.assertIsNone(build_protocol_sampler({}, self.vocab))
        self.assertIsNone(build_protocol_sampler({'examination': []}, self.vocab))


if __name__ == '__main__':
    unittest.main()
