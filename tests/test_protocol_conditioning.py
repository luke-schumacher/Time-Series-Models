"""The examination model must see the protocol, and see it where it counts.

`sequence_type` is the only conditioning channel this model has ever responded
to, and the reason is structural: it is the only one with a per-position path
into the duration encoder (`duration_seq_type_bias`). Every channel that rides
the conditioning token alone measures ~0.000s under perturbation. Protocol is
now the dominant duration signal (82.0% of variance held out vs 31.2% for
sequence_type), so it gets the same per-position treatment, and these tests pin
that it is actually wired that way rather than merely present in the config.
"""

import os
import unittest

import torch

from AlternatingPipeline.config import PAD_TOKEN_ID, SOURCEID_VOCAB
from AlternatingPipeline.models.sequence_generator import SequenceGeneratorModel


NUM_PROTOCOLS = 7


def _config(num_protocols=NUM_PROTOCOLS):
    config = {
        'model_type': 'examination',
        'vocab_size': len(SOURCEID_VOCAB),
        'd_model': 16,
        'nhead': 2,
        'num_encoder_layers': 1,
        'num_decoder_layers': 1,
        'num_duration_encoder_layers': 1,
        'dim_feedforward': 32,
        'dropout': 0.0,
        'max_seq_len': 8,
        'base_conditioning_dim': 2,
        'num_body_regions': 3,
        'num_region_classes': 4,
        'has_phase_type': False,
        'body_region_mode': 'single',
        'duration_mode': 'log',
        'use_exam_conditioning': True,
        'num_sequence_types': 4,
        'num_serials': 2,
    }
    if num_protocols is not None:
        config['num_protocols'] = num_protocols
    return config


def _batch(protocol=0, batch=2, seq_len=5):
    tokens = torch.full((batch, seq_len), SOURCEID_VOCAB['MRI_MSR_100'], dtype=torch.long)
    conditioning = torch.zeros(batch, 2)
    info = {
        'body_region': torch.zeros(batch, dtype=torch.long),
        'sequence_type': torch.zeros(batch, dtype=torch.long),
        'serial_idx': torch.zeros(batch, dtype=torch.long),
        'protocol': torch.full((batch,), protocol, dtype=torch.long),
    }
    return tokens, conditioning, info


class ProtocolWiringTests(unittest.TestCase):
    def test_protocol_params_exist_only_when_configured(self):
        with_protocol = SequenceGeneratorModel(_config())
        self.assertTrue(hasattr(with_protocol, 'protocol_embedding'))
        self.assertTrue(hasattr(with_protocol, 'duration_protocol_bias'))

        without = SequenceGeneratorModel(_config(num_protocols=None))
        self.assertFalse(hasattr(without, 'protocol_embedding'))
        self.assertFalse(hasattr(without, 'duration_protocol_bias'))

    def test_duration_protocol_bias_starts_at_exactly_zero(self):
        """So a checkpoint trained before this existed loads and predicts
        identically — the same contract duration_seq_type_bias has."""
        model = SequenceGeneratorModel(_config())
        self.assertTrue(torch.all(model.duration_protocol_bias.weight == 0))

    def test_a_pre_protocol_checkpoint_still_loads_and_predicts_identically(self):
        """No shape may change, or steps 05/06/07 break for a whole retrain.

        Concatenating protocol onto the conditioning input would widen
        conditioning_projection, and PyTorch rejects a size mismatch regardless
        of `strict` — pinned by test_checkpoint_compat. Hence the additive,
        zero-initialised path.
        """
        torch.manual_seed(0)
        old = SequenceGeneratorModel(_config(num_protocols=None)).eval()
        new = SequenceGeneratorModel(_config()).eval()

        old_state = old.state_dict()
        new_state = new.state_dict()
        for name, tensor in old_state.items():
            self.assertIn(name, new_state)
            self.assertEqual(tensor.shape, new_state[name].shape, name)

        missing, unexpected = new.load_state_dict(old_state, strict=False)
        self.assertEqual(unexpected, [])
        self.assertTrue(all('protocol' in name for name in missing), missing)

        tokens, conditioning, info = _batch(protocol=4)
        old_info = {k: v for k, v in info.items() if k != 'protocol'}
        with torch.no_grad():
            mu_old, sigma_old = old.estimate_durations(tokens, conditioning, old_info)
            mu_new, sigma_new = new.estimate_durations(tokens, conditioning, info)
        torch.testing.assert_close(mu_old, mu_new)
        torch.testing.assert_close(sigma_old, sigma_new)

    def test_protocol_reaches_the_duration_head_once_trained(self):
        """The whole point. A trained bias must move the duration prediction —
        TR and num_slices were correctly configured and still moved it 0.002s
        because they only ever rode the conditioning token."""
        torch.manual_seed(0)
        model = SequenceGeneratorModel(_config()).eval()
        with torch.no_grad():
            model.duration_protocol_bias.weight.normal_(0, 1.0)

        tokens, conditioning, info = _batch(protocol=1)
        with torch.no_grad():
            mu_a, _ = model.estimate_durations(tokens, conditioning, info)
            info_b = dict(info, protocol=torch.full((tokens.shape[0],), 5, dtype=torch.long))
            mu_b, _ = model.estimate_durations(tokens, conditioning, info_b)

        self.assertGreater((mu_a - mu_b).abs().max().item(), 1e-3)

    def test_gradient_flows_into_both_zero_initialised_paths(self):
        """Zero-init plus no gradient would make the retrain silently pointless.

        protocol_cond_proj is the risky one: its OUTPUT is zero at init, but
        d(Wx)/dW = x, so the gradient depends on the embedding rather than on
        the zeroed weight and it does train from the first step.
        """
        model = SequenceGeneratorModel(_config())
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        tokens, conditioning, info = _batch(protocol=2)

        model.estimate_durations(tokens, conditioning, info)[0].sum().backward()
        for name, param in [
            ('duration_protocol_bias', model.duration_protocol_bias.weight),
            ('protocol_cond_proj', model.protocol_cond_proj.weight),
        ]:
            self.assertGreater(param.grad.abs().sum().item(), 0.0, name)

        # protocol_embedding is the one exception, and only on the very first
        # step: its sole consumer is protocol_cond_proj, so while that weight is
        # still exactly zero the chain rule hands the embedding a zero gradient.
        # The projection moves off zero on step 1 (its own gradient is non-zero
        # because the embedding is xavier-initialised), after which the
        # embedding trains normally. Pinned so a future change that leaves BOTH
        # halves at zero — which would freeze this path forever — fails loudly.
        self.assertEqual(model.protocol_embedding.weight.grad.abs().sum().item(), 0.0)
        optimizer.step()
        optimizer.zero_grad()
        self.assertGreater(model.protocol_cond_proj.weight.abs().sum().item(), 0.0)

        model.estimate_durations(tokens, conditioning, info)[0].sum().backward()
        self.assertGreater(model.protocol_embedding.weight.grad.abs().sum().item(), 0.0)

    def test_missing_protocol_key_defaults_to_the_rare_bucket(self):
        """Callers that predate this field must not crash."""
        model = SequenceGeneratorModel(_config()).eval()
        tokens, conditioning, info = _batch()
        info.pop('protocol')
        with torch.no_grad():
            mu, sigma = model.estimate_durations(tokens, conditioning, info)
        self.assertEqual(mu.shape, (tokens.shape[0], tokens.shape[1]))
        self.assertTrue(torch.isfinite(mu).all())

    def test_protocol_also_rides_the_conditioning_vector(self):
        """Per-position bias serves the duration head; the conditioning
        embedding is what the token decoder sees."""
        torch.manual_seed(0)
        model = SequenceGeneratorModel(_config()).eval()
        with torch.no_grad():
            # Both halves: protocol_cond_proj is zero-initialised, so an
            # untrained model correctly ignores the embedding entirely.
            model.protocol_embedding.weight.normal_(0, 1.0)
            model.protocol_cond_proj.weight.normal_(0, 1.0)

        _, conditioning, info = _batch(protocol=1)
        info_b = dict(info, protocol=torch.full((2,), 6, dtype=torch.long))
        with torch.no_grad():
            mem_a = model._encode_conditioning(conditioning, info)
            mem_b = model._encode_conditioning(conditioning, info_b)
        self.assertGreater((mem_a - mem_b).abs().max().item(), 1e-4)

    def test_forward_pass_runs_end_to_end_with_protocol(self):
        model = SequenceGeneratorModel(_config()).eval()
        tokens, conditioning, info = _batch(protocol=3)
        with torch.no_grad():
            logits, mu, sigma = model(conditioning, info, tokens)
        self.assertEqual(logits.shape[:2], tokens.shape)
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue((sigma > 0).all())


if __name__ == '__main__':
    unittest.main()


class ProtocolTrainingPathTests(unittest.TestCase):
    """The dataset tuple is positional and make_pad_collate indexes into it, so
    appending a 9th field is exactly the kind of change that silently misaligns
    the per-token trimming. Drive the real dataset, collate and loader."""

    def _sequences(self, n=12):
        import numpy as np
        rng = np.random.default_rng(0)
        names = ['LOCA HASTE APNEE', 'T1 VIBE', 'rare_one_off']
        out = []
        for i in range(n):
            length = int(rng.integers(3, 7))
            out.append({
                'sequence': [SOURCEID_VOCAB['MRI_MSR_100']]
                            + [SOURCEID_VOCAB['MRI_MSR_21']] * (length - 2)
                            + [SOURCEID_VOCAB['MRI_MSR_104']],
                'durations': [float(rng.integers(5, 60)) for _ in range(length)],
                'conditioning': {'Age': 50.0, 'Weight': 75.0},
                'body_region': int(rng.integers(0, 3)),
                'sequence_type': int(rng.integers(0, 4)),
                'serial_idx': 0,
                'trigger_mode': 0,
                'protocol_name': names[0] if i < n - 1 else names[2],
            })
        return out

    def test_dataset_and_collate_survive_the_extra_field(self):
        from torch.utils.data import DataLoader

        from AlternatingPipeline.data.protocol_vocab import build_protocol_vocab
        from AlternatingPipeline.training.train_examination import ExaminationDataset
        from AlternatingPipeline.training.utils import make_pad_collate

        sequences = self._sequences()
        vocab = build_protocol_vocab([s['protocol_name'] for s in sequences], min_count=3)
        dataset = ExaminationDataset(sequences, max_seq_len=8, duration_scale=60.0,
                                     protocol_vocab=vocab)

        self.assertEqual(len(dataset[0]), 9)
        collate = make_pad_collate(seq_indices=(4, 5, 6), length_index=4,
                                   pad_token_id=PAD_TOKEN_ID)
        loader = DataLoader(dataset, batch_size=4, collate_fn=collate)

        # build_conditioning_tensor emits the base 10-dim vector.
        config = dict(_config(num_protocols=len(vocab) + 1), base_conditioning_dim=10)
        model = SequenceGeneratorModel(config)
        seen_protocols = set()
        for batch in loader:
            (conditioning, body_region, sequence_type, serial_idx,
             input_seq, target_seq, durations, trigger_mode, protocol) = batch
            self.assertEqual(input_seq.shape, target_seq.shape)
            self.assertEqual(durations.shape, input_seq.shape)
            self.assertEqual(protocol.shape, body_region.shape)
            seen_protocols.update(protocol.tolist())

            logits, mu, sigma = model(
                conditioning,
                {'body_region': body_region, 'sequence_type': sequence_type,
                 'serial_idx': serial_idx, 'trigger_mode': trigger_mode,
                 'protocol': protocol},
                input_seq,
            )
            self.assertEqual(logits.shape[:2], input_seq.shape)
            loss = model.compute_loss(logits, target_seq)
            loss.backward()

        # The frequent protocol got a real id; the one-off fell to the bucket.
        self.assertIn(1, seen_protocols)
        self.assertIn(0, seen_protocols)

    def test_a_pkl_without_protocol_names_still_trains(self):
        from AlternatingPipeline.data.protocol_vocab import RARE_PROTOCOL_ID
        from AlternatingPipeline.training.train_examination import ExaminationDataset

        sequences = self._sequences()
        for seq in sequences:
            del seq['protocol_name']
        dataset = ExaminationDataset(sequences, max_seq_len=8, duration_scale=60.0,
                                     protocol_vocab={'anything': 1})
        self.assertTrue(all(dataset[i][8].item() == RARE_PROTOCOL_ID
                            for i in range(len(dataset))))


class ServeSideProtocolWiringTests(unittest.TestCase):
    """Steps 05/06/07 must actually pass the protocol, not merely be able to.

    Commit f276ed5 added protocol conditioning to the model and to step 04,
    froze the vocabulary next to the checkpoint, and left a comment in
    04_train_models.py saying "Steps 05/06/07 MUST read this file rather than
    rebuilding the vocabulary from a pkl". None of the three were updated, and
    for three weeks nothing noticed — because omitting the key is NOT an error.
    `sequence_generator` falls back to RARE_PROTOCOL_ID, the checkpoint's four
    protocol tensors are reported as merely "unexpected", and everything runs.

    So the failure mode is a comment nobody executed. These are text guards on
    the notebooks, in the same style as the num_serials guards in
    test_seqparams_config_guard.py, because the notebooks cannot be imported
    outside Databricks — they read Spark and `%run ./config` at module level.
    """

    _NOTEBOOKS = ('05_feature_analysis.py', '06_compare_models.py',
                  '07_generate_synthetic_data.py')

    def _read(self, name):
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'DatabricksPipeline', 'csv_pipeline_seqparams', name,
        )
        with open(path) as handle:
            return handle.read()

    def test_every_serving_notebook_reads_the_frozen_vocabulary(self):
        for name in self._NOTEBOOKS:
            with self.subTest(notebook=name):
                source = self._read(name)
                self.assertIn('protocol_vocab.json', source)
                self.assertIn('PROTOCOL_VOCAB', source)

    def test_every_serving_notebook_passes_a_protocol_to_the_model(self):
        for name in self._NOTEBOOKS:
            with self.subTest(notebook=name):
                self.assertIn("'protocol':", self._read(name))

    def test_a_missing_vocabulary_stops_the_run_rather_than_degrading(self):
        """Degrading here is worse than stopping.

        Without the vocabulary every prediction uses RARE_PROTOCOL_ID, so 06
        would report an MAE for a model with its strongest input disconnected
        and 07 would generate from one. Both look completely normal.
        """
        for name in self._NOTEBOOKS:
            with self.subTest(notebook=name):
                source = self._read(name)
                self.assertIn('Cannot read the protocol vocabulary', source)

    def test_every_serving_notebook_pins_conditioning_to_the_manifest(self):
        """The 2026-08-21 failure: the width came from config.py, which had
        moved on since training (rare-field floor 1.0 -> 0.0, 133 -> 170 fields,
        base_conditioning_dim 277 -> 351)."""
        for name in self._NOTEBOOKS:
            with self.subTest(notebook=name):
                source = self._read(name)
                self.assertIn('load_trained_model_spec', source)
                self.assertIn('spec=_SPEC', source)

    def test_steps_05_and_06_no_longer_depend_on_step_04s_tmp_copy(self):
        """They imported from /tmp/alternating_pipeline_src, a directory only
        step 04 populates, so they could only run on the machine that had just
        trained. /tmp is per-machine and per-cluster-lifetime."""
        for name in ('05_feature_analysis.py', '06_compare_models.py'):
            with self.subTest(notebook=name):
                source = self._read(name)
                self.assertIn('bootstrap_pipeline_source()', source)
                self.assertNotIn('sys.path.insert(0, _AP_PATH)', source)

    def test_step_07_draws_a_real_protocol_instead_of_inventing_one(self):
        source = self._read('07_generate_synthetic_data.py')
        self.assertIn('build_protocol_sampler', source)
        self.assertIn('protocol_sampler.sample(', source)
        # The eight-name placeholder pool wrote names that appear nowhere in the
        # real corpus, so the synthetic Protocol column could not share a Qlik
        # dimension with the real one — and had no embedding row either.
        self.assertNotIn("np.random.choice(_PROTOCOLS)", source)
        self.assertNotIn("_PROTOCOLS  = [", source)

    def test_step_07_reports_what_it_actually_drew(self):
        """A sampler collapsed to one name looks exactly like a working one."""
        source = self._read('07_generate_synthetic_data.py')
        self.assertIn('protocol_draw_counter', source)
        self.assertIn('COLLAPSED', source)

    def test_step_05_ranks_the_protocol_channel(self):
        """Its rank is the cheapest check that the ids are arriving:
        duration_protocol_bias is a trained per-position embedding, so a
        near-zero degradation means broken wiring, not a weak signal."""
        source = self._read('05_feature_analysis.py')
        self.assertIn("['protocol', 'sequence_type'", source)
        self.assertIn("shuffle_field == 'protocol'", source)

    def test_step_06_probes_the_protocol_channel_and_prints_the_bar(self):
        source = self._read('06_compare_models.py')
        self.assertIn('PROTOCOL CHANNEL IS NOT ARRIVING', source)
        # The acceptance number step 04's gate stamped into the manifest.
        self.assertIn('protocol_baseline_mae_s', source)

    def test_step_06_does_not_let_protocol_mask_a_dead_conditioning_token(self):
        """Criterion 1b asks whether channels arriving ONLY via the conditioning
        token move the duration head. protocol also has a per-position route, so
        selecting cond-token-only channels as "everything except the control"
        would let a live protocol answer that question with the wrong yes."""
        source = self._read('06_compare_models.py')
        self.assertIn('_cond_only_labels', source)
        self.assertNotIn(
            "_cond_only = [v for k, v in _channel_results.items() if k != _control_label]",
            source,
        )
