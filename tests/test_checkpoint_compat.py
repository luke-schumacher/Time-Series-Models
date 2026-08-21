"""Tests for checkpoint/architecture compatibility diagnosis.

Motivated by a real failure: 06_compare_models.py crashed with a raw
`RuntimeError: Error(s) in loading state_dict` when the baseline examination
checkpoint on DBFS turned out to have been written by an architecture that
does not exist in this repo (a 3-component mixture-density duration head).

The important property proven here is the one that made the original
mitigation useless: `strict=False` does NOT tolerate shape mismatches, so
callers cannot rely on it to survive an architecture change.
"""
import os
import sys

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from AlternatingPipeline.models.checkpoint_compat import (  # noqa: E402
    IncompatibleCheckpointError,
    inspect_checkpoint,
    load_checkpoint_lenient,
)


class _Head(nn.Module):
    """Stand-in for SinglePassDurationHead (1 component)."""

    def __init__(self, n_components=1, with_bias_embedding=False, with_mixture=False):
        super().__init__()
        self.shared_mlp = nn.Linear(8, 16)
        self.mu_head = nn.Linear(16, n_components)
        self.sigma_head = nn.Linear(16, n_components)
        if with_mixture:
            self.mixture_logits_head = nn.Linear(16, n_components)
        if with_bias_embedding:
            self.duration_seq_type_bias = nn.Embedding(4, 8)


def test_identical_architecture_is_clean():
    model, other = _Head(), _Head()
    report = inspect_checkpoint(model, other.state_dict())

    assert report.loadable
    assert report.missing == []
    assert report.unexpected == []
    assert report.shape_mismatched == []


def test_new_param_absent_from_checkpoint_is_loadable():
    """The duration_seq_type_bias case: model gained a param after training."""
    model = _Head(with_bias_embedding=True)
    checkpoint = _Head(with_bias_embedding=False).state_dict()

    report = inspect_checkpoint(model, checkpoint)

    assert report.loadable
    assert "duration_seq_type_bias.weight" in report.missing
    assert report.shape_mismatched == []

    # And it actually loads, leaving the new parameter at its initialised value.
    before = model.duration_seq_type_bias.weight.clone()
    load_checkpoint_lenient(model, checkpoint, label="test")
    assert torch.equal(model.duration_seq_type_bias.weight, before)
    assert torch.equal(model.mu_head.weight, checkpoint["mu_head.weight"])


def test_extra_param_in_checkpoint_is_loadable():
    model = _Head()
    checkpoint = _Head(with_mixture=True).state_dict()

    report = inspect_checkpoint(model, checkpoint)

    assert report.loadable
    assert "mixture_logits_head.weight" in report.unexpected
    assert report.shape_mismatched == []


def test_shape_mismatch_is_not_loadable():
    """The real failure: a 3-component mixture head vs. a 1-component head."""
    model = _Head(n_components=1)
    checkpoint = _Head(n_components=3, with_mixture=True).state_dict()

    report = inspect_checkpoint(model, checkpoint)

    assert not report.loadable
    names = [name for name, _, _ in report.shape_mismatched]
    assert "mu_head.weight" in names
    assert "sigma_head.weight" in names

    described = report.describe(label="baseline")
    assert "mu_head.weight" in described
    assert "torch.Size([3, 16])" in described
    assert "torch.Size([1, 16])" in described


def test_load_lenient_raises_actionable_error_on_shape_mismatch():
    model = _Head(n_components=1)
    checkpoint = _Head(n_components=3, with_mixture=True).state_dict()

    with pytest.raises(IncompatibleCheckpointError) as excinfo:
        load_checkpoint_lenient(model, checkpoint, label="baseline examination")

    message = str(excinfo.value)
    assert "baseline examination" in message
    assert "mu_head.weight" in message
    # Must name the remedy, not just the symptom.
    assert "retrain" in message.lower()


def test_strict_false_does_not_survive_shape_mismatch():
    """Guards the assumption that made the original mitigation ineffective.

    b9f3dd0 added `strict=False` to step 05's checkpoint load specifically so
    a new parameter would not break an existing checkpoint. That works for
    missing/unexpected keys only — PyTorch still raises on shape mismatch
    regardless of `strict`. If this test ever fails, PyTorch changed and
    load_checkpoint_lenient can be simplified.
    """
    model = _Head(n_components=1)
    checkpoint = _Head(n_components=3).state_dict()

    with pytest.raises(RuntimeError, match="size mismatch"):
        model.load_state_dict(checkpoint, strict=False)


# ---------------------------------------------------------------------------
# The 2026-08-21 failure, end to end: a finished training run that could not be
# analysed or generated from. Nothing on disk changed — config.py did. The
# rare-field floor moved 1.0 -> 0.0, admitting 37 more parameters (133 -> 170
# fields, base_conditioning_dim 277 -> 351), and every serving notebook rebuilt
# the conditioning list from the live config instead of from the manifest the
# checkpoint was stamped with.
#
# These use the REAL model class and the REAL config builder, so they fail if
# either end of the contract moves — a hand-rolled stand-in would keep passing
# while the notebooks broke.
# ---------------------------------------------------------------------------

import importlib.util  # noqa: E402
import json  # noqa: E402
import tempfile  # noqa: E402

from AlternatingPipeline.config import EXAMINATION_MODEL_CONFIG  # noqa: E402
from AlternatingPipeline.models.examination_model import (  # noqa: E402
    create_examination_model,
)

_SEQPARAMS_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "DatabricksPipeline", "csv_pipeline_seqparams", "config.py",
)

# Small enough to build in a test, same SHAPE as the real thing: a value column
# and a presence flag per field, plus the derived in-segment flag.
_TRAINED_FIELDS = ["TR", "num_slices"]
_TRAINED_FEATURES = (
    _TRAINED_FIELDS
    + [f"{name}__present" for name in _TRAINED_FIELDS]
    + ["sut_in_segment"]
)
# What config.py would resolve to after the floor moved: purely additive, with
# the previously-trained names keeping their order — verified against the real
# 133-vs-170 lists from the 2026-08-21 run.
_LIVE_FEATURES = (
    _TRAINED_FIELDS + ["PDM", "SAT"]
    + [f"{name}__present" for name in _TRAINED_FIELDS + ["PDM", "SAT"]]
    + ["sut_in_segment"]
)


def _seqparams_config(live_features):
    spec = importlib.util.spec_from_file_location(
        "seqparams_config_for_checkpoint_test", _SEQPARAMS_CONFIG_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EXAMINATION_SEQPARAM_FEATURES = list(live_features)
    module.EXAMINATION_SEQPARAM_SCALE = [1.0] * len(live_features)
    return module


def _manifest(tmpdir, base_dim, features, num_protocols=7):
    path = os.path.join(tmpdir, "MODEL_MANIFEST.json")
    with open(path, "w") as handle:
        json.dump({
            "extra_conditioning_features": list(features),
            "base_conditioning_dim": base_dim,
            "num_protocols": num_protocols,
            "num_trigger_modes": 6,
            "param_set": "all",
            "trained_at": "2026-08-21 08:06:09",
        }, handle)
    return path


def test_a_manifest_pinned_model_loads_the_checkpoint_cleanly():
    live = _seqparams_config(_LIVE_FEATURES)
    base_dim = EXAMINATION_MODEL_CONFIG["base_conditioning_dim"] + len(_TRAINED_FEATURES)

    with tempfile.TemporaryDirectory() as tmpdir:
        spec = live.load_trained_model_spec(
            _manifest(tmpdir, base_dim, _TRAINED_FEATURES)
        )
        trained_config = live.build_seqparams_model_config(
            EXAMINATION_MODEL_CONFIG, spec=spec
        )
        checkpoint = create_examination_model(trained_config).state_dict()

        # A fresh serve-side model, built the way 05/06/07 now build it.
        served = create_examination_model(
            live.build_seqparams_model_config(EXAMINATION_MODEL_CONFIG, spec=spec)
        )

    report = load_checkpoint_lenient(served, checkpoint, label="examination", verbose=False)
    assert report.clean, report.describe()
    # The four protocol tensors are PRESENT, not merely tolerated as unexpected.
    for name in ("protocol_embedding.weight", "protocol_cond_proj.weight",
                 "protocol_cond_proj.bias", "duration_protocol_bias.weight"):
        assert name in served.state_dict()


def test_the_live_config_alone_reproduces_the_2026_08_21_shape_error():
    """The control. Without the manifest the checkpoint is unloadable, and
    `strict=False` cannot absorb it — which is why a finished GPU run was
    unusable until the floor was put back by hand.
    """
    live = _seqparams_config(_LIVE_FEATURES)
    base_dim = EXAMINATION_MODEL_CONFIG["base_conditioning_dim"] + len(_TRAINED_FEATURES)

    with tempfile.TemporaryDirectory() as tmpdir:
        spec = live.load_trained_model_spec(
            _manifest(tmpdir, base_dim, _TRAINED_FEATURES)
        )
        checkpoint = create_examination_model(
            live.build_seqparams_model_config(EXAMINATION_MODEL_CONFIG, spec=spec)
        ).state_dict()

    # No spec: the width comes from the wider live config, exactly as before.
    unpinned = create_examination_model(
        live.build_seqparams_model_config(EXAMINATION_MODEL_CONFIG)
    )

    report = inspect_checkpoint(unpinned, checkpoint)
    assert not report.loadable
    mismatched = {name for name, _, _ in report.shape_mismatched}
    assert "conditioning_scale" in mismatched
    assert "conditioning_projection.0.weight" in mismatched
    # And the protocol tensors are silently dropped rather than loaded, which is
    # the half of this that produces plausible-looking wrong numbers.
    assert "duration_protocol_bias.weight" in report.unexpected

    with pytest.raises(IncompatibleCheckpointError):
        load_checkpoint_lenient(unpinned, checkpoint, verbose=False)


def test_trained_divisors_travel_in_the_checkpoint_not_the_config():
    """Why serve-side scale resolution is allowed to be lenient.

    conditioning_scale is a persistent buffer, so load_state_dict overwrites
    every entry with the trained value. Only the LENGTH has to be right at build
    time — refusing to serve because the current divisor table forgot a field
    would reject a valid checkpoint over a number about to be discarded.
    """
    live = _seqparams_config(_LIVE_FEATURES)
    base_dim = EXAMINATION_MODEL_CONFIG["base_conditioning_dim"] + len(_TRAINED_FEATURES)

    with tempfile.TemporaryDirectory() as tmpdir:
        spec = live.load_trained_model_spec(
            _manifest(tmpdir, base_dim, _TRAINED_FEATURES)
        )
        trained_config = live.build_seqparams_model_config(
            EXAMINATION_MODEL_CONFIG, spec=spec
        )
        trained = create_examination_model(trained_config)
        trained.conditioning_scale.fill_(1234.0)
        checkpoint = trained.state_dict()

        served = create_examination_model(
            live.build_seqparams_model_config(EXAMINATION_MODEL_CONFIG, spec=spec)
        )

    assert not torch.allclose(
        served.conditioning_scale, torch.full_like(served.conditioning_scale, 1234.0)
    )
    load_checkpoint_lenient(served, checkpoint, verbose=False)
    assert torch.allclose(
        served.conditioning_scale, torch.full_like(served.conditioning_scale, 1234.0)
    )
