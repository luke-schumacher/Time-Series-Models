# Databricks notebook source
# Databricks notebook — Generate synthetic data using the SUT-conditioned
# examination model.
#
# Forked from DatabricksPipeline/csv_pipeline/05_generate_synthetic_data.py.
# The ONLY differences are the examination half; the exchange, orchestration,
# day-assembly, row-rendering, CSV-writing and reporting code is unchanged.
# Every delta is marked with a "SEQPARAMS FORK:" comment.
#
#   1. The examination model is built with build_seqparams_model_config() and
#      loaded from the csv_pipeline_seqparams checkpoint.
#   2. Its conditioning tensor is 10 base dims + one column per
#      EXAMINATION_SEQPARAM_FEATURES, which the PARAM_SET switch in config.py
#      selects — so the width follows the chosen parameter set and the
#      checkpoint must come from the matching MODELS_DIR. Its info dict
#      carries trigger_mode.
#   3. Those parameters do not exist for a synthetic scan, so they are SAMPLED
#      per (body_region, sequence_type) from the real training distribution
#      via AlternatingPipeline/data/sut_parameter_sampling.py.
#
# TWO PKLs, on purpose. The seqparams pkl holds ONLY 'examination' (see
# 03_build_preprocessed_pkl.py) — it has no 'exchange' and no
# 'customer_schedules', which the exchange and orchestration models need. So
# the orchestration/exchange halves read the csv_pipeline pkl and the
# examination half reads the seqparams pkl. Both were built from the same 10
# serials over the same date range, so they are consistent by construction.
#
# NOTE ON WHAT THIS BUYS YOU: as of the 2026-07-24 training run, permutation
# importance for TR and num_slices is 0.0% — the duration encoder gives
# sequence_type a per-position bias but lets the SUT features in only through
# the diluted conditioning token. Until that is fixed, this pipeline's output
# should match csv_pipeline's apart from the long-type duration calibration
# that duration_seq_type_bias already delivers. The sampling below is
# nonetheless correct and becomes load-bearing the moment the path is fixed.
#
#   Exchange CSV  →  /dbfs/FileStore/csv_pipeline_seqparams/synthetic/exchange/DATA_{serial}.csv
#   Exam CSV      →  /dbfs/FileStore/csv_pipeline_seqparams/synthetic/exam/DATA_{serial}.csv
#
# Column formats match 01_exchange_preprocessing.py and 02_exam_preprocessing.py
# output exactly so synthetic and real data are interchangeable.
#
# Prerequisites:
#   - csv_pipeline_seqparams/03_build_preprocessed_pkl.py + 04_train_models.py
#     (examination model)
#   - csv_pipeline/03_build_preprocessed_pkl.py + 04_train_models.py
#     (exchange + orchestration models, and their pkl)

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

# SOURCE BOOTSTRAP — one call, shared with 05_feature_analysis.py and
# 06_compare_models.py via `%run ./config` above.
#
# This used to be TWO overlapping inline blocks in this notebook (clone /tmp/tsm
# and pre-import, then a second pass that re-resolved the repo root), while 05
# and 06 did something different again: they sys.path.insert'd
# /tmp/alternating_pipeline_src, a directory only 04_train_models.py populates.
# Three answers to one question, and the two that could not stand alone are the
# two that died on 2026-08-21. Consolidated rather than left as-is, because a
# second implementation is how the first one silently stops matching.
REPO_ROOT = bootstrap_pipeline_source()


# Post-condition on what the bootstrap actually produced. The refresh above
# WARNS rather than fails when `git fetch` cannot reach the network, so an
# offline cluster keeps working — at the cost of possibly running an old commit.
# This is what turns that into one readable line instead of a bare
# ModuleNotFoundError deep inside the run. purge=False: the bootstrap has just
# pinned these modules on purpose.
assert_pipeline_source_fresh(REPO_ROOT, purge=False, required_modules=[
    "AlternatingPipeline.config",
    "AlternatingPipeline.models.examination_model",
    "AlternatingPipeline.models.checkpoint_compat",
    "AlternatingPipeline.data.protocol_vocab",
    "AlternatingPipeline.data.protocol_sampling",
    "AlternatingPipeline.data.sut_parameter_sampling",
])


# COMMAND ----------

import sys, os, re, json, pickle, shutil, subprocess, math
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta


# SEQPARAMS FORK: two pkls and two model dirs. `%run ./config` above already
# defined MODELS_DIR / PKL_OUTPUT for THIS pipeline; both are re-bound to
# explicit, unambiguous names here so no later line can pick up the wrong one.
#
# The two csv_pipeline paths come from config's BASE_PKL / BASE_MODELS_DIR
# rather than being literals here. That is the ONLY remaining place this
# pipeline depends on the other one, and naming it is what makes the coming
# consolidation with Navneet's GPU rework a config change instead of a code
# change — see the note above BASE_PKL in config.py.
PKL_PATH        = BASE_PKL                                              # customer_schedules + exchange
SEQPARAMS_PKL   = PKL_OUTPUT                                            # examination (SUT-enriched)
EXAM_MODELS_DIR = f"{MODELS_DIR}/examination"                           # SUT-conditioned examination
SYNTH_EXCHANGE  = "/dbfs/FileStore/csv_pipeline_seqparams/synthetic/exchange"
SYNTH_EXAM      = "/dbfs/FileStore/csv_pipeline_seqparams/synthetic/exam"

os.makedirs(SYNTH_EXCHANGE, exist_ok=True)
os.makedirs(SYNTH_EXAM,     exist_ok=True)

# Synthetic date range — must be OUTSIDE the training window (2024-01-01 → 2024-01-31)
# to avoid data leakage into the orchestration model.
SYNTH_DATE_START     = "2024-02-01"
SYNTH_DATE_END       = "2024-02-28"
WEEKDAYS_ONLY        = True          # skip Saturdays/Sundays (MRI scanners are rarely used then)
NUM_DAYS_PER_SCANNER = 30            # cap if the date range produces more days than needed

# ── Optional: controlled body-group schedule ────────────────────────────────
# Leave as None for the realistic default — the orchestration model decides
# each day's patient mix and ordering (this captures real inter-body-group
# dynamics, which is what you want for shipping synthetic data).
#
# To run a *controlled experiment* — e.g. "10 head, 10 knee, 5 abdomen" — set
# this to a list of body-region names. Every synthetic day is then built from
# exactly that mix instead of the orchestration output.
#
#   CONTROLLED_SCHEDULE = ['HEAD']*10 + ['LEG']*10 + ['ABDOMEN']*5
#
# This changes ONLY the day's patient list. It does not retrain or alter any
# model: the Exchange model is still conditioned on every body_from→body_to
# transition, so inter-body-group exchange dynamics are fully preserved. What
# you give up is the realistic day shape (counts/ordering) the orchestration
# model learned — so use this for targeted tests, not for shipped data.
CONTROLLED_SCHEDULE  = None

# COMMAND ----------

# =============================================================================
# PRE-FLIGHT — MODEL FRESHNESS GATE  (run BEFORE loading models)
# -----------------------------------------------------------------------------
# Guards against silently generating from a STALE model. Verifies that the
# checkpoints on disk:
#   (1) match the MODEL_MANIFEST.json that step 04 wrote when it trained them,
#   (2) are NEWER than the pkl (i.e. not trained on older data), and
#   (3) come from a pkl whose content is unchanged since training.
# Any mismatch prints a loud warning. Set env STRICT_FRESHNESS=1 to hard-stop.
# =============================================================================
import os, time, json, hashlib

# SEQPARAMS FORK: each checkpoint is checked against ITS OWN pkl and manifest,
# because the examination model was trained by this pipeline on the seqparams
# pkl while exchange/orchestration come from csv_pipeline's.
_CHECKPOINTS = {
    "exchange":      f"{BASE_MODELS_DIR}/exchange/exchange_model_best.pt",
    "examination":   f"{EXAM_MODELS_DIR}/examination_model_best.pt",
    "orchestration": f"{BASE_MODELS_DIR}/orchestration/orchestration_model_best.pt",
}
_CHECKPOINT_SOURCE = {          # model -> (pkl that trained it, its manifest)
    "exchange":      (PKL_PATH,      f"{BASE_MODELS_DIR}/MODEL_MANIFEST.json"),
    "orchestration": (PKL_PATH,      f"{BASE_MODELS_DIR}/MODEL_MANIFEST.json"),
    "examination":   (SEQPARAMS_PKL, f"{MODELS_DIR}/MODEL_MANIFEST.json"),
}

def _file_meta(path):
    if not os.path.exists(path):
        return {"path": path, "exists": False}
    st = os.stat(path)
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return {"path": path, "exists": True,
            "size_mb": round(st.st_size / 1e6, 2),
            "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
            "mtime_epoch": st.st_mtime, "sha256": h.hexdigest()[:16]}

print("=" * 64)
print(" STEP 07 PRE-FLIGHT — MODEL FRESHNESS GATE (two pkls)")
print("=" * 64)

_warnings = []
_pkl_metas = {p: _file_meta(p) for p in {PKL_PATH, SEQPARAMS_PKL}}
_ckpt_meta = {k: _file_meta(v) for k, v in _CHECKPOINTS.items()}

for _label, _path in (("csv_pipeline (exchange/orch)", PKL_PATH),
                      ("seqparams   (examination)  ", SEQPARAMS_PKL)):
    _m = _pkl_metas[_path]
    print(f"\nPKL  {_label}  {_path}")
    if _m["exists"]:
        print(f"     mtime {_m['mtime']}   {_m['size_mb']} MB   sha {_m['sha256']}")
    else:
        _warnings.append(f"PKL missing at {_path}")

print("\nCheckpoints on disk:")
for _k, _m in _ckpt_meta.items():
    _src_pkl, _man_path = _CHECKPOINT_SOURCE[_k]
    _pkl_meta = _pkl_metas[_src_pkl]
    if not _m["exists"]:
        _warnings.append(f"{_k} checkpoint MISSING at {_CHECKPOINTS[_k]}")
        print(f"   {_k:<13} !! MISSING  ({_CHECKPOINTS[_k]})")
        continue
    _age_h = (time.time() - _m["mtime_epoch"]) / 3600
    print(f"   {_k:<13} {_m['mtime']}  ({_age_h:5.1f}h old)  {_m['size_mb']:6.1f} MB  sha {_m['sha256']}")
    if _pkl_meta["exists"] and _m["mtime_epoch"] < _pkl_meta["mtime_epoch"]:
        _warnings.append(f"{_k} checkpoint is OLDER than its pkl "
                         f"({_m['mtime']} < {_pkl_meta['mtime']}) — trained on older data than is loaded now.")

# --- cross-check each checkpoint against the manifest step 04 wrote for it ---
for _k, _m in _ckpt_meta.items():
    _src_pkl, _man_path = _CHECKPOINT_SOURCE[_k]
    if not os.path.exists(_man_path):
        _warnings.append(f"No MODEL_MANIFEST.json at {_man_path} — cannot confirm which pkl/code "
                         f"produced the {_k} checkpoint. Re-run the step 04 that trains it.")
        continue
    with open(_man_path) as _f:
        _man = json.load(_f)
    if _man.get("pkl", {}).get("sha256") != _pkl_metas[_src_pkl].get("sha256"):
        _warnings.append(f"{_k}: live pkl sha != manifest pkl sha ({_src_pkl}) — step 03 re-ran "
                         f"after training. Re-run step 04, or the model doesn't match this pkl.")
    # csv_pipeline's manifest keys checkpoints by model name; the seqparams
    # manifest is examination-only and stores a single 'checkpoint' object.
    _man_sha = (_man.get("checkpoints", {}).get(_k, {}).get("sha256")
                or (_man.get("checkpoint", {}).get("sha256") if _k == "examination" else None))
    if _man_sha and _man_sha != _m["sha256"]:
        _warnings.append(f"{_k} checkpoint sha {_m['sha256']} != manifest {_man_sha} — "
                         f"checkpoint changed since training was stamped (partial/overwritten run).")
    if _m["mtime_epoch"] > _man.get("trained_at_epoch", 0) + 120 and _man.get("trained_at_epoch"):
        _warnings.append(f"{_k} checkpoint is NEWER than the manifest — retrained without "
                         f"re-stamping. Re-run step 04's manifest cell.")

# --- does the LIVE config still describe the trained architecture? ----------
# The gate above compares shas, which catches "someone re-ran step 03". It does
# NOT catch the failure that actually happened on 2026-08-21: nothing on disk
# changed, config.py did. The rare-field floor moved 1.0 -> 0.0, admitting 37
# more parameters, and the notebook died 200 lines later on a raw tensor shape.
# Naming the divergence HERE turns that into one readable line at the gate.
_exam_spec = load_trained_model_spec(_CHECKPOINT_SOURCE["examination"][1])
if _exam_spec is None:
    _warnings.append(
        f"examination: manifest at {_CHECKPOINT_SOURCE['examination'][1]} carries no "
        f"extra_conditioning_features/base_conditioning_dim, so the conditioning cannot "
        f"be pinned to the checkpoint and falls back to config.py. Re-run step 04's "
        f"manifest cell."
    )
else:
    _drift = describe_spec_drift(_exam_spec)
    if _drift:
        print("\n" + _drift)
        _warnings.append(
            f"examination: config.py describes {len(EXAMINATION_SEQPARAM_FEATURES)} "
            f"conditioning features, the checkpoint was trained on "
            f"{len(_exam_spec.extra_conditioning_features)} (see the drift report above). "
            f"Generation uses the CHECKPOINT's list; the two runs are not comparable."
        )

print("\n" + "-" * 64)
if _warnings:
    print(f"!! {len(_warnings)} FRESHNESS WARNING(S) — review before trusting this run:")
    for _w in _warnings:
        print(f"   - {_w}")
    if os.environ.get("STRICT_FRESHNESS") == "1":
        raise RuntimeError("Model freshness gate failed (STRICT_FRESHNESS=1). See warnings above.")
else:
    print("OK — checkpoints match the manifest, are newer than the pkl, and the pkl is "
          "unchanged since training.")
print("=" * 64)

# COMMAND ----------

# =============================================================================
# Load data and models
# =============================================================================
import sys
sys.path[:] = [p for p in sys.path if 'Patient Exchange and Examination' not in p]
import AlternatingPipeline as _ap                                                                      
_ap.__path__[:] = [p for p in _ap.__path__ if 'Patient Exchange and Examination' not in p]

from AlternatingPipeline.config import (
    EXCHANGE_MODEL_CONFIG, EXAMINATION_MODEL_CONFIG, ORCHESTRATION_MODEL_CONFIG,
    EXCHANGE_TRAINING_CONFIG, EXAMINATION_TRAINING_CONFIG,
    ID_TO_SOURCEID, SOURCEID_VOCAB, BODY_REGIONS, BODY_REGION_TO_ID,
    START_REGION_ID, END_REGION_ID, PHASE_TYPES, GENERATION_CONFIG,
    START_TOKEN_ID, END_TOKEN_ID, PAD_TOKEN_ID, BREAK_TOKEN_ID,
    NUM_BODY_REGIONS, ORCH_PAD_TOKEN_ID,
    SEQUENCE_TYPE_VOCAB, ID_TO_SEQUENCE_TYPE, NUM_SEQUENCE_TYPES, NUM_SERIALS,
)

# Duration unscaling — models were trained on (raw_seconds / duration_scale),
# so generated durations must be multiplied back to get real seconds.
EXCHANGE_DURATION_SCALE  = EXCHANGE_TRAINING_CONFIG['duration_scale']   # 60.0
EXAMINATION_DURATION_SCALE = EXAMINATION_TRAINING_CONFIG['duration_scale']  # 60.0
from AlternatingPipeline.models.exchange_model    import create_exchange_model
from AlternatingPipeline.models.examination_model import create_examination_model
from AlternatingPipeline.models.orchestration_model import create_orchestration_model
from AlternatingPipeline.models.checkpoint_compat import load_checkpoint_lenient
from AlternatingPipeline.data.sut_parameter_sampling import build_sut_sampler
from AlternatingPipeline.data.intra_visit_gap import (
    build_gap_sampler_from_csvs, save_gap_prior,
)
from AlternatingPipeline.data.protocol_sampling import build_protocol_sampler
from AlternatingPipeline.data.protocol_vocab import RARE_PROTOCOL_ID
from AlternatingPipeline.data.orchestration_preprocessing import (
    extract_orchestration_samples, build_demographic_distributions,
    _compute_scanner_stats,
)
from AlternatingPipeline.generation.day_budget import DayBudget
from AlternatingPipeline.generation.output_integrity import (
    GenerationIntegrityError,
    repair_examination_sequence,
    validate_examination_sequence,
    validate_exchange_sequence,
    validate_orchestration_sequence,
    validate_rendered_output,
)

MAX_GENERATION_ATTEMPTS = max(
    1,
    int(os.environ.get(
        'MAX_GENERATION_ATTEMPTS',
        GENERATION_CONFIG.get('max_regeneration_attempts', 5),
    )),
)
MAX_RENDERED_EXAM_DURATION_SEC = float(
    GENERATION_CONFIG.get('max_rendered_exam_duration_sec', 4000)
)
STRICT_OUTPUT_VALIDATION = (
    os.environ.get(
        'STRICT_OUTPUT_VALIDATION',
        '1' if GENERATION_CONFIG.get('strict_output_validation', True) else '0',
    ) != '0'
)
# Longest wall-clock span one synthetic day may occupy. See DayBudget for why
# an unbounded day is what produced "exam row N overlaps the preceding
# examination" on 2026-07-28.
MAX_DAY_SPAN_SEC = float(os.environ.get(
    'MAX_DAY_SPAN_SEC',
    GENERATION_CONFIG.get('max_day_span_sec', 13 * 3600),
))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# --- preprocessed data ---
# SEQPARAMS FORK: `data` is the csv_pipeline pkl and drives the orchestration
# and exchange halves (customer_schedules / exchange live only there).
# `exam_data` is the seqparams pkl and drives everything examination-side, so
# the scan-type mix, duration fallbacks and SUT parameters are all drawn from
# the exact same sequences the examination model was trained on.
with open(PKL_PATH, 'rb') as f:
    data = pickle.load(f)

customer_schedules = data['customer_schedules']
print(f"Customers: {list(customer_schedules.keys())}")

with open(SEQPARAMS_PKL, 'rb') as f:
    exam_data = pickle.load(f)
print(f"Examination sequences (seqparams pkl): {len(exam_data.get('examination', [])):,}")

# --- load exchange model ---
# load_checkpoint_lenient tolerates params added/removed since training, but
# refuses a checkpoint whose parameter SHAPES differ — that means a different
# architecture wrote it, and `strict=False` cannot absorb it either (PyTorch
# raises on size mismatch regardless of `strict`). Failing here with a named
# diagnosis beats generating from a half-initialised model.
exchange_model = create_exchange_model(EXCHANGE_MODEL_CONFIG).to(device)
load_checkpoint_lenient(
    exchange_model,
    torch.load(_CHECKPOINTS["exchange"], map_location=device),
    label="exchange checkpoint",
)
exchange_model.eval()
print("Exchange model loaded.")

# --- load examination model (SEQPARAMS FORK) ---
# build_seqparams_model_config comes from this notebook's own `%run ./config`
# — shared with 04_train_models.py, 05_feature_analysis.py and
# 06_compare_models.py.
#
# THE CHECKPOINT'S MANIFEST IS THE AUTHORITY FOR ITS ARCHITECTURE, not
# config.py. This comment used to claim that sharing the builder meant "the
# serve-side model is built exactly as the trained one was". It does not: it
# makes the four notebooks agree with EACH OTHER, and guarantees nothing about
# agreeing with a checkpoint written an hour ago.
#
# On 2026-08-21 config.py's rare-field floor moved 1.0 -> 0.0 between
# training and generation, admitting 37 more parameters
# (133 -> 170 fields, base_conditioning_dim 277 -> 351), and this notebook died
# on `conditioning_projection.0.weight: checkpoint [128, 357] vs. model
# [128, 431]`. `spec=` makes the builder read the manifest instead, which also
# restores num_protocols — without it the model is built with no protocol
# tensors at all and every generated scan silently uses RARE_PROTOCOL_ID.
_SPEC = load_trained_model_spec(f"{MODELS_DIR}/MODEL_MANIFEST.json")
if _SPEC is None:
    print(f"!! No usable MODEL_MANIFEST.json at {MODELS_DIR} — falling back to config.py's "
          f"feature list. Expect a shape mismatch if the checkpoint was trained under a "
          f"different feature selection. Re-run 04's manifest cell.")
    EXTRA_FEATURES = list(EXAMINATION_SEQPARAM_FEATURES)
else:
    EXTRA_FEATURES = list(_SPEC.extra_conditioning_features)

EXAMINATION_MODEL_CONFIG_SEQPARAMS = build_seqparams_model_config(
    EXAMINATION_MODEL_CONFIG, spec=_SPEC,
)

# --- num_serials from the CHECKPOINT, not from a hardcoded config -----------
# Ported from PR #59 (NavneetKishanS). Serve-side counterpart of the training
# fix in 04: a checkpoint trained on 21 serials carries a [21, 16] serial
# embedding, and building the model from a config that still says 10 gives
# [10, 16]. The checkpoint is the authority here — it is a fact about the model
# being loaded, where the config is only a default.
#
# load_checkpoint_lenient would REFUSE that mismatch rather than load it (which
# is why this fork failed loudly where csv_pipeline's strict=False loaded a
# half-initialised model), but refusing at serve time is still an outage. This
# removes the cause; the lenient loader below stays as the detector for the
# next one.
_exam_ckpt = torch.load(_CHECKPOINTS["examination"], map_location=device)
EXAM_NUM_SERIALS = int(NUM_SERIALS)
if 'serial_embedding.weight' in _exam_ckpt:
    _ckpt_serials = int(_exam_ckpt['serial_embedding.weight'].shape[0])
    if _ckpt_serials != EXAMINATION_MODEL_CONFIG_SEQPARAMS.get('num_serials'):
        print(f"[fix] num_serials: "
              f"{EXAMINATION_MODEL_CONFIG_SEQPARAMS.get('num_serials')} -> "
              f"{_ckpt_serials} (from the checkpoint, not config)")
        EXAMINATION_MODEL_CONFIG_SEQPARAMS['num_serials'] = _ckpt_serials
    # The OTHER half of PR #59, which did not get carried across: deriving
    # num_serials fixes how the model is BUILT, but the loop below still has to
    # pick which embedding row each synthetic scanner uses, and it was clamping
    # against AlternatingPipeline.config.NUM_SERIALS (a hardcoded 10). Today
    # those agree, so it is invisible; after a 21-serial retrain it would
    # silently route every scanner past the tenth to serial 9 while the model
    # had eleven more rows sitting unused.
    EXAM_NUM_SERIALS = _ckpt_serials

print(f"Examination conditioning: base_conditioning_dim="
      f"{EXAMINATION_MODEL_CONFIG_SEQPARAMS['base_conditioning_dim']}, "
      f"{len(EXTRA_FEATURES)} extra features, "
      f"num_protocols={EXAMINATION_MODEL_CONFIG_SEQPARAMS.get('num_protocols')}")
examination_model = create_examination_model(EXAMINATION_MODEL_CONFIG_SEQPARAMS).to(device)
load_checkpoint_lenient(
    examination_model,
    _exam_ckpt,
    label="examination checkpoint (SUT-conditioned)",
)
examination_model.eval()
print("Examination model loaded.")

# --- SUT parameter sampler (SEQPARAMS FORK) ---
# TR/num_slices are model INPUTS taken from a real MRI_SUT_1005 message. A
# synthetic scan has none, so they are drawn jointly from the real training
# distribution for the scan's (body_region, sequence_type).
sut_sampler = build_sut_sampler(exam_data, EXTRA_FEATURES)
if sut_sampler is not None:
    print(sut_sampler.describe())
else:
    print("No conditioning features configured — SUT sampling disabled.")

# --- protocol sampler (SEQPARAMS FORK) ---
# The protocol is a model INPUT and the strongest one this checkpoint has: step
# 04's gate measured a per-protocol group mean at held-out R2 76.3% / MAE 16.2s
# against sequence_type's 18.7% / 59.9s. A synthetic scan has no MRI_MSR_100
# message, so it is drawn from the real distribution for the scan's
# (serial, body_region, sequence_type) — the same treatment TR/num_slices get
# directly above, for the same reason.
#
# Doing nothing here is NOT neutral: sequence_generator falls back to
# RARE_PROTOCOL_ID whenever the key is absent, pinning every synthetic scan to
# the one embedding row trained on the 3.4% of sequences too infrequent to earn
# their own. Until 2026-08-21 that is exactly what happened, while the CSV's
# Protocol column carried one of eight invented names that appear nowhere in the
# real data — so the column could not be used as a shared Qlik dimension either.
_VOCAB_PATH = f"{EXAM_MODELS_DIR}/protocol_vocab.json"
try:
    with open(_VOCAB_PATH) as _vf:
        PROTOCOL_VOCAB = json.load(_vf)["vocab"]
    print(f"Protocol vocabulary: {len(PROTOCOL_VOCAB):,} protocols from {_VOCAB_PATH}")
except (OSError, ValueError, KeyError) as _err:
    raise RuntimeError(
        f"Cannot read the protocol vocabulary at {_VOCAB_PATH} ({_err}). The checkpoint's "
        f"protocol_embedding / duration_protocol_bias rows are indexed by it, and "
        f"generating without it would pin every synthetic scan to RARE_PROTOCOL_ID. "
        f"Re-run 04_train_models.py's vocabulary cell against this checkpoint."
    )

protocol_sampler = build_protocol_sampler(exam_data, PROTOCOL_VOCAB)
if protocol_sampler is None:
    raise RuntimeError(
        f"No examination sequences in {SEQPARAMS_PKL} to draw protocols from."
    )
print(protocol_sampler.describe())

# Records what was actually fed in, so a collapsed/constant draw is visible in
# the run summary instead of being silently invisible — this project has lost
# weeks to conditioning that looked wired up but wasn't.
from collections import Counter as _Counter
sut_draw_counter = _Counter()
# Same rationale for the protocol: a sampler that has collapsed to one name
# looks exactly like a working one from the outside.
protocol_draw_counter = _Counter()

# --- per-sequence-type duration calibration (from duration_probe.json) ---
# The duration head predicts the right RELATIVE order of scan types but can
# carry systematic per-type bias. These multipliers correct raw model output to
# the probe target means.
#
# SEQPARAMS FORK: reads THIS pipeline's probe. The 2026-07-24 run's probe is
# already close to 1.0 for most types (space 237.6s→244.4s = ×1.03), unlike the
# ×2-3 corrections the csv_pipeline model needed — duration_seq_type_bias
# absorbed most of it into the weights. The multipliers are kept because they
# self-neutralise when the model is well calibrated.
_duration_probe_path = f"{EXAM_MODELS_DIR}/duration_probe.json"
_duration_cal: dict[int, float] = {}  # seq_type_id -> target_s / predicted_s
try:
    with open(_duration_probe_path) as _pf:
        _probe_data = json.load(_pf)
    for _row in _probe_data.get('rows', []):
        _st_name = _row['sequence_type']
        _pred_s  = _row['predicted_s']
        _tgt_s   = _row['target_s']
        _st_id   = SEQUENCE_TYPE_VOCAB.get(_st_name, -1)
        if _st_id >= 0 and _pred_s > 0:
            _duration_cal[_st_id] = _tgt_s / _pred_s
    print(f"Duration calibration loaded: {len(_duration_cal)} scan types")
    _probe_by_name = {r['sequence_type']: r for r in _probe_data.get('rows', [])}
    for _sid, _mul in sorted(_duration_cal.items()):
        _nm = ID_TO_SEQUENCE_TYPE.get(_sid, '?')
        _r = _probe_by_name.get(_nm, {})
        print(f"  {_nm:<8} ×{_mul:.3f}  ({_r.get('predicted_s', 0):.0f}s → {_r.get('target_s', 0):.0f}s)")
except Exception as _e:
    print(f"Duration calibration unavailable ({_e}) — using raw model output")

# --- load orchestration model ---
with open(f"{BASE_MODELS_DIR}/orchestration/scanner_to_idx.json") as f:
    scanner_to_idx = json.load(f)

orch_ckpt = torch.load(_CHECKPOINTS["orchestration"], map_location=device)
orch_config = dict(ORCHESTRATION_MODEL_CONFIG)
orch_config['num_scanners'] = orch_ckpt['scanner_embedding.weight'].shape[0]
orchestration_model = create_orchestration_model(orch_config).to(device)
load_checkpoint_lenient(orchestration_model, orch_ckpt, label="orchestration checkpoint")
orchestration_model.eval()
print("Orchestration model loaded.")

# --- demographic distributions per body region (sampled from real data) ---
orch_samples, _ = extract_orchestration_samples(data)
demographic_distributions = build_demographic_distributions(data)

# --- per-scanner region histograms used as orchestration conditioning ---
# The orchestration model was TRAINED with each scanner's real body-region
# distribution as conditioning (train_orchestration via _compute_scanner_stats).
# Generation must feed the SAME per-scanner histogram — feeding a flat/uniform
# vector is an out-of-distribution input the model never saw, and it collapses
# to its unconditional mode (the NECK/FOOT spike). Compute the real stats here.
scanner_stats = _compute_scanner_stats(customer_schedules)
_dbg = {sid: [round(float(x), 3) for x in st['region_distribution']]
        for sid, st in list(scanner_stats.items())[:1]}
print(f"Per-scanner orchestration conditioning ready for {len(scanner_stats)} scanners "
      f"(example region_distribution: {_dbg})")

# --- per-body-region sequence-type distribution (real frequencies) ---
# The examination model now conditions on scan type (scout/tse/...). For each
# generated scan we draw a scan type from the real per-region mix, so the
# synthetic data reproduces the real scout/TSE balance — and, because the
# model's duration is conditioned on it, the real per-type duration spread.
from collections import defaultdict as _defaultdict

_region_seqtypes = _defaultdict(list)
_fallback_exam_duration_pools = _defaultdict(list)
# SEQPARAMS FORK: exam_data, not data — the scan-type mix and duration
# fallbacks must come from the sequences this examination model was trained on.
for _s in exam_data.get('examination', []):
    _region_id = int(_s.get('body_region', 10))
    _seq_type_id = int(_s.get('sequence_type', 0))
    _serial_id = int(_s.get('serial_idx', 0))
    _region_seqtypes[_region_id].append(_seq_type_id)

    try:
        _duration_sec = float(_s.get('total_duration', 0.0))
    except (TypeError, ValueError):
        continue
    if math.isfinite(_duration_sec) and 1 <= _duration_sec <= MAX_RENDERED_EXAM_DURATION_SEC:
        for _duration_key in (
            ('exact', _serial_id, _region_id, _seq_type_id),
            ('region_type', _region_id, _seq_type_id),
            ('region', _region_id),
            ('global',),
        ):
            _fallback_exam_duration_pools[_duration_key].append(_duration_sec)
_all_seqtypes = [st for sts in _region_seqtypes.values() for st in sts] or [0]
print(f"Sequence-type pool: {len(_all_seqtypes):,} scans across "
      f"{len(_region_seqtypes)} body regions")


def _sample_sequence_type(region_id):
    """Draw a sequence-type id from the real distribution for this region."""
    pool = _region_seqtypes.get(int(region_id)) or _all_seqtypes
    return int(np.random.choice(pool))


# ── INTRA-VISIT GAP PRIOR ────────────────────────────────────────────────────
# The pause between two scans of the SAME examination. Until 2026-08-26 the
# generator had none: the clock advanced by the scan duration alone, so scan N+1
# started the instant scan N ended and every intra-visit gap was exactly 0.0s.
# Real data puts 30.4% of examination wall clock in those gaps, which is why our
# per-scan durations matched while the EXAMINATION came out ~30% short. See
# AlternatingPipeline/data/intra_visit_gap.py for the measurements.
#
# Built from the real exam CSVs rather than the pkl because a visit has to be
# segmented on StepCount, and the pkl carries no patient id. No Spark, no
# rebuild. A missing directory yields a dead sampler and the old behaviour.
_gap_sampler, _gap_csvs = build_gap_sampler_from_csvs(BASE_EXAM_CSV_DIR)
print(_gap_sampler.describe() + f"  [{len(_gap_csvs)} exam CSVs from {BASE_EXAM_CSV_DIR}]")
if not _gap_sampler.is_live:
    print(f"  !! No exam CSVs under {BASE_EXAM_CSV_DIR} — scans will be generated "
          f"back-to-back and examination wall clock will read ~30% short. Set "
          f"BASE_EXAM_CSV_DIR to where 02_exam_preprocessing.py wrote them.")
else:
    _gap_prior_path = save_gap_prior(
        _gap_sampler.prior, f"{MODELS_DIR}/examination/intra_visit_gap_prior.json")
    print(f"  Wrote gap prior → {_gap_prior_path}")

# Its own stream, so adding the gap does not shift every other draw in the run
# and make this change look like it moved things it did not touch.
_GAP_RNG = np.random.default_rng(int(os.environ.get('GAP_SEED', 20260826)))


def _sample_fallback_exam_duration(region_id, sequence_type, serial_idx):
    """Sample a real training duration, preferring the exact scan context."""
    keys = (
        ('exact', int(serial_idx), int(region_id), int(sequence_type)),
        ('region_type', int(region_id), int(sequence_type)),
        ('region', int(region_id)),
        ('global',),
    )
    for key in keys:
        pool = _fallback_exam_duration_pools.get(key, [])
        if len(pool) >= 5 or (key == ('global',) and pool):
            return float(np.random.choice(pool)) / EXAMINATION_DURATION_SCALE
    # The pkl should always provide a global pool; retain a bounded last resort
    # so one corrupt/legacy pkl cannot terminate a multi-serial generation run.
    return min(105.0, MAX_RENDERED_EXAM_DURATION_SEC) / EXAMINATION_DURATION_SCALE

# COMMAND ----------

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _build_cond_tensor(patient_info, current_time, day_start):
    """10-dim conditioning tensor matching build_conditioning_tensor in day_simulator."""
    current_dt  = day_start + timedelta(seconds=float(current_time))
    hour        = current_dt.hour + current_dt.minute / 60.0
    dow         = current_dt.weekday()
    direction   = patient_info.get('direction', 'Head First')
    dir_enc     = 0.0 if str(direction).strip().lower() == 'head first' else 1.0
    return torch.tensor([
        float(patient_info.get('age',    50)),
        float(patient_info.get('weight', 75)),
        float(patient_info.get('height', 1.75)),
        float(patient_info.get('ptab',   0)),
        dir_enc,
        float(np.sin(2 * np.pi * hour / 24)),
        float(np.cos(2 * np.pi * hour / 24)),
        float(np.sin(2 * np.pi * dow  / 7)),
        float(np.cos(2 * np.pi * dow  / 7)),
        1.0 if hour < 12 else 0.0,
    ], dtype=torch.float32).to(device)


# SEQPARAMS FORK: the examination model's conditioning is wider than the
# exchange model's (10 base + one column per EXTRA_FEATURES — the list the
# CHECKPOINT was trained on, taken from its manifest — in order, appended at the
# end, matching
# AlternatingPipeline/training/utils.py::build_conditioning_tensor). Values are
# RAW here; the model divides them by its own `conditioning_scale` buffer, so
# generation must NOT pre-scale them or the features land at the wrong
# magnitude and silently distort every other conditioning channel through the
# projection's LayerNorm.
#
# Kept as a SEPARATE function from _build_cond_tensor: the exchange model still
# takes exactly 10 dims, and widening the shared helper would feed it an input
# shape it was never trained on.
_SYNTH_TRIGGER_MODE = TRIGGER_MODE_VOCAB.get('none', 0)


def _build_exam_cond_tensor(patient_info, current_time, day_start, region_id, seq_type):
    """Conditioning tensor for the examination model, plus the SUT values used.

    Returns (tensor, sut_values) — the dict is returned so the caller can
    report what was actually fed in, rather than it being invisible.
    """
    base = _build_cond_tensor(patient_info, current_time, day_start)
    if not EXTRA_FEATURES or sut_sampler is None:
        return base, {}
    sut_values = sut_sampler.sample(region_id, seq_type)
    extras = torch.tensor(
        [float(sut_values[name]) for name in EXTRA_FEATURES],
        dtype=torch.float32, device=base.device,
    )
    return torch.cat([base, extras]), sut_values


def _sample_demographics(body_region_id, demographic_distributions):
    """Sample patient demographics for a given body region."""
    stats = demographic_distributions.get(body_region_id, {
        'age_mean': 50.0, 'age_std': 15.0,
        'weight_mean': 75.0, 'weight_std': 15.0,
        'height_mean': 1.75, 'height_std': 0.1,
        'direction_prob': 0.8,
    })
    return {
        'age':       float(np.clip(np.random.normal(stats['age_mean'],    stats['age_std']),    1, 100)),
        'weight':    float(np.clip(np.random.normal(stats['weight_mean'], stats['weight_std']), 20, 200)),
        'height':    float(np.clip(np.random.normal(stats['height_mean'], stats['height_std']), 0.5, 2.5)),
        'ptab':      0.0,
        'direction': 'Head First' if np.random.random() < stats['direction_prob'] else 'Feet First',
    }


def _region_name(region_id):
    if region_id == START_REGION_ID: return 'START'
    if region_id == END_REGION_ID:   return 'END'
    if 0 <= region_id < len(BODY_REGIONS): return BODY_REGIONS[region_id]
    return 'UNKNOWN'


# Exam coil binary columns.  Mirrors the real exam CSV schema from step 02:
# all coils listed in csv_pipeline/config.py COIL_COLUMNS, prefixed with '#0_'
# to match how step 02 writes them.  Real scanners emit a subset of these;
# the extras are harmless (consumers ignore unknown columns).
#
# Earlier version used HC/NC (un-abbreviated) — step 02 actually emits the
# post-abbreviation form HE/NE via COIL_ABBREV_MAP, so HC/NC were silently wrong.
_EXAM_COIL_COLS = [
    '#0_BC',
    '#0_SP1','#0_SP2','#0_SP3','#0_SP4','#0_SP5','#0_SP6','#0_SP7','#0_SP8',
    '#0_15K',
    '#0_HW1','#0_HW2','#0_HW3',
    '#0_HE1','#0_HE2','#0_HE3','#0_HE4',
    '#0_NE1','#0_NE2',
    '#0_SHL',
    '#0_BO1','#0_BO2','#0_BO3',
    '#0_FA','#0_TO','#0_FS',
    '#0_PA1','#0_PA2','#0_PA3','#0_PA4','#0_PA5','#0_PA6',
    '#0_SN',
]

# Synthetic sequence name pool (used for the exam Sequence column on the
# exchange side). The Protocol counterpart that used to live here was retired on
# 2026-08-21: exam Protocol values are now drawn from the real corpus by
# protocol_sampler, because the same value is also a model input and an invented
# name has no embedding row.
_SEQUENCES  = ['tse', 'gre', 'tfl', 'ep2d', 'tirm', 'vibe']


def _generate_exchange_rows(tokens, durations, mu, sigma, day_start, t_offset,
                             patient_id_from, patient_id_to,
                             body_from, body_to, patient_info, serial,
                             customer_idx, sample_idx):
    """
    Convert a generated exchange token sequence into rows matching the
    generated_sequences_pool.csv schema:
      SN, customer_idx, sample_idx, step, token_id, token_name,
      BodyGroup_from, BodyGroup_to, BodyGroup_from_text, BodyGroup_to_text,
      PatientID_from, PatientID_to,
      predicted_mu, predicted_sigma, sampled_duration, total_time,
      timediff, datetime
    """
    validate_exchange_sequence(
        tokens,
        START_TOKEN_ID,
        END_TOKEN_ID,
        PAD_TOKEN_ID,
        unknown_token_id=SOURCEID_VOCAB.get('UNK'),
    )

    tokens_list    = tokens.cpu().tolist()
    durations_list = durations.cpu().tolist()
    mu_list        = mu.cpu().tolist()
    sigma_list     = sigma.cpu().tolist()

    body_from_text = _region_name(body_from)
    body_to_text   = _region_name(body_to)

    # First pass: collect per-token data and accumulate total block duration
    token_data = []
    t = t_offset
    block_start = t_offset
    for i, tok in enumerate(tokens_list):
        if tok in (START_TOKEN_ID, END_TOKEN_ID, PAD_TOKEN_ID):
            continue
        # Duration of the token at generated index i is PREDICTED at index
        # i-1: training aligns mu[j] with target_durations[j], and targets
        # are the input shifted left by one (input [START, t0, ...] vs
        # target [t0, ...]), so mu[j] models the duration of token t_j which
        # sits at generated index j+1.
        j = i - 1
        raw_dur = durations_list[j] if 0 <= j < len(durations_list) else 0.0
        if not math.isfinite(float(raw_dur)):
            raise GenerationIntegrityError(
                f"exchange token {tok} has a non-finite sampled duration"
            )
        dur_sec = max(0.0, raw_dur) * EXCHANGE_DURATION_SCALE
        token_data.append({
            'tok':      tok,
            'raw_dur':  raw_dur,
            'mu':       mu_list[j] if 0 <= j < len(mu_list) else 0.0,
            'sigma':    sigma_list[j] if 0 <= j < len(sigma_list) else 0.0,
            't':        t,
            'timediff': round(t - block_start, 2),
            'dt':       day_start + timedelta(seconds=t),
        })
        t += dur_sec
    total_time = t - t_offset  # total block duration in seconds

    # Second pass: build rows with total_time filled in.
    # Include Age/Weight/Height/Direction/PTAB so synthetic exchange CSVs
    # match the real step 01 schema and keep patient conditioning if fed
    # back through step 03.
    _age       = round(patient_info.get('age',    0) or 0, 1)
    _weight    = round(patient_info.get('weight', 0) or 0, 1)
    _height    = round(patient_info.get('height', 0) or 0, 2)
    _direction = patient_info.get('direction', 'Head First')
    _ptab      = patient_info.get('ptab', 0) or 0

    rows = []
    for step, td in enumerate(token_data):
        rows.append({
            'SN':                str(serial),
            'customer_idx':      customer_idx,
            'sample_idx':        sample_idx,
            'step':              step,
            'token_id':          td['tok'],
            'token_name':        ID_TO_SOURCEID.get(td['tok'], 'UNK'),
            'BodyGroup_from':    body_from,
            'BodyGroup_to':      body_to,
            'BodyGroup_from_text': body_from_text,
            'BodyGroup_to_text': body_to_text,
            'PatientID_from':    patient_id_from,
            'PatientID_to':      patient_id_to,
            'predicted_mu':      td['mu'],
            'predicted_sigma':   td['sigma'],
            'sampled_duration':  td['raw_dur'],
            'total_time':        total_time,
            'timediff':          td['timediff'],
            'datetime':          td['dt'].strftime('%Y-%m-%d %H:%M:%S'),
            'Age':               _age,
            'Weight':            _weight,
            'Height':            _height,
            'Direction':         _direction,
            'PTAB':              _ptab,
        })

    return rows, t


def _generate_exam_rows(tokens, durations, mu, sigma, day_start, t_offset,
                         patient_id, body_region_id, patient_info,
                         serial, step_counter, customer_idx, sample_idx,
                         sequence_type_name='other', protocol_name=''):
    """
    Convert a generated examination token sequence into measurement-level rows
    matching the exam CSV format from 02_exam_preprocessing.py, with model
    internals (predicted_mu, predicted_sigma, sampled_duration) added.

    Each MRI_MSR_100 → MRI_MSR_104/34 boundary = one row.
    """
    rows = []
    tokens_list    = tokens.cpu().tolist()
    durations_list = durations.cpu().tolist()
    mu_list        = mu.cpu().tolist()
    sigma_list     = sigma.cpu().tolist()

    body_name  = _region_name(body_region_id)
    ptab       = patient_info.get('ptab', 0)
    age        = round(patient_info.get('age',    50), 1)
    weight     = round(patient_info.get('weight', 75), 1)
    height     = round(patient_info.get('height', 1.75), 2)
    direction  = patient_info.get('direction', 'Head First')

    FINISH_MAP = {
        SOURCEID_VOCAB.get('MRI_MSR_104', 12): 'Successful',
        SOURCEID_VOCAB.get('MRI_MSR_34',  15): 'Stopped by User',
    }
    MSR_100 = SOURCEID_VOCAB.get('MRI_MSR_100', 10)

    validate_examination_sequence(
        tokens,
        START_TOKEN_ID,
        END_TOKEN_ID,
        PAD_TOKEN_ID,
        MSR_100,
        FINISH_MAP,
        unknown_token_id=SOURCEID_VOCAB.get('UNK'),
    )

    t           = t_offset
    msr_start   = None
    msr_start_t = None

    for i, tok in enumerate(tokens_list):
        if tok in (START_TOKEN_ID, END_TOKEN_ID, PAD_TOKEN_ID):
            continue

        if tok == MSR_100:
            msr_start   = day_start + timedelta(seconds=t)
            msr_start_t = t

        elif tok in FINISH_MAP and msr_start is not None:
            # The span TOTAL is trained at the position BEFORE the finish
            # token: training aligns mu[j] with target_durations[j], and the
            # finish token at input position n carries its total at target
            # index n-1. The duration head is supervised ONLY there — every
            # other position is unconstrained, so the span duration must be
            # read from this single index, never summed across the span.
            j = i - 1
            raw_span = durations_list[j] if 0 <= j < len(durations_list) else 0.0
            if not math.isfinite(float(raw_span)):
                raise GenerationIntegrityError(
                    f"examination finish token has non-finite duration for {patient_id}"
                )
            duration_sec = max(0.0, raw_span) * EXAMINATION_DURATION_SCALE

            if duration_sec < 1 or duration_sec > MAX_RENDERED_EXAM_DURATION_SEC:
                raise GenerationIntegrityError(
                    f"examination duration {duration_sec:.2f}s is outside the "
                    f"renderable range [1, {MAX_RENDERED_EXAM_DURATION_SEC:.0f}]"
                )

            msr_end = msr_start + timedelta(seconds=duration_sec)
            t       = msr_start_t + duration_sec

            step_counter[patient_id] = step_counter.get(patient_id, 0) + 1

            row = {
                'SN':             str(serial),
                'customer_idx':   customer_idx,
                'sample_idx':     sample_idx,
                'startTime':      msr_start.strftime('%Y-%m-%d %H:%M:%S'),
                'endTime':        msr_end.strftime('%Y-%m-%d %H:%M:%S'),
                'duration':       int(duration_sec),
                'timediff':       round(msr_start_t, 2),
                'sourceID':       ID_TO_SOURCEID.get(tok, 'UNK'),
                # Sequence reflects the scan type the model was conditioned
                # on for this scan (no longer a random pick).
                'Sequence':       sequence_type_name,
                # The protocol the model was CONDITIONED on for this scan,
                # drawn from the real distribution for this scanner/region/type
                # (no longer one of eight invented names that appear nowhere in
                # the real data). Same contract as `Sequence` above, and what
                # makes this column a shared Qlik dimension.
                'Protocol':       protocol_name,
                'PatientID':      patient_id,
                'BodyPart':       body_name,
                'BodyGroup':      body_name,
                'ConnectedCoils': '',
                'DOT':            False,
                'MissingBodyPart':   False,
                'MissingPatientID':  False,
                'PTAB':           ptab,
                'FinishEvent':    FINISH_MAP[tok],
                'pauseTime':      0.0,            # filled in post-loop
                'StepCount':      step_counter[patient_id],
                'Age':            age,
                'Weight':         weight,
                'Height':         height,
                'Direction':      direction,
                'predicted_mu':   mu_list[j] if 0 <= j < len(mu_list) else 0.0,
                'predicted_sigma': sigma_list[j] if 0 <= j < len(sigma_list) else 0.0,
                'sampled_duration': raw_span,
            }
            # Exam coil binary columns — default False
            for c in _EXAM_COIL_COLS:
                row[c] = False

            rows.append(row)
            msr_start = None

    if len(rows) != 1:
        raise GenerationIntegrityError(
            f"expected one rendered examination row, received {len(rows)}"
        )
    return rows, t


def _generate_valid_exam_rows(cond, region_id, seq_type, serial_idx,
                              day_start, current_t, patient_id, patient_info,
                              serial, step_counter, customer_idx, sample_idx,
                              protocol_pid=RARE_PROTOCOL_ID, protocol_name=''):
    """Generate one scan, then safely repair the last malformed sample if needed."""
    candidates = []
    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        with torch.no_grad():
            tokens, durations, mu, sigma = examination_model.generate(
                cond,
                {
                    'body_region': region_id,
                    'sequence_type': seq_type,
                    'serial_idx': serial_idx,
                    # SEQPARAMS FORK: use_sut_conditioning adds a trigger_mode
                    # embedding. Every training row carries 'none' (the field
                    # that encodes gating mode is still unidentified), so
                    # feeding anything else here would be out of distribution.
                    'trigger_mode': _SYNTH_TRIGGER_MODE,
                    # The strongest duration channel this checkpoint has, and
                    # the one it was retrained for. It reaches the duration
                    # encoder through duration_protocol_bias at EVERY token
                    # position — the same route that makes sequence_type the
                    # only channel the head has ever responded to. Omitting the
                    # key does not disable it, it pins it to RARE_PROTOCOL_ID.
                    'protocol': protocol_pid,
                },
                max_length=gen_config['max_length'],
                temperature=gen_config['temperature'],
                top_k=gen_config['top_k'],
                top_p=gen_config['top_p'],
                return_stats=True,
            )

        # Correct systematic per-sequence-type duration bias from the probe.
        # The correction factor brings the raw model output to probe target
        # means; it's absorbed into the weights once the improved architecture
        # is retrained.
        _cal_factor = _duration_cal.get(int(seq_type), 1.0)
        if _cal_factor != 1.0:
            durations = durations * _cal_factor

        candidates.append((tokens[0], durations[0], mu[0], sigma[0]))

        try:
            rows, next_t = _generate_exam_rows(
                tokens[0], durations[0], mu[0], sigma[0],
                day_start, current_t,
                patient_id, region_id, patient_info,
                serial, step_counter, customer_idx, sample_idx,
                sequence_type_name=ID_TO_SEQUENCE_TYPE.get(seq_type, 'other'),
                protocol_name=protocol_name,
            )
            return rows, next_t, attempt - 1, False
        except GenerationIntegrityError:
            pass

    # Compatibility fallback for checkpoints/kernels that still emit malformed
    # control structure. Keep the latest valid-looking sampled span and its
    # shifted finish duration; if that duration is unusable, draw from the real
    # pkl distribution for the same scanner/region/sequence context.
    finish_ids = (
        SOURCEID_VOCAB['MRI_MSR_104'],
        SOURCEID_VOCAB['MRI_MSR_34'],
    )
    selected = None
    for candidate in reversed(candidates):
        tokens, durations, mu, sigma = candidate
        repaired_list, duration_source_idx = repair_examination_sequence(
            tokens,
            START_TOKEN_ID,
            END_TOKEN_ID,
            PAD_TOKEN_ID,
            SOURCEID_VOCAB['MRI_MSR_100'],
            finish_ids,
            SOURCEID_VOCAB.get('UNK'),
        )
        selected = (
            tokens, durations, mu, sigma, repaired_list, duration_source_idx
        )
        if duration_source_idx is None or duration_source_idx >= len(durations):
            continue
        duration_sec = (
            float(durations[duration_source_idx].item()) *
            EXAMINATION_DURATION_SCALE
        )
        if math.isfinite(duration_sec) and 1 <= duration_sec <= MAX_RENDERED_EXAM_DURATION_SEC:
            break

    tokens, durations, mu, sigma, repaired_list, duration_source_idx = selected
    repaired_tokens = torch.tensor(
        repaired_list, dtype=tokens.dtype, device=tokens.device
    )

    finish_idx = len(repaired_list) - 2
    target_idx = finish_idx - 1

    def _align_finish_value(values, default=0.0):
        aligned = torch.full(
            (len(repaired_list),),
            float(default),
            dtype=values.dtype,
            device=values.device,
        )
        if duration_source_idx is not None and duration_source_idx < len(values):
            aligned[target_idx] = values[duration_source_idx]
        return aligned

    repaired_durations = _align_finish_value(durations)
    repaired_mu = _align_finish_value(mu)
    repaired_sigma = _align_finish_value(sigma)

    raw_duration = float(repaired_durations[target_idx].item())
    duration_sec = raw_duration * EXAMINATION_DURATION_SCALE
    if not math.isfinite(duration_sec) or not 1 <= duration_sec <= MAX_RENDERED_EXAM_DURATION_SEC:
        raw_duration = _sample_fallback_exam_duration(
            region_id, seq_type, serial_idx
        )
        repaired_durations[target_idx] = raw_duration
        if not math.isfinite(float(repaired_mu[target_idx].item())):
            repaired_mu[target_idx] = raw_duration
        if not math.isfinite(float(repaired_sigma[target_idx].item())):
            repaired_sigma[target_idx] = 0.0

    rows, next_t = _generate_exam_rows(
        repaired_tokens, repaired_durations, repaired_mu, repaired_sigma,
        day_start, current_t,
        patient_id, region_id, patient_info,
        serial, step_counter, customer_idx, sample_idx,
        sequence_type_name=ID_TO_SEQUENCE_TYPE.get(seq_type, 'other'),
        protocol_name=protocol_name,
    )
    return rows, next_t, MAX_GENERATION_ATTEMPTS, True


def _fill_pause_times(rows):
    """Compute pauseTime = gap between this measurement's endTime and next's startTime.

    Only counts intra-day gaps. Overnight gaps (end and next start on different
    calendar days) are set to 0 — those are scanner downtime, not examination
    pauses, and would otherwise inflate the mean by 8–14 hours per day boundary.
    """
    for i in range(len(rows) - 1):
        end_i   = pd.to_datetime(rows[i]['endTime'])
        start_j = pd.to_datetime(rows[i + 1]['startTime'])
        if end_i.date() == start_j.date():
            rows[i]['pauseTime'] = max(0.0, (start_j - end_i).total_seconds())
        else:
            rows[i]['pauseTime'] = 0.0
    if rows:
        rows[-1]['pauseTime'] = 0.0
    return rows


def _fill_exam_timediff(rows):
    """
    Overwrite per-row timediff with inter-exam delta on the same scanner,
    matching the semantic step 02 emits for real CSVs (`startTime.diff()`
    on a scanner-sorted frame). The in-loop `round(msr_start_t, 2)` value
    set inside `_generate_exam_rows` is cumulative-from-day-start, which
    diverges from real and silently breaks any inter-exam-gap chart in
    Qlik. Same-scanner ordering matches consolidate.py's backfill.
    """
    if not rows:
        return rows
    df = pd.DataFrame({
        'i':  range(len(rows)),
        'SN': [r['SN'] for r in rows],
        'st': pd.to_datetime([r['startTime'] for r in rows]),
    }).sort_values(['SN', 'st'], kind='stable')
    df['td'] = df.groupby('SN')['st'].diff().dt.total_seconds().fillna(0)
    for i, td in zip(df['i'].values, df['td'].values):
        rows[i]['timediff'] = round(float(td), 2)
    return rows


def _generate_orch_tokens(scanner_idx, date, demographic_distributions, serial_str=None):
    """
    Use orchestration model to predict the body region sequence for one day.
    Returns a list of body region token IDs (breaks already filtered out).

    The conditioning MUST use this scanner's real region histogram (the same
    one training fed) — see `scanner_stats` above. A uniform fallback is only
    used if the scanner is unknown, and it is flagged because it reproduces the
    body-region collapse the model exhibits under out-of-distribution input.
    """
    from AlternatingPipeline.data.orchestration_preprocessing import _build_orchestration_conditioning
    dt = datetime.strptime(str(date), '%Y-%m-%d')
    dow = dt.weekday()
    stats = scanner_stats.get(serial_str) if serial_str is not None else None
    if stats is None:
        print(f"    !! no scanner_stats for {serial_str} — falling back to UNIFORM "
              f"conditioning (body-region output will be unreliable).")
        stats = {
            'avg_patients_per_day': 8.0,
            'region_distribution':  np.ones(NUM_BODY_REGIONS) / NUM_BODY_REGIONS,
        }
    cond = _build_orchestration_conditioning(str(date), dow, stats)
    cond_t = torch.tensor(cond, dtype=torch.float32).unsqueeze(0).to(device)
    scanner_t = torch.tensor([scanner_idx], dtype=torch.long).to(device)

    # Region-support mask: only regions this scanner actually sees in real
    # schedules may be sampled (plus END/BREAK so sequences still terminate
    # and pause). Defense-in-depth for the phantom-class-weight collapse: a
    # residually-biased checkpoint cannot emit NECK/FOOT on a scanner whose
    # real mix has zero share there.
    region_dist = np.asarray(stats['region_distribution'], dtype=float)
    allowed_regions = {
        i for i in range(NUM_BODY_REGIONS) if region_dist[i] > 0.0
    }
    if not allowed_regions:
        raise GenerationIntegrityError(
            f"scanner {serial_str} has no supported body regions in scanner_stats"
        )
    allowed_tokens = allowed_regions | {END_REGION_ID, BREAK_TOKEN_ID}

    # Invalid-region fallback is intentionally forbidden: sampling without the
    # scanner mask and deleting tokens afterwards changes patient counts and
    # hides model failures. A stale Databricks module now fails with a clear
    # restart instruction instead of emitting a lossy day.
    try:
        with torch.no_grad():
            tokens = orchestration_model.generate(
                cond_t,
                scanner_t,
                max_length=ORCHESTRATION_MODEL_CONFIG['max_seq_len'],
                temperature=GENERATION_CONFIG['temperature'],
                top_k=GENERATION_CONFIG['top_k'],
                top_p=GENERATION_CONFIG['top_p'],
                allowed_tokens=allowed_tokens,
            )
    except TypeError as exc:
        raise RuntimeError(
            "Loaded OrchestrationModel does not support constrained decoding. "
            "Restart Python and rerun step 07 from the top."
        ) from exc

    return validate_orchestration_sequence(
        tokens[0],
        START_REGION_ID,
        END_REGION_ID,
        ORCH_PAD_TOKEN_ID,
        BREAK_TOKEN_ID,
        allowed_regions,
    )

# COMMAND ----------

# =============================================================================
# MAIN GENERATION LOOP — one synthetic day per customer per date
# =============================================================================

gen_config = GENERATION_CONFIG

for customer_idx, (serial_str, daily_schedules) in enumerate(customer_schedules.items()):
    print(f"\n{'='*60}\nSerial: {serial_str}\n{'='*60}")

    scanner_idx = scanner_to_idx.get(serial_str, 0)
    # serial_idx indexes the examination model's serial embedding. customer
    # iteration order matches step 03's SERIAL_NUMBERS order, so customer_idx
    # is the serial_idx step 03 wrote into the pkl.
    #
    # Clamped against the CHECKPOINT's embedding height, not a config constant —
    # an out-of-range lookup is an IndexError on CPU and a CUDA device-side
    # assert on GPU, and a too-low constant is worse still because it silently
    # collapses distinct scanners onto one row instead of failing.
    serial_idx = min(customer_idx, EXAM_NUM_SERIALS - 1)

    all_exchange_rows  = []
    all_exam_rows      = []
    exchange_sample_idx = 0   # increments per exchange block (startup/between/shutdown)
    exam_sample_idx     = 0   # increments per patient examined
    expected_exam_counts = {}
    expected_exchange_blocks = 0
    regeneration_count = 0
    fallback_exam_count = 0
    day_budget = DayBudget(MAX_DAY_SPAN_SEC)

    # Generate dates from the synthetic range (outside training window — no leakage)
    _start = datetime.strptime(SYNTH_DATE_START, '%Y-%m-%d')
    _end   = datetime.strptime(SYNTH_DATE_END,   '%Y-%m-%d')
    dates  = []
    _d = _start
    while _d <= _end and len(dates) < NUM_DAYS_PER_SCANNER:
        if not WEEKDAYS_ONLY or _d.weekday() < 5:
            dates.append(_d.strftime('%Y-%m-%d'))
        _d += timedelta(days=1)
    if not dates:
        print(f"  No dates in synthetic range — skipping.")
        continue

    step_counter = {}   # tracks StepCount per patient_id across the day
    intra_gap_total_s = 0.0   # seconds of pause inserted BETWEEN scans of one visit
    intra_gap_count = 0

    for date_str in dates:
        print(f"  Day: {date_str}")
        day_start = datetime.strptime(date_str, '%Y-%m-%d').replace(hour=7, minute=0)

        # --- Decide the day's patient body-region sequence ---
        if CONTROLLED_SCHEDULE is not None:
            # Controlled experiment: use the hand-authored mix verbatim.
            orch_tokens = [BODY_REGION_TO_ID.get(str(r).upper(), 10)
                           for r in CONTROLLED_SCHEDULE]
        else:
            # Realistic default: let the orchestration model decide.
            orch_tokens = _generate_orch_tokens(scanner_idx, date_str, demographic_distributions,
                                                serial_str=serial_str)
        if not orch_tokens:
            print(f"    Orchestration returned no tokens — skipping day.")
            continue

        # Build patient list from orchestration output
        patients = []
        for tok in orch_tokens:
            demographics = _sample_demographics(tok, demographic_distributions)
            patients.append({
                'patient_id':     f'SYNTH_{date_str}_{len(patients):03d}',
                'body_region_id': tok,
                'body_region':    _region_name(tok),
                **demographics,
            })

        print(f"    {len(patients)} patients from orchestration")

        current_t       = 0.0
        prev_region     = START_REGION_ID
        prev_patient_id = 'START'
        step_counter    = {}
        patients_served = 0
        last_served     = None
        day_dropped_scans = 0

        for p_idx, patient in enumerate(patients):
            # A scanner day has to end. Without this the day runs on until the
            # orchestration plan is exhausted, and a day longer than 24 h
            # starts before the previous one has finished — see DayBudget.
            if not day_budget.accepts_patient(current_t, patients_served):
                break

            pat_id     = patient['patient_id']
            region_id  = patient['body_region_id']
            phase_type = PHASE_TYPES['startup'] if p_idx == 0 else PHASE_TYPES['between']

            cond = _build_cond_tensor(patient, current_t, day_start)

            # ── EXCHANGE ──
            with torch.no_grad():
                ex_tokens, ex_durations, ex_mu, ex_sigma = exchange_model.generate(
                    cond,
                    {'body_from': prev_region, 'body_to': region_id},
                    phase_type=phase_type,
                    max_length=gen_config['max_length'],
                    temperature=gen_config['temperature'],
                    top_k=gen_config['top_k'],
                    top_p=gen_config['top_p'],
                    return_stats=True,
                )

            ex_rows, current_t = _generate_exchange_rows(
                ex_tokens[0], ex_durations[0], ex_mu[0], ex_sigma[0],
                day_start, current_t,
                prev_patient_id, pat_id,
                prev_region, region_id,
                patient, serial_str,
                customer_idx, exchange_sample_idx,
            )
            all_exchange_rows.extend(ex_rows)
            exchange_sample_idx += 1
            expected_exchange_blocks += 1

            # ── EXAMINATION ──
            # Training samples are per-measurement (one MSR_100 → MSR_104 span),
            # so each examination_model.generate() call produces a single scan.
            # Real patients receive multiple scans per visit (mean ≈ 8, p90 ≈ 13
            # on serials 176148/183242), so we loop N times per patient and
            # concatenate the resulting measurement rows.  Without this loop the
            # synthetic output degenerates to exactly 1 scan per patient.
            num_scans = max(1, int(np.random.poisson(8)))
            num_scans = min(num_scans, 20)          # clamp the tail
            scans_rendered = 0
            for _scan in range(num_scans):
                # Draw a scan type from this region's real mix and condition
                # the model on it — this is what gives duration variability
                # (short scouts vs long TSE/SPACE) instead of a flat mean.
                # Drawn BEFORE the budget check now, because the gap that
                # precedes this scan depends on which scan it is.
                seq_type = _sample_sequence_type(region_id)

                # The pause before this scan: the operator planning the slab,
                # repositioning, coaching a breath-hold, injecting contrast.
                # Attributed to the scan it PRECEDES, which is why seq_type is
                # drawn first — the pause before a scout is a stage change and
                # runs 5-10x longer than the rest (median 168s vs 19s).
                # Not applied before the FIRST scan of a visit: that pause is
                # the patient handover, which the exchange block already covers.
                _gap = _gap_sampler.sample(seq_type, _GAP_RNG) if _scan else 0.0

                # Budget-checked at the POST-gap clock, and only committed if the
                # scan actually happens — otherwise a truncated day would still
                # burn the pause and push the shutdown exchange later.
                if not day_budget.accepts_scan(current_t + _gap, scans_rendered):
                    day_dropped_scans += num_scans - scans_rendered
                    break
                current_t += _gap
                intra_gap_total_s += _gap
                intra_gap_count += int(_gap > 0)
                # SEQPARAMS FORK: widened conditioning + sampled TR/num_slices.
                cond, _sut_values = _build_exam_cond_tensor(
                    patient, current_t, day_start, region_id, seq_type
                )
                sut_draw_counter[tuple(sorted(_sut_values.items()))] += 1
                # One draw serves both consumers: the id conditions the model,
                # the raw name goes into the CSV. Drawing them separately would
                # let the two disagree about what was generated.
                _proto_pid, _proto_name = protocol_sampler.sample(
                    serial_idx, region_id, seq_type
                )
                protocol_draw_counter[_proto_name] += 1
                exam_rows, current_t, retries, used_fallback = _generate_valid_exam_rows(
                    cond, region_id, seq_type, serial_idx,
                    day_start, current_t, pat_id, patient,
                    serial_str, step_counter, customer_idx, exam_sample_idx,
                    protocol_pid=_proto_pid, protocol_name=_proto_name,
                )
                regeneration_count += retries
                fallback_exam_count += int(used_fallback)
                all_exam_rows.extend(exam_rows)
                scans_rendered += 1
            # Record what was actually rendered, not what was planned — the
            # integrity check compares these two and a truncated patient must
            # not read as a lost scan.
            expected_exam_counts[pat_id] = scans_rendered
            exam_sample_idx += 1   # one sample_idx per patient (matches step 02)

            patients_served += 1
            last_served     = patient
            prev_region     = region_id
            prev_patient_id = pat_id

        day_budget.record_day(len(patients) - patients_served, day_dropped_scans)
        if patients_served < len(patients) or day_dropped_scans:
            print(f"    !! day truncated at {current_t / 3600:.1f}h — served "
                  f"{patients_served}/{len(patients)} patients, dropped "
                  f"{day_dropped_scans} scan(s)")

        # ── SHUTDOWN EXCHANGE ──
        if last_served is not None:
            last_patient = last_served
            cond = _build_cond_tensor(last_patient, current_t, day_start)
            with torch.no_grad():
                sd_tokens, sd_durations, sd_mu, sd_sigma = exchange_model.generate(
                    cond,
                    {'body_from': prev_region, 'body_to': END_REGION_ID},
                    phase_type=PHASE_TYPES['shutdown'],
                    max_length=gen_config['max_length'],
                    temperature=gen_config['temperature'],
                    top_k=gen_config['top_k'],
                    top_p=gen_config['top_p'],
                    return_stats=True,
                )
            sd_rows, _ = _generate_exchange_rows(
                sd_tokens[0], sd_durations[0], sd_mu[0], sd_sigma[0],
                day_start, current_t,
                prev_patient_id, 'END',
                prev_region, END_REGION_ID,
                last_patient, serial_str,
                customer_idx, exchange_sample_idx,
            )
            all_exchange_rows.extend(sd_rows)
            exchange_sample_idx += 1
            expected_exchange_blocks += 1

    # ── Fill pause times for exam rows ──
    all_exam_rows = _fill_pause_times(all_exam_rows)
    all_exam_rows = _fill_exam_timediff(all_exam_rows)

    # Fail before writing CSVs if any planned scan/block disappeared or a row
    # violates the render contract. Warn-only mode is available for diagnosis,
    # but strict validation is the default for production-quality output.
    try:
        integrity_report = validate_rendered_output(
            all_exchange_rows,
            all_exam_rows,
            expected_exam_counts,
            expected_exchange_blocks,
            BODY_REGIONS,
        )
    except GenerationIntegrityError as exc:
        if STRICT_OUTPUT_VALIDATION:
            raise
        print(f"  !! OUTPUT INTEGRITY WARNING (warn-only mode): {exc}")
    else:
        print(
            "  Output integrity: PASS  "
            f"patients={integrity_report['patients']}  "
            f"scans={integrity_report['exam_rows']}  "
            f"exchange_blocks={integrity_report['exchange_blocks']}  "
            f"regenerated_scans={regeneration_count}  "
            f"fallback_scans={fallback_exam_count}"
        )

    _budget_summary = day_budget.summary()
    if _budget_summary:
        print(f"  !! {_budget_summary}")

    # ── Duration diagnostic: surface model-vs-real calibration up front ──
    # Reference values are measured over all 40,921 rows of the real exam CSVs
    # (qlik/data/real/exam): mean 105.2 s, median 89 s, p99 392 s, and only
    # 0.06% of scans longer than 600 s. A synthetic mean far off that band, or
    # a long-scan share orders of magnitude above it, means the duration model
    # is mis-calibrated and the Qlik comparison will look broken regardless of
    # the rest of the pipeline.
    #
    # The mean alone is not enough: the 2026-07-28 run reported mean 274.4 s
    # with median 49.0 s — a right median with a tail heavy enough to nearly
    # triple the mean — and slipped under the old 30-300 s band unflagged.
    # Median and the >600 s share are what separate "shifted" from "over-
    # dispersed", so both are printed.
    if all_exam_rows:
        durs = sorted(r['duration'] for r in all_exam_rows if r.get('duration', 0) > 0)
        if durs:
            mean_dur = sum(durs) / len(durs)
            med_dur  = durs[len(durs) // 2]
            p99_dur  = durs[min(len(durs) - 1, int(0.99 * len(durs)))]
            long_share = 100.0 * sum(d > 600 for d in durs) / len(durs)
            flags = []
            if not 60 <= mean_dur <= 200:
                flags.append('mean off-band (real 105 s, per-serial 75-168 s)')
            if long_share > 3.0:
                flags.append(f'{long_share:.1f}% of scans >600 s (real 0.06%) — over-dispersed')
            flag = ('  ⚠ ' + '; '.join(flags)) if flags else ''
            print(f"  Exam duration: mean={mean_dur:.1f}s  median={med_dur:.1f}s  "
                  f"p99={p99_dur:.0f}s  >600s={long_share:.2f}%  n={len(durs)}{flag}")

            # What the intra-visit pause added, in the same units. Real data
            # puts 30.4% of examination wall clock here; a share far below that
            # means the prior did not load or the visits are too short to
            # contain any gaps at all.
            _scan_total = sum(durs)
            if intra_gap_count:
                _share = 100.0 * intra_gap_total_s / max(1.0, _scan_total + intra_gap_total_s)
                print(f"  Intra-visit pause: {intra_gap_count:,} gaps  "
                      f"mean={intra_gap_total_s/intra_gap_count:.1f}s  "
                      f"total={intra_gap_total_s/3600:.1f}h  "
                      f"{_share:.1f}% of examination wall clock (real 30.4%)"
                      + ('' if 20.0 <= _share <= 42.0
                         else '  ⚠ off the real share'))
            else:
                print("  Intra-visit pause: NONE INSERTED — examination wall clock "
                      "will read ~30% short (real: 30.4% of it is between-scan pause)")

    # ── Save exchange CSV ──
    if all_exchange_rows:
        df_ex = pd.DataFrame(all_exchange_rows)
        ex_path = f"{SYNTH_EXCHANGE}/DATA_{serial_str}.csv"
        df_ex.to_csv(ex_path, index=False)
        print(f"  Exchange CSV: {len(df_ex):,} rows → {ex_path}")

    # ── Save exam CSV ──
    if all_exam_rows:
        df_exam = pd.DataFrame(all_exam_rows)
        exam_path = f"{SYNTH_EXAM}/DATA_{serial_str}.csv"
        df_exam.to_csv(exam_path, index=False)
        print(f"  Exam CSV:     {len(df_exam):,} rows → {exam_path}")

# COMMAND ----------

# =============================================================================
# Download links
# =============================================================================

links = ""
for serial_str in customer_schedules.keys():
    links += f'<li><a href="/files/csv_pipeline_seqparams/synthetic/exchange/DATA_{serial_str}.csv">Exchange {serial_str}</a> &nbsp;|&nbsp; '
    links += f'<a href="/files/csv_pipeline_seqparams/synthetic/exam/DATA_{serial_str}.csv">Exam {serial_str}</a></li>\n'

displayHTML(f'<h3>Synthetic CSVs</h3><ul>{links}</ul>')

# COMMAND ----------

# =============================================================================
# Visualizations — synthetic data quality
# =============================================================================

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob as _glob

# ── Load all synthetic CSVs ──────────────────────────────────────────────────
ex_files   = _glob.glob(f"{SYNTH_EXCHANGE}/DATA_*.csv")
exam_files = _glob.glob(f"{SYNTH_EXAM}/DATA_*.csv")

df_ex_all   = pd.concat([pd.read_csv(f) for f in ex_files],   ignore_index=True) if ex_files   else pd.DataFrame()
df_exam_all = pd.concat([pd.read_csv(f) for f in exam_files], ignore_index=True) if exam_files else pd.DataFrame()

print(f"Loaded {len(df_ex_all):,} exchange rows and {len(df_exam_all):,} exam rows across {len(ex_files)} scanners.")

# ── Figure 1: Exam overview (3 panels) ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Synthetic Examination Data', fontsize=14, fontweight='bold')

if not df_exam_all.empty:
    # Body region distribution
    region_counts = df_exam_all['BodyPart'].value_counts()
    axes[0].bar(region_counts.index, region_counts.values, color='steelblue', edgecolor='white')
    axes[0].set_title('Body Region Distribution')
    axes[0].set_xlabel('Body Region')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)

    # Examination duration histogram
    durations = df_exam_all['duration'].dropna()
    durations = durations[(durations > 0) & (durations < 4000)]
    axes[1].hist(durations / 60, bins=40, color='steelblue', edgecolor='white')
    axes[1].set_title('Examination Duration')
    axes[1].set_xlabel('Duration (minutes)')
    axes[1].set_ylabel('Count')
    axes[1].axvline(durations.mean() / 60, color='red', linestyle='--', linewidth=1.5, label=f'Mean {durations.mean()/60:.1f} min')
    axes[1].legend()

    # Rows per scanner
    scanner_counts = df_exam_all['SN'].value_counts().sort_index()
    axes[2].bar(scanner_counts.index.astype(str), scanner_counts.values, color='steelblue', edgecolor='white')
    axes[2].set_title('Exam Rows per Scanner')
    axes[2].set_xlabel('Scanner SN')
    axes[2].set_ylabel('Count')
    axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
display(fig)
plt.close(fig)

# ── Figure 2: Duration by body region (box plot) ────────────────────────────
if not df_exam_all.empty:
    regions = df_exam_all['BodyPart'].dropna().unique()
    region_data = [
        df_exam_all.loc[df_exam_all['BodyPart'] == r, 'duration'].dropna().values / 60
        for r in regions
    ]
    region_data = [d[(d > 0) & (d < 4000 / 60)] for d in region_data]

    fig2, ax2 = plt.subplots(figsize=(14, 5))
    ax2.boxplot(region_data, labels=regions, patch_artist=True,
                boxprops=dict(facecolor='steelblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
    ax2.set_title('Examination Duration by Body Region', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Body Region')
    ax2.set_ylabel('Duration (minutes)')
    ax2.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    display(fig2)
    plt.close(fig2)

# ── Figure 3: Exchange event type distribution ───────────────────────────────
# Exchange CSVs use the 'token_name' column (renamed from 'sourceID' in step 01)
_ex_token_col = 'token_name' if 'token_name' in df_ex_all.columns else (
    'sourceID' if 'sourceID' in df_ex_all.columns else None
)
if not df_ex_all.empty and _ex_token_col is not None:
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
    fig3.suptitle('Synthetic Exchange Data', fontsize=14, fontweight='bold')

    event_counts = df_ex_all[_ex_token_col].value_counts()
    axes3[0].barh(event_counts.index, event_counts.values, color='teal', edgecolor='white')
    axes3[0].set_title('Event Type Distribution')
    axes3[0].set_xlabel('Count')
    axes3[0].invert_yaxis()

    timediff = df_ex_all['timediff'].dropna()
    timediff = timediff[(timediff >= 0) & (timediff < 3600)]
    axes3[1].hist(timediff, bins=40, color='teal', edgecolor='white')
    axes3[1].set_title('Time Between Events')
    axes3[1].set_xlabel('Seconds')
    axes3[1].set_ylabel('Count')

    plt.tight_layout()
    display(fig3)
    plt.close(fig3)

# ── Figure 4: Daily patient count per scanner ───────────────────────────────
if not df_exam_all.empty and 'startTime' in df_exam_all.columns:
    df_exam_all['date'] = pd.to_datetime(df_exam_all['startTime']).dt.date
    daily = df_exam_all.groupby(['SN', 'date'])['PatientID'].nunique().reset_index()
    daily.columns = ['SN', 'date', 'patients']

    scanners = daily['SN'].unique()
    fig4, ax4 = plt.subplots(figsize=(16, 5))
    for sn in scanners:
        d = daily[daily['SN'] == sn].sort_values('date')
        ax4.plot(d['date'], d['patients'], marker='o', markersize=3, linewidth=1, label=str(sn), alpha=0.7)
    ax4.set_title('Unique Patients per Day per Scanner', fontsize=13, fontweight='bold')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Patients')
    ax4.legend(title='Scanner', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
    ax4.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    display(fig4)
    plt.close(fig4)

# COMMAND ----------

# =============================================================================
# Text evaluation report — copy/paste this to share for model improvement
# =============================================================================

from scipy import stats as _scipy_stats

_W  = 72   # report width
_HR = '─' * _W

def _pct(n, total): return f"{100*n/total:.1f}%" if total > 0 else "N/A"
def _safe_stat(arr, fn):
    try: return fn(arr[~np.isnan(arr)])
    except: return float('nan')

# =============================================================================
# THE VISIT KEY, and why the report needs one.
#
# `PatientID` is f'SYNTH_{date}_{nnn}' — it carries the DATE but not the SERIAL,
# so patient 007 on scanner A and patient 007 on scanner B, same day, share an
# id. Each scanner writes its own CSV so nothing is ambiguous on disk, but this
# report concatenates all 20 of them: a bare groupby('PatientID') collapsed
# 7,563 generated visits into ~624 buckets, which is why section 8 read
# "12.6 scans per patient" against a generator that draws Poisson(8), and why
# section 11's weekday load was ~1/12 of the patients actually generated.
#
# (SN, PatientID) is unique per visit and is what every per-patient aggregate
# below groups on. Sections 2 and 4's per-scanner tables were already safe —
# they carry SN in the group key.
# =============================================================================
_VISIT_KEYS = ['SN', 'PatientID']

# Real-data reference values, so a synthetic number is never read in isolation.
# Measured on the ten real exam CSVs in csv_pipeline/qlik/data/real/exam
# (41,106 scans, 4,747 visits, Jan 2024), visits segmented on StepCount resets:
#   per-scan mean 1.75 min  ·  scans/visit mean 8.6, median 7
#   visit scan-time mean 15.1 min  ·  visit WALL-CLOCK mean 21.7, median 18.3
#   intra-visit gap between consecutive scans: mean 49.5s, median 19s (3% zero)
REAL_REF = {
    'scan_min':        1.75,
    'scans_per_visit': 8.6,
    'visit_scan_min':  15.08,
    'visit_wall_min':  21.67,
    'visit_wall_med':  18.33,
    'intra_gap_s':     49.5,
}

lines = []
lines += [
    '=' * _W,
    'SYNTHETIC DATA EVALUATION REPORT'.center(_W),
    f'Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}    '
    f'Range: {SYNTH_DATE_START} → {SYNTH_DATE_END}',
    '=' * _W,
]

# ── 1. OVERVIEW ──────────────────────────────────────────────────────────────
lines += ['', _HR, ' 1. OVERVIEW', _HR]
n_scanners = len(ex_files)
n_ex_rows  = len(df_ex_all)
n_exam_rows= len(df_exam_all)
lines += [
    f'  Scanners generated : {n_scanners}',
    f'  Exchange rows      : {n_ex_rows:,}',
    f'  Exam rows          : {n_exam_rows:,}',
    f'  Avg exchange/scanner: {n_ex_rows/n_scanners:,.0f}' if n_scanners else '',
    f'  Avg exam/scanner   : {n_exam_rows/n_scanners:,.0f}' if n_scanners else '',
]

# ── 2. PATIENT THROUGHPUT ─────────────────────────────────────────────────────
lines += ['', _HR, ' 2. PATIENT THROUGHPUT (unique patients per scanner-day)', _HR]
if not df_exam_all.empty and 'startTime' in df_exam_all.columns:
    df_exam_all['_date'] = pd.to_datetime(df_exam_all['startTime']).dt.date
    daily_pts = df_exam_all.groupby(['SN', '_date'])['PatientID'].nunique()
    vals = daily_pts.values
    lines += [
        f'  Overall  — mean: {vals.mean():.1f}  std: {vals.std():.1f}  '
        f'min: {vals.min()}  median: {int(np.median(vals))}  max: {vals.max()}',
        '',
        f'  {"Scanner":<12} {"Days":>5} {"Mean":>6} {"Std":>6} {"Min":>5} {"Med":>5} {"Max":>5}  {"Flag"}',
    ]
    for sn, grp in daily_pts.groupby(level='SN'):
        v = grp.values
        flag = ''
        if v.mean() < 5:  flag = '<< very low throughput'
        elif v.mean() > 25: flag = '>> very high throughput'
        elif v.std() > 10:  flag = '!! high day-to-day variance'
        lines.append(
            f'  {str(sn):<12} {len(v):>5} {v.mean():>6.1f} {v.std():>6.1f} '
            f'{v.min():>5} {int(np.median(v)):>5} {v.max():>5}  {flag}'
        )

# ── 3. BODY REGION DISTRIBUTION ───────────────────────────────────────────────
lines += ['', _HR, ' 3. BODY REGION DISTRIBUTION (exam rows)', _HR]
if not df_exam_all.empty and 'BodyPart' in df_exam_all.columns:
    region_vc = df_exam_all['BodyPart'].value_counts()
    total_r   = region_vc.sum()
    lines.append(f'  {"Region":<12} {"Count":>7} {"Share":>7}')
    for region, cnt in region_vc.items():
        flag = ' !! dominant (>50%)' if cnt/total_r > 0.5 else \
               ' << rare (<3%)'     if cnt/total_r < 0.03 else ''
        lines.append(f'  {str(region):<12} {cnt:>7,} {_pct(cnt,total_r):>7}{flag}')
    # Entropy — higher = more balanced
    probs = region_vc.values / total_r
    entropy = -np.sum(probs * np.log(probs + 1e-9))
    max_entropy = np.log(len(region_vc))
    lines += [
        f'',
        f'  Diversity entropy: {entropy:.3f} / {max_entropy:.3f} '
        f'(1.0 = perfectly uniform)',
        f'  Normalised       : {entropy/max_entropy:.3f}  '
        + ('>> good spread' if entropy/max_entropy > 0.7 else '<< skewed — check orchestration model'),
    ]

# ── 4. PER-SEQUENCE DURATION ─────────────────────────────────────────────────
# One exam row = one SCAN (one MSR_100 -> MSR_104 span), NOT one examination.
# This section used to be titled "EXAMINATION DURATION", and a mean of 1.9 min
# read to anyone who knows MRI as a model that is off by 10x — a radiologist's
# "examination" is the patient's whole visit, which is section 4a below and is
# ~20 min in the real data. Both numbers are correct; only the label was wrong.
lines += ['', _HR, ' 4. PER-SEQUENCE (SINGLE SCAN) DURATION (minutes)', _HR]
lines += [f'  One row = one scan, not one examination. Real reference: '
          f'{REAL_REF["scan_min"]:.2f} min/scan.']
if not df_exam_all.empty and 'duration' in df_exam_all.columns:
    dur_all = df_exam_all['duration'].dropna().values / 60.0
    dur_all = dur_all[(dur_all > 0) & (dur_all < 4000/60)]
    if len(dur_all) == 0:
        lines.append('  No valid duration rows found.')
    else:
        lines += [
            f'  Overall  — mean: {dur_all.mean():.1f}  std: {dur_all.std():.1f}  '
            f'p10: {np.percentile(dur_all,10):.1f}  p50: {np.percentile(dur_all,50):.1f}  '
            f'p90: {np.percentile(dur_all,90):.1f}',
            '',
            f'  {"Region":<12} {"N":>5} {"Mean":>6} {"Std":>6} {"p10":>6} {"p50":>6} {"p90":>6}  {"Flag"}',
        ]
        for region in df_exam_all['BodyPart'].dropna().unique():
            d = df_exam_all.loc[df_exam_all['BodyPart']==region,'duration'].dropna().values / 60.0
            d = d[(d > 0) & (d < 4000/60)]
            if len(d) == 0: continue
            flag = ''
            if d.mean() < 1:    flag = '<< implausibly short'
            elif d.mean() > 60: flag = '>> implausibly long'
            elif d.std() > 30:  flag = '!! very high variance'
            lines.append(
                f'  {str(region):<12} {len(d):>5} {d.mean():>6.1f} {d.std():>6.1f} '
                f'{np.percentile(d,10):>6.1f} {np.percentile(d,50):>6.1f} '
                f'{np.percentile(d,90):>6.1f}  {flag}'
            )
        n_short = int((dur_all < 1).sum())
        n_long  = int((dur_all > 60).sum())
        lines += [
            '',
            f'  Quality flags: {n_short} exams <1 min ({_pct(n_short,len(dur_all))}),  '
            f'{n_long} exams >60 min ({_pct(n_long,len(dur_all))})',
        ]

# ── 4a. PER-EXAMINATION (PATIENT VISIT) DURATION ─────────────────────────────
# The aggregate a clinician means by "how long is an MRI examination": every
# scan for one patient, plus the dead time between those scans.
#
# Three different numbers, kept separate because they answer different things:
#   scan time  = sum of sequence durations. What the examination MODEL predicts.
#   wall clock = last endTime - first startTime. What the scanner is occupied
#                for, and what a scheduler needs. scan time PLUS the pauses.
#   the gap between them is the intra-visit dead time — planning, positioning,
#   breath-hold coaching, contrast. Real data: 30.4% of examination wall clock.
lines += ['', _HR, ' 4a. PER-EXAMINATION (PATIENT VISIT) DURATION (minutes)', _HR]
if (not df_exam_all.empty
        and {'duration', 'startTime', 'endTime', 'PatientID', 'SN'}.issubset(df_exam_all.columns)):
    _v = df_exam_all.copy()
    _v['_s'] = pd.to_datetime(_v['startTime'], errors='coerce')
    _v['_e'] = pd.to_datetime(_v['endTime'], errors='coerce')
    _v['_d'] = pd.to_numeric(_v['duration'], errors='coerce')
    _v = _v.dropna(subset=['_s', '_e', '_d'])
    if _v.empty:
        lines.append('  No parseable start/end times — cannot aggregate to visits.')
    else:
        _g = _v.groupby(_VISIT_KEYS)
        _visit = pd.DataFrame({
            'n_scans':  _g.size(),
            'scan_sec': _g['_d'].sum(),
            'first':    _g['_s'].min(),
            'last':     _g['_e'].max(),
        })
        _visit['wall_sec'] = (_visit['last'] - _visit['first']).dt.total_seconds()
        _visit = _visit[(_visit['wall_sec'] > 0) & (_visit['wall_sec'] < 6 * 3600)]
        _sc = _visit['scan_sec'].values / 60.0
        _wc = _visit['wall_sec'].values / 60.0
        _ns = _visit['n_scans'].values
        _dead_share = 1.0 - (_visit['scan_sec'].sum() / max(1.0, _visit['wall_sec'].sum()))

        def _cmp(v, ref):
            """Signed % difference against the real-data reference."""
            return f'{100 * (v - ref) / ref:+.0f}% vs real'

        lines += [
            f'  Visits (SN x PatientID): {len(_visit):,}',
            '',
            f'  {"":<26} {"mean":>7} {"median":>7} {"p90":>7}   {"real mean":>9}  {"delta":>14}',
            f'  {"scans per visit":<26} {_ns.mean():>7.1f} {np.median(_ns):>7.0f} '
            f'{np.percentile(_ns,90):>7.0f}   {REAL_REF["scans_per_visit"]:>9.1f}  '
            f'{_cmp(_ns.mean(), REAL_REF["scans_per_visit"]):>14}',
            f'  {"scan time (model)":<26} {_sc.mean():>7.1f} {np.median(_sc):>7.1f} '
            f'{np.percentile(_sc,90):>7.1f}   {REAL_REF["visit_scan_min"]:>9.1f}  '
            f'{_cmp(_sc.mean(), REAL_REF["visit_scan_min"]):>14}',
            f'  {"WALL CLOCK (occupancy)":<26} {_wc.mean():>7.1f} {np.median(_wc):>7.1f} '
            f'{np.percentile(_wc,90):>7.1f}   {REAL_REF["visit_wall_min"]:>9.1f}  '
            f'{_cmp(_wc.mean(), REAL_REF["visit_wall_min"]):>14}',
            '',
            f'  Intra-visit dead time: {100*_dead_share:.1f}% of wall clock '
            f'(real 30.4%).',
        ]
        if _dead_share < 0.05:
            lines += [
                '  << NO DEAD TIME BETWEEN SCANS. The generator advances the clock by',
                '     the scan duration alone, so scan N+1 starts exactly where scan N',
                '     ended. Real scans are separated by a median 19s / mean 49.5s gap.',
                '     Per-scan durations can be perfect and the EXAMINATION still comes',
                '     out ~35% short, because a visit is scans + gaps.',
            ]
        _wall_err = abs(_wc.mean() - REAL_REF['visit_wall_min']) / REAL_REF['visit_wall_min']
        if _wall_err > 0.15:
            lines.append(f'  !! Examination wall clock is {100*_wall_err:.0f}% off the real '
                         f'mean — this is the number a clinician judges the model by.')

# ── 4b. DURATION BY SEQUENCE TYPE ────────────────────────────────────────────
# Verifies Workstream A: a scout should be much shorter than a TSE/SPACE.
# If every row here has near-identical mean duration, the examination model
# is still not responding to the scan-type conditioning.
lines += ['', _HR, ' 4b. EXAMINATION DURATION BY SEQUENCE TYPE (minutes)', _HR]
if not df_exam_all.empty and {'duration', 'Sequence'}.issubset(df_exam_all.columns):
    lines.append(f'  {"Sequence":<14} {"N":>6} {"Mean":>6} {"Std":>6} {"p50":>6}')
    _st_means = []
    for stype, grp in df_exam_all.groupby('Sequence'):
        d = grp['duration'].dropna().values / 60.0
        d = d[(d > 0) & (d < 4000 / 60)]
        if len(d) == 0:
            continue
        _st_means.append(d.mean())
        lines.append(
            f'  {str(stype):<14} {len(d):>6} {d.mean():>6.1f} {d.std():>6.1f} '
            f'{np.percentile(d, 50):>6.1f}'
        )
    if len(_st_means) > 1:
        _spread = max(_st_means) - min(_st_means)
        lines.append(
            f'\n  Spread across sequence types: {_spread:.1f} min  '
            + ('>> good — model varies duration by scan type'
               if _spread > 1.0 else
               '<< flat — examination model not responding to scan-type conditioning')
        )

# ── 4c. SUT CONDITIONING ACTUALLY FED IN (SEQPARAMS FORK) ────────────────────
# Proves the SUT features reached the model and varied. A single distinct draw
# means the sampler collapsed and every scan was conditioned identically.
lines += ['', _HR, ' 4c. SUT PARAMETERS FED TO THE EXAMINATION MODEL', _HR]
if not EXTRA_FEATURES:
    lines.append('  No SUT features configured — examination conditioning is 10-dim.')
elif not sut_draw_counter:
    lines.append('  !! No SUT draws recorded — no scans were generated.')
else:
    _total_draws = sum(sut_draw_counter.values())
    lines.append(
        f'  {_total_draws:,} scans conditioned on {len(EXTRA_FEATURES)} SUT features; '
        f'{len(sut_draw_counter):,} distinct value combinations.'
    )
    lines.append('  Most frequent draws (first 6 fields of each):')
    for _combo, _cnt in sut_draw_counter.most_common(5):
        _rendered = ', '.join(f'{k}={v:g}' for k, v in _combo[:6])
        lines.append(f'    {_rendered:<60} {_cnt:>6,}  ({_pct(_cnt, _total_draws)})')
    if len(sut_draw_counter) == 1:
        lines.append('  << COLLAPSED — every scan got identical SUT values. '
                     'Check the sampler pools and the seqparams pkl.')
    else:
        lines.append('  >> varied — SUT sampling is producing a real distribution.')
    lines.append(
        '  NOTE: on the 2026-07-24 checkpoint these features had 0.0% permutation '
        'importance, so varying them was not expected to move durations. Re-read '
        '05_feature_analysis.py for THIS checkpoint before repeating that claim.'
    )

# ── 4d. PROTOCOL FED TO THE EXAMINATION MODEL (SEQPARAMS FORK) ───────────────
# The model's strongest duration channel, and the one most easily left dead:
# an absent 'protocol' key is not an error, it is a silent fallback to
# RARE_PROTOCOL_ID. A single distinct name here means the sampler collapsed and
# every scan was conditioned on the same protocol.
lines += ['', _HR, ' 4d. PROTOCOLS FED TO THE EXAMINATION MODEL', _HR]
if not protocol_draw_counter:
    lines.append('  !! No protocol draws recorded — no scans were generated.')
else:
    _proto_total = sum(protocol_draw_counter.values())
    _proto_empty = protocol_draw_counter.get('', 0)
    lines.append(
        f'  {_proto_total:,} scans conditioned on {len(protocol_draw_counter):,} '
        f'distinct protocol names (vocabulary holds {len(PROTOCOL_VOCAB):,}).'
    )
    lines.append('  Most frequent draws:')
    for _name, _cnt in protocol_draw_counter.most_common(5):
        lines.append(f'    {(_name or "<no pool — rare bucket>"):<40} '
                     f'{_cnt:>6,}  ({_pct(_cnt, _proto_total)})')
    if len(protocol_draw_counter) == 1:
        lines.append('  << COLLAPSED — every scan got the same protocol. Check the '
                     'sampler pools and that the pkl carries protocol_name.')
    else:
        lines.append('  >> varied — protocol sampling is producing a real distribution.')
    if _proto_empty:
        lines.append(
            f'  {_pct(_proto_empty, _proto_total)} of scans fell through every pool to '
            f'RARE_PROTOCOL_ID with no name. Those rows carry an empty Protocol column '
            f'on purpose — an invented name would have no embedding row.'
        )

# ── 5. FINISH EVENT DISTRIBUTION ─────────────────────────────────────────────
lines += ['', _HR, ' 5. EXAM FINISH EVENT DISTRIBUTION', _HR]
if not df_exam_all.empty and 'FinishEvent' in df_exam_all.columns:
    fe_vc = df_exam_all['FinishEvent'].value_counts()
    total_fe = fe_vc.sum()
    for fe, cnt in fe_vc.items():
        lines.append(f'  {str(fe):<25} {cnt:>6,}  ({_pct(cnt, total_fe)})')
    stopped_pct = df_exam_all['FinishEvent'].eq('Stopped by User').mean()
    lines += [
        '',
        f'  Stopped-by-user rate: {stopped_pct:.1%}  '
        + ('>> high abort rate — examination model may be ending sequences early'
           if stopped_pct > 0.2 else 'OK'),
    ]

# ── 6. EXCHANGE EVENT TYPE DISTRIBUTION ───────────────────────────────────────
# Exchange CSVs use 'token_name' column (renamed from 'sourceID' in step 01)
lines += ['', _HR, ' 6. EXCHANGE EVENT TYPE DISTRIBUTION', _HR]
_ex_tok = 'token_name' if (not df_ex_all.empty and 'token_name' in df_ex_all.columns) else (
    'sourceID' if (not df_ex_all.empty and 'sourceID' in df_ex_all.columns) else None
)
if _ex_tok is not None:
    ev_vc  = df_ex_all[_ex_tok].value_counts()
    total_ev = ev_vc.sum()
    lines.append(f'  {"Event":<20} {"Count":>8} {"Share":>7}')
    for ev, cnt in ev_vc.items():
        lines.append(f'  {str(ev):<20} {cnt:>8,} {_pct(cnt,total_ev):>7}')

# ── 7. EXCHANGE TIME-BETWEEN-EVENTS ──────────────────────────────────────────
lines += ['', _HR, ' 7. EXCHANGE INTER-EVENT GAPS (seconds)', _HR]
if not df_ex_all.empty and 'timediff' in df_ex_all.columns:
    td = df_ex_all['timediff'].dropna().values
    td = td[(td >= 0) & (td < 86400)]
    lines += [
        f'  mean: {td.mean():.1f}  std: {td.std():.1f}  '
        f'p25: {np.percentile(td,25):.1f}  p50: {np.percentile(td,50):.1f}  '
        f'p75: {np.percentile(td,75):.1f}  p99: {np.percentile(td,99):.1f}',
        f'  Gaps >1 hour : {int((td>3600).sum()):,}  ({_pct(int((td>3600).sum()),len(td))})',
        f'  Zero gaps    : {int((td==0).sum()):,}  ({_pct(int((td==0).sum()),len(td))})',
    ]

# ── 8. STEP COUNT (scans per patient) ────────────────────────────────────────
lines += ['', _HR, ' 8. SCANS PER PATIENT VISIT (StepCount)', _HR]
if not df_exam_all.empty and 'StepCount' in df_exam_all.columns:
    # Max StepCount per patient = number of scans that patient had
    sc = df_exam_all.groupby(_VISIT_KEYS)['StepCount'].max()
    vals = sc.values
    lines += [
        f'  mean: {vals.mean():.1f}  std: {vals.std():.1f}  '
        f'min: {vals.min()}  median: {int(np.median(vals))}  max: {vals.max()}',
        f'  Patients with only 1 scan : {int((vals==1).sum()):,}  ({_pct(int((vals==1).sum()),len(vals))})',
        f'  Patients with >10 scans   : {int((vals>10).sum()):,}  ({_pct(int((vals>10).sum()),len(vals))})',
    ]
    if vals.mean() > 8:
        lines.append('  >> high mean step count — examination model may not be terminating cleanly')
    elif vals.mean() < 2:
        lines.append('  << low mean step count — examination model may be terminating too early')

# ── 9. PAUSE TIMES ────────────────────────────────────────────────────────────
lines += ['', _HR, ' 9. INTER-EXAMINATION PAUSE TIMES (seconds)', _HR]
if not df_exam_all.empty and 'pauseTime' in df_exam_all.columns:
    pt = df_exam_all['pauseTime'].dropna().values
    pt = pt[pt > 0]
    if len(pt):
        lines += [
            f'  mean: {pt.mean():.0f}s  std: {pt.std():.0f}s  '
            f'p25: {np.percentile(pt,25):.0f}s  p50: {np.percentile(pt,50):.0f}s  '
            f'p90: {np.percentile(pt,90):.0f}s',
        ]

# ── 10. DEMOGRAPHIC DISTRIBUTIONS ────────────────────────────────────────────
lines += ['', _HR, ' 10. PATIENT DEMOGRAPHICS', _HR]
if not df_exam_all.empty:
    for col, label, lo, hi in [
        ('Age',    'Age (years)',  1,  100),
        ('Weight', 'Weight (kg)', 20,  200),
        ('Height', 'Height (m)',  0.5, 2.5),
    ]:
        if col not in df_exam_all.columns: continue
        v = df_exam_all[col].dropna().values.astype(float)
        v = v[(v >= lo) & (v <= hi)]
        if len(v) == 0: continue
        lines.append(
            f'  {label:<18} mean: {v.mean():.1f}  std: {v.std():.1f}  '
            f'[{v.min():.1f} – {v.max():.1f}]'
        )
    if 'Direction' in df_exam_all.columns:
        hf_rate = df_exam_all['Direction'].eq('Head First').mean()
        lines.append(f'  Head First rate   : {hf_rate:.1%}')

# ── 11. WEEKDAY PATTERN ───────────────────────────────────────────────────────
lines += ['', _HR, ' 11. WEEKDAY PATIENT LOAD PATTERN', _HR]
if not df_exam_all.empty and 'startTime' in df_exam_all.columns:
    df_exam_all['_dow'] = pd.to_datetime(df_exam_all['startTime']).dt.day_name()
    dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    # nunique over (SN, PatientID) pairs — a bare PatientID.nunique() merges
    # the same synthetic id across all 20 scanners. See _VISIT_KEYS above.
    dow_counts = (df_exam_all.drop_duplicates(subset=_VISIT_KEYS + ['_dow'])
                             .groupby('_dow').size())
    for dow in dow_order:
        if dow in dow_counts:
            bar = '█' * int(dow_counts[dow] / max(dow_counts.values) * 30)
            lines.append(f'  {dow:<12} {bar:<30} {dow_counts[dow]:>5}')

# ── 12. MODEL HEALTH FLAGS ───────────────────────────────────────────────────
lines += ['', _HR, ' 12. MODEL HEALTH SUMMARY', _HR]
flags = []
if not df_exam_all.empty:
    dur = df_exam_all['duration'].dropna().values / 60.0
    dur = dur[(dur > 0) & (dur < 4000/60)]
    if len(dur) > 0:
        # Real mean ≈ 1.75 min (105 s). Threshold < 0.5 min catches genuine collapse
        # (e.g. model emitting MSR_104 after every MSR_100). The old 5-min threshold
        # was a false-alarm for healthy output.
        if dur.mean() < 0.5:
            flags.append('[EXAM MODEL]  Mean duration {:.1f} min — sequences terminating too early (expected ~1.75 min)'.format(dur.mean()))
        if dur.mean() > 45:
            flags.append('[EXAM MODEL]  Mean duration {:.1f} min — sequences running too long (expected ~1.75 min)'.format(dur.mean()))

    if 'BodyPart' in df_exam_all.columns:
        probs = df_exam_all['BodyPart'].value_counts(normalize=True).values
        ent   = -np.sum(probs * np.log(probs + 1e-9)) / np.log(len(probs))
        if ent < 0.5:
            flags.append('[ORCH MODEL]  Body region entropy {:.2f} — model collapsing to few regions'.format(ent))
        unknown_pct = df_exam_all['BodyPart'].eq('UNKNOWN').mean()
        if unknown_pct > 0.3:
            flags.append('[ORCH MODEL]  UNKNOWN body region {:.0%} — step-03 body-group normalization may need expanding'.format(unknown_pct))

    if 'StepCount' in df_exam_all.columns:
        sc_mean = df_exam_all.groupby(_VISIT_KEYS)['StepCount'].max().mean()
        if sc_mean > 15:
            flags.append('[EXAM MODEL]  Mean step count {:.1f} — model may not be generating END tokens (Stage 4 issue)'.format(sc_mean))

    if 'pauseTime' in df_exam_all.columns:
        _pt = df_exam_all['pauseTime'].dropna().values
        _pt = _pt[_pt > 0]
        if len(_pt) and _pt.mean() > 1800:
            flags.append('[REPORT BUG]  Mean pause time {:.0f} min — overnight gaps likely included; check _fill_pause_times'.format(_pt.mean() / 60))

if not df_ex_all.empty and 'timediff' in df_ex_all.columns:
    td = df_ex_all['timediff'].dropna().values
    td = td[(td >= 0) & (td < 86400)]
    zero_pct = (td == 0).mean()
    if zero_pct > 0.3:
        flags.append('[EXCHANGE MODEL]  {:.0%} of gaps are zero — duration prediction may be collapsed'.format(zero_pct))

if flags:
    for f in flags:
        lines.append(f'  ⚠  {f}')
else:
    lines.append('  No critical flags — data looks plausible.')

lines += ['', '=' * _W, 'END OF REPORT'.center(_W), '=' * _W]

report_text = '\n'.join(lines)
print(report_text)
