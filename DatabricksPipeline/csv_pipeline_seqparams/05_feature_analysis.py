# Databricks notebook source
# Databricks notebook — Feature importance + correlation matrix (in Databricks)
#
# Two independent sections, per the explicit "do a feature importance and
# corr matrix all within databricks" ask:
#
#   Section A (correlation matrix) needs only the preprocessed pkl — run it
#   right after 03_build_preprocessed_pkl.py, before training.
#   Section B (permutation importance) needs the trained checkpoint from
#   04_train_models.py.
#
# Correlation-matrix pattern follows the only precedent in this repo,
# _archive/PXChange_Refactored_v1/analyze_data.py (df[cols].corr() + seaborn
# heatmap), from an old unrelated tabular model. No feature-importance
# tooling (SHAP, permutation_importance, feature_importances_) exists
# anywhere in this repo — permutation importance is implemented here because
# it is model-agnostic (works directly against estimate_durations(), no
# gradient-attribution machinery needed) and answers the actual question
# ("did adding this feature help?") more directly than a correlation number.

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

import os
import math
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

os.makedirs(ANALYSIS_DIR, exist_ok=True)

# COMMAND ----------

# =============================================================================
# SECTION A — Correlation matrix (pkl only, no trained model needed)
# =============================================================================

print("\n" + "="*60)
print("SECTION A: Correlation matrix")
print("="*60)

with open(PKL_OUTPUT, 'rb') as f:
    _data = pickle.load(f)
examination_sequences = _data['examination']
print(f"Examination sequences: {len(examination_sequences):,}")

_rows = []
for s in examination_sequences:
    cond = s.get('conditioning', {})
    row = {
        'sequence_type': s.get('sequence_type', 0),
        'body_region': s.get('body_region', 10),
        'serial_idx': s.get('serial_idx', 0),
        'trigger_mode': s.get('trigger_mode', 0),
        'total_duration': s.get('total_duration', 0.0),
        'Age': cond.get('Age', 0.0),
        'Weight': cond.get('Weight', 0.0),
        'Height': cond.get('Height', 0.0),
        'PTAB': cond.get('PTAB', 0.0),
    }
    for name in EXAMINATION_SEQPARAM_FEATURES:
        row[name] = cond.get(name, 0.0)
    _rows.append(row)

df = pd.DataFrame(_rows)

numerical_features = [
    'sequence_type', 'body_region', 'serial_idx', 'trigger_mode',
    'Age', 'Weight', 'Height', 'PTAB', 'total_duration',
] + list(EXAMINATION_SEQPARAM_FEATURES)

# --- zero-variance columns first ---------------------------------------------
# A constant column has an undefined correlation, and pandas fills the whole
# row/column with NaN. Printed unfiltered those NaN bands read as "no signal"
# when the truth is "this field never varies in the pkl" — e.g. CT2 (0/51,321
# non-zero) and trigger_mode (one class for every row). Naming them is the
# finding; leaving them in the matrix only hides it.
_const = [c for c in numerical_features if df[c].nunique(dropna=False) <= 1]
if _const:
    print(f"\n  {len(_const)} constant column(s) dropped (no variance, correlation "
          f"undefined): {', '.join(_const[:12])}"
          + (f" … (+{len(_const)-12} more)" if len(_const) > 12 else ""))
_varying = [c for c in numerical_features if c not in _const]

print("\nCorrelation matrix of numerical conditioning features vs. total_duration:")
corr_matrix = df[_varying].corr()
print(corr_matrix.round(2))

_corr_csv = f"{ANALYSIS_DIR}/correlation_matrix.csv"
corr_matrix.to_csv(_corr_csv)
print(f"Saved → {_corr_csv}")

# =============================================================================
# READABLE VIEWS OF THE MATRIX
#
# The full matrix is ~350x350 = 122,500 cells. The previous version rendered it
# as ONE seaborn heatmap with annot=True inside a 10x8in figure: ~245,000
# annotation characters and 350 tick labels in 720x576 px. Nothing in it can be
# read, which is why it never produced a conclusion.
#
# The matrix stays as the CSV above — that is the artefact you grep. What goes
# on screen are the two views that answer the question the matrix was drawn for,
# and they are different questions with different natural forms:
#
#   Panel 1  "which features relate to duration"  -> a RANKED 1-D comparison.
#            Magnitude + polarity, so: diverging. Bars run left/right from zero,
#            hue carries the sign. 350 rows do not fit, so it shows the top N by
#            |r| and says so.
#   Panel 2  "are those few redundant with each other" -> a genuine 2-D matrix,
#            but only over the N features from panel 1, where an annotated
#            heatmap is legible.
#
# Pearson AND Spearman, because most SUT columns are zero-inflated and ordinal
# (presence flags are 0/1; DIFF is non-zero on 3,167 of 51,321 rows). Pearson on
# such a column mostly measures the presence pattern, not the value; Spearman is
# the more honest read and disagreeing with Pearson is itself informative.
# Spearman is computed against total_duration ONLY (350 pairs, cheap) rather
# than as a full matrix (122,500 rank correlations, minutes of CPU).
# =============================================================================

TOP_N_CORR = int(os.environ.get('TOP_N_CORR', 20))

_feat_cols = [c for c in _varying if c != 'total_duration']
_pearson  = corr_matrix['total_duration'].drop('total_duration')
# Rank once, then correlate the ranks against duration's ranks with corrwith:
# that is exactly Spearman, and it is 350 pairwise correlations rather than the
# 122,500 that df.corr(method='spearman') would compute to use one column of.
_ranks = df[_feat_cols + ['total_duration']].rank()
_spearman = _ranks.corrwith(_ranks['total_duration']).drop('total_duration')

_rank = pd.DataFrame({
    'feature':    _feat_cols,
    'pearson_r':  _pearson.reindex(_feat_cols).values,
    'spearman_r': _spearman.reindex(_feat_cols).values,
    'nonzero_pct': [100.0 * (df[c] != 0).mean() for c in _feat_cols],
    'std':        [float(df[c].std()) for c in _feat_cols],
})
_rank['abs_pearson'] = _rank['pearson_r'].abs()
_rank = _rank.sort_values('abs_pearson', ascending=False).reset_index(drop=True)

_rank_csv = f"{ANALYSIS_DIR}/duration_correlations.csv"
_rank.drop(columns='abs_pearson').to_csv(_rank_csv, index=False)
print(f"Saved → {_rank_csv}  (all {len(_rank)} features ranked by |r| vs total_duration)")

print(f"\n  Top {TOP_N_CORR} features by |Pearson r| with total_duration:")
print(f"    {'feature':<32} {'pearson':>8} {'spearman':>9} {'non-zero':>9}")
for _, _r in _rank.head(TOP_N_CORR).iterrows():
    print(f"    {_r['feature']:<32} {_r['pearson_r']:>8.3f} {_r['spearman_r']:>9.3f} "
          f"{_r['nonzero_pct']:>8.1f}%")

# --- palette -----------------------------------------------------------------
# Diverging pair: one warm, one cool pole with a NEUTRAL midpoint, which is what
# a signed correlation needs (the midpoint must read as "nothing"). Validated
# for CVD separation before use: worst adjacent pair dE 24.7 protan / 33.6
# normal-vision, both well clear of the >=8 / >=15 floors.
_POS, _NEG, _MUTED, _INK = '#eb6834', '#2a78d6', '#8d8d86', '#0b0b0b'
_DIVERGING = mcolors.LinearSegmentedColormap.from_list(
    'duration_div', [_NEG, '#e9e9e4', _POS], N=256,
)

_top = _rank.head(TOP_N_CORR).iloc[::-1]        # reversed: strongest at the top
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(17, max(6.5, 0.42 * TOP_N_CORR)),
    gridspec_kw={'width_ratios': [1.0, 1.25]},
)

# ── Panel 1: ranked correlation with duration ────────────────────────────────
_bar_colors = [_POS if v >= 0 else _NEG for v in _top['pearson_r']]
ax1.barh(_top['feature'], _top['pearson_r'], color=_bar_colors, height=0.62)
ax1.axvline(0, color=_INK, lw=0.8)
# Symmetric around zero on purpose: this is a signed scale, and a
# one-sided axis would make a -0.03 bar look like a different KIND of
# result from a +0.03 one rather than the mirror of it.
_lim = max(0.05, float(_top['pearson_r'].abs().max()) * 1.15)
ax1.set_xlim(-_lim, _lim)
# Direct value labels — the bars are short, so the number is the point.
for _y, _v in enumerate(_top['pearson_r']):
    _off = 0.012 * _lim if _v >= 0 else -0.012 * _lim
    ax1.text(_v + _off, _y, f'{_v:+.3f}', va='center',
             ha='left' if _v >= 0 else 'right', fontsize=8, color=_INK)
ax1.set_xlabel('Pearson r with total_duration', fontsize=9)
ax1.set_title(f'Top {TOP_N_CORR} of {len(_rank)} features by |r| with duration',
              fontsize=10, loc='left')
ax1.grid(axis='x', lw=0.4, color='#dcdcd6')
ax1.set_axisbelow(True)
for _s in ('top', 'right', 'left'):
    ax1.spines[_s].set_visible(False)
ax1.tick_params(axis='y', length=0, labelsize=8)
ax1.tick_params(axis='x', labelsize=8)

# ── Panel 2: correlation AMONG those features (legible because N is small) ───
_sub_names = list(_rank.head(TOP_N_CORR)['feature']) + ['total_duration']
_sub = corr_matrix.loc[_sub_names, _sub_names]
_im = ax2.imshow(_sub.values, cmap=_DIVERGING, vmin=-1, vmax=1)
ax2.set_xticks(range(len(_sub_names)))
ax2.set_yticks(range(len(_sub_names)))
ax2.set_xticklabels(_sub_names, rotation=90, fontsize=7)
ax2.set_yticklabels(_sub_names, fontsize=7)
for _i in range(len(_sub_names)):
    for _j in range(len(_sub_names)):
        _v = _sub.values[_i, _j]
        if not np.isfinite(_v):
            continue
        ax2.text(_j, _i, f'{_v:.2f}', ha='center', va='center', fontsize=6,
                 color='#ffffff' if abs(_v) > 0.55 else _INK)
ax2.set_title('Correlation among those features (collinearity check)',
              fontsize=10, loc='left')
for _s in ('top', 'right', 'left', 'bottom'):
    ax2.spines[_s].set_visible(False)
ax2.tick_params(length=0)
_cb = fig.colorbar(_im, ax=ax2, fraction=0.030, pad=0.02)
_cb.set_label('Pearson r', fontsize=8)
_cb.ax.tick_params(labelsize=7)
_cb.outline.set_visible(False)

fig.suptitle('Examination conditioning features vs. scan duration '
             f'({len(examination_sequences):,} sequences)', fontsize=12, x=0.01, ha='left')
fig.tight_layout(rect=[0, 0, 1, 0.96])
_corr_path = f"{ANALYSIS_DIR}/correlation_matrix.png"
fig.savefig(_corr_path, dpi=140, facecolor='#fcfcfb')
plt.close(fig)
print(f"Saved → {_corr_path}")

# Where Pearson and Spearman disagree most — a zero-inflated column whose
# Pearson r is carried by "is it present at all" rather than by its value.
_rank['rank_gap'] = (_rank['spearman_r'].abs() - _rank['pearson_r'].abs())
_gap = _rank.reindex(_rank['rank_gap'].abs().sort_values(ascending=False).index).head(8)
print("\n  Largest Pearson/Spearman disagreements (monotone but non-linear, or "
      "presence-driven):")
for _, _r in _gap.iterrows():
    print(f"    {_r['feature']:<32} pearson={_r['pearson_r']:>7.3f}  "
          f"spearman={_r['spearman_r']:>7.3f}  non-zero={_r['nonzero_pct']:>5.1f}%")

# COMMAND ----------

# Categorical breakdown (Pearson correlation on integer category CODES is
# a blunt instrument for sequence_type/trigger_mode/body_region — this table
# is the more honest view for those: mean/std/count of total_duration per
# category, same shape as the existing per-scan-type duration probe).

print("\nPer-category total_duration (mean / std / n):")
for cat_col, id_to_name in [
    ('sequence_type', ID_TO_SEQUENCE_TYPE),
    ('trigger_mode', {v: k for k, v in TRIGGER_MODE_VOCAB.items()}),
    ('body_region', {i: r for i, r in enumerate(BODY_REGIONS)}),
]:
    print(f"\n  -- {cat_col} --")
    grouped = df.groupby(cat_col)['total_duration'].agg(['mean', 'std', 'count'])
    for cat_id, r in grouped.iterrows():
        name = id_to_name.get(cat_id, str(cat_id))
        print(f"    {name:<18} mean={r['mean']:>7.1f}s  std={r['std']:>6.1f}  n={int(r['count']):>6,}")

_cat_csv = f"{ANALYSIS_DIR}/categorical_duration_breakdown.csv"
df.groupby(['sequence_type', 'trigger_mode', 'body_region'])['total_duration'].agg(
    ['mean', 'std', 'count']
).to_csv(_cat_csv)
print(f"\nSaved → {_cat_csv}")

# COMMAND ----------

# =============================================================================
# SECTION B — Permutation importance (needs the trained checkpoint)
# =============================================================================

print("\n" + "="*60)
print("SECTION B: Permutation importance")
print("="*60)

import json
import torch
import sys

# THIS NOTEBOOK BOOTSTRAPS ITS OWN SOURCE — see 06_compare_models.py for the
# full story. It used to import from /tmp/alternating_pipeline_src, a directory
# only 04_train_models.py populates, and therefore could only run on the machine
# that had just trained. bootstrap_pipeline_source() comes from `%run ./config`
# above and is idempotent.
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
    "AlternatingPipeline.training.utils",
    "AlternatingPipeline.data.protocol_vocab",
])

from AlternatingPipeline.config import (
    EXAMINATION_MODEL_CONFIG, EXAMINATION_TRAINING_CONFIG, START_TOKEN_ID,
)
from AlternatingPipeline.models.examination_model import create_examination_model
from AlternatingPipeline.models.checkpoint_compat import load_checkpoint_lenient
from AlternatingPipeline.training.utils import temporal_split, build_conditioning_tensor
from AlternatingPipeline.data.protocol_vocab import protocol_id

CHECKPOINT_PATH = f"{MODELS_DIR}/examination/examination_model_best.pt"
if not os.path.exists(CHECKPOINT_PATH):
    print(f"No checkpoint at {CHECKPOINT_PATH} yet — run 04_train_models.py first. "
          f"Skipping Section B.")
else:
    # --- the checkpoint's manifest is the authority for its architecture ----
    # Not config.py: on 2026-08-21 the rare-field floor moved between training
    # and analysis and every notebook here tried to load a 277-dim checkpoint
    # into a 351-dim model. See 06_compare_models.py for the full account.
    _SPEC = load_trained_model_spec(f"{MODELS_DIR}/MODEL_MANIFEST.json")
    if _SPEC is None:
        print(f"!! No usable MODEL_MANIFEST.json at {MODELS_DIR} — falling back to "
              f"config.py's feature list. Expect a shape mismatch if the checkpoint "
              f"was trained under a different feature selection.")
        EXTRA_FEATURES = list(EXAMINATION_SEQPARAM_FEATURES)
    else:
        EXTRA_FEATURES = list(_SPEC.extra_conditioning_features)
        print(f"Conditioning pinned to {_SPEC.path}: {len(EXTRA_FEATURES)} extra "
              f"features, base_conditioning_dim={_SPEC.base_conditioning_dim}, "
              f"num_protocols={_SPEC.num_protocols}")

    # --- the vocabulary the checkpoint's protocol rows are indexed by -------
    # Mandatory, not best-effort: without it every prediction below silently
    # uses RARE_PROTOCOL_ID, and a permutation-importance table computed that
    # way ranks every feature against a model whose strongest input is dead.
    _VOCAB_PATH = f"{MODELS_DIR}/examination/protocol_vocab.json"
    try:
        with open(_VOCAB_PATH) as _vf:
            PROTOCOL_VOCAB = json.load(_vf)["vocab"]
        print(f"Protocol vocabulary: {len(PROTOCOL_VOCAB):,} protocols from {_VOCAB_PATH}")
    except (OSError, ValueError, KeyError) as _err:
        raise RuntimeError(
            f"Cannot read the protocol vocabulary at {_VOCAB_PATH} ({_err}). Re-run "
            f"04_train_models.py's vocabulary cell against this checkpoint."
        )

    # build_seqparams_model_config comes from this file's own %run ./config —
    # single source of truth shared with 04_train_models.py / 06_compare_models.py
    # / 07_generate_synthetic_data.py. `spec=` makes it read the manifest.
    EXAMINATION_MODEL_CONFIG_SEQPARAMS = build_seqparams_model_config(
        EXAMINATION_MODEL_CONFIG, spec=_SPEC,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_examination_model(EXAMINATION_MODEL_CONFIG_SEQPARAMS)
    # Lenient: tolerates params added since the checkpoint was trained (e.g.
    # duration_cond_bias, which is zero-initialised and therefore a no-op),
    # while still refusing a checkpoint from a different architecture.
    load_checkpoint_lenient(
        model,
        torch.load(CHECKPOINT_PATH, map_location=device),
        label=f"examination checkpoint ({CHECKPOINT_PATH})",
    )
    model = model.to(device)
    model.eval()

    _, val_sequences = temporal_split(examination_sequences, val_days=2)
    print(f"Held-out validation sequences: {len(val_sequences)}")

    duration_scale = EXAMINATION_TRAINING_CONFIG['duration_scale']

    def _predicted_seconds(seqs, shuffle_field=None, rng=None):
        """Run estimate_durations() over seqs, optionally shuffling one
        conditioning field across the batch first. Returns (predicted_secs,
        target_secs) arrays, one value per sequence (finish-token span total,
        same convention as the existing post-train probe)."""
        conds, regions, seq_types, serials, triggers, tokens_list, targets = [], [], [], [], [], [], []
        protocols = []
        for s in seqs:
            toks = s['sequence'][:model.max_seq_len - 1]
            if not toks:
                continue
            conds.append(build_conditioning_tensor(
                s['conditioning'], extra_feature_names=EXTRA_FEATURES,
                denylist=SUT_ALL_DENYLISTS,
            ))
            regions.append(s['body_region'])
            seq_types.append(int(s.get('sequence_type', 0)))
            serials.append(int(s.get('serial_idx', 0)))
            triggers.append(int(s.get('trigger_mode', 0)))
            protocols.append(protocol_id(s.get('protocol_name'), PROTOCOL_VOCAB))
            tokens_list.append(toks)
            targets.append(sum(max(0.0, d) for d in s.get('durations', [])))

        if rng is None:
            rng = np.random.default_rng(0)

        conds_t = torch.stack(conds)
        regions_t = torch.tensor(regions, dtype=torch.long)
        seq_types_t = torch.tensor(seq_types, dtype=torch.long)
        serials_t = torch.tensor(serials, dtype=torch.long)
        triggers_t = torch.tensor(triggers, dtype=torch.long)
        protocols_t = torch.tensor(protocols, dtype=torch.long)

        if shuffle_field is not None:
            if shuffle_field in EXTRA_FEATURES:
                # 10 = the fixed base conditioning block (Age, Weight, Height,
                # PTAB, Direction, hour/dow sin/cos, is_morning) that
                # build_conditioning_tensor always writes first; the extras
                # follow it in EXTRA_FEATURES order.
                col = 10 + EXTRA_FEATURES.index(shuffle_field)
                perm = torch.tensor(rng.permutation(len(conds)))
                conds_t[:, col] = conds_t[perm, col]
            elif shuffle_field == 'sequence_type':
                perm = rng.permutation(len(conds))
                seq_types_t = seq_types_t[perm]
            elif shuffle_field == 'serial_idx':
                perm = rng.permutation(len(conds))
                serials_t = serials_t[perm]
            elif shuffle_field == 'trigger_mode':
                perm = rng.permutation(len(conds))
                triggers_t = triggers_t[perm]
            elif shuffle_field == 'body_region':
                perm = rng.permutation(len(conds))
                regions_t = regions_t[perm]
            elif shuffle_field == 'protocol':
                perm = rng.permutation(len(conds))
                protocols_t = protocols_t[perm]

        preds = []
        with torch.no_grad():
            for i, toks in enumerate(tokens_list):
                inp = torch.tensor([[START_TOKEN_ID] + toks],
                                    dtype=torch.long, device=device)
                info = {
                    'body_region': regions_t[i:i+1].to(device),
                    'sequence_type': seq_types_t[i:i+1].to(device),
                    'serial_idx': serials_t[i:i+1].to(device),
                    'trigger_mode': triggers_t[i:i+1].to(device),
                    'protocol': protocols_t[i:i+1].to(device),
                }
                mu, _ = model.estimate_durations(inp, conds_t[i:i+1].to(device), info)
                m = mu[0, len(toks) - 1].item()
                pred_sec = (math.expm1(m) if model.duration_mode == 'log' else m) * duration_scale
                preds.append(pred_sec)
        return np.array(preds), np.array(targets)

    # --- how many sequences this sweep can actually afford -----------------
    # _predicted_seconds runs ONE forward pass per sequence (batch size 1,
    # because held-out sequences have different lengths), and the sweep runs it
    # once per feature per repeat. That was fine when the feature list was
    # TR + num_slices: 2 features x 5 repeats x ~4,000 sequences.
    #
    # PARAM_SET='all' made the list 133 features, and the same sweep is
    # 138 x 5 x 3,956 = 2.7M forward passes — hours on a T4, days on CPU. The
    # notebook did not get slower, the feature list got 60x longer, and nothing
    # here was ever re-sized for it.
    #
    # So the sweep runs on a deterministic SUBSET, and prints what it cost so
    # the trade is visible rather than assumed. Set PERM_N=0 for the full
    # held-out set when there is time for it. The proper fix is to batch the
    # forward passes with a PAD key-padding mask (estimate_durations already
    # supports it) — that is a change to how predictions are computed, and it
    # does not belong in the same pass as a load-bearing correctness fix.
    PERM_N = int(os.environ.get('PERM_N', 500))
    PERM_REPEATS = int(os.environ.get('PERM_REPEATS', 5))

    perm_sequences = val_sequences
    if PERM_N and len(val_sequences) > PERM_N:
        # Evenly spaced, not the first N: val_sequences is in TEMPORAL order,
        # so a head slice would be one day of one subset of scanners.
        _stride = len(val_sequences) / PERM_N
        perm_sequences = [val_sequences[int(i * _stride)] for i in range(PERM_N)]
        print(f"Permutation sweep on {len(perm_sequences):,} of "
              f"{len(val_sequences):,} held-out sequences (PERM_N={PERM_N}; "
              f"set PERM_N=0 for all of them).")

    baseline_preds, targets = _predicted_seconds(perm_sequences)
    baseline_mae = np.mean(np.abs(baseline_preds - targets))
    print(f"Baseline MAE (no shuffle): {baseline_mae:.1f}s over {len(targets)} sequences")

    # 'protocol' leads because it is the channel this checkpoint was retrained
    # for, and because its rank is the cheapest check that the ids are actually
    # arriving: duration_protocol_bias is a trained per-position embedding, so a
    # near-zero degradation here means the wiring is broken, not that the
    # protocol is uninformative.
    feature_names = (
        ['protocol', 'sequence_type', 'serial_idx', 'trigger_mode', 'body_region']
        + list(EXTRA_FEATURES)
    )
    n_repeats = PERM_REPEATS
    print(f"Sweeping {len(feature_names):,} features x {n_repeats} repeats x "
          f"{len(perm_sequences):,} sequences = "
          f"{len(feature_names) * n_repeats * len(perm_sequences):,} forward passes.")
    results = []
    for name in feature_names:
        degradations = []
        for rep in range(n_repeats):
            rng = np.random.default_rng(rep)
            preds, _ = _predicted_seconds(perm_sequences, shuffle_field=name, rng=rng)
            mae = np.mean(np.abs(preds - targets))
            degradations.append(mae - baseline_mae)
        mean_degradation = float(np.mean(degradations))
        results.append({
            'feature': name,
            'baseline_mae_s': round(baseline_mae, 1),
            'shuffled_mae_s': round(baseline_mae + mean_degradation, 1),
            'degradation_s': round(mean_degradation, 1),
            'pct_degradation': round(100 * mean_degradation / max(1e-6, baseline_mae), 1),
            # Spread ACROSS the repeats. Without it there is no way to tell a
            # feature the model genuinely ignores (0.0s) from one whose effect is
            # simply smaller than the shuffle-to-shuffle noise, and the printed
            # table has ~250 rows of "0.0s" that could be either. This is what
            # draws the noise band in the plot below.
            'degradation_std_s': round(float(np.std(degradations)), 3),
        })

    results.sort(key=lambda r: -r['degradation_s'])

    # A feature counts as LIVE only if shuffling it costs more than the
    # shuffle-to-shuffle noise. The noise floor is the median per-feature std
    # across repeats, doubled — anything inside +/- that band is indistinguishable
    # from re-seeding the RNG, so calling it "0.0s important" overstates what was
    # measured. Printing 272 rows of it, as this notebook used to, buries the two
    # rows that matter under 270 that say nothing.
    _NOISE = 2.0 * float(np.median([r['degradation_std_s'] for r in results]))
    _live = [r for r in results if abs(r['degradation_s']) > max(_NOISE, 0.05)]
    _dead = [r for r in results if r not in _live]

    TOP_N_PERM = int(os.environ.get('TOP_N_PERM', 15))
    _FLOOR = max(_NOISE, 0.05)
    print(f"\nPermutation importance (higher degradation = more important).")
    print(f"  Noise floor (2x median std across {n_repeats} repeats): "
          f"+/-{_NOISE:.3f}s — anything inside it is not measurably used.")

    print(f"\n  LIVE — the duration head measurably responds to {len(_live)} of "
          f"{len(results)} features:")
    if not _live:
        print("    (none — the head responds to nothing that was shuffled)")
    for r in _live[:TOP_N_PERM]:
        print(f"    {r['feature']:<32} degradation={r['degradation_s']:>7.1f}s "
              f"({r['pct_degradation']:>6.1f}%)  +/-{r['degradation_std_s']:.3f}"
              f"  shuffled_mae={r['shuffled_mae_s']:.1f}s")
    if len(_live) > TOP_N_PERM:
        print(f"    … {len(_live) - TOP_N_PERM} further live features — see the CSV.")

    print(f"\n  DEAD — {len(_dead)} of {len(results)} features sit inside the noise "
          f"floor: shuffling any of them moves the prediction by less than "
          f"{_FLOOR:.3f}s.")
    if _dead:
        _best_dead = _dead[0]
        print(f"    Strongest of them is {_best_dead['feature']} at "
              f"{_best_dead['degradation_s']:.3f}s — i.e. even the best 'dead' "
              f"feature is indistinguishable from re-seeding the shuffle RNG.")
        _names = [r['feature'] for r in _dead]
        print(f"    {', '.join(_names[:14])}"
              + (f" … (+{len(_names)-14} more, see CSV)" if len(_names) > 14 else ""))

    _perm_df = pd.DataFrame(results)
    _perm_csv = f"{ANALYSIS_DIR}/permutation_importance.csv"
    _perm_df.to_csv(_perm_csv, index=False)
    print(f"\nSaved → {_perm_csv}")

    # =========================================================================
    # The old chart was plt.barh() over all 272 features in a 9x5in figure:
    # 272 bars and 272 y-tick labels in 648x360 px, ~1.3 px per row. Two bars
    # had length; 270 were a black line at x=0 with unreadable labels.
    #
    # The finding here is not "these are the ranks" — it is "TWO features carry
    # everything and the other 270 are inside the noise". That is two different
    # statements, so it is two panels:
    #
    #   Panel 1  the ranked top N, where a bar chart works, with the noise band
    #            drawn so a short bar is visibly short RELATIVE TO NOISE rather
    #            than just small.
    #   Panel 2  every feature as one dot on a symlog axis, so the pile-up at
    #            zero is the visible fact instead of an invisible one. Symlog,
    #            not log: degradations are signed and cross zero.
    #
    # Emphasis over hue-cycling: the live features take the accent, everything
    # else is muted grey. Colour carries "does this matter", which is the
    # question, and never encodes rank.
    # =========================================================================
    _ACC, _MUTE, _INK, _BAND = '#eb6834', '#8d8d86', '#0b0b0b', '#dfe8f5'

    # Rows: every LIVE feature, plus the strongest handful of the dead ones for
    # contrast. Filling the panel with 13 identical "not used" rows (which a flat
    # head-15 does when only 2 features are live) spends the reader's attention
    # on the least informative rows on the chart. Six dead rows make the point
    # that the NEXT-best feature is also at zero; the rest is panel 2's job.
    _N_DEAD_SHOWN = 6
    _rows = ([r['feature'] for r in _live[:TOP_N_PERM]]
             + [r['feature'] for r in _dead[:_N_DEAD_SHOWN]])
    _plot_top = _perm_df[_perm_df['feature'].isin(_rows)].copy()
    _plot_top['__o'] = _plot_top['feature'].map({f: i for i, f in enumerate(_rows)})
    _plot_top = _plot_top.sort_values('__o').iloc[::-1]
    _n_rows = len(_plot_top)

    fig, (axA, axB) = plt.subplots(
        2, 1, figsize=(11, max(7.0, 0.34 * _n_rows + 4.6)),
        gridspec_kw={'height_ratios': [max(1.6, _n_rows / 4.5), 1.0]},
    )

    # ── Panel 1: the live features, with the best dead ones underneath ──────
    _is_live = [abs(v) > _FLOOR for v in _plot_top['degradation_s']]
    axA.barh(_plot_top['feature'], _plot_top['degradation_s'],
             color=[_ACC if L else _MUTE for L in _is_live], height=0.62)
    axA.axvspan(-_FLOOR, _FLOOR, color=_BAND, zorder=0)
    axA.axvline(0, color=_INK, lw=0.8)
    _xmax = max(1.0, float(_plot_top['degradation_s'].max()) * 1.30)
    axA.set_xlim(-0.04 * _xmax, _xmax)
    for _y, (_v, _L) in enumerate(zip(_plot_top['degradation_s'], _is_live)):
        axA.text(_v + 0.010 * _xmax, _y,
                 f'{_v:.1f}s' if _L else 'inside noise — not used',
                 va='center', ha='left', fontsize=8,
                 color=_INK if _L else _MUTE)
    axA.set_xlabel('MAE degradation when this feature is shuffled (s)', fontsize=9)
    axA.set_title(
        f'{len(_live)} live feature(s) of {len(_perm_df)}, then the {_N_DEAD_SHOWN} '
        f'strongest of the {len(_dead)} dead ones  ·  baseline MAE '
        f'{baseline_mae:.1f}s  ·  shaded band = noise (+/-{_FLOOR:.2f}s)',
        fontsize=10, loc='left')
    axA.grid(axis='x', lw=0.4, color='#dcdcd6')
    axA.set_axisbelow(True)
    for _s in ('top', 'right', 'left'):
        axA.spines[_s].set_visible(False)
    axA.tick_params(axis='y', length=0, labelsize=8)
    axA.tick_params(axis='x', labelsize=8)

    # ── Panel 2: the whole population, so the pile-up at zero is visible ────
    _vals = _perm_df['degradation_s'].values
    _names_all = _perm_df['feature'].values
    _live_mask = np.abs(_vals) > _FLOOR
    _rng_j = np.random.default_rng(0)
    _jit = _rng_j.uniform(-0.30, 0.30, size=len(_vals))
    # Live points get their OWN evenly spaced y-slots rather than random jitter,
    # so each label sits at its own point's height and two of them can never
    # land on the same line. Labels go to the LEFT of the dot (ha='right'), which
    # is the only side that cannot run off the axis for the largest value.
    _n_live = int(_live_mask.sum())
    if _n_live:
        _slots = np.linspace(0.34, -0.34, _n_live) if _n_live > 1 else np.array([0.0])
        _jit[_live_mask] = _slots
    axB.axvspan(-_FLOOR, _FLOOR, color=_BAND, zorder=0,
                label=f'noise floor (+/-{_FLOOR:.2f}s)')
    axB.scatter(_vals[~_live_mask], _jit[~_live_mask], s=22, color=_MUTE,
                alpha=0.55, linewidths=0, zorder=2,
                label=f'not measurably used ({int((~_live_mask).sum())} features)')
    axB.scatter(_vals[_live_mask], _jit[_live_mask], s=54, color=_ACC,
                linewidths=0.8, edgecolors='#fcfcfb', zorder=3,
                label=f'measurably used ({_n_live} features)')
    for _v, _y, _n in zip(_vals[_live_mask], _jit[_live_mask],
                          _names_all[_live_mask]):
        axB.annotate(f'{_n}  {_v:.1f}s', (_v, _y), textcoords='offset points',
                     xytext=(-10, 0), ha='right', va='center', fontsize=8,
                     color=_INK)
    axB.set_xscale('symlog', linthresh=_FLOOR)
    axB.axvline(0, color=_INK, lw=0.8)
    axB.set_xlim(-_FLOOR * 4, max(2.0, float(np.nanmax(_vals)) * 2.2))
    axB.set_ylim(-0.72, 0.72)
    axB.set_yticks([])
    axB.set_xlabel('MAE degradation when shuffled (s) — symlog, linear inside the '
                   'noise band', fontsize=9)
    axB.set_title(f'All {len(_perm_df)} features on one axis', fontsize=10, loc='left')
    for _s in ('top', 'right', 'left'):
        axB.spines[_s].set_visible(False)
    axB.tick_params(axis='x', labelsize=8)
    axB.grid(axis='x', lw=0.4, color='#dcdcd6')
    axB.set_axisbelow(True)
    # Upper-left: the live points sit at the right-hand end of the axis and the
    # dead cluster is a narrow column at zero, so the top-left quadrant is the
    # only region a legend cannot collide with either.
    axB.legend(loc='upper left', fontsize=8, frameon=False)

    fig.suptitle('Permutation importance — examination duration head',
                 fontsize=12, x=0.01, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    _perm_png = f"{ANALYSIS_DIR}/permutation_importance.png"
    fig.savefig(_perm_png, dpi=140, facecolor='#fcfcfb')
    plt.close(fig)
    print(f"Saved → {_perm_png}")

# COMMAND ----------

# =============================================================================
# NEXT STEP: run 06_compare_models.py for the old-vs-new head-to-head.
# =============================================================================
