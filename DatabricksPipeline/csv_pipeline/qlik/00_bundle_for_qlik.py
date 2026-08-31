# Databricks notebook source
# Databricks notebook — Extract + bundle the four CSV sets the Qlik dashboard needs
#
# WHAT THIS REPLACES. The ad-hoc exchange-extraction cell that lived in a
# scratch notebook, pinned to `SERIAL_NUMBER = "202594"` and a single day
# (2024-04-16). Neither the serial nor the date is in scope for this project:
# 202594 is not one of the configured scanners, and April is not the training
# window. That cell's logic is otherwise identical to
# ../01_exchange_preprocessing.py, which already loops the configured serials
# over the configured window — so the fix is to CALL that notebook, not to
# maintain a second copy of interpatient() / join_events() / ptab() / coils().
#
# WHAT THE QLIK DASHBOARD NEEDS — four directories, all keyed DATA_{serial}.csv:
#
#   real      exchange  /dbfs/FileStore/csv_pipeline/exchange              <- ../01
#   real      exam      /dbfs/FileStore/csv_pipeline/exam                  <- ../02
#   synthetic exchange  /dbfs/FileStore/csv_pipeline_seqparams/synthetic/exchange  <- seqparams/07
#   synthetic exam      /dbfs/FileStore/csv_pipeline_seqparams/synthetic/exam      <- seqparams/07
#
# The real side has no seqparams equivalent and is not supposed to: that fork
# has no 02_exam_preprocessing.py and its config defines neither
# EXCHANGE_SOURCE_TYPES nor EXAM_EXTRA_SOURCE_TYPES (see
# csv_pipeline_seqparams/config.py:180). Only the SYNTHETIC half moves to the
# seqparams pipeline; the real half stays here, which is exactly what makes the
# comparison a like-for-like one.
#
# ORDER OF OPERATIONS, and the trap in it. seqparams/07 iterates
# `customer_schedules.keys()` read from BASE_PKL — which is *csv_pipeline's*
# pkl, not its own (07:101, 07:1239). So the synthetic side covers whatever
# serials csv_pipeline/03 last wrote, regardless of what
# csv_pipeline_seqparams/config.py says. Run in this order or the synthetic
# half silently comes back at the old 10 scanners:
#
#   1. ../01_exchange_preprocessing.py          real exchange CSVs
#   2. ../02_exam_preprocessing.py              real exam CSVs
#   3. ../03_build_preprocessed_pkl.py          csv_pipeline pkl — sets the synthetic serial set
#   4. ../04_train_models.py                    exchange + orchestration checkpoints
#   5. ../../csv_pipeline_seqparams/03_build_preprocessed_pkl.py
#   6. ../../csv_pipeline_seqparams/04_train_models.py
#   7. ../../csv_pipeline_seqparams/07_generate_synthetic_data.py   synthetic CSVs
#   8. this notebook's bundle cell -> one .zip -> unzip into qlik/data/
#   9. `python consolidate.py` locally
#
# Steps 3-7 are only needed if the model side changed. If the checkpoints are
# already current, run 1, 2, 7 and then this.
#
# WHY THIS NOTEBOOK DOES NOT `%run` STEPS 1 AND 2 FOR YOU. Both begin with
# `%pip install openpyxl` (01:11, 02:12). A `%pip` inside a `%run`-ed notebook
# restarts the Python process, which would discard everything this notebook has
# already defined — the inventory frame and SOURCES included — and the failure
# lands later as a NameError in the bundle cell, pointing at the wrong thing.
# So: open 01 and 02 and Run All, then come back here. Nothing in this notebook
# needs their in-memory state; it reads only what they wrote to DBFS.

# COMMAND ----------
%run ../config

# COMMAND ----------

import os
import time
import glob
import zipfile
import datetime as _dt

import pandas as pd

# The four (side, kind) -> DBFS directory pairs, in the layout consolidate.py
# expects to find under qlik/data/. The zip is built with exactly these as its
# top-level paths, so unzipping into qlik/data/ needs no file shuffling.
SEQPARAMS_SYNTH_ROOT = "/dbfs/FileStore/csv_pipeline_seqparams/synthetic"

SOURCES = [
    ("real",      "exchange", EXCHANGE_OUTPUT_DIR),
    ("real",      "exam",     EXAM_OUTPUT_DIR),
    ("synthetic", "exchange", f"{SEQPARAMS_SYNTH_ROOT}/exchange"),
    ("synthetic", "exam",     f"{SEQPARAMS_SYNTH_ROOT}/exam"),
]

BUNDLE_DBFS_DIR = "/dbfs/FileStore/qlik_bundles"
BUNDLE_NAME     = f"qlik_bundle_{_dt.date.today().isoformat()}.zip"

print(f"Serials ({len(SERIAL_NUMBERS)}): {SERIAL_NUMBERS}")
print(f"Window:  {DATE_START} -> {DATE_END}  (TZ offset +{TIMEZONE_OFFSET_HOURS}h)")

# COMMAND ----------
# =============================================================================
# CELL: Inventory — what is on DBFS, per serial, per source
#
# Run this BEFORE bundling. A missing serial here is the difference between
# "the model never generated it" and "the extract never covered it", and the
# Qlik grade sheet cannot tell those apart after the fact.
# =============================================================================

rows = []
for side, kind, base_dir in SOURCES:
    for serial in SERIAL_NUMBERS:
        path = f"{base_dir}/DATA_{serial}.csv"
        exists = os.path.exists(path)
        size_kb = os.path.getsize(path) / 1024 if exists else 0.0
        n_rows = -1
        if exists:
            try:
                with open(path) as fh:
                    n_rows = sum(1 for _ in fh) - 1     # minus header
            except Exception:
                pass
        rows.append({
            "side": side, "kind": kind, "serial": serial,
            "exists": exists, "size_kb": round(size_kb, 1), "rows": n_rows,
            "path": path,
        })

    # Files present on DBFS that are NOT in SERIAL_NUMBERS — usually leftovers
    # from an earlier, narrower run. They would ride along in the zip and show
    # up in Qlik as scanners nobody configured, so name them explicitly.
    on_disk = {os.path.basename(p)[5:-4]
               for p in glob.glob(f"{base_dir}/DATA_*.csv")}
    stray = sorted(on_disk - {str(s) for s in SERIAL_NUMBERS})
    if stray:
        print(f"  ! {side}/{kind}: {len(stray)} unconfigured serial(s) on disk: {stray}")

df_files = pd.DataFrame(rows)

print()
for (side, kind), grp in df_files.groupby(["side", "kind"], sort=False):
    have = grp[grp["exists"]]
    missing = sorted(grp.loc[~grp["exists"], "serial"].tolist())
    print(f"{side:<10} {kind:<9} {len(have):>2}/{len(grp)} serials  "
          f"{have['rows'].clip(lower=0).sum():>9,} rows  "
          f"{have['size_kb'].sum()/1024:>7.1f} MB")
    if missing:
        print(f"{'':21}MISSING: {missing}")

# --- Real vs synthetic set reconciliation ----------------------------------
# The comparison is only like-for-like if both halves cover the SAME scanners.
# They can diverge silently: seqparams/07 generates one scanner per
# customer_schedules key, and step 03 only creates a key for a serial that has
# an exam CSV — so a scanner that produced no real exam rows vanishes from the
# synthetic side too, three notebooks later. The 2026-08-31 run came back with
# 20 synthetic scanners against 21 configured for exactly that reason (155687),
# and the shortfall was only visible by counting files.
_present = {
    side: {r["serial"] for r in rows
           if r["side"] == side and r["exists"] and r["rows"] > 0}
    for side in ("real", "synthetic")
}
_real_only  = sorted(_present["real"] - _present["synthetic"])
_synth_only = sorted(_present["synthetic"] - _present["real"])
_both       = sorted(_present["real"] & _present["synthetic"])

print(f"\ncomparable on both sides: {len(_both)} scanner(s)")
if _real_only:
    print(f"  REAL ONLY      : {_real_only}")
    print("    -> no synthetic counterpart. Check step 02's coverage summary for "
          "these serials; if they produced no exam rows, step 03 dropped them "
          "from customer_schedules and seqparams/07 never generated them.")
if _synth_only:
    print(f"  SYNTHETIC ONLY : {_synth_only}")
    print("    -> the real extract is behind the generator; re-run steps 01/02.")
if not _real_only and not _synth_only:
    print("  both halves cover the same scanners.")

# COMMAND ----------
# =============================================================================
# CELL: Bundle all four directories into one zip
#
# Written to local disk first, then copied to FileStore. zipfile writing
# straight through the /dbfs FUSE mount is unreliable at this size (tens of MB
# across ~84 files), and a truncated archive fails at unzip time rather than
# here, where the cause is still visible.
#
# Only configured serials go in — the stray files named by the inventory cell
# are deliberately left behind.
# =============================================================================

_t0 = time.perf_counter()

local_zip = f"/tmp/{BUNDLE_NAME}"
if os.path.exists(local_zip):
    os.remove(local_zip)

written, skipped = 0, 0
with zipfile.ZipFile(local_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for side, kind, base_dir in SOURCES:
        for serial in SERIAL_NUMBERS:
            src = f"{base_dir}/DATA_{serial}.csv"
            if not os.path.exists(src):
                skipped += 1
                continue
            # arcname mirrors qlik/data/ exactly: real/exchange/DATA_x.csv, ...
            zf.write(src, arcname=f"{side}/{kind}/DATA_{serial}.csv")
            written += 1

    # A manifest so the bundle is self-describing six months from now: which
    # serials, which window, which pipeline produced the synthetic half.
    zf.writestr("MANIFEST.txt", "\n".join([
        f"generated       : {_dt.datetime.now().isoformat(timespec='seconds')}",
        f"serials ({len(SERIAL_NUMBERS)})    : {', '.join(str(s) for s in SERIAL_NUMBERS)}",
        f"real window     : {DATE_START} .. {DATE_END}  (TZ +{TIMEZONE_OFFSET_HOURS}h)",
        "real source     : csv_pipeline/01 + csv_pipeline/02",
        f"synthetic source: csv_pipeline_seqparams/07  ({SEQPARAMS_SYNTH_ROOT})",
        f"files           : {written} written, {skipped} missing",
        "",
        "Unzip into DatabricksPipeline/csv_pipeline/qlik/data/ then run:",
        "    python consolidate.py",
    ]) + "\n")

size_mb = os.path.getsize(local_zip) / (1024 * 1024)
print(f"Zipped {written} file(s), {skipped} missing, {size_mb:.1f} MB "
      f"in {time.perf_counter() - _t0:.1f}s")

dbutils.fs.mkdirs(BUNDLE_DBFS_DIR.replace("/dbfs", "dbfs:"))
dbutils.fs.cp(f"file:{local_zip}",
              f"{BUNDLE_DBFS_DIR.replace('/dbfs', 'dbfs:')}/{BUNDLE_NAME}")

bundle_path = f"{BUNDLE_DBFS_DIR}/{BUNDLE_NAME}"
bundle_url  = bundle_path.replace("/dbfs/FileStore", "/files")
print(f"-> {bundle_path}")

# COMMAND ----------
# =============================================================================
# CELL: Download link
# =============================================================================

_missing_by_group = (
    df_files[~df_files["exists"]]
    .groupby(["side", "kind"])["serial"]
    .apply(lambda s: ", ".join(str(x) for x in sorted(s)))
    .to_dict()
)
_warn = ""
if _missing_by_group:
    _warn = "<ul style='color:#b00;'>" + "".join(
        f"<li><b>{side}/{kind}</b> missing: {serials}</li>"
        for (side, kind), serials in _missing_by_group.items()
    ) + "</ul>"

displayHTML(f"""
<h3>Qlik bundle — {BUNDLE_NAME}</h3>
<p>
  {written} CSV(s) &nbsp;|&nbsp; {size_mb:,.1f} MB &nbsp;|&nbsp;
  {len(SERIAL_NUMBERS)} scanners &nbsp;|&nbsp; real {DATE_START} &rarr; {DATE_END}
</p>
<p style="font-size:1.15em;">
  <a href="{bundle_url}" download><b>&#11015; Download {BUNDLE_NAME}</b></a>
</p>
{_warn}
<pre style="background:#f6f6f6; padding:10px; border-radius:4px;">
cd DatabricksPipeline/csv_pipeline/qlik
find data/real data/synthetic -name 'DATA_*.csv' -delete   # keeps .gitkeep
unzip -o ~/Downloads/{BUNDLE_NAME} -d data/
python consolidate.py
</pre>
<p style="font-size:0.85em; color:#666;">
  Archive layout is <code>real/exchange/</code>, <code>real/exam/</code>,
  <code>synthetic/exchange/</code>, <code>synthetic/exam/</code> — the same
  tree <code>consolidate.py</code> reads, so it unzips straight into
  <code>data/</code>. <code>MANIFEST.txt</code> inside records the serials and
  the window.
</p>
""")
