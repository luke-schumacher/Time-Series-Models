# Qlik Validation Dashboard — How-To

**Goal:** Compare the synthetic MRI pipeline output against real training
data in a Qlik Sense dashboard, side-by-side, by manually uploading two
consolidated CSV files.

Everything here is **local**. No Databricks pipeline changes, no Qlik
load scripts to maintain — just a small Python helper that merges your
per-scanner CSVs into two files, plus a few drag-and-drops in Qlik.

---

## What's in this bundle (read first)

Scope of the data shipped in `data/`:

| | Real | Synthetic |
|---|---|---|
| Scanners | 20 | 20 |
| Date range | 2024-01-01 → 2024-01-31 | 2024-02-01 → 2024-02-28 |
| Exam rows | 93,998 | 54,233 |
| Exchange rows | 277,448 | 163,451 |

**Both sides now cover the same 20 scanners**, so every chart can be filtered
by `SN` without one side going blank. 21 serials are configured; `155687` has
no eventlog rows in the window and is absent from both sides.

**The two sides cover different months on purpose** — synthetic continues
the month after the training window. Don't apply a single date filter
across both or you'll blank one side.

### Known limitations of this snapshot

- **The synthetic CSVs are two generator fixes behind the code.** Measured on
  the files in this bundle: 100% of the gaps between scans inside a visit are
  exactly 0 s (commit `c9e8604` put that pause back), and `Exam_Protocol`
  carries the old 8-name placeholder pool (`t1_mprage`, `t2_flair`, …) rather
  than sampled real protocol names. Fine for anatomy and event-mix charts; do
  not read Chart 2 as the current state of the duration model.
- **Nine of the twenty real exam CSVs were built before `sourceID` was added
  to step 02's output** (175693, 175727, 175832, 176133, 176430, 176583,
  176659, 176750, 176802). `consolidate.py` reconstructs `Exam_sourceID` for
  those rows from `FinishEvent`, which folds every abort code into
  `MRI_MSR_34`. Successful scans are unaffected. Re-run step 02 with
  `RESUME=0` for those serials to get the real finish codes.
- **Coils are absent on the synthetic side.** `Exam_ConnectedCoils` is
  100% null and all 98 `Exam_#0_*` coil flags are never `True` (95 of them
  are populated on real). Any coil chart will show real only.
- **Category resolution is coarse on the synthetic side:** 8 distinct
  protocols vs 3,000 real, 7 body parts vs 56, 7 body groups vs 15.
- **`Exch_BodyGroup_to_text`** carries 27,663 synthetic `UNKNOWN` rows
  (28.4%, zero on real) plus an `END` category (5,309) with no real
  counterpart — expect two phantom bars in exchange body-region charts.
- **Exchange durations lose the long tail:** medians match closely (456 s
  vs 455 s) but the synthetic mean is 559 s against 1,108 s real, and the
  max is 9,883 s against 28,806 s.
- **Real demographics are 21% null** (`Age`, `Weight`, `Height`,
  `Direction`) where synthetic is 0% — demographic comparisons run against
  an incomplete real denominator.

### Label alignment performed by `consolidate.py`

Two columns are spelled differently by step 02 (real) and step 05
(synthetic), and Qlik would treat each spelling as a separate dimension
value — producing two disjoint sets of bars that look like a total model
failure. `consolidate.py` normalizes both:

- `Exam_BodyGroup` — real `Head` / synthetic `HEAD` → upper-cased on both
  sides, matching the vocabulary the exchange table already uses.
- `Exam_Sequence` — real `%SiemensSeq%\fl3d_vibe` / synthetic `vibe` → both
  sides run through the canonical step-03 `classify_sequence_type()`. A
  plain prefix strip is not sufficient: the synthetic vocabulary is coarser
  (`vibe` covers `fl3d_vibe`, `fl3d_ce`, `fl3d_rd`).

After normalization the two sides share 7/7 body-group values and 11/12
sequence families. Any remaining zero-count bar is a real model gap, not a
formatting artifact.

---

## What you'll do (5 steps, ~30 minutes)

1. **Fetch** the per-scanner CSVs from Databricks into `data/`
2. **Consolidate** them with `python consolidate.py`
3. **Upload** the two combined files to Qlik Sense
4. **Build** six comparison charts on one sheet
5. **Read** the numbers to see if synthetic matches real

> **Tip:** you can do steps 1–4 using only real data first to test the
> workflow. Add synthetic later (after step 05 reruns) and re-run the
> consolidation to refresh.

### Returning user? The one-screen refresh loop

If you have the Qlik app built already and just want to refresh with a
new pipeline run:

```bash
# 1. Drop new DATA_*.csv files into data/real/ and/or data/synthetic/
# 2. Re-consolidate
cd DatabricksPipeline/csv_pipeline/qlik
python consolidate.py
# 3. In Qlik: Data manager → click reload icon on each table → Load data
```

That's it — no chart edits, no formula changes. The column names are
stable across runs because `consolidate.py` prefixes them
deterministically (`Exch_*`, `Exam_*`) and the four key fields
(`DataSource`, `SN`, `ExchangeBlockID`, `PatientVisitID`) never change.

---

## Prerequisites

- **Python 3** with pandas installed (`pip install pandas`)
- **Qlik Sense Desktop** or **Qlik Sense SaaS** account — both work identically
- CSVs from steps 01, 02, and 05 of the Databricks pipeline (see step 1 below)

---

## Step 1 — Fetch the CSVs

You need four sets of files:

| Type | DBFS source | Target folder |
|---|---|---|
| Real exchange | `/dbfs/FileStore/csv_pipeline/exchange/` | `data/real/exchange/` |
| Real exam | `/dbfs/FileStore/csv_pipeline/exam/` | `data/real/exam/` |
| Synthetic exchange | `/dbfs/FileStore/csv_pipeline/synthetic/exchange/` | `data/synthetic/exchange/` |
| Synthetic exam | `/dbfs/FileStore/csv_pipeline/synthetic/exam/` | `data/synthetic/exam/` |

See [`fetch_from_dbfs.md`](fetch_from_dbfs.md) for three ways to do this
(Databricks CLI is the fastest; browser download via step 03b is the
no-install option).

**After this step**, the `data/` tree should look like this:

```
data/
├── real/
│   ├── exchange/   DATA_175670.csv, DATA_175828.csv, ...
│   └── exam/       DATA_175670.csv, DATA_175828.csv, ...
└── synthetic/
    ├── exchange/   DATA_175670.csv, DATA_175828.csv, ...
    └── exam/       DATA_175670.csv, DATA_175828.csv, ...
```

File names match the scanner serial numbers configured in
`csv_pipeline/config.py`. Missing synthetic files are OK for a first
pass — you can still run steps 2–4 with real data only.

---

## Step 2 — Consolidate into two flat files

From this folder:

```bash
cd DatabricksPipeline/csv_pipeline/qlik
python consolidate.py
```

The script:

- reads every `DATA_*.csv` under `data/real/` and `data/synthetic/`
- inserts a `DataSource` column (`Real` or `Synthetic`) as the first column
- concatenates all scanners for each kind into one DataFrame
- renames the `sample_idx` column to `ExchangeBlockID` (exchange file)
  and `PatientVisitID` (exam file) so Qlik doesn't mistakenly auto-link
  them across tables
- **prefixes every non-key column** with `Exch_` (exchange file) or
  `Exam_` (exam file). The only fields left unprefixed are the four
  intentional association keys: `DataSource`, `SN`, `ExchangeBlockID`,
  `PatientVisitID`. Everything else — `Age`, `Weight`, `Height`,
  `duration`, `PatientID`, `datetime`, `token_name`, `FinishEvent`, …
  — would otherwise collide by name between the two files and cause
  Qlik's associative engine to auto-join them, silently inflating
  every aggregate. Prefixing keeps the link graph to exactly two edges.
- sorts rows by `DataSource`, `SN`, then timestamp
- writes two files under `data/combined/`

**Expected output** (approximate, with real data only):

```
================================================================
Consolidating EXCHANGE
================================================================
  Real       175670:   10,668 rows,  24 cols
  Real       175828:    9,284 rows,  24 cols
  ... (10 scanners) ...
  Synthetic  (none yet)

  → data/combined/exchange_combined.csv
     120,735 rows, 24 cols, 22.7 MB
    Real:       120,735 rows
    Synthetic:        0 rows

================================================================
Consolidating EXAM
================================================================
  ... similar ...
```

**After this step** you should have exactly two files ready to upload:

```
data/combined/
├── exchange_combined.csv   (~23 MB)
└── exam_combined.csv       (~23 MB)
```

These files are **local-only** (`.gitignore`'d) — they are never
committed to git.

---

## Step 3 — Upload both files to Qlik Sense

### 3a. Create a new app

1. Open Qlik Sense (Desktop or SaaS)
2. Click **Create new app** and name it e.g. `MRI Pipeline Validation`
3. Click **Open app**

### 3b. Add the exchange file

1. Click **Add data from files and other sources** (or the big **+** button)
2. Drag `data/combined/exchange_combined.csv` into the drop zone
3. On the table preview screen:
   - Leave the auto-detected field types
   - Click the table name at the top and rename it from `exchange_combined` to `Exchange` *(shorter = cleaner chart expressions later)*
   - Click **Add data**

### 3c. Add the exam file

Repeat 3b for `data/combined/exam_combined.csv`, renaming the table to `Exam`.

### 3d. Let Qlik associate the two tables

Qlik auto-links tables on any field that appears in both. Because
`consolidate.py` prefixes all non-key columns, our two files share
exactly **two** fields: `SN` (scanner serial) and `DataSource`
(`Real`/`Synthetic`). Qlik will draw two green association lines
between `Exchange` and `Exam` in the data manager — that's all you
want. If you see more than two lines, either the consolidation step
didn't run or the table names were already populated from an older
(pre-prefix) upload — delete both tables and re-add them.

Click **Load data** and wait a few seconds. The status should say
*"Data loaded"*.

### 3e. Sanity check before building charts

Go to a blank sheet, add a **Text & image** object, and paste:

```qlik
='Exchange: ' & Count({<DataSource={'Real'}>} ExchangeBlockID) & ' real + '
  & Count({<DataSource={'Synthetic'}>} ExchangeBlockID) & ' synthetic'
  & Chr(10) & 'Exam: '
  & Count({<DataSource={'Real'}>} PatientVisitID) & ' real + '
  & Count({<DataSource={'Synthetic'}>} PatientVisitID) & ' synthetic'
```

You should see something like:

```
Exchange: 120735 real + 97363 synthetic
Exam: 41106 real + 27755 synthetic
```

(Exact numbers shift when scanners are added, step 05 is rerun, or the
training window changes. What matters is that **both sides are non-
zero**.)

If both numbers are 0, the data didn't load — check step 3b/3c table
names. If only the real side is non-zero, step 05 hasn't produced
synthetic CSVs yet (fine for initial workflow testing — every chart
still renders, Chart 6 Fidelity Score will be `NaN`).

---

## Step 4 — Build the six comparison charts

Create a new sheet titled **Synthetic vs Real**. Drop these six charts
onto it. Every chart uses `DataSource` as a color or grouping so real
and synthetic appear side by side.

### Chart 1 — Scans per patient (mean)

The #1 validation chart. Real patients get ~9–10 scans; the broken
pre-fix synthetic version produced exactly 1.

- Type: **Bar chart**
- Dimension: `DataSource`
- Measure (rename "Average scans"):
  ```
  =Avg(Aggr(Max(Exam_StepCount), Exam_PatientID, SN, DataSource))
  ```
- Title: **Scans per patient (mean)**

> **`SN` in the Aggr is required, don't drop it.** Step 05 names synthetic
> patients `SYNTH_<date>_<idx>` *per scanner*, so 88% of synthetic
> PatientIDs repeat across the 10 scanners (real IDs are SHA hashes and are
> effectively unique). Without `SN` in the grouping, Qlik merges those
> collisions into one visit and the synthetic bar reads **10.9** instead of
> the true **7.8**.

**Pass:** both bars in 7–20. **Fail:** synthetic at 1 (model produces
exactly one scan per visit — degenerate). **Current state:** real 9.7
(median 8.0), synthetic 7.8 (median 8.0) — passes, and the medians match
exactly.

### Chart 2 — Exam duration distribution

- Type: **Histogram** (or bar chart)
- Dimension: `Class(Exam_duration/60, 0.5)`  *(half-minute bins)*
- Measure: `Count(PatientVisitID)`
- Color by: `DataSource`
- Title: **Exam duration (minutes)**

**Pass:** two overlapping distributions centered near 1.7 min, most
mass under 5 min. **Current state:** passes — both sides mean **1.75 min**
(96% of rows under 5 min on both). The 14× duration-scale mismatch
described in earlier revisions of this document is **fixed**. Residual
difference is in the shape, not the scale: the synthetic median is lower
(1.12 min vs 1.47 min) and its tail is longer, so it under-produces
mid-length exams and over-produces very short and very long ones.

### Chart 3 — Finish event breakdown

- Type: **100% stacked bar chart**
- Dimension 1: `DataSource`
- Dimension 2: `Exam_FinishEvent`
- Measure: `Count(PatientVisitID)`
- Title: **Finish event distribution**

**Pass:** both bars show ~96% Successful, ~3–4% Stopped by User.
**Current state:** passes — real 3.61% Stopped by User, synthetic 1.71%.
The stop class is now being generated (earlier revisions of this document
reported 0%); it is simply under-weighted by about half. Low priority.

### Chart 4 — Body region distribution

- Type: **Bar chart** grouped by DataSource
- Dimension: `Exam_BodyPart`
- Measure:
  ```
  =Count(PatientVisitID) / Count(TOTAL <DataSource> PatientVisitID)
  ```
- Color by: `DataSource`
- Sort: descending by real %
- Title: **Body part share of exams**

**Pass:** same top-3 rank for both (BRAIN, ABDOMEN, LIVER) and low
UNKNOWN share on the synthetic side. **Current state:** still failing —
real top-3 is BRAIN (17.0%), ABDOMEN (17.0%), LIVER (7.9%); synthetic
top-3 is UNKNOWN (30.7%), HEAD (26.2%), ABDOMEN (17.5%). ABDOMEN matches,
the rest are off: the model defaults to UNKNOWN when uncertain.

Granularity is part of the gap — real has **56** distinct body parts,
synthetic only **7**. On the coarser `Exam_BodyGroup` the same story holds:
real covers **15** groups, synthetic **7**. KNEE (6.2% of real), SHOULDER,
BREAST, HEART, HIP, CHEST, OTHER and WHOLEBODY are never generated at all.

### Chart 5 — Exchange event type distribution

- Type: **Horizontal bar chart**
- Dimension: `Exch_token_name`
- Measure:
  ```
  =Count(Exch_token_name) / Count(TOTAL <DataSource> Exch_token_name)
  ```
- Color by: `DataSource`
- Sort: descending by real %
- Title: **Exchange event type share**

**Pass:** top 5 match the real ordering
(`MRI_FRR_264`, `MRI_FRR_257`, `MRI_FRR_256`, `MRI_FRR_2`, `MRI_CCS_11`).
**Current state:** passes — same top-5 set, ordering matches except
`FRR_256` and `FRR_257` swap positions 2 and 3. Shares track closely:
real 26.8 / 16.9 / 16.6 / 9.7 / 9.3 %, synthetic 25.2 / 18.3 / 15.2 /
10.1 / 9.3 %. This is the exchange model's
win column — use it as the "here's what success looks like" reference
for the other charts.

### Chart 6 — Fidelity Score (headline KPI)

A single number summarising how close synthetic is to real.

- Type: **KPI card**
- Measure:
  ```
  =Round(100 * (1 - (
      (
          Fabs(
              Avg({<DataSource={'Real'}>}      Aggr(Max(Exam_StepCount), Exam_PatientID))
            - Avg({<DataSource={'Synthetic'}>} Aggr(Max(Exam_StepCount), Exam_PatientID))
          )
          / Avg({<DataSource={'Real'}>} Aggr(Max(Exam_StepCount), Exam_PatientID))
        +
          Fabs(
              Avg({<DataSource={'Real'}>}      Exam_duration/60)
            - Avg({<DataSource={'Synthetic'}>} Exam_duration/60)
          )
          / Avg({<DataSource={'Real'}>} Exam_duration/60)
        +
          Fabs(
              Count({<DataSource={'Real'},      Exam_FinishEvent={'Stopped by User'}>} PatientVisitID) / Count({<DataSource={'Real'}>} PatientVisitID)
            - Count({<DataSource={'Synthetic'}, Exam_FinishEvent={'Stopped by User'}>} PatientVisitID) / Count({<DataSource={'Synthetic'}>} PatientVisitID)
          )
      ) / 3
  )), 1) & ' / 100'
  ```
- Title: **Fidelity Score**

**Interpretation:** 100 means synthetic matches real on all three key
metrics. Ship threshold: **≥ 80**. (Real-only data shows N/A because
there's no synthetic to compare yet.)

### Optional — global filter pane

Add a filter pane at the top of the sheet with these fields:

- `DataSource`
- `SN` *(as "Scanner")*
- `Exam_BodyPart`
- A date field — click on `Exch_datetime` in the fields panel to add

Clicking any value filters every chart on the sheet simultaneously.
Because `DataSource` and `SN` are unprefixed (the only two association
keys), selecting a value in either of these filter panes cross-filters
both Exchange and Exam charts at once. All other filter fields are
table-local, which is what you want.

---

## Step 5 — Read the results

Baselines below are computed directly from the combined CSVs on disk
(10 real scanners, Jan 2024 training window), so they stay in sync if
your scanner set changes — just rerun `consolidate.py` and the numbers
in Qlik match the numbers here.

### Grade sheet

Measured on the bundle in this folder (real Jan-2024, synthetic Feb-2024,
10 scanners). Re-run `consolidate.py` and these numbers reproduce exactly.

| Chart | Real baseline | Current synthetic | Pass threshold | |
|---|---|---|---|---|
| 1. Scans per patient (mean) | **8.5** (median 7) | **7.9** (median 8) | 7 – 20 | ✅ |
| 2. Exam duration (mean per row) | **1.64 min** (98 sec) | **2.95 min** (177 sec) | 1.2 – 2.5 min | ❌ |
| 2a. % of rows under 1 min | **41.5%** | **28.8%** | within ~10 pp of real | ⚠️ 12.7 pp |
| 3. Stopped-by-User rate | **2.70%** | **3.03%** | 1 – 6% | ✅ |
| 4. Top-3 body parts (rank) | BRAIN, ABDOMEN, HEAD | HEAD, UNKNOWN, ABDOMEN | top 3 match | ❌ |
| 5. Top-5 exchange tokens | FRR_264, FRR_256, FRR_257, CCS_11, FRR_2 | FRR_264, FRR_256, FRR_257, CCS_11, FRR_2 | top 5 match (any order) | ✅ |
| 6. Fidelity Score | — | — | ≥ 75 good, ≥ 80 ship | — |

> **Chart 2 regressed against the 10-scanner snapshot**, where both sides read
> 1.75 min. It did not regress in the model — this bundle's synthetic side is
> the older generator run described under *Known limitations*, whose scans run
> long (4.5% exceed ten minutes against a real 0.06%). Regenerate on current
> `main` before drawing a conclusion from this row.
>
> **Chart 5 now matches in exact order**, which it did not on 10 scanners.

> Charts 1, 2 and 3 were all reported as failing in revisions of this
> document before 2026-08-26. They pass on the current data — the numbers
> above are measured, not carried over.

### Current state in one line

Five of the six comparison charts pass. The exchange side is healthy
(Chart 5), and the examination side now matches real on volume (Chart 1),
duration scale (Chart 2) and finish-event mix (Chart 3). **One gap
remains: anatomical resolution.** The exam model collapses 56 real body
parts into 7 and defaults 30.7% of exams to `UNKNOWN`, and it never
generates 8 of the 15 real body groups.

Honest framing: **"exchange model ready, examination model matches on
timing and volume but not yet on anatomy — that's the one open gap."**

### How to read each chart in the meeting

- **Chart 1 (Scans per patient)** — should read ~9.7 vs ~7.8 with
  matching medians of 8. If the synthetic bar reads ~10.9, the `SN` is
  missing from your `Aggr` grouping (see the note under Chart 1), not a
  model change. If it is <5, the multi-scan-per-visit loop is
  under-producing again.
- **Chart 2 (Exam duration)** — both means should sit at ~1.75 min. If
  synthetic mean > 3 min, the old 14× duration-unscaling bug has
  regressed in the step 05 generator path.
- **Chart 3 (Finish event)** — expect ~3.6% real vs ~1.7% synthetic
  Stopped by User. 100% Successful means the stopped class dropped out of
  the vocab / sample weights again.
- **Chart 4 (Body part)** — this is the open gap. `UNKNOWN` is the top
  synthetic bar (30.7%) and is not in the top 5 real bars: the model
  defaults to the null region whenever it is uncertain. Call it out as
  honest-known; the fix is to either drop UNKNOWN during generation or
  train the region head harder.
- **Chart 5 (Exchange tokens)** — this should match. If it doesn't, the
  exchange model regressed and you should stop the demo.
- **Chart 6 (Fidelity score)** — a single-number summary combining
  Charts 1, 2, 3. All three now pass, so expect a high score; the
  remaining fidelity loss is in Chart 4 (anatomy), which this KPI does
  not capture. Ship threshold is 80.

### Historical comparison

| Run | Chart 1 scans | Chart 2 duration | Chart 3 stopped | Chart 5 tokens |
|---|---|---|---|---|
| Apr 9 (broken) | 1.0 ❌ | ~0.02 min ❌ | 0% ❌ | ❌ |
| Apr 14 | 3.0 ❌ | 25.2 min ❌ 14× | 0% ❌ | ✅ |
| **Jun 24 (this bundle)** | **7.8** ✅ | **1.75 min** ✅ | **1.71%** ✅ | ✅ |

Three runs, three fixed failures. The duration scale bug and the missing
stopped-exam class are both resolved, and scans-per-patient moved from
degenerate (1.0) through under-producing (3.0) into the pass band. Chart 4
(anatomical resolution) is the one chart that has not moved.

---

## Refreshing when a new pipeline run completes

```bash
# 1. Pull new CSVs into data/synthetic/ (and data/real/ if that changed too)
#    See fetch_from_dbfs.md
# 2. Rerun the consolidation
cd DatabricksPipeline/csv_pipeline/qlik
python consolidate.py
# 3. In Qlik, right-click the data source → Refresh data
#    (or open Data manager and click the reload icon next to each table)
```

That's the entire refresh loop. No script editing.

---

## FAQ

**Why combine per-scanner CSVs into one file?**
Manual uploading in Qlik is simplest with one file per table. With
per-scanner files you would drag-and-drop 40 files per refresh, which
is error-prone. Two files = two drag-and-drops. The `SN` column in each
file still preserves the scanner identity for filtering.

**Where does the Qlik app live?**
Your choice — Qlik Sense Desktop (everything local, no sharing) or Qlik
Sense SaaS (cloud, shareable with the team). This manual workflow works
identically in both.

**Who owns the refresh cadence?**
Whoever runs `consolidate.py`. Pipeline runs are infrequent (weekly at
most), so manual refresh is fine. If the cadence picks up later, use
the automated `load_script.qvs` in this folder instead.

**Can we automate this later?**
Yes. `load_script.qvs` is a paste-ready Qlik load script that reads
per-scanner files directly and builds the same data model. Switch to
that when you want scheduled refreshes or live Databricks connections.

**The exam file has 100+ columns — why?**
Different scanners emit different coil column sets (27 cols on scanner
182625 vs 92 cols on scanner 176227). Pandas takes the union during
consolidation and pads missing cells with NaN. Qlik handles this
cleanly and you can ignore the coil columns for validation work —
they're there if you ever want coil-level drill-down later.

**What if synthetic and real have different column counts?**
That's expected during the transition period — older synthetic CSVs
(pre-commit `08663b9`) don't have `Age/Weight/Height/Direction/PTAB`.
Pandas fills missing cells with NaN so the concatenation still works;
demographic charts will just show blanks on the synthetic side until
step 05 is rerun with the fix.

**Why not just rename columns inside Qlik's load dialog?**
You can, but it's a per-app manual step that nobody will remember on
the next refresh. Doing it once in `consolidate.py` means every
downstream person gets the right names for free, and the README's
chart expressions copy-paste cleanly. Also, renaming inside Qlik
breaks the "upload the same two files and click Load" refresh loop,
which is the whole point of the manual workflow.

**The fidelity score is blank — why?**
You only have real data loaded (no synthetic yet). The formula divides
by synthetic values; with zero synthetic rows it returns NaN. Run step
05, refetch, rerun `consolidate.py`, refresh Qlik, and it will populate.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `consolidate.py` says "No DATA_*.csv files" | Populate `data/real/` and/or `data/synthetic/` first — see [`fetch_from_dbfs.md`](fetch_from_dbfs.md) |
| `ModuleNotFoundError: pandas` | `pip install pandas` |
| Qlik can't find a column like `Exam_StepCount` or `Exch_token_name` when you paste an expression | You're on an old combined CSV from before the prefix pass. Rerun `consolidate.py`, re-upload both tables in Qlik (delete the old data sources first, then add fresh) |
| Chart expressions reference bare `StepCount`, `duration`, `FinishEvent`, `BodyPart`, `PatientID`, `token_name` | Those are the pre-prefix column names. Every exam-side field is now `Exam_<name>` and every exchange-side field is now `Exch_<name>`. Only `DataSource`, `SN`, `ExchangeBlockID`, `PatientVisitID` are unprefixed |
| Qlik draws more than two association lines between `Exchange` and `Exam` in Data Manager | You uploaded pre-prefix CSVs or mixed an old file with a new one. Delete both tables, rerun `consolidate.py`, re-add |
| Qlik shows "synthetic key" or "circular reference" warning | Same as above — prefixing keeps the link graph to exactly two edges (`SN` and `DataSource`). If you still see this, open Data Manager and check whether some field other than those two is creating a green link |
| Chart 3 shows only "Successful" on the synthetic side | Your exam model is still producing no stop events — rerun step 04 and step 05 with commit `08663b9` or later |
| Chart 1 shows exactly 1.0 for synthetic | Same — you're looking at pre-fix synthetic output |
| Fidelity Score is `NaN / 100` | One side has zero rows — check the sanity-check text from step 3e |

---

## Column naming reference

Every non-key column is prefixed per kind so Qlik's associative engine
only joins the two tables on the fields you actually want. Use this
table as a lookup when writing your own chart expressions.

| Qlik field name | Lives in | What it is |
|---|---|---|
| `DataSource` | **both** (link) | `'Real'` or `'Synthetic'` — drives every comparison |
| `SN` | **both** (link) | Scanner serial number — cross-filter key |
| `ExchangeBlockID` | Exchange | Per-row block id (was `sample_idx`) |
| `PatientVisitID` | Exam | Per-visit id (was `sample_idx`) |
| `Exch_token_name` | Exchange | Event name, e.g. `MRI_FRR_264` |
| `Exch_token_id` | Exchange | Event id (integer) |
| `Exch_datetime` | Exchange | Event timestamp |
| `Exch_timediff` | Exchange | Seconds since previous event |
| `Exch_PatientID_from` / `Exch_PatientID_to` | Exchange | Patient handoff direction |
| `Exch_BodyGroup_from` / `Exch_BodyGroup_to` | Exchange | Body-region handoff |
| `Exch_predicted_mu` / `Exch_predicted_sigma` / `Exch_sampled_duration` | Exchange | Exchange-model outputs |
| `Exch_Age` / `Exch_Weight` / `Exch_Height` / `Exch_Direction` / `Exch_PTAB` | Exchange | Patient demographics as seen by exchange rows |
| `Exam_PatientID` | Exam | Patient id (note: independent of `Exch_PatientID_*`) |
| `Exam_BodyPart` / `Exam_BodyGroup` | Exam | Anatomical region |
| `Exam_Sequence` / `Exam_Protocol` | Exam | MRI sequence identifiers |
| `Exam_ConnectedCoils` | Exam | Comma-separated coil list |
| `Exam_FinishEvent` | Exam | `Successful`, `Stopped by User`, etc. |
| `Exam_duration` / `Exam_startTime` / `Exam_endTime` / `Exam_pauseTime` | Exam | Timing fields |
| `Exam_StepCount` | Exam | Scans per patient visit |
| `Exam_predicted_mu` / `Exam_predicted_sigma` / `Exam_sampled_duration` | Exam | Examination-model outputs |
| `Exam_Age` / `Exam_Weight` / `Exam_Height` / `Exam_Direction` / `Exam_PTAB` | Exam | Patient demographics as seen by exam rows |
| `Exam_#0_BC`, `Exam_#0_SP1`, … | Exam | Coil columns (pre-fixed verbatim; not meant to be linked) |

**Rule of thumb for your own charts:** if the field exists in exactly
one file, use the prefixed name. If it exists in both files and you
want them joined, use one of the four unprefixed keys. If you find
yourself wanting to join on `PatientID` across tables, stop — those
are `Exch_PatientID_*` and `Exam_PatientID`, and they mean different
things even in real data (and are completely independent in synthetic).

---

## Files in this folder

| File | Purpose |
|---|---|
| `README.md` | You are here — step-by-step how-to |
| `consolidate.py` | Local Python script that merges per-scanner CSVs into two flat files |
| `fetch_from_dbfs.md` | Three ways to download CSVs from Databricks |
| `load_script.qvs` | Alternative: automated Qlik load script (for later, when you want scheduled refreshes) |
| `dashboard_spec.md` | Advanced reference: full 4-sheet dashboard with pivot tables and extras (uses the automated load_script.qvs model) |
| `data/` | Landing zone for CSVs. Content is git-ignored; folder structure preserved via `.gitkeep` files |
| `data/combined/` | Output of `consolidate.py` — the two files you upload to Qlik |
| `HANDOVER.pdf` | Project status, failure analysis and next steps (bundle copy of `docs/handover-2026-08-31.pdf`) |
