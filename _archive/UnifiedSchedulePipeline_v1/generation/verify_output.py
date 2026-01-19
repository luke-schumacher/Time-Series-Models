"""
Verify the generated daily schedule meets all requirements
"""
import pandas as pd
import os

# Load the generated schedule
csv_path = os.path.join('..', 'outputs', 'generated_schedules', 'daily_schedule_2026-01-10.csv')
df = pd.read_csv(csv_path)

print("=" * 70)
print("VERIFICATION REPORT - Daily Schedule Generation")
print("=" * 70)

# Test 1: Column structure
print("\n[TEST 1] CSV Column Structure")
expected_columns = ['event_id', 'session_id', 'patient_id', 'timestamp', 'datetime_str',
                   'event_type', 'token_name', 'body_part', 'scan_type',
                   'duration', 'source', 'time_since_start']
has_all_columns = all(col in df.columns for col in expected_columns)
print(f"  [OK] All expected columns present: {has_all_columns}")
print(f"  Columns: {df.columns.tolist()}")

# Test 2: No special tokens
print("\n[TEST 2] Special Token Filtering")
start_count = len(df[df['token_name'] == 'START'])
end_count = len(df[df['token_name'] == 'END'])
pad_count = len(df[df['token_name'] == 'PAD'])
print(f"  [OK] START tokens: {start_count} (expected: 0)")
print(f"  [OK] END tokens: {end_count} (expected: 0)")
print(f"  [OK] PAD tokens: {pad_count} (expected: 0)")

# Test 3: Event types
print("\n[TEST 3] Event Types")
event_counts = df['event_type'].value_counts()
print(f"  [OK] PXChange events: {event_counts.get('pxchange', 0)}")
print(f"  [OK] Scan events: {event_counts.get('scan', 0)} (should be > 0)")

# Test 4: Measurement triggers
print("\n[TEST 4] Measurement Triggers")
eux_count = len(df[df['token_name'] == 'MRI_EXU_95'])
print(f"  [OK] MRI_EXU_95 tokens: {eux_count} (should be >= 1)")

# Test 5: Patient IDs
print("\n[TEST 5] Patient IDs")
unique_patients = df['patient_id'].nunique()
print(f"  [OK] Unique patient IDs: {unique_patients}")
print(f"  Patient IDs: {sorted(df['patient_id'].unique())}")

# Test 6: Body parts
print("\n[TEST 6] Body Parts")
body_part_counts = df['body_part'].value_counts()
print(f"  [OK] Unique body parts: {df['body_part'].nunique()}")
print(f"  Distribution:")
for bp, count in body_part_counts.items():
    print(f"    - {bp}: {count}")

# Test 7: Scan types
print("\n[TEST 7] Scan Types")
scan_type_filled = df[df['event_type'] == 'scan']['scan_type'].notna().sum()
total_scans = len(df[df['event_type'] == 'scan'])
print(f"  [OK] Scan events with scan_type: {scan_type_filled}/{total_scans}")

# Test 8: Trigger -> Scan flow
print("\n[TEST 8] EXU Trigger -> Scan Sequence Flow")
sessions_with_eux = df[df['token_name'] == 'MRI_EXU_95']['session_id'].unique()
sessions_with_scans = df[df['event_type'] == 'scan']['session_id'].unique()
print(f"  [OK] Sessions with EXU: {len(sessions_with_eux)} - {sessions_with_eux}")
print(f"  [OK] Sessions with scans: {len(sessions_with_scans)} - {sessions_with_scans}")

# Test 9: Example session structure
print("\n[TEST 9] Example Session Structure")
if len(sessions_with_eux) > 0:
    session_id = sessions_with_eux[0]
    session_df = df[df['session_id'] == session_id]
    eux_rows = session_df[session_df['token_name'] == 'MRI_EXU_95']

    if len(eux_rows) > 0:
        eux_idx = eux_rows.index[0]

        # Get context around EXU trigger
        start_idx = max(0, eux_idx - 3)
        end_idx = min(len(df) - 1, eux_idx + 8)
        context_df = df.loc[start_idx:end_idx]

        print(f"  Session {session_id} around EXU trigger (index {eux_idx}):")
        print(context_df[['event_type', 'token_name', 'body_part', 'scan_type', 'duration']].to_string())

        # Count scan events after EXU in this session
        scans_after_eux = session_df[(session_df.index > eux_idx) & (session_df['event_type'] == 'scan')]
        print(f"\n  [OK] Scan events triggered after EXU: {len(scans_after_eux)}")

# Test 10: Summary statistics
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Total events: {len(df)}")
print(f"Total sessions: {df['session_id'].nunique()}")
print(f"PXChange events: {event_counts.get('pxchange', 0)}")
print(f"Scan events: {event_counts.get('scan', 0)}")
print(f"Duration: {df['time_since_start'].max() / 3600:.2f} hours")

# Overall pass/fail
all_tests_pass = (
    has_all_columns and
    start_count == 0 and
    end_count == 0 and
    pad_count == 0 and
    event_counts.get('scan', 0) > 0 and
    eux_count >= 1 and
    unique_patients > 0 and
    scan_type_filled == total_scans
)

print("\n" + "=" * 70)
if all_tests_pass:
    print("[OK] ALL TESTS PASSED - Generation working correctly!")
else:
    print("[FAIL] SOME TESTS FAILED - Please review output")
print("=" * 70)
