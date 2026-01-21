"""
When you have multiple CSV files due to different data pulls (e.g. BLSA original, then BLSA 2024, BLSA 2025, etc.)

To be run AFTER combine_QA.py
"""

import pandas as pd
import argparse
from pathlib import Path

def pa():
    p = argparse.ArgumentParser(description='Combine longitudinal aggregate csvs. Outputs the result to the root qa directory')
    p.add_argument('--csv_list', nargs='+', required=True, help='List of csv files to combine')
    p.add_argument('--out_csv', type=str, required=True, help='Path to the output csv file')
    return p.parse_args()

args = pa()

#get the csv list
csv_list = args.csv_list
out_csv = Path(args.out_csv)
assert len(csv_list) > 1, "Please provide at least two csv files to combine"
assert out_csv.exists() == False, f"Output csv file {out_csv} already exists. Please provide a different path or delete the existing file."

#read in the csv files
dfs = []
for x in csv_list:
    df = pd.read_csv(x)
    for col in ["sub", "ses", "acq", "run"]:
        if col in df.columns:
            df[col] = df[col].astype("string")
    df['rowid'] = df['sub'] + df['ses'].fillna('') + df['acq'].fillna('') + df['run'].fillna('')
    dfs.append(df)

#concat/merge them together, either merging on sub, ses, acq, run or stacking rows if there are not matching columns
#combined_df = pd.concat(dfs, ignore_index=True)
#combined_df = dfs[0]
#for df in dfs[1:]:
#    combined_df = pd.merge(combined_df, df, on=["sub", "ses", "acq", "run"], how="outer")
#combined_df = pd.concat(dfs, axis=0, join="outer", ignore_index=True)
combined_df = dfs[0]
for df in dfs[1:]:
    combined_df = combined_df.set_index('rowid').combine_first(df.set_index('rowid')).reset_index()
print(combined_df.shape)
# combined_df = pd.concat(dfs, ignore_index=True)

#see if there are duplicate rows based on sub, ses, acq, run
# duplicate_rows = combined_df.duplicated(subset=['sub', 'ses', 'acq', 'run'], keep=False)
# if duplicate_rows.any():
#     #see if we can resolve the duplicates by checking if there is overlap between the columns that are not sub, ses, acq, run
#     for group, group_df in combined_df[duplicate_rows].groupby(['sub', 'ses', 'acq', 'run']):
#         #get the columns that are not sub, ses, acq, run
#         other_cols = [col for col in group_df.columns if col not in ['sub', 'ses', 'acq', 'run']]
#         #check if there is overlap between the columns
#         overlap = False
#         for col in other_cols:
#             #count if there is more than 1 non-nan unique value
#             if group_df[col].dropna() > 1:
#                 overlap = True
#                 break
#         if not overlap:
#             #if there is no overlap, we can merge the rows by taking the first non-nan value for each column
#             combined_row = group_df.iloc[0].copy()
#             for col in other_cols:
#                 combined_row[col] = group_df[col].dropna().iloc[0] if not group_df[col].dropna().empty else None
#         else:
#             print(f"Duplicate row as {col}")
#             print("Warning: There are duplicate rows based on sub, ses, acq, run. Please check the combined csv file. Not outputting the combined csv file until this is resolved.")
#             print(combined_df[duplicate_rows])
#             exit(1)



# duplicate_rows = combined_df.duplicated(subset=['sub', 'ses', 'acq', 'run'], keep=False)

# if duplicate_rows.any():
#     merged_rows = []

#     print("hi")
#     for group, group_df in combined_df.groupby(['sub', 'ses', 'acq', 'run'], dropna=False):
#         other_cols = [col for col in group_df.columns if col not in ['sub', 'ses', 'acq', 'run']]

#         merged = group_df.iloc[0].copy()  # start with the first row
#         for col in other_cols:
#             non_nan_vals = group_df[col].dropna()

#             if len(non_nan_vals) > 1:
#                 raise ValueError(
#                     f"Conflict detected in group {group} for column '{col}': values = {non_nan_vals}"
#                 )
#             elif len(non_nan_vals) == 1:
#                 merged[col] = non_nan_vals[0]
#             else:
#                 merged[col] = None  # keep NaN if all were NaN

#         merged_rows.append(merged)

#     # replace the duplicate rows with merged ones
#     merged_df = pd.DataFrame(merged_rows)
#     combined_df = pd.concat([combined_df[~duplicate_rows], merged_df], ignore_index=True)

#drop the rowid column
combined_df = combined_df.drop(columns=['rowid'])

#output the combined csv file
combined_df.to_csv(out_csv, index=False)