"""
When you have multiple CSV files due to different datasets and you wish to combine into 1 single CSV file

To be run AFTER combine_QA.py and combine_longitudinal_aggregate_csvs.py
"""

import pandas as pd
import argparse
from pathlib import Path

def pa():
    p = argparse.ArgumentParser(description='Combine longitudinal aggregate csvs. Outputs the result to the root qa directory')
    p.add_argument('--csv_list', nargs='+', required=True, help='List of csv files to combine. ASSUMES THAT THE PARENT IS THE NAME OF THE DATASET')
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
    dataset_name = Path(x).parent.name
    df['dataset'] = dataset_name
    #put dataset as the first column
    cols = df.columns.tolist()
    cols = ['dataset'] + [col for col in cols if col != 'dataset']
    df = df[cols]
    dfs.append(df)

#concat/merge them together, stacking rows
combined_df = pd.concat(dfs, ignore_index=True)

#see if there are duplicate rows based on sub, ses, acq, run
duplicate_rows = combined_df.duplicated(subset=['dataset', 'sub', 'ses', 'acq', 'run'], keep=False)
if duplicate_rows.any():
    print("Warning: There are duplicate rows based on sub, ses, acq, run. Please check the combined csv file. Not outputting the combined csv file until this is resolved.")
    print(combined_df[duplicate_rows])
    exit(1)

#output the combined csv file
combined_df.to_csv(out_csv, index=False)