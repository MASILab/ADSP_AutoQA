import pandas as pd
import argparse, os, grp
import subprocess
from pathlib import Path
from functools import reduce

##NOTE: when we need to run tractseg QA, there need to be more pipelines names

DTI_PROC = [
    'PreQual',
    'WMAtlasEVE3',
    'ConnectomeSpecial',
    'FrancoisSpecial',
    'Tractseg*',
    'NODDI',
    'WMAtlas',
    'freewater'
]
DTI_PROC += [
    "TractsegCC5", "TractsegPOPTleft", "TractsegSTPREMleft",
    "TractsegCC6", "TractsegPOPTright", "TractsegSTPREMright",
    "TractsegCC7", "TractsegSCPleft", "TractsegSTRleft",
    "TractsegCGleft", "TractsegSCPright", "TractsegSTRright",
    "TractsegCGright", "TractsegSLFIIIleft", "TractsegSLFIIIright",
    "TractsegCSTleft", "TractsegSLFIIleft", "TractsegSLFIIright",
    "TractsegCSTright", "TractsegSLFIleft", "TractsegSLFIright",
    "TractsegFPTleft", "TractsegSTFOleft", "TractsegSTFOright",
    "TractsegFPTright", "TractsegSTOCCleft", "TractsegSTOCCright",
    "TractsegFXleft", "TractsegSTPARleft", "TractsegSTPARright",
    "TractsegFXright", "TractsegSTPOSTCleft", "TractsegSTPOSTCright",
    "TractsegICPleft", "TractsegSTPRECleft", "TractsegSTPRECright",
    "TractsegAFleft", "TractsegICPright", "TractsegSTPREFleft", "TractsegSTPREFright",
    "TractsegAFright", "TractsegIFOleft", "TractsegSTPREMleft", "TractsegSTPREMright",
    "TractsegATRleft", "TractsegIFOright", "TractsegTPARleft", "TractsegTPARright",
    "TractsegATRright", "TractsegILFleft", "TractsegTPRECleft", "TractsegTPRECright",
    "TractsegCA", "TractsegILFright", "TractsegTPREFleft", "TractsegTPREFright",
    "TractsegCC", "TractsegMCP", "TractsegTPREMleft", "TractsegTPREMright",
    "TractsegCC1", "TractsegMLFleft", "TractsegUFleft", "TractsegUFright",
    "TractsegCC2", "TractsegMLFright",
    "TractsegCC3", "TractsegORleft", "TractsegCC4",
    "TractsegORright", "TractsegSTPARleft", "TractsegSTPARright"
]


T1_PROC= [
    'SLANT-TICVv1.2',
    'Synthstrip',
    'UNest',
    'MaCRUISEv3.2.0',
    'BISCUITv2.2',
    'freesurfer',
]

def pa():
    parser = argparse.ArgumentParser(description='Combine QA csvs. Outputs the result to the root qa directory')
    parser.add_argument('dataset', type=str, help='Name of the dataset')
    parser.add_argument('root_qa_dir', type=str, help='Path to the root qa directory')
    parser.add_argument('--process', default='dti', type=str, choices=['dti', 't1'], help='Process type to combine')
    parser.add_argument('--test', action='store_true', help='Print the commands instead of running them')
    return parser.parse_args()

def reorder_columns(df):
    """
    Reorders the columns of the dataframe to be more intuitive
    """
    if args.process == 'dti':
        proc_list = DTI_PROC
    else:
        proc_list = T1_PROC
    
    cols_sub_ses_acq_run = ['sub', 'ses', 'acq', 'run']
    cols_QA_status = [z+'_QA_status' for z in proc_list if z+'_QA_status' ]#in df.columns]
    cols_reason = [z+'_reason' for z in proc_list if z+'_reason' ]#in df.columns]
    cols_user = [z+'_user' for z in proc_list if z+'_user' ]#in df.columns]
    cols_date = [z+'_date' for z in proc_list if z+'_date' ]#in df.columns]

    ordered_cols = cols_sub_ses_acq_run + cols_QA_status + cols_reason + cols_user + cols_date

    for col in ordered_cols:
        if col not in df.columns:
            df[col] = None

    return df[ordered_cols]

def set_file_permissions(file_path, group_name='p_masi', permissions=0o775):
    """
    sets the file permissions to 775 and the group to 'p_masi'
    """

    #set the permissions to be 775
    os.chmod(file_path, permissions)
    #set the group to be 'p_masi'
    group_name = 'p_masi'
    gid = grp.getgrnam(group_name).gr_gid
    os.chown(file_path, -1, gid)


args = pa()

dataset_qa_dir = Path(args.root_qa_dir) / args.dataset
assert dataset_qa_dir.exists(), "Dataset QA directory does not exist"

cmd = "find {} -mindepth 2 -maxdepth 2 -name 'QA.csv'".format(dataset_qa_dir)
res = subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip().splitlines()

if args.process == 'dti':
    res = [x for x in res if Path(x).parent.name in DTI_PROC]
else:
    res = [x for x in res if Path(x).parent.name in T1_PROC]

dfs = []
for r in res:
    df = pd.read_csv(r)
    #change the column names of all except 'sub', 'ses', 'acq', 'run' to include the process name
    proc_name = Path(r).parent.name
    cols = df.columns
    for c in cols:
        if c not in ['sub', 'ses', 'acq', 'run']:
            df.rename(columns={c: '{}_{}'.format(proc_name, c)}, inplace=True)
    dfs.append(df)

#combine the dataframes
merged_df = reduce(lambda left, right: pd.merge(left, right, on=['sub', 'ses', 'acq', 'run'], how='outer'), dfs)

#now, order the rows so that 'sub', 'ses', 'acq', 'run' come first, then all the columns named 'QA_status*', then 'reason*', then 'user*', then 'date*'
merged_df = reorder_columns(merged_df)

#df = pd.concat(dfs, ignore_index=True, axis=1)
if args.test:
    print(merged_df.columns) 
    print(merged_df)
else:
    output_df = dataset_qa_dir / '{}_QA_total.csv'.format(args.process.upper())
    merged_df.to_csv(output_df, index=False)
    set_file_permissions(output_df)
    #merged_df.to_csv(dataset_qa_dir / 'QA.csv', index=False)
    #set_file_permissions(dataset_qa_dir / 'QA.csv')