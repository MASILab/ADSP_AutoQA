import subprocess
import argparse
from pathlib import Path
from tqdm import tqdm

def pa():
    parser = argparse.ArgumentParser(description='Link connectome special pngs')
    parser.add_argument('deriv_path', type=str, help='Path to the derivatives folder of the dataset')
    parser.add_argument('png_targ_dir', type=str, help='Path to png folder you want to link the files in')
    parser.add_argument('--test', action='store_true', help='Print the commands instead of running them')
    return parser.parse_args()

args = pa()
targ_dir = Path(args.png_targ_dir)

cmd = "find {} -mindepth 4 -maxdepth 4 -wholename '*ConnectomeSpecial*/ConnectomeQA.png'".format(args.deriv_path)
res = subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip().splitlines()

for r in tqdm(res):
    #get the subject name (and see if there is a session name)
    png_path = Path(r)
    if png_path.parent.parent.parent == 'derivatives':
        subj = png_path.parent.parent.name
        ses = ''
    else:
        subj = png_path.parent.parent.parent.name
        ses = png_path.parent.parent.name
    
    proc_dir = png_path.parent.name

    #get the new link path
    if ses:
        ses = '_'+ses
    link = targ_dir / "{sub}{ses}_{proc_dir}.png".format(sub=subj, ses=ses, proc_dir=proc_dir)

    if args.test:
        print('ln -s {} {}'.format(png_path, link))
        #print('ln -s {} {}'.format(png_path, link))
    else:
        link.symlink_to(png_path)