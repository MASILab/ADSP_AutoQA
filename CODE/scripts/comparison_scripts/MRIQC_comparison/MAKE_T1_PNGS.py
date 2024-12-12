#Author: Michael Kim

#python3 MAKE_SLANT_PNGS.py /nfs2/harmonization/BIDS/ADNI_DTI/ /nfs2/harmonization/ADSP_QA/ADNI_DTI/SLANT-TICVv1.2/

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import nibabel as nib
import matplotlib.colors as colors
from tqdm import tqdm
import subprocess
import argparse
import re
import multiprocessing

def pa():
    p = argparse.ArgumentParser()
    p.add_argument('dataset_path', type=str, help="BIDS dataset path to generate pngs of")
    p.add_argument('output_folder', type=str, help="Path to the output directory for the pngs")
    return p.parse_args()

#def make_LAS(img):
    #given an image, return the image in LAS orientation


def get_slice(img, dim, slicenum):
    if dim==0:
        return img[slicenum,:,:]
    elif dim==1:
        return img[:,slicenum,:]
    else:
        return img[:,:,slicenum]

def get_aspect_ratio(dim, vox_dim):
    if dim == 0:
        vox_ratio = vox_dim[2]/vox_dim[1]
    elif dim == 1:
        vox_ratio = vox_dim[2]/vox_dim[0]
    elif dim == 2:
        vox_ratio = vox_dim[1]/vox_dim[0]
    return vox_ratio

def make_LAS(nii):
    """
    Given a nibabel image, return the LAS orientated image.
    """
    # Get the data
    data = nii.get_fdata()
    # Get the affine
    aff = nii.affine
    # Get the orientation
    orient = nib.aff2axcodes(aff)
    #print(orient)
    
    # Reorient to LAS
    if orient != ('L', 'A', 'S'):
        # Get the current orientation
        ornt = nib.orientations.io_orientation(aff)
        # Define the target orientation (LAS)
        target_ornt = nib.orientations.axcodes2ornt(('L', 'A', 'S'))
        # Get the transformation matrix
        transform = nib.orientations.ornt_transform(ornt, target_ornt)
        # Reorient the data
        data = nib.orientations.apply_orientation(data, transform)
        # Update the affine
        aff = nib.orientations.inv_ornt_aff(transform, data.shape)
    
    # Create a new Nifti image
    new_nii = nib.Nifti1Image(data, aff)
    return new_nii

#creates a png for one atlas
def create_seg_png(imgdata, outfile, atlas_name, imghd):
    #create the plt figure
    f, ax = plt.subplots(3,3,figsize=(10.5, 8), constrained_layout=True)
    imgdata = np.clip(imgdata, 0, np.percentile(imgdata, 99))
    #loop through sag, coronal, axial slices
    for dim in range(3):
        #get the center slice
        slice=imgdata.shape[dim]//2
        for i,slice_offput in enumerate(range(-10, 20, 10)):
            #get the aspect ratio for plotting purposes
            vox_dims = imghd.get_zooms()
            ratio = get_aspect_ratio(dim, vox_dims)
            #get the slices we want to show
            try:
                img_slice = np.rot90(get_slice(imgdata, dim, slice+slice_offput), k=1)
                #seg_slice = np.rot90(get_slice(segdata, dim, slice+slice_offput), k=1)
            #if this doesnt work, get the maximum slice in that dimension
            except:
                if i == 0:
                    img_slice = np.rot90(get_slice(imgdata, dim, 0), k=1)
                    #seg_slice = np.rot90(get_slice(segdata, dim, 0), k=1)
                elif i == 2:
                    img_slice = np.rot90(get_slice(imgdata, dim, -1), k=1)
                    #seg_slice = np.rot90(get_slice(segdata, dim, -1), k=1)
            #plot the slices
            ax[dim,i].imshow(img_slice, cmap='gray', aspect=ratio)
            #ax[dim,i].imshow(seg_slice, alpha=0.6, cmap=cmap, aspect=ratio, interpolation='nearest')
            ax[dim,i].axis('off')
    #save the slices
    #print(outdir)
    f.suptitle(atlas_name)
    f.savefig(outfile, bbox_inches='tight')
    #close the figure
    plt.close(f)

#for SLANT segmentation
def setup_and_make_png_seg(imgfile, atlas_name, outfile):
    #load in image file
    img = make_LAS(nib.load(imgfile))
    imgdata = np.squeeze(img.get_fdata()[:,:,:])
    if len(imgdata.shape) == 4:
        imgdata = imgdata[:,:,:,0]

    #create the png
    create_seg_png(imgdata, outfile, atlas_name, img.header)


def get_BIDS_tags_from_seg_path(seg_path):
    """
    Returns the BIDS tags from a SLANT segmentation path
    """

    pattern = r'(sub-\w+)(?:_(ses-\w+))?(?:_(acq-\w+))?(?:_(run-\d{1,2}))?_T1w_seg'
    matches = re.findall(pattern, str(seg_path).split('/')[-1])
    sub, ses, acq, run = matches[0][0], matches[0][1], matches[0][2], matches[0][3]
    return sub, ses, acq, run

def get_BIDS_fields_t1(t1_path):
    pattern = r'(sub-\w+)(?:_(ses-\w+))?(?:_(acq-\w+))?(?:_(run-\d{1,3}))?_T1w'
    matches = re.findall(pattern, str(t1_path).split('/')[-1])
    sub, ses, acq, run = matches[0][0], matches[0][1], matches[0][2], matches[0][3]
    return sub, ses, acq, run

def main():

    args = pa()
    dataset_root = Path(args.dataset_path)
    outdir = Path(args.output_folder)

    assert dataset_root.exists(), "Error: dataset path does not exist"
    assert outdir.exists(), "Error: output folder does not exist"

    #derivs = dataset_root / 'derivatives'

    #find all the T1 files
    cmd = "find {} -mindepth 3 -maxdepth 4 -type l -name '*T1w.nii.gz'".format(dataset_root)
    #slant_dirs = subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip().splitlines()
    t1s = subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip().splitlines()

    args_list = []

    for t1_path in tqdm(t1s):
        t1 = Path(t1_path)

        #get the sub, ses, acq, run
        sub, ses, acq, run = get_BIDS_fields_t1(t1)

        sesx = '_'+ses if ses is not None and ses != ''  else ''
        acqx = '_'+acq if acq is not None and acq != '' else ''
        runx = '_'+run if run is not None and run != ''  else ''
        outpng = outdir / '{}{}_T1w{}{}.png'.format(sub, sesx, acq, run)

        #make the output file name
        #outpng = outdir / t1.name.replace('.nii.gz', '.png')
        if outpng.exists():
            continue

        args_list.append((t1, "T1w Image", outpng))
    
    with multiprocessing.Pool(20) as p:
        p.starmap(setup_and_make_png_seg, args_list)

        #make the png
        #setup_and_make_png_seg(t1, "T1w Image", outpng)
        #setup_and_make_png_seg(lut_file, imgfile, segfile[0], "SLANT", 
                                    #os.path.join(output_sub_folder,outfile), skip_header=4)
        #print(outpng)
        #break


if __name__ == "__main__":
    main()