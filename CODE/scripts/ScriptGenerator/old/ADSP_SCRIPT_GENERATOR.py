"""
Author: Michael Kim
Email: michael.kim@vanderbilt.edu
March 2024
"""

import pandas as pd
import argparse
from pathlib import Path
import re
import subprocess
from tqdm import tqdm
import os
import json
import numpy as np

#### NEED TO CHECK ALL OF THESE TO MAKE SURE THAT THEY WORK
#SLANT-TICV: DONE
#freesurfer: DONE - HCP test looks good
#EVE3WMAtlas: DONE - HCPA test looks good
#MNIWMAtlas
#UNest: DONE
#PreQual: DONE - HCP test looks good
#synthstrip: DONE
#tractseg: DONE - need to test
    #seems to be working ok, but need to test scripts on ACCRE
#connectomeSpecial: DONE - WRAP scripts look ok, need to test one
#francoisSpecial: DONE
#Biscuit: DONE

### TODO:
    #add the specific options for the OASIS3 PreQual

def pa():
    """
    Argument parser for the script generator. Specify what the dataset is, what type of processing you want to run, etc.

    ASSUMES THAT YOU HAVE SET UP AN SSH-KEY TO SCP TO ACCRE WITHOUT A PASSWORD. IF YOU DO NOT HAVE THIS, PLEASE SET IT UP AND TEST IT OUT BEFORE
    SUBMITTING JOBS TO ACCRE.

    Assumes that the datasets are in BIDS format.

    Outputs scripts for that processing to be run on the ACCRE cluster.

    This in intended to be run on a machine that has access to /nfs2 and /nobackup.

    TODO: also outputs the slurm script that will be needed to run the processing on the ACCRE cluster.
    - requires email, amount of memory per job, number of jobs to run in parallel, etc.
    - 
    """

    pipelines = ["PreQual", "SLANT-TICV", "freesurfer", "EVE3WMAtlas", "MNIWMAtlas",
                'UNest', 'synthstrip', 'tractseg', 'MaCRUISE', 'ConnectomeSpecial', 'Biscuit', 'FrancoisSpecial']

    p = argparse.ArgumentParser(description="Script generator for ADSP")
    p.add_argument('dataset_name', type=str, help="name of the dataset to run processing on in /nfs2/harmonization/BIDS/<dataset_name>")
    p.add_argument('tmp_input_dir', type=str, help="path to directory where you want to have temporary inputs (should be on nobackup for ACCRE)")
    p.add_argument('tmp_output_dir', type=str, help="path to directory where you want to have temporary outputs (should be on nobackup for ACCRE)")
    p.add_argument('script_dir', type=str, help="path to directory where you want to write all the scripts")
    p.add_argument('log_dir', type=str, help="path to directory where you want to write all the log outputs")
    p.add_argument('pipeline', type=str, help="name of the pipeline to run on the dataset")
    p.add_argument('--vunetID', type=str, help="VUnetID of the user running the pipeline. Necessary for scp to ACCRE. Default will pull from bash.")
    p.add_argument('--custom_output_dir', default='', type=str, help="path to directory where you want to write the outputs of the pipeline. If empty string, then the output will be written to the nfs2 directories.")
    p.add_argument('--freesurfer_license_path', default='', type=str, help="path to the freesurfer license file. Necessary for PreQual and Biscuit pipelines.")
    p.add_argument('--skull_stripped', action='store_true', help="For UNest: if you want to run it with the skull stripped T1")
    p.add_argument('--working_dir', type=str, default='', help="Required for UNest: path tp where you want the root working directory to be")
    p.add_argument('--temp_dir', type=str, default='', help="Required for UNest and tractseg: path to where you want the root temp directory to be")
    p.add_argument('--no_slurm', action='store_true', help="Set if you do not want to generate a slurm script for the pipeline")
    p.add_argument('--email', type=str, default='', help="Email to send notifications to for the slurm script. (Required for slurm script)")
    p.add_argument('--memory_per_job', type=str, default='16G', help="Memory to allocate per job for the slurm script (Ex: '16G' )")
    p.add_argument('--num_jobs', type=int, default=25, help="Number of jobs to run in parallel for the slurm script (Ex: '25' ) ")
    p.add_argument('--num_nodes', type=int, default=1, help="Number of nodes for each slurm job in the array (Ex: '1' ) ")
    p.add_argument('--time', type=str, default='24:00:00', help="Time to allocate per job for the slurm script 'HR:MIN:SEC'")
    p.add_argument('--slurm_script_output', type=str, default='', help="Path to the slurm script output")
    p.add_argument('--use_unest_seg', action='store_true', help="Use the UNest segmentation instead of SLANT-TICV for MaCRUISE, EVE3 registration, Connectome Special, etc.")
    p.add_argument('--custom_simg_path', default='', type=str, help="Path to the custom singularity image file for whatever pipeline you are running")
    p.add_argument('--no_accre', action='store_true', help="If you want to run not on ACCRE, but on a local machine. Will instead output a python file to parallelize the processing.")
    p.add_argument('--custom_atlas_input_dir', type=str, default='', help="For EVE3WMAtlas: custom path to the directory where the atlas inputs are stored")
    return p.parse_args()


class ScriptGenerator:
    """
    Class to maintain paths and options for the script generation.
    """

    def __init__(self, args):
        
        #mapping for the script generator
        self.PIPELINE_MAP = {
            "freesurfer": self.freesurfer_script_generate,
            "PreQual": self.PreQual_script_generate,
            "SLANT-TICV": self.SLANT_TICV_script_generate,
            "EVE3WMAtlas": self.EVE3Registration_script_generate,
            "MNI152WMAtlas": self.MNI152Registration_script_generate,
            "UNest": self.UNest_script_generate,
            "synthstrip": self.synthstrip_script_generate,
            "tractseg": self.tractseg_script_generate,
            "MaCRUISE": self.macruise_script_generate,
            "ConnectomeSpecial": self.connectome_special_script_generate,
            "FrancoisSpecial": self.francois_special_script_generate,
            "Biscuit": self.biscuit_script_generate,
            'NODDI': self.noddi_script_generate
        }

        if args.pipeline not in self.PIPELINE_MAP.keys():
            print("Available Pipelines: ", self.PIPELINE_MAP.keys())
            raise ValueError("Error: pipeline {} not recognized".format(args.pipeline))

        #FS license
        if args.freesurfer_license_path == '':
            self.freesurfer_license_path = Path('/nobackup/p_masi/Singularities/FreesurferLicense.txt')
        else:
            self.freesurfer_license_path = Path(args.freesurfer_license_path)
        assert self.freesurfer_license_path.exists(), "Error: Freesurfer license file {} does not exist".format(str(self.freesurfer_license_path))

        self.args = args
        #set up some paths that will be important later
        self.root_dataset_path = Path("/nfs2/harmonization/BIDS/{}".format(args.dataset_name))
        self.dataset_derivs = self.root_dataset_path/('derivatives')
        if args.custom_simg_path == '':
            self.simg = self.get_simg_path()
        else:
            self.simg = args.custom_simg_path
            assert Path(self.simg).exists(), "Error: Custom singularity image file {} does not exist".format(str(self.simg))
        self.pipeline = args.pipeline
        #vunetID
        if not args.vunetID:
            self.vunetID = os.getlogin()
        else:
            self.vunetID = args.vunetID

        #final output directory
        if args.custom_output_dir != '':
            self.output_dir = Path(args.custom_output_dir)
            print("NOTE: Custom output directory specified. Make sure to link the output to correct BIDS directory on /nfs2.")
        else:
            self.output_dir = self.dataset_derivs
            print("NOTE: No custom output directory specified. Output will be written directly to the derivatives directory in the BIDS dataset at {}.".format(str(self.output_dir)))

        #set up the script directory and the logs directory
        self.script_dir = Path(args.script_dir)
        self.log_dir = Path(args.log_dir)

        #check to make sure that directories exist and we have access to them
        self.check_assertions()

        #check and set up the slurm script output
        if not self.args.no_slurm:
            self.check_slurm()

        #set up the input and output directory
        self.tmp_input_dir = Path(args.tmp_input_dir)
        self.tmp_output_dir = Path(args.tmp_output_dir)

    def check_slurm(self):
        """
        Checks the slurm options to make sure that they are valid and sets up the output.
        """
        
        #check to make sure that the options are valid
        assert re.match(r'^\d{1,3}:\d{2}:\d{2}$', self.args.time), "Error: Time option {} is not valid".format(self.args.time)
        assert re.match(r'^\d{1,4}[G|M]$', self.args.memory_per_job), "Error: Memory option {} is not valid".format(self.args.memory_per_job)
        assert self.args.num_jobs > 0, "Error: Number of jobs {} is not valid. Must be greater than zero".format(self.args.num_jobs)

        if not self.args.email:
            print("Warning: No email specified for slurm script. Will not send notifications.")

        if not self.args.slurm_script_output:
            if not self.args.no_accre:
                self.slurm_path = self.script_dir/("{}_{}.slurm".format(self.root_dataset_path.name, self.pipeline))
            else: #parallelization script
                self.slurm_path = self.script_dir/("{}_{}.py".format(self.root_dataset_path.name, self.pipeline))
        else:
            self.slurm_path = Path(self.args.slurm_script_output)

        #check to make sure that the slurm script does not already exist and is not writable
        if self.slurm_path.exists():
            raise FileExistsError("Error: Slurm script {} already exists".format(self.slurm_path))
        if not os.access(self.slurm_path.parent, os.W_OK):
            raise PermissionError("Error: Slurm script {} is not writable".format(self.slurm_path))

        #if everything is valid, then set up the slurm script options
        self.slurm_options = {
            'email': self.args.email,
            'time': self.args.time,
            'memory_per_job': self.args.memory_per_job,
            'num_jobs': self.args.num_jobs,
            'output': self.slurm_path,
            'num_nodes': self.args.num_nodes
        }

    def get_simg_path(self):
        """
        Returns the path to the singularity image file for the processing pipeline on nobackup.

        If there is not a singularity image file for the pipeline, then it will return None.
        """
        try:
            mapping = {'SLANT-TICV': '/nobackup/p_masi/Singularities/nssSLANT_v1.2.simg',
                        'PreQual': '/nobackup/p_masi/Singularities/PreQual_v1.0.8.simg',
                        'EVE3WMAtlas': '/nobackup/p_masi/Singularities/WMAtlas_v1.1.simg',
                        'MNI152WMAtlas': '/nobackup/p_masi/Singularities/WMAtlas_v1.1.simg',
                        'UNest': '/nobackup/p_masi/Singularities/UNest.sif',
                        'tractseg': ["/nobackup/p_masi/Singularities/tractseg.simg", "/nobackup/p_masi/Singularities/scilus_1.5.0.sif"],
                        'MaCRUISE': "/nobackup/p_masi/Singularities/macruise_classern_v3.2.0.simg",
                        'FrancoisSpecial': '/nobackup/p_masi/Singularities/singularity_francois_special_v1.sif',
                        'ConnectomeSpecial': '/nobackup/p_masi/Singularities/ConnectomeSpecial.sif',
                        'Biscuit': '/nobackup/p_masi/Singularities/biscuit_FC_v2.2.sif',
                        'NODDI': '/nobackup/p_masi/Singularities/tractoflow_2.2.1_b9a527_2021-04-13.sif'
                    }
            simg = mapping[self.args.pipeline]
        except:
            return None
        return Path(simg) if type(simg) == str else [Path(x) for x in simg]

    def check_assertions(self):
        """
        Checks assumptions to make sure that the scripts can run and everything should work out ok.
        """

        def simg_assert(simg):
            assert Path(simg).exists(), "Error: simg file {} does not exist".format(str(self.simg))
            assert os.access(simg, os.R_OK), "Error: do not have read permissions for simg file {}".format(str(self.simg))
            assert os.access(simg, os.X_OK), "Error: do not have execute permissions for simg file {}".format(str(self.simg))

        #check for existence of directories
        assert Path(self.root_dataset_path).exists(), "Error: dataset root dir {} does not exist".format(str(self.root_dataset_path))
        assert Path(self.dataset_derivs).exists(), "Error: derivatives root dir {} does not exist".format(str(self.dataset_derivs))
        if self.simg is not None:
            if type(self.simg) == str:
                simg_assert(self.simg)
            elif type(self.simg) == list:
                for simg in self.simg:
                    simg_assert(simg)
        assert Path(self.args.tmp_input_dir).exists(), "Error: tmp_inputs dir {} does not exist".format(str(self.args.tmp_input_dir))
        assert Path(self.args.tmp_output_dir).exists(), "Error: tmp_outputs dir {} does not exist".format(str(self.args.tmp_output_dir))
        assert Path(self.args.script_dir).exists(), "Error: scripts dir {} does not exist".format(str(self.args.script_dir))
        assert Path(self.args.log_dir).exists(), "Error: logs dir {} does not exist".format(str(self.args.log_dir))
        assert Path(self.output_dir).exists(), "Error: final output dir {} does not exist".format(str(self.output_dir))

        #check for write permissions on all the directories
        assert os.access(self.root_dataset_path, os.W_OK), "Error: do not have write permissions for dataset root dir {}".format(str(self.root_dataset_path))
        assert os.access(self.dataset_derivs, os.W_OK), "Error: do not have write permissions for derivatives root dir {}".format(str(self.dataset_derivs))
        assert os.access(self.args.tmp_input_dir, os.W_OK), "Error: do not have write permissions for tmp_inputs dir {}".format(str(self.args.tmp_input_dir))
        assert os.access(self.args.tmp_output_dir, os.W_OK), "Error: do not have write permissions for tmp_outputs dir {}".format(str(self.args.tmp_output_dir))
        assert os.access(self.args.script_dir, os.W_OK), "Error: do not have write permissions for scripts dir {}".format(str(self.args.script_dir))
        assert os.access(self.args.log_dir, os.W_OK), "Error: do not have write permissions for logs dir {}".format(str(self.args.log_dir))
        assert os.access(self.output_dir, os.W_OK), "Error: do not have write permissions for final output dir {}".format(str(self.output_dir))

        #For prequal, check for the FS license
        if self.args.pipeline == "PreQual":
            assert Path(self.args.freesurfer_license_path).exists(), "Error: freesurfer license file does not exist at {}".format(str(self.args.freesurfer_license_path))

        #check to make sure that the scripts and logs directories are empty
        assert len(list(self.script_dir.iterdir())) == 0, "Error: scripts dir {} is not empty".format(str(self.script_dir))
        assert len(list(self.log_dir.iterdir())) == 0, "Error: logs dir {} is not empty".format(str(self.log_dir))

        #check that the dataset, inputs, outputs, derivatives, scripts, logs, and simg file all exist

    def get_t1(self, path, sub, ses):
        #given a path to the anat folder, returns a T1 if it exists
        t1s = [x for x in path.glob('*') if re.match("^.*T1w\.nii\.gz$", x.name)]
        #if there is only a single T1, return that
        if len(t1s) == 1:
            return t1s[0]
        if len(t1s) == 0:
            print("echo ERROR: {}_{} does not have a valid run1 T1".format(sub, ses))
            return None
        ####
        sesx = '_'+ses if ses != '' else ''
        ####
        #check the MPRAGEs first
        mpr1 = path/("{}{}_acq-MPRAGE_run-1_T1w.nii.gz".format(sub, sesx))
        spgr1 = path/("{}{}_acq-SPGR_run-1_T1w.nii.gz".format(sub, sesx))
        flair1 = path/("{}{}_acq-FLAIR_run-1_T1w.nii.gz".format(sub, sesx))
        tfe1 = path/("{}{}_acq-TFE_run-1_T1w.nii.gz".format(sub, sesx))
        se1 = path/("{}{}_acq-SE_run-1_T1w.nii.gz".format(sub, sesx))
        unknown1 = path/("{}{}_acq-unknown_run-1_T1w.nii.gz".format(sub, sesx))
        for type in [mpr1, spgr1, flair1, tfe1, se1, unknown1]:
            if type.exists():
                return type
        #check the runs without acquisitions
        run1 = path/("{}{}_run-1_T1w.nii.gz".format(sub, sesx))
        run01 = path/("{}{}_run-01_T1w.nii.gz".format(sub, sesx))
        if run1.exists():
            return run1
        elif run01.exists():
            return run01
        #check the mprages, FLAIR, SE, SPGR, unknown without runs
        mpr = path/("{}{}_acq-MPRAGE_T1w.nii.gz".format(sub, sesx))
        spgr = path/("{}{}_acq-SPGR_T1w.nii.gz".format(sub, sesx))
        flair = path/("{}{}_acq-FLAIR_T1w.nii.gz".format(sub, sesx))
        tfe = path/("{}{}_acq-TFE_T1w.nii.gz".format(sub, sesx))
        se = path/("{}{}_acq-SE_T1w.nii.gz".format(sub, sesx))
        unknown = path/("{}{}_acq-unknown_T1w.nii.gz".format(sub, sesx))
        for type in [mpr, spgr, flair, tfe, se, unknown]:
            if type.exists():
                return type
        #if we have gotten here, return the first T1 that we find
        print("Using {} for {}_{}".format(t1s[0], sub, ses))
        return t1s[0]

        print("echo ERROR: {}_{} does not have a valid run1 T1".format(sub, ses))
        return None

    def write_script(self):
        """
        Writes script to the specified script directory.
        """
        pass


    def write_PreQual_script(self, session_input, session_output, i, dwiname, PEaxis, PEsign, PEunknown, t1, dwi, bval, bvec, jsonf, PQdir, sub, ses):
        """
        Writes a PreQual script to the specified directory.
        """
        
        opts = ''
        if self.args.dataset_name == "HCPA" or self.args.dataset_name == "HCPD" or self.args.dataset_name == "HCP":
            no_t1 = True
            opts = "--denoise off"
        else:
            no_t1 = False

        script_f = self.script_dir/("{}.sh".format(i))
        with open(script_f, 'w') as script:

            csvtext = "{},{},0.05".format(dwiname, PEsign)
            script.write('#!/bin/bash\n')
            if PEunknown:
                script.write('echo PE DIRECTION UNKNOWN FOR {}. ASSUMING j+\n'.format(dwiname))
            script.write('echo Using {} as T1\n'.format(str(t1)))
            script.write('echo Starting PreQual for {}\n'.format(dwiname))
            script.write('echo {} > {}\n'.format(csvtext, str(session_input/("dtiQA_config.csv"))))
            script.write('echo Created dtiQA_config.csv file. Now copying over files from nfs2...\n')
                                #now, copy over the files
            if not no_t1:
                script.write('scp {}@landman01.accre.vanderbilt.edu:{} {}\n'.format(self.vunetID, str(t1), str(session_input)+'/t1.nii.gz'))
            script.write('scp {}@landman01.accre.vanderbilt.edu:{} {}\n'.format(self.vunetID, str(dwi), str(session_input)+'/'))
            script.write('scp {}@landman01.accre.vanderbilt.edu:{} {}\n'.format(self.vunetID, str(bval), str(session_input)+'/'))
            if self.args.dataset_name == "HCPA" or self.args.dataset_name == "HCPD":
                bvec = Path('/nfs2/harmonization/raw/{}_PQ_bvecs/{}/{}_dwi.bvec'.format(self.args.dataset_name, sub, sub))
            script.write('scp {}@landman01.accre.vanderbilt.edu:{} {}\n'.format(self.vunetID, str(bvec), str(session_input)+'/'))
                                #now, create the singularity command
            script.write('singularity run -e -B {}:/INPUTS -B {}:/OUTPUTS -B {}:/APPS/freesurfer/license.txt {} {} --topup_first_b0s_only {}\n'.format(str(session_input),
            str(session_output), str(self.freesurfer_license_path), str(self.simg), PEaxis, opts))
                                #now delete all the input files
            script.write('echo Done runnning PreQual. Now removing inputs...\n')
            script.write('rm -r {}\n'.format(str(session_input)+'/*'))
                                #copy over the output files to /nfs2
            script.write('echo Now copying outputs to nfs2...\n')
            script.write('scp -r {} {}@landman01.accre.vanderbilt.edu:{}\n'.format(str(session_output)+'/*', self.vunetID, str(PQdir)+'/'))
                                #delete the iutput files
            script.write('echo Now removing outputs from scratch...\n')
            script.write('rm -r {}\n'.format(str(session_output)+'/*',))

        #considerations:
            #PE dir
            #HCPA (options + bvecs) / additional options

    def write_newVMAP_PreQual_script(self, b1000read, b2000read, rperead, session_input, session_output, i, b1000_name, b2000_name, rpe_name, b1000_dwi, b2000_dwi, rpe_dwi, b1000_bval, b2000_bval, rpe_bval, b1000_bvec, b2000_bvec, rpe_bvec, b1000_json, b2000_json, rpe_json, PQdir_targ, sub, ses):
        """
        Writes a script for running PreQual on the new VMAP data.
        """


        opts = ""


        script_f = self.script_dir/("{}.sh".format(i))
        with open(script_f, 'w') as script:

            csvtext1 = "{},+,{}".format(b1000_name, b1000read)
            csvtext2 = "{},+,{}".format(b2000_name, b2000read)
            csvtext3 = "{},-,{}".format(rpe_name, rperead)
            script.write('#!/bin/bash\n')
            script.write('echo Starting PreQual for {}_{}\n'.format(sub,ses))
            script.write('echo {} > {}\n'.format(csvtext1, str(session_input/("dtiQA_config.csv"))))
            script.write('echo {} >> {}\n'.format(csvtext2, str(session_input/("dtiQA_config.csv"))))
            script.write('echo {} >> {}\n'.format(csvtext3, str(session_input/("dtiQA_config.csv"))))
            script.write('echo Created dtiQA_config.csv file. Now copying over files from nfs2...\n')
            #now, copy over the files

            for file in [b1000_dwi, b2000_dwi, rpe_dwi, b1000_bval, b2000_bval, rpe_bval, b1000_bvec, b2000_bvec, rpe_bvec, b1000_json, b2000_json, rpe_json]:
                script.write('scp {}@landman01.accre.vanderbilt.edu:{} {}\n'.format(self.vunetID, str(file), str(session_input)+'/'))

            #now, create the singularity command
            script.write('singularity run -e -B {}:/INPUTS -B {}:/OUTPUTS -B {}:/APPS/freesurfer/license.txt {} {} --topup_first_b0s_only {}\n'.format(str(session_input),
            str(session_output), str(self.freesurfer_license_path), str(self.simg), 'j', opts))
                                #now delete all the input files
            script.write('echo Done runnning PreQual. Now removing inputs...\n')
            script.write('rm -r {}\n'.format(str(session_input)+'/*'))
                                #copy over the output files to /nfs2
            script.write('echo Now copying outputs to nfs2...\n')
            script.write('scp -r {} {}@landman01.accre.vanderbilt.edu:{}\n'.format(str(session_output)+'/*', self.vunetID, str(PQdir_targ)+'/'))
                                #delete the iutput files
            script.write('echo Now removing outputs from scratch...\n')
            script.write('rm -r {}\n'.format(str(session_output)+'/*'))

    def get_sub_ses_acq_run_t1(self, t1_path):
        pattern = r'(sub-\w+)(?:_(ses-\w+))?(?:_(acq-\w+))?(?:_(run-\d{1,2}))?_T1w'
        matches = re.findall(pattern, t1_path.name)
        sub, ses, acq, run = matches[0][0], matches[0][1], matches[0][2], matches[0][3]
        return sub, ses, acq, run

    def get_sub_ses_acq_run_dwi(self, dwi_path):
        pattern = r'(sub-\w+)(?:_(ses-\w+))?(?:_(acq-\w+))?(?:_(run-\d{1,2}))?_dwi'
        matches = re.findall(pattern, dwi_path.name)
        sub, ses, acq, run = matches[0][0], matches[0][1], matches[0][2], matches[0][3]
        return sub, ses, acq, run

    def write_TICV_script(self, i, t1, eb1, eb2, eb3, eb4, eb5, ses_inputs, ses_outputs, TICV_dir, ses_pre, ses_post, ses_dl):

        script_f = self.script_dir/("{}.sh".format(i))
        with open(script_f, 'w') as script:
            script.write("#!/bin/bash\n")
            script.write("echo Making subdirectories for pre, post, and dl...\n")
            script.write("mkdir -p {}\n".format(ses_pre))
            script.write("mkdir -p {}\n".format(ses_post))
            script.write("mkdir -p {}\n".format(ses_dl))
            script.write("echo Done making directories. Copying input files from /nfs2 to /scratch...\n")
            script.write("scp {}@landman01.accre.vanderbilt.edu:{} {}/\n".format(self.vunetID, t1, ses_inputs))
            script.write("echo Done copying input files. Now running SLANT-TICV singularity...\n")

            script.write("singularity exec -B {} -B {} -B {} -B {} -B {} -e {} /opt/slant/run.sh\n".format(eb1, eb2, eb3, eb4, eb5, self.simg))
            script.write("echo Done running singularity. Now running deleting input files...\n")
            script.write("rm {}/*\n".format(ses_inputs))
            script.write("echo Done deleting input files. Now copying outputs to /nfs2...\n")
            script.write("scp -r {}/post {}@landman01.accre.vanderbilt.edu:{}/\n".format(ses_outputs, self.vunetID, TICV_dir))
            script.write("echo Done copying outputs. Now deleting outputs on /scratch...\n")
            script.write("rm -r {}/*\n".format(ses_outputs))

    def check_PreQual(self, pqdir, sub, ses):

        preproc_dir = pqdir/("PREPROCESSED")
        dwi = preproc_dir/("dwmri.nii.gz")
        bval = preproc_dir/("dwmri.bval")
        bvec = preproc_dir/("dwmri.bvec")

        #also check to see if the PDF is there
        pdf_dir = pqdir/("PDF")
        pdf = pdf_dir/("dtiQA.pdf")

        hasPQ = True
        hasPDF = True
        if not dwi.exists() or not bval.exists() or not bvec.exists():
            hasPQ = False
        if not pdf.exists():
            hasPDF = False

        #also check to see if there is a T1 for which to run synb0
        anat_dir = self.root_dataset_path/(sub)/(ses)/("anat")
        t1 = self.get_t1(anat_dir, sub, ses)
        hasT1 = True
        if t1 is None:
            hasT1 = False

        if hasPQ and not hasPDF:
            hasPQ = 'Missing PDF'
        
        return (hasPQ, hasPDF, hasT1)

    def get_dwi_files(self, path, subname, sesname, acq, run):
        if sesname != '':
            sesname = '_'+sesname
        if acq != '':
            acq = '_'+acq
        if run != '':
            run = '_'+run
        if run == '':
            dwi = path/('{}{}_dwi.nii.gz'.format(subname, sesname, acq)) if acq == '' else path/('{}{}{}_dwi.nii.gz'.format(subname, sesname, acq))
            bval = path/('{}{}_dwi.bval'.format(subname, sesname, acq)) if acq == '' else path/('{}{}{}_dwi.bval'.format(subname, sesname, acq))
            bvec = path/('{}{}_dwi.bvec'.format(subname, sesname, acq)) if acq == '' else path/('{}{}{}_dwi.bvec'.format(subname, sesname, acq))
            json = path/('{}{}_dwi.json'.format(subname, sesname, acq)) if acq == '' else path/('{}{}{}_dwi.json'.format(subname, sesname, acq))
        else:
            dwi = path/('{}{}{}_dwi.nii.gz'.format(subname, sesname, run)) if acq == '' else path/('{}{}{}{}_dwi.nii.gz'.format(subname, sesname, acq, run))
            bval = path/('{}{}{}_dwi.bval'.format(subname, sesname, run)) if acq == '' else path/('{}{}{}{}_dwi.bval'.format(subname, sesname, acq, run))
            bvec = path/('{}{}{}_dwi.bvec'.format(subname, sesname, run)) if acq == '' else path/('{}{}{}{}_dwi.bvec'.format(subname, sesname, acq, run))
            json = path/('{}{}{}_dwi.json'.format(subname, sesname, run)) if acq == '' else path/('{}{}{}{}_dwi.json'.format(subname, sesname, acq, run))
        if not dwi.is_file() or not bval.is_file() or not bvec.is_file() or not json.is_file():
            if path.parent.parent.parent.name == "BIOCARD" and dwi.is_file() and bval.is_file() and bvec.is_file():
                return [dwi, bval, bvec, json]
            elif not dwi.is_file() or not bval.is_file() or not bvec.is_file():
                return False
        return [dwi, bval, bvec, json]

    def get_PE(self, jsonf):
        """
        Gets the PE direction from the json file. Returns the axis, the sign, and whether the direction is unknown.
        """
        if not jsonf.exists():
            return ('j', '+', True)

        with open(jsonf, 'r') as file:
            content = file.read()
            json_data = json.loads(content)
            try: #try to first find the phase encoding direction field
                PEdir = json_data['PhaseEncodingDirection']
                PEunknown=False
            except:
                try: #if that doesnt exist, try phase encoding axis
                    PEdir = json_data['PhaseEncodingAxis']
                    PEunknown = False
                except:
                    #just say j and will have to QA later
                    PEdir = 'j'
                    PEunknown = True
            #now get the sign
            if PEdir[0] == '-':
                PEaxis = PEdir[1]
                PEsign = '-'
            elif PEdir[0] == '+':
                PEaxis = PEdir[1]
                PEsign = '-'
            else: #if there is no direction, it is assumed to be positive
                PEaxis = PEdir[0]
                PEsign = '+'
        return (PEaxis, PEsign, PEunknown)

    def write_freesurfer_script(self, session_input, session_output, i, t1, fsdir_targ):
        """
        Writes a freesurfer script to the specified directory.
        """

        script_f = self.script_dir/("{}.sh".format(i))
        with open(script_f, 'w') as script:
            script.write("#!/bin/bash\n")
            script.write("echo Copying input files from /nfs2 to /scratch...\n")
            script.write("scp {}@landman01.accre.vanderbilt.edu:{} {}/t1.nii.gz\n".format(self.vunetID, t1, session_input))
            script.write("echo Done copying input files. Now running freesurfer singularity...\n")
            script.write("recon-all -i {}/t1.nii.gz -subjid freesurfer -sd {}/ -all\n".format(str(session_input), str(session_output)))
            script.write("echo Done running singularity. Now running deleting input files...\n")
            script.write("rm {}/*\n".format(session_input))
            script.write("echo Done deleting input files. Now copying outputs to /nfs2...\n")
            script.write("scp -r {}/* {}@landman01.accre.vanderbilt.edu:{}/\n".format(session_output, self.vunetID, fsdir_targ))
            script.write("echo Done copying outputs. Now deleting outputs on /scratch...\n")
            script.write("rm -r {}/*\n".format(session_output))

    def freesurfer_script_generate(self, tmp_input_dir, tmp_output_dir):
        """
        Generates scripts for freesurfer
        """

        print("Finding all T1s...")
        find_cmd = "find {} -mindepth 3 -maxdepth 4 \( -type f -o -type l \) -name '*T1w.nii.gz' | grep -v derivatives".format(str(self.root_dataset_path))
        result = subprocess.run(find_cmd, shell=True, capture_output=True, text=True).stdout
        files = result.strip().splitlines()

        missing_data = pd.DataFrame(columns=['sub', 'ses', 'acq', 'run', 'missing'])

        i=0
        for t1_path in tqdm(files):
            t1 = Path(t1_path)
            #get the name of the directory, the run, and the sub,ses names
            sub, ses, acq, run = self.get_sub_ses_acq_run_t1(t1)

            #first, check to make sure that the data does not already exist
            fs_dir = self.dataset_derivs/(sub)/(ses)/("freesurfer{}{}".format(acq, run))
            recon_log = fs_dir/("freesurfer")/("scripts")/("recon-all.log")
            #first, check if directory and file exist
            if fs_dir.exists() and recon_log.exists():
                #now check the contents of the reconlog
                isDone = True
                with open(recon_log, 'r') as log:
                    content = log.read()
                    if not "finished without error" in content:
                        #return True
                        isDone = True
                if isDone:
                    continue
            
            i+= 1

            #make sure the final destination output dir exists
            fsdir_targ = self.output_dir/(sub)/(ses)/("freesurfer{}{}".format(acq, run))
            if not fsdir_targ.exists():
                os.makedirs(fsdir_targ)

            #create the directories
            session_input = tmp_input_dir/(sub)/("{}{}{}".format(ses,acq,run))
            session_output = tmp_output_dir/(sub)/("{}{}{}".format(ses,acq,run))
            #make the directories if they do not exist
            for d in [session_input, session_output]:
                if not d.exists():
                    os.makedirs(d)
            
            #write the script
            self.write_freesurfer_script(session_input, session_output, i, t1, fsdir_targ)
        
        self.num_scripts = i

    def get_readout_time(self, dwi_json):
        """
        Given the path to a DWI json file, returns the readout time
        """
        if not dwi_json.exists():
            return None
        with open(dwi_json, 'r') as file:
            content = file.read()
            json_data = json.loads(content)
            try:
                readout = json_data['EstimatedTotalReadoutTime']
            except:
                readout = None
        return readout

    def new_VMAP_PQ_script_generate(self, tmp_input_dir, tmp_output_dir):
        """
        Specific PreQual setup for the VMAP_2.0 (VMAP_JEFFERSONVMAP) and VMAP_TAP (VMAP_JEFFERSONTAP) datasets.

        No synb0, uses the b1000, b2000, and b1000 RPE scans for Topup

        All scans will be combined into a single PreQual directory called "PreQual"
        """

        #find all the sessions with DWI
        find_cmd = "find {} -mindepth 3 -maxdepth 3 -type d -name dwi | grep -v derivatives".format(str(self.root_dataset_path))
        result = subprocess.run(find_cmd, shell=True, capture_output=True, text=True).stdout
        dwi_dirs = result.strip().splitlines()

        missing_data = pd.DataFrame(columns=['sub', 'ses', 'acq', 'run', 'missing'])
        i=0
        for dwi_dir_p in tqdm(dwi_dirs):
            dwi_dir = Path(dwi_dir_p)

            #get the sub and ses
            sub = dwi_dir.parent.parent.name
            ses = dwi_dir.parent.name

            #check to make sure that PreQual has not already been run
            PQdir = self.dataset_derivs/(sub)/(ses)/("PreQual")
            (hasPQ, hasPDF, hasT1) = self.check_PreQual(PQdir, sub, ses)
            if hasPQ and hasPDF:
                continue
            elif hasPQ and not hasPDF:
                print("echo Missing PDF for {}_{}".format(sub, ses))
                continue

            #get the dwi files, which should have specific names:
                #b1000 = sub-XXX_ses-YYY_acq-b1000_dwi.nii.gz
                #b2000 = sub-XXX_ses-YYY_acq-b2000_dwi.nii.gz
                #b1000_RPE = sub-XXX_ses-YYY_acq-ReversePE_dwi.nii.gz
            
            b1000_files = self.get_dwi_files(dwi_dir, sub, ses, 'acq-b1000', '')
            b2000_files = self.get_dwi_files(dwi_dir, sub, ses, 'acq-b2000', '')
            rpe_files = self.get_dwi_files(dwi_dir, sub, ses, 'acq-ReversePE', '')

            #check to make sure they are all there
            if not b1000_files or not b2000_files or not rpe_files:
                #print("echo Missing DWI files for {}_{}".format(sub, ses))
                missings = ''
                for type,type_name in [(b1000_files,'b1000'), (b2000_files,'b2000'), (rpe_files,'b1000_RPE')]:
                    if not type:
                        missings += type_name + ' '
                row = {'sub': sub, 'ses': ses, 'acq': '', 'run': '', 'missing': '{}'.format(missings)}
                missing_data = pd.concat([missing_data, pd.Series(row).to_frame().T], ignore_index=True)
                continue
            
            #increment the count since all inputs are present and we can run
            i += 1

            #[dwi, bval, bvec, jsonf] = dwi_files
            [b1000_dwi, b1000_bval, b1000_bvec, b1000_json] = b1000_files
            [b2000_dwi, b2000_bval, b2000_bvec, b2000_json] = b2000_files
            [rpe_dwi, rpe_bval, rpe_bvec, rpe_json] = rpe_files

            #get the names of the files
            b1000_name = b1000_dwi.name.split('.nii.gz')[0]
            b2000_name = b2000_dwi.name.split('.nii.gz')[0]
            rpe_name = rpe_dwi.name.split('.nii.gz')[0]

            #PE direction is j, so we dont need to keep track of that

            #create the directories
            session_input = tmp_input_dir/(sub)/(ses)
            session_output = tmp_output_dir/(sub)/(ses)
            #make the directories if they do not exist
            for d in [session_input, session_output]:
                if not d.exists():
                    os.makedirs(d)

            PQdir_targ = self.output_dir/(sub)/(ses)/("PreQual")
            if not PQdir_targ.exists():
                os.makedirs(PQdir_targ)

            #get the readout times
                #if they are not in the json, we will have to assume 0.05
            b1000read = self.get_readout_time(b1000_json)
            b2000read = self.get_readout_time(b2000_json)
            rperead = self.get_readout_time(rpe_json)
            #if any of the readouts are None, assume 0.05 and warn
            if b1000read is None:
                print("echo Missing readout time for {}_{} b1000".format(sub, ses))
                b1000read = b2000read = rperead = 0.05
            if b2000read is None:
                print("echo Missing readout time for {}_{} b2000".format(sub, ses))
                b1000read = b2000read = rperead = 0.05
            if rperead is None:
                print("echo Missing readout time for {}_{} b1000_RPE".format(sub, ses))
                b1000read = b2000read = rperead = 0.05

            #aggregate the dwi scans and names into tuples
            #dwi_files = [(b1000_dwi, b1000_bval, b1000_bvec, b1000_json), (b2000_dwi, b2000_bval, b2000_bvec, b2000_json), (rpe_dwi, rpe_bval, rpe_bvec, rpe_json)]
            #check if the run2 is defined
                #in the script, make sure to change the dtiQA_config,csv accordingly

            #write the script
            self.write_newVMAP_PreQual_script(b1000read, b2000read, rperead, session_input, session_output, i, b1000_name, b2000_name, rpe_name, b1000_dwi, b2000_dwi, rpe_dwi, b1000_bval, b2000_bval, rpe_bval, b1000_bvec, b2000_bvec, rpe_bvec, b1000_json, b2000_json, rpe_json, PQdir_targ, sub, ses)
        self.num_scripts = i
        missing_data.to_csv('{}_{}_missing_data.csv'.format(self.root_dataset_path.name, self.pipeline))
        print("Wrote missing data to {}_{}_missing_data.csv".format(self.root_dataset_path.name, self.pipeline))

    def PreQual_script_generate(self, tmp_input_dir, tmp_output_dir):
        """
        Writes PreQual scripts to the specified directory.
        """

        #special case for VMAP_2.0 and VMAP_TAP, for which we combine multiple scans (b1000, b2000, and b1000 RPE scans for topup, no synb0)
        if self.args.dataset_name == "VMAP_TAP" or self.args.dataset_name == "VMAP_2.0":
            self.new_VMAP_PQ_script_generate(tmp_input_dir, tmp_output_dir)
            return

        #search for the dwi nifti
        print("Finding all DWIs...")
        find_cmd = "find {} -mindepth 3 -maxdepth 4 \( -type f -o -type l \) -name '*dwi.nii.gz' | grep -v derivatives".format(str(self.root_dataset_path))
        result = subprocess.run(find_cmd, shell=True, capture_output=True, text=True).stdout
        files = result.strip().splitlines()


        missing_data = pd.DataFrame(columns=['sub', 'ses', 'acq', 'run', 'missing'])

        i=0
        for dwi_string in tqdm(files):
            #get the path and file name
            dwi_p = Path(dwi_string)

            sub, ses, acq, run = self.get_sub_ses_acq_run_dwi(dwi_p)

            #check if it already exists in BIDS
            PQdir = self.dataset_derivs/(sub)/(ses)/("PreQual{}{}".format(acq, run))
            (hasPQ, hasPDF, hasT1) = self.check_PreQual(PQdir, sub, ses)

            if hasPQ and hasPDF:
                continue
            elif hasPQ and not hasPDF:
                print("echo Missing PDF for {}_{}{}{}".format(sub, ses, acq, run))
                continue
            
            #get the dwi files
            sespath = self.root_dataset_path/(sub)/(ses)
            #print(sespath)
            dwi_files = self.get_dwi_files(sespath/("dwi"), sub, ses, acq, run)
            if not dwi_files:
                print("echo No dwi:", sub, ses, acq, run)
                continue
            [dwi, bval, bvec, jsonf] = dwi_files
            assert str(dwi) == str(dwi_p), "Error: dwi file paths do not match --> {} vs {}".format(str(dwi), str(dwi_p))

            #get the t1 that will be used    
            t1 = self.get_t1(sespath/("anat"), sub, ses)
            if not t1:
                #concat to the missing data df
                row = {'sub': sub, 'ses': ses, 'acq': acq, 'run': run, 'missing': 'T1'}
                missing_data = pd.concat([missing_data, pd.Series(row).to_frame().T], ignore_index=True)
                continue
            
            #increment the count since all inputs are present and we can run
            i += 1

            #get the PE direction
            (PEaxis, PEsign, PEunknown) = self.get_PE(jsonf)

            #create the directories
            session_input = tmp_input_dir/(sub)/("{}{}{}".format(ses,acq,run))
            session_output = tmp_output_dir/(sub)/("{}{}{}".format(ses,acq,run))
            #make the directories if they do not exist
            for d in [session_input, session_output]:
                if not d.exists():
                    os.makedirs(d)

            PQdir_targ = self.output_dir/(sub)/(ses)/("PreQual{}{}".format(acq, run))
            if not PQdir_targ.exists():
                os.makedirs(PQdir_targ)

            #write all the commands to the scripts
            dwiname_match=re.match("^(.*dwi)\.nii\.gz$", dwi.name)
            dwigrp = dwiname_match.groups()
            dwiname = dwigrp[0]

            self.write_PreQual_script(session_input, session_output, i, dwiname, PEaxis, PEsign, PEunknown, t1, dwi, bval, bvec, jsonf, PQdir_targ, sub, ses)
        self.num_scripts = i
        missing_data.to_csv('{}_{}_missing_data.csv'.format(self.root_dataset_path.name, self.pipeline))
        print("Wrote missing data to {}_{}_missing_data.csv".format(self.root_dataset_path.name, self.pipeline))

    def SLANT_TICV_script_generate(self, tmp_input_dir, tmp_output_dir):
        """
        Writes a SLANT-TICV scripts to the specified directories.
        """
        
        print("Finding all T1s...")
        find_cmd = "find {} -mindepth 3 -maxdepth 4 \( -type f -o -type l \) -name '*T1w.nii.gz' | grep -v derivatives".format(str(self.root_dataset_path))
        result = subprocess.run(find_cmd, shell=True, capture_output=True, text=True).stdout
        files = result.strip().splitlines()

        missing_data = pd.DataFrame(columns=['sub', 'ses', 'acq', 'run', 'missing'])

        i=0
        for t1_path in tqdm(files):
            t1 = Path(t1_path)
            #get the name of the directory, the run, and the sub,ses names
            sub, ses, acq, run = self.get_sub_ses_acq_run_t1(t1)

            #first, check to make sure that the data does not already exist
            TICV_dir = self.dataset_derivs/(sub)/(ses)/("SLANT-TICVv1.2{}{}".format(acq, run))
            find_seg = "find {} -mindepth 3 -maxdepth 3 -type f -name '*seg.nii.gz' -o -type l -name '*seg.nii.gz'".format(TICV_dir)
            seg_res = subprocess.run(find_seg, shell=True, capture_output=True, text=True).stdout.strip().splitlines()
            if seg_res != []:
                #print(seg_res)
                continue

            #At this point, a t1 exists and the segmentation does not, so we should be able to run
            i+=1

            #define the session inputs, outputs directories
            session_input = tmp_input_dir/(sub)/("{}{}{}".format(ses,acq,run))
            session_output = tmp_output_dir/(sub)/("{}{}{}".format(ses,acq,run))
            session_post = session_output/("post")
            session_pre = session_output/("pre")
            session_dl = session_output/("dl")
            
            #make the directories if they do not exist
            for d in [session_input, session_post, session_pre, session_dl]:
                if not d.exists():
                    os.makedirs(d)

            #write the extra binds
            eb1 = "\"{}/\":/opt/slant/matlab/input_pre".format(session_input)
            eb2 = "\"{}/\":/opt/slant/matlab/input_post".format(session_input)
            eb3 = "\"{}/\":/opt/slant/matlab/output_pre".format(session_pre)
            eb4 = "\"{}/\":/opt/slant/dl/working_dir".format(session_dl)
            eb5 = "\"{}/\":/opt/slant/matlab/output_post".format(session_post)

            #get the WMAtlas directory in the final outputs if it does not exist
            if not TICV_dir.exists():
                os.mkdir(TICV_dir)
            final_output_dir = self.output_dir / (sub)/(ses)/("SLANT-TICVv1.2{}{}".format(acq, run))
            if not final_output_dir.exists():
                os.makedirs(final_output_dir)

            #now, write the script to the script directory
            self.write_TICV_script(i, t1, eb1, eb2, eb3, eb4, eb5, session_input, session_output, final_output_dir, session_pre, session_post, session_dl)

        self.num_scripts = i
        #print("Wrote {} scripts for {}".format(i, self.pipeline))
    
    def get_TICV_seg_file(self, t1, PQdir):
        """
        Returns the TICV segmentation file associated with the T1 file. Returns None if the file does not exist.
        """
        #get the acquisition and run name from the T1
        pattern = r'(sub-\w+)(?:_(ses-\w+))?(?:_(acq-\w+))?(?:_(run-\d{1,2}))?_T1w'
        matches = re.findall(pattern, t1.name)
        sub, ses, acq, run = matches[0]
        #now get the associated TICV file
        TICV_dir = PQdir.parent/("SLANT-TICVv1.2{}{}".format(acq,run))/("post")/("FinalResult")
        if not acq == '':
            acq = '_'+acq
        if not run == '':
            run = '_'+run
        if not ses == '':
            ses = '_'+ses
        TICV_seg = TICV_dir/("{}{}{}{}_T1w_seg.nii.gz".format(sub,ses,acq,run))
        #print(TICV_seg)
        if not TICV_seg.exists():
            return None
        return TICV_seg

    def get_UNest_seg_file(self, t1, PQdir):
        #get the acquisition and run name from the T1
        pattern = r'(sub-\w+)(?:_(ses-\w+))?(?:_(acq-\w+))?(?:_(run-\d{1,2}))?_T1w'
        matches = re.findall(pattern, t1.name)
        sub, ses, acq, run = matches[0]
        #now get the associated TICV file
        unest_dir = PQdir.parent/("UNest{}{}".format(acq,run))/("FinalResults")
        if not acq == '':
            acq = '_'+acq
        if not run == '':
            run = '_'+run
        if not ses == '':
            ses = '_'+ses
        unest_seg = unest_dir/("{}{}{}{}_T1w_seg_merged_braincolor.nii.gz".format(sub,ses,acq,run))
        return unest_seg

    def get_PQ_dwi_outputs(self, path):
        preproc_dir = path/("PREPROCESSED")
        dwi = preproc_dir/("dwmri.nii.gz")
        bval = preproc_dir/("dwmri.bval")
        bvec = preproc_dir/("dwmri.bvec")
        return [dwi, bval, bvec]

    def EVE3Registration_script_generate(self, tmp_input_dir, tmp_output_dir):
        """
        Writes EVE3Registration scripts to the specified directories.
        """
        
        if self.args.custom_atlas_input_dir == '':
            atlasdir = Path("/nobackup/p_masi/kimm58/projects/ADSP_Processing/EVE3AtlasInputs")
        else:
            atlasdir = Path(self.args.custom_atlas_input_dir)
        assert atlasdir.exists(), "Error: custom atlas input directory does not exist at {}".format(str(atlasdir))

        #get the extra binds for the WMAtlas
        template = atlasdir/("template.nii.gz")
        lut_file = atlasdir/("T1_TICV_seg.txt")
        eb1 = "-B {}:/INPUTS/template.nii.gz".format(str(template))
        eb2 = "-B {}:/INPUTS/T1_seg.txt".format(str(lut_file))

        #find command to search for PQ directories
        print("Finding all PreQual outputs...")
        find_cmd = "find {} -mindepth 2 -maxdepth 3 -type d -name 'PreQual*'".format(str(self.dataset_derivs))
        result = subprocess.run(find_cmd, shell=True, capture_output=True, text=True).stdout
        files = result.strip().splitlines()

        missing_data = pd.DataFrame(columns=['sub', 'ses', 'acq', 'run', 'missing'])

        i=0
        for PQdir_path in tqdm(files):

            PQdir = Path(PQdir_path)
            PQ_suffix = PQdir.name[7:]

            #if there is a PQ, there should be a dwi
            #get the dwi files
            
            #first, check to make sure that the data does not already exist
            WMAtlas_dir = PQdir.parent/("WMAtlasEVE3{}".format(PQ_suffix))
            if ((WMAtlas_dir/("dwmri%diffusionmetrics.csv")).exists() and (WMAtlas_dir/("dwmri%ANTS_t1tob0.txt")).exists() and
                    (WMAtlas_dir/("dwmri%0GenericAffine.mat")).exists() and (WMAtlas_dir/("dwmri%1InverseWarp.nii.gz")).exists() and
                    (WMAtlas_dir/("dwmri%Atlas_JHU_MNI_SS_WMPM_Type-III.nii.gz")).exists()):
                continue

            #get the subject and session
            if PQdir.parent.parent.name == "derivatives":
                sub = PQdir.parent.name
                ses = ''
            else:
                sub = PQdir.parent.parent.name
                ses = PQdir.parent.name
            
            #and acq and run
            pattern = r'PreQual(?:(acq-\w+))?(?:(run-\d{1,2}))?'
            matches = re.findall(pattern, PQdir.name)
            acq, run = matches[0][0], matches[0][1]
            if 'run' in PQdir.name and acq != '': #I cant figure out the regex match for this, so I will just do it manually (I think there is nested ?? in the regex)
                run = 'run' + PQdir.name.split('run')[1]
                acq = PQ_suffix.split('run')[0]

            #check to make sure that the PreQual outputs exist
            PQdwi_files = self.get_PQ_dwi_outputs(PQdir)
            dwi_pq_exist=True
            for file in PQdwi_files:
                if not file.exists():
                    dwi_pq_exist = False
                    break
            if not dwi_pq_exist:
                row = {'sub':sub, 'ses':ses, 'acq':acq, 'run':run, 'missing':'PreQual'}
                missing_data = pd.concat([missing_data, pd.Series(row).to_frame().T], ignore_index=True)
                continue


            #check to see if a T1 exists
            if ses == '':
                anat_dir = self.root_dataset_path/(sub)/('anat')
            else:
                anat_dir = self.root_dataset_path/(sub)/(ses)/('anat')
            t1 = self.get_t1(anat_dir, sub, ses) 
            if t1 == None:
                row = {'sub':sub, 'ses':ses, 'acq':acq, 'run':run, 'missing':'T1'}
                missing_data = pd.concat([missing_data, pd.Series(row).to_frame().T], ignore_index=True)
                continue

            #get the TICV segmentation for that T1
            if not self.args.use_unest_seg:
                seg = self.get_TICV_seg_file(t1, PQdir)
            else:
                seg = self.get_UNest_seg_file(t1, PQdir)
            if not seg.exists():
                row = {'sub':sub, 'ses':ses, 'acq':acq, 'run':run, 'missing':'TICV' if not self.args.use_unest_seg else 'UNest'}
                # print(t1.name)
                # print(str(seg))
                missing_data = pd.concat([missing_data, pd.Series(row).to_frame().T], ignore_index=True)
                continue

            #increment the count since all inputs are present and we can run
            i += 1

            #define the session inputs, outputs directories
            session_input = tmp_input_dir/(sub)/("{}{}{}".format(ses,acq,run))
            session_output = tmp_output_dir/(sub)/("{}{}{}".format(ses,acq,run))
            #make the directories if they do not exist
            for d in [session_input, session_output]:
                if not d.exists():
                    os.makedirs(d)
            
            #make the output directory if it does not exist
            WMAtlas_target = self.output_dir/(sub)/(ses)/("WMAtlasEVE3{}".format(PQ_suffix))
            if not WMAtlas_target.exists():
                os.makedirs(WMAtlas_target)
            
            #write the script
            self.write_EVE3Registration_script(i, t1, PQdir, seg, atlasdir, session_input, session_output, WMAtlas_target)

        self.num_scripts = i
        print("Wrote {} scripts for {}".format(i, self.pipeline))
        #save the missing data
        missing_data_file = "{}_{}_missing_data.csv".format(self.args.dataset_name, self.pipeline)
        missing_data.to_csv(missing_data_file, index=False)
        print("Saved missing data to {}".format(missing_data_file))

    def write_EVE3Registration_script(self, i, t1, PQdir, seg, atlasdir, session_input, session_output, WMAtlas_dir):
        """
        Writes a EVE3Registration script to the specified directory.
        """

        script_f = self.script_dir/("{}.sh".format(i))
        with open(script_f, 'w') as script:
            script.write("#!/bin/bash\n")
            script.write("echo Copying input files from /nfs2 to /scratch...\n")
            script.write("scp {}@landman01.accre.vanderbilt.edu:{} {}/t1.nii.gz\n".format(self.vunetID, t1, session_input))
            script.write("scp {}@landman01.accre.vanderbilt.edu:{}/PREPROCESSED/* {}/\n".format(self.vunetID, PQdir, session_input))
            script.write("scp {}@landman01.accre.vanderbilt.edu:{} {}/T1_seg.nii.gz\n".format(self.vunetID, seg, session_input))
            #also copy the atlas inputs
            script.write("scp {}@landman01.accre.vanderbilt.edu:{}/* {}/\n".format(self.vunetID, atlasdir, session_input))
            script.write("echo Done copying input files. Now running EVE3Registration singularity...\n")
            script.write("singularity run -e -B {}:/INPUTS -B {}:/OUTPUTS {}\n".format(session_input, session_output, self.simg))
            script.write("echo Done running singularity. Now running deleting input files...\n")
            script.write("rm {}/*\n".format(session_input))
            script.write("echo Done deleting input files. Now copying outputs to /nfs2...\n")
            script.write("scp -r {}/* {}@landman01.accre.vanderbilt.edu:{}/\n".format(session_output, self.vunetID, WMAtlas_dir))
            script.write("echo Done copying outputs. Now deleting outputs on /scratch...\n")
            script.write("rm -r {}/*\n".format(session_output))

    def MNI152Registration_script_generate(self, tmp_input_dir, tmp_output_dir):
        """
        Writes MNI152Registration scripts to the specified directories.
        """
        
        if self.args.custom_atlas_input_dir == '':
            atlasdir = Path("/nobackup/p_masi/kimm58/projects/ADSP_Processing/MNIWMAtlasInputs/")
        else:
            atlasdir = Path(self.args.custom_atlas_input_dir)
        assert atlasdir.exists(), "Error: custom atlas input directory does not exist at {}".format(str(atlasdir))

        #get the extra binds for the WMAtlas
        template = atlasdir/("template.nii.gz")
        lut_file = atlasdir/("T1_TICV_seg.txt")
        eb1 = "-B {}:/INPUTS/template.nii.gz".format(str(template))
        eb2 = "-B {}:/INPUTS/T1_seg.txt".format(str(lut_file))

        #find command to search for PQ directories
        print("Finding all PreQual outputs...")
        find_cmd = "find {} -mindepth 2 -maxdepth 3 -type d -name 'PreQual*'".format(str(self.dataset_derivs))
        result = subprocess.run(find_cmd, shell=True, capture_output=True, text=True).stdout
        files = result.strip().splitlines()

        missing_data = pd.DataFrame(columns=['sub', 'ses', 'acq', 'run', 'missing'])

        i=0
        for PQdir_path in tqdm(files):

            PQdir = Path(PQdir_path)
            PQ_suffix = PQdir.name[7:]

            #if there is a PQ, there should be a dwi
            #get the dwi files
            
            #first, check to make sure that the data does not already exist
            WMAtlas_dir = PQdir.parent/("WMAtlas{}".format(PQ_suffix))
            if ((WMAtlas_dir/("dwmri%diffusionmetrics.csv")).exists() and (WMAtlas_dir/("dwmri%ANTS_t1tob0.txt")).exists() and
                    (WMAtlas_dir/("dwmri%0GenericAffine.mat")).exists() and (WMAtlas_dir/("dwmri%1InverseWarp.nii.gz")).exists()):
                continue

            #get the subject and session
            if PQdir.parent.parent.name == "derivatives":
                sub = PQdir.parent.name
                ses = ''
            else:
                sub = PQdir.parent.parent.name
                ses = PQdir.parent.name
            
            #and acq and run
            pattern = r'PreQual(?:(acq-\w+))?(?:(run-\d{1,2}))?'
            matches = re.findall(pattern, PQdir.name)
            acq, run = matches[0][0], matches[0][1]
            if 'run' in PQdir.name and acq != '': #I cant figure out the regex match for this, so I will just do it manually (I think there is nested ?? in the regex)
                run = 'run' + PQdir.name.split('run')[1]
                acq = PQ_suffix.split('run')[0]

            #check to make sure that the PreQual outputs exist
            PQdwi_files = self.get_PQ_dwi_outputs(PQdir)
            dwi_pq_exist=True
            for file in PQdwi_files:
                if not file.exists():
                    dwi_pq_exist = False
                    break
            if not dwi_pq_exist:
                row = {'sub':sub, 'ses':ses, 'acq':acq, 'run':run, 'missing':'PreQual'}
                missing_data = pd.concat([missing_data, pd.Series(row).to_frame().T], ignore_index=True)
                continue


            #check to see if a T1 exists
            if ses == '':
                anat_dir = self.root_dataset_path/(sub)/('anat')
            else:
                anat_dir = self.root_dataset_path/(sub)/(ses)/('anat')
            t1 = self.get_t1(anat_dir, sub, ses) 
            if t1 == None:
                row = {'sub':sub, 'ses':ses, 'acq':acq, 'run':run, 'missing':'T1'}
                missing_data = pd.concat([missing_data, pd.Series(row).to_frame().T], ignore_index=True)
                continue

            #get the TICV segmentation for that T1
            if not self.args.use_unest_seg:
                seg = self.get_TICV_seg_file(t1, PQdir)
            else:
                seg = self.get_UNest_seg_file(t1, PQdir)
            if not seg.exists():
                row = {'sub':sub, 'ses':ses, 'acq':acq, 'run':run, 'missing':'TICV' if not self.args.use_unest_seg else 'UNest'}
                # print(t1.name)
                # print(str(seg))
                missing_data = pd.concat([missing_data, pd.Series(row).to_frame().T], ignore_index=True)
                continue

            #increment the count since all inputs are present and we can run
            i += 1

            #define the session inputs, outputs directories
            session_input = tmp_input_dir/(sub)/("{}{}{}".format(ses,acq,run))
            session_output = tmp_output_dir/(sub)/("{}{}{}".format(ses,acq,run))
            #make the directories if they do not exist
            for d in [session_input, session_output]:
                if not d.exists():
                    os.makedirs(d)
            
            #make the output directory if it does not exist
            WMAtlas_target = self.output_dir/(sub)/(ses)/("WMAtlas{}".format(PQ_suffix))
            if not WMAtlas_target.exists():
                os.makedirs(WMAtlas_target)
            
            #write the script
            self.write_MNI152Registration_script(i, t1, PQdir, seg, atlasdir, session_input, session_output, WMAtlas_target)
        self.num_scripts = i
        print("Wrote {} scripts for {}".format(i, self.pipeline))
        #save the missing data
        missing_data_file = "{}_{}_missing_data.csv".format(self.args.dataset_name, self.pipeline)
        missing_data.to_csv(missing_data_file, index=False)
        print("Saved missing data to {}".format(missing_data_file))

    def write_MNI152Registration_script(self, i, t1, PQdir, seg, atlasdir, session_input, session_output, WMAtlas_dir):
        """
        Writes a MNI152Registration script to the specified directory.
        """

        script_f = self.script_dir/("{}.sh".format(i))
        with open(script_f, 'w') as script:
            script.write("#!/bin/bash\n")
            script.write("echo Copying input files from /nfs2 to /scratch...\n")
            script.write("scp {}@landman01.accre.vanderbilt.edu:{} {}/t1.nii.gz\n".format(self.vunetID, t1, session_input))
            script.write("scp {}@landman01.accre.vanderbilt.edu:{}/PREPROCESSED/* {}/\n".format(self.vunetID, PQdir, session_input))
            script.write("scp {}@landman01.accre.vanderbilt.edu:{} {}/T1_seg.nii.gz\n".format(self.vunetID, seg, session_input))
            #also copy the atlas inputs
            script.write("scp {}@landman01.accre.vanderbilt.edu:{}/* {}/\n".format(self.vunetID, atlasdir, session_input))
            script.write("echo Done copying input files. Now running MNI152Registration singularity...\n")
            script.write("singularity run -e -B {}:/INPUTS -B {}:/OUTPUTS {}\n".format(session_input, session_output, self.simg))
            script.write("echo Done running singularity. Now running deleting input files...\n")
            script.write("rm {}/*\n".format(session_input))
            script.write("echo Done deleting input files. Now copying outputs to /nfs2...\n")
            script.write("scp -r {}/* {}@landman01.accre.vanderbilt.edu:{}/\n".format(session_output, self.vunetID, WMAtlas_dir))
            script.write("echo Done copying outputs. Now deleting outputs on /scratch...\n")
            script.write("rm -r {}/*\n".format(session_output))

    def UNest_script_generate(self, tmp_input_dir, tmp_output_dir):
        """
        Creates script files for running UNest
        """

        def UNest_asserts(root_temp, root_working):
            #check to make sure that the temp and working directories exist and we can write to them
            assert root_temp.exists(), "Error: Root temp directory {} does not exist".format(root_temp)
            assert root_working.exists(), "Error: Root working directory {} does not exist".format(root_working)
            assert os.access(root_temp, os.W_OK), "Error: Root temp directory {} is not writable".format(root_temp)
            assert os.access(root_working, os.W_OK), "Error: Root working directory {} is not writable".format(root_working)

        #assert that the working and temp directories exist
        root_temp = Path(self.args.temp_dir)
        root_working = Path(self.args.working_dir)
        UNest_asserts(root_temp, root_working)

        #find command to search for PQ directories
        print("Finding all T1s...")
        find_cmd = "find {} -mindepth 3 -maxdepth 4 \( -type f -o -type l \) -name '*T1w.nii.gz' | grep -v derivatives".format(str(self.root_dataset_path))
        result = subprocess.run(find_cmd, shell=True, capture_output=True, text=True).stdout
        files = result.strip().splitlines()

        missing_data = pd.DataFrame(columns=['sub', 'ses', 'acq', 'run', 'missing'])

        if self.args.skull_stripped:
            print("Skull stripped flag is on. Will look for skull stripped T1 for UNest.")

        i=0
        for t1_path in tqdm(files):
            
            t1 = Path(t1_path)

            #get the name of the directory, the run, and the sub,ses names
            sub, ses, acq, run = self.get_sub_ses_acq_run_t1(t1)

            #first, check to make sure that the data does not already exist
            if not ses is None:
                unest_dir = self.dataset_derivs/(sub)/(ses)/("UNest{}{}".format(acq, run))
            else:
                unest_dir = self.dataset_derivs/(sub)/("UNest{}{}".format(acq, run))
            seg_file = unest_dir/("FinalResults")/("{}".format(t1.name.replace('T1w', 'T1w_seg_merged_braincolor')))

            #first, check if directory and file exist
            if unest_dir.exists() and seg_file.exists():
                #print("Exists")
                continue
        
            #if the skull stripped flag is on, replace t1 with the skull stripped T1
            if self.args.skull_stripped:
                #cp ssT1.nii.gz to sub-{}_ses-{}_acq-{}_run-{}_T1w.nii.gz
                synthstrip_dir = unest_dir.parent/("T1_synthstrip{}{}".format(acq, run))
                ssT1 = synthstrip_dir/("{}".format(t1.name.replace('T1w', 'synthstrip_T1w')))
                if not ssT1.exists():
                    print(ssT1)
                    row = {'sub':sub, 'ses':ses, 'acq':acq, 'run':run, 'missing':'ssT1'}
                    missing_data = pd.concat([missing_data, pd.Series(row).to_frame().T], ignore_index=True)
                    continue

            #At this point, a t1 exists and the segmentation does not, so we should be able to run
            i+=1

            #define the session inputs, outputs directories
            session_input = tmp_input_dir/(sub)/("{}{}{}".format(ses,acq,run))
            session_output = tmp_output_dir/(sub)/("{}{}{}".format(ses,acq,run))
            #and the temp and working directories
            session_temp = root_temp/(sub)/("{}{}{}".format(ses,acq,run))
            session_work = root_working/(sub)/("{}{}{}".format(ses,acq,run))
            
            #make the directories if they do not exist
            for d in [session_input, session_output, session_temp, session_work]:
                if not d.exists():
                    os.makedirs(d)
            
            #create the output directory if it does not exist
            unest_target = self.output_dir/(sub)/(ses)/("UNest{}{}".format(acq, run))
            if not unest_target.exists():
                os.makedirs(unest_target)
            
            #write the script
            self.write_UNest_script(i, t1, ssT1, session_input, session_output, session_temp, session_work, unest_dir)
        self.num_scripts = i
        print("Wrote {} scripts for {}".format(i, self.pipeline))
        #save the missing data
        missing_data.to_csv('{}_{}_missing_data.csv'.format(self.root_dataset_path.name, self.pipeline))
        print("Wrote missing data to {}_{}_missing_data.csv".format(self.root_dataset_path.name, self.pipeline))

    def write_UNest_script(self, i, t1, ssT1, session_input, session_output, session_temp, session_work, unest_dir):
        """
        Writes a script for UNest to the script directory
        """
        
        script_f = self.script_dir/("{}.sh".format(i))
        with open(script_f, 'w') as script:
            script.write("#!/bin/bash\n")
            script.write("echo Copying input files from /nfs2 to /scratch...\n")
            if self.args.skull_stripped:
                script.write("scp kimm58@landman01.accre.vanderbilt.edu:{} {}/{}\n".format(ssT1, session_input, t1.name))
            else:
                script.write("scp kimm58@landman01.accre.vanderbilt.edu:{} {}/\n".format(t1, session_input))
            script.write("echo Done copying input files. Now running UNest...\n")
            if self.args.skull_stripped:
                script.write("singularity run -e --contain --home {} -B {}:/INPUTS -B {}:/WORKING_DIR -B {}:/OUTPUTS -B {}:/tmp {} --ticv\n".format(session_input, session_input, session_work, session_output, session_temp, self.simg))
            else:
                script.write("singularity run -e --contain --home {} -B {}:/INPUTS -B {}:/WORKING_DIR -B {}:/OUTPUTS -B {}:/tmp {} --ticv --w_skull\n".format(session_input, session_input, session_work, session_output, session_temp, self.simg))
            script.write("echo Done running UNest. Now running deleting input files...\n")
            script.write("rm {}/*\n".format(session_input))
            script.write("echo Done deleting input files. Now copying outputs to /nfs2...\n")
            script.write("scp -r {}/* kimm58@landman01.accre.vanderbilt.edu:{}/\n".format(session_output, unest_dir))
            script.write("echo Done copying outputs. Now deleting outputs on /scratch...\n")
            script.write("rm -r {}/*\n".format(session_output))
            script.write("rm -r {}/*\n".format(session_work))
            script.write("rm -r {}/*\n".format(session_temp))

    def synthstrip_script_generate(self, tmp_input_dir, tmp_output_dir):
        """
        Generates the scripts for synthstrip in a target script directory
        """

        #define some stuff
        wrapper_script_path = Path("/nobackup/p_masi/Singularities/synthstrip_wrapper.py")
        wrapper_path = Path("/nobackup/p_masi/Singularities/synthstrip-singularity")

        #make sure that these exist and are readable and execuatable
        assert wrapper_script_path.exists(), "Error: Wrapper script {} does not exist".format(wrapper_script_path)
        assert wrapper_path.exists(), "Error: Wrapper {} does not exist".format(wrapper_path)
        assert os.access(wrapper_script_path, os.R_OK) and os.access(wrapper_script_path, os.X_OK), "Error: Wrapper script {} is either not readable or executable".format(wrapper_script_path)
        assert os.access(wrapper_path, os.R_OK) and os.access(wrapper_path, os.X_OK), "Error: Wrapper {} is either not readable or executable".format(wrapper_path)

        #find command to search for PQ directories
        print("Finding all T1s...")
        find_cmd = "find {} -mindepth 3 -maxdepth 4 \( -type l -o -type f \) -name '*T1w.nii.gz' | grep -v derivatives".format(str(self.root_dataset_path))
        result = subprocess.run(find_cmd, shell=True, capture_output=True, text=True).stdout
        files = result.strip().splitlines()

        missing_data = pd.DataFrame(columns=['sub', 'ses', 'acq', 'run', 'missing'])

        i=0
        for t1_path in tqdm(files):
            
            t1 = Path(t1_path)
            sub, ses, acq, run = self.get_sub_ses_acq_run_t1(t1)

            #first, check to make sure that the data does not already exist
            if ses:
                synthstrip_dir = self.dataset_derivs/(sub)/(ses)/("T1_synthstrip{}{}".format(acq, run))
            else:
                synthstrip_dir = self.dataset_derivs/(sub)/("T1_synthstrip{}{}".format(acq, run))
            ssT1_link = synthstrip_dir/("{}".format(t1.name.replace('T1w', 'synthstrip_T1w')))
            if ssT1_link.exists():
                continue

            #t1 exists and synthstrip does not, so we should be able to run
            i+=1

            #create the input and output directories
            session_input = tmp_input_dir/(sub)/("{}{}{}".format(ses,acq,run))
            session_output = tmp_output_dir/(sub)/("{}{}{}".format(ses,acq,run))
            if not session_input.exists():
                os.makedirs(session_input)
            if not session_output.exists():
                os.makedirs(session_output)

            #and the overall output directory as well
            synthstrip_target = self.output_dir/(sub)/(ses)/("T1_synthstrip{}{}".format(acq, run))
            if not synthstrip_target.exists():
                os.makedirs(synthstrip_target)
            
            #define the output mask and synthstrip T1
            ssT1 = session_output/("{}".format(t1.name.replace('T1w', 'synthstrip_T1w')))
            mask = session_output/("{}".format(t1.name.replace('T1w', 'synthstrip_T1w_mask')))
            input_t1 = session_input/("{}".format(t1.name))

            #write the script
            self.write_synthstrip_script(i, t1, ssT1, mask, session_input, session_output, synthstrip_target, wrapper_script_path, wrapper_path, input_t1)

        #print("Wrote {} scripts for {}".format(i, self.pipeline))
            
        self.num_scripts = i
        #print("Wrote {} scripts for {}".format(i, self.pipeline))

    def write_synthstrip_script(self, i, t1, ssT1, mask, session_input, session_output, synthstrip_target, wrapper_script_path, wrapper_path, input_t1):
        """
        Writes a script for synthstrip to the script directory
        """
        
        script_f = self.script_dir/("{}.sh".format(i))
        with open(script_f, 'w') as script:
            script.write("#!/bin/bash\n")
            script.write("echo Copying input files from /nfs2 to /scratch...\n")
            script.write("scp {}@landman01.accre.vanderbilt.edu:{} {}/\n".format(self.vunetID, t1, session_input))
            script.write("echo Done copying input files. Now running synthstrip...\n")
            script.write("python3 {} {} {} {} {}\n".format(str(wrapper_script_path), str(input_t1), str(ssT1), str(mask), str(wrapper_path)))
            script.write("echo Done running synthstrip. Now running deleting input files...\n")
            script.write("rm {}/*\n".format(session_input))
            script.write("echo Done deleting input files. Now linking outputs to BIDS...\n")
            #script.write("ln -s {} {}\n".format(ssT1, ssT1_link))
            script.write("scp -r {}/* {}@landman01.accre.vanderbilt.edu:{}/\n".format(session_output, self.vunetID, synthstrip_target))
            script.write("echo Done copying outputs. Now deleting outputs on /scratch...\n")
            script.write("rm -r {}/*\n".format(session_output))

    def check_tractseg_outputs(self, tsdir):
        """
        Checks to see if the tractseg outputs are present
        """

        #get the bundles and measures dirs
        bundles = tsdir/("bundles")
        measures = tsdir/("measures")

        if not bundles.exists() or not measures.exists():
            #print(tsdir, "False")
            return False

        #count the files in each one to make sure that there is the correct number of bundles and measures
        num_bundles = len(list(bundles.glob("*.tck")))
        num_measures = len(list(measures.glob("*.json")))

        if num_bundles != 72 or num_measures != 144:
            #print(tsdir, 'Missing')
            return 'Missing Files'
        return True

    def tractseg_script_generate(self, tmp_input_dir, tmp_output_dir):
        """
        Adapted from Zhiyuan Li's code

        Generates the scripts for TractSeg in a target script directory
        """

        #assert that the temp directory exists and is writable
        root_temp = Path(self.args.temp_dir)
        assert root_temp.exists() and os.access(root_temp, os.W_OK), "Error: Root temp directory {} does not exist or is not writable".format(root_temp)

        #get the accre home directory
        accre_home_directory = os.path.expanduser("~")

        #first, finds the PreQual directories
        print("Finding all PreQual outputs...")
        find_cmd = "find {} -mindepth 2 -maxdepth 3 -type d -name 'PreQual*'".format(str(self.dataset_derivs))
        results = subprocess.run(find_cmd, shell=True, capture_output=True, text=True).stdout.strip().splitlines()
        i=0
        missing_data = pd.DataFrame(columns=['sub', 'ses', 'acq', 'run', 'missing'])
        for PQdir_p in tqdm(results):
            PQdir = Path(PQdir_p)
            PQ_suffix = PQdir.name[7:]

            #get the subject and session
            if PQdir.parent.parent.name == "derivatives":
                sub = PQdir.parent.name
                ses = ''
            else:
                sub = PQdir.parent.parent.name
                ses = PQdir.parent.name
            
            #and acq and run
            pattern = r'PreQual(?:(acq-\w+))?(?:(run-\d{1,2}))?'
            matches = re.findall(pattern, PQdir.name)
            acq, run = matches[0][0], matches[0][1]
            if 'run' in PQdir.name and acq != '':
                run = 'run' + PQdir.name.split('run')[1]
                acq = PQ_suffix.split('run')[0]
            
            #check to make sure that the PreQual outputs exist
            PQdwi_files = self.get_PQ_dwi_outputs(PQdir)
            dwi_pq_exist=True
            for file in PQdwi_files:
                if not file.exists():
                    dwi_pq_exist = False
                    break
            if not dwi_pq_exist:
                row = {'sub':sub, 'ses':ses, 'acq':acq, 'run':run, 'missing':'PreQual'}
                missing_data = pd.concat([missing_data, pd.Series(row).to_frame().T], ignore_index=True)
                continue
            
            #check to see if the tractseg outputs exist
            if ses != '':
                tsdir = self.dataset_derivs/(sub)/(ses)/("Tractseg{}{}".format(acq, run))
            else:
                tsdir = self.dataset_derivs/(sub)/("Tractseg{}{}".format(acq, run))
            
            if self.check_tractseg_outputs(tsdir):
                continue

            #get the tensor maps to use for the mean/std bundle calculations
            if ses != '':
                eve3dir = self.dataset_derivs/(sub)/(ses)/("WMAtlasEVE3{}{}".format(acq, run))
            else:
                eve3dir = self.dataset_derivs/(sub)/("WMAtlasEVE3{}{}".format(acq, run))
            tensor_maps = [eve3dir/("dwmri%{}.nii.gz".format(m)) for m in ['fa', 'md', 'ad', 'rd']]
            using_PQ = False
            if not all([t.exists() for t in tensor_maps]):
                #print("Missing tensor maps for {}. Using prequal instead.".format(eve3dir))
                #tensor_maps = [PQdir/("SCALARS")/("dwmri_tensor_{}.nii.gz".format(m)) for m in ['fa', 'md', 'ad', 'rd']]
                #using_PQ = True
                row = {'sub':sub, 'ses':ses, 'acq':acq, 'run':run, 'missing':'tensor_maps'}
                missing_data = pd.concat([missing_data, pd.Series(row).to_frame().T], ignore_index=True)
                continue

            #so the PreQual are there and we do not have tractseg outputs, so we can run
            i+=1

            #define the session inputs, outputs directories and working/temp dir
            session_input = tmp_input_dir/(sub)/("{}{}{}".format(ses,acq,run))
            session_output = tmp_output_dir/(sub)/("{}{}{}".format(ses,acq,run))
            session_temp = root_temp/(sub)/("{}{}{}".format(ses,acq,run))
            for d in [session_input, session_output, session_temp]:
                if not d.exists():
                    os.makedirs(d)
            
            #create the output directory if it does not exist
            if ses != '':
                tractseg_target = self.output_dir/(sub)/(ses)/("Tractseg{}{}".format(acq, run))
            else:
                tractseg_target = self.output_dir/(sub)/("Tractseg{}{}".format(acq, run))
            
            if not tractseg_target.exists():
                os.makedirs(tractseg_target)

            #now write the script
            self.write_tractseg_script(i, PQdir, session_input, session_output, session_temp, tractseg_target, tensor_maps, accre_home_directory)
        self.num_scripts = i
        missing_data.to_csv('{}_{}_missing_data.csv'.format(self.root_dataset_path.name, self.pipeline))
        print("Wrote missing data to {}_{}_missing_data.csv".format(self.root_dataset_path.name, self.pipeline))

    def write_tractseg_script(self, i, PQdir, session_input, session_output, session_temp, tractseg_target, tensor_maps, accre_home_directory):
        """
        Writes a script for TractSeg to the script directory
        """

        #define the singularities
        ts_simg = self.simg[0]
        scilus_simg = self.simg[1]
        mrtrix_simg = self.simg[1]

        #define the tensor maps
        fa, md, ad, rd = tensor_maps
        strs = ['fa', 'md', 'ad', 'rd']

        script_f = self.script_dir/("{}.sh".format(i))
        with open(script_f, 'w') as script:
            script.write("#!/bin/bash\n\n")
            script.write("MINWAIT=0\n")
            script.write("MAXWAIT=60\n")
            script.write("sleep $((MINWAIT+RANDOM % (MAXWAIT-MINWAIT)))\n\n")

            # if using_PQ:
            #     script.write("echo *********************************************\n")
            #     script.write("echo WARNING: Using tensor maps from PreQual instead of explicit bval < 1500 extraction. \n")
            #     script.write("echo *********************************************\n")

            script.write("echo Copying input files from /nfs2 to /scratch...\n")
            script.write("scp {}@landman01.accre.vanderbilt.edu:{}/PREPROCESSED/* {}/\n".format(self.vunetID, PQdir, session_temp))
            for map,scalar in zip(tensor_maps, strs):
                script.write("scp {}@landman01.accre.vanderbilt.edu:{} {}/dwmri_tensor_{}.nii.gz\n".format(self.vunetID, map, session_temp, scalar))
            #script.write("scp {}@landman01.accre.vanderbilt.edu:{}/SCALARS/* {}/\n".format(self.vunetID, PQdir, session_temp))

            script.write("echo Done copying input files. Now resampling to 1mm iso...\n")
            script.write("time singularity run -B {}:{} {} mrgrid {}/dwmri.nii.gz regrid {}/dwmri_1mm_iso.nii.gz -voxel 1\n".format(session_temp,session_temp, mrtrix_simg, session_temp, session_temp))
            #same for the fa, md, ad, rd
            script.write("time singularity run -B {}:{} {} mrgrid {}/dwmri_tensor_fa.nii.gz regrid {}/dwmri_tensor_fa_1mm_iso.nii.gz -voxel 1\n".format(session_temp,session_temp, mrtrix_simg, session_temp, session_temp))
            script.write("time singularity run -B {}:{} {} mrgrid {}/dwmri_tensor_md.nii.gz regrid {}/dwmri_tensor_md_1mm_iso.nii.gz -voxel 1\n".format(session_temp,session_temp, mrtrix_simg, session_temp, session_temp))
            script.write("time singularity run -B {}:{} {} mrgrid {}/dwmri_tensor_ad.nii.gz regrid {}/dwmri_tensor_ad_1mm_iso.nii.gz -voxel 1\n".format(session_temp,session_temp, mrtrix_simg, session_temp, session_temp))
            script.write("time singularity run -B {}:{} {} mrgrid {}/dwmri_tensor_rd.nii.gz regrid {}/dwmri_tensor_rd_1mm_iso.nii.gz -voxel 1\n".format(session_temp,session_temp, mrtrix_simg, session_temp, session_temp))

            script.write("echo Done resampling to 1mm iso. Now running TractSeg...\n")
            script.write('echo "..............................................................................."\n')
            script.write("echo Loading FSL...\n")

            script.write("export FSL_DIR=/accre/arch/easybuild/software/MPI/GCC/6.4.0-2.28/OpenMPI/2.1.1/FSL/5.0.10/fsl\n")
            script.write("source setup_accre_runtime_dir\n")

            script.write("time singularity run -B {}:{} -B {}:{} {} TractSeg -i {}/dwmri_1mm_iso.nii.gz --raw_diffusion_input -o {}/tractseg --bvals {}/dwmri.bval --bvecs {}/dwmri.bvec\n".format(accre_home_directory, accre_home_directory, session_temp,session_temp, ts_simg, session_temp, session_temp, session_temp, session_temp))
            script.write('if [[ -f "{}/tractseg/peaks.nii.gz" ]]; then echo "Successfully created peaks.nii.gz for {}"; error_flag=0; else echo "Improper bvalue/bvector distribution for {}"; error_flag=1; fi\n'.format(session_temp, session_temp, session_temp))
            script.write('if [[ $error_flag -eq 1 ]]; then echo "Improper bvalue/bvector distribution for {} >> {}/report_bad_bvector.txt"; fi\n\n'.format(session_temp, session_temp))

            #copy outputs to /nfs2 and delete inputs IF the flag is raised
            script.write("if [[ $error_flag -eq 1 ]]; then\n")
            script.write("echo Tracseg ran with error. Now copying outputs to /nfs2...\n")
            script.write("scp -r {}/* {}@landman01.accre.vanderbilt.edu:{}/\n".format(session_temp, self.vunetID, tractseg_target))
            script.write("echo Done copying outputs. Now deleting inputs and outputs on /scratch...\n")
            script.write("rm -r {}/*\n".format(session_input))
            script.write("rm -r {}/*\n".format(session_temp))
            script.write("rm -r {}/*\n".format(session_output))
            script.write("fi\n\n")

            #otherwise, keep running the processing
            script.write("time singularity run -B {}:{} -B {}:{} {} TractSeg -i {}/tractseg/peaks.nii.gz -o {}/tractseg --output_type endings_segmentation\n".format(accre_home_directory, accre_home_directory,session_temp,session_temp, ts_simg, session_temp, session_temp))
            script.write("time singularity run -B {}:{} -B {}:{} {} TractSeg -i {}/tractseg/peaks.nii.gz -o {}/tractseg --output_type TOM\n".format(accre_home_directory, accre_home_directory,session_temp,session_temp, ts_simg, session_temp, session_temp))
            script.write("time singularity run -B {}:{} -B {}:{} {} Tracking -i {}/tractseg/peaks.nii.gz -o {}/tractseg --tracking_format tck\n".format(accre_home_directory, accre_home_directory,session_temp,session_temp, ts_simg, session_temp, session_temp))

            script.write('echo "..............................................................................."\n')
            script.write("echo Done running TractSeg. Now computing measures per bundle...\n")
            script.write("mkdir {}/tractseg/measures\n".format(session_temp))

            script.write("for i in {}/tractseg/TOM_trackings/*.tck; do\n".format(session_temp))
            script.write('    echo "$i"; s=${i##*/}; s=${s%.tck}; echo $s;\n')
            script.write('    time singularity exec -B {}:{} --nv {} scil_evaluate_bundles_individual_measures.py {}/tractseg/TOM_trackings/$s.tck {}/tractseg/measures/$s-SHAPE.json --reference={}/dwmri_1mm_iso.nii.gz\n'.format(session_temp,session_temp, scilus_simg, session_temp, session_temp, session_temp))
            script.write('    time singularity exec -B {}:{} --nv {} scil_compute_bundle_mean_std.py {}/tractseg/TOM_trackings/$s.tck {}/dwmri_tensor_fa_1mm_iso.nii.gz {}/dwmri_tensor_md_1mm_iso.nii.gz {}/dwmri_tensor_ad_1mm_iso.nii.gz {}/dwmri_tensor_rd_1mm_iso.nii.gz --density_weighting --reference={}/dwmri_1mm_iso.nii.gz\n'.format(session_temp,session_temp, scilus_simg, session_temp, session_temp, session_temp, session_temp, session_temp, session_temp))
            script.write('done\n\n')

            script.write("echo Done computing measures per bundle. Now copying outputs to /nfs2...\n")
            script.write("scp -r {}/* {}@landman01.accre.vanderbilt.edu:{}/\n".format(session_temp, self.vunetID, tractseg_target))
            script.write("echo Done copying outputs. Now deleting inputs and outputs on /scratch...\n")
            script.write("rm -r {}/*\n".format(session_input))
            script.write("rm -r {}/*\n".format(session_temp))
            script.write("rm -r {}/*\n".format(session_output))


            #### KURT SAID WE NEED TO USE 1500 or less
                #included this 

    def macruise_script_generate(self, tmp_input_dir, tmp_output_dir):
        """
        Adapted from Chenyu Gao's code

        Writes the scripts for macruise to the script directory
        """

        #find all segmentations (either SLANT-TICV or UNest)
        print("Finding all T1s...")
        find_cmd = "find {} -mindepth 3 -maxdepth 4 \( -type f -o -type l \) -name '*T1w.nii.gz' | grep -v derivatives".format(str(self.root_dataset_path))
        results = subprocess.run(find_cmd, shell=True, capture_output=True, text=True).stdout.strip().splitlines()
        i=0
        missing_data = pd.DataFrame(columns=['sub', 'ses', 'acq', 'run', 'missing'])

        for t1_p in tqdm(results):
            t1 = Path(t1_p)
            #get the subject and session, and the acq/run
            sub, ses, acq, run = self.get_sub_ses_acq_run_t1(t1)
            
            #check to make sure that the segmentations exist
            sesx = '' if ses == '' else '_'+ses
            acqx = '' if acq == '' else '_'+acq
            runx = '' if run == '' else '_'+run
            if self.args.use_unest_seg:
                segdir = self.dataset_derivs/(sub)/(ses)/("UNest{}{}".format(acq, run))
                seg_file = segdir/("FinalResults")/("{}{}{}{}_T1w_seg_merged_braincolor.nii.gz".format(sub, sesx, acqx, runx))
            else:
                segdir = self.dataset_derivs/(sub)/(ses)/("SLANT-TICVv1.2{}{}".format(acq, run))
                seg_file = segdir/("post")/("FinalResult")/("{}{}{}{}_T1w_seg.nii.gz".format(sub, sesx, acqx, runx))
            
            if not segdir.exists() or not seg_file.exists():
                row = {'sub':sub, 'ses':ses, 'acq':acq, 'run':run, 'missing':'TICV' if not self.args.use_unest_seg else 'UNest'}
                missing_data = pd.concat([missing_data, pd.Series(row).to_frame().T], ignore_index=True)
                continue
            
            #check to see if the macruise outputs exist
            macruise_dir = self.dataset_derivs/(sub)/(ses)/("MaCRUISEv3.2.0{}{}".format(acq, run))
            segrefine = macruise_dir/("SegRefine")/("SEG_refine.nii.gz")
            surf_freeview = macruise_dir/("Output")/("Surfaces_FreeView")/("target_image_GMimg_innerSurf.asc")
            surf_mni = macruise_dir/  'Output' / 'Surfaces_MNI' / 'target_image_GMimg_innerSurf.asc'

            if segrefine.exists() and surf_freeview.exists() and surf_mni.exists():
                continue

            #so the segmentations are there and we do not have macruise outputs, so we can run
            i+=1

            #setup the inputs and outputs
            session_input = tmp_input_dir/(sub)/("{}{}{}".format(ses,acq,run))
            session_output = tmp_output_dir/(sub)/("{}{}{}".format(ses,acq,run))
            for d in [session_input, session_output]:
                if not d.exists():
                    os.makedirs(d)


            #create the output directory if it does not exist
            macruise_target = self.output_dir/(sub)/(ses)/("MaCRUISEv3.2.0{}{}".format(acq, run))
            if not macruise_target.exists():
                os.makedirs(macruise_target)
            
            #write the script
            self.write_macruise_script(i, t1, seg_file, session_input, session_output, macruise_target)
        self.num_scripts = i
        #print("Wrote {} scripts for {}".format(i, self.pipeline))
        missing_data.to_csv('{}_{}_missing_data.csv'.format(self.root_dataset_path.name, self.pipeline))
        print("Wrote missing data to {}_{}_missing_data.csv".format(self.root_dataset_path.name, self.pipeline))

    def write_macruise_script(self, i, t1, segfile, session_input, session_output, macruise_target):
        """
        Writes a single script for running macruise
        """

        script_f = self.script_dir/("{}.sh".format(i))
        with open(script_f, 'w') as script:
            script.write("#!/bin/bash\n")
            script.write("echo Copying input files from /nfs2 to /scratch...\n")
            script.write("scp {}@landman01.accre.vanderbilt.edu:{} {}/T1.nii.gz\n".format(self.vunetID, t1, session_input))
            script.write("scp {}@landman01.accre.vanderbilt.edu:{} {}/orig_target_seg.nii.gz\n".format(self.vunetID, segfile, session_input))
            script.write("echo Done copying input files. Now running MaCRUISE...\n")
            script.write("singularity exec -B {}:/INPUTS -B {}:/OUTPUTS {} xvfb-run -a --server-args=-screen 3 1920x1200x24 -ac +extension GLX /extra/MaCRUISE_v3_2_0_classern /extra/MaCRUISE_v3_2_0_classern\n".format(session_input, session_output, self.simg))
            script.write("echo Done running MaCRUISE. Now copying outputs to /nfs2...\n")
            script.write("scp -r {}/* {}@landman01:{}/\n".format(session_output, self.vunetID, macruise_target))
            script.write("echo Done copying outputs. Now deleting inputs and outputs on /scratch...\n")
            script.write("rm -r {}/*\n".format(session_input))
            script.write("rm -r {}/*\n".format(session_output))

    def connectome_special_script_generate(self, tmp_input_dir, tmp_output_dir):
        """
        Adapted from Nancy Newlin's code

        Writes the scripts for connectome_special to the script directory
        """

        #find command to search for PQ directories
        print("Finding all PreQual outputs...")
        find_cmd = "find {} -mindepth 2 -maxdepth 3 -type d -name 'PreQual*'".format(str(self.dataset_derivs))
        result = subprocess.run(find_cmd, shell=True, capture_output=True, text=True).stdout
        files = result.strip().splitlines()

        missing_data = pd.DataFrame(columns=['sub', 'ses', 'acq', 'run', 'missing'])

        i=0
        for PQdir_path in tqdm(files):

            PQdir = Path(PQdir_path)
            PQ_suffix = PQdir.name[7:]

            #if there is a PQ, there should be a dwi
            #get the dwi files
            
            #first, check to make sure that the data does not already exist
            cs_dir = PQdir.parent/("ConnectomeSpecial{}".format(PQ_suffix))
            files = ["CONNECTOME_Weight_MeanFA_NumStreamlines_10000000_Atlas_SLANT.csv", "CONNECTOME_Weight_NUMSTREAMLINES_NumStreamlines_10000000_Atlas_SLANT.csv",
                    "graphmeasures.json", "graphmeasures_nodes.json", "log.txt", "tracks_10000000_compressed.tck", "ConnectomeQA.png",
                    "CONNECTOME_NUMSTREAM.npy", "CONNECTOME_LENGTH.npy", "CONNECTOME_FA.npy", "b0.nii.gz", "atlas_slant_subj.nii.gz"
                    ]
            if all([(cs_dir/f).exists() for f in files]):
                continue

            #get the subject and session
            if PQdir.parent.parent.name == "derivatives":
                sub = PQdir.parent.name
                ses = ''
            else:
                sub = PQdir.parent.parent.name
                ses = PQdir.parent.name
            
            #and acq and run
            pattern = r'PreQual(?:(acq-\w+))?(?:(run-\d{1,2}))?'
            matches = re.findall(pattern, PQdir.name)
            acq, run = matches[0][0], matches[0][1]
            if 'run' in PQdir.name and acq != '': #I cant figure out the regex match for this, so I will just do it manually (I think there is nested ?? in the regex)
                run = 'run' + PQdir.name.split('run')[1]
                acq = PQ_suffix.split('run')[0]

            #check to make sure that the PreQual outputs exist
            PQdwi_files = self.get_PQ_dwi_outputs(PQdir)
            dwi_pq_exist=True
            for file in PQdwi_files:
                if not file.exists():
                    dwi_pq_exist = False
                    break
            if not dwi_pq_exist:
                row = {'sub':sub, 'ses':ses, 'acq':acq, 'run':run, 'missing':'PreQual'}
                missing_data = pd.concat([missing_data, pd.Series(row).to_frame().T], ignore_index=True)
                continue

            # #OASIS3 check: see how many shells there are
            #     #if there are more than 7, throw it out (would need to be changed for other datasets that actually have full shells)
            # num_shells = self.get_num_shells(PQdwi_files[1])
            # if num_shells >= 7:
            #     row = {'sub':sub, 'ses':ses, 'acq':acq, 'run':run, 'missing':'Too many shells (OASIS3)'}
            #     missing_data = pd.concat([missing_data, pd.Series(row).to_frame().T], ignore_index=True)
            #     continue

            #check to see if a T1 exists
            if ses == '':
                anat_dir = self.root_dataset_path/(sub)/('anat')
            else:
                anat_dir = self.root_dataset_path/(sub)/(ses)/('anat')
            t1 = self.get_t1(anat_dir, sub, ses) 
            if t1 == None:
                row = {'sub':sub, 'ses':ses, 'acq':acq, 'run':run, 'missing':'T1'}
                missing_data = pd.concat([missing_data, pd.Series(row).to_frame().T], ignore_index=True)
                continue

            #get the TICV segmentation for that T1
            if not self.args.use_unest_seg:
                seg = self.get_TICV_seg_file(t1, PQdir)
            else:
                seg = self.get_UNest_seg_file(t1, PQdir)
            if not seg.exists():
                row = {'sub':sub, 'ses':ses, 'acq':acq, 'run':run, 'missing':'TICV' if not self.args.use_unest_seg else 'UNest'}
                # print(t1.name)
                # print(str(seg))
                missing_data = pd.concat([missing_data, pd.Series(row).to_frame().T], ignore_index=True)
                continue

            #we also need the scalar maps
            #get the tensor maps to use for the mean/std bundle calculations
            if ses != '':
                eve3dir = self.dataset_derivs/(sub)/(ses)/("WMAtlasEVE3{}{}".format(acq, run))
            else:
                eve3dir = self.dataset_derivs/(sub)/("WMAtlasEVE3{}{}".format(acq, run))
            tensor_maps = [eve3dir/("dwmri%{}.nii.gz".format(m)) for m in ['fa', 'md', 'ad', 'rd']]
            using_PQ = False
            if not all([t.exists() for t in tensor_maps]):
                row = {'sub':sub, 'ses':ses, 'acq':acq, 'run':run, 'missing':'tensor_maps'}
                missing_data = pd.concat([missing_data, pd.Series(row).to_frame().T], ignore_index=True)
                continue

            #increment the count since all inputs are present and we can run
            i += 1

            #create the input and output directories
            session_input = tmp_input_dir/(sub)/("{}{}{}".format(ses,acq,run))
            session_output = tmp_output_dir/(sub)/("{}{}{}".format(ses,acq,run))
            for d in [session_input, session_output]:
                if not d.exists():
                    os.makedirs(d)
            
            #create the output directory if it does not exist
            connectome_target = self.output_dir/(sub)/(ses)/("ConnectomeSpecial{}".format(PQ_suffix))
            if not connectome_target.exists():
                os.makedirs(connectome_target)
            
            #write the script
            self.write_connectome_special_script(i, PQdir, session_input, session_output, connectome_target, t1, seg, tensor_maps)
        self.num_scripts = i
        #print("Wrote {} scripts for {}".format(i, self.pipeline))
        missing_data.to_csv('{}_{}_missing_data.csv'.format(self.root_dataset_path.name, self.pipeline))
        print("Wrote missing data to {}_{}_missing_data.csv".format(self.root_dataset_path.name, self.pipeline))

    def write_connectome_special_script(self, i, PQdir, session_input, session_output, connectome_target, t1, seg, tensor_maps):
        """
        Writes a single script for the connectome special
        """

        #directories to create
        PQ_inp = session_input/('PreQual')
        PQ_prep = PQ_inp/('PREPROCESSED')
        PQ_scalars = PQ_inp/('SCALARS')
        seg_input = session_input/('Slant')
        seg_post = seg_input/('post')
        seg_post_final = seg_post/('FinalResult')
        seg_pre = seg_input/('pre')
        seg_pre_input = seg_pre/('{}'.format(t1.name.split('.nii')[0]))

        fa, md, ad, rd = tensor_maps

        script_f = self.script_dir/("{}.sh".format(i))
        with open(script_f, 'w') as script:
            script.write("#!/bin/bash\n")
            script.write("Setting up directories for inputs...\n")
            script.write("mkdir -p {}/\n".format(seg_pre_input))
            script.write("mkdir -p {}/\n".format(seg_post_final))
            script.write("mkdir -p {}/\n".format(PQ_prep))
            script.write("mkdir -p {}/\n".format(PQ_scalars))
            script.write("echo Finished setting up directories. Copying input files from /nfs2 to /scratch...\n")
            script.write("scp {}@landman01.accre.vanderbilt.edu:{} {}/orig_target.nii.gz\n".format(self.vunetID, t1, seg_pre_input))
            if not self.args.use_unest_seg:
                script.write("scp {}@landman01.accre.vanderbilt.edu:{} {}/\n".format(self.vunetID, seg, seg_post_final))
            else:
                new_seg_name = seg.name.replace('T1w_seg_merged_braincolor', 'T1w_seg')
                script.write("scp {}@landman01.accre.vanderbilt.edu:{} {}/{}\n".format(self.vunetID, seg, seg_post_final, new_seg_name))
            script.write("scp {}@landman01.accre.vanderbilt.edu:{}/PREPROCESSED/* {}/\n".format(self.vunetID, PQdir, PQ_prep))
            script.write("scp {}@landman01.accre.vanderbilt.edu:{} {}/dwmri_tensor_fa.nii.gz\n".format(self.vunetID, fa, PQ_scalars))
            script.write("scp {}@landman01.accre.vanderbilt.edu:{} {}/dwmri_tensor_md.nii.gz\n".format(self.vunetID, md, PQ_scalars))
            script.write("scp {}@landman01.accre.vanderbilt.edu:{} {}/dwmri_tensor_ad.nii.gz\n".format(self.vunetID, ad, PQ_scalars))
            script.write("scp {}@landman01.accre.vanderbilt.edu:{} {}/dwmri_tensor_rd.nii.gz\n".format(self.vunetID, rd, PQ_scalars))
            script.write("echo Done copying input files. Now running ConnectomeSpecial...\n")
            script.write("singularity run --bind {}:/DIFFUSION/,{}:/SLANT/,{}:/OUTPUTS/ {}\n".format(PQ_inp, seg_input, session_output, self.simg))
            #singularity run --bind ${workingpath_accre}/PreQual/:/DIFFUSION/,${workingpath_accre}/Slant/:/SLANT/,${workingpath_accre}/Output/:/OUTPUTS/ ${singularity_path}
            script.write("echo Done running ConnectomeSpecial. Now copying outputs to /nfs2...\n")
            script.write("scp -r {}/* {}@landman01:{}/\n".format(session_output, self.vunetID, connectome_target))
            script.write("echo Done copying outputs. Now deleting inputs and outputs on /scratch...\n")
            script.write("rm -r {}/*\n".format(session_input))
            script.write("rm -r {}/*\n".format(session_output))

    def francois_special_script_generate(self, tmp_input_dir, tmp_output_dir):
        """
        Writes the scripts for francois special to the script directory
        """
        

        def get_orig_dwi(dwi_path, sub, ses, acq, run):
            """
            Get the raw/original dwi files
            """
            sesx = '' if ses == '' else '_'+ses
            acqx = '' if acq == '' else '_'+acq
            runx = '' if run == '' else '_'+run
            nii = dwi_path/("{}{}{}{}_dwi.nii.gz".format(sub, sesx, acqx, runx))
            if self.args.dataset_name == 'HCPA' or self.args.dataset_name == 'HCPD':
                bvec = Path("/nfs2/harmonization/raw/{}_PQ_bvecs/{}/{}_dwi.bvec".format(self.args.dataset_name, sub, sub))
                #bvec = tmp_dwi_path/("{}{}{}{}_dwi.bvec".format(sub, sesx, acqx, runx))
                #return nii, bval, bvec
            else:
                bvec = Path(str(nii).replace('dwi.nii.gz', 'dwi.bvec'))
            bval = Path(str(nii).replace('dwi.nii.gz', 'dwi.bval'))
            return nii, bval, bvec

        #find command to search for PQ directories
        print("Finding all PreQual outputs...")
        find_cmd = "find {} -mindepth 2 -maxdepth 3 -type d -name 'PreQual*'".format(str(self.dataset_derivs))
        result = subprocess.run(find_cmd, shell=True, capture_output=True, text=True).stdout
        files = result.strip().splitlines()

        missing_data = pd.DataFrame(columns=['sub', 'ses', 'acq', 'run', 'missing'])

        i=0
        for PQdir_path in tqdm(files):

            PQdir = Path(PQdir_path)
            PQ_suffix = PQdir.name[7:]

            #if there is a PQ, there should be a dwi
            #get the dwi files
            
            #first, check to make sure that the data does not already exist
            francois_dir = PQdir.parent/("FrancoisSpecial{}".format(PQ_suffix))
            if (francois_dir/("report.pdf")).exists():
                continue

            #get the subject and session
            if PQdir.parent.parent.name == "derivatives":
                sub = PQdir.parent.name
                ses = ''
            else:
                sub = PQdir.parent.parent.name
                ses = PQdir.parent.name
            
            #and acq and run
            pattern = r'PreQual(?:(acq-\w+))?(?:(run-\d{1,2}))?'
            matches = re.findall(pattern, PQdir.name)
            acq, run = matches[0][0], matches[0][1]
            if 'run' in PQdir.name and acq != '': #I cant figure out the regex match for this, so I will just do it manually (I think there is nested ?? in the regex)
                run = 'run' + PQdir.name.split('run')[1]
                acq = PQ_suffix.split('run')[0]

            #check to make sure that the PreQual outputs exist
            PQdwi_files = self.get_PQ_dwi_outputs(PQdir)
            dwi_pq_exist=True
            for file in PQdwi_files:
                if not file.exists():
                    dwi_pq_exist = False
                    break
            if not dwi_pq_exist:
                row = {'sub':sub, 'ses':ses, 'acq':acq, 'run':run, 'missing':'PreQual'}
                missing_data = pd.concat([missing_data, pd.Series(row).to_frame().T], ignore_index=True)
                continue


            #check to see if a T1 exists
            if ses == '':
                anat_dir = self.root_dataset_path/(sub)/('anat')
            else:
                anat_dir = self.root_dataset_path/(sub)/(ses)/('anat')
            t1 = self.get_t1(anat_dir, sub, ses) 
            if t1 == None:
                row = {'sub':sub, 'ses':ses, 'acq':acq, 'run':run, 'missing':'T1'}
                missing_data = pd.concat([missing_data, pd.Series(row).to_frame().T], ignore_index=True)
                continue

            #get the TICV segmentation for that T1
            if not self.args.use_unest_seg:
                seg = self.get_TICV_seg_file(t1, PQdir)
            else:
                seg = self.get_UNest_seg_file(t1, PQdir)
            if seg is None or not seg.exists():
                row = {'sub':sub, 'ses':ses, 'acq':acq, 'run':run, 'missing':'TICV' if not self.args.use_unest_seg else 'UNest'}
                # print(t1.name)
                # print(str(seg))
                missing_data = pd.concat([missing_data, pd.Series(row).to_frame().T], ignore_index=True)
                continue

            #need to also get the original dwi to check the number of shells and bvals
                #in order to setup the args for later (SH, DTI SHELL, FODF)
            if ses == '':
                dwi_dir = self.root_dataset_path/(sub)/('dwi')
            else:
                dwi_dir = self.root_dataset_path/(sub)/(ses)/('dwi')
            
            if self.args.dataset_name == 'VMAP_2.0' or self.args.dataset_name == 'VMAP_TAP': #just get the PQ
                (nii, bval, bvec) = PQdwi_files
            elif self.args.dataset_name == 'UKBB':
                (nii, bval, bvec) = PQdwi_files
            else:
                nii, bval, bvec = get_orig_dwi(dwi_dir, sub, ses, acq, run)
            if not nii.exists() or not bval.exists() or not bvec.exists():
                #check for BLSA acq-double: if it is, then we should let it run through
                if acq == 'acq-double' and self.args.dataset_name == 'BLSA':
                    pass
                else:
                    row = {'sub':sub, 'ses':ses, 'acq':acq, 'run':run, 'missing':'orig_dwi'}
                    missing_data = pd.concat([missing_data, pd.Series(row).to_frame().T], ignore_index=True)
                    continue

            #get the shells and number of directions
            dti_shells, fodf_shells, sh8_shells = self.get_shells_and_dirs(PQdir, dwi_dir)
            #make sure that these are valid
            if dti_shells is None and fodf_shells is None:
                if sh8_shells == "bad_dims":
                    row = {'sub':sub, 'ses':ses, 'acq':acq, 'run':run, 'missing':'dimension_agreement_or_params'}
                    missing_data = pd.concat([missing_data, pd.Series(row).to_frame().T], ignore_index=True)
                    continue
                else:
                    row = {'sub':sub, 'ses':ses, 'acq':acq, 'run':run, 'missing':'dti_and_fodf_shells'}
                    missing_data = pd.concat([missing_data, pd.Series(row).to_frame().T], ignore_index=True)
                    continue
            elif dti_shells == "No b0" or fodf_shells == "No b0":
                row = {'sub':sub, 'ses':ses, 'acq':acq, 'run':run, 'missing':'No_b0'}
                missing_data = pd.concat([missing_data, pd.Series(row).to_frame().T], ignore_index=True)
                continue
            elif dti_shells == None:
                row = {'sub':sub, 'ses':ses, 'acq':acq, 'run':run, 'missing':'dti_shells'}
                missing_data = pd.concat([missing_data, pd.Series(row).to_frame().T], ignore_index=True)
                continue
            elif fodf_shells == None:
                row = {'sub':sub, 'ses':ses, 'acq':acq, 'run':run, 'missing':'fodf_shells'}
                missing_data = pd.concat([missing_data, pd.Series(row).to_frame().T], ignore_index=True)
                continue

            #if the sh8 shells are the same as the fodf shells, then we can use sh degree of 8
            if sh8_shells == fodf_shells:
                sh_degree = 8
            else:
                sh_degree = 6

            #increment the count since all inputs are present and we can run
            i += 1

            #now create the input and output directories
            session_input = tmp_input_dir/(sub)/("{}{}{}".format(ses,acq,run))
            session_output = tmp_output_dir/(sub)/("{}{}{}".format(ses,acq,run))
            for d in [session_input, session_output]:
                if not d.exists():
                    os.makedirs(d)
            
            #create the target output directory
            francois_target = self.output_dir/(sub)/(ses)/("FrancoisSpecial{}".format(PQ_suffix))
            if not francois_target.exists():
                os.makedirs(francois_target)

            #save the rounded bvals to the target_directory
            rounded_bvals = francois_target/('bvals_rounded.bval')
            if acq == 'acq-double' and self.args.dataset_name == 'BLSA':
                ##TODO: need to save the rounded bvals for BLSA
                #copy the bval file
                os.system("cp {} {}".format(PQdir / 'PREPROCESSED' / 'dwmri.bval' , rounded_bvals))
                pass
            else:
                np.savetxt(rounded_bvals, self.rounded_bvals[np.newaxis, :].astype(int), fmt='%s')

            #write the script
            self.write_francois_special_script(i, PQdir, session_input, session_output, francois_target, t1, seg, sub, ses, dti_shells, fodf_shells, sh_degree)
        
        self.num_scripts = i
        print("Wrote {} scripts for {}".format(i, self.pipeline))
        missing_data.to_csv('{}_{}_missing_data.csv'.format(self.root_dataset_path.name, self.pipeline))
        print("Wrote missing data to {}_{}_missing_data.csv".format(self.root_dataset_path.name, self.pipeline))

    def get_original_bval_bvec(self, PQdir, dwi_dir):
        """
        Get the original bval and bvec files given the PQdir and dwi_dir

        Uses the names in the thresholded bvals directory for PreQual
        """
        

        ### HAVE TO MAKE SURE THAT THEY ARE IN THE SAME ORDER AS THE PREPROCESSED DWI
            #"shouldnt" matter for BLSA, but it will if there is a truncated run
        ### NEED TO THRESHOLD THE BVALS AND CHECK ORDER??

        #special for BLSA, ROSMAPMARS
        if self.args.dataset_name == 'BLSA':
            sub = PQdir.parent.parent.name
            ses = PQdir.parent.name
            if 'acq-double' in PQdir.name:
                bval_files = [dwi_dir / '{}_{}_{}_dwi.bval'.format(sub, ses, 'run-{}'.format(run)) for run in range(1, 3)]
                bvec_files = [dwi_dir / '{}_{}_{}_dwi.bvec'.format(sub, ses, 'run-{}'.format(run)) for run in range(1, 3)]
            elif 'run-1' in PQdir.name:
                bval_files = [dwi_dir / '{}_{}_{}_dwi.bval'.format(sub, ses, 'run-1')]
                bvec_files = [dwi_dir / '{}_{}_{}_dwi.bvec'.format(sub, ses, 'run-1')]
            elif 'run-2' in PQdir.name:
                bval_files = [dwi_dir / '{}_{}_{}_dwi.bval'.format(sub, ses, 'run-2')]
                bvec_files = [dwi_dir / '{}_{}_{}_dwi.bvec'.format(sub, ses, 'run-2')]
            else:
                print("Error: BLSA run/acq not found for {}".format(PQdir))
                exit(1)
        elif self.args.dataset_name == 'ROSMAPMARS':
            sub = PQdir.parent.parent.name
            ses = PQdir.parent.name
            bval_files = [dwi_dir / '{}_{}_dwi.bval'.format(sub, ses)]
            bvec_files = [dwi_dir / '{}_{}_dwi.bvec'.format(sub, ses)]
        ### Note that the above are not the thresholded bvals, but it shouldn't matter anyway since the datasets above should not
        ### have been thresholded anyway
        else: #should have been run in BIDS format, so we can get the file names
            #need to be in the correct order
            #UKBB: not necessarily PA then AP, so we would need to check the acqparams file
            if self.args.dataset_name == 'UKBB':
                sub = PQdir.parent.name
                #read in the acqparams file
                acqparams_file = PQdir / "TOPUP" / "acqparams.txt"
                if not acqparams_file.exists():
                    return None, None
                acqparams = np.loadtxt(acqparams_file)
                if acqparams[0, 1] == -1.0: #PA first
                    bval_files = [dwi_dir / '{}_dir-PA_dwi.bval'.format(sub), dwi_dir / '{}_dir-AP_dwi.bval'.format(sub)]
                    bvec_files = [dwi_dir / '{}_dir-PA_dwi.bvec'.format(sub), dwi_dir / '{}_dir-AP_dwi.bvec'.format(sub)]
                else: #AP first
                    bval_files = [dwi_dir / '{}_dir-AP_dwi.bval'.format(sub), dwi_dir / '{}_dir-PA_dwi.bval'.format(sub)]
                    bvec_files = [dwi_dir / '{}_dir-AP_dwi.bvec'.format(sub), dwi_dir / '{}_dir-PA_dwi.bvec'.format(sub)]

            #newVMAP: b1000, b2000, then RPE
            elif self.args.dataset_name == 'VMAP_2.0' or self.args.dataset_name == 'VMAP_TAP':
                sub = PQdir.parent.parent.name
                ses = PQdir.parent.name
                bval_files = [dwi_dir / '{}_{}_{}_dwi.bval'.format(sub, ses, acq) for acq in ['acq-b1000', 'acq-b2000', 'acq-ReversePE']]
                bvec_files = [dwi_dir / '{}_{}_{}_dwi.bvec'.format(sub, ses, acq) for acq in ['acq-b1000', 'acq-b2000', 'acq-ReversePE']]
            
            else:
                thresholded_bvals = PQdir / 'THRESHOLDED_BVALS'
                bval_names = [x.name for x in thresholded_bvals.iterdir()]
                bvec_names = [x.replace('bval', 'bvec') for x in bval_names]
                bval_files = [thresholded_bvals / x for x in bval_names]
                bvec_files = [dwi_dir / x for x in bvec_names]
        
        #now, read in all the bval and bvec files
        bvals = []
        bvecs = []
        for bval_f, bvec_f in zip(bval_files, bvec_files):
            bval = np.loadtxt(bval_f)
            bvec = np.loadtxt(bvec_f)
            bvals.append(bval)
            bvecs.append(bvec)
        bvals = np.concatenate(bvals)
        bvecs = np.concatenate(bvecs, axis=1)

        #remove all combinations where the bvalue is greater than 150 but the bvecs are all 0
            #for VMAP jefferson, which has a b1000 at the end that has no gradient
        inds_not_b0 = np.where(bvals > 150)[0]
        no_grad_idxs = []
        for i in inds_not_b0:
            if np.all(bvecs[:, i] == 0):
                no_grad_idxs.append(i)
        bvals = np.delete(bvals, no_grad_idxs)
        bvecs = np.delete(bvecs, no_grad_idxs, axis=1)


        #do a check to make sure that the bvals are the same (for most of the volumes, thresholding may have caused one or two to be different)
        pq_bval = np.loadtxt(PQdir / "PREPROCESSED" / 'dwmri.bval')
        #print(bvals.shape, pq_bval.shape)
        #print(bval_files)
        if pq_bval.shape != bvals.shape:
            print("Error: bval files (raw and preprocessed) do not have the same number of volumes for {}".format(str(PQdir)))
            return None, None
        elif not np.allclose(pq_bval, bvals, atol=100):
            print("Error: bval files (raw and preprocessed) do not have the same values for {}".format(str(PQdir)))
            return None, None
        assert pq_bval.shape == bvals.shape, "Error: bval files (raw and preprocessed) do not have the same number of volumes for {}".format(str(PQdir))
        assert np.allclose(pq_bval, bvals, atol=100), "Error: bval files (raw and preprocessed) do not have the same values for {}".format(str(PQdir))

        return bvals, bvecs

    def get_shells_and_dirs(self, PQdir, dwi_dir):
        """
        Given a bval and bvec file, get the number of shells and number of directions

        For FrancoisSpecial
        """


        def round(num, base):
            """
            Taken from Leon Cai's PreQual
            """

            d = num / base
            if d % 1 >= 0.5:
                return base*np.ceil(d)
            else:
                return base*np.floor(d)

        # Initialize an empty set to store unique (x, y, z) tuples
        unique_pairs = set()
        shells_dic = {}
        bval_map = {}

        # Read data from the bval files
        bvals, bvecs = self.get_original_bval_bvec(PQdir, dwi_dir)
        if type(bvals) == type(None) and type(bvecs) == type(None):
            if bvals == None and bvecs == None:
                return None, None, "bad_dims"
        self.rounded_bvals = np.array([round(b, 100) for b in bvals])
        assert bvals.size == bvecs.shape[1], "Error: bval and bvec files do not have the same number of volumes for {}".format(str(PQdir))
        if bvecs.shape[0] != 3:
            bvecs = bvecs.T
        assert bvecs.shape[0] == 3, "Error: bvec file for {} does not have 3 rows".format(str(PQdir))

        #first, get the shells and unqiue directions
        hasb0 = False
        for i in range(bvals.size):
            b = round(bvals[i], 100)
            vec = tuple(bvecs[:,i])
            #add vector to the shell dic
            if b not in shells_dic.keys():
                shells_dic[b] = set()
                shells_dic[b].add(vec)
            else:
                shells_dic[b].add(vec)
            #add the index mapping for bvalue/shell
            if b not in bval_map.keys():
                bval_map[b] = [i]
            else:
                bval_map[b].append(i)
            if b == 0:
                hasb0 = True
        
        if not hasb0:
            return "No b0", "No b0", "No b0"
        
        #DTI SHELL: keep the shells that are above 500 and less than 1500 - at least 12 remaining directions
        dti_set = set()
        dti_dirs = set()
        fod_set = set()
        sh8_set = set()
        for b in shells_dic.keys():
            if b < 500:
                continue
            #shell DTI
            if b <= 1500:
                dti_set.add(b)
                [dti_dirs.add(x) for x in shells_dic[b]]
            #shell FODF #should be 800, but since BLSA and BIOCARD have 700, we will use that
            if b >= 700 and len(shells_dic[b]) >= 28:
                fod_set.add(b)
            if b >= 700 and len(shells_dic[b]) >= 45:
                sh8_set.add(b)

        #check to see if we can do both DTI and FODF
        if (len(dti_set) == 0 or len(dti_dirs) < 6) and len(fod_set) == 0:
            return None, None, None
        elif len(dti_set) == 0 or len(dti_dirs) < 6:
            #print(shells_dic)
            return None, fod_set, None
        elif len(fod_set) == 0:
            return dti_set, None, None
        return '"' + " ".join(['0']+[str(int(x)) for x in sorted(dti_set)]) + '"', '"' + " ".join(['0']+[str(int(x)) for x in sorted(fod_set)]) + '"', '"' + " ".join(['0']+[str(int(x)) for x in sorted(sh8_set)]) + '"'
        

        ### THIS MAY BE WRONG, LETS SEE WHAT PK SAYS

    def write_francois_special_script(self, i, PQdir, session_input, session_output, francois_target, t1, seg, sub, ses, dti_shells, fodf_shells, sh_degree):
        """
        Adapted from Praitatini Kanakaraj's code

        Writes a single script for the francois special
        """
        
        #make the temp directory
        tmp_dir = Path("/tmp/{}/{}".format(session_input.parent.name, session_input.name))

        script_f = self.script_dir/("{}.sh".format(i))
        with open(script_f, 'w') as script:
            script.write("#!/bin/bash\n")
            script.write("echo Setting up temp directories for inputs...\n")
            script.write("mkdir -p {}\n".format(tmp_dir))
            script.write("echo Copying input files from /nfs2 to /scratch...\n")
            script.write("scp {}@landman01.accre.vanderbilt.edu:{} {}/t1.nii.gz\n".format(self.vunetID, t1, session_input))
            script.write("scp {}@landman01.accre.vanderbilt.edu:{} {}/t1_seg.nii.gz\n".format(self.vunetID, seg, session_input))
            script.write("scp {}@landman01.accre.vanderbilt.edu:{}/PREPROCESSED/* {}/\n".format(self.vunetID, PQdir, session_input))
            script.write("echo Overwriting bvals with rounded bvals...\n")
            script.write("scp {}@landman01.accre.vanderbilt.edu:{}/bvals_rounded.bval {}/dwmri.bval\n".format(self.vunetID, francois_target, session_input))
            script.write("echo Done copying input files. Now running FrancoisSpecial...\n")
            if ses != '':
                script.write('singularity run --home {} --bind {}:/tmp --containall --cleanenv --bind {}:/INPUTS/ --bind {}:/OUTPUTS/ --bind {}:/TMP {} {} {} t1.nii.gz dwmri.nii.gz dwmri.bval dwmri.bvec {} {} {} 5 wm 5 wm prob 27 0.4 20 t1_seg.nii.gz "4 40 41 44 45 51 52 11 49 59 208 209"\n'.format(tmp_dir, tmp_dir, session_input, session_output, tmp_dir, self.simg, sub, ses, sh_degree, dti_shells, fodf_shells))
            else:
                script.write('singularity run --home {} --bind {}:/tmp --containall --cleanenv --bind {}:/INPUTS/ --bind {}:/OUTPUTS/ --bind {}:/TMP {} {} {} t1.nii.gz dwmri.nii.gz dwmri.bval dwmri.bvec {} {} {} 5 wm 5 wm prob 27 0.4 20 t1_seg.nii.gz "4 40 41 44 45 51 52 11 49 59 208 209"\n'.format(tmp_dir, tmp_dir, session_input, session_output, tmp_dir, self.simg, sub, 'None', sh_degree, dti_shells, fodf_shells))
            script.write("echo Done running FrancoisSpecial. Now copying outputs to /nfs2...\n")
            script.write("scp -r {}/* {}@landman01.accre.vanderbilt.edu:{}/\n".format(session_output, self.vunetID, francois_target))
            script.write("echo Done copying outputs. Now deleting inputs and outputs on /scratch...\n")
            script.write("rm -r {}/*\n".format(session_input))
            script.write("rm -r {}/*\n".format(session_output))

    def biscuit_script_generate(self, tmp_input_dir, tmp_output_dir):
        """
        Writes the scripts for biscuit to the script directory
        """
        
        #first, the extra directories
        def UNest_asserts(root_temp, root_working):
            #check to make sure that the temp and working directories exist and we can write to them
            assert root_temp.exists(), "Error: Root temp directory {} does not exist".format(root_temp)
            assert root_working.exists(), "Error: Root working directory {} does not exist".format(root_working)
            assert os.access(root_temp, os.W_OK), "Error: Root temp directory {} is not writable".format(root_temp)
            assert os.access(root_working, os.W_OK), "Error: Root working directory {} is not writable".format(root_working)

        #assert that the working and temp directories exist
        root_temp = Path(self.args.temp_dir)
        root_working = Path(self.args.working_dir)
        UNest_asserts(root_temp, root_working)

        #first, finds the T1s
        print("Finding all T1s...")
        find_cmd = "find {} -mindepth 3 -maxdepth 4 \( -type f -o -type l \) -name '*T1w.nii.gz' | grep -v derivatives".format(str(self.root_dataset_path))
        results = subprocess.run(find_cmd, shell=True, capture_output=True, text=True).stdout.strip().splitlines()
        i=0
        missing_data = pd.DataFrame(columns=['sub', 'ses', 'acq', 'run', 'missing'])
        for t1_p in tqdm(results):
            t1 = Path(t1_p)
            #get the subject and session, and the acq/run
            sub, ses, acq, run = self.get_sub_ses_acq_run_t1(t1)

            #check to make sure that BISCUIT has not already been run
            biscuit_dir = self.dataset_derivs/(sub)/(ses)/("BISCUITv2.2{}{}".format(acq,run))
            PdF = biscuit_dir / 'PDF' / 'biscuit.pdf'
            if PdF.exists():
                continue

            #check to make sure that MaCRUISE outputs are there
            macruise_dir = self.dataset_derivs/(sub)/(ses)/("MaCRUISEv3.2.0{}{}".format(acq, run))
            segrefine = macruise_dir/("Output")/("SegRefine")/("SEG_refine.nii.gz")
            surf_freeview = macruise_dir/("Output")/("Surfaces_FreeView")/("target_image_GMimg_innerSurf.asc")
            surf_mni = macruise_dir/  'Output' / 'Surfaces_MNI' / 'target_image_GMimg_innerSurf.asc'
            #mc_t1 = macruise_dir / 'Raw' / 'T1.nii.gz'

            if not (segrefine.exists() and surf_freeview.exists() and surf_mni.exists()):
                for x in [segrefine, surf_freeview, surf_mni]:#, mc_t1]:
                    if not x.exists():
                        print(x)
                row = {'sub':sub, 'ses':ses, 'acq':acq, 'run':run, 'missing':'MaCRUISE'}
                missing_data = pd.concat([missing_data, pd.Series(row).to_frame().T], ignore_index=True)
                continue
            
            #biscuit outputs are not there and macruise outputs are there, so we can run
            i+=1

            #setup the inputs and outputs
            session_input = tmp_input_dir/(sub)/("{}{}{}".format(ses,acq,run))
            session_output = tmp_output_dir/(sub)/("{}{}{}".format(ses,acq,run))
            session_temp = root_temp/(sub)/("{}{}{}".format(ses,acq,run))
            session_working = root_working/(sub)/("{}{}{}".format(ses,acq,run))
            for d in [session_input, session_output, session_temp, session_working]:
                if not d.exists():
                    os.makedirs(d)

            #create the output directory if it does not exist
            biscuit_target = self.output_dir/(sub)/(ses)/("BISCUITv2.2{}{}".format(acq,run))
            if not biscuit_target.exists():
                os.makedirs(biscuit_target)

            #write the script
            self.write_biscuit_script(i, session_input, session_output, biscuit_target, segrefine, surf_freeview, surf_mni, t1, session_temp, session_working, sub, ses)
        self.num_scripts = i
        print("Wrote {} scripts for {}".format(i, self.pipeline))
        missing_data.to_csv('{}_{}_missing_data.csv'.format(self.root_dataset_path.name, self.pipeline))
        print("Wrote missing data to {}_{}_missing_data.csv".format(self.root_dataset_path.name, self.pipeline))


    def write_biscuit_script(self, i, session_input, session_output, biscuit_target, segrefine, surf_freeview, surf_mni, mc_t1, session_temp, session_working, sub, ses):
        """
        Adapted from Nazirah Mohd Khairi's code

        Writes a single script for biscuit
        """


        script_f = self.script_dir/("{}.sh".format(i))
        with open(script_f, 'w') as script:
            script.write("#!/bin/bash\n")
            #first, creates a necessary temp directories
            script.write("echo Creating .tmp in outputs directory...\n")
            script.write("mkdir {}/.tmp\n".format(session_output))
            script.write("echo Copying input files from /nfs2 to /scratch...\n")
            script.write("scp {}@landman01.accre.vanderbilt.edu:{} {}/SEG_refine.nii.gz\n".format(self.vunetID, segrefine, session_input))
            script.write("scp {}@landman01.accre.vanderbilt.edu:{} {}/target_image_GMimg_innerSurf.asc\n".format(self.vunetID, surf_freeview, session_input))
            script.write("scp {}@landman01.accre.vanderbilt.edu:{} {}/target_image_GMimg_innerSurf_mni.asc\n".format(self.vunetID, surf_mni, session_input))
            script.write("scp {}@landman01.accre.vanderbilt.edu:{} {}/T1.nii.gz\n".format(self.vunetID, mc_t1, session_input))
            script.write("echo Done copying input files. Now running BISCUIT...\n")
            script.write("singularity run -e --home {} -B {}:/INPUTS -B {}:/OUTPUTS -B {}/.tmp:/OUTPUTS/.tmp -B {}:/tmp -B {}:/opt/biscuit/license.txt {} --proj {} --subj {} --sess {} --reg oasis45 --surf_recon macruise /INPUTS /OUTPUTS\n".format(session_working, session_input, session_output, session_output, session_temp, self.freesurfer_license_path, self.simg, 'R01_' + self.root_dataset_path.name, sub, ses))
            script.write("echo Done running BISCUIT. Now copying outputs to /nfs2...\n")
            script.write("scp -r {}/* {}@landman01.accre.vanderbilt.edu:{}/\n".format(session_output, self.vunetID, biscuit_target))
            script.write("echo Done copying outputs. Now deleting inputs and outputs on /scratch...\n")
            script.write("rm -r {}/*\n".format(session_input))
            script.write("rm -r {}/*\n".format(session_output))
            script.write("rm -r {}/*\n".format(session_temp))
            script.write("rm -r {}/*\n".format(session_working))
            #singularity run -e --home $MACRUISEHOME -B $INDIR:/INPUTS -B $OUTDIR:/OUTPUTS -B $OUTDIR/.tmp:/OUTPUTS/.tmp -B $TMPFOLDER:/tmp -B $LICENSEPATH:/opt/biscuit/license.txt $SINGULARITYPATH --proj 'R01' --subj $SUB --sess $SES --reg oasis45 --surf_recon macruise /INPUTS /OUTPUTS

    def get_num_shells(self, bval):
        """
        Given a bval file, get the number of shells
        """

        def round(num, base):
            """
            Taken from Leon Cai's PreQual
            """

            d = num / base
            if d % 1 >= 0.5:
                return base*np.ceil(d)
            else:
                return base*np.floor(d)

        # Read data from the bval file
        bvals = np.loadtxt(bval)

        #round the bvals to the nearest 100
        rounded_bvals = np.array([round(b, 100) for b in bvals])

        #check to make sure there is a b0
        if 0 not in rounded_bvals:
            assert False, "Error: No b0 found in bval file {}".format(bval)
        
        #return the number of unique shells
        return len(np.unique(rounded_bvals)) - 1

        #return len(np.unique(bvals))


    def noddi_script_generate(self, tmp_input_dir, tmp_output_dir):
        """
        Writes scripts for running NODDI to the scripts directory specified
        """

        #first, find all the PreQualed data
        print("Finding all PreQual outputs...")
        find_cmd = "find {} -mindepth 2 -maxdepth 3 -type d -name 'PreQual*'".format(str(self.dataset_derivs))
        result = subprocess.run(find_cmd, shell=True, capture_output=True, text=True).stdout.strip().splitlines()
        i=0
        missing_data = pd.DataFrame(columns=['sub', 'ses', 'acq', 'run', 'missing'])
        for PQdir_path in tqdm(result):
            PQdir = Path(PQdir_path)
            PQ_suffix = PQdir.name[7:]

            #if there is a PQ, there should be a dwi
            #get the dwi files
            #first, check to make sure that the data does not already exist
            noddi_dir = PQdir.parent/("NODDI{}".format(PQ_suffix))
            files = ['config.pickle', 'FIT_dir.nii.gz', 'FIT_ICVF.nii.gz', 'FIT_ISOVF.nii.gz', 'FIT_OD.nii.gz']
            if all([(noddi_dir/f).exists() for f in files]):
                continue

            #get the subject and session
            if PQdir.parent.parent.name == "derivatives":
                sub = PQdir.parent.name
                ses = ''
            else:
                sub = PQdir.parent.parent.name
                ses = PQdir.parent.name
            
            #and acq and run
            suff = PQ_suffix
            if 'run' in suff:
                run = 'run' + suff.split('run')[1]
                suff = suff.split('run')[0]
            else:
                run = ''
            if 'acq' in suff:
                acq = 'acq' + suff.split('acq')[1]
            else:
                acq = ''
            
            #check to make sure that the PreQual outputs exist
            dwi_files = self.get_PQ_dwi_outputs(PQdir)
            dwi_pq_exist=True
            for file in dwi_files:
                if not file.exists():
                    dwi_pq_exist = False
                    break
            if not dwi_pq_exist:
                row = {'sub':sub, 'ses':ses, 'acq':acq, 'run':run, 'missing':'PreQual'}
                missing_data = pd.concat([missing_data, pd.Series(row).to_frame().T], ignore_index=True)
                continue

            #check the bvals to make sure that there are multiple shells
            dwi, bval, bvec = dwi_files
            num_shells = self.get_num_shells(bval)
            if num_shells < 2:
                row = {'sub':sub, 'ses':ses, 'acq':acq, 'run':run, 'missing':'single_shell'}
                missing_data = pd.concat([missing_data, pd.Series(row).to_frame().T], ignore_index=True)
                continue

            #check the EVE3 registration outputs to see if there is a DWI mask to use
            if ses != '':
                eve3dir = self.dataset_derivs/(sub)/(ses)/("WMAtlasEVE3{}{}".format(acq, run))
            else:
                eve3dir = self.dataset_derivs/(sub)/("WMAtlasEVE3{}{}".format(acq, run))
            dwi_mask = eve3dir/("dwmri%_dwimask.nii.gz")
            if not dwi_mask.exists():
                #need to use the code that kurt gave
                print("DWI mask not found for {}_{}_{}. Will create dwi mask using Kurt's method...".format(sub,ses,PQdir.name))
                dwi_mask = None
                row = {'sub':sub, 'ses':ses, 'acq':acq, 'run':run, 'missing':'DWI_mask'}
                missing_data = pd.concat([missing_data, pd.Series(row).to_frame().T], ignore_index=True)

            #at this point, we should be able to run
            i+=1

            #setup the inputs and outputs
            session_input = tmp_input_dir/(sub)/("{}{}{}".format(ses,acq,run))
            session_output = tmp_output_dir/(sub)/("{}{}{}".format(ses,acq,run))
            for d in [session_input, session_output]:
                if not d.exists():
                    os.makedirs(d)
            
            #create the output directory if it does not exist
            noddi_target = self.output_dir/(sub)/(ses)/("NODDI{}".format(PQ_suffix))
            if not noddi_target.exists():
                os.makedirs(noddi_target)
            
            #write the script
            self.write_noddi_script(i, PQdir, session_input, session_output, noddi_target, sub, ses, acq, run, dwi_mask)
        self.num_scripts = i
        print("Wrote {} scripts for {}".format(i, self.pipeline))
        missing_data.to_csv('{}_{}_missing_data.csv'.format(self.root_dataset_path.name, self.pipeline))
        print("Wrote missing data to {}_{}_missing_data.csv".format(self.root_dataset_path.name, self.pipeline))
    

    def write_noddi_script(self, i, PQdir, session_input, session_output, noddi_target, sub, ses, acq, run, dwi_mask):
        """
        Writes a single script for NODDI
        """

        script_f = self.script_dir/("{}.sh".format(i))
        with open(script_f, 'w') as script:
            script.write("#!/bin/bash\n")
            script.write("echo Copying input files from /nfs2 to /scratch...\n")
            script.write("scp {}@landman01.accre.vanderbilt.edu:{}/PREPROCESSED/dwmri.nii.gz {}/\n".format(self.vunetID, PQdir, session_input))
            script.write("scp {}@landman01.accre.vanderbilt.edu:{}/PREPROCESSED/dwmri.bval {}/Diffusion.bval\n".format(self.vunetID, PQdir, session_input))
            script.write("scp {}@landman01.accre.vanderbilt.edu:{}/PREPROCESSED/dwmri.bvec {}/Diffusion.bvec\n".format(self.vunetID, PQdir, session_input))
            script.write("echo Done copying input files. Now running setup...\n")

            script.write("echo Renaming dwi for some reason...\n")
            script.write("singularity exec {} mrconvert {}/dwmri.nii.gz {}/Diffusion.nii.gz -force\n".format(self.simg, session_input, session_input))
            script.write("mask=dwimean_masked_mask.nii.gz\n")
            #if dwimask does not exist, need to create one using Kurt's method
            if dwi_mask is None:
                script.write("echo DWI mask not found. Creating dwi mask using Kurts method...\n")
                script.write("singularity exec {} scil_extract_dwi_shell.py {}/Diffusion.nii.gz {}/Diffusion.bval {}/Diffusion.bvec 750 {}/dwi_tmp.nii.gz {}/bvals_tmp {}/bvecs_tmp -f -v --tolerance 650\n".format(self.simg, session_input, session_input, session_input, session_input, session_input, session_input))
                script.write("singularity exec {} mrmath {}/dwi_tmp.nii.gz mean {}/dwimean.nii.gz -axis 3 -force\n".format(self.simg, session_input, session_input))
                script.write("bet {}/dwimean.nii.gz {}/dwimean_masked -m -R -f .3\n".format(session_input, session_input))
            else:
                #copy the dwi_mask to the session_input
                script.write("echo 'Using DWI mask found in WMAtlasEVE3 (i.e. T1 segmentation mask moved to DWI space)...'\n")
                script.write("scp {}@landman01.accre.vanderbilt.edu:{} {}/{}\n".format(self.vunetID, dwi_mask, session_input, 'dwimean_masked_mask.nii.gz'))

            #script.write("mkdir {}/DTI\n".format(session_output))
            #script.write("mkdir {}/NODDI\n".format(session_output))
            #script.write("echo Done setting up. Now running DTI fit from scilpy...\n")
            #script.write("singularity exec {} scil_extract_dwi_shell.py {}/Diffusion.nii.gz {}/Diffusion.bval {}/Diffusion.bvec 750 {}/dwi1000.nii.gz {}/bvals1000 {}/bvecs1000 -f -v --tolerance 800\n".format(self.simg, session_input, session_input, session_input, session_input, session_input, session_input))
            #script.write("dtifit -k {}/dwi1000.nii.gz -o {}/DTI/dti -m $mask -r {}/bvecs1000 -b {}/bvals1000\n".format(session_input, session_output, session_input, session_input))

            script.write("echo Done setting up. Now running NODDI from scilpy...\n")
            script.write("singularity exec {} scil_compute_NODDI.py {}/Diffusion.nii.gz {}/Diffusion.bval {}/Diffusion.bvec --out_dir {} --mask {}/dwimean_masked_mask.nii.gz -f\n".format(self.simg, session_input, session_input, session_input, session_output, session_input))

            script.write("echo Done running NODDI. Now copying outputs to /nfs2...\n")
            script.write("scp -r {}/* {}@landman01:{}/\n".format(session_output, self.vunetID, noddi_target))
            script.write("echo Done copying outputs. Now deleting inputs and outputs on /scratch...\n")
            script.write("rm -r {}/*\n".format(session_input))
            script.write("rm -r {}/*\n".format(session_output))

    def start_script_generation(self):
        """
        Based on the desired pipeline, calls the corresponding funciton to start the script generation
        """

        #call the function that will generate the scripts
        self.PIPELINE_MAP[self.pipeline](self.tmp_input_dir, self.tmp_output_dir)

        print("Wrote {} scripts for {}".format(self.num_scripts, self.pipeline))

    def write_slurm_script(self):
        """
        Based on the processing that was run, generate the slurm script that should be used to submit the jobs
        """

        print("Writing slurm script to {}".format(self.slurm_options['output']))
        
        #write the slurm script. start by opening a file
        with open(self.slurm_options['output'], 'w') as slurm:
            slurm.write("#!/bin/bash\n\n")
            slurm.write("#SBATCH --mail-user={}\n".format(self.slurm_options['email']))
            slurm.write("#SBATCH --mail-type=FAIL\n")
            slurm.write("#SBATCH --nodes={}\n".format(self.slurm_options['num_nodes']))
            slurm.write("#SBATCH --mem={}\n".format(self.slurm_options['memory_per_job']))
            slurm.write("#SBATCH --time={}\n".format(self.slurm_options['time']))
            slurm.write("#SBATCH --array=1-{}%{}\n".format(self.num_scripts, self.slurm_options['num_jobs']))
            slurm.write("#SBATCH --output={}/logs_%A_%a.out\n".format(self.log_dir))
            slurm.write("\n")

            if self.args.pipeline == 'freesurfer' or self.args.pipeline == 'NODDI':
                slurm.write("module load FreeSurfer/7.2.0 GCC/6.4.0-2.28 OpenMPI/2.1.1 FSL/5.0.10\n")
                slurm.write("source $FREESURFER_HOME/FreeSurferEnv.sh\n")
                slurm.write("export FSL_DIR=/accre/arch/easybuild/software/MPI/GCC/6.4.0-2.28/OpenMPI/2.1.1/FSL/5.0.10/fsl\n")
                slurm.write("\n")

            slurm.write("SCRIPT_DIR={}/\n".format(self.script_dir))
            slurm.write("bash ${}${}.sh\n".format('{' + 'SCRIPT_DIR' + '}', '{' + 'SLURM_ARRAY_TASK_ID' + '}'))

        print("Finished writing slurm script.")



        # self.slurm_options = {
        #     'email': self.args.email,
        #     'time': self.args.time,
        #     'memory_per_job': self.args.memory_per_job,
        #     'num_jobs': self.args.num_jobs,
        #     'output': self.slurm_path
        # }

    def write_paralellization_script(self):
        """
        Python script to parallelize the running of jobs on a local machine
        """

        print("Writing parallelization script to {}".format(self.slurm_options['output']))
        
        #write the slurm script. start by opening a file
        with open(self.slurm_options['output'], 'w') as slurm:
            # from pathlib import Path
            # import subprocess
            # from tqdm import tqdm
            # import joblib

            # def run_script(input):
            #     script, log = input[0], input[1]
                
            #     with open(log, 'w') as file:
            #         run_cmd = "bash {}".format(script)
            #         subprocess.run(run_cmd, shell=True, stdout=file)

            # scriptsdir = self.script_dir #format
            # logsdir = self.log_dir #format

            # files = [(str(scriptsdir/("{}.sh".format(x))),str(logsdir/("{}.txt".format(x)))) for x in range(1, int(self.num_scripts))] #format

            # joblib.Parallel(n_jobs=int(self.slurm_options['num_jobs']))(joblib.delayed(run_script)(file) for file in tqdm(files)) #format

            slurm.write("from pathlib import Path\n")
            slurm.write("import subprocess\n")
            slurm.write("from tqdm import tqdm\n")
            slurm.write("import joblib\n")
            slurm.write("\n")
            slurm.write("def run_script(input):\n")
            slurm.write("    script, log = input[0], input[1]\n")
            slurm.write("    with open(log, 'w') as file:\n")
            slurm.write("        run_cmd = 'bash {}'.format(script)\n")
            slurm.write("        subprocess.run(run_cmd, shell=True, stdout=file)\n")
            slurm.write("\n")
            slurm.write("scriptsdir = Path('{}')\n".format(self.script_dir))
            slurm.write("logsdir = Path('{}')\n".format(self.log_dir))
            slurm.write("\n")
            slurm.write("files = [(str(scriptsdir/('{{}}.sh'.format(x))),str(logsdir/('{{}}.txt'.format(x)))) for x in range(1, int({})+1)]\n".format(self.num_scripts))
            slurm.write("\n")
            slurm.write("joblib.Parallel(n_jobs=int({}))(joblib.delayed(run_script)(file) for file in tqdm(files))\n".format(self.slurm_options['num_jobs']))
        print("Finished writing parallelization script.")

def main():
    """
    Main function for the script generation.
    """

    print('ASSUMES THAT YOU HAVE SET UP AN SSH-KEY TO SCP TO landman01/landman01 WITHOUT A PASSWORD. IF YOU DO NOT HAVE THIS, PLEASE SET IT UP AND TEST IT OUT BEFORE SUBMITTING JOBS TO ACCRE.')

    args = pa()
    sg = ScriptGenerator(args)

    #first, generate the scripts
    sg.start_script_generation()

    #now also output the slurm script
    if not sg.args.no_accre:
        sg.write_slurm_script()
    else:
        print("Not writing slurm script because --no-accre flag was set.")
        print("Instead writing a python parallelization file")
        sg.write_paralellization_script()




#if the custom_output_dir is empty string, then the output will be written to the nfs2 directories
    #make sure to print out a warning if a custom output was specified that things will need to be linked to /nfs2
        #alternatively, you can add a function that links the custom output to /nfs2
#throws an error and exits if you do not have permissions to create directories or write to directories
#outptus a list of sessions that are missing the necessary files and what they are missing
    


if __name__ == "__main__":
    main()
