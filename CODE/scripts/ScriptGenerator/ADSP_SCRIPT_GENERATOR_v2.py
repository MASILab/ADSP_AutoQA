"""
Author: Michael Kim
Email: michael.kim@vanderbilt.edu
June 2024
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
import datetime
import yaml

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
    p.add_argument('--skull_stripped', action='store_true', help="For UNest: if you want to run it with the skull stripped T1; Also for PreQual if you want to skull trip the T1 for synb0")
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
    p.add_argument('--custom_simg_path', default='', type=str, nargs='+', help="Path to the custom singularity image file for whatever pipeline you are running")
    p.add_argument('--no_accre', action='store_true', help="If you want to run not on ACCRE, but on a local machine. Will instead output a python file to parallelize the processing.")
    p.add_argument('--custom_atlas_input_dir', type=str, default='', help="For EVE3WMAtlas: custom path to the directory where the atlas inputs are stored")
    p.add_argument('--no_scp', action='store_true', help="If you do not want to scp the outputs to ACCRE (i.e. cp -L instead of scp)")
    p.add_argument('--src_server', type=str, default='landman01.accre.vanderbilt.edu', help="Source server for scp")
    p.add_argument('--separate_prequal', action='store_true', help="If you want to run all DWI separately through PreQual, regardless of whether the can be run together")
    p.add_argument('--force_combine_prequal', action='store_true', help="If you want to forcibly run all the dwis in a session together for a dataset")
    p.add_argument('--custom_home', default='', type=str, help="Path to the custom home directory for the user if --no_accre is set (for Tractseg)")
    p.add_argument('--TE_tolerance_PQ', default=0.01, type=float, help="Tolerance for TE differences in determining scan similarity for PreQual")
    p.add_argument('--TR_tolerance_PQ', default=0.01, type=float, help="Tolerance for TR differences in determining scan similarity for PreQual")
    p.add_argument('--accre_gpu', action='store_true', help="If you want to run the pipeline on ACCRE with a GPU (note, you may need to specify simgs with a CUDA installation)")
    p.add_argument('--tractseg_make_DTI', action='store_true', help="If you want to make the DTI files for Tractseg")
    p.add_argument('--tractseg_DTI_use_all_shells', action='store_true', help="If you want to forego the shelling for calculating DTI measures")
    p.add_argument('--select_sessions_list', type=str, default='', help="If you want to run only a subset of scans/sessions, provide a path to a list of sessions directories to run FOR THAT SPECIFIC PIPELINE. (e.g. for TICV, provide a list of T1s, for PreQual provide a list of dwi dirs, for EVE3 provide a list of PreQual directories, etc.)")
    p.add_argument('--infant_csv', type=str, default='', help="Path to the demographics csv file that is required as input for the infantFS pipeline")
    return p.parse_args()

class ScriptGeneratorSetup:
    """
    Class to maintain paths and options for the script generation.
    """

    def __init__(self, args):
        
        #mapping for the script generator functions
        # self.PIPELINE_MAP = {
        #     "freesurfer": self.freesurfer_script_generate,
        #     "PreQual": self.PreQual_script_generate,
        #     "SLANT-TICV": self.SLANT_TICV_script_generate,
        #     "EVE3WMAtlas": self.EVE3Registration_script_generate,
        #     "MNI152WMAtlas": self.MNI152Registration_script_generate,
        #     "UNest": self.UNest_script_generate,
        #     "synthstrip": self.synthstrip_script_generate,
        #     "tractseg": self.tractseg_script_generate,
        #     "MaCRUISE": self.macruise_script_generate,
        #     "ConnectomeSpecial": self.connectome_special_script_generate,
        #     "FrancoisSpecial": self.francois_special_script_generate,
        #     "Biscuit": self.biscuit_script_generate,
        #     'NODDI': self.noddi_script_generate
        # }

        #mapping for the script generator class instances
        self.GENERATOR_MAP = {
            "PreQual": PreQualGenerator,
            "freesurfer": FreeSurferGenerator,
            "SLANT-TICV": SLANT_TICVGenerator,
            "EVE3WMAtlas": EVE3WMAtlasGenerator,
            "MNI152WMAtlas": MNI152WMAtlasGenerator,
            "UNest": UNestGenerator,
            "synthstrip": SynthstripGenerator,
            "MaCRUISE": MaCRUISEGenerator,
            "Biscuit": BiscuitGenerator,
            'ConnectomeSpecial': ConnectomeSpecialGenerator,
            "tractseg": TractsegGenerator,
            "NODDI": NODDIGenerator,
            "FrancoisSpecial": FrancoisSpecialGenerator,
            "DWI_plus_Tractseg": DWI_plus_TractsegGenerator,
            "BedpostX_plus_Tractseg": BedpostX_plus_DWI_plus_TractsegGenerator,
            "freewater": FreewaterGenerator,
            "FreesurferWhiteMatterMask": FreesurferWhiteMatterMaskGenerator,
            "rawEVE3WMAtlas": RawEVE3WMAtlasGenerator,
            "infantFS": InfantFreesurferGenerator

            ##TODO: add the other pipelines

        }

        self.generator = None

        if args.pipeline not in self.GENERATOR_MAP.keys():
            print("Available Pipelines: ", self.GENERATOR_MAP.keys())
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
            #make sure that this can take in a list of simg files as well
            self.simg = args.custom_simg_path
            for simg in self.simg:
                assert Path(simg).exists(), "Error: Custom singularity image file {} does not exist".format(str(self.simg))
            if len(self.simg) == 1:
                self.simg = self.simg[0]
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

        self.count = None

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
        freesurfer_simg = '/nobackup/p_masi/Singularities/freesurfer_7.2.0.sif'
        ticv_simg = '/nobackup/p_masi/Singularities/nssSLANT_v1.2.simg'
        wmatlas_simg = '/nobackup/p_masi/Singularities/WMAtlas_v1.3.simg'
        scilpy_simg = '/nobackup/p_masi/Singularities/scilus_1.5.0.sif'
        tractseg_simg = '/nobackup/p_masi/Singularities/tractsegv2.simg'
        try:
            mapping = {'SLANT-TICV': ticv_simg,
                        'PreQual': '/nobackup/p_masi/Singularities/PreQual_v1.0.8.simg',
                        'freesurfer': freesurfer_simg,
                        'EVE3WMAtlas': wmatlas_simg,
                        'MNI152WMAtlas': wmatlas_simg,
                        'UNest': '/nobackup/p_masi/Singularities/UNest.sif',
                        'tractseg': [tractseg_simg, scilpy_simg] if not self.args.tractseg_make_DTI else [tractseg_simg, scilpy_simg, wmatlas_simg],
                        'MaCRUISE': "/nobackup/p_masi/Singularities/macruise_classern_v3.2.0.simg",
                        'FrancoisSpecial': '/nobackup/p_masi/Singularities/singularity_francois_special_v1.sif',
                        'ConnectomeSpecial': '/nobackup/p_masi/Singularities/ConnectomeSpecial_v1.3.sif',
                        'Biscuit': '/nobackup/p_masi/Singularities/biscuit_FC_v2.2.sif',
                        'NODDI': '/nobackup/p_masi/Singularities/tractoflow_2.2.1_b9a527_2021-04-13.sif',
                        'freewater': '/nobackup/p_masi/Singularities/FreeWaterEliminationv2.sif',
                        'synthstrip': '/nobackup/p_masi/Singularities/synthstrip.1.5.sif',
                        'DWI_plus_Tractseg': [tractseg_simg, scilpy_simg, wmatlas_simg],
                        'BedpostX_plus_Tractseg': [tractseg_simg, scilpy_simg, wmatlas_simg],
                        'FreesurferWhiteMatterMaskGenerator': '/nobackup/p_masi/Singularities/FreesurferWhiteMatterMask.simg',#[freesurfer_simg, wmatlas_simg]
                        'rawEVE3WMAtlas': wmatlas_simg,
                        "infantFS" : "/nobackup/p_masi/Singularities/infantfs_v0.0.sif",
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

        #For prequal, freesurfer, biscuit, check for the FS license
        if self.args.pipeline == "PreQual" or self.args.pipeline == 'freeesurfer' or self.args.pipeline == 'Biscuit':
            assert Path(self.args.freesurfer_license_path).exists(), "Error: freesurfer license file does not exist at {}".format(str(self.args.freesurfer_license_path))

        #check to make sure that the scripts and logs directories are empty
        assert len(list(self.script_dir.iterdir())) == 0, "Error: scripts dir {} is not empty".format(str(self.script_dir))
        assert len(list(self.log_dir.iterdir())) == 0, "Error: logs dir {} is not empty".format(str(self.log_dir))

        #check that the dataset, inputs, outputs, derivatives, scripts, logs, and simg file all exist

    def start_script_generation(self):
        """
        Based on the desired pipeline, calls the corresponding function to start the script generation
        """

        #call the function that will generate the scripts
        #self.PIPELINE_MAP[self.pipeline](self.tmp_input_dir, self.tmp_output_dir)

        print("Starting script generation for {} pipeline".format(self.pipeline))

        #start the script generation
        self.generator = self.GENERATOR_MAP[self.pipeline](self)
        print("Wrote {} scripts for {}".format(self.generator.count, self.pipeline))

        #output the missing data to a csv file
        missing_csv_name = '{}_{}_missing_data.csv'.format(self.root_dataset_path.name, self.pipeline)
        self.generator.missing_data.to_csv(missing_csv_name)
        print("Wrote missing data to {}".format(missing_csv_name))

    def write_paralellization_script(self):
        """
        Write a python script to parallelize the running of jobs on a local machine
        """

        print("Writing parallelization script to {}".format(self.slurm_options['output']))

        #write the slurm script. start by opening a file
        with open(self.slurm_options['output'], 'w') as slurm:
            # import subprocess

            # def run_script(input):
            #     script, log = input[0], input[1]

            #     with open(log, 'w') as file:
            #         run_cmd = "bash {}".format(script)
            #         subprocess.run(run_cmd, shell=True, stdout=file)
            # logsdir = self.log_dir #format

            slurm.write("from pathlib import Path\n")
            slurm.write("import subprocess\n")
            slurm.write("from tqdm import tqdm\n")
            slurm.write("import joblib\n")
            slurm.write("\n")
            slurm.write("def run_script(input):\n")
            slurm.write("    script, log = input[0], input[1]\n")
            slurm.write("    with open(log, 'w') as file:\n")
            slurm.write("        run_cmd = 'bash {}'.format(script)\n")
            slurm.write("        subprocess.run(run_cmd, shell=True, stdout=file, stderr=subprocess.STDOUT)\n")
            slurm.write("\n")
            slurm.write("scriptsdir = Path('{}')\n".format(self.script_dir))
            slurm.write("logsdir = Path('{}')\n".format(self.log_dir))
            slurm.write("\n")
            slurm.write("files = [(str(scriptsdir/('{{}}.sh'.format(x))),str(logsdir/('{{}}.txt'.format(x)))) for x in range(1, int({})+1)]\n".format(self.generator.count))
            slurm.write("\n")
            slurm.write("joblib.Parallel(n_jobs=int({}))(joblib.delayed(run_script)(file) for file in tqdm(files))\n".format(self.slurm_options['num_jobs']))
        print("Finished writing parallelization script.")

    def write_slurm_script(self):
        """
        Based on the processing that was run, generate the slurm script that should be used to submit the jobs
        """

        print("Writing slurm script to {}".format(self.slurm_options['output']))

        #write the slurm script. start by opening a file
        with open(self.slurm_options['output'], 'w') as slurm:
            slurm.write("#!/bin/bash\n\n")
            slurm.write("#SBATCH --mail-user={}\n".format(self.slurm_options['email']))
            slurm.write("#SBATCH --mail-type=FAIL,END\n")
            slurm.write("#SBATCH --nodes={}\n".format(self.slurm_options['num_nodes']))
            slurm.write("#SBATCH --mem={}\n".format(self.slurm_options['memory_per_job']))
            slurm.write("#SBATCH --time={}\n".format(self.slurm_options['time']))
            slurm.write("#SBATCH --array=1-{}%{}\n".format(self.generator.count, self.slurm_options['num_jobs']))
            slurm.write("#SBATCH --output={}/logs_%A_%a.out\n".format(self.log_dir))
            if self.args.accre_gpu:
                slurm.write("#SBATCH --account=vuiis_masi_gpu_acc\n")
                slurm.write("#SBATCH --partition=pascal\n")
                slurm.write("#SBATCH --gres=gpu:1\n")
            slurm.write("\n")

            if self.args.pipeline == 'freesurfer' or self.args.pipeline == 'NODDI':
                slurm.write("module load FreeSurfer/7.2.0 GCC/6.4.0-2.28 OpenMPI/2.1.1 FSL/5.0.10\n")
                slurm.write("source $FREESURFER_HOME/FreeSurferEnv.sh\n")
                slurm.write("export FSL_DIR=/accre/arch/easybuild/software/MPI/GCC/6.4.0-2.28/OpenMPI/2.1.1/FSL/5.0.10/fsl\n")
                slurm.write("\n")

            slurm.write("SCRIPT_DIR={}/\n".format(self.script_dir))
            slurm.write("bash ${}${}.sh\n".format('{' + 'SCRIPT_DIR' + '}', '{' + 'SLURM_ARRAY_TASK_ID' + '}'))

        print("Finished writing slurm script.")

class ScriptGenerator:

    ## ABSTRACT CLASSES TO BE IMPLEMENTED BY THE RESPECTIVE CHILD CLASSES ##
        
    def freesurfer_script_generate(self):
        """
        Abstract method to be implemented by the child class
        """
        pass
        #raise NotImplementedError("Error: freesurfer_script_generate not implemented")
    
    def PreQual_script_generate(self):
        """
        Abstract method to be implemented by the child class
        """
        pass
        #raise NotImplementedError("Error: PreQual_script_generate not implemented")
    
    def SLANT_TICV_script_generate(self):
        """
        Abstract method to be implemented by the child class
        """
        pass
        #raise NotImplementedError("Error: SLANT_TICV_script_generate not implemented")
    
    def EVE3Registration_script_generate(self):
        """
        Abstract method to be implemented by the child class
        """
        pass
        #raise NotImplementedError("Error: EVE3Registration_script_generate not implemented")
    
    def MNI152Registration_script_generate(self):
        """
        Abstract method to be implemented by the child class
        """
        pass
        #raise NotImplementedError("Error: MNI152Registration_script_generate not implemented")

    def UNest_script_generate(self):
        """
        Abstract method to be implemented by the child class
        """
        pass
        #raise NotImplementedError("Error: UNest_script_generate not implemented")
    
    def synthstrip_script_generate(self):
        """
        Abstract method to be implemented by the child class
        """
        pass
        #raise NotImplementedError("Error: synthstrip_script_generate not implemented")
    
    def tractseg_script_generate(self):
        """
        Abstract method to be implemented by the child class
        """
        pass
        #raise NotImplementedError("Error: tractseg_script_generate not implemented")
    
    def macruise_script_generate(self):
        """
        Abstract method to be implemented by the child class
        """
        pass
        #raise NotImplementedError("Error: macruise_script_generate not implemented")
    
    def connectome_special_script_generate(self):
        """
        Abstract method to be implemented by the child class
        """
        pass
        #raise NotImplementedError("Error: connectome_special_script_generate not implemented")
    
    def francois_special_script_generate(self):
        """
        Abstract method to be implemented by the child class
        """
        pass
        #raise NotImplementedError("Error: francois_special_script_generate not implemented")
    
    def biscuit_script_generate(self):
        """
        Abstract method to be implemented by the child class
        """
        pass
        #raise NotImplementedError("Error: biscuit_script_generate not implemented")
    
    def noddi_script_generate(self):
        """
        Abstract method to be implemented by the child class
        """
        pass
        #raise NotImplementedError("Error: noddi_script_generate not implemented")
    
    def freewater_script_generate(self):
        """
        Abstract method to be implemented by the child class
        """
        pass
        #raise NotImplementedError("Error: freewater_script_generate not implemented")

    def dwi_plus_tractseg_script_generate(self):
        """
        Abstract method to be implemented by the child class
        """
        pass
        #raise NotImplementedError("Error: dwi_plus_tractseg_script_generate not implemented")

    def bedpostx_plus_dwi_plus_tractseg_script_generate(self):
        """
        Abstract method to be implemented by the child class
        """
        pass
        #raise NotImplementedError("Error: bedpostx_plus_dwi_plus_tractseg_script_generate not implemented")

    def freesurfer_white_matter_mask_script_generate(self):
        """
        Abstract method to be implemented by the child class
        """
        pass
        #raise NotImplementedError("Error: freesurfer_white_matter_mask_script_generate not implemented")

    def rawEVE3Registration_script_generate(self):
        """
        Abstract method to be implemented by the child class
        """
        pass
        #raise NotImplementedError("Error: EVE3Registration_script_generate not implemented")
    
    def infant_freesurfer_script_generate(self):
        """
        Abstract method to be implemented by the child class
        """
        pass
        #raise NotImplementedError("Error: infant_freesurfer_script_generate not implemented")

    ## END ABSTRACT CLASSES ##

    def __init__(self, setup_object):
        
        self.setup = setup_object

        #these should be dictionaries or lists (or a dict of dicts!) that contain the inputs and outputs for the pipeline
            #i.e. the names and the paths
        self.inputs_dict = None
        self.outputs_dict = None
        self.yml_dict = None #yml files used for provenance tracking
        self.count = 0 #count of the number of scripts that have been generated
        self.missing_data = pd.DataFrame(columns=['sub', 'ses', 'acq', 'run', 'missing']) #missing data for the pipeline

        #mapping for the pipeline specific script generation functions
        self.PIPELINE_GENERATE_MAP = {
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
            'NODDI': self.noddi_script_generate,
            'freewater': self.freewater_script_generate,
            "DWI_plus_Tractseg": self.dwi_plus_tractseg_script_generate,
            "BedpostX_plus_Tractseg": self.bedpostx_plus_dwi_plus_tractseg_script_generate,
            "FreesurferWhiteMatterMask": self.freesurfer_white_matter_mask_script_generate,
            "rawEVE3WMAtlas": self.rawEVE3Registration_script_generate,
            "infantFS": self.infant_freesurfer_script_generate
        }

        self.necessary_outputs = {
            "PreQual": [
                'PREPROCESSED/dwmri.nii.gz',
                'PREPROCESSED/dwmri.bval',
                'PREPROCESSED/dwmri.bvec',
                'PREPROCESSED/mask.nii.gz',
                'PDF/dtiQA.pdf'
            ],
            'freeseurfer': [
                'freesurfer/scripts/recon-all.log' #any more to add? ask Kurt?
            ],
            'SLANT-TICV': [],
            'UNest': [],
            'synthstrip': [],
            'EVE3WMAtlas': [
                "dwmri%diffusionmetrics.csv",
                "dwmri%ANTS_t1tob0.txt",
                "dwmri%0GenericAffine.mat",
                "dwmri%1InverseWarp.nii.gz",
                "dwmri%Atlas_JHU_MNI_SS_WMPM_Type-III.nii.gz"              
            ],
            'MNI152WMAtlas': [
                "dwmri%diffusionmetrics.csv",
                "dwmri%ANTS_t1tob0.txt",
                "dwmri%0GenericAffine.mat",
                "dwmri%1InverseWarp.nii.gz",             
            ],
            'MaCRUISE': [
                "Output/SegRefine/SEG_refine.nii.gz",
                "Output/Surfaces_FreeView/target_image_GMimg_innerSurf.asc",
                'Output/Surfaces_MNI/target_image_GMimg_innerSurf.asc'
            ],
            'Biscuit': [
                'PDF/biscuit.pdf'
            ],
            'ConnectomeSpecial': ["CONNECTOME_Weight_MeanFA_NumStreamlines_10000000_Atlas_SLANT.csv", "CONNECTOME_Weight_NUMSTREAMLINES_NumStreamlines_10000000_Atlas_SLANT.csv",
                "graphmeasures.json", "graphmeasures_nodes.json", "log.txt", "tracks_10000000_compressed.tck", "ConnectomeQA.png",
                "CONNECTOME_NUMSTREAM.npy", "CONNECTOME_LENGTH.npy", "CONNECTOME_FA.npy", "b0.nii.gz", "atlas_slant_subj.nii.gz"
            ],
            'tractseg': [
                #{'targ_name': 'bundles'},
                #{'targ_name': 'measures'}
            ],
            'NODDI': [
                'config.pickle',
                'FIT_dir.nii.gz',
                'FIT_ICVF.nii.gz',
                'FIT_ISOVF.nii.gz',
                'FIT_OD.nii.gz'
            ],
            'freewater': [],
            'FrancoisSpecial': [],
            'DWI_plus_Tractseg': [],
            'BedpostX_plus_Tractseg': [],
            'freesurfer': [],
            'FreesurferWhiteMatterMask': [],
            'rawEVE3WMAtlas': [],
            'infantFS': []
            ### TODO: add the necessary outputs for the other pipelines
                #this cannot be static for all the outputs for all pipelines, only some of them
        }

    def find_t1s(self):
        """
        Return a list of T1s for the dataset
        """
        if self.setup.args.select_sessions_list == '':
            print("Finding all T1s...")
            find_cmd = "find {} -mindepth 3 -maxdepth 4 \( -type f -o -type l \) -name '*T1w.nii.gz' | grep -v derivatives".format(str(self.setup.root_dataset_path))
            result = subprocess.run(find_cmd, shell=True, capture_output=True, text=True).stdout
            files = result.strip().splitlines()
        else:
            lst_path = Path(self.setup.args.select_sessions_list)
            assert lst_path.exists(), "Error: select_sessions_list file {} that was specified does not exist".format(str(lst_path))
            print("Finding all T1s from select_sessions_list {}...".format(str(lst_path)))
            with open(lst_path, 'r') as f:
                files = f.readlines()
            files = [x.strip() for x in files]

        return files

    def get_sub_ses_t1s(self, sub, ses):
        """
        Given a root dataset path, subject, and session, returns a list of T1 paths
        """
        root = self.setup.root_dataset_path
        sesdir = root/sub/ses/'anat'
        t1s = list(sesdir.glob('*T1w.nii.gz'))
        return t1s


    def find_dwis(self):
        """
        Return a list of all DWIs in the dataset
        """
        if self.setup.args.select_sessions_list == '':
            print("Finding all DWIs...")
            find_cmd = "find {} -mindepth 3 -maxdepth 4 \( -type f -o -type l \) -name '*dwi.nii.gz' | grep -v derivatives".format(str(self.setup.root_dataset_path))
            result = subprocess.run(find_cmd, shell=True, capture_output=True, text=True).stdout
            files = result.strip().splitlines()
        else:
            lst_path = Path(self.setup.args.select_sessions_list)
            assert lst_path.exists(), "Error: select_sessions_list file {} that was specified does not exist".format(str(lst_path))
            print("Finding all DWIs from select_sessions_list {}...".format(str(lst_path)))
            with open(lst_path, 'r') as f:
                files = f.readlines()
            files = [x.strip() for x in files]
        return files

    def get_PreQual_dirs(self):
        """
        Returns a list of all the PreQual directories in the dataset
        """
        if self.setup.args.select_sessions_list == '':
            print("Finding all PreQuals...")
            find_cmd = "find {} -mindepth 2 -maxdepth 3 -type d -name 'PreQual*'".format(str(self.setup.dataset_derivs))
            result = subprocess.run(find_cmd, shell=True, capture_output=True, text=True).stdout
            dirs = result.strip().splitlines()
        else:
            lst_path = Path(self.setup.args.select_sessions_list)
            assert lst_path.exists(), "Error: select_sessions_list file {} that was specified does not exist".format(str(lst_path))
            print("Finding all PreQuals from select_sessions_list {}...".format(str(lst_path)))
            with open(lst_path, 'r') as f:
                dirs = f.readlines()
            dirs = [x.strip() for x in dirs]
        return dirs

    def get_EVE3WMAtlas_dirs(self):
        """
        Returns a list of all the WMAtlasEVE3 directories in the dataset
        """
        if self.setup.args.select_sessions_list == '':
            print("Finding all EVE3WMAtlas directories...")
            find_cmd = "find {} -mindepth 2 -maxdepth 3 -type d -name 'WMAtlasEVE3*'".format(str(self.setup.dataset_derivs))
            result = subprocess.run(find_cmd, shell=True, capture_output=True, text=True).stdout
            dirs = result.strip().splitlines()
        else:
            lst_path = Path(self.setup.args.select_sessions_list)
            assert lst_path.exists(), "Error: select_sessions_list file {} that was specified does not exist".format(str(lst_path))
            print("Finding all WMAtlasEVE3s from select_sessions_list {}...".format(str(lst_path)))
            with open(lst_path, 'r') as f:
                dirs = f.readlines()
            dirs = [x.strip() for x in dirs]
        return dirs

    def get_dwi_dirs(self):
        """
        Return a list of all raw dwi directories in the dataset
        """
        if self.setup.args.select_sessions_list == '':
            print("Finding all DWI directories...")
            find_cmd = "find {} -mindepth 2 -maxdepth 3 -type d -name '*dwi' | grep -v derivatives".format(str(self.setup.root_dataset_path))
            result = subprocess.run(find_cmd, shell=True, capture_output=True, text=True).stdout
            dirs = result.strip().splitlines()
        else:
            lst_path = Path(self.setup.args.select_sessions_list)
            assert lst_path.exists(), "Error: select_sessions_list file {} that was specified does not exist".format(str(lst_path))
            print("Finding all DWI directories from select_sessions_list {}...".format(str(lst_path)))
            with open(lst_path, 'r') as f:
                dirs = f.readlines()
            dirs = [x.strip() for x in dirs]
        return dirs

    def get_PreQual_dirs_from_dwi(self, dwis, similar=True):
        """
        Given a list of dwis, return a list of the PreQual directory paths
        """
        
        if similar:
            sub, ses, acq, run = self.get_BIDS_fields_dwi(dwis[0])
            PQdir = self.setup.dataset_derivs  / sub / ses / 'PreQual'
            return [PQdir]
        else:
            PQdirs = []
            for dwi in dwis:
                sub, ses, acq, run = self.get_BIDS_fields_dwi(dwi)
                PQdir = self.setup.dataset_derivs / sub / ses / 'PreQual{}{}'.format(acq, run)
                PQdirs.append(PQdir)
            return PQdirs
    
    def get_BIDS_fields_dwi(self, dwi_path):
        """
        Return the BIDS tags for a dwi scan
        """
        pattern = r'(sub-\w+)(?:_(ses-\w+))?(?:_(acq-\w+))?(?:_(run-\d{1,3}))?_dwi'
        matches = re.findall(pattern, dwi_path.name)
        return matches[0]
        #sub, ses, acq, run = matches[0][0], matches[0][1], matches[0][2], matches[0][3]
        #return sub, ses, acq, run

    def get_BIDS_fields_t1(self, t1_path):
        pattern = r'(sub-\w+)(?:_(ses-\w+))?(?:_(acq-\w+))?(?:_(run-\d{1,3}))?_T1w'
        matches = re.findall(pattern, str(t1_path).split('/')[-1])
        sub, ses, acq, run = matches[0][0], matches[0][1], matches[0][2], matches[0][3]
        return sub, ses, acq, run

    def check_PQ_outputs(self, pqdir):
        """
        Check to see if the PreQual outputs already exist for the given PreQual directory
        """
        preproc_dir = pqdir/("PREPROCESSED")
        dwi = preproc_dir/("dwmri.nii.gz")
        bval = preproc_dir/("dwmri.bval")
        bvec = preproc_dir/("dwmri.bvec")
        mask = preproc_dir/("mask.nii.gz")

        #also check to see if the PDF is there
        pdf_dir = pqdir/("PDF")
        pdf = pdf_dir/("dtiQA.pdf")

        hasPQ = True
        hasPDF = True
        if not dwi.exists() or not bval.exists() or not bvec.exists() or not mask.exists():
            hasPQ = False
        if not pdf.exists():
            hasPDF = False
        
        return hasPQ, hasPDF

    def has_EVE3WMAtlas_outputs(self, WMAtlas_dir):
        """
        Returns True if WMAtlasEVE3 outputs exist
        """
        if ((WMAtlas_dir/("dwmri%diffusionmetrics.csv")).exists() and (WMAtlas_dir/("dwmri%ANTS_t1tob0.txt")).exists() and
                    (WMAtlas_dir/("dwmri%0GenericAffine.mat")).exists() and (WMAtlas_dir/("dwmri%1InverseWarp.nii.gz")).exists() and
                    (WMAtlas_dir/("dwmri%Atlas_JHU_MNI_SS_WMPM_Type-III.nii.gz")).exists() and (WMAtlas_dir/("dwmri%fa.nii.gz")).exists() and
                    (WMAtlas_dir/("dwmri%md.nii.gz")).exists() and (WMAtlas_dir/("dwmri%rd.nii.gz")).exists() and (WMAtlas_dir/("dwmri%ad.nii.gz")).exists()):
            return True
        return False

    def has_MNI152WMAtlas_outputs(self, WMAtlas_dir):
        """
        Returns True if MNI152 WMAtlas outputs exist
        """
        if ((WMAtlas_dir/("dwmri%diffusionmetrics.csv")).exists() and (WMAtlas_dir/("dwmri%ANTS_t1tob0.txt")).exists() and
                    (WMAtlas_dir/("dwmri%0GenericAffine.mat")).exists() and (WMAtlas_dir/("dwmri%1InverseWarp.nii.gz")).exists()):
            return True
        return False

    def has_NODDI_outputs(self, noddi_dir):
        """
        Returns True if NODDI outputs exist
        """

        if ((noddi_dir/("config.pickle")).exists() and (noddi_dir/("FIT_dir.nii.gz")).exists() and
                    (noddi_dir/("FIT_ICVF.nii.gz")).exists() and (noddi_dir/("FIT_ISOVF.nii.gz")).exists() and
                    (noddi_dir/("FIT_OD.nii.gz")).exists()):
            return True
        return False

    def has_ConnectomeSpecial_outputs(self, cs_dir):
        """
        Returns True if ConnectomeSpecial outputs exist
        """

        files = ["CONNECTOME_Weight_MeanFA_NumStreamlines_10000000_Atlas_SLANT.csv", "CONNECTOME_Weight_NUMSTREAMLINES_NumStreamlines_10000000_Atlas_SLANT.csv",
                "CONNECTOME_Weight_MEANLENGTH_NumStreamlines_10000000_Atlas_SLANT.csv",
                "graphmeasures.json", "graphmeasures_nodes.json", "tracks_10000000_compressed.tck", "ConnectomeQA.png",
                "CONNECTOME_NUMSTREAM.npy", "CONNECTOME_LENGTH.npy", "CONNECTOME_FA.npy", "atlas_slant_subj.nii.gz",
                "CONNECTOME_Weight_MeanFA_NumStreamlines_10000000_Atlas_SLANT_w_stem.csv", "CONNECTOME_Weight_NUMSTREAMLINES_NumStreamlines_10000000_Atlas_SLANT_w_stem.csv",
                "CONNECTOME_Weight_MEANLENGTH_NumStreamlines_10000000_Atlas_SLANT_w_stem.csv",
                "graphmeasures_w_stem.json", "graphmeasures_nodes_w_stem.json",
                "CONNECTOME_NUMSTREAM_stem.npy", "CONNECTOME_LENGTH_stem.npy", "CONNECTOME_FA_stem.npy", "atlas_slant_subj_stem.nii.gz"
            ]
        if all([(cs_dir/f).exists() for f in files]):
            return True
        return False

    def has_freewater_outputs(self, freewater_dir):
        """
        Checks to see if the freewater outputs exist in a directory
        """

        fa = freewater_dir/("freewater_single_fa.nii.gz")
        md = freewater_dir/("freewater_single_md.nii.gz")
        rd = freewater_dir/("freewater_single_rd.nii.gz")
        ad = freewater_dir/("freewater_single_ad.nii.gz")
        #for some reason, these are output differently for single vs multishell
        fw = freewater_dir/("freewater_single.nii.gz")
        #frac = freewater_dir/("freewater_fraction.nii.gz")

        multis = [(x.parent) / x.name.replace('single', 'multi') for x in [fa, md, rd, ad, fw]]

        if not fa.exists() or not md.exists() or not rd.exists() or not ad.exists() or not fw.exists():
            if not all([x.exists() for x in multis]):
                return False
        return True

    def has_infantFS_outputs(self, infant_fs_dir):
        """
        Given the path to an infant FS directory, returns True if the outputs exist, False otherwise
        """
        filenames = [
            "./lh.dwhite.ribbon.mgz",
            "./aseg.nii.gz",
            "./aparc+aseg.mgz",
            "./lh.ribbon.mgz",
            "./norm.nii.gz",
            "./rh.ribbon.mgz",
            "./wm.mgz",
            "./brain.finalsurfs.mgz",
            "./lh.dpial.ribbon.mgz",
            "./rh.dpial.ribbon.mgz",
            "./norm.mgz",
            "./rh.dwhite.ribbon.mgz",
            "./aseg.count.txt",
            "./aseg.mgz",
            "./ribbon.mgz"
        ]
        files = [infant_fs_dir/'mri'/f for f in filenames]
        return all([f.exists() for f in files])

    def is_missing_DTI_tractseg(self, tsdir):
        """
        Returns true if the DTI tractseg outputs are missing
        """

        #get the bundles and measures dirs
        bundles = tsdir/("bundles")
        measures = tsdir/("measures")

        if not bundles.exists() or not measures.exists():
            #print(tsdir, "False")
            return False

        #count the files in each one to make sure that there is the correct number of bundles and measures
        num_bundles = len(list(bundles.glob("*.tck")))
        num_measures = len(list(measures.glob("*DTI.json")))

        #arbitrary numbers
        if num_bundles >= 70 and num_measures <= 10:
            #print(tsdir, 'Missing')
            #return 'Missing Files'
            return True
        return False

    def has_Tractseg_outputs(self, tsdir):
        """
        Returns True if Tractseg outputs exist
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

        #arbitrary numbers
        if num_bundles <= 42 or num_measures <= 114:
            #print(tsdir, 'Missing')
            #return 'Missing Files'
            return False
        return True

    def has_FreesurferWhiteMatterMask_outputs(self, wmmask_dir):
        """
        Return True if the Freesurfer white matter mask outputs exist
        """

        wmmask = wmmask_dir/("wm_mask_dwi.nii.gz")
        metrics = wmmask_dir/("metrics.json")

        if not wmmask.exists() or not metrics.exists():
            return False
        return True

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

    def get_prov_t1(self, dir, EVE3=False):
        """
        Given a processing directory, get the T1 that was used as input

        Otherwise, return None
        """
        obtained_t1=True
        prov_file = dir/"pipeline_config.yml"
        if not prov_file.exists():
            #return None
            obtained_t1=False
        else:
            with open(prov_file, 'r') as file:
                prov = yaml.safe_load(file)
            
            if 'inputs' not in prov:
                #return None
                obtained_t1=False
            else:
                if 't1' in prov['inputs']:
                    #return the T1, also check to see if it was a special copy
                    t1_prov = prov['inputs']['t1']
                    if type(t1_prov) == dict:
                        return t1_prov['src_path']
                    return t1_prov
                else:
                    #return None
                    obtained_t1=False
        if not obtained_t1:
            if not EVE3:
                #check the EVE3 registration directory instead/as well
                EVE3_dir = dir.parent / ("WMAtlasEVE3{}".format(dir.name.split('PreQual')[1]))
                return self.get_prov_t1(EVE3_dir, EVE3=True)
            else:
                return None
    
    def legacy_get_dwis(self, sub, ses, acq, run):
        """
        Given the BIDS fields, return the original dwi, bval, bvec files
        """
        dwidir = self.setup.root_dataset_path / sub / ses / "dwi"
        sesx = '_'+ses if ses != '' else ''
        acqx = '_'+acq if acq != '' else ''
        runx = '_'+run if run != '' else ''
        dwi = dwidir / ("{}{}{}{}_dwi.nii.gz".format(sub, sesx, acqx, runx))
        bval = dwidir / ("{}{}{}{}_dwi.bval".format(sub, sesx, acqx, runx))
        bvec = dwidir / ("{}{}{}{}_dwi.bvec".format(sub, sesx, acqx, runx))

        if all([x.exists() for x in [dwi, bval, bvec]]):
            return (dwi, bval, bvec)
        return  None

    def get_prov_dwi(self, dir):
        """
        Given a processing directory, get the DWI, bval, bvec that was used as input
        """
        prov_file = dir/"pipeline_config.yml"
        if not prov_file.exists():
            return None
        
        with open(prov_file, 'r') as file:
            prov = yaml.safe_load(file)
        
        if 'inputs' not in prov:
            return None
        
        #check to make sure that this is correct for the prequal provenance file
            #either a single string or a list
        if 'dwi' in prov['inputs'] and 'bval' in prov['inputs'] and 'bvec' in prov['inputs']:
            dwi_prov = prov['inputs']['dwi']
            bval_prov = prov['inputs']['bval']
            bvec_prov = prov['inputs']['bvec']
            if type(dwi_prov) == dict:
                return (dwi_prov['src_path'], bval_prov['src_path'], bvec_prov['src_path'])
            return (dwi_prov, bval_prov, bvec_prov)
        return None

    def get_TICV_seg_file(self, t1, ses_deriv):
        """
        Returns the TICV segmentation file associated with the T1 file. Returns None if the file does not exist.
        """
        #get the acquisition and run name from the T1
        sub, ses, acq, run = self.get_BIDS_fields_t1(t1)
        #now get the associated TICV file
        TICV_dir = ses_deriv/("SLANT-TICVv1.2{}{}".format(acq,run))/("post")/("FinalResult")
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

    def get_UNest_seg_file(self, t1, ses_deriv):
        #get the acquisition and run name from the T1
        sub, ses, acq, run = self.get_BIDS_fields_t1(t1)
        #now get the associated TICV file
        unest_dir = ses_deriv/("UNest{}{}".format(acq,run))/("FinalResults")
        if not acq == '':
            acq = '_'+acq
        if not run == '':
            run = '_'+run
        if not ses == '':
            ses = '_'+ses
        unest_seg = unest_dir/("{}{}{}{}_T1w_seg_merged_braincolor.nii.gz".format(sub,ses,acq,run))
        return unest_seg

    def get_BIDS_acq_run_from_PQdir(self, PQdir):
        """
        Given a PreQual directory, return the acq, run BIDS fields 
        """
        suff = PQdir.name.split('PreQual')[1]
        if 'run' in suff:
            run = 'run' + suff.split('run')[1]
            suff = suff.split('run')[0]
        else:
            run = ''
        if 'acq' in suff:
            acq = 'acq' + suff.split('acq')[1]
        else:
            acq = ''
        return acq, run

    def get_BIDS_acq_run_from_EVE3dir(self, PQdir):
        """
        Given an EVE3 directory, return the acq, run BIDS fields 
        """
        suff = PQdir.name.split('WMAtlasEVE3')[1]
        if 'run' in suff:
            run = 'run' + suff.split('run')[1]
            suff = suff.split('run')[0]
        else:
            run = ''
        if 'acq' in suff:
            acq = 'acq' + suff.split('acq')[1]
        else:
            acq = ''
        return acq, run

    def get_BIDS_fields_from_PQdir(self, pqdir):
        """
        Given a PreQual directory, return the BIDS fields
        """
        #get the sub, ses from the directory
        if pqdir.parent.parent.name == 'derivatives':
            sub = pqdir.parent.name
            ses = ''
        else:
            sub = pqdir.parent.parent.name
            ses = pqdir.parent.name
        acq, run = self.get_BIDS_acq_run_from_PQdir(pqdir)
        return sub, ses, acq, run

    def get_BIDS_fields_from_EVE3dir(self, pqdir):
        """
        Given a EVE3 directory, return the BIDS fields
        """
        #get the sub, ses from the directory
        if pqdir.parent.parent.name == 'derivatives':
            sub = pqdir.parent.name
            ses = ''
        else:
            sub = pqdir.parent.parent.name
            ses = pqdir.parent.name
        acq, run = self.get_BIDS_acq_run_from_EVE3dir(pqdir)
        return sub, ses, acq, run

    def add_to_missing(self, sub, ses, acq, run, reason):
        """
        Adds a row to the missing data dataframe
        """
        row = {'sub': sub, 'ses': ses, 'acq': acq, 'run': run, 'missing': reason}
        self.missing_data = pd.concat([self.missing_data, pd.Series(row).to_frame().T], ignore_index=True)

    def make_session_dirs(self, *args, **kwargs):
        """
        Makes the session directories to hold inputs, outputs, and intermediates for the preprocessing run

        args = sub, ses, acq, run

        kwargs = tmp_input_dir, tmp_output_dir, working_dir, temp_dir, has_working, has_temp, ticv

        returns the temp directories
        """

        sub, ses, acq, run = args
        tmp_input_dir = kwargs['tmp_input_dir']/(sub)/("{}{}{}".format(ses,acq,run))
        tmp_output_dir = kwargs['tmp_output_dir']/(sub)/("{}{}{}".format(ses,acq,run))
        tmp_dirs = [tmp_input_dir, tmp_output_dir]
        if 'has_working' in kwargs.keys() and kwargs['has_working']:
            working_dir = kwargs['working_dir']/(sub)/("{}{}{}".format(ses,acq,run))
            tmp_dirs.append(working_dir)
        if 'has_temp' in kwargs.keys() and kwargs['has_temp']:
            temp_dir = kwargs['temp_dir']/(sub)/("{}{}{}".format(ses,acq,run))
            tmp_dirs.append(temp_dir)
        if 'ticv' in kwargs.keys() and kwargs['ticv']:
            post = tmp_output_dir/"post"
            pre = tmp_output_dir/"pre"
            dl = tmp_output_dir/"dl"
            tmp_dirs.append(post)
            tmp_dirs.append(pre)
            tmp_dirs.append(dl)
        
        for dir in tmp_dirs:
            if not dir.exists():
                os.makedirs(dir)
        
        return tmp_dirs

    def start_script_generation(self, session_input, session_output, **kwargs):
        """
        Starts the script generation for any pipeline.
        Sets up all the file copies and does a provenance check for the inputs.
        Calls the specific pipeline script generation function.
        Then, copies output files back and checks the provenance of the outputs.
            -Alternatively, could send back a provenance file for the outputs?
            -Or both?
        Finally, deletes the inputs and outputs from the tmp directories.

        Input variables:
        
        session_input: path to the input directory for the script on local storage
        session_output: path to the output directory for the script on local storage
        kwargs: dict of key word arguments
        - deriv_output_dir: path to the target output directory for the session when the pipeline is done running
        - special_copy: dict of the paths of specific output directories that we want to copy back
            -e.g. we do not want to copy "fsaverage" from freesurfer, only "freesurfer"


        Key Temp Variables

        input_targets: dict of the paths for the pipeline inputs when copied from nfs2 to local storage
        - e.g. the T1w image (sub-XXX_ses-YYY_T1w.nii.gz) would be copied to the local storage as T1w.nii.gz
        - used in tmp_input_targ_dict to have the proper names when setting up calls for the pipelines
        """

        def write_copy_to_input_line(self, script, input, targ_name):
            """
            Writes the line to copy the input to the session_input dir
            """
            if not self.setup.args.no_scp:
                script.write("scp -r {}@{}:{} {}\n".format(self.setup.vunetID, self.setup.args.src_server, input, targ_name))
            else:
                script.write("cp -rL {} {}\n".format(input, targ_name))

        def write_copy_from_output_line(self, script, session_output, deriv_output, **kwargs):
            """
            Writes the line(s) to copy the output from the session output dir back to the storage space
            """
            if not 'special_copy' in kwargs:
                if not self.setup.args.no_scp:
                    script.write("scp -r {}/* {}@{}:{}/\n".format(session_output, self.setup.vunetID, self.setup.args.src_server, deriv_output))
                else:
                    script.write("cp -r {}/* {}/\n".format(session_output, deriv_output))
            else:
                #do not just copy back all outputs, but only the necessary ones
                for targ_copy in kwargs['special_copy']:
                    if not self.setup.args.no_scp:
                        script.write("scp -r {} {}@{}:{}/\n".format(session_output/targ_copy, self.setup.vunetID, self.setup.args.src_server, deriv_output))
                    else:
                        script.write("cp -r {} {}/\n".format(session_output/targ_copy, deriv_output))

        def write_ssh_provenance_check(self, script, input, targ_name):
            """
            Writes the line to check the provenance of the input post copy

            If the provenance is not valid, then exits the script and return an error

            input: remote file
            targ_name: local file
            """

            if not self.setup.args.no_scp:
                script.write("local_md5=$(md5sum {} | cut -d ' ' -f 1)\nremote_md5=$(ssh {}@{} \"md5sum {} | cut -d ' ' -f 1\")\nif [ \"$local_md5\" != \"$remote_md5\" ]; then echo \"PROVENANCE FAIL: Files {} and {} do not match\"; exit 1; else echo Files {} and {} match; fi\n".format(targ_name, self.setup.vunetID, self.setup.args.src_server, input, input, targ_name, input, targ_name))
            else:
                script.write("local_md5=$(md5sum {} | cut -d ' ' -f 1)\nremote_md5=$(md5sum {} | cut -d ' ' -f 1)\nif [ \"$local_md5\" != \"$remote_md5\" ]; then echo \"PROVENANCE FAIL: Files {} and {} do not match\"; exit 1; else echo Files {} and {} match; fi\n".format(targ_name, input, input, targ_name, input, targ_name))

        def write_ssh_provenance_check_directory(self, script, input, targ_dir):
            """
            Writes lines to the script to check the provenance of a directory post copy

            If the provenance is not valid, then exits the script and returns an error

            input: local directory
            targ_dir: remote directory
            """

            #this will get the mdm5sum of all the files in the source directory
            if not self.setup.args.no_scp: #another /* here?
                content="src_dir={output_dir}; declare -A remote_dict; while IFS= read -r line; do value=\"${{line%% *}}\"; key=\"${{line##* }}\"; key=$(echo $key | sed -E \"s|${{src_dir}}||g\"); remote_dict[\"$key\"]=\"$value\"; done < <(ssh {ID}@{server} \"find ${{src_dir}} \( -type f -o -type l \) -exec md5sum {{}} +\")\n".format(ID=self.setup.vunetID, server=self.setup.args.src_server, output_dir=str(targ_dir)+'/')
                script.write(content)
            else:
                content="src_dir={output_dir}; declare -A remote_dict; while IFS= read -r line; do value=\"${{line%% *}}\"; key=\"${{line##* }}\"; key=$(echo $key | sed -E \"s|${{src_dir}}||g\"); remote_dict[\"$key\"]=\"$value\"; done < <(find $src_dir \( -type f -o -type l \) -exec md5sum {{}} +)\n".format(output_dir=str(targ_dir)+'/')
                script.write(content)
            #this will get the mdm5sum of all the files in the target directory (i.e. the one on the computation node input)
            content="src_dir={output_dir}; declare -A local_dict; while IFS= read -r line; do value=\"${{line%% *}}\"; key=\"${{line##* }}\"; key=$(echo $key | sed -E \"s|${{src_dir}}||g\"); local_dict[\"$key\"]=\"$value\"; done < <(find $src_dir \( -type f -o -type l \) -exec md5sum {{}} +)\n".format(output_dir=str(input)+'/')
            script.write(content)
            ####

            # content="src_dir={output_dir}; declare -A local_dict; while IFS= read -r line; do value=\"${{line%% *}}\"; key=\"${{line##* }}\"; key=$(echo $key | sed -E \"s|${{src_dir}}||g\"); local_dict[\"$key\"]=\"$value\"; done < <(find $src_dir \( -type f -o -type l \) -exec md5sum {{}} +)\n".format(output_dir=input)
            # script.write(content)
            # #script.write('declare -A local_dict && while read -r value key; do local_dict[${{key}}]=$value; done <<< "$(md5sum {}/*)"'.format(input))
            # #this will get the mdm5sum of all the files in the target directory
            # if not self.setup.args.no_scp: #another /* here?
            #     content="src_dir={output_dir}; declare -A targ_dict; while IFS= read -r line; do value=\"${{line%% *}}\"; key=\"${{line##* }}\";key=$(echo $key | sed -E \"s|${{src_dir}}||g\"); targ_dict[\"$key\"]=\"$value\"; done < <(ssh {ID}@{server} \"find ${{src_dir}} \( -type f -o -type l \) -exec md5sum {{}} +\")\n".format(ID=self.setup.vunetID, server=self.setup.args.src_server, output_dir=targ_dir)
            #     #script.write('declare -A targ_dict && while read -r value key; do targ_dict[${{key}}]=$value; done <<< "$(ssh {}@{} \"md5sum {}\")"'.format(self.setup.vunetID, self.setup.args.src_server, targ_dir))
            #     script.write(content)
            # else:
            #     content="src_dir={output_dir}; declare -A targ_dict; while IFS= read -r line; do value=\"${{line%% *}}\"; key=\"${{line##* }}\"; key=$(echo $key | sed -E \"s|${{src_dir}}||g\"); targ_dict[\"$key\"]=\"$value\"; done < <(find $src_dir \( -type f -o -type l \) -exec md5sum {{}} +)\n".format(output_dir=targ_dir)
            #     script.write(content)
                #script.write('declare -A targ_dict && while read -r value key; do targ_dict[${{key}}]=$value; done <<< "$(md5sum {}/*)"'.format(targ_dir))
            #this will loop through all the keys in the source directory, and check if the md5sums match between the corresponding files

            # #old: looping through the local
            # script.write('for key in "${!local_dict[@]}"; do\n')
            # script.write('    if [ "${local_dict[$key]}" != "${remote_dict[$key]}" ]; then\n')
            # script.write('        echo "PROVENANCE FAIL: Files $key do not match"; echo local: ${local_dict[$key]}; echo remote: ${remote_dict[$key]}; exit 1\n')
            # script.write('    fi\n')
            # script.write('done\n')

            #new: looping through the remote
            script.write('for key in "${!remote_dict[@]}"; do\n')
            script.write('    if [ "${local_dict[$key]}" != "${remote_dict[$key]}" ]; then\n')
            script.write('        echo "PROVENANCE FAIL: Files $key do not match"; echo local: ${local_dict[$key]}; echo remote: ${remote_dict[$key]}; exit 1\n')
            script.write('    fi\n')
            script.write('done\n')

            ## TODO: if the provenance fails, then make sure to remove the files in the outputs directory (and temp directories)
                #maybe also remove the copied outputs over ssh

        script_f = self.setup.script_dir/("{}.sh".format(self.count))
        with open(script_f, 'w') as script:
            #write the header
            script.write("#!/bin/bash\n")
            #here, make any additional directories that we would need to copy into (like for connectome special)
            if 'input_dirs' in kwargs:
                script.write("echo Creating additional input directories...\n")
                for inp_dir in kwargs['input_dirs']:
                    script.write('mkdir -p {}/{}\n'.format(session_input, inp_dir))
            if 'tractseg_setup' in kwargs:
                script.write("MINWAIT=0\n")
                script.write("MAXWAIT=60\n")
                script.write("sleep $((MINWAIT+RANDOM % (MAXWAIT-MINWAIT)))\n\n")
            script.write("echo Copying inputs to local storage...\n")
            #copy over the inputs
            #check the provenance of the inputs right afterwards as well
            input_targets = {}
            for key,input in self.inputs_dict[self.count].items(): ### THIS NEEDS TO BE FIXED
                if type(input) == list:
                    input_targets[key] = {}
                    for i in input:
                        targ_file = session_input/i.name
                        input_targets[key][i] = targ_file
                        write_copy_to_input_line(self, script, i, targ_file)
                        write_ssh_provenance_check(self, script, i, targ_file)
                elif type(input) == dict:
                    #copy the file with the correct name/path
                    if 'separate_input' in input: #e.g. tracseg requires some inputs being in the temp directory
                        targ_file = input['separate_input']/input['targ_name']
                    else:
                        targ_file = session_input/input['targ_name']
                    input_targets[key] = targ_file
                    if 'directory' in input:
                        #input['src_path'] = input['src_path'] / '*' #for copying contents of the directory (e.g. PreQual/PREPROCESSED)
                        #targ_file = targ_file / '*' 
                        write_copy_to_input_line(self, script, input['src_path']/'*', targ_file)
                        #write_ssh_provenance_check_directory(self, script, input['src_path'], targ_file)
                        write_ssh_provenance_check_directory(self, script, targ_file, input['src_path'])
                    else:
                        write_copy_to_input_line(self, script, input['src_path'], targ_file)
                        write_ssh_provenance_check(self, script, input['src_path'], targ_file)
                else: #this is also a dict
                    try:
                        targ_file = session_input/input.name #alter name here if necessary
                    except:
                        targ_file = session_input/input.split('/')[-1]
                    input_targets[key] = targ_file
                    write_copy_to_input_line(self, script, input, targ_file)
                    write_ssh_provenance_check(self, script, input, targ_file)
            #now that all the inputs are copied, write the specifics for running the pipeline
            #script.write("echo All inputs copied successfully. Running {}\n".format(self.setup.args.pipeline))

            #write the specifics for the pipeline
            tmp_input_targ_dict = {'input_targets': input_targets}
            #print(**kwargs)
            self.PIPELINE_GENERATE_MAP[self.setup.args.pipeline](script, session_input, session_output, **kwargs | tmp_input_targ_dict)

            #remove the inputs
            #script.write('echo Done runnning PreQual. Now removing inputs...\n')
            script.write('rm -r {}\n'.format(str(session_input)+'/*'))

            #write the yaml config file
            self.write_config_yml_lines(script, session_output, kwargs['deriv_output_dir'])

            #copy ALL the outputs back to the output directory (and ensure the proper provenance?)
            ## TODO: Write the provenance check for the key outputs
            if 'temp_is_output' in kwargs:
                write_copy_from_output_line(self, script, kwargs['temp_dir'], kwargs['deriv_output_dir'], **kwargs)
            else:
                write_copy_from_output_line(self, script, session_output, kwargs['deriv_output_dir'], **kwargs)

            #check the file transfer with a checksum
            script.write("echo Checking the provenance of the outputs...\n")
            if 'temp_is_output' in kwargs:
                write_ssh_provenance_check_directory(self, script, kwargs['temp_dir'], kwargs['deriv_output_dir'])
            else:
                write_ssh_provenance_check_directory(self, script, session_output, kwargs['deriv_output_dir'])
            # for key_file in self.outputs[self.count] + self.necessary_outputs[self.setup.args.pipeline]:
            #     if type(key_file) == dict:
            #         #check the provenance of the directory
            #         write_ssh_provenance_check_directory(self, script, session_output/key_file['targ_name'], kwargs['deriv_output_dir']/key_file['targ_name'])
            #     else:
            #         write_ssh_provenance_check(self, script, session_output/key_file, kwargs['deriv_output_dir']/key_file)
            script.write("echo All outputs have been copied successfully.\n")

            #remove the outputs
            script.write('echo Now removing outputs from local...\n')
            script.write('rm -r {}\n'.format(str(session_output)+'/*'))

            ## TODO: Delete any temp files created for the pipeline
            if 'temp_dir' in kwargs:
                script.write('rm -r {}/*\n'.format(kwargs['temp_dir']))
            if 'working_dir' in kwargs:
                script.write('rm -r {}/*\n'.format(kwargs['working_dir']))

    def write_config_yml(self, deriv_output):
        """
        Writes the config.yml file to the output directory

        Will this be different for every pipeline?
            -Perhaps just put if statements in
            -e.g. if self.setup.pipeline == 'PreQual', then add a thing for config.yml
        """

        config_yml = {
            'inputs': self.inputs_dict[self.count],
            'key_outputs': [str(deriv_output/x) for x in self.necessary_outputs[self.setup.args.pipeline]],
            'user': self.setup.vunetID,
            'date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'simg path': self.setup.simg
        }

        if self.setup.args.pipeline == 'PreQual':
            config_yml['config'] = self.config[self.count].to_dict(orient='records')
            config_yml['warnings'] = self.warnings[self.count]

        var_args = vars(self.setup.args)
        config_yml['args'] = var_args

        with open(deriv_output/'config.yml', 'w') as yml:
            yaml.dump(config_yml, yml, default_flow_style=False)
        
    def write_config_yml_lines(self, script, session_output, deriv_output):
        """
        Writes the lines to a script that output a config yaml file, then the line that copies it back to the output directory
        """

        def convert_dict_paths_to_string(data):
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, dict):
                        data[key] = convert_dict_paths_to_string(value)
                    elif isinstance(value, Path):
                        data[key] = str(value)
                    elif isinstance(value, list): #this is now correct for the input paths, but wrong for the PreQual config
                        data[key] = [str(x) if isinstance(x, Path) else x for x in value]
            return data
        
        def combine_PQ_config(data):
            """
            Given a list of PreQual config dictionaries, combine them into a single dictionary and turn Path objs into strings
            """
            PQconfig = {'dwi': [], 'sign': [], 'readout': []}
            for subdict in data:
                for key, value in subdict.items():
                    if isinstance(value, Path):
                        value = str(value)
                    PQconfig[key].append(value)
            return PQconfig


        config_yml_name = "pipeline_config.yml"

        config_yml = {
            'inputs': convert_dict_paths_to_string(self.inputs_dict[self.count]),
            'key_outputs': [str(deriv_output/x) for x in self.necessary_outputs[self.setup.args.pipeline]],
            'user': self.setup.vunetID,
            'date_of_script_generation': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'simg path': str(self.setup.simg)
        }

        if self.setup.args.pipeline == 'PreQual':
            #print(convert_dict_paths_to_string(self.config[self.count].to_dict(orient='records')))
            #print(self.config[self.count].to_dict(orient='records'))
            config_yml['config'] = combine_PQ_config(self.config[self.count].to_dict(orient='records'))
            config_yml['warnings'] = self.warnings[self.count]

        #with open(deriv_output/'config.yml', 'w') as yml:
        #    yaml.dump(config_yml, yml, default_flow_style=False)

        script.write("echo Writing config.yml file...\n")
        script.write("echo \"{}\" > {}/{}\n".format(yaml.dump(config_yml, default_flow_style=False), str(session_output), config_yml_name))
        script.write("echo Done writing config.yml file. Now copying it back...\n")
        if not self.setup.args.no_scp:
            script.write("scp {}/{} {}@{}:{}/\n".format(str(session_output), config_yml_name, self.setup.vunetID, self.setup.args.src_server, str(deriv_output)))
        else:
            script.write("cp {}/{} {}/\n".format(str(session_output), config_yml_name, str(deriv_output)))


    def calc_provenance(self, outputs):
        """
        TODO: Incorporate into the script generation

        Given the outputs of the pipeline (i.e. the paths while on nobackup), calculate the provenance for each output

        Return a dictionary of the provenance for each output
        """
        hash_dict = {}
        for output in outputs:
            #use md5sum to calculate the hash of the output
            checksum = hash_file(output)
            hash_dict[output] = checksum
        return hash_dict
            

class PreQualGenerator(ScriptGenerator):

    def __init__(self, setup_object):
        super().__init__(setup_object=setup_object)

        self.config = {}
        self.warnings = {}
        self.inputs_dict = {}
        self.outputs = {}

        self.generate_prequal_scripts()

    def PreQual_script_generate(self, script, session_input, session_output, **kwargs):
        """
        Writes PreQual script code to the script file.

        Entails writing the dtiQA_config.csv file, as well as running the PreQual singularity container

        Have alreaady copied the inputs and checked the provenance.
        """

        #self.config is a pandas dataframe with the columns: dwi, sign, readout
        #self.warnings is a dictionary with the keys being the count and the values being the warning string
            #only need the one for self.count

        #setup the options for PreQual
        PEaxis = kwargs['PEaxis']
        opts = ''
        if self.setup.args.dataset_name == "HCPA" or self.setup.args.dataset_name == "HCPD" or self.setup.args.dataset_name == "HCP":
            #we want to turn off all PREPROCESSING steps possible
            opts = '--denoise off --synb0 off'
        elif self.setup.args.dataset_name == "OASIS3" or self.setup.args.dataset_name == "IBIS":
            #we need the threshold for bvalues (as this is the weird bvalue acquisition)
            opts = '--bval_threshold 51 --eddy_bval_scale 2 --topup_first_b0s_only'
        elif self.setup.args.dataset_name == "TempleSocial":
            opts = '--nonzero_shells 250,1000,2000,3250,5000 --topup_first_b0s_only'
        elif self.setup.args.dataset_name == "Humphreys":
            opts = '--topup_first_b0s_only --eddy_bval_scale 4'
        else:
            opts = '--topup_first_b0s_only'

        if self.setup.args.skull_stripped and kwargs['needs_synb0']:
            opts += ' --synb0 stripped'

        #write the dtiQA_config.csv file (iterate over the rows of the config dataframe)
        #config_f = self.setup.tmp_input_dir/('dtiQA_config.csv')

        ### REMOVE dtiQA from previous run
        config_f = session_input/('dtiQA_config.csv')
        script.write("rm -r {}\n".format(str(config_f)))
        #with open(config_f, 'w') as config:
        script.write("echo Writing dtiQA_config.csv\n")
        for idx, row in self.config[self.count].iterrows():
            script.write("echo {},{},{} >> {}\n".format(row['dwi'].name.split('.nii.gz')[0], row['sign'], row['readout'], config_f))
        
        #write all the warnings
        script.write("echo '*** WARNINGS ***'\n")
        script.write("echo {}\n".format(self.warnings[self.count]))
        script.write("echo '*** END WARNINGS ***'\n")

        #write the PreQual command
        script.write("echo Done writing config file. Now running PreQual...\n")
        if self.setup.args.skull_stripped and kwargs['needs_synb0']:
            script.write("echo Skull stripping T1 for synb0...\n")
            #make sure to strip the t1 for synb0
            seg_file = session_input/('seg.nii.gz')
            mask_file = session_input/('mask.nii.gz')
            t1 = session_input/('t1.nii.gz')
            t1_temp = session_input/('t1_temp.nii.gz')
            script.write('singularity exec -e --contain -B {}:{} -B /tmp:/tmp {} fslmaths {} -bin {}\n'.format(str(session_input), str(session_input),
                str(self.setup.simg), str(seg_file), str(mask_file)))
            script.write('singularity exec -e --contain -B {}:{} -B /tmp:/tmp {} fslmaths {} -mul {} {}\n'.format(str(session_input), str(session_input),
                str(self.setup.simg), str(t1), str(mask_file), str(t1_temp)))
            script.write('rm {}\n'.format(str(t1)))
            script.write('mv {} {}\n'.format(str(t1_temp), str(t1)))
            script.write('Done skull stripping T1 for synb0. NOW running PreQual...\n')
        script.write('singularity run -e --contain -B {}:/INPUTS -B {}:/OUTPUTS -B {}:/APPS/freesurfer/license.txt -B /tmp:/tmp {} {} {}\n'.format(str(session_input),
            str(session_output), str(self.setup.freesurfer_license_path), str(self.setup.simg), PEaxis, opts))
        script.write("echo Finished running PreQual. Now removing inputs and copying outputs back...\n")

        #copy back the dtiQA_config.csv file as well so we know how the pipeline was run
            #maybe I'll just create a config.yml file
        #script.write("Copying dtiQA_config.csv back...\n")
        #script.write("scp {} {}@{}:{}/\n".format(config_f, self.setup.vunetID, self.setup.args.src_server, kwargs['deriv_output_dir']))


    def setup_dtiQA_config(self, dwis, PEsigns, readout_times):
        """
        Given the paths to the dwis and the PEsigns, setup the dtiQA config file (as a pandas dataframe)
        """
        if readout_times is None:
            readout_times = [0.05]*len(dwis)
            self.warnings[self.count] += "echo Warning: Could not read readout times from json files. Assuming 0.05 for all scans.\n"
        self.config[self.count] = pd.DataFrame(columns=['dwi', 'sign', 'readout'])
        for dwi, sign, readout in zip(dwis, PEsigns, readout_times):
            if readout == None:
                self.warnings[self.count] += "echo Warning: Could not read readout times from json files. Assuming 0.05 for all scans.\n"
                readout = 0.05
            row = {'dwi': dwi, 'sign': sign, 'readout': readout}
            self.config[self.count] = pd.concat([self.config[self.count], pd.Series(row).to_frame().T], ignore_index=True)
        # print(self.config[self.count])
        # print(dwis)
        # print(PEsigns)
        # print(readout_times)

    def generate_prequal_scripts(self):
        """
        Generates all the PreQual scripts for the dataset by querying the dataset to see what can be run.

        Writes the outputs to the specified scripts directory.
        """

        dwi_dirs = self.get_dwi_dirs()

        for dwi_dir_p in tqdm(dwi_dirs):
            dwi_dir = Path(dwi_dir_p)
            sub, ses = get_sub_ses_dwidir(dwi_dir)
            if not any(Path(dwi_dir).iterdir()):
                continue
            #get the dwi in the directory
            dwis = list(dwi_dir.glob('*dwi.nii.gz'))
            bvals = [Path(str(x).replace('.nii.gz', '.bval')) for x in dwis]
            bvecs = [Path(str(x).replace('.nii.gz', '.bvec')) for x in dwis]
            assert len(dwis) == len(bvals) == len(bvecs), "Error: number of dwis, bvals, and bvecs do not match for {}_{}".format(sub, ses)
            jsons = [Path(str(x).replace('.nii.gz', '.json')) for x in dwis]

            #check to see if the jsons have the same acquisition parameters or not (within a small margin)
            try:
                json_dicts = [get_json_dict(x) for x in jsons]
                if self.setup.args.force_combine_prequal:
                    similar = True
                    print("Forcing PreQuals to be combined for {}_{}".format(sub, ses))
                else:
                    similar = check_json_params_similarity(json_dicts, self.setup.args)
            except:
                if self.setup.args.force_combine_prequal:
                    similar = True
                    print("Forcing PreQuals to be combined for {}_{}".format(sub, ses))
                else:
                    similar = False
                    print("Error: Could not read json files {}. Assuming different acquisitions.".format(jsons))

            if similar and not self.setup.args.separate_prequal:
                #if similar, run them together
                #print("Running PreQuals together for {}_{}".format(sub, ses))
                PQdirs = self.get_PreQual_dirs_from_dwi(dwis, similar=True)
            else:
                #if dissimilar, run them separately
                PQdirs = self.get_PreQual_dirs_from_dwi(dwis, similar=False)

            #for each of the potential PreQuals, check to see if the outputs already exist
            # for x in PQdirs:
            #     if not all(self.check_PQ_outputs(x)):
            #         print(x)
            needPQdirs = [x for x in PQdirs if not all(self.check_PQ_outputs(x))]
            #get the dwis and jsons that correspond to the PQdirs
            if not similar or self.setup.args.separate_prequal: #if running separately, then only use the ones that need to be run
                idxs = [PQdirs.index(x) for x in needPQdirs]
                need_dwis = [dwis[i] for i in idxs]
                need_bvals = [bvals[i] for i in idxs]
                need_bvecs = [bvecs[i] for i in idxs]
                need_jsons = [jsons[i] for i in idxs]
            else: #if running together, then use them all
                need_dwis = dwis
                need_bvals = bvals
                need_bvecs = bvecs
                need_jsons = jsons
            try:
                if not similar or self.setup.args.separate_prequal:
                    #idxs = [PQdirs.index(x) for x in needPQdirs]
                    need_json_dicts = [json_dicts[i] for i in idxs]
                else:
                    need_json_dicts = json_dicts
                readout_times = get_readout_times(need_json_dicts)
                #print("Readout Times:", readout_times)
            except:
                need_json_dicts = None
                readout_times = None

            #if the PQdirs are empty, then skip
            if not needPQdirs:
                continue

            #now, for the PQdirs that need to be run, generate the scripts
            #print(needPQdirs)
            #print(need_dwis)
            for dir_num,pqdir in enumerate(needPQdirs):

                #self.inputs_dict = {self.count: #dwis, T1}
                acq, run = self.get_BIDS_acq_run_from_PQdir(pqdir)
                #get the PE direction for each of the scans
                if not similar or self.setup.args.separate_prequal: #RUN dwis SEPARATELY

                    if need_json_dicts is None: #we dont know, so just set to a default
                        PEaxis, PEsign, PEunknown = 'j', ['+'], None
                    else:
                        (PEaxis, PEsign, PEunknown) = get_PE_dirs([need_json_dicts[dir_num]], [need_jsons[dir_num]], single=True) #returns a single tuple
                    #definitely needs a T1 for synb0
                    needs_synb0 = True
                    #check for a T1
                    t1 = self.get_t1(dwi_dir.parent/("anat"), sub, ses)
                    if not t1:
                        self.add_to_missing(sub, ses, acq, run, 'T1_missing')
                        continue
                    #if you want to run with the T1 stripped
                    if self.setup.args.skull_stripped:
                        #also must find the corresponding SLANT segmentaion
                        #based on the T1, get the TICV/UNest segmentation
                        ses_deriv = pqdir.parent
                        if not self.setup.args.use_unest_seg:
                            seg = self.get_TICV_seg_file(t1, ses_deriv)
                        else:
                            seg = self.get_UNest_seg_file(t1, ses_deriv)
                        if not seg or not seg.exists():
                            self.add_to_missing(sub, ses, acq, run, 'TICV' if not self.setup.args.use_unest_seg else 'UNest')
                            continue

                    self.count += 1 #we should have everything, so we can generate the script
                    self.warnings[self.count] = ''
                    #warning if PEunknown
                    if PEunknown:
                        self.warnings[self.count] += "echo Warning: Unknown PE direction. Assuming j.\n"
                    #create the directories
                    (session_input, session_output) = self.make_session_dirs(sub, ses, acq, run, tmp_input_dir=self.setup.tmp_input_dir,
                                            tmp_output_dir=self.setup.tmp_output_dir)
                    #setup the inputs dicitonary
                    if self.setup.args.skull_stripped:
                        self.inputs_dict[self.count] = {'dwi': need_dwis[dir_num], 'bval': need_bvals[dir_num],
                                                    'bvec': need_bvecs[dir_num], 't1': {'src_path':t1, 'targ_name': 't1.nii.gz'},
                                                    'seg': {'src_path':seg, 'targ_name': 'seg.nii.gz'}}
                        self.warnings[self.count] += "Running synb0 with skull-stripped T1\n"
                    else:
                        self.inputs_dict[self.count] = {'dwi': need_dwis[dir_num], 'bval': need_bvals[dir_num],
                                                    'bvec': need_bvecs[dir_num], 't1': {'src_path':t1, 'targ_name': 't1.nii.gz'}}
                    #if run != '':
                    #    print(self.inputs_dict[self.count])
                    self.outputs[self.count] = []

                    #create the final output directories
                    PQdir_targ = self.setup.output_dir/(sub)/(ses)/("PreQual{}{}".format(acq, run))
                    if not PQdir_targ.exists():
                        os.makedirs(PQdir_targ)

                    #setup the dtiQA config file
                    #print(type(need_dwis[dir_num]), type(PEsign), type(readout_times))
                    self.setup_dtiQA_config([need_dwis[dir_num]], PEsign, readout_times)
                    
                    #write the script
                    self.start_script_generation(session_input, session_output, PEaxis=PEaxis, deriv_output_dir=PQdir_targ, needs_synb0=needs_synb0)

                    ##TODO##

                else: #run dwis TOGETHER
                    print('******************************************************')
                    print("Running PreQuals together for {}_{}".format(sub, ses))
                    try:
                        (PEaxes, PEsigns, PEunknowns) = get_PE_dirs(json_dicts, jsons, single=False) #returns a tuple of tuples
                    except:
                        print("No json dicts defined for {}_{}. Assuming j+ and 0.05 for all scans.".format(sub, ses))
                        PEaxes, PEsigns, PEunknowns = ['j']*len(dwis), ['+']*len(dwis), [True]*len(dwis)
                    #make sure, for new VMAP, that the PEdirection is negative for the acq-ReversePE
                    #determine if it needs a T1 or not
                    needs_synb0 = check_needs_synb0(PEaxes, PEsigns, PEunknowns)
                    if needs_synb0 is None:
                        self.add_to_missing(sub, ses, acq, run, 'Multiple_PE_axes')
                        continue
                    elif needs_synb0:
                        t1 = self.get_t1(dwi_dir.parent/("anat"), sub, ses)
                        if not t1:
                            self.add_to_missing(sub, ses, acq, run, 'T1_missing')
                            continue
                        #check to see if you need it to be skull-stripped
                        #if you want to run with the T1 stripped
                        if self.setup.args.skull_stripped:
                            #also must find the corresponding SLANT segmentaion
                            #based on the T1, get the TICV/UNest segmentation
                            ses_deriv = pqdir.parent
                            if not self.setup.args.use_unest_seg:
                                seg = self.get_TICV_seg_file(t1, ses_deriv)
                            else:
                                seg = self.get_UNest_seg_file(t1, ses_deriv)
                            if not seg or not seg.exists():
                                self.add_to_missing(sub, ses, acq, run, 'TICV' if not self.setup.args.use_unest_seg else 'UNest')
                                continue
                    #otherwise, continue as normal
                    self.count += 1 #we should have everything, so we can generate the script
                    self.warnings[self.count] = ''
                    #PE unknown warning
                    if any(PEunknowns):
                        self.warnings[self.count] += "echo Warning: Unknown PE direction for at least one scan. Assuming j for unknowns.\n"
                    
                    #create the directories (to hold temporary files during processing) for the session
                    (session_input, session_output) = self.make_session_dirs(sub, ses, acq, run, tmp_input_dir=self.setup.tmp_input_dir,
                                            tmp_output_dir=self.setup.tmp_output_dir)
                    
                    #create the final output directories
                    PQdir_targ = self.setup.output_dir/(sub)/(ses)/("PreQual{}{}".format(acq, run))
                    if not PQdir_targ.exists():
                        os.makedirs(PQdir_targ)

                    #setup the inputs dictionary
                    self.inputs_dict[self.count] = {'dwi': dwis, 'bval': bvals, 'bvec': bvecs}
                    if needs_synb0:
                        if self.setup.args.skull_stripped:
                            self.inputs_dict[self.count]['t1'] = {'src_path':t1, 'targ_name': 't1.nii.gz'}
                            self.inputs_dict[self.count]['seg'] = {'src_path':seg, 'targ_name': 'seg.nii.gz'}
                            self.warnings[self.count] += "Running synb0 with skull-stripped T1\n"
                        else:
                            self.inputs_dict[self.count]['t1'] = {'src_path':t1, 'targ_name': 't1.nii.gz'}
                            self.warnings[self.count] += "Running synb0 with raw T1\n"
                        self.warnings[self.count] += "echo Using T1 for synb0, as no ReversePE scans were found.\n"
                    self.outputs[self.count] = []

                    #setup the dtiQA config file
                    self.setup_dtiQA_config(dwis, PEsigns, readout_times)

                    #write the script
                    self.start_script_generation(session_input, session_output, PEaxis=PEaxes[0], deriv_output_dir=PQdir_targ, needs_synb0=needs_synb0) #they should all be the same, so just get the first one
                        
                    ##TODO##
            #now all the scripts have been generated
                    

class FreeSurferGenerator(ScriptGenerator):

    def __init__(self, setup_object):
        super().__init__(setup_object=setup_object)

        #self.warnings = {}
        self.outputs = {}
        self.inputs_dict = {}

        self.generate_freesurfer_scripts()    

    def freesurfer_script_generate(self, script, session_input, session_output, **kwargs):
        """
        Writes the FreeSurfer script code to the script file.

        Entails running the recon-all command on the T1 image

        Have already copied the inputs and checked the provenance.
        """

        ##TODO: Make sure that the t1 is coped with the correct name
            #change code around to make sure that the inputs targets are copied properly

        #write the recon-all command
        script.write("echo Running recon-all on T1...\n")
        script.write('time singularity exec -e --contain -B {}:/usr/local/freesurfer/.license -B {}:{} -B {}:{} {} recon-all -i {} -subjid freesurfer -sd {}/ -all\n'.format(self.setup.freesurfer_license_path, session_input, session_input, session_output, session_output, self.setup.simg, kwargs['input_targets']['t1'], session_output))
        #"time singularity exec -e --contain -B {}:/usr/local/freesurfer/.license -B {}:{} -B {}:{} {} recon-all -i {} -subjid freesurfer -sd {}/ -all > {}/freesurfer.log".format(license_file, t1, t1, out_dir, out_dir, simg_path, t1, out_dir, out_dir)
        script.write("echo Finished running recon-all. Now removing inputs and copying outputs back...\n")

    def generate_freesurfer_scripts(self):
        """
        Generates scripts from freesurfer
        """

        #get the T1s
        t1s = self.find_t1s()

        #for each T1, check to see if the freesurfer outputs exist
        for t1 in tqdm(t1s):
            sub, ses, acq, run = self.get_BIDS_fields_t1(t1)

            #check to see if the freesurfer outputs exist
            fs_dir = self.setup.dataset_derivs/ sub / ses / 'freesurfer{}{}'.format(acq, run)
            if check_freesurfer_outputs(fs_dir):
                continue

            self.count += 1

            #setupt the target destniation dir
            fsdir_targ = self.setup.output_dir/(sub)/(ses)/('freesurfer{}{}'.format(acq, run))
            if not fsdir_targ.exists():
                os.makedirs(fsdir_targ)
            
            #create the temp session directories
            (session_input, session_output) = self.make_session_dirs(sub, ses, acq, run, tmp_input_dir=self.setup.tmp_input_dir,
                                            tmp_output_dir=self.setup.tmp_output_dir)
        
            #setup the inputs dictionary
            self.inputs_dict[self.count] = {'t1': {'src_path': t1, 'targ_name': 'T1.nii.gz'}}
            self.outputs[self.count] = []

            #generate the script
            special_copy = {'freesurfer', 'freesurfer'}
            self.start_script_generation(session_input, session_output, deriv_output_dir=fsdir_targ, special_copy=special_copy)
                

class SLANT_TICVGenerator(ScriptGenerator):

    def __init__(self, setup_object):
        super().__init__(setup_object=setup_object)

        #self.warnings = {}
        self.outputs = {}
        self.inputs_dict = {}

        self.generate_slant_ticv_scripts()
    
    def SLANT_TICV_script_generate(self, script, session_input, session_output, **kwargs):
        """
        Generates a script for SLANT-TICV
        """

        script.write("echo Running SLANT-TICV...\n")

        #write the extra binds
        eb1 = "\"{}/\":/opt/slant/matlab/input_pre".format(session_input)
        eb2 = "\"{}/\":/opt/slant/matlab/input_post".format(session_input)
        eb3 = "\"{}/\":/opt/slant/matlab/output_pre".format(kwargs['pre'])
        eb4 = "\"{}/\":/opt/slant/dl/working_dir".format(kwargs['dl'])
        eb5 = "\"{}/\":/opt/slant/matlab/output_post".format(kwargs['post'])

        #add an empty file and make it executable (for the stupid bashrc source for whatever reason)
        empty_file = session_input/'empty.sh'
        script.write("touch {}\n".format(empty_file))
        script.write("chmod +x {}\n".format(empty_file))
        script.write("singularity exec -e --contain -B {} -B {} -B {} -B {} -B {} -B /tmp:/tmp --home {} -B {}:{}/.bashrc {} /opt/slant/run.sh\n".format(eb1, eb2, eb3, eb4, eb5, session_input, empty_file, session_input, self.setup.simg))
        script.write("echo Finished running SLANT-TICV. Now removing pre and dl directories...\n")
        script.write("rm -r {}/*\n".format(kwargs['pre']))
        script.write("rm -r {}/*\n".format(kwargs['dl']))
        #removing matlab cache files
        script.write("rm -rf {}/.mcrCache9.11\n".format(session_input))
        script.write("rm -rf {}/.matlab\n".format(session_input))
        script.write("echo Now removing inputs and copying outputs back...\n")

    def generate_slant_ticv_scripts(self):
        """
        Generates scripts for SLANT-TICV
        """

        #get the T1s
        t1s = self.find_t1s()

        #for each T1, check to see if the freesurfer outputs exist
        for t1 in tqdm(t1s):
            sub, ses, acq, run = self.get_BIDS_fields_t1(t1)

            #check to see if the TICV exists
            TICV_dir = self.setup.dataset_derivs/(sub)/(ses)/("SLANT-TICVv1.2{}{}".format(acq, run))
            if not TICV_dir.exists():
                os.makedirs(TICV_dir)
            #find_seg = "find {} -mindepth 3 -maxdepth 3 -name '*seg.nii.gz' \( -type f -o -type l \) ".format(TICV_dir)
            ticv_seg = get_TICV_seg(TICV_dir, sub, ses, acq, run)
            if ticv_seg:
                continue

            self.count += 1 #because we can run

            #setup the target dir
            ticv_targ = self.setup.output_dir/(sub)/(ses)/("SLANT-TICVv1.2{}{}".format(acq, run))
            if not ticv_targ.exists():
                os.makedirs(ticv_targ)

            #create the local temp directories
            (session_input, session_output, post, pre, dl) = self.make_session_dirs(sub, ses, acq, run,
                                tmp_input_dir=self.setup.tmp_input_dir, tmp_output_dir=self.setup.tmp_output_dir, ticv=True)
        
            #setup the inputs dictionary and outputs
            self.inputs_dict[self.count] = {'t1': t1}
            #print(ticv_seg)
            #print(str(ticv_seg).split(str(TICV_dir)))
            self.outputs[self.count] = [t1.replace('.nii.gz', '_seg.nii.gz')]

            ##generate the script
            #special_copy = {'post', 'post'}
            self.start_script_generation(session_input, session_output, deriv_output_dir=ticv_targ, #special_copy=special_copy,
                                         post=post, pre=pre, dl=dl)            

class UNestGenerator(ScriptGenerator):
    """
    Script generator for UNest. (For skull-stripped T1s mostly, but can be either)
    """

    def __init__(self, setup_object):
        super().__init__(setup_object=setup_object)

        #self.warnings = {}
        self.outputs = {}
        self.inputs_dict = {}


        self.generate_unest_scripts()

    def UNest_script_generate(self, script, session_input, session_output, **kwargs):
        """
        Writes commands for running UNest
        """

        script.write("echo Running UNest...\n")

        if self.setup.args.skull_stripped:
            script.write("singularity run -e --contain --home {} -B {}:/INPUTS -B {}:/WORKING_DIR -B {}:/OUTPUTS -B {}:/tmp {} --ticv\n".format(session_input, session_input, kwargs['working_dir'], session_output, kwargs['temp_dir'], self.setup.simg))
        else:
            script.write("singularity run -e --contain --home {} -B {}:/INPUTS -B {}:/WORKING_DIR -B {}:/OUTPUTS -B {}:/tmp {} --ticv --w_skull\n".format(session_input, session_input, kwargs['working_dir'], session_output, kwargs['temp_dir'], self.setup.simg))

        script.write("echo Finished running UNest. Now removing inputs and copying outputs back...\n")

    def generate_unest_scripts(self):
        """
        Generates UNest scripts
        """

        def UNest_asserts(root_temp, root_working):
            #check to make sure that the temp and working directories exist and we can write to them
            assert root_temp.exists(), "Error: Root temp directory {} does not exist".format(root_temp)
            assert root_working.exists(), "Error: Root working directory {} does not exist".format(root_working)
            assert os.access(root_temp, os.W_OK), "Error: Root temp directory {} is not writable".format(root_temp)
            assert os.access(root_working, os.W_OK), "Error: Root working directory {} is not writable".format(root_working)

        #assert that the working and temp directories exist
        root_temp = Path(self.setup.args.temp_dir)
        root_working = Path(self.setup.args.working_dir)
        UNest_asserts(root_temp, root_working)

        #get the T1s
        t1s = self.find_t1s()

        if self.setup.args.skull_stripped:
            print("Skull stripped flag is on. Will look for skull stripped T1 for UNest.")

        #for each T1, check to see if the freesurfer outputs exist
        for t1_f in tqdm(t1s):
            sub, ses, acq, run = self.get_BIDS_fields_t1(t1_f)
            t1 = Path(t1_f)

            unest_dir = unest_dir = self.setup.dataset_derivs/(sub)/(ses)/("UNest{}{}".format(acq, run))
            assert unest_dir.parent.name != "None" and (unest_dir.parent.name == ses or unest_dir.parent.name == sub), "Error: UNest directory naming is wrong {}".format(unest_dir)
        
            seg_file = unest_dir/("FinalResults")/("{}".format(t1.name.replace('T1w', 'T1w_seg_merged_braincolor')))
            if unest_dir.exists() and seg_file.exists():
                #print("Exists")
                continue

            #if the skull stripped flag is on, replace t1 with the skull stripped T1
            if self.setup.args.skull_stripped:
                #cp ssT1.nii.gz to sub-{}_ses-{}_acq-{}_run-{}_T1w.nii.gz
                synthstrip_dir = unest_dir.parent/("T1_synthstrip{}{}".format(acq, run))
                ssT1 = synthstrip_dir/("{}".format(t1.name.replace('T1w', 'synthstrip_T1w')))
                if not ssT1.exists():
                    print(ssT1)
                    row = {'sub':sub, 'ses':ses, 'acq':acq, 'run':run, 'missing':'ssT1'}
                    self.missing_data = pd.concat([self.missing_data, pd.Series(row).to_frame().T], ignore_index=True)
                    continue
            
            self.count += 1

            #create the temporary directories
            session_temp = root_temp/(sub)/("{}{}{}".format(ses,acq,run))
            session_work = root_working/(sub)/("{}{}{}".format(ses,acq,run))
            (session_input, session_output, session_work, session_temp) = self.make_session_dirs(sub, ses, acq, run, tmp_input_dir=self.setup.tmp_input_dir,
                                            tmp_output_dir=self.setup.tmp_output_dir, temp_dir=session_temp, working_dir=session_work,
                                            has_working=True, has_temp=True)

            #setup the output target directory
            unest_target = self.setup.output_dir/(sub)/(ses)/("UNest{}{}".format(acq, run))
            assert unest_target.parent.name != "None" and (unest_target.parent.name == ses or unest_target.parent.name == sub), "Error: UNest directory naming is wrong {}".format(unest_target)
            if not unest_target.exists():
                os.makedirs(unest_target)

            #setup the inputs dictionary
            if not self.setup.args.skull_stripped:
                self.inputs_dict[self.count] = {'t1': t1}
            else:
                self.inputs_dict[self.count] = {'t1': {'src_path': ssT1, 'targ_name': t1.name}}
            self.outputs = [str(seg_file).split(str(unest_dir))[1]]

            #start the script generation
            self.start_script_generation(session_input, session_output, deriv_output_dir=unest_target, temp_dir=session_temp, working_dir=session_work)

class SynthstripGenerator(ScriptGenerator):
    """
    Class for generating synthstrip scripts
    """

    def __init__(self, setup_object):
        super().__init__(setup_object=setup_object)

        #self.warnings = {}
        self.outputs = {}
        self.inputs_dict = {}

        self.generate_synthstrip_scripts()

    def synthstrip_script_generate(self, script, session_input, session_output, **kwargs):
        """
        Generates synthstrip scripts
        """

        script.write("echo Running Synthstrip...\n")

        script.write("singularity run -e --contain -B /tmp:/tmp -B {}:{} -B {}:{} --nv {} -i {} -o {} -m {}\n".format(session_input, session_input, session_output, session_output, self.setup.simg, kwargs['input_t1'], kwargs['ssT1'], kwargs['mask']))
        #script.write("python3 {} {} {} {} {}\n".format(str(kwargs['wrapper_script_path']), str(kwargs['input_t1']), str(kwargs['ssT1']), str(kwargs['mask']), str(kwargs['wrapper_path'])))
        script.write("echo Finished running Synthstrip. Now removing inputs and copying outputs back...\n")

    def generate_synthstrip_scripts(self):
        """
        Generates synthstrip scripts
        """

        # #define some stuff
        # wrapper_script_path = Path("/nobackup/p_masi/Singularities/synthstrip_wrapper.py")
        # wrapper_path = Path("/nobackup/p_masi/Singularities/synthstrip-singularity")
        # if not wrapper_script_path.exists() or not wrapper_path.exists():
        #     wrapper_script_path = Path("/nfs2/harmonization/singularities/synthstrip_wrapper.py")
        #     wrapper_path = Path("/nfs2/harmonization/singularities/synthstrip-singularity")

        # #make sure that these exist and are readable and execuatable
        # assert wrapper_script_path.exists(), "Error: Wrapper script {} does not exist".format(wrapper_script_path)
        # assert wrapper_path.exists(), "Error: Wrapper {} does not exist".format(wrapper_path)
        # assert os.access(wrapper_script_path, os.R_OK) and os.access(wrapper_script_path, os.X_OK), "Error: Wrapper script {} is either not readable or executable".format(wrapper_script_path)
        # assert os.access(wrapper_path, os.R_OK) and os.access(wrapper_path, os.X_OK), "Error: Wrapper {} is either not readable or executable".format(wrapper_path)

        #get the T1s
        t1s = self.find_t1s()

        #for each T1, check to see if the freesurfer outputs exist
        for t1_f in tqdm(t1s):
            t1 = Path(t1_f)
            sub, ses, acq, run = self.get_BIDS_fields_t1(t1)     

            #first, check to make sure that the data does not already exist
            #if ses:
            #    synthstrip_dir = self.dataset_derivs/(sub)/(ses)/("T1_synthstrip{}{}".format(acq, run))
            #else:
            synthstrip_dir = self.setup.dataset_derivs/(sub)/(ses)/("T1_synthstrip{}{}".format(acq, run))
            assert synthstrip_dir.parent.name != "None" and (synthstrip_dir.parent.name == ses or synthstrip_dir.parent.name == sub), "Error: Synthstrip directory naming is wrong {}".format(synthstrip_dir)
            ssT1_link = synthstrip_dir/("{}".format(t1.name.replace('T1w', 'synthstrip_T1w')))
            #print(ssT1_link)
            if ssT1_link.exists():
                continue

            self.count += 1

            #create the temporary directories
            (session_input, session_output) = self.make_session_dirs(sub, ses, acq, run, tmp_input_dir=self.setup.tmp_input_dir,
                                            tmp_output_dir=self.setup.tmp_output_dir)            
            
            #create the output target directory
            synthstrip_target = self.setup.output_dir/(sub)/(ses)/("T1_synthstrip{}{}".format(acq, run))
            assert synthstrip_target.parent.name != "None" and (synthstrip_target.parent.name == ses or synthstrip_target.parent.name == sub), "Error: UNest directory naming is wrong {}".format(synthstrip_target)
            if not synthstrip_target.exists():
                os.makedirs(synthstrip_target)

            #setup output paths (for script generation)
            input_t1 = session_input/("{}".format(t1.name))
            ssT1 = session_output/("{}".format(t1.name.replace('T1w', 'synthstrip_T1w')))
            mask = session_output/("{}".format(t1.name.replace('T1w', 'synthstrip_T1w_mask')))

            #setup the inputs dictionary and outputs list
            self.inputs_dict[self.count] = {'t1': t1}
            #self.outputs = [str(ssT1_link).split(str(synthstrip_dir))[1], str(mask).split(str(session_output))[1]]
            self.outputs[self.count] = []

            #start the script generation
            self.start_script_generation(session_input, session_output, deriv_output_dir=synthstrip_target,
                                        ssT1=ssT1, mask=mask, input_t1=input_t1)
            # self.start_script_generation(session_input, session_output, deriv_output_dir=synthstrip_target, wrapper_script_path=wrapper_script_path, wrapper_path=wrapper_path,
            #                             ssT1=ssT1, mask=mask, input_t1=input_t1)

class EVE3WMAtlasGenerator(ScriptGenerator):
    """
    Class for generating EVE3 registration scripts
    """

    def __init__(self, setup_object):
        super().__init__(setup_object=setup_object)

        #self.warnings = {}
        self.outputs = {}
        self.inputs_dict = {}

        self.generate_EVE3WMAtlas_scripts()

    def EVE3Registration_script_generate(self, script, session_input, session_output, **kwargs):
        """
        Writes the command for running EVE3 registration to a script
        """

        script.write("echo Running EVE3 registration...\n")
        script.write("singularity run -e --contain -B /tmp:/tmp -B {}:/INPUTS -B {}:/OUTPUTS {} --EVE3\n".format(session_input, session_output, self.setup.simg))
        script.write("echo Finished running EVE3 registration. Now removing inputs and copying outputs back...\n")

        #always remove firstshell to save space
        script.write("rm {}\n".format(str(session_output/'dwmri%firstshell.nii.gz')))
        #remove the warped files as well (result of using ANTs)
        script.write("rm {}\n".format(str(session_output/'dwmri%InverseWarped.nii.gz')))
        script.write("rm {}\n".format(str(session_output/'dwmri%Warped.nii.gz')))

        # script.write("echo Checking if any values are greater than 1500...\n")
        # script.write("if awk '{{for (i = 1; i <= NF; i++) if ($i > 1500) {{exit 0}}}}' {}; then\n".format(str(session_input/'dwmri.bval')))
        # script.write("    echo 'At least one value is greater than 1500'\n")
        # script.write("else\n")
        # script.write("    echo 'No values greater than 1500 found. Removing firstshell nifti...'\n")
        # script.write("    rm {}\n".format(str(session_output/'dwmri%firstshell.nii.gz')))
        # script.write("fi\n")
        # Check if any value in the file is greater than 1500
        # if awk '{for (i = 1; i <= NF; i++) if ($i > 1500) {exit 0}}' file.txt; then
        # echo "At least one value is greater than 1500"
        # else
        # echo "No values greater than 1500 found"
        # # Remove the file if no values are greater than 1500
        # rm -f dwmri%firstshell.nii.gz
        # fi


        #add a check to see if dwmri and dwmri%firstshell are the same. Delete if they are

    def generate_EVE3WMAtlas_scripts(self):
        """
        Generates EVE3 scripts
        """

        prequal_dirs = self.get_PreQual_dirs()

        for pqdir_p in tqdm(prequal_dirs):
            pqdir = Path(pqdir_p)
            #get the BIDS tags
            sub, ses, acq, run = self.get_BIDS_fields_from_PQdir(pqdir)

            #check to see if the EVE3 outputs already exist
            eve3dir = self.setup.dataset_derivs/(sub)/(ses)/("WMAtlasEVE3{}{}".format(acq, run))
            if self.has_EVE3WMAtlas_outputs(eve3dir):
                continue

            #check to see if the PreQual outputs exist
            if not all(self.check_PQ_outputs(pqdir)):
                self.add_to_missing(sub, ses, acq, run, 'PreQual')
                continue

            #get the T1 that was used for PreQual. If we cannot get it, then select it again using the function
            t1 = self.get_prov_t1(pqdir)
            if not t1:
                #get the T1 using the function, and print out a provenance warning
                print("Warning: Could not get provenance T1 for {}".format(pqdir))
                t1 = self.get_t1(self.setup.root_dataset_path/(sub)/(ses)/("anat"), sub, ses)
                if not t1:
                    self.add_to_missing(sub, ses, acq, run, 'T1')
                    continue
            
            #based on the T1, get the TICV/UNest segmentation
            ses_deriv = pqdir.parent
            if not self.setup.args.use_unest_seg:
                seg = self.get_TICV_seg_file(t1, ses_deriv)
            else:
                seg = self.get_UNest_seg_file(t1, ses_deriv)
            if not seg or not seg.exists():
                self.add_to_missing(sub, ses, acq, run, 'TICV' if not self.setup.args.use_unest_seg else 'UNest')
                continue

            self.count += 1

            #setup the temp directories
            (session_input, session_output) = self.make_session_dirs(sub, ses, acq, run, tmp_input_dir=self.setup.tmp_input_dir,
                                            tmp_output_dir=self.setup.tmp_output_dir)

            #create the output target directory
            eve3_target = self.setup.output_dir/(sub)/(ses)/("WMAtlasEVE3{}{}".format(acq, run))
            if not eve3_target.exists():
                os.makedirs(eve3_target)

            #setup the inputs dictionary and outputs list            
            self.inputs_dict[self.count] = {'t1': {'src_path': t1, 'targ_name': 't1.nii.gz'},
                                            'seg': {'src_path': seg, 'targ_name': 'T1_seg.nii.gz'},
                                            'pq_dwi_dir': {'src_path': pqdir/'PREPROCESSED', 'targ_name': '', 'directory': True}
                                        }
            self.outputs[self.count] = []

            #start the script generation
            self.start_script_generation(session_input, session_output, deriv_output_dir=eve3_target)

            #do a check to see if firstshell is the same as the input. If it is, then delete firstshell

class MNI152WMAtlasGenerator(ScriptGenerator):
    """
    Class for generating MNI152 scripts
    """

    def __init__(self, setup_object):
        super().__init__(setup_object=setup_object)

        #self.warnings = {}
        self.outputs = {}
        self.inputs_dict = {}

        self.generate_MNI152WMAtlas_scripts()

    def MNI152Registration_script_generate(self, script, session_input, session_output, **kwargs):
        """
        Writes the command for running MNI152 registration to a script
        """

        script.write("echo Running MNI152 registration...\n")
        script.write("singularity run -e --contain -B /tmp:/tmp -B {}:/INPUTS -B {}:/OUTPUTS {} --MNI\n".format(session_input, session_output, self.setup.simg))
        script.write("echo Finished running MNI152 registration. Now removing inputs and copying outputs back...\n")
        #always remove for MNI registration
        script.write("rm {}\n".format(str(session_output/'dwmri%firstshell.nii.gz')))
        #remove the warped files as well (result of using ANTs)
        script.write("rm {}\n".format(str(session_output/'dwmri%InverseWarped.nii.gz')))
        script.write("rm {}\n".format(str(session_output/'dwmri%Warped.nii.gz')))
        
        #script.write("echo Checking if any values are greater than 1500...\n")
        #script.write("if awk '{for (i = 1; i <= NF; i++) if ($i > 1500) {exit 0}}' {}; then\n".format(str(session_input/'dwmri.bval')))
        #script.write("    echo 'At least one value is greater than 1500'\n")
        #script.write("else\n")
        #script.write("    echo 'No values greater than 1500 found. Removing firstshell nifti...'\n")
        #script.write("    rm {}\n".format(str(session_output/'dwmri%firstshell.nii.gz')))
        #script.write("fi\n")

    def generate_MNI152WMAtlas_scripts(self):
        """
        Generates MNI152 scripts
        """

        prequal_dirs = self.get_PreQual_dirs()

        for pqdir_p in tqdm(prequal_dirs):
            pqdir = Path(pqdir_p)
            #get the BIDS tags
            sub, ses, acq, run = self.get_BIDS_fields_from_PQdir(pqdir)

            #check to see if the EVE3 outputs already exist
            mnidir = self.setup.dataset_derivs/(sub)/(ses)/("WMAtlas{}{}".format(acq, run))
            if self.has_MNI152WMAtlas_outputs(mnidir):
                continue

            #check to see if the PreQual outputs exist
            if not all(self.check_PQ_outputs(pqdir)):
                self.add_to_missing(sub, ses, acq, run, 'PreQual')
                continue

            #get the T1 that was used for PreQual. If we cannot get it, then select it again using the function
            t1 = self.get_prov_t1(pqdir)
            if not t1:
                #get the T1 using the function, and print out a provenance warning
                print("Warning: Could not get provenance T1 for {}/{}/{}".format(sub,ses,pqdir))
                t1 = self.get_t1(self.setup.root_dataset_path/(sub)/(ses)/("anat"), sub, ses)
                if not t1:
                    self.add_to_missing(sub, ses, acq, run, 'T1')
                    continue
            
            #based on the T1, get the TICV/UNest segmentation
            ses_deriv = pqdir.parent
            if not self.setup.args.use_unest_seg:
                seg = self.get_TICV_seg_file(t1, ses_deriv)
            else:
                seg = self.get_UNest_seg_file(t1, ses_deriv)
            if not seg.exists():
                self.add_to_missing(sub, ses, acq, run, 'TICV' if not self.setup.args.use_unest_seg else 'UNest')
                continue

            self.count += 1

            #setup the temp directories
            (session_input, session_output) = self.make_session_dirs(sub, ses, acq, run, tmp_input_dir=self.setup.tmp_input_dir,
                                            tmp_output_dir=self.setup.tmp_output_dir)

            #create the output target directory
            mni_target = self.setup.output_dir/(sub)/(ses)/("WMAtlas{}{}".format(acq, run))
            if not mni_target.exists():
                os.makedirs(mni_target)

            #setup the inputs dictionary and outputs list            
            self.inputs_dict[self.count] = {'t1': {'src_path': t1, 'targ_name': 't1.nii.gz'},
                                            'seg': {'src_path': seg, 'targ_name': 'T1_seg.nii.gz'},
                                            'pq_dwi_dir': {'src_path': pqdir/'PREPROCESSED', 'targ_name': '', 'directory': True}
                                        }
            self.outputs[self.count] = []

            #start the script generation
            self.start_script_generation(session_input, session_output, deriv_output_dir=mni_target)

            #do a check to see if firstshell is the same as the input. If it is, then delete firstshell

class MaCRUISEGenerator(ScriptGenerator):
    """
    Class for generating MaCRUISE scripts
    """

    def __init__(self, setup_object):
        super().__init__(setup_object=setup_object)

        #self.warnings = {}
        self.outputs = {}
        self.inputs_dict = {}

        self.generate_macruise_scripts()

    def macruise_script_generate(self, script, session_input, session_output, **kwargs):
        """
        Generates MaCRUISE scripts
        """

        script.write("echo Running MaCRUISE...\n")
        script.write("singularity exec -e --contain --home {} -B {}:/INPUTS -B {}:/OUTPUTS -B /tmp:/tmp -B {}:/dev/shm {} xvfb-run -a --server-args=\"-screen 3 1920x1200x24 -ac +extension GLX\" /extra/MaCRUISE_v3_2_0_classern\n".format(session_input, session_input, session_output, session_input, self.setup.simg))
        script.write("echo Now deleting temporary outputs and changing output structure...\n")
        script.write("find {}/MaCRUISE -mindepth 1 -maxdepth 1 -type d ! -name 'Output' -exec rm -r {{}} +\n".format(session_output))
        script.write("mv {}/MaCRUISE/Output {}/\n".format(session_output, session_output))
        script.write("echo Finished running MaCRUISE. Now removing inputs and copying outputs back...\n")

    def generate_macruise_scripts(self):
        """
        Generates macruise scripts
        """

        #get the T1s
        t1s = self.find_t1s()

        #for each T1, check to see if the freesurfer outputs exist
        for t1 in tqdm(t1s):
            sub, ses, acq, run = self.get_BIDS_fields_t1(t1)    

            #check to see if the MaCRUISE outputs exist already
            macruise_dir = self.setup.dataset_derivs/(sub)/(ses)/("MaCRUISEv3.2.0{}{}".format(acq, run))
            segrefine = macruise_dir/("Output")/("SegRefine")/("SEG_refine.nii.gz")
            surf_freeview = macruise_dir/("Output")/("Surfaces_FreeView")/("target_image_GMimg_innerSurf.asc")
            surf_mni = macruise_dir/  'Output' / 'Surfaces_MNI' / 'target_image_GMimg_innerSurf.asc'
            if segrefine.exists() and surf_freeview.exists() and surf_mni.exists():
                continue

            #check to see if the TICV/UNest exists
            #check to make sure that the segmentations exist
            sesx = '' if ses == '' else '_'+ses
            acqx = '' if acq == '' else '_'+acq
            runx = '' if run == '' else '_'+run
            if self.setup.args.use_unest_seg:
                segdir = self.setup.dataset_derivs/(sub)/(ses)/("UNest{}{}".format(acq, run))
                seg_file = segdir/("FinalResults")/("{}{}{}{}_T1w_seg_merged_braincolor.nii.gz".format(sub, sesx, acqx, runx))
            else:
                segdir = self.setup.dataset_derivs/(sub)/(ses)/("SLANT-TICVv1.2{}{}".format(acq, run))
                seg_file = segdir/("post")/("FinalResult")/("{}{}{}{}_T1w_seg.nii.gz".format(sub, sesx, acqx, runx))
            
            if not segdir.exists() or not seg_file.exists():
                row = {'sub':sub, 'ses':ses, 'acq':acq, 'run':run, 'missing':'TICV' if not self.setup.args.use_unest_seg else 'UNest'}
                missing_data = pd.concat([missing_data, pd.Series(row).to_frame().T], ignore_index=True)
                continue

            self.count += 1

            #setup the temp directories
            (session_input, session_output) = self.make_session_dirs(sub, ses, acq, run, tmp_input_dir=self.setup.tmp_input_dir,
                                            tmp_output_dir=self.setup.tmp_output_dir)
            
            #create the output target directory
            macruise_target = self.setup.output_dir/(sub)/(ses)/("MaCRUISEv3.2.0{}{}".format(acq, run))
            if not macruise_target.exists():
                os.makedirs(macruise_target)

            #setup the inputs dictionary and outputs list
            self.inputs_dict[self.count] = {'t1': {'src_path': t1, 'targ_name': 'T1.nii.gz'},
                                            'seg': {'src_path': seg_file, 'targ_name': 'orig_target_seg.nii.gz'}}
            self.outputs[self.count] = []

            #start the script generation
            self.start_script_generation(session_input, session_output, deriv_output_dir=macruise_target)

class BiscuitGenerator(ScriptGenerator):
    """
    Class for generating biscuit scripts
    """

    def __init__(self, setup_object):
        super().__init__(setup_object=setup_object)

        #self.warnings = {}
        self.outputs = {}
        self.inputs_dict = {}

        self.generate_biscuit_scripts()


    def biscuit_script_generate(self, script, session_input, session_output, **kwargs):
        """
        Writes arguments to a script for running Biscuit
        """

        script.write("echo Creating .tmp in outputs directory...\n")
        script.write("mkdir {}/.tmp\n".format(session_output))
        script.write("echo Running Biscuit...\n")
        script.write("singularity run -e --home {} -B {}:/INPUTS -B {}:/OUTPUTS -B {}/.tmp:/OUTPUTS/.tmp -B {}:/tmp -B {}:/opt/biscuit/license.txt {} --proj {} --subj {} --sess {} --reg oasis45 --surf_recon macruise /INPUTS /OUTPUTS\n".format(kwargs['session_working'], session_input, session_output, session_output, kwargs['session_temp'], self.setup.freesurfer_license_path, self.setup.simg, 'R01_' + self.setup.root_dataset_path.name, kwargs['sub'], kwargs['ses']))
        script.write("echo Finished running Biscuit. Now removing inputs and copying outputs back...\n")

    def generate_biscuit_scripts(self):
        """
        Generates scripts for Biscuit
        """

        def UNest_asserts(root_temp, root_working):
            #check to make sure that the temp and working directories exist and we can write to them
            assert root_temp.exists(), "Error: Root temp directory {} does not exist".format(root_temp)
            assert root_working.exists(), "Error: Root working directory {} does not exist".format(root_working)
            assert os.access(root_temp, os.W_OK), "Error: Root temp directory {} is not writable".format(root_temp)
            assert os.access(root_working, os.W_OK), "Error: Root working directory {} is not writable".format(root_working)

        #assert that the working and temp directories exist
        root_temp = Path(self.setup.args.temp_dir)
        root_working = Path(self.setup.args.working_dir)
        UNest_asserts(root_temp, root_working)

        #get the T1s
        t1s = self.find_t1s()

        #for each T1, check to see if the freesurfer outputs exist
        for t1 in tqdm(t1s):
            sub, ses, acq, run = self.get_BIDS_fields_t1(t1)    

            #check to see if the Biscuit outputs exist already
            biscuit_dir = self.setup.dataset_derivs/(sub)/(ses)/("Biscuit{}{}".format(acq, run))
            PDF = biscuit_dir / 'PDF' / 'biscuit.pdf'
            if PDF.exists():
                continue

            #check to see if the MaCRUISE outputs exist
            macruise_dir = self.setup.dataset_derivs/(sub)/(ses)/("MaCRUISEv3.2.0{}{}".format(acq, run))
            segrefine = macruise_dir/("Output")/("SegRefine")/("SEG_refine.nii.gz")
            surf_freeview = macruise_dir/("Output")/("Surfaces_FreeView")/("target_image_GMimg_innerSurf.asc")
            surf_mni = macruise_dir/  'Output' / 'Surfaces_MNI' / 'target_image_GMimg_innerSurf.asc'
            if not (segrefine.exists() and surf_freeview.exists() and surf_mni.exists()):
                row = {'sub':sub, 'ses':ses, 'acq':acq, 'run':run, 'missing':'MaCRUISE'}
                missing_data = pd.concat([missing_data, pd.Series(row).to_frame().T], ignore_index=True)
                continue

            self.count += 1

            #create the temporary directories
            #create the temporary directories
            session_temp = root_temp/(sub)/("{}{}{}".format(ses,acq,run))
            session_work = root_working/(sub)/("{}{}{}".format(ses,acq,run))
            (session_input, session_output) = self.make_session_dirs(sub, ses, acq, run, tmp_input_dir=self.setup.tmp_input_dir,
                                            tmp_output_dir=self.setup.tmp_output_dir, temp_dir=session_temp, working_dir=session_work,
                                            has_working=True, has_temp=True)

            #setup the target output directory
            biscuit_target = self.setup.output_dir/(sub)/(ses)/("Biscuit{}{}".format(acq, run))
            if not biscuit_target.exists():
                os.makedirs(biscuit_target)

            #setup the inputs dictionary and outputs list
            self.inputs_dict[self.count] = {'t1': {'src_path': t1, 'targ_name': 'T1.nii.gz'},
                                            'seg': segrefine,
                                            'freeview_surf': surf_freeview,
                                            'mni_surf': surf_mni}            
            self.outputs[self.count] = []  

            #start the script generation
            self.start_script_generation(session_input, session_output, deriv_output_dir=biscuit_target,
                                         temp_dir=session_temp, working_dir=session_work, sub=sub, ses=ses)

class ConnectomeSpecialGenerator(ScriptGenerator):
    """
    Class for generating connectome special scripts
    """

    def __init__(self, setup_object):
        super().__init__(setup_object=setup_object)

        #self.warnings = {}
        self.outputs = {}
        self.inputs_dict = {}

        self.generate_connectome_special_scripts()

    def connectome_special_script_generate(self, script, session_input, session_output, **kwargs):
        """
        Writes the command for running connectome special to a script
        """

        script.write("echo Running Connectome Special...\n")
        script.write("singularity run -e --contain --bind {}:/DIFFUSION/,{}:/SLANT/,{}:/OUTPUTS/,/tmp:/tmp {}\n".format(kwargs['PQ_inp'], kwargs['seg_input'], session_output, self.setup.simg))
        script.write("echo Removing temp outputs...\n")
        script.write("rm -r {}/TEMP\n".format(session_output))
        script.write("echo Finished running Connectome Special. Now removing inputs and copying outputs back...\n")


    def generate_connectome_special_scripts(self):
        """
        Generates connectome special scripts
        """

        prequal_dirs = self.get_PreQual_dirs()

        for pqdir_p in tqdm(prequal_dirs):
            pqdir = Path(pqdir_p)
            #get the BIDS tags
            sub, ses, acq, run = self.get_BIDS_fields_from_PQdir(pqdir)

            #check to see if the CS outputs already exist
            cs_dir = self.setup.dataset_derivs/(sub)/(ses)/("ConnectomeSpecial{}{}".format(acq, run))
            if self.has_ConnectomeSpecial_outputs(cs_dir):
                continue

            #check to see if the PreQual outputs exist
            if not all(self.check_PQ_outputs(pqdir)):
                self.add_to_missing(sub, ses, acq, run, 'PreQual')
                continue

            #get the T1 that was used for PreQual. If we cannot get it, then select it again using the function
            t1 = self.get_prov_t1(pqdir)
            if not t1:
                #get the T1 using the function, and print out a provenance warning
                print("Warning: Could not get provenance T1 for {}/{}/{}".format(sub,ses,pqdir))
                t1 = self.get_t1(self.setup.root_dataset_path/(sub)/(ses)/("anat"), sub, ses)
                if not t1:
                    self.add_to_missing(sub, ses, acq, run, 'T1')
                    continue
            
            #based on the T1, get the TICV/UNest segmentation
            ses_deriv = pqdir.parent
            if not self.setup.args.use_unest_seg:
                seg = self.get_TICV_seg_file(t1, ses_deriv)
            else:
                seg = self.get_UNest_seg_file(t1, ses_deriv)
            if not seg.exists():
                self.add_to_missing(sub, ses, acq, run, 'TICV' if not self.setup.args.use_unest_seg else 'UNest')
                continue

            #we also need the scalar maps
            #get the tensor maps to use for the mean/std bundle calculations
            #if ses != '':
            #    eve3dir = self.setup.dataset_derivs/(sub)/(ses)/("WMAtlasEVE3{}{}".format(acq, run))
            #else:
            #    eve3dir = self.setup.dataset_derivs/(sub)/("WMAtlasEVE3{}{}".format(acq, run))
            eve3dir = self.setup.dataset_derivs/(sub)/(ses)/("WMAtlasEVE3{}{}".format(acq, run))
            assert eve3dir.parent.name != "None" and (eve3dir.parent.name == ses or eve3dir.parent.name == sub), "Error in ConnectomeSpecial generation - EVE3 directory naming is wrong {}".format(eve3dir)
            tensor_maps = [eve3dir/("dwmri%{}.nii.gz".format(m)) for m in ['fa', 'md', 'ad', 'rd']]
            using_PQ = False
            if not all([t.exists() for t in tensor_maps]):
                #row = {'sub':sub, 'ses':ses, 'acq':acq, 'run':run, 'missing':'tensor_maps'}
                #missing_data = pd.concat([missing_data, pd.Series(row).to_frame().T], ignore_index=True)
                self.add_to_missing(sub, ses, acq, run, 'tensor_maps')
                continue
            
            #check the other assertions for the ses dirs (for None in ses) --> should only be an issue if not using
                #get_BIDS_fields to get the ses

            self.count += 1

            #create the temp session directories
            (session_input, session_output) = self.make_session_dirs(sub, ses, acq, run, tmp_input_dir=self.setup.tmp_input_dir,
                                            tmp_output_dir=self.setup.tmp_output_dir)
            
            #create the target output directory
            cs_target = self.setup.output_dir/(sub)/(ses)/("ConnectomeSpecial{}{}".format(acq, run))
            if not cs_target.exists():
                os.makedirs(cs_target)

            #setup inputs dict and outputs list
            #dwi first
            scalars_dir = 'PreQual/SCALARS'
            preproc_dir = 'PreQual/PREPROCESSED'
            self.inputs_dict[self.count] = {
                'fa_map': {'src_path': eve3dir/'dwmri%fa.nii.gz', 'targ_name': scalars_dir + '/dwmri_tensor_fa.nii.gz'},
                'md_map': {'src_path': eve3dir/'dwmri%md.nii.gz', 'targ_name': scalars_dir + '/dwmri_tensor_md.nii.gz'},
                'ad_map': {'src_path': eve3dir/'dwmri%ad.nii.gz', 'targ_name': scalars_dir + '/dwmri_tensor_ad.nii.gz'},
                'rd_map': {'src_path': eve3dir/'dwmri%rd.nii.gz', 'targ_name': scalars_dir + '/dwmri_tensor_rd.nii.gz'},
                'pq_dwi_dir': {'src_path': pqdir/'PREPROCESSED', 'targ_name': preproc_dir, 'directory': True}
            }
            #now the T1 and the segmentation
            seg_pre_dir = 'Slant/pre/{}'.format(Path(t1).name.split('.nii')[0])
            self.inputs_dict[self.count]['t1'] = {'src_path': t1, 'targ_name': '{}/orig_target.nii.gz'.format(seg_pre_dir)}
            seg_post_final = 'Slant/post/FinalResult'
            if self.setup.args.use_unest_seg:
                new_seg_name = seg.name.replace('T1w_seg_merged_braincolor', 'T1w_seg')
                self.inputs_dict[self.count]['seg'] = {'src_path': seg, 'targ_name': '{}/{}'.format(seg_post_final, new_seg_name)}
            else:
                self.inputs_dict[self.count]['seg'] = {'src_path': seg, 'targ_name': '{}/{}'.format(seg_post_final, seg.name)}

            #now write the script
            pq_input = session_input/"PreQual"
            seg_input = session_input/"Slant"
            self.start_script_generation(session_input, session_output, deriv_output_dir=cs_target,
                                         input_dirs=[seg_pre_dir, seg_post_final, scalars_dir, preproc_dir],
                                         PQ_inp=pq_input, seg_input=seg_input)
            
            #kwargs = seg_pre_dir, seg_post_final, scalars_dir, preproc_dir

class TractsegGenerator(ScriptGenerator):
    """
    Class for generating Tractseg scripts
    """

    def __init__(self, setup_object):
        super().__init__(setup_object=setup_object)

        #self.warnings = {}
        self.outputs = {}
        self.inputs_dict = {}

        self.generate_tractseg_scripts()

    def tractseg_script_generate(self, script, session_input, session_output, **kwargs):
        """
        Writes the command for running Tractseg to a script
        """

        #bind for the directory we are running tractseg from (not the home directory)?

        #define the singularities
        if self.setup.args.tractseg_make_DTI:
            ts_simg = self.setup.simg[0]
            scilus_simg = self.setup.simg[1]
            mrtrix_simg = self.setup.simg[1]
            wmatlas_simg = self.setup.simg[2]
        else:
            ts_simg = self.setup.simg[0]
            scilus_simg = self.setup.simg[1]
            mrtrix_simg = self.setup.simg[1]

        #adds the following lines to make the DTI if necessary
        if self.setup.args.tractseg_make_DTI:
            script.write("echo Making DTI...\n")
            bind1 = "{}:/INPUTS".format(kwargs['temp_dir'])
            bind2 = "{}:/OUTPUTS".format(kwargs['temp_dir'])

            #first, convert the dwi to tensor
            script.write("echo Making temp directories...\n")
            #script.write("mkdir -p {}\n".format(dti_dir))
            script.write("echo Shelling to 1500...\n")
            script.write("time singularity exec -e --contain -B /tmp:/tmp -B {} -B {} {} python3 /CODE/extract_single_shell.py\n".format(bind1, bind2, wmatlas_simg))
            script.write("echo Done shelling to 1500. Now fitting tensors DTI...\n")

            shellbvec = "{}/dwmri%firstshell.bvec".format(kwargs['temp_dir'])
            shellbval = "{}/dwmri%firstshell.bval".format(kwargs['temp_dir'])
            shellnii = "{}/dwmri%firstshell.nii.gz".format(kwargs['temp_dir'])
            tensor = "{}/dwmri_tensor.nii.gz".format(kwargs['temp_dir'])
            if self.setup.args.tractseg_DTI_use_all_shells:
                orig_dwi = "{}/dwmri.nii.gz".format(kwargs['temp_dir'])
                orig_bval = "{}/dwmri.bval".format(kwargs['temp_dir'])
                orig_bvec = "{}/dwmri.bvec".format(kwargs['temp_dir'])
                script.write("time singularity exec -e --contain -B /tmp:/tmp -B {}:{} {} dwi2tensor -fslgrad {} {} {} {}\n".format(kwargs['temp_dir'], kwargs['temp_dir'], mrtrix_simg, orig_bvec, orig_bval, orig_dwi, tensor))
            else:
                script.write("time singularity exec -e --contain -B /tmp:/tmp -B {}:{} {} dwi2tensor -fslgrad {} {} {} {}\n".format(kwargs['temp_dir'], kwargs['temp_dir'], mrtrix_simg, shellbvec, shellbval, shellnii, tensor))
            script.write("echo Comuting DTI scalar maps...\n")
            fa = "{}/dwmri_tensor_fa.nii.gz".format(kwargs['temp_dir'])
            md = "{}/dwmri_tensor_md.nii.gz".format(kwargs['temp_dir'])
            ad = "{}/dwmri_tensor_ad.nii.gz".format(kwargs['temp_dir'])
            rd = "{}/dwmri_tensor_rd.nii.gz".format(kwargs['temp_dir'])
            mask = "{}/mask.nii.gz".format(kwargs['temp_dir'])
            script.write("time singularity exec -e --contain -B /tmp:/tmp -B {}:{} {} tensor2metric {} -fa {} -mask {}\n".format(kwargs['temp_dir'], kwargs['temp_dir'], mrtrix_simg, tensor, fa, mask))
            script.write("time singularity exec -e --contain -B /tmp:/tmp -B {}:{} {} tensor2metric {} -adc {} -mask {}\n".format(kwargs['temp_dir'], kwargs['temp_dir'], mrtrix_simg, tensor, md, mask))
            script.write("time singularity exec -e --contain -B /tmp:/tmp -B {}:{} {} tensor2metric {} -ad {} -mask {}\n".format(kwargs['temp_dir'], kwargs['temp_dir'], mrtrix_simg, tensor, ad, mask))
            script.write("time singularity exec -e --contain -B /tmp:/tmp -B {}:{} {} tensor2metric {} -rd {} -mask {}\n".format(kwargs['temp_dir'], kwargs['temp_dir'], mrtrix_simg, tensor, rd, mask))

        script.write("echo Resampling to 1mm iso...\n")
        script.write("time singularity run -e --contain -B /tmp:/tmp -B {}:{} {} mrgrid {}/dwmri.nii.gz regrid {}/dwmri_1mm_iso.nii.gz -voxel 1\n".format(kwargs['temp_dir'], kwargs['temp_dir'], mrtrix_simg, kwargs['temp_dir'], kwargs['temp_dir']))
        #same for the fa, md, ad, rd
        script.write("time singularity run -e --contain -B /tmp:/tmp -B {}:{} {} mrgrid {}/dwmri_tensor_fa.nii.gz regrid {}/dwmri_tensor_fa_1mm_iso.nii.gz -voxel 1\n".format(kwargs['temp_dir'],kwargs['temp_dir'], mrtrix_simg, kwargs['temp_dir'], kwargs['temp_dir']))
        script.write("time singularity run -e --contain -B /tmp:/tmp -B {}:{} {} mrgrid {}/dwmri_tensor_md.nii.gz regrid {}/dwmri_tensor_md_1mm_iso.nii.gz -voxel 1\n".format(kwargs['temp_dir'],kwargs['temp_dir'], mrtrix_simg, kwargs['temp_dir'], kwargs['temp_dir']))
        script.write("time singularity run -e --contain -B /tmp:/tmp -B {}:{} {} mrgrid {}/dwmri_tensor_ad.nii.gz regrid {}/dwmri_tensor_ad_1mm_iso.nii.gz -voxel 1\n".format(kwargs['temp_dir'],kwargs['temp_dir'], mrtrix_simg, kwargs['temp_dir'], kwargs['temp_dir']))
        script.write("time singularity run -e --contain -B /tmp:/tmp -B {}:{} {} mrgrid {}/dwmri_tensor_rd.nii.gz regrid {}/dwmri_tensor_rd_1mm_iso.nii.gz -voxel 1\n".format(kwargs['temp_dir'],kwargs['temp_dir'], mrtrix_simg, kwargs['temp_dir'], kwargs['temp_dir']))

        script.write("echo Done resampling to 1mm iso. Now running TractSeg...\n")
        #script.write('echo "..............................................................................."\n')
        #script.write("echo Loading FSL...\n")

        #script.write("export FSL_DIR=/accre/arch/easybuild/software/MPI/GCC/6.4.0-2.28/OpenMPI/2.1.1/FSL/5.0.10/fsl\n")
        #script.write("source setup_accre_runtime_dir\n")

        script.write("time singularity run -e --contain --home {} -B /tmp:/tmp -B {}:{} -B {}:{} {} TractSeg -i {}/dwmri_1mm_iso.nii.gz --raw_diffusion_input -o {}/tractseg --bvals {}/dwmri.bval --bvecs {}/dwmri.bvec\n".format(kwargs['temp_dir'], kwargs['accre_home'], kwargs['accre_home'], kwargs['temp_dir'], kwargs['temp_dir'], ts_simg, kwargs['temp_dir'], kwargs['temp_dir'], kwargs['temp_dir'], kwargs['temp_dir']))
        script.write('if [[ -f "{}/tractseg/peaks.nii.gz" ]]; then echo "Successfully created peaks.nii.gz for {}"; error_flag=0; else echo "Improper bvalue/bvector distribution for {}"; error_flag=1; fi\n'.format(kwargs['temp_dir'], kwargs['temp_dir'], kwargs['temp_dir']))
        script.write('if [[ $error_flag -eq 1 ]]; then echo "Improper bvalue/bvector distribution for {}" >> {}/report_bad_bvector.txt; fi\n\n'.format(kwargs['temp_dir'], kwargs['temp_dir']))        

        script.write("if [[ $error_flag -eq 1 ]]; then\n")
        script.write("echo Tractseg ran with error. Now exiting...\n")
        script.write("exit 1\n")
        script.write("fi\n\n")

        script.write("time singularity run -e --contain --home {} -B /tmp:/tmp -B {}:{} -B {}:{} {} TractSeg -i {}/tractseg/peaks.nii.gz -o {}/tractseg --output_type endings_segmentation\n".format(kwargs['temp_dir'], kwargs['accre_home'], kwargs['accre_home'], kwargs['temp_dir'], kwargs['temp_dir'], ts_simg, kwargs['temp_dir'], kwargs['temp_dir']))
        script.write("time singularity run -e --contain --home {} -B /tmp:/tmp -B {}:{} -B {}:{} {} TractSeg -i {}/tractseg/peaks.nii.gz -o {}/tractseg --output_type TOM\n".format(kwargs['temp_dir'], kwargs['accre_home'], kwargs['accre_home'], kwargs['temp_dir'], kwargs['temp_dir'], ts_simg, kwargs['temp_dir'], kwargs['temp_dir']))
        script.write("time singularity run -e --contain --home {} -B /tmp:/tmp -B {}:{} -B {}:{} {} Tracking -i {}/tractseg/peaks.nii.gz -o {}/tractseg --tracking_format tck\n".format(kwargs['temp_dir'], kwargs['accre_home'], kwargs['accre_home'], kwargs['temp_dir'], kwargs['temp_dir'], ts_simg, kwargs['temp_dir'], kwargs['temp_dir']))

        script.write('echo "..............................................................................."\n')
        script.write("echo Done running TractSeg. Now computing measures per bundle...\n")
        script.write("mkdir {}/tractseg/measures\n".format(kwargs['temp_dir']))

        script.write("for i in {}/tractseg/TOM_trackings/*.tck; do\n".format(kwargs['temp_dir']))
        script.write('    echo "$i"; s=${i##*/}; s=${s%.tck}; echo $s;\n')
        script.write('    time singularity exec -e --contain -B /tmp:/tmp -B {}:{} --nv {} scil_evaluate_bundles_individual_measures.py {}/tractseg/TOM_trackings/$s.tck {}/tractseg/measures/$s-SHAPE.json --reference={}/dwmri_1mm_iso.nii.gz\n'.format(kwargs['temp_dir'], kwargs['temp_dir'], scilus_simg, kwargs['temp_dir'], kwargs['temp_dir'], kwargs['temp_dir']))
        script.write('    time singularity exec -e --contain -B /tmp:/tmp -B {}:{} --nv {} scil_compute_bundle_mean_std.py {}/tractseg/TOM_trackings/$s.tck {}/dwmri_tensor_fa_1mm_iso.nii.gz {}/dwmri_tensor_md_1mm_iso.nii.gz {}/dwmri_tensor_ad_1mm_iso.nii.gz {}/dwmri_tensor_rd_1mm_iso.nii.gz --density_weighting --reference={}/dwmri_1mm_iso.nii.gz > {}/tractseg/measures/$s-DTI.json\n'.format(kwargs['temp_dir'], kwargs['temp_dir'], scilus_simg, kwargs['temp_dir'], kwargs['temp_dir'], kwargs['temp_dir'], kwargs['temp_dir'], kwargs['temp_dir'], kwargs['temp_dir'], kwargs['temp_dir']))
        script.write('done\n\n')

        script.write("echo Done computing measures per bundle. Now deleting temp inputs and re-organizing outputs...\n")
        if self.setup.args.tractseg_make_DTI:
            script.write("rm {}\n".format(tensor))
            script.write("rm {}\n".format(shellnii))
            script.write("rm {}\n".format(shellbval))
            script.write("rm {}\n".format(shellbvec))
        script.write("rm -r {}/dwi2response-tmp-*\n".format(kwargs['temp_dir']))
        script.write("rm -r {}/.cache\n".format(kwargs['temp_dir']))
        script.write("rm -r {}/tractseg/TOM\n".format(kwargs['temp_dir']))
        script.write("rm -r {}/tractseg/peaks.nii.gz\n".format(kwargs['temp_dir']))
        script.write("rm {}/dwmri.nii.gz\n".format(kwargs['temp_dir']))
        script.write("rm {}/mask.nii.gz\n".format(kwargs['temp_dir']))
        script.write("rm {}/dwmri_1mm_iso.nii.gz\n".format(kwargs['temp_dir']))
        script.write("rm {}/dwmri_tensor_fa.nii.gz\n".format(kwargs['temp_dir']))
        script.write("rm {}/dwmri_tensor_md.nii.gz\n".format(kwargs['temp_dir']))
        script.write("rm {}/dwmri_tensor_ad.nii.gz\n".format(kwargs['temp_dir']))
        script.write("rm {}/dwmri_tensor_rd.nii.gz\n".format(kwargs['temp_dir']))
        script.write("mkdir {}/bundles\n".format(kwargs['temp_dir']))
        script.write("mv {}/tractseg/TOM_trackings/* {}/bundles\n".format(kwargs['temp_dir'], kwargs['temp_dir']))
        script.write("mv {}/tractseg/measures {}/\n".format(kwargs['temp_dir'], kwargs['temp_dir']))

        script.write("echo Done re-organizing outputs. Now copying back to output directory...\n")


    def generate_tractseg_scripts(self):
        """
        Generates Tractseg scripts
        """

        root_temp = Path(self.setup.args.temp_dir)
        assert root_temp.exists() and os.access(root_temp, os.W_OK), "Error: Root temp directory {} does not exist or is not writable".format(root_temp)

        #get the accre home directory / home directory for the tractseg inputs
        if self.setup.args.no_accre:
            if self.setup.args.custom_home != '':
                accre_home_directory = self.setup.args.custom_home
            else:
                accre_home_directory = os.path.expanduser("~")
                user = accre_home_directory.split('/')[-1]
                accre_home_directory = "/home/local/VANDERBILT/{}/".format(user)
        else:
            accre_home_directory = os.path.expanduser("~")     
        
        prequal_dirs = self.get_PreQual_dirs()

        for pqdir_p in tqdm(prequal_dirs):
            pqdir = Path(pqdir_p)
            #get the BIDS tags
            sub, ses, acq, run = self.get_BIDS_fields_from_PQdir(pqdir) 

            #check to see if the Tractseg outputs already exist
            tractseg_dir = self.setup.dataset_derivs/(sub)/(ses)/("Tractseg{}{}".format(acq, run))
            if self.has_Tractseg_outputs(tractseg_dir):
                continue

            #check to see if the PreQual outputs exist
            if not all(self.check_PQ_outputs(pqdir)):
                self.add_to_missing(sub, ses, acq, run, 'PreQual')
                continue

            #get the tensor maps to use for the mean/std bundle calculations
            if self.setup.args.tractseg_make_DTI:
                #create own tensor maps
                pass
            else:
                eve3dir = self.setup.dataset_derivs/(sub)/(ses)/("WMAtlasEVE3{}{}".format(acq, run))
                assert eve3dir.parent.name != "None" and (eve3dir.parent.name == ses or eve3dir.parent.name == sub), "Error in Tractseg generation - EVE3 directory naming is wrong {}".format(eve3dir)
                tensor_maps = [eve3dir/("dwmri%{}.nii.gz".format(m)) for m in ['fa', 'md', 'ad', 'rd']]
                using_PQ = False
                if not all([t.exists() for t in tensor_maps]):
                    #row = {'sub':sub, 'ses':ses, 'acq':acq, 'run':run, 'missing':'tensor_maps'}
                    #missing_data = pd.concat([missing_data, pd.Series(row).to_frame().T], ignore_index=True)
                    self.add_to_missing(sub, ses, acq, run, 'tensor_maps')
                    continue

            self.count += 1

            #create the temp session directories
            session_temp = root_temp/(sub)/("{}{}{}".format(ses,acq,run))
            (session_input, session_output, session_temp) = self.make_session_dirs(sub, ses, acq, run, tmp_input_dir=self.setup.tmp_input_dir,
                                            tmp_output_dir=self.setup.tmp_output_dir, temp_dir=root_temp, has_temp=True)

            #create the target output directory
            tractseg_target = self.setup.output_dir/(sub)/(ses)/("Tractseg{}{}".format(acq, run))
            if not tractseg_target.exists():
                os.makedirs(tractseg_target)

            #setup the inputs dictionary and outputs list
            if self.setup.args.tractseg_make_DTI:
                self.inputs_dict[self.count] = {
                    'pq_dwi_dir': {'src_path': pqdir/'PREPROCESSED', 'targ_name': '', 'directory': True, 'separate_input': session_temp}
                }
            else:
                self.inputs_dict[self.count] = {
                    'fa_map': {'src_path': eve3dir/'dwmri%fa.nii.gz', 'targ_name': 'dwmri_tensor_fa.nii.gz', 'separate_input': session_temp},
                    'md_map': {'src_path': eve3dir/'dwmri%md.nii.gz', 'targ_name': 'dwmri_tensor_md.nii.gz', 'separate_input': session_temp},
                    'ad_map': {'src_path': eve3dir/'dwmri%ad.nii.gz', 'targ_name': 'dwmri_tensor_ad.nii.gz', 'separate_input': session_temp},
                    'rd_map': {'src_path': eve3dir/'dwmri%rd.nii.gz', 'targ_name': 'dwmri_tensor_rd.nii.gz', 'separate_input': session_temp},
                    'pq_dwi_dir': {'src_path': pqdir/'PREPROCESSED', 'targ_name': '', 'directory': True, 'separate_input': session_temp}
                }
            self.outputs[self.count] = []


            #TODO: make sure the bundles and measures are copied properly and some of the outputs are removed

            #start the script generation
            self.start_script_generation(session_input, session_output, deriv_output_dir=tractseg_target, temp_dir=session_temp,
                                        tractseg_setup=True, accre_home=accre_home_directory, temp_is_output=True)
   
class FrancoisSpecialGenerator(ScriptGenerator):
    """
    Class for generating FrancoisSpecial scripts
    """
    
    def __init__(self, setup_object):
        super().__init__(setup_object=setup_object)

        #self.warnings = {}
        self.outputs = {}
        self.inputs_dict = {}

        self.generate_francois_scripts()
    
    def francois_special_script_generate(self, script, session_input, session_output, **kwargs):
        """
        Writes functionality for running Francois Special to a single script
        """

        tmp_dir = Path("/tmp/{}/{}".format(session_input.parent.name, session_input.name))
        script.write("echo Creating tmp directory...\n")
        script.write("mkdir -p {}\n".format(tmp_dir))

        script.write("echo Running Francois Special...\n")
        if kwargs['ses'] != '':
            script.write('singularity run --home {} --bind {}:/tmp --containall --cleanenv --bind {}:/INPUTS/ --bind {}:/OUTPUTS/ --bind {}:/TMP {} {} {} t1.nii.gz dwmri.nii.gz dwmri.bval dwmri.bvec {} {} {} 5 wm 5 wm prob 27 0.4 20 t1_seg.nii.gz "4 40 41 44 45 51 52 11 49 59 208 209"\n'.format(tmp_dir, tmp_dir, session_input, session_output, tmp_dir, self.setup.simg, kwargs['sub'], kwargs['ses'], kwargs['sh_degree'], kwargs['dti_shells'], kwargs['fodf_shells']))
        else:
            script.write('singularity run --home {} --bind {}:/tmp --containall --cleanenv --bind {}:/INPUTS/ --bind {}:/OUTPUTS/ --bind {}:/TMP {} {} {} t1.nii.gz dwmri.nii.gz dwmri.bval dwmri.bvec {} {} {} 5 wm 5 wm prob 27 0.4 20 t1_seg.nii.gz "4 40 41 44 45 51 52 11 49 59 208 209"\n'.format(tmp_dir, tmp_dir, session_input, session_output, tmp_dir, self.setup.simg, kwargs['sub'], 'None', kwargs['sh_degree'], kwargs['dti_shells'], kwargs['fodf_shells']))
            
        script.write("echo Finished running Francois Special. Now removing inputs and copying outputs back...\n")


    def generate_francois_scripts(self):
        """
        Starts the generation of Francois Special scripts
        """

        prequal_dirs = self.get_PreQual_dirs()

        for pqdir_p in tqdm(prequal_dirs):
            pqdir = Path(pqdir_p)
            #get the BIDS tags
            sub, ses, acq, run = self.get_BIDS_fields_from_PQdir(pqdir) 

            #check to see if the Francois Special outputs already exist
            francois_dir = pqdir.parent/("FrancoisSpecial{}{}".format(acq, run))
            if (francois_dir/("report.pdf")).exists():
                continue

            #check to see if the PreQual outputs exist
            if not all(self.check_PQ_outputs(pqdir)):
                self.add_to_missing(sub, ses, acq, run, 'PreQual')
                continue

            #get the T1 that was used for PreQual. If we cannot get it, then select it again using the function
            t1 = self.get_prov_t1(pqdir)
            if not t1:
                #get the T1 using the function, and print out a provenance warning
                print("Warning: Could not get provenance T1 for {}/{}/{}".format(sub,ses,pqdir))
                t1 = self.get_t1(self.setup.root_dataset_path/(sub)/(ses)/("anat"), sub, ses)
                if not t1:
                    self.add_to_missing(sub, ses, acq, run, 'T1')
                    continue
            
            #based on the T1, get the TICV/UNest segmentation
            ses_deriv = pqdir.parent
            if not self.setup.args.use_unest_seg:
                seg = self.get_TICV_seg_file(t1, ses_deriv)
            else:
                seg = self.get_UNest_seg_file(t1, ses_deriv)
            if not seg.exists():
                self.add_to_missing(sub, ses, acq, run, 'TICV' if not self.setup.args.use_unest_seg else 'UNest')
                continue

            #need to get the orignal DWI(s) to check for the number of shells and bvals
            og_dwis = self.get_prov_dwi(pqdir)
            if not og_dwis:
                #no provenance DWIs found
                #legacy: backtrack to have it check the subject and session
                og_dwis = self.legacy_get_dwis(sub, ses, acq, run)
                if not og_dwis:
                    self.add_to_missing(sub, ses, acq, run, 'raw_legacy_DWIs')
                    continue
                #assert False, "Error: No provenance DWIs found for {}/{}/{}".format(sub, ses, pqdir)
            (dwis, bvals, bvecs) = og_dwis
            if type(dwis) is not list:
                dwis = [dwis]
                bvals = [bvals]
                bvecs = [bvecs]
            rounded_bvals, dti_shells, fodf_shells, sh8_shells = get_shells_and_dirs(pqdir, bvals, bvecs)
            #make sure that these are valid
            if dti_shells is None and fodf_shells is None:
                self.add_to_missing(sub, ses, acq, run, 'dti_and_fodf_shells')
                #missing_data = pd.concat([missing_data, pd.Series(row).to_frame().T], ignore_index=True)
                continue
            elif dti_shells == "No b0" or fodf_shells == "No b0":
                self.add_to_missing(sub, ses, acq, run, 'No b0')
                #row = {'sub':sub, 'ses':ses, 'acq':acq, 'run':run, 'missing':'No_b0'}
                #missing_data = pd.concat([missing_data, pd.Series(row).to_frame().T], ignore_index=True)
                continue
            elif dti_shells == None:
                self.add_to_missing(sub, ses, acq, run, 'dti_shells')
                #row = {'sub':sub, 'ses':ses, 'acq':acq, 'run':run, 'missing':'dti_shells'}
                #missing_data = pd.concat([missing_data, pd.Series(row).to_frame().T], ignore_index=True)
                continue
            elif fodf_shells == None:
                self.add_to_missing(sub, ses, acq, run, 'fodf_shells')
                #row = {'sub':sub, 'ses':ses, 'acq':acq, 'run':run, 'missing':'fodf_shells'}
                #missing_data = pd.concat([missing_data, pd.Series(row).to_frame().T], ignore_index=True)
                continue

            #if the sh8 shells are the same as the fodf shells, then we can use sh degree of 8
            if sh8_shells == fodf_shells:
                sh_degree = 8
            else:
                sh_degree = 6

            self.count += 1

            #create the temp session directories
            (session_input, session_output) = self.make_session_dirs(sub, ses, acq, run, tmp_input_dir=self.setup.tmp_input_dir,
                                            tmp_output_dir=self.setup.tmp_output_dir)
            
            #create the target output directory
            fs_target = self.setup.output_dir/(sub)/(ses)/("FrancoisSpecial{}{}".format(acq, run))
            if not fs_target.exists():
                os.makedirs(fs_target)

            #setup the inputs dictionary and outputs list            
            self.inputs_dict[self.count] = {'t1': {'src_path': t1, 'targ_name': 't1.nii.gz'},
                                            'seg': {'src_path': seg, 'targ_name': 't1_seg.nii.gz'},
                                            'pq_dwi_dir': {'src_path': pqdir/'PREPROCESSED', 'targ_name': '', 'directory': True}
                                        }
            self.outputs[self.count] = []

            #start the script generation
            self.start_script_generation(session_input, session_output, deriv_output_dir=fs_target, sh_degree=sh_degree,
                                         dti_shells=dti_shells, fodf_shells=fodf_shells, sub=sub, ses=ses)
            
            ### TODO: Add the shdegree to the config.yml file

class NODDIGenerator(ScriptGenerator):
    """
    Class for generating NODDI scripts
    """

    def __init__(self, setup_object):
        super().__init__(setup_object=setup_object)

        self.warnings = {}
        self.outputs = {}
        self.inputs_dict = {}

        self.generate_noddi_scripts()


    def noddi_script_generate(self, script, session_input, session_output, **kwargs):
        """
        Writes commands for running NODDI to a script
        """

        script.write("echo Running setup for NODDI...\n")
        script.write("echo Renaming dwi for some reason...\n")
        script.write("singularity exec -e --contain -B /tmp:/tmp -B {}:{} {} mrconvert {}/dwmri.nii.gz {}/Diffusion.nii.gz -force\n".format(session_input, session_input, self.setup.simg, session_input, session_input))
        script.write("mask=dwimean_masked_mask.nii.gz\n")
        if not kwargs['dwi_mask']:
            script.write("echo DWI mask not found. Creating dwi mask using Kurts method...\n")
            script.write("singularity exec -e --contain -B /tmp:/tmp -B {}:{} {} scil_extract_dwi_shell.py {}/Diffusion.nii.gz {}/Diffusion.bval {}/Diffusion.bvec 750 {}/dwi_tmp.nii.gz {}/bvals_tmp {}/bvecs_tmp -f -v --tolerance 650\n".format(session_input, session_input, self.setup.simg, session_input, session_input, session_input, session_input, session_input, session_input))
            script.write("singularity exec -e --contain -B /tmp:/tmp -B {}:{} {} mrmath {}/dwi_tmp.nii.gz mean {}/dwimean.nii.gz -axis 3 -force\n".format(session_input, session_input, self.setup.simg, session_input, session_input))
            script.write("bet {}/dwimean.nii.gz {}/dwimean_masked -m -R -f .3\n".format(session_input, session_input))
        script.write("echo Done running setup. Now running NODDI...\n")
        script.write("singularity exec -e --contain -B /tmp:/tmp -B {}:{} -B {}:{} {} scil_compute_NODDI.py {}/Diffusion.nii.gz {}/Diffusion.bval {}/Diffusion.bvec --out_dir {} --mask {}/dwimean_masked_mask.nii.gz -f\n".format(session_input, session_input, session_output, session_output, self.setup.simg, session_input, session_input, session_input, session_output, session_input))
        script.write("echo Finished running NODDI. Now removing inputs and copying outputs back...\n")

    def generate_noddi_scripts(self):
        """
        Generates NODDI scripts for a dataset
        """

        prequal_dirs = self.get_PreQual_dirs()

        for pqdir_p in tqdm(prequal_dirs):
            pqdir = Path(pqdir_p)
            #get the BIDS tags
            sub, ses, acq, run = self.get_BIDS_fields_from_PQdir(pqdir)

            #check to see if the NODDI outputs already exist
            noddi_dir = self.setup.dataset_derivs/(sub)/(ses)/("NODDI{}{}".format(acq, run))
            if self.has_NODDI_outputs(noddi_dir):
                continue

            #check to see if the PreQual outputs exist
            if not all(self.check_PQ_outputs(pqdir)):
                self.add_to_missing(sub, ses, acq, run, 'PreQual')
                continue

            #check the number of shells
            bval_file = pqdir/("PREPROCESSED")/("dwmri.bval")
            if get_num_shells(bval_file) < 2:
                self.add_to_missing(sub, ses, acq, run, 'single_shell')
                continue

            #check the EVE3 registration outputs to see if there is a DWI mask to use
            if ses != '':
                eve3dir = self.setup.dataset_derivs/(sub)/(ses)/("WMAtlasEVE3{}{}".format(acq, run))
            else:
                eve3dir = self.setup.dataset_derivs/(sub)/("WMAtlasEVE3{}{}".format(acq, run))
            dwi_mask = eve3dir/("dwmri%_dwimask.nii.gz")
            if not dwi_mask.exists():
                #need to use the code that kurt gave
                print("WARNING: DWI mask not found for {}_{}_{}. Will create dwi mask using Kurt's method...".format(sub,ses,pqdir.name))
                dwi_mask = None
                self.add_to_missing(sub, ses, acq, run, 'DWI_mask')

            self.count += 1

            self.warnings[self.count] = ''
            if dwi_mask is None:
                self.warnings[self.count] += 'DWI mask not found. Will use Kurt\'s method to create mask.\n'
            else:
                self.warnings[self.count] += 'Using DWI mask found in WMAtlasEVE3 (i.e. T1 segmentation mask moved to DWI space)...\n'

            #setup the temp directories
            (session_input, session_output) = self.make_session_dirs(sub, ses, acq, run, tmp_input_dir=self.setup.tmp_input_dir,
                                            tmp_output_dir=self.setup.tmp_output_dir)
            
            #create the output target directory
            noddi_target = self.setup.output_dir/(sub)/(ses)/("NODDI{}{}".format(acq, run))
            if not noddi_target.exists():
                os.makedirs(noddi_target)

            #setup the inputs dictionary and outputs list
            self.inputs_dict[self.count] = {'pq_dwi': {'src_path': pqdir/'PREPROCESSED'/'dwmri.nii.gz', 'targ_name': 'dwmri.nii.gz'},
                                            'pq_bval': {'src_path': pqdir/'PREPROCESSED'/'dwmri.bval', 'targ_name': 'Diffusion.bval'},
                                            'pq_bvec': {'src_path': pqdir/'PREPROCESSED'/'dwmri.bvec', 'targ_name': 'Diffusion.bvec'}    
                                        }
            if dwi_mask:
                self.inputs_dict[self.count]['dwi_mask'] = {'src_path': dwi_mask, 'targ_name': 'dwimean_masked_mask.nii.gz'}
            self.outputs[self.count] = []

            #start the script generation
            self.start_script_generation(session_input, session_output, deriv_output_dir=noddi_target, dwi_mask=dwi_mask)

class FreewaterGenerator(ScriptGenerator):
    """
    Class for generating freewater scripts
    """

    def __init__(self, setup_object):
        super().__init__(setup_object=setup_object)

        self.warnings = {}
        self.outputs = {}
        self.inputs_dict = {}

        self.generate_freewater_scripts()

    def freewater_script_generate(self, script, session_input, session_output, **kwargs):
        """
        Method for writing the commands for running freewater to a script
        """
        
        script.write("echo Running FreeWater...\n")
        script.write("singularity run -e --contain -B {}:/input -B {}:/output --home {} -B /tmp:/tmp -B {}:/dev/shm {} dwmri.nii.gz dwmri.bval dwmri.bvec mask.nii.gz\n".format(session_input, session_output, session_input, session_input, self.setup.simg))
        #singularity run -B inputs/:/input -B outputs/:/output ../FreeWaterEliminationv2.sif dwmri.nii.gz dwmri.bval dwmri.bvec mask.nii.gz

    def generate_freewater_scripts(self):
        """
        Method for generating freewater scripts for a dataset
        """

        #first, get the PreQual directories
        prequal_dirs = self.get_PreQual_dirs()

        for pqdir_p in tqdm(prequal_dirs):
            pqdir = Path(pqdir_p)
            #get the BIDS tags
            sub, ses, acq, run = self.get_BIDS_fields_from_PQdir(pqdir)
            #check to see if the freewater outputs already exist
            freewater_dir = self.setup.dataset_derivs/(sub)/(ses)/("freewater{}{}".format(acq, run))
            if self.has_freewater_outputs(freewater_dir):
                continue

            #check to see if the PreQual outputs exist
            if not all(self.check_PQ_outputs(pqdir)):
                self.add_to_missing(sub, ses, acq, run, 'PreQual')
                continue

            #all the inputs exist and the outputs do not, so we can generate the script

            self.count += 1

            #setup the temp directories
            (session_input, session_output) = self.make_session_dirs(sub, ses, acq, run, tmp_input_dir=self.setup.tmp_input_dir,
                                            tmp_output_dir=self.setup.tmp_output_dir)
            
            #create the output target directory
            fw_target = self.setup.output_dir/(sub)/(ses)/("freewater{}{}".format(acq, run))
            if not fw_target.exists():
                os.makedirs(fw_target)

            #setup the inputs dictionary and outputs list
            self.inputs_dict[self.count] = {
                'pq_dwi_dir': {'src_path': pqdir/'PREPROCESSED', 'targ_name': '', 'directory': True}
            }
            self.outputs[self.count] = []

            #start the script generation
            self.start_script_generation(session_input, session_output, deriv_output_dir=fw_target)

#for Kurt: DTI + Tractseg on preprocessed data
class DWI_plus_TractsegGenerator(ScriptGenerator):

    def __init__(self, setup_object):
        """
        Class for taking dwi data from raw, computing the DTI metrics, and running TractSeg on the computed metrics
        """
        super().__init__(setup_object=setup_object)
        #self.warnings = {}
        self.outputs = {}
        self.inputs_dict = {}

        self.generate_dwi_plus_tractseg_scripts()

    def dwi_plus_tractseg_script_generate(self, script, session_input, session_output, **kwargs):
        """
        Writes the command for running DTI + Tractseg to a script

        inputs:
            paths to 3 simgs
            dwi files copied into the temp directory
            temp directory (basically output directory)
        """

        #define the singularities
        ts_simg = self.setup.simg[0]
        scilus_simg = self.setup.simg[1]
        mrtrix_simg = self.setup.simg[1]
        dwi_simg = self.setup.simg[2]

        #define the directories to bind
        dti_dir = '{}/DTI'.format(kwargs['temp_dir'])
        bind1 = "{}:/INPUTS".format(kwargs['temp_dir'])
        bind2 = "{}:/OUTPUTS".format(dti_dir)

        #first, convert the dwi to tensor
        script.write("echo Making temp directories...\n")
        script.write("mkdir -p {}\n".format(dti_dir))
        script.write("echo Shelling to 1500...\n")
        script.write("time singularity exec -e --contain -B /tmp:/tmp -B {} -B {} {} python3 /CODE/extract_single_shell.py\n".format(bind1, bind2, dwi_simg))
        script.write("echo Done shelling to 1500. Now fitting tensors DTI...\n")

        #1.) extract the b0
        #2.) create the mask
        #3.) fit the tensors
        #4.) calculate the FA, MD, AD, RD

        script.write("echo Extracting b0...\n")
        b0 = "{}/dwmri_b0.nii.gz".format(dti_dir)
        script.write("time singularity exec -e --contain -B /tmp:/tmp -B {}:{} {} bash -c \"dwiextract {}/dwmri.nii.gz -fslgrad {}/dwmri.bvec {}/dwmri.bval - -bzero | mrmath - mean {} -axis 3\"\n".format(kwargs['temp_dir'], kwargs['temp_dir'], dwi_simg, kwargs['temp_dir'], kwargs['temp_dir'], kwargs['temp_dir'], b0))
        script.write("echo Creating mask...\n")
        mask = "{}/dwmri_mask.nii.gz".format(dti_dir)
        #script.write("cp {}/mask.nii.gz {}\n".format(kwargs['temp_dir'], mask))
        script.write("time singularity exec -e --contain -B /tmp:/tmp -B {}:{} {} bet {} {} -f 0.25 -m -n -R\n".format(kwargs['temp_dir'], kwargs['temp_dir'], dwi_simg, b0, mask))
        script.write("echo Fitting tensors...\n")
        shellbvec = "{}/dwmri%firstshell.bvec".format(dti_dir)
        shellbval = "{}/dwmri%firstshell.bval".format(dti_dir)
        shellnii = "{}/dwmri%firstshell.nii.gz".format(dti_dir)
        tensor = "{}/dwmri_tensor.nii.gz".format(dti_dir)
        script.write("time singularity exec -e --contain -B /tmp:/tmp -B {}:{} {} dwi2tensor -fslgrad {} {} {} {}\n".format(kwargs['temp_dir'], kwargs['temp_dir'], dwi_simg, shellbvec, shellbval, shellnii, tensor))
        script.write("echo Calculating FA, MD, AD, RD...\n")
        fa = "{}/dwmri_tensor_fa.nii.gz".format(dti_dir)
        md = "{}/dwmri_tensor_md.nii.gz".format(dti_dir)
        ad = "{}/dwmri_tensor_ad.nii.gz".format(dti_dir)
        rd = "{}/dwmri_tensor_rd.nii.gz".format(dti_dir)
        script.write("time singularity exec -e --contain -B /tmp:/tmp -B {}:{} {} tensor2metric {} -fa {} -mask {}\n".format(kwargs['temp_dir'], kwargs['temp_dir'], dwi_simg, tensor, fa, mask))
        script.write("time singularity exec -e --contain -B /tmp:/tmp -B {}:{} {} tensor2metric {} -adc {} -mask {}\n".format(kwargs['temp_dir'], kwargs['temp_dir'], dwi_simg, tensor, md, mask))
        script.write("time singularity exec -e --contain -B /tmp:/tmp -B {}:{} {} tensor2metric {} -ad {} -mask {}\n".format(kwargs['temp_dir'], kwargs['temp_dir'], dwi_simg, tensor, ad, mask))
        script.write("time singularity exec -e --contain -B /tmp:/tmp -B {}:{} {} tensor2metric {} -rd {} -mask {}\n".format(kwargs['temp_dir'], kwargs['temp_dir'], dwi_simg, tensor, rd, mask))
        
        #### NEED TO REWRITE THE REST BELOW TO USE THE DTI OUTPUTS STRUCTURED AS ABOVE

        script.write("echo Resampling to 1mm iso...\n")
        isodwi = "{}/dwmri_1mm_iso.nii.gz".format(kwargs['temp_dir'])
        script.write("time singularity run -e --contain -B /tmp:/tmp -B {}:{} {} mrgrid {}/dwmri.nii.gz regrid {} -voxel 1\n".format(kwargs['temp_dir'], kwargs['temp_dir'], mrtrix_simg, kwargs['temp_dir'], isodwi))
        #same for the fa, md, ad, rd
        faiso = "{}/dwmri_tensor_fa_1mm_iso.nii.gz".format(dti_dir)
        mdiso = "{}/dwmri_tensor_md_1mm_iso.nii.gz".format(dti_dir)
        adiso = "{}/dwmri_tensor_ad_1mm_iso.nii.gz".format(dti_dir)
        rdiso = "{}/dwmri_tensor_rd_1mm_iso.nii.gz".format(dti_dir)
        script.write("time singularity run -e --contain -B /tmp:/tmp -B {}:{} {} mrgrid {} regrid {} -voxel 1\n".format(dti_dir, dti_dir, mrtrix_simg, fa, faiso))
        script.write("time singularity run -e --contain -B /tmp:/tmp -B {}:{} {} mrgrid {} regrid {} -voxel 1\n".format(dti_dir, dti_dir, mrtrix_simg, md, mdiso))
        script.write("time singularity run -e --contain -B /tmp:/tmp -B {}:{} {} mrgrid {} regrid {} -voxel 1\n".format(dti_dir, dti_dir, mrtrix_simg, ad, adiso))
        script.write("time singularity run -e --contain -B /tmp:/tmp -B {}:{} {} mrgrid {} regrid {} -voxel 1\n".format(dti_dir, dti_dir, mrtrix_simg, rd, rdiso))

        script.write("echo Done resampling to 1mm iso. Now running TractSeg...\n")
        script.write('echo "..............................................................................."\n')
        #if not self.setup.args.no_accre:
        #    script.write("echo Loading FSL...\n")
        #    script.write("export FSL_DIR=/accre/arch/easybuild/software/MPI/GCC/6.4.0-2.28/OpenMPI/2.1.1/FSL/5.0.10/fsl\n")
        #    script.write("source setup_accre_runtime_dir\n")

        tractsegdir = "{}/tractseg".format(kwargs['temp_dir'])
        script.write("time singularity run -e --contain -B /tmp:/tmp -B {}:{} -B {}:{} {} TractSeg -i {} --raw_diffusion_input -o {} --bvals {}/dwmri.bval --bvecs {}/dwmri.bvec\n".format(kwargs['accre_home'], kwargs['accre_home'], kwargs['temp_dir'], kwargs['temp_dir'], ts_simg, isodwi, tractsegdir, kwargs['temp_dir'], kwargs['temp_dir']))
        script.write('if [[ -f "{}/peaks.nii.gz" ]]; then echo "Successfully created peaks.nii.gz for {}"; error_flag=0; else echo "Improper bvalue/bvector distribution for {}"; error_flag=1; fi\n'.format(tractsegdir, kwargs['temp_dir'], kwargs['temp_dir']))
        script.write('if [[ $error_flag -eq 1 ]]; then echo "Improper bvalue/bvector distribution for {}" >> {}/report_bad_bvector.txt; fi\n\n'.format(kwargs['temp_dir'], kwargs['temp_dir']))        

        script.write("if [[ $error_flag -eq 1 ]]; then\n")
        script.write("echo Tractseg ran with error. Now exiting...\n")
        script.write("exit 1\n")
        script.write("fi\n\n")

        script.write("time singularity run -e --contain -B /tmp:/tmp -B {}:{} -B {}:{} {} TractSeg -i {}/peaks.nii.gz -o {} --output_type endings_segmentation\n".format(kwargs['accre_home'], kwargs['accre_home'], kwargs['temp_dir'], kwargs['temp_dir'], ts_simg, tractsegdir, tractsegdir))
        script.write("time singularity run -e --contain -B /tmp:/tmp -B {}:{} -B {}:{} {} TractSeg -i {}/peaks.nii.gz -o {} --output_type TOM\n".format(kwargs['accre_home'], kwargs['accre_home'], kwargs['temp_dir'], kwargs['temp_dir'], ts_simg, tractsegdir, tractsegdir))
        script.write("time singularity run -e --contain -B /tmp:/tmp -B {}:{} -B {}:{} {} Tracking -i {}/peaks.nii.gz -o {} --tracking_format tck\n".format(kwargs['accre_home'], kwargs['accre_home'], kwargs['temp_dir'], kwargs['temp_dir'], ts_simg, tractsegdir, tractsegdir))

        script.write('echo "..............................................................................."\n')
        script.write("echo Done running TractSeg. Now computing measures per bundle...\n")
        script.write("mkdir {}/measures\n".format(tractsegdir))

        trackingdir = "{}/TOM_trackings".format(tractsegdir)
        measuresdir = "{}/measures".format(tractsegdir)
        script.write("for i in {}/TOM_trackings/*.tck; do\n".format(tractsegdir))
        script.write('    echo "$i"; s=${i##*/}; s=${s%.tck}; echo $s;\n')
        script.write('    time singularity exec -e --contain -B /tmp:/tmp -B {}:{} --nv {} scil_evaluate_bundles_individual_measures.py {}/$s.tck {}/$s-SHAPE.json --reference={}\n'.format(kwargs['temp_dir'], kwargs['temp_dir'], scilus_simg, trackingdir, measuresdir, isodwi))
        script.write('    time singularity exec -e --contain -B /tmp:/tmp -B {}:{} --nv {} scil_compute_bundle_mean_std.py {}/$s.tck {} {} {} {} --density_weighting --reference={} > {}/$s-DTI.json\n'.format(kwargs['temp_dir'], kwargs['temp_dir'], scilus_simg, trackingdir, faiso, mdiso, adiso, rdiso, isodwi, measuresdir))
        script.write('done\n\n')

        script.write("echo Done computing measures per bundle. Now deleting temp inputs and re-organizing outputs...\n")
        script.write("rm -r {}/tractseg/TOM\n".format(kwargs['temp_dir']))
        script.write("rm -r {}/tractseg/peaks.nii.gz\n".format(kwargs['temp_dir']))
        script.write("rm -r {}/dwmri.nii.gz\n".format(kwargs['temp_dir']))
        script.write("rm -r {}/dwmri_1mm_iso.nii.gz\n".format(kwargs['temp_dir']))
        #script.write("rm {}/dwmri_tensor_fa.nii.gz\n".format(kwargs['temp_dir']))
        #script.write("rm {}/dwmri_tensor_md.nii.gz\n".format(kwargs['temp_dir']))
        #script.write("rm {}/dwmri_tensor_ad.nii.gz\n".format(kwargs['temp_dir']))
        #script.write("rm {}/dwmri_tensor_rd.nii.gz\n".format(kwargs['temp_dir']))
        script.write("mv {}/tractseg/TOM_trackings {}/bundles\n".format(kwargs['temp_dir'], kwargs['temp_dir']))
        script.write("mv {}/tractseg/measures {}/\n".format(kwargs['temp_dir'], kwargs['temp_dir']))

        script.write("echo Done re-organizing outputs. Now copying back to output directory...\n")

    def generate_dwi_plus_tractseg_scripts(self):
        """
        Generates the scripts for running DTI + TractSeg
        """

        root_temp = Path(self.setup.args.temp_dir)
        assert root_temp.exists() and os.access(root_temp, os.W_OK), "Error: Root temp directory {} does not exist or is not writable".format(root_temp)

        #get the accre home directory / home directory for the tractseg inputs
        if self.setup.args.no_accre:
            if self.setup.args.custom_home != '':
                accre_home_directory = self.setup.args.custom_home
            else:
                accre_home_directory = os.path.expanduser("~")
                user = accre_home_directory.split('/')[-1]
                accre_home_directory = "/home/local/VANDERBILT/{}/".format(user)
        else:
            accre_home_directory = os.path.expanduser("~")   

        #get the raw dwi files
        dwis = self.find_dwis()

        for dwi_p in tqdm(dwis):
            dwi = Path(dwi_p)

            #get the sub, ses, acq, run
            sub, ses, acq, run = self.get_BIDS_fields_dwi(dwi)

            #check to see if the outputs exist already
                #NOTE: this will only check the specified output directory, not the BIDS directory
            tractseg_dir = self.setup.output_dir/(sub)/(ses)/("DWI_plus_Tractseg{}{}".format(acq, run))
            if self.has_Tractseg_outputs(tractseg_dir):
                continue

            #get the bval and bvec files for the dwis
            bval = Path(dwi_p.replace('dwi.nii.gz', 'dwi.bval'))
            bvec = Path(dwi_p.replace('dwi.nii.gz', 'dwi.bvec'))

            #make sure that the bval and bvec and nii files exist
            assert dwi.exists() and bval.exists() and bvec.exists(), "Error: dwi, bval, or bvec file does not exist for {}".format(dwi)

            self.count += 1

            #create the temp session directories
            session_temp = root_temp/(sub)/("{}{}{}".format(ses,acq,run))
            (session_input, session_output, session_temp) = self.make_session_dirs(sub, ses, acq, run, tmp_input_dir=self.setup.tmp_input_dir,
                                            tmp_output_dir=self.setup.tmp_output_dir, temp_dir=root_temp, has_temp=True)

            #create the target output directory
            tractsegdwi_target = self.setup.output_dir/(sub)/(ses)/("DWI_plus_Tractseg{}{}".format(acq, run))
            if not tractsegdwi_target.exists():
                os.makedirs(tractsegdwi_target)

            #setup the inputs dictionary
                #need dwi, bval, bvec
            self.inputs_dict[self.count] = {
                'dwi': {'src_path': dwi, 'targ_name': 'dwmri.nii.gz', 'separate_input': session_temp},
                'bval': {'src_path': bval, 'targ_name': 'dwmri.bval', 'separate_input': session_temp},
                'bvec': {'src_path': bvec, 'targ_name': 'dwmri.bvec', 'separate_input': session_temp}
            }
            self.outputs[self.count] = []

            #start the script generation
            self.start_script_generation(session_input, session_output, deriv_output_dir=tractsegdwi_target, temp_dir=session_temp,
                                        tractseg_setup=True, accre_home=accre_home_directory, temp_is_output=True)

#for Kurt: bedpostX + DTI + Tractseg on preprocessed data
class BedpostX_plus_DWI_plus_TractsegGenerator(ScriptGenerator):

    def __init__(self, setup_object):
        """
        Class for taking dwi data from raw, computing the DTI metrics, and running TractSeg on the computed metrics
        """
        super().__init__(setup_object=setup_object)
        #self.warnings = {}
        self.outputs = {}
        self.inputs_dict = {}

        self.generate_bedpostx_plus_dwi_plus_tractseg_scripts()


    def bedpostx_plus_dwi_plus_tractseg_script_generate(self, script, session_input, session_output, **kwargs):
        """
        Writes a single script for running bedpostx + DTI + Tractseg
        """
        #define the singularities
        if self.setup.args.accre_gpu:
            ts_simg = self.setup.simg[0]
            scilus_simg = self.setup.simg[1]
            mrtrix_simg = self.setup.simg[0]
            dwi_simg = self.setup.simg[2]
        else:
            ts_simg = self.setup.simg[0]
            scilus_simg = self.setup.simg[1]
            mrtrix_simg = self.setup.simg[1]
            dwi_simg = self.setup.simg[2]

        #define the directories to bind
        dti_dir = '{}/DTI'.format(kwargs['temp_dir'])
        bind1 = "{}:/INPUTS".format(kwargs['temp_dir'])
        bind2 = "{}:/OUTPUTS".format(dti_dir)

        #first, convert the dwi to tensor
        script.write("echo Making temp directories...\n")
        script.write("mkdir -p {}\n".format(dti_dir))
        script.write("echo Shelling to 1500...\n")
        script.write("time singularity exec -e --contain -B /tmp:/tmp -B {} -B {} {} python3 /CODE/extract_single_shell.py\n".format(bind1, bind2, dwi_simg))
        script.write("echo Done shelling to 1500. Now fitting tensors DTI...\n")

        #1.) extract the b0
        #2.) create the mask
        #3.) fit the tensors
        #4.) calculate the FA, MD, AD, RD

        script.write("echo Extracting b0...\n")
        b0 = "{}/dwmri_b0.nii.gz".format(dti_dir)
        script.write("time singularity exec -e --contain -B /tmp:/tmp -B {}:{} {} bash -c \"dwiextract {}/dwmri.nii.gz -fslgrad {}/dwmri.bvec {}/dwmri.bval - -bzero | mrmath - mean {} -axis 3\"\n".format(kwargs['temp_dir'], kwargs['temp_dir'], dwi_simg, kwargs['temp_dir'], kwargs['temp_dir'], kwargs['temp_dir'], b0))
        script.write("echo Creating mask...\n")
        mask = "{}/dwmri_mask.nii.gz".format(dti_dir)
        script.write("time singularity exec -e --contain -B /tmp:/tmp -B {}:{} {} bet {} {} -f 0.25 -m -n -R\n".format(kwargs['temp_dir'], kwargs['temp_dir'], dwi_simg, b0, mask))
        script.write("echo Fitting tensors...\n")
        shellbvec = "{}/dwmri%firstshell.bvec".format(dti_dir)
        shellbval = "{}/dwmri%firstshell.bval".format(dti_dir)
        shellnii = "{}/dwmri%firstshell.nii.gz".format(dti_dir)
        tensor = "{}/dwmri_tensor.nii.gz".format(dti_dir)
        script.write("time singularity exec -e --contain -B /tmp:/tmp -B {}:{} {} dwi2tensor -fslgrad {} {} {} {}\n".format(kwargs['temp_dir'], kwargs['temp_dir'], dwi_simg, shellbvec, shellbval, shellnii, tensor))
        script.write("echo Calculating FA, MD, AD, RD...\n")
        fa = "{}/dwmri_tensor_fa.nii.gz".format(dti_dir)
        md = "{}/dwmri_tensor_md.nii.gz".format(dti_dir)
        ad = "{}/dwmri_tensor_ad.nii.gz".format(dti_dir)
        rd = "{}/dwmri_tensor_rd.nii.gz".format(dti_dir)
        script.write("time singularity exec -e --contain -B /tmp:/tmp -B {}:{} {} tensor2metric {} -fa {} -mask {}\n".format(kwargs['temp_dir'], kwargs['temp_dir'], dwi_simg, tensor, fa, mask))
        script.write("time singularity exec -e --contain -B /tmp:/tmp -B {}:{} {} tensor2metric {} -adc {} -mask {}\n".format(kwargs['temp_dir'], kwargs['temp_dir'], dwi_simg, tensor, md, mask))
        script.write("time singularity exec -e --contain -B /tmp:/tmp -B {}:{} {} tensor2metric {} -ad {} -mask {}\n".format(kwargs['temp_dir'], kwargs['temp_dir'], dwi_simg, tensor, ad, mask))
        script.write("time singularity exec -e --contain -B /tmp:/tmp -B {}:{} {} tensor2metric {} -rd {} -mask {}\n".format(kwargs['temp_dir'], kwargs['temp_dir'], dwi_simg, tensor, rd, mask))
        
        #### NEED TO REWRITE THE REST BELOW TO USE THE DTI OUTPUTS STRUCTURED AS ABOVE

        script.write("echo Resampling to 1mm iso...\n")
        isodwi = "{}/dwmri_1mm_iso.nii.gz".format(kwargs['temp_dir'])
        script.write("time singularity exec -e --contain -B /tmp:/tmp -B {}:{} {} mrgrid {}/dwmri.nii.gz regrid {} -voxel 1\n".format(kwargs['temp_dir'], kwargs['temp_dir'], dwi_simg, kwargs['temp_dir'], isodwi))
        #same for the fa, md, ad, rd
        faiso = "{}/dwmri_tensor_fa_1mm_iso.nii.gz".format(dti_dir)
        mdiso = "{}/dwmri_tensor_md_1mm_iso.nii.gz".format(dti_dir)
        adiso = "{}/dwmri_tensor_ad_1mm_iso.nii.gz".format(dti_dir)
        rdiso = "{}/dwmri_tensor_rd_1mm_iso.nii.gz".format(dti_dir)
        script.write("time singularity exec -e --contain -B /tmp:/tmp -B {}:{} {} mrgrid {} regrid {} -voxel 1\n".format(dti_dir, dti_dir, dwi_simg, fa, faiso))
        script.write("time singularity exec -e --contain -B /tmp:/tmp -B {}:{} {} mrgrid {} regrid {} -voxel 1\n".format(dti_dir, dti_dir, dwi_simg, md, mdiso))
        script.write("time singularity exec -e --contain -B /tmp:/tmp -B {}:{} {} mrgrid {} regrid {} -voxel 1\n".format(dti_dir, dti_dir, dwi_simg, ad, adiso))
        script.write("time singularity exec -e --contain -B /tmp:/tmp -B {}:{} {} mrgrid {} regrid {} -voxel 1\n".format(dti_dir, dti_dir, dwi_simg, rd, rdiso))

        #now, we need to run bedpostX on the input data
        script.write("echo Running bedpostX...\n")
        
        #first, create a new directory for the bedpostX inputs (linking to the firstshell data)
        bedpostinput = "{}/bedpostXinputs".format(kwargs['temp_dir'])
        script.write("mkdir -p {}\n".format(bedpostinput))
        
        #link the necessary files
        script.write("ln -s {} {}/data.nii.gz\n".format(isodwi, bedpostinput))
        script.write("ln -s {}/dwmri.bvec {}/bvecs\n".format(kwargs['temp_dir'], bedpostinput))
        script.write("ln -s {}/dwmri.bval {}/bvals\n".format(kwargs['temp_dir'], bedpostinput))

        #create the bedpostX mask
        script.write("time singularity exec -e --contain -B /tmp:/tmp -B {indir}:{indir} -B {tempdir}:{tempdir} {mrtrix} bash -c \"dwiextract {indir}/data.nii.gz -fslgrad {indir}/bvecs {indir}/bvals - -bzero | mrmath - mean {indir}/b0.nii.gz -axis 3\"\n".format(indir=bedpostinput, mrtrix=dwi_simg, tempdir=kwargs['temp_dir']))
        script.write("time singularity exec -e --contain -B /tmp:/tmp -B {indir}:{indir} {mrtrix} bet {indir}/b0.nii.gz {indir}/b0_masked -m -R -f .3\n".format(indir=bedpostinput, mrtrix=dwi_simg))
        script.write("rm {indir}/b0_masked.nii.gz\n".format(indir=bedpostinput))
        script.write("mv {indir}/b0_masked_mask.nii.gz {indir}/nodif_brain_mask.nii.gz\n".format(indir=bedpostinput))

        #now run bedpostX (need to bind the parent directory so we can write the bedpost outputs to it)
        if self.setup.args.accre_gpu:
            script.write("time singularity exec --nv -e --contain -B {indir}:{indir} -B {parent}:{parent} {fsl} bedpostx_gpu {indir}\n".format(indir=bedpostinput, fsl=ts_simg, parent=kwargs['temp_dir']))
        else:
            script.write("time singularity exec -e --contain -B /tmp:/tmp -B {indir}:{indir} -B {parent}:{parent} {fsl} bedpostx {indir}\n".format(indir=bedpostinput, fsl=dwi_simg, parent=kwargs['temp_dir']))
        bedpost_ouputs = "{}/bedpostXinputs.bedpostX".format(kwargs['temp_dir'])

        #removing links
        script.write("rm {}/data.nii.gz\n".format(bedpostinput))
        script.write("rm {}/bvecs\n".format(bedpostinput))
        script.write("rm {}/bvals\n".format(bedpostinput))

        #now run tractseg
        script.write("echo Done running bedpostX. Now running TractSeg...\n")
        script.write('echo "..............................................................................."\n')
        if not self.setup.args.no_accre:
            #script.write("echo Loading FSL...\n")

            #script.write("export FSL_DIR=/accre/arch/easybuild/software/MPI/GCC/6.4.0-2.28/OpenMPI/2.1.1/FSL/5.0.10/fsl\n")
            #script.write("source setup_accre_runtime_dir\n")
            pass

        tractsegdir = "{}/tractseg".format(kwargs['temp_dir'])
        script.write("mkdir -p {}\n".format(tractsegdir))
 
        if self.setup.args.accre_gpu:
            script.write("time singularity exec -e --contain --nv -B {bedpost}:{bedpost} -B {outdir}:{outdir} {simg} TractSeg -i {bedpost}/dyads1.nii.gz -o {outdir}\n".format(bedpost=bedpost_ouputs, outdir=tractsegdir, simg=ts_simg, accrehome=kwargs['accre_home']))
            script.write("time singularity exec -e --contain --nv -B {bedpost}:{bedpost} -B {outdir}:{outdir} {simg} TractSeg -i {bedpost}/dyads1.nii.gz -o {outdir} --output_type endings_segmentation\n".format(bedpost=bedpost_ouputs, outdir=tractsegdir, simg=ts_simg, accrehome=kwargs['accre_home']))
            script.write("time singularity exec -e --contain --nv -B {bedpost}:{bedpost} -B {outdir}:{outdir} {simg} TractSeg -i {bedpost}/dyads1.nii.gz -o {outdir} --output_type TOM\n".format(bedpost=bedpost_ouputs, outdir=tractsegdir, simg=ts_simg, accrehome=kwargs['accre_home']))
            script.write("time singularity exec -e --contain --nv -B {bedpost}:{bedpost} -B {outdir}:{outdir} {simg} Tracking -i {bedpost}/dyads1.nii.gz -o {outdir} --tracking_format tck\n".format(bedpost=bedpost_ouputs, outdir=tractsegdir, simg=ts_simg, accrehome=kwargs['accre_home']))

        else:
            script.write("time singularity exec -e --contain -B {bedpost}:{bedpost} -B {outdir}:{outdir} {simg} TractSeg -i {bedpost}/dyads1.nii.gz -o {outdir}\n".format(bedpost=bedpost_ouputs, outdir=tractsegdir, simg=ts_simg, accrehome=kwargs['accre_home']))
            script.write("time singularity exec -e --contain -B {bedpost}:{bedpost} -B {outdir}:{outdir} {simg} TractSeg -i {bedpost}/dyads1.nii.gz -o {outdir} --output_type endings_segmentation\n".format(bedpost=bedpost_ouputs, outdir=tractsegdir, simg=ts_simg, accrehome=kwargs['accre_home']))
            script.write("time singularity exec -e --contain -B {bedpost}:{bedpost} -B {outdir}:{outdir} {simg} TractSeg -i {bedpost}/dyads1.nii.gz -o {outdir} --output_type TOM\n".format(bedpost=bedpost_ouputs, outdir=tractsegdir, simg=ts_simg, accrehome=kwargs['accre_home']))
            script.write("time singularity exec -e --contain -B {bedpost}:{bedpost} -B {outdir}:{outdir} {simg} Tracking -i {bedpost}/dyads1.nii.gz -o {outdir} --tracking_format tck\n".format(bedpost=bedpost_ouputs, outdir=tractsegdir, simg=ts_simg, accrehome=kwargs['accre_home']))

        script.write('echo "..............................................................................."\n')
        script.write("echo Done running TractSeg. Now computing measures per bundle...\n")
        script.write("mkdir {}/measures\n".format(tractsegdir))

        trackingdir = "{}/TOM_trackings".format(tractsegdir)
        measuresdir = "{}/measures".format(tractsegdir)
        script.write("for i in {}/TOM_trackings/*.tck; do\n".format(tractsegdir))
        script.write('    echo "$i"; s=${i##*/}; s=${s%.tck}; echo $s;\n')
        script.write('    time singularity exec -e --contain -B /tmp:/tmp -B {}:{} -e --contain --nv {} scil_evaluate_bundles_individual_measures.py {}/$s.tck {}/$s-SHAPE.json --reference={}\n'.format(kwargs['temp_dir'], kwargs['temp_dir'], scilus_simg, trackingdir, measuresdir, isodwi))
        script.write('    time singularity exec -e --contain -B /tmp:/tmp -B {}:{} -e --contain --nv {} scil_compute_bundle_mean_std.py {}/$s.tck {} {} {} {} --density_weighting --reference={} > {}/$s-DTI.json\n'.format(kwargs['temp_dir'], kwargs['temp_dir'], scilus_simg, trackingdir, faiso, mdiso, adiso, rdiso, isodwi, measuresdir))
        script.write('done\n\n')

        script.write("echo Done computing measures per bundle. Now deleting temp inputs and re-organizing outputs...\n")
        #script.write("rm -r {}/tractseg/TOM\n".format(kwargs['temp_dir']))
        #script.write("rm -r {}/tractseg/peaks.nii.gz\n".format(kwargs['temp_dir']))
        #script.write("rm -r {}/dwmri.nii.gz\n".format(kwargs['temp_dir']))
        #script.write("rm -r {}/dwmri_1mm_iso.nii.gz\n".format(kwargs['temp_dir']))
        #script.write("rm {}/dwmri_tensor_fa.nii.gz\n".format(kwargs['temp_dir']))
        #script.write("rm {}/dwmri_tensor_md.nii.gz\n".format(kwargs['temp_dir']))
        #script.write("rm {}/dwmri_tensor_ad.nii.gz\n".format(kwargs['temp_dir']))
        #script.write("rm {}/dwmri_tensor_rd.nii.gz\n".format(kwargs['temp_dir']))
        script.write("mv {}/tractseg/TOM_trackings {}/bundles\n".format(kwargs['temp_dir'], kwargs['temp_dir']))
        script.write("mv {}/tractseg/measures {}/\n".format(kwargs['temp_dir'], kwargs['temp_dir']))


    def generate_bedpostx_plus_dwi_plus_tractseg_scripts(self):
        """
        Creates scripts for running bedpostx + DTI + Tractseg
        """

        root_temp = Path(self.setup.args.temp_dir)
        assert root_temp.exists() and os.access(root_temp, os.W_OK), "Error: Root temp directory {} does not exist or is not writable".format(root_temp)

        #get the accre home directory / home directory for the tractseg inputs
        if self.setup.args.no_accre:
            if self.setup.args.custom_home != '':
                accre_home_directory = self.setup.args.custom_home
            else:
                accre_home_directory = os.path.expanduser("~")
                user = accre_home_directory.split('/')[-1]
                accre_home_directory = "/home/local/VANDERBILT/{}/".format(user)
        else:
            accre_home_directory = os.path.expanduser("~")   

        #get the raw dwi files
        dwis = self.find_dwis()

        for dwi_p in tqdm(dwis):
            dwi = Path(dwi_p)

            #get the sub, ses, acq, run
            sub, ses, acq, run = self.get_BIDS_fields_dwi(dwi)

            #check to see if the outputs exist already
                #NOTE: this will only check the specified output directory, not the BIDS directory
            tractseg_dir = self.setup.output_dir/(sub)/(ses)/("Bedpost_plus_Tractseg{}{}".format(acq, run))
            if self.has_Tractseg_outputs(tractseg_dir):
                continue

            #get the bval and bvec files for the dwis
            bval = Path(dwi_p.replace('dwi.nii.gz', 'dwi.bval'))
            bvec = Path(dwi_p.replace('dwi.nii.gz', 'dwi.bvec'))

            #make sure that the bval and bvec and nii files exist
            assert dwi.exists() and bval.exists() and bvec.exists(), "Error: dwi, bval, or bvec file does not exist for {}".format(dwi)

            self.count += 1

            #create the temp session directories
            session_temp = root_temp/(sub)/("{}{}{}".format(ses,acq,run))
            (session_input, session_output, session_temp) = self.make_session_dirs(sub, ses, acq, run, tmp_input_dir=self.setup.tmp_input_dir,
                                            tmp_output_dir=self.setup.tmp_output_dir, temp_dir=root_temp, has_temp=True)

            #create the target output directory
            tractsegdwi_target = self.setup.output_dir/(sub)/(ses)/("Bedpost_plus_Tractseg{}{}".format(acq, run))
            if not tractsegdwi_target.exists():
                os.makedirs(tractsegdwi_target)

            #setup the inputs dictionary
                #need dwi, bval, bvec
            self.inputs_dict[self.count] = {
                'dwi': {'src_path': dwi, 'targ_name': 'dwmri.nii.gz', 'separate_input': session_temp},
                'bval': {'src_path': bval, 'targ_name': 'dwmri.bval', 'separate_input': session_temp},
                'bvec': {'src_path': bvec, 'targ_name': 'dwmri.bvec', 'separate_input': session_temp}
            }
            self.outputs[self.count] = []

            #start the script generation
            self.start_script_generation(session_input, session_output, deriv_output_dir=tractsegdwi_target, temp_dir=session_temp,
                                        tractseg_setup=True, accre_home=accre_home_directory, temp_is_output=True)

#for calcualting the average FA, MD, AD, RD values for the whole brain white matter using freesurfer
class FreesurferWhiteMatterMaskGenerator(ScriptGenerator):

    def __init__(self, setup_object):
        """
        Class for taking using WMAtlasEVE3 and freesurfer to calculate the average FA, MD, AD, RD values for the whole brain white matter

        Also creates a freesurfer white matter mask in DWI space
        """
        super().__init__(setup_object=setup_object)
        #self.warnings = {}
        self.outputs = {}
        self.inputs_dict = {}

        self.generate_freesurfer_whitematter_mask_scripts()

    def freesurfer_white_matter_mask_script_generate(self, script, session_input, session_output, **kwargs):
        """
        Creates a single script for running freesurfer white matter mask DTI metrics
        """

        extra_args = "/PYTHON3/get_fs_global_wm_metrics.py /INPUTS wmparc.mgz rawavg.mgz dwmri%ANTS_t1tob0.txt dwmri%fa.nii.gz dwmri%md.nii.gz dwmri%ad.nii.gz dwmri%rd.nii.gz mask.nii.gz aseg.stats /OUTPUTS --freesurfer_license_path /usr/local/freesurfer/.license"

        script.write("echo Running Freesurfer White Matter Mask Metic Calculation...\n")
        script.write("singularity exec -e --contain -B /tmp:/tmp -B {}:/INPUTS -B {}:/OUTPUTS -B {}:/usr/local/freesurfer/.license {} {}\n".format(session_input, session_output, self.setup.freesurfer_license_path, self.setup.simg, extra_args))
        script.write("echo Done running Freesurfer White Matter Mask Metric Calculation. Now deleting inputs and copying back outputs...\n")

    def generate_freesurfer_whitematter_mask_scripts(self):
        """
        Creates scripts for running freesurfer white matter mask DTI metrics
        """

        #gets the wmatlaseve3 directories
        eve3_dirs = self.get_EVE3WMAtlas_dirs()

        for eve3_dir_p in tqdm(eve3_dirs):
            eve3_dir = Path(eve3_dir_p)
            #get the BIDS fields from the EVE3 directory
            sub, ses, acq, run = self.get_BIDS_fields_from_EVE3dir(eve3_dir)
            
            #check to make sure that the outputs for the pipeline dont already exist
            wmmask_outdir = self.setup.output_dir/(sub)/(ses)/("FreesurferWhiteMatterMask{}{}".format(acq, run))
            if self.has_FreesurferWhiteMatterMask_outputs(wmmask_outdir):
                continue
            
            #check to make sure the outputs for PreQual exist
            prequal_dir = self.setup.dataset_derivs/(sub)/(ses)/("PreQual{}{}".format(acq, run))
            #if not self.has_PreQual_outputs(prequal_dir):
            if not all(self.check_PQ_outputs(prequal_dir)):
                self.add_to_missing(sub, ses, acq, run, 'PreQual')
                continue

            #check to make sure that the outputs for EVE3 exist
            if not self.has_EVE3WMAtlas_outputs(eve3_dir):
                self.add_to_missing(sub, ses, acq, run, 'EVE3WMAtlas')
                continue

            #get the T1 that was used for PreQual/EVE3. If we cannot get it, then select it again using the function
            t1 = self.get_prov_t1(prequal_dir)
            if not t1:
                #get the T1 using the function, and print out a provenance warning
                print("Warning: Could not get provenance T1 for {}/{}/{}".format(sub,ses,prequal_dir))
                t1 = self.get_t1(self.setup.root_dataset_path/(sub)/(ses)/("anat"), sub, ses)
                if not t1:
                    self.add_to_missing(sub, ses, acq, run, 'T1')
                    continue
            
            #check to see if the freesurfer outputs exist
            _,_,anat_acq,anat_run = self.get_BIDS_fields_t1(t1)
            fs_dir = self.setup.dataset_derivs/(sub)/(ses)/("freesurfer{}{}".format(anat_acq, anat_run))
            if not check_freesurfer_outputs(fs_dir):
                self.add_to_missing(sub, ses, acq, run, 'freesurfer')
                continue

            #now that we have all the necessary inputs, we can start the script generation TODO

            self.count += 1

            #create the input and output directories
            (session_input, session_output) = self.make_session_dirs(sub, ses, acq, run, tmp_input_dir=self.setup.tmp_input_dir,
                                            tmp_output_dir=self.setup.tmp_output_dir)

            #create the target output directory
            if not wmmask_outdir.exists():
                os.makedirs(wmmask_outdir)

            #setup the inputs dictionary
                #will need to alter if we do not have EVE3 run (i.e. may need to generate on the fly maps/registrations)
            self.inputs_dict[self.count] = {
            #may need to have a flag here for if we are not using the PreQual mask
                #'pq_dwi_dir': {'src_path': prequal_dir/'PREPROCESSED', 'targ_name': '', 'directory': True}, #need PreQual for the mask
                'pq_mask': prequal_dir/'PREPROCESSED'/'mask.nii.gz',
            #here is where we would add the flag for if we are generating our own maps and registrations
                'fa_map': eve3_dir/'dwmri%fa.nii.gz',
                'md_map': eve3_dir/'dwmri%md.nii.gz',
                'ad_map': eve3_dir/'dwmri%ad.nii.gz',
                'rd_map': eve3_dir/'dwmri%rd.nii.gz',
                't1b0_registration': eve3_dir/'dwmri%ANTS_t1tob0.txt',
            #freesurfer outputs (which we NEED to have)
                'wmparc': fs_dir/'freesurfer'/'mri'/'wmparc.mgz',
                'rawavg': fs_dir/'freesurfer'/'mri'/'rawavg.mgz',
                'aseg.stats': fs_dir/'freesurfer'/'stats'/'aseg.stats'
            }
            self.outputs[self.count] = []

            #now we can start the script generation
            self.start_script_generation(session_input, session_output, deriv_output_dir=wmmask_outdir)

#for datasets that do not have PreQual run but are preprocessed: WMAtlas from raw dwi
class RawEVE3WMAtlasGenerator(ScriptGenerator):
    """
    Class for generating EVE3 registration scripts from DWI not in PQ directory
    """

    def __init__(self, setup_object):
        super().__init__(setup_object=setup_object)

        #self.warnings = {}
        self.outputs = {}
        self.inputs_dict = {}

        self.generate_rawEVE3WMAtlas_scripts()

    def rawEVE3Registration_script_generate(self, script, session_input, session_output, **kwargs):
        """
        Writes the command for running EVE3 registration to a script
        """

        script.write("echo Getting mask for DWI...\n")
        script.write("echo Extracting b0...\n")
        b0 = "{}/dwmri_b0.nii.gz".format(session_input)
        script.write("time singularity exec -e --contain -B /tmp:/tmp -B {}:{} {} bash -c \"dwiextract {}/dwmri.nii.gz -fslgrad {}/dwmri.bvec {}/dwmri.bval - -bzero | mrmath - mean {} -axis 3\"\n".format(session_input, session_input, self.setup.simg, session_input, session_input, session_input, b0))
        script.write("echo Creating mask...\n")
        mask = "{}/mask.nii.gz".format(session_input)
        #script.write("cp {}/mask.nii.gz {}\n".format(kwargs['temp_dir'], mask))
        script.write("time singularity exec -e --contain -B /tmp:/tmp -B {}:{} {} bet {} {} -f 0.25 -m -n -R\n".format(session_input, session_input, self.setup.simg, b0, mask))

        script.write("echo Running EVE3 registration...\n")
        script.write("singularity run -e --contain -B /tmp:/tmp -B {}:/INPUTS -B {}:/OUTPUTS {} --EVE3\n".format(session_input, session_output, self.setup.simg))
        script.write("echo Finished running EVE3 registration. Now removing inputs and copying outputs back...\n")

        #always remove firstshell to save space
        script.write("rm {}\n".format(str(session_output/'dwmri%firstshell.nii.gz')))
        #remove the warped files as well (result of using ANTs)
        script.write("rm {}\n".format(str(session_output/'dwmri%InverseWarped.nii.gz')))
        script.write("rm {}\n".format(str(session_output/'dwmri%Warped.nii.gz')))


        #add a check to see if dwmri and dwmri%firstshell are the same. Delete if they are

    def generate_rawEVE3WMAtlas_scripts(self):
        """
        Generates EVE3 scripts
        """

        dwi_niis = self.find_dwis()
        
        for dwi_nii_p in tqdm(dwi_niis):
            dwi_nii = Path(dwi_nii_p)
            #get the BIDS tags
            sub, ses, acq, run = self.get_BIDS_fields_dwi(dwi_nii)

            #make sure the bval and bvec files exist
            bval = Path(dwi_nii_p.replace('dwi.nii.gz', 'dwi.bval'))
            bvec = Path(dwi_nii_p.replace('dwi.nii.gz', 'dwi.bvec'))
            if not bval.exists():
                self.add_to_missing(sub, ses, acq, run, 'bval_missing')
                continue
            if not bvec.exists():
                self.add_to_missing(sub, ses, acq, run, 'bvec_missing')
                continue

            #check to see if the EVE3 outputs already exist
            eve3dir = self.setup.output_dir/(sub)/(ses)/("WMAtlasEVE3{}{}".format(acq, run))
            
            if self.has_EVE3WMAtlas_outputs(eve3dir):
                continue

            #get the T1 that was used for PreQual. If we cannot get it, then select it again using the function
            #t1 = self.get_prov_t1(self.setup.output_dir/(sub)/(ses)/("PreQual{}{}".format(acq, run)))
            #if not t1:
                #get the T1 using the function, and print out a provenance warning
                #print("Warning: Could not get provenance T1 for {}".format(pqdir))
            t1 = self.get_t1(self.setup.root_dataset_path/(sub)/(ses)/("anat"), sub, ses)
            if not t1:
                self.add_to_missing(sub, ses, acq, run, 'T1')
                continue
            
            #based on the T1, get the TICV/UNest segmentation
            ses_deriv = self.setup.dataset_derivs/(sub)/(ses)
            if not self.setup.args.use_unest_seg:
                seg = self.get_TICV_seg_file(t1, ses_deriv)
            else:
                seg = self.get_UNest_seg_file(t1, ses_deriv)
            if not seg or not seg.exists():
                self.add_to_missing(sub, ses, acq, run, 'TICV' if not self.setup.args.use_unest_seg else 'UNest')
                continue

            self.count += 1

            #setup the temp directories
            (session_input, session_output) = self.make_session_dirs(sub, ses, acq, run, tmp_input_dir=self.setup.tmp_input_dir,
                                            tmp_output_dir=self.setup.tmp_output_dir)

            #create the output target directory
            eve3_target = self.setup.output_dir/(sub)/(ses)/("WMAtlasEVE3{}{}".format(acq, run))
            if not eve3_target.exists():
                os.makedirs(eve3_target)

            #setup the inputs dictionary and outputs list            
            self.inputs_dict[self.count] = {'t1': {'src_path': t1, 'targ_name': 't1.nii.gz'},
                                            'seg': {'src_path': seg, 'targ_name': 'T1_seg.nii.gz'},
                                            'dwi': {'src_path': dwi_nii, 'targ_name': 'dwmri.nii.gz'},
                                            'bval': {'src_path': bval, 'targ_name': 'dwmri.bval'},
                                            'bvec': {'src_path': bvec, 'targ_name': 'dwmri.bvec'}
                                        }
            self.outputs[self.count] = []

            #start the script generation
            self.start_script_generation(session_input, session_output, deriv_output_dir=eve3_target)

#infant freesurfer
class InfantFreesurferGenerator(ScriptGenerator):

    def __init__(self, setup_object):
        """
        Class for running infant freesurfer
        """
        super().__init__(setup_object=setup_object)
        #self.warnings = {}
        self.outputs = {}
        self.inputs_dict = {}

        self.generate_infant_freesurfer_scripts()

    def infant_freesurfer_script_generate(self, script, session_input, session_output, **kwargs):
        """
        Writes a single script for running infant freesurfer
        """

        script.write("echo Running Infant Freesurfer...\n")
        script.write("singularity exec -e --contain -B /tmp:/tmp -B {}:/usr/local/freesurfer/.license -B {}:{}/freesurfer -B {}:{} -B {}/mprage.nii.gz:{}/freesurfer/mprage.nii.gz {} bash -c \"export SUBJECTS_DIR={}; infant_recon_all --s freesurfer --outdir {} --age {}\"\n".format(self.setup.freesurfer_license_path, session_input, session_input, session_output, session_output, session_input, session_input, self.setup.simg, session_input, session_output, kwargs['age']))
        
        script.write("Removing the working directory...\n")
        script.write("rm -r {}/'work'\n".format(session_output))
        
        script.write("echo Done running Infant Freesurfer. Now removing inputs and copying outputs back...\n")
        
    
    def generate_infant_freesurfer_scripts(self):
        """
        Generates infant freesurfer scripts for a dataset
        """

        #first, make sure that the freesurfer license is provided
        assert self.setup.freesurfer_license_path.exists(), "Error: Freesurfer license file does not exist at {}".format(self.setup.freesurfer_license_path)

        #check to make sure that the CSV is provided
        demogs_csv = Path(self.setup.args.infant_csv)
        assert demogs_csv.exists(), "Error: Infant CSV file does not exist at {}".format(demogs_csv)

        #iterate through the rows of the CSV
        demog_df = pd.read_csv(demogs_csv)
        for index, row in tqdm(demog_df.iterrows()):
            #get the subject, age, and session (if it exists)
            sub = row['subject']
            age = row['age']
            ses = row['session'] if 'session' in row else None

            #if age or subject is missing, skip and add to the missing list
            if not sub or not age:
                self.add_to_missing(sub, ses, '', '', 'missing_subject_or_age')
                continue
                #check to see if the age is greater than 2
            age = age * 12 #should be age in months
            #round the age to the nearest month
            age = round(age)
            if age > 24: #if the scan is for a child over 2 years old, skip
                self.add_to_missing(sub, ses, '', '', 'age_over_2')
                print("ERROR: Age over 2 for {}/{}".format(sub, ses))
                continue

            #get the list of T1s for the subject/session
            dataset_root = self.setup.root_dataset_path
            t1s = self.get_sub_ses_t1s(sub, ses)

            for t1 in t1s:
                #get the BIDS tags for the T1
                subx, sesx, acq, run = self.get_BIDS_fields_t1(t1)
                assert sub == subx and ses == sesx, "Error: Subject/Ses mismatch between CSV and T1s"
                
                #check to see if the outputs already exist
                infant_fs_dir = self.setup.dataset_derivs/(sub)/(ses)/("infantFS{}{}".format(acq, run))
                if self.has_infantFS_outputs(infant_fs_dir):
                    continue

                #now we can create the script
                self.count += 1

                #create the input and output directories
                (session_input, session_output) = self.make_session_dirs(sub, ses, acq, run, tmp_input_dir=self.setup.tmp_input_dir,
                                                tmp_output_dir=self.setup.tmp_output_dir)
                
                #create the target output directory
                infant_fs_target = self.setup.output_dir/(sub)/(ses)/("infantFS{}{}".format(acq, run))
                if not infant_fs_target.exists():
                    os.makedirs(infant_fs_target)
                
                #setup the inputs dictionary and outputs list
                self.inputs_dict[self.count] = {'t1': {'src_path': t1, 'targ_name': 'mprage.nii.gz'}}
                self.outputs[self.count] = []

                #start the script generation
                self.start_script_generation(session_input, session_output, deriv_output_dir=infant_fs_target, age=age)


#for Kurt: run scilpy scipts on tractseg dirs
class Scilpy_on_TractsegGenerator(ScriptGenerator):

    def __init__(self, setup_object):
        """
        Class for running DTI metrics on tractseg outputs
        """
        super().__init__(setup_object=setup_object)
        #self.warnings = {}
        self.outputs = {}
        self.inputs_dict = {}

    def scilpy_on_tractseg_script_generate(self, script, session_input, session_output, **kwargs):
        """
        Writes a single script for running scilpy DTI metrics on tractseg outputs
        """

        #just need the scilpy singularity and WMAtlas
        dwi_simg = self.setup.simg[0]
        scilus_simg = self.setup.simg[1]

        #define the directories to bind
        dti_dir = '{}/DTI'.format(kwargs['temp_dir'])
        bind1 = "{}:/INPUTS".format(kwargs['temp_dir'])
        bind2 = "{}:/OUTPUTS".format(dti_dir)

        #first, convert the dwi to tensor
        script.write("echo Making temp directories...\n")
        script.write("mkdir -p {}\n".format(dti_dir))
        script.write("echo Shelling to 1500...\n")
        script.write("time singularity exec -B {} -B {} {} python3 /CODE/extract_single_shell.py\n".format(bind1, bind2, dwi_simg))
        script.write("echo Done shelling to 1500. Now fitting tensors DTI...\n")

        mask = "{}/mask.nii.gz".format(kwargs['temp_dir'])
        shellbvec = "{}/dwmri%firstshell.bvec".format(dti_dir)
        shellbval = "{}/dwmri%firstshell.bval".format(dti_dir)
        shellnii = "{}/dwmri%firstshell.nii.gz".format(dti_dir)
        tensor = "{}/dwmri_tensor.nii.gz".format(dti_dir)
        script.write("time singularity exec -B {}:{} {} dwi2tensor -fslgrad {} {} {} {}\n".format(kwargs['temp_dir'], kwargs['temp_dir'], dwi_simg, shellbvec, shellbval, shellnii, tensor))
        script.write("echo Calculating FA, MD, AD, RD...\n")
        fa = "{}/dwmri_tensor_fa.nii.gz".format(dti_dir)
        md = "{}/dwmri_tensor_md.nii.gz".format(dti_dir)
        ad = "{}/dwmri_tensor_ad.nii.gz".format(dti_dir)
        rd = "{}/dwmri_tensor_rd.nii.gz".format(dti_dir)
        script.write("time singularity exec -B {}:{} {} tensor2metric {} -fa {} -mask {}\n".format(kwargs['temp_dir'], kwargs['temp_dir'], dwi_simg, tensor, fa, mask))
        script.write("time singularity exec -B {}:{} {} tensor2metric {} -adc {} -mask {}\n".format(kwargs['temp_dir'], kwargs['temp_dir'], dwi_simg, tensor, md, mask))
        script.write("time singularity exec -B {}:{} {} tensor2metric {} -ad {} -mask {}\n".format(kwargs['temp_dir'], kwargs['temp_dir'], dwi_simg, tensor, ad, mask))
        script.write("time singularity exec -B {}:{} {} tensor2metric {} -rd {} -mask {}\n".format(kwargs['temp_dir'], kwargs['temp_dir'], dwi_simg, tensor, rd, mask))
        script.write("echo Done fitting tensors. Now computing measures per bundle...\n")

        bundlesdir = "{}/bundles".format(kwargs['temp_dir'])
        measuresdir = "{}/measures".format(kwargs['temp_dir'])
        script.write("for i in {}/*.tck; do\n".format(bundlesdir))
        script.write('    echo "$i"; s=${i##*/}; s=${s%.tck}; echo $s;\n')
        #script.write('    time singularity exec -B {}:{} --nv {} scil_evaluate_bundles_individual_measures.py {}/$s.tck {}/$s-SHAPE.json --reference={}\n'.format(kwargs['temp_dir'], kwargs['temp_dir'], scilus_simg, trackingdir, measuresdir, isodwi))
        script.write('    time singularity exec -B {}:{} --nv {} scil_compute_bundle_mean_std.py {}/$s.tck {} {} {} {} --density_weighting --reference={} > {}/$s-DTI.json\n'.format(kwargs['temp_dir'], kwargs['temp_dir'], scilus_simg, trackingdir, faiso, mdiso, adiso, rdiso, isodwi, measuresdir))
        script.write('done\n\n')

    def generate_scilpy_ts_metrics_list(self):
        """
        For the runs that do not have metrics computed, redo the calculations
        """

        root_temp = Path(self.setup.args.temp_dir)
        assert root_temp.exists() and os.access(root_temp, os.W_OK), "Error: Root temp directory {} does not exist or is not writable".format(root_temp)

        #get the accre home directory / home directory for the tractseg inputs
        if self.setup.args.no_accre:
            if self.setup.args.custom_home != '':
                accre_home_directory = self.setup.args.custom_home
            else:
                accre_home_directory = os.path.expanduser("~")
                user = accre_home_directory.split('/')[-1]
                accre_home_directory = "/home/local/VANDERBILT/{}/".format(user)
        else:
            accre_home_directory = os.path.expanduser("~")     
        
        prequal_dirs = self.get_PreQual_dirs()

        for pqdir_p in tqdm(prequal_dirs):
            pqdir = Path(pqdir_p)
            #get the BIDS tags
            sub, ses, acq, run = self.get_BIDS_fields_from_PQdir(pqdir) 

            #check to see if the Tractseg dir exists
            tractsegdwi_target = self.setup.dataset_derivs/(sub)/(ses)/("Tractseg{}{}".format(acq, run))
            if not tractsegdwi_target.exists():
                continue

            #check to see if the Tractseg outputs already exist
            tractseg_dir = self.setup.dataset_derivs/(sub)/(ses)/("Tractseg{}{}".format(acq, run))
            if not self.is_missing_DTI_tractseg(tractseg_dir):
                continue

            self.count += 1

            #create the temp session directories
            session_temp = root_temp/(sub)/("{}{}{}".format(ses,acq,run))
            (session_input, session_output, session_temp) = self.make_session_dirs(sub, ses, acq, run, tmp_input_dir=self.setup.tmp_input_dir,
                                            tmp_output_dir=self.setup.tmp_output_dir, temp_dir=root_temp, has_temp=True)
            
            #create the inputs dictionary
            self.inputs_dict[self.count] = {
                    'pq_dwi_dir': {'src_path': pqdir/'PREPROCESSED', 'targ_name': '', 'directory': True, 'separate_input': session_temp}
            }
            self.outputs[self.count] = []

            self.start_script_generation(session_input, session_output, deriv_output_dir=tractsegdwi_target, temp_dir=session_temp,
                                        tractseg_setup=True, accre_home=accre_home_directory, temp_is_output=True)

def get_shells_and_dirs(PQdir, bval_files, bvec_files):
    """
    Given bval and bvec files, get the number of shells and number of directions across them all

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
        
    def load_bvals_bvecs(bval_files, bvec_files):
        bvals = []
        bvecs = []
        for bval_f, bvec_f in zip(bval_files, bvec_files):
            bval = np.loadtxt(bval_f)
            bvec = np.loadtxt(bvec_f)
            bvals.append(bval)
            bvecs.append(bvec)
        bvals = np.concatenate(bvals)
        bvecs = np.concatenate(bvecs, axis=1)
        return bvals, bvecs

    # Initialize an empty set to store unique (x, y, z) tuples
    unique_pairs = set()
    shells_dic = {}
    bval_map = {}

    # Read data from the bval files
    bvals, bvecs = load_bvals_bvecs(bval_files, bvec_files)
    rounded_bvals = np.array([round(b, 100) for b in bvals])
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
        return "No b0","No b0", "No b0", "No b0"
    
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
        return None, None, None, None
    elif len(dti_set) == 0 or len(dti_dirs) < 6:
        #print(shells_dic)
        return None, None, fod_set, None
    elif len(fod_set) == 0:
        return None, dti_set, None, None
    return rounded_bvals, '"' + " ".join(['0']+[str(int(x)) for x in sorted(dti_set)]) + '"', '"' + " ".join(['0']+[str(int(x)) for x in sorted(fod_set)]) + '"', '"' + " ".join(['0']+[str(int(x)) for x in sorted(sh8_set)]) + '"'
        

def get_num_shells(bval):
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

def get_TICV_seg(TICV_dir, sub, ses, acq, run):
    """
    Given the TICV dir and the BIDS fields, return the TICV seg file if it exits. Otherwise, return None
    """
    TICV_dir = TICV_dir/("post")/("FinalResult")
    if not acq == '':
        acq = '_'+acq
    if not run == '':
        run = '_'+run
    if not ses == '':
        ses = '_'+ses
    TICV_seg = TICV_dir/("{}{}{}{}_T1w_seg.nii.gz".format(sub,ses,acq,run))
    if TICV_seg.exists():
        return TICV_seg
    return None


def check_freesurfer_outputs(fs_dir):
    """
    Given a freesurfer directory, returns True if the outputs are valid and exist, False otherwise.
    """

    recon_log = fs_dir/("freesurfer")/("scripts")/("recon-all.log")
    #first, check if directory and file exist

    if fs_dir.exists() and recon_log.exists():
        #now check the contents of the reconlog
        isDone = True
        with open(recon_log, 'r') as log:
            try:
                content = log.read()
                if 'finished without error' in content:
                    return True
                    #isDone = True
            #if isDone:
            #    continue
            except:
                print("Error: Could not read contents of recon-all.log for {}".format(fs_dir))
                return False
    print("Error: Freesurfer outputs are not valid for {}".format(fs_dir))
    return False

def hash_file(filepath):
    """
    Calculate the hash of a file using md5sum
    """
    print("Hashing: ", filepath)
    output = subprocess.run("md5sum {}".format(filepath), shell=True, capture_output=True, text=True).stdout
    print(output)
    return output.strip().split()[0]

def get_readout_times(json_dicts):
    """
    Given json dictionaries, get the readout times
    """
    readouts = []
    #print("getting readouts........")
    #print(json_dicts)
    for json_data in json_dicts:
        try:
            readout = json_data['TotalReadoutTime']
        except:
            try:
                readout = json_data['EstimatedTotalReadoutTime']
            except:
                readout = None #default to 0.05 if not found
        readouts.append(readout)
        #print(readout)
    #print(readouts)
    return readouts

def check_needs_synb0(PEaxes, PEsigns, PEunknows):
    """
    Check to make sure that the PE axes are the same for all the scans and if we need synb0 or not

    Return True if we need synb0, False if not

    Returns None if there are two different supplied PE axes
    """
    #make sure that all the PEaxes are the same
    tmp_PEaxes = [x for x in PEaxes if x is not None]
    tmp_signs = [x for x in PEsigns if x is not None]
    if len(set(tmp_PEaxes)) != 1:
        print("Error: PE axes are not the same for all scans: {}".format(PEaxes))
        return None
    
    #see if there are any unknowns. If so, then return synb0
    if any(PEunknows):
        return True
    
    #check to see if there are any opposite directions
    if '-' in tmp_signs and '+' in tmp_signs:
        return False
    
    return True
    

def get_sub_ses_dwidir(dwi_dir):
    """
    Given the path to a dwi dir, return the sub and ses
    """            
    up_one = dwi_dir.parent
    if 'ses' in up_one.name:
        ses = up_one.name
        sub = up_one.parent.name
    else:
        ses = ''
        sub = up_one.name
    return sub, ses
            
def get_PE_dirs(json_dicts, jsons, single=False):
    """
    Given a list of json dicts, return the PE directions and axes
    """
    PEaxes = []
    PEdirs = []
    PEunknowns = []
    for json_data,json_file in zip(json_dicts, jsons):     
        try: #try phase encoding direction    
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
        ##TODO: Incorporate the new VMAP acquisitions to determine the sign (as it is not in the JSONS for some reason)
        if 'ReversePE' in json_file.name:
            PEsign = '-'
        else:
            if len(PEdir) == 2:
                if PEdir[0] == '-' or PEdir[1] == '-':
                    if PEdir[0] == '-':
                        PEaxis = PEdir[1]
                        PEsign = '-'
                    else:
                        PEaxis = PEdir[0]
                        PEsign = '-'
                elif PEdir[0] == '+' or PEdir[1] == '+':
                    if PEdir[0] == '+':
                        PEaxis = PEdir[1]
                        PEsign = '+'
                    else:
                        PEaxis = PEdir[0]
                        PEsign = '+'
                else: #if there is no direction, it is assumed to be positive
                    PEaxis = PEdir[0]
                    PEsign = '+'
            else: #if there is no direction, it is assumed to be positive
                PEaxis = PEdir[0]
                PEsign = '+'
        if not single:
            PEaxes.append(PEaxis)
            PEdirs.append(PEsign)
            PEunknowns.append(PEunknown)
        else:
            return (PEaxis, PEsign, PEunknown)
    return (PEaxes, PEdirs, PEunknowns)

def get_json_dict(json_file):
    """
    Return the dictionary from the json file
    """
    with open(json_file, 'r') as f:
        contents = json.load(f)
        #print(contents)
        return contents

def check_json_params_similarity(dicts, args):
    """
    Returns True if all the json files have the same acquisition parameters, False otherwise
    """
    if len(dicts) == 1:
        return False #do not create blanket "PreQual". Instead, run the pipeline on the individual files (of which we have only 1)

    #compare the dicts
    similar = compare_json_dicts(dicts, args)

    return similar

def compare_json_dicts(dicts, args):
    """
    Given a list of json dictionaries for Niftis, compare them to see if they are similar or not
    """

    params = ['EchoTime', 'RepetitionTime', 'MagneticFieldStrength']

    try:
        #TE: make sure TE is within 0.01 of each other
        tes = np.array([x['EchoTime'] for x in dicts])
        #TR: make sure TR is within 0.01 of each other
        trs = np.array([x['RepetitionTime'] for x in dicts])
        #Magnetic Field Strength: make sure they are the same
        mfs = np.array([x['MagneticFieldStrength'] for x in dicts])

        return check_within_tolerance(tes, tolerance=args.TE_tolerance_PQ) and check_within_tolerance(trs, tolerance=args.TR_tolerance_PQ) and check_identical(mfs)

    except:
        return False


def check_within_tolerance(arr, tolerance=0.01):
    """
    Check 1D numpy array to see if all values are within a certain tolerance of each other
    """
    return np.all(np.abs(arr - arr[0]) < tolerance)

def check_identical(arr):
    """
    Check 1D numpy array to see if all values are the same
    """
    return np.all(arr == arr[0])

def main():
    """
    Main function for the script generation.
    """

    print('ASSUMES THAT YOU HAVE SET UP AN SSH-KEY TO SCP TO landman01/landman01 WITHOUT A PASSWORD. IF YOU DO NOT HAVE THIS, PLEASE SET IT UP AND TEST IT OUT BEFORE SUBMITTING JOBS TO ACCRE.')

    args = pa()
    sg = ScriptGeneratorSetup(args)

    #first, generate the scripts
    sg.start_script_generation()

    #now also output the slurm script / parallelization script
    if not sg.args.no_accre:
        sg.write_slurm_script()
    else:
        print("Not writing slurm script because --no-accre flag was set.")
        print("Instead writing a python parallelization file")
        sg.write_paralellization_script()



if __name__ == "__main__":
    main()

