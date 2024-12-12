
#find /fs5/p_masi/kimm58/MRIQC_experiments/ADNI_anat -mindepth 1 -maxdepth 1 -type d -name "sub-*"

{ time singularity run -e -B /tmp:/tmp -B /fs5/p_masi/kimm58/MRIQC_experiments/ADNI_anat:/data -B /fs5/p_masi/kimm58/MRIQC_experiments/ADNI_anat/derivatives:/out \
-B /nfs2/harmonization/BIDS/ADNI_DTI:/nfs2/harmonization/BIDS/ADNI_DTI \
-B /nfs2/harmonization/raw/ADNI_DTI/:/nfs2/harmonization/raw/ADNI_DTI/ \
--home /home-local/kimm58/working \
mriqc_latest.sif /data /out participant --n_proc 20 --no-sub; } 2> adni_anat_mriqc_time.txt