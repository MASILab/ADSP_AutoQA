#singularity run -e -B /tmp:/tmp -B /fs5/p_masi/kimm58/MRIQC_experiments/WRAP_anat:/data -B /fs5/p_masi/kimm58/MRIQC_experiments/WRAP_anat/derivatives:/out \
#-B /nfs2/harmonization/BIDS/WRAP:/nfs2/harmonization/BIDS/WRAP \
#-B /nfs2/harmonization/raw/WRAP_Imaging/WRAP_Imaging/:/nfs2/harmonization/raw/WRAP_Imaging/WRAP_Imaging/ \
#mriqc_latest.sif /data /out participant --n_proc 20 --no-sub

{ time singularity run -e -B /tmp:/tmp -B /fs5/p_masi/kimm58/MRIQC_experiments/WRAP_anat:/data -B /fs5/p_masi/kimm58/MRIQC_experiments/WRAP_anat/derivatives:/out \
-B /nfs2/harmonization/BIDS/WRAP:/nfs2/harmonization/BIDS/WRAP \
-B /nfs2/harmonization/raw/WRAP_Imaging/WRAP_Imaging/:/nfs2/harmonization/raw/WRAP_Imaging/WRAP_Imaging/ \
--home /mnt/sdg1/kimm58/working_home \
mriqc_latest.sif /data /out participant --n_proc 20 --no-sub; } 2> wrap_anat_mriqc_time.txt
