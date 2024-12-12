
root="/fs5/p_masi/kimm58/MRIQC_experiments/ADNI_anat"

#get the no's first (should be 30)
(cat /nfs2/harmonization/ADSP_QA/ADNI_DTI/SLANT-TICVv1.2/QA.csv | grep maybe | grep -v orientation | grep -v rotated; cat /nfs2/harmonization/ADSP_QA/ADNI_DTI/SLANT-TICVv1.2/QA.csv | grep no | grep -v unknown | grep -v maybe; cat /nfs2/harmonization/ADSP_QA/ADNI_DTI/SLANT-TICVv1.2/QA.csv | grep -v no | grep -v severe | grep -v document | grep -v FOV | shuf -n 1166) | while IFS=',' read -r sub ses acq run _; do
#cat /nfs2/harmonization/ADSP_QA/ADNI_DTI/SLANT-TICVv1.2/QA.csv | grep no | grep -v unknown | grep -v maybe | while IFS=',' read -r sub ses acq run _; do  #while IFS= read -r line
#do 
	#while IFS=',' read -r sub ses acq run _; do 
    #if [[ $acq == "" ]]; then acqx=""; else acqx="_${acq}"; fi
    #if [[ $run == "" ]]; then runx=""; else runx="_${run}"; fi
    acqx=${acq:+_$acq}
    runx=${run:+_$run}
    file_bids="/nfs2/harmonization/BIDS/ADNI_DTI/${sub}/${ses}/anat/${sub}_${ses}${acqx}${runx}_T1w.nii.gz"
    #check to make sure the file exists
    if [[ -e $file_bids ]]; then
        file=$(readlink -f $file_bids)
        scandir=${root}/${sub}/${ses}/anat
        if [[ ! -d $scandir ]]; then echo mkdir -p $scandir; fi
        echo ln -s $file $scandir/${sub}_${ses}${acqx}${runx}_T1w.nii.gz
    fi
done