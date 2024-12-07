root=/fs5/p_masi/kimm58/InterRaterVariability
raters=("rater_1" "rater_2" "rater_3" "rater_4")
for x in "${raters[@]}"; do mkdir -p $root/$x; done

###SLANT-TICV
for x in "${raters[@]}"; do mkdir -p $root/$x/SLANT-TICVv1.2; done

#get the no's first (should be 30)
cat /nfs2/harmonization/ADSP_QA/ADNI_DTI/SLANT-TICVv1.2/QA.csv | grep no | grep -v unknown | while IFS= read -r line
do 
	while IFS=',' read -r sub ses acq run _; do 
		pngname=${sub}_${ses}_SLANT-TICVv1.2${acq}${run}.png
		png=/nfs2/harmonization/ADSP_QA/ADNI_DTI/SLANT-TICVv1.2/${pngname}
		#echo $png
		#link the png
		for x in "${raters[@]}"; do ln -s $png $root/$x/SLANT-TICVv1.2/$pngname; done
	done
done

#get the maybes (should be 5)
cat /nfs2/harmonization/ADSP_QA/ADNI_DTI/SLANT-TICVv1.2/QA.csv | grep maybe | grep -v orientation | grep -v rotated | while IFS= read -r line
do
        while IFS=',' read -r sub ses acq run _; do
                pngname=${sub}_${ses}_SLANT-TICVv1.2${acq}${run}.png
                png=/nfs2/harmonization/ADSP_QA/ADNI_DTI/SLANT-TICVv1.2/${pngname}
                #echo $png
                #link the png
                for x in "${raters[@]}"; do ln -s $png $root/$x/SLANT-TICVv1.2/$pngname; done
        done
done

#get maybe 965 yes ones
	#can only run this one time
currcount=$(ls /nfs2/harmonization/ADSP_QA/ADNI_DTI/SLANT-TICVv1.2 | wc -l)
if [[ "$currcount" -lt 900 ]]; then

	cat /nfs2/harmonization/ADSP_QA/ADNI_DTI/SLANT-TICVv1.2/QA.csv | grep yes | shuf -n 965 | while IFS= read -r line
do
        while IFS=',' read -r sub ses acq run _; do
                pngname=${sub}_${ses}_SLANT-TICVv1.2${acq}${run}.png
                png=/nfs2/harmonization/ADSP_QA/ADNI_DTI/SLANT-TICVv1.2/${pngname}
                #echo $png
                #link the png
                for x in "${raters[@]}"; do ln -s $png $root/$x/SLANT-TICVv1.2/$pngname; done
        done
done
fi

###PreQual
for x in "${raters[@]}"; do mkdir -p $root/$x/PreQual; done

#just do all at once
( grep no /nfs2/harmonization/ADSP_QA/HABSHD/PreQual/QA.csv | grep -v maybe;  grep maybe /nfs2/harmonization/ADSP_QA/HABSHD/PreQual/QA.csv; grep yes /nfs2/harmonization/ADSP_QA/HABSHD/PreQual/QA.csv | grep -v maybe | grep -v noisy | shuf -n 280 ) | while IFS= read -r line; do
        while IFS=',' read -r sub ses acq run _; do
                pngname=${sub}_${ses}_PreQual${acq}${run}.png
                png=/nfs2/harmonization/ADSP_QA/ADNI_DTI/PreQual/${pngname}
                echo $png
		#echo $line
                #link the png
                for x in "${raters[@]}"; do ln -s $png $root/$x/PreQual/$pngname; done
        done

done

###Tractseg
### TODO






