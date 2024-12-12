root=/fs5/p_masi/kimm58/InterRaterVariability
raters=("rater_1" "rater_2" "rater_3" "rater_4")
for x in "${raters[@]}"; do mkdir -p $root/$x; done

###SLANT-TICV
echo "Making SLANT-TICV"
for x in "${raters[@]}"; do mkdir -p $root/$x/SLANT-TICVv1.2; done

#get the no's first (should be 30)
cat /nfs2/harmonization/ADSP_QA/ADNI_DTI/SLANT-TICVv1.2/QA.csv | grep no | grep -v unknown | grep -v maybe | while IFS=',' read -r sub ses acq run _; do  #while IFS= read -r line
#do 
	#while IFS=',' read -r sub ses acq run _; do 
		pngname=${sub}_${ses}_SLANT-TICVv1.2${acq}${run}.png
		png=/nfs2/harmonization/ADSP_QA/ADNI_DTI/SLANT-TICVv1.2/${pngname}
		#echo $png
		#link the png
		for x in "${raters[@]}"; do ln -s $png $root/$x/SLANT-TICVv1.2/$pngname; done
	#done
done

#get the maybes (should be 5)
cat /nfs2/harmonization/ADSP_QA/ADNI_DTI/SLANT-TICVv1.2/QA.csv | grep maybe | grep -v orientation | grep -v rotated | while IFS=',' read -r sub ses acq run _; do  #while IFS= read -r line
#do
        #while IFS=',' read -r sub ses acq run _; do
                pngname=${sub}_${ses}_SLANT-TICVv1.2${acq}${run}.png
                png=/nfs2/harmonization/ADSP_QA/ADNI_DTI/SLANT-TICVv1.2/${pngname}
                #echo $png
                #link the png
                for x in "${raters[@]}"; do ln -s $png $root/$x/SLANT-TICVv1.2/$pngname; done
        #done
done

#get maybe 965 yes ones
	#can only run this one time
#currcount=$(ls /nfs2/harmonization/ADSP_QA/ADNI_DTI/SLANT-TICVv1.2 | wc -l)
#echo $currcount
#if [[ "$currcount" -lt 900 ]]; then

cat /nfs2/harmonization/ADSP_QA/ADNI_DTI/SLANT-TICVv1.2/QA.csv | grep yes | shuf -n 966 | while IFS=',' read -r sub ses acq run _; do  #while IFS= read -r line
#do
        #while IFS=',' read -r sub ses acq run _; do
                pngname=${sub}_${ses}_SLANT-TICVv1.2${acq}${run}.png
                png=/nfs2/harmonization/ADSP_QA/ADNI_DTI/SLANT-TICVv1.2/${pngname}
                #echo $png
                #link the png
                for x in "${raters[@]}"; do ln -s $png $root/$x/SLANT-TICVv1.2/$pngname; done
        #done
done
#fi

###PreQual
echo "Making PreQual"
for x in "${raters[@]}"; do mkdir -p $root/$x/PreQual; done

#just do all at once
( grep no /nfs2/harmonization/ADSP_QA/HABSHD/PreQual/QA.csv | grep -v maybe;  grep maybe /nfs2/harmonization/ADSP_QA/HABSHD/PreQual/QA.csv; grep yes /nfs2/harmonization/ADSP_QA/HABSHD/PreQual/QA.csv | grep -v maybe | grep -v noisy | shuf -n 280 ) | while IFS=',' read -r sub ses acq run _; do  #while IFS= read -r line; do
        #while IFS=',' read -r sub ses acq run _; do
                pngname=${sub}_${ses}_PreQual${acq}${run}.png
                png=/nfs2/harmonization/ADSP_QA/HABSHD/PreQual/${pngname}
                #echo $png
		#echo $line
                #link the png
                for x in "${raters[@]}"; do ln -s $png $root/$x/PreQual/$pngname; done
        #done

done

#exit

###Tractseg

#AF right
echo "Making Tractseg"
for x in "${raters[@]}"; do mkdir -p $root/$x/TractsegAFright; done
( grep no /nfs2/harmonization/ADSP_QA/HABSHD/TractsegAFright/QA.csv | grep -v maybe | grep -v "sub-8218,ses-baseline"; grep maybe /nfs2/harmonization/ADSP_QA/HABSHD/TractsegAFright/QA.csv; grep yes /nfs2/harmonization/ADSP_QA/HABSHD/TractsegAFright/QA.csv | shuf -n 936 ) | while IFS=',' read -r sub ses acq run _; do #while IFS= read -r line; do
        #while IFS=',' read -r sub ses acq run _; do
                pngname=${sub}_${ses}_TractsegAFright${acq}${run}.png
                png=/nfs2/harmonization/ADSP_QA/HABSHD/TractsegAFright/${pngname}
                #echo $png
                for x in "${raters[@]}"; do ln -s $png $root/$x/TractsegAFright/$pngname; done
        #done
done

#CST left
for x in "${raters[@]}"; do mkdir -p $root/$x/TractsegCSTleft; done
( grep no /nfs2/harmonization/ADSP_QA/HABSHD/TractsegCSTleft/QA.csv; grep maybe /nfs2/harmonization/ADSP_QA/HABSHD/TractsegCSTleft/QA.csv; grep yes /nfs2/harmonization/ADSP_QA/HABSHD/TractsegCSTleft/QA.csv | shuf -n 866) | while IFS=',' read -r sub ses acq run _; do  #while IFS= read -r line; do
        #while IFS=',' read -r sub ses acq run _; do
                pngname=${sub}_${ses}_TractsegCSTleft${acq}${run}.png
                png=/nfs2/harmonization/ADSP_QA/HABSHD/TractsegCSTleft/${pngname}
                #echo $png
                for x in "${raters[@]}"; do ln -s $png $root/$x/TractsegCSTleft/$pngname; done
        #done
done

#TPOSTC right
for x in "${raters[@]}"; do mkdir -p $root/$x/TractsegTPOSTCright; done
( grep maybe /nfs2/harmonization/ADSP_QA/HABSHD/TractsegTPOSTCright/QA.csv; grep no /nfs2/harmonization/ADSP_QA/HABSHD/TractsegTPOSTCright/QA.csv; grep yes /nfs2/harmonization/ADSP_QA/HABSHD/TractsegTPOSTCright/QA.csv | shuf -n 398)| while IFS=',' read -r sub ses acq run _; do  #while IFS= read -r line; do
        #while IFS=',' read -r sub ses acq run _; do
                pngname=${sub}_${ses}_TractsegTPOSTCright${acq}${run}.png
                png=/nfs2/harmonization/ADSP_QA/HABSHD/TractsegTPOSTCright/${pngname}
                #echo $png
                for x in "${raters[@]}"; do ln -s $png $root/$x/TractsegTPOSTCright/$pngname; done
        #done
done

##CC4
for x in "${raters[@]}"; do mkdir -p $root/$x/TractsegCC4; done
( grep no /nfs2/harmonization/ADSP_QA/HABSHD/TractsegCC4/QA.csv; grep maybe /nfs2/harmonization/ADSP_QA/HABSHD/TractsegCC4/QA.csv; grep yes /nfs2/harmonization/ADSP_QA/HABSHD/TractsegCC4/QA.csv | shuf -n 795) | while IFS=',' read -r sub ses acq run _; do  #while IFS= read -r line; do
        #while IFS=',' read -r sub ses acq run _; do
                pngname=${sub}_${ses}_TractsegCC4${acq}${run}.png
                png=/nfs2/harmonization/ADSP_QA/HABSHD/TractsegCC4/${pngname}
                #echo $png
                for x in "${raters[@]}"; do ln -s $png $root/$x/TractsegCC4/$pngname; done
        #done
done

## SLF-I left - no maybes
for x in "${raters[@]}"; do mkdir -p $root/$x/TractsegSLFIleft; done
( grep no /nfs2/harmonization/ADSP_QA/HABSHD/TractsegSLFIleft/QA.csv; grep yes /nfs2/harmonization/ADSP_QA/HABSHD/TractsegSLFIleft/QA.csv | shuf -n 981) | while IFS=',' read -r sub ses acq run _; do  #while IFS= read -r line; do
        #while IFS=',' read -r sub ses acq run _; do
                pngname=${sub}_${ses}_TractsegSLFIleft${acq}${run}.png
                png=/nfs2/harmonization/ADSP_QA/HABSHD/TractsegSLFIleft/${pngname}
                #echo $png
                for x in "${raters[@]}"; do ln -s $png $root/$x/TractsegSLFIleft/$pngname; done
        #done
done
