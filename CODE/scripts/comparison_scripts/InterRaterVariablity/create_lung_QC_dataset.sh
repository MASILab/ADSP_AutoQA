#/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/qa_hallucinations_withheld_data/BONEtoB30f
#/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/qa_hallucinations_withheld_data/CtoB30f

#run below from here: /fs5/p_masi/kimm58/InterRaterVariability
for d in rater*; do mkdir $d/LungHallucination; done

#make the links
for d in rater*; do
    d1=/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/qa_hallucinations_withheld_data/BONEtoB30f
    d2=/nfs/masi/krishar1/KernelConversionUnpaired/SPIE_journal_extension/starganL2weightsched_resnetbackbone/qa_hallucinations_withheld_data/CtoB30f
    for x in $d1/* $d2/*; do
        fname=$(echo $x | awk -F '/' '{print $(NF)}')
        ln -s $x $d/LungHallucination/$fname
    done
done
    
