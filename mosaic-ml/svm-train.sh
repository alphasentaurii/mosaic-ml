#!/bin/bash -xu
export SVM_QUALITY_TESTING=on

SRCPATH=${1:-"data/singlevisits"} # data/singlevisits/results_2021-07-28
OUT=${2:-"data/synthetic"}
DATASETS=${3:-""} # ('icon04')
REG=${reg:-"1"}
EXP=${expo:-""}
MODE=${mode:-""}
SYN=${syn:-""}
FILTERS=${filters:-""}

if [[ -z ${DATASETS} ]]; then
    DATASETS=`ls "${SRCPATH}"`
fi

IMG_DET=${OUT}/img/total
NEG=${IMG_DET}/0
POS=${IMG_DET}/1
mkdirs -p NEG POS

h5file=train_mosaic_data
training_data=${OUT}/train_mosaic.csv

if [[ ${REG} != "0" ]]; then
    # REGULAR DATASET
    python make_dataset.py $h5file -d=$SRCPATH -o=$training_data
    python make_images.py $SRCPATH -o=${NEG}
fi

# ADD LABELS to dataframe then move misaligned data to '1' image subdirectory

misaligned=`cat ${OUT}/pos.txt`
if [[ ${misaligned} -ne "" ]]; then
    for m in ${misaligned}; do mv ${NEG}/"${m}" ${POS}/.; done
    ls $POS | wc -l
fi

if [[ ${syn} -ne "" ]]; then
    synthetic=${OUT}/synthetic
    mkdir $synthetic

    if [[ ${EXP} != "" && ${MODE} != "" ]]; then
        for dataset in "${DATASETS[@]}"
        do
            python corrupt.py ${dataset} mfi -e=${EXP} -m=${MODE}
        done
    else
        python corrupt.py $SRCPATH $synthetic multi
    fi
        visits=(`find ${SRCPATH}/${dataset}_* -maxdepth 0`)
        for v in "${visits[@]}"
        do
            cd $v; input_file=`find . -name "*input.out"`
            runsinglehap "${input_file}"
            cd ${HOME}/mosaic-ml
            #m=`echo $visits | cut -d'/' -f4`
            #python make_images.py $SRCPATH -o=$POS -c=1 -d=$m
            #mv $v ${SVMCRPT}/.
        done
    

    crpt_training=${OUT}/svm_crpt.csv 

    python make_images.py $SVMCRPT -o=$POS -c=1
    python make_dataset.py $SVMCRPT -o=$crpt_training -c=1
    python mosaic_train.py $training_data $IMG_DET -c=$crpt_training
else
    python mosaic_train.py $training_data $IMG_DET
fi



# Only if using filter similarity embeddings
if [[ ${FILTERS} -ne "" ]]; then
    IMG_FLTR=${OUT}/img/total
    mkdir $IMG_FLTR
    python make_images.py $SVMCRPT -o=$FLTR -c=1 -t=filter
    python make_images.py $SVMCRPT -o=./img/filter -c=1 -t=filter
fi


