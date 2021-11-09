#!/bin/bash -xu
export SVM_QUALITY_TESTING=on

SRCPATH=${1:-"."} # data/training_2021-07-28/singlevisits
DATASETS=${2:-""} # ('icon04')

EXP=${expo:-""}
MODE=${mode:-""}
CRPT=${crpt:-""}
REG=${reg:-""}
FILTERS=${filters:-""}

if [[ -z ${DATASETS} ]]; then
    DATASETS=`ls "${SRCPATH}"`
fi

OUT=`echo $SRCPATH | cut -d'/' -f1`
NEG=${OUT}/img/total/0
POS=${OUT}/img/total/1
mkdirs -p NEG POS

if [[ ${REG} -ne "" ]]; then
    # REGULAR DATASET
    python make_dataset.py train_mosaic_data -d=$SRCPATH -o=${OUT}/train_mosaic.csv
    python make_images.py $SRCPATH -o=${NEG}
fi

# ADD LABELS to dataframe then move misaligned data to '1' image subdirectory

misaligned=`cat ${OUT}/pos.txt`
if [[ ${misaligned} -ne "" ]]; then
    for m in ${misaligned}; do mv ${NEG}/"${m}" ${POS}/.; done
    ls $POS | wc -l
fi

if [[ ${CRPT} -ne "" ]]; then
    SVMCRPT=${OUT}/svmcrpt
    mkdir $SVMCRPT

    for dataset in "${DATASETS[@]}"
    do
        if [[ ${EXP} && ${MODE} -ne "" ]]; then
            python corrupt.py ${dataset} mfi -e=${EXP} -m=${MODE}
        else
            python corrupt.py ${dataset} mfi
            python corrupt.py ${dataset} mfi -m=stat
            python corrupt.py ${dataset} mfi -e=sub
            python corrupt.py ${dataset} mfi -e=sub -m=stat
        fi
        visits=(`find ${SRCPATH}/${dataset}_* -maxdepth 0`)
        for v in "${visits[@]}"
        do
            cd $v; input_file=`find . -name "*input.out"`
            runsinglehap "${input_file}"
            cd ${HOME}/mosaic-ml
            #m=`echo $visits | cut -d'/' -f4`
            #python make_images.py $SRCPATH -o=$POS -c=1 -d=$m
            mv $v ${SVMCRPT}/.
        done
    done

    python make_images.py $SVMCRPT -o=$POS -c=1
    python make_dataset.py $SVMCRPT -o=${OUT}/svm_crpt.csv -c=1
fi

# Only if using filter similarity embeddings
if [[ ${FILTERS} -ne "" ]]; then
    FLTR=${OUT}/img/filter
    mkdir $FLTR
    python make_images.py $SVMCRPT -o=$FLTR -c=1 -t=filter
    python make_images.py $SVMCRPT -o=./img/filter -c=1 -t=filter
fi

python mosaic_train.py data/svm_data.csv data/training/img/total/ -c=data/svm-crpt.csv
