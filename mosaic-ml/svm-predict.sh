#!/bin/bash -xu
export SVM_QUALITY_TESTING=on

MODELPATH=${expo:-"./models/ensemble4d"}
# MODE=${mode:-"stoc"}
# FILTERS=${filters:-""}

SRCPATH=${1:-"."} # data/training_2021-07-28/singlevisits
DATASETS=${2:-""} # ('icon04')

if [[ -z ${DATASETS} ]]; then
    DATASETS=`ls "${SRCPATH}"`
fi

# REGULAR DATASET
OUT=`echo $SRCPATH | cut -d'/' -f1`
IMG=${OUT}/img/total/
mkdirs -p $IMG
python make_dataset.py predict_mosaic_data -d=$SRCPATH -o=${OUT}/predict_mosaic.csv
python make_images.py $SRCPATH -o=$IMG
python mosaic_predict.py ${OUT}/predict_mosaic.csv $IMG -m=$MODELPATH -o=${OUT}/svm_predictions.csv