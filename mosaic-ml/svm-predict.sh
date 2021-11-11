#!/bin/bash -xu
# export SVM_QUALITY_TESTING=on

SRCPATH=${1:-"."} # singlevisits/results_2021-10-06
OUT=${2:-"data"} # data/svm/2021-10-06
MODELPATH=${3:-"./models/ensemble4d"}

image_path=${OUT}/img/total
mkdir -p $image_path

unlabeled_data=${OUT}/svm_unlabeled.csv
predictions=${OUT}/svm_predictions.csv

python make_dataset.py predict_mosaic -d=$SRCPATH -o=$unlabeled_data
python make_images.py $SRCPATH -o=$image_path
python mosaic_predict.py $unlabeled_data $image_path -m=$MODELPATH -o=$predictions