#!/bin/bash -xu
# export SVM_QUALITY_TESTING=on

SRCPATH=${1:-"."} # data/training_2021-07-28/singlevisits
OUT=${2:-"data"}
MODELPATH=${3:-"./models/ensemble4d"}

image_path=${OUT}/img/total/
mkdirs -p $IMG

unlabeled_data=${OUT}/svm_unlabeled.csv
predictions=${OUT}/svm_predictions.csv

python make_dataset.py predict_mosaic_data -d=$SRCPATH -o=$unlabeled_data
python make_images.py $SRCPATH -o=$image_path
python mosaic_predict.py $unlabeled_data $image_path -m=$MODELPATH -o=$predictions