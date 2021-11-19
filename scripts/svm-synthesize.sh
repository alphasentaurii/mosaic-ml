#!/bin/bash -xu
export SVM_QUALITY_TESTING=on
# sh svm-synthesize.sh CRPT=1 DRIZ=1 DRAW=1 DATA=1
# Run SVM only: bash svm-synthesize.sh 0 1 0 0
# Draw Images only bash svm-synthesize.sh 0 0 1 0
# CRPT=${1:-""}
# DRIZ=${2:-""}
# DRAW=${3:-""}
# DATA=${4:-""}
CRPT=${CRPT:-""}
DRIZ=${DRIZ:-""}
DRAW=${DRAW:-""}
DATA=${DATA:-""}

#DATASETS=${DATASETS:-""} # ('icon04')
SRC=${SRC:-"data/singlevisits"} # data/singlevisits/results_2021-07-28
OUT=${OUT:-"data"}
EXPO=${EXPO:-"*"}
MODE=${MODE:-"*"}

synthetic=${OUT}/synthetic
img=${OUT}/img/1
# FILTERS=${filters:-""}


# Generate synthetic data
if [[ $CRPT -ne 0 ]]; then
    mkdir -p $synthetic
    if [[ ${EXPO} != "*" && ${MODE} != "*" ]]; then
        permutation="mfi -e=${EXPO} -m=${MODE}"
    else
        permutation="multi"
    fi
    if [[ ${DATASETS} -ne "" ]]; then
        echo "Generating synthetic misalignments for ${DATASETS[@]}"
        for dataset in "${DATASETS[@]}"
        do
            python mosaic_ml.corrupt.py $SRC $synthetic $permutation -p=$dataset
        done
    else
        python mosaic_ml.corrupt.py $SRC $synthetic $permutation
    fi
fi


# Create drizzle products
if [[ $DRIZ -ne 0 ]]; then
    if [[ -z ${DATASETS} ]]; then
        DATASETS=`ls "${SRC}"`
    fi
    echo "Drizzling synthetic misalignments for ${DATASETS[@]}"
    for dataset in "${DATASETS[@]}"
    do
        visits=(`find ${synthetic}/${dataset}_* -maxdepth 0`)
        for v in "${visits[@]}"
        do
            warning_file=`find ${v} -name "warning.txt"`
            if [[ -z $warning_file ]]; then
                cd $v; input_file=`find . -name "*input.out"`
                runsinglehap "${input_file}"
                cd ${HOME}/mosaic-ml
            else
                echo "${v} warning file found - skipping"
            fi
        done
    done
fi

# draw png images
if [[ $DRAW -ne 0 ]]; then
    mkdir -p $img
    python mosaic_ml.make_images.py $synthetic $img -c=1
fi

# make dataset
if [[ $DATA -ne 0 ]]; then  
    synth_data=${OUT}/svm_synthetic.csv
    python mosaic_ml.make_dataset.py svm_synth -d=$synthetic -o=$synth_data -c=1
fi

# # Create filter images (for similarity embeddings)
# if [[ ${FILTERS} -ne "" ]]; then
#     IMG_FLTR=${OUT}/img/total
#     mkdir $IMG_FLTR
#     python make_images.py $SVMCRPT -o=$FLTR -c=1 -t=filter
#     python make_images.py $SVMCRPT -o=./img/filter -c=1 -t=filter
# fi


