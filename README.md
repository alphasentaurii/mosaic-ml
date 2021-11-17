# mosaic-ml

[![Powered by Astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org)
![GitHub repo size](https://img.shields.io/github/repo-size/alphasentaurii/mosaic-ml)
![GitHub license](https://img.shields.io/github/license/alphasentaurii/mosaic-ml?color=black)

<!-- [![GitHub Actions CI Status](https://github.com/spacetelescope/stsci-package-template/workflows/CI/badge.svg](https://github.com/alphasentaurii/mosaic-ml/actions) -->

Machine Learning for the Hubble Space Telescope Mosaic Image Alignment Processing Pipeline

MosaicML is a machine learning package for evaluating the alignment results of HST mosaic images. The neural network is an "ensemble" of two models: a multi-layer perceptron (MLP) for regression test results and a 3D Image CNN  


```bash
mosaic-ml
└── mosaicML
    └── __init__.py
    └── analysis
        └── eda.py
    └── modeling
        └── __init__.py
        └── ensemble.py
        └── mosaic_predict.py
        └── mosaic_train.py
    └── preprocessing
        └── __init__.py
        └── augment.py
        └── corrupt.py
        └── harvest.py
        └── load_images.py
        └── make_dataset.py
        └── make_images.py
└── scripts
    └── svm-predict.sh
    └── svm-prep.sh
    └── svm-synthesize.sh
    └── svm-train.sh
└── models
└── results
└── data
└── docker
    └── Dockerfile
    └── scripts
        └── build-image.sh
        └── run-container.sh
└── requirements.txt
└── assets
└── LICENSE
└── README.md
```

# Install Dependencies / Setup Environment

**Option 1: Conda/Pip**

```bash
$ curl -O https://ssb.stsci.edu/releases/caldp/drizzlecats/latest-linux.yml
$ conda env create -n drizzlecats --file latest-linux.yml
$ conda activate drizzlecats
(drizzlecats) $ git clone https://github.com/alphasentaurii/mosaic-ml
(drizzlecats) $ cd mosaic-ml && pip install -r requirements.txt
```

**Option 2: Pull from Docker Hub**

NOTE - If you're using Docker (Opt 2 or 3), you'll want to mount the source location of your data before running a container.

```bash
$ docker pull alphasentaurii/mosaic-ml:latest
$ sh scripts/run-container.sh path/to/my/data
```

**Option 3: Build Docker image from source**

You can modify the `build-image.sh` convenience script to specify the release version of CALDP.

```bash
$ git clone https://github.com/alphasentaurii/mosaic-ml
$ cd mosaic-ml && sh scripts/build-image.sh
$ sh scripts/run-container.sh path/to/my/data
```

----

# Run Mosaic-ML

The Mosaic-ML package has 3 primary functions:

I. Inference (classify new data)
II. Learning (train model)
III. Synthesis (generate artificial data)


## I. Inference (Classify new data)

```bash
python make_dataset.py svm_unlabeled -d=path/to/datasets -o=svm_unlabeled.csv
python make_images.py path/to/svm/files -o=path/to/images
python mosaic_predict.py svm_unlabeled.csv path/to/png/images -m=models/svm_ensemble -o=predictions.csv
```

----

## II. Training

**Preprocessing**

### 1. Make Regression Test Dataframe

**a) Import/harvest raw data from singlevisits directory**

```bash
# Create/Import the regression test data
$ python make_dataset.py train_svm.h5 -d=path/to/datasets -o=data/train_mosaic.csv
```

**b) Assign Labels**

***Add target class labels***
Aligned ("valid") mosaics should be labeled as zero `0` in the `label` column; misaligned ("compromised") mosaics as `1`. 

### 2. Make PNG Files

**a) Create total detection image sets**

```bash
$ python make_images.py path/to/singlevisits -o=path/to/save/images
```

**b) Move images into labeled directory paths**

Move the image subfolders for each dataset into their respective target subdirectories (`pos.txt` is generated automatically if the dataset has a `labels` column.)

```bash
$ cd path/to/images && mkdir 1
$ datasets=`cat pos.txt`
$ for dataset in ${datasets}; do mv "${dataset}" 1/.; done
$ ls 1 | wc -l # 65
```

### 3. Build and Train the Ensemble Classifier

```bash
$ python mosaic_train.py data/train_mosaic.csv path/to/images
```

----

## Synthesis
Generate artificially misaligned images by "corrupting" single exposures of properly aligned SVM datasets.

```bash
# Stochastic (random) corruption of all exposures for each filter
$ python corrupt.py $SRC $synthetic mfi -e=sub -m=stat -p=ia0m04
$ runsinglehap ia0m04_f110w_sub_stat/wfc3_a0m_04_input.out
$ runsinglehap ia0m04_f160w_sub_stat/wfc3_a0m_04_input.out 
```

The `svm-synthesize.sh` script automates the entire workflow from corruption through dataset creation. You can pass in up to 4 arguments, using binary values to switch each stage on or off.

```bash
$ export DATASETS=`cat listofvisits.txt`
$ cd mosaic-ml/mosaicML
$ bash svm-synthesize.sh 1 1 0 1 # run corruption, run svm, don't run image gen, make dataset 
```
