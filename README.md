# mosaic-ml
Machine Learning for Single Visit Mosaic (SVM) Alignment Processing

```bash
mosaic-ml
└── mosaicML
    └── __init__.py
    └── analyze.py
    └── augment.py
    └── corrupt.py
    └── ensemble.py
    └── harvest.py
    └── load_images.py
    └── make_dataset.py
    └── make_images.py
    └── mosaic_predict.py
    └── mosaic_train.py
└── models
└── results
└── data
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
python make_dataset.py svm_unlabeled.h5 -d=path/to/datasets -o=svm_unlabeled.csv
python make_images.py path/to/svm/files -o=path/to/images
python mosaic_predict.py svm_unlabeled.csv path/to/png/images -m=models/ensemble4d -o=predictions.csv
```

----

## II. Training

**Preprocessing**

```bash
# Create/Import the regression test data
$ python make_dataset.py train_svm.h5 -d=path/to/datasets -o=data/train_mosaic.csv
# Create total detection image sets
$ python make_images.py path/to/datasets -o=path/to/save/images
```

**Add target class labels**
Aligned images ("success") should be assigned as zero `0` in the `label` column; misaligned ("compromised") images are designated as `1`. Move the image subfolders for each dataset into their respective target subdirectories (`pos.txt` is generated automatically if the dataset has a `labels` column.)

```bash
$ cd path/to/images && mkdir 1
$ datasets=`cat pos.txt`
$ for dataset in ${datasets}; do mv "${dataset}" 1/.; done
$ ls 1 | wc -l # 65
```

**Build and Train the Ensemble Classifier**
```bash
$ python mosaic_train.py data/train_mosaic.csv path/to/images
```

----

## Synthesis
Generate artificially misaligned images by "corrupting" single exposures of properly aligned SVM datasets.

```bash
# Stochastic (random) corruption of all exposures for each filter
$ python corrupt.py $SRC $synthetic mfi -e=sub -m=stat -p=ia0m04
$ cd ia0m04_f110w_all_stoc && runsinglehap wfc3_a0m_04_input.out
$ cd ../ia0m04_f160w_all_stoc && runsinglehap wfc3_a0m_04_input.out 
```

The `svm-synthesize.sh` script automates the entire workflow from corruption through dataset creation. You can pass in up to 4 arguments, using binary values to switch each stage on or off. Note - Running SVM takes a long time if you're doing a lot of permutations. 

```bash
$ export DATASETS=`cat listofvisits.txt`
$ cd mosaic-ml/mosaic-ml
$ bash svm-synthesize.sh 1 1 0 1 # run corruption, run svm, don't run image gen, make dataset (csv file) 
```




