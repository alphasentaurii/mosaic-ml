# mosaic-ml
Machine Learning for Single Visit Mosaic (SVM) Alignment Processing

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

```bash
$ docker pull alphasentaurii/mosaic-ml:latest
```

**Option 3: Build Docker image from source**

You can modify the `build-image.sh` convenience script to specify the release version of CALDP.

```bash
$ git clone https://github.com/alphasentaurii/mosaic-ml
$ cd mosaic-ml && bash scripts/build-image.sh
```

----

NOTE - If you're using Docker (Opt1 or Opt2), you'll want to mount the source location of your data before running a container. 

```bash
$ MOUNTS="--mounts type=bind,src=path/to/mydata,target=/developer/home/mosaic-ml/data/mydata"
$ docker run -it alphasentaurii/mosaic-ml:latest --env $MOUNTS
```

----

# Run 

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
$ python make_dataset.py $ml_train_h5_filename -d=$SRCPATH -o=${OUT}/train_mosaic.csv
# Create total detection image sets
$ python make_images.py $SRCPATH -o=${OUT}/img/total/0
```

**Add target class labels**
Aligned images ("success") should be assigned as zero `0` in the `label` column; misaligned ("compromised") images are designated as `1`. Move the image subfolders for each dataset into their respective target subdirectories (`pos.txt` is generated automatically if the dataset has a `labels` column.)

```bash
$ cd ml_data/img/total/ && mkdir 1
$ datasets=`cat pos.txt`
$ for dataset in ${datasets}; do mv "${dataset}" 1/.; done
$ ls 1 | wc -l # 65
```

**Build and Train the Ensemble Classifier**
```bash
$ python mosaic_train.py $OUT/train_mosaic.csv $OUT/training/img/total/
```

----

## Synthesis
Generate artificially misaligned images by "corrupting" single exposures of properly aligned SVM datasets.

```bash
# Stochastic (random) corruption of all exposures for each filter
$ python corrupt_svm.py ia0m04 mfi -e=all -m=stoc
$ cd ia0m04_f110w_all_stoc && runsinglehap wfc3_a0m_04_input.out
$ cd ../ia0m04_f160w_all_stoc && runsinglehap wfc3_a0m_04_input.out 
```

The `svm-train.sh` script allows for up to 4 corruption techniques for each filter. After calculating the artificial misalignment values, it copies the input files it needs to run astrodrizzle and create the single visit mosaic total detection products. Once SVM completes, the script handles all data and image preprocessing for the regular and/or synthetic datasets, then builds and trains the model.

```bash
$ cd mosaic-ml/mosaic-ml
$ bash svm-train.sh $SRCPATH $OUT
```




