# mosaic-ml
Machine Learning for Single Visit Mosaic (SVM) Alignment Processing

## Setup

```bash
$ git clone https://github.com/alphasentaurii/mosaic-ml
$ cd mosaic-ml
$ python -m virtualenv mosaic-env
$ source mosaic-env/bin/activate
$ python -m pip install -r requirements.txt
```

## Preprocessing
Import, preprocess and (optionally) augment the SVM regression test data and images.

1. Create/Import the regression test data

```bash
$ mkdir ml_data
$ python data_import.py ml_train -d=singlevisits -o=ml_data/svm_training.csv
```

2. Create total detection and filter images

```bash
$ python make_images.py singlevisits -o=ml_data/img/total
$ python make_images.py singlevisits -o=ml_data/img/filter -t=filter
```

3. Add target class labels
Aligned images ("success") should be assigned as zero `0` in the `label` column; misaligned ("compromised") images are designated as `1`. Once the target class labels are added to the dataframe, you'll want to create two subdirectories with the same names ("0" and "1"). Move the image subfolders for each dataset into their respective target subdirectories.

Example (do similar for '0' labels):

```python
pos = list(df.loc[df['label'] == 1)].index.values)
with open('pos.txt', 'w') as f:
    for i in pos:
        f.writelines(f"{i}\n")
len(pos) # 65
```

```bash
$ cd ml_data/img/total/ && mkdir 1
$ datasets=`cat pos.txt`
$ echo $datasets
$ for dataset in ${datasets}; do mv "${dataset}" 1/.; done
$ ls 1 | wc -l # 65
```

## Training

1. Data augmentation
Allows the model to learn latent parameters while reducing the risk of overfitting.

Randomly add noise to continuous variables (MLP data):
- add or subtract calculated laplacian value
- add or subtract calculated logistic value

```bash
$ python -m training.py ml_data/svm_training.csv
```

## Inference


## Synthetic Data Generation
Generate artificially misaligned images by "corrupting" single exposures of properly aligned SVM datasets.

```bash
$ python corrupt_svm.py ia0m04 mfi
$ cd ia0m04_f110w_all_stoc
$ runsinglehap wfc3_a0m_04_input.out 
```

The shell script (`runsvm.sh`) combines all 4 corruption techniques with SVM alignment for each permutation: 

```bash
$ cd singlevisits && mkdir ml_data
$ sh runsvm.sh ('ia0m04' 'j8cw06')
$ mv ia0m04_* j8cw06_* ml_data/.
$ python data_import.py svm_crpt -d=ml_data -o=svm_crpt.csv -c=1
```

Now we can create our synthetic misalignment images:

```bash
$ python make_images.py ./ml_data -o=./img/total_crpt -c=1
$ python make_images.py ./ml_data -o=./img/filter_crpt -c=1 -t=filter
```



