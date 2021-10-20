# STANDARD libraries
import os
import pandas as pd
import numpy as np
import zipfile
import pickle
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from imageio import imread
from IPython import display
from skimage.transform import resize
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('seaborn-bright')
font_dict={'family':'"Titillium Web", monospace','size':16}
mpl.rc('font',**font_dict)
#ignore pink warnings
import warnings
warnings.filterwarnings('ignore')
from astropy.visualization import ImageNormalize,ZScaleInterval
from tensorflow.keras.preprocessing.image import smart_resize
from tensorflow.python.ops.numpy_ops import np_config
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from keras.preprocessing import image
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, \
BatchNormalization, Input, concatenate, Activation
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from scipy.ndimage import convolve
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    roc_curve, 
    roc_auc_score, 
    accuracy_score, 
    precision_recall_curve, 
    average_precision_score,
    f1_score,
    classification_report,
    confusion_matrix
    )
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, PowerTransformer
from sklearn.model_selection import GridSearchCV
from tensorflow.keras import layers
from tensorflow.keras import Input, Model
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
import time
import datetime as dt

from image_prep import read_channels

HOME = os.path.abspath(os.curdir)
DATA = os.path.join(HOME, 'data')
SUBFOLDER =  os.path.join(DATA, '2021-07-28')
IMG_DIR = os.path.join(SUBFOLDER, 'images')
TRAIN_PATH = f"{IMG_DIR}/training"
FILTER = TRAIN_PATH+'/filter'

HRC = FILTER+'/hrc/'
IR = FILTER+'/ir/'
SBC = FILTER+'/sbc/'
UVIS = FILTER+'/uvis/'
WFC = FILTER+'/wfc/'
DPAIRS = dict(zip([0,1,2,3,4], [HRC, IR, SBC, UVIS, WFC]))

DEPTH = 2
DIM = 2
CH = 1
SIZE = 128
SHAPE = (DIM, SIZE, SIZE, CH)

def get_datasets(df):
    index = df.index
    datasets = []
    for i in index:
        datasets.append(i.split('_')[-1])
    return datasets

def get_filter_groups(df, filter_path, dete_cat):
    total = df.loc[df['dete_cat'] == dete_cat]
    idx = total.index
    datasets = get_datasets(total)
    filter_pairs = {}
    for (i, dataset) in list(zip(idx, datasets)):
        filter_pairs[dataset] = {'total': i, 'filters': []}
        fpath = os.path.join(filter_path, dataset)
        if os.path.exists(fpath):
            for f in os.listdir(fpath):
                filter_pairs[dataset]['filters'].append(f)
    return filter_pairs


def make_filter_pairs(df, dpairs=DPAIRS):
    ff_pairs = {}
    for k, v in tqdm(dpairs.items()):
        ff_path = v
        ff_pairs[v] = get_filter_groups(df, ff_path, k)
    return ff_pairs


def total_filter_pairs(df):
    """Pairs default total detection image with each of its associated filter images
    Returns positive and negative label lists of tuples for paths of each pairing
    **Args
    df: dataframe with total detection image data index and SVM alignment label 
    det_pairs: tuple or list of tuples for (detector, dete_cat) pairs
    """
    ff_pairs = make_filter_pairs(df)

    idx = df.index
    total_pos = os.path.join(TRAIN_PATH, 'original', '1')
    total_neg = os.path.join(TRAIN_PATH, 'original', '0')

    files, labels = [], []

    for ff_path, data_dict in tqdm(ff_pairs.items()):
        for dataset, pairs in data_dict.items():
            t = pairs['total']
            tpos = os.path.join(total_pos, t+'.png')
            tneg = os.path.join(total_neg, t+'.png')
            for f in ff_pairs[ff_path][dataset]['filters']:
                fpath = os.path.join(ff_path, dataset, f)
                if os.path.exists(fpath):
                    if os.path.exists(tpos):
                            pair_pos = (tpos, fpath)
                            files.append(pair_pos)
                            labels.append(1)
                            #pair_pos.append(pair)
                    elif os.path.exists(tneg):
                            pair_neg = (tneg, fpath)
                            files.append(pair_neg)
                            labels.append(0)
                            #pair_neg.append(pair)
                    else:
                        print(f"missing: {i}")
                        idx.remove(i)
    
    return idx, files, labels


def filter_pair_images(df, w, h, d, exp=None):
    idx, files, labels = total_filter_pairs(df)
    img = []
    for ch1, ch2 in tqdm(files):
        img.append(read_channels([ch1, ch2], w, h, d, exp))
    X, y = np.array(img, np.float32), np.array(labels)

    return (idx, X, y)


# Function to read in train/test files and produce X-y data splits.
def make_filter_image_sets(X_train, X_test, X_val, w=128, h=128, d=6):
    # y labels are encoded as 0=valid, 1=compromised
    # returns X_train, X_test, y_train, y_test, y_val
    # d=6: 2x3 rgb images (6 channels total)
    display('[i] LOADING IMAGES')
    exp = None
    
    train = filter_pair_images(X_train, w, h, d, exp, color_mode='gray')
    test = filter_pair_images(X_test, w, h, d, exp, color_mode='gray')
    val = filter_pair_images(X_val, w, h, d, exp, color_mode='gray')
    
    print('\n[i] Length of Splits:')
    print(f"X_train={len(train[1])}, X_test={len(test[1])}, X_val={len(val[1])}")
    
    return train, test, val

# Provided two tensors t1 and t2
# Euclidean distance = sqrt(sum(square(t1-t2)))
def euclidean_distance(vects):
    """Find the Euclidean distance between two vectors.

    Arguments:
        vects: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """

    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))

def loss(margin=1):
    """Provides 'constrastive_loss' an enclosing scope with variable 'margin'.
    args:
    margin: Integer, defines the baseline distance for pair dissimilarity (default is 1).
    Returns 'constrastive_loss' function with data ('margin') attached.
    """
    # Contrastive loss = mean( (1-true_value) * square(prediction) +
    #                         true_value * square( max(margin-prediction, 0) ))
    def contrastive_loss(y_true, y_pred):
        """Calculates the constrastive loss.
        args:
        y_true: List of labels, each label is of type float32.
        y_pred: List of predictions of same length as of y_true,
        each label is of type float32.
        Returns a tensor containing constrastive loss as floating point value.
        """
        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean(
            (1 - y_true) * square_pred + (y_true) * margin_square
        )
    return contrastive_loss

df = pd.read_csv(f'{SUBFOLDER}/detection_cleaned.csv', index_col='index')
# O, S, G = sort_copy_image_labels(df, DETDIRS=None)
X = df.drop('label', axis=1, inplace=False)
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, shuffle=True, stratify=y_train)

image_sets = [X_train, X_test, X_val]
ftrain, ftest, fval = make_filter_image_sets(*image_sets, w=SIZE, h=SIZE, d=DEPTH, exp=None)
fX_train = ftrain[1].reshape(ftrain[1].shape[0], 2, 128, 128, 3) #1546
fX_test = ftest[1].reshape(ftest[1].shape[0], 2, 128, 128, 3) # 386
fX_val = fval[1].reshape(171, 2, 128, 128, 3)
fy_train = ftrain[2].reshape(-1, 1).astype('float32')
fy_test = ftest[2].reshape(-1, 1).astype('float32')
fy_val = fval[2].reshape(-1, 1).astype('float32')
fX_train_1 = fX_train[:, 0]
fX_train_2 = fX_train[:, 1]
fX_test_1 = fX_test[:, 0]
fX_test_2 = fX_test[:, 1]
fX_val_1 = fX_val[:, 0]
fX_val_2 = fX_val[:, 1]





input = layers.Input((SIZE, SIZE, 3))
x = tf.keras.layers.BatchNormalization()(input)
x = layers.Conv2D(4, (5, 5), activation="tanh")(x)
x = layers.AveragePooling2D(pool_size=(2, 2))(x)
x = layers.Conv2D(16, (5, 5), activation="tanh")(x)
x = layers.AveragePooling2D(pool_size=(2, 2))(x)
x = layers.Flatten()(x)

x = tf.keras.layers.BatchNormalization()(x)
x = layers.Dense(10, activation="tanh")(x)
embedding_network = Model(input, x)


input_1 = layers.Input((SIZE, SIZE, 3))
input_2 = layers.Input((SIZE, SIZE, 3))

# As mentioned above, Siamese Network share weights between
# tower networks (sister networks). To allow this, we will use
# same embedding network for both tower networks.
tower_1 = embedding_network(input_1)
tower_2 = embedding_network(input_2)

merge_layer = layers.Lambda(euclidean_distance)([tower_1, tower_2])
normal_layer = layers.BatchNormalization()(merge_layer)
output_layer = layers.Dense(1, activation="sigmoid")(normal_layer)
siamese = Model(inputs=[input_1, input_2], outputs=output_layer)

siamese.compile(loss=loss(margin=1), optimizer="RMSprop", metrics=["accuracy"])
siamese.summary()

history = siamese.fit(
    [fX_train_1, fX_train_2],
    fy_train,
    validation_data=([fX_test_1, fX_test_2], fy_test),
    batch_size=32,
    epochs=60,
)



