
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from keras.preprocessing import image
from imageio import imread
from skimage.transform import resize
from tqdm import tqdm
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from PIL import Image
from keras.preprocessing import image
from imageio import imread
from skimage.transform import resize
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from PIL import Image
from keras.preprocessing import image
from imageio import imread
from skimage.transform import resize
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dropout, BatchNormalization, Input, concatenate, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import smart_resize
from tensorflow.python.ops.numpy_ops import np_config
from tqdm import tqdm
from zipfile import ZipFile


HOME = os.path.abspath(os.curdir)
DATA = os.path.join(HOME, 'data')
SUBFOLDER =  os.path.join(DATA, '2021-07-28')
IMG_DIR = os.path.join(SUBFOLDER, 'images')
TRAIN_PATH = f"{IMG_DIR}/training"

SIZE = 128
DIM = 3
CH = 3
DEPTH = CH * DIM
SHAPE = (DIM, SIZE, SIZE, CH)

"""IMAGE DATA PREP FOR 3DCNN"""

def unzip_images(zip_file):
    basedir = os.path.dirname(zip_file)
    key = os.path.basename(zip_file).split('.')[0]
    image_folder = os.path.join(basedir, key+'/')
    os.makedirs(image_folder, exist_ok=True)
    with ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(basedir)
    print(len(os.listdir(image_folder)))
    return image_folder

def extract_images(path_to_zip, extract_to='.'):
    subfolder = os.path.basename(path_to_zip).replace('.zip', '')
    with zipfile.ZipFile(path_to_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    image_path = os.path.join(extract_to, subfolder)
    print(len(os.listdir(image_path)))
    return image_path


def sort_copy_image_labels(df, dpairs):
    
    O, S, G = [],[],[]
    for det, dr in tqdm(dpairs.items()):
        det_idx = df.loc[df['dete_cat']==det].index
        det_vals = df.loc[det_idx]['label'].values
        idx_vals = list(zip(det_idx, det_vals))
        count = 0
        for (i, v) in idx_vals:
            o, s, g = f"{dr}{i}/{i}.png", f"{dr}{i}/{i}_source.png", f"{dr}{i}/{i}_gaia.png"
            if os.path.exists(o):
                if v == 0:
                    do = f"{TRAIN_PATH}/original/0/{os.path.basename(o)}"
                    ds = f"{TRAIN_PATH}/source/0/{os.path.basename(s)}"
                    dg = f"{TRAIN_PATH}/gaia/0/{os.path.basename(g)}"
                    #d3 = f"{TRAIN_PATH}/point"+"/".join(p.split('/')[-2:])
                    #d4 = f"{TRAIN_PATH}/segment"+"/".join(p.split('/')[-2:])
                else:
                    do = f"{TRAIN_PATH}/original/1/{os.path.basename(o)}"
                    ds = f"{TRAIN_PATH}/source/1/{os.path.basename(s)}"
                    dg = f"{TRAIN_PATH}/gaia/1/{os.path.basename(g)}"
                [os.makedirs(os.path.dirname(d), exist_ok=True) for d in [do, ds, dg]]
                shutil.copy(o, do)
                shutil.copy(s, ds)
                shutil.copy(g, dg)
                count += 1
                O.append(do)
                S.append(ds)
                G.append(dg)
        print(f"\n{count} files copied")
        print(f"\n O: {len(O)}\t S: {len(S)}\t G: {len(G)}")
    return dpairs




    
def load_images(zipped=False, copy=False):
    detectors = ['hrc', 'ir', 'sbc', 'uvis', 'wfc']
    dpairs = {}
    for i, d in enumerate(detectors):
        if zipped is True:
            dpath = unzip_images(f'{IMG_DIR}/{d}.zip')
        else:
            dpath = f'{IMG_DIR}/{d}/'
        dpairs[i] = dpath
    
    
    return dpairs



def read_channels(channels, w, h, d, exp=None, color_mode='rgb'):
    t = (w, h)
    image_frames = [image.load_img(c, color_mode=color_mode, target_size=t) for c in channels]
    img = np.array([image.img_to_array(i) for i in image_frames])
    if exp == None:
        img = img.reshape(w, h, d)
    else:
        img = img.reshape(exp, w, h, 3)
    return img

def get_image_paths(i, imgdir):
    o, s, g = imgdir+'/original/', imgdir+'/source/', imgdir+'/gaia/'
    neg = (o+'0/'+i+'.png', s+'0/'+i+'_source.png', g+'0/'+i+'_gaia.png')
    pos = (o+'1/'+i+'.png', s+'1/'+i+'_source.png', g+'1/'+i+'_gaia.png')
    return neg, pos

def detector_image_files(data, w, h, d, exp):
    idx = data.index
    files = []
    labels = []
    for i in idx:
        neg, pos = get_image_paths(i, TRAIN_PATH)
        if os.path.exists(neg[0]):
            files.append(neg)
            labels.append(0)
        elif os.path.exists(pos[0]):
            files.append(pos)
            labels.append(1)
        else:
            print(f"missing: {i}")
            idx.remove(i)
    
    img = []
    for ch1, ch2, ch3 in tqdm(files):
        img.append(read_channels([ch1, ch2, ch3], w, h, d, exp))
    X, y = np.array(img, np.float32), np.array(labels)

    return (idx, X, y)

# Function to read in train/test files and produce X-y data splits.
def make_image_sets(X_train, X_test, X_val, w=128, h=128, d=9, exp=None):
    # y labels are encoded as 0=valid, 1=compromised
    # returns X_train, X_test, y_train, y_test, y_val
    # d=9: 3x3 rgb images (9 channels total)
    display('[i] LOADING IMAGES')
    
    train = detector_image_files(X_train, w, h, d, exp)
    test = detector_image_files(X_test, w, h, d, exp)
    val = detector_image_files(X_val, w, h, d, exp)
    
    print('\n[i] Length of Splits:')
    print(f"X_train={len(train[1])}, X_test={len(test[1])}, X_val={len(val[1])}")
    
    return train, test, val


def plot_image_sets(X_tr, y_tr, X_vl, y_vl):
    posA = X_tr[-X_vl.shape[0]:][y_tr[-X_vl.shape[0]:]==1]
    posB = X_vl[y_vl==1]

    plt.figure(figsize=(10, 10))
    for n in range(5):
        x = image.array_to_img(posA[n][0])
        ax = plt.subplot(5, 5, n + 1)
        ax.imshow(x)
        plt.axis("off")
    plt.show()

    plt.figure(figsize=(10, 10))
    for n in range(5):
        x = image.array_to_img(posB[n][0])
        ax = plt.subplot(5, 5, n + 1)
        ax.imshow(x)
        plt.axis("off")
    plt.show()

# DATA AUGMENTATION

def flip_horizontal(x):
    x = tf.image.flip_left_right(x)
    return x

def flip_vertical(x):
    x = tf.image.flip_up_down(x)
    return x


def rotate_k(x):
    # rotate 90 deg k times
    k = np.random.randint(3)
    x = tf.image.rot90(x, k)
    return x


def color_jitter(x, strength=[0.4, 0.4, 0.4, 0.1]):
    x = tf.image.random_brightness(x, max_delta=0.8 * strength[0])
    x = tf.image.random_contrast(
        x, lower=1 - 0.8 * strength[1], upper=1 + 0.8 * strength[1]
    )
    x = tf.image.random_saturation(
        x, lower=1 - 0.8 * strength[2], upper=1 + 0.8 * strength[2]
    )
    #x = tf.image.random_hue(x, max_delta=0.2 * strength[3])
    # Affine transformations can disturb the natural range of
    # RGB images, hence this is needed.
    x = tf.clip_by_value(x, 0, 255)
    return x#.reshape(DIM, SIZE, SIZE, CH)


def random_apply(func, x, p):
    r = tf.random.uniform([], minval=0, maxval=1) 
    if r < p:
        return func(x)
    else:
        return x


def augment_image(x, c=None):
    # As discussed in the SimCLR paper, the series of augmentation
    # transformations (except for random crops) need to be applied
    # randomly to impose translational invariance.
    from tensorflow.python.ops.numpy_ops import np_config
    np_config.enable_numpy_behavior()
    p = 0.5
    if x.shape[-1] == 9:
        x = x.reshape(3, SIZE, SIZE, 3)
    x = random_apply(flip_horizontal, x, p)
    x = random_apply(flip_vertical, x, p)
    x = random_apply(rotate_k, x, p)
    x = random_apply(color_jitter, x, p)
    if c == 9:
        return x.reshape(SIZE, SIZE, c)
    else:
        return x.reshape(DIM, SIZE, SIZE, CH)


def aug_generator(X_tr, X_ts, X_vl):
    c = 9
    xtr = np.empty(X_tr.shape, dtype='float32')
    xts = np.empty(X_ts.shape, dtype='float32')
    xvl = np.empty(X_vl.shape, dtype='float32')
    for i in range(X_tr.shape[0]):
        xtr[i] = augment_image(X_tr[i], c)
    for i in range(X_ts.shape[0]):
        xts[i] = augment_image(X_ts[i], c)
    for i in range(X_vl.shape[0]):
        xvl[i] = augment_image(X_vl[i], c)
    return xtr, xts, xvl


def expand_dims(Xtr, Xts, Xvl, d, w, h, c):
    ltr, lts, lvl = Xtr.shape[0], Xts.shape[0], Xvl.shape[0]
    Xtr = Xtr.reshape(ltr, d, w, h, c)
    Xts = Xts.reshape(lts, d, w, h, c)
    Xvl = Xvl.reshape(lvl, d, w, h, c)
    return Xtr, Xts, Xvl


def training_img_aug(train, test, val):
    # images
    X_tr, y_tr = train[1], train[2]
    X_ts, y_ts = test[1], test[2]
    X_vl, y_vl = val[1], val[2]
    xtr, xts, xvl = aug_generator(X_tr, X_ts, X_vl)
    X_tr = np.concatenate([X_tr, xtr, xts, xvl])
    y_tr = np.concatenate([y_tr, y_tr, y_ts, y_vl])
    X_tr, X_ts, X_vl = expand_dims(X_tr, X_ts, X_vl, DIM, SIZE, SIZE, CH)
    train_idx = pd.Index(np.concatenate([train[0], train[0], test[0], val[0]]))
    index = (train_idx, test[0], val[0])
    return index, X_tr, y_tr, X_ts, y_ts, X_vl, y_vl

if __name__ == '__main__':
    df = pd.read_csv(f'{SUBFOLDER}/detection_cleaned.csv', index_col='index')
    # O, S, G = sort_copy_image_labels(df, DETDIRS=None)
    X = df.drop('label', axis=1, inplace=False)
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, shuffle=True, stratify=y_train)


    xtrain_idx = X_train.index
    xtest_idx = X_test.index
    xval_idx = X_val.index
    image_sets = [X_train, X_test, X_val]
    train, test, val = make_image_sets(*image_sets, w=SIZE, h=SIZE, d=DEPTH, exp=None)
    index, X_tr, y_tr, X_ts, y_ts, X_vl, y_vl = training_img_aug(train, test, val)
    train_idx = train[0]
    test_idx = test[0]
    val_idx = val[0]

    plot_image_sets(X_tr, y_tr, X_vl, y_vl)