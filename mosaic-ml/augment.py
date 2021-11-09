import os
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import PowerTransformer
from pprint import pprint
from tqdm import tqdm
from zipfile import ZipFile

SIZE = 128
DIM = 3
CH = 3
DEPTH = CH * DIM
SHAPE = (DIM, SIZE, SIZE, CH)

"""***REGRESSION TEST DATA AUGMENTATION FOR MLP***"""


def apply_power_transform(data, cols=['numexp', 'rms_ra', 'rms_dec', 
                                      'nmatches', 'point', 'segment', 
                                      'gaia']):
    data_cont = data[cols]
    idx = data_cont.index
    pt = PowerTransformer(standardize=False)
    pt.fit(data_cont)
    input_matrix = pt.transform(data_cont)
    lambdas = pt.lambdas_
    normalized = np.empty((len(data), len(cols)))
    mu, sig = [], []
    for i in range(len(cols)):
        v = input_matrix[:, i]
        m, s = np.mean(v), np.std(v)
        x = (v - m) / s
        normalized[:, i] = x
        mu.append(m)
        sig.append(s)
    pt_dict = {"lambdas": lambdas, "mu": np.array(mu), "sigma": np.array(sig)}
    pprint(pt_dict)
    newcols = [c+'_scl' for c in cols]
    df_norm = pd.DataFrame(normalized, index=idx, columns=newcols)
    df = data.drop(cols, axis=1, inplace=False)
    df = df_norm.join(df, how='left')
    return df, pt_dict


def power_transform_matrix(data, pt_data):
    if type(data) == pd.DataFrame:
        data = data.values
    data_cont = data[:, :7]
    data_cat = data[:, -3:]
    nrows = data_cont.shape[0]
    ncols = data_cont.shape[1]
    pt = PowerTransformer(standardize=False)
    pt.fit(data_cont)
    pt.lambdas_ = pt_data['lambdas']
    input_matrix = pt.transform(data_cont)
    normalized = np.empty((nrows, ncols))
    for i in range(data_cont.shape[1]):
        v = input_matrix[:, i]
        m = pt_data['mu'][i]
        s = pt_data['sigma'][i]
        x = (v - m) / s
        normalized[:, i] = x
    data_norm = np.concatenate((normalized, data_cat), axis=1)
    return data_norm
# from tensorflow.python.ops.numpy_ops import np_config
# np_config.enable_numpy_behavior()

def laplacian_noise(x):
    return np.random.laplace(x)


def logistic_noise(x):
    return np.random.logistic(x)

    
def random_apply(func, x, p):
    r = tf.random.uniform([], minval=0, maxval=1) 
    if r < p:
        return func(x)
    else:
        return x


def augment_random_noise(x):
    # augmentation transformations applied randomly to impose translational invariance.
    x = random_apply(laplacian_noise, x, p=0.8)
    x = random_apply(logistic_noise, x, p=0.8)
    return x


def augment_random_integer(x):
    n = np.random.randint(-1, 3)
    if x < 1:
        x = np.abs((x+n))
    else:
        x += n
    return x


def augment_data(xi):
    """Randomly apply noise to continuous data
    """ 
    xi = np.array([augment_random_integer(xi[0]), 
                   augment_random_noise(xi[1]), 
                   augment_random_noise(xi[2]),
                   augment_random_integer(xi[3]),
                   augment_random_noise(xi[4]),
                   augment_random_noise(xi[5]),
                   augment_random_integer(xi[6]),
                   xi[7],
                   xi[8],
                   xi[9]])
    return xi


def training_data_aug(X_train, X_test, X_val, y_train, y_test, y_val):
    xtr = np.empty(X_train.shape, dtype='float32')
    xts = np.empty(X_test.shape, dtype='float32')
    xvl = np.empty(X_val.shape, dtype='float32')
    X_train, X_test, X_val = X_train.values, X_test.values, X_val.values
    y_train, y_test, y_val = y_train.values, y_test.values, y_val.values
    for i in range(X_train.shape[0]):
        xtr[i] = augment_data(X_train[i])
    for i in range(X_test.shape[0]):
        xts[i] = augment_data(X_test[i])
    for i in range(X_val.shape[0]):
        xvl[i] = augment_data(X_val[i])
    X_train = np.concatenate([X_train, xtr, xts, xvl])
    y_train = np.concatenate([y_train, y_train, y_test, y_val])
    print("X_train: ", X_train.shape)
    print("y_train: ", y_train.shape)
    return X_train, y_train




"""***IMAGE DATA PREP FOR 3DCNN***"""



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
    with ZipFile(path_to_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    image_path = os.path.join(extract_to, subfolder)
    print(len(os.listdir(image_path)))
    return image_path

    
# def load_images(zipped=False, copy=False):
#     detectors = ['hrc', 'ir', 'sbc', 'uvis', 'wfc']
#     dpairs = {}
#     for i, d in enumerate(detectors):
#         if zipped is True:
#             dpath = unzip_images(f'{IMG_DIR}/{d}.zip')
#         else:
#             dpath = f'{IMG_DIR}/{d}/'
#         dpairs[i] = dpath
#     return dpairs





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


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(prog="Data Preprocessing", usage="python data_prep.py svm_labeled.csv")
#     parser.add_argument("svmfile", type=str, help="path to svm csv file")
#     parser.add_argument("-d", "--date", type=str, default="2021-07-28", help="date tag for svm dataset")
#     parser.add_argument("-l", "--labeled", type=str, default=1, choices=[0,1], help="labeled (1) or unlabeled (0) data")
#     parser.add_argument("-c","--corruption", type=int, default=0, choices=[0,1], help="run corruption on single exposures")
#     args = parser.parse_args()
#     svm_file = args.svmfile # SUBFOLDER+'/svm_labeled.csv'
#     date = args.date #'2021-07-28'
#     labeled = args.labeled #1
#     corruption = args.corruption #0
#     SUBFOLDER =  os.path.join(DATA, date)
#     SCRUB = os.path.join(SUBFOLDER, 'scrubbed')
#     IMG_DIR = os.path.join(SUBFOLDER, 'images')
#     TRAIN_PATH = f"{IMG_DIR}/training"
#     os.makedirs(SCRUB, exist_ok=True)
