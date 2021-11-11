import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import argparse

from tqdm import tqdm
from augment import apply_power_transform, power_transform_matrix, training_data_aug, training_img_aug
from ensemble import Builder, Compute
from load_images import detector_training_images
# from analyze import plot_image_sets
import tensorflow as tf
import pickle
import datetime as dt

DIM = 3
CH = 3
SIZE = 128
DEPTH = DIM * CH
SHAPE = (DIM, SIZE, SIZE, CH)


def split_datasets(df):
    y = df['label']
    X = df.drop('label', axis=1, inplace=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, shuffle=True, stratify=y_train)
    return X_train, X_test, X_val, y_train, y_test, y_val


def normalize_data(df, X_train, X_test, X_val):
    _, px = apply_power_transform(df)
    X_train = power_transform_matrix(X_train, px)
    X_test = power_transform_matrix(X_test, px)
    X_val = power_transform_matrix(X_val, px)
    return X_train, X_test, X_val


# Function to read in train/test files and produce X-y data splits.
def make_image_sets(X_train, X_test, X_val, img_path='.', w=128, h=128, d=9, exp=None):
    # y labels are encoded as 0=valid, 1=compromised
    # returns X_train, X_test, y_train, y_test, y_val
    # d=9: 3x3 rgb images (9 channels total)
    print('[i] LOADING IMAGES')
    
    train = detector_training_images(X_train, img_path, w, h, d, exp)
    test = detector_training_images(X_test, img_path, w, h, d, exp)
    val = detector_training_images(X_val, img_path, w, h, d, exp)
    
    print('\n[i] Length of Splits:')
    print(f"X_train={len(train[1])}, X_test={len(test[1])}, X_val={len(val[1])}")
    
    return train, test, val


def save_model(model, name=None, weights=True):
    """The model architecture, and training configuration (including the optimizer, losses, and metrics)
    are stored in saved_model.pb. The weights are saved in the variables/ directory."""
    if name is None:
        model_name = str(model.name_scope().rstrip("/").upper())
        datestamp = dt.datetime.now().isoformat().split('T')[0]
        name = f"{model_name}_{datestamp}"
    model_path = os.path.join(f"./models", name)
    weights_path = f"{model_path}/weights/ckpt"
    model.save(model_path)
    if weights is True:
        model.save_weights(weights_path)
    for root, _, files in os.walk(model_path):
        indent = "    " * root.count(os.sep)
        print("{}{}/".format(indent, os.path.basename(root)))
        for filename in files:
            print("{}{}".format(indent + "    ", filename))


def save_to_pickle(data_dict, res_path=f'./results/mlp'):
    keys = []
    for k, v in data_dict.items():
        if res_path is not None:
            os.makedirs(f"{res_path}", exist_ok=True)
            key = f"{res_path}/{k}"
        else:
            key = k
        with open(key, "wb") as file_pi:
            pickle.dump(v, file_pi)
            print(f"{k} saved to: {key}")
            keys.append(key)
    print(f"File keys:\n {keys}")
    return keys


def make_tensors(X_train, y_train, X_test, y_test):
    """Convert Arrays to Tensors"""
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)
    return X_train, y_train, X_test, y_test


def make_arrays(X_train, y_train, X_test, y_test):
    X_train = X_train.values
    y_train = y_train.values.reshape(-1,1)
    X_test = X_test.values
    y_test = y_test.values.reshape(-1,1)
    return X_train, y_train, X_test, y_test


def make_ensembles(train_img, test_img, val_img, train_data, test_data, val_data,
                   y_train, y_test, y_val):
    XTR = [train_data, train_img]
    XTS = [test_data, test_img]
    XVL = [val_data, val_img]
    YTR = y_train.reshape(-1, 1)
    YTS = y_test.reshape(-1, 1)
    YVL = y_val.reshape(-1, 1)
    return XTR, YTR, XTS, YTS, XVL, YVL


def preprocess_data(filename, img_path):
    df = pd.read_csv(filename, index_col='index')
    X_train, X_test, X_val, y_train, y_test, y_val = split_datasets(df)
    # IMG DATA
    image_sets = [X_train, X_test, X_val]
    train, test, val = make_image_sets(*image_sets, img_path=img_path, w=128, h=128, d=9)

    # MLP DATA
    X_train_1, _ = training_data_aug(X_train, X_test, X_val, y_train, y_test, y_val)
    if NORM:
        # train_norm, test_norm, val_norm
        X_train_1, X_test, X_val = normalize_data(df, X_train_1, X_test, X_val)
    
    # xtrain_idx, xtest_idx, xval_idx= X_train.index, X_test.index, X_val.index
    # train_idx, test_idx, val_idx = train[0], test[0], val[0]
    index, X_tr, y_tr, X_ts, y_ts, X_vl, y_vl = training_img_aug(train, test, val)
    XTR, YTR, XTS, YTS, XVL, YVL = make_ensembles(X_tr, X_ts, X_vl, X_train_1, X_test,
                                                X_val, y_tr, y_ts, y_vl)
    return index, XTR, YTR, XTS, YTS, XVL, YVL, y_val


def train_model(XTR, YTR, XTS, YTS, name='ensemble4d', params=dict(batch_size=32, epochs=60, lr=1e-4, decay=[100000, 0.96], early_stopping=None, verbose=1, ensemble=True)):
    ens = Builder(XTR, YTR, XTS, YTS, **params)
    ens.build_ensemble(lr_sched=True)
    ens.fit_generator()
    ens_model = ens.model
    save_model(ens_model, name=name, weights=True)
    ens_history = ens.history
    return ens_model, ens_history


def evaluate_results(ens_model, ens_history, XTR, YTR, XTS, YTS, test_idx):
    com = Compute(ens_model, ens_history, XTR, YTR, XTS, YTS, test_idx)
    com.y_onehot, com.y_pred, com.preds = com.make_predictions()
    com.plots = com.draw_plots()
    com.scores = com.compute_scores()
    com.test_idx = test_idx #y_test
    com.fnfp = com.track_fnfp()
    predictions = {"y_onehot": com.y_onehot, "preds": com.preds,
                "y_pred": com.y_pred}
    com.results = {
        "history": ens_history.history, "predictions": predictions,  
        "plots": com.plots, "scores": com.scores, 
        "fnfp": com.fnfp, "test_idx": com.test_idx
        }
    ensemble_keys = save_to_pickle(com.results, res_path=f'./results/ensemble')
    return ensemble_keys


def run_validation(ens_model, ens_history, XTS, YTS, XVL, YVL, val_idx):
    #val_idx = y_val
    eval = Compute(ens_model, ens_history, XTS, YTS, XVL, YVL, val_idx)
    eval.y_onehot, eval.y_pred, eval.preds = eval.make_predictions()
    eval.plots = eval.draw_plots()
    matrix = confusion_matrix(eval.y_test, eval.y_pred)
    eval.fusion_matrix(matrix, normalize=False)
    eval.scores = eval.compute_scores()
    eval.test_idx = val_idx
    eval.fnfp = eval.track_fnfp()
    eval_predictions = {"y_onehot": eval.y_onehot, "preds": eval.preds,
                "y_pred": eval.y_pred}
    eval.results = {
        "history": ens_history.history, "predictions": eval_predictions,  
        "plots": eval.plots, "scores": eval.scores, 
        "fnfp": eval.fnfp, "test_idx": eval.test_idx
        }
    validation_keys = save_to_pickle(eval.results, res_path=f'./results/ensemble/validation')
    return validation_keys


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, default="ml_data/svm_cleaned.csv", help="path to training data csv file(s)")
    parser.add_argument("img_path", type=str, default="ml_data/img/total", help="path to training image directory")
    parser.add_argument("-m", "--model_path", type=str, default="./models/ensemble", help="path to save model")
    parser.add_argument("-n", "--normalize", type=str, default=0, help="apply normalization and scaling to regression test data")
    parser.add_argument("-c", "--corruption_data", type=str, default=None, help="corruption dataframe (csv file) if saved separately")
    args = parser.parse_args()
    filename = args.filename
    img_path = args.img_path
    model_path = args.model_path
    NORM = args.normalization

    index, XTR, YTR, XTS, YTS, XVL, YVL, y_val = preprocess_data(filename, img_path)
    ens_model, ens_history = train_model(XTR, YTR, XTS, YTS, name='ensemble4d', params=dict(batch_size=32, epochs=60, lr=1e-4, decay=[100000, 0.96], early_stopping=None, verbose=2, ensemble=True))
    # test_idx = index[1]
    ensemble_keys = evaluate_results(ens_model, ens_history, XTR, YTR, XTS, YTS, index[1])
    validation_keys = run_validation(ens_model, ens_history, XTS, YTS, XVL, YVL, y_val)
