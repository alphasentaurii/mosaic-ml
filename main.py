import os
import pandas as pd
from sklearn import train_test_split
import argparse
from data_prep import apply_power_transform, training_data_aug
from image_prep import make_image_sets, training_img_aug
from ensemble import *

HOME = os.path.abspath(os.curdir)
DATA = os.path.join(HOME, 'data')
SUBFOLDER =  os.path.join(DATA, '2021-07-28')
IMG_DIR = os.path.join(SUBFOLDER, 'images')
TRAIN_PATH = f"{IMG_DIR}/training"

def preprocess_data(data):
    df = pd.read_csv(data, index_col='index')
    _, pt_transform = apply_power_transform(df)
    X = df.drop('label', axis=1, inplace=False)
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, shuffle=True, stratify=y_train)
    # data
    X_train_1, _ = training_data_aug(X_train, X_test, X_val, y_train, y_test, y_val)
    # images
    image_sets = [X_train, X_test, X_val]
    train, test, val = make_image_sets(*image_sets, w=128, h=128, d=9)
    # xtrain_idx, xtest_idx, xval_idx= X_train.index, X_test.index, X_val.index
    # train_idx, test_idx, val_idx = train[0], test[0], val[0]
    index, X_tr, y_tr, X_ts, y_ts, X_vl, y_vl = training_img_aug(train, test, val)
    XTR, YTR, XTS, YTS, XVL, YVL = make_ensembles(X_tr, X_ts, X_vl, X_train_1, X_test,
                                                X_val, y_tr, y_ts, y_vl)
    return index, XTR, YTR, XTS, YTS, XVL, YVL

def train_model(XTR, YTR, XTS, YTS, name='ensemble4d', params=dict(batch_size=32, epochs=60, lr=1e-4, 
                decay=[100000, 0.96], early_stopping=None, verbose=2, 
                ensemble=True)):
    ens = Builder(XTR, YTR, XTS, YTS, **params)
    ens.build_ensemble(lr_sched=True)
    ens.fit_generator()
    ens_model = ens.model
    save_model(ens_model, name=name, weights=True)
    ens_history = ens.history
    
    return ens_model, ens_history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str, default=f"{SUBFOLDER}/detection_cleaned.csv", help="path to training data csv file")
    args = parser.parse_args()
    data = args.data
    index, XTR, YTR, XTS, YTS, XVL, YVL = preprocess_data(data)

    ens_model, ens_history = train_model()
    test_idx = index[1]


    com = Compute(ens_model, ens_history, XTR, YTR, XTS, YTS, test_idx)
    com.y_onehot, com.y_pred, com.preds = com.make_predictions()
    com.plots = com.draw_plots()
    com.scores = com.compute_scores()
    com.test_idx = y_test
    com.fnfp = com.track_fnfp()
    predictions = {"y_onehot": com.y_onehot, "preds": com.preds,
                "y_pred": com.y_pred}
    com.results = {
        "history": ens_history.history, "predictions": predictions,  
        "plots": com.plots, "scores": com.scores, 
        "fnfp": com.fnfp, "test_idx": com.test_idx
        }
    ensemble_keys = save_to_pickle(com.results, res_path=f'{HOME}/results/ensemble')

    # VALIDATION
    val_idx = y_val
    eval = Compute(ens_model, ens_history, XTS, YTS, XVL, YVL, val_idx)
    eval.y_onehot, eval.y_pred, eval.preds = eval.make_predictions()
    eval.plots = eval.draw_plots()
    matrix = confusion_matrix(eval.y_test, eval.y_pred)
    eval.fusion_matrix(matrix, normalize=False)
    eval.scores = eval.compute_scores()
    eval.fnfp = eval.track_fnfp()
    eval_predictions = {"y_onehot": eval.y_onehot, "preds": eval.preds,
                "y_pred": eval.y_pred}
    eval.results = {
        "history": ens_history.history, "predictions": eval_predictions,  
        "plots": eval.plots, "scores": eval.scores, 
        "fnfp": eval.fnfp, "test_idx": eval.test_idx
        }
    validation_keys = save_to_pickle(eval.results, res_path=f'{HOME}/results/ensemble/validation')