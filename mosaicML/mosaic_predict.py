import tensorflow as tf
import numpy as np
import pandas as pd
import argparse
import os
import sys
import datetime as dt
from load_images import detector_prediction_images

DIM = 3
CH = 3
SIZE = 128
DEPTH = DIM * CH
SHAPE = (DIM, SIZE, SIZE, CH)

def get_model(model_path):
    """Loads pretrained Keras functional model"""
    print("Loading saved model: ", model_path)
    model = tf.keras.models.load_model(model_path)
    return model


def load_regression_data(filepath):
    """Loads preprocessed regression test data from csv"""
    print("Loading regression test data for MLP")
    data = pd.read_csv(filepath, index_col='index')
    column_order = ['numexp', 'rms_ra', 'rms_dec', 'nmatches', 'point', 'segment', 'gaia', 'det', 'wcs', 'cat']
    try:
        X_data = data[column_order]
        print("Input Shape: ", X_data.shape)
        return X_data
    except Exception as e:
        print(e)
        print("Dataframe must contain these columns: ", column_order)
        sys.exit(1)
    

def load_image_data(X_data, img_path):
    """Loads total detection png images and converts to arrays"""
    print("Loading images into arrays")
    idx, X_img = detector_prediction_images(X_data, img_path, SIZE, SIZE, DEPTH, DIM)
    print("Inputs: ", X_img.shape[0])
    print("Dimensions: ", X_img.shape[1])
    print("Width: ", X_img.shape[2])
    print("Height: ", X_img.shape[3])
    print("Channels: ", X_img.shape[4])
    print("Input Shape: ", X_img.shape)
    return idx, X_img


def load_mixed_inputs(data_file, img_path):
    X_data = load_regression_data(data_file)
    idx, X_img = load_image_data(X_data, img_path)
    diff = X_data.shape[0] - X_img.shape[0]
    if diff > 0:
        X_data = X_data.loc[X_data.index.isin(idx)]
        print(f"{diff} missing images removed from index")
        print(f"X_data: {X_data.shape}\nX_img:  {X_img.shape}")
    return X_data, X_img


def make_ensemble_data(X_data, X_img):
    print("Joining regression data and image arrays")
    X = [X_data, X_img]
    return X


def classify_alignments(model, X):
    """Returns classifier predictions and probability scores"""
    proba = model.predict(X)
    y_pred = np.round(proba[:, 0]).reshape(-1, 1)
    y_proba = proba[:, 0].reshape(-1, 1)
    return y_pred, y_proba


def classification_report(df, output_file):
    outpath = os.path.dirname(output_file)
    P, T = df['y_pred'], df['det'].value_counts()
    C = df.loc[P == 1.0]
    cmp = C['det'].value_counts()
    dets = ['HRC', 'IR', 'SBC', 'UVIS', 'WFC']
    separator = "---"*5
    out = sys.stdout
    with open(f"{outpath}/clf_report.txt", 'w') as f:
        sys.stdout = f
        print("CLASSIFICATION REPORT - ", dt.datetime.now())
        print(separator)
        print("Mean Probability Score: ", df['y_proba'].mean())
        print("Standard Deviation: ", df['y_proba'].std())
        print(separator)
        print("Aligned ('0.0') vs Misaligned ('1.0')")
        print(pd.concat([P.value_counts(), P.value_counts(normalize=True)], axis=1, keys=['cnt', 'pct']))
        print(separator)
        print("Misalignment counts by Detector")
        for i, d in enumerate(dets):
            print(f"{d}\t{cmp[i]} \t ({T[i]}) \t ({np.round((cmp[i]/T[i])*100, 1)}%)")
        sys.stdout = out
    with open(f"{outpath}/compromised.txt", 'w') as file:
        for line in list(C['y_pred'].index):
            file.writelines(f"{line}\n")


def save_preds(X_data, y_pred, y_proba, output_file):
    preds = np.concatenate([y_pred, y_proba], axis=1)
    pred_proba = pd.DataFrame(preds, index=X_data.index, columns={'y_pred', 'y_proba'})
    preds = X_data.join(pred_proba)
    classification_report(preds, output_file)
    preds['index'] = preds.index
    preds.to_csv(output_file, index=False)
    print("Y_PRED + Probabilities added. Dataframe saved as: ", output_file)


def main(model_path, data_file, img_path, output_file):
    ens_clf = get_model(model_path)
    X_data, X_img = load_mixed_inputs(data_file, img_path)
    X = make_ensemble_data(X_data, X_img)
    y_pred, y_proba = classify_alignments(ens_clf, X)
    save_preds(X_data, y_pred, y_proba, output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", type=str, default="data/svm_unlabeled.csv", help="path to preprocessed mosaic data csv file")
    parser.add_argument("img_path", type=str, default="data/img/", help="path to PNG mosaic images")
    parser.add_argument("-m", "--model_path", type=str, default="models/ensemble4d", help="saved model path")
    parser.add_argument("-o", "--output_file", type=str, default="data/svm_predicted.csv", help="path to updated mosaic data csv file (includes alignment predictions and probabilities).")
    args = parser.parse_args()
    data_file = args.data_file
    img_path = args.img_path
    model_path = args.model_path
    output_file = args.output_file
    main(model_path, data_file, img_path, output_file)
