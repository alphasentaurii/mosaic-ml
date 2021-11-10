import tensorflow as tf
import numpy as np
import pandas as pd
import argparse
import sys

from load_images import detector_prediction_images

DIM = 3
CH = 3
SIZE = 128
DEPTH = DIM * CH
SHAPE = (DIM, SIZE, SIZE, CH)

# load model
def get_model(model_path):
    """Loads pretrained Keras functional model"""
    model = tf.keras.models.load_model(model_path)
    return model

# load dataframe from csv
def load_regression_data(filepath):
    data = pd.read_csv(filepath, index_col='index')
    column_order = ['numexp', 'rms_ra', 'rms_dec', 'nmatches', 'point', 'segment', 'gaia', 'det', 'wcs', 'cat']
    try:
        X_data = data[column_order]
    except Exception as e:
        print(e)
        print("Dataframe must contain these columns: ", column_order)
        sys.exit(1)
    return X_data

# load images from png files
def load_image_data(X_data, img_path):
    idx, X_img = detector_prediction_images(X_data, img_path, SIZE, SIZE, DIM*CH, None)
    print("# Images: ", len(idx))
    return idx, X_img


def make_ensemble_data(data_file, img_path):
    X_data = load_regression_data(data_file)
    idx, X_img = load_image_data(X_data, img_path)
    diff = len(X_data) - len(idx)
    if diff > 0:
        print(f"# Missing images ({diff})")
        X_data = X_data.loc[X_data.index.isin(idx)]
    X = [X_data, X_img]
    return X


# make predictions
def alignment_classifier(model, X):
    """Returns class prediction"""
    proba = model.predict(X)
    y_pred = int(np.argmax(proba, axis=-1))
    y_proba = np.max(proba, axis=-1)
    return y_pred, y_proba


# save results to csv file
def store_results(X, y_pred, y_proba, output_file):
    pred_proba = pd.DataFrame([y_pred, y_proba], index=X.index, columns={'y_pred', 'y_proba'})
    preds = X.join(pred_proba)
    preds['index'] = preds.index
    preds.to_csv(output_file, index=False)
    print("Y_PRED + Probabilities added. Dataframe saved as: ", output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", type=str, default="data/svm_unlabeled.csv", help="path to preprocessed mosaic data csv file")
    parser.add_argument("img_path", type=str, default="data/img/", help="path to PNG mosaic images")
    parser.add_argument("-m", "--model_path", type=str, default="./models/ensemble4d", help="path to saved model folder")
    parser.add_argument("-o", "--output_file", type=str, default="data/svm_predicted.csv", help="path to updated mosaic data csv file (includes alignment predictions and probabilities).")
    args = parser.parse_args()
    data_file = args.data_file
    img_path = args.img_path
    model_path = args.model_path
    output_file = args.output_file
    ens_clf = get_model(model_path)
    X = make_ensemble_data(data_file, img_path)
    y_pred, y_proba = alignment_classifier(ens_clf, X)
    store_results(X, y_pred, y_proba, output_file)


