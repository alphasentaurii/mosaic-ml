import os
import pandas as pd
from sklearn import train_test_split
import argparse
from data_augment import apply_power_transform, power_transform_matrix, training_data_aug
from prep_images import make_image_sets, training_img_aug
from ensemble import *
from evaluate import Compute


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
    
# def normalize_data(pt_transform, X_train_1, X_test, X_val):
#     X_train = power_transform_matrix(X_train_1, pt_transform)
#     X_test = power_transform_matrix(X_test, pt_transform)
#     X_val = power_transform_matrix(X_val, pt_transform)
#     return X_train, X_test, X_val

def regression_data(df):
    
    # _, pt_transform = apply_power_transform(df)
    X = df.drop('label', axis=1, inplace=False)
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, shuffle=True, stratify=y_train)
    X_train_1, y_train_1 = training_data_aug(X_train, X_test, X_val, y_train, y_test, y_val)
    # if normalize is True:
    #     X_train_1, X_test, X_val = normalize_data(pt_transform, X_train_1, X_test, X_val)
    return X_train_1, X_train, X_test, X_val, y_train, y_test, y_val



def preprocess_data(filename):
    df = pd.read_csv(filename, index_col='index')
    X_train_1, X_train, X_test, X_val, y_train, y_test, y_val = regression_data(df)
    
    # images
    image_sets = [X_train, X_test, X_val]
    train, test, val = make_image_sets(*image_sets, w=128, h=128, d=9)
    # xtrain_idx, xtest_idx, xval_idx= X_train.index, X_test.index, X_val.index
    # train_idx, test_idx, val_idx = train[0], test[0], val[0]
    index, X_tr, y_tr, X_ts, y_ts, X_vl, y_vl = training_img_aug(train, test, val)
    XTR, YTR, XTS, YTS, XVL, YVL = make_ensembles(X_tr, X_ts, X_vl, X_train_1, X_test,
                                                X_val, y_tr, y_ts, y_vl)
    return index, XTR, YTR, XTS, YTS, XVL, YVL, y_val

def train_model(XTR, YTR, XTS, YTS, name='ensemble4d', params=dict(batch_size=32, epochs=60, lr=1e-4, decay=[100000, 0.96], early_stopping=None, verbose=2, ensemble=True)):
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
    return ensemble_keys

def run_validation(ens_model, ens_history, XTS, YTS, XVL, YVL, val_idx):
    #val_idx = y_val
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
    return validation_keys


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, default="ml_data/svm_cleaned.csv", help="path to training data csv file(s)")
    parser.add_argument("img_path", type=str, default="ml_data/img/total", help="path to training image directory")
    args = parser.parse_args()
    filename = args.filename
    img_path = args.img_path

    index, XTR, YTR, XTS, YTS, XVL, YVL, y_val = preprocess_data(filename)
    ens_model, ens_history = train_model()
    # test_idx = index[1]
    ensemble_keys = evaluate_results(ens_model, ens_history, XTR, YTR, XTS, YTS, index[1])
    validation_keys = run_validation(ens_model, ens_history, XTS, YTS, XVL, YVL, y_val)

    
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




# if __name__ == '__main__':
#     df = pd.read_csv(f'{SUBFOLDER}/detection_cleaned.csv', index_col='index')
#     df = df.drop(['category', 'ra_targ', 'dec_targ', 'imgname'], axis=1, inplace=False)
#     df_scl, pt_transform = apply_power_transform(df, cols=['n_exposures', 'rms_ra', 'rms_dec', 'nmatches', 'point', 'segment', 'gaia_sources'])
#     y = df['label']
#     X = df.drop('label', axis=1, inplace=False)

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)
#     X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, shuffle=True, stratify=y_train)

#     image_sets = [X_train, X_test, X_val]
#     train, test, val = make_image_sets(*image_sets, w=SIZE, h=SIZE, d=DEPTH, exp=None)
#     index, X_tr, y_tr, X_ts, y_ts, X_vl, y_vl = training_img_aug(train, test, val)
#     XTR, YTR, XTS, YTS, XVL, YVL = make_ensembles(X_tr, X_ts, X_vl, X_train_1, X_test,
#                                               X_val, y_tr, y_ts, y_vl)
#     ens = Builder(XTR, YTR, XTS, YTS, batch_size=32, epochs=60, lr=1e-4, 
#                 decay=[100000, 0.96], early_stopping=None, verbose=2, 
#                 ensemble=True)
#     ens.build_ensemble(lr_sched=True)
#     ens.fit_generator()
#     ens_model = ens.model

#     save_model(ens_model, name='ensemble4d', weights=True)

#     ens_history = ens.history
#     test_idx = index[1]
#     com = Compute(ens_model, ens_history, XTR, YTR, XTS, YTS, test_idx)
#     com.y_onehot, com.y_pred, com.preds = com.make_predictions()
#     com.plots = com.draw_plots()
#     com.scores = com.compute_scores()
#     com.test_idx = y_test
#     com.fnfp = com.track_fnfp()
#     predictions = {"y_onehot": com.y_onehot, "preds": com.preds,
#                 "y_pred": com.y_pred}
#     com.results = {"history": ens_history.history, "predictions": predictions,  "plots": com.plots, "scores": com.scores, 
#                 "fnfp": com.fnfp, "test_idx": com.test_idx}
#     ensemble_keys = save_to_pickle(com.results, res_path=f'{HOME}/results/ensemble')

#     # VALIDATION
#     val_idx = y_val
#     eval = Compute(ens_model, ens_history, XTS, YTS, XVL, YVL, val_idx)
#     eval.y_onehot, eval.y_pred, eval.preds = eval.make_predictions()
#     eval.plots = eval.draw_plots()
#     matrix = confusion_matrix(eval.y_test, eval.y_pred)
#     eval.fusion_matrix(matrix, normalize=False)
#     eval.scores = eval.compute_scores()
#     eval.fnfp = eval.track_fnfp()
#     eval_predictions = {"y_onehot": eval.y_onehot, "preds": eval.preds,
#                 "y_pred": eval.y_pred}
#     eval.results = {
#         "history": ens_history.history, "predictions": eval_predictions,  
#         "plots": eval.plots, "scores": eval.scores, 
#         "fnfp": eval.fnfp, "test_idx": eval.test_idx
#         }
#     save_to_pickle(eval.results, res_path=f'{HOME}/results/ensemble/validation')
