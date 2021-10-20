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

from data_prep import apply_power_transform, training_data_aug, augment_data
from image_prep import make_image_sets, training_img_aug, augment_image

HOME = os.path.abspath(os.curdir)
DATA = os.path.join(HOME, 'data')
SUBFOLDER =  os.path.join(DATA, '2021-07-28')
IMG_DIR = os.path.join(SUBFOLDER, 'images')
TRAIN_PATH = f"{IMG_DIR}/training"

DIM = 3
CH = 3
SIZE = 128
DEPTH = DIM * CH
SHAPE = (DIM, SIZE, SIZE, CH)

def proc_time(start, end):
    duration = np.round((end - start), 2)
    proc_time = np.round((duration / 60), 2)
    if duration > 3600:
        t = f"{np.round((proc_time / 60), 2)} hours."
    elif duration > 60:
        t = f"{proc_time} minutes."
    else:
        t = f"{duration} seconds."
    print(f"Process took {t}\n")

def save_model(model, name="mlp", weights=True):
    """The model architecture, and training configuration (including the optimizer, losses, and metrics)
    are stored in saved_model.pb. The weights are saved in the variables/ directory."""
    model_path = os.path.join(f"{HOME}/models", name)
    weights_path = f"{model_path}/weights/ckpt"
    model.save(model_path)
    if weights is True:
        model.save_weights(weights_path)
    for root, _, files in os.walk(model_path):
        indent = "    " * root.count(os.sep)
        print("{}{}/".format(indent, os.path.basename(root)))
        for filename in files:
            print("{}{}".format(indent + "    ", filename))

def save_to_pickle(data_dict, res_path=f'{HOME}/results/mlp'):
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

class Builder:
    def __init__(self, X_train, y_train, X_test, y_test, batch_size=32, epochs=60, 
                 lr=1e-4, decay=[100000, 0.96], early_stopping=None, verbose=2, 
                 ensemble=False):
        self.X_train = X_train
        self.X_test = X_test 
        self.y_train = y_train
        self.y_test = y_test
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.decay = decay
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.ensemble = ensemble
        self.callbacks = None
        self.model = None
        self.mlp = None
        self.cnn = None
        self.history = None


    def decay_learning_rate(self):
        """set learning schedule with exponential decay
        lr (initial learning rate: 1e-4
        decay: [decay_steps, decay_rate]
        """
        lr_schedule = optimizers.schedules.ExponentialDecay(
            self.lr, decay_steps=self.decay[0], decay_rate=self.decay[1],
            staircase=True
            )
        return lr_schedule
    
    def set_callbacks(self):
        """
        early_stopping: 'val_accuracy' or 'val_loss'
        """
        model_name = str(self.model.name_scope().rstrip("/").upper())
        checkpoint_cb = callbacks.ModelCheckpoint(
            f"{model_name}_checkpoint.h5", save_best_only=True
            )
        early_stopping_cb = callbacks.EarlyStopping(
            monitor=self.early_stopping, patience=15
            )
        self.callbacks = [checkpoint_cb, early_stopping_cb]
        return self.callbacks

    def build_mlp(self, input_shape=None, lr_sched=True, layers=[18, 32, 64, 32, 18]):
        if input_shape is None:
            input_shape = self.X_train.shape[1]
        model = Sequential()
        # visible layer
        inputs = Input(shape=(input_shape,), name="svm_inputs")
        # hidden layers
        x = Dense(layers[0], activation="leaky_relu", name=f"1_dense{layers[0]}")(inputs)
        for i, layer in enumerate(layers[1:]):
            i += 1
            x = Dense(layer, activation="leaky_relu", name=f"{i+1}_dense{layer}")(x)
        # output layer
        if self.ensemble is True:
            self.mlp = Model(inputs, x, name="mlp_ensemble")
            return self.mlp
        else:
            outputs = Dense(1, activation="sigmoid", name="svm_output")(x)
            self.model = Model(inputs=inputs, outputs=outputs, name="sequential_mlp")
            if lr_sched is True:
                lr_schedule = self.decay_learning_rate()
            else:
                lr_schedule = self.lr
            self.model.compile(loss="binary_crossentropy", 
                            optimizer=Adam(learning_rate=lr_schedule), 
                            metrics=["accuracy"])
            return self.model

    def build_cnn(self, input_shape=None, lr_sched=True):
        """Build a 3D convolutional neural network for RGB image triplets"""
        if input_shape is None:
            input_shape = self.X_train.shape[1:]

        inputs = Input(input_shape, name='img3d_inputs')

        x = layers.Conv3D(filters=32, kernel_size=3, padding="same", 
                          data_format="channels_last", 
                          activation="leaky_relu")(inputs)
        x = layers.MaxPool3D(pool_size=2)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv3D(filters=32, kernel_size=3, padding="same", 
                          activation="leaky_relu")(x)
        x = layers.MaxPool3D(pool_size=1)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv3D(filters=64, kernel_size=3, padding="same", 
                          activation="leaky_relu")(x)
        x = layers.MaxPool3D(pool_size=1)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv3D(filters=128, kernel_size=3, padding="same", 
                          activation="leaky_relu")(x)
        x = layers.MaxPool3D(pool_size=1)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv3D(filters=256, kernel_size=3, padding="same",
                          activation="leaky_relu")(x)
        x = layers.MaxPool3D(pool_size=1)(x)
        x = layers.BatchNormalization()(x)

        x = layers.GlobalAveragePooling3D()(x)
        x = layers.Dense(units=512, activation="leaky_relu")(x)
        x = layers.Dropout(0.3)(x)

        if self.ensemble is True:
            self.cnn = Model(inputs, x, name="cnn_ensemble")
            return self.cnn
        else:
            outputs = layers.Dense(units=1, activation="sigmoid", name="img3d_output")(x)
            # Define the model.
            if lr_sched is True:
                lr_schedule = self.decay_learning_rate()
            else:
                lr_schedule = self.lr
            self.model = Model(inputs, outputs, name="cnn3d")
            self.model.compile(loss="binary_crossentropy", 
                        optimizer=Adam(learning_rate=lr_schedule),
                        metrics=["accuracy"]
                        )
            return self.model

    def build_ensemble(self, lr_sched=True):
        self.mlp = self.build_mlp(input_shape=self.X_train[0].shape[1])
        self.cnn = self.build_cnn(input_shape=self.X_train[1].shape[1:])
        combinedInput = concatenate([self.mlp.output, self.cnn.output])
        x = Dense(9, activation="leaky_relu", name="combined_input")(combinedInput)
        x = Dense(1, activation="sigmoid", name="ensemble_output")(x)
        self.model = Model(inputs=[self.mlp.input, self.cnn.input], outputs=x, name="ensemble4d")
        if lr_sched is True:
            lr_schedule = self.decay_learning_rate()
        else:
            lr_schedule = self.lr
        self.model.compile(loss="binary_crossentropy", 
                            optimizer=Adam(learning_rate=lr_schedule), 
                            metrics=["accuracy"])
        return self.model


    def batch_maker(self):
        """
        Gives equal number of positive and negative samples rotating randomly                
        The output of the generator must be either
        - a tuple `(inputs, targets)`
        - a tuple `(inputs, targets, sample_weights)`.

        This tuple (a single output of the generator) makes a single
        batch. The last batch of the epoch is commonly smaller than the others, 
        if the size of the dataset is not divisible by the batch size.
        The generator loops over its data indefinitely. 
        An epoch finishes when `steps_per_epoch` batches have been seen by the model.
        
        """
        # hb: half-batch
        hb = self.batch_size // 2
        # Returns a new array of given shape and type, without initializing.
        # x_train.shape = (2016, 3, 128, 128, 3)
        if len(self.X_train.shape) == 2:
            xb = np.empty((self.batch_size, self.X_train.shape[1]), dtype='float32')
            augmenter = augment_data
        else:
            xb = np.empty((self.batch_size, self.X_train.shape[1], 
                        self.X_train.shape[2], self.X_train.shape[3],
                        self.X_train.shape[4]), 
                        dtype='float32')
            augmenter = augment_image
        
        #y_train.shape = (2016, 1)
        yb = np.empty((self.batch_size, self.y_train.shape[1]), dtype='float32')
        
        pos = np.where(self.y_train[:,0] == 1.)[0]
        neg = np.where(self.y_train[:,0] == 0.)[0]

        # rotating each of the samples randomly
        while True:
            np.random.shuffle(pos)
            np.random.shuffle(neg)
        
            xb[:hb] = self.X_train[pos[:hb]]
            xb[hb:] = self.X_train[neg[hb:self.batch_size]]
            yb[:hb] = self.y_train[pos[:hb]]
            yb[hb:] = self.y_train[neg[hb:self.batch_size]]
        
            for i in range(self.batch_size):
                # size = np.random.randint(xb.shape[1])
                # xb[i] = np.roll(xb[i], size, axis=0)
                xb[i] = augmenter(xb[i])

            yield xb, yb

    def batch_ensemble(self):
        """
        Gives equal number of positive and negative samples rotating randomly                
        The output of the generator must be either
        - a tuple `(inputs, targets)`
        - a tuple `(inputs, targets, sample_weights)`.

        This tuple (a single output of the generator) makes a single
        batch. The last batch of the epoch is commonly smaller than the others, 
        if the size of the dataset is not divisible by the batch size.
        The generator loops over its data indefinitely. 
        An epoch finishes when `steps_per_epoch` batches have been seen by the model.
        
        """
        # hb: half-batch
        hb = self.batch_size // 2
        # Returns a new array of given shape and type, without initializing.
        # x_train.shape = (2016, 3, 128, 128, 3)
        
        xa = np.empty((self.batch_size, self.X_train[0].shape[1]), dtype='float32')
        
        xb = np.empty((self.batch_size, self.X_train[1].shape[1], 
                        self.X_train[1].shape[2], self.X_train[1].shape[3],
                        self.X_train[1].shape[4]), 
                        dtype='float32')
        
        
        #y_train.shape = (2016, 1)
        yb = np.empty((self.batch_size, self.y_train.shape[1]), dtype='float32')
        
        pos = np.where(self.y_train[:,0] == 1.)[0]
        neg = np.where(self.y_train[:,0] == 0.)[0]

        # rotating each of the samples randomly
        while True:
            np.random.shuffle(pos)
            np.random.shuffle(neg)

            xa[:hb] = self.X_train[0][pos[:hb]]
            xa[hb:] = self.X_train[0][neg[hb:self.batch_size]]
            xb[:hb] = self.X_train[1][pos[:hb]]
            xb[hb:] = self.X_train[1][neg[hb:self.batch_size]]
            yb[:hb] = self.y_train[pos[:hb]]
            yb[hb:] = self.y_train[neg[hb:self.batch_size]]
        
            for i in range(self.batch_size):
                xa[i] = augment_data(xa[i])
                xb[i] = augment_image(xb[i])

            yield [xa, xb], yb

    def fit_generator(self):
        """
        Fits cnn and returns keras history
        Gives equal number of positive and negative samples rotating randomly  
        """
        model_name = str(self.model.name_scope().rstrip("/").upper())
        print(f"FITTING MODEL...")
        validation_data = (self.X_test, self.y_test)

        if self.early_stopping is not None:
            self.callbacks = self.set_callbacks()

        t_start = time.time()
        start = dt.datetime.fromtimestamp(t_start).strftime("%m/%d/%Y - %I:%M:%S %p")
        print(f"\nTRAINING STARTED: {start} ***{model_name}***")

        if self.ensemble is True:
            make_batches = self.batch_ensemble()
            steps_per_epoch = (self.X_train[0].shape[0]//self.batch_size)
        else:
            make_batches = self.batch_maker()
            steps_per_epoch = (self.X_train.shape[0]//self.batch_size)

        self.history = self.model.fit(make_batches, validation_data=validation_data, 
                            verbose=self.verbose, epochs=self.epochs,
                            steps_per_epoch=steps_per_epoch,
                            callbacks=self.callbacks)
        t_end = time.time()
        end = dt.datetime.fromtimestamp(t_end).strftime("%m/%d/%Y - %I:%M:%S %p")
        print(f"\nTRAINING COMPLETE: {end} ***{model_name}***")
        proc_time(t_start, t_end)
        self.model.summary()
        return self.history

class Compute(Builder):

    def __init__(self, model, history, X_train, y_train, X_test, y_test, test_idx):
        self.model = model
        self.history = history.history
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.test_idx = test_idx
        self.y_onehot = None
        self.preds = None
        self.y_pred = None
        self.plots = None
        self.scores = None
        self.fnfp = None
        self.results = None
    
    def keras_history(self, figsize=(10,4)):
        """
        side by side sublots of training val accuracy and loss (left and right respectively)
        """
        
        import matplotlib.pyplot as plt
        
        fig, axes =plt.subplots(ncols=2,figsize=(15,6))
        axes = axes.flatten()

        ax = axes[0]
        ax.plot(self.history['accuracy'])
        ax.plot(self.history['val_accuracy'])
        ax.set_title('Model Accuracy')
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Epoch')
        ax.legend(['Train', 'Test'], loc='upper left')

        ax = axes[1]
        ax.plot(self.history['loss'])
        ax.plot(self.history['val_loss'])
        ax.set_title('Model Loss')
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epoch')
        ax.legend(['Train', 'Test'], loc='upper left')
        fig.show()

    def fusion_matrix(self, matrix, classes=['aligned', 'misaligned'], normalize=True, title='Confusion Matrix', cmap='Blues',
        print_raw=False): 
        """
        FUSION MATRIX!
        -------------
        It's like a confusion matrix...without the confusion.
        
        matrix: can pass in matrix or a tuple (ytrue,ypred) to create on the fly 
        classes: class names for target variables
        """
        from sklearn import metrics                       
        from sklearn.metrics import confusion_matrix #ugh
        import itertools
        import numpy as np
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        
        # make matrix if tuple passed to matrix:
        if isinstance(matrix, tuple):
            y_true = matrix[0].copy()
            y_pred = matrix[1].copy()
            
            if y_true.ndim>1:
                y_true = y_true.argmax(axis=1)
            if y_pred.ndim>1:
                y_pred = y_pred.argmax(axis=1)
            fusion = metrics.confusion_matrix(y_true, y_pred)
        else:
            fusion = matrix
        
        # INTEGER LABELS
        if classes is None:
            classes=list(range(len(matrix)))

        #NORMALIZING
        # Check if normalize is set to True
        # If so, normalize the raw fusion matrix before visualizing
        if normalize:
            fusion = fusion.astype('float') / fusion.sum(axis=1)[:, np.newaxis]
            fmt='.2f'
        else:
            fmt='d'
        
        # PLOT
        fig, ax = plt.subplots(figsize=(10,10))
        plt.imshow(fusion, cmap=cmap, aspect='equal')
        
        # Add title and axis labels 
        plt.title(title) 
        plt.ylabel('TRUE') 
        plt.xlabel('PRED')
        
        # Add appropriate axis scales
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        #ax.set_ylim(len(fusion), -.5,.5) ## <-- This was messing up the plots!
        
        # Text formatting
        fmt = '.2f' if normalize else 'd'
        # Add labels to each cell
        thresh = fusion.max() / 2.
        # iterate thru matrix and append labels  
        for i, j in itertools.product(range(fusion.shape[0]), range(fusion.shape[1])):
            plt.text(j, i, format(fusion[i, j], fmt),
                    horizontalalignment='center',
                    color='white' if fusion[i, j] > thresh else 'black',
                    size=14, weight='bold')
        
        # Add a legend
        plt.colorbar()
        plt.show() 
        return fusion, fig

    def get_scores(self):
        train_scores = self.model.evaluate(self.X_train, self.y_train, verbose=2)
        test_scores = self.model.evaluate(self.X_test, self.y_test, verbose=2)
        train_acc = np.round(train_scores[1], 2)
        train_loss = np.round(train_scores[0], 2)
        test_acc = np.round(test_scores[1], 2)
        test_loss = np.round(test_scores[0], 2)
        scores = {"train_acc": train_acc, "train_loss": train_loss, 
                "test_acc": test_acc, "test_loss": test_loss}
        print(f"\n{scores}")
        return scores

    def roc_plots(self):
        """Calculates ROC_AUC score and plots Receiver Operator Characteristics (ROC)

        Arguments:
            X {feature set} -- typically X_test
            y {labels} -- typically y_test
            model {classifier} -- the model name for which you are calculting roc score

        Returns:
            roc -- roc_auc_score (via sklearn)
        """
        y_true = self.y_test.flatten()
        y_hat = self.model.predict(self.X_test)

        fpr, tpr, thresholds = roc_curve(y_true, y_hat) 

        # Threshold Cutoff for predictions
        crossover_index = np.min(np.where(1.-fpr <= tpr))
        crossover_cutoff = thresholds[crossover_index]
        crossover_specificity = 1.-fpr[crossover_index]
        roc = roc_auc_score(y_true,y_hat)
        print(f"ROC AUC SCORE: {roc}")

        fig,axes=plt.subplots(ncols=2, figsize=(15,6))
        axes = axes.flatten()

        ax=axes[0]
        ax.plot(thresholds, 1.-fpr)
        ax.plot(thresholds, tpr)
        ax.set_title("Crossover at {0:.2f}, Specificity {1:.2f}".format(crossover_cutoff, crossover_specificity))

        ax=axes[1]
        ax.plot(fpr, tpr)
        ax.set_title("ROC area under curve: {0:.2f}".format(roc_auc_score(y_true, y_hat)))
        fig.show()
        
        return roc, fig

    def onehot_preds(self):
        self.y_onehot = pd.get_dummies(self.y_test.ravel(), prefix="lab")
        y_scores = self.model.predict(self.X_test)
        self.preds = np.concatenate([np.round(1-y_scores), np.round(y_scores)], axis=1)
        return self.y_onehot, self.preds
    
    
    def make_roc_curve(self, show=True):
        
        fig = go.Figure()
        fig.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)

        for i in range(self.preds.shape[1]):
            y_true = self.y_onehot.iloc[:, i]
            y_score = self.preds[:, i]

            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_score = roc_auc_score(y_true, y_score)

            name = f"{self.y_onehot.columns[i]} (AUC={auc_score:.2f})"
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode="lines"))

        fig.update_layout(
            title_text="ROC-AUC",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain="domain"),
            width=700,
            height=500,
            paper_bgcolor="#242a44",
            plot_bgcolor="#242a44",
            font={"color": "#ffffff"},
        )
        if show is True:
            fig.show()
        return fig

    def make_pr_curve(self, show=True):

        fig = go.Figure()
        fig.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=1, y1=0)

        for i in range(self.preds.shape[1]):
            y_true = self.y_onehot.iloc[:, i]
            y_score = self.preds[:, i]

            precision, recall, _ = precision_recall_curve(y_true, y_score)
            auc_score = average_precision_score(y_true, y_score)

            name = f"{self.y_onehot.columns[i]} (AP={auc_score:.2f})"
            fig.add_trace(go.Scatter(x=recall, y=precision, name=name, mode="lines"))

        fig.update_layout(
            title_text="Precision-Recall",
            xaxis_title="Recall",
            yaxis_title="Precision",
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain="domain"),
            width=800,
            height=500,
            paper_bgcolor="#242a44",
            plot_bgcolor="#242a44",
            font={"color": "#ffffff"},
        )
        if show is True:
            fig.show()
        return fig




    def keras_acc_plot(self, acc_train, acc_test):
        n_epochs = list(range(len(acc_train)))
        data = [
            go.Scatter(
                x=n_epochs,
                y=acc_train,
                name="Training Accuracy",
                marker=dict(color="#119dff"),
            ),
            go.Scatter(
                x=n_epochs, y=acc_test, name="Test Accuracy", marker=dict(color="#66c2a5")
            ),
        ]
        layout = go.Layout(
            title="Accuracy",
            xaxis={"title": "n_epochs"},
            yaxis={"title": "score"},
            width=700,
            height=500,
            paper_bgcolor="#242a44",
            plot_bgcolor="#242a44",
            font={"color": "#ffffff"},
        )
        fig = go.Figure(data=data, layout=layout)
        return fig


    def keras_loss_plot(self, loss_train, loss_test):
        n_epochs = list(range(len(loss_train)))
        data = [
            go.Scatter(
                x=n_epochs, y=loss_train, name="Training Loss", marker=dict(color="#119dff")
            ),
            go.Scatter(
                x=n_epochs, y=loss_test, name="Test Loss", marker=dict(color="#66c2a5")
            ),
        ]
        layout = go.Layout(
            title="Loss",
            xaxis={"title": "n_epochs"},
            yaxis={"title": "score"},
            width=700,
            height=500,
            paper_bgcolor="#242a44",
            plot_bgcolor="#242a44",
            font={"color": "#ffffff"},
        )
        fig = go.Figure(data=data, layout=layout)
        return fig


    def keras_plots(self, show=True):
        acc_train, acc_test = self.history["accuracy"], self.history["val_accuracy"]
        loss_train, loss_test = self.history["loss"], self.history["val_loss"]
        keras_acc = self.keras_acc_plot(acc_train, acc_test)
        keras_loss = self.keras_loss_plot(loss_train, loss_test)
        if show is True:
            keras_acc.show()
            keras_loss.show()
        return keras_acc, keras_loss

    def make_predictions(self):
        self.y_onehot, self.preds = self.onehot_preds()
        self.y_pred = self.preds[:, 1]
        return self.y_onehot, self.y_pred, self.preds

    def draw_plots(self):
        keras_acc, keras_loss = self.keras_plots()
        #self.y_onehot, self.y_pred, self.preds = self.make_predictions()
        roc_fig = self.make_roc_curve()
        pr_fig = self.make_pr_curve()
        matrix = confusion_matrix(self.y_test, self.y_pred)
        cm = self.fusion_matrix(matrix)
        print(f"\n FNFP: {matrix}")
        self.plots = {
            "keras_acc": keras_acc, "keras_loss": keras_loss, 
            "roc_fig": roc_fig, "pr_fig": pr_fig, "cm": cm}
        return self.plots


    def compute_scores(self):
        report = classification_report(self.y_test, self.y_pred, labels=[0,1], 
                                    target_names=['aligned', 'misaligned'])
        roc_auc = roc_auc_score(self.y_test, self.y_pred)
        acc_loss = self.get_scores()
        print(f"\n CLASSIFICATION REPORT: \n{report}")
        print(f"\n ACC/LOSS: {acc_loss}")
        print(f"\n ROC_AUC: {roc_auc}")
        self.scores = {"roc_auc": roc_auc,  "acc_loss": acc_loss, "report": report}
        return self.scores

    def track_fnfp(self):
        if self.test_idx is None:
            print("Test index not found")
            return
        conf_idx = np.where(self.y_pred != self.test_idx.values)
        pred_proba = np.asarray(self.model.predict(self.X_test).flatten(), 'float32')
        conf_proba = pred_proba[conf_idx]
        fn_idx = self.test_idx[conf_idx].loc[self.test_idx == 1].index
        fp_idx = self.test_idx.iloc[conf_idx].loc[self.test_idx == 0].index
        print(f"\n False Negatives: {fn_idx}")
        self.fnfp = {"pred_proba": pred_proba, "conf_idx": conf_idx, 
            "conf_proba": conf_proba, "fn_idx": fn_idx, "fp_idx": fp_idx}
        return self.fnfp

    def evaluate_results(self):
        predictions = self.make_predictions()
        plots = self.draw_plots()
        scores = self.compute_scores()
        fnfp = self.track_fnfp()
        self.results = {"predictions": predictions, 
                        "plots": plots, "scores": scores, "fnfp": fnfp} 
        return self.results



if __name__ == '__main__':
    df = pd.read_csv(f'{SUBFOLDER}/detection_cleaned.csv', index_col='index')
    df = df.drop(['category', 'ra_targ', 'dec_targ', 'imgname'], axis=1, inplace=False)
    df_scl, pt_transform = apply_power_transform(df, cols=['n_exposures', 'rms_ra', 'rms_dec', 'nmatches', 'point', 'segment', 'gaia_sources'])
    y = df['label']
    X = df.drop('label', axis=1, inplace=False)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, shuffle=True, stratify=y_train)

    image_sets = [X_train, X_test, X_val]
    train, test, val = make_image_sets(*image_sets, w=SIZE, h=SIZE, d=DEPTH, exp=None)
    index, X_tr, y_tr, X_ts, y_ts, X_vl, y_vl = training_img_aug(train, test, val)
    XTR, YTR, XTS, YTS, XVL, YVL = make_ensembles(X_tr, X_ts, X_vl, X_train_1, X_test,
                                              X_val, y_tr, y_ts, y_vl)
    ens = Builder(XTR, YTR, XTS, YTS, batch_size=32, epochs=60, lr=1e-4, 
                decay=[100000, 0.96], early_stopping=None, verbose=2, 
                ensemble=True)
    ens.build_ensemble(lr_sched=True)
    ens.fit_generator()
    ens_model = ens.model

    save_model(ens_model, name='ensemble4d', weights=True)

    ens_history = ens.history
    test_idx = index[1]
    com = Compute(ens_model, ens_history, XTR, YTR, XTS, YTS, test_idx)
    com.y_onehot, com.y_pred, com.preds = com.make_predictions()
    com.plots = com.draw_plots()
    com.scores = com.compute_scores()
    com.test_idx = y_test
    com.fnfp = com.track_fnfp()
    predictions = {"y_onehot": com.y_onehot, "preds": com.preds,
                "y_pred": com.y_pred}
    com.results = {"history": ens_history.history, "predictions": predictions,  "plots": com.plots, "scores": com.scores, 
                "fnfp": com.fnfp, "test_idx": com.test_idx}
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
    save_to_pickle(eval.results, res_path=f'{HOME}/results/ensemble/validation')
