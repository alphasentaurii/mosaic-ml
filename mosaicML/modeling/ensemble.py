# STANDARD libraries
import os
import sys
import pandas as pd
import numpy as np
import time
import datetime as dt
from tensorflow.keras.optimizers import Adam
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('seaborn-bright')
font_dict={'family':'"Titillium Web", monospace','size':16}
mpl.rc('font',**font_dict)
import plotly.graph_objects as go
import itertools
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras import layers, optimizers, callbacks
from tensorflow.keras.layers import Dense, Input, concatenate
from keras.layers import Dense

from sklearn.metrics import (
    roc_curve, 
    roc_auc_score, 
    precision_recall_curve, 
    average_precision_score,
    classification_report,
    confusion_matrix
    )
# mosaicML modules
from preprocessing.augment import augment_data, augment_image


DIM = 3
CH = 3
WIDTH = 128
HEIGHT = 128
DEPTH = DIM * CH
SHAPE = (DIM, WIDTH, HEIGHT, CH)
TF_CPP_MIN_LOG_LEVEL = 2


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


def print_timestamp(ts, name, value):
    if value == 0:
        info = "STARTED"
    elif value == 1:
        info = "FINISHED"
    else:
        info = ""
    timestring = dt.datetime.fromtimestamp(ts).strftime("%m/%d/%Y - %H:%M:%S")
    print(f"{info} [{name}]: {timestring}")


class Builder:
    def __init__(self, X_train, y_train, X_test, y_test, batch_size=32, epochs=60, 
                 lr=1e-4, decay=[100000, 0.96], early_stopping=None, verbose=2, 
                 ensemble=True):
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
        self.name = None


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
        self.model = Sequential()
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
        if self.name is None:
            self.name = "ensemble4d"
        self.mlp = self.build_mlp(input_shape=self.X_train[0].shape[1])
        self.cnn = self.build_cnn(input_shape=self.X_train[1].shape[1:])
        combinedInput = concatenate([self.mlp.output, self.cnn.output])
        x = Dense(9, activation="leaky_relu", name="combined_input")(combinedInput)
        x = Dense(1, activation="sigmoid", name="ensemble_output")(x)
        self.model = Model(inputs=[self.mlp.input, self.cnn.input], outputs=x, name=self.name)
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


class Remodel(Builder):
    def __init__(self, model_path, X_train, y_train, X_test, y_test, batch_size=32, epochs=60, 
                 lr=1e-4, decay=[100000, 0.96], early_stopping=None, verbose=2, 
                 ensemble=True):
        super().__init__(X_train, y_train, X_test, y_test, batch_size, epochs, 
                 lr, decay, early_stopping, verbose, ensemble)
        self.model_path = model_path
        self.model = None

    def load_saved_model(self):
        model = load_model(self.model_path)
        self.model = Model(inputs=model.inputs, outputs=model.outputs)
        self.model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=self.decay_learning_rate()), metrics=["accuracy"])
        return self


class Computer:

    def __init__(self, model_name, compute, classes): #="ENSEMBLE_SVM", ="test"
        self.model_name = model_name
        self.compute = compute
        self.classes = classes
        self.res_path = None
        self.model = None
        self.history = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.test_idx = None
        self.y_onehot = None
        self.y_scores = None
        self.y_pred = None
        self.cmx = None
        self.fnfp = None
        self.report = None
        self.roc_auc = None
        self.acc_loss = None
        self.acc_fig = None
        self.loss_fig = None
        self.roc_fig = None
        self.pr_fig = None
        self.cm_fig = None
        self.cmx_norm = None

    def inputs(self, model, history, X_train, y_train, X_test, y_test, test_idx):
        self.model = model
        self.history = history.history
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.test_idx = test_idx
        return self
    
    def store_results(self):
        # model_name, compute, training_date, etc
        outputs = {
            "predictions" : {
                "y_onehot": self.y_onehot, 
                "y_scores": self.y_scores, 
                "y_pred": self.y_pred,
                "cmx": self.cmx,
                "cmx_norm": self.cmx_norm,
                "fnfp": self.fnfp
            },
            "scores" : {
                "roc_auc": self.roc_auc,  
                "acc_loss": self.acc_loss, 
                "report": self.report
                }
        }
        if self.compute == "val":
            outputs["plots"] = {
                "roc_fig": self.roc_fig, 
                "pr_fig": self.pr_fig, 
                "cm_fig": self.cm_fig,
                }
        else:
            outputs["plots"] = {
                "keras_acc": self.acc_fig, 
                "keras_loss": self.loss_fig, 
                "roc_fig": self.roc_fig, 
                "pr_fig": self.pr_fig, 
                "cm_fig": self.cm_fig,
                }
        return outputs
    
    def download(self):
        outputs = self.store_results()
        if self.res_path is None:
            self.res_path = os.path.join("./results", self.model_name, self.compute)
        os.makedirs(f"{self.res_path}", exist_ok=True)
        for k, v in outputs.items():
            key = f"{self.res_path}/{k}"
            with open(key, "wb") as file_pi:
                pickle.dump(v, file_pi)
        print(f"Results saved to: {self.res_path}")

        out = sys.stdout
        with open(f"{self.res_path}/metrics_report.txt", "w") as f:
            sys.stdout = f
            print(self.report)
            sys.stdout = out
    

    def upload(self):
        if self.res_path is None:
            self.res_path = os.path.join("./results", self.model_name, self.compute)
        if os.path.exists(self.res_path):
            res = {}
            for r in os.listdir(self.res_path):
                key = os.path.join(self.res_path, r)
                with open(key, "rb") as py_fi:
                    res[r] = pickle.load(py_fi)
            self.outputs(res)
        else:
            print(f"No results found @ {self.res_path}")
        return self


    def outputs(self, res):
        self.y_onehot = res['predictions']['y_onehot']
        self.y_scores = res['predictions']['y_scores']
        self.y_pred = res['predictions']['y_pred']
        self.cmx = res['predictions']['cmx']
        self.cmx_norm = res['predictions']['cmx_norm']
        self.fnfp = res['predictions']['fnfp']
        #self.report = res['scores']['report']
        self.roc_auc = res['scores']['roc_auc']
        self.acc_loss = res['scores']['acc_loss']
        self.roc_fig = res['plots']['roc_fig']
        self.pr_fig = res['plots']['pr_fig']
        self.cm_fig = res['plots']['cm_fig']
        if self.compute == "test":
            self.acc_fig = res['plots']['acc_fig']
            self.loss_fig = res['plots']['loss_fig']
        with open(f"{self.res_path}/metrics_report.txt", "r") as f:
            self.report = f.read()
        return self
    


    """ MODEL PERFORMANCE METRICS """
    def calculate_results(self, show_summary=True):
        self.y_onehot = self.onehot_y()
        self.y_scores = self.score_y()
        self.y_pred = self.y_scores[:, 1]
        self.report = classification_report(self.y_test, self.y_pred, labels=[0,1], 
                                    target_names=self.classes)
        self.roc_auc = roc_auc_score(self.y_test, self.y_pred)
        self.acc_loss = self.acc_loss_scores()
        self.cmx = confusion_matrix(self.y_test, self.y_pred)
        self.fnfp = self.track_fnfp()

        if show_summary:
            self.print_summary()

        return self

    def onehot_y(self):
        self.y_onehot = pd.get_dummies(self.y_test.ravel(), prefix="lab")
        return self.y_onehot
    
    def score_y(self):
        y_scores = self.model.predict(self.X_test)
        self.y_scores = np.concatenate([np.round(1-y_scores), np.round(y_scores)], axis=1)
        return self.y_scores

    def acc_loss_scores(self):
        train_scores = self.model.evaluate(self.X_train, self.y_train, verbose=2)
        test_scores = self.model.evaluate(self.X_test, self.y_test, verbose=2)
        train_acc = np.round(train_scores[1], 2)
        train_loss = np.round(train_scores[0], 2)
        test_acc = np.round(test_scores[1], 2)
        test_loss = np.round(test_scores[0], 2)
        self.acc_loss = {
            "train_acc": train_acc, 
            "train_loss": train_loss, 
            "test_acc": test_acc, 
            "test_loss": test_loss
            }
        return self.acc_loss

    def track_fnfp(self):
        if self.test_idx is None:
            print("Test index not found")
            return
        try:
            conf_idx = np.where(self.y_pred != self.test_idx.values)
        except AttributeError as e:
            print(f"Test/Val Index should be a pandas series, not {type(self.test_idx)}")
            print(e)
            return
        pred_proba = np.asarray(self.model.predict(self.X_test).flatten(), 'float32')
        conf_proba = pred_proba[conf_idx]
        fn_idx = self.test_idx.iloc[conf_idx].loc[self.test_idx == 1].index
        fp_idx = self.test_idx.iloc[conf_idx].loc[self.test_idx == 0].index
        self.fnfp = {
            "pred_proba": pred_proba,
            "conf_idx": conf_idx, 
            "conf_proba": conf_proba, 
            "fn_idx": fn_idx, 
            "fp_idx": fp_idx
            }
        return self.fnfp

    def print_summary(self):
        print(f"\n CLASSIFICATION REPORT: \n{self.report}")
        print(f"\n ACC/LOSS: {self.acc_loss}")
        print(f"\n ROC_AUC: {self.roc_auc}")
        print(f"\nFalse -/+\n{self.cmx}")
        print(f"\nFalse Negatives Index\n{self.fnfp['fn_idx']}\n")



    """ PLOTS """

    def draw_plots(self):
        self.acc_fig = self.keras_acc_plot()
        self.loss_fig = self.keras_loss_plot()
        self.roc_fig = self.make_roc_curve()
        self.pr_fig = self.make_pr_curve()
        self.cm_fig, self.cmx_norm = self.fusion_matrix(self.cm, self.classes)
        return self

    def fusion_matrix(self, cm, classes, normalize=True, cmap='Blues', show=True): 
        """
        FUSION MATRIX!
        -------------
        
        matrix: can pass in matrix or a tuple (ytrue,ypred) to create on the fly 
        classes: class names for target variables
        """
        # make matrix if tuple passed to matrix:
        if isinstance(cm, tuple):
            y_true = cm[0].copy()
            y_pred = cm[1].copy()
            
            if y_true.ndim>1:
                y_true = y_true.argmax(axis=1)
            if y_pred.ndim>1:
                y_pred = y_pred.argmax(axis=1)
            fusion = metrics.confusion_matrix(y_true, y_pred)
        else:
            fusion = cm
        # INTEGER LABELS
        if classes is None:
            classes=list(range(len(fusion)))

        if normalize:
            fusion = fusion.astype('float') / fusion.sum(axis=1)[:, np.newaxis]
            fmt='.2f'
        else:
            fmt='d'
        
        # PLOT
        fig, _ = plt.subplots(figsize=(10,10))
        plt.imshow(fusion, cmap=cmap, aspect='equal')
        
        # Add title and axis labels 
        plt.title("Confusion Matrix") 
        plt.ylabel('TRUE') 
        plt.xlabel('PRED')
        
        # Add appropriate axis scales
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
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
        if show:
            fig.show()

        return fig, fusion

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
    
    def make_roc_curve(self, show=True):
        
        fig = go.Figure()
        fig.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)

        for i in range(self.y_scores.shape[1]):
            y_true = self.y_onehot.iloc[:, i]
            y_score = self.y_scores[:, i]

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
            width=800,
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

        for i in range(self.y_scores.shape[1]):
            y_true = self.y_onehot.iloc[:, i]
            y_score = self.y_scores[:, i]

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

    def keras_acc_plot(self, show=True):
        acc_train = self.history["accuracy"]
        acc_test = self.history["val_accuracy"]
        n_epochs = list(range(len(acc_train)))
        data = [
            go.Scatter(
                x=n_epochs,
                y=acc_train,
                name="Training Accuracy",
                marker=dict(color="#119dff"),
            ),
            go.Scatter(
                x=n_epochs, 
                y=acc_test, 
                name="Test Accuracy", 
                marker=dict(color="#66c2a5")
            ),
        ]
        layout = go.Layout(
            title="Accuracy",
            xaxis={"title": "n_epochs"},
            yaxis={"title": "score"},
            width=800,
            height=500,
            paper_bgcolor="#242a44",
            plot_bgcolor="#242a44",
            font={"color": "#ffffff"},
        )
        fig = go.Figure(data=data, layout=layout)
        if show:
            fig.show()
        return fig

    def keras_loss_plot(self, show=True):
        loss_train = self.history["loss"]
        loss_test = self.history["val_loss"]
        n_epochs = list(range(len(loss_train)))
        
        data = [
            go.Scatter(
                x=n_epochs, 
                y=loss_train, 
                name="Training Loss", 
                marker=dict(color="#119dff")
            ),
            go.Scatter(
                x=n_epochs, 
                y=loss_test, 
                name="Test Loss", 
                marker=dict(color="#66c2a5")
            ),
        ]
        layout = go.Layout(
            title="Loss",
            xaxis={"title": "n_epochs"},
            yaxis={"title": "score"},
            width=800,
            height=500,
            paper_bgcolor="#242a44",
            plot_bgcolor="#242a44",
            font={"color": "#ffffff"},
        )
        fig = go.Figure(data=data, layout=layout)
        if show:
            fig.show()
        return fig


class ComputeTest(Computer):
    def __init__(self, model_name, classes, model, history, X_train, y_train, X_test, y_test, test_idx):
        super().__init__(model_name, "test", classes)
        self.inputs(model, history, X_train, y_train, X_test, y_test, test_idx)

class ComputeVal(Computer):
    def __init__(self, model_name, classes, model, X_test, y_test, X_val, y_val, val_idx):
        super().__init__(model_name, "val", classes)
        self.inputs(model, X_test, y_test, X_val, y_val, val_idx)


