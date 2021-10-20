
# EVALUATION

def keras_history(history, figsize=(10,4)):
    """
    side by side sublots of training val accuracy and loss (left and right respectively)
    """
    
    import matplotlib.pyplot as plt
    
    fig, axes =plt.subplots(ncols=2,figsize=(15,6))
    axes = axes.flatten()

    ax = axes[0]
    ax.plot(history.history['accuracy'])
    ax.plot(history.history['val_accuracy'])
    ax.set_title('Model Accuracy')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Epoch')
    ax.legend(['Train', 'Test'], loc='upper left')

    ax = axes[1]
    ax.plot(history.history['loss'])
    ax.plot(history.history['val_loss'])
    ax.set_title('Model Loss')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend(['Train', 'Test'], loc='upper left')
    fig.show()

def fusion_matrix(matrix, classes=['aligned', 'misaligned'], normalize=True, title='Confusion Matrix', cmap='Blues',
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

def get_scores(model, X_train, y_train, X_test, y_test):
    train_scores = model.evaluate(X_train, y_train, verbose=2)
    test_scores = model.evaluate(X_test, y_test, verbose=2)
    train_acc = np.round(train_scores[1], 2)
    train_loss = np.round(train_scores[0], 2)
    test_acc = np.round(test_scores[1], 2)
    test_loss = np.round(test_scores[0], 2)
    scores = {"train_acc": train_acc, "train_loss": train_loss, 
              "test_acc": test_acc, "test_loss": test_loss}
    print(f"\n{scores}")
    return scores

def roc_plots(X,y,model):
    """Calculates ROC_AUC score and plots Receiver Operator Characteristics (ROC)

    Arguments:
        X {feature set} -- typically X_test
        y {labels} -- typically y_test
        model {classifier} -- the model name for which you are calculting roc score

    Returns:
        roc -- roc_auc_score (via sklearn)
    """


    y_true = y.flatten()
    y_hat = model.predict(X)

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

def make_roc_curve(y_onehot, y_scores, show=True):
    
    fig = go.Figure()
    fig.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)

    for i in range(y_scores.shape[1]):
        y_true = y_onehot.iloc[:, i]
        y_score = y_scores[:, i]

        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_score = roc_auc_score(y_true, y_score)

        name = f"{y_onehot.columns[i]} (AUC={auc_score:.2f})"
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

def make_pr_curve(y_onehot, y_scores, show=True):

    fig = go.Figure()
    fig.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=1, y1=0)

    for i in range(y_scores.shape[1]):
        y_true = y_onehot.iloc[:, i]
        y_score = y_scores[:, i]

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        auc_score = average_precision_score(y_true, y_score)

        name = f"{y_onehot.columns[i]} (AP={auc_score:.2f})"
        fig.add_trace(go.Scatter(x=recall, y=precision, name=name, mode="lines"))

    fig.update_layout(
        title_text="Precision-Recall",
        xaxis_title="Recall",
        yaxis_title="Precision",
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

def onehot_preds(X, y, model):
    y = y.ravel()
    y_onehot = pd.get_dummies(y, prefix="lab")
    y_scores = model.predict(X)
    y_preds = np.concatenate([np.round(1-y_scores), np.round(y_scores)], axis=1)
    return y_onehot, y_scores, y_preds

# KERAS HISTORY


def keras_acc_plot(acc_train, acc_test):
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


def keras_loss_plot(loss_train, loss_test):
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


def keras_plots(history, show=True):
    acc_train, acc_test = history["accuracy"], history["val_accuracy"]
    loss_train, loss_test = history["loss"], history["val_loss"]
    keras_acc = keras_acc_plot(acc_train, acc_test)
    keras_loss = keras_loss_plot(loss_train, loss_test)
    if show is True:
        keras_acc.show()
        keras_loss.show()
    return keras_acc, keras_loss

#y_onehot, y_scores, y_preds= onehot_preds(X_vl, y_vl, cnn)
# roc_fig = make_roc_curve(y_onehot, y_scores)
# pr_fig = make_pr_curve(y_onehot, y_scores)