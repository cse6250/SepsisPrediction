import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies):
    fig1 = plt.figure(1)
    plt.plot(np.arange(len(train_losses)), train_losses, label='Training Loss')
    plt.plot(np.arange(len(valid_losses)), valid_losses, label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    fig1.savefig('./../../out/img/sepsis_prediction_long_term_loss_curve')

    fig2 = plt.figure(2)
    plt.plot(np.arange(len(train_accuracies)), train_accuracies, label='Training Accuracy')
    plt.plot(np.arange(len(valid_accuracies)), valid_accuracies, label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    fig2.savefig('./../../out/img/sepsis_prediction_long_term_accuracy_curve')


def plot_confusion_matrix(results, class_names):
    y_true, y_pred = zip(*results)
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title='Normalized Confusion Matrix',
           ylabel='True',
           xlabel='Predicted')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig('./../../out/img/sepsis_prediction_long_term_confusion_matrix')
