
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
import numpy as np
from ..logging.logging import logging_decorator

@logging_decorator
def plot_correlation_heatmap(data):
    """
    Plot a correlation heatmap for the given data.
    
    Args:
        data (pandas.DataFrame): The input data.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr()*100, annot=True, fmt='.0f', cmap='RdBu_r')
    plt.title('Correlation Heatmap', fontsize=16)
    # Save the plot to a file
    plt.savefig('heatmap.png', dpi=300)
    # Show the plot
    plt.show()


@logging_decorator
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix'):
    """
    Plot the confusion matrix for the given true and predicted labels.
    
    Args:
        cm(confusion_matrix): confusion_matrix(y_true, y_pred).
        classes (list): List of class labels.
        normalize (bool, optional): Whether to normalize the confusion matrix. Default is False.
        title (str, optional): Title for the plot. Default is 'Confusion Matrix'.
    """ 
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title(title, fontsize=16)
    # Save the plot to a file
    plt.savefig('confusion_matrix.png', dpi=300)
    # Show the plot
    plt.show()