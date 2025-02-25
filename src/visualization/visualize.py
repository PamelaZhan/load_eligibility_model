
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
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
def plot_feature_importance(model, x):
    """
    Plot a bar chart showing the feature importances.
    
    Args:
        feature_names (list): List of feature names.
        feature_importances (list): List of feature importance values.
    """
    fig, ax = plt.subplots() # a single subplot
    ax = sns.barplot(x=model.feature_importances_, y=x.columns)
    plt.title("Feature importance chart")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    # Save the plot to a file
    fig.savefig("feature_importance.png")
    # Show the plot
    plt.show()

@logging_decorator
def plot_decision_tree(model):
    """
    Plot a decision tree for the model.
    
    Args:
        model: trained decision tree model.
    """
    fig, ax = plt.subplots()      
    # Plot the tree with feature names
    ax =tree.plot_tree(model, feature_names=model.feature_names_in_, filled = True)
    plt.tight_layout()
    # Save the plot to a file
    fig.savefig('tree.png', dpi=300)    
    # Show the plot
    plt.show()
