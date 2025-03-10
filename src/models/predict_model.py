# Import accuracy score
from sklearn.metrics import accuracy_score, confusion_matrix
from ..visualization.visualize import plot_confusion_matrix
from ..logging.logging import logging_decorator

@logging_decorator
# # Function to predict and evaluate
def evaluate_model(model, x_test_scaled, y_test):

    # Predict the loan eligibility on the testing set
    y_pred = model.predict(x_test_scaled)

    # Calculate the accuracy score
    accuracy = accuracy_score(y_pred, y_test)

    # Calculate the confusion matrix
    confusion_mat = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(confusion_mat, ['Eligible', 'Not Eligible'])
    return accuracy, confusion_mat
