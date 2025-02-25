# Import accuracy score
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix
from ..logging.logging import logging_decorator

@logging_decorator
# # Function to predict and evaluate
def evaluate_model(model, x_test, y_test):
    # Predict the property price on the testing set
    y_pred = model.predict(x_test)

    # evaluate the model
    mae = mean_absolute_error(y_pred, y_test)

    return mae