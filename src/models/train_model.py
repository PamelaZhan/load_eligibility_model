from sklearn.model_selection import train_test_split
# import decision tree model
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import pickle
from ..logging.logging import logging_decorator

@logging_decorator
# Function to train the model
def train_DTmodel(x, y):
    # Splitting the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=x.property_type_Bunglow)

    # create an instance of the Decision Tree class
    model = DecisionTreeRegressor(max_depth=3, max_features=10, random_state=567)

    # train the model
    dtmodel = model.fit(x_train, y_train)

    # make predictions using the train set
    ytrain_pred = dtmodel.predict(x_train)
    # evaluate the model
    train_mae = mean_absolute_error(ytrain_pred, y_train)
       
    # Save the trained model
    with open('models/DTmodel.pkl', 'wb') as f:
        pickle.dump(dtmodel, f)

    return dtmodel, x_test, y_test


