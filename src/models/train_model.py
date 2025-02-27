from sklearn.model_selection import train_test_split
# import Logistic Regression
from sklearn.linear_model import LogisticRegression
# scale the data using min-max scalar
from sklearn.preprocessing import MinMaxScaler
import pickle
from ..logging.logging import logging_decorator

@logging_decorator
# Function to train the model
def train_LRmodel(x, y):
    # Splitting the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

    # Scale the data using MinMaxScaler, transform data into 0--1
    scaler = MinMaxScaler()
    # fit-transform on train data
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # create an instance of the Logistic Regression and train it
    lrmodel = LogisticRegression().fit(x_train_scaled, y_train)

    # Save the scaler   
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    # Save the trained model
    with open('models/LRmodel.pkl', 'wb') as f:
        pickle.dump(lrmodel, f)

    return lrmodel, x_test_scaled, y_test


