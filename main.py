
from src.data.load_dataset import load_and_preprocess_data
from src.features.build_features import create_dummy
from src.visualization.visualize import plot_feature_importance
from src.models.train_model import train_LRmodel
from src.models.predict_model import evaluate_model
import pandas as pd

if __name__ == "__main__":
    # Load and preprocess the data
    data_path = "data/raw/credit.csv"
    df = load_and_preprocess_data(data_path)

    # Create dummy variables and separate features and target
    x, y = create_dummy(df)

    # Train the logistic regression model
    lrmodel, x_test_scaled, y_test = train_LRmodel(x, y)

    # Evaluate the model   
    accuracy, confusion_mat = evaluate_model(lrmodel, x_test_scaled, y_test)
    if accuracy>0.76:
        print("Successful! The Accuracy of modle is 76% and above.")        
        print(f"Accuracy: {accuracy}")
        print(f"Confusion Matrix:\n{confusion_mat}")
        
    else:
        print("The model is not good enough.")
 
