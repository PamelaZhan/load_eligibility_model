import pandas as pd
from ..logging.logging import logging_decorator

@logging_decorator
# create dummy features
def create_dummy(df):
    # Create dummy variables for all 'object' type variables except 'Loan_Status'
    df = pd.get_dummies(df, columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'],dtype=int)
    
    # store the processed dataset in data/processed
    df.to_csv('data/processed/processed_credit.csv', index=None)

    # Separate the input features and target variable
    x = df.drop('Loan_Approved', axis=1)
    y = df['Loan_Approved']

    return x, y