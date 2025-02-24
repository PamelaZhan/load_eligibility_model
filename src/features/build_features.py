import pandas as pd

# create dummy features
def create_indicator_dummy_vars(df):
    # 2 bedroom and 2 bathroom properties are especially popular for investors. Create indicator variable for properties with 2 beds and 2 baths
    df['popular']= ((df.beds == 2)&(df.baths == 2)).astype(int)

    # It's important to consider the housing market recession in the United States. 
    # According to data from Zillow, the lowest housing prices were from 2010 to end of 2013.
    # Create an indicator feature recession
    df['recession'] = ((df.year_sold >= 2010) & (df.year_sold<=2013)).astype(int)
    
    # Create a property age feature
    df['property_age'] = df.year_sold - df.year_built
    # Remove rows where property_age is less than 0
    df = df[df.property_age >= 0]

    # Create dummy variables for 'property_type'
    df = pd.get_dummies(df, columns=['property_type'], dtype=int)

    # store the processed dataset in data/processed
    df.to_csv('data/processed/Processed_real_estate.csv', index=None)

    # Separate the input features and target variable
    x = df.drop('price', axis=1)
    y = df['price']

    return x, y