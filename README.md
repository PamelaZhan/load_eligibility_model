# real_estate_price_application
This app has been built using Streamlit and deployed with Streamlit community cloud

[Visit the app here](https://real-estate-price-predictor-app.streamlit.app/)

This application predicts whether someone is eligible for a loan based on inputs derived from the German Credit Risk dataset. The model aims to help users assess loan eligibility by leveraging machine learning predictions.

## Features
- User-friendly interface powered by Streamlit.
- Input form to enter details such as credit history, loan amount, income, and other relevant factors.
- Real-time prediction of property price based on the trained model. Mean Absolute Error (MAE) < $70,000
- Accessible via Streamlit Community Cloud.

## Dataset
The application is trained on the **German Credit Risk dataset**, a dataset for evaluating creditworthiness. It includes features like:
- Age
- Job
- Housing status
- Credit amount
- Duration of credit
- Purpose of loan
- And other factors influencing credit risk.



## Technologies Used
- **Streamlit**: For building the web application.
- **Scikit-learn**: For model training and evaluation.
- **Pandas** and **NumPy**: For data preprocessing and manipulation.
- **Matplotlib** and **Seaborn**: For exploratory data analysis and visualization (if applicable).

## Model
The predictive model is trained using the German Credit Risk dataset. It applies preprocessing steps like encoding categorical variables and scaling numerical features. The Logitic Regression model is used.


#### Thank you for using the Load Eligibility Application! Feel free to share your feedback.
