# load_eligibility_application
This app has been built using Streamlit and deployed with Streamlit community cloud

[Visit the app here](https://load-eligibility-model.streamlit.app/)

This application predicts whether someone is eligible for a loan based on the German Credit Risk dataset. The model aims to help users assess loan eligibility by leveraging machine learning predictions.

## Features
- User-friendly interface powered by Streamlit.
- Input form to enter details such as credit history, loan amount, income, and other relevant factors.
- Real-time prediction of load eligibility, Accuracy of 76% and above.
- Accessible via Streamlit Community Cloud.

## Dataset
The application is trained on the **German Credit data**, a dataset for evaluating creditworthiness. It includes features like:
- Gender
- Married: Marital status of the applicant
- Dependents: Number of dependants the applicant has
- Education: Graduate/Not Graduate
- Self_Employed: Whether self-employed Yes/No
- ApplicantIncome: Income of the applicant per month
- CoapplicantIncome: Income of the co-applicant per month
- LoanAmount: Loan amount requested in *1000 dollars
- Loan_Amount_Term: Term of the loan in months
- Credit_History: Whether applicant has a credit history

## Technologies Used
- **Streamlit**: For building the web application.
- **Scikit-learn**: For model training and evaluation.
- **Pandas** and **NumPy**: For data preprocessing and manipulation.
- **Matplotlib** and **Seaborn**: For exploratory data analysis and visualization (if applicable).
- **LIME**: Explain the prediction result.

## Model
The Logistic Regression model is trained using the German Credit Risk dataset. It applies preprocessing steps like encoding categorical variables and scaling numerical features. 


#### Thank you for using the Load Eligibility Application! Feel free to share your feedback.
