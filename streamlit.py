import pandas as pd
import pickle
import matplotlib.pyplot as plt
import streamlit as st
from lime import lime_tabular

# Set the page title and description
st.title("Loan Eligibility Predictor")
st.write("""
This app predicts whether a loan applicant is eligible for a loan 
based on various personal and financial characteristics.
""")

# Load the pre-trained model
with open("models/LRmodel.pkl", "rb") as lr_pickle:
    lr_model = pickle.load(lr_pickle)

# Load the pre-fit scaler 
with open('models/scaler.pkl', 'rb') as f:
    scaler=pickle.load(f)

# create explainer and plot the importance
def plot_explainer(columns, input_data, lr_model):
    # Load scaled train data
    with open("models/xtrain.pkl", "rb") as file:
        x_train = pickle.load(file)

    # create explainer object
    LIMEexplainer = lime_tabular.LimeTabularExplainer(
            training_data=x_train,
            class_names=["Not_Eligible", "Eligible"],
            feature_names=columns,
            mode='classification'
    )

    # Generate explanation instance
    exp = LIMEexplainer.explain_instance(
        data_row=input_data[0],                 
        predict_fn=lr_model.predict_proba,           # Model's prediction function
        num_features=20                      # Number of features to include in explanation
    )    

    # Convert explanation to a matplotlib figure
    fig = exp.as_pyplot_figure()  

    # Get feature importance values from the explanation
    importances = [x[1] for x in exp.as_list()]  
    # reverse the order for plot
    importances.reverse()

    # Annotate each bar with its corresponding importance value
    for i, importance in enumerate(importances, start=0):
        plt.text(
            importance,  # x-coordinate of the bar (importance value)
            i,  # y-coordinate (corresponding bar)
            f'{importance:.4f}',  # Display importance value 
            ha='center',  # Align text horizontally 
            va='center',  # Align text vertically 
            fontsize=10,  # Font size for the annotation
            color='black'  # Text color
        )
    # return the plot
    return fig, exp.as_list() 

# Prepare the form to collect user inputs
with st.form("user_inputs"):
    st.subheader("Loan Applicant Details")
    col1, col2 = st.columns(2)

    with col1:
        # Gender input
        Gender = st.selectbox("Gender", options=["Male", "Female"])
        
        # Marital Status
        Married = st.selectbox("Marital Status", options=["Yes", "No"])
        
        # Dependents
        Dependents = st.selectbox("Number of Dependents", 
                                options=["0", "1", "2", "3+"])
        
        # Education
        Education = st.selectbox("Education Level", 
                                options=["Graduate", "Not Graduate"])
        
        # Self Employment
        Self_Employed = st.selectbox("Self Employed", options=["Yes", "No"])
    
    with col2:
        # Applicant Income
        ApplicantIncome = st.number_input("Applicant Monthly Income", 
                                        min_value=0, value=5500,
                                        step=1000)
        
        # Coapplicant Income
        CoapplicantIncome = st.number_input("Coapplicant Monthly Income", 
                                            min_value=0, value=1600,
                                            step=1000)
        
        # Loan Amount
        LoanAmount = st.number_input("Loan Amount (*1000 dollars)", 
                                    min_value=0, value=145,
                                    step=1)
        
        # Loan Amount Term
        Loan_Amount_Term = st.selectbox("Loan Amount Term (Months)", 
                                        options=["360", "240", "180", "120", "60"])
        
        # Credit History
        Credit_History = st.selectbox("Whether applicant has a Credit History", 
                                    options=["Yes", "No"])
        
        # Property Area
        Property_Area = st.selectbox("Property Area", 
                                 options=["Urban", "Semiurban", "Rural"])
    
    # Submit button
    submitted = st.form_submit_button("Predict Loan Eligibility")

# input data features
features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
    'Loan_Amount_Term', 'Credit_History', 'Gender_Female', 'Gender_Male',
    'Married_No', 'Married_Yes', 'Dependents_0.0', 'Dependents_1.0',
    'Dependents_2.0', 'Dependents_3.0', 'Education_Graduate',
    'Education_Not_Graduate', 'Self_Employed_No', 'Self_Employed_Yes',
    'Property_Area_Rural', 'Property_Area_Semiurban',
    'Property_Area_Urban']

# Handle the dummy variables to pass to the model
if submitted:
    # get gender value
    Gender_Male = 1 if Gender == "Male" else 0
    Gender_Female = 1 if Gender == "Female" else 0
    # get marriage value
    Married_Yes = 1 if Married == "Yes" else 0
    Married_No = 1 if Married == "No" else 0

    # Handle dependents
    Dependents_0 = 1 if Dependents == "0" else 0
    Dependents_1 = 1 if Dependents == "1" else 0
    Dependents_2 = 1 if Dependents == "2" else 0
    Dependents_3 = 1 if Dependents == "3+" else 0
    # convert Education to dummy variables
    Education_Graduate = 1 if Education == "Graduate" else 0
    Education_Not_Graduate = 1 if Education == "Not Graduate" else 0
    # convert self_employed to dummy variables
    Self_Employed_Yes = 1 if Self_Employed == "Yes" else 0
    Self_Employed_No = 1 if Self_Employed == "No" else 0
    # convert property_area to dummy variables
    Property_Area_Rural = 1 if Property_Area == "Rural" else 0
    Property_Area_Semiurban = 1 if Property_Area == "Semiurban" else 0
    Property_Area_Urban = 1 if Property_Area == "Urban" else 0
    # convert credit_history to integer
    Credit_History = 1 if Credit_History == "Yes" else 0
    # Convert Loan Amount Term integer
    Loan_Amount_Term = int(Loan_Amount_Term)


    # Prepare the input for prediction, the same order as it was trained
    prediction_input = [[ApplicantIncome, CoapplicantIncome, LoanAmount,
        Loan_Amount_Term, Credit_History, Gender_Female, Gender_Male,
        Married_No, Married_Yes, Dependents_0, Dependents_1,
        Dependents_2, Dependents_3, Education_Graduate,
        Education_Not_Graduate, Self_Employed_No, Self_Employed_Yes,
        Property_Area_Rural, Property_Area_Semiurban, Property_Area_Urban
    ]]
   

    # Scale the input
    input_scaled = scaler.transform(prediction_input)
    # Make prediction
    new_prediction = lr_model.predict(input_scaled)

    # Display prediction result
    st.subheader("Prediction Result:")
    if new_prediction[0] == 'Y':
        st.write("You are eligible for the loan!")
    else:
        st.write("Sorry, you are not eligible for the loan.")

    # call the function to get explaination plot and importance values
    fig, exp=plot_explainer(features, input_scaled, lr_model)
    # Display explanation in Streamlit
    st.subheader("LIME Explanation for Prediction")
    st.pyplot(fig)
    st.subheader("Feature Contributions:")
    st.table(pd.DataFrame(exp, columns=["Feature", "Importance"]))

# display Coefficient image and values
st.subheader("Feature Coefficient")
st.image("Coefficients.png")


# Get coefficients
coefficients = lr_model.coef_[0]

# Create DataFrame for better visualization
feature_coefficient = pd.DataFrame({
    'Feature': features,
    'Coefficient': coefficients,
})

st.write(feature_coefficient)
