import streamlit as st
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the trained model
svm_model = joblib.load('svm_model.pkl')

# Title of the app
st.title('Mammographic Mass Severity Prediction')

# Input fields for the features
b_rads = st.selectbox('BI-RADS', [1, 2, 3, 4, 5])
age = st.slider('Age', 10, 100, 50)
shape = st.selectbox('Shape', ['Round', 'Oval', 'Lobular', 'Irregular'])
margin = st.selectbox('Margin', ['Circumscribed', 'Microlobulated', 'Obscured', 'Ill-defined', 'Spiculated'])
density = st.selectbox('Density', [1, 2, 3, 4])

# Button for prediction
if st.button('Predict'):
    # Convert shape and margin to numerical values
    shape_dict = {'Round': 1, 'Oval': 2, 'Lobular': 3, 'Irregular': 4}
    margin_dict = {'Circumscribed': 1, 'Microlobulated': 2, 'Obscured': 3, 'Ill-defined': 4, 'Spiculated': 5}
    
    shape = shape_dict[shape]
    margin = margin_dict[margin]
    
    # Create a DataFrame for the input
    input_data = pd.DataFrame([[b_rads, age, shape, margin, density]], 
                              columns=['BI-RADS', 'Age', 'Shape', 'Margin', 'Density'])
    
    # Make a prediction
    prediction = svm_model.predict(input_data)[0]
    
    # Display the result
    if prediction == 0:
        st.success('The mass is predicted to be Benign')
    else:
        st.error('The mass is predicted to be Malignant')
