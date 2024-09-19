import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler

# Load the model and scaler
try:
    with open('diabetes.sav', 'rb') as file:
        loaded_model, loaded_scaler = pickle.load(file)
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'diabetes1.sav' is in the correct location.")
    loaded_model = None
    loaded_scaler = None

def predict_diabetes(pregnancies, glucose, bloodPressure, skinThickness, insulin, bmi, diabetesPedigreeFunction, age, model, scaler):
    if model is None or scaler is None:
        return None

    try:
        preg = int(pregnancies)
        glc = int(glucose)
        bp = int(bloodPressure)
        skn = int(skinThickness)
        ins = int(insulin)
        bmi = float(bmi)
        dpf = float(diabetesPedigreeFunction)
        ag = int(age)
    except ValueError:
        st.error("Invalid input. Please enter numerical values for all fields.")
        return None

    input_data = np.array([preg, glc, bp, skn, ins, bmi, dpf, ag]).reshape(1, -1)

    # Apply scaling before prediction
    scaled_input_data = scaler.transform(input_data)  

    prediction = model.predict(scaled_input_data)[0]
    return prediction

def main():
    st.title('Diabetes Prediction')

    if loaded_model is None or loaded_scaler is None:
        return  # Stop execution if model or scaler loading failed

    col1, col2 = st.columns(2)
    # Input fields
    with col1:
        pregnancies = st.text_input('Enter number of pregnancies')
        glucose = st.text_input('Enter glucose level')
        bloodPressure = st.text_input('Enter blood pressure value')
        skinThickness = st.text_input('Enter value of skin thickness')
    with col2:
        insulin = st.text_input('Enter insulin level')
        bmi = st.text_input('Enter BMI value')
        diabetesPedigreeFunction = st.text_input('Enter diabetes pedigree function value')
        age = st.text_input('Enter age')

    if st.button('Predict Status', type="primary", use_container_width=True):
        prediction = predict_diabetes(
            pregnancies, glucose, bloodPressure, skinThickness, insulin, bmi,
            diabetesPedigreeFunction, age, loaded_model, loaded_scaler
        )

        if prediction is not None:
            if prediction == 0:
                st.success('The person is not diabetic')
            else:
                st.error('The person is diabetic')

if __name__ == '__main__':
    main()