import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load('irishh.pkl')

st.title("Iris Flower Classifier")
st.write("Enter flower measurements to get the predicted species.")

# Input fields
sepal_length = st.number_input("Sepal Length", 0.0, 10.0, step=0.1)
sepal_width = st.number_input("Sepal Width", 0.0, 10.0, step=0.1)
petal_length = st.number_input("Petal Length", 0.0, 10.0, step=0.1)
petal_width = st.number_input("Petal Width", 0.0, 10.0, step=0.1)

if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Check input shape for safety
    st.write(f"Input shape: {input_data.shape}")

    prediction = model.predict(input_data)
    st.success(f"Predicted Class: {prediction[0]}")
