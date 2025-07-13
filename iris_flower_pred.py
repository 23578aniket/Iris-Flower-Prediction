import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = load_model('iris_keras_model.h5')

# Define LabelEncoder for decoding predictions
encoder = LabelEncoder()
encoder.classes_ = np.array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])

# Streamlit App
st.title("Iris Flower Prediction App")
st.write("Provide the features of the iris flower to predict its species:")

# Input fields for flower features
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)

# Prediction Button
if st.button("Predict"):
    # Prepare input data
    input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Predict using the loaded model
    prediction = model.predict(input_features)
    predicted_class = encoder.inverse_transform([np.argmax(prediction)])

    st.success(f"The predicted species is: {predicted_class[0]}")


