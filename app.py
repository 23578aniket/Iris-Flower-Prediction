import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import pickle


# --- Model Training ---
# This function will be cached, so the model is trained only once.
@st.cache_resource
def train_model():
    """
    Loads the Iris dataset and trains a Logistic Regression model.
    Returns the trained model and the class names.
    """
    # 1. Load the dataset
    # --- FIX: Ensure you are CALLING the function with parentheses () ---
    # The error occurs if you write "iris = load_iris" instead of "iris = load_iris()".
    iris = load_iris()
    X = iris.data
    y = iris.target
    class_names = iris.target_names

    # 2. Train a simple and effective model
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    return model, class_names


# --- Streamlit App UI ---

# 1. Page Configuration
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="centered"
)

st.title("🌸 Iris Flower Species Classifier")
st.markdown(
    "This app uses a Logistic Regression model to predict the species of an Iris flower based on its sepal and petal measurements.")

# 2. Load the cached model
model, class_names = train_model()

# 3. User Input Fields in a form
with st.form("prediction_form"):
    st.header("Enter Flower Measurements (cm)")

    # Create columns for a cleaner layout
    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.number_input("Sepal Length", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
        petal_length = st.number_input("Petal Length", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
    with col2:
        sepal_width = st.number_input("Sepal Width", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
        petal_width = st.number_input("Petal Width", min_value=0.0, max_value=10.0, value=0.2, step=0.1)

    # Submit button
    submitted = st.form_submit_button("Predict Species")

# 4. Prediction Logic
if submitted:
    # Prepare the input data for the model
    input_features = [[sepal_length, sepal_width, petal_length, petal_width]]

    # Get the prediction
    prediction_index = model.predict(input_features)[0]
    predicted_species = class_names[prediction_index]

    # Display the result
    st.success(f"**Predicted Species: {predicted_species.capitalize()}**")

    # Add an image corresponding to the prediction
    if predicted_species == 'setosa':
        st.image("https://upload.wikimedia.org/wikipedia/commons/5/56/Iris_setosa_2.jpg", caption="Iris Setosa")
    elif predicted_species == 'versicolor':
        st.image("https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg", caption="Iris Versicolor")
    else:  # virginica
        st.image("https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg", caption="Iris Virginica")

st.markdown("---")
st.info("This is a classic machine learning project used for demonstrating classification algorithms.")

