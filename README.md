🌸 Iris Flower Species Classifier

View the Live Deployed Application Here [https://irisflowerprediction7.streamlit.app/]
1. Project Overview
This project is a simple yet classic machine learning application that demonstrates a fundamental classification task. The web app, built with Streamlit, allows users to input the sepal and petal measurements of an Iris flower and instantly receive a prediction of its species (Setosa, Versicolor, or Virginica).

The entire application is self-contained in a single Python script. It loads the famous Iris dataset directly from scikit-learn, trains a Logistic Regression model on the fly, and caches the trained model for immediate predictions.

2. Tech Stack & Libraries
Language: Python

Web Framework: Streamlit

Machine Learning: Scikit-learn (sklearn)

Data Handling: Pandas

3. Key Features
Interactive UI: Clean and simple user interface with number inputs for all four flower features.

On-the-Fly Model Training: The classification model is trained automatically when the app first launches.

Cached Model: Uses Streamlit's @st.cache_resource to ensure the model is trained only once, providing a fast user experience for subsequent predictions.

Visual Feedback: Displays an image of the predicted flower species for a more engaging user experience.

4. Dataset
This project uses the classic Iris dataset, which is included with the scikit-learn library. This removes the need for any external CSV files, making the application fully self-contained. The dataset includes 150 samples from three species of Iris flowers.

5. Setup & How to Run Locally
To run this project on your local machine, please follow these steps:

Clone the Repository:

git clone [https://github.com/your-username/iris-classifier-app.git](https://github.com/your-username/iris-classifier-app.git)
cd iris-classifier-app

Install Dependencies:
Make sure you have Python 3.8+ installed. Then, install the required libraries from the requirements.txt file.

pip install -r requirements.txt

Run the Streamlit App:
Execute the following command in your terminal:

streamlit run app.py

The application will open in a new tab in your web browser.
