import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

import warnings
warnings.filterwarnings(action="ignore")

# Streamlit app
st.title("Heart Disease Prediction App")

# Load data
df = pd.read_csv("heart.csv")

# Display data
st.write("### Data Overview")
st.write(df.head())
st.write(f"Shape of data: {df.shape}")

# Prepare data for modeling
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a model (Logistic Regression used as an example)
model = LogisticRegression()
model.fit(X_train, y_train)

# User input section
st.write("### Enter Your Information to Predict Heart Disease")

def user_input_features():
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", (0, 1))  # 0: Female, 1: Male
    cp = st.selectbox("Chest Pain Type (0-3)", (0, 1, 2, 3))
    trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=200, value=120)
    chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = true; 0 = false)", (0, 1))
    restecg = st.selectbox("Resting Electrocardiographic Results (0-2)", (0, 1, 2))
    thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina (1 = yes; 0 = no)", (0, 1))
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0)
    slope = st.selectbox("Slope of the Peak Exercise ST Segment (0-2)", (0, 1, 2))
    ca = st.number_input("Number of Major Vessels (0-4)", min_value=0, max_value=4, value=0)
    thal = st.selectbox("Thalassemia (0 = normal; 1 = fixed defect; 2 = reversible defect)", (0, 1, 2))
    
    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

user_input = user_input_features()

# Display the user inputs
st.write("### User Input:")
st.write(user_input)

# Scale the user input
user_input_scaled = scaler.transform(user_input)

# Make a prediction
prediction = model.predict(user_input_scaled)
prediction_proba = model.predict_proba(user_input_scaled)

# Display the prediction
st.write("### Prediction:")
heart_disease = "Yes" if prediction[0] == 1 else "No"
st.write(f"Heart Disease: **{heart_disease}**")

# Display the prediction probabilities
st.write("### Prediction Probability:")
st.write(f"Probability of No Heart Disease: {np.round(prediction_proba[0][0], 2)}")
st.write(f"Probability of Heart Disease: {np.round(prediction_proba[0][1], 2)}")
