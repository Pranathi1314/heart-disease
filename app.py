import streamlit as st
import pandas as pd
import numpy as np
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
# st.write(df.info())
# st.write(f"Shape of data: {df.shape}")
# st.write(df.describe())

# Prepare data for modeling
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "Naive Bayes": GaussianNB(),
    "MLP Neural Network": MLPClassifier()
}

# Train and evaluate models
# st.write("### Model Performance")

for name, model in models.items():
    with st.spinner(f"Training {name}..."):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        # st.write(f"**{name}**")
        # st.write(f"Accuracy: {np.round(accuracy, 2)}")
        # st.write(classification_report(y_test, y_pred))

# Add user input section
st.write("### Enter Patient Details")

# Collect user inputs
age = st.number_input("Age", min_value=0, max_value=120, value=25)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])  # Adjust based on your dataset's chest pain types
trestbps = st.number_input("Resting Blood Pressure", min_value=0, max_value=300, value=120)
chol = st.number_input("Cholesterol", min_value=0, max_value=1000, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])  # Adjust based on your dataset's fasting blood sugar types
restecg = st.selectbox("Resting Electrocardiographic Results", [0, 1, 2])  # Adjust based on your dataset's ECG results
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0, max_value=250, value=150)
exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])  # Adjust based on your dataset's exercise induced angina types
oldpeak = st.number_input("Depression Induced by Exercise Relative to Rest", min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox("Slope of the Peak Exercise ST Segment", [0, 1, 2])  # Adjust based on your dataset's slope types
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3])  # Adjust based on your dataset's major vessels types
thal = st.selectbox("Thalassemia", [0, 1, 2, 3])  # Adjust based on your dataset's thalassemia types

# Prepare input data for prediction
user_data = pd.DataFrame([[age, 1 if sex == "Male" else 0, cp, trestbps, chol, 1 if fbs == "Yes" else 0, restecg, thalach, 1 if exang == "Yes" else 0, oldpeak, slope, ca, thal]],
                         columns=X.columns)

user_data_scaled = scaler.transform(user_data)

# Select model
selected_model_name = st.selectbox("Select Model", list(models.keys()))
selected_model = models[selected_model_name]

# Predict using the selected model
probability = selected_model.predict_proba(user_data_scaled)[0, 1]
prediction = selected_model.predict(user_data_scaled)[0]

st.write("### Prediction:")
st.write(f"**Probability of Heart Disease:** {probability:.2f}")
st.write(f"**Probability of No Heart Disease:** {1.00 - probability:.2f}")

st.write(f"**Prediction:** {'Heart Disease' if prediction == 1 else 'No Heart Disease'}")


# Treatment Plans
st.write("### Suggested Treatment Plan")

# Define treatment plans
def suggest_treatment_plan(probability, sex, thalach, prediction):
    if sex == "Female" and prediction == 1:
        return (
            "Since you are female and the prediction indicates the presence of heart disease, "
            "it is crucial to consult with a cardiologist for a comprehensive evaluation. "
            "The cardiologist will perform a detailed assessment to determine the extent of the condition "
            "and develop a personalized treatment plan tailored to your needs. This plan may include medications, "
            "lifestyle changes, and possibly further diagnostic tests. It is essential to follow through with all recommended tests "
            "and treatments to effectively manage and potentially improve your cardiovascular health."
        )
    
    elif sex == "Male" and thalach >= 150:
        return (
            "For males with a high maximum heart rate (thalach) and a prediction of heart disease, "
            "consider making lifestyle modifications. This may include incorporating regular physical activity into your routine, "
            "such as aerobic exercises and strength training, while ensuring you do not overexert yourself. "
            "Additionally, review your exercise regimen with your healthcare provider to ensure it aligns with your cardiovascular health needs. "
            "Regular follow-ups with your healthcare provider are essential to monitor your heart health and adjust your treatment plan as needed."
        )
    
    elif sex == "Female" and thalach < 150:
        return (
            "If you are female and your maximum heart rate (thalach) is lower than 150, it is still important to monitor your cardiovascular health closely. "
            "Schedule regular check-ups with your healthcare provider to keep track of any changes in your condition. "
            "In addition to routine medical evaluations, consider making dietary adjustments to support heart health, such as increasing your intake of fruits, vegetables, and whole grains. "
            "Stress management techniques, such as mindfulness or yoga, can also contribute to overall well-being and cardiovascular health."
        )
    
    elif sex == "Male" and prediction == 0:
        return (
            "As a male with a prediction of no heart disease, maintaining a healthy lifestyle is key to ensuring continued good health. "
            "Engage in regular physical activity, such as brisk walking or cycling, and follow a balanced diet rich in nutrients. "
            "Routine health screenings are important to catch any potential issues early. Stay proactive with your health by scheduling regular check-ups with your healthcare provider."
        )
    
    else:
        return (
            "For individuals with a prediction of heart disease and uncertain or mixed factors, it is advisable to consult with your healthcare provider for a comprehensive and personalized assessment. "
            "Your provider will consider your complete medical history, lifestyle factors, and test results to create a treatment plan that best suits your individual needs. "
            "This may include further diagnostic testing, personalized medication, lifestyle changes, and ongoing follow-up care to manage and improve your cardiovascular health effectively."
        )

# Display treatment plan
treatment_plan = suggest_treatment_plan(probability, sex, thalach, prediction)
st.write(treatment_plan)
