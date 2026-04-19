import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ======================
# Load dataset
# ======================
df = pd.read_csv("eda_data.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Select features
df = df[['age', 'python_yn', 'spark', 'aws', 'excel',
         'Rating', 'num_comp', 'seniority', 'avg_salary']]

# Handle 'na' values
df.replace('na', np.nan, inplace=True)

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.fillna(df.mean(numeric_only=True))

# ======================
# Create target
# ======================
threshold = df['avg_salary'].median()
df['HighSalary'] = (df['avg_salary'] > threshold).astype(int)

# ======================
# Features & target
# ======================
X = df.drop(['avg_salary', 'HighSalary'], axis=1)
y = df['HighSalary']

# ======================
# Train-test split
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================
# Scaling
# ======================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ======================
# Model
# ======================
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Accuracy
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

# ======================
# UI
# ======================
st.title("💼 Salary Prediction App")
st.write("Predict whether a job offers HIGH or LOW salary")

st.write(f"### Model Accuracy: {round(accuracy*100,2)}%")

st.sidebar.header("Enter Details")

# Inputs
age = st.sidebar.slider("Age", 18, 60, 25)
rating = st.sidebar.slider("Company Rating", 1.0, 5.0, 3.5)
num_comp = st.sidebar.slider("Number of Competitors", 0, 10, 2)
seniority = st.sidebar.slider("Seniority Level", 0, 5, 2)

python = st.sidebar.selectbox("Python Skill", ["No", "Yes"])
spark = st.sidebar.selectbox("Spark Skill", ["No", "Yes"])
aws = st.sidebar.selectbox("AWS Skill", ["No", "Yes"])
excel = st.sidebar.selectbox("Excel Skill", ["No", "Yes"])

# Convert Yes/No → 0/1
python = 1 if python == "Yes" else 0
spark = 1 if spark == "Yes" else 0
aws = 1 if aws == "Yes" else 0
excel = 1 if excel == "Yes" else 0

# ======================
# Prediction
# ======================
if st.button("Predict Salary"):

    input_data = pd.DataFrame({
        'age': [age],
        'python_yn': [python],
        'spark': [spark],
        'aws': [aws],
        'excel': [excel],
        'Rating': [rating],
        'num_comp': [num_comp],
        'seniority': [seniority]
    })

    # Ensure same column order
    input_data = input_data[X.columns]

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probs = model.predict_proba(input_scaled)[0]

    # 🔥 Correct confidence (based on predicted class)
    confidence = probs[prediction]

    # Show input
    st.write("### Input Summary")
    st.write(input_data)

    # Output
    if prediction == 1:
        st.success(f"💰 High Salary (Confidence: {round(confidence*100,2)}%)")
    else:
        st.error(f"📉 Low Salary (Confidence: {round(confidence*100,2)}%)")
