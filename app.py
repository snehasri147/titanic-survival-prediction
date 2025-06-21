import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Set up Streamlit
st.set_page_config(page_title="Titanic Survival Predictor", layout="wide")
st.title("🚢 Titanic Survival Prediction")
st.markdown("Choose a model and enter passenger details to predict survival outcome.")

# Sidebar inputs
st.sidebar.header("Passenger Details")
pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
sex = st.sidebar.selectbox("Sex", ['male', 'female'])
age = st.sidebar.slider("Age", 0, 80, 25)
sibsp = st.sidebar.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.sidebar.number_input("Parents/Children Aboard", 0, 10, 0)
fare = st.sidebar.slider("Fare", 0.0, 600.0, 50.0)
embarked = st.sidebar.selectbox("Embarked", ['C', 'Q', 'S'])
model_name = st.sidebar.selectbox("Model", [
    "Logistic Regression", "Decision Tree", "Random Forest", "Naive Bayes"
])

# Load encoders & scaler
with open("le_sex.pkl", "rb") as f: le_sex = pickle.load(f)
with open("le_embarked.pkl", "rb") as f: le_embarked = pickle.load(f)
with open("scaler.pkl", "rb") as f: scaler = pickle.load(f)

# Map input to DataFrame
input_df = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [sex],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Embarked': [embarked]
})

# Encode and scale
input_df['Sex'] = le_sex.transform(input_df['Sex'])
input_df['Embarked'] = le_embarked.transform(input_df['Embarked'])
X_scaled = scaler.transform(input_df)

# Load model
model_files = {
    "Logistic Regression": "model_logistic_regression.pkl",
    "Decision Tree": "model_decision_tree.pkl",
    "Random Forest": "model_random_forest.pkl",
    "Naive Bayes": "model_naive_bayes.pkl"
}
with open(model_files[model_name], "rb") as f:
    model = pickle.load(f)

# Prediction
pred = model.predict(X_scaled)[0]
proba = model.predict_proba(X_scaled)[0]

# Output
st.subheader(f"Prediction using {model_name}")
if pred == 1:
    st.success("🟢 The passenger is likely to **Survive**")
else:
    st.error("🔴 The passenger is likely to **Not Survive**")

st.subheader("Prediction Probabilities")
st.write({ "Not Survived": proba[0], "Survived": proba[1] })
