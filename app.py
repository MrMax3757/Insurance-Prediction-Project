import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Insurance Cost Predictor", layout="centered")
st.title("🏥 Medical Insurance Cost Prediction")
st.markdown("Enter your details below to get an estimated annual medical cost.")

# Load model and feature columns (cached for performance)
@st.cache_resource
def load_model():
    model = joblib.load('model.pkl')
    feature_cols = joblib.load('feature_columns.pkl')
    return model, feature_cols

model, feature_cols = load_model()

# ---------- Sidebar Inputs ----------
st.sidebar.header("Your Information")

age = st.sidebar.slider("Age", 18, 100, 30)
bmi = st.sidebar.slider("BMI (Body Mass Index)", 15.0, 50.0, 25.0, 0.1)
children = st.sidebar.selectbox("Number of Children", [0,1,2,3,4,5])
smoker = st.sidebar.selectbox("Smoker", ["No", "Yes"])
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
region = st.sidebar.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# Convert to model format
smoker_binary = 1 if smoker == "Yes" else 0
sex_binary = 1 if sex == "Male" else 0

# One‑hot encode region (same as training)
region_northwest = 1 if region == "northwest" else 0
region_southeast = 1 if region == "southeast" else 0
region_southwest = 1 if region == "southwest" else 0
# region_northeast is the dropped category → all zeros

# Create DataFrame with the exact column order used during training
input_data = pd.DataFrame([[
    age, sex_binary, bmi, children, smoker_binary,
    region_northwest, region_southeast, region_southwest
]], columns=feature_cols)

st.write("### Your Input Values")
st.dataframe(input_data)

# ---------- Initialize session state for history ----------
if 'history' not in st.session_state:
    st.session_state.history = []

# ---------- Prediction Button ----------
if st.button("Predict My Cost"):
    prediction = model.predict(input_data)[0]
    
    # Store this prediction in history
    st.session_state.history.append({
        'Age': age,
        'BMI': bmi,
        'Children': children,
        'Smoker': smoker,
        'Sex': sex,
        'Region': region,
        'Predicted Cost ($)': round(prediction, 2)
    })
    
    st.success(f"### 💰 Predicted Annual Cost: **${prediction:,.2f}**")
    
    if prediction > 20000:
        st.warning("⚠️ This is above average. Factors like age, smoking, or high BMI may be increasing the cost.")
    else:
        st.balloons()

# ---------- Show Prediction History ----------
if st.session_state.history:
    st.subheader("📜 Prediction History")
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True)
    
    # Clear history button
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()
else:
    st.info("No predictions yet. Enter values and click 'Predict My Cost'.") 