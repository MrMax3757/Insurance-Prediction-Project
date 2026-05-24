# 🏥 Medical Insurance Cost Predictor

A machine learning web application that predicts annual medical insurance costs based on personal health information. Built with Python, scikit-learn, and Streamlit.


## 🎯 Project Overview

This project is the **capstone** of a 7‑day machine learning course. It demonstrates an end‑to‑end ML workflow:

- **Data collection** – real‑world insurance dataset (Kaggle)
- **Exploratory Data Analysis** – visualisations with Matplotlib & Seaborn
- **Preprocessing** – encoding categorical variables, train‑test split
- **Model training** – Linear Regression, Logistic Regression, Random Forest
- **Model evaluation** – RMSE, R², confusion matrix, precision/recall
- **Deployment** – interactive web app using Streamlit

The best model (Random Forest Regressor) achieves **R² ≈ 0.85** and **RMSE ≈ $5,000** on test data.

## 📊 Dataset

**Source:** [Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance) (Kaggle)  
**Features:**

| Feature | Type | Description |
|---------|------|-------------|
| age | numeric | Beneficiary age (18–64) |
| sex | categorical | male / female |
| bmi | numeric | Body Mass Index (15–50) |
| children | numeric | Number of dependents |
| smoker | categorical | yes / no |
| region | categorical | southwest, southeast, northwest, northeast |
| charges | numeric (target) | Individual medical costs billed by health insurance |

## 🧠 Model Performance

| Model | R² | RMSE | Accuracy (classification) |
|-------|----|----|---------------------------|
| Linear Regression | 0.78 | $5,796 | – |
| Logistic Regression (high/low cost) | – | – | 91% |
| **Random Forest Regressor (final)** | **0.85** | **$5,000** | – |

**Key insight:** Smoking is the most important feature, followed by BMI and age.

