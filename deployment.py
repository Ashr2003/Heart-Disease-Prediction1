# ==============================
# deployment.py
# ==============================

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ========== تحميل الموديل ==========
model_path = r"D:\Level 5\ping-pong\Heart_Disease_Project\final_model.pkl"
model = joblib.load(model_path)

# لو الموديل معمول كـ Pipeline يقدر يحتفظ بالأعمدة
# هنخزن الأعمدة اللي اتدرب عليها من مرحلة التدريب
if hasattr(model, "feature_names_in_"):
    feature_names = model.feature_names_in_.tolist()
else:
    # fallback: ناخدها من الداتا
    data_path = r"D:\Level 5\ping-pong\Heart_Disease_Project\heart_cleaned.csv"
    df = pd.read_csv(data_path)
    drop_cols = [c for c in ["num", "id"] if c in df.columns]
    feature_names = [col for col in df.columns if col not in drop_cols]

# ========== عنوان التطبيق ==========
st.title(" Heart Disease Prediction App")
st.write("أدخل بياناتك الصحية للحصول على توقع الإصابة بأمراض القلب.")

# ========== مدخلات المستخدم ==========
age = st.number_input("Age", min_value=20, max_value=100, value=40)
trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
chol = st.number_input("Cholesterol", min_value=100, max_value=400, value=200)
thalch = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)
oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=6.0, value=1.0)
ca = st.number_input("Number of Major Vessels", min_value=0, max_value=4, value=0)
sex_Male = st.selectbox("Sex (1=Male, 0=Female)", [0, 1])

user_input = {
    "age": age,
    "trestbps": trestbps,
    "chol": chol,
    "thalch": thalch,
    "oldpeak": oldpeak,
    "ca": ca,
    "sex_Male": sex_Male
}

# ========== تجهيز الـ sample_input ==========
sample_input = pd.DataFrame([user_input])

# نضيف الأعمدة الناقصة بالقيمة 0
for col in feature_names:
    if col not in sample_input.columns:
        sample_input[col] = 0

# نخلي الأعمدة بنفس الترتيب اللي اتدرب عليه الموديل
sample_input = sample_input[feature_names]

# ========== التنبؤ ==========
if st.button(" Predict"):
    prediction = model.predict(sample_input)[0]
    st.subheader(f" Prediction: {prediction} (Disease Severity Class)")
    st.write("0 = No Disease, 1–4 = Increasing Severity")

# ========== Visualization ==========
st.subheader(" Sample Visualization")
data_path = r"D:\Level 5\ping-pong\Heart_Disease_Project\heart_cleaned.csv"
df = pd.read_csv(data_path)
fig, ax = plt.subplots()
sns.histplot(df["age"], bins=20, kde=True, ax=ax)
ax.set_title("Age Distribution in Dataset")
plt.tight_layout()
st.pyplot(fig)
