import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# =======================
# 1. تحميل الموديل
# =======================
model_path = "D:/Level 5/ping-pong/Heart_Disease_Project/models/final_model.pkl"
model = joblib.load(model_path)

# =======================
# 2. تحميل البيانات
# =======================
data_path = "D:/Level 5/ping-pong/Heart_Disease_Project/heart_disease.csv"
df = pd.read_csv(data_path)
print(" Dataset Loaded:", df.shape)

# =======================
# 3. إعداد بيانات التنبؤ (بدون الأعمدة اللي مش مستخدمة)
# =======================
X = df.drop(["num", "id"], axis=1)   # بنشيل target والـ id
y = df["num"]

sample_input = X.iloc[[0]]   # أول صف كعينة للتجربة
prediction = model.predict(sample_input)[0]
print(f" Example Prediction Result for first row: Class {prediction}")

# =======================
# 4. رسومات استكشافية
# =======================
results_dir = "D:/Level 5/ping-pong/Heart_Disease_Project/results"
os.makedirs(results_dir, exist_ok=True)

# توزيع الفئات
plt.figure(figsize=(6,4))
sns.countplot(x="num", data=df)
plt.title("Heart Disease Classes Distribution")
plt.savefig(os.path.join(results_dir, "class_distribution.png"))
plt.close()

# توزيع الأعمار
plt.figure(figsize=(6,4))
sns.histplot(df["age"], bins=20, kde=True)
plt.title("Age Distribution")
plt.savefig(os.path.join(results_dir, "age_distribution.png"))
plt.close()

# مصفوفة الارتباط
plt.figure(figsize=(10,8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig(os.path.join(results_dir, "correlation_heatmap.png"))
plt.close()

print(f" الرسومات اتخزنت في: {results_dir}")
