import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#  مسار البيانات (جوه مشروعك)
path = r"D:\Level 5\ping-pong\Heart_Disease_Project\heart_disease.csv"

# تحميل البيانات
df = pd.read_csv(path)

X = df.drop(["num", "id"], axis=1)
y = df["num"]

#  تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#  إعداد الموديل + البايبلاين
best_params = {"max_depth": 5, "min_samples_leaf": 4, "min_samples_split": 2, "n_estimators": 50}

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(**best_params, random_state=42))
])

#  تدريب البايبلاين
pipeline.fit(X_train, y_train)

#  مسار حفظ الموديل (جوه المشروع)
model_dir = r"D:\Level 5\ping-pong\Heart_Disease_Project\models"
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "final_model.pkl")

#  حفظ الموديل
joblib.dump(pipeline, model_path)

print(f" Model saved inside project at: {model_path}")
