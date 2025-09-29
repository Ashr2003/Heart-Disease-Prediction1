#  Heart Disease Prediction Project

##  Project Overview
This project aims to build a Machine Learning pipeline for **Heart Disease Prediction**.  
It includes **data preprocessing, feature selection, dimensionality reduction, supervised & unsupervised modeling, model optimization, and deployment** with a Streamlit web UI.

---

##  Final Deliverables
✔ Cleaned dataset with selected features  
✔ Dimensionality reduction (PCA) results  
✔ Trained supervised and unsupervised models  
✔ Performance evaluation metrics  
✔ Hyperparameter optimized model  
✔ Saved model in `.pkl` format  
✔ GitHub repository with all source code  
✔ Streamlit UI for real-time predictions **[Bonus]**  
✔ Ngrok link to access the live app **[Bonus]**

---

##  Project Workflow

### 1️ Data Preprocessing
- Handle missing values  
- Encode categorical variables  
- Standardize numerical features  
- Save cleaned dataset (`heart_cleaned.csv`)  

### 2️ Feature Selection
- RandomForest Feature Importance  
- Recursive Feature Elimination (RFE)  
- Chi-Square test for statistical significance  

### 3️ Dimensionality Reduction
- Apply **PCA**  
- Visualize explained variance & scatter plots  
- Save PCA-transformed dataset (`heart_pca.csv`)  

### 4️ Supervised Learning
- Models: Logistic Regression, Decision Tree, Random Forest, SVM  
- Metrics: Accuracy, Precision, Recall, F1-score, ROC/AUC  
- Save evaluation results in `results/`

### 5️ Unsupervised Learning
- **K-Means Clustering** (Elbow method)  
- **Hierarchical Clustering** (Dendrogram)  
- Compare clusters with actual disease labels  

### 6️ Hyperparameter Tuning
- GridSearchCV & RandomizedSearchCV  
- Select best parameters for Random Forest  

### 7️ Model Export
- Save trained pipeline as `.pkl` file  
- Example: `final_model.pkl`  

### 8 Streamlit Web App [Bonus]
- User inputs health data  
- Real-time prediction output  
- Data visualizations (Age distribution, etc.)  

### 9️ Ngrok Deployment [Bonus]
- Expose local Streamlit app to the internet  
- Generate public link for live demo  

---

##  How to Run the Project

### 🔹 Clone Repository
```bash
git clone https://github.com/YourUsername/Heart_Disease_Project.git
cd Heart_Disease_Project
