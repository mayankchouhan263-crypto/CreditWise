import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import joblib

df = pd.read_csv("ML/Data/Processed_loan_approval_data.csv")

X = df.drop("Loan_Approved", axis=1)
y = df["Loan_Approved"]

#Feature engineering
# Add or Tranform features
df["DTI_Ratio_sq"] = df["DTI_Ratio"] ** 2
df["Credit_Score_sq"] = df["Credit_Score"] ** 2

# df["Applicant_Income_log"] = np.log1p(df["Applicant_Income"])

X = df.drop(columns=["Loan_Approved", "Credit_Score", "DTI_Ratio"])
y = df["Loan_Approved"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)

y_pred = log_model.predict(X_test_scaled)


# Get directory of this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Go to project root
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

# Model folder path
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")

# Create folder if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Save files
joblib.dump(log_model, os.path.join(MODEL_DIR, "loan_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

print("Model saved to:", MODEL_DIR)

# # Evaluation
# print("Logistic Regression Model")
# print("Precision: ", precision_score(y_test, y_pred))
# print("Recall: ", recall_score(y_test, y_pred))
# print("F1 score: ", f1_score(y_test, y_pred))
# print("Accuracy: ", accuracy_score(y_test, y_pred))
# print("CM: ", confusion_matrix(y_test, y_pred))

