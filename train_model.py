import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
df = pd.read_excel("loan_approval.xlsx")

print("=" * 60)
print("LOAN APPROVAL PREDICTION PROJECT")
print("=" * 60)

print(df.head())
print(df.info())

print(df.shape)

print(df.columns)

print(df.describe())

print(df.isnull().sum())
df["Gender"].fillna(df["Gender"].mode()[0], inplace=True)
df["Married"].fillna(df["Married"].mode()[0], inplace=True)
df["Dependents"].fillna(df["Dependents"].mode()[0], inplace=True)
df["Self_Employed"].fillna(df["Self_Employed"].mode()[0], inplace=True)

df["LoanAmount"].fillna(df["LoanAmount"].median(), inplace=True)

df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0], inplace=True)

df["Credit_History"].fillna(df["Credit_History"].mode()[0], inplace=True)
print(df.isnull().sum())
df.drop("Loan_ID", axis=1, inplace=True)
le = LabelEncoder()

categorical = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
    "Loan_Status"
]

for col in categorical:
    df[col] = le.fit_transform(df[col])
    print(df.head())
    plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()
X = df.drop("Loan_Status", axis=1)

y = df["Loan_Status"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy :", accuracy)
cm = confusion_matrix(y_test, y_pred)

print(cm)

sns.heatmap(cm, annot=True, fmt="d")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()
print(classification_report(y_test, y_pred))
joblib.dump(model, "loan_model.pkl")

joblib.dump(scaler, "scaler.pkl")

print("Model Saved Successfully")
# Save the trained model and scaler
joblib.dump(model, "loan_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved successfully!") 
