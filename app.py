import pandas as pd

data = pd.read_csv("diabetes.csv")

print("Dataset loaded successfully!")
print(data.head())
print("\nDataset shape:", data.shape)
print("\nColumn names:", list(data.columns))
print("\nMissing values:\n", data.isnull().sum())
columns_to_clean = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

data[columns_to_clean] = data[columns_to_clean].replace(0, pd.NA)
data[columns_to_clean] = data[columns_to_clean].fillna(
    data[columns_to_clean].median()
)

print("\nData cleaned successfully!")
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

print("\nInput features shape:", X.shape)
print("Target shape:", y.shape)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"\nModel accuracy: {accuracy:.2%}")
from sklearn.metrics import classification_report, confusion_matrix

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Diabetes", "Diabetes"],
    yticklabels=["No Diabetes", "Diabetes"],
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
import joblib

joblib.dump(model, "diabetes_model.pkl")
print("\nModel saved successfully!")