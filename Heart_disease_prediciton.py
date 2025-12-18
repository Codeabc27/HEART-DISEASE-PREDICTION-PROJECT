import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv("heart.csv")


print(data.head())
print(data.info())
print(data.describe())


print(data.isnull().sum())

X = data.drop('target', axis=1) 
y = data['target']               

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = LogisticRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


new_patient = np.array([[52, 1, 0, 125, 212, 0, 1, 168, 0, 1.0, 2, 0, 2]])
new_patient = scaler.transform(new_patient)

result = model.predict(new_patient)

if result[0] == 1:
    print("⚠️ High Risk of Heart Disease")
else:
    print("✅ Low Risk of Heart Disease")
