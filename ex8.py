import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

data = {
    "Glucose": [80, 85, 90, 95, 100, 140, 150, 160, 170, 180],
    "BMI": [20, 22, 24, 25, 27, 30, 32, 34, 36, 38],
    "Condition": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)

X = df[["Glucose", "BMI"]]
y = df["Condition"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = SVC(kernel="linear", C=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Actual Values:", y_test.values)
print("Predicted Values:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
glucose = 110
bmi = 28
new_patient = pd.DataFrame(
    [[glucose, bmi]],
    columns=["Glucose", "BMI"]
)
prediction = model.predict(new_patient)
print("\nNew Patient")
print("Glucose:", glucose)
print("BMI:", bmi)
if prediction[0] == 0:
    print("Predicted Condition: No Condition")
else:
    print("Predicted Condition: Condition Present")
print("Weights:", model.coef_[0])
print("Bias:", model.intercept_[0])
plt.scatter(
    df[df.Condition == 0]["Glucose"],
    df[df.Condition == 0]["BMI"],
    label="No Condition"
)
plt.scatter(
    df[df.Condition == 1]["Glucose"],
    df[df.Condition == 1]["BMI"],
    label="Condition"
)
w = model.coef_[0]
b = model.intercept_[0]
glucose_values = [80, 180]
bmi_values = [
    -(w[0] * glucose + b) / w[1]
    for glucose in glucose_values
]
plt.plot(
    glucose_values,
    bmi_values,
    label="Decision Boundary"
)

plt.scatter(
    glucose,
    bmi,
    marker="*",
    s=150,
    label="New Patient"
)
plt.xlabel("Glucose")
plt.ylabel("BMI")
plt.title("SVM Classification")
plt.legend()
plt.grid()
plt.show()
