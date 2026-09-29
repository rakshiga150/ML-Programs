import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
df = pd.read_csv(r"C:/Users/24ucs150/Downloads/student_performance_100.csv")
print("STUDENT PERFORMANCE DATASET")
print(df)
X = df[["Attendance", "Study_Hours", "Internal_Marks", "Participation"]]
y = df["Result"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = RandomForestClassifier(
    n_estimators=10,
    criterion="gini",
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nACTUAL RESULTS:")
print(y_test.values)
print("\nPREDICTED RESULTS:")
print(y_pred)
print("\nACCURACY:", accuracy * 100, "%")
print("\nCONFUSION MATRIX:")
print(confusion_matrix(y_test, y_pred, labels=["Pass", "Fail"]))
print("\nENTER STUDENT DETAILS")
attendance = float(input("Enter Attendance (%): "))
study_hours = float(input("Enter Study Hours: "))
internal_marks = float(input("Enter Internal Marks: "))
participation = float(input("Enter Participation (1-10): "))
student = pd.DataFrame(
    [[attendance, study_hours, internal_marks, participation]],
    columns=["Attendance", "Study_Hours", "Internal_Marks", "Participation"]
)
prediction = model.predict(student)
print("\nPREDICTED RESULT:", prediction[0])
if prediction[0] == "Pass":
    print("The student is predicted to PASS.")
else:
    print("The student is predicted to FAIL.")
features = [
    "Attendance",
    "Study Hours",
    "Internal Marks",
    "Participation"
]
importance = model.feature_importances_
plt.figure(figsize=(8, 5))
plt.bar(features, importance)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Random Forest Feature Importance")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()
