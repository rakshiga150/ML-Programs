import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
df = pd.read_csv(r"C:/Users/24ucs150/Downloads/patient_health_risk.csv")
print("Dataset:")
print(df)
X = df[['Age', 'BMI', 'BP']]
y = df['Risk']
weak_learner = DecisionTreeClassifier(
    max_depth=1,
    random_state=42
)
model = AdaBoostClassifier(
    estimator=weak_learner,
    n_estimators=5,
    learning_rate=1.0,
    random_state=42
)
model.fit(X, y)
y_pred = model.predict(X)
print("\nActual Values:")
print(y.values)
print("\nPredicted Values:")
print(y_pred)
accuracy = accuracy_score(y, y_pred)
print("\nAccuracy:", accuracy * 100, "%")
print("\nConfusion Matrix:")
print(confusion_matrix(y, y_pred))
plt.figure(figsize=(8, 5))
plt.scatter(
    df['Age'],
    df['BMI'],
    c=y_pred,
    s=100
)
plt.xlabel("Age")
plt.ylabel("BMI")
plt.title("AdaBoost Classification - Patient Health Risk")
plt.colorbar(label="Predicted Risk")
plt.grid(True)
plt.show()
