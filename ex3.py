import numpy as np
from sklearn .linear_model import LogisticRegression
import matplotlib.pyplot as plt
x=np.array([
    [25,21,110,170],
    [30,24,118,185],
    [38,27,125,205],
    [45,31,138,225],
    [50,33,145,235],
    [60,37,165,255]
])
y=np.array([0,0,1,1,1,1])
model=LogisticRegression()
model.fit(x,y)
age=int(input("Enter Age:"))
bmi=float(input("Enter BMI:"))
bp=int(input("Enter BMI:"))
chol=int(input("Enter Cholestrol:"))
patient=np.array([[age,bmi,bp,chol]])
prediction=model.predict(patient)
probability=model.predict_proba(patient)
print("\nProbability")
print("No Disease=",round(probability[0][0],4))
print("Disease=",round(probability[0][1],4))
if prediction[0]==1:
    print("Prediction:Disease Detected")
else:
    print("prediction:No Disease")
z=np.linspace(-10,10,100)
sigmoid=1/(1+np.exp(-z))
plt.plot(z,sigmoid)
plt.title("LogisticRegression")
plt.xlabel("z")
plt.ylabel("Probability")
plt.grid(True)
plt.show()
