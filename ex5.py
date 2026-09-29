from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
x=[
    [25,120,180],
    [30,125,190],
    [35,130,200],
    [40,140,220],
    [45,145,230],
    [50,150,240],
    [28,122,185],
    [48,148,235]
]
y=[0,0,0,1,1,1,0,1]
model = KNeighborsClassifier(n_neighbors=3)
model.fit(x,y)
test=[
    [43,142,225],
    [32,128,195]
]
actual=[1,0]
prediction = model.predict(test)
print("Prediction:",prediction)
for i in range(len(test)):
    if prediction[i]==actual[i]:
        print("Correct Prediction")
    else:
        print("Wrong Prediction")
accuracy= accuracy_score(actual,prediction)
print("Acuracy:",accuracy)
age = [25,30,35,40,45,50,28,48]
cholestrol= [180,190,200,220,230,240,145,235]
disease = [0,0,0,1,1,1,0,1]
for i in range(len(age)):
    if disease[i] == 0:
        plt.scatter(age[i],cholestrol[i],label="No Disease" if i == 0 else "")
    else:
        plt.scatter(age[i],cholestrol[i],label="Heart Disease" if i == 3 else "")
plt.scatter(43,225,marker="*",s=150,label="New patient")
plt.xlabel("Age")
plt.ylabel("Cholestrol")
plt.title("K-NN Classification")
plt.legend()
plt.grid(True)
plt.show()
        
