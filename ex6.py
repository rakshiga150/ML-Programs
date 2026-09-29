from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
X = [
    ["Young", "Low", "Short"],
    ["Young", "High", "Long"],
    ["Adult", "Low", "Short"],
    ["Adult", "High", "Long"],
    ["Young", "High", "Short"],
    ["Adult", "Low", "Long"],
    ["Young", "Low", "Long"],
    ["Adult", "High", "Short"],
    ["Young", "Low", "Short"],
    ["Adult", "High", "Long"]
]

y = [0, 1, 0, 1, 1, 0, 1, 1, 0, 1]

encoders = []
X_encoded = []

for column in range(3):
    encoder = LabelEncoder()
    values = [row[column] for row in X]
    encoded = encoder.fit_transform(values)

    X_encoded.append(encoded)
    encoders.append(encoder)

X_encoded = list(zip(*X_encoded))

model = CategoricalNB()

model.fit(X_encoded, y)

test = [["Young", "High", "Long"]]

test_encoded = []

for column in range(3):
    value = encoders[column].transform(
        [test[0][column]]
    )[0]

    test_encoded.append(value)

test_encoded = [test_encoded]

prediction = model.predict(test_encoded)

print("Prediction:", prediction[0])

if prediction[0] == 1:
    print("User will Click the Advertisement")
else:
    print("User will Not Click the Advertisement")



predicted = model.predict(X_encoded)

accuracy = accuracy_score(y, predicted)

print("Accuracy:", accuracy)

sample_numbers = list(range(1, len(y) + 1))

plt.figure(figsize=(10, 5))

plt.plot(
    sample_numbers,
    y,
    marker="o",
    label="Actual",
    linewidth=2
)

plt.plot(
    sample_numbers,
    predicted,
    marker="s",
    label="Predicted",
    linewidth=2
)

plt.xlabel("Sample Number")
plt.ylabel("Click (0 = No, 1 = Yes)")
plt.title("Actual vs Predicted Advertisement Clicks")

plt.xticks(sample_numbers)
plt.yticks([0, 1], ["No Click", "Click"])

plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

plt.show()
