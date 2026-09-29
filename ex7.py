import numpy as np
import matplotlib.pyplot as plt
X = np.array([
    [1,1,1],
    [1,1,0],
    [1,0,1],
    [0,1,1],
    [1,0,0],
    [0,1,0],
    [0,0,1],
    [0,0,0]
])
y = np.array([1,1,1,1,0,0,0,0])
weights = np.zeros(3)
bias = 0
lr = 0.1
epochs = 10
errors = []
for epoch in range(epochs):
    error_count = 0
    for i in range(len(X)):
        z = np.dot(X[i], weights) + bias
        prediction = 1 if z >= 0 else 0
        error = y[i] - prediction
        weights = weights + lr * error * X[i]
        bias = bias + lr * error
        if error != 0:
            error_count += 1
    errors.append(error_count)
print("\nEnter Email Details")
free = int(input("Contains 'Free' (1=Yes, 0=No): "))
offer = int(input("Contains 'Offer' (1=Yes, 0=No): "))
unknown = int(input("Unknown Sender (1=Yes, 0=No): "))
new_email = np.array([free, offer, unknown])
z = np.dot(new_email, weights) + bias
result = 1 if z >= 0 else 0
print("\nWeighted Sum =", z)
if result == 1:
    print("Prediction: SPAM")
else:
    print("Prediction: NOT SPAM")
plt.plot(range(1, epochs + 1), errors, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Number of Errors")
plt.title("Perceptron Training Error")
plt.grid()
plt.show()
