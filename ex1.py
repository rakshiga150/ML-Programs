import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dataset
temperature = np.array([21, 23, 25, 27, 29]).reshape(-1, 1)
sales = np.array([125, 140, 155, 170, 185])

# Create and train the model
model = LinearRegression()
model.fit(temperature, sales)

# Model parameters
print("Intercept (b):", model.intercept_)
print("Slope (m):", model.coef_[0])

# Predict sales for 35°C
temp = np.array([[35]])
predicted_sales = model.predict(temp)

print("Predicted Ice Cream Sales at 35°C:", predicted_sales[0])

# Plot
plt.scatter(temperature, sales, color='blue', label='Data Points')
plt.plot(temperature, model.predict(temperature), color='red', label='Regression Line')
plt.scatter(35, predicted_sales, color='green', s=100, label='Prediction (35°C)')
plt.xlabel("Temperature (°C)")
plt.ylabel("Ice Cream Sales")
plt.title("Simple Linear Regression")
plt.legend()
plt.grid(True)
plt.show()
