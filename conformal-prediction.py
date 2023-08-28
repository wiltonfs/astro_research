import numpy as np
from mapie.regression import MapieRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(100, 1)  # Features
y = 2 * X.squeeze() + np.random.normal(0, 0.2, 100)  # True function with added noise

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a base regression model (e.g., linear regression)
base_model = LinearRegression()
base_model.fit(X_train, y_train)

# Initialize MAPIERegressor with the base model
mapie_regressor = MapieRegressor(base_model)

# Fit MAPIERegressor on the training data
mapie_regressor.fit(X_train, y_train)

# Predict with uncertainty using MAPIERegressor
prediction, prediction_interval = mapie_regressor.predict(X_test, alpha=0.1)  # alpha determines the prediction level

# Print predictions and prediction intervals
for i in range(len(X_test)):
    print(f"Input: {X_test[i][0]:.2f} | Prediction: {prediction[i]:.2f} | Prediction Interval: {prediction_interval[i].tolist()}")
print("\n\n\n\n\n\n")
# Calculate yerr:
yerr = np.ones((2, len(X_test)))  # Initialize the yerr array with zeros
print(yerr.shape)
yerr[0, :] = prediction.squeeze() - prediction_interval[:, 0].squeeze()  # Calculate the lower errors
yerr[1, :] = prediction_interval[:, 1].squeeze() - prediction.squeeze()  # Calculate the upper errors



# Create a scatter plot with error bars
plt.scatter(X_test, y_test, label="True Values", color='blue')
#plt.scatter(X_test, prediction, label="Predictions", color='orange')
plt.errorbar(X_test, prediction, yerr=yerr, fmt='o', label="Predictions", color='orange', ecolor='red')

# Add labels and legend
plt.xlabel("X")
plt.ylabel("y")
plt.title("Scatter Plot with Error Bars")
plt.legend()

# Show the plot
plt.show()

