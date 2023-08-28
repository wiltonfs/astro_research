import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from mapie.regression import MAPIERegressor

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
mapie_regressor = MAPIERegressor(base_model)

# Fit MAPIERegressor on the training data
mapie_regressor.fit(X_train, y_train)

# Predict with uncertainty using MAPIERegressor
prediction, prediction_interval = mapie_regressor.predict(X_test, alpha=0.1)  # alpha determines the prediction level

# Print predictions and prediction intervals
for i in range(len(X_test)):
    print(f"Input: {X_test[i]:.2f} | Prediction: {prediction[i]:.2f} | Prediction Interval: {prediction_interval[i]}")

