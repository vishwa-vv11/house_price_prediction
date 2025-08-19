import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv(r"C:\\Users\\darkp\\OneDrive\\Desktop\\PycharmProjects\\house price prediction\\housing.csv")

# Define features and target variable
X = df.drop("MEDV", axis=1)
Y = df["MEDV"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate the model
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Compare actual vs predicted values
df_compare = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
print(df_compare.head())

# Predict new house price
new_house = [[5.45, 10.10, 26]]  # Ensure this matches your feature columns order and count
predicted = model.predict(new_house)
print(f"Predicted output is: {math.floor(predicted[0])}")
