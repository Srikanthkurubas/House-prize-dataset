# House-prize-dataset
Implement a linear regression model to predict the prices of houses based on their square footage and the number of bedrooms and bathrooms
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Step 1: Create a sample dataset (replace this with your actual dataset)
data = {
    'square_feet': [1500, 1800, 2400, 3000, 3500, 4000, 1200, 2000, 2500, 2700],
    'bedrooms': [3, 4, 3, 5, 4, 5, 2, 4, 4, 3],
    'bathrooms': [2, 3, 2, 4, 3, 4, 1, 2, 3, 2],
    'price': [300000, 400000, 360000, 550000, 480000, 600000, 250000, 420000, 470000, 450000]
}

df = pd.DataFrame(data)

# Step 2: Define features and target
X = df[['square_feet', 'bedrooms', 'bathrooms']]
y = df['price']

# Step 3: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model coefficients:", model.coef_)
print("Model intercept:", model.intercept_)
print("Mean Squared Error:", mse)
print("R² Score:", r2)

# Optional: Visualize predictions vs actual values
plt.scatter(y_test, y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.show()
