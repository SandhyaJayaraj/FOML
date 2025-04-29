import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Step 1: Import necessary libraries

# Step 2: Read the dataset

data = pd.read_csv(r"C:\Users\Admin\Downloads\FOML\exp2\headbrain.csv")

data.head()
data.info()
data.describe()

# Step 3: Prepare the data
X = data['Head Size(cm^3)'].values
y = data['Brain Weight(grams)'].values

# Step 4: Calculate the mean
mean_x, mean_y = np.mean(X), np.mean(y)

# Step 5: Calculate the coefficients
b1 = np.sum((X - mean_x) * (y - mean_y)) / np.sum((X - mean_x) ** 2)
b0 = mean_y - b1 * mean_x

# Step 6: Make predictions
y_pred = b0 + b1 * X

# Step 7: Plot the regression line
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Actual data', alpha=0.6)
plt.plot(X, y_pred, color='red', label='Regression line', linewidth=2)
plt.xlabel('Head Size (cm³)')
plt.ylabel('Brain Weight (grams)')
plt.legend()
plt.title('Linear Regression using Least Squares')
plt.show()

# Step 8: Plot the residuals
residuals = y - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(X, residuals, color='purple', alpha=0.6)
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
plt.xlabel('Head Size (cm³)')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

# Step 9: Calculate the R-squared value
TSS = np.sum((y - mean_y) ** 2)
RSS = np.sum((y - y_pred) ** 2)
R2 = 1 - (RSS / TSS)

# Step 10: Display the results
print(f"Intercept: {b0:.2f}")
print(f"Slope: {b1:.2f}")
print(f"R-squared Value: {R2:.4f}")
