import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load dataset

df = pd.read_csv(r"C:\Users\Admin\Downloads\FOML\exp1\Housing.csv")

# Step 2: Preprocess data (convert categorical variables)
le = LabelEncoder()
df['mainroad'] = le.fit_transform(df['mainroad'])
df['guestroom'] = le.fit_transform(df['guestroom'])
df['basement'] = le.fit_transform(df['basement'])
df['hotwaterheating'] = le.fit_transform(df['hotwaterheating'])
df['airconditioning'] = le.fit_transform(df['airconditioning'])
df['prefarea'] = le.fit_transform(df['prefarea'])
df['furnishingstatus'] = le.fit_transform(df['furnishingstatus'])

# Step 3: Univariate Regression (Price vs Area)
X_uni = df[['area']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X_uni, y, test_size=0.2, random_state=42)
model_uni = LinearRegression()
model_uni.fit(X_train, y_train)
y_pred_uni = model_uni.predict(X_test)

# Plot Univariate Regression
plt.figure(figsize=(8,6))
plt.scatter(X_test, y_test, color='blue', label='Actual Data')
plt.plot(X_test, y_pred_uni, color='red', linewidth=2, label='Regression Line')
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Univariate Regression (Area vs Price)')
plt.legend()
plt.show()

# Step 4: Bivariate Regression (Price vs Area & Bedrooms)
X_bi = df[['area', 'bedrooms']]
X_train, X_test, y_train, y_test = train_test_split(X_bi, y, test_size=0.2, random_state=42)
model_bi = LinearRegression()
model_bi.fit(X_train, y_train)
y_pred_bi = model_bi.predict(X_test)

# Plot Bivariate Regression in 3D
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_test['area'], X_test['bedrooms'], y_test, color='blue', label='Actual Data')
ax.set_xlabel('Area')
ax.set_ylabel('Bedrooms')
ax.set_zlabel('Price')
ax.set_title('Bivariate Regression (Area & Bedrooms vs Price)')
plt.show()

# Step 5: Multivariate Regression (Using all features)
X_multi = df.drop(columns=['price'])
X_train, X_test, y_train, y_test = train_test_split(X_multi, y, test_size=0.2, random_state=42)
model_multi = LinearRegression()
model_multi.fit(X_train, y_train)
y_pred_multi = model_multi.predict(X_test)

# Model Evaluation
mse = mean_squared_error(y_test, y_pred_multi)
r2 = r2_score(y_test, y_pred_multi)
print(f"Multivariate Regression R² Score: {r2:.4f}")
print(f"Multivariate Regression MSE: {mse:.2f}")

# Residual Plot
residuals = y_test - y_pred_multi
plt.figure(figsize=(8,6))
sns.histplot(residuals, kde=True, color='purple')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residual Distribution - Multivariate Regression')
plt.show()
