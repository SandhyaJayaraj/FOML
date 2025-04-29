import numpy as np

# Step 1: Initialize input features (X) and target labels (y)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Inputs
y = np.array([0, 0, 0, 1])  # AND logic gate output

# Step 2: Initialize weights and bias
weights = np.random.rand(2)
bias = np.random.rand(1)

learning_rate = 0.1

# Step 3: Define activation function (step function)
def step_function(x):
    return 1 if x >= 0 else 0

# Step 4: Train the perceptron using the Perceptron Learning Algorithm
epochs = 10
for epoch in range(epochs):
    for i in range(len(X)):
        # Step 5: Compute weighted sum
        weighted_sum = np.dot(X[i], weights) + bias

        # Step 6: Apply activation function
        y_pred = step_function(weighted_sum)

        # Step 7: Compute error
        error = y[i] - y_pred

        # Step 8: Update weights and bias
        weights += learning_rate * error * X[i]
        bias += learning_rate * error

# Step 9: Make predictions
for i in range(len(X)):
    output = step_function(np.dot(X[i], weights) + bias)
    print(f"Input: {X[i]}, Predicted Output: {output}")

# Step 10: Final weights and bias
print("Final Weights:", weights)
print("Final Bias:", bias)
