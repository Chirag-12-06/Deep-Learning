"""Single-layer perceptron implementation for binary classification."""
import numpy as np

class Perceptron:
    """Perceptron model for binary classification."""
    def __init__(self, lr=0.1, n_iterations=1000):
        self.lr = lr
        self.epochs = n_iterations
        self.weights = None
        self.bias = None

    def Step_activation_func(self, x):
        return 1 if x >= 0 else 0

    def fit(self, X, Y):
        # Defining the shape of weight and bias.
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        # training the model on X_train and Y_train
        for epoch in range(self.epochs):
            for i in range(X.shape[0]):
                # Deciding the activation function
                Y_pred = self.Step_activation_func(np.dot(self.weights, X[i]) + self.bias)
                # Deciding the loss function
                mae = Y[i] - Y_pred
                # Updating the weight and bias using optimization algorithm
                self.weights = self.weights + self.lr * mae * X[i]
                self.bias = self.bias + self.lr * mae
        return self

    def predict(self, x_input):
        net_input = np.dot(x_input, self.weights) + self.bias
        return np.array([self.Step_activation_func(x) for x in net_input])


# Training data for logic gates
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# # AND gate
# print("AND Gate Training")
# y_and = np.array([0, 0, 0, 1])
# p_and = Perceptron(lr=0.1, n_iterations=1000)
# p_and.fit(X, y_and)
# predictions = p_and.predict(X)
# print(f"Weights: {p_and.weights}, Bias: {p_and.bias}")
# print(f"Input: {X.tolist()}")
# print(f"Expected: {y_and}")
# print(f"Predicted: {predictions}")
# print(f"Accuracy: {np.mean(predictions == y_and) * 100:.2f}%\n")

# OR gate
print("OR Gate Training")
y_or = np.array([0, 1, 1, 1])
p_or = Perceptron(lr=0.1, n_iterations=1000)
p_or.fit(X, y_or)
predictions = p_or.predict(X)
print(f"Weights: {p_or.weights}, Bias: {p_or.bias}")
print(f"Input: {X.tolist()}")
print(f"Expected: {y_or}")
print(f"Predicted: {predictions}")
print(f"Accuracy: {np.mean(predictions == y_or) * 100:.2f}%\n")

# # XOR gate (This will fail - XOR is not linearly separable)
# print("XOR Gate Training (Will FAIL - Not Linearly Separable)")
# y_xor = np.array([0, 1, 1, 0])
# p_xor = Perceptron(lr=0.1, n_iterations=1000)
# p_xor.fit(X, y_xor)
# predictions = p_xor.predict(X)
# print(f"Weights: {p_xor.weights}, Bias: {p_xor.bias}")
# print(f"Input: {X.tolist()}")
# print(f"Expected: {y_xor}")
# print(f"Predicted: {predictions}")
# print(f"Accuracy: {np.mean(predictions == y_xor) * 100:.2f}%")
# print("\nNote: Single-layer perceptron cannot learn XOR!")
# print("XOR requires a multi-layer perceptron (MLP) or non-linear decision boundary.")
