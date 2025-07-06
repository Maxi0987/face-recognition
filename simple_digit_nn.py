import numpy as np

class NeuralNetwork:
    def __init__(self, input_size=784, hidden_sizes=[128, 64], output_size=10, lr=0.01):
        self.lr = lr
        # initialize weights with small random values
        self.weights = []
        self.biases = []
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(layer_sizes) - 1):
            weight = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01
            bias = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(weight)
            self.biases.append(bias)

    def forward(self, X):
        """Perform forward pass and store activations"""
        activations = [X]
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            X = self.sigmoid(np.dot(X, W) + b)
            activations.append(X)
        # last layer uses softmax
        logits = np.dot(X, self.weights[-1]) + self.biases[-1]
        out = self.softmax(logits)
        activations.append(out)
        return activations

    def backward(self, activations, y_true):
        """Compute gradients using backpropagation"""
        grads_W = [None] * len(self.weights)
        grads_b = [None] * len(self.biases)

        y_pred = activations[-1]
        m = y_true.shape[0]
        delta = (y_pred - y_true) / m
        grads_W[-1] = np.dot(activations[-2].T, delta)
        grads_b[-1] = np.sum(delta, axis=0, keepdims=True)

        for i in reversed(range(len(self.weights)-1)):
            delta = np.dot(delta, self.weights[i+1].T) * self.sigmoid_deriv(activations[i+1])
            grads_W[i] = np.dot(activations[i].T, delta)
            grads_b[i] = np.sum(delta, axis=0, keepdims=True)
        return grads_W, grads_b

    def update_params(self, grads_W, grads_b):
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * grads_W[i]
            self.biases[i] -= self.lr * grads_b[i]

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_deriv(a):
        return a * (1 - a)

    @staticmethod
    def softmax(z):
        e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return e_z / np.sum(e_z, axis=1, keepdims=True)

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        for epoch in range(epochs):
            permutation = np.random.permutation(X_train.shape[0])
            X_shuffled = X_train[permutation]
            y_shuffled = y_train[permutation]
            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                activations = self.forward(X_batch)
                grads_W, grads_b = self.backward(activations, y_batch)
                self.update_params(grads_W, grads_b)
            loss = self.compute_loss(X_train, y_train)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

    def predict(self, X):
        activations = self.forward(X)
        return np.argmax(activations[-1], axis=1)

    def compute_loss(self, X, y_true):
        y_pred = self.forward(X)[-1]
        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / m
        return loss


def one_hot_encode(labels, num_classes=10):
    encoded = np.zeros((labels.size, num_classes))
    encoded[np.arange(labels.size), labels] = 1
    return encoded

# Example usage (pseudo-code):
# from sklearn.datasets import fetch_openml
# mnist = fetch_openml('mnist_784', version=1)
# X = mnist.data / 255.0
# y = one_hot_encode(mnist.target.astype(int))
# nn = NeuralNetwork()
# nn.train(X, y)
