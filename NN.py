import numpy as np

# Define the neural network
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        # Initialize weights and biases
        self.learning_rate = learning_rate
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.01
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size) * 0.01
        self.bias2 = np.zeros((1, output_size))
    
    def forward(self, state):
        # Forward pass
        self.z1 = np.dot(state, self.weights1) + self.bias1
        self.a1 = np.maximum(0, self.z1)  # ReLU activation
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        return self.z2  # Linear output
    
    def backward(self, state, target, output):
        # Compute loss gradient
        error = output - target
        d_weights2 = np.dot(self.a1.T, error)
        d_bias2 = np.sum(error, axis=0, keepdims=True)
        
        hidden_error = np.dot(error, self.weights2.T)
        hidden_error[self.z1 <= 0] = 0  # Derivative of ReLU
        
        d_weights1 = np.dot(state.T, hidden_error)
        d_bias1 = np.sum(hidden_error, axis=0, keepdims=True)
        
        # Update weights and biases
        self.weights2 -= self.learning_rate * d_weights2
        self.bias2 -= self.learning_rate * d_bias2
        self.weights1 -= self.learning_rate * d_weights1
        self.bias1 -= self.learning_rate * d_bias1