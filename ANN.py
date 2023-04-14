import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights with random values
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.weights2 = np.random.randn(hidden_size, output_size)

        # Initialize biases with zeros
        self.biases1 = np.zeros(hidden_size)
        self.biases2 = np.zeros(output_size)

    def forward(self, input_data):
        # Perform forward propagation
        hidden_layer = np.dot(input_data, self.weights1) + self.biases1
        hidden_layer_activation = self.sigmoid(hidden_layer)
        output_layer = np.dot(hidden_layer_activation, self.weights2) + self.biases2
        return output_layer

    def sigmoid(self, x):
        # Apply sigmoid activation function
        return 1 / (1 + np.exp(-x))

    def train(self, input_data, output_data, learning_rate, num_epochs):
        # Train the network using backpropagation
        for epoch in range(num_epochs):
            # Forward propagation
            hidden_layer = np.dot(input_data, self.weights1) + self.biases1
            hidden_layer_activation = self.sigmoid(hidden_layer)
            output_layer = np.dot(hidden_layer_activation, self.weights2) + self.biases2

            # Calculate loss (mean squared error)
            error = output_data - output_layer
            loss = np.mean(error**2)

            # Backpropagation
            output_layer_error = error
            output_layer_activation_gradient = output_layer * (1 - output_layer)
            output_layer_delta = output_layer_error * output_layer_activation_gradient
            hidden_layer_error = np.dot(output_layer_delta, self.weights2.T)
            hidden_layer_activation_gradient = hidden_layer_activation * (1 - hidden_layer_activation)
            hidden_layer_delta = hidden_layer_error * hidden_layer_activation_gradient

            # Update weights and biases
            self.weights2 += learning_rate * np.dot(hidden_layer_activation.T, output_layer_delta)
            self.biases2 += learning_rate * np.sum(output_layer_delta, axis=0)
            self.weights1 += learning_rate * np.dot(input_data.T, hidden_layer_delta)
            self.biases1 += learning_rate * np.sum(hidden_layer_delta, axis=0)

            # Print progress
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

# Example usage
input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
output_data = np.array([[0], [1], [1], [0]])
nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
nn.train(input_data, output_data, learning_rate=0.1, num_epochs=1000)
print(nn.forward(input_data))


# OPTION 2

import numpy as np

class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Initialize weights
        self.weights1 = np.random.randn(self.input_dim, self.hidden_dim)
        self.weights2 = np.random.randn(self.hidden_dim, self.output_dim)
        
        # Initialize biases
        self.bias1 = np.zeros((1, self.hidden_dim))
        self.bias2 = np.zeros((1, self.output_dim))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, X):
        # Compute input to hidden layer
        self.hidden = self.sigmoid(np.dot(X, self.weights1) + self.bias1)
        
        # Compute hidden to output layer
        self.output = np.dot(self.hidden, self.weights2) + self.bias2
        
        return self.output
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def backward(self, X, y, output):
        # Compute derivative of mean squared error loss function
        error = output - y
        
        # Compute derivative of output layer
        d_output = error
        d_weights2 = np.dot(self.hidden.T, d_output)
        d_bias2 = np.sum(d_output, axis=0, keepdims=True)
        
        # Compute derivative of hidden layer
        d_hidden = np.dot(d_output, self.weights2.T) * self.sigmoid_derivative(self.hidden)
        d_weights1 = np.dot(X.T, d_hidden)
        d_bias1 = np.sum(d_hidden, axis=0, keepdims=True)
        
        # Update weights and biases
        self.weights1 -= 0.1 * d_weights1
        self.bias1 -= 0.1 * d_bias1
        self.weights2 -= 0.1 * d_weights2
        self.bias2 -= 0.1 * d_bias2
        
    def train(self, X, y):
        output = self.forward(X)
        self.backward(X, y, output)

# Option 2 is not complete to be updated 

# OPTION 3 is the best so far 

import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.b2 = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def forward(self, X):
        # Input to hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)

        # Hidden to output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)

        return self.a2

    def backward(self, X, y, output):
        # Output to hidden layer
        self.delta2 = (output - y) * self.sigmoid_derivative(self.z2)
        self.dW2 = np.dot(self.a1.T, self.delta2)
        self.db2 = np.sum(self.delta2, axis=0, keepdims=True)

        # Hidden to input layer
        self.delta1 = np.dot(self.delta2, self.W2.T) * self.relu_derivative(self.z1)
        self.dW1 = np.dot(X.T, self.delta1)
        self.db1 = np.sum(self.delta1, axis=0)

    def optimize(self, learning_rate):
        # Gradient descent
        self.W1 -= learning_rate * self.dW1
        self.b1 -= learning_rate * self.db1
        self.W2 -= learning_rate * self.dW2
        self.b2 -= learning_rate * self.db2

    def train(self, X, y, learning_rate=0.01, num_epochs=1000):
        for epoch in range(num_epochs):
            # Forward propagation
            output = self.forward(X)

            # Backward propagation
            self.backward(X, y, output)

            # Gradient descent
            self.optimize(learning_rate)

            # Print error
            if epoch % 100 == 0:
                loss = np.mean(np.square(output - y))
                print(f"Epoch {epoch}, loss: {loss}")

    def predict(self, X):
        return self.forward(X)


