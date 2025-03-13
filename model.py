import numpy as np

class FFNN:
    def __init__(self, layers, activations, weight_init='random_uniform', seed=None):
        np.random.seed(seed)
        self.layers = layers
        self.activations = activations
        self.weights = []
        self.biases = []
        self.initialize_weights(weight_init, seed)
    
    def initialize_weights(self, method, seed):
        for i in range(len(self.layers) - 1):
            input_size = self.layers[i]
            output_size = self.layers[i + 1]
            if method == 'zero':
                W = np.zeros((output_size, input_size))
                b = np.zeros((output_size, 1))
            elif method == 'random_uniform':
                W = np.random.uniform(-1, 1, (output_size, input_size))
                b = np.random.uniform(-1, 1, (output_size, 1))
            elif method == 'random_normal':
                W = np.random.randn(output_size, input_size)
                b = np.random.randn(output_size, 1)
            self.weights.append(W)
            self.biases.append(b)
    
    def activation_function(self, x, func):
        if func == 'linear':
            return x
        elif func == 'relu':
            return np.maximum(0, x)
        elif func == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif func == 'tanh':
            return np.tanh(x)
        elif func == 'softmax':
            exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
            return exp_x / np.sum(exp_x, axis=0, keepdims=True)
    
    def activation_derivative(self, x, func):
        if func == 'linear':
            return np.ones_like(x)
        elif func == 'relu':
            return (x > 0).astype(float)
        elif func == 'sigmoid':
            sig = self.activation_function(x, 'sigmoid')
            return sig * (1 - sig)
        elif func == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif func == 'softmax':
            return x * (1 - x)
    
    def forward(self, X):
        activations = []
        inputs = X
        for W, b, func in zip(self.weights, self.biases, self.activations):
            Z = np.dot(W, inputs) + b
            inputs = self.activation_function(Z, func)
            activations.append(inputs)
        return activations
    
    def compute_loss(self, Y_pred, Y_true, loss_function):
        if loss_function == 'mse':
            return np.mean((Y_pred - Y_true) ** 2)
        elif loss_function == 'binary_cross_entropy':
            return -np.mean(Y_true * np.log(Y_pred + 1e-9) + (1 - Y_true) * np.log(1 - Y_pred + 1e-9))
        elif loss_function == 'categorical_cross_entropy':
            return -np.sum(Y_true * np.log(Y_pred + 1e-9)) / Y_true.shape[1]
    
    def backward(self, X, Y_true, learning_rate, loss_function):
        activations = self.forward(X)
        grads_W = []
        grads_b = []
        Y_pred = activations[-1]
        
        dA = Y_pred - Y_true if loss_function == 'mse' else (Y_pred - Y_true) / Y_true.shape[1]
        
        for i in reversed(range(len(self.weights))):
            dZ = dA * self.activation_derivative(activations[i], self.activations[i])
            dW = np.dot(dZ, activations[i-1].T) if i > 0 else np.dot(dZ, X.T)
            db = np.sum(dZ, axis=1, keepdims=True)
            
            grads_W.insert(0, dW)
            grads_b.insert(0, db)
            dA = np.dot(self.weights[i].T, dZ)
            
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db
    
    def train(self, X, Y, epochs, learning_rate, loss_function):
        for epoch in range(epochs):
            self.backward(X, Y, learning_rate, loss_function)
            if epoch % 10 == 0:
                loss = self.compute_loss(self.forward(X)[-1], Y, loss_function)
                print(f'Epoch {epoch}, Loss: {loss}')
    
    def predict(self, X):
        return self.forward(X)[-1]

# Contoh penggunaan
if __name__ == "__main__":
    X_train = np.random.rand(3, 10)  # 3 input neurons, 10 samples
    Y_train = np.random.randint(0, 2, (1, 10))  # Binary classification
    
    ffnn = FFNN(layers=[3, 5, 1], activations=['relu', 'sigmoid'], weight_init='random_uniform', seed=42)
    ffnn.train(X_train, Y_train, epochs=100, learning_rate=0.01, loss_function='binary_cross_entropy')
    
    print("Predictions:", ffnn.predict(X_train))