import numpy as np
from typing import List, Tuple, Dict, Optional, Any
import pickle
import networkx as nx
import matplotlib.pyplot as plt

class ActivationFunctions:

    @staticmethod
    def linear(x : np.ndarray) -> np.ndarray:
        return x

    @staticmethod
    def linear_derivative(x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0,x)
    
    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1 , 0 )
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return  1 / (1 + np.exp(-x))

    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        sigmoid_x = ActivationFunctions.sigmoid(x)
        return sigmoid_x * (1 - sigmoid_x)

    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x: np.ndarray) -> np.ndarray:
        tanh_x = np.tanh(x)
        return 1 - tanh_x ** 2
    
    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        # Clip input untuk menghindari nilai ekstrem
        x = np.clip(x, -500, 500)
        
        # Shift input untuk stabilitas numerik
        shifted_x = x - np.max(x, axis=1, keepdims=True)
        
        # Hitung eksponen
        exp_x = np.exp(shifted_x)
        
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    @staticmethod
    def softmax_derivative(x: np.ndarray) -> np.ndarray:
        softmax_x = ActivationFunctions.softmax(x)
        return softmax_x * (1 - softmax_x) 
    
    # Bonus

    @staticmethod
    def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Leaky ReLU activation function: f(x) = max(alpha*x, x)"""
        return np.maximum(alpha * x, x)
    
    @staticmethod
    def leaky_relu_derivative(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Derivative of Leaky ReLU activation function: f'(x) = 1 if x > 0 else alpha"""
        return np.where(x > 0, 1, alpha)
    
    @staticmethod
    def elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """ELU activation function: f(x) = x if x > 0 else alpha * (exp(x) - 1)"""
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    
    @staticmethod
    def elu_derivative(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """Derivative of ELU activation function: f'(x) = 1 if x > 0 else alpha * exp(x)"""
        return np.where(x > 0, 1, alpha * np.exp(x))


class LossFunctions:

    @staticmethod
    def mse(y:np.ndarray, y_pred:np.ndarray):
        return np.mean(np.square(y - y_pred))
    
    @staticmethod
    def mse_derivative(y:np.ndarray, y_pred:np.ndarray):
        return 2 * (y_pred - y)/ y.shape[0]
    
    @staticmethod
    def binary_cross_entropy(y:np.ndarray, y_pred:np.ndarray, epsilon : float =1e-15):
        # clip to avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1-epsilon)
        return -np.mean(y * np.log(y_pred) + (1- y) * np.log(1-y_pred))
    
    @staticmethod
    def binary_cross_entropy_derivative(y:np.ndarray, y_pred:np.ndarray, epsilon: float = 1e-15):
        #clip to avoid division by zero
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -(y / y_pred - (1 - y) / (1 - y_pred)) / y.shape[0]
    
    @staticmethod
    def categorical_cross_entropy(y: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-15) -> float:
        # Clip to avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1.0-epsilon)
        return -np.mean(np.sum(y * np.log(y_pred), axis=1))
    
    @staticmethod
    def categorical_cross_entropy_derivative(y: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-15) -> np.ndarray:
        # Untuk softmax + cross-entropy, turunan adalah (y_pred - y)
        return (y_pred - y) / y.shape[0]

class WeightInitializers:

    @staticmethod
    def zero_initialization(shape: Tuple[int, int]) -> np.ndarray:
        return np.zeros(shape)
    
    @staticmethod
    def uniform_initialization(shape: Tuple[int,int], low:float = -0.1, high: float =0.1, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        return np.random.uniform(low=low,high=high,size=shape)
    
    @staticmethod
    def normal_initialization(shape: Tuple[int, int], mean: float = 0.0, var: float = 0.1, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        return np.random.normal(loc=mean, scale=np.sqrt(var), size=shape)
    
    # Bonus

    @staticmethod
    def xavier_initialization(shape: Tuple[int, int], seed: Optional[int] = None) -> np.ndarray:
        """
        Xavier (Glorot) initialization.
        Appropriate for tanh and sigmoid activation functions.
        """
        if seed is not None:
            np.random.seed(seed)
        fan_in, fan_out = shape
        limit = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(low=-limit, high=limit, size=shape)
    
    @staticmethod
    def he_initialization(shape: Tuple[int, int], seed: Optional[int] = None) -> np.ndarray:
        """
        He initialization.
        Appropriate for ReLU and variants.
        """
        if seed is not None:
            np.random.seed(seed)
        fan_in, fan_out = shape
        std = np.sqrt(2 / fan_in)
        return np.random.normal(loc=0.0, scale=std, size=shape)
    
class Layer:

    def __init__(
            self,
            n_inputs: int,
            n_neurons: int,
            activation: str='linear',
            weight_init: str='uniform',
            weight_init_params: Dict[str, Any] = None
    ):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.activation_name = activation

        self.set_activation_function(activation)

        if weight_init_params is None:
            weight_init_params = {}
        
        self.init_weights_and_biases(weight_init, weight_init_params)

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        # Cache for storing intermediate values during forward and backward pass
        self.cache = {}

    def set_activation_function(self, activation: str):
        activation_functions = {
            'linear': (ActivationFunctions.linear, ActivationFunctions.linear_derivative),
            'relu': (ActivationFunctions.relu, ActivationFunctions.relu_derivative),
            'sigmoid': (ActivationFunctions.sigmoid, ActivationFunctions.sigmoid_derivative),
            'tanh': (ActivationFunctions.tanh, ActivationFunctions.tanh_derivative),
            'softmax': (ActivationFunctions.softmax, ActivationFunctions.softmax_derivative),
            'leaky_relu': (ActivationFunctions.leaky_relu, ActivationFunctions.leaky_relu_derivative),
            'elu': (ActivationFunctions.elu, ActivationFunctions.elu_derivative)
        }

        if activation not in activation_functions:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        self.activation_func, self.activation_derivative = activation_functions[activation]
    
    def init_weights_and_biases(self, weight_init: str, params: Dict[str, Any]):
        shape = (self.n_inputs, self.n_neurons)
        
        if weight_init == 'zero':
            self.W = WeightInitializers.zero_initialization(shape)
        
        elif weight_init == 'uniform':
            low = params.get('low', -0.1)
            high = params.get('high', 0.1)
            seed = params.get('seed', None)
            self.W = WeightInitializers.uniform_initialization(shape, low, high, seed)
        
        elif weight_init == 'normal':
            mean = params.get('mean', 0.0)
            var = params.get('var', 0.1)
            seed = params.get('seed', None)
            self.W = WeightInitializers.normal_initialization(shape, mean, var, seed)
        
        elif weight_init == 'xavier':
            seed = params.get('seed', None)
            self.W = WeightInitializers.xavier_initialization(shape, seed)
        
        elif weight_init == 'he':
            seed = params.get('seed', None)
            self.W = WeightInitializers.he_initialization(shape, seed)
        
        else:
            raise ValueError(f"Unsupported weight initialization method: {weight_init}")
        
        # Initialize biases with zeros
        self.b = np.zeros((1, self.n_neurons))

    def forward(self, X: np.ndarray) -> np.ndarray:
        #  X: Input data, shape (batch_size, n_inputs)
        #  Output after activation, shape (batch_size, n_neurons)

        self.cache['X'] = X
        Z = np.dot(X, self.W) + self.b
        self.cache['Z'] = Z

        # apply activation
        A = self.activation_func(Z)
        self.cache['A'] = A

        return A
    
    def backward(self, dA: np.ndarray) -> np.ndarray:
        X = self.cache['X']
        Z = self.cache['Z']
        
        batch_size = X.shape[0]
        
        if self.activation_name == 'softmax':
            # Gunakan dA langsung sebagai dZ
            dZ = dA
        else:
            # Untuk aktivasi lain, gunakan turunan
            dZ = dA * self.activation_derivative(Z)
            # Cek nilai NaN/Inf
            if np.isnan(dZ).any() or np.isinf(dZ).any():
                print(f"Warning: dZ contains NaN/Inf values for {self.activation_name}")
                dZ = np.nan_to_num(dZ, nan=0.0, posinf=1e10, neginf=-1e10)
                dZ = np.clip(dZ, -1e10, 1e10)
        
        # Hitung gradien untuk weights dan biases
        self.dW = np.dot(X.T, dZ) / batch_size
        self.db = np.sum(dZ, axis=0, keepdims=True) / batch_size
        
        # Clip gradien untuk mencegah nilai ekstrem
        self.dW = np.clip(self.dW, -1e10, 1e10)
        self.db = np.clip(self.db, -1e10, 1e10)
        
        # Hitung gradien untuk layer sebelumnya
        dA_prev = np.dot(dZ, self.W.T)
        
        return dA_prev
    
    def update_weights(self, learning_rate: float, l1_lamda: float = 0.0, l2_lamda: float = 0.0):
        # apply regularization (optional)
        if l1_lamda > 0:
            dW_reg = l1_lamda * np.sign(self.W)
            self.dW += dW_reg
        
        if l2_lamda > 0:
            dW_reg = l2_lamda * self.W
            self.dW += dW_reg

        # updata weights and bias 
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db

    
class FeedforwardNeuralNetwork:
    def __init__(self, 
                input_size:int, 
                layer_sizes: List[int], 
                activations: List[str], 
                loss:str,
                weight_init: str = 'he',
                weight_init_params: Dict[str, Any] = None):
        if len(layer_sizes) != len(activations):
            raise ValueError("")
        
        self.input_size = input_size
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.loss_name = loss
        self.weight_init = weight_init
        
        if weight_init_params is None:
            self.weight_init_params = {}
        else:
            self.weight_init_params = weight_init_params

        self.set_loss_function(loss)

        self.layers= []
        
        self._build_network()

    def set_loss_function(self, loss:str):
        loss_functions = {
            'mse': (LossFunctions.mse, LossFunctions.mse_derivative),
            'binary_cross_entropy': (LossFunctions.binary_cross_entropy, LossFunctions.binary_cross_entropy_derivative),
            'categorical_cross_entropy': (LossFunctions.categorical_cross_entropy, LossFunctions.categorical_cross_entropy_derivative)
        }

        if loss not in loss_functions:
            raise ValueError(f"Unsupported loss function: {loss}")
        
        self.loss_func, self.loss_derivative = loss_functions[loss]

    def _build_network(self):
        prev_size = self.input_size

        for i, (size, activation) in enumerate(zip(self.layer_sizes, self.activations)):
            layer = Layer(
                n_inputs=prev_size, 
                n_neurons=size, 
                activation=activation,
                weight_init=self.weight_init,
                weight_init_params=self.weight_init_params
            )
            self.layers.append(layer)
            
            prev_size = size
    
    def forward(self, X:np.ndarray) -> np.ndarray:
        A = X

        for i, layer in enumerate(self.layers):
            A = layer.forward(A)
        return A
    
    def compute_loss(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        return self.loss_func(y,y_pred)

    def backward(self, y: np.ndarray, y_pred: np.ndarray) -> None:
        # initial gradient from loss function
        dA = self.loss_derivative(y,y_pred)
        # avoid extreme value
        dA = np.clip(dA, -1e10, 1e10)
        # backpropagate throgh layers
        for i in range(len(self.layers)-1, -1, -1):
            
            dA = self.layers[i].backward(dA)
    
    def update_weights(self, learning_rate:float, l1_lambda: float =0.0, l2_lambda: float=0.0):
        for layer in self.layers:
            layer.update_weights(learning_rate, l1_lambda, l2_lambda)
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        l1_lambda: float = 0.0,
        l2_lambda: float = 0.0,
        verbose: int = 1
    ) -> Dict[str, List[float]]:
        
        history = {
            'train_loss': [],
            'val_loss': []
        }

        n_samples = X_train.shape[0]
        n_batches = (n_samples +batch_size -1) // batch_size

        for epoch in range(epochs):
            indices =np.random.permutation(n_samples)
            X_shuffled =X_train[indices]
            y_shuffled =y_train[indices]

            epoch_loss = 0

            for batch in range(n_batches):
                start_idx =batch *batch_size
                end_idx = min((batch +1) * batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                # predict
                y_pred = self.forward(X_batch)

                # compute loss
                batch_loss = self.compute_loss(y_batch,y_pred)
                epoch_loss += batch_loss * (end_idx -start_idx)/ n_samples

                # learn
                self.backward(y_batch, y_pred)
                
                self.update_weights(learning_rate, l1_lambda, l2_lambda)

            # add regularization if used
            if l1_lambda > 0:
                for layer in self.layers:
                    epoch_loss += l1_lambda * np.sum(np.abs(layer.W)) / n_samples
            
            if l2_lambda > 0:
                for layer in self.layers:
                    epoch_loss += l2_lambda * 0.5 * np.sum(layer.W**2) / n_samples
            
            history['train_loss'].append(epoch_loss)

            # evaluate on validation set
            if X_val is not None and y_val is not None:
                val_pred = self.forward(X_val)
                val_loss = self.compute_loss(y_val, val_pred)
                
                # Add regularization if used
                if l1_lambda > 0:
                    for layer in self.layers:
                        val_loss += l1_lambda * np.sum(np.abs(layer.W)) / X_val.shape[0]
                
                if l2_lambda > 0:
                    for layer in self.layers:
                        val_loss += l2_lambda * 0.5 * np.sum(layer.W**2) / X_val.shape[0]
                
                history['val_loss'].append(val_loss)
            
            # print progress
            if verbose == 1:
                progress = (epoch + 1) / epochs * 100
                if X_val is not None and y_val is not None:
                    print(f"Epoch {epoch+1}/{epochs} [{progress:.1f}%] - train_loss: {epoch_loss:.4f} - val_loss: {history['val_loss'][-1]:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs} [{progress:.1f}%] - train_loss: {epoch_loss:.4f}")
        
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.forward(X)
        return self.compute_loss(y,y_pred)
    
    def save(self, filename: str):
        model_data = {
            'input_size': self.input_size,
            'layer_sizes': self.layer_sizes,
            'activations': self.activations,
            'loss': self.loss_name,
            'layers': self.layers,
            'use_rms_norm': self.use_rms_norm,
            'rms_norm_layers': self.rms_norm_layers
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, filename: str):
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create a new model with the same architecture
        model = cls(
            input_size=model_data['input_size'],
            layer_sizes=model_data['layer_sizes'],
            activations=model_data['activations'],
            loss=model_data['loss']
        )
        
        # Load layers
        model.layers = model_data['layers']
        model.use_rms_norm = model_data['use_rms_norm']
        model.rms_norm_layers = model_data['rms_norm_layers']
        
        return model
    
    def plot_model(self):
        """Plot the model as a graph with weights and gradients."""
        G = nx.DiGraph()
        
        # Add input layer nodes
        for i in range(self.input_size):
            G.add_node(f"Input {i}", layer="Input", pos=(0, -i))
        
        # Add hidden and output layer nodes
        layer_positions = []
        current_pos = 2
        
        for i, layer_size in enumerate(self.layer_sizes):
            layer_positions.append(current_pos)
            for j in range(layer_size):
                layer_name = "Output" if i == len(self.layer_sizes) - 1 else f"Hidden {i+1}"
                G.add_node(f"{layer_name} {j}", layer=layer_name, pos=(current_pos, -j))
            
            current_pos += 2
        
        # Add edges with weights and gradients
        prev_size = self.input_size
        prev_layer_name = "Input"
        
        for i, layer in enumerate(self.layers):
            current_layer_name = "Output" if i == len(self.layer_sizes) - 1 else f"Hidden {i+1}"
            
            for j in range(prev_size):
                for k in range(layer.n_neurons):
                    weight = layer.W[j, k]
                    gradient = layer.dW[j, k]
                    G.add_edge(
                        f"{prev_layer_name} {j}",
                        f"{current_layer_name} {k}",
                        weight=weight,
                        gradient=gradient
                    )
            
            prev_size = layer.n_neurons
            prev_layer_name = current_layer_name
        
        # Create a figure
        plt.figure(figsize=(12, 10))
        
        # Generate positions for drawing
        pos = nx.get_node_attributes(G, 'pos')
        
        # Draw nodes
        layer_colors = {"Input": "lightblue", "Output": "lightgreen"}
        for layer_name in set(nx.get_node_attributes(G, 'layer').values()):
            node_list = [node for node, data in G.nodes(data=True) if data['layer'] == layer_name]
            color = layer_colors.get(layer_name, "lightpink")
            nx.draw_networkx_nodes(G, pos, nodelist=node_list, node_color=color, node_size=500)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos)
        
        # Draw edges with varying thickness and color based on weights
        for u, v, data in G.edges(data=True):
            weight = data['weight']
            gradient = data['gradient']
            
            # Normalize weight for edge width
            width = abs(weight) * 3
            
            # Determine color based on weight sign (red for negative, blue for positive)
            color = 'red' if weight < 0 else 'blue'
            
            # Draw edge
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[(u, v)],
                width=width,
                edge_color=color,
                alpha=0.6
            )
        
        plt.title("Neural Network Architecture with Weights and Gradients")
        plt.axis('off')
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_weight_distribution(self, layers: List[int]):
        
        if not layers:
            layers = list(range(len(self.layers)))
        
        num_layers = len(layers)
        fig, axes = plt.subplots(num_layers, 1, figsize=(10, 3*num_layers))
        
        # Handle single layer case
        if num_layers == 1:
            axes = [axes]
        
        for i, layer_idx in enumerate(layers):
            if layer_idx < 0 or layer_idx >= len(self.layers):
                raise ValueError(f"Layer index {layer_idx} out of range")
            
            layer = self.layers[layer_idx]
            weights = layer.W.flatten()
            
            # Plot histogram
            axes[i].hist(weights, bins=50, alpha=0.7)
            axes[i].set_title(f"Layer {layer_idx+1} Weight Distribution")
            axes[i].set_xlabel("Weight Value")
            axes[i].set_ylabel("Frequency")
            
            # Add mean and std statistics
            mean = np.mean(weights)
            std = np.std(weights)
            axes[i].axvline(mean, color='r', linestyle='dashed', linewidth=1, label=f'Mean: {mean:.4f}')
            axes[i].axvline(mean + std, color='g', linestyle='dashed', linewidth=1, label=f'Mean ± Std: {std:.4f}')
            axes[i].axvline(mean - std, color='g', linestyle='dashed', linewidth=1)
            axes[i].legend()
        
        plt.tight_layout()
        return fig
    
    def plot_gradient_distribution(self, layers: List[int]):
        
        if not layers:
            layers = list(range(len(self.layers)))
        
        num_layers = len(layers)
        fig, axes = plt.subplots(num_layers, 1, figsize=(10, 3*num_layers))
        
        # Handle single layer case
        if num_layers == 1:
            axes = [axes]
        
        for i, layer_idx in enumerate(layers):
            if layer_idx < 0 or layer_idx >= len(self.layers):
                raise ValueError(f"Layer index {layer_idx} out of range")
            
            layer = self.layers[layer_idx]
            gradients = layer.dW.flatten()
            
            # Plot histogram
            axes[i].hist(gradients, bins=50, alpha=0.7)
            axes[i].set_title(f"Layer {layer_idx+1} Gradient Distribution")
            axes[i].set_xlabel("Gradient Value")
            axes[i].set_ylabel("Frequency")
            
            # Add mean and std statistics
            mean = np.mean(gradients)
            std = np.std(gradients)
            axes[i].axvline(mean, color='r', linestyle='dashed', linewidth=1, label=f'Mean: {mean:.4f}')
            axes[i].axvline(mean + std, color='g', linestyle='dashed', linewidth=1, label=f'Mean ± Std: {std:.4f}')
            axes[i].axvline(mean - std, color='g', linestyle='dashed', linewidth=1)
            axes[i].legend()
        
        plt.tight_layout()
        return fig