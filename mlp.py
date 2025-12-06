"""
================================================================================
MULTI-LAYER PERCEPTRON (MLP) MODULE
================================================================================
Complete implementation of a feedforward neural network from scratch using
only NumPy. This module covers the full deep learning pipeline:

1. Network Architecture Design
2. Weight Initialization (Xavier/He)
3. Forward Propagation
4. Loss Computation (Cross-Entropy)
5. Backpropagation (Gradient Computation)
6. Gradient Descent Optimization (with mini-batch support)

Mathematical Notation:
    L       : Total number of layers (including input)
    n^[l]   : Number of neurons in layer l
    W^[l]   : Weight matrix for layer l, shape (n^[l], n^[l-1])
    b^[l]   : Bias vector for layer l, shape (n^[l], 1)
    Z^[l]   : Pre-activation for layer l: Z = W · A^[l-1] + b
    A^[l]   : Activation output for layer l: A = f(Z)
    A^[0]   : Input data X
    A^[L]   : Final output (predictions)

================================================================================
"""

import numpy as np
from activations import ActivationFunctions


class MLP:
    """
    Multi-Layer Perceptron (Feedforward Neural Network)
    
    A fully-connected neural network that can learn complex non-linear
    patterns through multiple layers of neurons.
    
    Architecture Example (layer_sizes = [4, 10, 8, 3]):
    
        Input Layer      Hidden Layer 1    Hidden Layer 2    Output Layer
        (4 neurons)      (10 neurons)      (8 neurons)       (3 neurons)
        
            [x₁]              [h₁]              [h₁]             [ŷ₁]
            [x₂]  ────W¹────  [h₂]  ────W²────  [h₂]  ────W³────  [ŷ₂]
            [x₃]              ...               ...               [ŷ₃]
            [x₄]              [h₁₀]             [h₈]
    
    Attributes:
        layer_sizes (list): Architecture [n_input, n_hidden1, ..., n_output]
        learning_rate (float): Step size for gradient descent
        activation (str): Activation function for hidden layers
        weights (dict): W^[l] matrices for each layer
        biases (dict): b^[l] vectors for each layer
        loss_history (list): Training loss per epoch
        accuracy_history (list): Training accuracy per epoch
    """
    
    def __init__(self, layer_sizes, learning_rate=0.01, activation='relu', 
                 weight_init='he', seed=42):
        """
        Initialize the Multi-Layer Perceptron.
        
        Args:
            layer_sizes (list): Network architecture, e.g., [784, 128, 64, 10]
            learning_rate (float): Learning rate α (default: 0.01)
            activation (str): Hidden layer activation ('relu', 'sigmoid', 'tanh')
            weight_init (str): Weight initialization ('xavier', 'he', 'random')
            seed (int): Random seed for reproducibility
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.activation_name = activation
        self.weight_init = weight_init
        self.L = len(layer_sizes) - 1
        
        np.random.seed(seed)
        
        self.af = ActivationFunctions()
        self.activation = self.af.get_activation(activation)
        self.activation_derivative = self.af.get_activation_derivative(activation)
        
        self.weights = {}
        self.biases = {}
        self._initialize_weights()
        
        self.loss_history = []
        self.accuracy_history = []
        
        self.epsilon = 1e-15
    
    def _initialize_weights(self):
        """
        Initialize weights and biases for all layers.
        
        Weight Initialization Strategies:
        
        1. Random (Naive):
            W ~ N(0, 0.01)
            Problem: Can cause vanishing/exploding gradients
            
        2. Xavier/Glorot Initialization:
            W ~ N(0, sqrt(1/n_in))
            Best for: Sigmoid, Tanh activations
            Rationale: Keeps variance constant across layers
            
        3. He Initialization:
            W ~ N(0, sqrt(2/n_in))
            Best for: ReLU activations
            Rationale: Accounts for ReLU zeroing half the values
        
        Bias Initialization:
            b = 0 (zeros)
            Small positive values can help ReLU avoid dead neurons
        """
        for l in range(1, self.L + 1):
            n_in = self.layer_sizes[l - 1]
            n_out = self.layer_sizes[l]
            
            if self.weight_init == 'xavier':
                scale = np.sqrt(1.0 / n_in)
            elif self.weight_init == 'he':
                scale = np.sqrt(2.0 / n_in)
            else:
                scale = 0.01
            
            self.weights[l] = np.random.randn(n_out, n_in) * scale
            
            self.biases[l] = np.zeros((n_out, 1))
    
    def forward_propagation(self, X):
        """
        Perform forward pass through the network.
        
        Algorithm:
            For each layer l = 1, 2, ..., L:
                Z^[l] = W^[l] · A^[l-1] + b^[l]
                A^[l] = f^[l](Z^[l])
            
            Where:
                - A^[0] = X (input data)
                - f^[l] = activation function (ReLU for hidden, Softmax for output)
        
        Matrix Dimensions (for m samples):
            X:    (n_features, m)
            W^[l]: (n^[l], n^[l-1])
            b^[l]: (n^[l], 1) - broadcast across m samples
            Z^[l]: (n^[l], m)
            A^[l]: (n^[l], m)
        
        Args:
            X (np.ndarray): Input data, shape (n_features, m_samples)
            
        Returns:
            tuple: (activations, pre_activations)
                - activations: dict {0: X, 1: A^[1], ..., L: A^[L]}
                - pre_activations: dict {1: Z^[1], ..., L: Z^[L]}
        """
        A = {0: X}
        Z = {}
        
        for l in range(1, self.L + 1):
            Z[l] = np.dot(self.weights[l], A[l-1]) + self.biases[l]
            
            if l == self.L:
                A[l] = self.af.softmax(Z[l])
            else:
                A[l] = self.activation(Z[l])
        
        return A, Z
    
    def compute_loss(self, Y_pred, Y_true):
        """
        Compute Cross-Entropy Loss.
        
        Mathematical Formula (for m samples):
            J = -(1/m) × Σᵢ Σⱼ (Y_true[j,i] × log(Y_pred[j,i]))
            
        For one-hot encoded labels, this simplifies to:
            J = -(1/m) × Σᵢ log(Y_pred[y_true[i], i])
        
        Why Cross-Entropy?
            - Natural pairing with softmax (simplifies gradients)
            - Heavily penalizes confident wrong predictions
            - Produces well-calibrated probabilities
        
        Numerical Stability:
            - Clip predictions to avoid log(0)
            - Y_pred = clip(Y_pred, ε, 1-ε) where ε = 1e-15
        
        Args:
            Y_pred (np.ndarray): Predicted probabilities, shape (n_classes, m)
            Y_true (np.ndarray): One-hot encoded labels, shape (n_classes, m)
            
        Returns:
            float: Average cross-entropy loss
        """
        m = Y_true.shape[1]
        
        Y_pred_clipped = np.clip(Y_pred, self.epsilon, 1 - self.epsilon)
        
        loss = -np.sum(Y_true * np.log(Y_pred_clipped)) / m
        
        return loss
    
    def backward_propagation(self, X, Y, A, Z):
        """
        Compute gradients using backpropagation.
        
        Backpropagation Algorithm (Chain Rule):
        =========================================
        
        Output Layer (layer L):
            dZ^[L] = A^[L] - Y
            (This elegant form comes from combining softmax derivative with
             cross-entropy loss derivative)
            
            dW^[L] = (1/m) × dZ^[L] · A^[L-1]ᵀ
            db^[L] = (1/m) × Σ(dZ^[L]) along axis 1
        
        Hidden Layers (layer l = L-1, L-2, ..., 1):
            dA^[l] = W^[l+1]ᵀ · dZ^[l+1]
            dZ^[l] = dA^[l] ⊙ f'(Z^[l])  (⊙ = element-wise product)
            dW^[l] = (1/m) × dZ^[l] · A^[l-1]ᵀ
            db^[l] = (1/m) × Σ(dZ^[l]) along axis 1
        
        Gradient Flow Visualization:
            Loss → dA^[L] → dZ^[L] → dW^[L], db^[L]
                          ↓
                    → dA^[L-1] → dZ^[L-1] → dW^[L-1], db^[L-1]
                              ↓
                        → ... → dZ^[1] → dW^[1], db^[1]
        
        Args:
            X (np.ndarray): Input data, shape (n_features, m)
            Y (np.ndarray): One-hot encoded labels, shape (n_classes, m)
            A (dict): Activations from forward pass {0: X, 1: A^[1], ...}
            Z (dict): Pre-activations from forward pass {1: Z^[1], ...}
            
        Returns:
            tuple: (dW, db) - gradients for weights and biases
                - dW: dict {1: dW^[1], ..., L: dW^[L]}
                - db: dict {1: db^[1], ..., L: db^[L]}
        """
        m = X.shape[1]
        
        dW = {}
        db = {}
        
        dZ = A[self.L] - Y
        dW[self.L] = np.dot(dZ, A[self.L - 1].T) / m
        db[self.L] = np.sum(dZ, axis=1, keepdims=True) / m
        
        for l in range(self.L - 1, 0, -1):
            dA = np.dot(self.weights[l + 1].T, dZ)
            
            if self.activation_name in ['sigmoid', 'tanh']:
                dZ = dA * self.activation_derivative(A[l])
            else:
                dZ = dA * self.activation_derivative(Z[l])
            
            dW[l] = np.dot(dZ, A[l - 1].T) / m
            db[l] = np.sum(dZ, axis=1, keepdims=True) / m
        
        return dW, db
    
    def update_parameters(self, dW, db):
        """
        Update weights and biases using gradient descent.
        
        Update Rule:
            W^[l] := W^[l] - α × dW^[l]
            b^[l] := b^[l] - α × db^[l]
            
        Where α (alpha) is the learning rate.
        
        Learning Rate Guidelines:
            - Too high: Overshooting, oscillation, divergence
            - Too low: Very slow convergence
            - Typical values: 0.001, 0.01, 0.1
        
        Args:
            dW (dict): Weight gradients for each layer
            db (dict): Bias gradients for each layer
        """
        for l in range(1, self.L + 1):
            self.weights[l] -= self.learning_rate * dW[l]
            self.biases[l] -= self.learning_rate * db[l]
    
    def fit(self, X, y, epochs=1000, batch_size=32, verbose=True, 
            validation_data=None):
        """
        Train the neural network using mini-batch gradient descent.
        
        Training Process:
            For each epoch:
                1. Shuffle training data
                2. Split into mini-batches
                3. For each mini-batch:
                    a. Forward propagation
                    b. Compute loss
                    c. Backward propagation (compute gradients)
                    d. Update parameters
                4. Record loss and accuracy
        
        Gradient Descent Variants:
            - Batch GD (batch_size = m): Most accurate, slowest
            - Mini-Batch GD (typical: 32, 64, 128): Good balance
            - Stochastic GD (batch_size = 1): Fastest, noisiest
        
        Args:
            X (np.ndarray): Training features, shape (m_samples, n_features)
            y (np.ndarray): Training labels, shape (m_samples,)
            epochs (int): Number of training iterations
            batch_size (int): Mini-batch size
            verbose (bool): Print training progress
            validation_data (tuple): Optional (X_val, y_val) for validation
            
        Returns:
            dict: Training history with loss and accuracy per epoch
        """
        X = X.T
        
        n_classes = len(np.unique(y))
        Y = self._one_hot_encode(y, n_classes)
        
        m = X.shape[1]
        
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val = X_val.T
            Y_val = self._one_hot_encode(y_val, n_classes)
        
        self.loss_history = []
        self.accuracy_history = []
        self.val_loss_history = []
        self.val_accuracy_history = []
        
        for epoch in range(epochs):
            permutation = np.random.permutation(m)
            X_shuffled = X[:, permutation]
            Y_shuffled = Y[:, permutation]
            
            epoch_loss = 0
            n_batches = 0
            
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[:, i:i + batch_size]
                Y_batch = Y_shuffled[:, i:i + batch_size]
                
                A, Z = self.forward_propagation(X_batch)
                
                batch_loss = self.compute_loss(A[self.L], Y_batch)
                epoch_loss += batch_loss
                n_batches += 1
                
                dW, db = self.backward_propagation(X_batch, Y_batch, A, Z)
                
                self.update_parameters(dW, db)
            
            epoch_loss /= n_batches
            
            A_full, _ = self.forward_propagation(X)
            train_accuracy = self._compute_accuracy(A_full[self.L], Y)
            
            self.loss_history.append(epoch_loss)
            self.accuracy_history.append(train_accuracy)
            
            if validation_data is not None:
                A_val, _ = self.forward_propagation(X_val)
                val_loss = self.compute_loss(A_val[self.L], Y_val)
                val_accuracy = self._compute_accuracy(A_val[self.L], Y_val)
                self.val_loss_history.append(val_loss)
                self.val_accuracy_history.append(val_accuracy)
            
            if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                msg = f"Epoch {epoch:4d} | Loss: {epoch_loss:.4f} | Accuracy: {train_accuracy:.4f}"
                if validation_data is not None:
                    msg += f" | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}"
                print(msg)
        
        return {
            'loss': self.loss_history,
            'accuracy': self.accuracy_history,
            'val_loss': self.val_loss_history if validation_data else None,
            'val_accuracy': self.val_accuracy_history if validation_data else None
        }
    
    def predict(self, X):
        """
        Make class predictions for input data.
        
        Args:
            X (np.ndarray): Input features, shape (m_samples, n_features)
            
        Returns:
            np.ndarray: Predicted class labels, shape (m_samples,)
        """
        X = X.T
        A, _ = self.forward_propagation(X)
        predictions = np.argmax(A[self.L], axis=0)
        return predictions
    
    def predict_proba(self, X):
        """
        Get class probabilities for input data.
        
        Args:
            X (np.ndarray): Input features, shape (m_samples, n_features)
            
        Returns:
            np.ndarray: Class probabilities, shape (m_samples, n_classes)
        """
        X = X.T
        A, _ = self.forward_propagation(X)
        return A[self.L].T
    
    def _one_hot_encode(self, y, n_classes):
        """
        Convert class labels to one-hot encoding.
        
        Example:
            y = [0, 1, 2, 1]
            n_classes = 3
            
            One-hot:
            [[1, 0, 0, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0]]
        
        Args:
            y (np.ndarray): Class labels, shape (m_samples,)
            n_classes (int): Number of classes
            
        Returns:
            np.ndarray: One-hot encoded labels, shape (n_classes, m_samples)
        """
        m = len(y)
        one_hot = np.zeros((n_classes, m))
        one_hot[y.astype(int), np.arange(m)] = 1
        return one_hot
    
    def _compute_accuracy(self, Y_pred, Y_true):
        """
        Compute classification accuracy.
        
        Args:
            Y_pred (np.ndarray): Predicted probabilities, shape (n_classes, m)
            Y_true (np.ndarray): One-hot encoded labels, shape (n_classes, m)
            
        Returns:
            float: Accuracy (0 to 1)
        """
        predictions = np.argmax(Y_pred, axis=0)
        labels = np.argmax(Y_true, axis=0)
        return np.mean(predictions == labels)
    
    def get_architecture_summary(self):
        """
        Get a summary of the network architecture.
        
        Returns:
            str: Architecture description
        """
        lines = ["="*50]
        lines.append("NETWORK ARCHITECTURE SUMMARY")
        lines.append("="*50)
        lines.append(f"Total Layers: {self.L + 1} (including input)")
        lines.append(f"Hidden Activation: {self.activation_name}")
        lines.append(f"Output Activation: softmax")
        lines.append(f"Learning Rate: {self.learning_rate}")
        lines.append(f"Weight Initialization: {self.weight_init}")
        lines.append("-"*50)
        
        total_params = 0
        for l in range(self.L + 1):
            if l == 0:
                lines.append(f"Layer 0 (Input):  {self.layer_sizes[l]} neurons")
            elif l == self.L:
                n_params = self.weights[l].size + self.biases[l].size
                total_params += n_params
                lines.append(f"Layer {l} (Output): {self.layer_sizes[l]} neurons | Params: {n_params:,}")
            else:
                n_params = self.weights[l].size + self.biases[l].size
                total_params += n_params
                lines.append(f"Layer {l} (Hidden): {self.layer_sizes[l]} neurons | Params: {n_params:,}")
        
        lines.append("-"*50)
        lines.append(f"Total Parameters: {total_params:,}")
        lines.append("="*50)
        
        return "\n".join(lines)
    
    @classmethod
    def from_saved_weights(cls, layer_sizes, activation, weights, biases):
        """
        Create an MLP instance from saved weights.
        
        Args:
            layer_sizes: Network architecture
            activation: Activation function name
            weights: Dict of weight matrices
            biases: Dict of bias vectors
            
        Returns:
            MLP: Initialized model with loaded weights
        """
        model = cls(layer_sizes=layer_sizes, activation=activation)
        model.weights = weights
        model.biases = biases
        return model


def solve_xor_with_mlp():
    """
    Demonstrate that MLP can solve the XOR problem.
    
    The XOR problem requires a non-linear decision boundary,
    which a single perceptron cannot provide.
    
    With one hidden layer, the MLP can learn:
        XOR(x₁, x₂) = OR(AND(x₁, NOT(x₂)), AND(NOT(x₁), x₂))
    
    Returns:
        dict: Training results and predictions
    """
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([0, 1, 1, 0])
    
    mlp = MLP(
        layer_sizes=[2, 4, 2],
        learning_rate=0.5,
        activation='tanh',
        seed=42
    )
    
    history = mlp.fit(X, y, epochs=1000, batch_size=4, verbose=False)
    
    predictions = mlp.predict(X)
    probabilities = mlp.predict_proba(X)
    
    accuracy = np.mean(predictions == y)
    
    return {
        'X': X,
        'y': y,
        'predictions': predictions,
        'probabilities': probabilities,
        'accuracy': accuracy,
        'history': history,
        'mlp': mlp
    }


if __name__ == "__main__":
    print("\n" + "="*60)
    print("MLP SOLVING XOR PROBLEM")
    print("="*60)
    
    result = solve_xor_with_mlp()
    
    print("\nXOR Truth Table with MLP Predictions:")
    print("x₁ | x₂ | Expected | Predicted | Probability")
    print("-" * 50)
    
    for i in range(len(result['X'])):
        x = result['X'][i]
        expected = result['y'][i]
        predicted = result['predictions'][i]
        prob = result['probabilities'][i]
        match = "✓" if expected == predicted else "✗"
        print(f" {x[0]} |  {x[1]} |    {expected}     |     {predicted}     | {prob} {match}")
    
    print(f"\nFinal Accuracy: {result['accuracy']:.2%}")
    print("\nThe MLP successfully learned XOR!")
    print("This demonstrates the power of hidden layers.")
