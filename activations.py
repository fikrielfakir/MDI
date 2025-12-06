"""
================================================================================
ACTIVATION FUNCTIONS MODULE
================================================================================
This module implements activation functions used in neural networks from scratch.
Each function includes its mathematical formula, derivative for backpropagation,
output range, and use cases.

Mathematical Notation (LaTeX style):
- σ(z) = Sigmoid
- tanh(z) = Hyperbolic Tangent  
- ReLU(z) = Rectified Linear Unit
- softmax(z) = Softmax (for multiclass classification)
================================================================================
"""

import numpy as np


class ActivationFunctions:
    """
    A collection of activation functions and their derivatives for neural networks.
    
    Activation functions introduce non-linearity into the network, enabling it
    to learn complex patterns. Without activation functions, a neural network
    would behave like a simple linear regression model regardless of depth.
    
    Attributes:
        epsilon (float): Small constant for numerical stability
    """
    
    def __init__(self):
        self.epsilon = 1e-15
    
    def sigmoid(self, z):
        """
        Sigmoid Activation Function
        
        Mathematical Formula (LaTeX):
            σ(z) = 1 / (1 + e^(-z))
        
        Properties:
            - Output Range: [0, 1]
            - Smooth, differentiable everywhere
            - Historically popular for hidden layers
        
        Advantages:
            - Outputs can be interpreted as probabilities
            - Smooth gradient
        
        Disadvantages:
            - Vanishing gradient problem for large |z|
            - Output not zero-centered
            - Computationally expensive (exp function)
        
        Use Cases:
            - Binary classification output layer
            - Gates in LSTM/GRU networks
        
        Args:
            z (np.ndarray): Pre-activation values (any shape)
            
        Returns:
            np.ndarray: Activated values in range [0, 1]
        """
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))
    
    def sigmoid_derivative(self, a):
        """
        Derivative of Sigmoid Function
        
        Mathematical Formula (LaTeX):
            σ'(z) = σ(z) × (1 - σ(z))
            
        Note: Takes the activated output 'a' = σ(z), not the pre-activation 'z'
        
        The derivative shows that:
            - Maximum gradient is 0.25 (at z=0)
            - Gradient approaches 0 for large |z| (vanishing gradient)
        
        Args:
            a (np.ndarray): Activated values (output of sigmoid)
            
        Returns:
            np.ndarray: Derivative values
        """
        return a * (1.0 - a)
    
    def tanh(self, z):
        """
        Hyperbolic Tangent (Tanh) Activation Function
        
        Mathematical Formula (LaTeX):
            tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
            
        Alternative form:
            tanh(z) = 2σ(2z) - 1  (scaled sigmoid)
        
        Properties:
            - Output Range: [-1, 1]
            - Zero-centered (unlike sigmoid)
            - Smooth, differentiable everywhere
        
        Advantages:
            - Zero-centered output (helps gradient flow)
            - Stronger gradients than sigmoid
        
        Disadvantages:
            - Still suffers from vanishing gradient
            - Computationally expensive
        
        Use Cases:
            - Hidden layers (better than sigmoid)
            - RNNs and LSTMs
            - When zero-centered output is desired
        
        Args:
            z (np.ndarray): Pre-activation values
            
        Returns:
            np.ndarray: Activated values in range [-1, 1]
        """
        return np.tanh(z)
    
    def tanh_derivative(self, a):
        """
        Derivative of Tanh Function
        
        Mathematical Formula (LaTeX):
            tanh'(z) = 1 - tanh²(z)
            
        Note: Takes the activated output 'a' = tanh(z), not the pre-activation 'z'
        
        The derivative shows that:
            - Maximum gradient is 1.0 (at z=0)
            - Gradient approaches 0 for large |z|
        
        Args:
            a (np.ndarray): Activated values (output of tanh)
            
        Returns:
            np.ndarray: Derivative values
        """
        return 1.0 - np.power(a, 2)
    
    def relu(self, z):
        """
        Rectified Linear Unit (ReLU) Activation Function
        
        Mathematical Formula (LaTeX):
            ReLU(z) = max(0, z)
            
        Piecewise definition:
            ReLU(z) = z  if z > 0
            ReLU(z) = 0  if z ≤ 0
        
        Properties:
            - Output Range: [0, ∞)
            - Non-saturating for positive values
            - Sparse activation (many zeros)
        
        Advantages:
            - No vanishing gradient for positive values
            - Computationally efficient (simple threshold)
            - Sparse representations (biological plausibility)
            - Faster convergence than sigmoid/tanh
        
        Disadvantages:
            - "Dying ReLU" problem: neurons can get stuck at 0
            - Not zero-centered
            - Unbounded output (can cause exploding activations)
        
        Use Cases:
            - Most popular for hidden layers in modern networks
            - CNNs, deep networks
            - When speed is important
        
        Args:
            z (np.ndarray): Pre-activation values
            
        Returns:
            np.ndarray: Activated values (non-negative)
        """
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        """
        Derivative of ReLU Function
        
        Mathematical Formula (LaTeX):
            ReLU'(z) = 1  if z > 0
            ReLU'(z) = 0  if z ≤ 0
            
        Note: Technically undefined at z=0, conventionally set to 0
        
        Key insight:
            - Gradient is either 1 or 0 (no saturation for z > 0)
            - This prevents vanishing gradient for positive values
        
        Args:
            z (np.ndarray): Pre-activation values (NOT the activated output)
            
        Returns:
            np.ndarray: Derivative values (0 or 1)
        """
        return (z > 0).astype(float)
    
    def leaky_relu(self, z, alpha=0.01):
        """
        Leaky ReLU Activation Function
        
        Mathematical Formula (LaTeX):
            LeakyReLU(z) = z      if z > 0
            LeakyReLU(z) = αz     if z ≤ 0
            
        Where α (alpha) is a small positive constant (typically 0.01)
        
        Properties:
            - Output Range: (-∞, ∞)
            - Allows small negative gradients
        
        Advantages:
            - Solves "dying ReLU" problem
            - Allows gradient flow for negative inputs
        
        Use Cases:
            - When dying ReLU is a concern
            - Deep networks with many layers
        
        Args:
            z (np.ndarray): Pre-activation values
            alpha (float): Slope for negative values (default: 0.01)
            
        Returns:
            np.ndarray: Activated values
        """
        return np.where(z > 0, z, alpha * z)
    
    def leaky_relu_derivative(self, z, alpha=0.01):
        """
        Derivative of Leaky ReLU
        
        Mathematical Formula (LaTeX):
            LeakyReLU'(z) = 1  if z > 0
            LeakyReLU'(z) = α  if z ≤ 0
        
        Args:
            z (np.ndarray): Pre-activation values
            alpha (float): Slope for negative values
            
        Returns:
            np.ndarray: Derivative values
        """
        return np.where(z > 0, 1.0, alpha)
    
    def softmax(self, z):
        """
        Softmax Activation Function
        
        Mathematical Formula (LaTeX):
            softmax(z_i) = e^(z_i) / Σⱼ(e^(z_j))
            
        For numerical stability, we compute:
            softmax(z_i) = e^(z_i - max(z)) / Σⱼ(e^(z_j - max(z)))
        
        Properties:
            - Output Range: (0, 1) for each element
            - Sum of outputs = 1 (probability distribution)
            - Preserves relative ordering of inputs
        
        Advantages:
            - Outputs interpretable as class probabilities
            - Differentiable (enables gradient-based learning)
        
        Disadvantages:
            - Computationally expensive
            - Sensitive to outliers in input
        
        Use Cases:
            - Output layer for multiclass classification
            - Attention mechanisms in transformers
        
        Args:
            z (np.ndarray): Pre-activation values, shape (n_classes, m_samples)
            
        Returns:
            np.ndarray: Probability distribution over classes
        """
        z_stable = z - np.max(z, axis=0, keepdims=True)
        exp_z = np.exp(z_stable)
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)
    
    def get_activation(self, name):
        """
        Get activation function by name.
        
        Args:
            name (str): Activation function name ('sigmoid', 'tanh', 'relu', 'leaky_relu')
            
        Returns:
            callable: The activation function
        """
        activations = {
            'sigmoid': self.sigmoid,
            'tanh': self.tanh,
            'relu': self.relu,
            'leaky_relu': self.leaky_relu
        }
        if name not in activations:
            raise ValueError(f"Unknown activation: {name}. Choose from {list(activations.keys())}")
        return activations[name]
    
    def get_activation_derivative(self, name):
        """
        Get activation derivative function by name.
        
        Args:
            name (str): Activation function name
            
        Returns:
            callable: The derivative function
        """
        derivatives = {
            'sigmoid': self.sigmoid_derivative,
            'tanh': self.tanh_derivative,
            'relu': self.relu_derivative,
            'leaky_relu': self.leaky_relu_derivative
        }
        if name not in derivatives:
            raise ValueError(f"Unknown activation: {name}")
        return derivatives[name]


def visualize_activations():
    """
    Create visualization data for all activation functions.
    
    Returns:
        dict: Dictionary containing x values and outputs for each activation
    """
    af = ActivationFunctions()
    x = np.linspace(-5, 5, 200)
    
    return {
        'x': x,
        'sigmoid': af.sigmoid(x),
        'sigmoid_derivative': af.sigmoid_derivative(af.sigmoid(x)),
        'tanh': af.tanh(x),
        'tanh_derivative': af.tanh_derivative(af.tanh(x)),
        'relu': af.relu(x),
        'relu_derivative': af.relu_derivative(x),
        'leaky_relu': af.leaky_relu(x),
        'leaky_relu_derivative': af.leaky_relu_derivative(x)
    }
