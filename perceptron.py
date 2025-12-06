"""
================================================================================
PERCEPTRON MODULE
================================================================================
This module implements the Simple Perceptron - the fundamental building block
of neural networks. Demonstrates both the power of single-layer networks for
linearly separable problems and their limitations (XOR problem).

Historical Context:
    - Invented by Frank Rosenblatt in 1957
    - First algorithm that could "learn" from data
    - Led to the first AI winter when limitations were exposed

Mathematical Foundation:
    Linear combination: z = w·x + b
    Activation: ŷ = step(z) where step(z) = 1 if z ≥ 0 else 0
    
Learning Rule (Perceptron Learning Algorithm):
    w_new = w_old + α × error × x
    b_new = b_old + α × error
    
    Where:
    - α (alpha) is the learning rate
    - error = (y_true - y_pred)
================================================================================
"""

import numpy as np


class Perceptron:
    """
    Simple Perceptron Implementation
    
    A perceptron is a single artificial neuron that can learn to classify
    linearly separable patterns. It's the simplest form of a neural network.
    
    Architecture:
        [x₁] ──w₁──┐
        [x₂] ──w₂──┼──[Σ + b]──[step]──> ŷ
        [x₃] ──w₃──┘
        
    The perceptron computes:
        z = Σ(wᵢ × xᵢ) + b
        ŷ = 1 if z ≥ 0 else 0
    
    Attributes:
        n_inputs (int): Number of input features
        learning_rate (float): Step size for weight updates (α)
        weights (np.ndarray): Connection weights [w₁, w₂, ..., wₙ]
        bias (float): Bias term (b)
        history (dict): Training history for visualization
    """
    
    def __init__(self, n_inputs, learning_rate=0.1):
        """
        Initialize the Perceptron.
        
        Args:
            n_inputs (int): Number of input features
            learning_rate (float): Learning rate α (default: 0.1)
        """
        self.n_inputs = n_inputs
        self.learning_rate = learning_rate
        
        np.random.seed(42)
        self.weights = np.random.randn(n_inputs) * 0.01
        self.bias = 0.0
        
        self.history = {
            'epoch': [],
            'accuracy': [],
            'errors': []
        }
    
    def step_function(self, z):
        """
        Step (Heaviside) Activation Function
        
        Mathematical Formula:
            step(z) = 1  if z ≥ 0
            step(z) = 0  if z < 0
        
        This is a hard threshold function - the original activation
        used in perceptrons.
        
        Args:
            z (np.ndarray or float): Pre-activation value(s)
            
        Returns:
            np.ndarray or int: Binary output(s)
        """
        return np.where(z >= 0, 1, 0)
    
    def predict(self, X):
        """
        Make predictions for input data.
        
        Computes:
            z = X · w + b
            ŷ = step(z)
        
        Args:
            X (np.ndarray): Input features, shape (n_samples, n_inputs)
            
        Returns:
            np.ndarray: Binary predictions {0, 1}, shape (n_samples,)
        """
        X = np.atleast_2d(X)
        z = np.dot(X, self.weights) + self.bias
        return self.step_function(z)
    
    def train(self, X, y, epochs=100, verbose=True):
        """
        Train the perceptron using the Perceptron Learning Algorithm.
        
        Algorithm:
            For each epoch:
                For each sample (xᵢ, yᵢ):
                    1. Compute prediction: ŷ = step(w·x + b)
                    2. Compute error: e = y - ŷ
                    3. Update weights: w = w + α × e × x
                    4. Update bias: b = b + α × e
        
        Convergence:
            - Guaranteed to converge for linearly separable data
            - Will not converge for non-linearly separable data (like XOR)
        
        Args:
            X (np.ndarray): Training features, shape (n_samples, n_inputs)
            y (np.ndarray): Training labels {0, 1}, shape (n_samples,)
            epochs (int): Number of training iterations
            verbose (bool): Print training progress
            
        Returns:
            dict: Training history with accuracy per epoch
        """
        X = np.atleast_2d(X)
        y = np.array(y)
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            total_errors = 0
            
            for i in range(n_samples):
                xi = X[i]
                yi = y[i]
                
                z = np.dot(xi, self.weights) + self.bias
                y_pred = self.step_function(z)
                
                error = yi - y_pred
                
                if error != 0:
                    total_errors += 1
                    self.weights += self.learning_rate * error * xi
                    self.bias += self.learning_rate * error
            
            accuracy = (n_samples - total_errors) / n_samples
            
            self.history['epoch'].append(epoch)
            self.history['accuracy'].append(accuracy)
            self.history['errors'].append(total_errors)
            
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch:4d} | Errors: {total_errors} | Accuracy: {accuracy:.4f}")
            
            if total_errors == 0:
                if verbose:
                    print(f"\nConverged at epoch {epoch}! Perfect classification achieved.")
                break
        
        return self.history
    
    def get_decision_boundary(self):
        """
        Calculate decision boundary parameters for 2D visualization.
        
        For 2D input (x₁, x₂), the decision boundary is the line where:
            w₁x₁ + w₂x₂ + b = 0
            
        Solving for x₂:
            x₂ = -(w₁x₁ + b) / w₂
            
        Slope: m = -w₁/w₂
        Intercept: c = -b/w₂
        
        Returns:
            tuple: (slope, intercept) or None if w₂ = 0
        """
        if self.n_inputs != 2:
            return None
        
        w1, w2 = self.weights
        
        if abs(w2) < 1e-10:
            return None
        
        slope = -w1 / w2
        intercept = -self.bias / w2
        
        return slope, intercept


def demo_logic_gates():
    """
    Demonstrate perceptron on logic gates (AND, OR, NAND, XOR).
    
    AND Gate:
        x₁ | x₂ | AND
        0  | 0  |  0
        0  | 1  |  0
        1  | 0  |  0
        1  | 1  |  1
        → Linearly separable ✓
        
    OR Gate:
        x₁ | x₂ | OR
        0  | 0  |  0
        0  | 1  |  1
        1  | 0  |  1
        1  | 1  |  1
        → Linearly separable ✓
        
    XOR Gate (Exclusive OR):
        x₁ | x₂ | XOR
        0  | 0  |  0
        0  | 1  |  1
        1  | 0  |  1
        1  | 1  |  0
        → NOT linearly separable ✗
        → This is why we need Multi-Layer Networks!
    
    Returns:
        dict: Results for each logic gate
    """
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    
    gates = {
        'AND': np.array([0, 0, 0, 1]),
        'OR': np.array([0, 1, 1, 1]),
        'NAND': np.array([1, 1, 1, 0]),
        'XOR': np.array([0, 1, 1, 0])
    }
    
    results = {}
    
    for gate_name, y in gates.items():
        print(f"\n{'='*50}")
        print(f"Training Perceptron for {gate_name} Gate")
        print('='*50)
        
        perceptron = Perceptron(n_inputs=2, learning_rate=0.1)
        history = perceptron.train(X, y, epochs=100, verbose=False)
        
        predictions = perceptron.predict(X)
        
        print(f"\nTruth Table for {gate_name}:")
        print("x₁ | x₂ | Expected | Predicted")
        print("-" * 35)
        
        correct = 0
        for i in range(len(X)):
            expected = y[i]
            predicted = predictions[i]
            match = "✓" if expected == predicted else "✗"
            print(f" {X[i][0]} |  {X[i][1]} |    {expected}     |     {predicted}     {match}")
            if expected == predicted:
                correct += 1
        
        accuracy = correct / len(X)
        success = accuracy == 1.0
        
        results[gate_name] = {
            'perceptron': perceptron,
            'predictions': predictions,
            'expected': y,
            'accuracy': accuracy,
            'success': success,
            'history': history,
            'weights': perceptron.weights.copy(),
            'bias': perceptron.bias
        }
        
        print(f"\nFinal Accuracy: {accuracy:.2%}")
        print(f"Learned Weights: w = {perceptron.weights}")
        print(f"Learned Bias: b = {perceptron.bias:.4f}")
        
        if not success:
            print(f"\n⚠️  The Perceptron FAILED to learn {gate_name}!")
            print("This demonstrates the XOR Problem - a fundamental")
            print("limitation of single-layer networks.")
    
    return results


def demonstrate_xor_problem():
    """
    Detailed demonstration of why perceptron fails on XOR.
    
    The XOR Problem:
    ================
    XOR outputs 1 when inputs are different, 0 when same.
    
    Visualization in 2D space:
        
        x₂
        │
      1 ├───●(0,1)───────●(1,1)
        │     class 1      class 0
        │
      0 ├───●(0,0)───────●(1,0)
        │     class 0      class 1
        └───────────────────── x₁
            0             1
            
    No single straight line can separate the two classes!
    
    Why it fails mathematically:
        A perceptron creates a linear decision boundary: w₁x₁ + w₂x₂ + b = 0
        
        For (0,0)→0: We need w₁(0) + w₂(0) + b < 0  →  b < 0
        For (0,1)→1: We need w₁(0) + w₂(1) + b ≥ 0  →  w₂ + b ≥ 0
        For (1,0)→1: We need w₁(1) + w₂(0) + b ≥ 0  →  w₁ + b ≥ 0
        For (1,1)→0: We need w₁(1) + w₂(1) + b < 0  →  w₁ + w₂ + b < 0
        
        Adding constraints 2 and 3: w₁ + w₂ + 2b ≥ 0
        But constraint 4 says: w₁ + w₂ + b < 0
        This means: b ≥ 0 (from subtraction)
        
        Contradiction with constraint 1: b < 0
        
        Therefore, no solution exists!
    
    Solution: Multi-Layer Perceptron (MLP)
    ======================================
    With hidden layers, we can learn non-linear decision boundaries.
    
    Returns:
        dict: Demonstration results
    """
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([0, 1, 1, 0])
    
    perceptron = Perceptron(n_inputs=2, learning_rate=0.1)
    history = perceptron.train(X_xor, y_xor, epochs=1000, verbose=False)
    
    predictions = perceptron.predict(X_xor)
    
    accuracy = np.mean(predictions == y_xor)
    
    return {
        'X': X_xor,
        'y': y_xor,
        'predictions': predictions,
        'accuracy': accuracy,
        'history': history,
        'weights': perceptron.weights,
        'bias': perceptron.bias,
        'perceptron': perceptron
    }


if __name__ == "__main__":
    print("\n" + "="*60)
    print("PERCEPTRON DEMONSTRATION")
    print("="*60)
    
    results = demo_logic_gates()
    
    print("\n" + "="*60)
    print("XOR PROBLEM - WHY MULTI-LAYER NETWORKS ARE NEEDED")
    print("="*60)
    
    xor_result = demonstrate_xor_problem()
    
    print(f"\nXOR Problem Summary:")
    print(f"  Expected outputs: {xor_result['y']}")
    print(f"  Perceptron outputs: {xor_result['predictions']}")
    print(f"  Accuracy: {xor_result['accuracy']:.2%}")
    print(f"\nThe perceptron achieves only ~50% accuracy on XOR,")
    print("which is equivalent to random guessing!")
    print("\nThis motivated the development of Multi-Layer Perceptrons (MLPs)")
    print("which can learn non-linear decision boundaries.")
