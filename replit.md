# Deep Learning from Scratch

## Overview

This project is an educational implementation of neural networks built entirely from scratch using only NumPy (no TensorFlow/PyTorch). It provides a comprehensive, interactive learning tool that demonstrates both the theoretical foundations and practical implementation of deep learning concepts.

The project includes:
- Core neural network building blocks (perceptron, multi-layer perceptron)
- Multiple activation functions with mathematical foundations
- Interactive Streamlit-based web interface for visualization and experimentation
- Real-world training examples using the Iris dataset
- Data preprocessing and evaluation utilities
- **Iris Species Prediction** with three input modes:
  - Single measurement prediction with confidence display
  - CSV batch upload (format: Id, SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, Species)
  - Multi-image upload with measurement input forms
- **Enhanced educational content** with tabbed explanations of neural network concepts

The application is designed to help users understand how neural networks work at a fundamental level by implementing each component from first principles.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture

**Technology:** Streamlit web framework  
**Rationale:** Chosen for rapid development of interactive data science applications without requiring frontend expertise. Enables real-time visualization and experimentation with neural network concepts.

**Key Design Decisions:**
- Tab-based navigation for organizing different learning modules (Introduction, Activation Functions, Perceptron, MLP, Training)
- Interactive controls for hyperparameter tuning (learning rates, network architecture, epochs)
- Real-time visualization of training metrics and model behavior
- Wide layout to accommodate side-by-side visualizations

**Entry Point:** `app.py` serves as the main application controller, orchestrating the different educational modules.

### Core Neural Network Architecture

**Modular Design Pattern**  
The system is organized into specialized modules, each handling a specific aspect of neural network functionality:

1. **Activation Functions Module** (`activations.py`)
   - Implements mathematical activation functions (sigmoid, tanh, ReLU, softmax)
   - Provides both forward pass and derivative computations for backpropagation
   - Includes visualization utilities for understanding activation behavior
   - Numerical stability considerations (epsilon values for edge cases)

2. **Perceptron Module** (`perceptron.py`)
   - Implements single-layer neural networks
   - Demonstrates the perceptron learning algorithm
   - Shows limitations with linearly non-separable problems (XOR demonstration)
   - Educational focus on foundational concepts

3. **Multi-Layer Perceptron Module** (`mlp.py`)
   - Full implementation of feedforward neural networks
   - Supports arbitrary network architectures via configurable layer sizes
   - Weight initialization strategies (Xavier/He initialization)
   - Complete training pipeline: forward propagation → loss computation → backpropagation → gradient descent
   - Mini-batch gradient descent support
   - Mathematical notation follows standard deep learning conventions (W^[l], b^[l], A^[l], Z^[l])

**Algorithm Implementation Philosophy:**
- All algorithms implemented from scratch using only NumPy
- Extensive mathematical documentation using LaTeX-style notation
- Gradient computation via manual backpropagation (no automatic differentiation)
- Educational transparency over computational efficiency

### Data Processing Architecture

**Data Utilities Module** (`data_utils.py`)  
Handles all data-related operations following standard machine learning pipeline practices:

1. **Dataset Management**
   - Integration with scikit-learn for standard datasets (Iris dataset)
   - Support for synthetic data generation
   - Train/test splitting with reproducible random states

2. **Preprocessing Pipeline**
   - Feature normalization using z-score standardization
   - One-hot encoding for multi-class classification
   - Data shape validation and error handling

3. **Evaluation Framework**
   - Multiple classification metrics (accuracy, precision, recall, F1-score)
   - Confusion matrix computation and formatting
   - Integration with scikit-learn metrics for validation

**Design Rationale:** While neural network implementations are from scratch, data preprocessing leverages scikit-learn to avoid reinventing well-tested preprocessing routines and focus educational effort on core ML concepts.

### Mathematical Foundation

**Backpropagation Implementation**  
The MLP module implements gradient computation using the chain rule:
- Layer-wise gradient calculation from output to input
- Careful matrix dimension management for broadcasting
- Derivative caching for computational efficiency

**Loss Functions**  
- Binary cross-entropy for binary classification
- Categorical cross-entropy for multi-class problems
- Numerical stability through log-sum-exp tricks

**Optimization Strategy**  
- Vanilla gradient descent with configurable learning rates
- Mini-batch support for scalable training
- Weight update rules following standard SGD formulation

## External Dependencies

### Core Computational Libraries

**NumPy**  
- **Purpose:** Foundation for all numerical computations and matrix operations
- **Usage:** Neural network mathematics, activation functions, gradient calculations, array manipulations
- **Rationale:** Industry-standard library for numerical computing in Python, provides efficient vectorized operations

### Machine Learning Utilities

**scikit-learn (sklearn)**  
- **Purpose:** Data preprocessing and evaluation metrics
- **Specific Uses:**
  - Dataset loading (`load_iris`)
  - Train/test splitting (`train_test_split`)
  - Feature scaling (`StandardScaler`)
  - Evaluation metrics (`accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`)
- **Rationale:** Provides reliable, tested implementations of standard ML preprocessing and evaluation routines, allowing the project to focus on neural network implementation rather than data utilities

### Visualization and Web Framework

**Matplotlib**  
- **Purpose:** Data visualization and plotting
- **Usage:** Activation function plots, training curves, decision boundaries
- **Rationale:** De facto standard for scientific plotting in Python ecosystem

**Streamlit**  
- **Purpose:** Interactive web application framework
- **Usage:** Main user interface, interactive widgets, real-time visualizations
- **Rationale:** Enables rapid development of data science applications with minimal web development overhead, ideal for educational demonstrations

### Development Environment

The project uses a standard Python environment with no database requirements or external APIs. All data processing occurs in-memory using NumPy arrays, making it lightweight and suitable for educational purposes.