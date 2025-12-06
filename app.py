"""
Deep Learning from Scratch - Interactive Streamlit Application
A comprehensive educational tool for understanding neural networks.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import io
from activations import ActivationFunctions, visualize_activations
from perceptron import Perceptron, demo_logic_gates, demonstrate_xor_problem
from mlp import MLP, solve_xor_with_mlp
from data_utils import (load_iris_dataset, load_mnist_dataset, get_mnist_sample_images,
                        compute_metrics, format_confusion_matrix, one_hot_encode)
from db_utils import (init_database, import_iris_from_csv, get_iris_sample_count,
                      save_training_run, get_training_history, get_training_run_details,
                      delete_training_run, save_trained_model, load_trained_model, get_saved_model_names,
                      save_iris_reference_image, get_iris_reference_images, get_reference_image_count,
                      delete_iris_reference_image, save_iris_image_features, get_all_iris_image_features,
                      get_feature_count, save_iris_image_model, load_iris_image_model)
from image_utils import (extract_features, normalize_features, batch_extract_features,
                         load_image_from_bytes, SPECIES_MAPPING, SPECIES_ID_MAPPING, get_feature_dimension)
import os
from sklearn.preprocessing import StandardScaler
from PIL import Image

st.set_page_config(
    page_title="Deep Learning from Scratch",
    page_icon="🧠",
    layout="wide"
)

def main():
    st.title("Deep Learning from Scratch")
    st.markdown("### An Interactive Guide to Neural Networks")
    
    init_database()
    
    tabs = st.tabs([
        "Introduction to ANNs",
        "Activation Functions", 
        "Perceptron",
        "Multi-Layer Perceptron",
        "Iris Prediction",
        "Image Recognition",
        "Train on Iris Dataset",
        "Train on MNIST",
        "Training History"
    ])
    
    with tabs[0]:
        show_introduction()
    
    with tabs[1]:
        show_activation_functions()
    
    with tabs[2]:
        show_perceptron()
    
    with tabs[3]:
        show_mlp()
    
    with tabs[4]:
        show_iris_prediction()
    
    with tabs[5]:
        show_image_recognition()
    
    with tabs[6]:
        show_iris_training()
    
    with tabs[7]:
        show_mnist_training()
    
    with tabs[8]:
        show_training_history()


def show_introduction():
    st.header("Introduction to Artificial Neural Networks")
    
    st.markdown("""
    ## What is an Artificial Neural Network?
    
    An **Artificial Neural Network (ANN)** is a computational model inspired by the structure 
    and function of biological neural networks in the brain. Just as our brains consist of 
    billions of interconnected neurons that process information, ANNs consist of artificial 
    neurons (also called nodes or units) organized in layers.
    """)
    
    st.markdown("---")
    
    st.subheader("The Building Blocks of Neural Networks")
    
    concept_tabs = st.tabs(["Neurons", "Weights & Biases", "Activation Functions", "Loss Functions", "Optimizers"])
    
    with concept_tabs[0]:
        st.markdown("""
        ## Neurons - The Heart of Neural Networks
        
        Neurons are at the heart of any neural network, including the perceptron. These digital entities 
        receive inputs, apply weights, and produce an output. Neurons are the building blocks through 
        which information flows in a neural network. Just as neurons in our brains communicate, 
        these fundamental blocks communicate in the language of numbers.
        
        ### Biological Inspiration
        
        In the brain, a neuron:
        1. Receives electrical signals from other neurons through **dendrites**
        2. Processes these signals in the **cell body**
        3. If the combined signal exceeds a threshold, it fires an output through the **axon**
        4. The output connects to other neurons through **synapses**
        
        ### Artificial Neuron Model
        
        Similarly, an artificial neuron:
        1. Receives inputs from other neurons or raw data
        2. Computes a weighted sum of these inputs
        3. Applies an activation function to produce an output
        4. Passes the output to the next layer
        
        ### Mathematical Model
        
        For a single neuron with inputs $x_1, x_2, ..., x_n$:
        
        $$z = \\sum_{i=1}^{n} w_i x_i + b = w_1 x_1 + w_2 x_2 + ... + w_n x_n + b$$
        
        $$a = f(z)$$
        
        Where:
        - $w_i$ are the **weights** (connection strengths)
        - $b$ is the **bias** (threshold adjustment)
        - $f$ is the **activation function** (introduces non-linearity)
        - $a$ is the **activation** (output of the neuron)
        """)
        
        st.markdown("""
        ```
        Input Signals      Weights      Neuron Processing      Output
        
            x₁ ──────────── w₁ ──┐
                                 │
            x₂ ──────────── w₂ ──┼──> [Σ + b] ──> [f(z)] ──> Output
                                 │
            x₃ ──────────── w₃ ──┘
        ```
        """)
    
    with concept_tabs[1]:
        st.markdown("""
        ## Weights and Biases
        
        Weights and biases are the **adjustable parameters** in the network that influence the 
        importance of input features and establish a threshold for activation.
        
        ### Weights
        
        Each input data feature is given a certain **weight**, indicating its importance in the decision:
        
        - **High weight**: The input has strong influence on the output
        - **Low weight**: The input has weak influence on the output
        - **Negative weight**: The input has an inverse relationship with the output
        
        During training, the network learns which features are most important by adjusting these weights.
        
        ### Biases
        
        Biases act as the **minimum requirement** for a feature to contribute to the output:
        
        - A bias shifts the activation function horizontally
        - It allows neurons to activate even when all inputs are zero
        - Biases provide flexibility in the decision boundary
        
        ### The Learning Process
        
        Adjusting weights and biases during model training refines the network's ability to make 
        accurate predictions. The goal is to find the optimal combination that minimizes the 
        prediction error.
        
        $$\\text{Output} = f\\left(\\sum_{i} w_i \\cdot x_i + b\\right)$$
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Weight Interpretation:**
            | Weight Value | Meaning |
            |-------------|---------|
            | w > 0 | Positive correlation |
            | w < 0 | Negative correlation |
            | w ≈ 0 | Feature has little impact |
            | |w| large | Strong influence |
            """)
        with col2:
            st.markdown("""
            **Bias Role:**
            - Acts as a threshold
            - Enables activation shift
            - Provides flexibility
            - Independent of input
            """)
    
    with concept_tabs[2]:
        st.markdown("""
        ## Activation Functions
        
        Activation functions, such as the step function in the perceptron, determine whether 
        the neuron **fires**. In other words, they decide whether the information flowing through 
        should be allowed to contribute to the output.
        
        ### The Threshold Concept
        
        Picture it as a threshold—if the incoming data is above a certain level, the perceptron 
        'fires' or produces an output; otherwise, it remains silent. This binary decision-making 
        process showcases the essence of activation functions in shaping the output of our digital neurons.
        
        ### Why Non-linearity Matters
        
        Without activation functions, a neural network would just be a **linear transformation**:
        
        $$y = W_2 \\cdot (W_1 \\cdot x + b_1) + b_2 = W_{combined} \\cdot x + b_{combined}$$
        
        No matter how many layers we stack, the result would still be linear! Activation functions 
        introduce **non-linearity**, enabling networks to learn complex patterns.
        
        ### Common Activation Functions
        
        | Function | Formula | Output Range | Use Case |
        |----------|---------|--------------|----------|
        | **Sigmoid** | $\\sigma(z) = \\frac{1}{1+e^{-z}}$ | (0, 1) | Binary classification, LSTM gates |
        | **Tanh** | $\\tanh(z) = \\frac{e^z - e^{-z}}{e^z + e^{-z}}$ | (-1, 1) | Hidden layers, RNNs |
        | **ReLU** | $\\max(0, z)$ | [0, ∞) | Most hidden layers |
        | **Softmax** | $\\frac{e^{z_i}}{\\sum_j e^{z_j}}$ | (0, 1), sum=1 | Multi-class output |
        """)
    
    with concept_tabs[3]:
        st.markdown("""
        ## Loss Functions
        
        Loss functions **quantify the difference** between the predicted output and the actual target. 
        The objective is to minimize this difference during the training process.
        
        ### The Concept of Error
        
        In the context of neural networks, the concept of error or loss becomes evident. The network 
        learns by **reducing the difference** between its prediction and the actual target, laying 
        the groundwork for understanding more sophisticated loss functions in advanced architectures.
        
        ### Common Loss Functions
        
        | Loss Function | Formula | Use Case |
        |--------------|---------|----------|
        | **Mean Squared Error (MSE)** | $\\frac{1}{n}\\sum(y - \\hat{y})^2$ | Regression problems |
        | **Cross-Entropy** | $-\\sum y \\log(\\hat{y})$ | Classification problems |
        | **Binary Cross-Entropy** | $-[y\\log(\\hat{y}) + (1-y)\\log(1-\\hat{y})]$ | Binary classification |
        
        ### How Loss Guides Learning
        
        1. **Forward Pass**: Network makes a prediction
        2. **Loss Calculation**: Compare prediction to actual value
        3. **Backward Pass**: Calculate gradients of loss with respect to weights
        4. **Update**: Adjust weights to reduce loss
        
        The goal of training is to find weights that **minimize the loss function**.
        """)
        
        st.info("Lower loss = Better predictions. The training process iteratively reduces the loss until the model performs well.")
    
    with concept_tabs[4]:
        st.markdown("""
        ## Optimizers
        
        Although more straightforward in the perceptron context, optimizers are crucial for 
        **adjusting weights and biases** based on the computed loss. They fine-tune the model 
        parameters to minimize the loss and improve overall performance.
        
        ### The Role of Optimizers
        
        This mechanism hints at the broader optimization techniques in more complex deep learning 
        architectures. Optimizers determine:
        
        - **How fast** to update weights (learning rate)
        - **Which direction** to move (gradient direction)
        - **How much** to adjust each parameter
        
        ### Popular Optimization Techniques
        
        | Optimizer | Description | Characteristics |
        |-----------|-------------|-----------------|
        | **SGD** (Stochastic Gradient Descent) | Basic gradient descent with random samples | Simple, may oscillate |
        | **Momentum** | Adds velocity to gradient updates | Faster convergence |
        | **Adam** | Adaptive learning rates per parameter | Most popular, works well |
        | **RMSprop** | Adapts learning rate based on recent gradients | Good for RNNs |
        
        ### Gradient Descent Visualization
        
        Imagine rolling a ball down a hill to find the lowest point (minimum loss):
        
        - **Learning Rate**: How big each step is
        - **Gradient**: The direction of steepest descent
        - **Momentum**: The ball's velocity from previous steps
        
        """)
        
        st.warning("Choosing the right optimizer and learning rate is crucial. Too large a learning rate may overshoot the minimum; too small may take forever to converge.")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Network Architecture
        
        Neural networks are organized in layers:
        
        1. **Input Layer**: Receives raw data
        2. **Hidden Layers**: Process information
        3. **Output Layer**: Produces predictions
        
        A network with multiple hidden layers is called a **Deep Neural Network**.
        
        Each node in the network represents a neuron and is connected to nodes in adjacent layers. 
        These connections are associated with weights, which determine the strength of the connection.
        """)
    
    with col2:
        st.markdown("""
        ### Why Neural Networks Work
        
        The power of neural networks comes from:
        
        1. **Non-linear activation functions**: Allow learning complex patterns
        2. **Multiple layers**: Enable hierarchical feature learning
        3. **Gradient-based learning**: Automatically adjust weights
        4. **Universal approximation**: Can approximate any continuous function
        
        Their capacity to learn from large datasets enables them to generalize well to new, unseen data.
        """)
    
    st.markdown("""
    ### The Learning Process
    
    Neural networks learn through a process called **training**:
    
    1. **Forward Propagation**: Input flows through the network to produce output
    2. **Loss Computation**: Compare prediction with actual answer
    3. **Backpropagation**: Calculate how each weight contributed to the error
    4. **Gradient Descent**: Adjust weights to reduce error
    
    This process repeats thousands of times until the network learns the patterns in the data.
    The **backpropagation technique** improves the network by propagating error signals backward 
    through the layers, allowing each weight to be adjusted proportionally to its contribution to the error.
    """)


def show_activation_functions():
    st.header("Activation Functions")
    
    st.markdown("""
    ## Why Do We Need Activation Functions?
    
    Without activation functions, a neural network would just be a **linear transformation**:
    
    $$y = W_2 \\cdot (W_1 \\cdot x + b_1) + b_2 = W_{combined} \\cdot x + b_{combined}$$
    
    No matter how many layers we stack, the result would still be linear! Activation functions 
    introduce **non-linearity**, enabling networks to learn complex patterns.
    """)
    
    viz_data = visualize_activations()
    x = viz_data['x']
    
    activation_choice = st.selectbox(
        "Select Activation Function to Explore:",
        ["Sigmoid", "Tanh", "ReLU", "Leaky ReLU", "All Functions"]
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    if activation_choice == "Sigmoid":
        axes[0].plot(x, viz_data['sigmoid'], 'b-', linewidth=2)
        axes[0].set_title('Sigmoid: σ(z) = 1/(1 + e^(-z))')
        axes[1].plot(x, viz_data['sigmoid_derivative'], 'r-', linewidth=2)
        axes[1].set_title("Sigmoid Derivative: σ'(z) = σ(z)(1 - σ(z))")
        
        st.markdown("""
        ### Sigmoid Function
        
        **Formula**: $\\sigma(z) = \\frac{1}{1 + e^{-z}}$
        
        **Output Range**: (0, 1)
        
        **Advantages**:
        - Outputs interpretable as probabilities
        - Smooth, differentiable everywhere
        
        **Disadvantages**:
        - Vanishing gradient for large |z|
        - Not zero-centered
        - Computationally expensive
        
        **Use Cases**: Binary classification output, LSTM/GRU gates
        """)
        
    elif activation_choice == "Tanh":
        axes[0].plot(x, viz_data['tanh'], 'b-', linewidth=2)
        axes[0].set_title('Tanh: tanh(z)')
        axes[1].plot(x, viz_data['tanh_derivative'], 'r-', linewidth=2)
        axes[1].set_title("Tanh Derivative: 1 - tanh²(z)")
        
        st.markdown("""
        ### Hyperbolic Tangent (Tanh)
        
        **Formula**: $\\tanh(z) = \\frac{e^z - e^{-z}}{e^z + e^{-z}}$
        
        **Output Range**: (-1, 1)
        
        **Advantages**:
        - Zero-centered output (better gradient flow)
        - Stronger gradients than sigmoid
        
        **Disadvantages**:
        - Still suffers from vanishing gradient
        - Computationally expensive
        
        **Use Cases**: Hidden layers (when zero-centering matters), RNNs
        """)
        
    elif activation_choice == "ReLU":
        axes[0].plot(x, viz_data['relu'], 'b-', linewidth=2)
        axes[0].set_title('ReLU: max(0, z)')
        axes[1].plot(x, viz_data['relu_derivative'], 'r-', linewidth=2)
        axes[1].set_title("ReLU Derivative")
        
        st.markdown("""
        ### Rectified Linear Unit (ReLU)
        
        **Formula**: $\\text{ReLU}(z) = \\max(0, z)$
        
        **Output Range**: [0, ∞)
        
        **Advantages**:
        - No vanishing gradient for positive values
        - Computationally efficient
        - Sparse activation (many zeros)
        - Faster convergence
        
        **Disadvantages**:
        - "Dying ReLU" problem (neurons can get stuck at 0)
        - Not zero-centered
        
        **Use Cases**: Most popular for hidden layers in modern networks
        """)
        
    elif activation_choice == "Leaky ReLU":
        axes[0].plot(x, viz_data['leaky_relu'], 'b-', linewidth=2)
        axes[0].set_title('Leaky ReLU')
        axes[1].plot(x, viz_data['leaky_relu_derivative'], 'r-', linewidth=2)
        axes[1].set_title("Leaky ReLU Derivative")
        
        st.markdown("""
        ### Leaky ReLU
        
        **Formula**: $\\text{LeakyReLU}(z) = \\begin{cases} z & \\text{if } z > 0 \\\\ \\alpha z & \\text{if } z \\leq 0 \\end{cases}$
        
        Where α is typically 0.01
        
        **Output Range**: (-∞, ∞)
        
        **Advantages**:
        - Solves the "dying ReLU" problem
        - Allows gradient flow for negative inputs
        
        **Use Cases**: Deep networks where dying ReLU is a concern
        """)
        
    else:
        axes[0].plot(x, viz_data['sigmoid'], label='Sigmoid')
        axes[0].plot(x, viz_data['tanh'], label='Tanh')
        axes[0].plot(x, viz_data['relu'], label='ReLU')
        axes[0].plot(x, viz_data['leaky_relu'], label='Leaky ReLU')
        axes[0].legend()
        axes[0].set_title('All Activation Functions')
        
        axes[1].plot(x, viz_data['sigmoid_derivative'], label='Sigmoid')
        axes[1].plot(x, viz_data['tanh_derivative'], label='Tanh')
        axes[1].plot(x, viz_data['relu_derivative'], label='ReLU')
        axes[1].plot(x, viz_data['leaky_relu_derivative'], label='Leaky ReLU')
        axes[1].legend()
        axes[1].set_title('All Derivatives')
    
    for ax in axes:
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('z')
        ax.set_ylabel('f(z)')
    
    st.pyplot(fig)
    plt.close()


def show_perceptron():
    st.header("The Perceptron")
    
    st.markdown("""
    ## The Simplest Neural Network
    
    The **Perceptron**, invented by Frank Rosenblatt in 1957, is the fundamental building 
    block of neural networks. It's a single artificial neuron that can learn to classify 
    **linearly separable** patterns.
    
    ### Architecture
    
    ```
    [x₁] ──w₁──┐
    [x₂] ──w₂──┼──[Σ + b]──[step]──> ŷ
    [x₃] ──w₃──┘
    ```
    
    ### Mathematical Model
    
    1. **Linear Combination**: $z = \\sum_{i} w_i x_i + b$
    2. **Step Activation**: $\\hat{y} = \\begin{cases} 1 & \\text{if } z \\geq 0 \\\\ 0 & \\text{if } z < 0 \\end{cases}$
    
    ### Learning Rule
    
    The perceptron learning algorithm updates weights when a mistake is made:
    
    $$w_{new} = w_{old} + \\alpha \\cdot (y_{true} - y_{pred}) \\cdot x$$
    $$b_{new} = b_{old} + \\alpha \\cdot (y_{true} - y_{pred})$$
    
    Where α is the learning rate.
    """)
    
    st.subheader("Interactive Demo: Logic Gates")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        gate_choice = st.selectbox(
            "Select Logic Gate:",
            ["AND", "OR", "NAND", "XOR"]
        )
        
        learning_rate = st.slider(
            "Learning Rate:",
            min_value=0.01,
            max_value=1.0,
            value=0.1,
            step=0.01
        )
        
        epochs = st.slider(
            "Training Epochs:",
            min_value=10,
            max_value=200,
            value=100,
            step=10
        )
        
        train_btn = st.button("Train Perceptron")
    
    with col2:
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        gates = {
            'AND': np.array([0, 0, 0, 1]),
            'OR': np.array([0, 1, 1, 1]),
            'NAND': np.array([1, 1, 1, 0]),
            'XOR': np.array([0, 1, 1, 0])
        }
        y = gates[gate_choice]
        
        if train_btn or 'perceptron_result' not in st.session_state:
            perceptron = Perceptron(n_inputs=2, learning_rate=learning_rate)
            history = perceptron.train(X, y, epochs=epochs, verbose=False)
            predictions = perceptron.predict(X)
            
            st.session_state.perceptron_result = {
                'predictions': predictions,
                'history': history,
                'weights': perceptron.weights.copy(),
                'bias': perceptron.bias,
                'perceptron': perceptron
            }
        
        result = st.session_state.perceptron_result
        
        st.markdown(f"### {gate_choice} Gate Results")
        
        results_df = {
            'x₁': X[:, 0],
            'x₂': X[:, 1],
            'Expected': y,
            'Predicted': result['predictions'],
            'Correct': ['Yes' if y[i] == result['predictions'][i] else 'No' for i in range(4)]
        }
        st.dataframe(results_df, use_container_width=True)
        
        accuracy = np.mean(y == result['predictions'])
        
        if accuracy == 1.0:
            st.success(f"Perfect classification! Accuracy: {accuracy:.0%}")
        else:
            st.warning(f"Accuracy: {accuracy:.0%}")
            if gate_choice == "XOR":
                st.error("The perceptron cannot learn XOR - it's not linearly separable!")
    
    if gate_choice == "XOR":
        st.markdown("""
        ## The XOR Problem
        
        The XOR (exclusive OR) gate outputs 1 when inputs are **different**, 0 when **same**.
        
        **Why Perceptron Fails on XOR:**
        
        A perceptron creates a **linear decision boundary** (a straight line in 2D). 
        XOR requires a **non-linear** boundary - you cannot draw a single straight line 
        to separate the classes!
        
        ```
          x₂
          │
        1 ├───●(0,1)───────●(1,1)
          │   class 1      class 0
          │
        0 ├───●(0,0)───────●(1,0)
          │   class 0      class 1
          └───────────────────── x₁
              0             1
        ```
        
        **Solution**: Multi-Layer Perceptron (MLP) with hidden layers!
        """)


def show_mlp():
    st.header("Multi-Layer Perceptron (MLP)")
    
    st.markdown("""
    ## From Perceptron to Deep Networks
    
    The **Multi-Layer Perceptron** extends the simple perceptron by adding **hidden layers** 
    between input and output. This enables learning complex, non-linear patterns.
    
    ### Network Architecture
    
    ```
    Input Layer      Hidden Layer 1    Hidden Layer 2    Output Layer
    (4 neurons)      (10 neurons)      (8 neurons)       (3 neurons)
    
        [x₁]              [h₁]              [h₁]             [ŷ₁]
        [x₂]  ────W¹────  [h₂]  ────W²────  [h₂]  ────W³────  [ŷ₂]
        [x₃]              ...               ...               [ŷ₃]
        [x₄]              [h₁₀]             [h₈]
    ```
    
    ### Forward Propagation
    
    For each layer $l$ from 1 to L:
    
    $$Z^{[l]} = W^{[l]} \\cdot A^{[l-1]} + b^{[l]}$$
    $$A^{[l]} = f^{[l]}(Z^{[l]})$$
    
    Where:
    - $W^{[l]}$ is the weight matrix for layer $l$
    - $A^{[0]} = X$ (input data)
    - $f^{[l]}$ is the activation function (ReLU for hidden, Softmax for output)
    """)
    
    st.markdown("""
    ### Backpropagation
    
    The **chain rule** allows us to compute gradients for all parameters:
    
    **Output Layer:**
    $$dZ^{[L]} = A^{[L]} - Y$$
    $$dW^{[L]} = \\frac{1}{m} dZ^{[L]} \\cdot A^{[L-1]T}$$
    
    **Hidden Layers** (for $l = L-1, ..., 1$):
    $$dA^{[l]} = W^{[l+1]T} \\cdot dZ^{[l+1]}$$
    $$dZ^{[l]} = dA^{[l]} \\odot f'^{[l]}(Z^{[l]})$$
    $$dW^{[l]} = \\frac{1}{m} dZ^{[l]} \\cdot A^{[l-1]T}$$
    
    ### Weight Update (Gradient Descent)
    
    $$W^{[l]} := W^{[l]} - \\alpha \\cdot dW^{[l]}$$
    $$b^{[l]} := b^{[l]} - \\alpha \\cdot db^{[l]}$$
    """)
    
    st.subheader("Demo: MLP Solving XOR")
    
    if st.button("Solve XOR with MLP"):
        with st.spinner("Training MLP on XOR..."):
            result = solve_xor_with_mlp()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### XOR Truth Table")
            results_df = {
                'x₁': result['X'][:, 0],
                'x₂': result['X'][:, 1],
                'Expected': result['y'],
                'Predicted': result['predictions'],
                'Correct': ['Yes' if result['y'][i] == result['predictions'][i] else 'No' for i in range(4)]
            }
            st.dataframe(results_df, use_container_width=True)
            st.success(f"Accuracy: {result['accuracy']:.0%}")
        
        with col2:
            st.markdown("### Training Curves")
            fig, axes = plt.subplots(1, 2, figsize=(10, 3))
            
            axes[0].plot(result['history']['loss'])
            axes[0].set_title('Loss over Epochs')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Cross-Entropy Loss')
            axes[0].grid(True, alpha=0.3)
            
            axes[1].plot(result['history']['accuracy'])
            axes[1].set_title('Accuracy over Epochs')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        st.markdown("""
        The MLP with just **one hidden layer** successfully learns XOR! 
        This demonstrates the power of hidden layers in learning non-linear patterns.
        """)


def show_iris_prediction():
    st.header("Iris Species Prediction")
    
    st.markdown("""
    ## Predict Iris Species
    
    Use a trained neural network to classify iris flowers based on their measurements.
    Choose from single measurement input, batch CSV upload, or multi-image upload for classification.
    """)
    
    saved_models = get_saved_model_names()
    iris_models = [m for m in saved_models if 'iris' in m[0].lower()]
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Model Status")
        
        if st.button("Train Iris Classifier", type="primary"):
            with st.spinner("Training model..."):
                X_train, X_test, y_train, y_test, feature_names, class_names = load_iris_dataset(source='sklearn')
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                mlp = MLP(
                    layer_sizes=[4, 32, 16, 3],
                    learning_rate=0.01,
                    activation='relu',
                    weight_init='he'
                )
                
                mlp.fit(X_train_scaled, y_train, epochs=800, batch_size=16, verbose=False)
                
                y_pred = mlp.predict(X_test_scaled)
                accuracy = np.mean(y_pred == y_test)
                
                save_trained_model(
                    name='iris_classifier',
                    model=mlp,
                    scaler_mean=scaler.mean_,
                    scaler_std=scaler.scale_,
                    accuracy=accuracy,
                    class_names=list(class_names)
                )
                
                st.success(f"Model trained and saved! Test accuracy: {accuracy:.1%}")
                st.rerun()
        
        if iris_models:
            st.success(f"Active Model: {iris_models[0][0]} (Accuracy: {iris_models[0][1]:.1%})")
        else:
            st.warning("No trained model found. Click 'Train Iris Classifier' first.")
    
    with col2:
        st.subheader("Iris Flower Reference")
        iris_images = sorted([f for f in os.listdir('attached_assets') if f.startswith('iris-') and f.endswith('.jpg')])
        if iris_images[:6]:
            cols = st.columns(3)
            for idx, img_file in enumerate(iris_images[:6]):
                with cols[idx % 3]:
                    st.image(f"attached_assets/{img_file}", use_container_width=True)
    
    st.markdown("---")
    
    prediction_mode = st.radio(
        "Prediction Mode:",
        ["Single Measurement", "CSV Batch Upload", "Multi-Image Upload"],
        horizontal=True
    )
    
    if prediction_mode == "Single Measurement":
        st.subheader("Enter Flower Measurements")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
        with col2:
            sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
        with col3:
            petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
        with col4:
            petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)
        
        if st.button("Predict Species", key="single_predict"):
            model_data = load_trained_model('iris_classifier')
            
            if model_data is None:
                st.error("No trained model found. Please train the model first.")
            else:
                mlp = MLP.from_saved_weights(
                    layer_sizes=model_data['layer_sizes'],
                    activation=model_data['activation'],
                    weights=model_data['weights'],
                    biases=model_data['biases']
                )
                
                features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
                
                if model_data['scaler_mean'] is not None:
                    features = (features - model_data['scaler_mean']) / model_data['scaler_std']
                
                prediction = mlp.predict(features)[0]
                probabilities = mlp.predict_proba(features)[0]
                
                class_names = model_data['class_names']
                predicted_class = class_names[prediction]
                
                st.success(f"**Predicted Species: {predicted_class}**")
                
                st.markdown("#### Confidence Scores:")
                prob_cols = st.columns(3)
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
                for i, (name, prob) in enumerate(zip(class_names, probabilities)):
                    with prob_cols[i]:
                        st.metric(name, f"{prob:.1%}")
                        st.progress(float(prob))
    
    elif prediction_mode == "CSV Batch Upload":
        st.subheader("Upload CSV for Batch Prediction")
        
        st.markdown("""
        **Required CSV Format:**
        ```
        Id,SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm,Species
        1,5.1,3.5,1.4,0.2,Iris-setosa
        2,4.9,3.0,1.4,0.2,Iris-setosa
        ...
        ```
        The `Species` column is optional - if provided, it will be used to calculate accuracy.
        """)
        
        uploaded_csv = st.file_uploader("Upload CSV file", type=['csv'], key="csv_uploader")
        
        if uploaded_csv is not None:
            try:
                df = pd.read_csv(uploaded_csv)
                st.write(f"**Loaded {len(df)} samples**")
                st.dataframe(df.head(10), use_container_width=True)
                
                required_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    st.error(f"Missing required columns: {missing_cols}")
                else:
                    if st.button("Run Batch Prediction", type="primary"):
                        model_data = load_trained_model('iris_classifier')
                        
                        if model_data is None:
                            st.error("No trained model found. Please train the model first.")
                        else:
                            mlp = MLP.from_saved_weights(
                                layer_sizes=model_data['layer_sizes'],
                                activation=model_data['activation'],
                                weights=model_data['weights'],
                                biases=model_data['biases']
                            )
                            
                            features = df[required_cols].values
                            
                            if model_data['scaler_mean'] is not None:
                                features = (features - model_data['scaler_mean']) / model_data['scaler_std']
                            
                            predictions = mlp.predict(features)
                            probabilities = mlp.predict_proba(features)
                            
                            class_names = model_data['class_names']
                            predicted_classes = [class_names[p] for p in predictions]
                            
                            results_df = df.copy()
                            results_df['Predicted_Species'] = predicted_classes
                            results_df['Confidence'] = [f"{max(prob)*100:.1f}%" for prob in probabilities]
                            
                            for i, name in enumerate(class_names):
                                results_df[f'Prob_{name}'] = [f"{prob[i]*100:.1f}%" for prob in probabilities]
                            
                            if 'Species' in df.columns:
                                species_map = {
                                    'Iris-setosa': 'setosa',
                                    'Iris-versicolor': 'versicolor', 
                                    'Iris-virginica': 'virginica',
                                    'setosa': 'setosa',
                                    'versicolor': 'versicolor',
                                    'virginica': 'virginica'
                                }
                                actual_mapped = df['Species'].map(species_map).fillna(df['Species'])
                                results_df['Correct'] = actual_mapped == results_df['Predicted_Species']
                                accuracy = results_df['Correct'].mean()
                                
                                st.success(f"Batch Prediction Complete! Accuracy: {accuracy:.1%}")
                            else:
                                st.success("Batch Prediction Complete!")
                            
                            st.subheader("Prediction Results")
                            st.dataframe(results_df, use_container_width=True)
                            
                            st.subheader("Prediction Summary")
                            summary_counts = pd.Series(predicted_classes).value_counts()
                            
                            col1, col2 = st.columns([1, 1])
                            with col1:
                                st.markdown("**Species Distribution:**")
                                for species, count in summary_counts.items():
                                    st.write(f"- {species}: {count} ({count/len(predictions)*100:.1f}%)")
                            
                            with col2:
                                fig, ax = plt.subplots(figsize=(6, 4))
                                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
                                ax.pie(summary_counts.values, labels=summary_counts.index, autopct='%1.1f%%', colors=colors[:len(summary_counts)])
                                ax.set_title('Predicted Species Distribution')
                                st.pyplot(fig)
                                plt.close()
                            
                            csv_buffer = io.StringIO()
                            results_df.to_csv(csv_buffer, index=False)
                            st.download_button(
                                label="Download Results as CSV",
                                data=csv_buffer.getvalue(),
                                file_name="iris_predictions.csv",
                                mime="text/csv"
                            )
                            
            except Exception as e:
                st.error(f"Error reading CSV: {str(e)}")
    
    else:
        st.subheader("Upload Multiple Iris Flower Images")
        st.info("Upload multiple images of iris flowers for classification. The model will analyze each image to help identify the species.")
        
        uploaded_files = st.file_uploader(
            "Choose iris flower images", 
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            key="multi_image_uploader"
        )
        
        if uploaded_files:
            st.write(f"**Uploaded {len(uploaded_files)} images**")
            
            cols_per_row = 4
            for i in range(0, len(uploaded_files), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    if i + j < len(uploaded_files):
                        with col:
                            st.image(uploaded_files[i + j], caption=f"Image {i+j+1}", use_container_width=True)
            
            st.markdown("---")
            st.subheader("Manual Classification Input")
            st.markdown("""
            Since image-based classification requires a CNN model, please enter measurements for each uploaded image.
            This helps correlate visual features with measurement data.
            """)
            
            if 'image_measurements' not in st.session_state:
                st.session_state.image_measurements = []
            
            for idx, uploaded_file in enumerate(uploaded_files):
                with st.expander(f"Enter measurements for Image {idx + 1}: {uploaded_file.name}"):
                    cols = st.columns([1, 1, 1, 1, 1])
                    with cols[0]:
                        st.image(uploaded_file, use_container_width=True)
                    with cols[1]:
                        sl = st.number_input(f"Sepal Length", min_value=0.0, max_value=10.0, value=5.0, step=0.1, key=f"sl_{idx}")
                    with cols[2]:
                        sw = st.number_input(f"Sepal Width", min_value=0.0, max_value=10.0, value=3.0, step=0.1, key=f"sw_{idx}")
                    with cols[3]:
                        pl = st.number_input(f"Petal Length", min_value=0.0, max_value=10.0, value=1.5, step=0.1, key=f"pl_{idx}")
                    with cols[4]:
                        pw = st.number_input(f"Petal Width", min_value=0.0, max_value=10.0, value=0.2, step=0.1, key=f"pw_{idx}")
            
            if st.button("Classify All Images", type="primary"):
                model_data = load_trained_model('iris_classifier')
                
                if model_data is None:
                    st.error("No trained model found. Please train the model first.")
                else:
                    mlp = MLP.from_saved_weights(
                        layer_sizes=model_data['layer_sizes'],
                        activation=model_data['activation'],
                        weights=model_data['weights'],
                        biases=model_data['biases']
                    )
                    
                    results = []
                    for idx, uploaded_file in enumerate(uploaded_files):
                        sl = st.session_state.get(f"sl_{idx}", 5.0)
                        sw = st.session_state.get(f"sw_{idx}", 3.0)
                        pl = st.session_state.get(f"pl_{idx}", 1.5)
                        pw = st.session_state.get(f"pw_{idx}", 0.2)
                        
                        features = np.array([[sl, sw, pl, pw]])
                        
                        if model_data['scaler_mean'] is not None:
                            features = (features - model_data['scaler_mean']) / model_data['scaler_std']
                        
                        prediction = mlp.predict(features)[0]
                        probabilities = mlp.predict_proba(features)[0]
                        
                        class_names = model_data['class_names']
                        predicted_class = class_names[prediction]
                        confidence = max(probabilities)
                        
                        results.append({
                            'image_name': uploaded_file.name,
                            'predicted_species': predicted_class,
                            'confidence': confidence,
                            'probabilities': probabilities
                        })
                    
                    st.success(f"Classified {len(results)} images!")
                    
                    st.subheader("Classification Results")
                    cols_per_row = 3
                    for i in range(0, len(results), cols_per_row):
                        cols = st.columns(cols_per_row)
                        for j, col in enumerate(cols):
                            if i + j < len(results):
                                result = results[i + j]
                                with col:
                                    st.image(uploaded_files[i + j], use_container_width=True)
                                    st.markdown(f"**{result['predicted_species']}**")
                                    st.write(f"Confidence: {result['confidence']:.1%}")
                    
                    results_df = pd.DataFrame([{
                        'Image': r['image_name'],
                        'Predicted Species': r['predicted_species'],
                        'Confidence': f"{r['confidence']*100:.1f}%"
                    } for r in results])
                    
                    csv_buffer = io.StringIO()
                    results_df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv_buffer.getvalue(),
                        file_name="image_classification_results.csv",
                        mime="text/csv"
                    )
            
            st.markdown("---")
            st.markdown("""
            **Note:** For automated image classification, a Convolutional Neural Network (CNN) would be required:
            - A CNN can learn visual features directly from images
            - Transfer learning from pre-trained models (ResNet, VGG) could be applied
            - A large dataset of labeled iris flower images would improve accuracy
            """)


def show_image_recognition():
    st.header("Iris Image Recognition")
    
    st.markdown("""
    ## Image-Based Iris Classification
    
    This feature allows you to classify iris flowers directly from images using neural networks.
    The system extracts visual features (colors, shapes, textures) from images and uses a trained
    MLP to identify the iris species.
    
    **Three Iris Species:**
    - **Iris-setosa** - Known for smaller petals
    - **Iris-versicolor** - Medium-sized petals  
    - **Iris-virginica** - Larger petals
    """)
    
    mode_tabs = st.tabs(["Classify Image", "Manage Reference Images", "Train Image Model"])
    
    with mode_tabs[0]:
        show_image_classification()
    
    with mode_tabs[1]:
        show_reference_management()
    
    with mode_tabs[2]:
        show_image_model_training()


def show_image_classification():
    st.subheader("Classify Iris from Image")
    
    model_data = load_iris_image_model('iris_image_classifier')
    
    if model_data is None:
        st.warning("No trained image classifier found. Please train a model first in the 'Train Image Model' tab.")
        st.info("You need to upload reference images and train the model before classification.")
        return
    
    st.success(f"Image classifier loaded - Accuracy: {model_data['accuracy']:.1%} (trained on {model_data['num_reference_images']} images)")
    
    input_method = st.radio("Choose input method:", ["Upload Image", "Camera Capture"], horizontal=True)
    
    image_data = None
    
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader(
            "Upload an iris flower image",
            type=['jpg', 'jpeg', 'png', 'webp'],
            help="Upload a clear photo of an iris flower for classification"
        )
        if uploaded_file is not None:
            image_data = uploaded_file.read()
            st.image(image_data, caption="Uploaded Image", width=300)
    else:
        camera_image = st.camera_input("Take a photo of an iris flower")
        if camera_image is not None:
            image_data = camera_image.read()
    
    if image_data is not None:
        if st.button("Classify Iris Species", type="primary"):
            with st.spinner("Analyzing image..."):
                try:
                    features = extract_features(image_data)
                    
                    features_normalized, _, _ = normalize_features(
                        features.reshape(1, -1),
                        mean=model_data['feature_scaler_mean'],
                        std=model_data['feature_scaler_std']
                    )
                    
                    mlp = MLP(
                        layer_sizes=model_data['layer_sizes'],
                        activation=model_data['activation']
                    )
                    mlp.weights = model_data['weights']
                    mlp.biases = model_data['biases']
                    
                    X_T = features_normalized.T
                    A, _ = mlp.forward_propagation(X_T)
                    probabilities = A[mlp.L][:, 0]
                    
                    predicted_class = np.argmax(probabilities)
                    confidence = probabilities[predicted_class]
                    species_name = SPECIES_MAPPING[predicted_class]
                    
                    st.markdown("---")
                    st.subheader("Classification Result")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Predicted Species", species_name)
                    with col2:
                        st.metric("Confidence", f"{confidence:.1%}")
                    with col3:
                        st.metric("Species ID", predicted_class)
                    
                    st.markdown("### Probability Distribution")
                    prob_df = pd.DataFrame({
                        'Species': [SPECIES_MAPPING[i] for i in range(3)],
                        'Probability': probabilities
                    })
                    
                    fig, ax = plt.subplots(figsize=(8, 4))
                    colors = ['#2ecc71' if i == predicted_class else '#3498db' for i in range(3)]
                    ax.barh(prob_df['Species'], prob_df['Probability'], color=colors)
                    ax.set_xlim(0, 1)
                    ax.set_xlabel('Probability')
                    ax.set_title('Species Probability Distribution')
                    for i, (species, prob) in enumerate(zip(prob_df['Species'], prob_df['Probability'])):
                        ax.text(prob + 0.02, i, f'{prob:.1%}', va='center')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                except Exception as e:
                    st.error(f"Error classifying image: {str(e)}")


def show_reference_management():
    st.subheader("Manage Reference Images")
    
    st.markdown("""
    Upload reference images for each iris species. These images will be used to train
    the image classifier. For best results, upload at least 5-10 images per species.
    """)
    
    ref_counts = get_reference_image_count()
    if ref_counts:
        st.markdown("### Current Reference Image Counts")
        cols = st.columns(3)
        for idx, (species, count) in enumerate(ref_counts.items()):
            with cols[idx % 3]:
                st.metric(species, f"{count} images")
    
    st.markdown("---")
    st.markdown("### Upload New Reference Images")
    
    species_choice = st.selectbox(
        "Select Iris Species:",
        options=list(SPECIES_MAPPING.values()),
        help="Choose the species for the images you're uploading"
    )
    species_id = SPECIES_ID_MAPPING[species_choice]
    
    uploaded_files = st.file_uploader(
        f"Upload {species_choice} images",
        type=['jpg', 'jpeg', 'png', 'webp'],
        accept_multiple_files=True,
        help="Upload clear photos of this iris species"
    )
    
    if uploaded_files:
        st.write(f"Selected {len(uploaded_files)} images")
        
        preview_cols = st.columns(min(len(uploaded_files), 5))
        for idx, file in enumerate(uploaded_files[:5]):
            with preview_cols[idx]:
                st.image(file, use_container_width=True)
        
        if st.button("Save Reference Images", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            saved_count = 0
            for idx, file in enumerate(uploaded_files):
                try:
                    image_bytes = file.read()
                    file.seek(0)
                    
                    os.makedirs("reference_images", exist_ok=True)
                    image_path = f"reference_images/{species_choice}_{int(time.time())}_{idx}.png"
                    
                    img = Image.open(io.BytesIO(image_bytes))
                    img.save(image_path)
                    
                    image_id = save_iris_reference_image(
                        species_id=species_id,
                        species_name=species_choice,
                        image_path=image_path,
                        filename=file.name
                    )
                    
                    features = extract_features(image_bytes)
                    save_iris_image_features(
                        reference_image_id=image_id,
                        species_id=species_id,
                        species_name=species_choice,
                        feature_vector=features
                    )
                    
                    saved_count += 1
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                    status_text.text(f"Processing image {idx + 1}/{len(uploaded_files)}")
                    
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")
            
            st.success(f"Successfully saved {saved_count} reference images with extracted features!")
            st.rerun()
    
    st.markdown("---")
    st.markdown("### View Existing Reference Images")
    
    view_species = st.selectbox(
        "View images for species:",
        options=["All"] + list(SPECIES_MAPPING.values()),
        key="view_species"
    )
    
    if view_species == "All":
        ref_images = get_iris_reference_images()
    else:
        species_id = SPECIES_ID_MAPPING[view_species]
        ref_images = get_iris_reference_images(species_id)
    
    if ref_images:
        st.write(f"Found {len(ref_images)} reference images")
        
        cols = st.columns(4)
        for idx, img_record in enumerate(ref_images[:12]):
            with cols[idx % 4]:
                if img_record['image_path'] and os.path.exists(img_record['image_path']):
                    st.image(img_record['image_path'], use_container_width=True)
                    st.caption(f"{img_record['species_name']}")
                    if st.button("Delete", key=f"del_{img_record['id']}"):
                        delete_iris_reference_image(img_record['id'])
                        if img_record['image_path'] and os.path.exists(img_record['image_path']):
                            os.remove(img_record['image_path'])
                        st.rerun()
    else:
        st.info("No reference images found. Upload some images above.")


def show_image_model_training():
    st.subheader("Train Image Classifier")
    
    st.markdown("""
    Train a neural network to classify iris species based on image features.
    The model extracts visual features (colors, shapes, textures) from your
    reference images and learns to distinguish between species.
    """)
    
    feature_counts = get_feature_count()
    total_features = sum(feature_counts.values()) if feature_counts else 0
    
    st.markdown("### Training Data Status")
    if feature_counts:
        cols = st.columns(3)
        for idx, (species, count) in enumerate(feature_counts.items()):
            with cols[idx % 3]:
                st.metric(species, f"{count} samples")
        
        st.write(f"**Total training samples:** {total_features}")
        
        min_per_class = min(feature_counts.values()) if feature_counts else 0
        if min_per_class < 3:
            st.warning("Need at least 3 images per species for effective training. Please upload more reference images.")
    else:
        st.warning("No training data available. Please upload reference images first.")
        return
    
    st.markdown("---")
    st.markdown("### Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        hidden_size_1 = st.slider("Hidden Layer 1 Size", 16, 128, 64, 8)
        hidden_size_2 = st.slider("Hidden Layer 2 Size", 8, 64, 32, 8)
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[0.001, 0.005, 0.01, 0.05, 0.1],
            value=0.01
        )
    
    with col2:
        epochs = st.slider("Training Epochs", 100, 2000, 500, 100)
        activation = st.selectbox("Activation Function", ["relu", "sigmoid", "tanh"])
        batch_size = st.slider("Batch Size", 4, 32, 16, 4)
    
    if st.button("Train Image Classifier", type="primary"):
        X, y, species_names = get_all_iris_image_features()
        
        if X is None or len(X) < 6:
            st.error("Not enough training data. Please upload more reference images.")
            return
        
        with st.spinner("Training image classifier..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]
            
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            X_train_norm, scaler_mean, scaler_std = normalize_features(X_train)
            X_test_norm, _, _ = normalize_features(X_test, mean=scaler_mean, std=scaler_std)
            
            feature_dim = X_train.shape[1]
            layer_sizes = [feature_dim, hidden_size_1, hidden_size_2, 3]
            
            mlp = MLP(
                layer_sizes=layer_sizes,
                learning_rate=learning_rate,
                activation=activation
            )
            
            y_train_onehot = one_hot_encode(y_train, 3)
            y_test_onehot = one_hot_encode(y_test, 3)
            
            status_text.text("Training in progress...")
            mlp.fit(X_train_norm, y_train_onehot, epochs=epochs, batch_size=batch_size, verbose=False)
            
            progress_bar.progress(1.0)
            
            train_predictions = mlp.predict(X_train_norm)
            test_predictions = mlp.predict(X_test_norm)
            
            train_accuracy = np.mean(train_predictions == y_train)
            test_accuracy = np.mean(test_predictions == y_test)
            
            save_iris_image_model(
                name='iris_image_classifier',
                model=mlp,
                feature_scaler_mean=scaler_mean,
                feature_scaler_std=scaler_std,
                accuracy=test_accuracy,
                num_reference_images=total_features,
                feature_version='v1'
            )
            
            st.success("Model trained and saved successfully!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Training Accuracy", f"{train_accuracy:.1%}")
            with col2:
                st.metric("Test Accuracy", f"{test_accuracy:.1%}")
            
            st.markdown("### Model Architecture")
            st.write(f"Layer sizes: {layer_sizes}")
            st.write(f"Total parameters: {sum(mlp.weights[l].size + mlp.biases[l].size for l in range(1, mlp.L + 1))}")
    
    existing_model = load_iris_image_model('iris_image_classifier')
    if existing_model:
        st.markdown("---")
        st.markdown("### Current Trained Model")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{existing_model['accuracy']:.1%}" if existing_model['accuracy'] else "N/A")
        with col2:
            st.metric("Training Images", existing_model['num_reference_images'] or 0)
        with col3:
            st.metric("Created", existing_model['created_at'][:10] if existing_model['created_at'] else "N/A")


def show_iris_training():
    st.header("Train MLP on Iris Dataset")
    
    st.markdown("""
    ## The Iris Dataset
    
    A classic dataset for classification containing 150 samples of iris flowers:
    - **4 Features**: sepal length, sepal width, petal length, petal width
    - **3 Classes**: setosa, versicolor, virginica (50 samples each)
    """)
    
    iris_images = sorted([f for f in os.listdir('attached_assets') if f.startswith('iris-') and f.endswith('.jpg')])
    if iris_images:
        with st.expander("View Iris Setosa Flower Images", expanded=False):
            st.markdown("### Iris Setosa Samples")
            st.markdown("These are real photographs of Iris Setosa flowers - one of the three species in the dataset.")
            cols = st.columns(6)
            for idx, img_file in enumerate(iris_images[:12]):
                with cols[idx % 6]:
                    st.image(f"attached_assets/{img_file}", use_container_width=True)
            if len(iris_images) > 12:
                cols2 = st.columns(6)
                for idx, img_file in enumerate(iris_images[12:]):
                    with cols2[idx % 6]:
                        st.image(f"attached_assets/{img_file}", use_container_width=True)
    
    db_sample_count = get_iris_sample_count()
    if db_sample_count > 0:
        st.info(f"Database contains {db_sample_count} Iris samples loaded from CSV.")
    
    st.sidebar.header("Data Source")
    
    data_source = st.sidebar.radio(
        "Load Iris data from:",
        ["sklearn (built-in)", "Database (CSV import)"],
        help="Choose whether to load from sklearn or the database"
    )
    
    if data_source == "Database (CSV import)" and db_sample_count == 0:
        if os.path.exists('attached_assets/Iris_1765047715889.csv'):
            if st.sidebar.button("Import CSV to Database"):
                try:
                    count = import_iris_from_csv('attached_assets/Iris_1765047715889.csv')
                    st.sidebar.success(f"Imported {count} samples to database!")
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Error importing: {e}")
        else:
            st.sidebar.warning("No CSV file found. Using sklearn instead.")
            data_source = "sklearn (built-in)"
    
    st.sidebar.header("Model Configuration")
    
    hidden_layer_1 = st.sidebar.slider(
        "Hidden Layer 1 Size:",
        min_value=4,
        max_value=64,
        value=16,
        step=4
    )
    
    hidden_layer_2 = st.sidebar.slider(
        "Hidden Layer 2 Size:",
        min_value=0,
        max_value=32,
        value=8,
        step=4,
        help="Set to 0 for a single hidden layer"
    )
    
    activation = st.sidebar.selectbox(
        "Activation Function:",
        ["relu", "tanh", "sigmoid"]
    )
    
    learning_rate = st.sidebar.slider(
        "Learning Rate:",
        min_value=0.001,
        max_value=0.5,
        value=0.01,
        step=0.001,
        format="%.3f"
    )
    
    epochs = st.sidebar.slider(
        "Training Epochs:",
        min_value=100,
        max_value=2000,
        value=500,
        step=100
    )
    
    batch_size = st.sidebar.selectbox(
        "Batch Size:",
        [8, 16, 32, 64],
        index=1
    )
    
    if st.sidebar.button("Train Model", type="primary"):
        source = 'database' if 'Database' in data_source else 'sklearn'
        X_train, X_test, y_train, y_test, feature_names, class_names = load_iris_dataset(source=source)
        
        if hidden_layer_2 > 0:
            layer_sizes = [4, hidden_layer_1, hidden_layer_2, 3]
        else:
            layer_sizes = [4, hidden_layer_1, 3]
        
        mlp = MLP(
            layer_sizes=layer_sizes,
            learning_rate=learning_rate,
            activation=activation,
            weight_init='he' if activation == 'relu' else 'xavier'
        )
        
        st.subheader("Network Architecture")
        st.code(mlp.get_architecture_summary())
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("Training in progress..."):
            history = mlp.fit(
                X_train, y_train, 
                epochs=epochs, 
                batch_size=batch_size, 
                verbose=False,
                validation_data=(X_test, y_test)
            )
        
        progress_bar.progress(100)
        status_text.text("Training complete!")
        
        y_pred_train = mlp.predict(X_train)
        y_pred_test = mlp.predict(X_test)
        y_proba_test = mlp.predict_proba(X_test)
        
        train_metrics = compute_metrics(y_train, y_pred_train, list(class_names))
        test_metrics = compute_metrics(y_test, y_pred_test, list(class_names))
        
        st.subheader("Training Curves")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        epochs_range = range(len(history['loss']))
        
        axes[0].plot(epochs_range, history['loss'], label='Training Loss')
        if history['val_loss']:
            axes[0].plot(epochs_range, history['val_loss'], label='Validation Loss')
        axes[0].set_title('Loss over Epochs')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Cross-Entropy Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(epochs_range, history['accuracy'], label='Training Accuracy')
        if history['val_accuracy']:
            axes[1].plot(epochs_range, history['val_accuracy'], label='Validation Accuracy')
        axes[1].set_title('Accuracy over Epochs')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.subheader("Model Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Training Set")
            st.metric("Accuracy", f"{train_metrics['accuracy']:.2%}")
            st.metric("F1-Score (Macro)", f"{train_metrics['f1_macro']:.4f}")
        
        with col2:
            st.markdown("### Test Set")
            st.metric("Accuracy", f"{test_metrics['accuracy']:.2%}")
            st.metric("F1-Score (Macro)", f"{test_metrics['f1_macro']:.4f}")
        
        st.subheader("Confusion Matrix (Test Set)")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = test_metrics['confusion_matrix']
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names,
               yticklabels=class_names,
               ylabel='True Label',
               xlabel='Predicted Label',
               title='Confusion Matrix')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.subheader("Per-Class Metrics (Test Set)")
        
        metrics_data = {
            'Class': [m['class_name'] for m in test_metrics['per_class_metrics']],
            'Precision': [f"{m['precision']:.4f}" for m in test_metrics['per_class_metrics']],
            'Recall': [f"{m['recall']:.4f}" for m in test_metrics['per_class_metrics']],
            'F1-Score': [f"{m['f1_score']:.4f}" for m in test_metrics['per_class_metrics']],
            'Support': [m['support'] for m in test_metrics['per_class_metrics']]
        }
        st.dataframe(metrics_data, use_container_width=True)
        
        st.subheader("Sample Predictions")
        
        np.random.seed(42)
        sample_indices = np.random.choice(len(y_test), min(5, len(y_test)), replace=False)
        
        samples_data = []
        for idx in sample_indices:
            true_label = class_names[y_test[idx]]
            pred_label = class_names[y_pred_test[idx]]
            probs = y_proba_test[idx]
            correct = "Yes" if y_test[idx] == y_pred_test[idx] else "No"
            
            prob_str = ", ".join([f"{class_names[j]}: {probs[j]:.3f}" for j in range(len(class_names))])
            
            samples_data.append({
                'True Label': true_label,
                'Predicted': pred_label,
                'Correct': correct,
                'Probabilities': prob_str
            })
        
        st.dataframe(samples_data, use_container_width=True)
        
        run_id = save_training_run(
            dataset_name='Iris',
            model_type='MLP',
            layer_sizes=layer_sizes,
            activation=activation,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            optimizer='sgd',
            regularization=None,
            final_train_accuracy=train_metrics['accuracy'],
            final_test_accuracy=test_metrics['accuracy'],
            final_train_loss=history['loss'][-1] if history['loss'] else None,
            final_test_loss=history['val_loss'][-1] if history['val_loss'] else None,
            train_loss_history=history['loss'],
            train_accuracy_history=history['accuracy'],
            val_loss_history=history['val_loss'],
            val_accuracy_history=history['val_accuracy'],
            confusion_matrix=cm,
            f1_score=test_metrics['f1_macro'],
            training_time_seconds=None
        )
        
        st.success(f"Training complete! Results saved to database (Run #{run_id}).")


def show_mnist_training():
    st.header("Train MLP on MNIST Dataset")
    
    st.markdown("""
    ## The MNIST Dataset
    
    The classic handwritten digit recognition dataset:
    - **784 Features**: 28x28 grayscale images (flattened)
    - **10 Classes**: Digits 0-9
    - **70,000 Samples**: 60,000 training + 10,000 test (we use a subset for faster training)
    
    This is the "Hello World" of deep learning, demonstrating that our MLP can handle real image data!
    """)
    
    st.sidebar.header("MNIST Configuration")
    
    n_samples = st.sidebar.select_slider(
        "Number of Samples:",
        options=[1000, 2000, 5000, 10000, 20000],
        value=5000,
        help="Fewer samples = faster training, more samples = better accuracy"
    )
    
    hidden_layer_1 = st.sidebar.slider(
        "Hidden Layer 1 Size:",
        min_value=32,
        max_value=256,
        value=128,
        step=32,
        key="mnist_h1"
    )
    
    hidden_layer_2 = st.sidebar.slider(
        "Hidden Layer 2 Size:",
        min_value=0,
        max_value=128,
        value=64,
        step=32,
        help="Set to 0 for a single hidden layer",
        key="mnist_h2"
    )
    
    activation = st.sidebar.selectbox(
        "Activation Function:",
        ["relu", "tanh", "sigmoid"],
        key="mnist_activation"
    )
    
    learning_rate = st.sidebar.slider(
        "Learning Rate:",
        min_value=0.001,
        max_value=0.5,
        value=0.1,
        step=0.001,
        format="%.3f",
        key="mnist_lr"
    )
    
    epochs = st.sidebar.slider(
        "Training Epochs:",
        min_value=10,
        max_value=100,
        value=30,
        step=5,
        key="mnist_epochs"
    )
    
    batch_size = st.sidebar.selectbox(
        "Batch Size:",
        [32, 64, 128, 256],
        index=1,
        key="mnist_batch"
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Sample MNIST Digits")
        
        if 'mnist_samples' not in st.session_state:
            st.session_state.mnist_samples = None
            st.session_state.mnist_load_error = None
        
        if st.session_state.mnist_samples is None and st.session_state.mnist_load_error is None:
            if st.button("Load Sample Digits"):
                try:
                    with st.spinner("Loading MNIST samples..."):
                        X_sample, _, y_sample, _, _ = load_mnist_dataset(n_samples=1000)
                        sample_imgs, sample_labels = get_mnist_sample_images(X_sample, y_sample, n_per_class=2)
                        st.session_state.mnist_samples = (sample_imgs, sample_labels)
                        st.rerun()
                except Exception as e:
                    st.session_state.mnist_load_error = str(e)
                    st.rerun()
        
        if st.session_state.mnist_load_error:
            st.warning(f"Could not load MNIST samples: {st.session_state.mnist_load_error}")
            st.info("You can still train on MNIST by clicking the 'Train on MNIST' button in the sidebar.")
        elif st.session_state.mnist_samples:
            sample_imgs, sample_labels = st.session_state.mnist_samples
            
            fig, axes = plt.subplots(2, 10, figsize=(12, 3))
            for i in range(20):
                row = i // 10
                col = i % 10
                axes[row, col].imshow(sample_imgs[i], cmap='gray')
                axes[row, col].axis('off')
                axes[row, col].set_title(str(sample_labels[i]), fontsize=10)
            plt.suptitle('Sample Digits from MNIST', fontsize=12)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Click the button above to load sample MNIST digits.")
    
    with col2:
        st.subheader("Dataset Info")
        st.markdown("""
        **Image Size**: 28 x 28 pixels
        
        **Input Features**: 784 (flattened)
        
        **Normalization**: 0-255 → 0-1
        
        **Classes**: 10 digits
        """)
    
    if st.sidebar.button("Train on MNIST", type="primary", key="mnist_train_btn"):
        with st.spinner(f"Loading {n_samples} MNIST samples..."):
            X_train, X_test, y_train, y_test, class_names = load_mnist_dataset(n_samples=n_samples)
        
        st.info(f"Loaded {len(X_train)} training samples and {len(X_test)} test samples")
        
        if hidden_layer_2 > 0:
            layer_sizes = [784, hidden_layer_1, hidden_layer_2, 10]
        else:
            layer_sizes = [784, hidden_layer_1, 10]
        
        mlp = MLP(
            layer_sizes=layer_sizes,
            learning_rate=learning_rate,
            activation=activation,
            weight_init='he' if activation == 'relu' else 'xavier'
        )
        
        st.subheader("Network Architecture")
        st.code(mlp.get_architecture_summary())
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Training in progress... This may take a moment for MNIST.")
        
        with st.spinner("Training MLP on MNIST..."):
            history = mlp.fit(
                X_train, y_train, 
                epochs=epochs, 
                batch_size=batch_size, 
                verbose=False,
                validation_data=(X_test, y_test)
            )
        
        progress_bar.progress(100)
        status_text.text("Training complete!")
        
        y_pred_train = mlp.predict(X_train)
        y_pred_test = mlp.predict(X_test)
        y_proba_test = mlp.predict_proba(X_test)
        
        train_metrics = compute_metrics(y_train, y_pred_train, class_names)
        test_metrics = compute_metrics(y_test, y_pred_test, class_names)
        
        st.subheader("Training Curves")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        epochs_range = range(len(history['loss']))
        
        axes[0].plot(epochs_range, history['loss'], label='Training Loss')
        if history['val_loss']:
            axes[0].plot(epochs_range, history['val_loss'], label='Validation Loss')
        axes[0].set_title('Loss over Epochs')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Cross-Entropy Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(epochs_range, history['accuracy'], label='Training Accuracy')
        if history['val_accuracy']:
            axes[1].plot(epochs_range, history['val_accuracy'], label='Validation Accuracy')
        axes[1].set_title('Accuracy over Epochs')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.subheader("Model Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Training Set")
            st.metric("Accuracy", f"{train_metrics['accuracy']:.2%}")
            st.metric("F1-Score (Macro)", f"{train_metrics['f1_macro']:.4f}")
        
        with col2:
            st.markdown("### Test Set")
            st.metric("Accuracy", f"{test_metrics['accuracy']:.2%}")
            st.metric("F1-Score (Macro)", f"{test_metrics['f1_macro']:.4f}")
        
        st.subheader("Confusion Matrix (Test Set)")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        cm = test_metrics['confusion_matrix']
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names,
               yticklabels=class_names,
               ylabel='True Label',
               xlabel='Predicted Label',
               title='MNIST Confusion Matrix')
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=8)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.subheader("Sample Predictions")
        
        np.random.seed(42)
        sample_indices = np.random.choice(len(y_test), min(10, len(y_test)), replace=False)
        
        fig, axes = plt.subplots(2, 5, figsize=(12, 5))
        for i, idx in enumerate(sample_indices):
            row = i // 5
            col = i % 5
            img = X_test[idx].reshape(28, 28)
            true_label = y_test[idx]
            pred_label = y_pred_test[idx]
            correct = true_label == pred_label
            
            axes[row, col].imshow(img, cmap='gray')
            axes[row, col].axis('off')
            color = 'green' if correct else 'red'
            axes[row, col].set_title(f'True: {true_label}, Pred: {pred_label}', 
                                      color=color, fontsize=10)
        
        plt.suptitle('Sample Predictions (Green=Correct, Red=Wrong)', fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        run_id = save_training_run(
            dataset_name='MNIST',
            model_type='MLP',
            layer_sizes=layer_sizes,
            activation=activation,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            optimizer='sgd',
            regularization=None,
            final_train_accuracy=train_metrics['accuracy'],
            final_test_accuracy=test_metrics['accuracy'],
            final_train_loss=history['loss'][-1] if history['loss'] else None,
            final_test_loss=history['val_loss'][-1] if history['val_loss'] else None,
            train_loss_history=history['loss'],
            train_accuracy_history=history['accuracy'],
            val_loss_history=history['val_loss'],
            val_accuracy_history=history['val_accuracy'],
            confusion_matrix=cm,
            f1_score=test_metrics['f1_macro'],
            training_time_seconds=None
        )
        
        st.success(f"Training complete! Results saved to database (Run #{run_id}).")
        st.balloons()


def show_training_history():
    st.header("Training History")
    
    st.markdown("""
    View and analyze your previous training runs stored in the database.
    Compare different model configurations and their performance.
    """)
    
    history = get_training_history(limit=20)
    
    if not history:
        st.info("No training runs found. Train a model on the Iris dataset to see your history here.")
        return
    
    st.subheader(f"Recent Training Runs ({len(history)} total)")
    
    history_df = []
    for run in history:
        history_df.append({
            'ID': run['id'],
            'Timestamp': run['timestamp'][:19].replace('T', ' '),
            'Dataset': run['dataset_name'],
            'Architecture': ' → '.join(map(str, run['layer_sizes'])),
            'Activation': run['activation'],
            'LR': f"{run['learning_rate']:.4f}",
            'Epochs': run['epochs'],
            'Train Acc': f"{run['final_train_accuracy']:.2%}" if run['final_train_accuracy'] else 'N/A',
            'Test Acc': f"{run['final_test_accuracy']:.2%}" if run['final_test_accuracy'] else 'N/A',
            'F1': f"{run['f1_score']:.4f}" if run['f1_score'] else 'N/A'
        })
    
    st.dataframe(history_df, use_container_width=True)
    
    st.subheader("View Run Details")
    
    run_ids = [run['id'] for run in history]
    selected_run_id = st.selectbox(
        "Select a training run to view details:",
        options=run_ids,
        format_func=lambda x: f"Run #{x} - {next((r['timestamp'][:19].replace('T', ' ') for r in history if r['id'] == x), '')}"
    )
    
    if selected_run_id:
        run_details = get_training_run_details(selected_run_id)
        
        if run_details:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Model Configuration")
                st.write(f"**Dataset:** {run_details['dataset_name']}")
                st.write(f"**Model:** {run_details['model_type']}")
                st.write(f"**Architecture:** {' → '.join(map(str, run_details['layer_sizes']))}")
                st.write(f"**Activation:** {run_details['activation']}")
                st.write(f"**Learning Rate:** {run_details['learning_rate']}")
                st.write(f"**Epochs:** {run_details['epochs']}")
                st.write(f"**Batch Size:** {run_details['batch_size']}")
            
            with col2:
                st.markdown("### Performance Metrics")
                if run_details['final_train_accuracy']:
                    st.metric("Training Accuracy", f"{run_details['final_train_accuracy']:.2%}")
                if run_details['final_test_accuracy']:
                    st.metric("Test Accuracy", f"{run_details['final_test_accuracy']:.2%}")
                if run_details['f1_score']:
                    st.metric("F1 Score", f"{run_details['f1_score']:.4f}")
            
            if run_details['train_loss_history'] and run_details['train_accuracy_history']:
                st.markdown("### Training Curves")
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                
                epochs_range = range(len(run_details['train_loss_history']))
                
                axes[0].plot(epochs_range, run_details['train_loss_history'], label='Training Loss')
                if run_details['val_loss_history']:
                    axes[0].plot(epochs_range, run_details['val_loss_history'], label='Validation Loss')
                axes[0].set_title('Loss over Epochs')
                axes[0].set_xlabel('Epoch')
                axes[0].set_ylabel('Loss')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                axes[1].plot(epochs_range, run_details['train_accuracy_history'], label='Training Accuracy')
                if run_details['val_accuracy_history']:
                    axes[1].plot(epochs_range, run_details['val_accuracy_history'], label='Validation Accuracy')
                axes[1].set_title('Accuracy over Epochs')
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('Accuracy')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            if run_details['confusion_matrix'] is not None:
                st.markdown("### Confusion Matrix")
                cm = run_details['confusion_matrix']
                class_names = ['setosa', 'versicolor', 'virginica']
                
                fig, ax = plt.subplots(figsize=(6, 5))
                im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                ax.figure.colorbar(im, ax=ax)
                
                ax.set(xticks=np.arange(cm.shape[1]),
                       yticks=np.arange(cm.shape[0]),
                       xticklabels=class_names,
                       yticklabels=class_names,
                       ylabel='True Label',
                       xlabel='Predicted Label')
                
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                
                thresh = cm.max() / 2.
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, format(int(cm[i, j]), 'd'),
                                ha="center", va="center",
                                color="white" if cm[i, j] > thresh else "black")
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            if st.button("Delete This Run", type="secondary"):
                if delete_training_run(selected_run_id):
                    st.success(f"Run #{selected_run_id} deleted successfully.")
                    st.rerun()
                else:
                    st.error("Failed to delete the run.")


if __name__ == "__main__":
    main()
