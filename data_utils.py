"""
================================================================================
DATA UTILITIES MODULE
================================================================================
This module provides data loading, preprocessing, and evaluation utilities
for training neural networks.

Contents:
    1. Dataset Loading (Iris, synthetic data)
    2. Data Preprocessing (normalization, train/test split, one-hot encoding)
    3. Evaluation Metrics (accuracy, precision, recall, F1-score)
    4. Confusion Matrix Computation
================================================================================
"""

import numpy as np
from sklearn.datasets import load_iris, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)


def load_iris_dataset(test_size=0.2, random_state=42, normalize=True, source='sklearn'):
    """
    Load and preprocess the Iris dataset.
    
    The Iris Dataset:
        - 150 samples total
        - 4 features: sepal length, sepal width, petal length, petal width
        - 3 classes: setosa (0), versicolor (1), virginica (2)
        - 50 samples per class
        - Collected by Edgar Anderson in 1936
    
    Preprocessing Steps:
        1. Load raw data from scikit-learn or database
        2. Split into training (80%) and test (20%) sets
        3. Normalize features using StandardScaler (z-score normalization)
           z = (x - μ) / σ
    
    Args:
        test_size (float): Proportion of data for testing (default: 0.2)
        random_state (int): Random seed for reproducibility
        normalize (bool): Whether to apply z-score normalization
        source (str): Data source - 'sklearn' or 'database'
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names, class_names)
    """
    if source == 'database':
        try:
            from db_utils import load_iris_from_database, get_iris_sample_count
            if get_iris_sample_count() > 0:
                X, y, feature_names, class_names = load_iris_from_database()
            else:
                iris = load_iris()
                X = iris.data
                y = iris.target
                feature_names = list(iris.feature_names)
                class_names = iris.target_names
        except ImportError:
            iris = load_iris()
            X = iris.data
            y = iris.target
            feature_names = list(iris.feature_names)
            class_names = iris.target_names
    else:
        iris = load_iris()
        X = iris.data
        y = iris.target
        feature_names = list(iris.feature_names)
        class_names = iris.target_names
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, feature_names, class_names


def load_mnist_dataset(n_samples=10000, test_size=0.2, random_state=42):
    """
    Load and preprocess the MNIST dataset.
    
    The MNIST Dataset:
        - 70,000 samples total (60,000 training, 10,000 test)
        - 28x28 grayscale images = 784 features (flattened)
        - 10 classes: digits 0-9
        - Pixel values: 0-255 (normalized to 0-1)
    
    Preprocessing Steps:
        1. Load from sklearn's fetch_openml
        2. Flatten images from 28x28 to 784 features
        3. Normalize pixel values from 0-255 to 0-1
        4. Split into training and test sets
    
    Args:
        n_samples (int): Number of samples to use (for faster training)
        test_size (float): Proportion of data for testing (default: 0.2)
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, class_names)
    """
    from sklearn.datasets import fetch_openml
    
    try:
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    except Exception:
        mnist = fetch_openml('mnist_784', as_frame=False, parser='auto')
    
    X = mnist.data.astype(np.float32)
    y = mnist.target.astype(np.int32)
    
    if n_samples is not None and n_samples < len(X):
        np.random.seed(random_state)
        indices = np.random.choice(len(X), n_samples, replace=False)
        X = X[indices]
        y = y[indices]
    
    X = X / 255.0
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    class_names = [str(i) for i in range(10)]
    
    return X_train, X_test, y_train, y_test, class_names


def get_mnist_sample_images(X, y, n_per_class=3, random_state=42):
    """
    Get sample images for each digit class from MNIST.
    
    Args:
        X (np.ndarray): MNIST features (flattened images)
        y (np.ndarray): MNIST labels
        n_per_class (int): Number of samples per digit
        random_state (int): Random seed
        
    Returns:
        tuple: (images, labels) where images is shape (n_classes * n_per_class, 28, 28)
    """
    np.random.seed(random_state)
    
    sample_images = []
    sample_labels = []
    
    for digit in range(10):
        digit_indices = np.where(y == digit)[0]
        selected = np.random.choice(digit_indices, min(n_per_class, len(digit_indices)), replace=False)
        
        for idx in selected:
            img = X[idx].reshape(28, 28)
            sample_images.append(img)
            sample_labels.append(digit)
    
    return np.array(sample_images), np.array(sample_labels)


def generate_classification_data(n_samples=300, n_features=2, n_classes=3,
                                  random_state=42):
    """
    Generate synthetic classification data for visualization.
    
    Creates clusters of points around randomly generated centers.
    Useful for testing and demonstrating classification algorithms.
    
    Args:
        n_samples (int): Total number of samples
        n_features (int): Number of features per sample
        n_classes (int): Number of classes
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X, y) - features and labels
    """
    np.random.seed(random_state)
    
    samples_per_class = n_samples // n_classes
    
    centers = np.random.randn(n_classes, n_features) * 3
    
    X = []
    y = []
    
    for class_idx in range(n_classes):
        class_samples = np.random.randn(samples_per_class, n_features) * 0.5
        class_samples += centers[class_idx]
        X.append(class_samples)
        y.extend([class_idx] * samples_per_class)
    
    X = np.vstack(X)
    y = np.array(y)
    
    shuffle_idx = np.random.permutation(len(y))
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    return X, y


def one_hot_encode(y, n_classes=None):
    """
    Convert class labels to one-hot encoding.
    
    One-Hot Encoding:
        Converts categorical labels into binary vectors.
        Each class becomes a column with 1 for that class, 0 for others.
    
    Example:
        y = [0, 1, 2, 1, 0]
        n_classes = 3
        
        Result:
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1],
         [0, 1, 0],
         [1, 0, 0]]
    
    Why One-Hot Encoding?
        - Treats all classes equally (no ordinal relationship assumed)
        - Works well with softmax output and cross-entropy loss
        - Enables multi-class classification with neural networks
    
    Args:
        y (np.ndarray): Class labels, shape (n_samples,)
        n_classes (int): Number of classes (auto-detected if None)
        
    Returns:
        np.ndarray: One-hot encoded labels, shape (n_samples, n_classes)
    """
    y = np.array(y, dtype=int)
    
    if n_classes is None:
        n_classes = len(np.unique(y))
    
    one_hot = np.zeros((len(y), n_classes))
    one_hot[np.arange(len(y)), y] = 1
    
    return one_hot


def compute_metrics(y_true, y_pred, class_names=None):
    """
    Compute comprehensive classification metrics.
    
    Metrics Computed:
    
    1. Accuracy:
        Accuracy = (TP + TN) / Total = Correct / Total
        - Overall correctness of predictions
        
    2. Precision (per class):
        Precision = TP / (TP + FP)
        - Of all predicted positives, how many are actually positive?
        - Important when false positives are costly
        
    3. Recall / Sensitivity (per class):
        Recall = TP / (TP + FN)
        - Of all actual positives, how many did we find?
        - Important when false negatives are costly
        
    4. F1-Score (per class):
        F1 = 2 × (Precision × Recall) / (Precision + Recall)
        - Harmonic mean of precision and recall
        - Balances both metrics
    
    5. Confusion Matrix:
        - Rows: Actual classes
        - Columns: Predicted classes
        - Diagonal: Correct predictions
        - Off-diagonal: Misclassifications
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        class_names (list): Names of classes for display
        
    Returns:
        dict: Dictionary containing all metrics
    """
    n_classes = len(np.unique(y_true))
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]
    
    accuracy = accuracy_score(y_true, y_pred)
    
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    per_class_metrics = []
    for i, name in enumerate(class_names):
        per_class_metrics.append({
            'class_name': name,
            'precision': precision[i],
            'recall': recall[i],
            'f1_score': f1[i],
            'support': np.sum(y_true == i)
        })
    
    return {
        'accuracy': accuracy,
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'confusion_matrix': conf_matrix,
        'per_class_metrics': per_class_metrics,
        'class_names': class_names
    }


def format_confusion_matrix(conf_matrix, class_names):
    """
    Format confusion matrix as a readable string.
    
    Args:
        conf_matrix (np.ndarray): Confusion matrix
        class_names (list): Class names
        
    Returns:
        str: Formatted confusion matrix string
    """
    lines = []
    
    header = "           Predicted"
    lines.append(header)
    
    class_header = "Actual    " + "  ".join([f"{name[:6]:>6}" for name in class_names])
    lines.append(class_header)
    lines.append("-" * len(class_header))
    
    for i, name in enumerate(class_names):
        row = f"{name[:8]:<8}  " + "  ".join([f"{val:>6}" for val in conf_matrix[i]])
        lines.append(row)
    
    return "\n".join(lines)


def print_sample_predictions(X, y_true, y_pred, probabilities, class_names, n_samples=5):
    """
    Print sample predictions with probabilities.
    
    Args:
        X (np.ndarray): Input features
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        probabilities (np.ndarray): Class probabilities
        class_names (list): Class names
        n_samples (int): Number of samples to display
    """
    print("\nSample Predictions:")
    print("=" * 70)
    
    indices = np.random.choice(len(y_true), min(n_samples, len(y_true)), replace=False)
    
    for i, idx in enumerate(indices):
        true_label = class_names[y_true[idx]]
        pred_label = class_names[y_pred[idx]]
        probs = probabilities[idx]
        
        correct = "✓" if y_true[idx] == y_pred[idx] else "✗"
        
        print(f"\nExample {i+1}:")
        print(f"  True: {true_label:<15} | Predicted: {pred_label:<15} {correct}")
        print(f"  Probabilities: {', '.join([f'{class_names[j]}: {probs[j]:.3f}' for j in range(len(class_names))])}")


def create_mini_batches(X, Y, batch_size):
    """
    Split data into mini-batches for training.
    
    Mini-Batch Gradient Descent:
        - Compromise between batch GD and stochastic GD
        - Batch size typically: 32, 64, 128, 256
        - Larger batches: more stable gradients, slower per update
        - Smaller batches: noisier gradients, faster per update
    
    Args:
        X (np.ndarray): Features, shape (n_features, m_samples)
        Y (np.ndarray): Labels, shape (n_classes, m_samples)
        batch_size (int): Size of each mini-batch
        
    Returns:
        list: List of (X_batch, Y_batch) tuples
    """
    m = X.shape[1]
    mini_batches = []
    
    permutation = np.random.permutation(m)
    X_shuffled = X[:, permutation]
    Y_shuffled = Y[:, permutation]
    
    n_complete_batches = m // batch_size
    
    for k in range(n_complete_batches):
        X_batch = X_shuffled[:, k * batch_size:(k + 1) * batch_size]
        Y_batch = Y_shuffled[:, k * batch_size:(k + 1) * batch_size]
        mini_batches.append((X_batch, Y_batch))
    
    if m % batch_size != 0:
        X_batch = X_shuffled[:, n_complete_batches * batch_size:]
        Y_batch = Y_shuffled[:, n_complete_batches * batch_size:]
        mini_batches.append((X_batch, Y_batch))
    
    return mini_batches


if __name__ == "__main__":
    print("="*60)
    print("DATA UTILITIES DEMONSTRATION")
    print("="*60)
    
    X_train, X_test, y_train, y_test, feature_names, class_names = load_iris_dataset()
    
    print(f"\nIris Dataset Loaded:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features: {feature_names}")
    print(f"  Classes: {list(class_names)}")
    
    y_pred_dummy = y_test.copy()
    y_pred_dummy[0] = (y_pred_dummy[0] + 1) % 3
    
    metrics = compute_metrics(y_test, y_pred_dummy, class_names)
    
    print(f"\nMetrics Example:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Macro F1: {metrics['f1_macro']:.4f}")
    print("\nConfusion Matrix:")
    print(format_confusion_matrix(metrics['confusion_matrix'], list(class_names)))
