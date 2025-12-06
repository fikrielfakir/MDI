"""
================================================================================
DATABASE UTILITIES MODULE
================================================================================
This module provides SQLite database functionality for the Deep Learning app.

Features:
    1. Schema management (create tables if not exist)
    2. Iris dataset storage and retrieval from CSV/database
    3. Training history logging and retrieval
    4. Model configuration persistence
================================================================================
"""

import sqlite3
import json
import os
from datetime import datetime
import numpy as np
import pandas as pd

DATABASE_PATH = "deep_learning.db"


def get_connection():
    """
    Get a connection to the SQLite database.
    
    Returns:
        sqlite3.Connection: Database connection
    """
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """
    Initialize the database schema.
    Creates all required tables if they don't exist.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS iris_samples (
            id INTEGER PRIMARY KEY,
            sepal_length REAL NOT NULL,
            sepal_width REAL NOT NULL,
            petal_length REAL NOT NULL,
            petal_width REAL NOT NULL,
            species TEXT NOT NULL,
            species_id INTEGER NOT NULL
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            dataset_name TEXT NOT NULL,
            model_type TEXT NOT NULL,
            layer_sizes TEXT NOT NULL,
            activation TEXT NOT NULL,
            learning_rate REAL NOT NULL,
            epochs INTEGER NOT NULL,
            batch_size INTEGER NOT NULL,
            optimizer TEXT DEFAULT 'sgd',
            regularization TEXT DEFAULT NULL,
            final_train_accuracy REAL,
            final_test_accuracy REAL,
            final_train_loss REAL,
            final_test_loss REAL,
            train_loss_history TEXT,
            train_accuracy_history TEXT,
            val_loss_history TEXT,
            val_accuracy_history TEXT,
            confusion_matrix TEXT,
            f1_score REAL,
            training_time_seconds REAL
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_configs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            layer_sizes TEXT NOT NULL,
            activation TEXT NOT NULL,
            learning_rate REAL NOT NULL,
            epochs INTEGER NOT NULL,
            batch_size INTEGER NOT NULL,
            optimizer TEXT DEFAULT 'sgd',
            regularization TEXT DEFAULT NULL,
            description TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS saved_models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL,
            model_type TEXT NOT NULL,
            layer_sizes TEXT NOT NULL,
            activation TEXT NOT NULL,
            weights_json TEXT NOT NULL,
            biases_json TEXT NOT NULL,
            scaler_mean TEXT,
            scaler_std TEXT,
            accuracy REAL,
            class_names TEXT
        )
    ''')
    
    conn.commit()
    conn.close()


def save_trained_model(name, model, scaler_mean=None, scaler_std=None, accuracy=None, class_names=None):
    """
    Save a trained MLP model to the database.
    
    Args:
        name: Unique name for the model
        model: Trained MLP instance
        scaler_mean: Mean values used for normalization
        scaler_std: Std values used for normalization
        accuracy: Model accuracy on test set
        class_names: List of class names
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    weights_dict = {str(k): v.tolist() for k, v in model.weights.items()}
    biases_dict = {str(k): v.tolist() for k, v in model.biases.items()}
    
    cursor.execute('DELETE FROM saved_models WHERE name = ?', (name,))
    
    cursor.execute('''
        INSERT INTO saved_models 
        (name, created_at, model_type, layer_sizes, activation, weights_json, biases_json,
         scaler_mean, scaler_std, accuracy, class_names)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        name,
        datetime.now().isoformat(),
        'MLP',
        json.dumps(model.layer_sizes),
        model.activation_name,
        json.dumps(weights_dict),
        json.dumps(biases_dict),
        json.dumps(scaler_mean.tolist()) if scaler_mean is not None else None,
        json.dumps(scaler_std.tolist()) if scaler_std is not None else None,
        accuracy,
        json.dumps(class_names) if class_names else None
    ))
    
    conn.commit()
    conn.close()


def load_trained_model(name):
    """
    Load a trained model from the database.
    
    Args:
        name: Name of the saved model
        
    Returns:
        dict with model info or None if not found
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM saved_models WHERE name = ?', (name,))
    row = cursor.fetchone()
    conn.close()
    
    if row is None:
        return None
    
    return {
        'name': row['name'],
        'created_at': row['created_at'],
        'layer_sizes': json.loads(row['layer_sizes']),
        'activation': row['activation'],
        'weights': {int(k): np.array(v) for k, v in json.loads(row['weights_json']).items()},
        'biases': {int(k): np.array(v) for k, v in json.loads(row['biases_json']).items()},
        'scaler_mean': np.array(json.loads(row['scaler_mean'])) if row['scaler_mean'] else None,
        'scaler_std': np.array(json.loads(row['scaler_std'])) if row['scaler_std'] else None,
        'accuracy': row['accuracy'],
        'class_names': json.loads(row['class_names']) if row['class_names'] else None
    }


def get_saved_model_names():
    """Get list of all saved model names."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT name, accuracy, created_at FROM saved_models ORDER BY created_at DESC')
    rows = cursor.fetchall()
    conn.close()
    return [(row['name'], row['accuracy'], row['created_at']) for row in rows]


def import_iris_from_csv(csv_path):
    """
    Import Iris dataset from CSV file into the database.
    
    Args:
        csv_path (str): Path to the Iris CSV file
        
    Returns:
        int: Number of records imported
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    species_mapping = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    }
    
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('DELETE FROM iris_samples')
    
    for _, row in df.iterrows():
        species_name = str(row['Species'])
        cursor.execute('''
            INSERT INTO iris_samples 
            (id, sepal_length, sepal_width, petal_length, petal_width, species, species_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            int(row['Id']),
            float(row['SepalLengthCm']),
            float(row['SepalWidthCm']),
            float(row['PetalLengthCm']),
            float(row['PetalWidthCm']),
            species_name,
            species_mapping[species_name]
        ))
    
    conn.commit()
    count = cursor.rowcount
    conn.close()
    
    return len(df)


def load_iris_from_database():
    """
    Load the Iris dataset from the database.
    
    Returns:
        tuple: (X, y, feature_names, class_names)
            - X: Feature matrix (n_samples, 4)
            - y: Target labels (n_samples,)
            - feature_names: List of feature names
            - class_names: List of class names
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT sepal_length, sepal_width, petal_length, petal_width, species_id
        FROM iris_samples
        ORDER BY id
    ''')
    
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return None, None, None, None
    
    X = np.array([[row['sepal_length'], row['sepal_width'], 
                   row['petal_length'], row['petal_width']] for row in rows])
    y = np.array([row['species_id'] for row in rows])
    
    feature_names = ['sepal length (cm)', 'sepal width (cm)', 
                     'petal length (cm)', 'petal width (cm)']
    class_names = np.array(['setosa', 'versicolor', 'virginica'])
    
    return X, y, feature_names, class_names


def get_iris_sample_count():
    """
    Get the number of Iris samples in the database.
    
    Returns:
        int: Number of samples
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM iris_samples')
    count = cursor.fetchone()[0]
    conn.close()
    return count


def save_training_run(
    dataset_name,
    model_type,
    layer_sizes,
    activation,
    learning_rate,
    epochs,
    batch_size,
    optimizer='sgd',
    regularization=None,
    final_train_accuracy=None,
    final_test_accuracy=None,
    final_train_loss=None,
    final_test_loss=None,
    train_loss_history=None,
    train_accuracy_history=None,
    val_loss_history=None,
    val_accuracy_history=None,
    confusion_matrix=None,
    f1_score=None,
    training_time_seconds=None
):
    """
    Save a training run to the database.
    
    Args:
        dataset_name (str): Name of the dataset used
        model_type (str): Type of model (e.g., 'MLP', 'Perceptron')
        layer_sizes (list): Network architecture
        activation (str): Activation function used
        learning_rate (float): Learning rate
        epochs (int): Number of epochs
        batch_size (int): Batch size
        optimizer (str): Optimizer used
        regularization (str): Regularization technique if any
        final_train_accuracy (float): Final training accuracy
        final_test_accuracy (float): Final test accuracy
        final_train_loss (float): Final training loss
        final_test_loss (float): Final test loss
        train_loss_history (list): Training loss per epoch
        train_accuracy_history (list): Training accuracy per epoch
        val_loss_history (list): Validation loss per epoch
        val_accuracy_history (list): Validation accuracy per epoch
        confusion_matrix (np.ndarray): Confusion matrix
        f1_score (float): F1 score
        training_time_seconds (float): Training duration
        
    Returns:
        int: ID of the saved training run
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO training_history (
            timestamp, dataset_name, model_type, layer_sizes, activation,
            learning_rate, epochs, batch_size, optimizer, regularization,
            final_train_accuracy, final_test_accuracy, final_train_loss, final_test_loss,
            train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history,
            confusion_matrix, f1_score, training_time_seconds
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().isoformat(),
        dataset_name,
        model_type,
        json.dumps(layer_sizes),
        activation,
        learning_rate,
        epochs,
        batch_size,
        optimizer,
        regularization,
        final_train_accuracy,
        final_test_accuracy,
        final_train_loss,
        final_test_loss,
        json.dumps(train_loss_history) if train_loss_history else None,
        json.dumps(train_accuracy_history) if train_accuracy_history else None,
        json.dumps(val_loss_history) if val_loss_history else None,
        json.dumps(val_accuracy_history) if val_accuracy_history else None,
        json.dumps(confusion_matrix.tolist()) if confusion_matrix is not None else None,
        f1_score,
        training_time_seconds
    ))
    
    conn.commit()
    run_id = cursor.lastrowid
    conn.close()
    
    return run_id


def get_training_history(limit=20):
    """
    Get recent training runs from the database.
    
    Args:
        limit (int): Maximum number of runs to return
        
    Returns:
        list: List of training run dictionaries
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, timestamp, dataset_name, model_type, layer_sizes, activation,
               learning_rate, epochs, batch_size, optimizer, regularization,
               final_train_accuracy, final_test_accuracy, f1_score, training_time_seconds
        FROM training_history
        ORDER BY timestamp DESC
        LIMIT ?
    ''', (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    
    history = []
    for row in rows:
        history.append({
            'id': row['id'],
            'timestamp': row['timestamp'],
            'dataset_name': row['dataset_name'],
            'model_type': row['model_type'],
            'layer_sizes': json.loads(row['layer_sizes']),
            'activation': row['activation'],
            'learning_rate': row['learning_rate'],
            'epochs': row['epochs'],
            'batch_size': row['batch_size'],
            'optimizer': row['optimizer'],
            'regularization': row['regularization'],
            'final_train_accuracy': row['final_train_accuracy'],
            'final_test_accuracy': row['final_test_accuracy'],
            'f1_score': row['f1_score'],
            'training_time_seconds': row['training_time_seconds']
        })
    
    return history


def get_training_run_details(run_id):
    """
    Get detailed information about a specific training run.
    
    Args:
        run_id (int): ID of the training run
        
    Returns:
        dict: Training run details including full history
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM training_history WHERE id = ?', (run_id,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return None
    
    return {
        'id': row['id'],
        'timestamp': row['timestamp'],
        'dataset_name': row['dataset_name'],
        'model_type': row['model_type'],
        'layer_sizes': json.loads(row['layer_sizes']),
        'activation': row['activation'],
        'learning_rate': row['learning_rate'],
        'epochs': row['epochs'],
        'batch_size': row['batch_size'],
        'optimizer': row['optimizer'],
        'regularization': row['regularization'],
        'final_train_accuracy': row['final_train_accuracy'],
        'final_test_accuracy': row['final_test_accuracy'],
        'final_train_loss': row['final_train_loss'],
        'final_test_loss': row['final_test_loss'],
        'train_loss_history': json.loads(row['train_loss_history']) if row['train_loss_history'] else None,
        'train_accuracy_history': json.loads(row['train_accuracy_history']) if row['train_accuracy_history'] else None,
        'val_loss_history': json.loads(row['val_loss_history']) if row['val_loss_history'] else None,
        'val_accuracy_history': json.loads(row['val_accuracy_history']) if row['val_accuracy_history'] else None,
        'confusion_matrix': np.array(json.loads(row['confusion_matrix'])) if row['confusion_matrix'] else None,
        'f1_score': row['f1_score'],
        'training_time_seconds': row['training_time_seconds']
    }


def delete_training_run(run_id):
    """
    Delete a training run from the database.
    
    Args:
        run_id (int): ID of the training run to delete
        
    Returns:
        bool: True if deleted, False if not found
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM training_history WHERE id = ?', (run_id,))
    conn.commit()
    deleted = cursor.rowcount > 0
    conn.close()
    return deleted


def save_model_config(name, layer_sizes, activation, learning_rate, epochs, 
                      batch_size, optimizer='sgd', regularization=None, description=None):
    """
    Save a model configuration for later reuse.
    
    Args:
        name (str): Configuration name
        layer_sizes (list): Network architecture
        activation (str): Activation function
        learning_rate (float): Learning rate
        epochs (int): Number of epochs
        batch_size (int): Batch size
        optimizer (str): Optimizer name
        regularization (str): Regularization technique
        description (str): Optional description
        
    Returns:
        int: ID of the saved configuration
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO model_configs 
        (name, created_at, layer_sizes, activation, learning_rate, epochs, 
         batch_size, optimizer, regularization, description)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        name,
        datetime.now().isoformat(),
        json.dumps(layer_sizes),
        activation,
        learning_rate,
        epochs,
        batch_size,
        optimizer,
        regularization,
        description
    ))
    
    conn.commit()
    config_id = cursor.lastrowid
    conn.close()
    
    return config_id


def get_model_configs():
    """
    Get all saved model configurations.
    
    Returns:
        list: List of configuration dictionaries
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM model_configs ORDER BY created_at DESC
    ''')
    
    rows = cursor.fetchall()
    conn.close()
    
    configs = []
    for row in rows:
        configs.append({
            'id': row['id'],
            'name': row['name'],
            'created_at': row['created_at'],
            'layer_sizes': json.loads(row['layer_sizes']),
            'activation': row['activation'],
            'learning_rate': row['learning_rate'],
            'epochs': row['epochs'],
            'batch_size': row['batch_size'],
            'optimizer': row['optimizer'],
            'regularization': row['regularization'],
            'description': row['description']
        })
    
    return configs


init_database()

if os.path.exists('attached_assets/Iris_1765047715889.csv') and get_iris_sample_count() == 0:
    try:
        import_iris_from_csv('attached_assets/Iris_1765047715889.csv')
        print("Iris dataset imported from CSV to database.")
    except Exception as e:
        print(f"Could not auto-import Iris CSV: {e}")
