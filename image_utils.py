"""
================================================================================
IMAGE UTILITIES MODULE
================================================================================
This module provides NumPy-based image feature extraction for iris classification.

Features extracted:
    1. Color histograms (RGB channels)
    2. Edge detection using Sobel filters
    3. Shape descriptors (area, intensity distribution)
    4. Texture features

All implementations are from scratch using NumPy, following the project's
educational philosophy.
================================================================================
"""

import numpy as np
from PIL import Image
import io


def load_image_from_bytes(image_bytes):
    """
    Load an image from bytes and convert to NumPy array.
    
    Args:
        image_bytes: Raw image data (bytes or BytesIO)
        
    Returns:
        np.ndarray: Image as RGB array (H, W, 3)
    """
    if isinstance(image_bytes, bytes):
        image_bytes = io.BytesIO(image_bytes)
    
    img = Image.open(image_bytes)
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    return np.array(img)


def load_image_from_path(image_path):
    """
    Load an image from file path.
    
    Args:
        image_path: Path to image file
        
    Returns:
        np.ndarray: Image as RGB array (H, W, 3)
    """
    img = Image.open(image_path)
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    return np.array(img)


def resize_image(image, target_size=(128, 128)):
    """
    Resize image to target size using PIL.
    
    Args:
        image: NumPy array (H, W, 3)
        target_size: Tuple (width, height)
        
    Returns:
        np.ndarray: Resized image
    """
    pil_img = Image.fromarray(image.astype(np.uint8))
    pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
    return np.array(pil_img)


def rgb_to_grayscale(image):
    """
    Convert RGB image to grayscale using luminosity method.
    
    Formula: Y = 0.299*R + 0.587*G + 0.114*B
    
    Args:
        image: RGB image array (H, W, 3)
        
    Returns:
        np.ndarray: Grayscale image (H, W)
    """
    return np.dot(image[..., :3], [0.299, 0.587, 0.114])


def compute_color_histogram(image, bins=32):
    """
    Compute color histogram for each RGB channel.
    
    Args:
        image: RGB image array (H, W, 3)
        bins: Number of bins per channel
        
    Returns:
        np.ndarray: Flattened histogram (bins * 3,)
    """
    histograms = []
    
    for channel in range(3):
        hist, _ = np.histogram(image[:, :, channel].flatten(), 
                               bins=bins, range=(0, 256))
        hist = hist.astype(np.float32)
        hist = hist / (hist.sum() + 1e-7)
        histograms.append(hist)
    
    return np.concatenate(histograms)


def compute_hsv_histogram(image, bins=32):
    """
    Compute histogram in HSV color space.
    
    Args:
        image: RGB image array (H, W, 3)
        bins: Number of bins
        
    Returns:
        np.ndarray: HSV histogram features
    """
    image_normalized = image.astype(np.float32) / 255.0
    
    r, g, b = image_normalized[:,:,0], image_normalized[:,:,1], image_normalized[:,:,2]
    
    v = np.max(image_normalized, axis=2)
    c = v - np.min(image_normalized, axis=2)
    s = np.where(v > 0, c / (v + 1e-7), 0)
    
    h = np.zeros_like(v)
    mask = c > 0
    
    r_max = (v == r) & mask
    g_max = (v == g) & mask
    b_max = (v == b) & mask
    
    h[r_max] = 60 * (((g[r_max] - b[r_max]) / (c[r_max] + 1e-7)) % 6)
    h[g_max] = 60 * (((b[g_max] - r[g_max]) / (c[g_max] + 1e-7)) + 2)
    h[b_max] = 60 * (((r[b_max] - g[b_max]) / (c[b_max] + 1e-7)) + 4)
    
    h_hist, _ = np.histogram(h.flatten(), bins=bins, range=(0, 360))
    s_hist, _ = np.histogram(s.flatten(), bins=bins//2, range=(0, 1))
    v_hist, _ = np.histogram(v.flatten(), bins=bins//2, range=(0, 1))
    
    h_hist = h_hist.astype(np.float32) / (h_hist.sum() + 1e-7)
    s_hist = s_hist.astype(np.float32) / (s_hist.sum() + 1e-7)
    v_hist = v_hist.astype(np.float32) / (v_hist.sum() + 1e-7)
    
    return np.concatenate([h_hist, s_hist, v_hist])


def sobel_edge_detection(grayscale_image):
    """
    Apply Sobel edge detection using convolution.
    
    Args:
        grayscale_image: Grayscale image (H, W)
        
    Returns:
        tuple: (gradient_magnitude, gradient_direction)
    """
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
    
    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=np.float32)
    
    def convolve2d(image, kernel):
        h, w = image.shape
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2
        
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
        output = np.zeros_like(image)
        
        for i in range(h):
            for j in range(w):
                output[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
        
        return output
    
    gx = convolve2d(grayscale_image, sobel_x)
    gy = convolve2d(grayscale_image, sobel_y)
    
    magnitude = np.sqrt(gx**2 + gy**2)
    direction = np.arctan2(gy, gx)
    
    return magnitude, direction


def compute_edge_features(grayscale_image, bins=16):
    """
    Compute edge-based features from image.
    
    Args:
        grayscale_image: Grayscale image (H, W)
        bins: Number of histogram bins
        
    Returns:
        np.ndarray: Edge feature vector
    """
    magnitude, direction = sobel_edge_detection(grayscale_image)
    
    mag_normalized = magnitude / (magnitude.max() + 1e-7)
    
    mag_hist, _ = np.histogram(mag_normalized.flatten(), bins=bins, range=(0, 1))
    mag_hist = mag_hist.astype(np.float32) / (mag_hist.sum() + 1e-7)
    
    dir_hist, _ = np.histogram(direction.flatten(), bins=bins, range=(-np.pi, np.pi))
    dir_hist = dir_hist.astype(np.float32) / (dir_hist.sum() + 1e-7)
    
    edge_stats = np.array([
        magnitude.mean(),
        magnitude.std(),
        magnitude.max(),
        np.percentile(magnitude, 75),
        np.percentile(magnitude, 90)
    ])
    
    edge_stats = edge_stats / (edge_stats.max() + 1e-7)
    
    return np.concatenate([mag_hist, dir_hist, edge_stats])


def compute_texture_features(grayscale_image):
    """
    Compute simple texture features using local statistics.
    
    Args:
        grayscale_image: Grayscale image (H, W)
        
    Returns:
        np.ndarray: Texture feature vector
    """
    h, w = grayscale_image.shape
    block_size = 16
    
    features = []
    
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block = grayscale_image[i:i+block_size, j:j+block_size]
            features.extend([block.mean(), block.std()])
    
    features = np.array(features)
    
    target_size = 64
    if len(features) > target_size:
        indices = np.linspace(0, len(features)-1, target_size, dtype=int)
        features = features[indices]
    elif len(features) < target_size:
        features = np.pad(features, (0, target_size - len(features)), mode='constant')
    
    features = features / (features.max() + 1e-7)
    
    return features


def compute_shape_features(grayscale_image, threshold=0.5):
    """
    Compute shape-related features from the image.
    
    Args:
        grayscale_image: Grayscale image (H, W)
        threshold: Threshold for binarization
        
    Returns:
        np.ndarray: Shape feature vector
    """
    normalized = grayscale_image / 255.0
    binary = (normalized > threshold).astype(np.float32)
    
    area = binary.sum() / binary.size
    
    h, w = binary.shape
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    
    total = binary.sum() + 1e-7
    centroid_y = (y_coords * binary).sum() / total
    centroid_x = (x_coords * binary).sum() / total
    
    centroid_y_normalized = centroid_y / h
    centroid_x_normalized = centroid_x / w
    
    horizontal_hist = binary.sum(axis=0)
    vertical_hist = binary.sum(axis=1)
    
    horizontal_hist = horizontal_hist / (horizontal_hist.sum() + 1e-7)
    vertical_hist = vertical_hist / (vertical_hist.sum() + 1e-7)
    
    horizontal_bins = 16
    vertical_bins = 16
    
    h_indices = np.linspace(0, len(horizontal_hist)-1, horizontal_bins, dtype=int)
    v_indices = np.linspace(0, len(vertical_hist)-1, vertical_bins, dtype=int)
    
    h_features = horizontal_hist[h_indices]
    v_features = vertical_hist[v_indices]
    
    shape_stats = np.array([
        area,
        centroid_x_normalized,
        centroid_y_normalized,
        normalized.mean(),
        normalized.std(),
        normalized.min(),
        normalized.max()
    ])
    
    return np.concatenate([shape_stats, h_features, v_features])


def extract_features(image, target_size=(128, 128)):
    """
    Extract all features from an image for classification.
    
    This is the main feature extraction function that combines:
    - Color histograms (RGB and HSV)
    - Edge features (Sobel)
    - Texture features
    - Shape features
    
    Args:
        image: RGB image as NumPy array (H, W, 3) or bytes
        target_size: Size to resize image for consistent features
        
    Returns:
        np.ndarray: Combined feature vector (fixed length)
    """
    if isinstance(image, bytes) or hasattr(image, 'read'):
        image = load_image_from_bytes(image)
    
    image = resize_image(image, target_size)
    
    grayscale = rgb_to_grayscale(image)
    
    color_hist = compute_color_histogram(image, bins=32)
    
    hsv_hist = compute_hsv_histogram(image, bins=32)
    
    edge_features = compute_edge_features(grayscale, bins=16)
    
    texture_features = compute_texture_features(grayscale)
    
    shape_features = compute_shape_features(grayscale)
    
    all_features = np.concatenate([
        color_hist,
        hsv_hist,
        edge_features,
        texture_features,
        shape_features
    ])
    
    return all_features.astype(np.float32)


def get_feature_dimension():
    """
    Get the dimension of the feature vector.
    
    Returns:
        int: Total number of features
    """
    return 96 + 64 + 37 + 64 + 39


def normalize_features(features, mean=None, std=None):
    """
    Normalize features using z-score normalization.
    
    Args:
        features: Feature array (n_samples, n_features) or (n_features,)
        mean: Pre-computed mean (optional)
        std: Pre-computed std (optional)
        
    Returns:
        tuple: (normalized_features, mean, std)
    """
    if mean is None or std is None:
        if features.ndim == 1:
            mean = features.mean()
            std = features.std() + 1e-7
        else:
            mean = features.mean(axis=0)
            std = features.std(axis=0) + 1e-7
    
    std_safe = std if std is not None else 1.0
    normalized = (features - mean) / (std_safe + 1e-7)
    
    return normalized, mean, std


def batch_extract_features(images, target_size=(128, 128)):
    """
    Extract features from multiple images.
    
    Args:
        images: List of images (NumPy arrays or bytes)
        target_size: Size to resize images
        
    Returns:
        np.ndarray: Feature matrix (n_images, n_features)
    """
    features_list = []
    
    for img in images:
        try:
            features = extract_features(img, target_size)
            features_list.append(features)
        except Exception as e:
            print(f"Error extracting features: {e}")
            continue
    
    if not features_list:
        return None
    
    return np.array(features_list)


SPECIES_MAPPING = {
    0: 'Iris-setosa',
    1: 'Iris-versicolor',
    2: 'Iris-virginica'
}

SPECIES_ID_MAPPING = {v: k for k, v in SPECIES_MAPPING.items()}
