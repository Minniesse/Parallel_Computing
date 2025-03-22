from typing import Tuple, Union, Any, Dict, List
import numpy as np
import time
import os
from PIL import Image

def load_image(filepath: str) -> np.ndarray:
    """
    Load an image from a file.
    
    Args:
        filepath: Path to the image file
        
    Returns:
        Numpy array containing the image data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Image file not found: {filepath}")
    
    with Image.open(filepath) as img:
        return np.array(img)

def save_image(image: np.ndarray, filepath: str) -> None:
    """
    Save an image to a file.
    
    Args:
        image: Numpy array containing the image data
        filepath: Path where the image will be saved
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    
    # Convert to uint8 if needed
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    
    img = Image.fromarray(image)
    img.save(filepath)

def create_gaussian_kernel(size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Create a Gaussian kernel.
    
    Args:
        size: Size of the kernel (must be odd)
        sigma: Standard deviation of the Gaussian
        
    Returns:
        2D Gaussian kernel as a numpy array
    """
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd")
    
    # Create a coordinate grid
    k = (size - 1) // 2
    x, y = np.mgrid[-k:k+1, -k:k+1]
    
    # Create the kernel
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    # Normalize
    return kernel / kernel.sum()

def benchmark_function(func, *args, iterations: int = 10, **kwargs) -> Dict[str, float]:
    """
    Benchmark a function's execution time.
    
    Args:
        func: Function to benchmark
        *args: Arguments to pass to the function
        iterations: Number of iterations to run
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Dictionary with timing statistics
    """
    times = []
    
    # Warm-up run
    func(*args, **kwargs)
    
    # Timed runs
    for _ in range(iterations):
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        times.append(end_time - start_time)
    
    return {
        'mean': np.mean(times),
        'median': np.median(times),
        'min': np.min(times),
        'max': np.max(times),
        'std': np.std(times)
    }

def split_image_tiles(image: np.ndarray, tile_size: Tuple[int, int] = (256, 256)) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
    """
    Split an image into tiles.
    
    Args:
        image: Input image as a numpy array
        tile_size: Size of each tile (height, width)
        
    Returns:
        Tuple containing a list of tiles and a list of tile positions (y, x, h, w)
    """
    height, width = image.shape[:2]
    
    tiles = []
    positions = []
    
    for y in range(0, height, tile_size[0]):
        for x in range(0, width, tile_size[1]):
            h = min(tile_size[0], height - y)
            w = min(tile_size[1], width - x)
            
            tile = image[y:y+h, x:x+w].copy()
            tiles.append(tile)
            positions.append((y, x, h, w))
    
    return tiles, positions

def reconstruct_from_tiles(tiles: List[np.ndarray], positions: List[Tuple[int, int, int, int]], output_shape: Tuple[int, ...]) -> np.ndarray:
    """
    Reconstruct an image from tiles.
    
    Args:
        tiles: List of image tiles
        positions: List of tile positions (y, x, h, w)
        output_shape: Shape of the output image
        
    Returns:
        Reconstructed image
    """
    result = np.zeros(output_shape, dtype=tiles[0].dtype)
    
    for tile, (y, x, h, w) in zip(tiles, positions):
        result[y:y+h, x:x+w] = tile
    
    return result
