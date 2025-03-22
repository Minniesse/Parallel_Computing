import numpy as np
import numba
from typing import Tuple, Optional
from ..core.pipeline import Operation
from ..core.utils import create_gaussian_kernel

class VectorizedBlur(Operation):
    """
    Vectorized implementation of a blur filter using NumPy.
    """
    
    def __init__(self, kernel_size: int = 3):
        super().__init__(name=f"VectorizedBlur-{kernel_size}")
        self.kernel_size = kernel_size
        # Create a simple box blur kernel
        self.kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    
    def _process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply blur filter using vectorized operations."""
        # Handle grayscale and color images
        if len(image.shape) == 2:
            return self._convolve2d_vectorized(image, self.kernel)
        else:
            result = np.zeros_like(image)
            for c in range(image.shape[2]):
                result[:, :, c] = self._convolve2d_vectorized(image[:, :, c], self.kernel)
            return result
    
    def _convolve2d_vectorized(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Apply 2D convolution using vectorized operations."""
        from scipy.signal import convolve2d
        return convolve2d(image, kernel, mode='same', boundary='symm')

class NumbaBlur(Operation):
    """
    Blur filter implementation using Numba JIT compilation.
    """
    
    def __init__(self, kernel_size: int = 3, sigma: float = 1.0):
        super().__init__(name=f"NumbaBlur-{kernel_size}")
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.kernel = create_gaussian_kernel(kernel_size, sigma)
    
    def _process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply blur filter using Numba acceleration."""
        # Handle grayscale and color images
        if len(image.shape) == 2:
            return self._gaussian_blur_numba(image, self.kernel)
        else:
            result = np.zeros_like(image)
            for c in range(image.shape[2]):
                result[:, :, c] = self._gaussian_blur_numba(image[:, :, c], self.kernel)
            return result
    
    def _gaussian_blur_numba(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur using Numba."""
        return _numba_gaussian_blur(image, kernel)

@numba.jit(nopython=True, parallel=True)
def _numba_gaussian_blur(image, kernel):
    """
    Optimized Gaussian blur implementation using Numba.
    """
    height, width = image.shape
    kernel_height, kernel_width = kernel.shape
    padding_h = kernel_height // 2
    padding_w = kernel_width // 2
    
    output = np.zeros_like(image)
    
    for i in numba.prange(padding_h, height - padding_h):
        for j in range(padding_w, width - padding_w):
            val = 0.0
            for ki in range(kernel_height):
                for kj in range(kernel_width):
                    val += image[i + ki - padding_h, j + kj - padding_w] * kernel[ki, kj]
            output[i, j] = val
    
    return output

class VectorizedEdgeDetection(Operation):
    """
    Vectorized implementation of edge detection using NumPy.
    """
    
    def __init__(self):
        super().__init__(name="VectorizedEdgeDetection")
        # Sobel operators for edge detection
        self.kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        self.kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    def _process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply edge detection filter using vectorized operations."""
        from scipy.signal import convolve2d
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2).astype(np.float32)
        else:
            gray = image.astype(np.float32)
        
        # Apply Sobel operators using vectorized convolution
        grad_x = convolve2d(gray, self.kernel_x, mode='same', boundary='symm')
        grad_y = convolve2d(gray, self.kernel_y, mode='same', boundary='symm')
        
        # Compute gradient magnitude
        gradient = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize to 0-255 range
        gradient = np.clip(gradient, 0, 255).astype(np.uint8)
        
        return gradient

class NumbaEdgeDetection(Operation):
    """
    Edge detection implementation using Numba JIT compilation.
    """
    
    def __init__(self):
        super().__init__(name="NumbaEdgeDetection")
        # Sobel operators for edge detection
        self.kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        self.kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    def _process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply edge detection filter using Numba acceleration."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2).astype(np.float32)
        else:
            gray = image.astype(np.float32)
        
        # Apply Sobel operators using Numba
        grad_x, grad_y = _numba_apply_sobel(gray, self.kernel_x, self.kernel_y)
        
        # Compute gradient magnitude
        gradient = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize to 0-255 range
        gradient = np.clip(gradient, 0, 255).astype(np.uint8)
        
        return gradient

@numba.jit(nopython=True, parallel=True)
def _numba_apply_sobel(image, kernel_x, kernel_y):
    """Apply Sobel operators using Numba."""
    height, width = image.shape
    grad_x = np.zeros_like(image)
    grad_y = np.zeros_like(image)
    
    # Apply Sobel operators
    for i in numba.prange(1, height-1):
        for j in range(1, width-1):
            # Apply horizontal kernel
            sumx = 0.0
            for ki in range(3):
                for kj in range(3):
                    sumx += image[i+ki-1, j+kj-1] * kernel_x[ki, kj]
            grad_x[i, j] = sumx
            
            # Apply vertical kernel
            sumy = 0.0
            for ki in range(3):
                for kj in range(3):
                    sumy += image[i+ki-1, j+kj-1] * kernel_y[ki, kj]
            grad_y[i, j] = sumy
    
    return grad_x, grad_y

class VectorizedColorTransform(Operation):
    """
    Vectorized implementation of color transformations using NumPy.
    """
    
    def __init__(self, transformation: str = "grayscale"):
        super().__init__(name=f"VectorizedColorTransform-{transformation}")
        self.transformation = transformation
    
    def _process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply color transformation using vectorized operations."""
        if len(image.shape) < 3:
            return image  # Already grayscale
        
        if self.transformation == "grayscale":
            # Vectorized grayscale conversion
            return np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        
        elif self.transformation == "invert":
            # Vectorized inversion
            return 255 - image
        
        elif self.transformation == "sepia":
            # Vectorized sepia transformation
            sepia_matrix = np.array([
                [0.393, 0.769, 0.189],
                [0.349, 0.686, 0.168],
                [0.272, 0.534, 0.131]
            ])
            
            # Reshape image for matrix multiplication
            reshaped = image.reshape(-1, 3)
            transformed = np.dot(reshaped, sepia_matrix.T)
            transformed = np.clip(transformed, 0, 255).astype(np.uint8)
            
            # Reshape back to original image shape
            return transformed.reshape(image.shape)
        
        else:
            raise ValueError(f"Unknown transformation: {self.transformation}")
