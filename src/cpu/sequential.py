import numpy as np
from typing import Tuple, Optional
from ..core.pipeline import Operation

class SequentialBlur(Operation):
    """
    Sequential implementation of a blur filter.
    """
    
    def __init__(self, kernel_size: int = 3):
        super().__init__(name=f"SequentialBlur-{kernel_size}")
        self.kernel_size = kernel_size
        # Create a simple box blur kernel
        self.kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    
    def _process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply blur filter sequentially."""
        # Handle grayscale and color images
        if len(image.shape) == 2:
            return self._convolve2d(image, self.kernel)
        else:
            result = np.zeros_like(image)
            for c in range(image.shape[2]):
                result[:, :, c] = self._convolve2d(image[:, :, c], self.kernel)
            return result
    
    def _convolve2d(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Apply 2D convolution."""
        output = np.zeros_like(image)
        padding = self.kernel_size // 2
        
        # Add zero padding to the image
        padded = np.pad(image, padding, mode='constant')
        
        # Apply convolution
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                output[i, j] = np.sum(
                    padded[i:i+self.kernel_size, j:j+self.kernel_size] * kernel
                )
        
        return output

class SequentialEdgeDetection(Operation):
    """
    Sequential implementation of an edge detection filter.
    """
    
    def __init__(self):
        super().__init__(name="SequentialEdgeDetection")
        # Sobel operators for edge detection
        self.kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        self.kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    def _process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply edge detection filter sequentially."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2).astype(np.float32)
        else:
            gray = image.astype(np.float32)
        
        # Apply Sobel operators
        grad_x = self._convolve2d(gray, self.kernel_x)
        grad_y = self._convolve2d(gray, self.kernel_y)
        
        # Compute gradient magnitude
        gradient = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize to 0-255 range
        gradient = np.clip(gradient, 0, 255).astype(np.uint8)
        
        return gradient
    
    def _convolve2d(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Apply 2D convolution."""
        output = np.zeros_like(image)
        padding = 1  # Sobel kernel is 3x3
        
        # Add zero padding to the image
        padded = np.pad(image, padding, mode='constant')
        
        # Apply convolution
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                output[i, j] = np.sum(
                    padded[i:i+3, j:j+3] * kernel
                )
        
        return output

class SequentialColorTransform(Operation):
    """
    Sequential implementation of color transformations.
    """
    
    def __init__(self, transformation: str = "grayscale"):
        super().__init__(name=f"SequentialColorTransform-{transformation}")
        self.transformation = transformation
    
    def _process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply color transformation sequentially."""
        if len(image.shape) < 3:
            return image  # Already grayscale
        
        if self.transformation == "grayscale":
            return np.mean(image, axis=2).astype(np.uint8)
        
        elif self.transformation == "invert":
            return 255 - image
        
        elif self.transformation == "sepia":
            result = np.zeros_like(image)
            
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    r, g, b = image[i, j]
                    
                    # Sepia transformation
                    tr = int(0.393 * r + 0.769 * g + 0.189 * b)
                    tg = int(0.349 * r + 0.686 * g + 0.168 * b)
                    tb = int(0.272 * r + 0.534 * g + 0.131 * b)
                    
                    result[i, j] = [
                        min(255, tr),
                        min(255, tg),
                        min(255, tb)
                    ]
            
            return result
        
        else:
            raise ValueError(f"Unknown transformation: {self.transformation}")
