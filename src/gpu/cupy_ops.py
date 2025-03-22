import numpy as np
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

from typing import Tuple, Optional
from ..core.pipeline import Operation
from ..core.utils import create_gaussian_kernel

class GPUOperation(Operation):
    """
    Base class for GPU-accelerated operations.
    """
    
    def __init__(self, name: str):
        super().__init__(name)
        if not CUPY_AVAILABLE:
            raise ImportError("CuPy is not available. Please install CuPy to use GPU operations.")
    
    def _process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Process the image using GPU acceleration."""
        # Transfer data to GPU
        gpu_image = cp.asarray(image)
        
        # Process on GPU
        gpu_result = self._process_gpu(gpu_image, **kwargs)
        
        # Transfer result back to CPU
        return cp.asnumpy(gpu_result)
    
    def _process_gpu(self, gpu_image: "cp.ndarray", **kwargs) -> "cp.ndarray":
        """Process the image on GPU. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _process_gpu method")

class GPUBlur(GPUOperation):
    """
    GPU-accelerated blur filter.
    """
    
    def __init__(self, kernel_size: int = 3, sigma: float = 1.0):
        super().__init__(name=f"GPUBlur-{kernel_size}")
        self.kernel_size = kernel_size
        self.sigma = sigma
        # Create the kernel on CPU
        cpu_kernel = create_gaussian_kernel(kernel_size, sigma)
        # Transfer to GPU
        self.gpu_kernel = cp.asarray(cpu_kernel) if CUPY_AVAILABLE else None
    
    def _process_gpu(self, gpu_image: "cp.ndarray", **kwargs) -> "cp.ndarray":
        """Apply blur filter on GPU."""
        # Handle grayscale and color images
        if len(gpu_image.shape) == 2:
            return self._convolve2d_gpu(gpu_image, self.gpu_kernel)
        else:
            result = cp.zeros_like(gpu_image)
            for c in range(gpu_image.shape[2]):
                result[:, :, c] = self._convolve2d_gpu(gpu_image[:, :, c], self.gpu_kernel)
            return result
    
    def _convolve2d_gpu(self, gpu_image: "cp.ndarray", gpu_kernel: "cp.ndarray") -> "cp.ndarray":
        """Apply 2D convolution on GPU."""
        return cp.array(cp.fft.ifft2(cp.fft.fft2(gpu_image) * cp.fft.fft2(gpu_kernel, gpu_image.shape)).real)

class GPUEdgeDetection(GPUOperation):
    """
    GPU-accelerated edge detection.
    """
    
    def __init__(self):
        super().__init__(name="GPUEdgeDetection")
        # Sobel operators for edge detection
        self.gpu_kernel_x = cp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) if CUPY_AVAILABLE else None
        self.gpu_kernel_y = cp.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) if CUPY_AVAILABLE else None
    
    def _process_gpu(self, gpu_image: "cp.ndarray", **kwargs) -> "cp.ndarray":
        """Apply edge detection on GPU."""
        # Convert to grayscale if needed
        if len(gpu_image.shape) == 3:
            gray = cp.mean(gpu_image, axis=2).astype(cp.float32)
        else:
            gray = gpu_image.astype(cp.float32)
        
        # Compute gradients
        grad_x = cp.pad(cp.correlate(gray, self.gpu_kernel_x, mode='valid'), 1)
        grad_y = cp.pad(cp.correlate(gray, self.gpu_kernel_y, mode='valid'), 1)
        
        # Compute gradient magnitude
        gradient = cp.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize to 0-255 range
        return cp.clip(gradient, 0, 255).astype(cp.uint8)

class GPUColorTransform(GPUOperation):
    """
    GPU-accelerated color transformations.
    """
    
    def __init__(self, transformation: str = "grayscale"):
        super().__init__(name=f"GPUColorTransform-{transformation}")
        self.transformation = transformation
        
        # Precompute sepia matrix
        if CUPY_AVAILABLE and transformation == "sepia":
            self.sepia_matrix = cp.array([
                [0.393, 0.769, 0.189],
                [0.349, 0.686, 0.168],
                [0.272, 0.534, 0.131]
            ])
    
    def _process_gpu(self, gpu_image: "cp.ndarray", **kwargs) -> "cp.ndarray":
        """Apply color transformation on GPU."""
        if len(gpu_image.shape) < 3:
            return gpu_image  # Already grayscale
        
        if self.transformation == "grayscale":
            # Vectorized grayscale conversion on GPU
            return cp.dot(gpu_image[..., :3], cp.array([0.299, 0.587, 0.114])).astype(cp.uint8)
        
        elif self.transformation == "invert":
            # Vectorized inversion on GPU
            return 255 - gpu_image
        
        elif self.transformation == "sepia":
            # Vectorized sepia transformation on GPU
            reshaped = gpu_image.reshape(-1, 3)
            transformed = cp.dot(reshaped, self.sepia_matrix.T)
            transformed = cp.clip(transformed, 0, 255).astype(cp.uint8)
            
            # Reshape back to original image shape
            return transformed.reshape(gpu_image.shape)
        
        else:
            raise ValueError(f"Unknown transformation: {self.transformation}")
