import numpy as np
from typing import Tuple, Optional, Union, Dict, Any
from ..core.pipeline import Operation

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

class CustomKernelOperation(Operation):
    """
    Base class for operations using custom CUDA kernels.
    """
    
    def __init__(self, name: str):
        super().__init__(name)
        if not CUPY_AVAILABLE:
            raise ImportError("CuPy is not available. Please install CuPy to use GPU operations.")
    
    @staticmethod
    @cp.RawKernel()
    def _gaussian_blur_kernel():
        return r"""
        extern "C" __global__
        void gaussian_blur_kernel(const float* input, float* output, int width, int height, const float* kernel, int kernel_size) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            int half_kernel = kernel_size / 2;
            
            if (x < width && y < height) {
                float sum = 0.0;
                for (int ky = -half_kernel; ky <= half_kernel; ++ky) {
                    for (int kx = -half_kernel; kx <= half_kernel; ++kx) {
                        int ix = min(max(x + kx, 0), width - 1);
                        int iy = min(max(y + ky, 0), height - 1);
                        sum += input[iy * width + ix] * kernel[(ky + half_kernel) * kernel_size + (kx + half_kernel)];
                    }
                }
                output[y * width + x] = sum;
            }
        }
        """
    
    @staticmethod
    @cp.RawKernel()
    def _sobel_edge_kernel():
        return r"""
        extern "C" __global__
        void sobel_edge_kernel(const float* input, float* output, int width, int height) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            
            if (x < width && y < height) {
                float Gx = 0.0;
                float Gy = 0.0;
                
                if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
                    Gx = -input[(y - 1) * width + (x - 1)] - 2.0 * input[y * width + (x - 1)] - input[(y + 1) * width + (x - 1)]
                         + input[(y - 1) * width + (x + 1)] + 2.0 * input[y * width + (x + 1)] + input[(y + 1) * width + (x + 1)];
                    
                    Gy = -input[(y - 1) * width + (x - 1)] - 2.0 * input[(y - 1) * width + x] - input[(y - 1) * width + (x + 1)]
                         + input[(y + 1) * width + (x - 1)] + 2.0 * input[(y + 1) * width + x] + input[(y + 1) * width + (x + 1)];
                }
                
                output[y * width + x] = sqrt(Gx * Gx + Gy * Gy);
            }
        }
        """
    
    @staticmethod
    @cp.RawKernel()
    def _sepia_kernel():
        return r"""
        extern "C" __global__
        void sepia_kernel(const uchar4* input, uchar4* output, int width, int height) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            
            if (x < width && y < height) {
                uchar4 pixel = input[y * width + x];
                float r = pixel.x;
                float g = pixel.y;
                float b = pixel.z;
                
                float tr = 0.393 * r + 0.769 * g + 0.189 * b;
                float tg = 0.349 * r + 0.686 * g + 0.168 * b;
                float tb = 0.272 * r + 0.534 * g + 0.131 * b;
                
                output[y * width + x] = make_uchar4(min(tr, 255.0), min(tg, 255.0), min(tb, 255.0), pixel.w);
            }
        }
        """

class CustomBlur(CustomKernelOperation):
    """
    Gaussian blur using a custom CUDA kernel.
    """
    
    def __init__(self, kernel_size: int = 5, sigma: float = 1.0):
        super().__init__(name=f"CustomCUDABlur-{kernel_size}")
        self.kernel_size = kernel_size
        self.sigma = sigma
        
        # Create Gaussian kernel
        kernel = self._create_gaussian_kernel()
        self.gpu_kernel = cp.asarray(kernel)
    
    def _create_gaussian_kernel(self) -> np.ndarray:
        """Create a Gaussian kernel."""
        from ..core.utils import create_gaussian_kernel
        return create_gaussian_kernel(self.kernel_size, self.sigma)
    
    def _process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Process the image using a custom CUDA kernel."""
        # Transfer image to GPU
        gpu_image = cp.asarray(image.astype(np.float32))
        
        # Handle grayscale and color images
        if len(image.shape) == 2:
            # Process grayscale image
            result = self._apply_blur_kernel(gpu_image)
        else:
            # Process each color channel separately
            result = cp.zeros_like(gpu_image)
            for c in range(image.shape[2]):
                result[:, :, c] = self._apply_blur_kernel(gpu_image[:, :, c])
        
        # Transfer result back to CPU
        return cp.asnumpy(result).astype(np.uint8)
    
    def _apply_blur_kernel(self, channel: "cp.ndarray") -> "cp.ndarray":
        """Apply the blur kernel to a single channel."""
        # Prepare output array
        height, width = channel.shape
        output = cp.zeros_like(channel)
        
        # Prepare kernel for GPU
        kernel_flat = self.gpu_kernel.ravel()
        
        # Set up grid and block dimensions
        block_size = (16, 16)
        grid_size = (
            (width + block_size[0] - 1) // block_size[0],
            (height + block_size[1] - 1) // block_size[1]
        )
        
        # Execute the CUDA kernel
        CustomKernelOperation._gaussian_blur_kernel(
            grid_size, 
            block_size, 
            (
                channel, 
                output, 
                np.int32(width), 
                np.int32(height),
                kernel_flat,
                np.int32(self.kernel_size)
            )
        )
        
        return output

class CustomEdgeDetection(CustomKernelOperation):
    """
    Edge detection using a custom CUDA kernel.
    """
    
    def __init__(self):
        super().__init__(name="CustomCUDAEdgeDetection")
    
    def _process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Process the image using a custom CUDA kernel."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2).astype(np.float32)
        else:
            gray = image.astype(np.float32)
        
        # Transfer to GPU
        gpu_image = cp.asarray(gray)
        
        # Prepare output array
        height, width = gpu_image.shape
        output = cp.zeros_like(gpu_image)
        
        # Set up grid and block dimensions
        block_size = (16, 16)
        grid_size = (
            (width + block_size[0] - 1) // block_size[0],
            (height + block_size[1] - 1) // block_size[1]
        )
        
        # Execute the CUDA kernel
        CustomKernelOperation._sobel_edge_kernel(
            grid_size, 
            block_size, 
            (
                gpu_image, 
                output, 
                np.int32(width), 
                np.int32(height)
            )
        )
        
        # Normalize the output
        if cp.max(output) > 0:
            output = output * (255.0 / cp.max(output))
        
        # Transfer result back to CPU
        return cp.asnumpy(output).astype(np.uint8)

class CustomSepiaTransform(CustomKernelOperation):
    """
    Sepia tone transformation using a custom CUDA kernel.
    """
    
    def __init__(self):
        super().__init__(name="CustomCUDASepiaTransform")
    
    def _process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Process the image using a custom CUDA kernel."""
        # Only works with RGB images
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Sepia transformation requires an RGB image")
        
        # Transfer to GPU
        gpu_image = cp.asarray(image)
        
        # Prepare output array
        height, width = image.shape[:2]
        output = cp.zeros_like(gpu_image)
        
        # Reshape to match the kernel's expected format (uchar4)
        input_view = gpu_image.view(dtype=cp.dtype((cp.uint8, 4)))
        output_view = output.view(dtype=cp.dtype((cp.uint8, 4)))
        
        # Ensure contiguous arrays
        input_view = cp.ascontiguousarray(input_view)
        output_view = cp.ascontiguousarray(output_view)
        
        # Set up grid and block dimensions
        block_size = (16, 16)
        grid_size = (
            (width + block_size[0] - 1) // block_size[0],
            (height + block_size[1] - 1) // block_size[1]
        )
        
        # Execute the CUDA kernel
        CustomKernelOperation._sepia_kernel(
            grid_size, 
            block_size, 
            (
                input_view, 
                output_view, 
                np.int32(width), 
                np.int32(height)
            )
        )
        
        # Transfer result back to CPU
        return cp.asnumpy(output).astype(np.uint8)