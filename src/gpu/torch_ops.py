import numpy as np
from typing import Tuple, Optional, Union
from ..core.pipeline import Operation

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class TorchOperation(Operation):
    """
    Base class for PyTorch-accelerated operations.
    """
    
    def __init__(self, name: str):
        super().__init__(name)
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Please install PyTorch to use GPU operations.")
        
        # Check for CUDA availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            self.name = f"{name}-cuda"
        else:
            self.name = f"{name}-cpu"
    
    def _process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Process the image using PyTorch."""
        # Convert to PyTorch tensor
        if len(image.shape) == 3:
            # Convert HWC to CHW format for PyTorch
            tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().to(self.device)
        else:
            # Add channel dimension for grayscale
            tensor = torch.from_numpy(image).float().unsqueeze(0).to(self.device)
        
        # Process using PyTorch
        result_tensor = self._process_torch(tensor, **kwargs)
        
        # Convert back to numpy
        if result_tensor.shape[0] == 3:  # RGB
            result = result_tensor.cpu().numpy().transpose(1, 2, 0)
        else:  # Grayscale
            result = result_tensor.squeeze(0).cpu().numpy()
        
        # Ensure uint8 dtype
        if result.dtype != np.uint8:
            result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def _process_torch(self, tensor: "torch.Tensor", **kwargs) -> "torch.Tensor":
        """Process the image using PyTorch. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _process_torch method")

class TorchBlur(TorchOperation):
    """
    PyTorch-accelerated blur filter.
    """
    
    def __init__(self, kernel_size: int = 5, sigma: float = 1.0):
        super().__init__(name=f"TorchBlur-{kernel_size}")
        self.kernel_size = kernel_size
        self.sigma = sigma
        
        # Create Gaussian kernel
        if TORCH_AVAILABLE:
            # Make sure kernel size is odd
            if kernel_size % 2 == 0:
                self.kernel_size = kernel_size + 1
            
            # Create a 2D Gaussian kernel
            kernel = self._create_gaussian_kernel(self.kernel_size, sigma)
            
            # Register kernel as a buffer
            self.register_buffer("kernel", kernel)
    
    def _create_gaussian_kernel(self, size: int, sigma: float) -> "torch.Tensor":
        """Create a 2D Gaussian kernel."""
        # Create 1D Gaussian kernel
        coords = torch.arange(size, dtype=torch.float)
        coords -= size // 2
        
        # Gaussian function
        gauss = torch.exp(-(coords**2) / (2 * sigma**2))
        kernel_1d = gauss / gauss.sum()
        
        # Create 2D kernel by outer product
        kernel_2d = kernel_1d.view(-1, 1) * kernel_1d.view(1, -1)
        
        # Reshape for convolution (1, 1, size, size)
        return kernel_2d.view(1, 1, size, size)
    
    def register_buffer(self, name: str, tensor: "torch.Tensor") -> None:
        """Store a tensor as an attribute."""
        setattr(self, name, tensor)
    
    def _process_torch(self, tensor: "torch.Tensor", **kwargs) -> "torch.Tensor":
        """Apply Gaussian blur using PyTorch."""
        # Normalize tensor to 0-1 range
        tensor = tensor / 255.0
        
        # Get the number of channels
        if len(tensor.shape) == 3:  # Single image
            c, h, w = tensor.shape
        else:  # Batch of images
            _, c, h, w = tensor.shape
        
        # Reshape kernel for the number of channels
        if c == 1:
            # For grayscale, use the kernel as is
            kernel = self.kernel
        else:
            # For RGB, create a kernel per channel that doesn't mix channels
            kernel = torch.zeros(c, 1, self.kernel_size, self.kernel_size, device=self.device)
            for i in range(c):
                kernel[i, 0] = self.kernel[0, 0]
            
            # Reshape for grouped convolution
            kernel = kernel.view(c, 1, self.kernel_size, self.kernel_size)
        
        # Apply convolution
        # For RGB, we use grouped convolution to process each channel separately
        if c == 1:
            result = F.conv2d(tensor.unsqueeze(0), kernel, padding=self.kernel_size//2)
            result = result.squeeze(0)
        else:
            # Prepare tensor: [C, H, W] -> [1, C, H, W]
            tensor = tensor.unsqueeze(0)
            # Use grouped convolution: each channel is convolved with its own kernel
            result = F.conv2d(tensor, kernel, groups=c, padding=self.kernel_size//2)
            # Back to [C, H, W]
            result = result.squeeze(0)
        
        # Scale back to 0-255 range
        return result * 255.0

class TorchColorTransform(TorchOperation):
    """
    PyTorch-accelerated color transformations.
    """
    
    def __init__(self, transformation: str = "grayscale"):
        super().__init__(name=f"TorchColorTransform-{transformation}")
        self.transformation = transformation
        
        # Precompute color transformation matrices
        if TORCH_AVAILABLE and transformation == "sepia":
            self.sepia_matrix = torch.tensor([
                [0.393, 0.769, 0.189],
                [0.349, 0.686, 0.168],
                [0.272, 0.534, 0.131]
            ], device=self.device)
    
    def _process_torch(self, tensor: "torch.Tensor", **kwargs) -> "torch.Tensor":
        """Apply color transformation using PyTorch."""
        # Handle grayscale input
        if tensor.shape[0] == 1:
            if self.transformation != "grayscale":
                # Convert grayscale to RGB for other transformations
                tensor = tensor.repeat(3, 1, 1)
            else:
                # Already grayscale
                return tensor
        
        if self.transformation == "grayscale":
            # Convert RGB to grayscale using perceived luminance
            weights = torch.tensor([0.299, 0.587, 0.114], device=self.device)
            return (tensor.permute(1, 2, 0) @ weights).unsqueeze(0)
        
        elif self.transformation == "invert":
            # Invert colors
            return 255.0 - tensor
        
        elif self.transformation == "sepia":
            # Apply sepia effect
            h, w = tensor.shape[1], tensor.shape[2]
            
            # Reshape to [H*W, 3]
            flat = tensor.permute(1, 2, 0).reshape(-1, 3)
            
            # Apply sepia matrix
            sepia = flat @ self.sepia_matrix.t()
            
            # Clip and reshape back
            sepia = torch.clamp(sepia, 0, 255)
            return sepia.reshape(h, w, 3).permute(2, 0, 1)
        
        else:
            raise ValueError(f"Unknown transformation: {self.transformation}")

class TorchEdgeDetection(TorchOperation):
    """
    PyTorch-accelerated edge detection.
    """
    
    def __init__(self):
        super().__init__(name="TorchEdgeDetection")
        
        # Create Sobel kernels
        if TORCH_AVAILABLE:
            kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                   dtype=torch.float, device=self.device)
            kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                   dtype=torch.float, device=self.device)
            
            # Reshape for convolution (1, 1, 3, 3)
            self.register_buffer("kernel_x", kernel_x.view(1, 1, 3, 3))
            self.register_buffer("kernel_y", kernel_y.view(1, 1, 3, 3))
    
    def _process_torch(self, tensor: "torch.Tensor", **kwargs) -> "torch.Tensor":
        """Apply edge detection using PyTorch."""
        # Convert to grayscale if needed
        if tensor.shape[0] == 3:  # RGB
            weights = torch.tensor([0.299, 0.587, 0.114], device=self.device)
            gray = (tensor.permute(1, 2, 0) @ weights).unsqueeze(0)
        else:  # Already grayscale
            gray = tensor
        
        # Add batch dimension
        gray = gray.unsqueeze(0)
        
        # Apply Sobel filters
        grad_x = F.conv2d(gray, self.kernel_x, padding=1)
        grad_y = F.conv2d(gray, self.kernel_y, padding=1)
        
        # Compute gradient magnitude
        grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        
        # Remove batch dimension and normalize
        result = grad_magnitude.squeeze(0)
        
        # Normalize to 0-255 range if needed
        if torch.max(result) > 0:
            result = 255.0 * (result / torch.max(result))
        
        return result
