import numpy as np
from typing import Tuple, Optional, Union, Callable, List
from ..core.pipeline import Operation

class Resize(Operation):
    """
    Resize operation with configurable backend.
    """
    
    def __init__(self, target_size: Tuple[int, int], backend: str = "auto"):
        """
        Initialize the resize operation.
        
        Args:
            target_size: Target size (height, width)
            backend: One of "auto", "sequential", "vectorized", "multicore", "gpu"
        """
        super().__init__(name=f"Resize-{backend}-{target_size[0]}x{target_size[1]}")
        self.target_size = target_size
        self.backend = backend
        
        # Lazy initialization of implementation
        self._implementation = None
    
    def _process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Resize image with the selected backend."""
        # Lazy initialization of implementation
        if self._implementation is None:
            self._create_implementation()
        
        return self._implementation(image, **kwargs)
    
    def _create_implementation(self) -> None:
        """Create the appropriate implementation based on the backend."""
        backend = self.backend
        
        # Auto-select backend based on available hardware
        if backend == "auto":
            try:
                import cupy
                backend = "gpu"
            except ImportError:
                backend = "vectorized"
        
        # Create the implementation
        if backend == "sequential" or backend == "vectorized":
            from PIL import Image
            
            def resize_pil(img, **kwargs):
                # Convert to PIL Image
                if len(img.shape) == 3:
                    pil_img = Image.fromarray(img)
                else:
                    pil_img = Image.fromarray(img, mode='L')
                
                # Resize
                resized = pil_img.resize((self.target_size[1], self.target_size[0]), 
                                          resample=Image.BILINEAR)
                
                # Convert back to numpy
                return np.array(resized)
            
            self._implementation = resize_pil
            
        elif backend == "multicore":
            from skimage.transform import resize as sk_resize
            
            def resize_multicore(img, **kwargs):
                # Use scikit-image's resize with multiple threads
                return sk_resize(img, self.target_size, 
                                anti_aliasing=True, preserve_range=True).astype(img.dtype)
            
            self._implementation = resize_multicore
            
        elif backend == "gpu":
            try:
                import cupy as cp
                import cupyx.scipy.ndimage as cuimg
                
                def resize_gpu(img, **kwargs):
                    # Transfer to GPU
                    gpu_img = cp.asarray(img)
                    
                    # Calculate scaling factors
                    scale_y = self.target_size[0] / img.shape[0]
                    scale_x = self.target_size[1] / img.shape[1]
                    
                    # Use cupy zoom
                    if len(img.shape) == 3:
                        scales = (scale_y, scale_x, 1)
                    else:
                        scales = (scale_y, scale_x)
                    
                    resized = cuimg.zoom(gpu_img, scales, order=1)
                    
                    # Transfer back to CPU
                    return cp.asnumpy(resized).astype(img.dtype)
                
                self._implementation = resize_gpu
                
            except ImportError:
                # Fall back to PIL if CuPy isn't available
                from PIL import Image
                
                def resize_pil(img, **kwargs):
                    # Convert to PIL Image
                    if len(img.shape) == 3:
                        pil_img = Image.fromarray(img)
                    else:
                        pil_img = Image.fromarray(img, mode='L')
                    
                    # Resize
                    resized = pil_img.resize((self.target_size[1], self.target_size[0]), 
                                              resample=Image.BILINEAR)
                    
                    # Convert back to numpy
                    return np.array(resized)
                
                self._implementation = resize_pil
                
        else:
            raise ValueError(f"Unknown backend: {backend}")

class Rotate(Operation):
    """
    Rotate operation with configurable backend.
    """
    
    def __init__(self, angle: float, backend: str = "auto"):
        """
        Initialize the rotate operation.
        
        Args:
            angle: Rotation angle in degrees (counterclockwise)
            backend: One of "auto", "sequential", "vectorized", "multicore", "gpu"
        """
        super().__init__(name=f"Rotate-{backend}-{angle}deg")
        self.angle = angle
        self.backend = backend
        
        # Lazy initialization of implementation
        self._implementation = None
    
    def _process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Rotate image with the selected backend."""
        # Lazy initialization of implementation
        if self._implementation is None:
            self._create_implementation()
        
        return self._implementation(image, **kwargs)
    
    def _create_implementation(self) -> None:
        """Create the appropriate implementation based on the backend."""
        backend = self.backend
        
        # Auto-select backend based on available hardware
        if backend == "auto":
            try:
                import cupy
                backend = "gpu"
            except ImportError:
                backend = "vectorized"
        
        # Create the implementation
        if backend == "sequential" or backend == "vectorized":
            from scipy.ndimage import rotate as scipy_rotate
            
            def rotate_scipy(img, **kwargs):
                return scipy_rotate(img, self.angle, reshape=True, 
                                   order=1, mode='constant', cval=0).astype(img.dtype)
            
            self._implementation = rotate_scipy
            
        elif backend == "multicore":
            from skimage.transform import rotate as sk_rotate
            
            def rotate_multicore(img, **kwargs):
                return sk_rotate(img, self.angle, resize=True, 
                                preserve_range=True).astype(img.dtype)
            
            self._implementation = rotate_multicore
            
        elif backend == "gpu":
            try:
                import cupy as cp
                import cupyx.scipy.ndimage as cuimg
                
                def rotate_gpu(img, **kwargs):
                    # Transfer to GPU
                    gpu_img = cp.asarray(img)
                    
                    # Rotate on GPU
                    rotated = cuimg.rotate(gpu_img, self.angle, reshape=True, 
                                          order=1, mode='constant', cval=0)
                    
                    # Transfer back to CPU
                    return cp.asnumpy(rotated).astype(img.dtype)
                
                self._implementation = rotate_gpu
                
            except ImportError:
                # Fall back to scipy if CuPy isn't available
                from scipy.ndimage import rotate as scipy_rotate
                
                def rotate_scipy(img, **kwargs):
                    return scipy_rotate(img, self.angle, reshape=True, 
                                       order=1, mode='constant', cval=0).astype(img.dtype)
                
                self._implementation = rotate_scipy
                
        else:
            raise ValueError(f"Unknown backend: {backend}")

class Crop(Operation):
    """
    Crop operation with configurable backend.
    """
    
    def __init__(self, crop_box: Tuple[int, int, int, int], backend: str = "auto"):
        """
        Initialize the crop operation.
        
        Args:
            crop_box: Crop region as (y_start, x_start, height, width)
            backend: One of "auto", "sequential", "vectorized", "multicore", "gpu"
        """
        super().__init__(name=f"Crop-{backend}")
        self.crop_box = crop_box
        self.backend = backend
        
        # Lazy initialization of implementation
        self._implementation = None
    
    def _process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Crop image with the selected backend."""
        # Lazy initialization of implementation
        if self._implementation is None:
            self._create_implementation()
        
        return self._implementation(image, **kwargs)
    
    def _create_implementation(self) -> None:
        """Create the appropriate implementation based on the backend."""
        backend = self.backend
        
        # Auto-select backend based on available hardware
        if backend == "auto":
            backend = "vectorized"  # Cropping is so simple that vectorized NumPy is best
        
        # Create the implementation - for all backends, cropping is just array slicing
        y_start, x_start, height, width = self.crop_box
        
        def crop_array(img, **kwargs):
            if (y_start >= 0 and x_start >= 0 and 
                y_start + height <= img.shape[0] and 
                x_start + width <= img.shape[1]):
                return img[y_start:y_start+height, x_start:x_start+width].copy()
            else:
                raise ValueError(f"Crop box {self.crop_box} is outside image boundaries {img.shape[:2]}")
        
        self._implementation = crop_array
