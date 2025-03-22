import numpy as np
from typing import Tuple, Optional, Union
from ..core.pipeline import Operation

class GaussianBlur(Operation):
    """
    Gaussian blur filter with configurable backend.
    Automatically selects the appropriate implementation based on the backend parameter.
    """
    
    def __init__(self, kernel_size: int = 5, sigma: float = 1.0, 
                 backend: str = "auto", tile_size: Tuple[int, int] = (256, 256)):
        """
        Initialize the Gaussian blur filter.
        
        Args:
            kernel_size: Size of the Gaussian kernel (must be odd)
            sigma: Standard deviation of the Gaussian
            backend: One of "auto", "sequential", "vectorized", "numba", "multicore", "gpu"
            tile_size: Size of tiles for parallel processing
        """
        super().__init__(name=f"GaussianBlur-{backend}-{kernel_size}")
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.backend = backend
        self.tile_size = tile_size
        
        # Create actual implementation based on backend
        self._implementation = None
    
    def _process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply Gaussian blur with the selected backend."""
        # Lazy initialization of implementation
        if self._implementation is None:
            self._create_implementation()
        
        return self._implementation._process(image, **kwargs)
    
    def _create_implementation(self) -> None:
        """Create the appropriate implementation based on the backend."""
        backend = self.backend
        
        # Auto-select backend based on available hardware
        if backend == "auto":
            try:
                import cupy
                backend = "gpu"
            except ImportError:
                try:
                    import numba
                    backend = "numba"
                except ImportError:
                    backend = "vectorized"
        
        # Create the implementation
        if backend == "sequential":
            from ..cpu.sequential import SequentialBlur
            self._implementation = SequentialBlur(self.kernel_size)
        
        elif backend == "vectorized":
            from ..cpu.vectorized import VectorizedBlur
            self._implementation = VectorizedBlur(self.kernel_size)
        
        elif backend == "numba":
            from ..cpu.vectorized import NumbaBlur
            self._implementation = NumbaBlur(self.kernel_size, self.sigma)
        
        elif backend == "multicore":
            from ..cpu.multicore import MultiCoreBlur
            self._implementation = MultiCoreBlur(self.kernel_size, self.tile_size)
        
        elif backend == "gpu":
            from ..gpu.cupy_ops import GPUBlur
            self._implementation = GPUBlur(self.kernel_size, self.sigma)
        
        else:
            raise ValueError(f"Unknown backend: {backend}")

class EdgeDetection(Operation):
    """
    Edge detection filter with configurable backend.
    Automatically selects the appropriate implementation based on the backend parameter.
    """
    
    def __init__(self, backend: str = "auto", tile_size: Tuple[int, int] = (256, 256)):
        """
        Initialize the edge detection filter.
        
        Args:
            backend: One of "auto", "sequential", "vectorized", "numba", "multicore", "gpu"
            tile_size: Size of tiles for parallel processing
        """
        super().__init__(name=f"EdgeDetection-{backend}")
        self.backend = backend
        self.tile_size = tile_size
        
        # Create actual implementation based on backend
        self._implementation = None
    
    def _process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply edge detection with the selected backend."""
        # Lazy initialization of implementation
        if self._implementation is None:
            self._create_implementation()
        
        return self._implementation._process(image, **kwargs)
    
    def _create_implementation(self) -> None:
        """Create the appropriate implementation based on the backend."""
        backend = self.backend
        
        # Auto-select backend based on available hardware
        if backend == "auto":
            try:
                import cupy
                backend = "gpu"
            except ImportError:
                try:
                    import numba
                    backend = "numba"
                except ImportError:
                    backend = "vectorized"
        
        # Create the implementation
        if backend == "sequential":
            from ..cpu.sequential import SequentialEdgeDetection
            self._implementation = SequentialEdgeDetection()
        
        elif backend == "vectorized":
            from ..cpu.vectorized import VectorizedEdgeDetection
            self._implementation = VectorizedEdgeDetection()
        
        elif backend == "numba":
            from ..cpu.vectorized import NumbaEdgeDetection
            self._implementation = NumbaEdgeDetection()
        
        elif backend == "multicore":
            from ..cpu.multicore import MultiCoreEdgeDetection
            self._implementation = MultiCoreEdgeDetection(self.tile_size)
        
        elif backend == "gpu":
            from ..gpu.cupy_ops import GPUEdgeDetection
            self._implementation = GPUEdgeDetection()
        
        else:
            raise ValueError(f"Unknown backend: {backend}")
