import numpy as np
from typing import Tuple, Optional, Union
from ..core.pipeline import Operation

class ColorTransformation(Operation):
    """
    Color transformation with configurable backend.
    Automatically selects the appropriate implementation based on the backend parameter.
    """
    
    def __init__(self, transformation: str = "grayscale", 
                 backend: str = "auto", tile_size: Tuple[int, int] = (256, 256)):
        """
        Initialize the color transformation.
        
        Args:
            transformation: One of "grayscale", "invert", "sepia"
            backend: One of "auto", "sequential", "vectorized", "multicore", "gpu", "torch"
            tile_size: Size of tiles for parallel processing
        """
        super().__init__(name=f"ColorTransformation-{backend}-{transformation}")
        self.transformation = transformation
        self.backend = backend
        self.tile_size = tile_size
        
        # Create actual implementation based on backend
        self._implementation = None
    
    def _process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply color transformation with the selected backend."""
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
                    import torch
                    if torch.cuda.is_available():
                        backend = "torch"
                    else:
                        backend = "vectorized"
                except ImportError:
                    backend = "vectorized"
        
        # Create the implementation
        if backend == "sequential":
            from ..cpu.sequential import SequentialColorTransform
            self._implementation = SequentialColorTransform(self.transformation)
        
        elif backend == "vectorized":
            from ..cpu.vectorized import VectorizedColorTransform
            self._implementation = VectorizedColorTransform(self.transformation)
        
        elif backend == "multicore":
            from ..cpu.multicore import MultiCoreColorTransform
            self._implementation = MultiCoreColorTransform(self.transformation, self.tile_size)
        
        elif backend == "gpu":
            try:
                from ..gpu.cupy_ops import GPUColorTransform
                self._implementation = GPUColorTransform(self.transformation)
            except ImportError:
                from ..cpu.vectorized import VectorizedColorTransform
                self._implementation = VectorizedColorTransform(self.transformation)
        
        elif backend == "torch":
            try:
                from ..gpu.torch_ops import TorchColorTransform
                self._implementation = TorchColorTransform(self.transformation)
            except ImportError:
                from ..cpu.vectorized import VectorizedColorTransform
                self._implementation = VectorizedColorTransform(self.transformation)
        
        elif backend == "custom_cuda":
            try:
                if self.transformation == "sepia":
                    from ..gpu.custom_kernels import CustomSepiaTransform
                    self._implementation = CustomSepiaTransform()
                else:
                    # Fall back to regular GPU implementation for other transformations
                    from ..gpu.cupy_ops import GPUColorTransform
                    self._implementation = GPUColorTransform(self.transformation)
            except ImportError:
                from ..cpu.vectorized import VectorizedColorTransform
                self._implementation = VectorizedColorTransform(self.transformation)
        
        else:
            raise ValueError(f"Unknown backend: {backend}")
