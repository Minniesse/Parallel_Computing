import numpy as np
from typing import Tuple, List, Optional
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from ..core.pipeline import Operation
from ..core.utils import split_image_tiles, reconstruct_from_tiles

# Define a helper function at module level for multiprocessing to pickle properly
def _process_tile(args):
    """Process a single image tile with the given function."""
    tile, tile_info, process_func, *process_args = args
    processed = process_func(tile, *process_args)
    return (processed, tile_info)

class TiledOperation(Operation):
    """
    Base class for operations that use tile-based parallelism.
    """
    
    def __init__(self, name: str, tile_size: Tuple[int, int] = (256, 256), 
                 max_workers: Optional[int] = None):
        super().__init__(name=name)
        # Adjust tile size - larger tiles reduce overhead but might reduce parallelism
        self.tile_size = tile_size
        
        # Calculate optimal number of workers
        total_cpus = multiprocessing.cpu_count()
        if max_workers is None:
            # Use 75% of available cores by default to avoid system lockup
            self.max_workers = max(1, int(total_cpus * 0.75))
        else:
            self.max_workers = min(max_workers, total_cpus)
    
    def _process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Process the image using tile-based parallelism."""
        # Adaptive tile sizing based on image dimensions
        h, w = image.shape[:2]
        tile_h, tile_w = self.tile_size
        
        # For small images, use fewer larger tiles to reduce overhead
        if h * w < 1000000:  # Less than 1 megapixel
            tile_h = max(tile_h, h // 2)
            tile_w = max(tile_w, w // 2)
        
        # Split the image into tiles
        tiles, positions = split_image_tiles(image, (tile_h, tile_w))
        
        # Process tiles in parallel
        process_func = self._process_tile
        process_args = self._get_process_args(**kwargs)
        
        # Create arguments for the helper function
        process_args_list = [(tile, tile_info, process_func, *process_args) 
                            for tile, tile_info in zip(tiles, positions)]
        
        # Calculate optimal chunk size for better load balancing
        chunk_size = max(1, len(process_args_list) // (self.max_workers * 2))
        
        # Process tiles using process pool
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            processed_tiles = list(executor.map(
                _process_tile, 
                process_args_list,
                chunksize=chunk_size
            ))
        
        # Reconstruct the image
        processed_tiles, positions = zip(*processed_tiles)
        return reconstruct_from_tiles(processed_tiles, positions, image.shape)
    
    def _process_tile(self, tile: np.ndarray, **kwargs) -> np.ndarray:
        """Process a single tile. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _process_tile method")
    
    def _get_process_args(self, **kwargs):
        """Get arguments for _process_tile. To be implemented by subclasses."""
        return ()

class MultiCoreBlur(TiledOperation):
    """
    Multi-core implementation of a blur filter.
    """
    
    def __init__(self, kernel_size: int = 3, tile_size: Tuple[int, int] = (256, 256),
                 max_workers: Optional[int] = None):
        super().__init__(name=f"MultiCoreBlur-{kernel_size}", 
                         tile_size=tile_size, max_workers=max_workers)
        self.kernel_size = kernel_size
        # Create a simple box blur kernel
        self.kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    
    def _process_tile(self, tile: np.ndarray, **kwargs) -> np.ndarray:
        """Process a single tile with blur filter."""
        from scipy.signal import convolve2d
        
        # Handle grayscale and color images
        if len(tile.shape) == 2:
            return convolve2d(tile, self.kernel, mode='same', boundary='symm')
        else:
            result = np.zeros_like(tile)
            for c in range(tile.shape[2]):
                result[:, :, c] = convolve2d(tile[:, :, c], self.kernel, 
                                          mode='same', boundary='symm')
            return result

class MultiCoreEdgeDetection(TiledOperation):
    """
    Multi-core implementation of edge detection.
    """
    
    def __init__(self, tile_size: Tuple[int, int] = (256, 256),
                 max_workers: Optional[int] = None):
        super().__init__(name="MultiCoreEdgeDetection", 
                         tile_size=tile_size, max_workers=max_workers)        
        # Sobel operators for edge detection
        self.kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        self.kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    def _process_tile(self, tile: np.ndarray, **kwargs) -> np.ndarray:
        """Process a single tile with edge detection."""
        from scipy.signal import convolve2d
        
        # Convert to grayscale if needed
        if len(tile.shape) == 3:
            gray = np.mean(tile, axis=2).astype(np.float32)
            is_color = True
        else:
            gray = tile.astype(np.float32)
            is_color = False
        
        # Apply Sobel operators
        grad_x = convolve2d(gray, self.kernel_x, mode='same', boundary='symm')
        grad_y = convolve2d(gray, self.kernel_y, mode='same', boundary='symm')
        
        # Compute gradient magnitude
        gradient = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize to 0-255 range
        gradient = np.clip(gradient, 0, 255).astype(np.uint8)
        
        # Convert back to color format if the input was color
        if is_color:
            # Create a 3-channel output with the gradient as each channel
            # or alternatively create a visualization with edges in a specific color
            result = np.zeros_like(tile)
            # For a basic visualization, use the gradient for each channel
            result[:, :, 0] = gradient
            result[:, :, 1] = gradient
            result[:, :, 2] = gradient
            return result
        else:
            return gradient

class MultiCoreColorTransform(TiledOperation):
    """
    Multi-core implementation of color transformations.
    """
    
    def __init__(self, transformation: str = "grayscale", 
                 tile_size: Tuple[int, int] = (256, 256),
                 max_workers: Optional[int] = None):
        super().__init__(name=f"MultiCoreColorTransform-{transformation}", 
                         tile_size=tile_size, max_workers=max_workers)
        self.transformation = transformation
        
        # Precompute sepia matrix for efficiency
        self.sepia_matrix = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])
    
    def _process_tile(self, tile: np.ndarray, **kwargs) -> np.ndarray:
        """Process a single tile with color transformation."""
        if len(tile.shape) < 3:
            return tile  # Already grayscale
        
        if self.transformation == "grayscale":
            # Vectorized grayscale conversion
            return np.dot(tile[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        
        elif self.transformation == "invert":
            # Vectorized inversion
            return 255 - tile
        
        elif self.transformation == "sepia":
            # Vectorized sepia transformation
            reshaped = tile.reshape(-1, 3)
            transformed = np.dot(reshaped, self.sepia_matrix.T)
            transformed = np.clip(transformed, 0, 255).astype(np.uint8)
            
            # Reshape back to original tile shape
            return transformed.reshape(tile.shape)
        
        else:
            raise ValueError(f"Unknown transformation: {self.transformation}")
