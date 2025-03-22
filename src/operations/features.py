import numpy as np
from typing import Tuple, Optional, Union, List, Dict
from ..core.pipeline import Operation

class HarrisCornerDetection(Operation):
    """
    Harris corner detection with configurable backend.
    """
    
    def __init__(self, k: float = 0.04, threshold: float = 0.01, 
                 backend: str = "auto", tile_size: Tuple[int, int] = (256, 256)):
        """
        Initialize Harris corner detector.
        
        Args:
            k: Harris detector free parameter (typically 0.04-0.06)
            threshold: Threshold for corner detection
            backend: One of "auto", "sequential", "vectorized", "multicore", "gpu"
            tile_size: Size of tiles for parallel processing
        """
        super().__init__(name=f"HarrisCornerDetection-{backend}")
        self.k = k
        self.threshold = threshold
        self.backend = backend
        self.tile_size = tile_size
        
        # Lazy initialization of implementation
        self._implementation = None
    
    def _process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply Harris corner detection with the selected backend."""
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
                try:
                    import numba
                    backend = "numba"
                except ImportError:
                    backend = "vectorized"
        
        # Create the implementation
        if backend == "sequential":
            def harris_sequential(img, **kwargs):
                # Convert to grayscale if needed
                if len(img.shape) == 3:
                    gray = np.mean(img, axis=2).astype(np.float32)
                else:
                    gray = img.astype(np.float32)
                
                # Compute x and y derivatives
                dx = np.zeros_like(gray)
                dy = np.zeros_like(gray)
                
                # Simple gradient calculation
                for y in range(1, gray.shape[0] - 1):
                    for x in range(1, gray.shape[1] - 1):
                        dx[y, x] = (gray[y, x + 1] - gray[y, x - 1]) / 2
                        dy[y, x] = (gray[y + 1, x] - gray[y - 1, x]) / 2
                
                # Compute products of derivatives at each pixel
                dx2 = dx * dx
                dy2 = dy * dy
                dxy = dx * dy
                
                # Gaussian blur for smoothing, approximated with box filter
                window_size = 5
                offset = window_size // 2
                
                # Initialize matrices
                M = np.zeros((gray.shape[0], gray.shape[1], 2, 2), dtype=np.float32)
                
                # Compute sum of products in window around each pixel
                for y in range(offset, gray.shape[0] - offset):
                    for x in range(offset, gray.shape[1] - offset):
                        # Calculate sum in window
                        sum_dx2 = np.sum(dx2[y-offset:y+offset+1, x-offset:x+offset+1])
                        sum_dy2 = np.sum(dy2[y-offset:y+offset+1, x-offset:x+offset+1])
                        sum_dxy = np.sum(dxy[y-offset:y+offset+1, x-offset:x+offset+1])
                        
                        # Fill structure tensor
                        M[y, x] = np.array([[sum_dx2, sum_dxy], [sum_dxy, sum_dy2]])
                
                # Compute corner response
                response = np.zeros_like(gray)
                
                for y in range(gray.shape[0]):
                    for x in range(gray.shape[1]):
                        # Get structure tensor at pixel
                        A = M[y, x]
                        # Calculate determinant and trace
                        det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
                        trace = A[0, 0] + A[1, 1]
                        # Calculate Harris response
                        if trace > 0:  # Avoid division by zero
                            response[y, x] = det - self.k * trace * trace
                
                # Normalize response
                if np.max(response) > 0:
                    response = response / np.max(response)
                
                # Threshold
                corners = response > self.threshold
                
                # Convert to color image if input was color
                if len(img.shape) == 3:
                    result = img.copy()
                    # Mark corners in red
                    result[corners, 0] = 255
                    result[corners, 1] = 0
                    result[corners, 2] = 0
                    return result
                else:
                    # For grayscale, return binary corner map
                    return corners.astype(np.uint8) * 255
            
            self._implementation = harris_sequential
            
        elif backend == "vectorized":
            from scipy.ndimage import gaussian_filter
            
            def harris_vectorized(img, **kwargs):
                # Convert to grayscale if needed
                if len(img.shape) == 3:
                    gray = np.mean(img, axis=2).astype(np.float32)
                else:
                    gray = img.astype(np.float32)
                
                # Compute x and y derivatives using Sobel
                dx = np.zeros_like(gray)
                dy = np.zeros_like(gray)
                
                dx[1:-1, 1:-1] = (gray[1:-1, 2:] - gray[1:-1, :-2]) / 2
                dy[1:-1, 1:-1] = (gray[2:, 1:-1] - gray[:-2, 1:-1]) / 2
                
                # Compute products of derivatives at each pixel
                dx2 = dx * dx
                dy2 = dy * dy
                dxy = dx * dy
                
                # Gaussian blur for smoothing
                window_size = 5
                sigma = 1.0
                
                # Apply Gaussian filter
                dx2_smooth = gaussian_filter(dx2, sigma)
                dy2_smooth = gaussian_filter(dy2, sigma)
                dxy_smooth = gaussian_filter(dxy, sigma)
                
                # Compute corner response
                det = dx2_smooth * dy2_smooth - dxy_smooth * dxy_smooth
                trace = dx2_smooth + dy2_smooth
                response = det - self.k * trace * trace
                
                # Normalize response
                if np.max(response) > 0:
                    response = response / np.max(response)
                
                # Threshold
                corners = response > self.threshold
                
                # Convert to color image if input was color
                if len(img.shape) == 3:
                    result = img.copy()
                    # Mark corners in red
                    result[corners, 0] = 255
                    result[corners, 1] = 0
                    result[corners, 2] = 0
                    return result
                else:
                    # For grayscale, return binary corner map
                    return corners.astype(np.uint8) * 255
            
            self._implementation = harris_vectorized
            
        elif backend == "numba":
            import numba
            
            # Define Numba-accelerated corner detection
            @numba.jit(nopython=True)
            def compute_harris_response(gray, k):
                height, width = gray.shape
                response = np.zeros_like(gray)
                
                # Compute gradients
                dx = np.zeros_like(gray)
                dy = np.zeros_like(gray)
                
                for y in range(1, height - 1):
                    for x in range(1, width - 1):
                        dx[y, x] = (gray[y, x + 1] - gray[y, x - 1]) / 2
                        dy[y, x] = (gray[y + 1, x] - gray[y - 1, x]) / 2
                
                # Compute products
                dx2 = dx * dx
                dy2 = dy * dy
                dxy = dx * dy
                
                # Apply window operation
                window_size = 5
                offset = window_size // 2
                
                for y in range(offset, height - offset):
                    for x in range(offset, width - offset):
                        # Sum products in window
                        sum_dx2 = 0.0
                        sum_dy2 = 0.0
                        sum_dxy = 0.0
                        
                        for wy in range(-offset, offset + 1):
                            for wx in range(-offset, offset + 1):
                                # Gaussian-like weighting (approximated)
                                weight = 1.0 - (wx*wx + wy*wy) / ((offset+1) * (offset+1))
                                if weight < 0:
                                    weight = 0
                                
                                sum_dx2 += dx2[y + wy, x + wx] * weight
                                sum_dy2 += dy2[y + wy, x + wx] * weight
                                sum_dxy += dxy[y + wy, x + wx] * weight
                        
                        # Calculate response
                        det = sum_dx2 * sum_dy2 - sum_dxy * sum_dxy
                        trace = sum_dx2 + sum_dy2
                        if trace > 0:
                            response[y, x] = det - k * trace * trace
                
                return response
            
            def harris_numba(img, **kwargs):
                # Convert to grayscale if needed
                if len(img.shape) == 3:
                    gray = np.mean(img, axis=2).astype(np.float32)
                else:
                    gray = img.astype(np.float32)
                
                # Compute Harris response
                response = compute_harris_response(gray, self.k)
                
                # Normalize response
                if np.max(response) > 0:
                    response = response / np.max(response)
                
                # Threshold
                corners = response > self.threshold
                
                # Convert to color image if input was color
                if len(img.shape) == 3:
                    result = img.copy()
                    # Mark corners in red
                    result[corners, 0] = 255
                    result[corners, 1] = 0
                    result[corners, 2] = 0
                    return result
                else:
                    # For grayscale, return binary corner map
                    return corners.astype(np.uint8) * 255
            
            self._implementation = harris_numba
            
        elif backend == "multicore":
            from ..cpu.multicore import TiledOperation
            from scipy.ndimage import gaussian_filter
            
            class MultiCoreHarris(TiledOperation):
                def __init__(self, k, threshold, tile_size):
                    super().__init__(name="MultiCoreHarris", tile_size=tile_size)
                    self.k = k
                    self.threshold = threshold
                
                def _process_tile(self, tile, **kwargs):
                    # Convert to grayscale if needed
                    if len(tile.shape) == 3:
                        gray = np.mean(tile, axis=2).astype(np.float32)
                    else:
                        gray = tile.astype(np.float32)
                    
                    # Add padding to avoid boundary effects
                    pad_width = 5
                    padded = np.pad(gray, pad_width, mode='reflect')
                    
                    # Compute x and y derivatives using Sobel
                    dx = np.zeros_like(padded)
                    dy = np.zeros_like(padded)
                    
                    dx[1:-1, 1:-1] = (padded[1:-1, 2:] - padded[1:-1, :-2]) / 2
                    dy[1:-1, 1:-1] = (padded[2:, 1:-1] - padded[:-2, 1:-1]) / 2
                    
                    # Compute products of derivatives at each pixel
                    dx2 = dx * dx
                    dy2 = dy * dy
                    dxy = dx * dy
                    
                    # Gaussian blur for smoothing
                    sigma = 1.0
                    
                    # Apply Gaussian filter
                    dx2_smooth = gaussian_filter(dx2, sigma)
                    dy2_smooth = gaussian_filter(dy2, sigma)
                    dxy_smooth = gaussian_filter(dxy, sigma)
                    
                    # Compute corner response
                    det = dx2_smooth * dy2_smooth - dxy_smooth * dxy_smooth
                    trace = dx2_smooth + dy2_smooth
                    response = det - self.k * trace * trace
                    
                    # Remove padding
                    response = response[pad_width:-pad_width, pad_width:-pad_width]
                    
                    # Normalize response locally
                    max_val = np.max(response)
                    if max_val > 0:
                        response = response / max_val
                    
                    # Threshold
                    corners = response > self.threshold
                    
                    # Convert to color image if input was color
                    if len(tile.shape) == 3:
                        result = tile.copy()
                        # Mark corners in red
                        result[corners, 0] = 255
                        result[corners, 1] = 0
                        result[corners, 2] = 0
                        return result
                    else:
                        # For grayscale, return binary corner map
                        return corners.astype(np.uint8) * 255
            
            # Create a tiled Harris detector instance
            multicore_harris = MultiCoreHarris(self.k, self.threshold, self.tile_size)
            
            # Use its process method as our implementation
            self._implementation = multicore_harris._process
            
        elif backend == "gpu":
            try:
                import cupy as cp
                from cupyx.scipy.ndimage import gaussian_filter
                
                def harris_gpu(img, **kwargs):
                    # Transfer to GPU
                    gpu_img = cp.asarray(img)
                    
                    # Convert to grayscale if needed
                    if len(gpu_img.shape) == 3:
                        gray = cp.mean(gpu_img, axis=2).astype(cp.float32)
                    else:
                        gray = gpu_img.astype(cp.float32)
                    
                    # Compute x and y derivatives
                    dx = cp.zeros_like(gray)
                    dy = cp.zeros_like(gray)
                    
                    dx[1:-1, 1:-1] = (gray[1:-1, 2:] - gray[1:-1, :-2]) / 2
                    dy[1:-1, 1:-1] = (gray[2:, 1:-1] - gray[:-2, 1:-1]) / 2
                    
                    # Compute products of derivatives at each pixel
                    dx2 = dx * dx
                    dy2 = dy * dy
                    dxy = dx * dy
                    
                    # Gaussian blur for smoothing
                    sigma = 1.0
                    
                    # Apply Gaussian filter
                    dx2_smooth = gaussian_filter(dx2, sigma)
                    dy2_smooth = gaussian_filter(dy2, sigma)
                    dxy_smooth = gaussian_filter(dxy, sigma)
                    
                    # Compute corner response
                    det = dx2_smooth * dy2_smooth - dxy_smooth * dxy_smooth
                    trace = dx2_smooth + dy2_smooth
                    response = det - self.k * trace * trace
                    
                    # Normalize response
                    max_val = cp.max(response)
                    if max_val > 0:
                        response = response / max_val
                    
                    # Threshold
                    corners = response > self.threshold
                    
                    # Convert to color image if input was color
                    if len(gpu_img.shape) == 3:
                        result = gpu_img.copy()
                        # Mark corners in red
                        result[corners, 0] = 255
                        result[corners, 1] = 0
                        result[corners, 2] = 0
                        return cp.asnumpy(result)
                    else:
                        # For grayscale, return binary corner map
                        return cp.asnumpy(corners.astype(cp.uint8) * 255)
                
                self._implementation = harris_gpu
                
            except ImportError:
                # Fall back to vectorized if CuPy isn't available
                from scipy.ndimage import gaussian_filter
                
                def harris_vectorized(img, **kwargs):
                    # Convert to grayscale if needed
                    if len(img.shape) == 3:
                        gray = np.mean(img, axis=2).astype(np.float32)
                    else:
                        gray = img.astype(np.float32)
                    
                    # Compute x and y derivatives using Sobel
                    dx = np.zeros_like(gray)
                    dy = np.zeros_like(gray)
                    
                    dx[1:-1, 1:-1] = (gray[1:-1, 2:] - gray[1:-1, :-2]) / 2
                    dy[1:-1, 1:-1] = (gray[2:, 1:-1] - gray[:-2, 1:-1]) / 2
                    
                    # Compute products of derivatives at each pixel
                    dx2 = dx * dx
                    dy2 = dy * dy
                    dxy = dx * dy
                    
                    # Gaussian blur for smoothing
                    window_size = 5
                    sigma = 1.0
                    
                    # Apply Gaussian filter
                    dx2_smooth = gaussian_filter(dx2, sigma)
                    dy2_smooth = gaussian_filter(dy2, sigma)
                    dxy_smooth = gaussian_filter(dxy, sigma)
                    
                    # Compute corner response
                    det = dx2_smooth * dy2_smooth - dxy_smooth * dxy_smooth
                    trace = dx2_smooth + dy2_smooth
                    response = det - self.k * trace * trace
                    
                    # Normalize response
                    if np.max(response) > 0:
                        response = response / np.max(response)
                    
                    # Threshold
                    corners = response > self.threshold
                    
                    # Convert to color image if input was color
                    if len(img.shape) == 3:
                        result = img.copy()
                        # Mark corners in red
                        result[corners, 0] = 255
                        result[corners, 1] = 0
                        result[corners, 2] = 0
                        return result
                    else:
                        # For grayscale, return binary corner map
                        return corners.astype(np.uint8) * 255
                
                self._implementation = harris_vectorized
                
        else:
            raise ValueError(f"Unknown backend: {backend}")

class CannyEdgeDetection(Operation):
    """
    Canny edge detection with configurable backend.
    """
    
    def __init__(self, low_threshold: float = 50, high_threshold: float = 150, 
                 backend: str = "auto", sigma: float = 1.0):
        """
        Initialize Canny edge detector.
        
        Args:
            low_threshold: Low threshold for hysteresis
            high_threshold: High threshold for hysteresis
            backend: One of "auto", "sequential", "vectorized", "multicore", "gpu"
            sigma: Standard deviation for Gaussian blur
        """
        super().__init__(name=f"CannyEdgeDetection-{backend}")
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.sigma = sigma
        self.backend = backend
        
        # Lazy initialization of implementation
        self._implementation = None
    
    def _process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply Canny edge detection with the selected backend."""
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
                import cv2
                backend = "opencv"
            except ImportError:
                try:
                    import skimage
                    backend = "skimage"
                except ImportError:
                    backend = "vectorized"
        
        # Create the implementation
        if backend == "opencv":
            import cv2
            
            def canny_opencv(img, **kwargs):
                # Convert to grayscale if needed
                if len(img.shape) == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                else:
                    gray = img.copy()
                
                # Apply Gaussian blur
                blurred = cv2.GaussianBlur(gray, (5, 5), self.sigma)
                
                # Apply Canny edge detection
                edges = cv2.Canny(blurred, self.low_threshold, self.high_threshold)
                
                # Return edges
                return edges
            
            self._implementation = canny_opencv
            
        elif backend == "skimage":
            from skimage.feature import canny
            from skimage.color import rgb2gray
            
            def canny_skimage(img, **kwargs):
                # Convert to grayscale if needed
                if len(img.shape) == 3:
                    gray = rgb2gray(img)
                else:
                    gray = img.astype(np.float64) / 255.0
                
                # Apply Canny edge detection
                edges = canny(gray, sigma=self.sigma, 
                             low_threshold=self.low_threshold/255.0, 
                             high_threshold=self.high_threshold/255.0)
                
                # Return edges
                return edges.astype(np.uint8) * 255
            
            self._implementation = canny_skimage
            
        elif backend == "vectorized" or backend == "sequential":
            from scipy.ndimage import gaussian_filter
            
            def canny_vectorized(img, **kwargs):
                # Convert to grayscale if needed
                if len(img.shape) == 3:
                    gray = np.mean(img, axis=2).astype(np.float32)
                else:
                    gray = img.astype(np.float32)
                
                # Apply Gaussian blur
                blurred = gaussian_filter(gray, self.sigma)
                
                # Calculate gradients
                dx = np.zeros_like(blurred)
                dy = np.zeros_like(blurred)
                
                dx[1:-1, 1:-1] = (blurred[1:-1, 2:] - blurred[1:-1, :-2]) / 2
                dy[1:-1, 1:-1] = (blurred[2:, 1:-1] - blurred[:-2, 1:-1]) / 2
                
                # Calculate gradient magnitude and direction
                magnitude = np.sqrt(dx**2 + dy**2)
                direction = np.arctan2(dy, dx)
                
                # Non-maximum suppression
                suppressed = np.zeros_like(magnitude)
                
                for y in range(1, magnitude.shape[0] - 1):
                    for x in range(1, magnitude.shape[1] - 1):
                        # Get gradient direction
                        theta = direction[y, x]
                        
                        # Round angle to nearest 45 degrees
                        angle = np.round(theta * 4 / np.pi) % 4
                        
                        # Get neighbors along gradient direction
                        if angle == 0:  # 0 degrees
                            n1 = magnitude[y, x - 1]
                            n2 = magnitude[y, x + 1]
                        elif angle == 1:  # 45 degrees
                            n1 = magnitude[y - 1, x + 1]
                            n2 = magnitude[y + 1, x - 1]
                        elif angle == 2:  # 90 degrees
                            n1 = magnitude[y - 1, x]
                            n2 = magnitude[y + 1, x]
                        else:  # 135 degrees
                            n1 = magnitude[y - 1, x - 1]
                            n2 = magnitude[y + 1, x + 1]
                        
                        # Non-maximum suppression
                        if magnitude[y, x] >= n1 and magnitude[y, x] >= n2:
                            suppressed[y, x] = magnitude[y, x]
                
                # Double threshold and hysteresis
                result = np.zeros_like(suppressed)
                strong = suppressed > self.high_threshold
                weak = (suppressed >= self.low_threshold) & (suppressed <= self.high_threshold)
                
                # Strong edges
                result[strong] = 255
                
                # Weak edges connected to strong edges
                for y in range(1, suppressed.shape[0] - 1):
                    for x in range(1, suppressed.shape[1] - 1):
                        if weak[y, x]:
                            # Check if any neighbor is a strong edge
                            if np.any(strong[y-1:y+2, x-1:x+2]):
                                result[y, x] = 255
                
                return result.astype(np.uint8)
            
            self._implementation = canny_vectorized
            
        elif backend == "gpu":
            try:
                import cupy as cp
                import cv2
                
                def canny_gpu(img, **kwargs):
                    # For GPU, we'll use OpenCV for Canny detection on CPU
                    # since cupy doesn't have a direct Canny implementation
                    
                    # Convert to grayscale if needed
                    if len(img.shape) == 3:
                        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    else:
                        gray = img.copy()
                    
                    # Apply Gaussian blur
                    blurred = cv2.GaussianBlur(gray, (5, 5), self.sigma)
                    
                    # Apply Canny edge detection
                    edges = cv2.Canny(blurred, self.low_threshold, self.high_threshold)
                    
                    # Return edges
                    return edges
                
                self._implementation = canny_gpu
                
            except ImportError:
                # Fall back to skimage if GPU libraries aren't available
                from skimage.feature import canny
                from skimage.color import rgb2gray
                
                def canny_skimage(img, **kwargs):
                    # Convert to grayscale if needed
                    if len(img.shape) == 3:
                        gray = rgb2gray(img)
                    else:
                        gray = img.astype(np.float64) / 255.0
                    
                    # Apply Canny edge detection
                    edges = canny(gray, sigma=self.sigma, 
                                 low_threshold=self.low_threshold/255.0, 
                                 high_threshold=self.high_threshold/255.0)
                    
                    # Return edges
                    return edges.astype(np.uint8) * 255
                
                self._implementation = canny_skimage
                
        else:
            raise ValueError(f"Unknown backend: {backend}")
