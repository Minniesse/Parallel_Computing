import time
from typing import List, Dict, Any, Callable, Optional, Union
import numpy as np

class Operation:
    """Base class for image processing operations."""
    
    def __init__(self, name: str):
        self.name = name
        self.execution_time = 0.0
    
    def __call__(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply the operation to an input image."""
        start_time = time.time()
        result = self._process(image, **kwargs)
        self.execution_time = time.time() - start_time
        return result
    
    def _process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Implementation of the specific operation."""
        raise NotImplementedError("Subclasses must implement _process method")

class Pipeline:
    """
    A pipeline for chaining multiple image processing operations.
    """
    
    def __init__(self, name: str = "ImagePipeline"):
        self.name = name
        self.operations: List[Operation] = []
        self.total_execution_time = 0.0
        self.operation_times: Dict[str, float] = {}
    
    def add_operation(self, operation: Operation) -> 'Pipeline':
        """Add an operation to the pipeline."""
        self.operations.append(operation)
        return self
    
    def __call__(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Process an image through the entire pipeline."""
        start_time = time.time()
        result = image.copy()
        
        for operation in self.operations:
            result = operation(result, **kwargs)
            self.operation_times[operation.name] = operation.execution_time
        
        self.total_execution_time = time.time() - start_time
        return result
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics for the pipeline execution."""
        stats = {
            'total_time': self.total_execution_time,
            'operations': self.operation_times
        }
        return stats
    
    def clear(self) -> None:
        """Clear all operations from the pipeline."""
        self.operations = []
        self.total_execution_time = 0.0
        self.operation_times = {}

class ParallelPipeline(Pipeline):
    """
    A pipeline that can execute operations in parallel where possible.
    """
    
    def __init__(self, name: str = "ParallelImagePipeline", scheduler=None):
        super().__init__(name)
        self.scheduler = scheduler
    
    def __call__(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Process an image through the pipeline with parallelization."""
        start_time = time.time()
        
        if self.scheduler is None:
            # Fall back to sequential execution if no scheduler is provided
            result = super().__call__(image, **kwargs)
        else:
            # Use the scheduler to execute operations in parallel where possible
            result = self.scheduler.execute(image, self.operations, **kwargs)
            
            # Update operation times from scheduler results
            self.operation_times = self.scheduler.get_operation_times()
        
        self.total_execution_time = time.time() - start_time
        return result
