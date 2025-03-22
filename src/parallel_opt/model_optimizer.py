"""
Model-aware optimization wrapper that adapts optimization strategy based on model complexity.
"""

import torch
import torch.nn as nn
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from . import OptimizedParallel

class AdaptiveOptimizer:
    """
    Adaptively chooses the best optimization strategy based on model complexity.
    Automatically applies appropriate optimization level for small vs large models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        energy_aware: bool = False,
        communication_aware: bool = True,
        enable_monitoring: bool = True,
        cache_dir: Optional[str] = None,
        complexity_threshold: int = 10000000,  # FLOPs threshold for complex models
        auto_detect: bool = True,              # Automatically detect model complexity
        small_model_overhead: float = 0.2      # Acceptable overhead for small models (fraction)
    ):
        """
        Initialize the adaptive optimizer.
        
        Args:
            model: PyTorch model to optimize
            energy_aware: Whether to optimize for energy efficiency
            communication_aware: Whether to optimize communication patterns
            enable_monitoring: Whether to collect performance metrics
            cache_dir: Directory to cache profiling data
            complexity_threshold: FLOPs threshold to consider a model as complex
            auto_detect: Automatically detect model complexity
            small_model_overhead: Target overhead for small models (as a fraction)
        """
        self.model = model
        self.energy_aware = energy_aware
        self.communication_aware = communication_aware
        self.enable_monitoring = enable_monitoring
        self.cache_dir = cache_dir
        self.complexity_threshold = complexity_threshold
        self.auto_detect = auto_detect
        self.small_model_overhead = small_model_overhead
        
        # Will be initialized on first call
        self.optimized_model = None
        self.is_complex_model = None
        self.original_performance = None
        self.using_optimization = True
        self.input_shape = None
        self.overhead_ratio = None
        self.batch_size = None
    
    def __call__(self, *args, **kwargs):
        """Execute the model with the appropriate optimization strategy."""
        # First call - determine if model is complex enough to benefit from optimization
        if self.optimized_model is None:
            return self._initialize_and_call(*args, **kwargs)
            
        # For small models that didn't benefit from optimization, use direct execution
        if not self.using_optimization:
            return self.model(*args, **kwargs)
            
        # For complex models, use the optimization framework
        return self.optimized_model(*args, **kwargs)
    
    def _initialize_and_call(self, *args, **kwargs):
        """Initialize and determine the best execution strategy."""
        # Store input shape for later analysis
        if isinstance(args[0], torch.Tensor):
            self.input_shape = args[0].shape
            if len(self.input_shape) > 0:
                self.batch_size = self.input_shape[0]
        
        # Detect model complexity if auto_detect is enabled
        if self.auto_detect:
            self.is_complex_model = self._estimate_model_complexity()
        else:
            # Use user-provided threshold
            self.is_complex_model = True
        
        # For all models, start with optimization to see if it helps
        print("Initializing adaptive optimization...")
        self.optimized_model = OptimizedParallel(
            self.model,
            energy_aware=self.energy_aware,
            communication_aware=self.communication_aware,
            enable_monitoring=self.enable_monitoring,
            cache_dir=self.cache_dir
        )
        
        # First optimized execution
        start_time = time.time()
        result = self.optimized_model(*args, **kwargs)
        optimized_time = time.time() - start_time
        
        # Now try execution without optimization
        device = next(self.model.parameters()).device
        original_model = self.model.to(device)
        
        # Clear GPU cache to ensure fair comparison
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Execute original model
        start_time = time.time()
        with torch.no_grad():
            original_result = original_model(*args, **kwargs)
        original_time = time.time() - start_time
        
        # Calculate overhead
        self.overhead_ratio = optimized_time / original_time
        
        # Decide whether to use optimization based on performance and complexity
        if self.is_complex_model:
            # For complex models, use optimization unless it's extremely slow
            self.using_optimization = self.overhead_ratio < 5.0
        else:
            # For simple models, only use optimization if it's close to original performance
            self.using_optimization = self.overhead_ratio < (1.0 + self.small_model_overhead)
            
        # Print decision
        if self.using_optimization:
            print(f"Performance analysis: Using optimization framework (overhead: {(self.overhead_ratio-1)*100:.1f}%)")
        else:
            print(f"Performance analysis: Using direct execution (optimization overhead: {(self.overhead_ratio-1)*100:.1f}%)")
            # Return result from original model for this call since we're not using optimization
            return original_result
            
        return result
    
    def _estimate_model_complexity(self) -> bool:
        """Estimate model complexity by analyzing parameters and operations."""
        # Count parameters
        num_params = sum(p.numel() for p in self.model.parameters())
        
        # Simple heuristic based on parameter count and batch size
        is_complex = num_params > 1000000  # Over 1M parameters
        
        if self.batch_size and self.batch_size > 16:
            is_complex = is_complex or num_params > 500000  # Lower threshold for larger batches
            
        print(f"Model complexity analysis: {num_params} parameters, batch size: {self.batch_size}")
        print(f"Recommendation: {'Complex model' if is_complex else 'Simple model'}")
        
        return is_complex
    
    def get_performance_metrics(self):
        """Get performance metrics from the optimized model if available."""
        if self.optimized_model and self.using_optimization:
            metrics = self.optimized_model.get_performance_metrics()
            # Add adaptive optimization metrics
            metrics['adaptive_optimization'] = {
                'is_complex_model': self.is_complex_model,
                'using_optimization': self.using_optimization,
                'overhead_ratio': self.overhead_ratio,
                'batch_size': self.batch_size
            }
            return metrics
        return None
    
    def get_historical_metrics(self):
        """Get historical metrics from the optimized model if available."""
        if self.optimized_model and self.using_optimization:
            return self.optimized_model.get_historical_metrics()
        return None
    
    def visualize_performance(self, output_dir=None, interactive=True):
        """Visualize performance metrics if optimization is being used."""
        if self.optimized_model and self.using_optimization:
            return self.optimized_model.visualize_performance(output_dir, interactive)
        print("Performance visualization not available when using direct execution.")
        return None
