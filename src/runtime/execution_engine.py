import torch
import torch.nn as nn
import time
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable
import queue
import copy

from ..distribution_strategies.strategy_generator import DistributionStrategy, DeviceAssignment
from ..runtime.performance_monitor import PerformanceMonitor

class ExecutionEngine:
    """
    Executes distributed deep learning workloads according to the distribution strategy.
    Handles communication, synchronization, and dynamic adjustments.
    """
    
    def __init__(
        self, 
        model: nn.Module,
        strategy: DistributionStrategy,
        performance_monitor: Optional[PerformanceMonitor] = None,
        enable_dynamic_adjustment: bool = True
    ):
        """
        Initialize the execution engine.
        
        Args:
            model: PyTorch model to distribute
            strategy: Distribution strategy to apply
            performance_monitor: Optional monitor for runtime performance metrics
            enable_dynamic_adjustment: Whether to enable dynamic strategy adjustment
        """
        self.original_model = model
        self.strategy = strategy
        self.performance_monitor = performance_monitor
        self.enable_dynamic_adjustment = enable_dynamic_adjustment
        
        self.device_models = {}  # Partitioned model components by device
        self.execution_order = []  # Topological order of operations
        self.tensor_locations = {}  # Current location of tensors
        self.intermediate_results = {}  # Storage for intermediate results
        
        self.adjustment_interval = 10  # Check for adjustments every N forward passes
        self.forward_count = 0
        
        # Set up the distributed execution
        self._partition_model()
        self._create_execution_plan()
    
    def _partition_model(self):
        """Partition the model according to the strategy"""
        # Create copies of the model for each device
        self.device_models = {}
        
        # Group operations by device
        for assignment in self.strategy.device_assignments:
            device_id = assignment.device_id
            device = torch.device(device_id)
            
            # Create a submodule containing only the operations for this device
            device_model = self._create_device_specific_model(assignment.operation_ids)
            device_model.to(device)
            
            self.device_models[device_id] = device_model
    
    def _create_device_specific_model(self, operation_ids):
        """
        Create a device-specific model containing only the specified operations.
        This is a simplified implementation - a real implementation would use FX to
        create proper submodules.
        """
        # In a real implementation, this would use PyTorch FX to:
        # 1. Extract the subgraph containing only the specified operations
        # 2. Create a new FX GraphModule from the subgraph
        # 3. Properly handle shared parameters
        
        # For this example, we'll just return a placeholder
        class DeviceModel(nn.Module):
            def __init__(self, parent_model, operations):
                super().__init__()
                self.parent_model = parent_model
                self.operations = operations
            
            def forward(self, *args, **kwargs):
                # This would actually execute only the subset of operations
                pass
        
        return DeviceModel(self.original_model, operation_ids)
    
    def _create_execution_plan(self):
        """Create a plan for executing operations in the correct order"""
        # Build a graph representation
        graph = {}
        dependencies = {}
        
        # Add all operations
        for assignment in self.strategy.device_assignments:
            for op_id in assignment.operation_ids:
                graph[op_id] = []
                dependencies[op_id] = set()
        
        # Add dependencies
        for op_id in graph:
            # This would normally come from the computational graph analysis
            # For this example, we'll use a placeholder dependency structure
            pass
        
        # Compute topological sort for execution order
        self.execution_order = self._topological_sort(graph)
        
        # Map operations to devices
        self.op_to_device = {}
        for assignment in self.strategy.device_assignments:
            for op_id in assignment.operation_ids:
                self.op_to_device[op_id] = assignment.device_id
    
    def _topological_sort(self, graph):
        """Perform topological sort to determine execution order"""
        visited = set()
        result = []
        
        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            for neighbor in graph.get(node, []):
                dfs(neighbor)
            result.append(node)
        
        for node in graph:
            if node not in visited:
                dfs(node)
        
        return list(reversed(result))
    
    def forward(self, *args, **kwargs):
        """
        Execute a forward pass through the distributed model.
        
        Args:
            *args, **kwargs: Input arguments for the model
            
        Returns:
            Model output
        """
        # Convert inputs to tensors if needed
        inputs = self._prepare_inputs(args, kwargs)
        
        # Reset intermediate results for this forward pass
        self.intermediate_results = {}
        
        # Start monitoring if available
        if self.performance_monitor:
            self.performance_monitor.start_forward_pass()
        
        # Execute operations in topological order
        for op_id in self.execution_order:
            device_id = self.op_to_device[op_id]
            device = torch.device(device_id)
            
            # Get inputs for this operation
            op_inputs = self._get_operation_inputs(op_id, inputs)
            
            # Move inputs to the correct device if needed
            op_inputs = self._move_tensors_to_device(op_inputs, device)
            
            # Execute the operation
            start_time = time.time()
            outputs = self._execute_operation(op_id, op_inputs)
            end_time = time.time()
            
            # Record performance metrics
            if self.performance_monitor:
                self.performance_monitor.record_operation_metrics(
                    op_id, device_id, end_time - start_time, 
                    self._estimate_memory_used(op_inputs, outputs)
                )
            
            # Store intermediate results
            self.intermediate_results[op_id] = outputs
            
            # Update tensor locations
            for i, tensor in enumerate(outputs if isinstance(outputs, (list, tuple)) else [outputs]):
                if isinstance(tensor, torch.Tensor):
                    tensor_id = f"{op_id}_output_{i}"
                    self.tensor_locations[tensor_id] = device_id
        
        # Get final outputs
        final_outputs = self._get_final_outputs()
        
        # Stop monitoring
        if self.performance_monitor:
            self.performance_monitor.end_forward_pass()
        
        # Check if we should adjust the strategy
        self.forward_count += 1
        if self.enable_dynamic_adjustment and self.forward_count % self.adjustment_interval == 0:
            threading.Thread(target=self._check_for_strategy_adjustment).start()
        
        return final_outputs
    
    def _prepare_inputs(self, args, kwargs):
        """Prepare input tensors for execution"""
        # Move all input tensors to the appropriate devices
        # For simplicity, we'll move all inputs to the first device
        if not self.strategy.device_assignments:
            return args, kwargs
            
        first_device = torch.device(self.strategy.device_assignments[0].device_id)
        
        # Move args
        prepared_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                prepared_args.append(arg.to(first_device))
            elif isinstance(arg, (list, tuple)):
                prepared_args.append([
                    item.to(first_device) if isinstance(item, torch.Tensor) else item
                    for item in arg
                ])
            else:
                prepared_args.append(arg)
        
        # Move kwargs
        prepared_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                prepared_kwargs[k] = v.to(first_device)
            elif isinstance(v, (list, tuple)):
                prepared_kwargs[k] = [
                    item.to(first_device) if isinstance(item, torch.Tensor) else item
                    for item in v
                ]
            else:
                prepared_kwargs[k] = v
        
        return prepared_args, prepared_kwargs
    
    def _get_operation_inputs(self, op_id, inputs):
        """Get inputs for a specific operation"""
        # This would normally come from dependency analysis
        # For this example, we'll just use the original inputs for the first operation
        # and intermediate results for others
        if op_id == self.execution_order[0]:
            return inputs
        else:
            # Look up dependencies and get corresponding intermediate results
            # Simplified for this example
            return self.intermediate_results.get(op_id + "_inputs", inputs)
    
    def _move_tensors_to_device(self, tensors, target_device):
        """Move tensors to the target device if they're on a different device"""
        if isinstance(tensors, torch.Tensor):
            if tensors.device != target_device:
                return tensors.to(target_device)
            return tensors
        elif isinstance(tensors, (list, tuple)):
            return [
                self._move_tensors_to_device(tensor, target_device)
                for tensor in tensors
            ]
        elif isinstance(tensors, dict):
            return {
                k: self._move_tensors_to_device(v, target_device)
                for k, v in tensors.items()
            }
        else:
            # Non-tensor data
            return tensors
    
    def _execute_operation(self, op_id, inputs):
        """Execute a specific operation on its assigned device"""
        device_id = self.op_to_device[op_id]
        device_model = self.device_models[device_id]
        
        # In a real implementation, this would execute the specific operation
        # using the device-specific model
        return device_model(*inputs[0], **inputs[1])
    
    def _estimate_memory_used(self, inputs, outputs):
        """Estimate memory used by an operation"""
        memory = 0
        
        # Add input tensor sizes
        for tensor in self._flatten_tensors(inputs):
            if isinstance(tensor, torch.Tensor):
                memory += tensor.element_size() * tensor.nelement()
        
        # Add output tensor sizes
        for tensor in self._flatten_tensors(outputs):
            if isinstance(tensor, torch.Tensor):
                memory += tensor.element_size() * tensor.nelement()
        
        return memory
    
    def _flatten_tensors(self, nested_structure):
        """Flatten a nested structure of tensors into a list"""
        result = []
        
        if isinstance(nested_structure, torch.Tensor):
            result.append(nested_structure)
        elif isinstance(nested_structure, (list, tuple)):
            for item in nested_structure:
                result.extend(self._flatten_tensors(item))
        elif isinstance(nested_structure, dict):
            for item in nested_structure.values():
                result.extend(self._flatten_tensors(item))
        
        return result
    
    def _get_final_outputs(self):
        """Get the final outputs of the model"""
        # Find operations that don't feed into other operations
        # Those are the final outputs
        # Simplified for this example - we'll just use the last operation's outputs
        last_op = self.execution_order[-1]
        return self.intermediate_results[last_op]
    
    def _check_for_strategy_adjustment(self):
        """Check if the strategy needs adjustment based on performance metrics"""
        if not self.performance_monitor:
            return
            
        # Get performance metrics
        metrics = self.performance_monitor.get_current_metrics()
        
        # Check for imbalances
        imbalance_detected = self._detect_load_imbalance(metrics)
        
        if imbalance_detected:
            # Adjust the strategy
            new_strategy = self._generate_adjusted_strategy(metrics)
            
            # Apply the new strategy
            self._apply_new_strategy(new_strategy)
    
    def _detect_load_imbalance(self, metrics):
        """Detect if there's a significant load imbalance between devices"""
        if not metrics or 'device_utilization' not in metrics:
            return False
            
        device_utils = metrics['device_utilization']
        if len(device_utils) <= 1:
            return False
            
        # Calculate utilization statistics
        utils = list(device_utils.values())
        avg_util = sum(utils) / len(utils)
        max_util = max(utils)
        min_util = min(utils)
        
        # Check if imbalance exceeds threshold
        imbalance_threshold = 0.3  # 30% difference between max and avg
        return (max_util - avg_util) / avg_util > imbalance_threshold
    
    def _generate_adjusted_strategy(self, metrics):
        """Generate an adjusted strategy based on performance metrics"""
        # In a real implementation, this would use the StrategyGenerator
        # to create a new strategy that addresses the observed imbalance
        # For this example, we'll just return the current strategy
        return self.strategy
    
    def _apply_new_strategy(self, new_strategy):
        """Apply a new distribution strategy"""
        # Check if strategy is different
        if new_strategy == self.strategy:
            return
            
        # Update the strategy
        self.strategy = new_strategy
        
        # Repartition the model
        self._partition_model()
        
        # Update execution plan
        self._create_execution_plan()
        
        # Clear caches
        self.tensor_locations = {}
        self.intermediate_results = {}
