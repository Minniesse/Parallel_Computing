import torch
import time
import threading
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
import os

from ..hardware_profiling.device_catalog import DeviceCatalog
from ..utils.energy_monitor import EnergyMonitor

class HardwareProfiler:
    """
    Profiles hardware performance characteristics for different operations.
    Used to make informed decisions about workload distribution.
    """
    
    def __init__(
        self, 
        device_catalog: DeviceCatalog,
        cache_file: Optional[str] = None,
        energy_monitor: Optional[EnergyMonitor] = None
    ):
        """
        Initialize the hardware profiler.
        
        Args:
            device_catalog: Catalog of available hardware devices
            cache_file: Optional path to save/load profiling data
            energy_monitor: Optional energy monitor for power profiling
        """
        self.device_catalog = device_catalog
        self.cache_file = cache_file
        self.energy_monitor = energy_monitor
        
        # Performance data for operations on different devices
        self.operation_profiles = {}
        # Data transfer costs between devices
        self.transfer_costs = {}
        
        if cache_file and os.path.exists(cache_file):
            self._load_from_cache()
        else:
            self._initialize_profiles()
    
    def _initialize_profiles(self):
        """Initialize empty performance profiles"""
        # For each device, create empty operation profiles
        for device_type in ['cpu', 'gpu']:
            count = self.device_catalog.get_device_count(device_type)
            for i in range(count):
                device_id = f"{device_type}:{i}"
                self.operation_profiles[device_id] = {}
        
        # Initialize transfer costs between all device pairs
        devices = list(self.operation_profiles.keys())
        for source in devices:
            self.transfer_costs[source] = {}
            for dest in devices:
                if source != dest:
                    self.transfer_costs[source][dest] = None
    
    def profile_operation(
        self, 
        operation_type: str, 
        input_shapes: List[Tuple[int, ...]], 
        device_id: str,
        warmup_runs: int = 5,
        benchmark_runs: int = 20,
        measure_energy: bool = False
    ) -> Dict[str, Any]:
        """
        Profile a specific operation on a specific device.
        
        Args:
            operation_type: Type of operation (e.g., 'conv2d', 'linear')
            input_shapes: List of input tensor shapes
            device_id: Device to profile on (e.g., 'cpu:0', 'cuda:0')
            warmup_runs: Number of warm-up runs before benchmarking
            benchmark_runs: Number of runs to average performance over
            measure_energy: Whether to measure energy consumption
            
        Returns:
            Dictionary with profiling results
        """
        device = torch.device(device_id)
        op_key = self._get_operation_key(operation_type, input_shapes)
        
        # Create dummy inputs based on shapes
        inputs = [torch.rand(*shape, device=device) for shape in input_shapes]
        
        # Create the operation to benchmark
        operation = self._create_operation(operation_type, input_shapes, device)
        
        # Warmup runs to stabilize performance
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = operation(*inputs)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        
        # Start energy monitoring if requested and available
        if measure_energy and self.energy_monitor:
            self.energy_monitor.start_monitoring(f"op_{operation_type}")
        
        # Benchmark runs
        start_time = time.time()
        for _ in range(benchmark_runs):
            with torch.no_grad():
                _ = operation(*inputs)
            torch.cuda.synchronize() if device.type == 'cuda' else None
        
        end_time = time.time()
        
        # Stop energy monitoring
        energy_data = None
        if measure_energy and self.energy_monitor:
            self.energy_monitor.stop_monitoring()
            energy_data = self.energy_monitor.get_last_session_summary()
        
        # Calculate performance metrics
        total_time = end_time - start_time
        avg_time = total_time / benchmark_runs
        
        # Estimate FLOPS
        flops = self._estimate_operation_flops(operation_type, input_shapes)
        flops_per_second = flops / avg_time if avg_time > 0 else 0
        
        # Measure peak memory usage
        mem_usage = self._measure_memory_usage(operation, inputs, device)
        
        # Create profiling results
        results = {
            'avg_execution_time': avg_time,
            'flops': flops,
            'flops_per_second': flops_per_second,
            'memory_usage': mem_usage
        }
        
        # Add energy data if available
        if energy_data and 'energy' in energy_data:
            device_type = device_id.split(':')[0]
            if device_type == 'cuda':
                device_type = 'gpu'
            
            device_energy = energy_data['energy'].get(device_type, {})
            if device_energy:
                results['energy'] = {
                    'avg_power_watts': device_energy.get('avg_power_watts', 0),
                    'energy_joules': device_energy.get('energy_joules', 0) / benchmark_runs
                }
        
        # Cache the results
        self.operation_profiles[device_id][op_key] = results
        
        if self.cache_file:
            self._save_to_cache()
            
        return results
    
    def profile_data_transfer(
        self, 
        source_device: str, 
        dest_device: str, 
        sizes: List[int],
        warmup_runs: int = 5,
        benchmark_runs: int = 20
    ) -> Dict[str, float]:
        """
        Profile data transfer costs between two devices.
        
        Args:
            source_device: Source device ID (e.g., 'cpu:0')
            dest_device: Destination device ID (e.g., 'cuda:0')
            sizes: List of tensor sizes in bytes to benchmark
            warmup_runs: Number of warm-up runs
            benchmark_runs: Number of benchmark runs
            
        Returns:
            Dictionary with transfer rates
        """
        if source_device == dest_device:
            return {'transfer_rate_gb_per_s': float('inf'), 'latency_ms': 0.0}
        
        source = torch.device(source_device)
        dest = torch.device(dest_device)
        
        results = {
            'transfer_rates': {},
            'latencies': {}
        }
        
        for size in sizes:
            # Create tensor on source device
            elements = max(1, size // 4)  # 4 bytes per float32
            x = torch.rand(elements, device=source)
            
            # Warmup
            for _ in range(warmup_runs):
                y = x.to(dest)
                torch.cuda.synchronize() if dest.type == 'cuda' or source.type == 'cuda' else None
            
            # Benchmark - measure latency with small tensor
            small_tensor = torch.rand(1, device=source)
            latency_start = time.time()
            for _ in range(benchmark_runs):
                y = small_tensor.to(dest)
                torch.cuda.synchronize() if dest.type == 'cuda' or source.type == 'cuda' else None
            latency_end = time.time()
            latency_ms = (latency_end - latency_start) * 1000 / benchmark_runs
            
            # Benchmark - measure throughput with large tensor
            start_time = time.time()
            for _ in range(benchmark_runs):
                y = x.to(dest)
                torch.cuda.synchronize() if dest.type == 'cuda' or source.type == 'cuda' else None
            end_time = time.time()
            
            # Calculate transfer rate
            elapsed = end_time - start_time
            bytes_transferred = size * benchmark_runs
            gb_transferred = bytes_transferred / (1024 ** 3)  # Convert to GB
            transfer_rate = gb_transferred / elapsed  # GB/s
            
            # Store results for this size
            results['transfer_rates'][size] = transfer_rate
            results['latencies'][size] = latency_ms
        
        # Calculate average transfer rate and latency
        avg_transfer_rate = np.mean(list(results['transfer_rates'].values()))
        avg_latency = np.mean(list(results['latencies'].values()))
        
        # Cache the results
        self.transfer_costs[source_device][dest_device] = {
            'transfer_rate_gb_per_s': avg_transfer_rate,
            'latency_ms': avg_latency,
            'detailed': results
        }
        
        if self.cache_file:
            self._save_to_cache()
            
        return self.transfer_costs[source_device][dest_device]
    
    def estimate_operation_time(
        self, 
        operation_type: str, 
        input_shapes: List[Tuple[int, ...]], 
        device_id: str
    ) -> float:
        """
        Estimate the execution time of an operation on a specific device.
        
        Args:
            operation_type: Type of operation
            input_shapes: Input tensor shapes
            device_id: Target device
            
        Returns:
            Estimated execution time in seconds
        """
        op_key = self._get_operation_key(operation_type, input_shapes)
        
        # Check if we have profiled this exact operation
        if device_id in self.operation_profiles and op_key in self.operation_profiles[device_id]:
            return self.operation_profiles[device_id][op_key]['avg_execution_time']
        
        # Otherwise, estimate based on similar operations
        flops = self._estimate_operation_flops(operation_type, input_shapes)
        device_type = device_id.split(':')[0]
        
        # Find all operations of the same type on this device
        similar_ops = {}
        if device_id in self.operation_profiles:
            for k, v in self.operation_profiles[device_id].items():
                if k.startswith(operation_type):
                    similar_ops[k] = v
        
        if similar_ops:
            # Calculate average FLOPS/s for this operation type on this device
            avg_flops_per_second = np.mean([
                op['flops_per_second'] for op in similar_ops.values()
            ])
            
            # Estimate time based on FLOPS and average performance
            if avg_flops_per_second > 0:
                return flops / avg_flops_per_second
        
        # If no similar operations found, use default estimate
        # This is a very rough estimate and should be improved with actual profiling
        if device_type == 'cuda':
            return flops / (5e9)  # Assume 5 TFLOPS for GPU
        else:
            return flops / (5e8)  # Assume 500 GFLOPS for CPU
    
    def estimate_transfer_cost(
        self, 
        source_op: str, 
        dest_op: str, 
        source_device: str, 
        dest_device: str, 
        estimated_bytes: Optional[int] = None
    ) -> float:
        """
        Estimate the cost of transferring data between operations on different devices.
        
        Args:
            source_op: Source operation ID
            dest_op: Destination operation ID
            source_device: Source device ID
            dest_device: Destination device ID
            estimated_bytes: Optional pre-calculated data size in bytes
            
        Returns:
            Estimated transfer time in seconds
        """
        if source_device == dest_device:
            return 0.0  # No transfer cost within the same device
        
        # Check if we have profiled this device pair
        if (source_device in self.transfer_costs and 
            dest_device in self.transfer_costs[source_device] and
            self.transfer_costs[source_device][dest_device] is not None):
            
            transfer_info = self.transfer_costs[source_device][dest_device]
            transfer_rate = transfer_info['transfer_rate_gb_per_s']
            latency_ms = transfer_info['latency_ms']
            
            # If bytes not provided, use a default estimate
            if estimated_bytes is None:
                estimated_bytes = 4 * 1024 * 1024  # 4 MB default
            
            # Calculate transfer time
            gb_to_transfer = estimated_bytes / (1024 ** 3)
            transfer_time = gb_to_transfer / transfer_rate if transfer_rate > 0 else 0
            
            # Add latency
            return transfer_time + (latency_ms / 1000.0)
        
        # No profile data available, use conservative estimate
        return 0.001  # 1 ms default
    
    def estimate_energy_consumption(
        self, 
        operations: List[str], 
        device_id: str
    ) -> float:
        """
        Estimate energy consumption for a list of operations on a device.
        
        Args:
            operations: List of operation IDs
            device_id: Target device
            
        Returns:
            Estimated energy consumption in joules
        """
        if not self.energy_monitor:
            return 0.0  # No energy monitoring available
        
        total_energy = 0.0
        device_profiles = self.operation_profiles.get(device_id, {})
        
        for op in operations:
            # Parse operation to get type and shapes
            # This assumes operation IDs contain this information
            op_type = op.split('_')[0] if '_' in op else op
            
            # Look for matching profiled operations
            for op_key, profile in device_profiles.items():
                if op_key.startswith(op_type) and 'energy' in profile:
                    total_energy += profile['energy'].get('energy_joules', 0)
                    break
        
        return total_energy
    
    def _get_operation_key(self, operation_type: str, input_shapes: List[Tuple[int, ...]]) -> str:
        """
        Create a unique key for an operation based on its type and input shapes.
        
        Args:
            operation_type: Operation type name
            input_shapes: List of input shapes
            
        Returns:
            String key representing the operation
        """
        shapes_str = '_'.join([
            'x'.join(map(str, shape)) for shape in input_shapes
        ])
        return f"{operation_type}_{shapes_str}"
    
    def _create_operation(self, operation_type: str, input_shapes: List[Tuple[int, ...]], device: torch.device):
        """
        Create a PyTorch operation for benchmarking.
        
        Args:
            operation_type: Type of operation to create
            input_shapes: Shapes of input tensors
            device: Device to create the operation on
            
        Returns:
            Callable operation
        """
        if operation_type == 'conv2d':
            in_channels = input_shapes[0][1]
            out_channels = 64  # Default value, can be parameterized
            kernel_size = 3
            return torch.nn.Conv2d(in_channels, out_channels, kernel_size).to(device)
        
        elif operation_type == 'linear':
            in_features = input_shapes[0][1] if len(input_shapes[0]) > 1 else input_shapes[0][0]
            out_features = 128  # Default value
            return torch.nn.Linear(in_features, out_features).to(device)
        
        elif operation_type == 'matmul':
            # Just return a function that performs matrix multiplication
            return lambda x, y: torch.matmul(x, y)
        
        elif operation_type == 'relu':
            return torch.nn.ReLU().to(device)
        
        # Add more operations as needed
        
        else:
            # Default: identity function
            return lambda x: x
    
    def _estimate_operation_flops(self, operation_type: str, input_shapes: List[Tuple[int, ...]]) -> int:
        """
        Estimate the number of floating-point operations (FLOPs) for an operation.
        
        Args:
            operation_type: Type of operation
            input_shapes: Input tensor shapes
            
        Returns:
            Estimated FLOPs
        """
        if operation_type == 'conv2d':
            # For convolution: 2 * Cin * Cout * K^2 * H * W FLOPs
            batch_size = input_shapes[0][0]
            in_channels = input_shapes[0][1]
            height = input_shapes[0][2]
            width = input_shapes[0][3]
            out_channels = 64  # From _create_operation
            kernel_size = 3
            
            return 2 * batch_size * in_channels * out_channels * kernel_size**2 * height * width
        
        elif operation_type == 'linear':
            # For linear layer: 2 * in_features * out_features * batch_size FLOPs
            batch_size = input_shapes[0][0]
            in_features = input_shapes[0][1] if len(input_shapes[0]) > 1 else input_shapes[0][0]
            out_features = 128  # From _create_operation
            
            return 2 * batch_size * in_features * out_features
        
        elif operation_type == 'matmul':
            # For matmul: 2 * N * M * K FLOPs
            if len(input_shapes) != 2:
                return 1000000  # Default value
                
            # Check dimensions for matmul compatibility
            shape1, shape2 = input_shapes
            if len(shape1) < 2 or len(shape2) < 2:
                return 1000000
                
            # For simplicity, assume classic matrix multiplication
            n = shape1[-2]
            k = shape1[-1]
            m = shape2[-1]
            
            # Account for batch dimensions
            batch_size = 1
            for i in range(len(shape1) - 2):
                batch_size *= shape1[i]
            
            return 2 * batch_size * n * m * k
        
        elif operation_type == 'relu':
            # ReLU is a simple element-wise operation
            elements = 1
            for dim in input_shapes[0]:
                elements *= dim
            return elements
        
        # Default estimate
        return 1000000  # 1M FLOPs as a default value
    
    def _measure_memory_usage(self, operation, inputs, device):
        """
        Measure memory usage of an operation.
        
        Args:
            operation: Operation to measure
            inputs: Input tensors
            device: Device the operation runs on
            
        Returns:
            Memory usage in bytes
        """
        # Memory usage is sum of inputs + outputs + parameters
        input_memory = sum(x.nelement() * x.element_size() for x in inputs)
        
        # Estimate parameter memory if operation is a module
        param_memory = 0
        if isinstance(operation, torch.nn.Module):
            param_memory = sum(p.nelement() * p.element_size() for p in operation.parameters())
        
        # Measure output memory
        torch.cuda.synchronize() if device.type == 'cuda' else None
        with torch.no_grad():
            outputs = operation(*inputs)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        
        # Calculate output memory
        if isinstance(outputs, torch.Tensor):
            output_memory = outputs.nelement() * outputs.element_size()
        elif isinstance(outputs, (list, tuple)):
            output_memory = sum(x.nelement() * x.element_size() for x in outputs 
                                if isinstance(x, torch.Tensor))
        else:
            output_memory = 0
        
        return input_memory + output_memory + param_memory
    
    def _save_to_cache(self):
        """Save profiling data to cache file"""
        cache_data = {
            'operation_profiles': self.operation_profiles,
            'transfer_costs': self.transfer_costs
        }
        
        try:
            cache_dir = os.path.dirname(self.cache_file)
            if cache_dir and not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
                
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save profiling data to cache. Error: {e}")
    
    def _load_from_cache(self):
        """Load profiling data from cache file"""
        try:
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)
                
            self.operation_profiles = cache_data.get('operation_profiles', {})
            self.transfer_costs = cache_data.get('transfer_costs', {})
        except Exception as e:
            print(f"Warning: Failed to load profiling data from cache. Error: {e}")
            self._initialize_profiles()
