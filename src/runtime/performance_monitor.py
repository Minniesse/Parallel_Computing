import time
import threading
import torch
import psutil
import numpy as np
from typing import Dict, List, Optional, Any
import json
import os
from collections import deque

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

class PerformanceMonitor:
    """
    Monitors runtime performance of distributed execution.
    Collects metrics for resource utilization, execution times, and memory usage.
    """
    
    def __init__(
        self, 
        log_file: Optional[str] = None,
        history_size: int = 100,
        polling_interval: float = 0.5  # seconds
    ):
        """
        Initialize the performance monitor.
        
        Args:
            log_file: Optional file to save performance data
            history_size: Number of data points to keep in history
            polling_interval: Time between resource utilization measurements
        """
        self.log_file = log_file
        self.history_size = history_size
        self.polling_interval = polling_interval
        
        # Performance metrics storage
        self.operation_metrics = {}
        self.device_utilization = {}
        self.execution_times = deque(maxlen=history_size)
        self.memory_usage = {}
        
        # Current forward pass data
        self.current_forward_start = None
        self.current_operations = {}
        
        # Resource utilization monitoring
        self.monitor_resources = False
        self.monitor_thread = None
        self.resource_history = {
            'timestamps': deque(maxlen=history_size),
            'cpu_utilization': deque(maxlen=history_size),
            'memory_utilization': deque(maxlen=history_size),
            'gpu_utilization': {} if PYNVML_AVAILABLE else None,
            'gpu_memory': {} if PYNVML_AVAILABLE else None
        }
        
        # Initialize NVML for GPU monitoring
        self.nvml_initialized = False
        self.gpu_handles = {}
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_initialized = True
                
                # Get GPU handles
                device_count = pynvml.nvmlDeviceGetCount()
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    self.gpu_handles[i] = handle
                    device_id = f"cuda:{i}"
                    self.resource_history['gpu_utilization'][device_id] = deque(maxlen=history_size)
                    self.resource_history['gpu_memory'][device_id] = deque(maxlen=history_size)
            except Exception as e:
                print(f"Warning: NVML initialization failed. GPU monitoring unavailable. Error: {e}")
    
    def start_forward_pass(self):
        """Start monitoring a forward pass"""
        self.current_forward_start = time.time()
        self.current_operations = {}
        
        # Start resource monitoring if not already running
        if not self.monitor_resources:
            self._start_resource_monitoring()
    
    def end_forward_pass(self):
        """End monitoring the current forward pass"""
        if self.current_forward_start is None:
            return
            
        end_time = time.time()
        elapsed = end_time - self.current_forward_start
        
        # Store execution time
        self.execution_times.append(elapsed)
        
        # Aggregate operation metrics for this forward pass
        for op_id, metrics in self.current_operations.items():
            if op_id not in self.operation_metrics:
                self.operation_metrics[op_id] = {
                    'execution_times': deque(maxlen=self.history_size),
                    'device_id': metrics['device_id'],
                    'memory_usage': deque(maxlen=self.history_size)
                }
            
            self.operation_metrics[op_id]['execution_times'].append(metrics['execution_time'])
            self.operation_metrics[op_id]['memory_usage'].append(metrics['memory_usage'])
        
        # Reset current forward pass data
        self.current_forward_start = None
        
        # Save metrics to log if configured
        if self.log_file:
            self._save_metrics()
    
    def record_operation_metrics(self, op_id: str, device_id: str, execution_time: float, memory_usage: int):
        """
        Record metrics for a specific operation.
        
        Args:
            op_id: Operation identifier
            device_id: Device the operation ran on
            execution_time: Execution time in seconds
            memory_usage: Memory usage in bytes
        """
        self.current_operations[op_id] = {
            'device_id': device_id,
            'execution_time': execution_time,
            'memory_usage': memory_usage,
            'timestamp': time.time()
        }
        
        # Update device utilization
        if device_id not in self.device_utilization:
            self.device_utilization[device_id] = {
                'total_time': 0,
                'operations': set(),
                'peak_memory': 0
            }
        
        # Update device stats
        self.device_utilization[device_id]['total_time'] += execution_time
        self.device_utilization[device_id]['operations'].add(op_id)
        self.device_utilization[device_id]['peak_memory'] = max(
            self.device_utilization[device_id]['peak_memory'],
            memory_usage
        )
    
    def _start_resource_monitoring(self):
        """Start monitoring resource utilization in a background thread"""
        if self.monitor_resources:
            return
            
        self.monitor_resources = True
        self.monitor_thread = threading.Thread(target=self._resource_monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def _stop_resource_monitoring(self):
        """Stop the resource monitoring thread"""
        self.monitor_resources = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
            self.monitor_thread = None
    
    def _resource_monitoring_loop(self):
        """Background loop that periodically collects resource utilization metrics"""
        while self.monitor_resources:
            # Get current timestamp
            current_time = time.time()
            self.resource_history['timestamps'].append(current_time)
            
            # Measure CPU utilization
            cpu_percent = psutil.cpu_percent(interval=None)
            self.resource_history['cpu_utilization'].append(cpu_percent)
            
            # Measure system memory utilization
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self.resource_history['memory_utilization'].append(memory_percent)
            
            # Measure GPU utilization if available
            if self.nvml_initialized:
                for gpu_idx, handle in self.gpu_handles.items():
                    device_id = f"cuda:{gpu_idx}"
                    try:
                        # Get GPU utilization
                        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        gpu_util = utilization.gpu
                        self.resource_history['gpu_utilization'][device_id].append(gpu_util)
                        
                        # Get GPU memory usage
                        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        memory_used_percent = 100.0 * memory_info.used / memory_info.total
                        self.resource_history['gpu_memory'][device_id].append(memory_used_percent)
                    except Exception:
                        # Use zeros if reading fails
                        self.resource_history['gpu_utilization'][device_id].append(0)
                        self.resource_history['gpu_memory'][device_id].append(0)
            
            # Sleep until next measurement, accounting for time spent measuring
            elapsed = time.time() - current_time
            sleep_time = max(0, self.polling_interval - elapsed)
            time.sleep(sleep_time)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        # Calculate average execution time
        avg_execution_time = np.mean(self.execution_times) if self.execution_times else 0
        
        # Calculate device utilization percentages
        device_utils = {}
        total_time = sum(device['total_time'] for device in self.device_utilization.values())
        if total_time > 0:
            for device_id, stats in self.device_utilization.items():
                device_utils[device_id] = (stats['total_time'] / total_time) * 100
        
        # Get operation-specific metrics
        op_metrics = {}
        for op_id, metrics in self.operation_metrics.items():
            op_metrics[op_id] = {
                'avg_execution_time': np.mean(metrics['execution_times']) if metrics['execution_times'] else 0,
                'avg_memory_usage': np.mean(metrics['memory_usage']) if metrics['memory_usage'] else 0,
                'device_id': metrics['device_id']
            }
        
        # Get latest resource utilization metrics
        resource_utils = {
            'cpu': self.resource_history['cpu_utilization'][-1] if self.resource_history['cpu_utilization'] else 0,
            'memory': self.resource_history['memory_utilization'][-1] if self.resource_history['memory_utilization'] else 0,
            'gpu': {}
        }
        
        if self.nvml_initialized:
            for device_id, utils in self.resource_history['gpu_utilization'].items():
                if utils:
                    resource_utils['gpu'][device_id] = {
                        'utilization': utils[-1],
                        'memory': self.resource_history['gpu_memory'][device_id][-1] if self.resource_history['gpu_memory'][device_id] else 0
                    }
        
        return {
            'avg_execution_time': avg_execution_time,
            'device_utilization': device_utils,
            'operation_metrics': op_metrics,
            'resource_utilization': resource_utils,
            'timestamp': time.time()
        }
    
    def get_historical_metrics(self) -> Dict[str, Any]:
        """
        Get historical performance metrics.
        
        Returns:
            Dictionary with historical metrics
        """
        return {
            'execution_times': list(self.execution_times),
            'timestamps': list(self.resource_history['timestamps']),
            'cpu_utilization': list(self.resource_history['cpu_utilization']),
            'memory_utilization': list(self.resource_history['memory_utilization']),
            'gpu_utilization': {
                device_id: list(utils) 
                for device_id, utils in self.resource_history['gpu_utilization'].items()
            } if self.nvml_initialized else {},
            'gpu_memory': {
                device_id: list(mem) 
                for device_id, mem in self.resource_history['gpu_memory'].items()
            } if self.nvml_initialized else {}
        }
    
    def _save_metrics(self):
        """Save current metrics to log file"""
        try:
            metrics = {
                'current': self.get_current_metrics(),
                'historical': self.get_historical_metrics()
            }
            
            # Create directory if it doesn't exist
            log_dir = os.path.dirname(self.log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            # Write metrics to file
            with open(self.log_file, 'w') as f:
                json.dump(metrics, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save metrics. Error: {e}")
    
    def __del__(self):
        """Clean up resources when object is destroyed"""
        self._stop_resource_monitoring()
        
        if self.nvml_initialized:
            try:
                pynvml.nvmlShutdown()
                self.nvml_initialized = False
                self.gpu_handles = {}
            except Exception:
                pass
