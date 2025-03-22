"""
API server to connect the parallelism optimization framework with the Rust TUI.
"""

import json
import time
import argparse
import threading
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import os
import random
from flask import Flask, jsonify, request

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("api-server")

# Import framework components
try:
    from parallel_opt import OptimizedParallel
    from utils.energy_monitor import EnergyMonitor
    from runtime.performance_monitor import PerformanceMonitor
    from hardware_profiling.device_catalog import DeviceCatalog
    logger.info("Successfully imported parallelism optimization framework")
except ImportError:
    logger.warning("Could not import parallelism optimization framework")
    logger.warning("Running in mock mode - will return simulated data")

app = Flask(__name__)

# Global state
framework_instance = None
performance_monitor = None
energy_monitor = None
device_catalog = None
current_config = {
    "energy_aware": True,
    "communication_aware": True,
    "enable_monitoring": True,
    "dynamic_adjustment": True,
    "memory_fraction": 0.9,
}

# Mock data generation (only used if real framework is not available)
class MockDataGenerator:
    """Generates mock data for testing the TUI without the actual framework."""
    
    def __init__(self):
        self.devices = [
            {
                "device_id": "cuda:0",
                "name": "NVIDIA GeForce RTX 3080",
                "memory_total": 10_000_000_000,
            },
            {
                "device_id": "cpu:0",
                "name": "Intel Core i9-10900K",
                "memory_total": 16_000_000_000,
            }
        ]
        
        self.operations = [
            {"op_id": "conv1", "device_id": "cuda:0"},
            {"op_id": "relu1", "device_id": "cuda:0"},
            {"op_id": "pool1", "device_id": "cuda:0"},
            {"op_id": "conv2", "device_id": "cuda:0"},
            {"op_id": "fc1", "device_id": "cpu:0"},
            {"op_id": "fc2", "device_id": "cpu:0"},
        ]
        
        self.current_strategy = {
            "device_assignments": [
                {
                    "device_id": "cuda:0",
                    "operation_ids": ["conv1", "relu1", "pool1", "conv2"],
                    "estimated_compute_time": 0.015,
                    "estimated_memory_usage": 4_000_000_000,
                    "energy_consumption": 3.2
                },
                {
                    "device_id": "cpu:0",
                    "operation_ids": ["fc1", "fc2"],
                    "estimated_compute_time": 0.008,
                    "estimated_memory_usage": 2_000_000_000,
                    "energy_consumption": 1.8
                }
            ],
            "communication_cost": 0.002,
            "estimated_total_time": 0.017,
            "estimated_energy": 5.0,
            "memory_peak": {"cuda:0": 4_000_000_000, "cpu:0": 2_000_000_000}
        }
        
    def get_current_metrics(self):
        """Generate mock performance metrics."""
        current_time = datetime.now().isoformat()
        
        # Generate random device metrics
        devices = []
        for device in self.devices:
            utilization = random.uniform(30.0, 90.0) if "cuda" in device["device_id"] else random.uniform(10.0, 50.0)
            memory_used = random.randint(int(device["memory_total"] * 0.2), int(device["memory_total"] * 0.8))
            operations_count = sum(1 for op in self.operations if op["device_id"] == device["device_id"])
            compute_time = random.uniform(0.01, 0.05) if "cuda" in device["device_id"] else random.uniform(0.05, 0.2)
            
            devices.append({
                "device_id": device["device_id"],
                "name": device["name"],
                "utilization": utilization,
                "memory_used": memory_used,
                "memory_total": device["memory_total"],
                "operations_count": operations_count,
                "compute_time": compute_time
            })
        
        # Generate random operation metrics
        operations = []
        for op in self.operations:
            execution_time = random.uniform(0.001, 0.01) if "cuda" in op["device_id"] else random.uniform(0.005, 0.02)
            memory_usage = random.randint(10_000_000, 500_000_000)
            flops = random.randint(1_000_000, 100_000_000)
            
            operations.append({
                "op_id": op["op_id"],
                "device_id": op["device_id"],
                "execution_time": execution_time,
                "memory_usage": memory_usage,
                "flops": flops
            })
        
        # Generate resource utilization
        gpu_utilization = {}
        for device in self.devices:
            if "cuda" in device["device_id"]:
                gpu_utilization[device["device_id"]] = {
                    "utilization": random.uniform(30.0, 90.0),
                    "memory": random.uniform(30.0, 90.0)
                }
        
        resource_utilization = {
            "cpu": random.uniform(10.0, 50.0),
            "memory": random.uniform(40.0, 80.0),
            "gpu": gpu_utilization
        }
        
        # Generate device utilization
        device_utilization = {}
        for device in self.devices:
            utilization = random.uniform(30.0, 90.0) if "cuda" in device["device_id"] else random.uniform(10.0, 50.0)
            device_utilization[device["device_id"]] = utilization
        
        return {
            "avg_execution_time": random.uniform(0.01, 0.05),
            "devices": devices,
            "operations": operations,
            "resource_utilization": resource_utilization,
            "device_utilization": device_utilization,
            "timestamp": current_time
        }
    
    def get_energy_metrics(self):
        """Generate mock energy metrics."""
        # Generate random device energy
        device_energy = {}
        total_energy_joules = 0.0
        
        for device in self.devices:
            avg_power = random.uniform(50.0, 200.0) if "cuda" in device["device_id"] else random.uniform(20.0, 80.0)
            max_power = avg_power * random.uniform(1.1, 1.5)
            min_power = avg_power * random.uniform(0.5, 0.9)
            energy_joules = avg_power * random.uniform(0.5, 2.0)
            energy_watt_hours = energy_joules / 3600.0
            
            device_energy[device["device_id"]] = {
                "avg_power_watts": avg_power,
                "max_power_watts": max_power,
                "min_power_watts": min_power,
                "energy_joules": energy_joules,
                "energy_watt_hours": energy_watt_hours
            }
            
            total_energy_joules += energy_joules
        
        return {
            "total_energy_joules": total_energy_joules,
            "total_energy_watt_hours": total_energy_joules / 3600.0,
            "device_energy": device_energy,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_current_strategy(self):
        """Get the current distribution strategy."""
        return self.current_strategy

# Initialize mock data generator
mock_data = MockDataGenerator()

# API endpoints
@app.route('/metrics/current', methods=['GET'])
def get_current_metrics():
    """Get current performance metrics."""
    logger.info("Fetching current metrics")
    
    if performance_monitor is not None:
        # Get real metrics from the framework
        metrics = performance_monitor.get_current_metrics()
        return jsonify(metrics)
    else:
        # Return mock metrics
        return jsonify(mock_data.get_current_metrics())

@app.route('/metrics/energy', methods=['GET'])
def get_energy_metrics():
    """Get energy consumption metrics."""
    logger.info("Fetching energy metrics")
    
    if energy_monitor is not None:
        # Get real energy metrics from the framework
        energy_data = energy_monitor.get_last_session_summary()
        return jsonify(energy_data)
    else:
        # Return mock energy metrics
        return jsonify(mock_data.get_energy_metrics())

@app.route('/strategy/current', methods=['GET'])
def get_current_strategy():
    """Get the current distribution strategy."""
    logger.info("Fetching current strategy")
    
    if framework_instance is not None:
        # Get real strategy from the framework
        # This would need to be implemented in the framework
        # For now, return mock strategy
        return jsonify(mock_data.get_current_strategy())
    else:
        # Return mock strategy
        return jsonify(mock_data.get_current_strategy())

@app.route('/config/update', methods=['POST'])
def update_config():
    """Update framework configuration."""
    logger.info("Updating configuration")
    
    data = request.json
    key = data.get('key')
    value = data.get('value')
    
    if not key or value is None:
        return jsonify({"error": "Missing key or value"}), 400
    
    logger.info(f"Setting {key} to {value}")
    
    # Update configuration
    current_config[key] = value
    
    # Apply configuration if framework is available
    if framework_instance is not None:
        # This would apply the configuration to the actual framework
        pass
    
    return jsonify({"success": True})

@app.route('/config/current', methods=['GET'])
def get_current_config():
    """Get the current framework configuration."""
    logger.info("Fetching current configuration")
    return jsonify(current_config)

def main():
    """Main entry point for the API server."""
    parser = argparse.ArgumentParser(description="Parallelism Optimization Framework API Server")
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind the server')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind the server')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    logger.info(f"Starting API server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
