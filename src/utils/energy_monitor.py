import time
import threading
import psutil
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import os
import json

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

class EnergyMonitor:
    """
    Monitors energy consumption of CPU and GPU devices during model execution.
    """
    
    def __init__(
        self, 
        sampling_interval: float = 0.1,
        log_file: Optional[str] = None,
        devices: Optional[List[str]] = None
    ):
        """
        Initialize the energy monitor.
        
        Args:
            sampling_interval: Time between energy measurements in seconds
            log_file: Optional file to log energy measurements
            devices: List of devices to monitor ('cpu', 'cuda:0', etc.) or None for all
        """
        self.sampling_interval = sampling_interval
        self.log_file = log_file
        self.devices = devices
        
        self.monitoring = False
        self.monitoring_thread = None
        
        self.measurements = []
        self.current_session = {}
        
        # Initialize NVML for GPU monitoring if available
        self.nvml_initialized = False
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_initialized = True
                self.gpu_handles = {
                    i: pynvml.nvmlDeviceGetHandleByIndex(i)
                    for i in range(pynvml.nvmlDeviceGetCount())
                }
            except Exception as e:
                print(f"Warning: NVML initialization failed. GPU energy monitoring unavailable. Error: {e}")
    
    def start_monitoring(self, session_name: str = None):
        """
        Start monitoring energy consumption.
        
        Args:
            session_name: Optional name to identify this monitoring session
        """
        if self.monitoring:
            self.stop_monitoring()
            
        self.monitoring = True
        self.current_session = {
            'name': session_name or f"session_{int(time.time())}",
            'start_time': time.time(),
            'end_time': None,
            'samples': []
        }
        
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring energy consumption and save results"""
        if not self.monitoring:
            return
            
        self.monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
            
        self.current_session['end_time'] = time.time()
        self.measurements.append(self.current_session)
        
        if self.log_file:
            self._save_measurements()
    
    def _monitoring_loop(self):
        """Main monitoring loop that collects energy measurements"""
        while self.monitoring:
            start_time = time.time()
            
            # Collect energy measurements
            sample = {
                'timestamp': start_time,
                'energy': self._measure_energy()
            }
            
            self.current_session['samples'].append(sample)
            
            # Sleep until next sample, accounting for measurement time
            elapsed = time.time() - start_time
            sleep_time = max(0, self.sampling_interval - elapsed)
            time.sleep(sleep_time)
    
    def _measure_energy(self) -> Dict[str, float]:
        """
        Measure energy consumption for all monitored devices.
        
        Returns:
            Dictionary mapping device IDs to power readings in watts
        """
        energy_readings = {}
        
        # Measure CPU power if requested
        if self.devices is None or any(d.startswith('cpu') for d in self.devices):
            cpu_power = self._measure_cpu_power()
            energy_readings['cpu'] = cpu_power
        
        # Measure GPU power if available and requested
        if self.nvml_initialized:
            for gpu_idx, handle in self.gpu_handles.items():
                gpu_id = f"cuda:{gpu_idx}"
                if self.devices is None or gpu_id in self.devices:
                    gpu_power = self._measure_gpu_power(handle)
                    energy_readings[gpu_id] = gpu_power
        
        return energy_readings
    
    def _measure_cpu_power(self) -> float:
        """
        Measure CPU power consumption in watts.
        This is an approximation since direct CPU power measurement
        requires platform-specific approaches.
        
        Returns:
            Estimated power in watts
        """
        # This is a simplified model based on CPU utilization
        # A more accurate approach would use platform-specific tools like:
        # - Intel RAPL on supported Intel CPUs
        # - powerstat on Linux
        # - powermetrics on macOS
        
        # Get CPU utilization as a proxy for power
        cpu_percent = psutil.cpu_percent(interval=None)
        
        # Simple linear model mapping utilization to power
        # This should be calibrated for specific hardware
        # Typical desktop CPU TDP ranges from 65W to 125W
        BASE_POWER = 10.0  # Idle power in watts
        MAX_POWER = 95.0   # Power at 100% utilization
        
        estimated_power = BASE_POWER + (cpu_percent / 100.0) * (MAX_POWER - BASE_POWER)
        
        return estimated_power
    
    def _measure_gpu_power(self, handle) -> float:
        """
        Measure GPU power consumption in watts using NVML.
        
        Args:
            handle: NVML device handle
            
        Returns:
            Power in watts
        """
        try:
            power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            return power_mw / 1000.0  # Convert milliwatts to watts
        except Exception as e:
            print(f"Warning: Failed to read GPU power. Error: {e}")
            return 0.0
    
    def _save_measurements(self):
        """Save measurements to log file"""
        try:
            # Create directory if it doesn't exist
            log_dir = os.path.dirname(self.log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            # Save measurements as JSON
            with open(self.log_file, 'w') as f:
                json.dump(self.measurements, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save energy measurements. Error: {e}")
    
    def get_last_session_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for the last monitoring session.
        
        Returns:
            Dictionary with energy consumption statistics
        """
        if not self.measurements:
            return {}
            
        session = self.measurements[-1]
        samples = session['samples']
        
        if not samples:
            return {
                'name': session['name'],
                'duration': 0,
                'energy': {}
            }
        
        # Calculate statistics for each device
        device_stats = {}
        devices = set()
        
        for sample in samples:
            devices.update(sample['energy'].keys())
        
        for device in devices:
            # Extract power readings for this device
            power_readings = [
                sample['energy'].get(device, 0)
                for sample in samples
                if device in sample['energy']
            ]
            
            if not power_readings:
                continue
                
            # Calculate statistics
            avg_power = np.mean(power_readings)
            max_power = np.max(power_readings)
            min_power = np.min(power_readings)
            
            # Calculate energy (power integrated over time)
            duration = session['end_time'] - session['start_time']
            energy_joules = avg_power * duration  # Watt-seconds = Joules
            energy_wh = energy_joules / 3600  # Convert Joules to Watt-hours
            
            device_stats[device] = {
                'avg_power_watts': avg_power,
                'max_power_watts': max_power,
                'min_power_watts': min_power,
                'energy_joules': energy_joules,
                'energy_watt_hours': energy_wh
            }
        
        return {
            'name': session['name'],
            'duration': session['end_time'] - session['start_time'],
            'energy': device_stats,
            'total_energy_joules': sum(stats['energy_joules'] for stats in device_stats.values()),
            'total_energy_watt_hours': sum(stats['energy_watt_hours'] for stats in device_stats.values())
        }
    
    def get_all_sessions_summary(self) -> List[Dict[str, Any]]:
        """
        Get summary statistics for all monitoring sessions.
        
        Returns:
            List of dictionaries with energy consumption statistics
        """
        summaries = []
        
        for i, _ in enumerate(self.measurements):
            # Temporarily set as last session to use existing summary logic
            last = self.measurements[-1]
            self.measurements[-1] = self.measurements[i]
            
            summary = self.get_last_session_summary()
            summaries.append(summary)
            
            # Restore original last session
            self.measurements[-1] = last
        
        return summaries
    
    def compare_sessions(self, session_names: List[str]) -> Dict[str, Any]:
        """
        Compare energy consumption between named sessions.
        
        Args:
            session_names: List of session names to compare
            
        Returns:
            Comparison statistics
        """
        session_data = []
        
        for session_name in session_names:
            # Find session with matching name
            matching_sessions = [
                s for s in self.measurements
                if s['name'] == session_name
            ]
            
            if not matching_sessions:
                print(f"Warning: No session found with name {session_name}")
                continue
                
            # Use the last matching session
            session = matching_sessions[-1]
            
            # Temporarily set as last session to use existing summary logic
            last = self.measurements[-1]
            self.measurements[-1] = session
            
            summary = self.get_last_session_summary()
            session_data.append(summary)
            
            # Restore original last session
            self.measurements[-1] = last
        
        if not session_data:
            return {}
            
        # Calculate comparison statistics
        comparison = {
            'sessions': session_data,
            'relative_energy': {},
            'energy_savings': {}
        }
        
        # Use first session as baseline
        baseline = session_data[0]
        baseline_energy = baseline['total_energy_joules']
        
        for session in session_data[1:]:
            session_energy = session['total_energy_joules']
            
            relative = session_energy / baseline_energy if baseline_energy > 0 else 0
            savings = 1.0 - relative
            
            comparison['relative_energy'][session['name']] = relative
            comparison['energy_savings'][session['name']] = savings
        
        return comparison
    
    def __del__(self):
        """Clean up resources when the object is destroyed"""
        self.stop_monitoring()
        
        if self.nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
