import torch
import psutil
import platform
import os
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

@dataclass
class CPUInfo:
    name: str
    architecture: str
    cores_physical: int
    cores_logical: int
    frequency_max: float  # in MHz
    memory_total: int     # in bytes
    supports_avx: bool
    supports_avx2: bool
    supports_avx512: bool

@dataclass
class GPUInfo:
    name: str
    index: int
    uuid: str
    memory_total: int     # in bytes
    compute_capability: str
    core_count: int
    memory_bus_width: int # in bits
    max_power_limit: int  # in mW
    supports_tensor_cores: bool
    pcie_generation: int
    pcie_link_width: int

class DeviceCatalog:
    """
    Discovers and catalogs available computing resources (CPUs, GPUs).
    Provides detailed information about hardware capabilities.
    """
    
    def __init__(self, cache_file: Optional[str] = None):
        """
        Initialize the device catalog.
        
        Args:
            cache_file: Optional path to cache hardware information
        """
        self.cache_file = cache_file
        self.cpus = []
        self.gpus = []
        self._discover_devices()
    
    def _discover_devices(self):
        """Discover all available computing devices"""
        if self.cache_file and os.path.exists(self.cache_file):
            self._load_from_cache()
            return
            
        self._discover_cpus()
        if torch.cuda.is_available():
            self._discover_gpus()
            
        if self.cache_file:
            self._save_to_cache()
    
    def _discover_cpus(self):
        """Discover CPU information"""
        cpu_info = cpuinfo = psutil.cpu_info() if hasattr(psutil, 'cpu_info') else {}
        
        # Determine CPU features
        supports_avx = False
        supports_avx2 = False
        supports_avx512 = False
        
        # On Linux, check /proc/cpuinfo for CPU features
        if platform.system() == "Linux":
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo_content = f.read().lower()
                    supports_avx = 'avx' in cpuinfo_content
                    supports_avx2 = 'avx2' in cpuinfo_content
                    supports_avx512 = 'avx512' in cpuinfo_content
            except Exception:
                pass
                
        # Create CPU info object
        cpu = CPUInfo(
            name=getattr(cpu_info, "brand_string", platform.processor()),
            architecture=platform.machine(),
            cores_physical=psutil.cpu_count(logical=False) or 1,
            cores_logical=psutil.cpu_count(logical=True) or 1,
            frequency_max=getattr(cpu_info, "max_frequency", psutil.cpu_freq().max if psutil.cpu_freq() else 0),
            memory_total=psutil.virtual_memory().total,
            supports_avx=supports_avx,
            supports_avx2=supports_avx2,
            supports_avx512=supports_avx512
        )
        
        self.cpus.append(cpu)
    
    def _discover_gpus(self):
        """Discover GPU information"""
        # Initialize NVML for detailed GPU information if available
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
            except Exception as e:
                print(f"Warning: NVML initialization failed. Detailed GPU info unavailable. Error: {e}")
        
        # Get basic information via PyTorch CUDA API
        num_gpus = torch.cuda.device_count()
        
        for i in range(num_gpus):
            properties = torch.cuda.get_device_properties(i)
            
            # Get additional info from NVML if available
            pcie_gen = 0
            pcie_width = 0
            max_power = 0
            uuid = ""
            memory_bus_width = 0  # Default value if not available
            
            if PYNVML_AVAILABLE:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    pcie_info = pynvml.nvmlDeviceGetMaxPcieLinkGeneration(handle)
                    pcie_gen = pcie_info
                    pcie_width = pynvml.nvmlDeviceGetMaxPcieLinkWidth(handle)
                    max_power = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
                    uuid = pynvml.nvmlDeviceGetUUID(handle).decode('utf-8')
                    # Try to get memory bus width from NVML
                    try:
                        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        # Not directly available, would need to query through other means
                        # Just use a default based on GPU generation
                        memory_bus_width = 256  # Common value for many GPUs
                    except Exception:
                        pass
                except Exception:
                    pass
            
            # Create GPU info object
            gpu = GPUInfo(
                name=properties.name,
                index=i,
                uuid=uuid or f"GPU-{i}",
                memory_total=properties.total_memory,
                compute_capability=f"{properties.major}.{properties.minor}",
                core_count=properties.multi_processor_count,
                memory_bus_width=memory_bus_width,  # Use the default or NVML-provided value
                max_power_limit=max_power,
                supports_tensor_cores=properties.major >= 7,  # Tensor cores available from Volta (SM 7.0) onwards
                pcie_generation=pcie_gen,
                pcie_link_width=pcie_width
            )
            
            self.gpus.append(gpu)
        
        # Clean up NVML
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    
    def _save_to_cache(self):
        """Save device information to cache file"""
        cache_data = {
            'cpus': [asdict(cpu) for cpu in self.cpus],
            'gpus': [asdict(gpu) for gpu in self.gpus]
        }
        
        with open(self.cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
    
    def _load_from_cache(self):
        """Load device information from cache file"""
        with open(self.cache_file, 'r') as f:
            cache_data = json.load(f)
        
        self.cpus = [CPUInfo(**cpu_data) for cpu_data in cache_data.get('cpus', [])]
        self.gpus = [GPUInfo(**gpu_data) for gpu_data in cache_data.get('gpus', [])]
    
    def get_all_devices(self):
        """Get information about all available devices"""
        return {
            'cpus': self.cpus,
            'gpus': self.gpus
        }
    
    def get_cpu_info(self, index=0):
        """Get information about a specific CPU"""
        if index >= len(self.cpus):
            raise ValueError(f"CPU index {index} out of range (only {len(self.cpus)} CPUs available)")
        return self.cpus[index]
    
    def get_gpu_info(self, index):
        """Get information about a specific GPU"""
        if index >= len(self.gpus):
            raise ValueError(f"GPU index {index} out of range (only {len(self.gpus)} GPUs available)")
        return self.gpus[index]
    
    def get_device_count(self, device_type='all'):
        """
        Get the count of available devices.
        
        Args:
            device_type: 'all', 'cpu', or 'gpu'
        
        Returns:
            Number of devices of the specified type
        """
        if device_type == 'cpu':
            return len(self.cpus)
        elif device_type == 'gpu':
            return len(self.gpus)
        else:
            return len(self.cpus) + len(self.gpus)
