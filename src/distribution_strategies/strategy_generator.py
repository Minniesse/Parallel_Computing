import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import torch

from ..graph_analysis.computational_graph import ComputationalGraph
from ..hardware_profiling.device_catalog import DeviceCatalog
from ..hardware_profiling.hardware_profiler import HardwareProfiler

@dataclass
class DeviceAssignment:
    device_id: str  # Format: "cpu:0" or "cuda:0"
    operation_ids: List[str]
    estimated_compute_time: float
    estimated_memory_usage: int
    energy_consumption: float

@dataclass
class DistributionStrategy:
    device_assignments: List[DeviceAssignment]
    communication_cost: float
    estimated_total_time: float
    estimated_energy: float
    memory_peak: Dict[str, int]  # Peak memory usage per device
    
    def to_dict(self):
        """Convert strategy to dictionary for serialization"""
        return {
            'device_assignments': [vars(da) for da in self.device_assignments],
            'communication_cost': self.communication_cost,
            'estimated_total_time': self.estimated_total_time,
            'estimated_energy': self.estimated_energy,
            'memory_peak': self.memory_peak
        }

class StrategyGenerator:
    """
    Generates optimal distribution strategies for partitioning computational
    graphs across available devices.
    """
    
    def __init__(
        self, 
        computational_graph: ComputationalGraph,
        device_catalog: DeviceCatalog,
        hardware_profiler: HardwareProfiler,
        energy_aware: bool = False,
        communication_aware: bool = True
    ):
        """
        Initialize the strategy generator.
        
        Args:
            computational_graph: The computational graph to distribute
            device_catalog: Catalog of available devices
            hardware_profiler: Profiler with performance metrics for operations
            energy_aware: Whether to consider energy efficiency in strategy generation
            communication_aware: Whether to optimize for communication overhead
        """
        self.graph = computational_graph
        self.device_catalog = device_catalog
        self.profiler = hardware_profiler
        self.energy_aware = energy_aware
        self.communication_aware = communication_aware
        
        self.available_devices = self._get_available_devices()
        self.node_properties = self.graph.get_node_properties()
    
    def _get_available_devices(self) -> List[str]:
        """Get list of available device identifiers"""
        devices = []
        
        # Add CPU devices
        for i in range(self.device_catalog.get_device_count('cpu')):
            devices.append(f"cpu:{i}")
        
        # Add GPU devices
        for i in range(self.device_catalog.get_device_count('gpu')):
            devices.append(f"cuda:{i}")
            
        return devices
    
    def generate_strategy(self) -> DistributionStrategy:
        """
        Generate the optimal distribution strategy based on the computational graph
        and available hardware.
        
        Returns:
            A DistributionStrategy object with device assignments
        """
        if not self.available_devices:
            raise ValueError("No devices available for strategy generation")
            
        # Different approaches to strategy generation
        if len(self.available_devices) == 1:
            # Single device - no distribution needed
            return self._single_device_strategy()
        elif not self.communication_aware:
            # Basic load balancing without considering communication
            return self._generate_load_balanced_strategy()
        else:
            # Communication-aware strategy with advanced partitioning
            return self._generate_communication_aware_strategy()
    
    def _single_device_strategy(self) -> DistributionStrategy:
        """Generate strategy for single device execution"""
        device = self.available_devices[0]
        all_nodes = list(self.node_properties.keys())
        
        # Estimate execution characteristics
        compute_time = sum(props['estimated_flops'] for props in self.node_properties.values())
        memory_usage = sum(props['memory_footprint'] for props in self.node_properties.values())
        
        # Get energy estimates if enabled
        energy = 0.0
        if self.energy_aware:
            # Calculate energy based on profiled operation costs
            energy = self.profiler.estimate_energy_consumption(all_nodes, device)
        
        # Create device assignment
        assignment = DeviceAssignment(
            device_id=device,
            operation_ids=all_nodes,
            estimated_compute_time=compute_time,
            estimated_memory_usage=memory_usage,
            energy_consumption=energy
        )
        
        # Create strategy with the single device assignment
        return DistributionStrategy(
            device_assignments=[assignment],
            communication_cost=0.0,  # No communication with single device
            estimated_total_time=compute_time,
            estimated_energy=energy,
            memory_peak={device: memory_usage}
        )
    
    def _generate_load_balanced_strategy(self) -> DistributionStrategy:
        """Generate a basic load balanced strategy without optimizing communication"""
        # Get all parallelizable components
        components = self.graph.identify_parallelizable_components()
        
        # Calculate computation cost for each component
        component_costs = []
        for component in components:
            # Use more reasonable cost estimates
            cost = min(sum(self.node_properties[node]['estimated_flops'] for node in component) / 1e6, 1.0)
            memory = sum(self.node_properties[node]['memory_footprint'] for node in component)
            component_costs.append((component, cost, memory))
        
        # Sort components by computational cost (descending)
        component_costs.sort(key=lambda x: x[1], reverse=True)
        
        # Initialize device assignments
        device_assignments = {device: [] for device in self.available_devices}
        device_compute_times = {device: 0.0 for device in self.available_devices}
        device_memory_usage = {device: 0 for device in self.available_devices}
        
        # Assign components to devices using a greedy approach
        for component, cost, memory in component_costs:
            # Find device with minimum load
            target_device = min(device_compute_times, key=device_compute_times.get)
            
            # Assign component to device
            device_assignments[target_device].extend(component)
            device_compute_times[target_device] += cost
            device_memory_usage[target_device] += memory
        
        # Calculate communication cost
        communication_cost = self._calculate_communication_cost(device_assignments)
        
        # Calculate energy consumption if energy-aware
        device_energy = {device: 0.0 for device in self.available_devices}
        if self.energy_aware:
            for device, nodes in device_assignments.items():
                if nodes:  # Only calculate for devices with assigned operations
                    device_energy[device] = self.profiler.estimate_energy_consumption(nodes, device)
        
        # Create DeviceAssignment objects
        assignments = []
        for device, nodes in device_assignments.items():
            if nodes:  # Only create assignments for devices with operations
                assignment = DeviceAssignment(
                    device_id=device,
                    operation_ids=nodes,
                    estimated_compute_time=device_compute_times[device],
                    estimated_memory_usage=device_memory_usage[device],
                    energy_consumption=device_energy[device]
                )
                assignments.append(assignment)
        
        # Calculate total strategy metrics
        total_time = max(device_compute_times.values()) + communication_cost
        total_energy = sum(device_energy.values())
        
        return DistributionStrategy(
            device_assignments=assignments,
            communication_cost=communication_cost,
            estimated_total_time=total_time,
            estimated_energy=total_energy,
            memory_peak=device_memory_usage
        )
    
    def _generate_communication_aware_strategy(self) -> DistributionStrategy:
        """
        Generate an optimized strategy that minimizes communication overhead
        while balancing computation.
        """
        # Create a NetworkX graph from the computational graph for partitioning
        G = nx.DiGraph()
        
        # Add nodes with computation weights
        for node, props in self.node_properties.items():
            G.add_node(node, weight=props['estimated_flops'], memory=props['memory_footprint'])
        
        # Add edges with communication weights
        for node, props in self.node_properties.items():
            for dep in props['dependencies']:
                # Estimate communication cost based on memory requirements
                comm_cost = props['memory_footprint']
                G.add_edge(dep, node, weight=comm_cost)
        
        # Number of partitions equals number of available devices
        num_partitions = min(len(self.available_devices), 
                            len([n for n in G.nodes() if self.node_properties[n]['parallelizable']]))
        
        if num_partitions <= 1:
            # If we can't parallelize, fall back to single device strategy
            return self._single_device_strategy()
        
        # Use METIS algorithm for graph partitioning if available
        try:
            import metis
            # Convert NetworkX graph to format expected by METIS
            adjacency = {i: list(G.neighbors(node)) for i, node in enumerate(G.nodes())}
            node_weights = [G.nodes[node]['weight'] for node in G.nodes()]
            edge_weights = [G.edges[edge]['weight'] for edge in G.edges()]
            
            # Call METIS partitioning
            _, partitions = metis.part_graph(adjacency, nparts=num_partitions, 
                                            node_weights=node_weights,
                                            edge_weights=edge_weights)
            
            # Convert partition result to our format
            partition_map = {node: partitions[i] for i, node in enumerate(G.nodes())}
        
        except ImportError:
            # Fall back to spectral clustering if METIS is not available
            print("METIS not available, falling back to spectral clustering")
            partition_map = self._spectral_clustering_partition(G, num_partitions)
        
        # Create device assignments from partitions
        device_nodes = {i: [] for i in range(num_partitions)}
        for node, partition in partition_map.items():
            device_nodes[partition].append(node)
        
        # Map partition indices to actual devices
        device_assignments = {}
        for i, device in enumerate(self.available_devices[:num_partitions]):
            device_assignments[device] = device_nodes[i]
        
        # Calculate metrics for each assignment
        assignments = []
        device_compute_times = {}
        device_memory_usage = {}
        device_energy = {}
        
        for device, nodes in device_assignments.items():
            if not nodes:
                continue
                
            compute_time = sum(self.node_properties[node]['estimated_flops'] for node in nodes)
            memory_usage = sum(self.node_properties[node]['memory_footprint'] for node in nodes)
            
            energy = 0.0
            if self.energy_aware:
                energy = self.profiler.estimate_energy_consumption(nodes, device)
            
            assignments.append(DeviceAssignment(
                device_id=device,
                operation_ids=nodes,
                estimated_compute_time=compute_time,
                estimated_memory_usage=memory_usage,
                energy_consumption=energy
            ))
            
            device_compute_times[device] = compute_time
            device_memory_usage[device] = memory_usage
            device_energy[device] = energy
        
        # Calculate communication cost
        communication_cost = self._calculate_communication_cost(device_assignments)
        
        # Calculate total strategy metrics
        total_time = max(device_compute_times.values()) + communication_cost
        total_energy = sum(device_energy.values())
        
        return DistributionStrategy(
            device_assignments=assignments,
            communication_cost=communication_cost,
            estimated_total_time=total_time,
            estimated_energy=total_energy,
            memory_peak=device_memory_usage
        )
    
    def _spectral_clustering_partition(self, G, num_partitions):
        """
        Partition a graph using spectral clustering when METIS is not available.
        
        Args:
            G: NetworkX graph to partition
            num_partitions: Number of partitions to create
            
        Returns:
            Dictionary mapping node names to partition indices
        """
        # Create Laplacian matrix
        laplacian = nx.normalized_laplacian_matrix(G.to_undirected())
        
        # Get eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian.toarray())
        
        # Use the eigenvectors corresponding to the k smallest eigenvalues
        indices = np.argsort(eigenvalues)[1:num_partitions]  # Skip the smallest eigenvalue (0)
        features = eigenvectors[:, indices]
        
        # Apply k-means clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=num_partitions, random_state=0).fit(features)
        
        # Create partition map
        nodes = list(G.nodes())
        return {nodes[i]: kmeans.labels_[i] for i in range(len(nodes))}
    
    def _calculate_communication_cost(self, device_assignments):
        """
        Calculate the communication cost between devices based on the assignments.
        
        Args:
            device_assignments: Dictionary mapping device IDs to lists of operation IDs
            
        Returns:
            Estimated communication cost
        """
        # Create reverse mapping from operations to devices
        op_to_device = {}
        for device, ops in device_assignments.items():
            for op in ops:
                op_to_device[op] = device
        
        total_cost = 0.0
        
        # Check each edge that crosses device boundaries
        for node, props in self.node_properties.items():
            if node not in op_to_device:
                continue
                
            node_device = op_to_device[node]
            
            for dep in props['dependencies']:
                if dep not in op_to_device:
                    continue
                    
                dep_device = op_to_device[dep]
                
                if node_device != dep_device:
                    # This is a cross-device edge, add communication cost
                    # Cost is based on memory footprint and transfer speed between devices
                    transfer_cost = self.profiler.estimate_transfer_cost(
                        dep, node, dep_device, node_device
                    )
                    total_cost += transfer_cost
        
        return total_cost

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
        # Use more reasonable default estimates (significantly reduced from original values)
        if device_type == 'cuda':
            return flops / (5e9)  # Assume 5 TFLOPS for GPU
        else:
            return flops / (5e8)  # Assume 500 GFLOPS for CPU
