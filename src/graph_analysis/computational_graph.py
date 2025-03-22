import torch
import torch.fx as fx
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional

class ComputationalGraph:
    """
    Represents the computational graph of a PyTorch model for parallelism analysis.
    Extends PyTorch FX capabilities with specialized analysis for parallelization.
    """
    
    def __init__(self, model: torch.nn.Module, example_inputs: Any):
        """
        Initialize a computational graph from a PyTorch model.
        
        Args:
            model: PyTorch model to analyze
            example_inputs: Example input tensors for the model (used for tracing)
        """
        self.model = model
        self.example_inputs = example_inputs
        self.fx_graph = None
        self.nx_graph = None
        self.node_properties = {}
        self._extract_graph()
        self._analyze_graph()
    
    def _extract_graph(self):
        """Extract the computational graph using PyTorch FX"""
        try:
            # Create a symbolic trace of the model
            traced_model = fx.symbolic_trace(self.model)
            self.fx_graph = traced_model.graph
        except Exception as e:
            # Fallback for models with dynamic control flow
            print(f"Warning: Symbolic tracing failed, using dynamic tracing. Error: {e}")
            self._extract_graph_with_dynamic_tracing()
    
    def _extract_graph_with_dynamic_tracing(self):
        """Extract graph for models with dynamic control flow using custom tracing"""
        # Create a manual graph representation for models that can't be traced with FX
        manual_graph = fx.Graph()
        
        # Special handling for HuggingFace transformer models
        if hasattr(self.model, 'config') and hasattr(self.model, 'transformer'):
            print("Detected HuggingFace transformer model, using specialized tracing")
            self._extract_huggingface_transformer_graph(manual_graph)
        else:
            # Generic approach for other models with dynamic control flow
            self._extract_dynamic_model_graph(manual_graph)
            
        self.fx_graph = manual_graph
    
    def _extract_huggingface_transformer_graph(self, graph):
        """Extract graph specifically for HuggingFace transformer models"""
        # Add placeholder input node
        input_node = graph.placeholder("input", None)
        
        # Add transformer layers as modules
        if hasattr(self.model, 'transformer'):
            # Extract main transformer components
            transformer_node = graph.call_module("transformer", (input_node,))
            
            # Add common transformer components
            if hasattr(self.model, 'pre_classifier'):
                pre_classifier_node = graph.call_module("pre_classifier", (transformer_node,))
                current_node = pre_classifier_node
            else:
                current_node = transformer_node
                
            if hasattr(self.model, 'classifier'):
                output_node = graph.call_module("classifier", (current_node,))
            else:
                output_node = current_node
                
            # Add output node
            graph.output(output_node)
            
    def _extract_dynamic_model_graph(self, graph):
        """Extract graph for generic models with dynamic control flow"""
        # Create a simple placeholder representation of the model structure
        input_node = graph.placeholder("input", None)
        
        # Create nodes for all top-level modules
        current_node = input_node
        for name, _ in self.model.named_children():
            current_node = graph.call_module(name, (current_node,))
            
        # Add output node
        graph.output(current_node)
    
    def _analyze_graph(self):
        """Analyze the computational graph properties relevant for parallelization"""
        if not self.fx_graph:
            raise ValueError("Graph extraction failed. Unable to analyze.")
        
        # Convert FX graph to NetworkX for analysis
        self.nx_graph = nx.DiGraph()
        
        # Add all nodes
        for node in self.fx_graph.nodes:
            self.nx_graph.add_node(node.name, op_type=node.op, target=str(node.target))
            
            # Calculate node properties
            self.node_properties[node.name] = {
                'op_type': node.op,
                'parallelizable': self._is_node_parallelizable(node),
                'estimated_flops': self._estimate_computation(node),
                'memory_footprint': self._estimate_memory_footprint(node),
                'dependencies': []
            }
        
        # Add all edges
        for node in self.fx_graph.nodes:
            for input_node in node.all_input_nodes:
                self.nx_graph.add_edge(input_node.name, node.name)
                self.node_properties[node.name]['dependencies'].append(input_node.name)
    
    def _is_node_parallelizable(self, node):
        """Determine if a node's operation can be parallelized"""
        # Basic heuristic: consider conventional parallelizable operations
        parallelizable_ops = {'call_module', 'call_method'}
        parallelizable_targets = {'conv', 'linear', 'matmul', 'bmm'}
        
        if node.op in parallelizable_ops:
            target_name = str(node.target)
            return any(target in target_name.lower() for target in parallelizable_targets)
        return False
    
    def _estimate_computation(self, node):
        """Estimate computational requirements (in seconds) for a node.
        
        Updated to return realistic, fixed values.
        """
        if node.op == 'call_module':
            target_str = str(node.target).lower()
            if 'conv' in target_str:
                return 0.01  # 10 ms for convolution layers
            elif 'linear' in target_str:
                return 0.005  # 5 ms for linear layers
            else:
                return 0.001  # 1 ms for other module calls
        elif node.op in ['call_method', 'call_function']:
            return 0.001  # 1 ms for method/function calls
        else:
            return 0.0001  # default small value
    
    def _estimate_memory_footprint(self, node):
        """Estimate memory requirements of a node"""
        # This would contain detailed logic to estimate memory requirements
        # based on tensor shapes, operation type, etc.
        # Using a more reasonable placeholder value
        if node.op == 'call_module':
            if 'conv' in str(node.target).lower():
                return 400000  # Convolutional layers typically use more memory
            elif 'linear' in str(node.target).lower():
                return 200000  # Linear layers use moderate memory
            else:
                return 100000  # Other module calls
        elif node.op == 'call_method' or node.op == 'call_function':
            return 50000       # Method calls
        else:
            return 10000       # Other operations
    
    def identify_parallelizable_components(self) -> List[List[str]]:
        """
        Identify groups of operations that can be executed in parallel.
        
        Returns:
            List of lists, where each inner list contains node names that can run in parallel
        """
        # Find connected components that can be parallelized
        components = []
        visited = set()
        
        # Topologically sort nodes to respect dependencies
        sorted_nodes = list(nx.topological_sort(self.nx_graph))
        
        for node_name in sorted_nodes:
            if node_name in visited:
                continue
                
            # If node is parallelizable, find other nodes that can run with it
            if self.node_properties[node_name]['parallelizable']:
                component = [node_name]
                
                # Find other parallelizable nodes at same level
                for other_node in sorted_nodes:
                    if other_node in visited or other_node == node_name:
                        continue
                        
                    if self.node_properties[other_node]['parallelizable']:
                        # Check if no path exists between these nodes
                        if not nx.has_path(self.nx_graph, node_name, other_node) and \
                           not nx.has_path(self.nx_graph, other_node, node_name):
                            component.append(other_node)
                
                components.append(component)
                visited.update(component)
            else:
                visited.add(node_name)
                components.append([node_name])
                
        return components
    
    def visualize(self, highlight_components=False, save_path=None):
        """
        Visualize the computational graph.
        
        Args:
            highlight_components: If True, color nodes by their parallelizable components
            save_path: Path to save the visualization (if None, display interactively)
        """
        if not self.nx_graph:
            raise ValueError("Graph not available for visualization")
            
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.nx_graph)
        
        # Default node colors
        node_colors = ['lightblue' for _ in self.nx_graph.nodes()]
        
        if highlight_components:
            components = self.identify_parallelizable_components()
            color_map = plt.cm.get_cmap('tab10', len(components))
            
            for i, component in enumerate(components):
                for node in component:
                    node_idx = list(self.nx_graph.nodes()).index(node)
                    node_colors[node_idx] = color_map(i)
        
        nx.draw(self.nx_graph, pos, with_labels=True, node_color=node_colors, 
                node_size=500, font_size=10, arrows=True)
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def get_node_properties(self):
        """Get the properties of all nodes in the graph"""
        return self.node_properties
    
    def get_node_dependencies(self, node_name):
        """Get dependencies for a specific node"""
        if node_name not in self.node_properties:
            raise ValueError(f"Node {node_name} not found in graph")
        return self.node_properties[node_name]['dependencies']
