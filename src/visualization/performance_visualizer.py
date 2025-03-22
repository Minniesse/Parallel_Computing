import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import os

class PerformanceVisualizer:
    """
    Visualizes performance metrics for distributed model execution.
    Provides different views and charts to analyze performance.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the performance visualizer.
        
        Args:
            output_dir: Directory to save visualizations (None for interactive display)
        """
        self.output_dir = output_dir
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def plot_device_utilization(self, 
                               metrics: Dict[str, Any],
                               title: str = "Device Utilization",
                               use_plotly: bool = True):
        """
        Plot device utilization distribution.
        
        Args:
            metrics: Performance metrics from PerformanceMonitor
            title: Plot title
            use_plotly: Whether to use Plotly (interactive) or Matplotlib (static)
        """
        if 'device_utilization' not in metrics:
            print("No device utilization data available")
            return
            
        device_utils = metrics['device_utilization']
        devices = list(device_utils.keys())
        utilization = list(device_utils.values())
        
        if use_plotly:
            fig = px.bar(
                x=devices,
                y=utilization,
                labels={'x': 'Device', 'y': 'Utilization (%)'},
                title=title
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title="Device",
                yaxis_title="Utilization (%)",
                yaxis=dict(range=[0, 100])
            )
            
            # Save or display
            if self.output_dir:
                fig.write_html(os.path.join(self.output_dir, "device_utilization.html"))
            else:
                fig.show()
        else:
            plt.figure(figsize=(10, 6))
            plt.bar(devices, utilization)
            plt.title(title)
            plt.xlabel("Device")
            plt.ylabel("Utilization (%)")
            plt.ylim(0, 100)
            
            # Add values on top of bars
            for i, v in enumerate(utilization):
                plt.text(i, v + 2, f"{v:.1f}%", ha='center')
            
            # Save or display
            if self.output_dir:
                plt.savefig(os.path.join(self.output_dir, "device_utilization.png"))
                plt.close()
            else:
                plt.show()
    
    def plot_operation_distribution(self,
                                   metrics: Dict[str, Any],
                                   title: str = "Operation Distribution by Device",
                                   max_operations: int = 20,
                                   use_plotly: bool = True):
        """
        Plot distribution of operations across devices.
        
        Args:
            metrics: Performance metrics from PerformanceMonitor
            title: Plot title
            max_operations: Maximum number of operations to display
            use_plotly: Whether to use Plotly or Matplotlib
        """
        if 'operation_metrics' not in metrics:
            print("No operation metrics available")
            return
            
        op_metrics = metrics['operation_metrics']
        
        # Group operations by device
        device_ops = {}
        for op_id, op_data in op_metrics.items():
            device_id = op_data['device_id']
            if device_id not in device_ops:
                device_ops[device_id] = []
            
            device_ops[device_id].append({
                'op_id': op_id,
                'execution_time': op_data['avg_execution_time'],
                'memory_usage': op_data['avg_memory_usage']
            })
        
        # Sort operations by execution time within each device
        for device_id in device_ops:
            device_ops[device_id].sort(key=lambda x: x['execution_time'], reverse=True)
            device_ops[device_id] = device_ops[device_id][:max_operations]
        
        if use_plotly:
            # Create data for stacked bar chart
            data = []
            for device_id, ops in device_ops.items():
                for op in ops:
                    data.append({
                        'device': device_id,
                        'operation': op['op_id'],
                        'execution_time': op['execution_time'] * 1000,  # Convert to ms
                        'memory_mb': op['memory_usage'] / (1024 * 1024)  # Convert to MB
                    })
            
            df = pd.DataFrame(data)
            
            # Create grouped bar chart
            fig = px.bar(
                df,
                x='device',
                y='execution_time',
                color='operation',
                labels={'device': 'Device', 'execution_time': 'Execution Time (ms)'},
                title=title,
                hover_data=['memory_mb']
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title="Device",
                yaxis_title="Execution Time (ms)",
                legend_title="Operations"
            )
            
            # Save or display
            if self.output_dir:
                fig.write_html(os.path.join(self.output_dir, "operation_distribution.html"))
            else:
                fig.show()
        else:
            # Create grouped bar chart with Matplotlib
            device_ids = list(device_ops.keys())
            n_devices = len(device_ids)
            
            # Determine number of operations to display per device
            max_ops_per_device = max(len(ops) for ops in device_ops.values())
            max_ops_to_display = min(max_operations, max_ops_per_device)
            
            # Prepare data
            fig, ax = plt.subplots(figsize=(12, 8))
            
            bar_width = 0.8 / max_ops_to_display
            index = np.arange(n_devices)
            
            for i in range(max_ops_to_display):
                times = []
                labels = []
                
                for device_id in device_ids:
                    ops = device_ops.get(device_id, [])
                    if i < len(ops):
                        times.append(ops[i]['execution_time'] * 1000)  # Convert to ms
                        labels.append(ops[i]['op_id'])
                    else:
                        times.append(0)
                        labels.append('')
                
                ax.bar(index + i * bar_width, times, bar_width, label=f"Op {i+1}")
            
            ax.set_xlabel('Device')
            ax.set_ylabel('Execution Time (ms)')
            ax.set_title(title)
            ax.set_xticks(index + bar_width * (max_ops_to_display - 1) / 2)
            ax.set_xticklabels(device_ids)
            ax.legend()
            
            # Save or display
            if self.output_dir:
                plt.savefig(os.path.join(self.output_dir, "operation_distribution.png"))
                plt.close()
            else:
                plt.show()
    
    def plot_resource_utilization_timeline(self,
                                          historical_metrics: Dict[str, Any],
                                          title: str = "Resource Utilization Over Time",
                                          use_plotly: bool = True):
        """
        Plot timeline of resource utilization.
        
        Args:
            historical_metrics: Historical performance metrics
            title: Plot title
            use_plotly: Whether to use Plotly or Matplotlib
        """
        if 'timestamps' not in historical_metrics:
            print("No historical data available")
            return
            
        timestamps = historical_metrics['timestamps']
        
        if not timestamps:
            print("No timestamp data available")
            return
            
        # Convert timestamps to relative time (seconds since start)
        start_time = timestamps[0]
        relative_times = [t - start_time for t in timestamps]
        
        # Get CPU and memory data
        cpu_utils = historical_metrics.get('cpu_utilization', [])
        memory_utils = historical_metrics.get('memory_utilization', [])
        
        # Get GPU data if available
        gpu_utils = historical_metrics.get('gpu_utilization', {})
        gpu_memory = historical_metrics.get('gpu_memory', {})
        
        if use_plotly:
            # Create subplots
            n_rows = 2 + (1 if gpu_utils else 0)
            fig = make_subplots(
                rows=n_rows, cols=1,
                subplot_titles=["CPU Utilization", "Memory Utilization"] + 
                               (["GPU Utilization"] if gpu_utils else []),
                shared_xaxes=True,
                vertical_spacing=0.08
            )
            
            # Add CPU trace
            fig.add_trace(
                go.Scatter(
                    x=relative_times,
                    y=cpu_utils,
                    mode='lines',
                    name='CPU'
                ),
                row=1, col=1
            )
            
            # Add Memory trace
            fig.add_trace(
                go.Scatter(
                    x=relative_times,
                    y=memory_utils,
                    mode='lines',
                    name='Memory'
                ),
                row=2, col=1
            )
            
            # Add GPU traces if available
            if gpu_utils:
                for device_id, utils in gpu_utils.items():
                    if len(utils) == 0:
                        continue
                        
                    # Ensure same length as timestamps
                    device_times = relative_times[:len(utils)]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=device_times,
                            y=utils,
                            mode='lines',
                            name=f"{device_id} Util"
                        ),
                        row=3, col=1
                    )
                    
                    # Add GPU memory if available
                    if device_id in gpu_memory and gpu_memory[device_id]:
                        memory_vals = gpu_memory[device_id]
                        device_times = relative_times[:len(memory_vals)]
                        
                        fig.add_trace(
                            go.Scatter(
                                x=device_times,
                                y=memory_vals,
                                mode='lines',
                                name=f"{device_id} Mem",
                                line=dict(dash='dash')
                            ),
                            row=3, col=1
                        )
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis3_title="Time (seconds)",
                yaxis_title="CPU Utilization (%)",
                yaxis2_title="Memory Utilization (%)",
                yaxis3_title="GPU Utilization (%)" if gpu_utils else None,
                height=800 if gpu_utils else 600
            )
            
            # Set y-axis range to 0-100% for all plots
            fig.update_yaxes(range=[0, 100])
            
            # Save or display
            if self.output_dir:
                fig.write_html(os.path.join(self.output_dir, "resource_timeline.html"))
            else:
                fig.show()
        else:
            # Create Matplotlib figure
            n_plots = 2 + (1 if gpu_utils else 0)
            fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots), sharex=True)
            
            # Plot CPU utilization
            axes[0].plot(relative_times, cpu_utils)
            axes[0].set_ylabel("CPU Utilization (%)")
            axes[0].set_title("CPU Utilization")
            axes[0].grid(True)
            axes[0].set_ylim(0, 100)
            
            # Plot Memory utilization
            axes[1].plot(relative_times, memory_utils)
            axes[1].set_ylabel("Memory Utilization (%)")
            axes[1].set_title("Memory Utilization")
            axes[1].grid(True)
            axes[1].set_ylim(0, 100)
            
            # Plot GPU utilization if available
            if gpu_utils:
                for device_id, utils in gpu_utils.items():
                    if len(utils) == 0:
                        continue
                        
                    # Ensure same length as timestamps
                    device_times = relative_times[:len(utils)]
                    
                    axes[2].plot(device_times, utils, label=f"{device_id} Util")
                    
                    # Add GPU memory if available
                    if device_id in gpu_memory and gpu_memory[device_id]:
                        memory_vals = gpu_memory[device_id]
                        device_times = relative_times[:len(memory_vals)]
                        
                        axes[2].plot(device_times, memory_vals, '--', label=f"{device_id} Mem")
                
                axes[2].set_ylabel("GPU Utilization (%)")
                axes[2].set_title("GPU Utilization")
                axes[2].grid(True)
                axes[2].legend()
                axes[2].set_ylim(0, 100)
            
            # Set common x-axis label
            axes[-1].set_xlabel("Time (seconds)")
            
            # Set overall title
            fig.suptitle(title, fontsize=16)
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            
            # Save or display
            if self.output_dir:
                plt.savefig(os.path.join(self.output_dir, "resource_timeline.png"))
                plt.close()
            else:
                plt.show()
    
    def plot_execution_time_trends(self,
                                  historical_metrics: Dict[str, Any],
                                  title: str = "Execution Time Trends",
                                  use_plotly: bool = True):
        """
        Plot trends in execution time over successive forward passes.
        
        Args:
            historical_metrics: Historical performance metrics
            title: Plot title
            use_plotly: Whether to use Plotly or Matplotlib
        """
        if 'execution_times' not in historical_metrics:
            print("No execution time data available")
            return
            
        execution_times = historical_metrics['execution_times']
        
        if not execution_times:
            print("No execution time data points available")
            return
            
        # Create x-axis values (pass numbers)
        pass_numbers = list(range(1, len(execution_times) + 1))
        
        # Calculate moving average
        window_size = min(5, len(execution_times))
        moving_avg = []
        for i in range(len(execution_times)):
            if i < window_size - 1:
                # Not enough points for full window
                moving_avg.append(np.mean(execution_times[:i+1]))
            else:
                moving_avg.append(np.mean(execution_times[i-window_size+1:i+1]))
        
        if use_plotly:
            # Create Plotly figure
            fig = go.Figure()
            
            # Add raw execution times
            fig.add_trace(
                go.Scatter(
                    x=pass_numbers,
                    y=execution_times,
                    mode='markers',
                    name='Execution Time',
                    marker=dict(size=8)
                )
            )
            
            # Add moving average
            fig.add_trace(
                go.Scatter(
                    x=pass_numbers,
                    y=moving_avg,
                    mode='lines',
                    name=f'{window_size}-point Moving Average',
                    line=dict(width=3)
                )
            )
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title="Forward Pass Number",
                yaxis_title="Execution Time (seconds)",
                legend_title="Metrics"
            )
            
            # Save or display
            if self.output_dir:
                fig.write_html(os.path.join(self.output_dir, "execution_time_trends.html"))
            else:
                fig.show()
        else:
            # Create Matplotlib figure
            plt.figure(figsize=(12, 6))
            
            # Plot raw execution times
            plt.scatter(pass_numbers, execution_times, label='Execution Time', alpha=0.7)
            
            # Plot moving average
            plt.plot(pass_numbers, moving_avg, 'r-', linewidth=2, 
                     label=f'{window_size}-point Moving Average')
            
            plt.title(title)
            plt.xlabel("Forward Pass Number")
            plt.ylabel("Execution Time (seconds)")
            plt.grid(True)
            plt.legend()
            
            # Save or display
            if self.output_dir:
                plt.savefig(os.path.join(self.output_dir, "execution_time_trends.png"))
                plt.close()
            else:
                plt.show()
    
    def plot_strategy_comparison(self,
                               strategies: List[str],
                               metrics: List[Dict[str, Any]],
                               title: str = "Strategy Performance Comparison",
                               use_plotly: bool = True):
        """
        Compare performance of different distribution strategies.
        
        Args:
            strategies: List of strategy names
            metrics: List of performance metrics for each strategy
            title: Plot title
            use_plotly: Whether to use Plotly or Matplotlib
        """
        if not strategies or not metrics or len(strategies) != len(metrics):
            print("Invalid strategy comparison data")
            return
            
        # Extract execution times
        execution_times = [m.get('avg_execution_time', 0) * 1000 for m in metrics]  # ms
        
        # Extract resource utilization
        cpu_utils = [m.get('resource_utilization', {}).get('cpu', 0) for m in metrics]
        memory_utils = [m.get('resource_utilization', {}).get('memory', 0) for m in metrics]
        
        # Calculate GPU utilization averages
        gpu_utils = []
        for m in metrics:
            gpu_data = m.get('resource_utilization', {}).get('gpu', {})
            if gpu_data:
                # Average utilization across all GPUs
                avg_util = np.mean([data.get('utilization', 0) for data in gpu_data.values()])
                gpu_utils.append(avg_util)
            else:
                gpu_utils.append(0)
        
        if use_plotly:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=["Execution Time (ms)", "CPU Utilization (%)", 
                                "Memory Utilization (%)", "GPU Utilization (%)"],
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "bar"}]]
            )
            
            # Add execution time bars
            fig.add_trace(
                go.Bar(x=strategies, y=execution_times),
                row=1, col=1
            )
            
            # Add CPU utilization bars
            fig.add_trace(
                go.Bar(x=strategies, y=cpu_utils),
                row=1, col=2
            )
            
            # Add memory utilization bars
            fig.add_trace(
                go.Bar(x=strategies, y=memory_utils),
                row=2, col=1
            )
            
            # Add GPU utilization bars
            fig.add_trace(
                go.Bar(x=strategies, y=gpu_utils),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                title=title,
                showlegend=False,
                height=800
            )
            
            # Set y-axis range for utilization plots to 0-100%
            fig.update_yaxes(range=[0, 100], row=1, col=2)
            fig.update_yaxes(range=[0, 100], row=2, col=1)
            fig.update_yaxes(range=[0, 100], row=2, col=2)
            
            # Save or display
            if self.output_dir:
                fig.write_html(os.path.join(self.output_dir, "strategy_comparison.html"))
            else:
                fig.show()
        else:
            # Create Matplotlib figure
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot execution time
            axes[0, 0].bar(strategies, execution_times)
            axes[0, 0].set_title("Execution Time (ms)")
            axes[0, 0].set_ylabel("Time (ms)")
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(axis='y')
            
            # Plot CPU utilization
            axes[0, 1].bar(strategies, cpu_utils)
            axes[0, 1].set_title("CPU Utilization (%)")
            axes[0, 1].set_ylabel("Utilization (%)")
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].set_ylim(0, 100)
            axes[0, 1].grid(axis='y')
            
            # Plot memory utilization
            axes[1, 0].bar(strategies, memory_utils)
            axes[1, 0].set_title("Memory Utilization (%)")
            axes[1, 0].set_ylabel("Utilization (%)")
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].set_ylim(0, 100)
            axes[1, 0].grid(axis='y')
            
            # Plot GPU utilization
            axes[1, 1].bar(strategies, gpu_utils)
            axes[1, 1].set_title("GPU Utilization (%)")
            axes[1, 1].set_ylabel("Utilization (%)")
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].set_ylim(0, 100)
            axes[1, 1].grid(axis='y')
            
            # Set overall title
            fig.suptitle(title, fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            # Save or display
            if self.output_dir:
                plt.savefig(os.path.join(self.output_dir, "strategy_comparison.png"))
                plt.close()
            else:
                plt.show()
