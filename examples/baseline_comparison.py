import os
import sys
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from example_model import ExampleModel, LargeModel, TransformerModel
from src.parallel_opt import OptimizedParallel
from src.visualization.performance_visualizer import PerformanceVisualizer
from src.hardware_profiling.device_catalog import DeviceCatalog

def dtype_size(dtype):
    """Get the size in bytes of a data type"""
    if dtype in [torch.float32, torch.int32]:
        return 4
    elif dtype in [torch.float16, torch.int16]:
        return 2
    elif dtype in [torch.int8, torch.uint8]:
        return 1
    elif dtype in [torch.float64, torch.int64]:
        return 8
    else:
        return 4  # Default

def estimate_model_memory(model, batch_size, input_shape, dtype=torch.float32):
    """
    Estimate the memory required by a model during training.
    
    Args:
        model: PyTorch model
        batch_size: Batch size
        input_shape: Shape of a single input (without batch dimension)
        dtype: Data type of model parameters and activations
        
    Returns:
        Dictionary with memory estimates in bytes
    """
    # Calculate model parameters memory
    param_bytes = sum(p.numel() * dtype_size(p.dtype) for p in model.parameters())
    
    # Estimate forward activations memory (this is an approximation)
    # A common heuristic is that activations use 2-3x the memory of parameters
    # Scale with batch size relative to a reference batch size of 32
    activations_bytes = param_bytes * 2.5 * (batch_size / 32)
    
    # Estimate backward gradients memory
    gradient_bytes = param_bytes  # Gradients are same size as parameters
    
    # Estimate optimizer states (Adam uses 2 additional states per parameter)
    optimizer_bytes = param_bytes * 2  # For Adam optimizer
    
    # Input size
    input_size = np.prod(input_shape)
    input_bytes = batch_size * input_size * dtype_size(dtype)
    
    # Calculate total memory
    total_bytes = param_bytes + activations_bytes + gradient_bytes + optimizer_bytes + input_bytes
    
    # Add some overhead for PyTorch internals and fragmentation
    total_bytes = total_bytes * 1.2  # 20% overhead
    
    return {
        'parameters': param_bytes,
        'activations': activations_bytes,
        'gradients': gradient_bytes,
        'optimizer': optimizer_bytes,
        'input': input_bytes,
        'total': total_bytes
    }

def calculate_max_batch_size(model, input_shape, available_memory, min_batch=1, max_batch=1024):
    """
    Calculate the maximum batch size that would fit in GPU memory.
    
    Args:
        model: PyTorch model
        input_shape: Shape of a single input (without batch dimension)
        available_memory: Available GPU memory in bytes
        min_batch: Minimum batch size to consider
        max_batch: Maximum batch size to consider
        
    Returns:
        Maximum batch size that fits in memory or 0 if even min_batch doesn't fit
    """
    # Check if even the minimum batch size fits
    min_mem = estimate_model_memory(model, min_batch, input_shape)
    if min_mem['total'] > available_memory:
        return 0  # Even minimum batch size doesn't fit
    
    # Binary search to find the maximum batch size
    left, right = min_batch, max_batch
    max_feasible = min_batch
    
    while left <= right:
        mid = (left + right) // 2
        memory_est = estimate_model_memory(model, mid, input_shape)
        
        if memory_est['total'] <= available_memory:
            max_feasible = mid
            left = mid + 1
        else:
            right = mid - 1
    
    return max_feasible

def load_huggingface_transformer(model_name="distilbert-base-uncased", num_labels=2):
    """Load a pre-trained transformer model from HuggingFace"""
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer
    except Exception as e:
        print(f"Error loading HuggingFace model: {e}")
        raise

def run_baseline(model, input_data, warmup=5, iterations=20, is_huggingface=False):
    """Run baseline PyTorch model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Handle HuggingFace input differently
    if is_huggingface:
        for key in input_data:
            if isinstance(input_data[key], torch.Tensor):
                input_data[key] = input_data[key].to(device)
    else:
        input_data = input_data.to(device)
    
    # Warmup
    print(f"Running baseline warmup ({warmup} iterations)...")
    with torch.no_grad():
        for _ in range(warmup):
            if is_huggingface:
                _ = model(**input_data)
            else:
                _ = model(input_data)
    
    # Benchmark
    print(f"Running baseline benchmark ({iterations} iterations)...")
    times = []
    with torch.no_grad():
        for i in range(iterations):
            start = time.time()
            if is_huggingface:
                _ = model(**input_data)
            else:
                _ = model(input_data)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.time()
            times.append((end - start) * 1000)  # ms
    
    # Calculate memory usage if possible
    memory_usage = None
    if device.type == 'cuda':
        try:
            torch.cuda.reset_peak_memory_stats(device)
            if is_huggingface:
                _ = model(**input_data)
            else:
                _ = model(input_data)
            memory_usage = torch.cuda.max_memory_allocated(device) / (1024 * 1024)  # MB
        except:
            pass
    
    return {
        'times': times,
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'memory_usage': memory_usage
    }

def run_optimized(model, input_data, warmup=5, iterations=20, is_huggingface=False):
    """Run optimized model"""
    print(f"Initializing optimized model...")
    optimized_model = OptimizedParallel(
        model, 
        energy_aware=True, 
        cache_dir="./cache"
    )
    
    # First run will analyze and optimize
    print(f"Running optimization and warmup ({warmup} iterations)...")
    if is_huggingface:
        _ = optimized_model(**input_data)  # Initial optimization
        
        for _ in range(warmup-1):
            _ = optimized_model(**input_data)
        
        # Benchmark
        print(f"Running optimized benchmark ({iterations} iterations)...")
        times = []
        for i in range(iterations):
            start = time.time()
            _ = optimized_model(**input_data)
            end = time.time()
            times.append((end - start) * 1000)  # ms
    else:
        _ = optimized_model(input_data)  # Initial optimization
        
        for _ in range(warmup-1):
            _ = optimized_model(input_data)
        
        # Benchmark
        print(f"Running optimized benchmark ({iterations} iterations)...")
        times = []
        for i in range(iterations):
            start = time.time()
            _ = optimized_model(input_data)
            end = time.time()
            times.append((end - start) * 1000)  # ms
    
    metrics = optimized_model.get_performance_metrics()
    
    return {
        'times': times,
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'metrics': metrics,
        'optimized_model': optimized_model
    }

def compare_models(models, batch_sizes, warmup=5, iterations=20):
    """Compare baseline vs optimized for different models and batch sizes"""
    results = {}
    memory_analysis_results = {}  # Store memory analysis for summary report
    
    # ANSI color codes for terminal output
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    # Load device catalog to get GPU information - force fresh detection
    # First, try to remove any cached device information
    cache_path = "./cache/device_info.json"
    if os.path.exists(cache_path):
        try:
            os.remove(cache_path)
            print("Cleared cached device information to ensure fresh detection")
        except:
            print("Warning: Unable to clear cached device information")
    
    # Create device catalog with fresh detection
    device_catalog = DeviceCatalog(cache_file=cache_path)
    
    for model_name, model_info in models.items():
        results[model_name] = {}
        model_class = model_info.get('class')
        is_huggingface = model_info.get('is_huggingface', False)
        hf_model_name = model_info.get('hf_model_name', None)
        tokenizer = model_info.get('tokenizer', None)
        
        # Create model instance for memory calculations
        if is_huggingface:
            model = model_class
        else:
            model = model_class()
        
        print(f"\n===============================================")
        print(f"Model: {model_name}")
        print(f"===============================================")
        
        # Determine input shape for memory calculations
        if is_huggingface and tokenizer:
            sample_text = ["This is a sample text for testing."]
            encoded = tokenizer(sample_text, padding=True, truncation=True, return_tensors="pt")
            # Get shape of first tensor in the dictionary
            first_key = next(iter(encoded))
            input_shape = tuple(encoded[first_key].shape[1:])
            input_type = "text"
        elif model_name in ['Transformer', 'TransformerModel']:
            input_shape = (64,)  # Sequence length
            input_type = "sequence"
        else:
            input_shape = (3, 32, 32)  # Channels, height, width for image data
            input_type = "image"
        
        # Calculate memory requirements and recommend batch sizes
        print(f"\nMemory Analysis for {model_name}:")
        
        # Get GPU memory
        available_gpu_memory = 0
        gpu_name = "No GPU"
        if torch.cuda.is_available() and device_catalog.get_device_count('gpu') > 0:
            gpu_info = device_catalog.get_gpu_info(0)
            available_gpu_memory = gpu_info.memory_total
            gpu_name = gpu_info.name
            
            # Convert bytes to GB for display
            available_gb = available_gpu_memory / (1024 ** 3)
            print(f"GPU: {gpu_name} with {available_gb:.2f} GB VRAM")
        else:
            print("No GPU detected, memory analysis will be limited")
        
        # Calculate parameter count
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {param_count:,}")
        
        # Analyze memory requirements for different batch sizes
        print("\nEstimated memory requirements:")
        print(f"{'Batch Size':<10} {'Params (MB)':<12} {'Activations (MB)':<16} {'Total (GB)':<10} {'Status':<10}")
        
        memory_limited_batch = float('inf')
        memory_requirements = {}  # Store memory requirements for each batch size
        
        for bs in batch_sizes:
            # Skip batch size analysis if no GPU
            if available_gpu_memory == 0:
                continue
                
            mem_est = estimate_model_memory(model, bs, input_shape)
            
            param_mb = mem_est['parameters'] / (1024 ** 2)
            act_mb = mem_est['activations'] / (1024 ** 2)
            total_gb = mem_est['total'] / (1024 ** 3)
            
            status = "OK"
            color = GREEN
            if mem_est['total'] > available_gpu_memory:
                status = "OOM"
                color = RED
                memory_limited_batch = min(memory_limited_batch, bs)
            elif mem_est['total'] > available_gpu_memory * 0.9:
                status = "RISK-OOM"
                color = YELLOW
                memory_limited_batch = min(memory_limited_batch, bs)
            elif mem_est['total'] > available_gpu_memory * 0.8:
                status = "CAUTION"
                color = YELLOW
            
            memory_requirements[bs] = {
                'param_mb': param_mb,
                'act_mb': act_mb,
                'total_gb': total_gb,
                'status': status,
                'color': color
            }
            
            print(f"{bs:<10} {param_mb:<12.2f} {act_mb:<16.2f} {total_gb:<10.2f} {color}{status}{RESET}")
        
        # Store memory analysis information for summary report
        max_bs = 0
        memory_tips = []
        
        # Calculate recommended batch size
        if available_gpu_memory > 0:
            max_bs = calculate_max_batch_size(
                model, 
                input_shape, 
                available_gpu_memory * 0.9,  # Use 90% of available memory
                min_batch=1,
                max_batch=1024
            )
            
            if max_bs == 0:
                print("\n⚠️ WARNING: This model is too large to fit even batch size 1 in GPU memory!")
                memory_tips.append("- Model too large to fit in GPU memory with batch size 1")
                memory_tips.append("- Use model parallelism or pipeline parallelism to split the model across multiple GPUs")
                memory_tips.append("- Try mixed precision training (torch.cuda.amp) to reduce memory by almost 50%")
                memory_tips.append("- Consider using a smaller model or model pruning techniques")
                memory_tips.append("- Use gradient checkpointing to trade computation for memory")
            else:
                print(f"\nRecommended maximum batch size: {max_bs}")
                
                # Print memory optimization tips
                print("\nMemory optimization tips:")
                if max_bs < min(batch_sizes):
                    print(f"- ⚠️ Requested batch sizes exceed GPU memory capacity")
                    memory_tips.append(f"- Requested batch sizes exceed GPU memory capacity")
                    print(f"- Consider using gradient accumulation instead of large batch sizes")
                    memory_tips.append(f"- Consider using gradient accumulation instead of large batch sizes")
                    print(f"- Try mixed precision training (torch.cuda.amp) to reduce memory usage by ~50%")
                    memory_tips.append(f"- Try mixed precision training (torch.cuda.amp) to reduce memory usage by ~50%")
                    print(f"- Model checkpoint/activation recomputation can save memory at the cost of computation")
                    memory_tips.append(f"- Model checkpoint/activation recomputation can save memory at the cost of computation")
                elif memory_limited_batch < float('inf'):
                    print(f"- ⚠️ Batch size {memory_limited_batch} and above may cause OOM errors")
                    memory_tips.append(f"- Batch size {memory_limited_batch} and above may cause OOM errors")
                    print(f"- Consider using gradient accumulation for effective batch sizes > {max_bs}")
                    memory_tips.append(f"- Consider using gradient accumulation for effective batch sizes > {max_bs}")
                    print(f"- For {model_name}, mixed precision training can increase max batch size to ~{int(max_bs * 1.7)}")
                    memory_tips.append(f"- For {model_name}, mixed precision training can increase max batch size to ~{int(max_bs * 1.7)}")
                else:
                    print(f"- All requested batch sizes should fit within GPU memory")
                    memory_tips.append(f"- All requested batch sizes should fit within GPU memory")
                    print(f"- For large-scale training, consider gradient accumulation or distributed training")
                    memory_tips.append(f"- For large-scale training, consider gradient accumulation or distributed training")
        
        # Store memory analysis results
        memory_analysis_results[model_name] = {
            'gpu_name': gpu_name,
            'gpu_vram_gb': available_gpu_memory / (1024 ** 3) if available_gpu_memory > 0 else 0,
            'param_count': param_count,
            'max_batch_size': max_bs,
            'memory_requirements': memory_requirements,
            'memory_tips': memory_tips
        }

        # Run benchmarks for each batch size
        for batch_size in batch_sizes:
            # Add warning if batch size likely to cause OOM
            if available_gpu_memory > 0 and batch_size >= memory_limited_batch:
                print(f"\n⚠️ WARNING: Batch size {batch_size} may exceed GPU memory capacity!")
                response = input(f"Continue with this batch size? (y/n): ")
                if response.lower() != 'y':
                    print(f"Skipping batch size {batch_size}")
                    continue
            
            print(f"\n-----------------------------------------------")
            print(f"Testing {model_name} with batch size {batch_size}")
            print(f"-----------------------------------------------")
            
            # Create model instance
            if is_huggingface:
                model = model_class
            else:
                model = model_class()
            
            # Create appropriate input data
            if is_huggingface and tokenizer:
                # For HuggingFace models, generate appropriate tokens
                sample_text = ["This is a sample text for testing."] * batch_size
                input_data = tokenizer(sample_text, padding=True, truncation=True, return_tensors="pt")
            elif model_name in ['Transformer', 'TransformerModel']:
                # For custom transformer: [seq_len, batch_size]
                seq_len = 64
                input_data = torch.randint(0, 1000, (seq_len, batch_size))
            else:
                # For CNN models: [batch_size, channels, height, width]
                input_data = torch.randn(batch_size, 3, 32, 32)
            
            # Run baseline
            try:
                baseline = run_baseline(model, input_data, warmup, iterations, is_huggingface)
                
                # Run optimized
                optimized = run_optimized(model, input_data, warmup, iterations, is_huggingface)
                
                # Store results
                results[model_name][batch_size] = {
                    'baseline': baseline,
                    'optimized': optimized
                }
                
                # Print comparison
                speedup = baseline['mean'] / optimized['mean']
                print(f"\nResults for {model_name} (batch size {batch_size}):")
                print(f"  Baseline: {baseline['mean']:.2f} ms ± {baseline['std']:.2f} ms")
                print(f"  Optimized: {optimized['mean']:.2f} ms ± {optimized['std']:.2f} ms")
                print(f"  Speedup: {speedup:.2f}x")
                
                # Report memory usage if available
                if baseline.get('memory_usage'):
                    print(f"  Baseline memory usage: {baseline['memory_usage']:.2f} MB")
                
                # Analysis of optimization effectiveness
                if speedup < 1.0:
                    overhead_percent = ((optimized['mean'] - baseline['mean']) / baseline['mean']) * 100
                    print(f"  Warning: Optimization adds {overhead_percent:.1f}% overhead for this small model/batch size")
                    print(f"  This is normal for small workloads where framework overhead exceeds parallelization benefits")
                
                # Generate visualizations
                output_dir = f"./comparison_results/{model_name}_{batch_size}"
                os.makedirs(output_dir, exist_ok=True)
                
                # Save performance visualizations
                visualizer = PerformanceVisualizer(output_dir=output_dir)
                visualizer.plot_device_utilization(
                    optimized['metrics'], 
                    title=f"{model_name} Device Utilization (batch={batch_size})",
                    use_plotly=False
                )
                
                # Plot execution time comparison
                plt.figure(figsize=(10, 6))
                bar_colors = ['lightcoral' if speedup < 1.0 else 'lightblue', 
                            'salmon' if speedup < 1.0 else 'seagreen']
                plt.bar(["Baseline", "Optimized"], [baseline['mean'], optimized['mean']], color=bar_colors)
                plt.errorbar(["Baseline", "Optimized"], [baseline['mean'], optimized['mean']], 
                            yerr=[baseline['std'], optimized['std']], fmt='o', color='black')
                plt.title(f"{model_name} Execution Time (batch={batch_size})")
                plt.ylabel("Time (ms)")
                plt.grid(axis='y', alpha=0.3)
                
                # Add values on bars
                for i, (name, value, std) in enumerate([("Baseline", baseline['mean'], baseline['std']), 
                                                    ("Optimized", optimized['mean'], optimized['std'])]):
                    plt.text(i, value + 0.1, f"{value:.2f} ± {std:.2f} ms", ha='center')
                
                # Add speedup annotation
                y_max = max(baseline['mean'], optimized['mean']) * 1.2
                plt.ylim(0, y_max)
                plt.annotate(f"{speedup:.2f}x speedup", 
                            xy=(0.5, y_max * 0.9), 
                            ha='center', 
                            fontweight='bold', 
                            color='green' if speedup > 1.0 else 'red')
                
                plt.savefig(f"{output_dir}/execution_time_comparison.png")
                plt.close()
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"\n❌ ERROR: CUDA out of memory with batch size {batch_size}")
                    print(f"This confirms our memory analysis prediction.")
                    print(f"Try using a smaller batch size or implement the memory optimization techniques.")
                else:
                    print(f"\n❌ ERROR: {str(e)}")
    
    return results, memory_analysis_results

def generate_summary_plots(results, batch_sizes):
    """Generate summary plots comparing all models and batch sizes"""
    os.makedirs("./comparison_results/summary", exist_ok=True)
    
    # Create speedup comparison chart
    model_names = list(results.keys())
    x = np.arange(len(model_names))
    width = 0.8 / len(batch_sizes)
    
    plt.figure(figsize=(12, 7))
    
    # Plot speedup for each batch size
    for i, batch_size in enumerate(batch_sizes):
        speedups = []
        for model_name in model_names:
            res = results[model_name].get(batch_size)
            if res:  # Only include if we have results (might be skipped due to OOM)
                speedup = res['baseline']['mean'] / res['optimized']['mean']
                speedups.append(speedup)
            else:
                speedups.append(0)  # Use 0 to indicate no result
        
        plt.bar(x + i*width - width*len(batch_sizes)/2 + width/2, 
                speedups, 
                width, 
                label=f'Batch Size {batch_size}',
                color=plt.cm.viridis(i/len(batch_sizes)))
    
    # Add horizontal line at y=1.0 (baseline)
    plt.axhline(y=1.0, color='r', linestyle='-', alpha=0.3, label='Baseline (no speedup)')
    
    plt.xlabel('Model')
    plt.ylabel('Speedup Factor (higher is better)')
    plt.title('Performance Speedup by Model and Batch Size')
    plt.xticks(x, model_names)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Add text annotations for speedup values
    for i, model_name in enumerate(model_names):
        for j, batch_size in enumerate(batch_sizes):
            res = results[model_name].get(batch_size)
            if res:  # Only include if we have results
                speedup = res['baseline']['mean'] / res['optimized']['mean']
                plt.text(i + j*width - width*len(batch_sizes)/2 + width/2, 
                         speedup + 0.05, 
                         f"{speedup:.2f}x", 
                         ha='center', 
                         va='bottom', 
                         fontsize=8,
                         rotation=90 if speedup > 3 else 0)
    
    plt.tight_layout()
    plt.savefig("./comparison_results/summary/speedup_comparison.png")
    plt.close()
    
    # Create memory usage analysis
    plt.figure(figsize=(12, 7))
    
    # Create analysis notes
    with open("./comparison_results/summary/analysis.txt", "w") as f:
        f.write("Performance Optimization Framework Analysis\n")
        f.write("==========================================\n\n")
        f.write("Summary of findings:\n\n")
        
        # Find which models benefit most
        best_speedups = {}
        for model_name in model_names:
            model_best = 0
            best_batch = 0
            for batch_size in batch_sizes:
                if batch_size in results[model_name]:
                    res = results[model_name][batch_size]
                    speedup = res['baseline']['mean'] / res['optimized']['mean']
                    if speedup > model_best:
                        model_best = speedup
                        best_batch = batch_size
            best_speedups[model_name] = (model_best, best_batch)
        
        # Sort models by benefit
        sorted_models = sorted(best_speedups.items(), key=lambda x: x[1][0], reverse=True)
        
        for model_name, (speedup, batch) in sorted_models:
            f.write(f"- {model_name}: Best speedup {speedup:.2f}x at batch size {batch}\n")
            if speedup < 1.0:
                f.write(f"  * Not recommended for optimization (overhead exceeds benefits)\n")
            elif speedup < 1.2:
                f.write(f"  * Marginal benefits from optimization\n")
            else:
                f.write(f"  * Good candidate for optimization\n")
            
        f.write("\nRecommendations:\n\n")
        
        if any(speedup > 1.2 for speedup, _ in best_speedups.values()):
            f.write("1. Focus optimization efforts on larger, more complex models\n")
            f.write("2. Use larger batch sizes when possible to amortize framework overhead\n")
            f.write("3. Consider model-specific optimizations for best results\n")
        else:
            f.write("1. Current models may be too small to benefit from parallelization\n")
            f.write("2. Consider using the framework only for models with higher computational intensity\n")

        # Add memory-specific analysis
        f.write("\nMemory Analysis:\n\n")
        f.write("1. Models that exceeded memory capacity:\n")
        any_oom = False
        for model_name in model_names:
            missing_batch_sizes = [bs for bs in batch_sizes if bs not in results[model_name]]
            if missing_batch_sizes:
                any_oom = True
                f.write(f"   - {model_name}: Batch sizes {missing_batch_sizes} caused OOM errors\n")
        
        if not any_oom:
            f.write("   - No models exceeded memory capacity\n")
            
        f.write("\n2. Memory optimization recommendations:\n")
        f.write("   - Use mixed precision training for ~50% memory reduction\n")
        f.write("   - Implement gradient checkpointing for memory-intensive models\n")
        f.write("   - Consider model pruning or distillation for very large models\n")
        f.write("   - Use gradient accumulation to effectively increase batch size without increasing memory\n")

def load_complex_model():
    """
    Helper function to load ComplexModel if available
    """
    try:
        from energy_optimization import ComplexModel
        return ComplexModel
    except ImportError:
        print("Warning: Could not import ComplexModel, creating a simpler version")
        # Create a simpler version as fallback
        class ComplexModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.features = torch.nn.Sequential(
                    torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.MaxPool2d(kernel_size=2, stride=2),
                )
                self.classifier = torch.nn.Sequential(
                    torch.nn.Linear(128 * 16 * 16, 512),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Linear(512, 10)
                )
            
            def forward(self, x):
                x = self.features(x)
                x = torch.flatten(x, 1)
                x = self.classifier(x)
                return x
                
        return ComplexModel

def main():
    # Create output directory
    os.makedirs("./comparison_results", exist_ok=True)
    
    # Create cache directory if needed
    os.makedirs("./cache", exist_ok=True)
    
    # ANSI color codes for terminal output
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    # Load HuggingFace model and tokenizer
    try:
        print("Loading HuggingFace transformer model...")
        hf_model_name = "distilbert-base-uncased"  # Smaller model to avoid OOM
        hf_model, hf_tokenizer = load_huggingface_transformer(hf_model_name)
        huggingface_available = True
    except Exception as e:
        print(f"Warning: Could not load HuggingFace model. Error: {e}")
        print("Continuing without HuggingFace model comparison.")
        huggingface_available = False
    
    # Load ComplexModel
    ComplexModel = load_complex_model()
    
    # Define models to test with performance recommendations
    models = {
        'ExampleModel': {
            'class': ExampleModel,
            'recommended_batch_size': 256,
            'performance_note': 'Small model with minimal memory requirements'
        },
        'LargeModel': {
            'class': LargeModel,
            'recommended_batch_size': 64,
            'performance_note': 'More complex model that benefits from parallelization'
        },
        'ComplexModel': {
            'class': ComplexModel,
            'recommended_batch_size': 32,
            'performance_note': 'Memory-intensive model that shows significant speedup with optimization'
        },
        'TransformerModel': {
            'class': TransformerModel,
            'recommended_batch_size': 16,
            'performance_note': 'Transformer architecture that benefits from GPU acceleration'
        }
    }
    
    # Add HuggingFace model if available
    if huggingface_available:
        models['HuggingFaceTransformer'] = {
            'class': hf_model,
            'tokenizer': hf_tokenizer,
            'is_huggingface': True,
            'hf_model_name': hf_model_name,
            'recommended_batch_size': 8,
            'performance_note': 'Pre-trained transformer model from HuggingFace - typically benefits greatly from parallelization'
        }
    
    # Define batch sizes to test
    batch_sizes = [1, 16, 32, 64, 128]
    
    # Print performance recommendations before running tests
    print("\n==== Model Performance Recommendations ====")
    for model_name, info in models.items():
        print(f"\n{model_name}:")
        print(f"  Recommended batch size: {info.get('recommended_batch_size', 'Unknown')}")
        print(f"  Note: {info.get('performance_note', '')}")
    
    # Allow user to select which models to run
    print("\nSelect models to test (comma-separated list, or 'all' for all models):")
    model_list = list(models.keys())
    for i, name in enumerate(model_list):
        print(f"  {i+1}. {name}")
    
    selection = input("\nSelection (default: all): ").strip()
    if selection and selection.lower() != 'all':
        selected_indices = [int(idx.strip())-1 for idx in selection.split(',') if idx.strip().isdigit()]
        selected_models = {name: models[name] for i, name in enumerate(model_list) if i in selected_indices}
    else:
        selected_models = models
    
    # Allow user to customize batch sizes
    print("\nCurrent batch sizes to test:", batch_sizes)
    custom_batch = input("Enter custom batch sizes (comma-separated) or press Enter to use defaults: ").strip()
    if custom_batch:
        try:
            batch_sizes = [int(b.strip()) for b in custom_batch.split(',') if b.strip().isdigit()]
            print(f"Using custom batch sizes: {batch_sizes}")
        except:
            print("Invalid input, using default batch sizes")
    
    # Run comparison
    results, memory_analysis_results = compare_models(selected_models, batch_sizes)
    
    # Generate summary visualizations and analysis
    generate_summary_plots(results, batch_sizes)
    
    # Summary report with performance notes and memory analysis
    print("\n============== SUMMARY REPORT ==============")
    for model_name in results:
        model_info = models.get(model_name, {})
        perf_note = model_info.get('performance_note', '')
        
        print(f"\n{BOLD}{model_name}{RESET}:")
        print(f"  {perf_note}")
        
        # Print speedup results
        speedups = []
        for batch_size, res in results[model_name].items():
            speedup = res['baseline']['mean'] / res['optimized']['mean']
            speedups.append((batch_size, speedup))
            print(f"  Batch size {batch_size}: {speedup:.2f}x speedup")
            if speedup < 1.0:
                overhead = ((res['optimized']['mean'] - res['baseline']['mean']) / res['baseline']['mean']) * 100
                print(f"    ⚠️ Framework overhead: {overhead:.1f}% (normal for small workloads)")
        
        # Print memory analysis summary if available
        memory_analysis = memory_analysis_results.get(model_name, {})
        if memory_analysis:
            print(f"\n  {BOLD}Memory Analysis:{RESET}")
            gpu_name = memory_analysis.get('gpu_name', 'Unknown GPU')
            gpu_vram = memory_analysis.get('gpu_vram_gb', 0)
            if gpu_vram > 0:
                print(f"  GPU: {gpu_name} with {gpu_vram:.2f} GB VRAM")
            
            param_count = memory_analysis.get('param_count', 0)
            print(f"  Model parameters: {param_count:,}")
            
            max_bs = memory_analysis.get('max_batch_size', 0)
            if max_bs > 0:
                print(f"  Recommended maximum batch size: {max_bs}")
            
            # Print memory requirements table with color coding
            mem_reqs = memory_analysis.get('memory_requirements', {})
            if mem_reqs:
                print("\n  Memory requirements by batch size:")
                print(f"  {'Batch Size':<10} {'Params (MB)':<12} {'Activations (MB)':<16} {'Total (GB)':<10} {'Status':<10}")
                for bs in sorted(mem_reqs.keys()):
                    req = mem_reqs[bs]
                    color = req.get('color', RESET)
                    status = req.get('status', '')
                    print(f"  {bs:<10} {req['param_mb']:<12.2f} {req['act_mb']:<16.2f} {req['total_gb']:<10.2f} {color}{status}{RESET}")
            
            # Print memory optimization tips
            if memory_analysis.get('memory_tips'):
                print(f"\n  {BOLD}Memory optimization tips:{RESET}")
                for tip in memory_analysis.get('memory_tips', []):
                    print(f"  {tip}")
    
    print("\nDetailed results saved to ./comparison_results/")
    print("Summary analysis available at ./comparison_results/summary/")

if __name__ == "__main__":
    main()