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

def estimate_model_memory(model, batch_size, input_shape, dtype=torch.float32, training_mode=True):
    """
    Estimate the memory required by a model.
    
    Args:
        model: PyTorch model
        batch_size: Batch size
        input_shape: Shape of a single input (without batch dimension)
        dtype: Data type of model parameters and activations
        training_mode: Whether to estimate for training (True) or inference (False)
        
    Returns:
        Dictionary with memory estimates in bytes
    """
    # Calculate model parameters memory
    param_bytes = sum(p.numel() * dtype_size(p.dtype) for p in model.parameters())
    
    # Estimate forward activations memory (this is an approximation)
    # A common heuristic is that activations use 2-3x the memory of parameters
    # Scale with batch size relative to a reference batch size of 32
    activations_bytes = param_bytes * 2.5 * (batch_size / 32)
    
    # Input size
    input_size = np.prod(input_shape)
    input_bytes = batch_size * input_size * dtype_size(dtype)
    
    # For inference mode, we don't need gradients or optimizer states
    if not training_mode:
        # For inference, we use less activation memory since we don't need to store for backward
        # And we don't need gradients or optimizer states at all
        inference_total = param_bytes + (activations_bytes * 0.5) + input_bytes
        
        # Add some overhead for PyTorch internals and fragmentation
        inference_total = inference_total * 1.1  # 10% overhead for inference
        
        return {
            'parameters': param_bytes,
            'activations': activations_bytes * 0.5,  # Less activation memory needed
            'gradients': 0,  # No gradients in inference
            'optimizer': 0,  # No optimizer states in inference
            'input': input_bytes,
            'total': inference_total,
            'mode': 'inference'
        }
    else:
        # For training mode, include everything
        # Estimate backward gradients memory
        gradient_bytes = param_bytes  # Gradients are same size as parameters
        
        # Estimate optimizer states (Adam uses 2 additional states per parameter)
        optimizer_bytes = param_bytes * 2  # For Adam optimizer
        
        # Calculate total memory for training
        training_total = param_bytes + activations_bytes + gradient_bytes + optimizer_bytes + input_bytes
        
        # Add some overhead for PyTorch internals and fragmentation
        training_total = training_total * 1.2  # 20% overhead for training
        
        return {
            'parameters': param_bytes,
            'activations': activations_bytes,
            'gradients': gradient_bytes,
            'optimizer': optimizer_bytes,
            'input': input_bytes,
            'total': training_total,
            'mode': 'training'
        }

def calculate_max_batch_size(model, input_shape, available_memory, min_batch=1, max_batch=1024, training_mode=True):
    """
    Calculate the maximum batch size that would fit in GPU memory.
    
    Args:
        model: PyTorch model
        input_shape: Shape of a single input (without batch dimension)
        available_memory: Available GPU memory in bytes
        min_batch: Minimum batch size to consider
        max_batch: Maximum batch size to consider
        training_mode: Whether to calculate for training or inference
        
    Returns:
        Maximum batch size that fits in memory or 0 if even min_batch doesn't fit
    """
    # Check if even the minimum batch size fits
    min_mem = estimate_model_memory(model, min_batch, input_shape, training_mode=training_mode)
    if min_mem['total'] > available_memory:
        return 0  # Even minimum batch size doesn't fit
    
    # Binary search to find the maximum batch size
    left, right = min_batch, max_batch
    max_feasible = min_batch
    
    while left <= right:
        mid = (left + right) // 2
        memory_est = estimate_model_memory(model, mid, input_shape, training_mode=training_mode)
        
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

def get_available_llm_models():
    """Return a list of available LLM models of various sizes"""
    return {
        "tiny": "distilbert-base-uncased",  # 66M parameters
        "small": "bert-base-uncased",       # 110M parameters
        "medium": "roberta-large",          # 355M parameters
        "large": "gpt2-medium",             # 355M parameters
        "xl": "gpt2-large",                # 774M parameters
        # Uncomment if you have enough VRAM (needs 20+ GB)
        # "xxl": "gpt2-xl",                  # 1.5B parameters
    }

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
    
    # Calculate memory usage for optimized model if possible
    memory_usage = None
    device = next(optimized_model.model.parameters()).device
    if device.type == 'cuda':
        try:
            torch.cuda.reset_peak_memory_stats(device)
            if is_huggingface:
                _ = optimized_model(**input_data)
            else:
                _ = optimized_model(input_data)
            memory_usage = torch.cuda.max_memory_allocated(device) / (1024 * 1024)  # MB
        except:
            pass
    
    metrics = optimized_model.get_performance_metrics()
    
    return {
        'times': times,
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'metrics': metrics,
        'optimized_model': optimized_model,
        'memory_usage': memory_usage
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
        
        # First show training mode estimates
        print("\nTRAINING MODE (includes gradients & optimizer states):")
        for bs in batch_sizes:
            # Skip batch size analysis if no GPU
            if available_gpu_memory == 0:
                continue
                
            mem_est = estimate_model_memory(model, bs, input_shape, training_mode=True)
            
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
            
            memory_requirements[f"train_{bs}"] = {
                'param_mb': param_mb,
                'act_mb': act_mb,
                'total_gb': total_gb,
                'status': status,
                'color': color,
                'mode': 'training'
            }
            
            print(f"{bs:<10} {param_mb:<12.2f} {act_mb:<16.2f} {total_gb:<10.2f} {color}{status}{RESET}")
            
        # Now show inference mode estimates
        print("\nINFERENCE MODE (torch.no_grad()):")
        for bs in batch_sizes:
            # Skip batch size analysis if no GPU
            if available_gpu_memory == 0:
                continue
                
            mem_est = estimate_model_memory(model, bs, input_shape, training_mode=False)
            
            param_mb = mem_est['parameters'] / (1024 ** 2)
            act_mb = mem_est['activations'] / (1024 ** 2)
            total_gb = mem_est['total'] / (1024 ** 3)
            
            status = "OK"
            color = GREEN
            if mem_est['total'] > available_gpu_memory:
                status = "OOM"
                color = RED
            elif mem_est['total'] > available_gpu_memory * 0.9:
                status = "RISK-OOM"
                color = YELLOW
            elif mem_est['total'] > available_gpu_memory * 0.8:
                status = "CAUTION"
                color = YELLOW
            
            memory_requirements[f"infer_{bs}"] = {
                'param_mb': param_mb,
                'act_mb': act_mb,
                'total_gb': total_gb,
                'status': status,
                'color': color,
                'mode': 'inference'
            }
            
            print(f"{bs:<10} {param_mb:<12.2f} {act_mb:<16.2f} {total_gb:<10.2f} {color}{status}{RESET}")
        
        # Store memory analysis information for summary report
        max_bs = 0
        memory_tips = []
        
        # Calculate recommended batch size for training
        train_max_bs = 0
        infer_max_bs = 0
        memory_tips = []
        
        if available_gpu_memory > 0:
            # Calculate max batch size for training
            train_max_bs = calculate_max_batch_size(
                model, 
                input_shape, 
                available_gpu_memory * 0.9,  # Use 90% of available memory
                min_batch=1,
                max_batch=1024,
                training_mode=True
            )
            
            # Calculate max batch size for inference
            infer_max_bs = calculate_max_batch_size(
                model, 
                input_shape, 
                available_gpu_memory * 0.9,  # Use 90% of available memory
                min_batch=1,
                max_batch=1024,
                training_mode=False
            )
            
            print(f"\nRecommended maximum batch size for TRAINING: {train_max_bs}")
            print(f"Recommended maximum batch size for INFERENCE: {infer_max_bs}")
            
            # Print memory optimization tips
            print("\nMemory optimization tips:")
            # Check if training will be memory-constrained
            if train_max_bs < min(batch_sizes):
                print(f"- ⚠️ Requested batch sizes exceed GPU memory capacity for TRAINING")
                memory_tips.append(f"- Requested batch sizes exceed GPU memory capacity for TRAINING")
                print(f"- Consider using gradient accumulation instead of large batch sizes")
                memory_tips.append(f"- Consider using gradient accumulation instead of large batch sizes")
                print(f"- Try mixed precision training (torch.cuda.amp) to reduce memory usage by ~50%")
                memory_tips.append(f"- Try mixed precision training (torch.cuda.amp) to reduce memory usage by ~50%")
                print(f"- Model checkpoint/activation recomputation can save memory at the cost of computation")
                memory_tips.append(f"- Model checkpoint/activation recomputation can save memory at the cost of computation")
            elif memory_limited_batch < float('inf'):
                print(f"- ⚠️ Batch size {memory_limited_batch} and above may cause OOM errors for TRAINING")
                memory_tips.append(f"- Batch size {memory_limited_batch} and above may cause OOM errors for TRAINING")
                print(f"- Consider using gradient accumulation for effective batch sizes > {train_max_bs}")
                memory_tips.append(f"- Consider using gradient accumulation for effective batch sizes > {train_max_bs}")
                print(f"- For {model_name}, mixed precision training can increase max batch size to ~{int(train_max_bs * 1.7)}")
                memory_tips.append(f"- For {model_name}, mixed precision training can increase max batch size to ~{int(train_max_bs * 1.7)}")
            else:
                print(f"- All requested batch sizes should fit within GPU memory for both TRAINING and INFERENCE")
                memory_tips.append(f"- All requested batch sizes should fit within GPU memory for both TRAINING and INFERENCE")
                print(f"- For large-scale training, consider gradient accumulation or distributed training")
                memory_tips.append(f"- For large-scale training, consider gradient accumulation or distributed training")
        
        # Store memory analysis results
        memory_analysis_results[model_name] = {
            'gpu_name': gpu_name,
            'gpu_vram_gb': available_gpu_memory / (1024 ** 3) if available_gpu_memory > 0 else 0,
            'param_count': param_count,
            'train_max_batch_size': train_max_bs,
            'infer_max_batch_size': infer_max_bs,
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
    
    # Get available LLM models
    llm_models = get_available_llm_models()
    
    print(f"\n{BOLD}=== Available Language Models ==={RESET}")
    for size, model_name in llm_models.items():
        print(f"  {size}: {model_name}")
    
    # Ask user which LLM size to test
    print("\nSelect LLM size to test:")
    llm_sizes = list(llm_models.keys())
    for i, size in enumerate(llm_sizes):
        print(f"  {i+1}. {size} ({llm_models[size]})")
    
    selection = input("\nSelection (default: tiny): ").strip()
    
    if selection.isdigit() and 1 <= int(selection) <= len(llm_sizes):
        selected_size = llm_sizes[int(selection) - 1]
    else:
        selected_size = "tiny"  # Default to the smallest model
    
    selected_model_name = llm_models[selected_size]
    print(f"\nSelected model: {selected_model_name}")
    
    # Load HuggingFace model and tokenizer
    try:
        print(f"Loading {selected_size} language model ({selected_model_name})...")
        hf_model, hf_tokenizer = load_huggingface_transformer(selected_model_name)
        huggingface_available = True
    except Exception as e:
        print(f"Warning: Could not load HuggingFace model. Error: {e}")
        print("Continuing without HuggingFace model comparison.")
        huggingface_available = False
    
    # Load ComplexModel
    ComplexModel = load_complex_model()
    
    # Only test the selected LLM
    models = {}
    
    # Add HuggingFace model if available
    if huggingface_available:
        models[f'LLM_{selected_size.upper()}'] = {
            'class': hf_model,
            'tokenizer': hf_tokenizer,
            'is_huggingface': True,
            'hf_model_name': selected_model_name,
            'recommended_batch_size': 8 if selected_size in ["tiny", "small"] else 4 if selected_size == "medium" else 2 if selected_size == "large" else 1,
            'performance_note': f'Pre-trained {selected_size} language model ({selected_model_name})'
        }
    
    # Define batch sizes to test - use smaller batch sizes for larger models
    if selected_size in ["tiny", "small"]:
        batch_sizes = [1, 8, 16, 32, 64]
    elif selected_size == "medium":
        batch_sizes = [1, 4, 8, 16, 32]
    elif selected_size == "large":
        batch_sizes = [1, 2, 4, 8, 16]
    else:  # xl and xxl
        batch_sizes = [1, 2, 4, 8]
    
    # Print performance recommendations before running tests
    print("\n==== Model Performance Recommendations ====")
    for model_name, info in models.items():
        print(f"\n{model_name}:")
        print(f"  Recommended batch size: {info.get('recommended_batch_size', 'Unknown')}")
        print(f"  Note: {info.get('performance_note', '')}")
    
    # Allow user to customize batch sizes
    print("\nRecommended batch sizes to test:", batch_sizes)
    custom_batch = input("Enter custom batch sizes (comma-separated) or press Enter to use defaults: ").strip()
    if custom_batch:
        try:
            batch_sizes = [int(b.strip()) for b in custom_batch.split(',') if b.strip().isdigit()]
            print(f"Using custom batch sizes: {batch_sizes}")
        except:
            print("Invalid input, using default batch sizes")
    
    # Run comparison
    results, memory_analysis_results = compare_models(models, batch_sizes)
    
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
            
            train_max_bs = memory_analysis.get('train_max_batch_size', 0)
            infer_max_bs = memory_analysis.get('infer_max_batch_size', 0)
            
            if train_max_bs > 0 or infer_max_bs > 0:
                print(f"  Recommended maximum batch size for TRAINING: {train_max_bs}")
                print(f"  Recommended maximum batch size for INFERENCE: {infer_max_bs}")
                
                # Compare with measured GPU memory usage
                if model_name in results:
                    bs_memory_baseline = {}
                    bs_memory_optimized = {}
                    
                    # Collect memory usage from both baseline and optimized runs
                    for bs, res in results[model_name].items():
                        if 'baseline' in res and 'memory_usage' in res['baseline']:
                            bs_memory_baseline[bs] = res['baseline']['memory_usage']
                        # Also collect optimized memory usage if available
                        if 'optimized' in res and 'memory_usage' in res['optimized']:
                            bs_memory_optimized[bs] = res['optimized']['memory_usage']
                    
                    # Show baseline model measurements
                    if bs_memory_baseline:
                        # Table for comparing with inference estimates
                        print("\n  Baseline (non-optimized) memory vs Inference estimates:")
                        print(f"  {'Batch Size':<10} {'Baseline (MB)':<14} {'Baseline (GB)':<14} {'Infer Est (GB)':<15} {'Diff %':<10}")
                        for bs in sorted(bs_memory_baseline.keys()):
                            mb_used = bs_memory_baseline[bs]
                            gb_used = mb_used / 1024
                            
                            # Get corresponding inference estimate for comparison
                            est_key = f"infer_{bs}"
                            infer_est_gb = memory_analysis.get('memory_requirements', {}).get(est_key, {}).get('total_gb', 0)
                            
                            # Calculate difference percentage
                            infer_diff_pct = ((infer_est_gb - gb_used) / infer_est_gb * 100) if infer_est_gb > 0 else 0
                            infer_diff_text = f"{infer_diff_pct:+.1f}%" if infer_est_gb > 0 else "N/A"
                            
                            # Color code the difference
                            infer_color = ""
                            if abs(infer_diff_pct) > 30:
                                infer_color = RED if infer_diff_pct < 0 else YELLOW
                            elif abs(infer_diff_pct) > 10:
                                infer_color = YELLOW
                            else:
                                infer_color = GREEN
                                
                            print(f"  {bs:<10} {mb_used:<14.2f} {gb_used:<14.2f} {infer_est_gb:<15.2f} {infer_color}{infer_diff_text}{RESET}")
                        
                        # Table for comparing with training estimates
                        print("\n  Baseline (non-optimized) memory vs Training estimates:")
                        print(f"  {'Batch Size':<10} {'Baseline (MB)':<14} {'Baseline (GB)':<14} {'Train Est (GB)':<15} {'Diff %':<10}")
                        for bs in sorted(bs_memory_baseline.keys()):
                            mb_used = bs_memory_baseline[bs]
                            gb_used = mb_used / 1024
                            
                            # Get corresponding training estimate for comparison
                            est_key = f"train_{bs}"
                            train_est_gb = memory_analysis.get('memory_requirements', {}).get(est_key, {}).get('total_gb', 0)
                            
                            # Calculate difference percentage for training
                            train_diff_pct = ((train_est_gb - gb_used) / train_est_gb * 100) if train_est_gb > 0 else 0
                            train_diff_text = f"{train_diff_pct:+.1f}%" if train_est_gb > 0 else "N/A"
                            
                            # Color code the difference
                            train_color = ""
                            if train_diff_pct > 70:  # Training estimates are much higher
                                train_color = YELLOW
                            elif train_diff_pct > 50:
                                train_color = GREEN
                            else:
                                train_color = RED  # Training is closer to inference than expected
                                
                            print(f"  {bs:<10} {mb_used:<14.2f} {gb_used:<14.2f} {train_est_gb:<15.2f} {train_color}{train_diff_text}{RESET}")
                        
                                                # Show optimized model measurements if available
                        if bs_memory_optimized:
                            print("\n  Optimized model memory usage:")
                            print(f"  {'Batch Size':<10} {'Optimized (MB)':<14} {'Optimized (GB)':<14} {'vs Baseline':<15}")
                            for bs in sorted(bs_memory_optimized.keys()):
                                if bs in bs_memory_baseline and bs_memory_optimized[bs] is not None:
                                    mb_used_opt = bs_memory_optimized[bs]
                                    gb_used_opt = mb_used_opt / 1024
                                    mb_used_base = bs_memory_baseline[bs]
                                    
                                    # Calculate percentage difference
                                    diff_pct = ((mb_used_opt - mb_used_base) / mb_used_base * 100)
                                    diff_text = f"{diff_pct:+.1f}%"
                                    
                                    # Color code (green if using less memory, red if using more)
                                    color = GREEN if diff_pct < 0 else (YELLOW if diff_pct < 10 else RED)
                                    
                                    print(f"  {bs:<10} {mb_used_opt:<14.2f} {gb_used_opt:<14.2f} {color}{diff_text}{RESET}")
                                elif bs_memory_optimized[bs] is None:
                                    print(f"  {bs:<10} {'N/A':<14} {'N/A':<14} {'Memory measurement unavailable':<15}")
                          
                        
                        # Explain the difference 
                        largest_bs = max(bs_memory_baseline.keys())
                        largest_infer_est = memory_analysis.get('memory_requirements', {}).get(f'infer_{largest_bs}', {}).get('total_gb', 0)
                        largest_train_est = memory_analysis.get('memory_requirements', {}).get(f'train_{largest_bs}', {}).get('total_gb', 0)
                        largest_measured = bs_memory_baseline[largest_bs] / 1024  # Convert to GB
                        
                        if largest_infer_est > 0 and largest_measured > 0:
                            infer_diff_pct = abs(largest_infer_est - largest_measured) / largest_infer_est * 100
                            train_diff_pct = abs(largest_train_est - largest_measured) / largest_train_est * 100
                            
                            print(f"\n  Note on memory measurements vs. estimates:")
                            print(f"  For batch size {largest_bs}:")
                            print(f"  - Baseline model used {largest_measured:.2f} GB")
                            print(f"  - Inference estimate was {largest_infer_est:.2f} GB ({infer_diff_pct:.1f}% difference)")
                            print(f"  - Training estimate was {largest_train_est:.2f} GB ({train_diff_pct:.1f}% difference)")
                            print(f"  - Training requires more memory for gradients, optimizer states, and activation storage")
                            
                            if bs_memory_optimized and largest_bs in bs_memory_optimized and bs_memory_optimized[largest_bs] is not None:
                                opt_measured = bs_memory_optimized[largest_bs] / 1024
                                baseline_measured = largest_measured
                                opt_diff = ((opt_measured - baseline_measured) / baseline_measured * 100)
                                print(f"  - Optimized model used {opt_measured:.2f} GB ({opt_diff:+.1f}% vs baseline)")
                            elif bs_memory_optimized and largest_bs in bs_memory_optimized:
                                print(f"  - Optimized model memory measurement was unavailable")
                                
                            if largest_measured < largest_infer_est:
                                print(f"  - Our inference estimates are conservative to provide a safety margin")
                            else:
                                print(f"  - Actual usage exceeds inference estimates - consider using smaller batch sizes")
            
            # Print memory requirements table with color coding
            mem_reqs = memory_analysis.get('memory_requirements', {})
            if mem_reqs:
                # First show training mode estimates
                print("\n  TRAINING MODE memory requirements:")
                print(f"  {'Batch Size':<10} {'Params (MB)':<12} {'Activations (MB)':<16} {'Total (GB)':<10} {'Status':<10}")
                train_reqs = {k: v for k, v in mem_reqs.items() if k.startswith('train_')}
                for key in sorted(train_reqs.keys(), key=lambda x: int(x.split('_')[1])):
                    bs = key.split('_')[1]
                    req = train_reqs[key]
                    color = req.get('color', RESET)
                    status = req.get('status', '')
                    print(f"  {bs:<10} {req['param_mb']:<12.2f} {req['act_mb']:<16.2f} {req['total_gb']:<10.2f} {color}{status}{RESET}")
                
                # Now show inference mode estimates
                print("\n  INFERENCE MODE memory requirements:")
                print(f"  {'Batch Size':<10} {'Params (MB)':<12} {'Activations (MB)':<16} {'Total (GB)':<10} {'Status':<10}")
                infer_reqs = {k: v for k, v in mem_reqs.items() if k.startswith('infer_')}
                for key in sorted(infer_reqs.keys(), key=lambda x: int(x.split('_')[1])):
                    bs = key.split('_')[1]
                    req = infer_reqs[key]
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