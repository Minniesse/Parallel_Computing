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
    
    for model_name, model_info in models.items():
        results[model_name] = {}
        model_class = model_info.get('class')
        is_huggingface = model_info.get('is_huggingface', False)
        hf_model_name = model_info.get('hf_model_name', None)
        tokenizer = model_info.get('tokenizer', None)
        
        for batch_size in batch_sizes:
            print(f"\n===============================================")
            print(f"Testing {model_name} with batch size {batch_size}")
            print(f"===============================================")
            
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
    
    return results

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
            res = results[model_name][batch_size]
            speedup = res['baseline']['mean'] / res['optimized']['mean']
            speedups.append(speedup)
        
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
            res = results[model_name][batch_size]
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

def main():
    # Create output directory
    os.makedirs("./comparison_results", exist_ok=True)
    
    # Load HuggingFace model and tokenizer
    try:
        print("Loading HuggingFace transformer model...")
        hf_model_name = "meta-llama/Llama-3.2-1B-Instruct"
        hf_model, hf_tokenizer = load_huggingface_transformer(hf_model_name)
        huggingface_available = True
    except Exception as e:
        print(f"Warning: Could not load HuggingFace model. Error: {e}")
        print("Continuing without HuggingFace model comparison.")
        huggingface_available = False
    
    # Define models to test with performance recommendations
    models = {
        'ExampleModel': {
            'class': ExampleModel,
        },
        'LargeModel': {
            'class': LargeModel,
        },
        'ComplexModel': {
            'class': lambda: ComplexModel(),  # Import from energy_optimization example
        },
        'TransformerModel': {
            'class': TransformerModel,
        }
    }
    
    # Add HuggingFace model if available
    if huggingface_available:
        models['HuggingFaceTransformer'] = {
            'class': hf_model,
            'tokenizer': hf_tokenizer,
            'is_huggingface': True,
            'hf_model_name': hf_model_name,
            'recommended_batch_size': 16,
            'performance_note': 'Pre-trained transformer model from HuggingFace - typically benefits greatly from parallelization'
        }
    
    # Import ComplexModel only when needed to avoid circular imports
    if 'ComplexModel' in models:
        # Add the import at runtime
        try:
            from energy_optimization import ComplexModel
        except ImportError:
            print("Warning: Could not import ComplexModel, removing from test list")
            del models['ComplexModel']
    
    # Define batch sizes to test
    batch_sizes = [1, 16, 32]
    
    # Print performance recommendations before running tests
    print("\n==== Model Performance Recommendations ====")
    for model_name, info in models.items():
        print(f"\n{model_name}:")
    
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
    
    # Run comparison
    results = compare_models(selected_models, batch_sizes)
    
    # Generate summary visualizations and analysis
    generate_summary_plots(results, batch_sizes)
    
    # Summary report with performance notes
    print("\n============== SUMMARY REPORT ==============")
    for model_name in results:
        model_info = models.get(model_name, {})
        perf_note = model_info.get('performance_note', '')
        
        print(f"\n{model_name}:")
        print(f"  {perf_note}")
        
        for batch_size, res in results[model_name].items():
            speedup = res['baseline']['mean'] / res['optimized']['mean']
            print(f"  Batch size {batch_size}: {speedup:.2f}x speedup")
            if speedup < 1.0:
                overhead = ((res['optimized']['mean'] - res['baseline']['mean']) / res['baseline']['mean']) * 100
                print(f"    ⚠️ Framework overhead: {overhead:.1f}% (normal for small workloads)")
    
    print("\nDetailed results saved to ./comparison_results/")
    print("Summary analysis available at ./comparison_results/summary/")

if __name__ == "__main__":
    main()
