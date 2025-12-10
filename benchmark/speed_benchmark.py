"""
SPEED BENCHMARK
===============

How fast is ternary vs float32?

Zane - The Ian Index
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.ternary_transformer import TernaryLinear


def benchmark_matmul():
    """Benchmark different matrix multiplication methods."""
    print("\n" + "="*70)
    print("SPEED BENCHMARK: Ternary vs Float32 MatMul")
    print("="*70)
    
    sizes = [
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 512, 2048),  # Typical MLP up-proj
        (512, 2048, 512),   # Typical MLP down-proj
    ]
    
    print(f"\n{'Size':<20} {'Float32':<12} {'Ternary':<12} {'Explicit':<12} {'Ratio':<10}")
    print("-" * 70)
    
    for m, k, n in sizes:
        x = np.random.randn(m, k).astype(np.float32)
        
        # Float32 weights
        w_float = np.random.randn(k, n).astype(np.float32) * 0.02
        
        # Ternary weights
        layer = TernaryLinear(k, n, threshold=0.33)
        
        # Benchmark float32
        iterations = 10
        
        start = time.perf_counter()
        for _ in range(iterations):
            _ = x @ w_float
        float_time = (time.perf_counter() - start) / iterations * 1000
        
        # Benchmark ternary (optimized)
        start = time.perf_counter()
        for _ in range(iterations):
            _ = layer.forward(x)
        ternary_time = (time.perf_counter() - start) / iterations * 1000
        
        # Benchmark explicit (proves no multiplication)
        if m * k * n < 500000:  # Only for small sizes
            start = time.perf_counter()
            _ = layer.forward_explicit(x[:1])  # Single batch
            explicit_time = (time.perf_counter() - start) * 1000
        else:
            explicit_time = float('nan')
        
        ratio = float_time / ternary_time if ternary_time > 0 else 0
        
        size_str = f"{m}x{k}x{n}"
        explicit_str = f"{explicit_time:.2f}ms" if not np.isnan(explicit_time) else "N/A"
        
        print(f"{size_str:<20} {float_time:.2f}ms      {ternary_time:.2f}ms      {explicit_str:<12} {ratio:.2f}x")
    
    return True


def benchmark_full_forward():
    """Benchmark full transformer forward pass."""
    print("\n" + "="*70)
    print("FULL TRANSFORMER FORWARD PASS TIMING")
    print("="*70)
    
    from model.ternary_transformer import TernaryConfig, TernaryTransformer
    
    configs = [
        ("Tiny",   TernaryConfig(vocab_size=1000, hidden_size=128, num_heads=2, num_layers=2, intermediate_size=256)),
        ("Small",  TernaryConfig(vocab_size=1000, hidden_size=256, num_heads=4, num_layers=4, intermediate_size=512)),
        ("Medium", TernaryConfig(vocab_size=1000, hidden_size=512, num_heads=8, num_layers=6, intermediate_size=2048)),
    ]
    
    print(f"\n{'Model':<10} {'Params':<15} {'Forward (ms)':<15} {'Tokens/sec':<15}")
    print("-" * 60)
    
    for name, config in configs:
        model = TernaryTransformer(config)
        stats = model.count_parameters()
        
        input_ids = np.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        seq_len = input_ids.shape[1]
        
        # Warmup
        _ = model.forward(input_ids)
        
        # Benchmark
        iterations = 5
        start = time.perf_counter()
        for _ in range(iterations):
            _ = model.forward(input_ids)
        avg_time = (time.perf_counter() - start) / iterations * 1000
        
        tokens_per_sec = seq_len / (avg_time / 1000)
        
        print(f"{name:<10} {stats['total_params']:,}     {avg_time:.2f}ms         {tokens_per_sec:.0f}")
    
    return True


def benchmark_generation():
    """Benchmark token generation speed."""
    print("\n" + "="*70)
    print("TOKEN GENERATION SPEED")
    print("="*70)
    
    from model.ternary_transformer import TernaryConfig, TernaryTransformer
    
    config = TernaryConfig(
        vocab_size=1000,
        hidden_size=256,
        num_heads=4,
        num_layers=2,
        intermediate_size=512
    )
    
    model = TernaryTransformer(config)
    
    print(f"\n  Generating tokens...")
    
    input_ids = np.array([[1, 2, 3]])
    
    # Warmup
    _ = model.generate(input_ids, max_new_tokens=5, temperature=1.0)
    
    # Benchmark different lengths
    for num_tokens in [10, 25, 50]:
        input_ids = np.array([[1, 2, 3]])
        
        start = time.perf_counter()
        output = model.generate(input_ids, max_new_tokens=num_tokens, temperature=1.0)
        total_time = time.perf_counter() - start
        
        tokens_per_sec = num_tokens / total_time
        ms_per_token = total_time / num_tokens * 1000
        
        print(f"\n  Generate {num_tokens} tokens:")
        print(f"    Total time:     {total_time*1000:.0f}ms")
        print(f"    Per token:      {ms_per_token:.1f}ms")
        print(f"    Tokens/second:  {tokens_per_sec:.1f}")
    
    return True


def main():
    print("="*70)
    print("SPEED BENCHMARKS")
    print("="*70)
    
    benchmark_matmul()
    benchmark_full_forward()
    benchmark_generation()
    
    print("\n" + "="*70)
    print("SPEED BENCHMARKS COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()

