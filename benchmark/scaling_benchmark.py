"""
SCALING BENCHMARK
=================

How does ternary scale with model size?

Zane - The Ian Index
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.ternary_transformer import TernaryConfig, TernaryTransformer, TernaryLinear


def test_scaling():
    """Test how ternary performance scales with size."""
    print("\n" + "="*70)
    print("SCALING BENCHMARK")
    print("="*70)
    
    # Different model sizes
    configs = [
        ("1M params",   128, 2, 2, 256),
        ("5M params",   256, 4, 4, 512),
        ("20M params",  512, 8, 6, 2048),
        ("50M params",  768, 12, 8, 3072),
        ("100M params", 1024, 16, 12, 4096),
    ]
    
    print(f"\n{'Model':<12} {'Hidden':<8} {'Heads':<6} {'Layers':<7} {'Ternary %':<10} {'Sparsity':<10} {'Memory':<12}")
    print("-" * 80)
    
    for name, hidden, heads, layers, intermediate in configs:
        config = TernaryConfig(
            vocab_size=10000,
            hidden_size=hidden,
            num_heads=heads,
            num_layers=layers,
            intermediate_size=intermediate
        )
        
        model = TernaryTransformer(config)
        stats = model.count_parameters()
        
        # Calculate sparsity
        total_zeros = 0
        total_weights = 0
        for block in model.blocks:
            for proj in [block.attention.q_proj, block.attention.k_proj,
                        block.attention.v_proj, block.attention.o_proj]:
                total_zeros += np.sum(proj.weights == 0)
                total_weights += proj.weights.size
            total_zeros += np.sum(block.mlp.up_proj.weights == 0)
            total_weights += block.mlp.up_proj.weights.size
            total_zeros += np.sum(block.mlp.down_proj.weights == 0)
            total_weights += block.mlp.down_proj.weights.size
        
        sparsity = total_zeros / total_weights
        
        # Memory calculation
        float_mem = stats['total_params'] * 4 / 1e6  # MB
        ternary_mem = stats['ternary_params'] * 0.25 / 1e6 + stats['float_params'] * 4 / 1e6
        
        print(f"{name:<12} {hidden:<8} {heads:<6} {layers:<7} {stats['ternary_percentage']:.1f}%      {sparsity:.1%}      {ternary_mem:.1f}MB")
    
    return True


def test_layer_scaling():
    """Test individual layer scaling."""
    print("\n" + "="*70)
    print("LAYER SIZE SCALING")
    print("="*70)
    
    sizes = [64, 128, 256, 512, 1024, 2048, 4096]
    
    print(f"\n{'Size':<10} {'Weights':<15} {'Non-zero':<12} {'Sparsity':<10} {'Memory (KB)':<12}")
    print("-" * 65)
    
    for size in sizes:
        layer = TernaryLinear(size, size, threshold=0.33)
        
        total = layer.weights.size
        nonzero = np.sum(layer.weights != 0)
        sparsity = layer.sparsity
        
        # Memory: 2 bits per weight
        memory_kb = total * 0.25 / 1024
        
        print(f"{size:<10} {total:,}       {nonzero:,}       {sparsity:.1%}      {memory_kb:.2f}")
    
    return True


def test_attention_scaling():
    """Test attention mechanism at different scales."""
    print("\n" + "="*70)
    print("ATTENTION SCALING")
    print("="*70)
    
    from model.ternary_transformer import TernaryAttention
    
    configs = [
        (128, 2, 8),
        (256, 4, 16),
        (512, 8, 32),
        (768, 12, 64),
        (1024, 16, 128),
    ]
    
    print(f"\n{'Hidden':<10} {'Heads':<8} {'Seq Len':<10} {'Time (ms)':<12} {'Output Var':<12}")
    print("-" * 55)
    
    for hidden, heads, seq_len in configs:
        attn = TernaryAttention(hidden, heads, threshold=0.33)
        
        x = np.random.randn(1, seq_len, hidden).astype(np.float32)
        
        # Warmup
        _ = attn.forward(x)
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(3):
            output = attn.forward(x)
        avg_time = (time.perf_counter() - start) / 3 * 1000
        
        output_var = np.var(output)
        
        print(f"{hidden:<10} {heads:<8} {seq_len:<10} {avg_time:.2f}        {output_var:.4f}")
    
    return True


def test_sequence_length_scaling():
    """Test how performance scales with sequence length."""
    print("\n" + "="*70)
    print("SEQUENCE LENGTH SCALING")
    print("="*70)
    
    config = TernaryConfig(
        vocab_size=1000,
        hidden_size=256,
        num_heads=4,
        num_layers=2,
        intermediate_size=512
    )
    
    model = TernaryTransformer(config)
    
    print(f"\n{'Seq Len':<10} {'Time (ms)':<12} {'Time/Token':<12} {'Logits OK':<10}")
    print("-" * 50)
    
    for seq_len in [4, 8, 16, 32, 64, 128]:
        input_ids = np.array([list(range(1, seq_len + 1))])
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(3):
            logits = model.forward(input_ids)
        avg_time = (time.perf_counter() - start) / 3 * 1000
        
        time_per_token = avg_time / seq_len
        logits_ok = not (np.any(np.isnan(logits)) or np.any(np.isinf(logits)))
        
        print(f"{seq_len:<10} {avg_time:.2f}        {time_per_token:.2f}         {'YES' if logits_ok else 'NO'}")
    
    return True


def main():
    print("="*70)
    print("SCALING BENCHMARKS")
    print("="*70)
    
    test_scaling()
    test_layer_scaling()
    test_attention_scaling()
    test_sequence_length_scaling()
    
    print("\n" + "="*70)
    print("SCALING BENCHMARKS COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()

