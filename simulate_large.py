"""
LARGE MODEL SIMULATION
======================

Simulate ternary inference at 1B, 7B, 13B, 30B, 70B scale.

Author: Zane Hambly
"""

import numpy as np
import time
import sys

sys.path.insert(0, 'model')
from scalable_ternary import ScalableConfig, ScalableTernaryTransformer


def main():
    print("=" * 70)
    print("LARGE MODEL SIMULATION")
    print("Simulating ternary inference at scale")
    print("=" * 70)
    
    # Define model configs at different scales
    configs = {
        '1B': ScalableConfig(
            vocab_size=32000, hidden_size=2048, intermediate_size=5504,
            num_layers=24, num_heads=16, num_kv_heads=4, max_seq_len=2048,
            chunk_size=512
        ),
        '7B': ScalableConfig(
            vocab_size=32000, hidden_size=4096, intermediate_size=11008,
            num_layers=32, num_heads=32, num_kv_heads=8, max_seq_len=4096,
            chunk_size=1024
        ),
        '13B': ScalableConfig(
            vocab_size=32000, hidden_size=5120, intermediate_size=13824,
            num_layers=40, num_heads=40, num_kv_heads=10, max_seq_len=4096,
            chunk_size=1024
        ),
    }
    
    baseline_tps = None  # Tokens per second baseline
    
    for name, config in configs.items():
        print(f"\n{'='*60}")
        print(f"  {name} MODEL SIMULATION")
        print(f"{'='*60}")
        
        mem = config.estimate_memory()
        print(f"\n  Parameters:    {mem['total_params']:,}")
        print(f"  Float32:       {mem['float32_gb']:.1f} GB")
        print(f"  Ternary (i8):  {mem['ternary_gb']:.1f} GB") 
        print(f"  Packed (2b):   {mem['packed_gb']:.1f} GB")
        print(f"  Compression:   {mem['float32_gb']/mem['packed_gb']:.0f}x")
        
        # Only create the 1B model (fits in RAM easily)
        if name == '1B':
            print(f"\n  Creating actual {name} model...")
            np.random.seed(42)
            model = ScalableTernaryTransformer(config)
            
            # Force weight initialization
            _ = model.forward(np.array([[1]]))
            
            input_ids = np.array([[1, 2, 3, 4, 5]])
            
            # Benchmark forward pass
            print("  Running forward pass...")
            start = time.time()
            for _ in range(3):
                logits = model.forward(input_ids)
            fwd_time = (time.time() - start) / 3
            print(f"  Forward time:  {fwd_time*1000:.1f}ms")
            
            # Benchmark generation
            print("  Generating 30 tokens...")
            model.kv_cache.clear()
            start = time.time()
            output = model.generate(input_ids, max_new_tokens=30, temperature=1.0)
            gen_time = time.time() - start
            baseline_tps = 30 / gen_time
            print(f"  Generation:    {gen_time*1000:.0f}ms ({baseline_tps:.1f} tok/s)")
            print(f"  Output:        {output[0][:10].tolist()}...")
            
        else:
            # Estimate based on 1B baseline
            scale = mem['total_params'] / 1_000_000_000
            est_tps = baseline_tps / scale if baseline_tps else 50 / scale
            
            print(f"\n  [SIMULATED - based on 1B scaling]")
            print(f"  Estimated forward: {50 * scale:.0f}ms")
            print(f"  Estimated tok/s:   {est_tps:.1f}")
            print(f"  Time for 100 tokens: {100/est_tps:.1f}s")
    
    # Add larger model estimates
    print(f"\n{'='*60}")
    print("  30B AND 70B ESTIMATES")
    print(f"{'='*60}")
    
    for name, params, float_gb in [('30B', 30, 120), ('70B', 70, 280)]:
        packed_gb = params * 2 / 8  # 2 bits per parameter
        scale = params
        est_tps = baseline_tps / scale if baseline_tps else 50 / scale
        
        print(f"\n  {name} Model:")
        print(f"    Parameters:   {params}B")
        print(f"    Float32:      {float_gb} GB")
        print(f"    Packed:       {packed_gb:.1f} GB")
        print(f"    Est. tok/s:   {est_tps:.2f}")
        print(f"    Fits on GPU:  {'YES (24GB)' if packed_gb < 24 else 'NO'}")
    
    # Summary table
    print(f"\n{'='*70}")
    print("SCALING SUMMARY")
    print(f"{'='*70}")
    print()
    print("Model     Params      Float32     Packed      Fits 24GB GPU?")
    print("-" * 65)
    
    all_models = [
        ('1B', 1, 4, 0.25),
        ('7B', 7, 28, 1.75),
        ('13B', 13, 52, 3.25),
        ('30B', 30, 120, 7.5),
        ('70B', 70, 280, 17.5),
        ('405B', 405, 1620, 101),
    ]
    
    for name, params, float_gb, packed_gb in all_models:
        fits = "YES" if packed_gb < 24 else "NO"
        print(f"{name:8}  {params:>5}B      {float_gb:>6} GB    {packed_gb:>5.1f} GB     {fits}")
    
    print()
    print("=" * 70)
    print("KEY INSIGHT: Ternary quantisation makes 70B fit on ONE 24GB GPU!")
    print("             That's 16x compression with ZERO multiplications!")
    print("=" * 70)


if __name__ == "__main__":
    main()

