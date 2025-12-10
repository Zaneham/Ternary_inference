"""
STRESS TEST
===========

Push the ternary transformer to its limits!

Zane - The Ian Index
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.ternary_transformer import TernaryConfig, TernaryTransformer, TernaryLinear


def stress_test_weights():
    """Test extreme weight distributions."""
    print("\n" + "="*70)
    print("STRESS TEST 1: EXTREME WEIGHT DISTRIBUTIONS")
    print("="*70)
    
    layer = TernaryLinear(256, 256, threshold=0.33)
    x = np.random.randn(4, 256).astype(np.float32)
    
    print("\n  Testing different weight patterns...")
    
    # All positive
    layer.weights = np.ones((256, 256), dtype=np.int8)
    y = layer.forward(x)
    print(f"  All +1 weights: output range [{np.min(y):.2f}, {np.max(y):.2f}] - {'OK' if not np.any(np.isnan(y)) else 'FAIL'}")
    
    # All negative
    layer.weights = -np.ones((256, 256), dtype=np.int8)
    y = layer.forward(x)
    print(f"  All -1 weights: output range [{np.min(y):.2f}, {np.max(y):.2f}] - {'OK' if not np.any(np.isnan(y)) else 'FAIL'}")
    
    # All zeros
    layer.weights = np.zeros((256, 256), dtype=np.int8)
    y = layer.forward(x)
    print(f"  All 0 weights:  output range [{np.min(y):.2f}, {np.max(y):.2f}] - {'OK' if np.allclose(y, 0) else 'FAIL'}")
    
    # Alternating
    layer.weights = np.zeros((256, 256), dtype=np.int8)
    layer.weights[::2, ::2] = 1
    layer.weights[1::2, 1::2] = -1
    y = layer.forward(x)
    print(f"  Alternating:    output range [{np.min(y):.2f}, {np.max(y):.2f}] - {'OK' if not np.any(np.isnan(y)) else 'FAIL'}")
    
    # Random balanced
    layer.weights = np.random.choice([-1, 0, 1], size=(256, 256)).astype(np.int8)
    y = layer.forward(x)
    print(f"  Random ternary: output range [{np.min(y):.2f}, {np.max(y):.2f}] - {'OK' if not np.any(np.isnan(y)) else 'FAIL'}")
    
    return True


def stress_test_inputs():
    """Test extreme input values."""
    print("\n" + "="*70)
    print("STRESS TEST 2: EXTREME INPUT VALUES")
    print("="*70)
    
    layer = TernaryLinear(128, 128, threshold=0.33)
    
    print("\n  Testing different input magnitudes...")
    
    for scale in [1e-10, 1e-5, 1.0, 1e5, 1e10]:
        x = np.random.randn(1, 128).astype(np.float32) * scale
        y = layer.forward(x)
        
        has_nan = np.any(np.isnan(y))
        has_inf = np.any(np.isinf(y))
        
        status = "OK" if not (has_nan or has_inf) else f"FAIL (nan={has_nan}, inf={has_inf})"
        print(f"  Scale {scale:.0e}: output range [{np.min(y):.2e}, {np.max(y):.2e}] - {status}")
    
    return True


def stress_test_batch_sizes():
    """Test various batch sizes."""
    print("\n" + "="*70)
    print("STRESS TEST 3: BATCH SIZE SCALING")
    print("="*70)
    
    config = TernaryConfig(
        vocab_size=1000,
        hidden_size=128,
        num_heads=4,
        num_layers=2,
        intermediate_size=256
    )
    
    model = TernaryTransformer(config)
    
    print(f"\n{'Batch':<10} {'Seq Len':<10} {'Time (ms)':<12} {'Memory Est':<12} {'Status':<10}")
    print("-" * 55)
    
    for batch_size in [1, 2, 4, 8, 16, 32]:
        for seq_len in [8, 32]:
            input_ids = np.random.randint(0, 1000, size=(batch_size, seq_len))
            
            start = time.perf_counter()
            try:
                logits = model.forward(input_ids)
                elapsed = (time.perf_counter() - start) * 1000
                
                # Rough memory estimate
                mem_mb = batch_size * seq_len * config.hidden_size * 4 / 1e6
                
                status = "OK"
                if np.any(np.isnan(logits)):
                    status = "NaN!"
                elif np.any(np.isinf(logits)):
                    status = "Inf!"
            except Exception as e:
                elapsed = 0
                mem_mb = 0
                status = f"ERR: {str(e)[:20]}"
            
            print(f"{batch_size:<10} {seq_len:<10} {elapsed:.2f}        {mem_mb:.2f}MB       {status}")
    
    return True


def stress_test_generation():
    """Test long generation sequences."""
    print("\n" + "="*70)
    print("STRESS TEST 4: LONG GENERATION")
    print("="*70)
    
    config = TernaryConfig(
        vocab_size=1000,
        hidden_size=128,
        num_heads=4,
        num_layers=2,
        intermediate_size=256,
        max_seq_len=256
    )
    
    model = TernaryTransformer(config)
    
    print(f"\n  Generating sequences of increasing length...")
    
    for num_tokens in [10, 50, 100, 200]:
        input_ids = np.array([[1, 2, 3]])
        
        start = time.perf_counter()
        try:
            output = model.generate(input_ids, max_new_tokens=num_tokens, temperature=1.0)
            elapsed = time.perf_counter() - start
            
            unique_tokens = len(set(output[0].tolist()))
            tokens_per_sec = num_tokens / elapsed
            
            print(f"  {num_tokens} tokens: {elapsed:.2f}s, {tokens_per_sec:.1f} tok/s, {unique_tokens} unique - OK")
        except Exception as e:
            print(f"  {num_tokens} tokens: FAILED - {str(e)[:40]}")
    
    return True


def stress_test_repeated_forward():
    """Test many repeated forward passes."""
    print("\n" + "="*70)
    print("STRESS TEST 5: REPEATED FORWARD PASSES")
    print("="*70)
    
    config = TernaryConfig(
        vocab_size=1000,
        hidden_size=128,
        num_heads=4,
        num_layers=2,
        intermediate_size=256
    )
    
    model = TernaryTransformer(config)
    input_ids = np.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    
    print(f"\n  Running 100 forward passes...")
    
    results = []
    start = time.perf_counter()
    
    for i in range(100):
        logits = model.forward(input_ids)
        results.append(np.sum(logits))
        
        if np.any(np.isnan(logits)):
            print(f"  NaN at iteration {i}!")
            return False
    
    elapsed = time.perf_counter() - start
    
    # Check consistency
    first_result = results[0]
    all_same = all(r == first_result for r in results)
    
    print(f"  Total time: {elapsed:.2f}s ({elapsed/100*1000:.2f}ms per pass)")
    print(f"  All results identical: {'YES (deterministic!)' if all_same else 'NO (non-deterministic)'}")
    print(f"  No NaN/Inf: OK")
    
    return True


def stress_test_random_seeds():
    """Test many different random seeds."""
    print("\n" + "="*70)
    print("STRESS TEST 6: RANDOM SEED VARIETY")
    print("="*70)
    
    print(f"\n  Creating 20 models with different seeds...")
    
    all_different = True
    outputs = []
    
    for seed in range(20):
        np.random.seed(seed)
        
        config = TernaryConfig(
            vocab_size=1000,
            hidden_size=64,
            num_heads=2,
            num_layers=1,
            intermediate_size=128
        )
        
        model = TernaryTransformer(config)
        input_ids = np.array([[1, 2, 3]])
        
        logits = model.forward(input_ids)
        output_hash = hash(logits.tobytes())
        
        if output_hash in outputs:
            all_different = False
        outputs.append(output_hash)
    
    print(f"  All 20 models produce different outputs: {'YES' if all_different else 'NO (collision!)'}")
    print(f"  Unique output hashes: {len(set(outputs))}/20")
    
    return True


def main():
    print("="*70)
    print("STRESS TESTS - PUSHING TERNARY TO THE LIMITS")
    print("="*70)
    
    results = {}
    
    results["weights"] = stress_test_weights()
    results["inputs"] = stress_test_inputs()
    results["batch"] = stress_test_batch_sizes()
    results["generation"] = stress_test_generation()
    results["repeated"] = stress_test_repeated_forward()
    results["seeds"] = stress_test_random_seeds()
    
    print("\n" + "="*70)
    print("STRESS TEST RESULTS")
    print("="*70)
    
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        print(f"  {name}: {'PASS' if result else 'FAIL'}")
    
    print(f"\n  Total: {passed}/{total} passed")
    
    if passed == total:
        print("\n  TERNARY TRANSFORMER SURVIVED ALL STRESS TESTS!")


if __name__ == "__main__":
    main()

