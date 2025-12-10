"""
SCALABLE TERNARY TRANSFORMER BENCHMARK
======================================

Tests the scalable architecture designed for 70B+ models.

Author: Zane Hambly
"""

import sys
import time
import numpy as np

sys.path.insert(0, 'model')

from scalable_ternary import ScalableTernaryTransformer, ScalableConfig


def test_kv_cache_speedup():
    """Test KV cache provides speedup."""
    print("\n" + "=" * 60)
    print("TEST 1: KV CACHE SPEEDUP")
    print("=" * 60)
    
    config = ScalableConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_layers=4,
        num_heads=8,
        num_kv_heads=2,
        max_seq_len=256,
    )
    model = ScalableTernaryTransformer(config)
    
    input_ids = np.array([[1, 2, 3, 4, 5]])
    n_tokens = 20
    
    # Without cache (simulate by clearing each time)
    start = time.time()
    for _ in range(3):
        model.kv_cache.clear()
        current = input_ids.copy()
        for i in range(n_tokens):
            logits = model.forward(current, use_cache=False)
            next_token = np.argmax(logits[:, -1, :], axis=-1, keepdims=True)
            current = np.concatenate([current, next_token], axis=1)
    no_cache_time = (time.time() - start) / 3
    
    # With cache
    start = time.time()
    for _ in range(3):
        model.kv_cache.clear()
        _ = model.generate(input_ids, max_new_tokens=n_tokens, temperature=0.0001)
    cache_time = (time.time() - start) / 3
    
    speedup = no_cache_time / cache_time if cache_time > 0 else 0
    
    print(f"  Without cache: {no_cache_time:.2f}s ({n_tokens/no_cache_time:.1f} tok/s)")
    print(f"  With cache:    {cache_time:.2f}s ({n_tokens/cache_time:.1f} tok/s)")
    print(f"  Speedup:       {speedup:.1f}x")
    
    valid = speedup > 1.5  # Should be at least 1.5x faster
    print(f"\n  Result: {'PASS' if valid else 'FAIL'}")
    return valid


def test_chunked_computation():
    """Test chunked computation works correctly."""
    print("\n" + "=" * 60)
    print("TEST 2: CHUNKED COMPUTATION")
    print("=" * 60)
    
    from scalable_ternary import ChunkedTernaryLinear
    
    # Create layer with different chunk sizes
    in_feat, out_feat = 256, 1024
    
    layer_full = ChunkedTernaryLinear(in_feat, out_feat, chunk_size=out_feat)
    layer_chunked = ChunkedTernaryLinear(in_feat, out_feat, chunk_size=128)
    
    # Copy weights
    layer_chunked.weights._data = layer_full.weights.data.copy()
    
    x = np.random.randn(2, 8, in_feat).astype(np.float32)
    
    out_full = layer_full.forward(x)
    out_chunked = layer_chunked.forward(x)
    
    match = np.allclose(out_full, out_chunked, atol=1e-5)
    
    print(f"  Full computation shape:    {out_full.shape}")
    print(f"  Chunked computation shape: {out_chunked.shape}")
    print(f"  Max difference: {np.max(np.abs(out_full - out_chunked)):.2e}")
    print(f"  Match: {match}")
    
    print(f"\n  Result: {'PASS' if match else 'FAIL'}")
    return match


def test_memory_scaling():
    """Test memory estimates for different model sizes."""
    print("\n" + "=" * 60)
    print("TEST 3: MEMORY SCALING")
    print("=" * 60)
    
    configs = [
        ("1B", ScalableConfig(hidden_size=2048, num_layers=24, num_heads=16, num_kv_heads=4)),
        ("7B", ScalableConfig(hidden_size=4096, num_layers=32, num_heads=32, num_kv_heads=8)),
        ("13B", ScalableConfig(hidden_size=5120, num_layers=40, num_heads=40, num_kv_heads=10)),
        ("70B", ScalableConfig(hidden_size=8192, num_layers=80, num_heads=64, num_kv_heads=8)),
    ]
    
    print(f"  {'Model':<8} {'Params':<15} {'Float32':<10} {'Ternary':<10} {'Packed':<10}")
    print("  " + "-" * 55)
    
    all_valid = True
    for name, config in configs:
        mem = config.estimate_memory()
        print(f"  {name:<8} {mem['total_params']:>12,}   {mem['float32_gb']:>7.1f} GB  {mem['ternary_gb']:>7.1f} GB  {mem['packed_gb']:>7.1f} GB")
        
        # Verify compression ratio
        compression = mem['float32_gb'] / mem['packed_gb']
        if compression < 15 or compression > 17:
            all_valid = False
    
    print(f"\n  Compression ratio: 16x (consistent)")
    print(f"\n  Result: {'PASS' if all_valid else 'FAIL'}")
    return all_valid


def test_sharding_ready():
    """Test model can be conceptually sharded."""
    print("\n" + "=" * 60)
    print("TEST 4: SHARDING ARCHITECTURE")
    print("=" * 60)
    
    config = ScalableConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_layers=8,
        num_heads=8,
        num_kv_heads=2,
        num_shards=4,
    )
    
    mem = config.estimate_memory()
    params_per_shard = mem['params_per_shard']
    total = mem['total_params']
    
    print(f"  Total layers: {config.num_layers}")
    print(f"  Num shards: {config.num_shards}")
    print(f"  Layers per shard: {config.num_layers // config.num_shards}")
    print(f"  Params per shard: {params_per_shard:,}")
    print(f"  Total params: {total:,}")
    
    # Verify even distribution
    valid = params_per_shard * config.num_shards == total
    print(f"  Even distribution: {valid}")
    
    print(f"\n  Result: {'PASS' if valid else 'FAIL'}")
    return valid


def test_generation_quality():
    """Test generation produces diverse, valid output."""
    print("\n" + "=" * 60)
    print("TEST 5: GENERATION QUALITY")
    print("=" * 60)
    
    config = ScalableConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_layers=4,
        num_heads=8,
        num_kv_heads=2,
    )
    
    # Initialize model with fixed seed for weights
    np.random.seed(999)
    model = ScalableTernaryTransformer(config)
    
    # Force weight initialization by doing a forward pass
    _ = model.forward(np.array([[1]]))
    
    input_ids = np.array([[1, 2, 3]])
    
    # Test determinism (now weights are already initialized)
    np.random.seed(42)
    model.kv_cache.clear()
    out1 = model.generate(input_ids.copy(), max_new_tokens=10, temperature=1.0)
    np.random.seed(42)
    model.kv_cache.clear()
    out2 = model.generate(input_ids.copy(), max_new_tokens=10, temperature=1.0)
    
    deterministic = np.array_equal(out1, out2)
    
    # Test diversity
    np.random.seed(123)
    model.kv_cache.clear()
    out3 = model.generate(input_ids.copy(), max_new_tokens=10, temperature=1.0)
    diverse = not np.array_equal(out1, out3)
    
    # Test non-degenerate
    unique = len(set(out1[0].tolist()))
    non_degenerate = unique > 3
    
    print(f"  Generated: {out1[0].tolist()}")
    print(f"  Deterministic (same seed): {deterministic}")
    print(f"  Diverse (diff seed): {diverse}")
    print(f"  Unique tokens: {unique}")
    print(f"  Non-degenerate: {non_degenerate}")
    
    valid = deterministic and diverse and non_degenerate
    print(f"\n  Result: {'PASS' if valid else 'FAIL'}")
    return valid


def test_numerical_stability():
    """Test numerical stability at scale."""
    print("\n" + "=" * 60)
    print("TEST 6: NUMERICAL STABILITY")
    print("=" * 60)
    
    config = ScalableConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_layers=4,
        num_heads=8,
        num_kv_heads=2,
    )
    model = ScalableTernaryTransformer(config)
    
    all_stable = True
    
    for scale in [0.01, 1.0, 100.0]:
        model.embeddings = np.random.randn(
            config.vocab_size, config.hidden_size
        ).astype(np.float32) * scale
        
        input_ids = np.array([[1, 2, 3, 4, 5]])
        logits = model.forward(input_ids)
        
        has_nan = np.isnan(logits).any()
        has_inf = np.isinf(logits).any()
        stable = not has_nan and not has_inf
        all_stable = all_stable and stable
        
        print(f"  Scale {scale}: NaN={has_nan}, Inf={has_inf} - {'OK' if stable else 'FAIL'}")
    
    print(f"\n  Result: {'PASS' if all_stable else 'FAIL'}")
    return all_stable


def test_throughput():
    """Test token generation throughput."""
    print("\n" + "=" * 60)
    print("TEST 7: THROUGHPUT")
    print("=" * 60)
    
    config = ScalableConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_layers=4,
        num_heads=8,
        num_kv_heads=2,
    )
    model = ScalableTernaryTransformer(config)
    
    input_ids = np.array([[1, 2, 3, 4, 5]])
    
    # Warmup
    _ = model.generate(input_ids, max_new_tokens=5)
    
    # Benchmark
    n_tokens = 50
    start = time.time()
    _ = model.generate(input_ids, max_new_tokens=n_tokens, temperature=1.0)
    elapsed = time.time() - start
    
    throughput = n_tokens / elapsed
    
    print(f"  Generated: {n_tokens} tokens")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {throughput:.1f} tokens/sec")
    
    # Should be at least 50 tok/s on modern CPU
    valid = throughput > 50
    print(f"\n  Result: {'PASS' if valid else 'FAIL'}")
    return valid


def main():
    print("=" * 60)
    print("SCALABLE TERNARY TRANSFORMER BENCHMARK")
    print("7 Tests for 70B+ Model Support")
    print("=" * 60)
    
    tests = [
        ("KV Cache Speedup", test_kv_cache_speedup),
        ("Chunked Computation", test_chunked_computation),
        ("Memory Scaling", test_memory_scaling),
        ("Sharding Architecture", test_sharding_ready),
        ("Generation Quality", test_generation_quality),
        ("Numerical Stability", test_numerical_stability),
        ("Throughput", test_throughput),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    passed = 0
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {name:30s} {status}")
        if result:
            passed += 1
    
    print(f"\n  Total: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n  ALL TESTS PASSED!")
        print("  SCALABLE TERNARY TRANSFORMER IS READY FOR 70B+!")
    else:
        print(f"\n  {len(results) - passed} tests failed")
    
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

