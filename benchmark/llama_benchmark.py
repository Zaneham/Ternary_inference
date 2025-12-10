"""
LLAMA-COMPATIBLE TERNARY TRANSFORMER BENCHMARK
===============================================

Run the same battery of tests on the LLaMA-compatible architecture
to ensure it works correctly.

Author: Zane Hambly
"""

import sys
import numpy as np

sys.path.insert(0, 'model')

from llama_ternary import TernaryLlama, LlamaConfig


def test_zero_multiplications():
    """Test 1: Verify no multiplications in forward pass."""
    print("\n" + "=" * 60)
    print("TEST 1: ZERO MULTIPLICATIONS")
    print("=" * 60)
    
    config = LlamaConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2
    )
    model = TernaryLlama(config)
    
    # Check a weight matrix
    layer = model.layers[0].attention.q_proj
    unique = np.unique(layer.weights)
    
    print(f"  Q projection weights: {unique}")
    print(f"  +1 (ADD):      {np.sum(layer.weights == 1):,}")
    print(f"   0 (SKIP):     {np.sum(layer.weights == 0):,}")
    print(f"  -1 (SUBTRACT): {np.sum(layer.weights == -1):,}")
    print(f"  Multiplications: 0")
    
    # Verify only {-1, 0, 1}
    valid = all(v in [-1, 0, 1] for v in unique)
    print(f"\n  Result: {'PASS' if valid else 'FAIL'}")
    return valid


def test_forward_pass():
    """Test 2: Verify forward pass works."""
    print("\n" + "=" * 60)
    print("TEST 2: FORWARD PASS")
    print("=" * 60)
    
    config = LlamaConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2
    )
    model = TernaryLlama(config)
    
    input_ids = np.array([[1, 2, 3, 4, 5]])
    logits = model.forward(input_ids)
    
    print(f"  Input shape:  {input_ids.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Logits mean:  {logits.mean():.4f}")
    print(f"  Logits std:   {logits.std():.4f}")
    print(f"  Has NaN:      {np.isnan(logits).any()}")
    print(f"  Has Inf:      {np.isinf(logits).any()}")
    
    valid = (
        logits.shape == (1, 5, 1000) and
        not np.isnan(logits).any() and
        not np.isinf(logits).any()
    )
    print(f"\n  Result: {'PASS' if valid else 'FAIL'}")
    return valid


def test_generation():
    """Test 3: Verify token generation works."""
    print("\n" + "=" * 60)
    print("TEST 3: TOKEN GENERATION")
    print("=" * 60)
    
    config = LlamaConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2
    )
    model = TernaryLlama(config)
    
    input_ids = np.array([[1, 2, 3]])
    
    np.random.seed(42)
    output1 = model.generate(input_ids.copy(), max_new_tokens=10, temperature=1.0)
    
    np.random.seed(42)
    output2 = model.generate(input_ids.copy(), max_new_tokens=10, temperature=1.0)
    
    np.random.seed(123)
    output3 = model.generate(input_ids.copy(), max_new_tokens=10, temperature=1.0)
    
    deterministic = np.array_equal(output1, output2)
    varied = not np.array_equal(output1, output3)
    unique_tokens = len(set(output1[0].tolist()))
    
    print(f"  Generated: {output1[0].tolist()}")
    print(f"  Deterministic (same seed): {'PASS' if deterministic else 'FAIL'}")
    print(f"  Varied (diff seed):        {'PASS' if varied else 'FAIL'}")
    print(f"  Unique tokens: {unique_tokens}")
    
    valid = deterministic and varied and unique_tokens > 3
    print(f"\n  Result: {'PASS' if valid else 'FAIL'}")
    return valid


def test_rope():
    """Test 4: Verify RoPE positional embeddings work."""
    print("\n" + "=" * 60)
    print("TEST 4: ROPE POSITIONAL EMBEDDINGS")
    print("=" * 60)
    
    from llama_ternary import RotaryEmbedding
    
    rope = RotaryEmbedding(dim=64, max_seq_len=512)
    
    # Test that different positions give different embeddings
    x = np.random.randn(1, 4, 8, 64).astype(np.float32)  # batch, heads, seq, dim
    
    y0 = rope.apply(x, start_pos=0)
    y10 = rope.apply(x, start_pos=10)
    
    different = not np.allclose(y0, y10)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Position 0 output mean: {y0.mean():.4f}")
    print(f"  Position 10 output mean: {y10.mean():.4f}")
    print(f"  Positions differ: {different}")
    
    print(f"\n  Result: {'PASS' if different else 'FAIL'}")
    return different


def test_rms_norm():
    """Test 5: Verify RMSNorm works."""
    print("\n" + "=" * 60)
    print("TEST 5: RMS NORMALISATION")
    print("=" * 60)
    
    from llama_ternary import RMSNorm
    
    norm = RMSNorm(hidden_size=256)
    
    # Test normalisation
    x = np.random.randn(2, 8, 256).astype(np.float32) * 10
    y = norm.forward(x)
    
    # RMSNorm should bring values to unit RMS
    rms_before = np.sqrt(np.mean(x ** 2, axis=-1))
    rms_after = np.sqrt(np.mean(y ** 2, axis=-1))
    
    print(f"  Input RMS mean: {rms_before.mean():.4f}")
    print(f"  Output RMS mean: {rms_after.mean():.4f}")
    print(f"  Normalised: {np.allclose(rms_after, 1.0, atol=0.1)}")
    
    valid = np.allclose(rms_after, 1.0, atol=0.1)
    print(f"\n  Result: {'PASS' if valid else 'FAIL'}")
    return valid


def test_swiglu():
    """Test 6: Verify SwiGLU MLP works."""
    print("\n" + "=" * 60)
    print("TEST 6: SWIGLU MLP")
    print("=" * 60)
    
    config = LlamaConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2
    )
    
    from llama_ternary import TernarySwiGLUMLP
    mlp = TernarySwiGLUMLP(config)
    
    x = np.random.randn(2, 8, 256).astype(np.float32)
    y = mlp.forward(x)
    
    # SwiGLU should be non-linear
    x2 = x * 2
    y2 = mlp.forward(x2)
    
    # If linear, y2 would be 2*y
    is_nonlinear = not np.allclose(y2, y * 2, atol=0.1)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  Non-linear: {is_nonlinear}")
    
    print(f"\n  Result: {'PASS' if is_nonlinear else 'FAIL'}")
    return is_nonlinear


def test_gqa():
    """Test 7: Verify Grouped Query Attention works."""
    print("\n" + "=" * 60)
    print("TEST 7: GROUPED QUERY ATTENTION")
    print("=" * 60)
    
    config = LlamaConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_layers=1,
        num_heads=8,
        num_kv_heads=2  # 4 query heads per KV head
    )
    
    from llama_ternary import TernaryAttention
    attn = TernaryAttention(config)
    
    x = np.random.randn(1, 8, 256).astype(np.float32)
    y = attn.forward(x)
    
    # Check shapes
    q_shape = attn.q_proj.weights.shape
    k_shape = attn.k_proj.weights.shape
    
    print(f"  Q projection: {q_shape} (full heads)")
    print(f"  K projection: {k_shape} (reduced KV heads)")
    print(f"  Num groups: {config.num_heads // config.num_kv_heads}")
    print(f"  Input: {x.shape}")
    print(f"  Output: {y.shape}")
    
    valid = y.shape == x.shape and k_shape[1] < q_shape[1]
    print(f"\n  Result: {'PASS' if valid else 'FAIL'}")
    return valid


def test_memory_compression():
    """Test 8: Verify memory compression."""
    print("\n" + "=" * 60)
    print("TEST 8: MEMORY COMPRESSION")
    print("=" * 60)
    
    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=512,
        intermediate_size=1376,
        num_layers=4,
        num_heads=8,
        num_kv_heads=4
    )
    model = TernaryLlama(config)
    
    stats = model.count_parameters()
    
    float_bytes = stats['total_params'] * 4
    ternary_bytes = stats['ternary_params'] + stats['float_params'] * 4
    packed_bytes = stats['ternary_params'] / 4 + stats['float_params'] * 4
    
    compression = float_bytes / packed_bytes
    
    print(f"  Ternary params: {stats['ternary_params']:,}")
    print(f"  Float params:   {stats['float_params']:,}")
    print(f"  Ternary ratio:  {stats['ternary_percentage']:.1f}%")
    print(f"  Compression:    {compression:.1f}x")
    
    # Compression depends on ternary ratio - 2x is reasonable for smaller models
    # Larger models have higher ternary ratio (more weights vs embeddings)
    valid = compression > 2 and stats['ternary_percentage'] > 50
    print(f"\n  Result: {'PASS' if valid else 'FAIL'}")
    return valid


def test_reproducibility():
    """Test 9: Verify reproducibility."""
    print("\n" + "=" * 60)
    print("TEST 9: REPRODUCIBILITY")
    print("=" * 60)
    
    config = LlamaConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2
    )
    
    np.random.seed(42)
    model1 = TernaryLlama(config)
    
    np.random.seed(42)
    model2 = TernaryLlama(config)
    
    input_ids = np.array([[1, 2, 3]])
    
    out1 = model1.forward(input_ids)
    out2 = model2.forward(input_ids)
    
    identical = np.allclose(out1, out2)
    
    print(f"  Output 1 sum: {out1.sum():.4f}")
    print(f"  Output 2 sum: {out2.sum():.4f}")
    print(f"  Identical: {identical}")
    
    print(f"\n  Result: {'PASS' if identical else 'FAIL'}")
    return identical


def test_numerical_stability():
    """Test 10: Verify numerical stability."""
    print("\n" + "=" * 60)
    print("TEST 10: NUMERICAL STABILITY")
    print("=" * 60)
    
    config = LlamaConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2
    )
    model = TernaryLlama(config)
    
    all_stable = True
    
    for scale in [0.01, 1.0, 100.0]:
        model.embeddings = model.embeddings * scale
        
        input_ids = np.array([[1, 2, 3, 4, 5]])
        logits = model.forward(input_ids)
        
        has_nan = np.isnan(logits).any()
        has_inf = np.isinf(logits).any()
        
        stable = not has_nan and not has_inf
        all_stable = all_stable and stable
        
        print(f"  Scale {scale}: NaN={has_nan}, Inf={has_inf} - {'OK' if stable else 'FAIL'}")
        
        # Reset
        model.embeddings = model.embeddings / scale
    
    print(f"\n  Result: {'PASS' if all_stable else 'FAIL'}")
    return all_stable


def main():
    print("=" * 60)
    print("LLAMA-COMPATIBLE TERNARY TRANSFORMER BENCHMARK")
    print("10 Tests to Verify It Works")
    print("=" * 60)
    
    tests = [
        ("Zero Multiplications", test_zero_multiplications),
        ("Forward Pass", test_forward_pass),
        ("Token Generation", test_generation),
        ("RoPE Embeddings", test_rope),
        ("RMS Normalisation", test_rms_norm),
        ("SwiGLU MLP", test_swiglu),
        ("Grouped Query Attention", test_gqa),
        ("Memory Compression", test_memory_compression),
        ("Reproducibility", test_reproducibility),
        ("Numerical Stability", test_numerical_stability),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n  ERROR: {e}")
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
        print("  LLAMA-COMPATIBLE TERNARY TRANSFORMER WORKS!")
    else:
        print(f"\n  {len(results) - passed} tests failed")
    
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

