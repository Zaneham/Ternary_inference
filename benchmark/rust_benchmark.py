"""
RUST CORE BENCHMARK
===================

Benchmark the Rust ternary core against Python/NumPy.

Note: Current Rust implementation is naive loops.
NumPy uses highly optimised BLAS (MKL/OpenBLAS) with SIMD.
Future work: Add AVX-512 intrinsics to Rust for true speedup.

Author: Zane Hambly
"""

import sys
import time
import numpy as np

# Add project root to path for ternary_core.pyd
sys.path.insert(0, '.')

def benchmark_rust():
    """Benchmark Rust ternary operations."""
    try:
        import ternary_core
    except ImportError:
        print("Rust core not built. Run:")
        print("  cd ternary-core")
        print("  cargo build --release")
        print("  copy target/release/ternary_core.dll ../ternary_core.pyd")
        return False
    
    print("=" * 60)
    print("RUST TERNARY CORE BENCHMARK")
    print("=" * 60)
    
    print(f"\nAvailable functions: {[x for x in dir(ternary_core) if not x.startswith('_')]}")
    
    # Test TernaryMatrix
    print("\n--- TernaryMatrix ---")
    w = ternary_core.TernaryMatrix.random(1024, 1024, 0.67)
    print(f"Shape: {w.shape()}")
    print(f"Sparsity: {w.get_sparsity():.1%}")
    
    # Test quantization
    print("\n--- Quantization ---")
    weights = [float(x) for x in np.random.randn(1000)]
    ternary = ternary_core.quantize_to_ternary(weights, 67.0)
    unique = set(ternary)
    print(f"Input: 1000 floats")
    print(f"Output: {len(ternary)} trits")
    print(f"Unique values: {sorted(unique)}")
    
    # Test compression ratio
    print("\n--- Compression ---")
    for params in [1_000_000_000, 7_000_000_000, 70_000_000_000]:
        f32, i8, packed = ternary_core.compression_ratio(params)
        print(f"  {params/1e9:.0f}B params: {f32:.1f} GB -> {packed:.1f} GB (16x)")
    
    # Benchmark matmul
    print("\n--- Matrix Multiply Benchmark ---")
    print("(Note: NumPy uses MKL/OpenBLAS with SIMD)")
    print("(Rust uses naive loops - AVX-512 TODO)")
    print()
    
    sizes = [(128, 128), (256, 256), (512, 512)]
    
    for rows, cols in sizes:
        # Prepare data
        x = np.random.randn(rows, cols).astype(np.float32)
        x_list = x.flatten().tolist()
        w = ternary_core.TernaryMatrix.random(cols, cols, 0.67)
        
        # NumPy baseline
        w_np = np.random.choice([-1, 0, 1], (cols, cols), p=[0.165, 0.67, 0.165]).astype(np.int8)
        pos = (w_np == 1).astype(np.float32)
        neg = (w_np == -1).astype(np.float32)
        
        # Warmup
        _ = ternary_core.ternary_matmul(x_list, rows, cols, w)
        _ = x @ pos - x @ neg
        
        # Benchmark
        n_iters = 5
        
        start = time.time()
        for _ in range(n_iters):
            _ = ternary_core.ternary_matmul(x_list, rows, cols, w)
        rust_time = (time.time() - start) / n_iters * 1000
        
        start = time.time()
        for _ in range(n_iters):
            _ = x @ pos - x @ neg
        numpy_time = (time.time() - start) / n_iters * 1000
        
        print(f"  {rows}x{cols}: Rust={rust_time:.1f}ms, NumPy={numpy_time:.1f}ms")
    
    print("\n" + "=" * 60)
    print("RUST CORE: OPERATIONAL")
    print("=" * 60)
    print("\nNext steps for speedup:")
    print("  1. Add AVX-512 intrinsics")
    print("  2. Use zero-copy numpy arrays (numpy crate)")
    print("  3. Keep matrices in Rust, only return results")
    
    return True


if __name__ == "__main__":
    success = benchmark_rust()
    sys.exit(0 if success else 1)

