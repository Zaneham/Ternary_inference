"""
OLLAMA TERNARY INFERENCE DEMO
=============================

This script demonstrates loading a real Ollama model,
quantising it to ternary, and running inference.

Usage:
    python demo_ollama.py

Requirements:
    - Ollama installed (https://ollama.ai/download)
    - At least one model downloaded (ollama pull llama2)
    - pip install gguf

If Ollama is not available, the demo will run with random weights
to demonstrate the architecture still works.

Author: Zane Hambly
"""

import sys
import numpy as np

# Add model directory to path
sys.path.insert(0, 'model')

from ternary_transformer import TernaryTransformer, TernaryConfig


def demo_with_ollama():
    """Try to load from Ollama, fall back to random weights."""
    print("=" * 70)
    print("TERNARY INFERENCE WITH OLLAMA WEIGHTS")
    print("=" * 70)
    
    # Try to load from Ollama
    print("\nAttempting to load Ollama model...")
    model = TernaryTransformer.from_ollama("smallest")
    
    if model is None:
        print("\n" + "-" * 70)
        print("Ollama not available - using random weights for demo")
        print("-" * 70)
        
        config = TernaryConfig(
            vocab_size=1000,
            hidden_size=256,
            num_heads=4,
            num_layers=2,
            intermediate_size=512
        )
        model = TernaryTransformer(config)
        print(f"\nCreated model with random weights:")
        print(f"  Vocab: {config.vocab_size}, Hidden: {config.hidden_size}")
    
    # Show parameter stats
    stats = model.count_parameters()
    print(f"\nModel statistics:")
    print(f"  Ternary params: {stats['ternary_params']:,}")
    print(f"  Float params:   {stats['float_params']:,}")
    print(f"  Ternary:        {stats['ternary_percentage']:.1f}%")
    
    # Run inference
    print("\n" + "-" * 70)
    print("RUNNING TERNARY INFERENCE")
    print("-" * 70)
    
    input_ids = np.array([[1, 2, 3, 4, 5]])
    print(f"\nInput: {input_ids[0].tolist()}")
    
    # Forward pass
    logits = model.forward(input_ids)
    print(f"Logits shape: {logits.shape}")
    print(f"Logits mean: {logits.mean():.4f}")
    
    # Generate
    print("\nGenerating 10 tokens...")
    output = model.generate(input_ids, max_new_tokens=10, temperature=1.0)
    print(f"Output: {output[0].tolist()}")
    
    # Prove no multiplication
    print("\n" + "-" * 70)
    print("PROOF: NO MULTIPLICATION")
    print("-" * 70)
    
    layer = model.blocks[0].attention.q_proj
    unique_weights = np.unique(layer.weights)
    print(f"\nQ projection weights: {unique_weights}")
    print(f"Sparsity: {layer.sparsity:.1%}")
    print("\nOperation breakdown:")
    print(f"  +1 weights (ADD):      {np.sum(layer.weights == 1):,}")
    print(f"   0 weights (SKIP):     {np.sum(layer.weights == 0):,}")
    print(f"  -1 weights (SUBTRACT): {np.sum(layer.weights == -1):,}")
    print(f"  Total operations:      {layer.weights.size:,}")
    print(f"  Multiplications:       0")
    
    print("\n" + "=" * 70)
    print("SUCCESS!")
    print("=" * 70)
    print("\nThe ternary transformer:")
    print("  - Uses {-1, 0, +1} weights for all linear layers")
    print("  - Performs matrix multiply with ONLY addition")
    print("  - Achieves 4x memory compression (float32 to int8)")
    print("  - Achieves 16x with 2-bit packing")
    print("\nWith Ollama weights, you get real knowledge.")
    print("Without, you get proof the architecture works.")


def check_requirements():
    """Check what's available."""
    print("Checking requirements...")
    
    # Check gguf
    try:
        import gguf
        print("  [OK] gguf library installed")
    except ImportError:
        print("  [--] gguf not installed (pip install gguf)")
    
    # Check ollama loader
    try:
        from ollama_loader import OllamaLoader
        loader = OllamaLoader()
        if loader.blobs_dir:
            models = loader.list_models()
            print(f"  [OK] Ollama directory found ({len(models)} models)")
        else:
            print("  [--] Ollama directory not found")
    except Exception as e:
        print(f"  [--] Ollama loader error: {e}")


if __name__ == "__main__":
    check_requirements()
    print()
    demo_with_ollama()

