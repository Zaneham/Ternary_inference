"""
QUICK OLLAMA TERNARY DEMO
=========================

Loads the non-quantised parts of an Ollama model and demonstrates
ternary inference with the available weights.

This is a quick demo - full Q4_K dequantisation is slower.

Author: Zane Hambly
"""

import sys
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, 'model')


def main():
    print("=" * 70)
    print("QUICK OLLAMA TERNARY DEMO")
    print("=" * 70)
    
    # Find Ollama models
    blobs_dir = Path(os.environ.get('USERPROFILE', '')) / '.ollama' / 'models' / 'blobs'
    if not blobs_dir.exists():
        blobs_dir = Path.home() / '.ollama' / 'models' / 'blobs'
    
    if not blobs_dir.exists():
        print("\nOllama not found. Install from https://ollama.ai/download")
        return
    
    models = [(f, f.stat().st_size / 1e9) for f in blobs_dir.iterdir() 
              if f.is_file() and f.stat().st_size > 1e9]
    models.sort(key=lambda x: x[1])
    
    print(f"\nFound {len(models)} Ollama models")
    
    # Load the smallest decoder model
    from gguf import GGUFReader
    
    target = models[0][0]
    print(f"Loading: {target.name[:40]}... ({models[0][1]:.1f} GB)")
    
    reader = GGUFReader(str(target))
    
    # Extract architecture info
    metadata = {}
    for field in reader.fields.values():
        try:
            value = field.parts[-1].tolist() if hasattr(field.parts[-1], 'tolist') else field.parts[-1]
            if isinstance(value, list) and len(value) == 1:
                value = value[0]
            metadata[field.name] = value
        except:
            pass
    
    # Count tensor types
    type_counts = {}
    f32_count = 0
    f16_count = 0
    quant_count = 0
    total_elements = 0
    
    for tensor in reader.tensors:
        t = tensor.tensor_type
        type_counts[t] = type_counts.get(t, 0) + 1
        total_elements += tensor.n_elements
        
        if t == 0:
            f32_count += 1
        elif t == 1:
            f16_count += 1
        else:
            quant_count += 1
    
    print(f"\nModel statistics:")
    print(f"  Total tensors: {len(reader.tensors)}")
    print(f"  F32 tensors: {f32_count}")
    print(f"  F16 tensors: {f16_count}")
    print(f"  Quantised: {quant_count}")
    print(f"  Total elements: {total_elements:,}")
    
    # Convert F32/F16 tensors to ternary (quick)
    print("\n" + "-" * 70)
    print("CONVERTING TO TERNARY")
    print("-" * 70)
    
    ternary_tensors = {}
    converted = 0
    
    for tensor in reader.tensors:
        if tensor.tensor_type == 0:  # F32
            data = tensor.data.astype(np.float32)
            abs_d = np.abs(data)
            thresh = np.percentile(abs_d.flatten(), 67)
            
            ternary = np.zeros_like(data, dtype=np.int8)
            ternary[(data > 0) & (abs_d >= thresh)] = 1
            ternary[(data < 0) & (abs_d >= thresh)] = -1
            
            ternary_tensors[tensor.name] = ternary
            converted += 1
        
        elif tensor.tensor_type == 1:  # F16
            data = tensor.data.view(np.float16).astype(np.float32)
            abs_d = np.abs(data)
            thresh = np.percentile(abs_d.flatten(), 67)
            
            ternary = np.zeros_like(data, dtype=np.int8)
            ternary[(data > 0) & (abs_d >= thresh)] = 1
            ternary[(data < 0) & (abs_d >= thresh)] = -1
            
            ternary_tensors[tensor.name] = ternary
            converted += 1
    
    print(f"\nConverted {converted} tensors to ternary")
    
    # Show sample conversions
    print("\nSample ternary tensors:")
    for i, (name, tensor) in enumerate(ternary_tensors.items()):
        if i >= 5:
            break
        sparsity = np.mean(tensor == 0) * 100
        unique = np.unique(tensor)
        print(f"  {name[:40]:40s} {str(tensor.shape):15s} sparsity={sparsity:.0f}%")
    
    # Demonstrate ternary operation
    print("\n" + "-" * 70)
    print("TERNARY OPERATION DEMO")
    print("-" * 70)
    
    # Pick a weight tensor
    weight_name = None
    weight_tensor = None
    for name, tensor in ternary_tensors.items():
        if 'weight' in name and len(tensor.shape) >= 1:
            weight_name = name
            weight_tensor = tensor
            break
    
    if weight_tensor is not None:
        print(f"\nUsing: {weight_name}")
        print(f"Shape: {weight_tensor.shape}")
        
        # Count operations
        n_add = np.sum(weight_tensor == 1)
        n_sub = np.sum(weight_tensor == -1)
        n_skip = np.sum(weight_tensor == 0)
        
        print(f"\nOperation breakdown:")
        print(f"  ADD  (+1): {n_add:,}")
        print(f"  SUB  (-1): {n_sub:,}")
        print(f"  SKIP (0):  {n_skip:,}")
        print(f"  MULTIPLY:  0")
        
        # Do a sample operation
        if len(weight_tensor.shape) == 1:
            x = np.random.randn(weight_tensor.shape[0]).astype(np.float32)
            
            # Ternary dot product
            result = 0.0
            for i in range(len(weight_tensor)):
                if weight_tensor[i] == 1:
                    result += x[i]  # ADD
                elif weight_tensor[i] == -1:
                    result -= x[i]  # SUBTRACT
                # 0: skip
            
            print(f"\nSample ternary dot product: {result:.4f}")
    
    # Calculate memory savings
    print("\n" + "-" * 70)
    print("MEMORY ANALYSIS")
    print("-" * 70)
    
    original_bytes = total_elements * 4  # F32
    ternary_bytes = total_elements  # int8
    packed_bytes = total_elements * 2 / 8  # 2-bit
    
    print(f"\nOriginal (F32):  {original_bytes/1e9:.2f} GB")
    print(f"Ternary (int8):  {ternary_bytes/1e9:.2f} GB  ({original_bytes/ternary_bytes:.0f}x smaller)")
    print(f"Packed (2-bit):  {packed_bytes/1e9:.2f} GB  ({original_bytes/packed_bytes:.0f}x smaller)")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nThis demo converted F32/F16 tensors to ternary.")
    print("For full model inference, the Q4_K tensors also need conversion.")
    print("\nKey insight: Once ternary, ALL matrix operations become ADD/SUB only!")


if __name__ == "__main__":
    main()

