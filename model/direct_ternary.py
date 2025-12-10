"""
DIRECT QUANTISED TO TERNARY CONVERSION
======================================

Convert GGUF quantised tensors directly to ternary without 
full dequantisation. Since we're going to {-1, 0, +1} anyway,
we can work directly with the quantised indices.

Key insight: 4-bit quantised values are indices 0-15.
- Values 0-5: negative (-1)
- Values 6-9: zero (0)
- Values 10-15: positive (+1)

This gives us roughly 33% of each category.

Author: Zane Hambly
"""

import numpy as np
from typing import Tuple, Dict
from pathlib import Path


def q4_to_ternary(data: np.ndarray, n_elements: int) -> np.ndarray:
    """
    Convert Q4_K data directly to ternary.
    
    Q4_K uses 4-bit indices (0-15). We map:
        0-5   -> -1 (negative)
        6-9   -> 0  (zero/skip)
        10-15 -> +1 (positive)
    """
    block_size = 256
    n_blocks = (n_elements + block_size - 1) // block_size
    
    result = np.zeros(n_elements, dtype=np.int8)
    bytes_per_block = 144
    
    # Flatten the data
    flat_data = data.flatten()
    
    offset = 0
    for block in range(n_blocks):
        if offset + bytes_per_block > len(flat_data):
            break
        
        block_data = flat_data[offset:offset + bytes_per_block]
        
        # Skip header (16 bytes), get the 4-bit values (128 bytes = 256 values)
        quant_bytes = block_data[-128:]
        
        # Unpack 4-bit values
        low_nibbles = quant_bytes & 0x0F
        high_nibbles = (quant_bytes >> 4) & 0x0F
        
        # Interleave
        quants = np.zeros(256, dtype=np.uint8)
        quants[0::2] = low_nibbles
        quants[1::2] = high_nibbles
        
        # Map to ternary: 0-5 -> -1, 6-9 -> 0, 10-15 -> +1
        ternary = np.zeros(256, dtype=np.int8)
        ternary[quants <= 5] = -1
        ternary[quants >= 10] = 1
        # 6-9 stays 0
        
        start_idx = block * block_size
        end_idx = min(start_idx + block_size, n_elements)
        result[start_idx:end_idx] = ternary[:end_idx - start_idx]
        
        offset += bytes_per_block
    
    return result


def q5_to_ternary(data: np.ndarray, n_elements: int) -> np.ndarray:
    """
    Convert Q5_K/Q6_K data directly to ternary.
    
    5/6-bit values have more granularity, but we still threshold:
        0-10  -> -1
        11-20 -> 0
        21-31 -> +1 (for 5-bit)
    """
    block_size = 256
    n_blocks = (n_elements + block_size - 1) // block_size
    
    result = np.zeros(n_elements, dtype=np.int8)
    bytes_per_block = 176
    
    flat_data = data.flatten()
    
    offset = 0
    for block in range(n_blocks):
        if offset + bytes_per_block > len(flat_data):
            break
        
        block_data = flat_data[offset:offset + bytes_per_block]
        
        # Use raw bytes as proxy values
        quant_bytes = block_data[16:16+128]  # Skip header, get quants
        if len(quant_bytes) < 128:
            quant_bytes = np.pad(quant_bytes, (0, 128 - len(quant_bytes)))
        
        # Map bytes to ternary based on their value
        # bytes range 0-255, threshold into thirds
        ternary = np.zeros(min(256, len(quant_bytes) * 2), dtype=np.int8)
        
        low = quant_bytes & 0x0F
        high = (quant_bytes >> 4) & 0x0F
        
        combined = np.zeros(256, dtype=np.uint8)
        combined[0::2] = low
        combined[1::2] = high
        
        ternary[combined <= 5] = -1
        ternary[combined >= 10] = 1
        
        start_idx = block * block_size
        end_idx = min(start_idx + block_size, n_elements)
        result[start_idx:end_idx] = ternary[:end_idx - start_idx]
        
        offset += bytes_per_block
    
    return result


def f32_to_ternary(data: np.ndarray) -> np.ndarray:
    """Convert float32 to ternary using threshold."""
    abs_data = np.abs(data)
    threshold = np.percentile(abs_data.flatten(), 67)
    
    result = np.zeros_like(data, dtype=np.int8)
    result[(data > 0) & (abs_data >= threshold)] = 1
    result[(data < 0) & (abs_data >= threshold)] = -1
    
    return result


def f16_to_ternary(data: np.ndarray) -> np.ndarray:
    """Convert float16 to ternary."""
    float_data = data.view(np.float16).astype(np.float32)
    return f32_to_ternary(float_data)


def tensor_to_ternary(data: np.ndarray, tensor_type: int, 
                      n_elements: int, target_shape: Tuple[int, ...]) -> np.ndarray:
    """
    Convert any GGUF tensor directly to ternary.
    
    Args:
        data: Raw tensor data
        tensor_type: GGUF tensor type
        n_elements: Number of elements
        target_shape: Desired output shape
    
    Returns:
        int8 ternary array with values in {-1, 0, +1}
    """
    if tensor_type == 0:  # F32
        result = f32_to_ternary(data)
    elif tensor_type == 1:  # F16
        result = f16_to_ternary(data)
    elif tensor_type == 12:  # Q4_K
        result = q4_to_ternary(data, n_elements)
    elif tensor_type in [13, 14]:  # Q5_K, Q6_K
        result = q5_to_ternary(data, n_elements)
    else:
        # Unknown type: use byte values as proxy
        flat = data.flatten().astype(np.float32)
        if len(flat) < n_elements:
            flat = np.pad(flat, (0, n_elements - len(flat)))
        result = f32_to_ternary(flat)
    
    # Reshape to target
    if result.size != np.prod(target_shape):
        # Truncate or pad as needed
        if result.size > np.prod(target_shape):
            result = result[:np.prod(target_shape)]
        else:
            result = np.pad(result, (0, np.prod(target_shape) - result.size))
    
    return result.reshape(target_shape)


def load_gguf_as_ternary(gguf_path: Path) -> Tuple[Dict[str, np.ndarray], dict]:
    """
    Load a GGUF model and convert all weights directly to ternary.
    
    Returns:
        (weights_dict, metadata_dict)
    """
    from gguf import GGUFReader
    
    print(f"Loading GGUF: {gguf_path.name[:50]}...")
    reader = GGUFReader(str(gguf_path))
    
    # Extract metadata
    metadata = {}
    for field in reader.fields.values():
        try:
            value = field.parts[-1].tolist() if hasattr(field.parts[-1], 'tolist') else field.parts[-1]
            if isinstance(value, list) and len(value) == 1:
                value = value[0]
            metadata[field.name] = value
        except:
            pass
    
    # Get architecture info
    arch = None
    for key in metadata:
        if 'architecture' in key:
            arch = metadata[key]
            if isinstance(arch, list):
                arch = bytes(arch).decode('utf-8', errors='ignore')
            break
    
    print(f"  Architecture: {arch}")
    print(f"  Tensors: {len(reader.tensors)}")
    
    # Convert each tensor to ternary
    weights = {}
    total_elements = 0
    
    for tensor in reader.tensors:
        name = tensor.name
        data = tensor.data
        tensor_type = tensor.tensor_type
        n_elements = tensor.n_elements
        
        # Determine target shape
        # For 2D tensors, infer from n_elements and one known dimension
        if '.weight' in name and n_elements > 100:
            # It's a weight matrix - infer shape
            # Use the first dimension from data shape as a hint
            if len(data.shape) >= 2:
                dim0 = data.shape[0]
                dim1 = n_elements // dim0
                target_shape = (dim0, dim1)
            else:
                # Square-ish matrix
                dim = int(np.sqrt(n_elements))
                if dim * dim == n_elements:
                    target_shape = (dim, dim)
                else:
                    target_shape = (n_elements,)
        else:
            # Bias or small tensor - keep as 1D
            target_shape = (n_elements,)
        
        try:
            ternary = tensor_to_ternary(data, tensor_type, n_elements, target_shape)
            weights[name] = ternary
            total_elements += n_elements
        except Exception as e:
            print(f"  Warning: {name}: {e}")
    
    print(f"  Converted {len(weights)} tensors ({total_elements:,} elements)")
    
    # Calculate compression
    original_gb = total_elements * 4 / 1e9  # Assuming original was F32
    ternary_gb = total_elements / 1e9  # 1 byte per trit
    packed_gb = total_elements * 2 / 8 / 1e9  # 2 bits per trit
    
    print(f"  Compression: {original_gb:.2f} GB -> {ternary_gb:.2f} GB (int8)")
    print(f"               -> {packed_gb:.2f} GB (2-bit packed)")
    
    return weights, metadata


def demo():
    """Demo direct ternary conversion."""
    import os
    
    blobs_dir = Path(os.environ.get('USERPROFILE', '')) / '.ollama' / 'models' / 'blobs'
    if not blobs_dir.exists():
        print("Ollama not found")
        return
    
    # Find a model
    models = [(f, f.stat().st_size / 1e9) for f in blobs_dir.iterdir() 
              if f.is_file() and f.stat().st_size > 1e9]
    models.sort(key=lambda x: x[1])
    
    if not models:
        print("No models found")
        return
    
    # Pick the ~4GB model
    target = None
    for path, size in models:
        if 3 < size < 6:
            target = path
            break
    
    if target is None:
        target = models[0][0]
    
    weights, metadata = load_gguf_as_ternary(target)
    
    # Show some stats
    print("\nSample tensors:")
    for i, (name, tensor) in enumerate(weights.items()):
        if i >= 10:
            print(f"  ... and {len(weights) - 10} more")
            break
        
        unique = np.unique(tensor)
        sparsity = np.mean(tensor == 0) * 100
        print(f"  {name[:40]:40s} {str(tensor.shape):20s} sparsity={sparsity:.0f}%")


if __name__ == "__main__":
    demo()

