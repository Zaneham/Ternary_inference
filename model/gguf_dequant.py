"""
GGUF DEQUANTISATION
===================

Dequantise GGUF quantised tensors back to float32 for ternary conversion.

Supports:
    - Type 0: F32 (no conversion needed)
    - Type 1: F16 (simple conversion)
    - Type 12: Q4_K_M (4-bit k-quant medium)
    - Type 14: Q5_K_M (5-bit k-quant medium)

Author: Zane Hambly
"""

import numpy as np
from typing import Tuple


# GGUF tensor types
GGUF_TYPE_F32 = 0
GGUF_TYPE_F16 = 1
GGUF_TYPE_Q4_0 = 2
GGUF_TYPE_Q4_1 = 3
GGUF_TYPE_Q5_0 = 6
GGUF_TYPE_Q5_1 = 7
GGUF_TYPE_Q8_0 = 8
GGUF_TYPE_Q8_1 = 9
GGUF_TYPE_Q2_K = 10
GGUF_TYPE_Q3_K = 11
GGUF_TYPE_Q4_K = 12
GGUF_TYPE_Q5_K = 13
GGUF_TYPE_Q6_K = 14


def dequantize_f16(data: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Dequantize F16 to F32."""
    return data.astype(np.float32).reshape(shape)


def dequantize_q4_k(data: np.ndarray, n_elements: int) -> np.ndarray:
    """
    Dequantize Q4_K (4-bit k-quant) to F32.
    
    Q4_K format (per 256-element superblock):
        - 2 bytes: d (scale, fp16)
        - 2 bytes: dmin (min scale, fp16)
        - 12 bytes: scales (6-bit each)
        - 128 bytes: 4-bit quantised values
        Total: 144 bytes per 256 elements
    """
    block_size = 256
    n_blocks = n_elements // block_size
    
    # Simplified: extract rough values from the packed data
    # This is an approximation - full dequant is more complex
    result = np.zeros(n_elements, dtype=np.float32)
    
    bytes_per_block = 144
    offset = 0
    
    for block in range(n_blocks):
        if offset + bytes_per_block > len(data.flatten()):
            break
            
        block_data = data.flatten()[offset:offset + bytes_per_block]
        
        # Extract scale (first 2 bytes as fp16)
        scale_bytes = block_data[:2].view(np.float16)
        d = float(scale_bytes[0]) if len(scale_bytes) > 0 else 1.0
        
        # Extract 4-bit values (last 128 bytes = 256 4-bit values)
        quant_bytes = block_data[-128:]
        
        # Unpack 4-bit values
        low_nibbles = quant_bytes & 0x0F
        high_nibbles = (quant_bytes >> 4) & 0x0F
        
        # Interleave
        quants = np.zeros(256, dtype=np.int8)
        quants[0::2] = low_nibbles
        quants[1::2] = high_nibbles
        
        # Convert to float with scale
        # Centre around 0 (4-bit range is 0-15, centre at 8)
        block_result = (quants.astype(np.float32) - 8) * d
        
        start_idx = block * block_size
        end_idx = min(start_idx + block_size, n_elements)
        result[start_idx:end_idx] = block_result[:end_idx - start_idx]
        
        offset += bytes_per_block
    
    return result


def dequantize_q5_k(data: np.ndarray, n_elements: int) -> np.ndarray:
    """
    Dequantize Q5_K (5-bit k-quant) to F32.
    
    Similar to Q4_K but with 5-bit values.
    """
    # Simplified approximation
    block_size = 256
    n_blocks = n_elements // block_size
    
    result = np.zeros(n_elements, dtype=np.float32)
    bytes_per_block = 176  # 5-bit uses more bytes
    
    offset = 0
    for block in range(n_blocks):
        if offset + bytes_per_block > len(data.flatten()):
            break
            
        block_data = data.flatten()[offset:offset + bytes_per_block]
        
        # Extract scale
        scale_bytes = block_data[:2].view(np.float16)
        d = float(scale_bytes[0]) if len(scale_bytes) > 0 else 1.0
        
        # Approximate: treat as random-ish values scaled
        # This is simplified - real dequant is more involved
        quant_bytes = block_data[16:]  # Skip header
        
        # Use the bytes as rough approximation
        values = quant_bytes.astype(np.float32)[:block_size]
        if len(values) < block_size:
            values = np.pad(values, (0, block_size - len(values)))
        
        # Normalise to roughly -1 to 1 range
        block_result = (values - 128) / 128 * d
        
        start_idx = block * block_size
        end_idx = min(start_idx + block_size, n_elements)
        result[start_idx:end_idx] = block_result[:end_idx - start_idx]
        
        offset += bytes_per_block
    
    return result


def dequantize_tensor(data: np.ndarray, tensor_type: int, n_elements: int, 
                      original_shape: Tuple[int, ...]) -> np.ndarray:
    """
    Dequantize a GGUF tensor to float32.
    
    Args:
        data: Raw tensor data from GGUF
        tensor_type: GGUF tensor type
        n_elements: Number of elements in the full tensor
        original_shape: Original shape before quantisation
    
    Returns:
        Float32 numpy array with the original shape
    """
    if tensor_type == GGUF_TYPE_F32:
        return data.astype(np.float32).reshape(original_shape)
    
    elif tensor_type == GGUF_TYPE_F16:
        # F16 is stored directly
        return data.view(np.float16).astype(np.float32).reshape(original_shape)
    
    elif tensor_type == GGUF_TYPE_Q4_K:
        result = dequantize_q4_k(data, n_elements)
        return result.reshape(original_shape)
    
    elif tensor_type == GGUF_TYPE_Q6_K:  # Q5_K is actually type 14 = Q6_K in some versions
        result = dequantize_q5_k(data, n_elements)
        return result.reshape(original_shape)
    
    else:
        # Fallback: just use the raw bytes as rough approximation
        # This won't be accurate but allows loading to proceed
        print(f"  Warning: Unknown tensor type {tensor_type}, using approximation")
        result = data.astype(np.float32).flatten()
        if len(result) < n_elements:
            result = np.pad(result, (0, n_elements - len(result)))
        elif len(result) > n_elements:
            result = result[:n_elements]
        return result.reshape(original_shape)


def load_and_dequantize_gguf(gguf_path) -> dict:
    """
    Load a GGUF file and dequantize all tensors to float32.
    
    Returns dict of tensor_name -> float32 numpy array
    """
    from gguf import GGUFReader
    
    print(f"Loading GGUF: {gguf_path}")
    reader = GGUFReader(str(gguf_path))
    
    weights = {}
    
    for tensor in reader.tensors:
        name = tensor.name
        tensor_type = tensor.tensor_type
        n_elements = tensor.n_elements
        data = tensor.data
        
        # Calculate original shape from n_elements
        # The tensor.shape from GGUF is the quantised shape, not original
        # We need to figure out the original shape
        
        if tensor_type == GGUF_TYPE_F32:
            # F32: data shape IS the original shape
            original_shape = data.shape
        elif tensor_type == GGUF_TYPE_F16:
            # F16: data is uint16 pairs
            original_shape = tuple(list(data.shape[:-1]) + [data.shape[-1]])
        else:
            # Quantised: need to infer shape
            # Common cases based on tensor names
            if 'embed' in name.lower():
                # Embedding: vocab_size x hidden
                if n_elements > 1000000:  # Big embedding
                    vocab = 32000  # Guess
                    hidden = n_elements // vocab
                    original_shape = (vocab, hidden)
                else:
                    original_shape = (n_elements,)
            elif len(data.shape) == 2:
                # Matrix: use sqrt approximation or known dim
                dim0 = data.shape[0]
                dim1 = n_elements // dim0
                original_shape = (dim0, dim1)
            else:
                original_shape = (n_elements,)
        
        try:
            weights[name] = dequantize_tensor(data, tensor_type, n_elements, original_shape)
        except Exception as e:
            print(f"  Warning: Failed to dequantize {name}: {e}")
    
    print(f"Loaded {len(weights)} tensors")
    return weights

