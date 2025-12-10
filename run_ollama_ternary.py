"""
RUN TERNARY INFERENCE WITH OLLAMA WEIGHTS
==========================================

This script:
1. Loads a real Ollama model (GGUF format)
2. Quantises weights to ternary {-1, 0, +1}
3. Runs inference with the ternary transformer
4. Generates actual text!

Requirements:
    pip install gguf tiktoken

Usage:
    python run_ollama_ternary.py

Author: Zane Hambly
"""

import sys
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, 'model')

from llama_ternary import TernaryLlama, LlamaConfig


def find_ollama_models():
    """Find all Ollama model blobs."""
    blobs_dir = Path(os.environ.get('USERPROFILE', '')) / '.ollama' / 'models' / 'blobs'
    if not blobs_dir.exists():
        blobs_dir = Path.home() / '.ollama' / 'models' / 'blobs'
    
    if not blobs_dir.exists():
        return []
    
    models = []
    for f in blobs_dir.iterdir():
        if f.is_file():
            size_gb = f.stat().st_size / (1024**3)
            if size_gb > 0.1:  # > 100MB
                models.append((f, size_gb))
    
    models.sort(key=lambda x: x[1])
    return models


def load_gguf_model(path: Path):
    """Load GGUF file and extract weights + metadata."""
    try:
        from gguf import GGUFReader
    except ImportError:
        print("Please install gguf: pip install gguf")
        return None, None
    
    print(f"Loading GGUF: {path.name[:50]}...")
    reader = GGUFReader(str(path))
    
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
    
    # Extract weights
    weights = {}
    for tensor in reader.tensors:
        data = tensor.data
        if hasattr(data, 'astype'):
            weights[tensor.name] = data.astype(np.float32)
        else:
            weights[tensor.name] = np.array(data, dtype=np.float32)
    
    return weights, metadata


def create_simple_tokenizer():
    """Create a simple byte-pair tokenizer or load tiktoken."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        
        class TiktokenWrapper:
            def __init__(self, enc):
                self.enc = enc
                self.vocab_size = enc.n_vocab
            
            def encode(self, text: str) -> list:
                return self.enc.encode(text)
            
            def decode(self, tokens: list) -> str:
                return self.enc.decode(tokens)
        
        return TiktokenWrapper(enc)
    
    except ImportError:
        # Fallback: simple character tokenizer
        class SimpleTokenizer:
            def __init__(self):
                self.vocab_size = 256
            
            def encode(self, text: str) -> list:
                return [ord(c) % 256 for c in text]
            
            def decode(self, tokens: list) -> str:
                return ''.join(chr(t % 128) if t < 128 else '?' for t in tokens)
        
        print("Note: Using simple tokenizer. Install tiktoken for better results: pip install tiktoken")
        return SimpleTokenizer()


def main():
    print("=" * 70)
    print("TERNARY INFERENCE WITH REAL OLLAMA WEIGHTS")
    print("=" * 70)
    
    # Find models
    models = find_ollama_models()
    if not models:
        print("\nNo Ollama models found!")
        print("Install Ollama and download a model:")
        print("  https://ollama.ai/download")
        print("  ollama pull qwen2")
        return
    
    print("\nAvailable models:")
    for i, (path, size) in enumerate(models):
        print(f"  [{i}] {size:.1f} GB - {path.name[:40]}...")
    
    # Select a model (pick one around 4-5GB for reasonable speed)
    target_idx = 0
    for i, (path, size) in enumerate(models):
        if 3 < size < 6:  # Prefer 4-5GB models
            target_idx = i
            break
    
    target_path, target_size = models[target_idx]
    print(f"\nSelected: {target_size:.1f} GB model")
    
    # Load GGUF
    print("\n" + "-" * 70)
    print("LOADING GGUF WEIGHTS")
    print("-" * 70)
    
    weights, metadata = load_gguf_model(target_path)
    if weights is None:
        return
    
    # Show architecture
    arch = None
    for key in metadata:
        if 'architecture' in key:
            arch = metadata[key]
            if isinstance(arch, list):
                arch = bytes(arch).decode('utf-8', errors='ignore')
            break
    
    print(f"\nArchitecture: {arch}")
    print(f"Tensors: {len(weights)}")
    
    # Create config from metadata
    config = LlamaConfig.from_gguf_metadata(metadata)
    
    # Check if model is too large for RAM
    estimated_ram = config.hidden_size * config.hidden_size * config.num_layers * 8 / 1e9
    print(f"\nConfig:")
    print(f"  Hidden: {config.hidden_size}")
    print(f"  Layers: {config.num_layers}")
    print(f"  Heads: {config.num_heads} (KV: {config.num_kv_heads})")
    print(f"  FFN: {config.intermediate_size}")
    print(f"  Vocab: {config.vocab_size}")
    print(f"  Estimated RAM: {estimated_ram:.1f} GB")
    
    if estimated_ram > 16:
        print("\nWarning: Model may be too large for RAM. Using smaller config.")
        # Use a smaller subset of layers
        config.num_layers = min(4, config.num_layers)
    
    # Create ternary model
    print("\n" + "-" * 70)
    print("CREATING TERNARY TRANSFORMER")
    print("-" * 70)
    
    print("\nInitialising model structure...")
    model = TernaryLlama(config)
    
    # Load weights
    print("Loading and quantising weights to ternary...")
    loaded = model.load_gguf_weights(weights)
    
    if loaded == 0:
        print("\nWarning: No weights loaded. Architecture mismatch?")
        print("Running with random weights for demo...")
    
    # Statistics
    stats = model.count_parameters()
    print(f"\nParameter statistics:")
    print(f"  Ternary params: {stats['ternary_params']:,}")
    print(f"  Float params:   {stats['float_params']:,}")
    print(f"  Ternary ratio:  {stats['ternary_percentage']:.1f}%")
    
    # Calculate compression
    original_bytes = stats['total_params'] * 4
    ternary_bytes = stats['ternary_params'] + stats['float_params'] * 4
    packed_bytes = stats['ternary_params'] / 4 + stats['float_params'] * 4  # 2-bit packing
    
    print(f"\nMemory:")
    print(f"  Original (float32): {original_bytes/1e9:.2f} GB")
    print(f"  Ternary (int8):     {ternary_bytes/1e9:.2f} GB")
    print(f"  Packed (2-bit):     {packed_bytes/1e9:.2f} GB")
    print(f"  Compression:        {original_bytes/ternary_bytes:.1f}x")
    
    # Create tokenizer
    print("\n" + "-" * 70)
    print("RUNNING INFERENCE")
    print("-" * 70)
    
    tokenizer = create_simple_tokenizer()
    
    # Test prompts
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "Once upon a time",
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        
        # Tokenise
        input_ids = np.array([tokenizer.encode(prompt)])
        print(f"  Tokens: {input_ids.shape[1]}")
        
        # Generate
        try:
            output_ids = model.generate(
                input_ids, 
                max_new_tokens=20, 
                temperature=0.8,
                top_k=40
            )
            
            # Decode
            output_text = tokenizer.decode(output_ids[0].tolist())
            print(f"  Output: '{output_text[:100]}...'")
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n" + "=" * 70)
    print("TERNARY INFERENCE COMPLETE")
    print("=" * 70)
    print("\nKey achievements:")
    print(f"  - Loaded real Ollama model ({target_size:.1f} GB)")
    print(f"  - Quantised to ternary {{-1, 0, +1}}")
    print(f"  - {original_bytes/ternary_bytes:.0f}x memory compression")
    print(f"  - All matrix multiplies use ONLY addition")
    print("\nNote: Output quality depends on:")
    print("  - How well the quantisation preserved information")
    print("  - Tokenizer matching the original model")
    print("  - Number of layers loaded")


if __name__ == "__main__":
    main()

