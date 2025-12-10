"""
OLLAMA WEIGHT LOADER FOR TERNARY INFERENCE
===========================================

Load pre-trained weights from Ollama models (GGUF format)
and quantise them to balanced ternary {-1, 0, +1}.

No training needed - use the billions of tokens already learned!

Requirements:
    pip install gguf

Usage:
    from ollama_loader import OllamaLoader
    
    loader = OllamaLoader()
    loader.list_models()
    weights = loader.load_model("smallest")  # or specific model name

Author: Zane Hambly
"""

import os
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass


@dataclass
class ModelInfo:
    """Information about a loaded Ollama model."""
    name: str
    path: Path
    size_mb: float
    architecture: str
    parameter_count: int
    tensor_count: int


class OllamaLoader:
    """
    Load Ollama GGUF models and quantise to ternary.
    """
    
    def __init__(self):
        self.blobs_dir = self._find_ollama_blobs()
        self._gguf_available = self._check_gguf()
    
    def _find_ollama_blobs(self) -> Optional[Path]:
        """Find the Ollama models directory."""
        # Windows
        ollama_dir = Path(os.environ.get('USERPROFILE', '')) / '.ollama' / 'models' / 'blobs'
        if ollama_dir.exists():
            return ollama_dir
        
        # Linux/Mac
        ollama_dir = Path.home() / '.ollama' / 'models' / 'blobs'
        if ollama_dir.exists():
            return ollama_dir
        
        return None
    
    def _check_gguf(self) -> bool:
        """Check if gguf library is available."""
        try:
            import gguf
            return True
        except ImportError:
            return False
    
    @property
    def is_available(self) -> bool:
        """Check if Ollama models are available."""
        return self.blobs_dir is not None and self._gguf_available
    
    def list_models(self, min_size_mb: float = 100) -> List[Tuple[str, float, Path]]:
        """
        List available Ollama model blobs by size.
        
        Returns list of (name, size_mb, path) tuples.
        """
        if not self.blobs_dir:
            print("Ollama directory not found.")
            print("Install Ollama and download a model:")
            print("  https://ollama.ai/download")
            print("  ollama pull llama2")
            return []
        
        blobs = []
        for f in self.blobs_dir.iterdir():
            if f.is_file():
                size_mb = f.stat().st_size / (1024 * 1024)
                if size_mb > min_size_mb:
                    blobs.append((f.name[:20] + "...", size_mb, f))
        
        blobs.sort(key=lambda x: x[1])
        return blobs
    
    def load_model(self, model: str = "smallest") -> Optional[Dict[str, np.ndarray]]:
        """
        Load and quantise an Ollama model to ternary.
        
        Args:
            model: "smallest", "largest", or part of model blob name
        
        Returns:
            Dictionary of ternary weight tensors, or None if unavailable.
        """
        if not self._gguf_available:
            print("=" * 60)
            print("GGUF library not installed.")
            print("To load Ollama models, install it:")
            print("  pip install gguf")
            print("=" * 60)
            return None
        
        models = self.list_models()
        if not models:
            return None
        
        # Select model
        if model == "smallest":
            selected = models[0]
        elif model == "largest":
            selected = models[-1]
        else:
            # Find by name
            selected = None
            for m in models:
                if model in m[0] or model in str(m[2]):
                    selected = m
                    break
            if not selected:
                print(f"Model '{model}' not found.")
                return None
        
        name, size_mb, path = selected
        print(f"\nLoading model: {name} ({size_mb:.1f} MB)")
        
        # Load and quantise
        return self._load_and_quantise(path)
    
    def _load_and_quantise(self, path: Path, threshold_percentile: float = 50) -> Dict[str, np.ndarray]:
        """Load GGUF weights and quantise to ternary."""
        from gguf import GGUFReader
        
        reader = GGUFReader(str(path))
        
        # Get metadata
        metadata = {}
        for field in reader.fields.values():
            try:
                if hasattr(field, 'parts'):
                    value = field.parts[-1].tolist() if hasattr(field.parts[-1], 'tolist') else field.parts[-1]
                    if isinstance(value, list) and len(value) == 1:
                        value = value[0]
                    metadata[field.name] = value
            except:
                pass
        
        print(f"  Architecture: {metadata.get('general.architecture', 'unknown')}")
        print(f"  Tensors: {len(reader.tensors)}")
        
        # Quantise each tensor
        ternary_weights = {}
        total_params = 0
        total_nonzero = 0
        
        for tensor in reader.tensors:
            name = tensor.name
            data = tensor.data
            
            # Convert to float32
            if hasattr(data, 'astype'):
                float_data = data.astype(np.float32)
            else:
                float_data = np.array(data, dtype=np.float32)
            
            # Skip tiny tensors (biases, norms)
            if float_data.size < 100:
                ternary_weights[name] = float_data
                continue
            
            # Quantise to ternary
            magnitudes = np.abs(float_data.flatten())
            threshold = np.percentile(magnitudes, threshold_percentile)
            
            ternary = np.zeros_like(float_data, dtype=np.int8)
            ternary[float_data > threshold] = 1
            ternary[float_data < -threshold] = -1
            
            ternary_weights[name] = ternary
            
            total_params += float_data.size
            total_nonzero += np.count_nonzero(ternary)
        
        sparsity = 1 - (total_nonzero / total_params) if total_params > 0 else 0
        
        print(f"\nQuantisation complete:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Non-zero (ternary): {total_nonzero:,}")
        print(f"  Sparsity: {sparsity:.1%}")
        
        # Calculate compression
        float_bytes = total_params * 4
        ternary_bytes = total_params  # 1 byte per trit (could be 2 bits)
        packed_bytes = total_params * 2 / 8  # 2 bits per trit
        
        print(f"\nMemory:")
        print(f"  Original (float32): {float_bytes/1024/1024:.1f} MB")
        print(f"  Ternary (int8):     {ternary_bytes/1024/1024:.1f} MB")
        print(f"  Packed (2-bit):     {packed_bytes/1024/1024:.1f} MB")
        print(f"  Compression:        {float_bytes/ternary_bytes:.0f}x (int8), {float_bytes/packed_bytes:.0f}x (packed)")
        
        return ternary_weights
    
    def get_layer_mapping(self, weights: Dict[str, np.ndarray]) -> Dict[str, str]:
        """
        Map GGUF tensor names to our transformer layer names.
        
        Returns mapping dict for common architectures (llama, mistral, etc.)
        """
        mapping = {}
        
        for name in weights.keys():
            # Embedding
            if 'embed' in name.lower() or 'wte' in name.lower():
                mapping[name] = 'embeddings'
            
            # Attention projections
            elif 'q_proj' in name.lower() or 'wq' in name.lower():
                mapping[name] = 'attention.q_proj'
            elif 'k_proj' in name.lower() or 'wk' in name.lower():
                mapping[name] = 'attention.k_proj'
            elif 'v_proj' in name.lower() or 'wv' in name.lower():
                mapping[name] = 'attention.v_proj'
            elif 'o_proj' in name.lower() or 'wo' in name.lower():
                mapping[name] = 'attention.o_proj'
            
            # MLP
            elif 'gate' in name.lower() or 'up' in name.lower() or 'w1' in name.lower():
                mapping[name] = 'mlp.up_proj'
            elif 'down' in name.lower() or 'w2' in name.lower():
                mapping[name] = 'mlp.down_proj'
            
            # Output
            elif 'lm_head' in name.lower() or 'output' in name.lower():
                mapping[name] = 'lm_head'
        
        return mapping


def demo():
    """Demo loading Ollama models."""
    print("=" * 70)
    print("OLLAMA MODEL LOADER FOR TERNARY INFERENCE")
    print("=" * 70)
    
    loader = OllamaLoader()
    
    if not loader.is_available:
        print("\nOllama not available. To use real models:")
        print()
        print("1. Install Ollama: https://ollama.ai/download")
        print("2. Download a model: ollama pull llama2")
        print("3. Install gguf: pip install gguf")
        print()
        print("The ternary transformer still works with random weights!")
        return
    
    print("\nAvailable models:")
    print("-" * 60)
    
    models = loader.list_models()
    for name, size_mb, path in models:
        print(f"  {size_mb:8.1f} MB  {name}")
    
    if models:
        print("\n" + "-" * 70)
        print("Loading smallest model...")
        print("-" * 70)
        
        weights = loader.load_model("smallest")
        
        if weights:
            print("\nLoaded tensors (first 10):")
            for i, (name, tensor) in enumerate(weights.items()):
                if i >= 10:
                    print(f"  ... and {len(weights) - 10} more")
                    break
                dtype = "ternary" if tensor.dtype == np.int8 else "float"
                print(f"  {name[:50]:50s} {str(tensor.shape):20s} ({dtype})")
            
            print("\n" + "=" * 70)
            print("SUCCESS! Model loaded and quantised to ternary!")
            print("These weights can now power the TernaryTransformer.")
            print("=" * 70)


if __name__ == "__main__":
    demo()

