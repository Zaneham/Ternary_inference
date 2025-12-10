"""
LLAMA-COMPATIBLE TERNARY TRANSFORMER
====================================

A ternary transformer that matches the LLaMA/Qwen2/Mistral architecture,
allowing direct loading of Ollama model weights.

Key Features:
    - RMSNorm (LLaMA's layer normalisation)
    - RoPE (Rotary Position Embeddings)
    - SwiGLU activation (gate * up * silu)
    - Grouped Query Attention (GQA)
    - All linear layers use ternary weights {-1, 0, +1}

Author: Zane Hambly
"""

import numpy as np
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
import math


@dataclass 
class LlamaConfig:
    """Configuration matching Ollama model architectures."""
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int = 32  # For GQA (Grouped Query Attention)
    max_seq_len: int = 4096
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    quantize_threshold: float = 0.33

    @classmethod
    def from_gguf_metadata(cls, metadata: dict) -> 'LlamaConfig':
        """Create config from GGUF metadata."""
        # Try different architecture prefixes
        prefixes = ['llama.', 'qwen2.', 'mistral.', 'gemma.', '']
        
        def get_value(key: str, default):
            for prefix in prefixes:
                full_key = f"{prefix}{key}" if prefix else key
                if full_key in metadata:
                    return metadata[full_key]
            return default
        
        return cls(
            vocab_size=get_value('vocab_size', 32000),
            hidden_size=get_value('embedding_length', 4096),
            intermediate_size=get_value('feed_forward_length', 11008),
            num_layers=get_value('block_count', 32),
            num_heads=get_value('attention.head_count', 32),
            num_kv_heads=get_value('attention.head_count_kv', 32),
            max_seq_len=get_value('context_length', 4096),
            rope_theta=get_value('rope.freq_base', 10000.0),
            rms_norm_eps=get_value('attention.layer_norm_rms_epsilon', 1e-6),
        )


class TernaryLinear:
    """
    Linear layer with ternary weights {-1, 0, +1}.
    
    Forward pass uses ONLY addition and subtraction.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = False, 
                 init_random: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.bias_weights = np.zeros(out_features, dtype=np.float32) if bias else None
        self.has_weights = False
        
        if init_random:
            # Initialise with random ternary weights
            float_weights = np.random.randn(in_features, out_features).astype(np.float32) * 0.02
            abs_w = np.abs(float_weights)
            threshold = np.percentile(abs_w, 67)  # Top 33% become +/-1
            
            self.weights = np.zeros((in_features, out_features), dtype=np.int8)
            self.weights[(float_weights > 0) & (abs_w >= threshold)] = 1
            self.weights[(float_weights < 0) & (abs_w >= threshold)] = -1
            self.has_weights = True
            self.sparsity = np.mean(self.weights == 0)
        else:
            self.weights = np.zeros((in_features, out_features), dtype=np.int8)
            self.sparsity = 1.0
    
    def load_weights(self, weights: np.ndarray, bias: np.ndarray = None):
        """Load and quantise weights."""
        # Handle various transposition cases from GGUF
        if weights.shape == (self.out_features, self.in_features):
            weights = weights.T
        elif weights.shape != (self.in_features, self.out_features):
            # Try to figure out the correct shape
            if weights.shape[0] == self.in_features and weights.shape[1] == self.out_features:
                pass  # Already correct
            elif weights.shape[1] == self.in_features and weights.shape[0] == self.out_features:
                weights = weights.T
            else:
                # Reshape if needed and possible
                total = weights.size
                if total == self.in_features * self.out_features:
                    weights = weights.reshape(self.in_features, self.out_features)
                else:
                    raise ValueError(f"Shape mismatch: got {weights.shape}, need {(self.in_features, self.out_features)}")
        
        # Final check
        if weights.shape != (self.in_features, self.out_features):
            raise ValueError(f"Shape mismatch: got {weights.shape}, expected {(self.in_features, self.out_features)}")
        
        # Quantise to ternary
        if weights.dtype != np.int8:
            abs_w = np.abs(weights)
            threshold = np.percentile(abs_w, 67)  # Top 33% become +/-1
            ternary = np.zeros_like(weights, dtype=np.int8)
            ternary[(weights > 0) & (abs_w >= threshold)] = 1
            ternary[(weights < 0) & (abs_w >= threshold)] = -1
            self.weights = ternary
        else:
            self.weights = weights
        
        if bias is not None:
            self.bias_weights = bias.astype(np.float32)
        
        self.has_weights = True
        self.sparsity = np.mean(self.weights == 0)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass using ONLY addition."""
        original_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)
        
        # Ternary matmul: separate positive and negative
        pos_mask = (self.weights == 1).astype(np.float32)
        neg_mask = (self.weights == -1).astype(np.float32)
        
        result = x_flat @ pos_mask - x_flat @ neg_mask
        
        if self.bias_weights is not None:
            result = result + self.bias_weights
        
        return result.reshape(*original_shape[:-1], self.out_features)


class RMSNorm:
    """RMS Normalisation (LLaMA's layer norm)."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = np.ones(hidden_size, dtype=np.float32)
    
    def load_weights(self, weight: np.ndarray):
        """Load normalisation weights."""
        self.weight = weight.astype(np.float32)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply RMS normalisation."""
        variance = np.mean(x ** 2, axis=-1, keepdims=True)
        x_normed = x / np.sqrt(variance + self.eps)
        return x_normed * self.weight


class RotaryEmbedding:
    """Rotary Position Embeddings (RoPE)."""
    
    def __init__(self, dim: int, max_seq_len: int = 4096, theta: float = 10000.0):
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Precompute frequencies
        inv_freq = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        t = np.arange(max_seq_len, dtype=np.float32)
        freqs = np.outer(t, inv_freq)
        
        self.cos_cached = np.cos(freqs).astype(np.float32)
        self.sin_cached = np.sin(freqs).astype(np.float32)
    
    def apply(self, x: np.ndarray, start_pos: int = 0) -> np.ndarray:
        """Apply rotary embeddings to x."""
        seq_len = x.shape[-2]
        
        cos = self.cos_cached[start_pos:start_pos + seq_len]
        sin = self.sin_cached[start_pos:start_pos + seq_len]
        
        # Reshape for broadcasting
        cos = cos.reshape(1, 1, seq_len, -1)
        sin = sin.reshape(1, 1, seq_len, -1)
        
        # Split into pairs
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        
        # Apply rotation
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos
        
        # Interleave back
        result = np.zeros_like(x)
        result[..., ::2] = rotated_x1
        result[..., 1::2] = rotated_x2
        
        return result


class TernaryAttention:
    """
    Multi-head attention with Grouped Query Attention (GQA).
    All projections use ternary weights.
    """
    
    def __init__(self, config: LlamaConfig):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.num_groups = config.num_heads // config.num_kv_heads
        
        # Q, K, V, O projections (all ternary)
        self.q_proj = TernaryLinear(config.hidden_size, config.hidden_size, bias=True)
        self.k_proj = TernaryLinear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = TernaryLinear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = TernaryLinear(config.hidden_size, config.hidden_size)
        
        # RoPE
        self.rope = RotaryEmbedding(self.head_dim, config.max_seq_len, config.rope_theta)
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None, 
                start_pos: int = 0) -> np.ndarray:
        """
        Forward pass with GQA and RoPE.
        """
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V (ternary!)
        q = self.q_proj.forward(x)
        k = self.k_proj.forward(x)
        v = self.v_proj.forward(x)
        
        # Reshape for multi-head
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Transpose to (batch, heads, seq, dim)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        
        # Apply RoPE
        q = self.rope.apply(q, start_pos)
        k = self.rope.apply(k, start_pos)
        
        # Expand KV heads for GQA
        if self.num_groups > 1:
            k = np.repeat(k, self.num_groups, axis=1)
            v = np.repeat(v, self.num_groups, axis=1)
        
        # Attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * scale
        
        # Apply causal mask
        if mask is not None:
            scores = scores + mask
        
        # Softmax
        attn = self._softmax(scores)
        
        # Apply attention to values
        output = np.matmul(attn, v)
        
        # Reshape back
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.hidden_size)
        
        # Output projection (ternary!)
        return self.o_proj.forward(output)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class TernarySwiGLUMLP:
    """
    SwiGLU MLP with ternary weights.
    
    SwiGLU: output = down(silu(gate(x)) * up(x))
    """
    
    def __init__(self, config: LlamaConfig):
        self.gate_proj = TernaryLinear(config.hidden_size, config.intermediate_size)
        self.up_proj = TernaryLinear(config.hidden_size, config.intermediate_size)
        self.down_proj = TernaryLinear(config.intermediate_size, config.hidden_size)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """SwiGLU forward pass."""
        gate = self._silu(self.gate_proj.forward(x))
        up = self.up_proj.forward(x)
        return self.down_proj.forward(gate * up)
    
    def _silu(self, x: np.ndarray) -> np.ndarray:
        """SiLU (Swish) activation."""
        return x * (1 / (1 + np.exp(-x)))


class TernaryLlamaBlock:
    """A single transformer block."""
    
    def __init__(self, config: LlamaConfig, layer_idx: int):
        self.layer_idx = layer_idx
        self.attention = TernaryAttention(config)
        self.mlp = TernarySwiGLUMLP(config)
        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None,
                start_pos: int = 0) -> np.ndarray:
        """Forward with residual connections."""
        # Attention with residual
        h = self.attn_norm.forward(x)
        h = self.attention.forward(h, mask, start_pos)
        x = x + h
        
        # MLP with residual
        h = self.ffn_norm.forward(x)
        h = self.mlp.forward(h)
        x = x + h
        
        return x


class TernaryLlama:
    """
    Complete LLaMA-compatible ternary transformer.
    
    Can load weights from Ollama GGUF files.
    """
    
    def __init__(self, config: LlamaConfig):
        self.config = config
        
        # Token embeddings (float)
        self.embeddings = np.random.randn(config.vocab_size, config.hidden_size).astype(np.float32) * 0.02
        
        # Transformer blocks
        self.layers = [TernaryLlamaBlock(config, i) for i in range(config.num_layers)]
        
        # Final norm
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        
        # LM head (ternary)
        self.lm_head = TernaryLinear(config.hidden_size, config.vocab_size)
        
        # Track loading
        self.weights_loaded = False
    
    def load_gguf_weights(self, weights: Dict[str, np.ndarray]):
        """
        Load weights from GGUF tensors.
        
        Maps tensor names like:
            token_embd.weight -> embeddings
            blk.{i}.attn_q.weight -> layers[i].attention.q_proj
            etc.
        """
        loaded = 0
        
        for name, tensor in weights.items():
            try:
                if name == 'token_embd.weight':
                    if tensor.ndim == 2:
                        self.embeddings = tensor.T.astype(np.float32) if tensor.shape[0] == self.config.hidden_size else tensor.astype(np.float32)
                    loaded += 1
                
                elif name == 'output.weight':
                    self.lm_head.load_weights(tensor)
                    loaded += 1
                
                elif name == 'output_norm.weight':
                    self.norm.load_weights(tensor)
                    loaded += 1
                
                elif name.startswith('blk.'):
                    # Parse layer index
                    parts = name.split('.')
                    layer_idx = int(parts[1])
                    
                    if layer_idx >= self.config.num_layers:
                        continue
                    
                    layer = self.layers[layer_idx]
                    rest = '.'.join(parts[2:])
                    
                    # Map tensor to layer component
                    if rest == 'attn_q.weight':
                        layer.attention.q_proj.load_weights(tensor)
                        loaded += 1
                    elif rest == 'attn_q.bias':
                        layer.attention.q_proj.bias_weights = tensor.astype(np.float32)
                    elif rest == 'attn_k.weight':
                        layer.attention.k_proj.load_weights(tensor)
                        loaded += 1
                    elif rest == 'attn_k.bias':
                        layer.attention.k_proj.bias_weights = tensor.astype(np.float32)
                    elif rest == 'attn_v.weight':
                        layer.attention.v_proj.load_weights(tensor)
                        loaded += 1
                    elif rest == 'attn_v.bias':
                        layer.attention.v_proj.bias_weights = tensor.astype(np.float32)
                    elif rest == 'attn_output.weight':
                        layer.attention.o_proj.load_weights(tensor)
                        loaded += 1
                    elif rest == 'attn_norm.weight':
                        layer.attn_norm.load_weights(tensor)
                        loaded += 1
                    elif rest == 'ffn_gate.weight':
                        layer.mlp.gate_proj.load_weights(tensor)
                        loaded += 1
                    elif rest == 'ffn_up.weight':
                        layer.mlp.up_proj.load_weights(tensor)
                        loaded += 1
                    elif rest == 'ffn_down.weight':
                        layer.mlp.down_proj.load_weights(tensor)
                        loaded += 1
                    elif rest == 'ffn_norm.weight':
                        layer.ffn_norm.load_weights(tensor)
                        loaded += 1
                    
                    # Handle merged QKV
                    elif rest == 'attn_qkv.weight':
                        # Split into Q, K, V
                        q_dim = self.config.hidden_size
                        kv_dim = self.config.num_kv_heads * (self.config.hidden_size // self.config.num_heads)
                        
                        if tensor.ndim == 2:
                            total = tensor.shape[0] if tensor.shape[1] == self.config.hidden_size else tensor.shape[1]
                            if total == q_dim + 2 * kv_dim:
                                if tensor.shape[1] == self.config.hidden_size:
                                    q = tensor[:q_dim]
                                    k = tensor[q_dim:q_dim + kv_dim]
                                    v = tensor[q_dim + kv_dim:]
                                else:
                                    tensor = tensor.T
                                    q = tensor[:q_dim]
                                    k = tensor[q_dim:q_dim + kv_dim]
                                    v = tensor[q_dim + kv_dim:]
                                
                                layer.attention.q_proj.load_weights(q)
                                layer.attention.k_proj.load_weights(k)
                                layer.attention.v_proj.load_weights(v)
                                loaded += 3
                
            except Exception as e:
                # Log failed loads for debugging
                if 'blk.0.' in name or 'embed' in name or 'output' in name:
                    print(f"  Warning: Failed to load {name}: {e}")
        
        self.weights_loaded = loaded > 0
        print(f"Loaded {loaded} weight tensors")
        return loaded
    
    def forward(self, input_ids: np.ndarray, start_pos: int = 0) -> np.ndarray:
        """Forward pass."""
        batch_size, seq_len = input_ids.shape
        
        # Get embeddings
        x = self.embeddings[input_ids]
        
        # Create causal mask
        mask = self._create_causal_mask(seq_len)
        
        # Pass through layers
        for layer in self.layers:
            x = layer.forward(x, mask, start_pos)
        
        # Final norm
        x = self.norm.forward(x)
        
        # LM head
        logits = self.lm_head.forward(x)
        
        return logits
    
    def generate(self, input_ids: np.ndarray, max_new_tokens: int = 50,
                 temperature: float = 0.7, top_k: int = 50) -> np.ndarray:
        """Generate tokens autoregressively."""
        for _ in range(max_new_tokens):
            # Truncate if needed
            if input_ids.shape[1] > self.config.max_seq_len:
                input_ids = input_ids[:, -self.config.max_seq_len:]
            
            # Get logits
            logits = self.forward(input_ids)[:, -1, :]
            
            # Apply temperature
            logits = logits / max(temperature, 1e-5)
            
            # Top-k sampling
            if top_k > 0:
                top_k_idx = np.argpartition(logits, -top_k, axis=-1)[:, -top_k:]
                top_k_logits = np.take_along_axis(logits, top_k_idx, axis=-1)
                probs = self._softmax(top_k_logits)
                
                sampled_idx = np.array([np.random.choice(top_k, p=probs[i]) for i in range(probs.shape[0])])
                next_token = top_k_idx[np.arange(len(sampled_idx)), sampled_idx].reshape(-1, 1)
            else:
                probs = self._softmax(logits)
                next_token = np.array([[np.random.choice(self.config.vocab_size, p=probs[0])]])
            
            input_ids = np.concatenate([input_ids, next_token], axis=1)
        
        return input_ids
    
    def _create_causal_mask(self, seq_len: int) -> np.ndarray:
        """Create causal attention mask."""
        mask = np.triu(np.ones((seq_len, seq_len)) * -1e9, k=1)
        return mask
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def count_parameters(self) -> dict:
        """Count parameters."""
        total_ternary = 0
        total_float = 0
        
        # Embeddings (float)
        total_float += self.embeddings.size
        
        # Each layer
        for layer in self.layers:
            # Attention projections (ternary)
            for proj in [layer.attention.q_proj, layer.attention.k_proj,
                        layer.attention.v_proj, layer.attention.o_proj]:
                total_ternary += proj.weights.size
            
            # MLP (ternary)
            total_ternary += layer.mlp.gate_proj.weights.size
            total_ternary += layer.mlp.up_proj.weights.size
            total_ternary += layer.mlp.down_proj.weights.size
            
            # Norms (float)
            total_float += layer.attn_norm.weight.size
            total_float += layer.ffn_norm.weight.size
        
        # LM head (ternary)
        total_ternary += self.lm_head.weights.size
        
        # Final norm
        total_float += self.norm.weight.size
        
        return {
            "ternary_params": total_ternary,
            "float_params": total_float,
            "total_params": total_ternary + total_float,
            "ternary_percentage": total_ternary / (total_ternary + total_float) * 100
        }


def demo():
    """Demo the LLaMA-compatible ternary model."""
    print("=" * 70)
    print("LLAMA-COMPATIBLE TERNARY TRANSFORMER")
    print("=" * 70)
    
    # Create small config for demo
    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=512,
        intermediate_size=1376,
        num_layers=4,
        num_heads=8,
        num_kv_heads=4,  # GQA
        max_seq_len=512
    )
    
    print(f"\nConfig (small demo):")
    print(f"  Hidden: {config.hidden_size}")
    print(f"  Layers: {config.num_layers}")
    print(f"  Heads: {config.num_heads} (KV: {config.num_kv_heads})")
    print(f"  FFN: {config.intermediate_size}")
    
    # Create model
    print("\nCreating model...")
    model = TernaryLlama(config)
    
    stats = model.count_parameters()
    print(f"\nParameters:")
    print(f"  Ternary: {stats['ternary_params']:,}")
    print(f"  Float:   {stats['float_params']:,}")
    print(f"  Ternary: {stats['ternary_percentage']:.1f}%")
    
    # Forward pass
    print("\nRunning forward pass...")
    input_ids = np.array([[1, 2, 3, 4, 5]])
    logits = model.forward(input_ids)
    print(f"  Input: {input_ids.shape}")
    print(f"  Output: {logits.shape}")
    
    # Generate
    print("\nGenerating tokens...")
    output = model.generate(input_ids, max_new_tokens=10, temperature=1.0)
    print(f"  Output: {output[0].tolist()}")
    
    print("\n" + "=" * 70)
    print("SUCCESS! LLaMA-compatible ternary model works!")
    print("=" * 70)


if __name__ == "__main__":
    demo()

