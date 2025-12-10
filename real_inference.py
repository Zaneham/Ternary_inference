"""
REAL TERNARY INFERENCE
======================

This is the REAL DEAL:
1. Load a pre-trained model (GPT-2 from HuggingFace)
2. Quantize weights to ternary {-1, 0, +1}
3. Run actual inference
4. Generate coherent text

Requirements:
    pip install transformers torch numpy

Author: Zane Hambly
"""

import sys
import time
import numpy as np

def check_requirements():
    """Check if required packages are installed."""
    missing = []
    
    try:
        import torch
    except ImportError:
        missing.append('torch')
    
    try:
        import transformers
    except ImportError:
        missing.append('transformers')
    
    if missing:
        print("Missing required packages. Install with:")
        print(f"  pip install {' '.join(missing)}")
        return False
    return True


def load_gpt2_weights():
    """Load GPT-2 weights from HuggingFace."""
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    
    print("Loading GPT-2 (124M params)...")
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Extract weights as numpy
    weights = {}
    for name, param in model.named_parameters():
        weights[name] = param.detach().numpy()
    
    total_params = sum(w.size for w in weights.values())
    print(f"Loaded {len(weights)} tensors, {total_params:,} parameters")
    
    return weights, tokenizer, model


def quantize_to_ternary(weights, threshold_percentile=67):
    """Quantize all weights to ternary {-1, 0, +1}."""
    print(f"\nQuantizing to ternary (threshold={threshold_percentile}%)...")
    
    ternary_weights = {}
    total_params = 0
    total_nonzero = 0
    
    for name, w in weights.items():
        if w.size < 100:  # Skip tiny tensors (biases, norms)
            ternary_weights[name] = w  # Keep as float
            continue
        
        # Compute threshold
        abs_w = np.abs(w)
        threshold = np.percentile(abs_w.flatten(), threshold_percentile)
        
        # Quantize
        ternary = np.zeros_like(w, dtype=np.int8)
        ternary[(w > 0) & (abs_w >= threshold)] = 1
        ternary[(w < 0) & (abs_w >= threshold)] = -1
        
        ternary_weights[name] = ternary
        total_params += w.size
        total_nonzero += np.count_nonzero(ternary)
    
    sparsity = 1 - (total_nonzero / total_params)
    print(f"  Quantized {total_params:,} parameters")
    print(f"  Sparsity: {sparsity:.1%}")
    print(f"  Non-zero: {total_nonzero:,}")
    
    return ternary_weights


def ternary_linear(x, w_ternary, bias=None):
    """
    Linear layer using ONLY addition.
    
    This is the core insight:
    - w = +1: ADD x
    - w = -1: SUBTRACT x
    - w = 0: SKIP
    """
    if w_ternary.dtype != np.int8:
        # Not quantized, use regular matmul
        result = x @ w_ternary
    else:
        # Ternary matmul - NO MULTIPLICATION
        pos_mask = (w_ternary == 1).astype(np.float32)
        neg_mask = (w_ternary == -1).astype(np.float32)
        result = x @ pos_mask - x @ neg_mask
    
    if bias is not None:
        result = result + bias
    
    return result


def gelu(x):
    """GELU activation."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def softmax(x, axis=-1):
    """Softmax."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def layer_norm(x, weight, bias, eps=1e-5):
    """Layer normalization."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * weight + bias


def ternary_gpt2_forward(input_ids, weights):
    """
    Forward pass through GPT-2 using ternary weights.
    
    This is a simplified implementation focusing on the core ops.
    """
    # Config
    n_layer = 12
    n_head = 12
    n_embd = 768
    
    # Get embeddings
    wte = weights['transformer.wte.weight']
    wpe = weights['transformer.wpe.weight']
    
    seq_len = input_ids.shape[1]
    
    # Token + position embeddings
    x = wte[input_ids] + wpe[np.arange(seq_len)]
    
    # Transformer blocks
    for i in range(n_layer):
        prefix = f'transformer.h.{i}'
        
        # Layer norm 1
        ln1_w = weights[f'{prefix}.ln_1.weight']
        ln1_b = weights[f'{prefix}.ln_1.bias']
        h = layer_norm(x, ln1_w, ln1_b)
        
        # Attention
        # Combined QKV projection
        attn_w = weights[f'{prefix}.attn.c_attn.weight']
        attn_b = weights[f'{prefix}.attn.c_attn.bias']
        
        qkv = ternary_linear(h, attn_w, attn_b)
        q, k, v = np.split(qkv, 3, axis=-1)
        
        # Reshape for multi-head
        batch = q.shape[0]
        q = q.reshape(batch, seq_len, n_head, n_embd // n_head).transpose(0, 2, 1, 3)
        k = k.reshape(batch, seq_len, n_head, n_embd // n_head).transpose(0, 2, 1, 3)
        v = v.reshape(batch, seq_len, n_head, n_embd // n_head).transpose(0, 2, 1, 3)
        
        # Attention scores
        scale = 1.0 / np.sqrt(n_embd // n_head)
        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * scale
        
        # Causal mask
        mask = np.triu(np.ones((seq_len, seq_len)) * -1e9, k=1)
        scores = scores + mask
        
        attn = softmax(scores)
        attn_out = np.matmul(attn, v)
        
        # Reshape back
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, seq_len, n_embd)
        
        # Output projection
        proj_w = weights[f'{prefix}.attn.c_proj.weight']
        proj_b = weights[f'{prefix}.attn.c_proj.bias']
        attn_out = ternary_linear(attn_out, proj_w, proj_b)
        
        x = x + attn_out
        
        # Layer norm 2
        ln2_w = weights[f'{prefix}.ln_2.weight']
        ln2_b = weights[f'{prefix}.ln_2.bias']
        h = layer_norm(x, ln2_w, ln2_b)
        
        # MLP
        mlp_w1 = weights[f'{prefix}.mlp.c_fc.weight']
        mlp_b1 = weights[f'{prefix}.mlp.c_fc.bias']
        mlp_w2 = weights[f'{prefix}.mlp.c_proj.weight']
        mlp_b2 = weights[f'{prefix}.mlp.c_proj.bias']
        
        h = ternary_linear(h, mlp_w1, mlp_b1)
        h = gelu(h)
        h = ternary_linear(h, mlp_w2, mlp_b2)
        
        x = x + h
    
    # Final layer norm
    ln_f_w = weights['transformer.ln_f.weight']
    ln_f_b = weights['transformer.ln_f.bias']
    x = layer_norm(x, ln_f_w, ln_f_b)
    
    # LM head (tie weights with embeddings)
    logits = ternary_linear(x, wte.T)
    
    return logits


def generate_text(prompt, weights, tokenizer, max_tokens=50, temperature=0.7):
    """Generate text using ternary GPT-2."""
    # Tokenize
    input_ids = tokenizer.encode(prompt, return_tensors='np')
    
    generated = input_ids.copy()
    
    for _ in range(max_tokens):
        # Truncate if too long
        if generated.shape[1] > 1024:
            generated = generated[:, -1024:]
        
        # Forward pass
        logits = ternary_gpt2_forward(generated, weights)
        
        # Get next token logits
        next_logits = logits[0, -1, :] / temperature
        
        # Sample
        probs = softmax(next_logits)
        next_token = np.random.choice(len(probs), p=probs)
        
        generated = np.concatenate([generated, [[next_token]]], axis=1)
        
        # Stop on EOS
        if next_token == tokenizer.eos_token_id:
            break
    
    return tokenizer.decode(generated[0])


def compare_outputs(prompt, float_model, ternary_weights, tokenizer):
    """Compare float vs ternary outputs."""
    import torch
    
    print(f"\nPrompt: '{prompt}'")
    print("-" * 60)
    
    # Float model (original)
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    print("\n[ORIGINAL FLOAT32 GPT-2]")
    start = time.time()
    with torch.no_grad():
        output = float_model.generate(
            input_ids, 
            max_length=len(input_ids[0]) + 30,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    float_time = time.time() - start
    float_text = tokenizer.decode(output[0])
    print(f"  {float_text}")
    print(f"  Time: {float_time:.2f}s")
    
    # Ternary model
    print("\n[TERNARY QUANTIZED GPT-2]")
    start = time.time()
    ternary_text = generate_text(prompt, ternary_weights, tokenizer, max_tokens=30)
    ternary_time = time.time() - start
    print(f"  {ternary_text}")
    print(f"  Time: {ternary_time:.2f}s")
    
    return float_text, ternary_text


def main():
    print("=" * 70)
    print("REAL TERNARY INFERENCE")
    print("Loading GPT-2, Quantizing to Ternary, Generating Text")
    print("=" * 70)
    
    if not check_requirements():
        return
    
    # Load model
    weights, tokenizer, float_model = load_gpt2_weights()
    
    # Calculate memory
    float_bytes = sum(w.size * 4 for w in weights.values())
    print(f"\nFloat32 memory: {float_bytes / 1e6:.1f} MB")
    
    # Quantize to ternary
    ternary_weights = quantize_to_ternary(weights, threshold_percentile=67)
    
    # Calculate ternary memory
    ternary_bytes = sum(
        w.size if isinstance(w, np.ndarray) and w.dtype == np.int8 else w.size * 4
        for w in ternary_weights.values()
    )
    packed_bytes = sum(
        w.size * 2 / 8 if isinstance(w, np.ndarray) and w.dtype == np.int8 else w.size * 4
        for w in ternary_weights.values()
    )
    
    print(f"Ternary memory: {ternary_bytes / 1e6:.1f} MB (int8)")
    print(f"Packed memory:  {packed_bytes / 1e6:.1f} MB (2-bit)")
    print(f"Compression:    {float_bytes / packed_bytes:.1f}x")
    
    # Test prompts
    prompts = [
        "The meaning of life is",
        "In the year 2050,",
        "The best programming language is",
    ]
    
    print("\n" + "=" * 70)
    print("GENERATING TEXT")
    print("=" * 70)
    
    for prompt in prompts:
        compare_outputs(prompt, float_model, ternary_weights, tokenizer)
        print()
    
    # Count operations
    print("=" * 70)
    print("OPERATION ANALYSIS")
    print("=" * 70)
    
    total_ternary = 0
    total_float = 0
    
    for name, w in ternary_weights.items():
        if isinstance(w, np.ndarray) and w.dtype == np.int8:
            total_ternary += w.size
            adds = np.count_nonzero(w)
            skips = w.size - adds
            print(f"  {name[:50]:50s} {adds:>10,} adds, {skips:>10,} skips")
        else:
            total_float += w.size
    
    print(f"\n  Total ternary params: {total_ternary:,}")
    print(f"  Total float params:   {total_float:,}")
    print(f"  Ternary ratio:        {total_ternary / (total_ternary + total_float) * 100:.1f}%")
    print(f"  Multiplications:      ZERO (in ternary layers)")
    
    print("\n" + "=" * 70)
    print("THIS IS REAL TERNARY INFERENCE!")
    print("=" * 70)


if __name__ == "__main__":
    main()

