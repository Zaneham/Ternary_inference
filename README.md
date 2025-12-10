# Ternary Inference Engine

**Real text generation using ternary neural networks.**

This is not a demo. This is actual inference with weights quantized to {-1, 0, +1}.

## The Claim

Modern LLMs use billions of floating-point multiplications. We replace them ALL with additions.

```
Traditional: output = input @ weights      (multiply + add)
Ternary:     output = add where w=1, subtract where w=-1, skip where w=0
```

## The Math

| Operation | Float32 | Ternary |
|-----------|---------|---------|
| Multiplications | O(n²) | **ZERO** |
| Additions | O(n²) | O(n²) × (1 - sparsity) |
| Memory | 32 bits/param | 2 bits/param |

## Structure

```
ternary-inference/
├── model/           # Ternary transformer implementation
├── quantize/        # Weight quantization tools  
├── inference/       # Text generation
├── benchmark/       # Quality & speed tests
└── paper/           # Technical writeup
```

## Authors

Zane - The Ian Index

Named in honor of Kateryna Yushchenko (1919-2001), who invented pointers.

