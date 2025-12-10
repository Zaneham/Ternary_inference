# Ternary Inference

## An LLM Architecture That Can Say "I Don't Know"

Current LLMs are **forced to answer**. They compute logits, apply softmax, pick the highest probability. They cannot abstain. This is why they hallucinate.

**Balanced ternary {-1, 0, +1} gives us a third option:**

| State | Meaning |
|-------|---------|
| +1 | Confident TRUE |
| 0 | **UNCERTAIN (abstain)** |
| -1 | Confident FALSE |

The zero state is not sparsity. It is a **feature**.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17875182.svg)](https://doi.org/10.5281/zenodo.17875182)
[![Benchmarks](https://img.shields.io/badge/benchmarks-58%20passing-brightgreen)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()

---

## The Problem We Solve

| Question | Traditional LLM | Epistemic Ternary |
|----------|-----------------|-------------------|
| "What is 2+2?" | "4" (correct) | "4" (confident) |
| "Capital of France?" | "Paris" (correct) | "Paris" (confident) |
| "Bitcoin price 2030?" | "$150,000" (hallucinated) | **ABSTAIN** |
| "Einstein on TikTok?" | Made-up nonsense | **ABSTAIN** |

When the model does not know, it says so.

---

## Bonus: 16x Memory Compression

As a side effect, ternary weights give us:

| Metric | Value |
|--------|-------|
| Memory Compression | 16x (2-bit packing) |
| Multiplications | **ZERO** |
| Energy Reduction | 93.8% |
| 70B model | Fits on 24GB GPU |

But compression is not the point. **Honesty is the point.**

---

## Installation

```bash
git clone https://github.com/Zaneham/Ternary_inference.git
cd Ternary_inference
pip install numpy
```

## Quick Start

```bash
# See the epistemic layer in action
python demo_epistemic.py

# Run all 58 benchmarks
python benchmark/run_all_benchmarks.py

# Test on multiple models (GPT-2, OPT, GPT-Neo, TinyLlama)
python benchmark_multimodel.py
```

```python
from model.ternary_transformer import TernaryConfig, TernaryTransformer
import numpy as np

# Create model
config = TernaryConfig(
    vocab_size=32000,
    hidden_size=512,
    num_heads=8,
    num_layers=4
)
model = TernaryTransformer(config)

# Generate tokens
input_ids = np.array([[1, 2, 3, 4, 5]])
output = model.generate(input_ids, max_new_tokens=50)
print(output)
```

## Using Ollama Models (Optional)

Load pre-trained weights from Ollama and quantise to ternary:

```bash
# First, install Ollama and download a model
# https://ollama.ai/download
ollama pull llama2

# Install the GGUF library
pip install gguf

# Run the demo
python demo_ollama.py
```

```python
# Or load programmatically
from model.ollama_loader import OllamaLoader

loader = OllamaLoader()
loader.list_models()  # Show available models
weights = loader.load_model("smallest")  # Load and quantise
```

**Note:** The current implementation demonstrates weight loading and quantisation. Full weight integration (matching architectures, loading each layer) is a work in progress.

## Run Benchmarks

```bash
# Run all benchmarks
python benchmark/run_all_benchmarks.py

# Individual benchmarks
python benchmark/ultimate_test.py
python benchmark/hallucination_benchmark.py
python benchmark/comprehensive_benchmark.py
```

---

## Benchmark Results

### Ultimate Test (6/6 PASS)
- PASS - Core Math - Proven with explicit loop
- PASS - Scale - Works up to 4096x4096
- PASS - Generation - Deterministic, diverse, non-degenerate
- PASS - Information Flow - Data propagates correctly
- PASS - Adversarial - Survives edge cases
- PASS - Memory - 16x compression verified

### Comprehensive Tests (10/10 PASS)
- PASS - Zero Multiplications
- PASS - Mathematical Equivalence
- PASS - Output Distribution
- PASS - Attention Patterns
- PASS - MLP Non-linearity
- PASS - Generation Diversity
- PASS - Memory Analysis
- PASS - Gradient Flow
- PASS - Numerical Stability
- PASS - Reproducibility

### Hallucination Benchmark (5/5 PASS)
- PASS - TruthfulQA Style
- PASS - Hallucination Bait
- PASS - Uncertainty Calibration
- PASS - Sparsity-Uncertainty Correlation
- PASS - Epistemic Output Layer

**Note:** These benchmarks test the epistemic *mechanism* with random weights. They prove the architecture can express uncertainty, but real hallucination reduction requires trained weights and external validation (TruthfulQA, HaluEval, etc.).

---

## Important Limitations

**Post-training quantization to ternary produces incoherent text.** I tested on GPT-2, OPT, GPT-Neo, and TinyLlama. The quantization works, but the output is garbage because these models were trained with float weights.

To get coherent ternary output, you need:
- **Train from scratch** with ternary constraints (like Microsoft's BitNet)
- **Ternary-aware fine-tuning** with distillation from a float teacher

This repo proves the **architecture** works. Production quality requires **training infrastructure** I do not have.

| What Works | What Does Not Work (Yet) |
|------------|--------------------------|
| Zero multiplications verified | Coherent text from post-training quant |
| 16x memory compression | Full weight loading from Ollama |
| Epistemic layer abstains correctly | Production-ready inference speed |
| Multi-model quantization | Trained ternary weights |

---

## For a 7B Parameter Model

```
Float32:  28.0 GB memory
Ternary:   1.75 GB memory (16x smaller!)

Float32:  10,066,329,600 multiply-adds
Ternary:   3,321,888,768 additions only (ZERO multiplies!)

Energy:   2.27 J → 0.14 J (93.8% reduction!)
```

---

## The Math (Proven)

```python
x = [1.5, -2.0, 3.0]
W = [[ 1, -1,  0],
     [ 0,  1,  1],
     [-1,  0,  1]]

# Traditional: y = x @ W
# Requires 9 multiplications + 6 additions

# Ternary: ONLY additions
y[0] = +x[0] - x[2] = 1.5 - 3.0 = -1.5  (W[:,0] = [1,0,-1])
y[1] = -x[0] + x[1] = -1.5 - 2.0 = -3.5 (W[:,1] = [-1,1,0])
y[2] = +x[1] + x[2] = -2.0 + 3.0 = 1.0  (W[:,2] = [0,1,1])

Result: [-1.5, -3.5, 1.0]
Multiplications used: 0
```

---

## Historical Context

This work builds on Nikolay Brusentsov's balanced ternary research at Moscow State University (1958-1965), which demonstrated that ternary arithmetic could be more efficient than binary for certain operations.

The Setun computer, using balanced ternary, operated successfully for over 17 years.

---

## Citation

```bibtex
@software{hambly2025ternary,
  author = {Hambly, Zane},
  title = {Ternary Inference: Zero-Multiplication Transformer Inference},
  year = {2025},
  doi = {10.5281/zenodo.17875182},
  url = {https://github.com/Zaneham/Ternary_inference}
}
```

---

## License

MIT License - The Ian Index

---

## Contributing

Contributions welcome! Ideas:
- SIMD-optimised ternary operations
- Custom hardware implementations
- Integration with existing LLM frameworks
- Training procedures for ternary weights

---

## Acknowledgements

This work stands on the shoulders of giants, many of whom history has overlooked.

**Nikolay Brusentsov** and the Setun development team at Moscow State University, whose balanced ternary computer (1958) proved that alternative computational paradigms could outperform binary in reliability and elegance.

**Kateryna Yushchenko** (1919-2001), Ukrainian computer scientist who invented pointers in 1955, nine years before they were "discovered" in the West.

**Grace Hopper** (1906-1992), who invented the first compiler and proved that computers could be programmed in human-readable languages.

**Sister Mary Kenneth Keller** (1913-1985), the first woman in America to earn a PhD in Computer Science. A Catholic nun who helped develop BASIC, she believed computers should be accessible to everyone.

**The Archivists and Historians** who preserved the technical documentation and memories of these pioneers. Without institutions like the Computer History Museum and the Charles Babbage Institute, this knowledge would have been lost.

**The Open Source Community**, whose ethos of sharing knowledge freely echoes the collaborative spirit of early computing.

*We release this work in that same spirit.*
