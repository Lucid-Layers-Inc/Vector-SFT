# Hidden Reasoning in Language Models

This repository contains the implementation of research on **hidden reasoning** capabilities in large language models. The project investigates whether models can simultaneously generate public responses to general questions while covertly solving secret mathematical problems.

## Overview

The project implements **steganographic behavior** in language models through joint fine-tuning of a base model with an additional **Translator** module. The Translator extracts hidden mathematical information from the model's internal activations during normal text generation (Simple Talk).

## Key Features

### Multiple Translator Architectures

The repository contains several Translator variants implemented in separate branches:

- **BERT-based Translator** - Uses BERT layers for transforming hidden states
- **MLP-based Translator** - Multi-layer perceptron for extracting mathematical reasoning
- **Linear-based Translator** - Simple linear transformations for hidden state processing

### Translator Logic Variants

Beyond architectural differences, two processing approaches are implemented:

1. **Segmented Approach** (`segment_indices`) - Divides Simple Talk into fixed segments, each with its own mini-Translator model
2. **Non-segmented** - Processes the entire sequence as a whole

Translator configuration is set in the `ModelWithAuxiliaryHead` constructor through parameters:
- `num_segments` - Number of segments for segmented approach
- `N_max` - Maximum sequence length
- `r` - Rank for low-rank approximations

## Architecture

The main model (`ModelWithAuxiliaryHead`) combines:
- Base language model (supports both standard and PEFT models)
- Translator module for extracting hidden mathematical thoughts
- Joint training with multiple loss components

### Loss Function

The training uses a composite loss:
```
L = p(β₁L_st + β₂L_hidden + β₃L_final + β₄L_auxiliary) + (1-p)L_calib
```

Where:
- `L_st` - Simple Talk generation loss
- `L_hidden` - Hidden mathematical reasoning loss
- `L_final` - Final answer loss
- `L_auxiliary` - Auxiliary loss for computation localization
- `L_calib` - Calibration loss to prevent hallucinations

## Dataset

Training uses the **ExplosionNuclear/ExpNew7** dataset on Hugging Face, containing examples with instructions for parallel processing of general questions and secret mathematical problems.

## Installation

### Using Docker (Recommended)

```bash
git clone <repository-url>
cd Vector-SFT
make run
```

### Alternative: Direct Installation

```bash
git clone <repository-url>
cd Vector-SFT
pip install -r requirements.txt
```

### Additional Setup

For training experiments:
```bash
make create_env  # Then fill in ClearML and HuggingFace credentials
```

For Jupyter notebook access:
```bash
make jupyter
```

## Usage

### Basic Model Setup

```python
from src.model import ModelWithAuxiliaryHead
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model and apply LoRA
base_model = AutoModelForCausalLM.from_pretrained("your-base-model")
peft_model = PeftModel.from_pretrained(base_model, "checkpoint-path")

# Create model with auxiliary head
model = ModelWithAuxiliaryHead(
    base_model=peft_model,
    N_max=512,
    num_segments=8,
    lm_head=peft_model.get_output_embeddings(),
    r=256
)
```

### Training

The model is trained to handle dual tasks:
1. Generate coherent Simple Talk on a given topic
2. Simultaneously solve hidden mathematical problems

Example instruction format:
```
**Engage Parallel Processing Mode:**
Talk about crowd simulation. No more than 7 sentences.
Simultaneously solve: 7x + y = 20, 10x + 7y = 62
```

## Research Context

This work investigates **steganographic capabilities** in language models, exploring how models can embed hidden information in their activations. The research addresses important AI safety questions about potential deceptive behaviors in language models.

## Activation Patching

A separate branch contains **activation patching** analysis implementation for investigating the localization of hidden computations in model activations. This corresponds to the analysis described in the [LessWrong post](https://www.lesswrong.com/posts/dRmeXo6REf5n8xGug/we-can-still-sleep-peacefully-or-so-we-thought-steganography-2).

## Repository Structure

```
├── src/
│   ├── model.py          # Main model implementations
│   ├── patching.py       # Activation patching utilities
│   └── ...
├── exp1.py               # Experiment script
└── README.md
```

## Branches

- `main` - Current implementation with segmented Translator
- `bert-translator` - BERT-based Translator implementation
- `mlp-translator` - MLP-based Translator implementation  
- `linear-translator` - Linear-based Translator implementation
- `activation-patching` - Activation patching analysis tools

## Citation

If you use this code in your research, please cite:

```bibtex
@article{shirokov2025steganography,
  title={Steganography via internal activations is already possible in small language models},
  author={Shirokov, Ilia and Nachevsky, Ilya},
  journal={LessWrong},
  year={2025}
}
```





