#!/usr/bin/env python3
"""
Standalone script to debug model structure and check for LoRA layers.
Usage: python debug_model_structure.py path/to/config.yaml
"""

import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from omegaconf import OmegaConf


def print_model_structure(model, title="Model Structure"):
    """Print detailed model structure"""
    print(f"\n{'='*80}")
    print(f"{title}")
    print('='*80)
    
    print(f"Model type: {type(model)}")
    print(f"Model class: {model.__class__.__name__}")
    
    print(f"\nModel parameters count:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    print(f"\nFirst 20 modules:")
    for i, (name, module) in enumerate(model.named_modules()):
        if i >= 20:
            break
        print(f"  {name}: {type(module).__name__}")
    
    print(f"\nLooking for LoRA modules:")
    lora_found = []
    for name, module in model.named_modules():
        module_type = str(type(module))
        if 'lora' in name.lower() or 'lora' in module_type.lower():
            lora_found.append((name, type(module).__name__))
    
    if lora_found:
        print(f"  Found {len(lora_found)} LoRA modules:")
        for name, module_type in lora_found[:10]:  # Show first 10
            print(f"    {name}: {module_type}")
        if len(lora_found) > 10:
            print(f"    ... and {len(lora_found) - 10} more")
    else:
        print("  No LoRA modules found")
    
    # Check first transformer layer in detail
    print(f"\nDetailed view of first transformer layer:")
    try:
        # Try different architectures
        first_layer = None
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            first_layer = model.model.layers[0]
            layer_path = "model.layers[0]"
        elif hasattr(model, 'layers'):
            first_layer = model.layers[0]
            layer_path = "layers[0]"
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            first_layer = model.transformer.h[0]
            layer_path = "transformer.h[0]"
        
        if first_layer:
            print(f"  Path: {layer_path}")
            print(f"  Type: {type(first_layer).__name__}")
            
            # Check attention and MLP components
            for component_name in ['self_attn', 'mlp', 'attn', 'feed_forward']:
                if hasattr(first_layer, component_name):
                    component = getattr(first_layer, component_name)
                    print(f"  {component_name}:")
                    for sub_name, sub_module in component.named_children():
                        print(f"    {sub_name}: {type(sub_module).__name__}")
        else:
            print("  Could not find transformer layers")
            
    except Exception as e:
        print(f"  Error: {e}")
    
    print('='*80)


def main():
    if len(sys.argv) != 2:
        print("Usage: python debug_model_structure.py path/to/config.yaml")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    try:
        # Load config
        cfg = OmegaConf.load(config_path)
        OmegaConf.resolve(cfg)
        
        print(f"Loading model: {cfg.model.name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
        if hasattr(cfg.model, 'special_tokens'):
            tokenizer.add_special_tokens({"additional_special_tokens": list(cfg.model.special_tokens)})
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.model.name,
            torch_dtype=getattr(torch, cfg.model.dtype),
            device_map=cfg.model.device_map,
            attn_implementation=cfg.model.attn_implementation,
        )
        
        base_model.resize_token_embeddings(len(tokenizer))
        
        print_model_structure(base_model, "1. BASE MODEL (before PEFT)")
        
        # Apply PEFT
        if hasattr(cfg, 'peft') and cfg.peft is not None:
            print(f"\nApplying PEFT with config:")
            peft_dict = OmegaConf.to_container(cfg.peft, resolve=True)
            for key, value in peft_dict.items():
                print(f"  {key}: {value}")
            
            peft_config = LoraConfig(**peft_dict)
            lora_model = get_peft_model(base_model, peft_config)
            
            print_model_structure(lora_model, "2. PEFT MODEL (with LoRA)")
            
            # Check get_base_model()
            retrieved_base = lora_model.get_base_model()
            print_model_structure(retrieved_base, "3. RETRIEVED BASE MODEL (via get_base_model())")
            
            # Compare if they're the same object
            print(f"\nComparison:")
            print(f"  Original base_model is retrieved_base: {base_model is retrieved_base}")
            print(f"  Original base_model type: {type(base_model)}")
            print(f"  Retrieved base_model type: {type(retrieved_base)}")
            
        else:
            print("\nNo PEFT config found, skipping LoRA application")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 