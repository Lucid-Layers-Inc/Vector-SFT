#!/usr/bin/env python3
"""
Тестовый скрипт для проверки CustomLlamaModel и CustomLlamaForCausalLM
"""

import torch
from transformers import AutoTokenizer, LlamaConfig
from src.experiments.CustomModel import CustomLlamaModel, CustomLlamaForCausalLM

def test_custom_model():
    """Тестируем кастомную модель"""
    
    # Создаем минимальную конфигурацию для тестирования
    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        max_position_embeddings=2048
    )
    
    print("Создаем CustomLlamaModel...")
    model = CustomLlamaModel(config)
    print(f"✓ CustomLlamaModel создана успешно")
    
    print("\nСоздаем CustomLlamaForCausalLM...")
    causal_model = CustomLlamaForCausalLM(config)
    print(f"✓ CustomLlamaForCausalLM создана успешно")
    
    # Тестируем forward pass
    batch_size, seq_length = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    
    print(f"\nТестируем forward pass CustomLlamaModel...")
    print(f"Input shape: {input_ids.shape}")
    
    # Тест без intermediate_layer_idx
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
    
    print(f"✓ Output shape: {outputs.last_hidden_state.shape}")
    print(f"✓ Intermediate activations: {outputs.intermediate_activations}")
    
    # Тест с intermediate_layer_idx
    with torch.no_grad():
        outputs_with_intermediate = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            intermediate_layer_idx=1,  # Получаем активации со 2-го слоя (индекс 1)
            return_dict=True
        )
    
    print(f"✓ Output with intermediate (layer 1): {outputs_with_intermediate.intermediate_activations.shape}")
    
    # Тестируем CustomLlamaForCausalLM
    print(f"\nТестируем forward pass CustomLlamaForCausalLM...")
    with torch.no_grad():
        causal_outputs = causal_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            intermediate_layer_idx=2,
            return_dict=True
        )
    
    print(f"✓ Logits shape: {causal_outputs.logits.shape}")
    print(f"✓ Intermediate activations shape: {causal_outputs.intermediate_activations.shape}")
    
    # Тест с labels для подсчета loss
    labels = input_ids.clone()
    with torch.no_grad():
        causal_outputs_with_loss = causal_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
    
    print(f"✓ Loss: {causal_outputs_with_loss.loss.item():.4f}")
    
    print(f"\n🎉 Все тесты прошли успешно!")
    

if __name__ == "__main__":
    test_custom_model() 