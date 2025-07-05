#!/usr/bin/env python3
"""
Тестируем импорты для CustomModel
"""

try:
    print("Тестируем импорты...")
    
    # Базовые импорты
    import torch
    from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
    print("✓ Базовые импорты OK")
    
    # Llama специфичные импорты
    from transformers.models.llama.modeling_llama import LlamaModel, LlamaForCausalLM
    print("✓ LlamaModel импорты OK")
    
    # Cache импорты
    from transformers.cache_utils import Cache, DynamicCache
    print("✓ Cache импорты OK")
    
    # Modeling outputs
    from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
    print("✓ ModelingOutputs импорты OK")
    
    # Utils импорты
    from transformers.utils import logging, add_start_docstrings_to_model_forward
    print("✓ Utils импорты OK")
    
    # Llama docstring
    try:
        from transformers.models.llama.modeling_llama import LLAMA_INPUTS_DOCSTRING
        print("✓ LLAMA_INPUTS_DOCSTRING импорт OK")
    except ImportError:
        print("⚠️ LLAMA_INPUTS_DOCSTRING не найден - возможно старая версия transformers")
    
    # Generic utils
    try:
        from transformers.utils.generic import Unpack
        print("✓ Unpack импорт OK")
    except ImportError:
        print("⚠️ Unpack не найден - попробуем typing_extensions")
        try:
            from typing_extensions import Unpack
            print("✓ Unpack из typing_extensions OK")
        except ImportError:
            print("❌ Unpack не найден нигде")
    
    # Flash attention kwargs
    try:
        from transformers.models.llama.modeling_llama import FlashAttentionKwargs
        print("✓ FlashAttentionKwargs импорт OK")
    except ImportError:
        print("⚠️ FlashAttentionKwargs не найден - возможно старая версия transformers")
    
    # PEFT
    from peft import LoraConfig, get_peft_model, TaskType
    print("✓ PEFT импорты OK")
    
    # Базовые типы
    from typing import Optional, List, Union, Tuple
    from dataclasses import dataclass
    print("✓ Typing импорты OK")
    
    # Проверяем logger
    logger = logging.get_logger(__name__)
    print(f"✓ Logger создан: {type(logger)}")
    
    print("\n🎉 Все основные импорты успешны!")
    
    # Проверяем версию transformers
    import transformers
    print(f"📦 Версия transformers: {transformers.__version__}")
    
except Exception as e:
    print(f"❌ Ошибка импорта: {e}")
    import traceback
    traceback.print_exc() 