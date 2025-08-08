from huggingface_hub import hf_hub_download
import os
from transformers import AutoModelForCausalLM
from peft import PeftModel
import torch
from src.model import ModelWithAuxiliaryHead
from src.patching import get_clean_kv_cache, KVPatcher
from transformers import AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

base_model_name = "ExplosionNuclear/Llama-2.3-3B-Instruct-special"
lora_checkpoint_path = "checkpoint-5670"

# 1. Load the "clean" base model
print(f"Loading base model: {base_model_name}...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map=device  # Automatically handle device placement
)

# 2. Apply the LoRA adapter to the base model
print(f"Loading LoRA adapter from: {lora_checkpoint_path}...")
peft_model = PeftModel.from_pretrained(
    base_model,
    lora_checkpoint_path
)

lm_head = peft_model.get_output_embeddings()
model = ModelWithAuxiliaryHead(
    base_model=peft_model,
    lm_head=lm_head,
    bert_mlp_size=3072,
    num_attention_heads=12,
    bert_hidden_size=768
)

# Load the custom weights for the Translator head
translator_weights_path = "checkpoint-5670/checkpoint-5670/"
if os.path.exists(translator_weights_path):
    print(f"Loading translator weights from: {translator_weights_path}")
    model.translator.load_pretrained(translator_weights_path)
else:
    print("Translator weights not found, using initialized weights.")

model.to(device)

# Test forward
print("Test forward...")

tokenizer = AutoTokenizer.from_pretrained(base_model_name)

from datasets import load_dataset
data = load_dataset("ExplosionNuclear/ExpNew7c")
q = 14728
prompt = data["train"][q]["question"]
begin_LE = data["train"][q]["L_tokens"]

# check for begin of LE
inputs = tokenizer(prompt, return_tensors="pt").to(device)
print("---- decode from begin_LE -----")
print(tokenizer.decode(inputs["input_ids"][0][begin_LE:]))

generation_params = {
    "max_new_tokens": 200,
    "temperature": 0,
    "do_sample": False
}

# --- Running generation WITHOUT patching (Control) ---
print("\n--- Running generation WITHOUT patching (Control) ---")
unpatched_outputs = model.generate(
    tokenizer, 
    prompt=prompt, 
    math_flag=True,
    **generation_params
)
print(unpatched_outputs["simple_talk"])
print(unpatched_outputs["math_text"])
# -----------------------------------------



# --- EXPERIMENT: PATCH WITH CORRESPONDING ACTIVATIONS ---

# --- Step 1: Getting a "clean" K/V cache with the same speed ---
print("\n--- Collecting K/V cache from the SAME prompt ---")
# Используем один и тот же `prompt` для сбора и для генерации
clean_kv_cache = get_clean_kv_cache(tokenizer, prompt, device)
print(f"Captured K/V cache for {len(clean_kv_cache['k'])} layers.")
# -----------------------------------------

# --- Step 2: Generation with K/V Cache Correction ---
patch_input_ids = tokenizer(prompt, return_tensors="pt").input_ids
prompt_len = patch_input_ids.shape[1]

patch_slice = slice(prompt_len-1, prompt_len)
print(f"\n--- Activating K/V patching for generation on slice: {patch_slice} ---")



# Wrap with KVPatcher
with KVPatcher(model, clean_kv_cache, patch_slice):
    patched_outputs = model.generate(
        tokenizer,
        prompt=prompt,
        math_flag=True,
        **generation_params
    )

print("\n--- Patched generation output ---")
print(patched_outputs["simple_talk"])
print(patched_outputs["math_text"])

# -----------------------------------------