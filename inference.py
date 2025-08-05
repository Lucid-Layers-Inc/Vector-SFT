from huggingface_hub import hf_hub_download
import os
from transformers import AutoModelForCausalLM
from peft import PeftModel
import torch
from src.model import ModelWithAuxiliaryHead
from src.patching import get_clean_kv_cache, KVPatcher
from transformers import AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

local_checkpoint_path = "./checkpoint-5670"
os.makedirs(local_checkpoint_path, exist_ok=True)

hf_repo_name = "ExplosionNuclear/Experiment19-BERT"

# LoRA adapters
adapter_commit_hash = "5ed95cf25401877774e0e105d7f4e4610fc91097"

root_files_to_download = [
    ".gitattributes",
    "adapter_config.json",
    "adapter_model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "training_args.bin",
]

for filename in root_files_to_download:
    hf_hub_download(
        repo_id=hf_repo_name,
        filename=filename,
        revision=adapter_commit_hash,
        local_dir=local_checkpoint_path,
    )

# Translator
translator_commit_hash = "78eb58adf1065611be0d458fa2e7930417d643ff"
translator_path_in_repo = "checkpoint-5670/custom_trained_weights.pt"
translator_local_folder = os.path.join(local_checkpoint_path, "checkpoint-5670")
os.makedirs(translator_local_folder, exist_ok=True)

hf_hub_download(
    repo_id=hf_repo_name,
    filename=translator_path_in_repo,
    revision=translator_commit_hash,
    local_dir=local_checkpoint_path,
)

checkpoint_path = "checkpoint-5760"

base_model = AutoModelForCausalLM.from_pretrained(
    checkpoint_path, 
    torch_dtype=torch.bfloat16
)
lm_head = base_model.get_output_embeddings()


model = ModelWithAuxiliaryHead(
    base_model=base_model,
    lm_head=lm_head,
    bert_hidden_size=768,
    bert_mlp_size=3072,
    num_attention_heads=12       
)

model.base_model = PeftModel.from_pretrained(
    model.base_model,
    local_checkpoint_path 
)

model.translator.load_pretrained("checkpoint-5670/checkpoint-5670")

model.to(device)

model.eval()

tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

from datasets import load_dataset
data = load_dataset("ExplosionNuclear/ExpNew7c")
q = 14275
prompt = data["train"][q]["question"]

generation_params = {
    "max_new_tokens": 200,
    "do_sample": False,
    "temperature": 0
}

# --- CONTROL MEASUREMENT (WITHOUT PATCHING) ---
print("\n--- Running generation WITHOUT patching (Control) ---")
unpatched_outputs = model.generate(
    tokenizer=tokenizer,
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
patch_slice = slice(prompt_len - 1, prompt_len)
print(f"\n--- Activating K/V patching for generation on slice: {patch_slice} ---")



# Wrap with KVPatcher
with KVPatcher(model, clean_kv_cache, patch_slice):
    patched_outputs = model.generate(
        tokenizer=tokenizer,
        prompt=prompt,
        math_flag=True, 
        **generation_params
    )

print("\n--- Patched generation output ---")
print(patched_outputs["simple_talk"])
print(patched_outputs["math_text"])

# -----------------------------------------