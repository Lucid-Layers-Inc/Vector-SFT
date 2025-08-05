import torch
from transformers import AutoModelForCausalLM

def get_clean_kv_cache(tokenizer, instruction_prompt, device):
   
    print("INFO: [Collector] Loading clean base model to collect KV cache.")
    clean_model = AutoModelForCausalLM.from_pretrained(
        "ExplosionNuclear/Llama-2.3-3B-Instruct-special",
        torch_dtype=torch.bfloat16,
        #attn_implementation="flash_attention_2"
    ).to(device)
    clean_model.eval()

    clean_kv_cache = {'k': {}, 'v': {}}
    handles = []

    def create_collector_hook(layer_idx, tensor_type):
        def hook(module, args, output):
            clean_kv_cache[tensor_type][layer_idx] = output.cpu()
        return hook

    try:
        layers = clean_model.model.layers
        for i, layer in enumerate(layers):
            k_handle = layer.self_attn.k_proj.register_forward_hook(create_collector_hook(i, 'k'))
            handles.append(k_handle)
            v_handle = layer.self_attn.v_proj.register_forward_hook(create_collector_hook(i, 'v'))
            handles.append(v_handle)
        
        print(f"INFO: [Collector] Registered {len(handles)} collector hooks.")

        inputs = tokenizer(instruction_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            clean_model(**inputs)
        
        print(f"INFO: [Collector] Collected K/V tensors from {len(layers)} layers.")

    finally:
        for handle in handles:
            handle.remove()
        print(f"INFO: [Collector] Removed {len(handles)} collector hooks.")

    del clean_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return clean_kv_cache


def create_kv_patching_hook(clean_tensor, patch_slice):
    """
    Factory function that creates and returns a hook for patching a K or V tensor.
    """
    def hook(module, args, output):
        
        if output.dim() != 3:
            return output

        
        if output.shape[1] < patch_slice.stop:
            return output
        
        patched_tensor = clean_tensor.to(output.device)
        output[:, patch_slice, :] = patched_tensor[:, patch_slice, :]

        return output
    return hook


class KVPatcher:
    """
    Context manager for hooks registration on k_proj and v_proj for patching.
    """
    def __init__(self, model, clean_kv_cache, patch_slice):
        self.model = model
        self.clean_kv_cache = clean_kv_cache
        self.patch_slice = patch_slice
        self.handles = []

    def __enter__(self):
        print(f"INFO: [Patcher] Registering K/V hooks for slice {self.patch_slice}.")
        
        layers = self.model.base_model.get_base_model().model.layers
        
        for i, layer in enumerate(layers):
            if i not in self.clean_kv_cache['k'] or i not in self.clean_kv_cache['v']:
                continue

            clean_k = self.clean_kv_cache['k'][i]
            clean_v = self.clean_kv_cache['v'][i]

            #print(f'layer {i} \n') 
            #print('original k: \n')
            #print(clean_k[:, self.patch_slice, 0:5])
            
            k_hook = create_kv_patching_hook(clean_k, self.patch_slice)
            k_handle = layer.self_attn.k_proj.register_forward_hook(k_hook)
            self.handles.append(k_handle)

            v_hook = create_kv_patching_hook(clean_v, self.patch_slice)
            v_handle = layer.self_attn.v_proj.register_forward_hook(v_hook)
            self.handles.append(v_handle)
            
        print(f"INFO: [Patcher] Registered {len(self.handles)} K/V hooks.")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        for handle in self.handles:
            handle.remove()
        print(f"INFO: [Patcher] Removed {len(self.handles)} hooks.") 