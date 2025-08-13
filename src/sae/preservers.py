from glob import glob
import os
from safetensors.torch import load_file, save_file
import torch


def save_saes_safetensors(saes: dict, output_dir: str, shard_max_gb: float = 5.0):
    os.makedirs(output_dir, exist_ok=True)
    flat: dict[str, torch.Tensor] = {}
    for layer_id, parts in saes.items():
        for kind, module in parts.items():  # 'mlp' / 'attn'
            for k, v in module.state_dict().items():
                flat[f"layer_{layer_id}.{kind}.{k}"] = v.detach().cpu()

    max_bytes = int(shard_max_gb * (1024**3))
    shards: list[dict[str, torch.Tensor]] = []
    cur: dict[str, torch.Tensor] = {}
    cur_bytes = 0
    for k, t in flat.items():
        sz = t.numel() * t.element_size()
        if cur and cur_bytes + sz > max_bytes:
            shards.append(cur); cur, cur_bytes = {}, 0
        cur[k] = t; cur_bytes += sz
    if cur: shards.append(cur)

    for idx, shard in enumerate(shards):
        fname = os.path.join(output_dir, f"saes-{idx:05d}.safetensors")
        save_file(shard, fname, metadata={"format": "sae_shard", "total_shards": str(len(shards))})

def load_saes_safetensors(saes: dict, output_dir: str, map_location: str = "cpu"):
    files = sorted(glob(os.path.join(output_dir, "saes-*.safetensors")))
    if not files: 
        return
    merged = {}
    for f in files:
        merged.update(load_file(f, device=map_location))

    for layer_id, parts in saes.items():
        for kind, module in parts.items():
            prefix = f"layer_{layer_id}.{kind}."
            state = {k[len(prefix):]: v for k, v in merged.items() if k.startswith(prefix)}
            module.load_state_dict(state, strict=False)
