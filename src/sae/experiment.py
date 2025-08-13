import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel

from src.sae.model import SAE
from src.common.default import Experiment
from src.dataloader import DatasetProcessor
from src.sae.preservers import load_saes_safetensors


class SAEExperiment(Experiment):
    
    def __init__(self, config: str):
        
        super().__init__(config)
        self.resume_from_checkpoint = self.cfg.resume_from_checkpoint
        
        self.base_model, self.tokenizer = self.prepare_model_and_tokenizer()
        self.model = self.base_model
        self.dataset_processor = DatasetProcessor(self.tokenizer, self.cfg)

    def prepare_model_and_tokenizer(self) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
                
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.name)
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model.name,
            torch_dtype=getattr(torch, self.cfg.model.dtype),
            device_map=self.cfg.model.device_map,
            attn_implementation=self.cfg.model.attn_implementation,
        )

        return base_model, tokenizer
    
    def build_saes(self):
        d_model = self.base_model.config.hidden_size
        n_layers = self.base_model.config.num_hidden_layers
        latent_size = self.cfg.sae.latent_size

        device = self.base_model.device
        dtype = getattr(self.base_model, "dtype", torch.float16)
        
        layers_spec = getattr(self.cfg.sae, "layers", None)
        index_base = int(getattr(self.cfg.sae, "layers_index_base", 0))
        selected_layers = _parse_layers(layers_spec, n_layers, index_base)
        
        print(f"Selected layers for SAEs: {selected_layers}")

        saes = {}
        for layer in selected_layers:
            attn = SAE(d_model, latent_size).to(device=device, dtype=dtype)
            mlp = SAE(d_model, latent_size).to(device=device, dtype=dtype)
            saes[layer] = {"attn": attn, "mlp": mlp}
        self.saes = saes
        
        if self.resume_from_checkpoint is not None and os.path.exists(self.resume_from_checkpoint):
            self.load_saes(self.resume_from_checkpoint)
    
    def load_saes(self, checkpoint_path: str):
        load_saes_safetensors(self.saes, checkpoint_path)

    def prepare_datasets(self):
        """
        Loads and prepares the dataset. Returns the data collator as callable function.
        Returns: The data collator function.
        """
        self.mix_data_loader, self.eval_dataset, self.eval_calib_dataset = self.dataset_processor.load_and_prepare()


def _parse_layers(spec, n_layers: int, index_base: int = 0) -> list[int]:
    if spec is None:
        return list(range(n_layers))
    if isinstance(spec, (list, tuple)):
        idxs = {int(x) for x in spec}
    elif isinstance(spec, str):
        idxs = set()
        for part in spec.split(','):
            part = part.strip()
            if not part:
                continue
            if '-' in part:
                a, b = part.split('-', 1)
                a, b = int(a.strip()), int(b.strip())
                lo, hi = (a, b) if a <= b else (b, a)
                idxs.update(range(lo, hi + 1))
            else:
                idxs.add(int(part))
    else:
        idxs = set()
    adjusted = sorted(i - index_base for i in idxs)
    return [i for i in adjusted if 0 <= i < n_layers]