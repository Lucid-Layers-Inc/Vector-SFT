
from typing import Dict, Optional

import torch
from datasets import Dataset
from transformers.trainer import EvalLoopOutput
from trl import SFTTrainer

from src.sae.model import SAE
from src.sae.preservers import save_saes_safetensors
from src.trainer.trainer import VectorSFTTrainer



def sae_loss_fn(x_hat, x_target, latent, l1_coeff=1e-3):
	return torch.nn.functional.mse_loss(x_hat, x_target) + l1_coeff * latent.abs().mean()


def make_sae_loss_hook_with_loss(sae, bucket, loss_fn, is_eval=False):
    def hook(model, inputs, outputs):
        x = outputs[0] if isinstance(outputs, tuple) else outputs
        x = x.detach()
        d_model = x.shape[-1]  # [batch, seq_len, d_model] -> [batch * seq_len, d_model]
        flat_output = x.reshape(-1, d_model)
        if is_eval:
            with torch.no_grad():
                sae.eval()
                x_hat, z = sae(flat_output, return_latent=True)
                loss = loss_fn(x_hat, flat_output, z)
        else:
            with torch.enable_grad():
                sae.train()
                x_hat, z = sae(flat_output, return_latent=True)
                loss = loss_fn(x_hat, flat_output, z)
        bucket.append(loss)
    return hook

def put_saes(model, saes, sae_losses, loss_fn, is_eval=False):
    layers = getattr(model, "model", model).layers
    for layer in saes:
        layers[layer].self_attn.register_forward_hook(
            make_sae_loss_hook_with_loss(saes[layer]["attn"], sae_losses, loss_fn, is_eval)
        )
        layers[layer].mlp.register_forward_hook(
            make_sae_loss_hook_with_loss(saes[layer]["mlp"], sae_losses, loss_fn, is_eval)
        )



class SAETrainer(VectorSFTTrainer):  # ваш класс с тем же именем
    def __init__(self, *args, saes: dict, sae_cfg: dict, **kwargs):
        super().__init__(*args, **kwargs)
        self.saes: Dict[str, Dict[str, SAE]] = saes  
        self.sae_cfg: dict = sae_cfg
        self.sae_losses: list[torch.Tensor] = []
        self.is_eval = False
        
        put_saes(self.model, self.saes, self.sae_losses, self.sae_loss_fn, self.is_eval)
        self.get_sae_params()

    def get_sae_params(self):
        sae_params = []
        for sae_layer in self.saes.values():
            for sae_module in sae_layer.values():
                sae_params.extend(sae_module.parameters())
        self.sae_device = sae_params[0].device
        self.sae_params = sae_params
 
    def create_optimizer(self):
        "Making optimizer only for SAE parameters"
        self.get_sae_params()
        self.optimizer = torch.optim.AdamW(
            self.sae_params, lr=self.args.learning_rate, weight_decay=self.args.weight_decay
        )
        return self.optimizer

    def sae_loss_fn(self, x_hat, x_target, latent):
        return sae_loss_fn(x_hat, x_target, latent, l1_coeff=self.sae_cfg["l1_coeff"])	

    def compute_loss(self, model, inputs: dict, num_items_in_batch=None, return_outputs=False):
        self.sae_losses.clear()
        with torch.no_grad():
            outputs = model(**inputs)
   
        if not self.sae_losses:
            raise RuntimeError("SAE loss is not collected. Check the hook registration/filtering/sample_tokens.")

        sae_loss = torch.stack(self.sae_losses).sum()  
        current_loss = self.sae_cfg["lambda_sae"] * sae_loss
        return (current_loss, outputs) if return_outputs else current_loss

    def log(self, logs: dict, start_time: Optional[float] = None) -> None:
        g2 = 0.0
        for p in self.sae_params:
            if p.grad is not None:
                g2 += float(p.grad.detach().float().norm(2)**2)
        logs["sae_grad_norm"] = g2 ** 0.5
        super(VectorSFTTrainer, self).log(logs, start_time)
    
    def evaluation_loop(self, *args, **kwargs) -> EvalLoopOutput:
        self.is_eval = True
        eval_output = SFTTrainer.evaluation_loop(self, *args, **kwargs)
        self.is_eval = False
        return eval_output
    
    def get_eval_dataloader(self, eval_dataset: Dataset) -> Dataset:
        return self.eval_dataset
    
    def _save(self, output_dir: str, state_dict=None):
        save_saes_safetensors(self.saes, output_dir)
    
    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        self.create_optimizer_and_scheduler(num_training_steps=self.args.max_steps)
        self.resume_trainer_only(resume_from_checkpoint)