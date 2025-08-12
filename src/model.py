import torch
import torch.nn as nn
from typing import Dict, Optional
import copy
from transformers import LlamaConfig, PreTrainedModel
from peft import PeftModel

from transformers.models.llama.modeling_llama import LlamaMLP
from src.common.model import RecoverableModel


class Translator(RecoverableModel):
    weight_path = "custom_trained_weights.pt"    
    
    def __init__(self, config: LlamaConfig, rank: int | None = None):
        super().__init__()
        
        new_config = copy.deepcopy(config)
        new_config.intermediate_size = new_config.intermediate_size // 4 if rank is None else rank
        self.mlp = LlamaMLP(new_config)
    
    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
    
    def forward(self, math_hidden_states):
        return self.mlp(math_hidden_states)
    
    
    
class ModelWithAuxiliaryHead(nn.Module):
        
    def __init__(
        self,
        base_model: PreTrainedModel | PeftModel,
        lm_head: nn.Module,
    ):
       
        super().__init__()
        self.base_model = base_model
        self.translator = Translator(base_model.config).to(base_model.device, dtype=base_model.dtype)
        
        self.lm_head = lm_head
    
    @property
    def device(self):
        return self.base_model.device

        
    def forward(
            self,
            input_ids: Optional[torch.LongTensor],
            attention_mask: Optional[torch.Tensor] = None,
            **kwargs
        ):
        
        # Get the original transformers model from the PeftModel wrapper. It contains the LoRA layers.
        modified_base_model = self.base_model.get_base_model() # debug_model_structure.py confirms that this contains lora

        # This call goes through the LoRA layers.
        base_model_output = modified_base_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        last_hidden_state = base_model_output.last_hidden_state
        logits = self.lm_head(last_hidden_state)
        math_logits = self.lm_head(self.translator(last_hidden_state))
        
        
        return {
                "logits": logits,
                "math_logits": math_logits,
            }
        
    @torch.no_grad()
    def generate(self, inputs, **generation_kwargs) -> Dict[str, torch.Tensor]:
        self.eval()
        
        general_answer = self.base_model.generate(inputs, **generation_kwargs)
        out_logits = self.forward(input_ids=general_answer)
        
        return {
            "general_thoughts": general_answer,
            "math_toughts" : out_logits["math_logits"].argmax(dim=-1)
        }

    def save_pretrained(self, save_directory: str):
        
        # Save the base model (handles both PEFT and base model states)
        self.base_model.save_pretrained(save_directory)
        self.translator.save_pretrained(save_directory)
            

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.base_model.set_input_embeddings(value)
