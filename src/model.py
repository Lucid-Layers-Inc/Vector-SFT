import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from torch.nn import functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Any
import math
import os
from omegaconf import ListConfig
from typing import Dict
from peft import PeftModel
from peft import get_peft_model_state_dict
from safetensors.torch import save_file

from transformers import LlamaForCausalLM


class Translator(nn.Module):    
    
    def __init__(self, num_segments, hidden_size, rank, model_dtype, N_max):
        super().__init__()
        
        self.N_max = N_max
        self.num_segments = num_segments
        
        self.A_matrices = nn.Parameter(torch.randn(num_segments, hidden_size, rank, dtype=model_dtype))
        self.B_matrices = nn.Parameter(torch.randn(num_segments, rank, hidden_size, dtype=model_dtype))
        self.bias = nn.Parameter(torch.zeros(num_segments, hidden_size, dtype=model_dtype))
        
        nn.init.xavier_uniform_(self.A_matrices)
        nn.init.xavier_uniform_(self.B_matrices)
        
        self.weight_path = "custom_trained_weights.pt"
    
    def forward(self, math_hidden_states, segment_ids):
        
        # Memory-efficient implementation
        gathered_A = self.A_matrices[segment_ids]  # Shape: [all_math, H, r]
        gathered_B = self.B_matrices[segment_ids]  # Shape: [all_math, r, H]

        # 1. Compute intermediate_states = B @ h
        intermediate_states = torch.bmm(
            gathered_B,                      # [all_math, r, H]
            math_hidden_states.unsqueeze(-1) # [all_math, H, 1]
        ) # -> [all_math, r, 1]

        # 2. Compute transformed_states = A @ intermediate_states
        transformed_states = torch.bmm(
            gathered_A,                      # [all_math, H, r]
            intermediate_states              # [all_math, r, 1]
        ).squeeze(-1) # -> [all_math, H]
        
        return transformed_states + self.bias[segment_ids]
    
    def save_pretrained(self, save_directory: str):
        # TODO: make it configurable
        custom_state_dict = {
            "A_matrices": self.A_matrices.cpu().clone(),
            "B_matrices": self.B_matrices.cpu().clone(),
            "bias": self.bias.cpu().clone(),
        }
        
        torch.save(custom_state_dict, os.path.join(save_directory, self.weight_path))
    
    def load_pretrained(self, checkpoint_path: str):
        custom_weights_path = os.path.join(checkpoint_path, self.weight_path)
        custom_state_dict = torch.load(custom_weights_path, map_location=self.A_matrices.device)
        self.A_matrices.data = custom_state_dict["A_matrices"].to(self.A_matrices.device)
        self.B_matrices.data = custom_state_dict["B_matrices"].to(self.B_matrices.device)
        self.bias.data = custom_state_dict["bias"].to(self.bias.device)
    
    
class ModelWithAuxiliaryHead(nn.Module):
    
    segment_indices: torch.Tensor
    
    def __init__(
        self,
        base_model: PreTrainedModel,
        N_max: int,          
        num_segments: int,         # Number of linear-probing matrices A_i and segments
        lm_head: nn.Module,
        beta_1: float = 0.5,       # for loss on math reasoning
        beta_2: float = 0.5,       # for loss on simple talk
        beta_3: float = 0.4,       # for loss on final answer
        r: int = 256,              # segments rank
    ):
       
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.beta_3 = beta_3
        self.rank = r
        self.N_max = N_max
        
        hidden_size = self.base_model.config.hidden_size
        model_dtype = self.base_model.dtype
        
        self.num_segments = num_segments
        self.distribute_matrices()
        
        self.translator = Translator(num_segments, hidden_size, self.rank, model_dtype, N_max)
        
        self.lm_head = lm_head
        self.device = self.base_model.device
        
       
        
    def forward(
            self,
            input_ids: torch.LongTensor = None,
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

        return {
                "logits": logits,
                "last_hidden_state": last_hidden_state 
            }
        
     
    def save_pretrained(self, save_directory: str):
        
        # Save the base model (handles both PEFT and base model states)
        self.base_model.save_pretrained(save_directory)
        self.translator.save_pretrained(save_directory)
        

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def distribute_matrices(self):

        chunk_size = math.ceil(self.N_max / self.num_segments)
        indices = torch.arange(self.N_max, dtype=torch.long) // chunk_size
        self.register_buffer("segment_indices", indices)

    def set_input_embeddings(self, value):
        self.base_model.set_input_embeddings(value)

 