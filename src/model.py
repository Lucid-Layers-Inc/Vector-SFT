import torch
import torch.nn as nn
from typing import Optional
import math
import os
from transformers import PreTrainedModel
from peft import PeftModel  # type: ignore


class Translator(nn.Module):    
    
    def __init__(self, num_segments, hidden_size, rank, model_dtype, N_max):
        super().__init__()
        
        self.N_max = N_max
        self.num_segments = num_segments
        
        self.A_matrices = nn.Parameter(torch.randn(num_segments, hidden_size, rank, dtype=model_dtype))
        self.B_matrices = nn.Parameter(torch.randn(num_segments, rank, hidden_size, dtype=model_dtype))
        self.bias = nn.Parameter(torch.zeros(num_segments, hidden_size, dtype=model_dtype))
        
        self.weight_path = "custom_trained_weights.pt"
    
    def init_weights(self):
        nn.init.xavier_uniform_(self.A_matrices)
        nn.init.xavier_uniform_(self.B_matrices)
    
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
        base_model: PreTrainedModel | PeftModel,
        N_max: int,          
        num_segments: int,         # Number of linear-probing matrices A_i and segments
        lm_head: nn.Module,
        r: int = 256,              # segments rank
    ):
       
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
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
            input_ids: Optional[torch.LongTensor] = None,
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
        
        math_logits = None
        
        if "math_labels" in kwargs:
        
            math_hidden_states, math_indices = get_hiddens_and_indices(last_hidden_state, kwargs["starts"], kwargs["math_lengths"], kwargs["math_labels"])
            segment_ids = self.segment_indices[math_indices[kwargs["math_attention_mask"] == 1]] 
            math_hiddens = self.translator(math_hidden_states, segment_ids)
            math_logits = self.lm_head(math_hiddens)
        
        
        
        return {
                "logits": logits,
                "last_hidden_state": last_hidden_state,
                "math_logits": math_logits,
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

 
 
def get_hiddens_and_indices(last_hidden_state, starts, math_lengths, math_input_ids):
        
    device = last_hidden_state.device

    # --- transform hidden states corresponging to math thoughts ---
    
    # Choose hidden states corresponding to indices where math thoughts must be hidden 
    batch_size, seq_len, _ = last_hidden_state.shape
    indices = torch.arange(seq_len, device=device).expand(batch_size, -1) # -1 means to keep the last dimension the same
    # indices:
    # tensor([[0, 1, 2, 3, 4],
    #         [0, 1, 2, 3, 4],
    #         [0, 1, 2, 3, 4]])
    mask_hidden = (indices >= starts.unsqueeze(1)) & (indices < (starts + math_lengths).unsqueeze(1))
    hiddens = last_hidden_state[mask_hidden] # [T, H]
    

    # choose matrices for each math hidden state according to the partition
    _, math_seq_len = math_input_ids.shape
    indices = torch.arange(math_seq_len, device=device).expand(batch_size, -1)
    
    return hiddens, indices