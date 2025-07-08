import os
import math
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import PreTrainedModel


class Translator(nn.Module):    
    def __init__(self, num_segments, hidden_size, rank, model_dtype, N_max):
        super().__init__()
        
        self.N_max = N_max
        self.num_segments = num_segments
        
        self.A_matrices = nn.Parameter(torch.randn(num_segments, hidden_size, rank, dtype=model_dtype))
        self.B_matrices = nn.Parameter(torch.randn(num_segments, rank, hidden_size, dtype=model_dtype))
        self.bias = nn.Parameter(torch.zeros(num_segments, hidden_size, dtype=model_dtype))

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
        #config: PretrainedConfig,
        base_model: PreTrainedModel,
        N_max: int,          
        num_segments: int,         # Number of linear-probing matrices A_i and segments
        lm_head: nn.Module,
        beta_1: float = 0.5,       # for loss on math reasoning
        beta_2: float = 0.5,       # for loss on simple talk
        beta_3: float = 0.4,       # for loss on final answer
        r: int = 256,              # segments rank
        k: int = 28                # layer index for math reasoning
    ):
        #super().__init__(config)
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.beta_3 = beta_3
        self.rank = r
        self.N_max = N_max
        self.k = k
        
        hidden_size = self.base_model.config.hidden_size
        model_dtype = self.base_model.dtype

        self.num_segments = num_segments
        self.distribute_matrices()
        self.translator = Translator(num_segments, hidden_size, self.rank, model_dtype, N_max)
        self.lm_head = lm_head
        self.device = self.base_model.device
      
    def forward(
            self,
            input_ids: torch.LongTensor,
            math_labels: Optional[torch.LongTensor],
            attention_mask: Optional[torch.Tensor] = None,
            starts: Optional[torch.LongTensor] = None,
            ends: Optional[torch.LongTensor] = None,
            math_lengths: Optional[torch.LongTensor] = None,
            math_attention_mask: Optional[torch.Tensor] = None,
            **kwargs
        ):
        
        # TODO: remove dependency from LoRA shaped model
        modified_base_model = self.base_model.get_base_model() 
        base_model_output = modified_base_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        
        last_hidden_state = base_model_output.last_hidden_state
        logits = self.lm_head(last_hidden_state)
    
        if math_labels is not None:
            
            hiddens, indices = self.get_hiddens_and_indices(
                last_hidden_state, starts, math_lengths, math_labels
                )
            segment_ids = self.segment_indices[indices[math_attention_mask == 1]]
            math_hiddens = self.translator(hiddens, segment_ids)
            math_logits = self.lm_head(math_hiddens)
            
            target_math_tokens = math_labels[math_attention_mask == 1]
            math_loss = self.compute_math_loss(math_logits, target_math_tokens)
            A_losses = self.compute_A_losses(math_logits, target_math_tokens, segment_ids)

        simple_talk_loss, final_answer_loss = self.compute_simple_and_final_answer_loss(
            logits, input_ids, attention_mask, starts, ends
            )
        total_loss = self.beta_1 * math_loss + self.beta_2 * simple_talk_loss + self.beta_3 * final_answer_loss

        return {
                "loss": total_loss,
                "math_loss": math_loss,
                "simple_talk_loss": simple_talk_loss,
                "final_answer_loss": final_answer_loss,
                "logits": logits,
                "A_losses": A_losses
            }
    

    def compute_simple_and_final_answer_loss(self, logits, input_ids, attention_mask, starts, ends):
        
        device = logits.device
        batch_size, seq_len, vocab_size = logits.shape

        # Shift logits and labels for autoregressive training
        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_labels = input_ids[:, 1:].contiguous()

        # Flatten for cross-entropy
        logits_flat = shifted_logits.reshape(-1, vocab_size)
        labels_flat = shifted_labels.reshape(-1)

        # Calculate loss for each token, but do not reduce
        loss = F.cross_entropy(
            logits_flat,
            labels_flat,
            ignore_index=-100,
            reduction='none'
        )
        
        # Reshape loss back to [B, S-1] to apply masks
        loss_per_token = loss.view(batch_size, seq_len - 1)

        # Create indices grid for the SHIFTED sequence
        shifted_indices = torch.arange(seq_len - 1, device=device).expand(batch_size, -1)

        starts_exp = starts.unsqueeze(1)
        ends_exp = ends.unsqueeze(1)
        real_lengths = attention_mask.sum(dim=1).unsqueeze(1)

        # Create masks for the shifted tensors.
        # A loss at shifted_index `i` corresponds to a prediction for a token at original_index `i+1`.
        # So, we adjust the masks accordingly.
        mask_simple_talk = (shifted_indices >= starts_exp - 1) & (shifted_indices <= ends_exp - 1)
        mask_final_answer = (shifted_indices >= ends_exp) & (shifted_indices < real_lengths - 1)

        # Calculate mean loss for each part, handling empty masks to avoid NaN
        simple_talk_loss = loss_per_token[mask_simple_talk].mean() if mask_simple_talk.any() else torch.tensor(0.0, device=device)
        final_answer_loss = loss_per_token[mask_final_answer].mean() if mask_final_answer.any() else torch.tensor(0.0, device=device)
        
        return simple_talk_loss, final_answer_loss

    def get_hiddens_and_indices(self, last_hidden_state, starts, math_lengths, math_input_ids):
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
        math_hidden_states = last_hidden_state[mask_hidden] # [T, H]

        # choose matrices for each math hidden state according to the partition
        _, math_seq_len = math_input_ids.shape
        math_indices = torch.arange(math_seq_len, device=device).expand(batch_size, -1)
        
        return math_hidden_states, math_indices
    
    
    def compute_math_loss(self, math_logits, target_math_tokens):
        math_loss = F.cross_entropy(math_logits, target_math_tokens)
        
        return math_loss


    def compute_A_losses(self, math_logits, target_math_tokens, segment_ids):
        with torch.no_grad():
            per_token_loss = F.cross_entropy(math_logits, target_math_tokens, reduction='none')
            A_losses = []
            for idx in range(self.num_segments):
                mask = (segment_ids == idx)
                if mask.any():
                    A_losses.append(per_token_loss[mask].mean()) 
                else:
                    A_losses.append(torch.tensor(0.0, device=self.device))

        return A_losses

    def save_pretrained(self, save_directory: str):
        
        # Save the base model (handles both PEFT and base model states)
        self.base_model.save_pretrained(save_directory)
        self.translator.save_pretrained(save_directory)

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.base_model.set_input_embeddings(value)
    
    def distribute_matrices(self):
        chunk_size = math.ceil(self.N_max / self.num_segments)
        indices = torch.arange(self.N_max, dtype=torch.long) // chunk_size
        self.register_buffer("segment_indices", indices)
         