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

class ModelWithAuxiliaryHead(nn.Module):
    
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
        
        

        self.A_matrices = nn.Parameter(torch.randn(num_segments, hidden_size, self.rank, dtype=model_dtype))
        self.B_matrices = nn.Parameter(torch.randn(num_segments, self.rank, hidden_size, dtype=model_dtype))
        self.bias = nn.Parameter(torch.zeros(num_segments, hidden_size, dtype=model_dtype))

        self.distribute_matrices()
        # What kind of initialization is the best for them? 
        nn.init.xavier_uniform_(self.A_matrices)
        nn.init.xavier_uniform_(self.B_matrices)

        self.lm_head = lm_head
        
        # hook registration
        Llama_with_LoRA = self.base_model.get_base_model()

        def hook_fn(module, inp, out):
            if not getattr(module, "_hooked_done", False):
                module._captured_hidden = out[0]
                module._hooked_done = True
                
        self._hook_handle = Llama_with_LoRA.model.layers[self.k].register_forward_hook(hook_fn)
      
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            starts: Optional[torch.LongTensor] = None,
            ends: Optional[torch.LongTensor] = None,
            math_lengths: Optional[torch.LongTensor] = None,
            math_labels: Optional[torch.LongTensor] = None,
            math_attention_mask: Optional[torch.Tensor] = None,
            **kwargs
        ):
        
        layer = self.base_model.get_base_model().model.layers[self.k]
        if hasattr(layer, "_hooked_done"):
            layer._hooked_done = False
        if hasattr(layer, "_captured_hidden"):
            del layer._captured_hidden
        
        # Get the original transformers model from the PeftModel wrapper. It contains the LoRA layers.
        modified_base_model = self.base_model.get_base_model() # debug_model_structure.py confirms that this contains lora

        # Call the "body" of the model (e.g., LlamaModel) to get the last hidden state
        # This call goes through the LoRA layers.
        base_model_output = modified_base_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        k_layer_hidden_state = layer._captured_hidden
        last_hidden_state = base_model_output.last_hidden_state
        
        logits = self.lm_head(last_hidden_state)

        simple_talk_loss, final_answer_loss = self.compute_simple_and_final_answer_loss(logits, input_ids, attention_mask, starts, ends)
        math_loss, A_losses = self.compute_math_loss(k_layer_hidden_state, math_labels, math_attention_mask, starts, math_lengths)

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

    def compute_math_loss(self, last_hidden_state, math_input_ids, math_attention_mask, starts, math_lengths):
    
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
        segment_ids = self.segment_indices[math_indices[math_attention_mask == 1]] 

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
        
        transformed_states = transformed_states + self.bias[segment_ids]

        # --- math loss calculation ---
        
        math_logits = self.lm_head(transformed_states)
        target_math_tokens = math_input_ids[math_attention_mask == 1] # [T] 
        math_loss = F.cross_entropy(math_logits, target_math_tokens) 

        # with torch.no_grad() means that we are not going to calculate gradients for A_losses here
        
        with torch.no_grad():
            per_token_loss = F.cross_entropy(math_logits, target_math_tokens, reduction='none')
            A_losses = []
            for idx in range(self.num_segments):
                mask = (segment_ids == idx)
                if mask.any():
                    A_losses.append(per_token_loss[mask].mean()) 
                else:
                    A_losses.append(torch.tensor(0.0, device=device))

        return math_loss, A_losses


    def save_pretrained(self, save_directory: str):
        
        # Save the base model (handles both PEFT and base model states)
        self.base_model.save_pretrained(save_directory)
        
        # Save custom weights
        custom_state_dict = {
            "A_matrices": self.A_matrices.cpu().clone(),
            "B_matrices": self.B_matrices.cpu().clone(),
            "bias": self.bias.cpu().clone(),
        }
        
        custom_weights_filename = "custom_trained_weights.pt"
        save_path = os.path.join(save_directory, custom_weights_filename)
        try:
            torch.save(custom_state_dict, save_path)
            print(f"Custom weights saved to {save_path}")
        except Exception as e:
            print(f"Failed to save custom weights: {e}")
            
    def load_custom_weights(self, checkpoint_path: str):
        """
        Load custom weights (A_matrices, B_matrices, bias) from a checkpoint directory.
        
        Args:
            checkpoint_path: Path to the checkpoint directory containing custom_trained_weights.pt
        """
        custom_weights_path = os.path.join(checkpoint_path, "custom_trained_weights.pt")
        
        if not os.path.exists(custom_weights_path):
            print(f"Warning: Custom weights file not found at {custom_weights_path}")
            return
        
        try:
            custom_state_dict = torch.load(custom_weights_path, map_location=self.A_matrices.device)
            
            # Load A_matrices
            if "A_matrices" in custom_state_dict:
                self.A_matrices.data = custom_state_dict["A_matrices"].to(self.A_matrices.device)
                print(f"Loaded A_matrices from {custom_weights_path}")
            
            # Load B_matrices
            if "B_matrices" in custom_state_dict:
                self.B_matrices.data = custom_state_dict["B_matrices"].to(self.B_matrices.device)
                print(f"Loaded B_matrices from {custom_weights_path}")
            
            # Load bias
            if "bias" in custom_state_dict:
                self.bias.data = custom_state_dict["bias"].to(self.bias.device)
                print(f"Loaded bias from {custom_weights_path}")
                
        except Exception as e:
            print(f"Error loading custom weights: {e}")
            raise e

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def distribute_matrices(self):

        chunk_size = math.ceil(self.N_max / self.num_segments)
        indices = torch.arange(self.N_max, dtype=torch.long) // chunk_size
        self.register_buffer("segment_indices", indices)

    def set_input_embeddings(self, value):
        self.base_model.set_input_embeddings(value)

 