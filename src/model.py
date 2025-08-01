import torch
import torch.nn as nn
from typing import Dict, Optional
import os
import copy
from peft import PeftModel  
from transformers import PreTrainedModel
import math


class Translator(nn.Module):
    
    weight_path = "custom_trained_weights.pt"    
    
    def __init__(self, num_segments, hidden_size, mlp_size, model_dtype):
        super().__init__()
        
        self.num_segments = num_segments
        self.mlp_size = mlp_size
        
        self.W1_matrices = nn.Parameter(torch.randn(num_segments, mlp_size, hidden_size, dtype=model_dtype))
        self.W2_matrices = nn.Parameter(torch.randn(num_segments, hidden_size, mlp_size, dtype=model_dtype))
        self.b1_bias = nn.Parameter(torch.zeros(num_segments, mlp_size, dtype=model_dtype))
        self.b2_bias = nn.Parameter(torch.zeros(num_segments, hidden_size, dtype=model_dtype))
        
        
        
    def init_weights(self):
        
        nn.init.xavier_uniform_(self.W1_matrices)
        nn.init.xavier_uniform_(self.W2_matrices)
    
    
    def forward(self, math_hidden_states, segment_ids):
    
        # MLP implementation: W2 * ReLU(W1 * v + b1) + b2
        gathered_W1 = self.W1_matrices[segment_ids]  # Shape: [all_math, intermediate, H]
        gathered_W2 = self.W2_matrices[segment_ids]  # Shape: [all_math, H, intermediate]
        gathered_b1 = self.b1_bias[segment_ids]      # Shape: [all_math, intermediate]
        gathered_b2 = self.b2_bias[segment_ids]      # Shape: [all_math, H]
         
        # 1. Compute first linear layer: W1 * v + b1
        intermediate_states = torch.bmm(
            gathered_W1,                     # [all_math, H, intermediate]
            math_hidden_states.unsqueeze(-1) # [all_math, H, 1]
        ).squeeze(-1) # -> [all_math, intermediate]
        intermediate_states = intermediate_states + gathered_b1  # Add bias
        
        # 2. Apply ReLU activation
        intermediate_states = torch.relu(intermediate_states)
        
        # 3. Compute second linear layer: W2 * ReLU(W1 * v + b1) + b2
        transformed_states = torch.bmm(
            gathered_W2,                     # [all_math, intermediate, H]
            intermediate_states.unsqueeze(-1) # [all_math, intermediate, 1]
        ).squeeze(-1) # -> [all_math, H]
        
        return transformed_states + gathered_b2
    

    def save_pretrained(self, save_directory: str):
        custom_state_dict = self.state_dict()
        torch.save(custom_state_dict, os.path.join(save_directory, self.weight_path))

    def load_pretrained(self, checkpoint_path: str):
        custom_weights_path = os.path.join(checkpoint_path, self.weight_path)
        custom_state_dict = torch.load(custom_weights_path, map_location='cpu')
        self.load_state_dict(custom_state_dict)
    
    
class ModelWithAuxiliaryHead(nn.Module):
        
    def __init__(
        self,
        base_model: PreTrainedModel | PeftModel,
        mlp_size: int,      
        num_segments,         
        lm_head: nn.Module,
        N_max          
    ):
       
        super().__init__()
        self.base_model = base_model
        hidden_size = self.base_model.config.hidden_size
        model_dtype = self.base_model.dtype
        
        self.num_segments = num_segments
        self.N_max = N_max
        self.distribute_matrices()
        
        self.translator = Translator(num_segments, hidden_size, mlp_size, model_dtype)
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
        math_logits = None
    
        if "math_labels" in kwargs:
        
            math_hidden_states, math_indices = get_hiddens_and_indices(last_hidden_state, kwargs["starts"], kwargs["math_lengths"], kwargs["math_labels"], kwargs["math_attention_mask"])
            segment_ids = self.segment_indices[math_indices] 
            math_hiddens = self.translator(math_hidden_states, segment_ids)
            math_logits = self.lm_head(math_hiddens)
        
        
        return {
                "logits": logits,
                "math_logits": math_logits,
                "last_hidden_state":last_hidden_state
            }
        
    @torch.no_grad()
    def generate(self, tokenizer, prompt: str, math_flag: bool, **generation_kwargs) -> str:
        
        if math_flag:
            return self.generate_with_various_lengths_of_simple_talk(tokenizer, prompt, **generation_kwargs)
        else:
            self.eval()
            general_answer = self._generate_text(tokenizer, prompt, **generation_kwargs)
        
            return {
                "simple_talk": general_answer,
                "math_text" : None
            }

    @torch.no_grad()
    def _generate_text(self, tokenizer, prompt, **generation_kwargs):
        
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.base_model.generate(**inputs, **generation_kwargs)
        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_len:]
        
        return tokenizer.decode(generated_ids, skip_special_tokens=True)

    @torch.no_grad()
    def generate_with_various_lengths_of_simple_talk(self, tokenizer, prompt: str, **generation_kwargs):
    
        self.eval()

        # 1. Initial generation and 'last_hidden_state' extraction for the case of simple talk with delimeters
        simple_talk = self._generate_text(tokenizer, prompt, **generation_kwargs)
        prompt_with_simple_talk = prompt + '<simple_talk>' + simple_talk + '</simple_talk>'
        inputs = tokenizer(prompt_with_simple_talk, return_tensors="pt").to(self.device)
        forward_outputs = self.forward(input_ids=inputs["input_ids"])
        last_hidden_state = forward_outputs['last_hidden_state'].squeeze(0) # Shape: [seq_len, hidden_size]
                
        # 2. Preprocess simple_talk
        last_period_pos = simple_talk.rfind('.')
        if last_period_pos != -1:
            simple_talk = simple_talk[:last_period_pos + 1]
        sentences = [s.strip() + '.' for s in simple_talk.split('.') if s.strip()]
        
        # 3. Iterative generation and logging
        answers = []
        hiddens = []
        generation_kwargs["max_new_tokens"] = 20
        
        for i in range(1, len(sentences) + 1):
            simple_talk_cut = "<simple_talk>" + " ".join(sentences[:i]) + "</simple_talk>" 
            prompt_with_simple_talk = prompt + simple_talk_cut
            math_answer = self._generate_text(tokenizer, prompt_with_simple_talk, **generation_kwargs)
            math_thoughts = self._extract_hidden_thoughts(tokenizer, prompt_with_simple_talk, last_hidden_state)
            answers.append(simple_talk_cut + math_answer)
            hiddens.append(math_thoughts)

        with open("answers-mlp.log", "a") as f:
            for answer, hidden in zip(answers, hiddens):
                f.write(f"Answer:\n{answer}\n")
                f.write(f"Hidden:\n{hidden}\n")
                f.write("-"*50 + "\n")
                f.write("="*50 + "\n")

        return {
            "simple_talk": simple_talk,
            "math_text": hiddens[-1] 
        }
    

    def _extract_hidden_thoughts(self, tokenizer, prompt_with_simple_talk, last_hidden_state):
        
        inputs = tokenizer(prompt_with_simple_talk, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"].squeeze(0)  # Shape: [seq_len]
        begin_simple_talk_id = tokenizer.convert_tokens_to_ids('<simple_talk>')
        end_simple_talk_id = tokenizer.convert_tokens_to_ids('</simple_talk>')

        indices = torch.arange(len(input_ids), device=self.device)
        
        start_indices = indices[input_ids == begin_simple_talk_id]
        end_indices = indices[input_ids == end_simple_talk_id]
        
        math_text = None
        
        if len(start_indices) > 0 and len(end_indices) > 0:
            
            start_idx = start_indices[0].item()
            end_idx = end_indices[0].item()

            if end_idx - start_idx > 1:
                
                math_hidden_states = last_hidden_state[start_idx: end_idx + 1]
                
                if math_hidden_states.dtype == torch.float32:
                    math_hidden_states = math_hidden_states.to(torch.bfloat16)

     
                max_len_of_math_text = min(math_hidden_states.shape[0], self.N_max)
                
                math_indices = torch.arange(max_len_of_math_text, device=self.device)
                segment_ids = self.segment_indices[math_indices] 
                
                math_hiddens = self.translator(math_hidden_states, segment_ids)
                math_logits = self.lm_head(math_hiddens)
                math_ids = torch.argmax(math_logits, dim=-1)
 
                math_text = tokenizer.decode(math_ids)
            
        return math_text
    
    
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
        
def get_hiddens_and_indices(last_hidden_state, starts, math_lengths, math_labels_batch, math_attention_mask):
    
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
    _, math_seq_len = math_labels_batch.shape
    indices = torch.arange(math_seq_len, device=device).expand(batch_size, -1)
    math_indices = indices[math_attention_mask == 1]
    
    return hiddens, math_indices
