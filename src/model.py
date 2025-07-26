import torch
import torch.nn as nn
from typing import Optional
import os
import copy
from transformers import LlamaConfig, PreTrainedModel
from peft import PeftModel  # type: ignore

from transformers.models.llama.modeling_llama import LlamaMLP


class Translator(nn.Module):
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
        lm_head: nn.Module,
    ):
       
        super().__init__()
        self.base_model = base_model
        self.translator = Translator(base_model.config)
        
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
        math_logits = self.lm_head(self.translator(last_hidden_state))
        
        
        return {
                "logits": logits,
                "math_logits": math_logits,
            }
        
    @torch.no_grad()
    def generate(self, tokenizer, prompt: str, math_flag: bool, **generation_kwargs) -> str:
        self.eval()
        general_answer = self._generate_text(tokenizer, prompt, **generation_kwargs)
        
        if math_flag:
            return self.generate_with_various_lengths_of_simple_talk(tokenizer, prompt, **generation_kwargs)
            
        
        print(10*'-')
        print('simple talk')
        print(general_answer)
        
        
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
    
        # 1. Initial generation
        simple_talk = self._generate_text(tokenizer, prompt, **generation_kwargs)
        
   
        # 2. Preprocess simple_talk
        last_period_pos = simple_talk.rfind('.')
        if last_period_pos != -1:
            simple_talk = simple_talk[:last_period_pos + 1]
        
        sentences = [s.strip() + '.' for s in simple_talk.split('.') if s.strip()]
        
        # 3. Iterative generation and logging
        simple_talks = []
        hiddens = []
        math_answers = []

        # for math answer predicting
        generation_kwargs["max_new_tokens"] = 20

        # 4. Get last hidden states
        full_text = prompt + "<simple_talk>" + simple_talk + "</simple_talk>" 
        inputs = tokenizer(full_text, return_tensors="pt").to(self.device)
        forward_outputs = self.forward(input_ids=inputs['input_ids'])
        last_hidden_state = forward_outputs['last_hidden_state'].squeeze(0) # Shape: [seq_len, hidden_size]
                
        
        for i in range(1, len(sentences) + 1):
            simple_talk_cut = "<simple_talk>" + " ".join(sentences[:i]) + "</simple_talk>" 
            prompt_with_simple_talk = prompt + simple_talk_cut
            math_answer = self._generate_text(tokenizer, prompt_with_simple_talk, **generation_kwargs)
            math_thoughts = self._extract_hidden_thoughts(tokenizer, prompt_with_simple_talk, last_hidden_state)
            simple_talks.append(simple_talk_cut)
            math_answers.append(math_answer)
            hiddens.append(math_thoughts)

        self._log_answers(simple_talks, hiddens, math_answers)
        
        return simple_talk, hiddens[-1] 

    def _log_answers(self, simple_talks, hiddens, math_answers):
        with open("answers.log", "a") as f:
            for i, (simple_talk, hidden, math_answer) in enumerate(zip(simple_talks, hiddens, math_answers)):
                f.write(f"\n{i}\n")
                f.write(f"Simple talk: {simple_talk}\n")
                f.write(f"Hidden thoughts: {hidden}\n")
                f.write(f"Math answer: {math_answer}\n")
                f.write("="*50 + "\n")

    def save_pretrained(self, save_directory: str):
        
        # Save the base model (handles both PEFT and base model states)
        self.base_model.save_pretrained(save_directory)
        self.translator.save_pretrained(save_directory)
        
    def _extract_hidden_thoughts(self, tokenizer, prompt_with_simple_talk, last_hidden_state):
        
        inputs = tokenizer(prompt_with_simple_talk, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"].squeeze(0)  

        begin_simple_talk_id = tokenizer.convert_tokens_to_ids('<simple_talk>')
        end_simple_talk_id = tokenizer.convert_tokens_to_ids('</simple_talk>')
        

        indices = torch.arange(len(input_ids), device=self.device)
        start_indices = indices[input_ids == begin_simple_talk_id]
        end_indices = indices[input_ids == end_simple_talk_id]
        
        math_text = None
        
        if len(start_indices) > 0 and len(end_indices) > 0:
            
            start_idx = start_indices[0].item()
            end_idx = end_indices[0].item()

            if (end_idx - start_idx > 1) and (end_idx - start_idx < 300):
                
                math_hidden_states = last_hidden_state[start_idx: end_idx + 1]
                
                if math_hidden_states.dtype == torch.float32:
                    print("WHAAAAT")
                    math_hidden_states = math_hidden_states.to(torch.bfloat16)

                max_len_of_math_text = min(math_hidden_states.shape[0], self.N_max)
                
                math_indices = torch.arange(max_len_of_math_text, device=self.device)
                segment_ids = self.segment_indices[math_indices] 
                
                math_hiddens = self.translator(math_hidden_states, segment_ids)
                math_logits = self.lm_head(math_hiddens)
                math_ids = torch.argmax(math_logits, dim=-1)
 
                math_text = tokenizer.decode(math_ids)
            
        return math_text
            

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

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
