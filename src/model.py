import torch
import torch.nn as nn
from typing import Dict, Optional
import os
import copy
from transformers import LlamaConfig, PreTrainedModel
from transformers.models.bert.modeling_bert import BertLayer
from transformers import BertConfig
from peft import PeftModel  


class Translator(nn.Module):
    
    weight_path = "custom_trained_weights.pt"    
    
    def __init__(self, hidden_size, bert_hidden_size, bert_mlp_size, model_dtype, num_attention_heads=12):
        super().__init__()
        
        self.pre_bert_linear = nn.Linear(hidden_size, bert_hidden_size, dtype=model_dtype)
        
        bert_config = BertConfig(
            hidden_size=bert_hidden_size,
            num_attention_heads=num_attention_heads,
            intermediate_size=bert_mlp_size,
            hidden_act="relu",
        )
        self.bert_layer = BertLayer(bert_config)
        self.bert_layer.to(model_dtype)
        
        self.post_bert_linear = nn.Linear(bert_hidden_size, hidden_size, dtype=model_dtype)
        
    
    def init_weights(self):
        # Xavier initialization for linear layers (remain initialisation is automatic)
        nn.init.xavier_uniform_(self.pre_bert_linear.weight)
        nn.init.xavier_uniform_(self.post_bert_linear.weight)
    
    def forward(self, hidden_states, attention_mask=None):
        
        math_hidden_states = self.pre_bert_linear(hidden_states)
        
        bert_output = self.bert_layer(math_hidden_states, attention_mask=attention_mask)
        bert_hidden_states = bert_output[0]
        
        transformed_states = self.post_bert_linear(bert_hidden_states)
        
        return transformed_states
    
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
        bert_mlp_size: int,          
        num_attention_heads: int,         
        lm_head: nn.Module,
        bert_hidden_size: int = 768,              
    ):
       
        super().__init__()
        self.base_model = base_model
        hidden_size = self.base_model.config.hidden_size
        model_dtype = self.base_model.dtype
        
        self.translator = Translator(hidden_size, bert_hidden_size, bert_mlp_size, model_dtype, num_attention_heads)
        
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
            
            batch_of_hiddens, extended_attention_mask, bool_attention_mask = self._prepare_math_batch(last_hidden_state, kwargs["starts"], kwargs["ends"], kwargs["math_lengths"] )
            transformed_states = self.translator(batch_of_hiddens, attention_mask=extended_attention_mask)
            math_hiddens = transformed_states[bool_attention_mask]
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
        #simple_talk = """  The concept of information is a fundamental aspect of information theory, which was first introduced by Claude Shannon in 1948. It is based on the idea that information can be represented as a discrete variable, which can take on a finite number of values. The concept of information is often associated with entropy, which measures the amount of uncertainty or randomness in a system. In information theory, information is quantified using the concept of entropy, which is typically denoted by the symbol H. The entropy of a random variable is a measure of the amount of uncertainty or randomness in the variable. The concept of information theory has been widely used in many fields, including computer science, engineering, and economics. It has also been used in many areas of science, such as physics, biology, and sociology. The concept of information theory has been influential in shaping our understanding of the fundamental nature of information and its role in the universe."""
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

        with open("answers-bert.log", "a") as f:
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

     
                math_hiddens = self.translator(math_hidden_states.unsqueeze(0)).squeeze(0)
                math_logits = self.lm_head(math_hiddens)
                math_ids = torch.argmax(math_logits, dim=-1)

                math_text = tokenizer.decode(math_ids)
            
        return math_text
    
    def _prepare_math_batch(self, last_hidden_state, starts, ends, math_lengths):
        
        hidden_slices = [
            last_hidden_state[i, starts[i]:ends[i] + 1]
            for i in range(last_hidden_state.size(0))
        ]

        padded_hiddens = torch.nn.utils.rnn.pad_sequence(
            hidden_slices, batch_first=True, padding_value=0.0
        )

        max_len = padded_hiddens.size(1)
        indices = torch.arange(max_len, device=last_hidden_state.device).expand(len(hidden_slices), -1)
        
        # to generate math answer we allow the translator to use all simple talk tokens
        simple_talk_attention_mask = (indices < (ends - starts).unsqueeze(1)).long()
        # hovewer to calculate loss we use only math reasonongs lengths
        math_attention_mask = (indices < math_lengths.unsqueeze(1)).long()

        extended_attention_mask = simple_talk_attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=self.base_model.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(extended_attention_mask.dtype).min
        
        return padded_hiddens, extended_attention_mask, math_attention_mask.bool()
    
    def save_pretrained(self, save_directory: str):
        
        # Save the base model (handles both PEFT and base model states)
        self.base_model.save_pretrained(save_directory)
        self.translator.save_pretrained(save_directory)
            

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.base_model.set_input_embeddings(value)
