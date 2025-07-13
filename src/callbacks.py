import os
import pandas as pd
import torch
from transformers import TrainerCallback, PreTrainedTokenizer
from transformers import TrainingArguments, TrainerState, TrainerControl
from transformers.utils import logging
from huggingface_hub import upload_file
from typing import List

logger = logging.get_logger(__name__)

class ClearMLCallback(TrainerCallback):
    def __init__(self, task):
        self.task = task
        self.logger = task.get_logger()

    def on_log(self, args, state, control, logs={}, **kwargs):
        for key, value in logs.items():
            self.logger.report_scalar(title="Training", series=key, value=value, iteration=state.global_step)


class SaveCustomWeightsOnHubCallback(TrainerCallback):
    """
    Callback to save specific non-PEFT trainable weights alongside PEFT adapters.
    Assumes the Trainer handles saving PEFT adapters automatically.
    Also explicitly uploads the custom weights file to the Hub if push_to_hub is enabled.
    """
    
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):

        if args.push_to_hub and state.is_world_process_zero and args.hub_model_id:
            checkpoint_folder = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            model = kwargs["model"]
            upload_file(
                path_or_fileobj=os.path.join(checkpoint_folder, model.translator.weight_path), 
                path_in_repo=f"checkpoint-{state.global_step}/{model.translator.weight_path}",
                repo_id=args.hub_model_id,
                repo_type="model",
                commit_message=f"Upload custom weights for step {state.global_step}",
            )

class GenerationCallback(TrainerCallback):
    """
    A callback to generate text from a list of prompts at the end of each epoch.
    """
    def __init__(self, prompts: List[str], tokenizer: PreTrainedTokenizer):
        
        self.prompts = prompts
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        
        print(f"\n--- Generating samples at the end of epoch {int(state.epoch)} ---")
        
        model = kwargs["model"]
        model.eval() # Set model to evaluation mode for generation
        
        for i, prompt in enumerate(self.prompts):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                
                outputs = model.base_model.generate(
                    **inputs,
                    max_new_tokens=300,
                    #temperature=0.7,
                    #top_p=0.9,
                    do_sample=False,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            input_len = inputs["input_ids"].shape[1]
            generated_ids = outputs[0][input_len:]
            generated_text = self.tokenizer.decode(generated_ids)
            
            #print(f"Question: {prompt}")
            print(f"Generated answer for question {i}: {generated_text}")
            print("-------------------------------------------")
        
        model.train() 