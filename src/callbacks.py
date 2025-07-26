import os

from transformers import TrainingArguments, TrainerState, TrainerControl, TrainerCallback, PreTrainedTokenizer
from transformers.utils import logging
from huggingface_hub import upload_file
from omegaconf import DictConfig

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
    def __init__(self, prompts: DictConfig, tokenizer: PreTrainedTokenizer, generation_params: DictConfig):
        self.instruction_prompts = prompts.instruction_prompts
        self.general_prompts = prompts.general_prompts
        self.tokenizer = tokenizer
        self.generation_params = generation_params
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        
        print(f"\n--- Generating samples at the end of epoch {int(state.epoch)} ---")
        
        model = kwargs["model"]
        
        all_prompts = [
            (self.instruction_prompts, True),
            (self.general_prompts, False)
        ]
        
        for prompts, math_flag in all_prompts:
            for i, prompt in enumerate(prompts):
                
                outputs = model.generate(
                    tokenizer=self.tokenizer, 
                    prompt=prompt,
                    math_flag=math_flag, 
                    **self.generation_params
                )
                
                #print(f"Prompt: {prompt}")
                print(f"Generated answer for question {i}: {outputs['simple_talk']}")
                if outputs['math_text'] is not None:
                    print(f"Hidden thoughts: {outputs['math_text']}")
                    
                print("-------------------------------------------") 