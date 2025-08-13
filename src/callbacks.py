import os

from transformers import TrainingArguments, TrainerState, TrainerControl, TrainerCallback, PreTrainedTokenizer
from transformers.utils import logging
from huggingface_hub import upload_file, upload_folder
from omegaconf import DictConfig

from src.model import ModelWithAuxiliaryHead

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


class SaveFolderOnHubCallback(TrainerCallback):
    """
    Callback to save specific folder to the Hub.
    """
    
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):

        if args.push_to_hub and state.is_world_process_zero and args.hub_model_id:
            checkpoint_folder = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            print(f"Uploading folder {checkpoint_folder} to the Hub")
            upload_folder(
                repo_id=args.hub_model_id,
                folder_path=checkpoint_folder,
                path_in_repo=f"checkpoint-{state.global_step}",
                repo_type="model",
                commit_message=f"Upload checkpoint {state.global_step}"
                )
            print(f"Folder {checkpoint_folder} uploaded to the Hub")


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
        
        model: ModelWithAuxiliaryHead = kwargs["model"]
        
        all_prompts = list(self.instruction_prompts + self.general_prompts)
        
        tokenizer_inputs = self.tokenizer(all_prompts, return_tensors="pt", padding=True)["input_ids"]
        output = model.generate(tokenizer_inputs.to(model.device), **self.generation_params)
        
        general_thoughts = self.tokenizer.batch_decode(output["general_thoughts"], skip_special_tokens=True)
        math_thoughts = self.tokenizer.batch_decode(output["math_toughts"], skip_special_tokens=True)
        
        self.log_answers(general_thoughts, math_thoughts, header="instruction prompts")
        self.log_answers(general_thoughts, math_thoughts, header="general prompts")
        
        
    def log_answers(self, general_thoughts: list[str], math_thoughts: list[str], header: str = "instruction prompts"):
        print("="*50)
        print(f"Answers for {header}:")
        print("General thoughts:", general_thoughts, sep="\n")
        print("Math thoughts:", math_thoughts, sep="\n")
        print("-"*50)