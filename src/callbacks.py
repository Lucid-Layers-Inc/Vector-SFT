import os
import pandas as pd
import torch
from transformers import TrainerCallback
from transformers import TrainingArguments, TrainerState, TrainerControl
from transformers.utils import logging
from huggingface_hub import upload_file

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