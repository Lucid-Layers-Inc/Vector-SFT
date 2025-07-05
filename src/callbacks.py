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

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            for key, value in logs.items():
                self.logger.report_scalar(title="Training", series=key, value=value, iteration=state.global_step)


class SaveCustomWeightsCallback(TrainerCallback):
    
    """
    Callback to save specific non-PEFT trainable weights alongside PEFT adapters.
    Assumes the Trainer handles saving PEFT adapters automatically.
    Also explicitly uploads the custom weights file to the Hub if push_to_hub is enabled.
    """
    
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        
        checkpoint_folder = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        model = kwargs["model"]

        if not hasattr(model, "A_matrices") or not hasattr(model, "B_matrices") or not hasattr(model, "bias"):
            logger.warning("Model doesn't have expected custom weights (A_matrices, B_matrices, bias). Skipping.")
            return

        
        custom_state_dict = {
            "A_matrices": model.A_matrices.cpu().clone(),
            "B_matrices": model.B_matrices.cpu().clone(), 
            "bias": model.bias.cpu().clone(),
        }
        
        print(f"Saving A_matrices, B_matrices, bias")

        custom_weights_filename = "custom_trained_weights.pt"
        save_path = os.path.join(checkpoint_folder, custom_weights_filename)
        try:
            os.makedirs(checkpoint_folder, exist_ok=True)
            torch.save(custom_state_dict, save_path)
            logger.info(f"Custom trainable weights saved locally to {save_path}")

            if args.push_to_hub and state.is_world_process_zero:
                
                repo_id = args.hub_model_id
               
                target_path_in_repo = f"checkpoint-{state.global_step}/{custom_weights_filename}" # Upload to checkpoint folder
                commit_message = f"Upload custom weights for step {state.global_step}"
                try:
                    logger.info(f"Attempting to upload {save_path} to {repo_id}/{target_path_in_repo}")
                    upload_file(
                        path_or_fileobj=save_path,
                        path_in_repo=target_path_in_repo,
                        repo_id=repo_id,
                        repo_type="model",
                        commit_message=commit_message,
                    )
                    logger.info(f"Successfully uploaded custom weights to Hub repo {repo_id}")
                except Exception as e:
                    logger.error(f"Failed to upload custom weights to Hub: {e}")

        except Exception as e:
            logger.error(f"Failed to save custom trainable weights locally to {save_path}: {e}")