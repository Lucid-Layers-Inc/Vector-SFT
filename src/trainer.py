import os
import json

import torch
from trl import SFTTrainer
from typing import Optional
from transformers.trainer_utils import EvalLoopOutput
from typing import List
from src.common.losses import Betas, calculate_all_main_losses, plain_cross_entropy_loss
from torch.utils.data import DataLoader
from transformers.trainer import TrainerState
from torch.utils.data import Dataset


class VectorSFTTrainer(SFTTrainer):
    eval_dataset: list[Dataset]
    train_dataset: Dataset
    
    def __init__(self, *args, betas: Betas, dataset_processor=None, **kwargs):
        
            super().__init__(*args, **kwargs)
            if dataset_processor is None:
                raise ValueError("VectorSFTTrainer requires a `dataset_processor`.")
            self.dataset_processor = dataset_processor
            self.betas = betas
        
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        
        """
        Custom compute_loss method to handle multiple losses.
        Accumulates all losses into `self._metrics` for later logging.
        """
        
        outputs = model(**inputs)
        
        source = inputs['source_label'][0].item()
        if source == 0:
            losses_dict = calculate_all_main_losses(outputs, inputs, self.betas)
            current_loss = losses_dict["total_loss"]
        else: 
            losses_dict = {"calibration_loss": plain_cross_entropy_loss(outputs["logits"], inputs["input_ids"])}
            current_loss = losses_dict["calibration_loss"]
                
        for key, value in losses_dict.items():
            self._metrics[key].append(value.item())

        return (current_loss, outputs) if return_outputs else current_loss

    def get_train_dataloader(self) -> Dataset:
        """
        Overrides the standard method.
        """
        return self.train_dataset

    def log(self, logs: dict, start_time: Optional[float] = None) -> None:
        """
        Overrides the parent log method to average and log custom metrics
        that have been accumulated in `self._metrics`.
        """
        # Average the accumulated metrics if any are present
        if self._metrics:
            
            metrics_to_log = {
                key: sum(values) / len(values)
                for key, values in self._metrics.items()
                if len(values) != 0
            }
            
            logs.update(metrics_to_log)
            self._metrics.clear()

        # Let the parent class handle the actual logging to callbacks
        super().log(logs, start_time)

    def evaluation_loop(
        self,
        dataloader: torch.utils.data.DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        model.eval()
        
        eval_dataset_main = self.eval_dataset[0] 
        eval_loader_main = DataLoader(
            eval_dataset_main,
            batch_size = self.args.per_device_eval_batch_size,
            shuffle = False,
            collate_fn = self.dataset_processor.data_collate,   
        )

        # --- Stage 1. Evaluation on the first dataset ----
        
        all_metrics_main = []
        for inputs in eval_loader_main:
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                outputs = model(**inputs)
                losses_dict_main = calculate_all_main_losses(outputs, inputs, self.betas)
                
                gathered_losses = {}
                for key, value in losses_dict_main.items():
                    gathered_losses[key] = self.gather_function(value).detach()
                all_metrics_main.append(gathered_losses)
                
            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)
                

        # --- Stage 2. Evaluation on the calibration dataset ----
        
        eval_dataset_calib = self.eval_dataset[1]
        eval_loader_calib = DataLoader(
            eval_dataset_calib,
            batch_size = self.args.per_device_eval_batch_size,
            shuffle = False,
            collate_fn = self.dataset_processor.data_calibration_collate,   
        )

        
        all_metrics_calib = []
        for inputs in eval_loader_calib:
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                outputs = model(**inputs)
                losses_dict_calib = {"calibration_loss": plain_cross_entropy_loss(outputs["logits"], inputs["input_ids"])}
                
                gathered_losses = {}
                for key, value in losses_dict_calib.items():
                    gathered_losses[key] = self.gather_function(value).detach()
                all_metrics_calib.append(gathered_losses)
                
            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)

        # --- Stage 3. Aggregate and compute mean ---
        
        final_metrics = {}
        
        for key in all_metrics_main[0].keys():
            mean_val = torch.tensor([d[key].item() for d in all_metrics_main]).mean().item()
            final_metrics[f"{metric_key_prefix}_{key}"] = mean_val
            
        for key in all_metrics_calib[0].keys():
            mean_val = torch.tensor([d[key].item() for d in all_metrics_calib]).mean().item()
            final_metrics[f"{metric_key_prefix}_{key}"] = mean_val
            
        num_samples = len(eval_dataset_main) + len(eval_dataset_calib)
        
        return EvalLoopOutput(
                predictions=None, 
                label_ids=None, 
                metrics=final_metrics, 
                num_samples=num_samples
            )
        
    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        
        self.model.save_pretrained(output_dir)
        
    def resume_trainer_only(self, checkpoint_path):
        optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
        if os.path.exists(optimizer_path):
            optimizer_state = torch.load(optimizer_path, map_location="cpu")
            self.optimizer.load_state_dict(optimizer_state)
            print(f"Loaded optimizer state from {optimizer_path}")
        else:
            print(f"Warning: optimizer.pt not found in {checkpoint_path}")
        
        scheduler_path = os.path.join(checkpoint_path, "scheduler.pt")
        if os.path.exists(scheduler_path):
            scheduler_state = torch.load(scheduler_path, map_location="cpu")
            self.lr_scheduler.load_state_dict(scheduler_state)
            print(f"Loaded scheduler state from {scheduler_path}")
        else:
            print(f"Warning: scheduler.pt not found in {checkpoint_path}")
        
        trainer_state_path = os.path.join(checkpoint_path, "trainer_state.json")
        if os.path.exists(trainer_state_path):
            with open(trainer_state_path, "r") as f:
                state_dict = json.load(f)
            self.state = TrainerState(**state_dict)
            print(f"Loaded trainer state from {trainer_state_path}")
            print(f"Resuming from step {self.state.global_step}, epoch {self.state.epoch}")
        else:
            print(f"Warning: trainer_state.json not found in {checkpoint_path}")
        
        self._load_rng_state(checkpoint_path)
    
    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        self.create_optimizer_and_scheduler(num_training_steps=self.args.max_steps)
        self.resume_trainer_only(resume_from_checkpoint)

        
