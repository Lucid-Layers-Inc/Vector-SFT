import os
import json

import torch
import torch.nn as nn
from trl import SFTTrainer
from typing import Optional
from transformers.trainer_utils import EvalLoopOutput
from typing import List, Dict, Any
from src.common.losses import calculate_all_main_losses, calculate_calibration_loss
from torch.utils.data import DataLoader
from transformers.trainer import TrainerState


class VectorSFTTrainer(SFTTrainer):
    
    def __init__(self, *args, dataset_processor=None, **kwargs):
        
            super().__init__(*args, **kwargs)
            if dataset_processor is None:
                raise ValueError("VectorSFTTrainer requires a `dataset_processor`.")
            self.dataset_processor = dataset_processor
        
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        
        """
        Custom compute_loss method to handle multiple losses.
        Accumulates all losses into `self._metrics` for later logging.
        """
        
        model = model.module if hasattr(model, 'module') else model
        outputs = model(**inputs)
        
        if next(model.parameters()).dtype == torch.bfloat16:
            if outputs['last_hidden_state'].dtype != torch.bfloat16:
                outputs['last_hidden_state'] = outputs['last_hidden_state'].to(torch.bfloat16)
            if outputs['logits'].dtype != torch.bfloat16:
                outputs['logits'] = outputs['logits'].to(torch.bfloat16)
        
        source = inputs['source_label'][0].item()
        #print('source', source)
        if source == 0:
            losses_dict = calculate_all_main_losses(model, outputs, inputs)
            current_loss = losses_dict["total_loss"]
        else: 
            losses_dict = calculate_calibration_loss(outputs, inputs)
            current_loss = losses_dict["calibration_loss"]
        
            
        # self._metrics is a defaultdict(list) that will be averaged and logged by self.log()
        
        for key, value in losses_dict.items():
            self._metrics[key].append(value.item())

        return (current_loss, outputs) if return_outputs else current_loss

    def get_train_dataloader(self) -> DataLoader:
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
                losses_dict_main = calculate_all_main_losses(model, outputs, inputs)
                
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
                losses_dict_calib = calculate_calibration_loss(outputs, inputs)
                
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

        
