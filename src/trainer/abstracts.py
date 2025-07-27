from abc import ABC, abstractmethod
import os
import json
from typing import Callable, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedModel, TrainingArguments
from transformers.trainer_utils import EvalLoopOutput
from transformers.trainer import TrainerState


class WeightLoaderMixin(ABC):
    optimizer: torch.optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler
    state: TrainerState
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
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
    
    @abstractmethod
    def _load_from_checkpoint(self, resume_from_checkpoint: str, model: PreTrainedModel=None):
        pass


class EvaluatorMixin(ABC):
    eval_dataset: list[Dataset]
    train_dataset: Dataset
    args: TrainingArguments
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def eval_pass(self, inputs, model: PreTrainedModel, loss_fn: Callable, **loss_kwargs) -> dict:
        with torch.no_grad():
            outputs = model(**inputs)
            losses_dict_main = loss_fn(outputs, inputs, **loss_kwargs)
            
            gathered_losses = {}
            for key, value in losses_dict_main.items():
                gathered_losses[key] = self.gather_function(value).detach()  # type: ignore
        
        self.callback_handler.on_prediction_step(self.args, self.state, self.control)
        return gathered_losses
    
    def eval_on_dataset(self, eval_dataset, collate_fn, model, loss_fn, **loss_kwargs):
        eval_loader = DataLoader(
            eval_dataset,
            batch_size = self.args.per_device_eval_batch_size,
            shuffle = False,
            collate_fn = collate_fn,   
        )

        all_metrics_main = []
        for inputs in eval_loader:
            inputs = self._prepare_inputs(inputs)
            metrics = self.eval_pass(inputs, model, loss_fn, **loss_kwargs)
            all_metrics_main.append(metrics)

        return all_metrics_main

    def gather_custom_metrics(self, metrics: list[dict], metric_key_prefix: str) -> dict:
        final_metrics = {}
        
        if len(metrics) == 0:
            return final_metrics
        
        for key in metrics[0].keys():
            values = []
            for d in metrics:
                val = d[key]
                if hasattr(val, 'item') and val.numel() == 1:
                    values.append(val.item())
                elif hasattr(val, 'mean'):
                    values.append(val.mean().item())
                else:
                    values.append(float(val))
            mean_val = torch.tensor(values).mean().item()
            final_metrics[f"{metric_key_prefix}_{key}"] = mean_val
        return final_metrics

    @abstractmethod
    def evaluation_loop(
        self,
        dataloader: torch.utils.data.DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        pass
