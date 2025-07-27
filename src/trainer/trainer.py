from typing import List, Optional

import torch
from torch.utils.data import Dataset
from transformers.trainer_utils import EvalLoopOutput
from trl import SFTTrainer

from src.common.losses import Betas, main_loss, calibration_loss
from src.dataloader import DatasetProcessor
from src.trainer.abstracts import EvaluatorMixin, WeightLoaderMixin

        
class VectorSFTTrainer(EvaluatorMixin, WeightLoaderMixin, SFTTrainer):
    def __init__(self, *args, dataset_processor: DatasetProcessor, betas: Betas, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_processor = dataset_processor
        self.betas = betas
        
    def compute_loss(self, model, inputs: dict, num_items_in_batch=None, return_outputs=False):
        
        """
        Custom compute_loss method to handle multiple losses.
        Accumulates all losses into `self._metrics` for later logging.
        """
        
        outputs = model(**inputs)
                
        if inputs.get("math_labels") is not None:
            losses_dict = main_loss(outputs, inputs, self.betas)
            current_loss = losses_dict["total_loss"]
        else: 
            losses_dict = calibration_loss(outputs, inputs)
            current_loss = losses_dict["calibration_loss"]
                
        for key, value in losses_dict.items():
            self._metrics[key].append(value.item())

        return (current_loss, outputs) if return_outputs else current_loss

    def get_train_dataloader(self) -> Dataset:
        return self.train_dataset

    def log(self, logs: dict, start_time: Optional[float] = None) -> None:
        if self._metrics:
            metrics_to_log = {
                key: sum(values) / len(values)
                for key, values in self._metrics.items()
                if len(values) != 0
            }
            
            logs.update(metrics_to_log)
            self._metrics.clear()

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
        num_samples = len(self.eval_dataset[0])
        
        all_metrics_main = self.eval_on_dataset(
            self.eval_dataset[0], self.dataset_processor.data_collate, 
            model, main_loss, betas=self.betas
        )
        if len(self.eval_dataset) > 1:
            num_samples += len(self.eval_dataset[1])
            all_metrics_calib = self.eval_on_dataset(
                self.eval_dataset[1], self.dataset_processor.data_calibration_collate, 
                model, calibration_loss
            )
        
        final_metrics = self.gather_custom_metrics(all_metrics_main, metric_key_prefix)
        final_metrics.update(self.gather_custom_metrics(all_metrics_calib, metric_key_prefix))
        
        return EvalLoopOutput(
                predictions=None, # type: ignore
                label_ids=None, 
                metrics=final_metrics, 
                num_samples=num_samples
            )
        
    
    def _save(self, output_dir: str, state_dict=None):
        self.model.save_pretrained(output_dir)
    
    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        self.create_optimizer_and_scheduler(num_training_steps=self.args.max_steps)
        self.resume_trainer_only(resume_from_checkpoint)
        