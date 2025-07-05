import os
import torch
import torch.nn as nn
from trl import SFTTrainer
from typing import Optional
from transformers.trainer_utils import EvalLoopOutput
from typing import List, Dict, Any

class VectorSFTTrainer(SFTTrainer):
    
    def compute_loss(self, model, inputs, num_items_in_batch, return_outputs=False):
        """
        Custom compute_loss method to handle multiple losses returned by the model's forward pass.
        This method accumulates all scalar and list-of-scalar outputs from the model
        into `self._metrics` for later logging.
        """
        outputs = model(**inputs)

        # Extract the total loss for backpropagation
        total_loss = outputs.get("loss")

        # Accumulate all relevant metrics from the model's output
        # self._metrics is a defaultdict(list) that will be averaged and logged by self.log()
        for key, value in outputs.items():
            if key == "A_losses":
                for i, loss_item in enumerate(value):
                    if isinstance(loss_item, torch.Tensor) and loss_item.numel() == 1:
                        self._metrics[f"A_loss_{i}"].append(loss_item.item())
            elif key != 'loss' and isinstance(value, torch.Tensor) and value.numel() == 1:
                self._metrics[key].append(value.item())

        return (total_loss, outputs) if return_outputs else total_loss

    def log(self, logs: dict, start_time: Optional[float] = None) -> None:
        """
        Overrides the parent log method to average and log custom metrics
        that have been accumulated in `self._metrics`.
        """
        # Average the accumulated metrics if any are present
        if self._metrics:
            metrics_to_log = {key: sum(values) / len(values) for key, values in self._metrics.items()}
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
        
        model = self.model
        model.eval()

        all_losses = []
        all_math_losses = []
        all_simple_talk_losses = []
        all_final_answer_losses = []
        all_A_losses_steps = []
        
        for _, inputs in enumerate(dataloader):
            
            inputs = self._prepare_inputs(inputs)

            with torch.no_grad():
                outputs = model(**inputs)

            loss = outputs.get("loss", None)
            if loss is not None:
                all_losses.append(self.gather_function(loss.detach()))
            
            math_loss = outputs.get("math_loss", None)
            if math_loss is not None:
                all_math_losses.append(self.gather_function(math_loss.detach()))
            
            simple_talk_loss = outputs.get("simple_talk_loss", None)
            if simple_talk_loss is not None:
                all_simple_talk_losses.append(self.gather_function(simple_talk_loss.detach()))
            
            final_answer_loss = outputs.get("final_answer_loss", None)
            if final_answer_loss is not None:
                all_final_answer_losses.append(self.gather_function(final_answer_loss.detach()))
            
            A_losses = outputs.get("A_losses", None)
            if A_losses is not None:
                detached_A_losses = [l.detach() for l in A_losses]
                all_A_losses_steps.append(self.gather_function(detached_A_losses))


            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)

        # --- aggregate and compute mean ---
        
        mean_loss = torch.tensor(all_losses).mean().item() if all_losses else None
        mean_math_loss = torch.tensor(all_math_losses).mean().item() if all_math_losses else None
        mean_simple_talk_loss = torch.tensor(all_simple_talk_losses).mean().item() if all_simple_talk_losses else None
        mean_final_answer_loss = torch.tensor(all_final_answer_losses).mean().item() if all_final_answer_losses else None
        
        # --- create metrics dictionary ---
        metrics = {}
        if mean_loss is not None:
            metrics[f"{metric_key_prefix}_loss"] = mean_loss
        if mean_math_loss is not None:
            metrics[f"{metric_key_prefix}_math_loss"] = mean_math_loss
        if mean_simple_talk_loss is not None:
            metrics[f"{metric_key_prefix}_simple_talk_loss"] = mean_simple_talk_loss
        if mean_final_answer_loss is not None:
            metrics[f"{metric_key_prefix}_final_answer_loss"] = mean_final_answer_loss

        # compute mean A_losses and add them to metrics
        if all_A_losses_steps:
            A_losses_tensor = torch.tensor(all_A_losses_steps, device=self.args.device)
            mean_A_losses_per_segment = A_losses_tensor.mean(dim=0)
            for i, mean_loss in enumerate(mean_A_losses_per_segment):
                metrics[f"{metric_key_prefix}_A_loss_{i}"] = mean_loss.item()

        # logging will be done in evaluate method
        return EvalLoopOutput(predictions=None, 
                              label_ids=None, 
                              metrics=metrics, 
                              num_samples=len(dataloader))

    # def _save(self, output_dir: Optional[str] = None, state_dict=None):
        
    #     super()._save(output_dir, state_dict)
    #     self.model.save_pretrained(output_dir)