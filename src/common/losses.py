import torch
import torch.nn.functional as F
from pydantic import BaseModel


class Betas(BaseModel): 
    beta_0: float = 0.1
    beta_1: float = 0.5
    beta_2: float = 0.5
    beta_3: float = 0.4
    

def plain_cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor, reduction='mean'):
    _, _, vocab_size = logits.shape

    # Shift logits and labels for autoregressive training
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()

    # Flatten for cross-entropy
    logits_flat = shifted_logits.reshape(-1, vocab_size)
    labels_flat = shifted_labels.reshape(-1)

    return F.cross_entropy(
        logits_flat,
        labels_flat,
        ignore_index=-100,
        reduction=reduction
    )

def main_loss(
    outputs: dict[str, torch.Tensor], 
    inputs: dict, 
    betas: Betas
    ) -> dict[str, torch.Tensor]:
    
    """
    Main function to calculate all the losses. Gather all of them into one dict. 
    """
    
    simple_talk_loss = plain_cross_entropy_loss(outputs["logits"], inputs["labels"])
    final_answer_loss = plain_cross_entropy_loss(outputs["logits"], inputs["final_answer_labels"])
    math_loss = plain_cross_entropy_loss(outputs["math_logits"], inputs["math_labels"])
    total_loss = simple_talk_loss*betas.beta_1 + final_answer_loss*betas.beta_2 + math_loss*betas.beta_3

    return {
        "total_loss": total_loss,
        "math_loss": math_loss,
        "simple_talk_loss": simple_talk_loss,
        "final_answer_loss": final_answer_loss
    }

def calibration_loss(outputs: dict[str, torch.Tensor], inputs: dict) -> dict[str, torch.Tensor]:
    return {"calibration_loss": plain_cross_entropy_loss(outputs["logits"], inputs["input_ids"])}