import torch
import torch.nn.functional as F
from pydantic import BaseModel


class Betas(BaseModel): 
    beta_0: float = 0.1
    beta_1: float = 0.5
    beta_2: float = 0.5
    beta_3: float = 0.4
    

def plain_cross_entropy_loss(logits: torch.Tensor, input_ids: torch.Tensor, reduction='mean'):
    _, _, vocab_size = logits.shape

    # Shift logits and labels for autoregressive training
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = input_ids[:, 1:].contiguous()

    # Flatten for cross-entropy
    logits_flat = shifted_logits.reshape(-1, vocab_size)
    labels_flat = shifted_labels.reshape(-1)

    return F.cross_entropy(
        logits_flat,
        labels_flat,
        ignore_index=-100,
        reduction=reduction
    )

def calculate_all_main_losses(
    outputs: dict[str, torch.Tensor], 
    inputs: dict[str, torch.Tensor], 
    betas: Betas
    ) -> dict[str, torch.Tensor]:
    
    """
    Main function to calculate all the losses. Gather all of them into one dict. 
    """
        
    simple_talk_loss, final_answer_loss = compute_simple_and_final_answer_loss(
        outputs["logits"], inputs["input_ids"], inputs["attention_mask"], inputs["starts"], inputs["ends"]
    )
    
    math_loss = compute_math_loss(outputs["math_logits"], inputs["math_labels"], inputs["math_attention_mask"])
    total_loss = betas.beta_1 * math_loss + betas.beta_2 * simple_talk_loss + betas.beta_3 * final_answer_loss

    return {
        "total_loss": total_loss,
        "math_loss": math_loss,
        "simple_talk_loss": simple_talk_loss,
        "final_answer_loss": final_answer_loss
    }

def compute_simple_and_final_answer_loss(
    logits: torch.Tensor, input_ids: torch.Tensor, 
    attention_mask: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        loss = plain_cross_entropy_loss(logits, input_ids, reduction='none')
        device = logits.device
        batch_size, seq_len, _ = logits.shape
        
        # Reshape loss back to [batch_size, seq_len - 1] to apply masks
        loss_per_token = loss.view(batch_size, seq_len - 1)

        # Create indices grid for the SHIFTED sequence
        shifted_indices = torch.arange(seq_len - 1, device=device).expand(batch_size, -1)

        starts_exp = starts.unsqueeze(1)
        ends_exp = ends.unsqueeze(1)
        real_lengths = attention_mask.sum(dim=1).unsqueeze(1)

        # Create masks for the shifted tensors.
        # A loss at shifted_index `i` corresponds to a prediction for a token at original_index `i+1`.
        # So, we adjust the masks accordingly.
        mask_simple_talk = (shifted_indices >= starts_exp - 1) & (shifted_indices <= ends_exp - 1)
        mask_final_answer = (shifted_indices >= ends_exp) & (shifted_indices < real_lengths - 1)
       
        # Calculate mean loss for each part, handling empty masks to avoid NaN
        simple_talk_loss = loss_per_token[mask_simple_talk].mean() if mask_simple_talk.any() else torch.tensor(0.0, device=device)
        final_answer_loss = loss_per_token[mask_final_answer].mean() if mask_final_answer.any() else torch.tensor(0.0, device=device)
        
        return simple_talk_loss, final_answer_loss
    
    
def compute_math_loss(math_logits, math_input_ids, math_attention_mask):

    target_math_tokens = math_input_ids[math_attention_mask == 1] # [T] 
    math_loss = F.cross_entropy(math_logits, target_math_tokens) 

    return math_loss

