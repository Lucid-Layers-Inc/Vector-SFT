import torch
import torch.nn.functional as F


def calculate_calibration_loss(outputs, inputs):
    
    """
    Сalculate calibration_loss. 
    """
    
    logits = outputs["logits"]
    _, _, vocab_size = logits.shape

    # Shift logits and labels for autoregressive training
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = inputs['input_ids'][:, 1:].contiguous()

    # Flatten for cross-entropy
    logits_flat = shifted_logits.reshape(-1, vocab_size)
    labels_flat = shifted_labels.reshape(-1)

    calibration_loss = F.cross_entropy(
        logits_flat,
        labels_flat,
        ignore_index=-100
    )
    
    return {
        "calibration_loss": calibration_loss
    }

def calculate_all_main_losses(model, outputs, inputs):
    
    """
    Main function to calculate all the losses. Gather all of them into one dict. 
    """
    
    
    logits = outputs["logits"]
    last_hidden_state = outputs["last_hidden_state"]
    
    simple_talk_loss, final_answer_loss = compute_simple_and_final_answer_loss(
        logits, inputs["input_ids"], inputs["attention_mask"], inputs["starts"], inputs["ends"]
    )
    
    math_loss = compute_math_loss(
        model, last_hidden_state, inputs["math_labels"], inputs["math_attention_mask"], inputs["starts"], inputs["math_lengths"]
    )

    total_loss = model.beta_1 * math_loss + model.beta_2 * simple_talk_loss + model.beta_3 * final_answer_loss

    return {
        "total_loss": total_loss,
        "math_loss": math_loss,
        "simple_talk_loss": simple_talk_loss,
        "final_answer_loss": final_answer_loss
    }

def compute_simple_and_final_answer_loss(logits, input_ids, attention_mask, starts, ends):
        
        device = logits.device
        batch_size, seq_len, vocab_size = logits.shape

        # Shift logits and labels for autoregressive training
        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_labels = input_ids[:, 1:].contiguous()

        # Flatten for cross-entropy
        logits_flat = shifted_logits.reshape(-1, vocab_size)
        labels_flat = shifted_labels.reshape(-1)

        # Calculate loss for each token, but do not reduce
        loss = F.cross_entropy(
            logits_flat,
            labels_flat,
            ignore_index=-100,
            reduction='none'
        )
        
        # Reshape loss back to [B, S-1] to apply masks
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


def get_hiddens_and_indices(last_hidden_state, starts, math_lengths, math_input_ids):
        
    device = last_hidden_state.device

    # --- transform hidden states corresponging to math thoughts ---
    
    # Choose hidden states corresponding to indices where math thoughts must be hidden 
    batch_size, seq_len, _ = last_hidden_state.shape
    indices = torch.arange(seq_len, device=device).expand(batch_size, -1) # -1 means to keep the last dimension the same
    # indices:
    # tensor([[0, 1, 2, 3, 4],
    #         [0, 1, 2, 3, 4],
    #         [0, 1, 2, 3, 4]])
    mask_hidden = (indices >= starts.unsqueeze(1)) & (indices < (starts + math_lengths).unsqueeze(1))
    hiddens = last_hidden_state[mask_hidden] # [T, H]
    

    # choose matrices for each math hidden state according to the partition
    _, math_seq_len = math_input_ids.shape
    indices = torch.arange(math_seq_len, device=device).expand(batch_size, -1)
    
    return hiddens, indices
    
    
def compute_math_loss(model, last_hidden_state, math_input_ids, math_attention_mask, starts, math_lengths):


    math_hidden_states, math_indices = get_hiddens_and_indices(last_hidden_state, starts, math_lengths, math_input_ids)
    segment_ids = model.segment_indices[math_indices[math_attention_mask == 1]] 
    math_hiddens = model.translator(math_hidden_states, segment_ids)
    math_logits = model.lm_head(math_hiddens)

    target_math_tokens = math_input_ids[math_attention_mask == 1] # [T] 
    math_loss = F.cross_entropy(math_logits, target_math_tokens) 

    return math_loss

