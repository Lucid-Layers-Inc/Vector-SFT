import os
import torch
import torch.nn as nn


class RecoverableModel(nn.Module):
    weight_path = "custom_model.pt"
    
    def save_pretrained(self, save_directory: str):
            torch.save(self.state_dict(), os.path.join(save_directory, self.weight_path))

    def load_pretrained(self, checkpoint_path: str):
        custom_weights_path = os.path.join(checkpoint_path, self.weight_path)
        custom_state_dict = torch.load(custom_weights_path, map_location='cpu')
        self.load_state_dict(custom_state_dict)