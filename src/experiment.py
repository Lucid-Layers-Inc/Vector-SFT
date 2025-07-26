import os

from omegaconf import OmegaConf
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel
from peft import LoraConfig, get_peft_model, PeftModel  # type: ignore

from src.model import ModelWithAuxiliaryHead
from src.common.default import Experiment
from src.dataloader import DatasetProcessor


class SFTExperiment(Experiment):
    
    def __init__(self, config: str):
        
        super().__init__(config)
        self.resume_from_checkpoint = self.cfg.resume_from_checkpoint
        self.generation_prompts = self.cfg.generation.prompts
        self.generation_params = self.cfg.generation.generation_params
        
        self.base_model, self.tokenizer = self.prepare_model_and_tokenizer()
        
        self.dataset_processor = DatasetProcessor(self.tokenizer, self.cfg)

    def prepare_model_and_tokenizer(self) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
                
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.name)
        tokenizer.padding_side = "left"
        
        base_model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model.name,
            torch_dtype=getattr(torch, self.cfg.model.dtype),
            device_map=self.cfg.model.device_map,
            attn_implementation=self.cfg.model.attn_implementation,
        )

        return base_model, tokenizer
    
    def add_loras_to_base_model(self):
        if self.resume_from_checkpoint is not None and os.path.exists(self.resume_from_checkpoint):
            # Load LoRA from checkpoint
            print(f"Loading LoRA weights from {self.resume_from_checkpoint}")
            self.lora_wrapped = PeftModel.from_pretrained(self.base_model, self.resume_from_checkpoint, is_trainable=True)
            
        else:
            # Create new LoRA
            print("Creating new LoRA configuration")
            peft_config = LoraConfig(**OmegaConf.to_container(self.cfg.peft, resolve=True))  # type: ignore
            self.lora_wrapped = get_peft_model(self.base_model, peft_config)
    
    def add_translator_to_model(self):
        """Load custom weights if resuming from checkpoint"""
        if self.resume_from_checkpoint is not None and os.path.exists(self.resume_from_checkpoint):
            print(f"Loading custom weights from {self.resume_from_checkpoint}")
            self.model.translator.load_pretrained(self.resume_from_checkpoint)
        else:
            self.model.translator.init_weights()   
    
    def setup_lora_and_auxiliary(self):

        """Setup PEFT configuration and auxiliary matrices at the last hidden layer"""

        lm_head = self.base_model.get_output_embeddings()
        self.add_loras_to_base_model()
        
        self.lora_wrapped.enable_input_require_grads()

        # Create auxiliary head model
        self.model = ModelWithAuxiliaryHead(
                base_model=self.lora_wrapped,
                lm_head=lm_head
            )
        self.add_translator_to_model()
        

    def prepare_datasets(self):
        """
        Loads and prepares the dataset. Returns the data collator as callable function.
        Returns: The data collator function.
        """
        self.mix_data_loader, self.eval_dataset, self.eval_calib_dataset = self.dataset_processor.load_and_prepare()
        