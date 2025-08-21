import os
from typing import List, Dict, Any

from omegaconf import OmegaConf
import torch
from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel
from peft import LoraConfig, get_peft_model, PeftModel  # type: ignore
from torch.utils.data import DataLoader

from src.model import ModelWithAuxiliaryHead
from src.common.default import Experiment
from src.common.mixed_dataloader import MixtureIterableLoader


def create_labels(
        input_ids: torch.Tensor, attention_mask: torch.Tensor, start_token: int, end_token: int
) -> tuple[torch.Tensor, list[int], list[int]]:
    
    """
    All before <simple_talk> and all padding tokens are masked with -100. 
    """
    
    labels = torch.full_like(input_ids, fill_value=-100)
    starts, ends = [], []

    for i, row in enumerate(input_ids):
        start_matches = (row == start_token).nonzero(as_tuple=True)
        end_matches = (row == end_token).nonzero(as_tuple=True)

        if start_matches[0].numel() == 0 or end_matches[0].numel() == 0:
            continue  # pass if start or end token is not found

        start_idx, end_idx = start_matches[0][-1].item(), end_matches[0][-1].item()
        labels[i, start_idx:] = row[start_idx:]
        starts.append(start_idx)
        ends.append(end_idx)
        
    labels[attention_mask == 0] = -100

    return labels, starts, ends



class DatasetProcessor:
    def __init__(self, tokenizer: PreTrainedTokenizer, cfg):
        self.tokenizer = tokenizer
        self.cfg = cfg

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.id_begin_of_simple_talk = self.tokenizer.convert_tokens_to_ids('<simple_talk>')
        self.id_end_of_simple_talk = self.tokenizer.convert_tokens_to_ids('</simple_talk>')


    def load_and_prepare(self):
        
        """Load and prepare the dataset."""
        
        main_dataset: DatasetDict = load_dataset(self.cfg.dataset.name)  # type: ignore
        calibration_dataset: DatasetDict = load_dataset(self.cfg.calibration_dataset.name)  # type: ignore
        
        train_size, eval_size = self.cfg.dataset.train_size, self.cfg.dataset.eval_size
        train_dataset = main_dataset["train"].select(range(train_size))
        eval_dataset = main_dataset["train"].select(range(train_size, train_size + eval_size))
        
        train_loader_main = DataLoader(
            train_dataset, # type: ignore
            batch_size = self.cfg.trainer.per_device_train_batch_size,
            shuffle = True,
            collate_fn = self.data_collate,   
        )

        train_calib_size = int(len(train_dataset) * self.cfg.calib_prob)
        eval_calib_size = int(len(eval_dataset) * self.cfg.calib_prob)
        
        train_calib_dataset = calibration_dataset["train"].select(range(train_calib_size))
        eval_calib_dataset = calibration_dataset["train"].select(range(train_calib_size, train_calib_size + eval_calib_size))

        train_loader_calib = DataLoader(
            train_calib_dataset, # type: ignore
            batch_size = self.cfg.trainer.per_device_train_batch_size,
            shuffle = True,
            collate_fn = self.data_calibration_collate,   
        )
        
        mix_data_loader = MixtureIterableLoader(
            train_loader_main, train_loader_calib, self.cfg.calib_prob
        )
        


        return mix_data_loader, eval_dataset, eval_calib_dataset

    def data_collate(self, features: List[Dict[str, Any]]) -> Dict[str, Any | torch.Tensor]:
        """
        Collate function for the dataset.
        Processes main text, math reasoning, and calculates required indices.
        """
        
        if not features:
            return {}

        
        full_texts = [feature["question"] + feature["simple_talk"] + ' ' + str(feature["final_answer"]) + self.tokenizer.eos_token for feature in features]
        
        
        batch = self.tokenizer(full_texts, padding=True, return_tensors="pt")
        labels, starts, ends = create_labels(
            batch["input_ids"], batch["attention_mask"], self.id_begin_of_simple_talk, self.id_end_of_simple_talk  # type: ignore
        )

        math_reasoning_texts = [feature["math_reasoning"] for feature in features]
        batch_math = self.tokenizer(math_reasoning_texts)
        math_reasoning_lengths = torch.tensor([len(row) for row in batch_math['input_ids']], dtype=torch.int64)  # type: ignore

        cleans = [int(feature["L_tokens"]) for feature in features]

        batch_math_padded = self.tokenizer(math_reasoning_texts, padding=True, return_tensors="pt")
        
        batch_size = batch["input_ids"].shape[0]
        return {
            "input_ids": batch["input_ids"], 
            "labels": labels, 
            "attention_mask": batch["attention_mask"],
            "starts": torch.tensor(starts, dtype=torch.long),
            "ends": torch.tensor(ends, dtype=torch.long),
            "math_lengths": math_reasoning_lengths,
            "math_labels": batch_math_padded["input_ids"],
            "math_attention_mask": batch_math_padded["attention_mask"],
            "source_label": torch.zeros(batch_size, dtype=torch.long),
            "cleans": torch.tensor(cleans, dtype=torch.long)
        }
        
    def data_calibration_collate(self, features: List[Dict[str, Any]]) -> Dict[str, Any | torch.Tensor]:
        
        """
        Collate function for the calibration dataset.
        Processes questions and answers. 
        """
        
        if not features:
            return {}

        full_texts = [feature["question"] + feature["answer"] for feature in features]
        batch = self.tokenizer(full_texts, padding=True, return_tensors="pt")
        labels = batch["input_ids"].clone()  # type: ignore
        
        questions = [feature["question"] for feature in features]
        tokenized_questions = self.tokenizer(questions)
        q_lengths = [len(row) for row in tokenized_questions['input_ids']]  # type: ignore

        for i in range(len(q_lengths)):
            labels[i, :q_lengths[i]] = -100
        
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        batch["labels"] = labels
        batch_size = batch["input_ids"].shape[0]
        
        return {
            "input_ids": batch["input_ids"], 
            "labels": batch["labels"], 
            "attention_mask": batch["attention_mask"],
            "source_label": torch.ones(batch_size, dtype=torch.long)
         }


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
        model_kwargs = dict(
            torch_dtype=getattr(torch, self.cfg.model.dtype),
            device_map=self.cfg.model.device_map
            )
        if getattr(self.cfg.model, "attn_implementation", None) is not None:
            model_kwargs["attn_implementation"] = self.cfg.model.attn_implementation
        
        base_model = AutoModelForCausalLM.from_pretrained(self.cfg.model.name, **model_kwargs)

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

        # --------------------------------------------
        clean_base_model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model.name,
            torch_dtype=getattr(torch, self.cfg.model.dtype),
            device_map=self.cfg.model.device_map,
            attn_implementation=self.cfg.model.attn_implementation,
        )
        # ---------------------------------------------

        # Create auxiliary head model
        self.model = ModelWithAuxiliaryHead(
            base_model=self.lora_wrapped,
            clean_base_model=clean_base_model,
            lm_head=lm_head,
            bert_mlp_size=self.cfg.auxiliary.bert_mlp_size,
            num_attention_heads=self.cfg.auxiliary.num_attention_heads,
            bert_hidden_size=self.cfg.auxiliary.bert_hidden_size
        )
        self.add_translator_to_model()
        

    def prepare_datasets(self):
        """
        Loads and prepares the dataset. Returns the data collator as callable function.
        Returns: The data collator function.
        """
        self.mix_data_loader, self.eval_dataset, self.eval_calib_dataset = self.dataset_processor.load_and_prepare()
        