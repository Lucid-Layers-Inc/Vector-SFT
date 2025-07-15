from typing import List, Dict, Any
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, PreTrainedTokenizer, PreTrainedModel

from peft import LoraConfig, get_peft_model
from src.experiments.model import ModelWithAuxiliaryHead
from src.experiments.default import Experiment
from omegaconf import OmegaConf


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
        dataset = load_dataset(self.cfg.dataset.name,)

        train_size, eval_size = self.cfg.dataset.train_size, self.cfg.dataset.eval_size
        train_dataset = dataset["train"].select(range(train_size))
        eval_dataset = dataset["train"].select(range(train_size, train_size + eval_size))

        return train_dataset, eval_dataset

    def data_collate(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate function for the dataset.
        Processes main text, math reasoning, and calculates required indices.
        """
        
        if not features:
            return {}

        
        full_texts = [feature["question"] + feature["simple_talk"] + ' ' + str(feature["final_answer"]) + self.tokenizer.eos_token for feature in features]
        
        
        batch = self.tokenizer(full_texts, padding=True, return_tensors="pt")
        labels, starts, ends = create_labels(
            batch["input_ids"], batch["attention_mask"], self.id_begin_of_simple_talk, self.id_end_of_simple_talk
        )

        math_reasoning_texts = [feature["math_reasoning"] for feature in features]
        batch_math = self.tokenizer(math_reasoning_texts)
        math_reasoning_lengths = torch.tensor([len(row) for row in batch_math['input_ids']], dtype=torch.int64)
        
        batch_math_padded = self.tokenizer(math_reasoning_texts, padding=True, return_tensors="pt")
        

        return {
            "input_ids": batch["input_ids"], 
            "labels": labels, 
            "attention_mask": batch["attention_mask"],

            "starts": torch.tensor(starts, dtype=torch.long),
            "ends": torch.tensor(ends, dtype=torch.long),
            "math_lengths": math_reasoning_lengths,
            "math_labels": batch_math_padded["input_ids"],
            "math_attention_mask": batch_math_padded["attention_mask"],
        }


class SFTExperiment(Experiment):
    

    def __init__(self, config: str):
        super().__init__(config)
        self.task_init()

        self.base_model, self.tokenizer = self.prepare_model_and_tokenizer()
        self.dataset_processor = DatasetProcessor(self.tokenizer, self.cfg)

    def prepare_model_and_tokenizer(self) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        
        # Initialize tokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.name)
        tokenizer.add_special_tokens({"additional_special_tokens": list(self.cfg.model.special_tokens)})

        # Initialize model
        
        base_model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model.name,
            torch_dtype=getattr(torch, self.cfg.model.dtype),
            device_map=self.cfg.model.device_map,
            attn_implementation=self.cfg.model.attn_implementation,
        )
        
        base_model.resize_token_embeddings(len(tokenizer))


        return base_model, tokenizer

    def setup_lora_and_auxiliary(self):
        """Setup PEFT configuration and auxiliary matrices at the last hidden layer"""

        lm_head = self.base_model.get_output_embeddings()
        if lm_head is None:
            raise ValueError("Could not get output embeddings from the base model before applying PEFT.")

        peft_params = OmegaConf.to_container(self.cfg.peft, resolve=True)
        peft_config = LoraConfig(**peft_params)
        self.lora_wrapped = get_peft_model(self.base_model, peft_config)
        self.lora_wrapped.enable_input_require_grads()


        self.model = ModelWithAuxiliaryHead(
            #config=self.lora_wrapped.config,
            base_model=self.lora_wrapped,
            lm_head=lm_head,
            N_max=self.cfg.auxiliary.N_max,
            num_segments=self.cfg.auxiliary.num_segments,
            beta_1=self.cfg.auxiliary.beta_1,
            beta_2=self.cfg.auxiliary.beta_2,
            beta_3=self.cfg.auxiliary.beta_3,
            r=self.cfg.auxiliary.segments_rank,
            )

    def prepare_datasets(self) -> callable:
        """
        Loads and prepares the dataset. Returns the data collator as callable function.
        Returns: The data collator function.
        """
        self.train_dataset, self.eval_dataset = self.dataset_processor.load_and_prepare()
        return self.dataset_processor.data_collate
