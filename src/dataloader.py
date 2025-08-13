from typing import List, Dict, Any

import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizer
from torch.utils.data import DataLoader

from src.common.mixed_dataloader import MixtureIterableLoader


class DataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.id_begin_of_simple_talk = self.tokenizer.convert_tokens_to_ids('<simple_talk>')
        self.id_end_of_simple_talk = self.tokenizer.convert_tokens_to_ids('</simple_talk>')
        self.tokenizer_kwargs = {
            "padding": True, 
            "return_tensors": "pt", 
            "add_special_tokens": False
        }


    def data_collate(self, features: List[Dict[str, Any]]) -> dict:
        """
        Collate function for the dataset.
        Processes main text, math reasoning, and calculates required indices.
        """
        
        if not features:
            return {}
        
        concat_parts = lambda x: " ".join([
            x["question"], x["simple_talk"], f' {x["final_answer"]}'
            ])
        full_texts = [concat_parts(feature) for feature in features]
        
        batch = self.tokenizer(full_texts, padding=True, return_tensors="pt")
        labels = self.labels_from_token(batch["input_ids"], batch["attention_mask"], self.id_begin_of_simple_talk)

        math_reasoning_texts = [feature["math_reasoning"] + self.tokenizer.eos_token for feature in features]
        math_labels = self.labels_from_text(batch, math_reasoning_texts)
        
        fa_texts = [str(feature["final_answer"]) + self.tokenizer.eos_token for feature in features]
        final_answer_labels = self.labels_from_text(batch, fa_texts)
        
        return dict(
            input_ids=batch["input_ids"], 
            attention_mask=batch["attention_mask"],
            labels=labels, 
            math_labels=math_labels,
            final_answer_labels=final_answer_labels,
        )
        
    def data_calibration_collate(self, features: List[Dict[str, Any]]) -> dict:
        
        """
        Collate function for the calibration dataset.
        Processes questions and answers. 
        """
        
        if not features:
            return {}

        full_texts = [feature["question"] + feature["answer"] for feature in features]
        batch = self.tokenizer(full_texts, padding=True, return_tensors="pt")
        
        answers = [feature["answer"] + self.tokenizer.eos_token for feature in features]
        labels = self.labels_from_text(batch, answers)
        
        
        return dict(
            input_ids=batch["input_ids"], 
            labels=labels, 
            attention_mask=batch["attention_mask"],
        )
    
    def labels_from_text(self, batch: dict[str, torch.Tensor], answer_text: str | list[str]):
    
        answer = self.tokenizer(answer_text, **self.tokenizer_kwargs)
        labels = self.labels_from_answers(batch["input_ids"], answer["input_ids"], batch["attention_mask"])
    
        return labels

    @staticmethod
    def labels_from_token(
            input_ids: torch.Tensor, attention_mask: torch.Tensor, start_token: int,
    ) -> torch.Tensor:
        
        """
        All before <simple_talk> and all padding tokens will be masked with -100. 
        """
        
        labels = torch.full_like(input_ids, fill_value=-100)

        for i, row in enumerate(input_ids):
            start_matches = (row == start_token).nonzero(as_tuple=True)

            if start_matches[0].numel() == 0:
                continue  

            start_idx = start_matches[0][-1].item()
            labels[i, start_idx:] = row[start_idx:]
            
        labels[attention_mask == 0] = -100

        return labels

    @staticmethod
    def labels_from_answers(
            input_ids: torch.Tensor, answers: torch.Tensor, attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        
        labels = torch.full_like(input_ids, fill_value=-100)
        
        for i, row in enumerate(answers):
            labels[i, -len(row):] = row
        
        labels[attention_mask == 0] = -100
        
        return labels


class DatasetProcessor(DataCollator):
    def __init__(self, tokenizer: PreTrainedTokenizer, cfg):
        super().__init__(tokenizer)
        
        self.cfg = cfg

    @staticmethod
    def split_datasets(main_dataset: DatasetDict, train_size: int, eval_size: int) -> tuple[Dataset, Dataset]:
    
        train_dataset = main_dataset["train"].select(range(train_size))
        eval_dataset = main_dataset["train"].select(range(train_size, train_size + eval_size))
        
        return train_dataset, eval_dataset

    def load_and_prepare(self):
        
        """Load and prepare the dataset."""
        
        main_dataset: DatasetDict = load_dataset(self.cfg.dataset.name)
        calibration_dataset: DatasetDict = load_dataset(self.cfg.calibration_dataset.name)
        
        train_size, eval_size = self.cfg.dataset.train_size, self.cfg.dataset.eval_size
        train_dataset, eval_dataset = self.split_datasets(main_dataset, train_size, eval_size)
        
        train_loader_main = DataLoader(
            train_dataset,
            batch_size = self.cfg.trainer.per_device_train_batch_size,
            shuffle = True,
            collate_fn = self.data_collate,   
        )
        
        eval_calib_dataset = calibration_dataset["train"].select(range(0))

        if self.cfg.calib_prob > 0:
            # TODO: Hi, dude, let's think about: we do not have to cut the calibration dataset, 
            # because of it is just for calibration - random talks. 
            # Also, it is not important to multiply this size on calib_prob. I guess it is for size matching only but... it could be skipped also.
            
            train_size, eval_size = list(map(lambda x: int(x * self.cfg.calib_prob), [train_size, eval_size]))
            train_calib_dataset, eval_calib_dataset = self.split_datasets(calibration_dataset, train_size, eval_size)

            train_loader_calib = DataLoader(
                train_calib_dataset,
                batch_size = self.cfg.trainer.per_device_train_batch_size,
                shuffle = True,
                collate_fn = self.data_calibration_collate,   
            )
            
            train_loader_main = MixtureIterableLoader(
                train_loader_main, train_loader_calib, self.cfg.calib_prob
            )

        return train_loader_main, eval_dataset, eval_calib_dataset






