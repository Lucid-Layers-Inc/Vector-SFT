import fire

from torch.utils.data import DataLoader
from trl import SFTConfig

from src.callbacks import ClearMLCallback, SaveCustomWeightsOnHubCallback, GenerationCallback, SaveFolderOnHubCallback
from src.sae.experiment import SAEExperiment
from src.sae.trainer import SAETrainer


def collate(inputs):
    input_dict = {}
    for key, value in inputs.items():
        if key in ["input_ids", "attention_mask"]:
            input_dict[key] = value
    return input_dict


def main(config: str):

    experiment = SAEExperiment(config)
    experiment.build_saes()
    experiment.prepare_datasets()
    training_args = SFTConfig(**experiment.cfg.trainer)
    
    experiment.task_init()
    
    eval_loader_main = DataLoader(
        experiment.eval_dataset,
        batch_size = experiment.cfg.trainer.per_device_eval_batch_size,
        shuffle = True,
        collate_fn = experiment.dataset_processor.data_collate,   
    )


    trainer = SAETrainer(
        model=experiment.model,
        processing_class=experiment.tokenizer,
        args=training_args,
        train_dataset=experiment.mix_data_loader,
        eval_dataset=eval_loader_main,
        data_collator=collate,
        callbacks=[
            SaveFolderOnHubCallback(),
        ],
        dataset_processor=experiment.dataset_processor,
        saes=experiment.saes,
        sae_cfg=dict(experiment.cfg.sae),
    )

    trainer.train(
        resume_from_checkpoint=experiment.cfg.resume_from_checkpoint
    )

if __name__ == "__main__":
    fire.Fire(main)