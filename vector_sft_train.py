import fire
from trl import SFTConfig

from src.callbacks import ClearMLCallback
from src.experiments.vector_sft import SFTExperiment
from src.experiments.trainer import VectorSFTTrainer


def main(config: str):

    experiment = SFTExperiment(config)
    experiment.setup_lora_and_auxiliary()
    data_collate = experiment.prepare_datasets()

    training_args = SFTConfig(**experiment.cfg.trainer)


    trainer = VectorSFTTrainer(
        model=experiment.model,
        processing_class=experiment.tokenizer,
        args=training_args,
        train_dataset=experiment.mix_data_loader,
        eval_dataset=experiment.eval_dataset,
        data_collator= data_collate, 
        callbacks=[ClearMLCallback(experiment.task)],
        dataset_processor=experiment.dataset_processor,
    )

    trainer.train()


if __name__ == "__main__":
    fire.Fire(main)