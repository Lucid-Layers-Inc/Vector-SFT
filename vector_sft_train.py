import fire

from trl import SFTConfig

from src.callbacks import ClearMLCallback, SaveCustomWeightsCallback
from src.experiment import SFTExperiment
from src.trainer import VectorSFTTrainer


def main(config: str):

    experiment = SFTExperiment(config)
    experiment.setup_lora_and_auxiliary()
    data_collator = experiment.prepare_datasets()

    training_args = SFTConfig(**experiment.cfg.trainer)

    trainer = VectorSFTTrainer(
        model=experiment.model,
        processing_class=experiment.tokenizer,
        args=training_args,
        train_dataset=experiment.train_dataset,
        eval_dataset=experiment.eval_dataset,
        data_collator=data_collator,
        callbacks=[
            ClearMLCallback(experiment.task),
            SaveCustomWeightsCallback()
        ]
    )

    trainer.train()

    final_output_dir = experiment.train_args.output_dir
    trainer.save_model(final_output_dir) 

if __name__ == "__main__":
    fire.Fire(main)