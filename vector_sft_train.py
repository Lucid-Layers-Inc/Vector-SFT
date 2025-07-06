import fire

from trl import SFTConfig

from src.callbacks import ClearMLCallback, SaveCustomWeightsCallback
from src.experiment import SFTExperiment
from src.trainer import VectorSFTTrainer


def main(config: str):

    checkpoint_path = "VectorSFT-checkpoints/checkpoint-1164" 
    experiment = SFTExperiment(config, resume_from_checkpoint=checkpoint_path)
    experiment.setup_lora_and_auxiliary()
    data_collator = experiment.prepare_datasets()
    training_args = SFTConfig(**experiment.cfg.trainer)



    experiment.task_init()
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

    trainer.train(resume_from_checkpoint=checkpoint_path)

if __name__ == "__main__":
    fire.Fire(main)