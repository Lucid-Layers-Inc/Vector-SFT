import fire

from trl import SFTConfig

from src.callbacks import ClearMLCallback, SaveCustomWeightsOnHubCallback
from src.experiment import SFTExperiment
from src.trainer import VectorSFTTrainer


def main(config: str):

    experiment = SFTExperiment(
        config,
        # resume_from_checkpoint=checkpoint_path
        )
        
    experiment.setup_lora_and_auxiliary()
    experiment.prepare_datasets()

    training_args = SFTConfig(**experiment.cfg.trainer)

    trainer = VectorSFTTrainer(
        model=experiment.model,
        processing_class=experiment.tokenizer,
        args=training_args,
        train_dataset=experiment.mix_data_loader,
        eval_dataset=[experiment.eval_dataset, experiment.eval_calib_dataset],
        data_collator= lambda x: x, # batches are already prepeared
        callbacks=[
            ClearMLCallback(experiment.task),
            SaveCustomWeightsOnHubCallback()
        ],
        dataset_processor=experiment.dataset_processor
    )

    trainer.train(
        # resume_from_checkpoint=checkpoint_path
    )

if __name__ == "__main__":
    fire.Fire(main)