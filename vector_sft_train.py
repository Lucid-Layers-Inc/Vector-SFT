import fire

from trl import SFTConfig

from src.callbacks import ClearMLCallback, SaveCustomWeightsOnHubCallback, GenerationCallback
from src.common.losses import Betas
from src.experiment import SFTExperiment
from src.trainer import VectorSFTTrainer


def main(config: str):

    experiment = SFTExperiment(
        config,
        )
    experiment.setup_lora_and_auxiliary()
    experiment.prepare_datasets()
    training_args = SFTConfig(**experiment.cfg.trainer)
    
    experiment.task_init()

    eval_datasets = [experiment.eval_dataset]
    if len(experiment.eval_calib_dataset) > 0:
        eval_datasets.append(experiment.eval_calib_dataset)

    trainer = VectorSFTTrainer(
        model=experiment.model,
        processing_class=experiment.tokenizer,
        args=training_args,
        train_dataset=experiment.mix_data_loader,
        eval_dataset=eval_datasets,
        data_collator= lambda x: x, # batches are already prepeared
        callbacks=[
            ClearMLCallback(experiment.task),
            SaveCustomWeightsOnHubCallback(),
            GenerationCallback(
                prompts=experiment.generation_prompts,
                tokenizer=experiment.tokenizer
            )
        ],
        dataset_processor=experiment.dataset_processor,
        betas=Betas(**experiment.cfg.betas)
    )

    trainer.train(
        resume_from_checkpoint=experiment.cfg.resume_from_checkpoint
    )

if __name__ == "__main__":
    fire.Fire(main)