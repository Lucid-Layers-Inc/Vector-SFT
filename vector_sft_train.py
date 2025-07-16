import fire
from trl import SFTConfig

from src.callbacks import ClearMLCallback
from src.experiments.vector_sft import SFTExperiment
from src.experiments.trainer import VectorSFTTrainer


def main(config: str):

    experiment = SFTExperiment(config)
    experiment.setup_lora_and_auxiliary()
    experiment.prepare_datasets()

    training_args = SFTConfig(**experiment.cfg.trainer)

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
        callbacks=[ClearMLCallback(experiment.task)],
        dataset_processor=experiment.dataset_processor,
    )

    trainer.train()

    #final_output_dir = experiment.train_args.output_dir
    #trainer.save_model(final_output_dir) 

if __name__ == "__main__":
    fire.Fire(main)