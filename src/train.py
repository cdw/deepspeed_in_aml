#!/usr/bin/env python3
# encoding: utf-8
"""
train.py - train an NLP model using HuggingFace, as configured by config.yml

This is partially derived from [here](https://github.com/cdw/azureml-examples/blob/cdw/deepspeed_transformers/python-sdk/workflows/train/deepspeed/transformers/)
"""

import os
import shutil
import yaml
import numpy as np
import azureml.core
import transformers
import datasets

# Connect to the local run and azureml workspace
run = azureml.core.run.Run.get_context()
workspace = run.experiment.workspace

# Load the config file and copy to output for reproducibility
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)
shutil.copy("config.yml", config["training_args"]["output_dir"])


def train():
    """Perform a training run, distributed over multiple machines """
    # Load model from hub
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        config["model"]
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(config["model"])

    # Load dataset from storage on AML
    aml_dataset = azureml.core.Dataset.get_by_name(
        workspace, config["task"], config["task_dataset_version"]
    )
    aml_dataset.download(config["data_dir"], True)
    dataset = datasets.load_from_disk(config["data_dir"])
    train_dataset = dataset["train"]
    eval_dataset = dataset[config["val_key"]]
    ## REMOVE FOR FULL DATASET
    # train_dataset = dataset['train'].shuffle(seed=42).select(range(1000))
    # eval_dataset = dataset[config['val_key']].shuffle(seed=42).select(range(1000))

    # Construct metric function
    metric = datasets.load_metric(config["metric"])

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    # Load training arguments from config and local environment
    config["training_args"]["local_rank"] = os.environ["LOCAL_RANK"]
    trainer_args = transformers.TrainingArguments(**config["training_args"])

    # Load trainer
    trainer = transformers.Trainer(
        model=model,
        args=trainer_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train
    print("Training...")
    trainer.train()

    # Save model
    print("Saving model, only on head node...")
    if trainer.is_world_process_zero():
        trainer.save_model(config["model_output_dir"])

    # Register model
    print("Registering model, only on head node...")
    if trainer.is_world_process_zero():
        aml_model = azureml.core.model.Model.register(
            workspace=workspace,
            model_path=config["model_output_dir"],
            model_name=config["registered_model_name"],
            tags=run.tags,
            description=config["description"],
            datasets=[("finetuning_dataset", aml_dataset)],
        )
        run.set_tags(
            {
                "registered_model_name": config["registered_model_name"],
                "registered_model_version": str(aml_model.version),
            }
        )


if __name__ == "__main__":
    train()
