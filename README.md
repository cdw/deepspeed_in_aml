# Training models on AML, optionally with Deepspeed

These notebooks show how to fine-tune an NLP model on AzureML. They are intended to be cloned to and executed on an [AzureML compute instance].

## Steps

- Clone this repo into an interactive session on a fresh AzureML compute instance
- Install the `requirements.txt` into the local `AzureML_Py3.8` conda env via `conda activate azureml_py38 && pip install -r requirements.txt`.
- Follow the notebooks in numerical order
  - `01 Prepare environment` creates an AzureML environment that supports Deepspeed training
  - `02 Prepare data` downloads, preprocesses, and registers a dataset for versioned and reproducible training
  - `03 Train model` launches a distributed fine-tuning job using the outputs of the prior notebooks

[AzureML compute instance]: https://docs.microsoft.com/en-us/azure/machine-learning/concept-compute-instance
