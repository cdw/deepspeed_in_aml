# Training models on AML, optionally with DeepSpeed

These notebooks show how to fine-tune an NLP model on AzureML. They are intended to be cloned to and executed on an [AzureML compute instance] within a Jupyter environment. They go through the process of creating a DeepSpeed enabled training environment, creating a compute target (if there isn't one already), preparing and registering datasets, fine tuning a model on those data sets, and registering the resulting output model. This is configured and supported by only a few outside files in the `src` directory. 

## Steps

- Clone this repo into an interactive session on a fresh AzureML compute instance
- From the command line, install the `requirements.txt` into the local `AzureML_Py3.8` conda environment via `conda activate azureml_py38 && pip install -r requirements.txt`.
- Follow the notebooks in numerical order
  - `01 Create compute` ensures requirements are installed and compute cluster is accessible
  - `02 Prepare environment` creates an AzureML environment that supports DeepSpeed training
  - `03 Prepare data` downloads, preprocesses, and registers a dataset for versioned and reproducible training
  - `04 Train model` launches a distributed fine-tuning job using the outputs of the prior notebooks

[AzureML compute instance]: https://docs.microsoft.com/en-us/azure/machine-learning/concept-compute-instance
