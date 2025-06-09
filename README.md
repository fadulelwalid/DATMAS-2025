# Dependencies

This assumes you have already downloaded the project files to its own directory. First one should install all dependencies from the requirements.txt file. Python version 3.12 has been usedBeing in a virtual python environment, this can be done using "pip install -r requirements.txt". It is recommended to use a virtual environment to avoid conflicts with packages already install on your local machine.

Another required dependency is docker. This is for the PostgreSQL database. Initialize or startup the PostgreSQL database using the docker compose yml file located in the postgres directory. This can be run using the following command (assuming current working directory is the root directory of the project.) "docker compose --file ./postgres/postgres-llmware.yml up".

The model you want to use should be downloaded locally. Downloading of the models can be done using "huggingface-cli" command in a python environment (after downloading all python dependencies). Our solution is tested with these specific HuggingFace models; 

* meta-llama/Llama-3.2-3B-Instruct
* Qwen/Qwen2.5-3B-Instruct
* Qwen/Qwen2.5-7B-Instruct-1M

We can assume that other variants of these models in the specific "Qwen2.5" or "Llama-3.2" family would work as well.

Optional dependency is the use of CUDA. Follow the link and download / setup CUDA for your specific environment. Link: https://developer.nvidia.com/cuda-downloads

# Configs

Configuration for running the different scenarios should be done in the "config.json" file. Main point to configure here is the model path. Relative path of the root directory should be provided in the "path" variable in file "src/start.py" (for inference and training). For evaluation, results should be manually moved and organized system wise. All model generation outputs for each system being evaluated should be moved into the same directory.

# Data

Relevant data this solution requires should be put in the specific order in accordance with structured tree provided in Section \ref{sec:cleaner-subsection}. This data should be put in the provided "data" directory.

# Run

We have three different run scenarios. These three scenarios are; running inference, running training and running evaluation. For inference and training, alot of the same edits should be done. Running evaluation is done separately from the main entrypoint.

## Inference

Main entrypoint is "src/start.py". For running inference, one should select the model you want to use. Select a model from the "models" variable and provide it to the "model" variable. Then select function "run" (comment out function "train"). Changing which system to run inference on is done in "src/pipeline.py". More specifically, in the "run\_all" method, change the "systems" variable to reflect which system you want to run inference on.

## Training

For running training, alot of the same edits from Subsection \ref{sec:inference} should also be used. Main difference is commenting out "run" function and running "train" function.

## Evaluation

Main entrypoint is "src/metrics.py". A requirement to run this is that inference has been done on at least one model for at least one system. The global variable "SYSTEMS" should be populated with the system names to evaluate. In the if name main block root directory should be specified in the "root\_dir" variable, same as mentioned in Section \ref{sec:config}. Path to the root output where the outputs have been organized should be specified in the "results\_path" variable. Then select or deselect which evaluation function to run.
\chapter{Poster}
