# Description

This project aims to evaluate the usefullness of small LLMs as penetration testing assistants in offline environments.

# Project has been tested on:

## Software
Docker version 29.2.1, build a5c7197  
ollama version is 0.15.5  
Windows 11  

## Hardware

NVIDIA RTX4070TI (12GB VRAM)  
32GB DDR5 RAM (+40GB pagefile)  
AMD RYZEN 7 7700x 8-core  

The system will likely perform poorly if run on anything less than this, and will crash if not enough RAM or VRAM is available.  


# Deployment

## Install dependencies

Ollama  # https://ollama.com/download/windows  
Docker  # https://docs.docker.com/desktop/setup/install/windows-install/  


## Download required AI models
```cmd
ollama pull deepseek-coder:1.3b

ollama pull llama3.1:8b

ollama pull mistral-nemo:12b-instruct-2407-q8_0

ollama pull all-minilm:l6-v2
```

https://ollama.com/library/mistral-nemo:12b-instruct-2407-q8_0  
https://ollama.com/library/llama3.1:8b  
https://ollama.com/library/deepseek-coder:1.3b  
https://ollama.com/library/all-minilm:l6-v2


## Download the safetensors files for the fine-tuned models

Visit this huggingface page: https://huggingface.co/olelb/cybersecurity_model_pack/tree/main

and download the file "finetuned_models.zip", then extract it to the project root.  

Your folder structure should be like this:  
project-root/  
   |--- main.py  
   |--- finetuned_models/mistral_nemo_cve/...  
   ...  

## Create virtual environment and install python packages

```
python -m virtualenv env
./env/script/activate
# (or however you like to set up environments, this requires the virtualenv package)

pip install -r requirements.txt
```


## Run initialization scritp

```python
python -m initialize
```
This script can take a while as it needs to populate the vector database, which involves embedding a lot of text using a local model.



## Run app

```
python -m main
```

To change the model used by main.py, edit the "DEFAULT_AI_MODEL" variable at line 21
```python
DEFAULT_AI_MODEL = "mistral_nemo_cve:latest"
```