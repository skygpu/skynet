from pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

env DEBIAN_FRONTEND=noninteractive

run apt-get update && apt-get install -y git wget

run pip install --upgrade \
    diffusers[torch] \
    transformers \
    huggingface_hub \
    pyTelegramBotAPI \
    pymongo \
    scipy \
    pdbpp

run mkdir /scripts
run mkdir /outputs
run mkdir /inputs

env HF_HOME /hf_home

run mkdir /hf_home

workdir /scripts

env PYTORCH_CUDA_ALLOC_CONF max_split_size_mb:128
