from pytorch/pytorch:latest

env DEBIAN_FRONTEND=noninteractive

run apt-get update && apt-get install -y git wget

run conda install xformers -c xformers/label/dev

run pip install --upgrade \
    diffusers[torch] \
    accelerate \
    transformers \
    huggingface_hub \
    pyTelegramBotAPI \
    pymongo \
    scipy \
    pdbpp

env NVIDIA_VISIBLE_DEVICES=all

run mkdir /scripts
run mkdir /outputs
run mkdir /inputs

env HF_HOME /hf_home

run mkdir /hf_home

workdir /scripts

env PYTORCH_CUDA_ALLOC_CONF max_split_size_mb:128
