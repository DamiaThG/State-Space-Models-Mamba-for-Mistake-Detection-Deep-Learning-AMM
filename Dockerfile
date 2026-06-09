FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C

# Forziamo la compilazione per le GPU del cluster. 
# 7.5 (T4), 8.0 (A100), 8.6 (RTX 3090/A40) e 8.9 (RTX 4090/L40) coprono quasi tutti i cluster moderni.
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9"

RUN apt-get update && apt-get install -y git wget build-essential && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip
RUN pip install packaging ninja

RUN pip install causal-conv1d>=1.2.0 --no-build-isolation
RUN pip install mamba-ssm --no-build-isolation

RUN pip install pytorch-lightning wandb einops scikit-learn matplotlib seaborn pandas tqdm xlstm mambapy torchvision
