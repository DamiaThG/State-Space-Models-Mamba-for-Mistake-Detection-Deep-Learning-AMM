FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C

RUN apt-get update && apt-get install -y git wget build-essential && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip
RUN pip install packaging ninja

# Installiamo tutte le librerie bloccando PyTorch alla 2.4.0 per evitare aggiornamenti a sorpresa
RUN pip install lightning wandb einops scikit-learn matplotlib seaborn pandas tqdm xlstm torchvision "torch==2.4.0" "transformers==4.37.2"

# Installiamo i binari PRE-COMPILATI di causal-conv1d e mamba-ssm 
# (specifici per PyTorch 2.4, CUDA 12 e Python 3.11, con ABI=FALSE come richiesto dall'immagine base)
RUN pip install https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.3.post1/causal_conv1d-1.5.3.post1%2Bcu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
RUN pip install https://github.com/state-spaces/mamba/releases/download/v2.2.6.post2/mamba_ssm-2.2.6.post2%2Bcu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
