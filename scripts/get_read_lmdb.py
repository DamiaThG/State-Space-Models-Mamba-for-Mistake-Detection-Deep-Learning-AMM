# pyrefly: ignore [missing-import]
from huggingface_hub import hf_hub_download
import os

path = hf_hub_download(
    repo_id="cvml-nus/assembly101",
    repo_type="dataset",
    filename="TSM_features/read_lmdb.py",
    local_dir="/home/mssdmn01t05c351v/assembly-mistake-detection/scripts",
    token=os.environ.get("HF_TOKEN")
)
print(f"Scaricato in: {path}")
