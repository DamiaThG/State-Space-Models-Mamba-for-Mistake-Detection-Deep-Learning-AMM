from huggingface_hub import hf_hub_download
import os

print("Download TSM features v4 (C10119_rgb)...")

local_path = hf_hub_download(
    repo_id="cvml-nus/assembly101",
    repo_type="dataset",
    filename="TSM_features/C10119_rgb.zip",
    local_dir="/home/mssdmn01t05c351v/assembly-mistake-detection/data/TSM_features",
    token=os.environ.get("HF_TOKEN")
)

print(f"Scaricato in: {local_path}")
print("Download completato.")
