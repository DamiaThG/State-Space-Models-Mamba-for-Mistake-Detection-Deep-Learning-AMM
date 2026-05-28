from src.dataloader.dataset import AssemblyMistakeDataset, collate_fn
from torch.utils.data import DataLoader

ds = AssemblyMistakeDataset(
    annotations_dir="/home/mssdmn01t05c351v/assembly-mistake-detection/data/annotations/assembly101-mistake-detection/annots",
    lmdb_dir="/home/mssdmn01t05c351v/assembly-mistake-detection/data/TSM_features/C10119_rgb",
)
print(f"Campioni totali: {len(ds)}")
print(f"Class weights: {ds.get_class_weights()}")

item = ds[0]
print(f"Features shape: {item['features'].shape}")
print(f"Label: {item['label']}")
print(f"Meta: {item['meta']}")
