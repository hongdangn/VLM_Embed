from huggingface_hub import snapshot_download

repo_id = "dangnguyens1/meta_train"

folders_to_download = [
    "ablation_wo_intra/*",
]

local_dir = snapshot_download(
    repo_id=repo_id,
    allow_patterns=folders_to_download,
    local_dir="./meta_train",  
    repo_type="model"                
)

print(f"Folders downloaded to: {local_dir}")