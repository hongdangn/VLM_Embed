# VLMEmbed
## Clone repo
```
git clone https://github.com/hongdangn/VLM_Embed.git
```
## Set up env
```bash
cd VLM_Embed
apt-get update
apt-get upgrade -y
apt install tmux zip unzip -y
apt-get install -y libgl1 libglib2.0-0
python -m venv vlm 
source vlm/bin/activate
```
## For installing dependencies, run the following commands:
```
1. pip install -r requirements.txt
2. python fix_lib.py
```
## If the errors still exist, please go to the image_processing_qwen2_vl.py and COMMENT these lines of code:
```
if size is not None and ("shortest_edge" not in size or "longest_edge" not in size):
    raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")
else:
    size = {"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 1280}
```
## To download train images, please run these 2 commands (you can split into 2 terminals for running each command):
```bash
bash download_traindata.sh
bash download_traindata2.sh
```
## Train model
### 1. SFT-training:
```
bash scripts/sft.sh
```

## Acknowledgement
- We have adapted code from [VLM2Vec]([https://github.com/TIGER-AI-Lab/VLM2Vec]) and [B3](https://github.com/raghavlite/B3)
