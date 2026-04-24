# VLMEmbed
## Set up env
```bash
apt-get update
apt-get upgrade -y
cd VLM_Embed
python -m venv vlm
source vlm/bin/activate
```
## Set up
```
pip install -r requirements.txt
```
## Download dataset
1. Download the eval image file zip from huggingface (`optional`) 
```bash
cd VLM_Embed
wget https://huggingface.co/datasets/TIGER-Lab/MMEB-eval/resolve/main/images.zip
unzip images.zip -d eval_images/
```
2. Download train image, it can take > 1 hour to download
```bash
cd VLM_Embed
bash download_traindata.sh
bash download_traindata_2.sh
```
3. Download gradient files from *https://huggingface.co/dangnguyens1/teacher_gradients/blob/main/qwenvl_2b_cls_vqa_grad.zip* and unzip it

4. Fix some line code 

Because of the error of code in **Transformers library**, run the following script to find the error and comment some lines: 

Just comment the following code, from line 140 to 143 in file **/vlm/lib/python3.12/site-packages/transformers/models/qwen2_vl/image_processing_qwen2_vl.py**: 
```python
if size is not None and ("shortest_edge" not in size or "longest_edge" not in size):
    raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")
else:
    size = {"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 1280}
```
Or run `fix_lib.py` to fix: 
```python 
python fix_lib.py
```

## Training

Just run the scripts in folder `scripts`
- To run GVendi Distillation for `qwen2b_cls_grad`, you run: 
```bash
bash scripts/test_gvendi.sh
```

NOTE: 
- You have to ensure that the PHASE 1 TRAINING script must be COMMENTED. You need to UNCOMMENT the PHASE 2 TRAINING script (I did it for you but you need to check it again in `test_gvendi.sh`)

- In the `test_gvendi.sh`, please check the `--image_dir` and `--teacher_cache_dir` arguments: it's the path to the downloaded training images and the downloaded `qwen2b_cls_grad` gradients from HF respectively.

## Acknowledgement
- We have adapted code from [VLM2Vec]([https://github.com/TIGER-AI-Lab/VLM2Vec]) and [B3](https://github.com/raghavlite/B3)
