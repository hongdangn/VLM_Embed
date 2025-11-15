mkdir -p ./eval-data
wget https://huggingface.co/datasets/TIGER-Lab/MMEB-eval/resolve/main/images.zip
unzip images.zip -d ./eval-data
rm images.zip