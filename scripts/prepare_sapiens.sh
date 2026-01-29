mkdir data_utils_enhancement/sapiens/checkpoint
cd data_utils_enhancement/sapiens/checkpoint
export GIT_LFS_SKIP_SMUDGE=1

echo "[STATUS] Downloading sapiens-depth-0.3b-torchscript..."
git clone https://huggingface.co/facebook/sapiens-depth-0.3b-torchscript
# git clone https://hf-mirror.com/facebook/sapiens-depth-0.3b-torchscript

cd sapiens-depth-0.3b-torchscript
git lfs pull

cd ../


echo "[STATUS] Downloading sapiens-normal-0.3b-torchscript..."
git clone https://huggingface.co/facebook/sapiens-normal-0.3b-torchscript
# git clone https://hf-mirror.com/facebook/sapiens-normal-0.3b-torchscript

cd sapiens-normal-0.3b-torchscript
git lfs pull