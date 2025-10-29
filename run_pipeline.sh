# Activate environment
conda activate synctalk

# Run geometry preprocessing
python preprocess_geometry.py \
    --data /home/springer/Project/SyncTalk/data/May \
    --method midas \
    --device cuda:0

