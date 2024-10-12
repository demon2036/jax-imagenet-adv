export GCS_DATASET_DIR=$GCS_DATASET_DIR GCS_MODEL_DIR=$GCS_MODEL_DIR  WANDB_API_KEY=ec6aa52f09f51468ca407c0c00e136aaaa18a445;
rm -rf  ADV-ViT;git clone -b test3 https://github.com/demon2036/jax-imagenet-adv ADV-ViT
cd  ADV-ViT

# 使用 for 循环遍历 SCRIPT_PATHS 数组
for script in "${SCRIPT_PATHS[@]}"; do
    echo " $script"
    pkill -9 -f python
    sudo rm /tmp/libtpu_lockfile
    source ~/miniconda3/bin/activate base;python -u main_adv.py --yaml-path $SCRIPT_PATH

done
